//! Layout planning, metadata emission, and block spooling for Cloud-Optimized GeoTIFFs (COG).

use std::fs::File;
use std::io::{self, Read, Seek, SeekFrom, Write};
use std::path::Path;

use tiff_core::{
    ByteOrder, Compression, PlanarConfiguration, Predictor, Tag, TagType, TagValue, TAG_SUB_IFDS,
};
use tiff_writer::{encoder, ImageBuilder, JpegOptions, LercOptions, TiffVariant};

use crate::cog::error::{Error, Result};
use crate::cog::sample::NumericSample;

pub(crate) fn checked_len_u64(len: usize, context: &str) -> Result<u64> {
    u64::try_from(len).map_err(|_| Error::Other(format!("{context} length exceeds u64::MAX")))
}

pub(crate) fn checked_add_u64(lhs: u64, rhs: u64, context: &str) -> Result<u64> {
    lhs.checked_add(rhs)
        .ok_or_else(|| Error::Other(format!("{context} overflow")))
}

pub(crate) fn checked_samples_per_pixel(bands: usize) -> Result<u16> {
    u16::try_from(bands).map_err(|_| {
        Error::InvalidConfig(format!(
            "band count {bands} exceeds TIFF SamplesPerPixel limit {}",
            u16::MAX
        ))
    })
}

pub(crate) fn checked_sample_count(dimensions: &[usize], context: &str) -> Result<usize> {
    dimensions
        .iter()
        .try_fold(1usize, |sample_count, &dimension| {
            sample_count
                .checked_mul(dimension)
                .ok_or_else(|| Error::Other(format!("{context} sample count overflows usize")))
        })
}

pub(crate) fn gdal_structural_metadata_bytes(planar_configuration: PlanarConfiguration) -> Vec<u8> {
    let mut payload = String::from(
        "LAYOUT=IFDS_BEFORE_DATA\n\
BLOCK_ORDER=ROW_MAJOR\n\
BLOCK_LEADER=SIZE_AS_UINT4\n\
BLOCK_TRAILER=LAST_4_BYTES_REPEATED\n\
KNOWN_INCOMPATIBLE_EDITION=NO\n",
    );
    if matches!(planar_configuration, PlanarConfiguration::Planar) {
        payload.push_str("INTERLEAVE=BAND\n");
    }
    payload.push(' ');
    format!(
        "GDAL_STRUCTURAL_METADATA_SIZE={:06} bytes\n{}",
        payload.len(),
        payload
    )
    .into_bytes()
}

pub(crate) fn gdal_block_leader(payload_len: usize, byte_order: ByteOrder) -> Result<Vec<u8>> {
    let block_len = u32::try_from(payload_len)
        .map_err(|_| Error::Other("COG block payload exceeds u32::MAX".into()))?;
    let mut leader = Vec::with_capacity(4);
    leader.extend_from_slice(&byte_order.write_u32(block_len));
    Ok(leader)
}

pub(crate) fn gdal_block_trailer(bytes: &[u8]) -> Vec<u8> {
    if bytes.len() >= 4 {
        bytes[bytes.len() - 4..].to_vec()
    } else {
        bytes.to_vec()
    }
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct CogBlockEncoding {
    pub compression: Compression,
    pub predictor: Predictor,
    pub samples_per_pixel: u16,
    pub row_width_pixels: usize,
    pub block_height: u32,
    pub lerc_options: Option<LercOptions>,
    pub jpeg_options: Option<JpegOptions>,
    pub jpeg_sampling: Option<[u16; 2]>,
    pub deflate_level: Option<u32>,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct TileWritePlan {
    pub tile_width: usize,
    pub tile_height: usize,
    pub planar_configuration: PlanarConfiguration,
    pub compression: Compression,
    pub predictor: Predictor,
    pub lerc_options: Option<LercOptions>,
    pub jpeg_options: Option<JpegOptions>,
    pub jpeg_sampling: Option<[u16; 2]>,
    pub deflate_level: Option<u32>,
    pub sparse: bool,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct CogBlockRecord {
    pub spool_offset: u64,
    pub logical_offset_delta: u64,
    pub logical_byte_count: u64,
    pub sparse: bool,
}

pub(crate) struct CogImage {
    pub builder: ImageBuilder,
    pub blocks: Vec<CogBlockRecord>,
    pub sub_ifd_count: usize,
}

pub(crate) struct PlannedCogImage {
    pub tags: Vec<Tag>,
    pub block_offsets: Vec<u64>,
    pub block_byte_counts: Vec<u64>,
}

pub(crate) struct CogLayout {
    pub base_offset: u64,
    pub is_bigtiff: bool,
    pub prefix_padding: usize,
    pub images: Vec<PlannedCogImage>,
}

pub(crate) struct BlockSpool {
    pub storage: File,
    pub len: u64,
}

impl BlockSpool {
    pub fn new(temp_dir: Option<&Path>) -> Result<Self> {
        Ok(Self {
            storage: match temp_dir {
                Some(path) => tempfile::tempfile_in(path)?,
                None => tempfile::tempfile()?,
            },
            len: 0,
        })
    }

    pub fn append_segmented(
        &mut self,
        prefix: &[u8],
        payload: &[u8],
        suffix: &[u8],
    ) -> Result<CogBlockRecord> {
        let spool_offset = self.len;
        let prefix_len = checked_len_u64(prefix.len(), "COG block prefix")?;
        let payload_len = checked_len_u64(payload.len(), "COG block payload")?;
        let suffix_len = checked_len_u64(suffix.len(), "COG block suffix")?;
        let physical_len = checked_add_u64(
            checked_add_u64(prefix_len, payload_len, "COG block size")?,
            suffix_len,
            "COG block size",
        )?;

        self.storage.seek(SeekFrom::End(0))?;
        self.storage.write_all(prefix)?;
        self.storage.write_all(payload)?;
        self.storage.write_all(suffix)?;
        self.len = checked_add_u64(self.len, physical_len, "COG spool length")?;

        Ok(CogBlockRecord {
            spool_offset,
            logical_offset_delta: prefix_len,
            logical_byte_count: payload_len,
            sparse: false,
        })
    }

    pub fn copy_into<W: Write + Seek>(
        &mut self,
        sink: &mut W,
        check: &impl Fn() -> Result<()>,
    ) -> Result<()> {
        self.storage.seek(SeekFrom::Start(0))?;
        sink.seek(SeekFrom::End(0))?;
        let mut buffer = [0u8; 64 * 1024];
        loop {
            check()?;
            let count = match self.storage.read(&mut buffer) {
                Err(e) if e.kind() == io::ErrorKind::Interrupted => continue,
                result => result?,
            };
            if count == 0 {
                break;
            }
            sink.write_all(&buffer[..count])?;
        }
        Ok(())
    }
}

pub(crate) fn compress_cog_block<T: NumericSample>(
    samples: &[T],
    block_index: usize,
    encoding: CogBlockEncoding,
) -> Result<Vec<u8>> {
    if matches!(encoding.compression, Compression::Lerc) {
        let opts = encoding.lerc_options.unwrap_or_default();
        tiff_writer::compress::compress_block_lerc(
            samples,
            encoding.row_width_pixels as u32,
            encoding.block_height,
            encoding.samples_per_pixel as u32,
            &opts,
            block_index,
        )
        .map_err(Into::into)
    } else {
        tiff_writer::compress::compress_block(
            samples,
            tiff_writer::compress::BlockEncodingOptions {
                byte_order: ByteOrder::LittleEndian,
                compression: encoding.compression,
                predictor: encoding.predictor,
                samples_per_pixel: encoding.samples_per_pixel,
                row_width_pixels: encoding.row_width_pixels,
                jpeg_options: encoding.jpeg_options.as_ref(),
                jpeg_sampling: encoding.jpeg_sampling,
                deflate_level: encoding.deflate_level,
            },
            block_index,
        )
        .map_err(Into::into)
    }
}

fn sub_ifds_tag(count: usize, is_bigtiff: bool) -> Result<Tag> {
    let count_u64 = checked_len_u64(count, "SubIFD offset count")?;
    if is_bigtiff {
        Ok(Tag {
            code: TAG_SUB_IFDS,
            tag_type: TagType::Ifd8,
            count: count_u64,
            value: TagValue::Long8(vec![0; count]),
        })
    } else {
        Ok(Tag::new(TAG_SUB_IFDS, TagValue::Long(vec![0; count])))
    }
}

fn build_cog_image_tags(image: &CogImage, is_bigtiff: bool) -> Result<Vec<Tag>> {
    let mut tags = image.builder.checked_build_tags(is_bigtiff)?;
    if image.sub_ifd_count > 0 {
        tags.push(sub_ifds_tag(image.sub_ifd_count, is_bigtiff)?);
        tags.sort_by_key(|tag| tag.code);
    }
    Ok(tags)
}

fn plan_cog_layout_for_variant(
    base_offset: u64,
    prefix_len: u64,
    images: &[CogImage],
    is_bigtiff: bool,
) -> Result<CogLayout> {
    let mut image_plans = Vec::with_capacity(images.len());
    let prefix_end = checked_add_u64(
        checked_add_u64(
            base_offset,
            encoder::header_len(is_bigtiff),
            "COG header size",
        )?,
        prefix_len,
        "COG prefix size",
    )?;
    let prefix_padding = usize::from(prefix_end % 2 != 0);
    let mut current = checked_add_u64(prefix_end, prefix_padding as u64, "COG prefix padding")?;

    for image in images {
        let expected_blocks = image.builder.checked_block_count()?;
        if image.blocks.len() != expected_blocks {
            return Err(Error::Other(format!(
                "COG image is missing block records: expected {expected_blocks}, got {}",
                image.blocks.len()
            )));
        }
        let tags = build_cog_image_tags(image, is_bigtiff)?;
        current = checked_add_u64(
            current,
            encoder::estimate_ifd_size(ByteOrder::LittleEndian, is_bigtiff, &tags)?,
            "COG IFD layout",
        )?;
        if !is_bigtiff {
            u32::try_from(current).map_err(|_| {
                Error::Tiff(tiff_writer::Error::ClassicOffsetOverflow { offset: current })
            })?;
        }
        image_plans.push(PlannedCogImage {
            tags,
            block_offsets: Vec::with_capacity(image.blocks.len()),
            block_byte_counts: Vec::with_capacity(image.blocks.len()),
        });
    }

    let data_start = current;
    for (image, planned) in images.iter().zip(image_plans.iter_mut()) {
        for block in &image.blocks {
            if block.sparse {
                planned.block_offsets.push(0);
                planned.block_byte_counts.push(0);
                continue;
            }
            let physical_start =
                checked_add_u64(data_start, block.spool_offset, "COG block physical offset")?;
            let logical_offset = checked_add_u64(
                physical_start,
                block.logical_offset_delta,
                "COG block logical offset",
            )?;
            if !is_bigtiff {
                u32::try_from(logical_offset).map_err(|_| {
                    Error::Tiff(tiff_writer::Error::ClassicOffsetOverflow {
                        offset: logical_offset,
                    })
                })?;
                u32::try_from(block.logical_byte_count).map_err(|_| {
                    Error::Tiff(tiff_writer::Error::ClassicByteCountOverflow {
                        byte_count: block.logical_byte_count,
                    })
                })?;
            }
            planned.block_offsets.push(logical_offset);
            planned.block_byte_counts.push(block.logical_byte_count);
        }
    }

    Ok(CogLayout {
        base_offset,
        is_bigtiff,
        prefix_padding,
        images: image_plans,
    })
}

pub(crate) fn plan_cog_layout(
    base_offset: u64,
    prefix_len: u64,
    variant: TiffVariant,
    images: &[CogImage],
) -> Result<CogLayout> {
    match variant {
        TiffVariant::Classic => plan_cog_layout_for_variant(base_offset, prefix_len, images, false),
        TiffVariant::BigTiff => plan_cog_layout_for_variant(base_offset, prefix_len, images, true),
        TiffVariant::Auto => {
            match plan_cog_layout_for_variant(base_offset, prefix_len, images, false) {
                Ok(layout) => Ok(layout),
                Err(Error::Tiff(
                    tiff_writer::Error::ClassicOffsetOverflow { .. }
                    | tiff_writer::Error::ClassicByteCountOverflow { .. },
                )) => plan_cog_layout_for_variant(base_offset, prefix_len, images, true),
                Err(err) => Err(err),
            }
        }
    }
}

pub(crate) fn emit_cog_metadata<W: Write + Seek>(
    sink: &mut W,
    prefix: &[u8],
    images: &[CogImage],
    layout: &CogLayout,
) -> Result<()> {
    sink.seek(SeekFrom::Start(layout.base_offset))?;
    encoder::write_header(sink, ByteOrder::LittleEndian, layout.is_bigtiff)?;
    sink.write_all(prefix)?;
    sink.write_all(&[0u8; 1][..layout.prefix_padding])?;

    let mut ifd_results = Vec::with_capacity(images.len());
    for (image, planned) in images.iter().zip(&layout.images) {
        let (offsets_tag_code, byte_counts_tag_code) = image.builder.offset_tag_codes();
        let ifd_result = encoder::write_ifd(
            sink,
            ByteOrder::LittleEndian,
            layout.is_bigtiff,
            &planned.tags,
            offsets_tag_code,
            byte_counts_tag_code,
            image.builder.checked_block_count()?,
        )?;
        ifd_results.push(ifd_result);
    }

    for (index, image) in images.iter().enumerate() {
        let planned = &layout.images[index];
        let ifd_result = &ifd_results[index];
        let (offsets_tag_code, byte_counts_tag_code) = image.builder.offset_tag_codes();

        let tag_offset = |code| {
            encoder::find_tag_value_offset(
                ifd_result.ifd_offset,
                layout.is_bigtiff,
                &planned.tags,
                code,
            )
        };
        if let Some(off) = tag_offset(offsets_tag_code) {
            encoder::patch_block_offsets(
                sink,
                ByteOrder::LittleEndian,
                layout.is_bigtiff,
                off,
                &planned.block_offsets,
            )?;
        }
        if let Some(off) = tag_offset(byte_counts_tag_code) {
            encoder::patch_block_byte_counts(
                sink,
                ByteOrder::LittleEndian,
                layout.is_bigtiff,
                off,
                &planned.block_byte_counts,
            )?;
        }
    }

    let first_ifd = ifd_results
        .first()
        .ok_or_else(|| Error::Other("COG layout contains no images".into()))?;
    encoder::patch_first_ifd(
        sink,
        layout.base_offset,
        ByteOrder::LittleEndian,
        layout.is_bigtiff,
        first_ifd.ifd_offset,
    )?;

    let sub_ifd_count = images.first().map(|image| image.sub_ifd_count).unwrap_or(0);
    if sub_ifd_count > 0 {
        let sub_ifd_offsets: Vec<u64> = ifd_results
            .iter()
            .skip(1)
            .take(sub_ifd_count)
            .map(|result| result.ifd_offset)
            .collect();
        if sub_ifd_offsets.len() != sub_ifd_count {
            return Err(Error::Other(format!(
                "COG SubIFD layout expected {sub_ifd_count} overview IFDs, found {}",
                sub_ifd_offsets.len()
            )));
        }
        let sub_ifd_data_offset = encoder::find_tag_value_offset(
            first_ifd.ifd_offset,
            layout.is_bigtiff,
            &layout.images[0].tags,
            TAG_SUB_IFDS,
        )
        .ok_or_else(|| Error::Other("COG base IFD is missing SubIFDs tag".into()))?;
        encoder::patch_block_offsets(
            sink,
            ByteOrder::LittleEndian,
            layout.is_bigtiff,
            sub_ifd_data_offset,
            &sub_ifd_offsets,
        )?;
    } else {
        for index in 1..ifd_results.len() {
            encoder::patch_next_ifd(
                sink,
                ByteOrder::LittleEndian,
                layout.is_bigtiff,
                ifd_results[index - 1].next_ifd_pointer_offset,
                ifd_results[index].ifd_offset,
            )?;
        }
    }

    sink.seek(SeekFrom::End(0))?;
    Ok(())
}
