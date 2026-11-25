use ndarray::Array3;
use std::io::{Cursor, Seek, SeekFrom, Write};
use std::sync::{
    atomic::{AtomicBool, AtomicUsize, Ordering},
    Arc,
};
use tiff::{decoder::Decoder, tags::Tag};
use xarray_cog::cog::{CogBuilder, Compression, GeoTiffBuilder, TiffVariant};

struct InterruptedSink {
    data: Cursor<Vec<u8>>,
    bytes: Arc<AtomicUsize>,
    cancelled: Arc<AtomicBool>,
    fail_write: bool,
}

impl Write for InterruptedSink {
    fn write(&mut self, data: &[u8]) -> std::io::Result<usize> {
        if self.fail_write && self.bytes.load(Ordering::Relaxed) >= 1 << 20 {
            return Err(std::io::Error::other("injected payload write failure"));
        }
        let count = self.data.write(data)?;
        if self.bytes.fetch_add(count, Ordering::Relaxed) + count >= 1 << 20 {
            self.cancelled.store(true, Ordering::Relaxed);
        }
        Ok(count)
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

impl Seek for InterruptedSink {
    fn seek(&mut self, position: SeekFrom) -> std::io::Result<u64> {
        self.data.seek(position)
    }
}

fn interrupted_payload_copy(fail_write: bool) {
    let scratch = tempfile::tempdir().unwrap();
    let bytes = Arc::new(AtomicUsize::new(0));
    let cancelled = Arc::new(AtomicBool::new(false));
    let sink = InterruptedSink {
        data: Cursor::new(Vec::new()),
        bytes: bytes.clone(),
        cancelled: cancelled.clone(),
        fail_write,
    };
    let cog = CogBuilder::new(
        GeoTiffBuilder::new(1024, 512)
            .tile_size(128, 128)
            .compression(Compression::None)
            .temp_dir(scratch.path()),
    )
    .overview_levels(vec![2, 4]);
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(2)
        .build()
        .unwrap();
    let mut writer = cog
        .pyramid_tile_writer::<f32, _>(sink, cog.pyramid_memory_bytes::<f32>().unwrap())
        .unwrap();
    for row in 0..4 {
        let tiles: Vec<_> = (0..8)
            .map(|col| (col, row, Array3::from_elem((128, 128, 1), 1f32)))
            .collect();
        writer.write_tiles(&tiles, &pool, &|| Ok(())).unwrap();
    }
    assert_eq!(bytes.load(Ordering::Relaxed), 0);
    let result = writer.finish(&pool, &|| {
        if !fail_write && cancelled.load(Ordering::Relaxed) {
            Err(xarray_cog::cog::Error::Other(
                "cancelled during payload copy".into(),
            ))
        } else {
            Ok(())
        }
    });
    let error = result.err().expect("payload copy must fail").to_string();
    assert!(
        error.contains(if fail_write {
            "injected payload write failure"
        } else {
            "cancelled during payload copy"
        }),
        "{error}"
    );
    let written = bytes.load(Ordering::Relaxed);
    assert!(
        ((1usize << 20)..(2usize << 20)).contains(&written),
        "{written} bytes"
    );
    assert_eq!(std::fs::read_dir(scratch.path()).unwrap().count(), 0);
}

#[test]
fn fused_copy_observes_cancellation_after_payload_progress() {
    interrupted_payload_copy(false);
}

#[test]
fn fused_copy_propagates_late_sink_error_and_releases_spools() {
    interrupted_payload_copy(true);
}

#[test]
fn sparse_fused_pyramid_records_zero_offsets_and_counts() {
    let cog = CogBuilder::new(GeoTiffBuilder::new(19, 21).tile_size(16, 16).sparse(true))
        .overview_levels(vec![2]);
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(2)
        .build()
        .unwrap();
    let mut writer = cog
        .pyramid_tile_writer::<u8, _>(
            Cursor::new(Vec::new()),
            cog.pyramid_memory_bytes::<u8>().unwrap(),
        )
        .unwrap();
    let tiles: Vec<_> = (0..4)
        .map(|i| {
            (
                i % 2,
                i / 2,
                Array3::zeros((16.min(21 - i / 2 * 16), 16.min(19 - i % 2 * 16), 1)),
            )
        })
        .collect();
    writer.write_tiles(&tiles, &pool, &|| Ok(())).unwrap();
    let bytes = writer.finish(&pool, &|| Ok(())).unwrap().into_inner();
    let mut decoder = Decoder::new(Cursor::new(bytes)).unwrap();
    for expected_blocks in [4, 1] {
        assert_eq!(
            decoder.get_tag_u64_vec(Tag::TileOffsets).unwrap(),
            vec![0; expected_blocks]
        );
        assert_eq!(
            decoder.get_tag_u64_vec(Tag::TileByteCounts).unwrap(),
            vec![0; expected_blocks]
        );
        if expected_blocks == 4 {
            decoder.next_image().unwrap();
        }
    }
}

#[test]
fn fused_subifds_reference_decodable_overviews_without_top_level_chain() {
    let cog = CogBuilder::new(
        GeoTiffBuilder::new(19, 21)
            .tile_size(16, 16)
            .tiff_variant(TiffVariant::Classic),
    )
    .overview_levels(vec![2, 4])
    .subifd_overviews();
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(2)
        .build()
        .unwrap();
    let mut writer = cog
        .pyramid_tile_writer::<u8, _>(
            Cursor::new(Vec::new()),
            cog.pyramid_memory_bytes::<u8>().unwrap(),
        )
        .unwrap();
    let tiles: Vec<_> = (0..4)
        .map(|i| {
            (
                i % 2,
                i / 2,
                Array3::from_shape_fn(
                    (16.min(21 - i / 2 * 16), 16.min(19 - i % 2 * 16), 1),
                    |(y, x, _)| (i / 2 * 16 + y + i % 2 * 16 + x) as u8,
                ),
            )
        })
        .collect();
    writer.write_tiles(&tiles, &pool, &|| Ok(())).unwrap();
    let bytes = writer.finish(&pool, &|| Ok(())).unwrap().into_inner();
    let mut decoder = Decoder::new(Cursor::new(&bytes)).unwrap();
    let offsets = decoder.get_tag_u64_vec(Tag::Unknown(330)).unwrap();
    assert_eq!(offsets.len(), 2);
    assert!(!decoder.more_images());
    for (offset, factor) in offsets.into_iter().zip([2usize, 4]) {
        let mut standalone = bytes.clone();
        standalone[4..8].copy_from_slice(&u32::try_from(offset).unwrap().to_le_bytes());
        let mut overview = Decoder::new(Cursor::new(standalone)).unwrap();
        let width = 19usize.div_ceil(factor);
        let height = 21usize.div_ceil(factor);
        assert_eq!(
            overview.dimensions().unwrap(),
            (width as u32, height as u32)
        );
        let tiff::decoder::DecodingResult::U8(values) = overview.read_image().unwrap() else {
            panic!("expected u8")
        };
        for y in 0..height {
            for x in 0..width {
                assert_eq!(values[y * width + x], ((y + x) * factor) as u8);
            }
        }
        assert!(!overview.more_images());
    }
}

#[test]
fn invalid_image_configuration_is_rejected_before_opening_scratch() {
    let scratch = tempfile::tempdir().unwrap();
    let missing = scratch.path().join("missing");
    let cog = CogBuilder::new(
        GeoTiffBuilder::new(19, 21)
            .tile_size(15, 16)
            .temp_dir(&missing),
    )
    .overview_levels(vec![2]);
    let error = cog
        .pyramid_tile_writer::<u16, _>(Cursor::new(Vec::new()), u64::MAX)
        .err()
        .expect("invalid tile size must be rejected");
    assert!(
        error
            .to_string()
            .contains("tile dimensions must be multiples of 16"),
        "{error}"
    );
    assert!(!missing.exists());
}
