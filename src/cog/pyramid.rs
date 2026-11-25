//! Ordered streaming nearest-neighbor pyramids without raw raster staging.

use std::io::{Seek, Write};

use ndarray::Array3;
use rayon::prelude::*;
use tiff_core::ByteOrder;

use crate::cog::builder::{CogBuilder, Resampling};
use crate::cog::error::{Error, Result};
use crate::cog::layout::{
    checked_add_u64, checked_len_u64, checked_sample_count, checked_samples_per_pixel,
    compress_cog_block, emit_cog_metadata, gdal_block_leader, gdal_block_trailer,
    gdal_structural_metadata_bytes, plan_cog_layout, BlockSpool, CogBlockEncoding, CogBlockRecord,
    CogImage, TileWritePlan,
};
use crate::cog::sample::{parse_nodata_value, NumericSample};

struct Frontier<T> {
    factor: usize,
    height: usize,
    row: usize,
    tiles: Vec<Vec<T>>,
    sums: Vec<Vec<f64>>,
    counts: Vec<Vec<u32>>,
}

/// Pixel-interleaved nearest pyramid writer accepting complete row-major base tiles.
pub struct CogPyramidWriter<T: NumericSample, W: Write + Seek> {
    sink: W,
    cog: CogBuilder,
    plan: TileWritePlan,
    images: Vec<CogImage>,
    spools: Vec<BlockSpool>,
    frontiers: Vec<Frontier<T>>,
    fill: T,
    nodata_value: Option<T>,
    next_tile: usize,
    across: usize,
    down: usize,
}

// Levels are validated by each caller.
pub(crate) fn pyramid_frontier_bytes(
    width: u32,
    tile_size: (u32, u32),
    bands: u32,
    sample_bytes: u64,
    resampling: Resampling,
    levels: &[u32],
) -> Result<u64> {
    let (tw, th) = (tile_size.0 as usize, tile_size.1 as usize);
    if tw == 0 || th == 0 {
        return Err(Error::InvalidConfig(
            "tile dimensions must be positive".into(),
        ));
    }
    let bytes_per_sample = checked_add_u64(
        sample_bytes,
        if resampling == Resampling::Average {
            12
        } else {
            0
        },
        "pyramid frontier sample bytes",
    )?;
    let mut bytes = 0u64;
    for &factor in levels {
        let across = (width as usize).div_ceil(factor as usize).div_ceil(tw);
        let samples = checked_sample_count(&[across, tw, th, bands as usize], "pyramid frontier")?;
        let level_bytes = checked_len_u64(samples, "pyramid frontier")?
            .checked_mul(bytes_per_sample)
            .ok_or_else(|| Error::InvalidConfig("pyramid frontier byte count overflow".into()))?;
        bytes = checked_add_u64(bytes, level_bytes, "pyramid frontier")?;
    }
    Ok(bytes)
}

impl CogBuilder {
    /// Bytes retained by padded overview tile-row frontiers, including average sums/counts.
    pub fn pyramid_memory_bytes<T: NumericSample>(&self) -> Result<u64> {
        pyramid_frontier_bytes(
            self.inner.width,
            (
                self.inner.tile_width.unwrap_or(256),
                self.inner.tile_height.unwrap_or(256),
            ),
            self.inner.bands,
            T::BYTES_PER_SAMPLE as u64,
            self.resampling,
            &self.normalized_overview_levels()?,
        )
    }

    /// Start a streaming pixel-interleaved pyramid within a frontier budget.
    pub fn pyramid_tile_writer<T: NumericSample, W: Write + Seek>(
        &self,
        sink: W,
        frontier_budget: u64,
    ) -> Result<CogPyramidWriter<T, W>> {
        if !matches!(
            self.resampling,
            Resampling::NearestNeighbor | Resampling::Average
        ) || self.inner.planar_configuration != tiff_core::PlanarConfiguration::Chunky
        {
            return Err(Error::InvalidConfig(
                "streaming pyramids require nearest or average resampling and pixel interleave"
                    .into(),
            ));
        }
        let required = self.pyramid_memory_bytes::<T>()?;
        if required > frontier_budget {
            return Err(Error::InvalidConfig(format!(
                "pyramid frontier requires {required} bytes, budget is {frontier_budget}"
            )));
        }
        let tw = self.inner.tile_width.unwrap_or(256);
        let th = self.inner.tile_height.unwrap_or(256);
        let levels = self.normalized_overview_levels()?;
        let images = self.build_images::<T>(&levels, tw, th)?;
        for image in &images {
            image.builder.validate()?;
        }
        let nodata_value = parse_nodata_value::<T>(&self.inner.nodata)?;
        let fill = nodata_value.unwrap_or_else(T::zero);
        let tile_samples = checked_sample_count(
            &[tw as usize, th as usize, self.inner.bands as usize],
            "pyramid tile",
        )?;
        let is_average = self.resampling == Resampling::Average;
        let mut frontiers = Vec::with_capacity(levels.len());
        let mut spools = Vec::with_capacity(images.len());
        for _ in &images {
            spools.push(BlockSpool::new(self.inner.temp_dir.as_deref())?);
        }
        for factor in levels {
            let width = (self.inner.width as usize).div_ceil(factor as usize);
            let num_tiles = width.div_ceil(tw as usize);
            frontiers.push(Frontier {
                factor: factor as usize,
                height: (self.inner.height as usize).div_ceil(factor as usize),
                row: 0,
                tiles: (0..num_tiles).map(|_| vec![fill; tile_samples]).collect(),
                sums: if is_average {
                    (0..num_tiles).map(|_| vec![0.0; tile_samples]).collect()
                } else {
                    Vec::new()
                },
                counts: if is_average {
                    (0..num_tiles).map(|_| vec![0; tile_samples]).collect()
                } else {
                    Vec::new()
                },
            });
        }
        let plan = TileWritePlan {
            tile_width: tw as usize,
            tile_height: th as usize,
            planar_configuration: self.inner.planar_configuration,
            compression: self.inner.compression,
            predictor: self.inner.predictor,
            lerc_options: self.inner.lerc_options,
            jpeg_options: self.inner.jpeg_options,
            jpeg_sampling: images[0].builder.jpeg_chroma_sampling(),
            deflate_level: self.inner.deflate_level,
            sparse: self.inner.sparse,
        };
        Ok(CogPyramidWriter {
            sink,
            cog: self.clone(),
            plan,
            images,
            spools,
            frontiers,
            fill,
            nodata_value,
            next_tile: 0,
            across: (self.inner.width as usize).div_ceil(tw as usize),
            down: (self.inner.height as usize).div_ceil(th as usize),
        })
    }
}

fn append_compressed(
    spool: &mut BlockSpool,
    compressed: Option<Vec<u8>>,
) -> Result<CogBlockRecord> {
    match compressed {
        Some(bytes) => spool.append_segmented(
            &gdal_block_leader(bytes.len(), ByteOrder::LittleEndian)?,
            &bytes,
            &gdal_block_trailer(&bytes),
        ),
        None => Ok(CogBlockRecord {
            spool_offset: 0,
            logical_offset_delta: 0,
            logical_byte_count: 0,
            sparse: true,
        }),
    }
}

fn compress<T: NumericSample>(
    samples: &[T],
    index: usize,
    plan: TileWritePlan,
    bands: usize,
) -> Result<Option<Vec<u8>>> {
    if plan.sparse && samples.iter().all(|v| *v == T::zero()) {
        return Ok(None);
    }
    Ok(Some(compress_cog_block(
        samples,
        index,
        CogBlockEncoding {
            compression: plan.compression,
            predictor: plan.predictor,
            samples_per_pixel: checked_samples_per_pixel(bands)?,
            row_width_pixels: plan.tile_width,
            block_height: plan.tile_height as u32,
            lerc_options: plan.lerc_options,
            jpeg_options: plan.jpeg_options,
            jpeg_sampling: plan.jpeg_sampling,
            deflate_level: plan.deflate_level,
        },
    )?))
}

impl<T: NumericSample, W: Write + Seek> CogPyramidWriter<T, W> {
    /// Compress base tiles in strict row-major order.
    pub fn write_tiles(
        &mut self,
        tiles: &[(usize, usize, Array3<T>)],
        pool: &rayon::ThreadPool,
        check: &impl Fn() -> Result<()>,
    ) -> Result<()> {
        check()?;
        let tw = self.plan.tile_width;
        let th = self.plan.tile_height;
        let bands = self.cog.inner.bands as usize;
        for (offset, (col, row, data)) in tiles.iter().enumerate() {
            let index = self
                .next_tile
                .checked_add(offset)
                .ok_or_else(|| Error::Other("tile index overflow".into()))?;
            if index >= self.across * self.down
                || *col != index % self.across
                || *row != index / self.across
            {
                return Err(Error::InvalidConfig(
                    "pyramid base tiles must be complete and in row-major order".into(),
                ));
            }
            let expected = (
                th.min(self.cog.inner.height as usize - row * th),
                tw.min(self.cog.inner.width as usize - col * tw),
                bands,
            );
            if data.dim() != expected {
                return Err(Error::InvalidConfig(format!(
                    "pyramid tile shape {:?}, expected {expected:?}",
                    data.dim()
                )));
            }
        }
        let mut offset = 0;
        while offset < tiles.len() {
            let count = (tiles.len() - offset)
                .min(pool.current_num_threads())
                .min(self.across - self.next_tile % self.across);
            let chunk = &tiles[offset..offset + count];
            check()?;
            let start = self.next_tile;
            let plan = self.plan;
            let fill = self.fill;
            let compressed: Vec<_> = pool.install(|| {
                chunk
                    .par_iter()
                    .enumerate()
                    .map(|(offset, (_, _, data))| {
                        if data.dim() == (th, tw, bands) {
                            if let Some(samples) = data.as_slice() {
                                return compress(samples, start + offset, plan, bands);
                            }
                        }
                        let mut padded = vec![fill; tw * th * bands];
                        crate::cog::raster_copy::copy_3d_chunky_region_into(
                            &data.view(),
                            crate::cog::raster_copy::Region {
                                row_start: 0,
                                col_start: 0,
                                rows: data.dim().0,
                                cols: data.dim().1,
                            },
                            &mut padded,
                            tw * bands,
                        );
                        compress(&padded, start + offset, plan, bands)
                    })
                    .collect()
            });
            for ((col, row, data), result) in chunk.iter().zip(compressed) {
                check()?;
                self.images[0]
                    .blocks
                    .push(append_compressed(&mut self.spools[0], result?)?);
                for frontier in &mut self.frontiers {
                    let factor = frontier.factor;
                    let y0 = row * th;
                    let x0 = col * tw;
                    if self.cog.resampling == Resampling::Average {
                        for by in 0..data.dim().0 {
                            let oy = (y0 + by) / factor;
                            let local_oy = oy % th;
                            for bx in 0..data.dim().1 {
                                let ox = (x0 + bx) / factor;
                                let tile_col = ox / tw;
                                let local_ox = ox % tw;
                                let base_idx = (local_oy * tw + local_ox) * bands;
                                for band in 0..bands {
                                    let val = data[(by, bx, band)];
                                    let f = val.to_f64();
                                    if self.nodata_value.is_none_or(|nd| val != nd) && !f.is_nan() {
                                        frontier.sums[tile_col][base_idx + band] += f;
                                        frontier.counts[tile_col][base_idx + band] += 1;
                                    }
                                }
                            }
                        }
                    } else {
                        for oy in y0.div_ceil(factor)..(y0 + data.dim().0).div_ceil(factor) {
                            for ox in x0.div_ceil(factor)..(x0 + data.dim().1).div_ceil(factor) {
                                let target = &mut frontier.tiles[ox / tw];
                                for band in 0..bands {
                                    target[((oy % th) * tw + ox % tw) * bands + band] =
                                        data[(oy * factor - y0, ox * factor - x0, band)];
                                }
                            }
                        }
                    }
                }
                self.next_tile += 1;
            }
            offset += count;
            if self.next_tile.is_multiple_of(self.across) {
                let consumed_rows =
                    (self.next_tile / self.across * th).min(self.cog.inner.height as usize);
                self.flush_frontiers(consumed_rows, pool, check)?;
            }
        }
        Ok(())
    }

    fn flush_frontiers(
        &mut self,
        consumed_rows: usize,
        pool: &rayon::ThreadPool,
        check: &impl Fn() -> Result<()>,
    ) -> Result<()> {
        let bands = self.cog.inner.bands as usize;
        let th = self.plan.tile_height;
        for (level, frontier) in self.frontiers.iter_mut().enumerate() {
            let first_y = frontier.row * th;
            if first_y >= frontier.height {
                continue;
            }
            let end_y = ((frontier.row + 1) * th).min(frontier.height);
            // Average needs a full box of source rows, except at the bottom edge.
            let complete_rows = if self.cog.resampling == Resampling::Average
                && consumed_rows < self.cog.inner.height as usize
            {
                consumed_rows / frontier.factor
            } else {
                consumed_rows.div_ceil(frontier.factor)
            };
            if complete_rows < end_y {
                continue;
            }
            if self.cog.resampling == Resampling::Average {
                for tile_idx in 0..frontier.tiles.len() {
                    for i in 0..frontier.tiles[tile_idx].len() {
                        let count = frontier.counts[tile_idx][i];
                        frontier.tiles[tile_idx][i] = if count == 0 {
                            self.fill
                        } else {
                            let val = T::from_f64(frontier.sums[tile_idx][i] / count as f64);
                            if count > 0 && self.nodata_value.is_some_and(|nd| val == nd) {
                                return Err(Error::Other(
                                    "averaged valid value collides with NoData; choose a different sentinel".into(),
                                ));
                            }
                            val
                        };
                    }
                    frontier.sums[tile_idx].fill(0.0);
                    frontier.counts[tile_idx].fill(0);
                }
            }
            let image = &mut self.images[level + 1];
            for chunk in frontier.tiles.chunks_mut(pool.current_num_threads()) {
                check()?;
                let start = image.blocks.len();
                let results: Vec<_> = pool.install(|| {
                    chunk
                        .par_iter()
                        .enumerate()
                        .map(|(offset, samples)| {
                            compress(samples, start + offset, self.plan, bands)
                        })
                        .collect()
                });
                for (samples, result) in chunk.iter_mut().zip(results) {
                    check()?;
                    image
                        .blocks
                        .push(append_compressed(&mut self.spools[level + 1], result?)?);
                    samples.fill(self.fill);
                }
            }
            frontier.row += 1;
        }
        Ok(())
    }

    /// Finish metadata and copy smallest-overview-first compressed payloads.
    pub fn finish(
        mut self,
        pool: &rayon::ThreadPool,
        check: &impl Fn() -> Result<()>,
    ) -> Result<W> {
        check()?;
        if self.next_tile != self.across * self.down {
            return Err(Error::InvalidConfig("pyramid is missing base tiles".into()));
        }
        self.flush_frontiers(self.cog.inner.height as usize, pool, check)?;
        let mut virtual_offset = 0;
        for index in (1..self.images.len()).rev().chain(std::iter::once(0)) {
            for block in &mut self.images[index].blocks {
                if !block.sparse {
                    block.spool_offset = checked_add_u64(
                        block.spool_offset,
                        virtual_offset,
                        "pyramid spool offset",
                    )?;
                }
            }
            virtual_offset = checked_add_u64(
                virtual_offset,
                self.spools[index].len,
                "pyramid spool length",
            )?;
        }
        let prefix = gdal_structural_metadata_bytes(self.plan.planar_configuration);
        let layout = plan_cog_layout(
            self.sink.stream_position()?,
            checked_len_u64(prefix.len(), "COG prefix")?,
            self.cog.inner.tiff_variant,
            &self.images,
        )?;
        check()?;
        emit_cog_metadata(&mut self.sink, &prefix, &self.images, &layout)?;
        for index in (1..self.images.len()).rev().chain(std::iter::once(0)) {
            self.spools[index].copy_into(&mut self.sink, check)?;
        }
        Ok(self.sink)
    }
}
