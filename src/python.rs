//! Optional Python adapter for the source-independent tile writer.
use crate::cog::{
    CogBuilder, CogPyramidWriter, Compression, Error, ExtraSample, GeoTiffBuilder, NumericSample,
    PhotometricInterpretation, Resampling, TiffVariant,
};
use numpy::{Element, PyReadonlyArray3};
use pyo3::types::PyList;
use pyo3::{exceptions::PyValueError, prelude::*};
use serde::Deserialize;
use std::{fs::File, path::PathBuf};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Config {
    width: u32,
    height: u32,
    bands: u32,
    dtype: String,
    epsg: u16,
    geographic: bool,
    origin: [f64; 2],
    scale: [f64; 2],
    nodata: Option<String>,
    tile_size: u32,
    compression: String,
    deflate_level: u32,
    resampling: String,
    overviews: Vec<u32>,
    overwrite: bool,
    memory_limit: u64,
    colorinterp: String,
    metadata: String,
    #[serde(default)]
    num_threads: Option<usize>,
}

fn invalid(e: impl std::fmt::Display) -> PyErr {
    PyValueError::new_err(e.to_string())
}

impl From<Error> for PyErr {
    fn from(error: Error) -> Self {
        match error {
            Error::Io(io) | Error::Tiff(tiff_writer::Error::Io(io)) => io.into(),
            other => invalid(other),
        }
    }
}

// The typed sink keeps NumPy validation and Rust buffer ownership together.
trait TileSink: Send + Sync {
    fn write_tiles(
        &mut self,
        py: Python<'_>,
        tiles: &Bound<'_, PyList>,
        c: &Config,
        start: usize,
        pool: &rayon::ThreadPool,
    ) -> PyResult<()>;
    fn finish(self: Box<Self>, pool: &rayon::ThreadPool) -> PyResult<()>;
}

impl<T: NumericSample + Element> TileSink for CogPyramidWriter<T, File> {
    fn write_tiles(
        &mut self,
        py: Python<'_>,
        tiles: &Bound<'_, PyList>,
        c: &Config,
        start: usize,
        pool: &rayon::ThreadPool,
    ) -> PyResult<()> {
        let across = (c.width as usize).div_ceil(c.tile_size as usize);
        let mut batch = Vec::with_capacity(tiles.len());
        for (offset, tile) in tiles.iter().enumerate() {
            py.check_signals()?;
            let array: PyReadonlyArray3<'_, T> = tile.extract()?;
            let view = array.as_array();
            let row = (start + offset) / across;
            let col = (start + offset) % across;
            let expected = (
                (c.tile_size as usize).min(c.height as usize - row * c.tile_size as usize),
                (c.tile_size as usize).min(c.width as usize - col * c.tile_size as usize),
                c.bands as usize,
            );
            if view.dim() != expected {
                return Err(invalid("unexpected raster tile shape"));
            }
            // Python may retain or mutate its arrays after this call. Native
            // workers only touch these owned copies while the GIL is released.
            batch.push((col, row, view.to_owned()));
        }
        py.detach(|| self.write_tiles(&batch, pool, &|| Ok(())))?;
        py.check_signals()
    }

    fn finish(self: Box<Self>, pool: &rayon::ThreadPool) -> PyResult<()> {
        (*self).finish(pool, &|| Ok(()))?;
        Ok(())
    }
}

/// Private context-managed output transaction. Dropping or closing it aborts.
#[pyclass(module = "xarray_cog._native")]
struct NativeWriter {
    sink: Option<Box<dyn TileSink>>,
    temp: Option<tempfile::NamedTempFile>,
    pool: Option<rayon::ThreadPool>,
    path: PathBuf,
    config: Config,
    completed: usize,
    count: usize,
    #[pyo3(get)]
    num_threads: usize,
    #[pyo3(get)]
    batch_size: usize,
    #[pyo3(get)]
    source_cache_budget: u64,
}

#[pymethods]
impl NativeWriter {
    #[new]
    fn new(py: Python<'_>, path: PathBuf, config: &str) -> PyResult<Self> {
        let c: Config = serde_json::from_str(config).map_err(invalid)?;
        if c.num_threads == Some(0) {
            return Err(invalid("num_threads must be positive"));
        }
        if c.width == 0
            || c.height == 0
            || c.bands == 0
            || c.bands > u16::MAX as u32
            || ![128, 256, 512, 1024, 2048].contains(&c.tile_size)
            || !c.origin.iter().all(|v| v.is_finite())
            || !c.scale.iter().all(|v| v.is_finite() && *v > 0.0)
            || c.epsg == 0
            || c.epsg == 32767
        {
            return Err(invalid(
                "invalid raster dimensions, grid, tile size or EPSG code",
            ));
        }
        let compression = match c.compression.as_str() {
            "deflate" => Compression::Deflate,
            "none" => Compression::None,
            _ => return Err(invalid("compression must be deflate or none")),
        };
        if !(1..=9).contains(&c.deflate_level) {
            return Err(invalid("deflate_level must be an integer from 1 to 9"));
        }
        let resampling = match c.resampling.as_str() {
            "nearest" => Resampling::NearestNeighbor,
            "average" => Resampling::Average,
            _ => return Err(invalid("resampling must be nearest or average")),
        };
        let rgb = match c.colorinterp.as_str() {
            "undefined" => false,
            "rgb" if c.bands == 3 => true,
            _ => {
                return Err(invalid(
                    "colorinterp must be undefined, or rgb with three bands",
                ))
            }
        };
        if c.overviews.iter().any(|v| *v < 2 || !v.is_power_of_two())
            || c.overviews.windows(2).any(|v| v[0] >= v[1])
        {
            return Err(invalid("overviews must be increasing powers of two"));
        }
        let mut builder = GeoTiffBuilder::new(c.width, c.height)
            .bands(c.bands)
            .tile_size(c.tile_size, c.tile_size)
            .compression(compression)
            .tiff_variant(TiffVariant::BigTiff)
            .pixel_scale(c.scale[0], c.scale[1])
            .origin(c.origin[0], c.origin[1])
            .photometric(if rgb {
                PhotometricInterpretation::Rgb
            } else {
                PhotometricInterpretation::MinIsBlack
            })
            .extra_samples(vec![
                ExtraSample::Unspecified;
                c.bands as usize - if rgb { 3 } else { 1 }
            ])
            .gdal_metadata(&c.metadata);
        if c.compression == "deflate" {
            builder = builder.deflate_level(c.deflate_level);
        }
        builder = if c.geographic {
            builder.geographic_epsg(c.epsg)
        } else {
            builder.projected_epsg(c.epsg)
        };
        if let Some(nodata) = &c.nodata {
            builder = builder.nodata(nodata);
        }
        let cog = CogBuilder::new(builder)
            .overview_levels(c.overviews.clone())
            .resampling(resampling);
        match c.dtype.as_str() {
            "uint8" => create::<u8>(py, path, c, cog),
            "uint16" => create::<u16>(py, path, c, cog),
            "int16" => create::<i16>(py, path, c, cog),
            "uint32" => create::<u32>(py, path, c, cog),
            "int32" => create::<i32>(py, path, c, cog),
            "float32" => create::<f32>(py, path, c, cog),
            "float64" => create::<f64>(py, path, c, cog),
            _ => Err(invalid("unsupported dtype")),
        }
    }

    fn __enter__(slf: PyRef<'_, Self>) -> PyResult<PyRef<'_, Self>> {
        if slf.sink.is_none() {
            return Err(invalid("writer is closed"));
        }
        Ok(slf)
    }

    fn __exit__(
        &mut self,
        _ty: &Bound<'_, PyAny>,
        _value: &Bound<'_, PyAny>,
        _traceback: &Bound<'_, PyAny>,
    ) {
        self.close();
    }

    fn close(&mut self) {
        self.sink.take();
        self.temp.take();
        self.pool.take();
    }

    fn write_tiles(&mut self, py: Python<'_>, tiles: &Bound<'_, PyList>) -> PyResult<()> {
        let result = self.write_batch(py, tiles);
        if result.is_err() {
            self.close();
        }
        result
    }

    fn finish(&mut self, py: Python<'_>) -> PyResult<()> {
        let result = self.finish_output(py);
        self.close();
        result
    }
}

impl NativeWriter {
    fn write_batch(&mut self, py: Python<'_>, tiles: &Bound<'_, PyList>) -> PyResult<()> {
        let sink = self
            .sink
            .as_mut()
            .ok_or_else(|| invalid("writer is closed"))?;
        let batch_len = tiles.len();
        if batch_len > self.count - self.completed {
            return Err(invalid("too many raster tiles"));
        }
        if tiles.is_empty() || tiles.len() > self.batch_size {
            return Err(invalid("tile batch must contain 1..batch_size tiles"));
        }
        sink.write_tiles(
            py,
            tiles,
            &self.config,
            self.completed,
            self.pool
                .as_ref()
                .ok_or_else(|| invalid("writer is closed"))?,
        )?;
        self.completed += batch_len;
        Ok(())
    }

    fn finish_output(&mut self, py: Python<'_>) -> PyResult<()> {
        let sink = self
            .sink
            .take()
            .ok_or_else(|| invalid("writer is closed"))?;
        if self.completed != self.count {
            return Err(invalid("missing raster tiles"));
        }
        py.check_signals()?;
        let pool = self
            .pool
            .as_ref()
            .ok_or_else(|| invalid("writer is closed"))?;
        py.detach(|| sink.finish(pool))?;
        py.check_signals()?;
        let temp = self
            .temp
            .take()
            .ok_or_else(|| invalid("writer is closed"))?;
        temp.as_file().sync_all()?;
        if self.config.overwrite {
            temp.persist(&self.path).map_err(|e| e.error)?;
        } else {
            temp.persist_noclobber(&self.path).map_err(|e| e.error)?;
        }
        Ok(())
    }
}

fn create<T: NumericSample + Element>(
    py: Python<'_>,
    path: PathBuf,
    c: Config,
    cog: CogBuilder,
) -> PyResult<NativeWriter> {
    let frontier = cog.pyramid_memory_bytes::<T>()?;
    let tile_bytes =
        u64::from(c.tile_size).pow(2) * u64::from(c.bands) * T::BYTES_PER_SAMPLE as u64;
    let across = (c.width as usize).div_ceil(c.tile_size as usize);
    let count = across
        .checked_mul((c.height as usize).div_ceil(c.tile_size as usize))
        .ok_or_else(|| invalid("tile count overflow"))?;
    let indexes = std::iter::once(1)
        .chain(c.overviews.iter().copied())
        .try_fold(0u64, |total, factor| {
            let x = u64::from(c.width)
                .div_ceil(u64::from(factor))
                .div_ceil(u64::from(c.tile_size));
            let y = u64::from(c.height)
                .div_ceil(u64::from(factor))
                .div_ceil(u64::from(c.tile_size));
            total.checked_add(x.checked_mul(y)?.checked_mul(256)?)
        })
        .ok_or_else(|| invalid("tile index memory estimate overflow"))?;
    let fixed = frontier
        .checked_add(indexes)
        .and_then(|v| v.checked_add(16 << 20))
        .ok_or_else(|| invalid("memory estimate overflow"))?;
    // Python batch, Rust copies, edge padding, encoded samples, compressed Vec
    // capacity (up to two buffers), plus codec workspace per worker. Reader/task
    // working allocations remain outside this estimate.
    let per_worker = tile_bytes
        .checked_mul(6)
        .and_then(|v| v.checked_add(1 << 20))
        .ok_or_else(|| invalid("memory estimate overflow"))?;
    let available = c.memory_limit.saturating_sub(fixed);
    if available < per_worker {
        return Err(invalid(format!(
            "writer memory estimate exceeds memory_limit of {} MiB",
            c.memory_limit as f64 / (1024.0 * 1024.0)
        )));
    }
    let requested = c
        .num_threads
        .unwrap_or_else(|| std::thread::available_parallelism().map_or(1, usize::from));
    // Auto selection leaves at least half the variable budget for source reuse,
    // unless even one worker needs more. Explicit ceilings use the full budget.
    let worker_budget = if c.num_threads.is_none() {
        (available / 2).max(per_worker)
    } else {
        available
    };
    let workers = requested
        .min(across)
        .min((worker_budget / per_worker) as usize)
        .max(1);
    let source_cache_budget = available - per_worker * workers as u64;
    if path.exists() && !c.overwrite {
        return Err(pyo3::exceptions::PyFileExistsError::new_err(
            path.display().to_string(),
        ));
    }
    py.check_signals()?;
    let parent = path
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(std::path::Path::new("."));
    let temp = tempfile::NamedTempFile::new_in(parent)?;
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(workers)
        .build()
        .map_err(invalid)?;
    // A cloned file handle avoids borrowing the temporary file from its owner.
    let sink = cog.pyramid_tile_writer::<T, _>(temp.as_file().try_clone()?, frontier)?;
    Ok(NativeWriter {
        sink: Some(Box::new(sink)),
        temp: Some(temp),
        pool: Some(pool),
        path,
        config: c,
        completed: 0,
        count,
        num_threads: workers,
        batch_size: workers,
        source_cache_budget,
    })
}

#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<NativeWriter>()?;
    Ok(())
}
