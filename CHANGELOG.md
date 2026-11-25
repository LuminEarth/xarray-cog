# Changelog

## Unreleased

- Export regular xarray DataArray and Dataset rasters through a Rust COG writer.
- Ship Python type annotations and a type stub for the native writer.
- Support NumPy and lazy Dask inputs, with reader backends installed by callers.
- Write local BigTIFF with DEFLATE (default level 7) or no compression,
  nearest or average overviews, CRS, NoData and band descriptions.
- Reuse decoded chunks across output tiles and bands with a bounded per-export cache.
- Coordinate bounded tile batches in Python with memory-aware parallel Rust compression.
- Optionally display tile progress and finalization status with `progress=True`.
- Validate raster grids and writer/cache memory estimates; install outputs atomically.
- Test source formats and output correctness independently, including Zarr and NetCDF.
