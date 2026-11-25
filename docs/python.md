# Python and xarray export

The Python package writes COGs directly through the Rust tile writer. It does not require GDAL or Rasterio at runtime and does not stage a Zarr store. CRS interpretation uses pyproj/PROJ.

See the [README](../README.md) for installation and a quick start, and
[Contributing](../CONTRIBUTING.md) for development setup.

## Export

```python
import xarray as xr
import xarray_cog  # registers .cog on DataArray and Dataset

# Select nonspatial dimensions yourself; calculations are preserved.
with xr.open_zarr('climate.zarr', decode_coords='all') as source:
    da = source['temperature'].isel(time=0, drop=True)
    da.cog.to_cog('temperature.tif', crs='EPSG:4326')

# Dataset variable order becomes TIFF band order.
with xr.open_zarr('image.zarr', decode_coords='all') as source:
    ds = source[['red', 'green', 'blue']].isel(time=0, drop=True)
    xarray_cog.write_cog(ds, 'rgb.tif', crs='EPSG:32632', colorinterp='rgb')

# Explicitly identify a DataArray's band dimension if not named "band".
cube.cog.to_cog('bands.tif', band_dim='channel')
```

Both entry points return the destination `pathlib.Path`. Output is local BigTIFF. Existing destinations raise `FileExistsError` unless `overwrite=True`; replacement happens only after successful finalization and synchronization. Python exceptions during reading leave the destination unchanged and release temporary files. SIGINT is checked while copying tiles, between batches and around finalization; a running native operation can delay interruption.

## Input rules

- `DataArray`: two spatial dimensions, optionally one explicitly named band dimension. `band` is detected automatically. Select other dimensions, including singleton time/level dimensions, with `.isel(..., drop=True)` first.
- `Dataset`: each data variable is a two-dimensional band. Select only raster variables; mark CF grid-mapping variables as coordinates with `set_coords` where needed. Bands share spatial dimensions, coordinates, dtype and CRS.
- Spatial dimensions default to `x`/`y`, `lon`/`lat` or `longitude`/`latitude`. Use `x_dim` and `y_dim` for other names. Coordinates must be finite, strictly monotonic, regularly spaced 1D pixel centers with at least two values per axis. Export normalizes to eastward columns and southward rows and derives outer bounds by half a pixel.
- Supported dtypes: uint8, uint16, int16, uint32, int32, float32 and float64, native byte order. No implicit mixed-band promotion or lossy casting. Use xarray `.astype(...)` explicitly when desired.
- CRS is read from `crs`, `epsg` or `epsg_code` attributes (including Dataset attributes), or a CF `grid_mapping` coordinate / `spatial_ref` coordinate. Conflicting CRS metadata fails. Explicit `crs` overrides metadata without reprojection. Only horizontal two-axis geographic/projected CRSs identifiable by a GeoTIFF-representable EPSG code are accepted.
- Current values are exported. `encoding` scale/offset and stored dtype are not reapplied. Raw `scale_factor`/`add_offset` attributes fail: decode with `xr.decode_cf` first.
- NaNs and scalar `_FillValue` / `missing_value` attributes indicate missing data. Stored fill values in `encoding` are not applied to already-decoded values. Floating output defaults to NaN NoData; integer output uses a shared source sentinel when present. Specify `nodata` to remap missing samples. The output sentinel must be exactly representable and must not collide with valid pixels. Infinities are preserved as sample values; infinite NoData is unsupported.
- Band names become TIFF descriptions. Other xarray metadata and nonspatial coordinates are not serialized. A COG is not a lossless container for an arbitrary Dataset.

## Arguments

`xarray_cog.write_cog(obj, path, **kwargs)` accepts the arguments below. With
`obj.cog.to_cog(path, **kwargs)`, the accessor supplies `obj` automatically.
All arguments after `path` are keyword-only.
The package ships type annotations and a stub for its compiled extension, so
editors and type checkers can inspect the function and accessor signatures.

| Argument | Default | Meaning and supported values |
|---|---|---|
| `obj` | Required | An xarray `DataArray` or `Dataset`; Dataset variables become bands in variable order |
| `path` | Required | Local destination path as a string or path-like object |
| `x_dim` | `None` | Horizontal dimension name; inferred from `x`, `lon`, or `longitude`. Specify it if ambiguous |
| `y_dim` | `None` | Vertical dimension name; inferred from `y`, `lat`, or `latitude`. Specify it if ambiguous |
| `band_dim` | `None` | DataArray band dimension; uses `band` if present, otherwise treats the array as one band. Must remain `None` for a Dataset |
| `crs` | `None` | CRS accepted by `pyproj.CRS.from_user_input`, such as `"EPSG:4326"`; inferred from metadata when omitted. An explicit value overrides metadata without reprojection |
| `nodata` | `None` | Output missing-value sentinel; inferred as NaN for floating data or a shared source sentinel for integer data when present. Must fit the dtype exactly and not collide with valid samples |
| `tile_size` | 512 | 128, 256, 512, 1024, 2048 |
| `compression` | `"deflate"` | `"deflate"`, `"none"` |
| `deflate_level` | 7 | Integer 1–9, ignored for uncompressed output |
| `resampling` | `"nearest"` | Overview resampling: `"nearest"` or `"average"`; does not resample base pixels |
| `overviews` | `"auto"` | `"auto"` adds powers-of-two reductions until the largest dimension fits within one tile; `"none"` disables overviews; or supply increasing powers of two, e.g. `[2, 4, 8]` |
| `colorinterp` | `"undefined"` | `"undefined"`, `"rgb"` (exactly three bands) |
| `memory_limit` | 512 | Memory budget in MiB (1 MiB = 1,048,576 bytes) for estimated writer buffers and cached input chunks. Accepts positive integers or floats. Total memory used by Python can exceed this budget because input arrays and temporary calculations also use memory |
| `overwrite` | `False` | Whether to replace an existing destination; replacement occurs only on success |
| `progress` | `False` | Boolean; show a text progress bar and finalization status on stderr when `True` |
| `num_threads` | `None` | Positive integer ceiling for Rust compression workers, or `None` for automatic selection from available CPUs and memory. Also limited by the number of tile columns. Does not configure Dask |

Python reads bounded batches in row-major order. The native writer copies each
batch into Rust-owned memory and releases the GIL while compressing tiles in
parallel and generating overviews. Python reads the next batch after the current
batch finishes; reading and compression do not overlap. Progress advances only
after a batch completes.

`num_threads=1` selects serial compression. A larger value is a ceiling: Rust
reduces the worker count if the memory budget or number of tile columns cannot
support it. With `num_threads=None`, automatic selection also limits workers to
available CPUs and leaves at least half the memory remaining after fixed writer
buffers for source caching, unless one worker requires more. Explicit ceilings
can use more of that memory for workers, leaving less for cached chunks. The
batch size equals the selected worker count; batches at row ends can be smaller.

Dask uses its separately configured local scheduler to compute missing input
chunks for a batch together. Dataset variables are computed separately. Native
worker settings do not set Dask's worker count.

Use `array.cog.to_cog('output.tif', progress=True)` to display completed output
tiles, followed by `Finalizing` while the file layout is completed and installed.
`Done` appears only after successful installation; an interrupted or failed export
shows `Stopped` if processing had started. The bar measures tile count, not elapsed
time or source chunks. Display updates are throttled, overwrite one line in a
terminal, and use separate lines when stderr is redirected. No extra dependency
is required, and exports are silent by default.

### Source chunk reuse and memory

Each export has one bounded LRU cache shared by all bands. The writer reserves
its estimated overview buffers, indexes and per-worker batch/compression workspace
first; the cache uses the remaining budget. Python converts `memory_limit` from
MiB to bytes for this accounting. Retained arrays own their memory and include a
per-entry bookkeeping allowance. The cache is cleared on success, failure or interruption.
Object-valued chunks are not retained because their Python payloads cannot be
accounted for by NumPy's buffer size.

For Dask inputs, the cache retains only the final NumPy results of the input
array's declared chunks (`array.chunks`). A chunk is computed when a tile first
needs it and reused for overlapping tiles and bands while it remains cached.
Dask manages graph optimization and can share dependencies within a batch;
intermediate task results are not retained between batches or separate Dataset
variables. Synchronous, threaded and other local Dask scheduling use the caller's
configuration; distributed scheduling is unsupported.

For best performance, preserve source-aligned chunks where practical and provide
enough cache memory to reuse chunks across output tiles. Rechunking or operations
that share upstream calculations across final chunks may repeat work. Complex
calculations are supported; calculations with independent chunks are a good fit.

For non-Dask inputs, the exporter uses xarray's `preferred_chunks` encoding
metadata when present, including chunks spanning multiple bands. Otherwise it
reads output-sized slices directly. Input stores are still opened by the caller.

The cache is bounded, not a guarantee that every chunk is read exactly once.
Entries larger than its capacity are not retained, and evicted entries may be
computed again. Increasing `memory_limit` leaves more room for reuse when the
writer estimate fits. Resident inputs, reader caches, Dask graphs, task working
arrays and temporary computations are outside this estimate; it is not a hard
limit on total process memory. Compressed blocks require temporary disk space. Benchmark the actual
input chunking and calculations for throughput and peak memory.

Reprojection, curvilinear/rotated grids, one-pixel axes, implicit alignment, generated masks/alpha, and distributed writing are outside this initial API.

## Input backends

Install the chosen reader separately. For `xr.open_zarr`, install Zarr, Dask and
any filesystem dependencies. Configure storage options through xarray:

```python
import xarray as xr
import xarray_cog

with xr.open_zarr('s3://bucket/input.zarr', chunks='auto',
                  storage_options={'anon': True}, decode_coords='all') as source:
    source['temperature'].isel(time=0, drop=True).cog.to_cog('out.tif', crs='EPSG:4326')
```

This example assumes an anonymously accessible store and an installed S3
filesystem backend such as `s3fs`.

Use `decode_coords='all'` to expose CF grid-mapping variables as coordinates.
If a mapping variable was not decoded as a coordinate, use
`source.set_coords('spatial_ref')` before selecting raster variables.
CF decoding happens in xarray; the exporter writes current values without
reapplying storage scale/offset. Keep file-backed datasets open until export
returns. Choose source chunks when lazy computation is useful; the exporter
does not rechunk the full input.
