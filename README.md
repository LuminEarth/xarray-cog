# xarray-cog

An xarray extension for saving `DataArray` and `Dataset` rasters as Cloud
Optimized GeoTIFFs (COGs), powered by Rust.

- **Built for large rasters on limited memory**: Streams chunked inputs (Zarr,
  NetCDF, Dask) tile-by-tile within a configurable `memory_limit`, so memory
  usage stays bounded instead of growing with the full raster size.
- **Fast and parallel**: Compresses tile batches in parallel in Rust—faster
  than GDAL-based exporters (`rioxarray`, `odc-geo`) in our
  [benchmarks](https://github.com/LuminEarth/xarray-cog/blob/main/docs/benchmarks/README.md).
- **No GDAL required**: Only depends on `numpy`, `xarray`, and `pyproj`; bring
  whichever xarray reader you already use.

## Install

Install with Python 3.10+ on Linux or macOS (Windows is not a supported release
target):

```sh
pip install xarray-cog
# or with uv:
uv add xarray-cog
```

To build from source instead, install Rust 1.91+ and a C/C++ toolchain and run
`pip install .`. See [Contributing](https://github.com/LuminEarth/xarray-cog/blob/main/CONTRIBUTING.md)
for development setup.

## Export an array or dataset

```python
import numpy as np
import xarray as xr
import xarray_cog  # registers .cog

array = xr.DataArray(
    np.arange(12, dtype='uint16').reshape(3, 4),
    dims=('y', 'x'),
    coords={'x': [10, 12, 14, 16], 'y': [50, 48, 46]},
    name='temperature',
)
array.cog.to_cog('temperature.tif', crs='EPSG:4326')

# Dataset variable order defines TIFF band order.
dataset = xr.Dataset({'temperature': array, 'adjusted': array + 2})
xarray_cog.write_cog(dataset, 'bands.tif', crs='EPSG:4326')
```

Coordinates must be regular 1D pixel centers, with any extra dimensions (such as
time or depth) selected before exporting. Passing `crs` sets output metadata
without reprojecting.

## Open lazy data with xarray

Install the reader separately (`h5netcdf` and `h5py` for this example), keep the
dataset open during export, and calculate values before writing:

```python
with xr.open_dataset('weather.nc', engine='h5netcdf', decode_coords='all') as source:
    with xr.set_options(keep_attrs=True):
        celsius = source['temperature'].isel(time=0, drop=True) - 273.15
    celsius.cog.to_cog('celsius.tif', crs='EPSG:4326')
```

See [input backends](https://github.com/LuminEarth/xarray-cog/blob/main/docs/python.md#input-backends)
for remote Zarr and CF metadata guidance.

## Defaults and docs

Writes local BigTIFF COGs with DEFLATE level 7, 512-pixel tiles, and
nearest-neighbour overviews. Existing destinations are refused unless
`overwrite=True`, and failed exports preserve any existing file. Tune memory usage with `memory_limit`, parallelism with `num_threads`, and
show progress with `progress=True`.

- [Python API and options](https://github.com/LuminEarth/xarray-cog/blob/main/docs/python.md)
- [Limitations](https://github.com/LuminEarth/xarray-cog/blob/main/docs/limitations.md) and [architecture](https://github.com/LuminEarth/xarray-cog/blob/main/docs/architecture.md)
- [Benchmarks](https://github.com/LuminEarth/xarray-cog/blob/main/docs/benchmarks/README.md)

MIT licensed. Adapted MIT-licensed code from `geotiff-writer` is listed in
[third-party notices](https://github.com/LuminEarth/xarray-cog/blob/main/THIRD_PARTY_NOTICES.md).
