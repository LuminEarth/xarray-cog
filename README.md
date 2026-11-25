# xarray-cog

An xarray extension for saving arrays and datasets as Cloud Optimized GeoTIFFs
(COGs), powered by Rust.

## Install

The package is not published yet. Install this checkout with Python 3.10+,
Rust 1.91+ and a C/C++ build toolchain:

```sh
python -m pip install .
```

The intended release command is `pip install xarray-cog` once a release exists.
The base runtime contains NumPy, xarray and pyproj; input readers are optional.
Linux wheel builds are checked in CI. Windows is not a supported release
target yet. See [Contributing](CONTRIBUTING.md) for development setup.

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

Coordinates must be regular one-dimensional pixel centers. Select time, depth and
other nonspatial dimensions before exporting. An explicit CRS assigns metadata;
it does not reproject. Calculated and CF-decoded values are written as they are.

## Open data with xarray

Install the reader separately (`h5netcdf` and `h5py` for this example). Keep the
dataset open until export returns. Xarray decodes packed values and fill values;
you can calculate new values before exporting.

```python
with xr.open_dataset('weather.nc', engine='h5netcdf', decode_coords='all') as source:
    with xr.set_options(keep_attrs=True):
        celsius = source['temperature'].isel(time=0, drop=True) - 273.15
    celsius.cog.to_cog('celsius.tif', crs='EPSG:4326')
```

Use the same approach with other xarray readers. See [input backends](docs/python.md#input-backends)
for remote Zarr and CF metadata guidance.

Defaults are DEFLATE **7**, 512-pixel tiles, nearest-neighbour overviews and local
BigTIFF output. Existing files are refused unless `overwrite=True`; failed reads
or writes preserve the destination. An export-local cache reuses decoded source
chunks across output tiles and bands. `memory_limit=512` sets the default 512 MiB budget
covering the writer's allocation
estimate and retained cache entries, not total process memory. Oversized or evicted
chunks may be read again; reader and calculation working memory is additional.
Rust compresses bounded batches in parallel; `num_threads` sets a worker ceiling,
with automatic selection by default. Pass `progress=True` to show a tile progress
bar and finalization status on stderr.

- [Python API and options](docs/python.md)
- [Limitations](docs/limitations.md) and [architecture](docs/architecture.md)
- [Benchmarks](docs/benchmarks/README.md)

MIT licensed. The internal writer includes adapted MIT-licensed code from
geotiff-writer; see [third-party notices](THIRD_PARTY_NOTICES.md).
