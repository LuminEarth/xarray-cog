"""Run in a fresh environment containing only the installed base wheel/sdist."""
import importlib.util
from pathlib import Path
import tempfile

import numpy as np
import xarray as xr
import xarray_cog

assert 'site-packages' in xarray_cog.__file__, xarray_cog.__file__
for name in ('zarr', 'dask', 'rasterio', 'osgeo'):
    assert importlib.util.find_spec(name) is None, name
with tempfile.TemporaryDirectory() as directory:
    data = xr.DataArray(np.arange(12, dtype='uint16').reshape(3, 4), dims=('y', 'x'),
                        coords={'x': [10, 12, 14, 16], 'y': [50, 48, 46]})
    for name, export in [('function', lambda path: xarray_cog.write_cog(data, path, crs='EPSG:4326')),
                         ('accessor', lambda path: data.cog.to_cog(path, crs='EPSG:4326'))]:
        output = export(Path(directory) / f'{name}.tif')
        assert output.read_bytes()[:4] in (b'II+\x00', b'MM\x00+')  # BigTIFF
print(f'Base-only installed package exported successfully: {xarray_cog.__file__}')
