"""Check the public package from a fresh interpreter, not an existing accessor."""
import subprocess
import sys


def test_fresh_process_exports(tmp_path):
    script = '''
from pathlib import Path
import sys
import numpy as np
import rasterio
import xarray as xr
import xarray_cog

assert xarray_cog.__all__ == ['write_cog']
assert not hasattr(xarray_cog, 'to_cog')

da = xr.DataArray(np.arange(12, dtype='uint16').reshape(3, 4), dims=('y', 'x'),
                  coords={'x': [10, 12, 14, 16], 'y': [50, 48, 46]}, attrs={'crs': 'EPSG:4326'})
for name, export in [('function', lambda p: xarray_cog.write_cog(da, p)),
                     ('accessor', da.cog.to_cog)]:
    path = export(Path(sys.argv[1]) / f'{name}.tif')
    with rasterio.open(path) as result:
        np.testing.assert_array_equal(result.read(1), da.values)
        assert result.crs.to_epsg() == 4326
        assert result.transform == rasterio.Affine(2, 0, 9, 0, -2, 51)
'''
    result = subprocess.run([sys.executable, '-c', script, str(tmp_path)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
