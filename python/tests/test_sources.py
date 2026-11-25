"""Read real input formats with xarray; the exporter only sees their arrays."""
from importlib.metadata import version

import dask
import dask.array as dsa
from dask.callbacks import Callback
import numpy as np
from pyproj import CRS
import pytest
import rasterio
import xarray as xr

import xarray_cog


def dataset():
    values = (np.arange(2 * 260 * 262).reshape(2, 260, 262) % 100).astype('float32') / 4 + 100
    values[:, 1, 2] = np.nan
    coords = {'time': [0, 1], 'y': 50 - np.arange(260) * 2, 'x': 10 + np.arange(262) * 2}
    a = xr.DataArray(values, dims=('time', 'y', 'x'), coords=coords, attrs={'grid_mapping': 'spatial_ref'})
    ds = xr.Dataset({'a': a, 'b': (a + 25).assign_attrs(a.attrs)})
    return ds.assign_coords(spatial_ref=xr.DataArray(0, attrs=CRS.from_epsg(4326).to_cf()))


def check(path, expected):
    with rasterio.open(path) as result:
        np.testing.assert_array_equal(result.read(), expected)
        assert result.dtypes == (str(expected.dtype),) * 2
        assert result.crs.to_epsg() == 4326
        assert result.transform == rasterio.Affine(2, 0, 9, 0, -2, 51)
        assert result.descriptions == ('b', 'a')
        assert np.isnan(result.nodata)
        assert result.overviews(1) == [2, 4]
        assert result.tags(ns='IMAGE_STRUCTURE')['LAYOUT'] == 'COG'
    for index, factor in enumerate((2, 4)):
        with rasterio.open(path, overview_level=index) as overview:
            np.testing.assert_array_equal(overview.read(), expected[:, ::factor, ::factor])


@pytest.mark.parametrize('backend', ['numpy', 'dask', 'zarr2', 'zarr3', 'netcdf'])
def test_selected_calculated_sources(tmp_path, backend):
    if backend == 'zarr3' and int(version('zarr').split('.')[0]) < 3:
        pytest.skip('Zarr v3 requires Python 3.11+ with supported Zarr releases')
    original = dataset()
    if backend.startswith('zarr'):
        source = tmp_path / 'source.zarr'
        original.to_zarr(source, zarr_format=int(backend[-1]), consolidated=False)
        opened = xr.open_zarr(source, consolidated=False, chunks={}, decode_coords='all')
    elif backend == 'netcdf':
        source = tmp_path / 'source.nc'
        encoding = {name: dict(dtype='int16', scale_factor=0.25, add_offset=100,
                               _FillValue=-32768) for name in ('a', 'b')}
        original.to_netcdf(source, engine='h5netcdf', encoding=encoding)
        opened = xr.open_dataset(source, engine='h5netcdf', chunks={}, decode_coords='all')
    else:
        opened = original.chunk({'time': 1, 'y': 192, 'x': 192}) if backend == 'dask' else original
    with opened:
        with xr.set_options(keep_attrs=True), dask.config.set(scheduler='synchronous'):
            selected = opened[['b', 'a']].isel(time=1, drop=True) + 2
            path = selected.cog.to_cog(tmp_path / 'out.tif', tile_size=128)
            expected = np.stack([original[name].isel(time=1).values + 2 for name in ('b', 'a')])
            check(path, expected.astype(selected['a'].dtype))


@pytest.mark.parametrize('chunk', [192, 256])
@pytest.mark.parametrize('fail_late', [False, True])
def test_lazy_reads_stay_local_and_failure_is_atomic(tmp_path, chunk, fail_late):
    reads = []
    finished_computes = []
    @dask.delayed
    def read_block(row, col):
        reads.append((row, col))
        if fail_late and (row, col) == (1, 1):
            raise RuntimeError('late source failure')
        yy, xx = np.indices((chunk, chunk))
        return ((yy + row * chunk) * 3 + xx + col * chunk).astype('int32')
    data = dsa.block([[dsa.from_delayed(read_block(row, col), shape=(chunk, chunk), dtype='int32')
                       for col in range(2)] for row in range(2)])
    da = xr.DataArray(data, dims=('y', 'x'), coords={'x': np.arange(chunk * 2),
                      'y': -np.arange(chunk * 2)}, attrs={'crs': 'EPSG:4326'})
    path = tmp_path / 'out.tif'
    path.write_bytes(b'original')
    with dask.config.set(scheduler='synchronous'), Callback(
        finish=lambda dsk, state, failed: finished_computes.append(list(reads))
    ):
        if fail_late:
            with pytest.raises(RuntimeError, match='late source failure'):
                da.cog.to_cog(path, tile_size=128, overwrite=True, num_threads=1)
            assert path.read_bytes() == b'original'
            assert list(tmp_path.iterdir()) == [path]
        else:
            da.cog.to_cog(path, tile_size=128, overwrite=True, num_threads=1)
            yy, xx = np.indices((chunk * 2, chunk * 2))
            with rasterio.open(path) as result:
                np.testing.assert_array_equal(result.read(1), yy * 3 + xx)
    # The first tile computed only its intersecting source chunk, even when
    # source chunks are larger than / misaligned with TIFF tiles.
    assert finished_computes[0] == [(0, 0)]


def test_missing_crs_and_curvilinear_coordinates(tmp_path):
    da = xr.DataArray(np.zeros((3, 4), dtype='float32'), dims=('y', 'x'),
                      coords={'x': [0, 1, 2, 3], 'y': [0, 1, 2]})
    with pytest.raises(ValueError, match='missing CRS'):
        xarray_cog.write_cog(da, tmp_path / 'missing.tif')
    curved = da.assign_coords(x=(('y', 'x'), np.arange(12).reshape(3, 4)))
    with pytest.raises(ValueError, match='1D'):
        xarray_cog.write_cog(curved, tmp_path / 'curved.tif', crs='EPSG:4326')
    assert not list(tmp_path.iterdir())
