"""Round trips use Rasterio/GDAL only as an independent test decoder."""
import importlib.util
import json

import numpy as np
import pytest
import rasterio
import xarray as xr


def array(dtype='float32', size=259):
    values = (np.arange(size * (size + 2)).reshape(size, size + 2) % 100).astype(dtype)
    return xr.DataArray(values, dims=('y', 'x'),
                        coords={'x': 10 + np.arange(size + 2) * 2,
                                'y': 50 - np.arange(size) * 2},
                        attrs={'crs': 'EPSG:4326'}, name='temperature')


def export(obj, path, **kwargs):
    import xarray_cog
    return xarray_cog.write_cog(obj, path, **kwargs)


def test_package_exists():
    assert importlib.util.find_spec('xarray_cog') is not None, 'Python extension is missing'


@pytest.mark.parametrize('dtype', ['uint8', 'uint16', 'int16', 'uint32', 'int32', 'float32', 'float64'])
def test_pixels_grid_overviews_and_dtype(tmp_path, dtype):
    da = array(dtype)
    path = export(da, tmp_path / 'out.tif', tile_size=128)
    with rasterio.open(path) as result:
        np.testing.assert_array_equal(result.read(1), da.values)
        assert result.crs.to_epsg() == 4326
        assert result.transform == rasterio.Affine(2, 0, 9, 0, -2, 51)
        assert result.dtypes == (dtype,)
        assert result.overviews(1) == [2, 4]
        assert result.tags(ns='IMAGE_STRUCTURE')['LAYOUT'] == 'COG'


def test_accessor_calculations_and_axis_reversal(tmp_path):
    import xarray_cog  # noqa: F401
    da = array(size=8)
    changed = (da + 7).assign_attrs(da.attrs).isel(y=slice(None, None, -1), x=slice(None, None, -1))
    changed.cog.to_cog(tmp_path / 'out.tif')
    with rasterio.open(tmp_path / 'out.tif') as result:
        np.testing.assert_array_equal(result.read(1), da.values + 7)


def test_dataset_order_and_dataarray_bands(tmp_path):
    import xarray_cog  # noqa: F401
    da = array(size=8)
    ds = xr.Dataset({'b': da + 2, 'a': da}, attrs=da.attrs)
    ds.cog.to_cog(tmp_path / 'ds.tif')
    cube = xr.concat([da + 2, da], dim='channel').assign_attrs(da.attrs)
    export(cube.transpose('x', 'channel', 'y'), tmp_path / 'cube.tif', band_dim='channel')
    with rasterio.open(tmp_path / 'ds.tif') as result, rasterio.open(tmp_path / 'cube.tif') as other:
        np.testing.assert_array_equal(result.read(), other.read())
        np.testing.assert_array_equal(result.read(1), da.values + 2)
        assert result.descriptions == ('b', 'a')


def test_nan_and_sentinel_average(tmp_path):
    da = array(size=260)
    da.values[:2, :2] = [[1, -999], [3, np.nan]]
    da.attrs['_FillValue'] = -999
    export(da, tmp_path / 'out.tif', resampling='average', tile_size=128, nodata=-999)
    with rasterio.open(tmp_path / 'out.tif') as result:
        assert result.nodata == -999
        assert result.read(1)[1, 1] == -999
    with rasterio.open(tmp_path / 'out.tif', overview_level=0) as result:
        assert result.read(1)[0, 0] == 2


def test_decoded_encoding_not_reapplied(tmp_path):
    da = array(size=8)
    da.encoding.update(scale_factor=0.1, add_offset=100, _FillValue=-32768, dtype='int16')
    export(da, tmp_path / 'out.tif')
    with rasterio.open(tmp_path / 'out.tif') as result:
        np.testing.assert_array_equal(result.read(1), da.values)
        assert result.dtypes == ('float32',)


@pytest.mark.parametrize('mutation, message', [
    (lambda da: da.assign_coords(x=np.arange(da.sizes['x']) ** 2), 'regular'),
    (lambda da: da.expand_dims(time=[0, 1]), 'dimension'),
    (lambda da: da.astype('int64'), 'dtype'),
    (lambda da: da.assign_attrs(crs='EPSG:4979'), '2D'),
    (lambda da: da.assign_attrs(scale_factor=2), 'decode'),
])
def test_rejects_unsupported_inputs(tmp_path, mutation, message):
    with pytest.raises((ValueError, TypeError), match=message):
        export(mutation(array(size=8)), tmp_path / 'out.tif')
    assert not list(tmp_path.iterdir())


def test_mixed_dataset_dtype_and_crs(tmp_path):
    da = array(size=8)
    ds = xr.Dataset({'a': da, 'b': da.astype('uint8')})
    with pytest.raises(ValueError, match='dtype'):
        export(ds, tmp_path / 'out.tif')
    ds = xr.Dataset({'a': da, 'b': da.assign_attrs(crs='EPSG:3857')})
    with pytest.raises(ValueError, match='CRS'):
        export(ds, tmp_path / 'out.tif')


def test_no_clobber_and_overwrite(tmp_path):
    path = tmp_path / 'out.tif'
    path.write_bytes(b'keep me')
    with pytest.raises(FileExistsError):
        export(array(size=8), path)
    assert path.read_bytes() == b'keep me'
    export(array(size=8), path, overwrite=True)
    with rasterio.open(path) as result:
        assert result.count == 1


def test_unavailable_spool_directory_raises_oserror(tmp_path, monkeypatch):
    path = tmp_path / 'out.tif'
    path.write_bytes(b'original')
    monkeypatch.setenv('TMPDIR', str(tmp_path / 'missing'))
    with pytest.raises(FileNotFoundError):
        export(array(size=8), path, overwrite=True)
    assert path.read_bytes() == b'original'
    assert list(tmp_path.iterdir()) == [path]


def test_lazy_current_values(tmp_path):
    da = array().chunk({'x': 128, 'y': 128})
    da = (da * 2).assign_attrs(crs='EPSG:4326')
    export(da, tmp_path / 'out.tif', tile_size=128)
    with rasterio.open(tmp_path / 'out.tif') as result:
        np.testing.assert_array_equal(result.read(1), da.compute().values)


def test_lazy_failure_preserves_destination(tmp_path):
    import dask.array as dsa
    from dask import delayed
    @delayed
    def fail():
        raise RuntimeError('source failed')
    da = array(size=8).copy(data=dsa.from_delayed(fail(), shape=(8, 10), dtype='float32'))
    path = tmp_path / 'out.tif'
    path.write_bytes(b'keep me')
    with pytest.raises(RuntimeError, match='source failed'):
        export(da, path, overwrite=True)
    assert path.read_bytes() == b'keep me'
    assert list(tmp_path.iterdir()) == [path]


def test_memory_budget_rejects_before_writing(tmp_path):
    with pytest.raises(ValueError, match='memory_limit of 1 MiB'):
        export(array(), tmp_path / 'out.tif', memory_limit=1)
    assert not list(tmp_path.iterdir())


def test_cf_grid_mapping(tmp_path):
    from pyproj import CRS
    da = array(size=8)
    da.attrs = {'grid_mapping': 'spatial_ref'}
    da = da.assign_coords(spatial_ref=xr.DataArray(0, attrs=CRS.from_epsg(32632).to_cf()))
    export(da, tmp_path / 'out.tif')
    with rasterio.open(tmp_path / 'out.tif') as result:
        assert result.crs.to_epsg() == 32632


def test_rgb_and_custom_dimensions(tmp_path):
    da = array('uint8', size=8).rename({'x': 'easting', 'y': 'northing'})
    ds = xr.Dataset({'red': da, 'green': da + 1, 'blue': da + 2}, attrs=da.attrs)
    export(ds, tmp_path / 'out.tif', x_dim='easting', y_dim='northing',
           crs='EPSG:32632', colorinterp='rgb', compression='none', overviews='none')
    with rasterio.open(tmp_path / 'out.tif') as result:
        assert [c.name for c in result.colorinterp] == ['red', 'green', 'blue']
        assert result.overviews(1) == []
        assert result.crs.to_epsg() == 32632


def test_nodata_collision_preserves_destination(tmp_path):
    path = tmp_path / 'out.tif'
    path.write_bytes(b'keep me')
    with pytest.raises(ValueError, match='collide'):
        export(array('uint8', size=8), path, nodata=5, overwrite=True)
    assert path.read_bytes() == b'keep me'
    assert list(tmp_path.iterdir()) == [path]


@pytest.mark.parametrize('nodata', [-1, 256, 1.5, np.nan, np.inf])
def test_unrepresentable_integer_nodata(tmp_path, nodata):
    with pytest.raises(ValueError, match='nodata'):
        export(array('uint8', size=8), tmp_path / 'out.tif', nodata=nodata)
    assert not list(tmp_path.iterdir())


def test_integer_source_nodata(tmp_path):
    da = array('int16', size=8)
    da.values[0, 0] = -99
    da.attrs['_FillValue'] = -99
    export(da, tmp_path / 'out.tif', nodata=-999)
    with rasterio.open(tmp_path / 'out.tif') as result:
        assert result.nodata == -999
        assert result.read(1)[0, 0] == -999


@pytest.mark.parametrize('kwargs', [dict(overviews=[4, 2]), dict(tile_size=16),
                                    dict(compression='jpeg'), dict(resampling='cubic'),
                                    dict(colorinterp='rgb')])
def test_invalid_writer_options(tmp_path, kwargs):
    with pytest.raises(ValueError):
        export(array(size=8), tmp_path / 'out.tif', **kwargs)
    assert not list(tmp_path.iterdir())


def native_config(**kwargs):
    config = dict(width=128, height=128, bands=1, dtype='uint8', epsg=4326,
                  geographic=True, origin=[0, 128], scale=[1, 1], nodata=None,
                  tile_size=128, compression='none', deflate_level=7, resampling='nearest', overviews=[],
                  overwrite=False, memory_limit=268435456, colorinterp='undefined', metadata='')
    config.update(kwargs)
    return json.dumps(config)


def native_write(path, config, tiles):
    from xarray_cog._native import NativeWriter
    with NativeWriter(path, config) as writer:
        for tile in tiles:
            writer.write_tiles([tile])
        writer.finish()


def test_memory_budget_includes_tile_indexes(tmp_path):
    with pytest.raises(ValueError, match='memory'):
        native_write(tmp_path / 'out.tif', native_config(width=1_000_000, memory_limit=17_000_000), iter([]))
    assert not list(tmp_path.iterdir())


def test_native_failure_cleans_up_after_written_tile(tmp_path):
    path = tmp_path / 'out.tif'
    path.write_bytes(b'keep me')
    def tiles():
        yield np.zeros((128, 128, 1), dtype='uint8')
        raise RuntimeError('tile two failed')
    with pytest.raises(RuntimeError, match='tile two failed'):
        native_write(path, native_config(width=256, overwrite=True), tiles())
    assert path.read_bytes() == b'keep me'
    assert list(tmp_path.iterdir()) == [path]


@pytest.mark.parametrize('tiles, error', [([], 'missing'),
    ([np.zeros((129, 128, 1), dtype='uint8')], 'shape'),
    ([np.zeros((128, 128, 1), dtype='uint8')] * 2, 'too many')])
def test_native_tile_validation(tmp_path, tiles, error):
    with pytest.raises(ValueError, match=error):
        native_write(tmp_path / 'out.tif', native_config(), iter(tiles))
    assert not list(tmp_path.iterdir())


def test_regular_float32_coordinates(tmp_path):
    x = np.arange(-180, 180, 0.1).astype('float32')
    da = xr.DataArray(np.zeros((2, len(x)), dtype='uint8'), dims=('lat', 'lon'),
                      coords={'lon': x, 'lat': np.array([50, 49.9], dtype='float32')},
                      attrs={'crs': 'EPSG:4326'})
    export(da, tmp_path / 'out.tif')
    with rasterio.open(tmp_path / 'out.tif') as result:
        assert result.transform.a == pytest.approx(0.1, abs=1e-7)
        assert result.transform.c == pytest.approx(-180.05, abs=1e-6)


@pytest.mark.parametrize('x', [
    np.float32(7_000_000) + np.array([0, 0.5, 1, 1.5, 2, 2.5, 6], dtype='float32'),
    float(2**53) + np.array([0, 2, 6, 10, 16, 20, 26]),
    np.array([0, 1, 2, 3.25, 4, 5, 6], dtype='float64') * 1e-16,
])
def test_rejects_irregular_pixel_centers(tmp_path, x):
    da = array(size=5).assign_coords(x=x)
    with pytest.raises(ValueError, match='regular'):
        export(da, tmp_path / 'out.tif')
    assert not list(tmp_path.iterdir())


def test_default_deflate_matches_level_seven(tmp_path):
    da = array(size=512)
    default = export(da, tmp_path / 'default.tif')
    seven = export(da, tmp_path / 'seven.tif', deflate_level=7)
    six = export(da, tmp_path / 'six.tif', deflate_level=6)
    assert default.read_bytes() == seven.read_bytes()
    # This repeating fixture distinguishes the encoder's levels 6 and 7.
    assert default.read_bytes() != six.read_bytes()
    for path in (default, seven, six):
        with rasterio.open(path) as result:
            np.testing.assert_array_equal(result.read(1), da.values)


@pytest.mark.parametrize('level', [0, 10, 1.5, True])
def test_invalid_deflate_level(tmp_path, level):
    with pytest.raises(ValueError, match='deflate_level'):
        export(array(size=8), tmp_path / 'out.tif', deflate_level=level)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize('level', [0, 10])
def test_native_invalid_deflate_level(tmp_path, level):
    with pytest.raises(ValueError, match='deflate_level must be'):
        native_write(tmp_path / 'out.tif', native_config(deflate_level=level), iter([]))
    assert not list(tmp_path.iterdir())


def test_interrupted_export_preserves_destination_and_next_export_works(tmp_path):
    path = tmp_path / 'out.tif'
    path.write_bytes(b'original')
    def tiles():
        yield np.zeros((128, 128, 1), dtype='uint8')
        raise KeyboardInterrupt()
    with pytest.raises(KeyboardInterrupt):
        native_write(path, native_config(width=256, overwrite=True), tiles())
    assert path.read_bytes() == b'original'
    assert list(tmp_path.iterdir()) == [path]
    export(array(size=8), path, overwrite=True)
    with rasterio.open(path) as result:
        np.testing.assert_array_equal(result.read(1), array(size=8).values)


def test_duplicate_float32_coordinates_rejected(tmp_path):
    da = array(size=8).assign_coords(x=np.array([1, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype='float32'))
    with pytest.raises(ValueError, match='monotonic|regular'):
        export(da, tmp_path / 'out.tif')
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize('accessor', [False, True])
@pytest.mark.parametrize('limit,expected_bytes', [(None, 536_870_912), (64, 67_108_864), (64.5, 67_633_152), (256, 268_435_456)])
def test_memory_limit_in_mebibytes(tmp_path, monkeypatch, accessor, limit, expected_bytes):
    import xarray_cog
    plans = []
    original = xarray_cog.NativeWriter
    def capture(path, config):
        plans.append(json.loads(config))
        return original(path, config)
    monkeypatch.setattr(xarray_cog, 'NativeWriter', capture)
    source = array(size=8)
    options = {} if limit is None else {'memory_limit': limit}
    path = tmp_path / 'out.tif'
    if accessor:
        source.cog.to_cog(path, **options)
    else:
        xarray_cog.write_cog(source, path, **options)
    assert plans[0]['memory_limit'] == expected_bytes
    with rasterio.open(path) as result:
        np.testing.assert_array_equal(result.read(1), source.values)


@pytest.mark.parametrize('limit', [0, -1, True, '256', float('nan'), float('inf'), 1e30, 0.0000001])
def test_invalid_memory_limit_mebibytes(tmp_path, limit):
    with pytest.raises(ValueError, match='memory_limit.*MiB'):
        export(array(size=8), tmp_path / 'out.tif', memory_limit=limit)
    assert not list(tmp_path.iterdir())
