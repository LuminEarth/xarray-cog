"""Computed input chunks are reused across tiles and bands within one export."""
from collections import Counter

import dask
import dask.array as da
import numpy as np
import pytest
import rasterio
import xarray as xr

import xarray_cog


def lazy_raster(reads, chunk=192, bands=3):
    @dask.delayed
    def read(row, col):
        reads.append((row, col))
        b, y, x = np.indices((bands, chunk, chunk))
        return (b * 10000 + (y + row * chunk) * 5 + x + col * chunk).astype('int32')
    blocks = [[da.from_delayed(read(row, col), shape=(bands, chunk, chunk), dtype='int32')
               for col in range(2)] for row in range(2)]
    data = da.concatenate([da.concatenate(row, axis=2) for row in blocks], axis=1)
    return xr.DataArray(data, dims=('band', 'y', 'x'),
                        coords={'band': np.arange(bands), 'y': -np.arange(chunk * 2),
                                'x': np.arange(chunk * 2)}, attrs={'crs': 'EPSG:4326'})


@pytest.mark.parametrize('scheduler', ['synchronous', 'threads'])
@pytest.mark.parametrize('chunk,rechunk,reverse', [(192, None, False), (256, None, True),
                                                   (192, 256, False)])
def test_chunked_calculations_export_correct_pixels(tmp_path, scheduler, chunk, rechunk, reverse):
    reads = []
    raster = lazy_raster(reads, chunk)
    if rechunk:
        raster = raster.chunk({'y': rechunk, 'x': rechunk})
    if reverse:
        raster = raster.isel(y=slice(None, None, -1), x=slice(None, None, -1))
    with xr.set_options(keep_attrs=True):
        raster = raster * 2 + 7
    with dask.config.set(scheduler=scheduler, num_workers=2):
        raster.cog.to_cog(tmp_path / 'out.tif', tile_size=128)
    b, y, x = np.indices((3, chunk * 2, chunk * 2))
    with rasterio.open(tmp_path / 'out.tif') as result:
        np.testing.assert_array_equal(result.read(), (b * 10000 + y * 5 + x) * 2 + 7)
    if rechunk is None:
        assert Counter(reads) == Counter({(row, col): 1 for row in range(2) for col in range(2)})
    else:
        # Upstream results shared by different final chunks are not retained.
        assert len(reads) > 4


def test_dataset_calculations_cache_their_own_final_chunks(tmp_path):
    reads = []
    source = lazy_raster(reads, bands=1).isel(band=0, drop=True)
    dataset = xr.Dataset({'a': source, 'b': source + 20}, attrs=source.attrs)
    with dask.config.set(scheduler='synchronous'):
        dataset.cog.to_cog(tmp_path / 'out.tif', tile_size=128)
    with rasterio.open(tmp_path / 'out.tif') as result:
        np.testing.assert_array_equal(result.read(2), result.read(1) + 20)
    assert len(reads) == 8  # Separate calculations retain separate final chunks.


def test_non_dask_netcdf_chunks_are_read_once(tmp_path, monkeypatch):
    from xarray.backends.h5netcdf_ import H5NetCDFArrayWrapper
    y, x = np.indices((384, 384))
    source = xr.DataArray((y * 5 + x).astype('int32'), dims=('y', 'x'),
                          coords={'y': -np.arange(384), 'x': np.arange(384)},
                          attrs={'crs': 'EPSG:4326'}, name='values')
    path = tmp_path / 'source.nc'
    source.to_netcdf(path, engine='h5netcdf', encoding={'values': {'chunksizes': (192, 192)}})
    reads = []
    original = H5NetCDFArrayWrapper.__getitem__
    def counted(self, key):
        if self.variable_name == 'values':
            reads.append(key)
        return original(self, key)
    monkeypatch.setattr(H5NetCDFArrayWrapper, '__getitem__', counted)
    with xr.open_dataset(path, engine='h5netcdf', chunks=None, cache=False) as opened:
        opened.cog.to_cog(tmp_path / 'out.tif', tile_size=128)
    with rasterio.open(tmp_path / 'out.tif') as result:
        np.testing.assert_array_equal(result.read(1), source.values)
    assert len(reads) == 4


def test_cache_budget_lru_and_owned_slices():
    import weakref
    from xarray_cog._source import _SourceCache
    cache = _SourceCache()
    cache.set_limit(70000)
    a, b, c = (np.full(4096, value, dtype='float64') for value in (1, 2, 3))
    cache._put('a', a)
    cache._put('b', b)
    assert cache._get('a') is not None  # a is now more recently used than b.
    cache._put('c', c)
    assert cache._get('b') is None
    assert cache._get('a') is not None and cache._get('c') is not None
    assert cache.bytes <= cache.limit
    cache._put('too-large', np.ones(100000, dtype='uint8'))
    assert cache._get('too-large') is None
    assert cache.bytes <= cache.limit
    cache.clear()
    parent = np.ones(100000, dtype='uint8')
    parent_ref = weakref.ref(parent)
    cache._put('slice', parent[:100])
    del parent
    assert parent_ref() is None  # A tiny cached slice must not retain its parent.
    assert cache._get('slice').flags.owndata
    cache.clear()
    assert cache.bytes == 0


@pytest.mark.parametrize('error', [RuntimeError, KeyboardInterrupt])
def test_cache_is_cleared_after_failure_and_between_exports(tmp_path, monkeypatch, error):
    from xarray_cog._source import _SourceCache
    caches = []
    original = _SourceCache.set_limit
    def record_limit(self, limit):
        caches.append(self)
        return original(self, limit)
    monkeypatch.setattr(_SourceCache, 'set_limit', record_limit)
    state = {'value': 1, 'fail': True}
    @dask.delayed
    def read(row, col):
        if state['fail'] and (row, col) == (1, 1):
            raise error('late chunk failed')
        return np.full((192, 192), state['value'], dtype='uint16')
    data = da.block([[da.from_delayed(read(row, col), shape=(192, 192), dtype='uint16')
                      for col in range(2)] for row in range(2)])
    raster = xr.DataArray(data, dims=('y', 'x'), coords={'y': -np.arange(384), 'x': np.arange(384)},
                          attrs={'crs': 'EPSG:4326'})
    output = tmp_path / 'out.tif'
    output.write_bytes(b'original')
    with dask.config.set(scheduler='synchronous'):
        with pytest.raises(error, match='late chunk failed'):
            raster.cog.to_cog(output, tile_size=128, overwrite=True)
        assert output.read_bytes() == b'original'
        assert list(tmp_path.iterdir()) == [output]
        assert caches[-1].bytes == 0
        state['fail'] = False
        for value in (2, 3):
            state['value'] = value
            raster.cog.to_cog(output, tile_size=128, overwrite=True)
            with rasterio.open(output) as result:
                assert np.all(result.read(1) == value)
            assert caches[-1].bytes == 0
    assert len({id(c) for c in caches}) == 3
    assert all(0 < c.limit < 512 * 1024**2 for c in caches)


def test_eviction_and_oversized_results_still_produce_correct_pixels(tmp_path, monkeypatch):
    from xarray_cog._source import _SourceCache
    original_limit, original_put = _SourceCache.set_limit, _SourceCache._put
    sizes = []
    def limit(self, available):
        original_limit(self, min(available, 200000))
    def put(self, key, value):
        original_put(self, key, value)
        sizes.append(self.bytes)
        assert self.bytes <= self.limit
    monkeypatch.setattr(_SourceCache, 'set_limit', limit)
    monkeypatch.setattr(_SourceCache, '_put', put)
    reads = []
    raster = lazy_raster(reads)
    with dask.config.set(scheduler='synchronous'):
        raster.cog.to_cog(tmp_path / 'out.tif', tile_size=128)
    b, y, x = np.indices((3, 384, 384))
    with rasterio.open(tmp_path / 'out.tif') as result:
        np.testing.assert_array_equal(result.read(), b * 10000 + y * 5 + x)
    assert len(reads) > 4  # A small cache must allow rereading instead of growing.
    assert sizes and max(sizes) <= 200000


def test_in_place_dask_calculation_cannot_mutate_cached_source(tmp_path):
    reads = []
    source = lazy_raster(reads, bands=1).isel(band=0, drop=True)
    def modify(block):
        block += 1
        return block
    changed = source.copy(data=source.data.map_blocks(modify, dtype=source.dtype))
    dataset = xr.Dataset({'changed': changed, 'original': source}, attrs=source.attrs)
    with dask.config.set(scheduler='synchronous'):
        dataset.cog.to_cog(tmp_path / 'out.tif', tile_size=128)
    y, x = np.indices((384, 384))
    with rasterio.open(tmp_path / 'out.tif') as result:
        np.testing.assert_array_equal(result.read(1), y * 5 + x + 1)
        np.testing.assert_array_equal(result.read(2), y * 5 + x)
    assert len(reads) == 8  # Separate calculations retain separate final chunks.


def test_non_dask_multiband_storage_chunks_are_shared(tmp_path, monkeypatch):
    from xarray.backends.h5netcdf_ import H5NetCDFArrayWrapper
    b, y, x = np.indices((3, 384, 384))
    source = xr.DataArray((b * 10000 + y * 5 + x).astype('int32'), dims=('band', 'y', 'x'),
                          coords={'band': [0, 1, 2], 'y': -np.arange(384), 'x': np.arange(384)},
                          attrs={'crs': 'EPSG:4326'}, name='values')
    path = tmp_path / 'source.nc'
    source.to_netcdf(path, engine='h5netcdf', encoding={'values': {'chunksizes': (3, 192, 192)}})
    reads = []
    original = H5NetCDFArrayWrapper.__getitem__
    def counted(self, key):
        if self.variable_name == 'values':
            reads.append(key)
        return original(self, key)
    monkeypatch.setattr(H5NetCDFArrayWrapper, '__getitem__', counted)
    with xr.open_dataset(path, engine='h5netcdf', chunks=None, cache=False) as opened:
        opened['values'].cog.to_cog(tmp_path / 'out.tif', tile_size=128)
    with rasterio.open(tmp_path / 'out.tif') as result:
        np.testing.assert_array_equal(result.read(), source.values)
    assert len(reads) == 4


def test_writer_budget_failure_happens_before_source_reads(tmp_path):
    reads = []
    with pytest.raises(ValueError, match='memory'):
        lazy_raster(reads).cog.to_cog(tmp_path / 'out.tif', tile_size=128, memory_limit=1)
    assert reads == []
    assert not list(tmp_path.iterdir())


def test_oversized_backend_chunk_reads_only_requested_tile(tmp_path, monkeypatch):
    from xarray.backends.h5netcdf_ import H5NetCDFArrayWrapper
    from xarray_cog._source import _SourceCache
    original_limit = _SourceCache.set_limit
    monkeypatch.setattr(_SourceCache, 'set_limit', lambda self, size: original_limit(self, 1000))
    y, x = np.indices((384, 384))
    source = xr.DataArray((y * 5 + x).astype('int32'), dims=('y', 'x'),
                          coords={'y': -np.arange(384), 'x': np.arange(384)},
                          attrs={'crs': 'EPSG:4326'}, name='values')
    path = tmp_path / 'source.nc'
    source.to_netcdf(path, engine='h5netcdf', encoding={'values': {'chunksizes': (384, 384)}})
    reads = []
    original = H5NetCDFArrayWrapper.__getitem__
    def counted(self, key):
        if self.variable_name == 'values':
            reads.append(key.tuple)
        return original(self, key)
    monkeypatch.setattr(H5NetCDFArrayWrapper, '__getitem__', counted)
    with xr.open_dataset(path, engine='h5netcdf', chunks=None, cache=False) as opened:
        opened.cog.to_cog(tmp_path / 'out.tif', tile_size=128)
    with rasterio.open(tmp_path / 'out.tif') as result:
        np.testing.assert_array_equal(result.read(1), source.values)
    assert len(reads) == 9
    assert all(s.stop - s.start <= 128 for key in reads for s in key)


def test_cache_does_not_retain_unaccounted_object_payloads():
    from xarray_cog._source import _SourceCache
    cache = _SourceCache()
    cache.set_limit(70000)
    values = np.empty(1, dtype=object)
    values[0] = bytearray(100000)
    cache._put('objects', values)
    assert cache._get('objects') is None
    assert cache.bytes == 0


def test_process_scheduler_reuses_source_chunks(tmp_path):
    counter = tmp_path / 'reads.txt'
    @dask.delayed
    def read():
        with counter.open('a') as stream:
            stream.write('read\n')
        return np.arange(192 * 192, dtype='int32').reshape(192, 192)
    data = da.from_delayed(read(), shape=(192, 192), dtype='int32')
    raster = xr.DataArray(data, dims=('y', 'x'),
                          coords={'y': -np.arange(192), 'x': np.arange(192)},
                          attrs={'crs': 'EPSG:4326'})
    with dask.config.set(scheduler='processes', num_workers=1):
        raster.cog.to_cog(tmp_path / 'out.tif', tile_size=128)
    assert counter.read_text().splitlines() == ['read']
    with rasterio.open(tmp_path / 'out.tif') as result:
        np.testing.assert_array_equal(result.read(1), np.arange(192 * 192).reshape(192, 192))


@pytest.mark.parametrize("capacity", [192 * 192 * 4 + 512, 10000000])
def test_only_final_declared_chunks_are_retained(capacity):
    from xarray_cog._source import _SourceCache
    reads = []
    source = lazy_raster(reads, chunk=192, bands=1).isel(band=0, drop=True)
    calculated = source * 2 + 7
    cache = _SourceCache()
    # Retain one final chunk, whether the cache fits one or many arrays.
    cache.set_limit(capacity)
    with dask.config.set(scheduler='synchronous'):
        first = cache.read(calculated, 0, 0, 64)
        second = cache.read(calculated, 64, 64, 64)
    y, x = np.indices((64, 64))
    np.testing.assert_array_equal(first, (y * 5 + x) * 2 + 7)
    np.testing.assert_array_equal(second, ((y + 64) * 5 + x + 64) * 2 + 7)
    assert reads == [(0, 0)]
    assert len(cache._entries) == 1
    retained = next(iter(cache._entries.values()))
    assert retained.shape == (192, 192)
    np.testing.assert_array_equal(retained, calculated[:192, :192].compute())


def test_rechunked_input_reuses_final_chunks_but_not_shared_intermediates():
    from xarray_cog._source import _SourceCache
    reads = []
    source = lazy_raster(reads, chunk=192, bands=1).isel(band=0, drop=True)
    calculated = (source * 2 + 7).chunk({'y': 96, 'x': 96})
    cache = _SourceCache()
    cache.set_limit(10000000)
    with dask.config.set(scheduler='synchronous'):
        for col in (0, 96, 0, 96):
            tile = cache.read(calculated, 0, col, 32)
            y, x = np.indices((32, 32))
            np.testing.assert_array_equal(tile, (y * 5 + x + col) * 2 + 7)
    # Each final chunk is computed once. Their common upstream chunk runs twice.
    assert reads == [(0, 0), (0, 0)]
    assert len(cache._entries) == 2
    assert all(value.shape == (96, 96) for value in cache._entries.values())
