"""Bounded batch exports preserve output and source reuse across worker counts."""
import json

import numpy as np
import pytest
import rasterio
import xarray as xr

import xarray_cog
from test_export import native_config


@pytest.mark.parametrize('resampling', ['nearest', 'average'])
def test_parallel_exports_match_serial_with_edge_tiles(tmp_path, resampling):
    y, x = np.indices((515, 901))
    values = np.stack([y * 3 + x, y + x * 2, x - y]).astype('float32')
    source = xr.DataArray(values, dims=('band', 'y', 'x'),
                          coords={'y': -np.arange(515), 'x': np.arange(901)},
                          attrs={'crs': 'EPSG:4326'})
    outputs = []
    for threads in (1, 2, 4, 8, None):
        path = tmp_path / f'{threads}.tif'
        source.cog.to_cog(path, tile_size=128, resampling=resampling, num_threads=threads)
        outputs.append(path.read_bytes())
        with rasterio.open(path) as result:
            np.testing.assert_array_equal(result.read(), values)
    assert all(output == outputs[0] for output in outputs)


@pytest.mark.parametrize('threads', [0, -1, True, 1.5, 'auto'])
def test_invalid_thread_limit(tmp_path, threads):
    source = xr.DataArray(np.ones((2, 2), dtype='uint8'), dims=('y', 'x'),
                          coords={'y': [1, 0], 'x': [0, 1]}, attrs={'crs': 'EPSG:4326'})
    with pytest.raises(ValueError, match='num_threads'):
        source.cog.to_cog(tmp_path / 'out.tif', num_threads=threads)
    assert not list(tmp_path.iterdir())


def test_native_writer_plans_and_closes_without_installing(tmp_path):
    from xarray_cog._native import NativeWriter
    path = tmp_path / 'out.tif'
    with NativeWriter(path, native_config(width=1024, num_threads=4)) as writer:
        assert writer.num_threads == writer.batch_size == 4
        assert 0 < writer.source_cache_budget < 268435456
        writer.write_tiles([np.ones((128, 128, 1), dtype='uint8')] * 4)
        assert not path.exists()
    assert not list(tmp_path.iterdir())
    with pytest.raises(ValueError, match='closed'):
        writer.finish()


def test_memory_budget_limits_workers(tmp_path):
    from xarray_cog._native import NativeWriter
    with NativeWriter(tmp_path / 'out.tif', native_config(width=1024, num_threads=8,
                      memory_limit=18 << 20)) as writer:
        assert 1 <= writer.num_threads < 8
        assert writer.batch_size == writer.num_threads
        assert writer.source_cache_budget >= 0


def test_batch_limit_and_failure_poison_writer(tmp_path):
    from xarray_cog._native import NativeWriter
    writer = NativeWriter(tmp_path / 'out.tif', native_config(width=512, num_threads=2))
    with pytest.raises(ValueError, match='batch'):
        writer.write_tiles([np.ones((128, 128, 1), dtype='uint8')] * 3)
    with pytest.raises(ValueError, match='closed'):
        writer.finish()
    assert not list(tmp_path.iterdir())


def test_native_no_clobber_race(tmp_path):
    from xarray_cog._native import NativeWriter
    path = tmp_path / 'out.tif'
    with NativeWriter(path, native_config()) as writer:
        writer.write_tiles([np.ones((128, 128, 1), dtype='uint8')])
        path.write_bytes(b'created while exporting')
        with pytest.raises(FileExistsError):
            writer.finish()
    assert path.read_bytes() == b'created while exporting'
    assert list(tmp_path.iterdir()) == [path]


def test_batch_reuses_dask_tasks_without_retention():
    import dask
    import dask.array as da
    from xarray_cog._source import _SourceCache
    calls = []
    @dask.delayed
    def read():
        calls.append(1)
        return np.arange(256 * 256, dtype='int32').reshape(256, 256)
    source = xr.DataArray(da.from_delayed(read(), shape=(256, 256), dtype='int32'), dims=('y', 'x'))
    cache = _SourceCache()  # No retention: sharing must happen in the batch graph.
    with dask.config.set(scheduler='synchronous'):
        tiles = cache.read_many(source, [(0, 0), (0, 128)], 128)
    assert calls == [1]
    np.testing.assert_array_equal(np.concatenate(tiles, axis=1),
                                  np.arange(256 * 256).reshape(256, 256)[:128])
    assert cache.bytes == 0


def test_python_reads_bounded_batches_and_updates_only_after_write(tmp_path, capsys, monkeypatch):
    original = xarray_cog.NativeWriter
    batch_lengths = []
    class FailingWriter:
        def __init__(self, *args):
            self.writer = original(*args)
        def __getattr__(self, name):
            return getattr(self.writer, name)
        def __enter__(self):
            return self
        def __exit__(self, *args):
            self.writer.close()
        def write_tiles(self, tiles):
            batch_lengths.append(len(tiles))
            self.writer.write_tiles(tiles)
            if len(batch_lengths) == 2:
                raise RuntimeError('batch failed')
    monkeypatch.setattr(xarray_cog, 'NativeWriter', FailingWriter)
    source = xr.DataArray(np.ones((256, 513), dtype='uint8'), dims=('y', 'x'),
                          coords={'y': -np.arange(256), 'x': np.arange(513)},
                          attrs={'crs': 'EPSG:4326'})
    with pytest.raises(RuntimeError, match='batch failed'):
        source.cog.to_cog(tmp_path / 'out.tif', tile_size=128, num_threads=2, progress=True)
    assert batch_lengths == [2, 2]
    err = capsys.readouterr().err
    assert '2/10 tiles | Stopped' in err
    assert 'Done' not in err
    assert not list(tmp_path.iterdir())


def test_native_dtype_error_aborts_and_releases_output(tmp_path):
    from xarray_cog._native import NativeWriter
    writer = NativeWriter(tmp_path / 'out.tif', native_config())
    with pytest.raises(TypeError):
        writer.write_tiles([np.ones((128, 128, 1), dtype='float32')])
    assert not list(tmp_path.iterdir())
    writer.close()  # Closing an already-failed writer is harmless.


def test_worker_budget_reserves_edge_and_encoding_buffers(tmp_path):
    from xarray_cog._native import NativeWriter
    budget = 256 << 20
    with NativeWriter(tmp_path / 'out.tif', native_config(width=4096, height=2047,
                      tile_size=2048, bands=3, compression='deflate', num_threads=2)) as writer:
        # Python input, Rust copy, full edge padding, serialized samples, and
        # compressed Vec capacity (up to two sample buffers) coexist.
        tile_bytes = 2048 * 2048 * 3
        minimum = (16 << 20) + 2 * (6 * tile_bytes + (1 << 20))
        assert budget - writer.source_cache_budget >= minimum


def test_mutating_submitted_list_cannot_change_completed_tile_count(tmp_path):
    import sys
    import threading
    from xarray_cog._native import NativeWriter
    tile = np.random.default_rng(5).integers(0, 256, (2048, 2048, 3), dtype='uint8')
    tiles = [tile]
    ready = threading.Event()
    thread = threading.Thread(target=lambda: (ready.wait(), tiles.clear()))
    thread.start()
    interval = sys.getswitchinterval()
    try:
        # The other Python thread can run once write_tiles releases the GIL.
        sys.setswitchinterval(100)
        with NativeWriter(tmp_path / 'out.tif', native_config(width=2048, height=2048,
                          tile_size=2048, bands=3, compression='deflate', num_threads=1)) as writer:
            ready.set()
            writer.write_tiles(tiles)
            thread.join()
            assert tiles == []
            writer.finish()
    finally:
        sys.setswitchinterval(interval)
        ready.set()
        thread.join()
    with rasterio.open(tmp_path / 'out.tif') as result:
        np.testing.assert_array_equal(result.read(), tile.transpose(2, 0, 1))
