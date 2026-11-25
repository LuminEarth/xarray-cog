"""Progress describes completed work without changing export behavior."""
import numpy as np
import pytest
import xarray as xr

import xarray_cog


def raster():
    return xr.DataArray(np.ones((192, 192), dtype='uint16'), dims=('y', 'x'),
                        coords={'y': -np.arange(192), 'x': np.arange(192)},
                        attrs={'crs': 'EPSG:4326'})


@pytest.mark.parametrize('terminal', [False, True])
def test_progress_is_opt_in_and_reports_finalization(tmp_path, capsys, monkeypatch, terminal):
    import sys
    monkeypatch.setattr(sys.stderr, 'isatty', lambda: terminal)
    source = raster()
    source.cog.to_cog(tmp_path / 'quiet.tif', tile_size=128)
    assert capsys.readouterr() == ('', '')
    source.cog.to_cog(tmp_path / 'visible.tif', tile_size=128, progress=True)
    out, err = capsys.readouterr()
    assert out == ''
    assert '0/4 tiles' in err and '4/4 tiles' in err
    assert err.index('Finalizing') < err.index('Done')
    assert err.endswith('\n')
    assert ('\r' in err) == terminal
    assert (tmp_path / 'quiet.tif').read_bytes() == (tmp_path / 'visible.tif').read_bytes()


@pytest.mark.parametrize('error', [RuntimeError, KeyboardInterrupt])
def test_progress_failure_preserves_destination(tmp_path, capsys, monkeypatch, error):
    from xarray_cog._source import _SourceCache
    original = _SourceCache.read
    calls = 0
    def read(self, *args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise error('source failed')
        return original(self, *args, **kwargs)
    monkeypatch.setattr(_SourceCache, 'read', read)
    path = tmp_path / 'out.tif'
    path.write_bytes(b'original')
    with pytest.raises(error, match='source failed'):
        raster().cog.to_cog(path, tile_size=128, overwrite=True, progress=True, num_threads=1)
    err = capsys.readouterr().err
    assert '1/4 tiles' in err and 'Stopped' in err
    assert 'Done' not in err and err.endswith('\n')
    assert path.read_bytes() == b'original'
    assert list(tmp_path.iterdir()) == [path]


def test_progress_requires_boolean(tmp_path):
    with pytest.raises(ValueError, match='progress must be a boolean'):
        xarray_cog.write_cog(raster(), tmp_path / 'out.tif', progress='yes')


def test_closed_progress_stream_does_not_prevent_export(tmp_path, monkeypatch):
    import io
    import sys
    stream = io.StringIO()
    stream.close()
    monkeypatch.setattr(sys, 'stderr', stream)
    path = raster().cog.to_cog(tmp_path / 'out.tif', progress=True)
    assert path.exists()


def test_finalization_failure_does_not_report_success(tmp_path, capsys, monkeypatch):
    original = xarray_cog.NativeWriter
    class FailOnFinish:
        def __init__(self, *args):
            self.writer = original(*args)
        def __getattr__(self, name):
            return getattr(self.writer, name)
        def __enter__(self):
            return self
        def __exit__(self, *args):
            self.writer.close()
        def finish(self):
            raise OSError('finalization failed')
    monkeypatch.setattr(xarray_cog, 'NativeWriter', FailOnFinish)
    with pytest.raises(OSError, match='finalization failed'):
        raster().cog.to_cog(tmp_path / 'out.tif', tile_size=128, progress=True)
    err = capsys.readouterr().err
    assert err.index('Finalizing') < err.index('Stopped')
    assert 'Done' not in err
    assert not list(tmp_path.iterdir())
