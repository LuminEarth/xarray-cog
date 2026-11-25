#!/usr/bin/env python3
"""Compare xarray_cog, rioxarray, and ODC on identical xarray rasters."""
from __future__ import annotations

import argparse
from array import array
from collections.abc import Iterable, Mapping, Sequence
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import signal
import statistics
import subprocess
import sys
import time
from typing import Any
import zlib

ROOT: Path = Path(__file__).resolve().parents[1]
MEMORY_BUDGET_MIB: int = 512
TILE_SIZE: int = 512
MISALIGNED_CHUNK_SIZE: int = 768
BAND_NAMES: tuple[str, ...] = ('Band1', 'Band2', 'Band3')
TOOLS: tuple[str, ...] = ('xarray_cog', 'rioxarray', 'odc', 'odc-windowed')
MODES: tuple[str, ...] = ('numpy', 'zarr', 'zarr-misaligned', 'netcdf')
BENCHMARK_PACKAGES: tuple[str, ...] = (
    'numpy',
    'xarray',
    'dask',
    'zarr',
    'rasterio',
    'rioxarray',
    'odc-geo',
    'h5netcdf',
    'tifffile',
)
WORKER_TIMEOUT_SECONDS: int = 1800
TIME_BINARY: Path = Path('/usr/bin/time')


def _write_zarr_array(
    root: Path,
    name: str,
    shape: Sequence[int],
    chunks: Sequence[int],
    dtype: str,
    dimensions: Sequence[str],
    attributes: Mapping[str, Any],
) -> Path:
    """Create a Zarr v2 array directory with metadata and attributes.

    Args:
        root: Parent Zarr group directory.
        name: Array directory name under ``root``.
        shape: Full array shape along each dimension.
        chunks: Chunk shape along each dimension.
        dtype: NumPy little-endian dtype string (e.g., ``'<f4'``).
        dimensions: Dimension names written to ``_ARRAY_DIMENSIONS``.
        attributes: Extra attributes merged into ``.zattrs``.

    Returns:
        Path to the created array directory.
    """
    directory = root / name
    directory.mkdir()
    zarray = {
        'zarr_format': 2,
        'shape': list(shape),
        'chunks': list(chunks),
        'dtype': dtype,
        'compressor': {'id': 'zlib', 'level': 6},
        'fill_value': 'NaN',
        'order': 'C',
        'filters': None,
    }
    zattrs = {'_ARRAY_DIMENSIONS': list(dimensions), **attributes}
    (directory / '.zarray').write_text(
        json.dumps(zarray) + '\n',
        encoding='utf-8',
    )
    (directory / '.zattrs').write_text(
        json.dumps(zattrs) + '\n',
        encoding='utf-8',
    )
    return directory


def _encode_zarr_chunk(values: Iterable[float], kind: str) -> bytes:
    """Encode numeric values as a little-endian zlib-compressed Zarr chunk.

    Args:
        values: Iterable of numeric values for the chunk in C order.
        kind: ``array.array`` typecode (``'d'`` for float64, ``'f'`` for
            float32).

    Returns:
        Zlib-compressed little-endian bytes for the chunk.
    """
    data = array(kind, values)
    if sys.byteorder != 'little':
        data.byteswap()
    return zlib.compress(data.tobytes(), 6)


def create_sample(root: Path, size: int) -> None:
    """Create a deterministic RGB Float32 Zarr v2 store using stdlib only.

    Args:
        root: Target directory path for the new Zarr store. Must not exist.
        size: Spatial width and height in pixels (must be at least 2).

    Raises:
        ValueError: If ``size`` is less than 2.
        FileExistsError: If ``root`` already exists.
    """
    if size < 2:
        raise ValueError('size must be at least 2')
    # Exclusive creation refuses an existing store, even an empty one.
    root.mkdir(parents=True, exist_ok=False)
    (root / '.zgroup').write_text('{"zarr_format":2}\n', encoding='utf-8')
    chunk = min(size, TILE_SIZE)
    axes = (
        ('x', -10, 20 / size, 'longitude', 'X'),
        ('y', 60, -20 / size, 'latitude', 'Y'),
    )
    for name, origin, step, standard, axis in axes:
        path = _write_zarr_array(
            root,
            name,
            [size],
            [size],
            '<f8',
            [name],
            {'standard_name': standard, 'axis': axis},
        )
        coords = (origin + (i + 0.5) * step for i in range(size))
        (path / '0').write_bytes(_encode_zarr_chunk(coords, 'd'))
    for band_index, band_name in enumerate(BAND_NAMES, start=1):
        path = _write_zarr_array(
            root,
            band_name,
            [size, size],
            [chunk, chunk],
            '<f4',
            ['y', 'x'],
            {'epsg': 4326},
        )
        for row in range(0, size, chunk):
            for col in range(0, size, chunk):
                values = (
                    float('nan')
                    if y >= size or x >= size or (x + y) % 97 == 0
                    else float((x * 7 + y * 11 + band_index * 19) % 256)
                    for y in range(row, row + chunk)
                    for x in range(col, col + chunk)
                )
                chunk_name = f'{row // chunk}.{col // chunk}'
                (path / chunk_name).write_bytes(
                    _encode_zarr_chunk(values, 'f'),
                )


def format_summary_table(summary: Sequence[Mapping[str, Any]]) -> str:
    """Format benchmark summary rows as an aligned plain-text table.

    Args:
        summary: Sequence of per-(mode, tool) summary dictionaries.

    Returns:
        Multi-line plain-text table with aligned columns.
    """
    headers = ('Mode', 'Tool', 'Median (s)', 'Peak RSS (MiB)', 'Output (MiB)')
    rows = [
        (
            str(row['mode']),
            str(row['tool']),
            f"{row['median_export_seconds']:.3f}",
            f"{row['max_rss_mib']:.1f}",
            f"{row['output_mib']:.3f}",
        )
        for row in summary
    ]
    widths = [
        max(len(header), *(len(r[i]) for r in rows)) if rows else len(header)
        for i, header in enumerate(headers)
    ]
    header_line = (
        f'{headers[0]:<{widths[0]}}  {headers[1]:<{widths[1]}}  '
        f'{headers[2]:>{widths[2]}}  {headers[3]:>{widths[3]}}  '
        f'{headers[4]:>{widths[4]}}'
    )
    divider = '  '.join('-' * width for width in widths)
    body_lines = [
        f'{r[0]:<{widths[0]}}  {r[1]:<{widths[1]}}  '
        f'{r[2]:>{widths[2]}}  {r[3]:>{widths[3]}}  {r[4]:>{widths[4]}}'
        for r in rows
    ]
    return '\n'.join([header_line, divider, *body_lines])


def verify(path: Path, source: Path, size: int) -> dict[str, Any]:
    """Verify TIFF layout, georeferencing, and all base/overview pixels.

    Args:
        path: Output Cloud Optimized GeoTIFF file to inspect.
        source: Synthetic Zarr v2 source store used to compute expected pixels.
        size: Base raster width and height in pixels.

    Returns:
        Dictionary recording checked sample counts per pyramid level.

    Raises:
        AssertionError: If TIFF structure, tags, offsets, or pixels mismatch.
    """
    import numpy as np
    import tifffile

    checked: list[int] = []
    previous: Any = None
    with tifffile.TiffFile(path) as tif:
        factors = [1]
        while size // factors[-1] > TILE_SIZE:
            factors.append(factors[-1] * 2)
        assert tif.is_bigtiff and len(tif.pages) == len(factors)
        max_ifd_offset = max(p.offset for p in tif.pages)
        min_data_offset = min(min(p.dataoffsets) for p in tif.pages)
        assert max_ifd_offset < min_data_offset
        for page, factor in zip(tif.pages, factors, strict=True):
            width = size // factor
            assert page.shape == (width, width, 3)
            assert page.dtype == np.dtype('float32')
            assert page.is_tiled
            assert page.tilewidth == TILE_SIZE
            assert page.tilelength == TILE_SIZE
            assert page.photometric == 1 and page.planarconfig == 1
            assert len(page.extrasamples) == 2
            assert all(int(v) == 0 for v in page.extrasamples)
            assert int(page.compression) in (8, 32946)
            assert page.tags.get(317) is None or page.tags[317].value == 1
            assert np.isnan(float(page.tags[42113].value))
            # GDAL overview IFDs may inherit georeferencing from the base.
            if factor == 1 or 33550 in page.tags:
                assert list(page.tags[33550].value) == [
                    20 / width,
                    20 / width,
                    0,
                ]
            if factor == 1 or 33922 in page.tags:
                assert list(page.tags[33922].value) == [0, 0, 0, -10, 60, 0]
            if factor == 1 or 34735 in page.tags:
                assert page.geotiff_tags['GeographicTypeGeoKey'] == 4326
            assert list(page.dataoffsets) == sorted(page.dataoffsets)
            if factor > 1:
                assert previous is not None
                assert max(page.dataoffsets) < min(previous.dataoffsets)
            samples = 0
            for segment, position, _ in page.segments(maxworkers=1):
                row, col = position[2:4]
                ys = np.arange(row, min(row + TILE_SIZE, width)) * factor
                xs = np.arange(col, min(col + TILE_SIZE, width)) * factor
                expected = np.empty((len(ys), len(xs), 3), dtype='<f4')
                for cy in np.unique(ys // TILE_SIZE):
                    iy = np.flatnonzero(ys // TILE_SIZE == cy)
                    for cx in np.unique(xs // TILE_SIZE):
                        ix = np.flatnonzero(xs // TILE_SIZE == cx)
                        for band_index, band_name in enumerate(BAND_NAMES):
                            raw = (source / f'{band_name}/{cy}.{cx}').read_bytes()
                            chunk = np.frombuffer(
                                zlib.decompress(raw),
                                dtype='<f4',
                            ).reshape(TILE_SIZE, TILE_SIZE)
                            patch = chunk[
                                np.ix_(ys[iy] % TILE_SIZE, xs[ix] % TILE_SIZE)
                            ]
                            expected[np.ix_(iy, ix, [band_index])] = patch[
                                ..., None
                            ]
                np.testing.assert_array_equal(
                    segment[0, : len(ys), : len(xs)],
                    expected,
                )
                samples += expected.size
            checked.append(samples)
            previous = page
    return {
        'samples_checked_per_level': checked,
        'layout_and_georeferencing_checked': True,
    }


def measure(
    command: Sequence[str],
    target: Path,
    prefix: Path,
) -> dict[str, Any]:
    """Run a worker command under ``/usr/bin/time`` and collect metrics.

    Args:
        command: Worker subprocess command and arguments.
        target: Expected output GeoTIFF path.
        prefix: File path prefix for ``.time`` and ``.log`` artifacts.

    Returns:
        Dictionary with elapsed time, sync time, peak RSS bytes, output bytes,
        and SHA-256 digest.

    Raises:
        RuntimeError: If the worker exits non-zero or peak RSS cannot be read.
    """
    timing = prefix.with_suffix('.time')
    log_path = prefix.with_suffix('.log')
    mac = sys.platform == 'darwin'
    timed = (
        [str(TIME_BINARY), '-l']
        if mac
        else [str(TIME_BINARY), '-v', '-o', str(timing)]
    )
    started = time.monotonic()
    with log_path.open('w', encoding='utf-8') as log:
        process = subprocess.Popen(
            [*timed, *command],
            stdout=log,
            stderr=log,
            start_new_session=True,
            env={**os.environ, 'LC_ALL': 'C'},
        )
        try:
            code = process.wait(timeout=WORKER_TIMEOUT_SECONDS)
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
    command_seconds = time.monotonic() - started
    if code:
        raise RuntimeError(f'Conversion failed ({code}); see {log_path}')
    sync_started = time.monotonic()
    with target.open('rb') as output:
        os.fsync(output.fileno())
    sync_seconds = time.monotonic() - sync_started
    elapsed = time.monotonic() - started
    timing_text = (
        log_path.read_text(encoding='utf-8')
        if mac
        else timing.read_text(encoding='utf-8')
    )
    pattern = (
        r'(\d+)\s+maximum resident set size'
        if mac
        else r'Maximum resident set size \(kbytes\):\s*(\d+)'
    )
    match = re.search(pattern, timing_text)
    if match is None:
        source_path = log_path if mac else timing
        raise RuntimeError(f'Cannot read peak RSS from {source_path}')
    with target.open('rb') as output:
        hasher = hashlib.sha256()
        for chunk in iter(lambda: output.read(1024 * 1024), b''):
            hasher.update(chunk)
        digest = hasher.hexdigest()
    return {
        'command': list(command),
        'elapsed_seconds': elapsed,
        'command_seconds': command_seconds,
        'final_sync_seconds': sync_seconds,
        'peak_rss_bytes': int(match[1]) * (1 if mac else 1024),
        'output_bytes': target.stat().st_size,
        'sha256': digest,
    }


def worker(args: argparse.Namespace) -> None:
    """Execute one timed export in an isolated subprocess.

    Args:
        args: Parsed worker CLI arguments containing ``source``, ``output``,
            ``mode``, ``tool``, ``gdal_level``, and ``threads``.
    """
    # Identical imports and source preparation for all exporters; outside timing.
    import dask
    import numpy as np
    import odc.geo.xr  # noqa: F401 - registers .odc accessor
    from odc.geo.cog import write_cog as odc_write_cog
    import rasterio
    import rioxarray  # noqa: F401 - registers .rio accessor
    import xarray as xr
    import xarray_cog

    chunks: dict[str, int] = (
        {'y': MISALIGNED_CHUNK_SIZE, 'x': MISALIGNED_CHUNK_SIZE}
        if args.mode in ('zarr-misaligned', 'netcdf')
        else {}
    )
    opened = (
        xr.open_dataset(args.source, engine='h5netcdf', chunks=chunks)
        if args.mode == 'netcdf'
        else xr.open_zarr(args.source, consolidated=False, chunks=chunks)
    )
    with opened, dask.config.set(scheduler='synchronous'):
        cube = opened[list(BAND_NAMES)].to_array(dim='band')
        if args.mode == 'numpy':
            cube.load()
        # Record storage chunks separately from chunks presented to exporters.
        # NumPy is eager even though preparation reads a chunked source store.
        input_layout: dict[str, Any] = {
            'input_shape': dict(cube.sizes),
            'input_storage_chunks': {
                name: dict(
                    zip(
                        opened[name].dims,
                        opened[name].encoding.get('chunksizes')
                        or opened[name].encoding['chunks'],
                        strict=True,
                    ),
                )
                for name in BAND_NAMES
            },
            'input_dask_chunks': (
                None
                if cube.chunks is None
                else {
                    dim: list(lengths)
                    for dim, lengths in zip(
                        cube.dims,
                        cube.chunks,
                        strict=True,
                    )
                }
            ),
        }
        cube.rio.write_crs(4326, inplace=True)
        cube.rio.write_nodata(np.nan, inplace=True)
        levels: list[int] = []
        factor = 2
        while max(cube.sizes['x'], cube.sizes['y']) // (factor // 2) > TILE_SIZE:
            levels.append(factor)
            factor *= 2
        assert cube.odc.geobox is not None
        native_plan: dict[str, Any] = {}
        if args.tool == 'xarray_cog':
            original_writer = xarray_cog.NativeWriter

            def record_plan(*arguments: Any) -> Any:
                """Intercept NativeWriter creation to record worker settings."""
                writer = original_writer(*arguments)
                native_plan.update(
                    native_workers=writer.num_threads,
                    batch_size=writer.batch_size,
                    source_cache_budget=writer.source_cache_budget,
                )
                return writer

            xarray_cog.NativeWriter = record_plan
        gdal_threads: int | str = (
            args.threads if args.threads is not None else 'ALL_CPUS'
        )
        with rasterio.Env(
            GDAL_CACHEMAX=MEMORY_BUDGET_MIB * 1024**2,
            GDAL_NUM_THREADS=str(gdal_threads),
        ):
            started = time.perf_counter()
            if args.tool == 'xarray_cog':
                xarray_cog.write_cog(
                    cube,
                    args.output,
                    tile_size=TILE_SIZE,
                    compression='deflate',
                    deflate_level=7,
                    resampling='nearest',
                    memory_limit=MEMORY_BUDGET_MIB,
                    num_threads=args.threads,
                )
            elif args.tool.startswith('odc'):
                odc_write_cog(
                    cube,
                    args.output,
                    blocksize=TILE_SIZE,
                    ovr_blocksize=TILE_SIZE,
                    overview_levels=levels,
                    overview_resampling='nearest',
                    use_windowed_writes=args.tool == 'odc-windowed',
                    intermediate_compression=False,
                    nodata=np.nan,
                    compress='DEFLATE',
                    zlevel=args.gdal_level,
                    predictor=1,
                    BIGTIFF='YES',
                    NUM_THREADS=gdal_threads,
                    interleave='pixel',
                    photometric='MINISBLACK',
                )
            else:
                cube.rio.to_raster(
                    args.output,
                    driver='COG',
                    COMPRESS='DEFLATE',
                    LEVEL=args.gdal_level,
                    PREDICTOR='NO',
                    OVERVIEW_COMPRESS='DEFLATE',
                    OVERVIEW_PREDICTOR='NO',
                    BLOCKSIZE=TILE_SIZE,
                    BIGTIFF='YES',
                    NUM_THREADS=gdal_threads,
                    RESAMPLING='NEAREST',
                )
            with args.output.open('rb') as output:
                os.fsync(output.fileno())
            seconds = time.perf_counter() - started
        # Inspect actual TIFF blocks outside the timed export, rather than just
        # reporting the requested tile size.
        with rasterio.open(args.output) as result:
            assert len(set(result.block_shapes)) == 1
            height, width = result.block_shapes[0]
            output_layout: dict[str, Any] = {
                'output_tile_shape': {
                    'y': height,
                    'x': width,
                    'band': result.count,
                },
                'output_interleave': result.profile['interleave'],
            }
        export_report = {
            'export_seconds': seconds,
            'requested_threads': args.threads,
            **native_plan,
            **input_layout,
            **output_layout,
        }
        args.output.with_suffix('.export.json').write_text(
            json.dumps(export_report),
            encoding='utf-8',
        )


def thread_limit(value: str) -> int | None:
    """Parse a positive worker ceiling or ``'auto'`` for automatic selection.

    Args:
        value: CLI option value string.

    Returns:
        Positive integer worker ceiling, or ``None`` when ``'auto'``.

    Raises:
        argparse.ArgumentTypeError: If ``value`` is invalid.
    """
    if value == 'auto':
        return None
    try:
        count = int(value)
        if count > 0:
            return count
    except ValueError:
        pass
    raise argparse.ArgumentTypeError('threads must be a positive integer or auto')


def _build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser for the benchmark runner."""
    examples = (
        'examples:\n'
        '  Quick smoke run:\n'
        '    python scripts/benchmark.py benchmark-results/smoke --size 1024 --repeats 1\n\n'
        '  Full exporter comparison:\n'
        '    python scripts/benchmark.py benchmark-results/python --size 8192 --repeats 3 --threads 4\n\n'
        '  Worker scaling on a subset of tools and modes:\n'
        '    python scripts/benchmark.py benchmark-results/scaling-4 --size 8192 --repeats 3 \\\n'
        '        --threads 4 --tools xarray_cog --modes numpy zarr-misaligned'
    )
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog=examples,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        'directory',
        type=Path,
        nargs='?',
        help='New output directory for generated inputs, TIFFs, and JSON reports',
    )
    parser.add_argument(
        '--size',
        type=int,
        default=8192,
        help='Raster width/height as a power of two >=512 (default: 8192)',
    )
    parser.add_argument(
        '--repeats',
        type=int,
        default=3,
        help='Timed repeats per mode/tool pair (default: 3)',
    )
    parser.add_argument(
        '--threads',
        type=thread_limit,
        default=1,
        help='Native/GDAL worker ceiling, or auto (default: 1)',
    )
    parser.add_argument(
        '--gdal-level',
        type=int,
        choices=(6, 7),
        default=7,
        help='DEFLATE level for rioxarray/ODC; xarray_cog always uses 7 (default: 7)',
    )
    parser.add_argument(
        '--tools',
        nargs='+',
        choices=TOOLS,
        default=list(TOOLS),
        help='Exporters to compare (default: all four)',
    )
    parser.add_argument(
        '--modes',
        nargs='+',
        choices=MODES,
        default=list(MODES),
        help='Input storage/chunking modes to benchmark (default: all four)',
    )
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--source', type=Path, help=argparse.SUPPRESS)
    parser.add_argument('--output', type=Path, help=argparse.SUPPRESS)
    parser.add_argument('--mode', help=argparse.SUPPRESS)
    parser.add_argument('--tool', help=argparse.SUPPRESS)
    return parser


def _resolve_source_revision() -> str:
    """Return the git revision for the benchmark environment metadata."""
    revision = os.environ.get('BENCH_SOURCE_REVISION')
    if revision:
        return revision
    try:
        return subprocess.check_output(
            ['git', '-C', str(ROOT), 'rev-parse', 'HEAD'],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return 'unknown'


def _check_prerequisites(parser: argparse.ArgumentParser) -> dict[str, str]:
    """Validate runtime tools and Python packages before creating outputs.

    Args:
        parser: Active CLI parser used to exit cleanly on missing prerequisites.

    Returns:
        Mapping of benchmark package names to installed version strings.
    """
    if not TIME_BINARY.exists():
        parser.exit(
            1,
            'Required timing binary /usr/bin/time was not found '
            '(GNU or BSD time is required).\n',
        )
    versions: dict[str, str] = {}
    missing: list[str] = []
    for name in BENCHMARK_PACKAGES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            missing.append(name)
    if missing:
        parser.exit(
            1,
            f"Missing benchmark packages: {', '.join(missing)}.\n"
            'Run with uv (uv run --no-editable --group benchmark ...) '
            'or install with: python -m pip install --group benchmark\n',
        )
    try:
        import xarray_cog._native  # noqa: F401 - verify native extension
    except ImportError as exc:
        parser.exit(
            1,
            f'Native xarray_cog extension is unavailable ({exc}).\n'
            'Build it first with: maturin develop --release --locked\n',
        )
    return versions


def main(argv: Sequence[str] | None = None) -> None:
    """Run the benchmark CLI or delegate to a timed worker subprocess.

    Args:
        argv: Optional sequence of CLI arguments (defaults to ``sys.argv[1:]``).
    """
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.worker:
        worker(args)
        return
    if args.size < TILE_SIZE or args.size & (args.size - 1) or args.repeats < 1:
        parser.error('size must be a power of two >=512; repeats must be positive')
    if sys.platform not in ('linux', 'darwin') or sys.flags.optimize:
        parser.error(
            'use Linux/macOS Python without -O (verification uses assertions)',
        )
    if args.directory is None or args.directory.exists():
        parser.error('provide a new result directory; existing paths are refused')

    versions = _check_prerequisites(parser)

    import rasterio
    import xarray as xr
    import xarray_cog._native

    native_file = xarray_cog._native.__file__
    assert native_file is not None
    native_sha256 = hashlib.sha256(Path(native_file).read_bytes()).hexdigest()

    directory = args.directory.resolve()
    directory.mkdir(parents=True)
    source = directory / 'input.zarr'
    print(f'Generating {args.size}x{args.size} synthetic fixture...', flush=True)
    create_sample(source, args.size)
    netcdf = directory / 'input.nc'
    if 'netcdf' in args.modes:
        with xr.open_zarr(source, consolidated=False, chunks={}) as data:
            encoding = {
                band: {
                    'chunksizes': (TILE_SIZE, TILE_SIZE),
                    'zlib': True,
                    'complevel': 6,
                }
                for band in BAND_NAMES
            }
            data.to_netcdf(netcdf, engine='h5netcdf', encoding=encoding)

    environment: dict[str, Any] = {
        'versions': versions,
        'gdal': rasterio.__gdal_version__,
        'platform': platform.platform(),
        'python': platform.python_version(),
        'size': args.size,
        'repeats': args.repeats,
        'source_revision': _resolve_source_revision(),
        'native_sha256': native_sha256,
        'mode': os.environ.get('BENCH_MODE', 'local'),
        'input_decoded_bytes': args.size**2 * 12,
        'deflate_levels': {
            tool: 7 if tool == 'xarray_cog' else args.gdal_level
            for tool in args.tools
        },
        'threads': args.threads,
        'tile_size': TILE_SIZE,
        'source_chunk_size': TILE_SIZE,
        'misaligned_dask_chunks': MISALIGNED_CHUNK_SIZE,
        'predictor': False,
        'overviews': 'nearest',
        'dask_scheduler': 'synchronous',
        'timing': 'export call plus fsync; excludes imports/input preparation',
        'rss': (
            'whole-process peak including source preparation; '
            'not incremental writer memory'
        ),
        'memory': (
            f'xarray-cog writer and source cache: {MEMORY_BUDGET_MIB} MiB; '
            f'GDAL cache: {MEMORY_BUDGET_MIB} MiB; '
            'total process memory is additional'
        ),
        'filesystem_cache': 'not flushed',
        'odc_intermediate_compression': False,
    }
    for name in ('memory.max', 'memory.swap.max', 'cpu.max'):
        path = Path('/sys/fs/cgroup') / name
        if path.exists():
            environment[name] = path.read_text(encoding='utf-8').strip()
    (directory / 'environment.json').write_text(
        json.dumps(environment, indent=2) + '\n',
        encoding='utf-8',
    )

    reports: list[dict[str, Any]] = []
    total_runs = len(args.modes) * args.repeats * len(args.tools)
    step = 0
    script_path = str(Path(__file__).resolve())
    thread_arg = str(args.threads) if args.threads is not None else 'auto'
    for mode in args.modes:
        for repeat in range(1, args.repeats + 1):
            order = args.tools if repeat % 2 else list(reversed(args.tools))
            for tool in order:
                step += 1
                target = directory / f'{mode}-{tool}-{repeat}.tif'
                command = [
                    sys.executable,
                    script_path,
                    '--worker',
                    '--source',
                    str(netcdf if mode == 'netcdf' else source),
                    '--mode',
                    mode,
                    '--tool',
                    tool,
                    '--output',
                    str(target),
                    '--gdal-level',
                    str(args.gdal_level),
                    '--threads',
                    thread_arg,
                ]
                metrics = measure(command, target, target.with_suffix(''))
                export_meta = json.loads(
                    target.with_suffix('.export.json').read_text(
                        encoding='utf-8',
                    ),
                )
                report: dict[str, Any] = {
                    'mode': mode,
                    'tool': tool,
                    'repeat': repeat,
                    **metrics,
                    **export_meta,
                }
                reports.append(report)
                (directory / 'comparison.json').write_text(
                    json.dumps(reports, indent=2) + '\n',
                    encoding='utf-8',
                )
                rss_mib = report['peak_rss_bytes'] / 2**20
                print(
                    f'[{step}/{total_runs}] {mode}/{tool} '
                    f'repeat {repeat}/{args.repeats}: '
                    f"{report['export_seconds']:.3f}s "
                    f'(peak RSS {rss_mib:.1f} MiB)',
                    flush=True,
                )

    verification: dict[str, Any] = {}
    summary: list[dict[str, Any]] = []
    layout_keys = (
        'input_shape',
        'input_storage_chunks',
        'input_dask_chunks',
        'output_tile_shape',
        'output_interleave',
    )
    native_keys = ('native_workers', 'batch_size', 'source_cache_budget')
    for mode in args.modes:
        for tool in args.tools:
            runs = [
                r for r in reports if r['mode'] == mode and r['tool'] == tool
            ]
            assert len({r['sha256'] for r in runs}) == 1, (
                f'{mode}/{tool}: outputs differ'
            )
            verification[f'{mode}/{tool}'] = verify(
                directory / f'{mode}-{tool}-1.tif',
                source,
                args.size,
            )
            layout = {key: runs[0][key] for key in layout_keys}
            assert all(
                all(run[key] == value for key, value in layout.items())
                for run in runs
            )
            native_meta = {
                key: runs[0][key] for key in native_keys if key in runs[0]
            }
            summary.append({
                'mode': mode,
                'tool': tool,
                **layout,
                'requested_threads': args.threads,
                **native_meta,
                'median_export_seconds': statistics.median(
                    r['export_seconds'] for r in runs
                ),
                'max_rss_mib': max(r['peak_rss_bytes'] for r in runs) / 2**20,
                'output_mib': runs[0]['output_bytes'] / 2**20,
            })

    (directory / 'verification.json').write_text(
        json.dumps(verification, indent=2) + '\n',
        encoding='utf-8',
    )
    (directory / 'summary.json').write_text(
        json.dumps(summary, indent=2) + '\n',
        encoding='utf-8',
    )
    print()
    print(format_summary_table(summary))
    print(
        f'\nAll base/overview pixels verified. Reports saved to {directory} '
        '(summary.json, comparison.json, verification.json, environment.json).',
    )


if __name__ == '__main__':
    main()
