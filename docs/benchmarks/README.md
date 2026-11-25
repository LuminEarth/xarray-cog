# Python export benchmarks

The shared runner (`scripts/benchmark.py`) compares `xarray_cog.write_cog`,
rioxarray's COG driver, and `odc.geo.cog.write_cog` (ordinary and windowed writes)
on an auto-generated synthetic raster.

## Quick start

Prerequisites: Linux or macOS, `/usr/bin/time` (GNU or BSD `time`), and either
[`uv`](https://docs.astral.sh/uv/) or a Rust/Python toolchain. Benchmark
dependencies are defined in the `[dependency-groups]` `benchmark` group in
[pyproject.toml](../../pyproject.toml).

With `uv` (pass `--no-editable` so `maturin` builds an optimized release
extension rather than a debug editable build):

```sh
# Quick smoke benchmark (1024x1024, 1 repeat)
uv run --no-editable --group benchmark python scripts/benchmark.py benchmark-results/smoke --size 1024 --repeats 1

# Full comparison (8192x8192, 3 repeats, 4 threads)
uv run --no-editable --group benchmark python scripts/benchmark.py benchmark-results/python --size 8192 --repeats 3 --threads 4
```

Without `uv` (`pip >= 25.1` is required for `--group`):

```sh
maturin develop --release --locked
python -m pip install --group benchmark
python scripts/benchmark.py benchmark-results/python --size 8192 --repeats 3 --threads 4
```

Existing output directories are refused so previous results are never overwritten.
Generated reports under `benchmark-results/` are git-ignored and not committed.

## Options

| Option | Default | Description |
|---|---|---|
| `directory` | *(required)* | New output directory for fixtures, GeoTIFFs, and JSON reports |
| `--size` | `8192` | Raster width and height in pixels (power of two `>= 512`) |
| `--repeats` | `3` | Timed repeats per `(mode, tool)` combination |
| `--threads` | `1` | Native/GDAL worker ceiling (`1`, `2`, ..., or `auto`) |
| `--gdal-level` | `7` | DEFLATE level (`6` or `7`) for rioxarray/ODC; `xarray_cog` always uses `7` |
| `--tools` | all four | Subset of `xarray_cog`, `rioxarray`, `odc`, `odc-windowed` |
| `--modes` | all four | Subset of `numpy`, `zarr`, `zarr-misaligned`, `netcdf` |

Run `python scripts/benchmark.py --help` to see all options and usage examples.

## Running in Docker

For a container run with one CPU and a 4 GiB memory/swap cap:

```sh
scripts/benchmark-docker.sh benchmark-results/docker --size 8192 --repeats 3
```

Additional flags after the output directory are forwarded to `benchmark.py`.
Set `BENCH_MEMORY` (for example `BENCH_MEMORY=8g`) to change the container cap; the
recorded cgroup values identify the active limits. Container measurements and
local measurements are separate experiments.

## Output files

Each run creates a new result directory containing:

- `input.zarr` (and `input.nc` when `netcdf` is selected): synthetic three-band
  Float32 fixture generated automatically by `benchmark.py`
- `summary.json`: per-`(mode, tool)` median export time, peak RSS, output size,
  and input/output chunk/tile layouts
- `comparison.json`: raw per-repeat timings, peak RSS, SHA-256 hashes, and worker
  metadata
- `verification.json`: independent TIFF layout, georeferencing, and pixel checks
- `environment.json`: package versions, GDAL version, native binary SHA-256, and
  benchmark configuration
- `<mode>-<tool>-<repeat>.tif`, `.log`, `.time`, `.export.json`: per-run outputs
  and diagnostics

## Methodology

The default raster is 8192×8192 with three Float32 bands (768 MiB decoded).
The matrix covers NumPy-loaded, Zarr/Dask, Zarr/Dask with 768-pixel chunks
misaligned with 512-pixel TIFF tiles, and NetCDF/Dask inputs. All exporters receive
the same three-band Float32 xarray values, grid and NoData. The synthetic fixture
is structured and compressible, so results do not represent every scientific
dataset. Benchmarks run separately from routine CI.

Settings: 512-pixel tiles, nearest overviews, pixel interleave, BigTIFF, no
predictor and DEFLATE **7** for all tools. `--threads` selects the native/GDAL
worker ceiling (default 1); use `--threads auto` for automatic selection. To reproduce a
comparison against GDAL's level 6, use `--gdal-level 6`; xarray-cog remains at 7,
and `environment.json` records every tool's level. ODC's intermediate compression
is disabled. xarray-cog uses a **512 MiB writer/cache budget**; GDAL uses a
**512 MiB cache**. Total process memory is measured separately. All memory sizes
below use MiB.

Runs alternate tool order. Export timing includes fsync and excludes imports and
input preparation. Peak memory measures the whole Python process, including resident
NumPy input and imports. Filesystem caches are not flushed. Every base and
overview pixel, layout and georeferencing is checked independently using the
shared TIFF verifier. Repeated output hashes must match. Each run and summary
records the input shape, per-variable storage chunk shape, complete Dask chunk
lengths (null for NumPy), actual output tile shape and interleave. Storage chunks
and Dask chunks are reported separately because they can differ. Reports also
record the requested thread ceiling and, for xarray-cog, the actual native worker
count, batch size and retained-source-cache budget. Dask uses the synchronous
scheduler in these comparisons to isolate the effect of native compression workers.

## Results: 8192×8192 pixels, 3 Float32 bands (2026-11-17)

**Every result below uses the same three-band raster: 8192 rows × 8192 columns ×
3 bands, or 768 MiB decoded.** Input modes change how those same pixel values are
stored and read; they do not change the raster dimensions or band count.

The results show two comparisons:

- **Exporters at four workers:** compare xarray-cog, rioxarray and ODC with a
  four-worker ceiling across all four input modes.
- **xarray-cog at different worker counts:** keep the exporter fixed and vary
  its worker ceiling for the NumPy and misaligned Zarr inputs. The four-worker
  row is reused from the exporter comparison; the one-worker row measures serial
  execution of the same implementation.

Both comparisons use the same optimized build, three repeats per setting,
DEFLATE 7 and nearest overviews. The xarray-cog writer/cache budget and GDAL
cache are each 512 MiB. Dask scheduling is synchronous throughout. The chunk layout below applies to both comparisons.

Measured at commit `7f2789e` on macOS 26.4.1 arm64 with Python 3.11.10 and
GDAL 3.10.3. Package versions: NumPy 2.4.6, xarray 2026.9.0, Dask 2026.8.0,
Zarr 3.1.6, Rasterio 1.4.4, rioxarray 0.19.0, odc-geo 0.5.4, h5netcdf 1.8.3
and tifffile 2026.3.3. The commands above rerun the comparison with the installed
versions; timings vary with hardware and software versions.

### Shared input chunks and output tiles

Spatial shapes below are height × width. Each source variable is one band;
output tiles interleave all three bands. On-disk chunking and Dask chunking are
separate: changing the Dask chunks does not rewrite the source store.

| Input mode | On-disk chunk per variable | Dask chunk presented to exporter | Output tile |
|---|---|---|---|
| NumPy | 512×512 (loaded before timing) | None; eager array | 512×512×3 bands |
| Zarr | 512×512 | 1 band × 512×512 | 512×512×3 bands |
| Zarr, misaligned | 512×512 | 1 band × 768×768 | 512×512×3 bands |
| NetCDF | 512×512 | 1 band × 768×768 | 512×512×3 bands |

The aligned Dask case has 16 chunks per spatial axis. The 768-pixel cases have
10 chunks of 768 plus one of 512 on each spatial axis, so their edge chunks are
smaller. Every TIFF overview also uses 512×512 tiles. Recorded chunk sizes come
from the input objects and output files; full per-axis chunk lengths are included
in each run and summary.

### Exporter comparison: four-worker ceiling

Every exporter is configured with a four-worker ceiling. For xarray-cog this
resolves to four Rust workers and batches of up to four tiles. The rioxarray and
ODC calls pass `NUM_THREADS=4` to GDAL through Rasterio. These are native
compression threads, not four Python processes; actual parallel activity depends
on the driver and stage of the export. GDAL supports multithreaded DEFLATE
compression, and its COG driver also supports parallel overview generation.
See the [GDAL COG driver options](https://gdal.org/en/stable/drivers/raster/cog.html).

#### Median export time (seconds)

| Input | xarray-cog (4 Rust workers) | rioxarray (4 GDAL threads) | ODC (4 GDAL threads) | ODC windowed (4 GDAL threads) |
|---|---:|---:|---:|---:|
| numpy | 1.982 | 5.831 | 5.141 | 4.363 |
| zarr | 3.067 | 3.773 | 3.768 | 3.984 |
| zarr-misaligned | 2.085 | 3.086 | 3.343 | 5.100 |
| netcdf | 2.629 | 3.172 | 3.659 | 6.757 |

#### Peak process memory (MiB)

| Input | xarray-cog (4 Rust workers) | rioxarray (4 GDAL threads) | ODC (4 GDAL threads) | ODC windowed (4 GDAL threads) |
|---|---:|---:|---:|---:|
| numpy | 1770.9 | 2697.8 | 4021.8 | 4663.0 |
| zarr | 635.3 | 3309.9 | 3175.4 | 3629.6 |
| zarr-misaligned | 787.0 | 2892.5 | 2536.4 | 3630.2 |
| netcdf | 787.3 | 2896.6 | 2537.6 | 3633.8 |

#### Output and verification

- xarray-cog: 11.035 MiB per output across all input modes.
- rioxarray: 11.148 MiB per output across all input modes.
- odc: 11.148 MiB per output across all input modes.
- odc-windowed: 11.148 MiB per output across all input modes.

All 48 exports completed. Repeat hashes matched for each input/exporter pair;
independent verification checked every base and overview pixel, metadata and
layout for each of the 16 distinct pairs. Overview factors were 2, 4, 8 and 16.

This structured, compressible workload measures the complete export call,
including lazy input reads, while NumPy loading happens before timing. Peak memory
includes imports and input preparation. Builds and test suites were completed
before timing, and exporters ran sequentially. Filesystem caches were not flushed.
Results describe this workload and host, not a universal performance ranking.

### xarray-cog comparison: varying the worker count

These runs use the **same NumPy and misaligned Zarr inputs** from the exporter
comparison: 8192×8192 pixels, three Float32 bands, and the shared chunk/tile
layout above. Only the requested Rust worker count changes. The total writer/cache
budget stays at 512 MiB, so increasing workers leaves less space for cached chunks.

The **four-worker row repeats the xarray-cog values above**: 1.982 seconds for
NumPy and 2.085 seconds for misaligned Zarr. The **one-worker row is the serial
baseline** for this same build and input data. This comparison does not include
rioxarray, ODC, aligned Zarr or NetCDF.

After running the four-worker exporter comparison, reproduce the remaining
worker settings with:

```sh
for workers in 1 2 8 auto; do
    python scripts/benchmark.py "benchmark-results/scaling-$workers" \
        --size 8192 --repeats 3 --threads "$workers" \
        --tools xarray_cog --modes numpy zarr-misaligned
done
```

| Requested workers | Actual workers / maximum batch tiles | Cache budget (MiB) | NumPy median (s) | Misaligned Zarr median (s) | NumPy peak memory (MiB) | Misaligned Zarr peak memory (MiB) |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 1 / 1 | 431.9 | 2.545 | 3.528 | 1742.2 | 816.2 |
| 2 | 2 / 2 | 412.9 | 1.779 | 2.519 | 1751.4 | 802.2 |
| 4 (same as above) | 4 / 4 | 374.9 | 1.982 | 2.085 | 1770.9 | 787.0 |
| 8 | 8 / 8 | 298.9 | 1.353 | 1.819 | 1807.2 | 719.6 |
| auto | 8 / 8 | 298.9 | 1.379 | 1.812 | 1807.2 | 718.9 |

All worker settings produced byte-identical files for each input mode. Every
base and overview pixel was independently verified. More workers reserve more
batch/compression workspace, leaving less room to retain source chunks. Reading,
calculations, overview accumulation and final file assembly can limit speedup;
these measurements do not promise linear scaling with core count.
