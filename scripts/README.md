# Scripts

Utility scripts for benchmarking, packaging verification, and script tests.

## Benchmarks

See [docs/benchmarks/README.md](../docs/benchmarks/README.md) for prerequisites,
options, methodology, and recorded results.

- [`benchmark.py`](benchmark.py) — Benchmark runner comparing
  `xarray_cog.write_cog`, `rioxarray`, `odc`, and `odc-windowed` across NumPy,
  Zarr, misaligned Zarr, and NetCDF inputs. Generates the synthetic Zarr v2
  fixture, measures export duration and peak RSS via `/usr/bin/time`, verifies
  output TIFF layout, georeferencing, and pixels, and writes JSON reports plus
  a summary table. Uses the `[dependency-groups]` `benchmark` group defined in
  [pyproject.toml](../pyproject.toml).
- [`benchmark.Dockerfile`](benchmark.Dockerfile) — Multi-stage Docker
  image that builds `xarray-cog` in release mode, installs the `benchmark`
  dependency group from [pyproject.toml](../pyproject.toml), and sets
  `benchmark.py` as the entrypoint.
- [`benchmark-docker.sh`](benchmark-docker.sh) — Builds
  `benchmark.Dockerfile` and runs `benchmark.py` inside a container with
  1 CPU and a configurable memory/swap limit (`BENCH_MEMORY`, default `4g`).

## Packaging verification

- [`smoke-installed.py`](smoke-installed.py) — Post-install smoke test run by
  `ci.yml`, `release.yml`, and `cibuildwheel` in a clean base-only
  environment. Verifies `xarray_cog` imports from `site-packages`, optional
  backends (`zarr`, `dask`, `rasterio`, `osgeo`) are absent, and both
  `xarray_cog.write_cog` and `.cog.to_cog` write valid BigTIFF files.

## Script tests

Run all script tests with:

```sh
python3 -m unittest discover -s scripts -p 'test_*.py'
```

- [`test_workflows.py`](test_workflows.py) — Stdlib-only unit tests for
  `benchmark.py` covering synthetic fixture generation (`create_sample`),
  overwrite refusal, CLI validation, `--help` output, and `format_summary_table`.
