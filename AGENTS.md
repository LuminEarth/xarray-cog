# AGENTS.md

Guidance for coding agents working in `xarray-cog`. See [README.md](README.md), [docs/architecture.md](docs/architecture.md), [docs/limitations.md](docs/limitations.md), and [CONTRIBUTING.md](CONTRIBUTING.md) for user-facing docs and setup details.

## Repository layout

- `src/cog/` — Rust COG/BigTIFF pyramid writer (`builder.rs`, `layout.rs`, `pyramid.rs`, `sample.rs`, `raster_copy.rs`, `error.rs`)
- `src/python.rs` — PyO3 native extension (`xarray_cog._native`)
- `python/xarray_cog/` — Python API, xarray accessor, source reading, and `_native.pyi` type stubs
- `tests/` — Rust integration tests
- `python/tests/` — Python end-to-end tests
- `scripts/test_*.py` — packaging and workflow verification tests

## Build and verification

After editing Rust code (`src/`), rebuild the extension before running Python tests:

```sh
maturin develop --release --locked
```

Run checks relevant to your changes (all checks before opening a PR):

```sh
python -m pytest python/tests -q
python -m mypy
python -m unittest discover -s scripts -p 'test_*.py'
cargo test --locked
cargo fmt --check
cargo clippy --features python --all-targets --locked -- -D warnings
RUSTDOCFLAGS='-D warnings' cargo doc --no-deps --locked
```

## Invariants

- **Keep runtime dependencies minimal**: Never add `rasterio`, `dask`, `zarr`, `h5netcdf`, `h5py`, or Rust `tiff` to base runtime dependencies (`pyproject.toml` `[project.dependencies]` or `Cargo.toml` `[dependencies]`).
- **Keep stubs and typing in sync**: Any PyO3 signature change in `src/python.rs` must update `python/xarray_cog/_native.pyi` and pass `mypy` (`disallow_untyped_defs = true`).
- **Preserve writer guarantees**: Do not add implicit reprojection, implicit casting, or lossy dtype promotion; failed reads or writes must preserve any existing destination file and clean up temporary files (see [docs/limitations.md](docs/limitations.md)).
- **Do not manually edit release metadata**: Versions in `Cargo.toml`, `Cargo.lock`, `pyproject.toml`, and entries in `CHANGELOG.md` are managed by Release Please.
- **PR titles**: Use Conventional Commits (`feat:`, `fix:`, `docs:`, etc.).
