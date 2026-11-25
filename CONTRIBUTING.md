# Contributing

Bug reports, feature requests, and pull requests are welcome.

Before starting work on a new feature or a larger change, please open an issue first to discuss the approach and make sure it fits the project's direction. Small bug fixes and documentation improvements can be opened directly as pull requests.

For security vulnerabilities, see [SECURITY.md](SECURITY.md) instead of opening a public issue.

## Development setup

Use [`uv`](https://docs.astral.sh/uv/) with Python 3.10+, Rust 1.91+, and a C/C++ toolchain. The package name is `xarray-cog` and the Python import name is `xarray_cog`.

### Python setup and checks

```sh
uv sync --extra test
uv run --extra test pytest python/tests -q
uv run --extra test mypy
uv run --extra test python -m unittest discover -s scripts -p 'test_*.py'
```

Pass `--extra test` on `uv run` so the `test` dependencies stay installed. After editing Rust code (`src/`), rebuild the extension with `uv sync --extra test --reinstall-package xarray-cog` (add `--no-editable` for an optimized release build).

### Rust checks

```sh
cargo test --locked
cargo fmt --check
cargo clippy --features python --all-targets --locked -- -D warnings
RUSTDOCFLAGS='-D warnings' cargo doc --no-deps --locked
```

Notes:
- On macOS, a custom compiler may need the active SDK and libc++ headers in `CXXFLAGS`.
- Keep test-only decoders (`rasterio`, the Rust `tiff` crate) out of base runtime dependencies.
- On Python 3.10, the test extra uses Zarr 2 and skips only the Zarr v3 source test.

## Pull requests

- Keep PRs focused and include tests for behavior changes.
- Ensure the Python and Rust checks above pass before requesting review.
- Use [Conventional Commit](https://www.conventionalcommits.org/) style for PR titles (for example, `feat: add an export option` or `fix: preserve nodata`).
