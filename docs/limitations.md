# Limitations

- Export requires regular 1D pixel-center coordinates, at least two per spatial
  axis, and a horizontal two-axis EPSG CRS. No curvilinear/rotated grids,
  reprojection or implicit alignment.
- A Dataset contains ordered 2D raster variables sharing grid, dtype and CRS.
  A DataArray may also have one band axis. Select all extra dimensions first.
- Supported dtypes are uint8, uint16, int16, uint32, int32, float32 and float64.
  Explicit casting and calculations belong to xarray.
- Local BigTIFF output only; DEFLATE or no compression; nearest/average
  overviews; numeric bands or three-band RGB. No generated masks or alpha.
- Raster samples, georeferencing, NoData and band descriptions are preserved.
  Arbitrary Dataset metadata, encodings and nonspatial coordinates are not.
- `memory_limit` is in MiB and covers estimated
  writer buffers and retained source-cache entries,
  not all memory used by Python. Readers and Dask can allocate additional working buffers.
  Oversized or evicted chunks may be read again. Dask execution uses a local
  scheduler; distributed scheduling is unsupported. Rust worker count is limited
  by memory and tile columns; source reads and native compression do not overlap.
- Temporary disk space is needed for compressed blocks and final output.
- SIGINT is checked while copying tiles, between batches and around finalization. A native operation
  can delay interruption. Exceptions preserve the existing destination and clean up
  owned temporary files; abrupt process termination cannot promise cleanup.
- Windows is not a supported release target.
