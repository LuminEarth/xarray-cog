"""Export regular xarray rasters through the GDAL-free Rust COG writer."""
from __future__ import annotations

import json
from collections.abc import Hashable, Iterable, Mapping, Sequence
from os import PathLike
from typing import Any, Literal, TypeAlias
from pathlib import Path
from xml.etree.ElementTree import Element, SubElement, tostring

import numpy as np
from numpy.typing import NDArray
import xarray as xr
from pyproj import CRS

from ._native import NativeWriter
from ._progress import _Progress
from ._source import _SourceCache

__all__ = ['write_cog']
_Raster: TypeAlias = xr.DataArray | xr.Dataset
_Number: TypeAlias = int | float | np.integer[Any] | np.floating[Any]
_NoData: TypeAlias = _Number | NDArray[Any]
_DTYPES = {'uint8', 'uint16', 'int16', 'uint32', 'int32', 'float32', 'float64'}


def _axis(obj: _Raster, requested: str | None, candidates: Sequence[str]) -> str:
    """Resolve an explicit or unambiguous spatial dimension.

    Args:
        obj: Xarray object whose dimensions are inspected.
        requested: Explicit dimension name, or None to infer it.
        candidates: Conventional dimension names to consider when inferring.

    Returns:
        The selected dimension name.

    Raises:
        ValueError: The requested dimension is missing or inference is ambiguous.
    """
    if requested is not None:
        if requested not in obj.dims:
            raise ValueError(f'spatial dimension {requested!r} is missing')
        return requested
    found = [name for name in candidates if name in obj.dims]
    if len(found) != 1:
        raise ValueError('specify x_dim and y_dim for ambiguous spatial dimensions')
    return found[0]


def _coordinate(obj: _Raster, dim: str) -> tuple[NDArray[np.float64], np.float64]:
    """Validate regular pixel centers and calculate their signed spacing.

    Args:
        obj: Xarray object containing the coordinate.
        dim: Spatial dimension with a one-dimensional coordinate.

    Returns:
        A tuple of float64 pixel centers and their signed spacing.

    Raises:
        ValueError: Coordinates are missing, nonnumeric, nonfinite, irregular,
            nonmonotonic, or contain fewer than two centers.
    """
    if dim not in obj.coords or obj[dim].dims != (dim,) or obj.sizes[dim] < 2:
        raise ValueError(f'{dim}: require a 1D coordinate with at least two pixel centers')
    raw = np.asarray(obj[dim].values)
    if not np.issubdtype(raw.dtype, np.number) or np.iscomplexobj(raw):
        raise ValueError(f'{dim}: coordinates must be real numeric pixel centers')
    values = raw.astype('float64')
    delta = np.diff(values)
    step = (values[-1] - values[0]) / (len(values) - 1)
    # Float32 pixel centers contain quantization noise.
    eps = np.finfo('float32').eps * 4 if raw.dtype == np.dtype('float32') else np.finfo('float64').eps * 32
    offsets = values - values[0]
    expected_offsets = step * np.arange(len(values))
    tolerance = eps * np.maximum(np.abs(values), abs(values[0])) + abs(step) * 1e-6
    # A large coordinate origin must never hide a displacement beyond half a pixel.
    tolerance = np.minimum(tolerance, abs(step) / 2)
    if (not np.isfinite(values).all() or not np.isfinite(delta).all() or step == 0
            or not np.all(delta > 0 if step > 0 else delta < 0)
            or not np.all(np.abs(offsets - expected_offsets) <= tolerance)):
        raise ValueError(f'{dim}: coordinates must be finite, monotonic and regular')
    return values, step


def _crs(da: xr.DataArray, parent_attrs: Mapping[Hashable, Any]) -> CRS:
    """Resolve a band's CRS from consistent attribute and CF metadata.

    Args:
        da: Raster band with optional CRS attributes or mapping coordinates.
        parent_attrs: Attributes inherited from the containing Dataset.

    Returns:
        The resolved pyproj.CRS.

    Raises:
        ValueError: CRS metadata is missing, conflicting, or references a
            missing grid-mapping coordinate.
        pyproj.exceptions.CRSError: A CRS definition cannot be interpreted.
    """
    candidates = []
    for attrs in (parent_attrs, da.attrs):
        for key in ('crs', 'epsg', 'epsg_code'):
            if key in attrs:
                candidates.append(CRS.from_user_input(attrs[key]))
    # Xarray can move grid_mapping into encoding during CF decoding.
    mapping = da.attrs.get('grid_mapping', da.encoding.get('grid_mapping'))
    if mapping:
        if mapping not in da.coords:
            raise ValueError(f'grid_mapping coordinate {mapping!r} is missing')
        candidates.append(CRS.from_cf(da.coords[mapping].attrs))
    elif 'spatial_ref' in da.coords:
        attrs = da.coords['spatial_ref'].attrs
        if attrs:
            candidates.append(CRS.from_cf(attrs))
    if not candidates:
        raise ValueError('missing CRS: supply crs="EPSG:code"')
    if any(value != candidates[0] for value in candidates[1:]):
        raise ValueError('conflicting CRS metadata')
    return candidates[0]


def _sentinel(da: xr.DataArray) -> _NoData | None:
    """Read a missing-value sentinel for the band's current values.

    Args:
        da: Raster band with optional _FillValue or missing_value attributes.

    Returns:
        The scalar sentinel, or None when neither attribute is present.

    Raises:
        ValueError: Sentinels are nonscalar or disagree.
    """
    # Encoding describes stored values; xarray may already have decoded these.
    values = [da.attrs[key] for key in ('_FillValue', 'missing_value') if key in da.attrs]
    if any(np.ndim(v) != 0 for v in values):
        raise ValueError('NoData metadata must be scalar')
    if len(values) == 2 and not _same(values[0], values[1]):
        raise ValueError('conflicting NoData metadata')
    return values[0] if values else None


def _same(a: _NoData, b: _NoData) -> bool | np.bool_ | NDArray[np.bool_]:
    """Compare numeric scalars, treating two NaNs as equal.

    Args:
        a: First scalar.
        b: Second scalar.

    Returns:
        Whether the scalars represent the same value or are both NaN.
    """
    return a == b or (np.isnan(a) and np.isnan(b))


def write_cog(
    obj: _Raster,
    path: str | PathLike[str],
    *,
    x_dim: str | None = None,
    y_dim: str | None = None,
    band_dim: str | None = None,
    crs: object | None = None,
    nodata: _NoData | None = None,
    tile_size: int = 512,
    compression: Literal['deflate', 'none'] = 'deflate',
    resampling: Literal['nearest', 'average'] = 'nearest',
    overviews: Literal['auto', 'none'] | Iterable[int] = 'auto',
    overwrite: bool = False,
    memory_limit: int | float = 512,
    colorinterp: Literal['undefined', 'rgb'] = 'undefined',
    deflate_level: int = 7,
    progress: bool = False,
    num_threads: int | None = None,
) -> Path:
    """Write an xarray raster to a local Cloud Optimized GeoTIFF.

    Dataset variables become bands in variable order. A DataArray may have a
    band dimension; select time, depth and other nonspatial dimensions before
    export. Bands must share a supported dtype, regular grid and horizontal CRS.
    Current values are written without reapplying storage scale or offset.

    Args:
        obj: DataArray or Dataset to export.
        path: Destination filesystem path, as a string or path-like object.
        x_dim: Horizontal dimension name. Defaults to inference from x, lon,
            or longitude.
        y_dim: Vertical dimension name. Defaults to inference from y, lat,
            or latitude.
        band_dim: DataArray band dimension. Defaults to band when present;
            otherwise the array is treated as a single band. Not used for Datasets.
        crs: CRS accepted by pyproj.CRS.from_user_input. Overrides input metadata
            without reprojection. Defaults to resolving the input's CRS metadata.
        nodata: Output missing-value sentinel, exactly representable in the
            input dtype. Defaults to NaN for floating data or a shared source
            sentinel for integer data, when present. Valid samples must not
            collide with the output sentinel.
        tile_size: Tile width and height in pixels: 128, 256, 512, 1024 or 2048.
            Defaults to 512.
        compression: Compression mode, 'deflate' or 'none'. Defaults to 'deflate'.
        resampling: Overview resampling, 'nearest' or 'average'. Defaults to
            'nearest'.
        overviews: 'auto', 'none', or increasing powers-of-two reduction factors.
            Defaults to 'auto', which adds levels until the largest dimension
            fits within one tile.
        overwrite: Whether to replace an existing destination after successful
            finalization. Defaults to False.
        memory_limit: Budget for estimated writer allocations and retained
            source chunks, in MiB. Defaults to 512.
            Accepts positive integers or floats. The cache uses the budget left
            after reserving writer buffers. Input arrays and temporary reader
            or Dask calculations use additional memory.
        colorinterp: 'undefined' for numeric bands or 'rgb' for exactly three
            bands. Defaults to 'undefined'.
        deflate_level: Integer compression level from 1 to 9. Defaults to 7;
            ignored when compression is 'none'.
        num_threads: Positive integer ceiling for Rust compression workers, or
            None to select automatically from available CPUs and memory. Workers
            are also limited by tile columns and the memory budget. This does
            not configure the Dask scheduler.
        progress: Show completed tiles and finalization status on stderr.
            Defaults to False. Tile progress is not a percentage of total time.

    Returns:
        pathlib.Path: The destination path after successful export.

    Raises:
        TypeError: obj is not a DataArray or Dataset.
        ValueError: Raster metadata, dtype or writer options are unsupported,
            NoData conflicts with valid samples, the writer budget is exceeded,
            or a distributed Dask scheduler is selected.
        pyproj.exceptions.CRSError: A CRS definition cannot be interpreted.
        FileExistsError: The destination exists and overwrite is False.
        OSError: Output or temporary-file I/O fails.

    Input-reader exceptions also propagate. Failed exports preserve an existing
    destination and clean up owned temporary files. See docs/python.md for the
    full input requirements and writer limitations.
    """
    if not isinstance(obj, (xr.DataArray, xr.Dataset)):
        raise TypeError('expected an xarray DataArray or Dataset')
    if type(progress) is not bool:
        raise ValueError('progress must be a boolean')
    if num_threads is not None and (type(num_threads) is not int or num_threads <= 0):
        raise ValueError('num_threads must be a positive integer or None')
    x_dim = _axis(obj, x_dim, ('x', 'lon', 'longitude'))
    y_dim = _axis(obj, y_dim, ('y', 'lat', 'latitude'))
    if x_dim == y_dim:
        raise ValueError('spatial dimensions must differ')
    x, dx = _coordinate(obj, x_dim)
    y, dy = _coordinate(obj, y_dim)
    if isinstance(obj, xr.Dataset):
        if band_dim is not None:
            raise ValueError('band_dim applies only to a DataArray')
        arrays = list(obj.data_vars.values())
        names = list(obj.data_vars)
        parent_attrs = obj.attrs
    else:
        band_dim = band_dim or ('band' if 'band' in obj.dims else None)
        if band_dim in (x_dim, y_dim):
            raise ValueError('band dimension must differ from spatial dimensions')
        if band_dim is not None:
            if band_dim not in obj.dims:
                raise ValueError('band dimension is missing')
            arrays = [obj.isel({band_dim: i}, drop=True) for i in range(obj.sizes[band_dim])]
            names = [str(v) for v in obj[band_dim].values]
        else:
            arrays = [obj]
            names = [str(obj.name or 'band_1')]
        parent_attrs = {}
    if not arrays:
        raise ValueError('at least one raster band is required')
    dtype = arrays[0].dtype
    if dtype.name not in _DTYPES or not dtype.isnative:
        raise ValueError(f'unsupported dtype {dtype}; explicitly cast before export')
    sources = []
    resolved_crs = []
    for da in arrays:
        if set(da.dims) != {y_dim, x_dim}:
            raise ValueError('select nonspatial dimensions before export; every band must be 2D')
        if da.dtype != dtype:
            raise ValueError('all bands must have the same dtype; explicitly cast before export')
        if 'scale_factor' in da.attrs or 'add_offset' in da.attrs:
            raise ValueError('decode scale_factor/add_offset with xarray.decode_cf before export')
        sources.append(_sentinel(da))
        resolved_crs.append(CRS.from_user_input(crs) if crs is not None else _crs(da, parent_attrs))
    if any(c != resolved_crs[0] for c in resolved_crs[1:]):
        raise ValueError('all bands must have the same CRS')
    resolved = resolved_crs[0]
    epsg = resolved.to_epsg()
    if (not (resolved.is_geographic or resolved.is_projected) or len(resolved.axis_info) != 2
            or epsg is None or not 0 < epsg < 65536 or epsg == 32767):
        raise ValueError('require a horizontal 2D EPSG CRS representable in GeoTIFF')
    if nodata is None:
        if np.issubdtype(dtype, np.floating):
            nodata = np.nan
        else:
            present = [v for v in sources if v is not None]
            if present:
                if any(not _same(v, present[0]) for v in present):
                    raise ValueError('different source NoData values: specify a common nodata')
                nodata = present[0]
    if nodata is not None:
        if np.ndim(nodata) != 0 or np.isinf(nodata):
            raise ValueError('nodata must be a finite scalar or floating NaN')
        if np.issubdtype(dtype, np.integer):
            info = np.iinfo(dtype)
            if not np.isfinite(nodata) or not info.min <= nodata <= info.max or int(nodata) != nodata:
                raise ValueError('nodata is not exactly representable in dtype')
        with np.errstate(over='ignore', invalid='ignore'):
            encoded = np.asarray(nodata).astype(dtype).item()
        if not _same(encoded, nodata):
            raise ValueError('nodata is not exactly representable in dtype')
    if type(tile_size) is not int or tile_size not in (128, 256, 512, 1024, 2048):
        raise ValueError('tile_size must be 128, 256, 512, 1024 or 2048')
    if type(deflate_level) is not int or not 1 <= deflate_level <= 9:
        raise ValueError('deflate_level must be an integer from 1 to 9')
    if (type(memory_limit) not in (int, float)
            or not 1 / 1024**2 <= memory_limit <= (2**64 - 1) // 1024**2):
        raise ValueError('memory_limit must be a finite positive number of MiB within the supported range')
    # Keep the public option in MiB; native writer and cache accounting use bytes.
    memory_limit_bytes = int(memory_limit * 1024**2)
    if isinstance(overviews, str):
        if overviews not in ('auto', 'none'):
            raise ValueError('overviews must be auto, none, or increasing powers of two')
        levels = []
        factor = 1
        if overviews == 'auto':
            while (max(len(x), len(y)) + factor - 1) // factor > tile_size:
                factor *= 2
                levels.append(factor)
    else:
        levels = list(overviews)
    # TIFF tag 42112 uses XML for band descriptions; ElementTree escapes names.
    root = Element('GDALMetadata')
    for i, name in enumerate(names):
        SubElement(root, 'Item', name='DESCRIPTION', sample=str(i), role='description').text = str(name)
    # Normalize to eastward columns and southward rows without resampling pixels.
    orientation = {
        x_dim: slice(None, None, -1) if dx < 0 else slice(None),
        y_dim: slice(None, None, -1) if dy > 0 else slice(None),
    }
    cube = None
    if isinstance(obj, xr.DataArray) and band_dim is not None:
        # Keep bands together so one storage chunk can serve several bands.
        cube = obj.transpose(band_dim, y_dim, x_dim).isel(orientation)
    else:
        arrays = [da.transpose(y_dim, x_dim).isel(orientation) for da in arrays]
    cache = _SourceCache()
    reporter = _Progress(((len(y) + tile_size - 1) // tile_size)
                         * ((len(x) + tile_size - 1) // tile_size)) if progress else None

    def read_tiles(windows: Sequence[tuple[int, int]]) -> list[NDArray[Any]]:
        """Read one bounded batch, sharing lazy tasks across its windows.

        Args:
            windows: Row/column offsets in output order.

        Returns:
            Interleaved NumPy tiles, including smaller tiles at image edges.
        """
        bands = []
        for band, (da, source) in enumerate(zip(arrays, sources)):
            values = (cache.read_many(cube, windows, tile_size, band=band) if cube is not None
                      else cache.read_many(da, windows, tile_size))
            for i, value in enumerate(values):
                missing = np.isnan(value)
                if source is not None:
                    missing |= value == source
                if nodata is not None:
                    if np.any((value == nodata) & ~missing):
                        raise ValueError('valid samples collide with destination nodata')
                    values[i] = np.where(missing, nodata, value).astype(dtype, copy=False)
            bands.append(values)
        return [np.stack(parts, axis=-1) for parts in zip(*bands)]

    # GeoTIFF locates the outer corner, while xarray coordinates are pixel centers.
    config = dict(width=len(x), height=len(y), bands=len(arrays), dtype=dtype.name,
                  epsg=epsg, geographic=resolved.is_geographic,
                  origin=[float(min(x) - abs(dx) / 2), float(max(y) + abs(dy) / 2)],
                  scale=[float(abs(dx)), float(abs(dy))],
                  nodata=None if nodata is None else str(nodata), tile_size=tile_size,
                  compression=compression, deflate_level=deflate_level,
                  resampling=resampling, overviews=levels,
                  overwrite=overwrite, memory_limit=memory_limit_bytes, colorinterp=colorinterp,
                  metadata=tostring(root, encoding='unicode'), num_threads=num_threads)
    path = Path(path)
    try:
        with NativeWriter(path, json.dumps(config, allow_nan=False)) as writer:
            cache.set_limit(writer.source_cache_budget)
            completed = 0
            if reporter is not None:
                reporter.update(0)
            for row in range(0, len(y), tile_size):
                for col in range(0, len(x), tile_size * writer.batch_size):
                    windows = [(row, x) for x in range(
                        col, min(len(x), col + tile_size * writer.batch_size), tile_size)]
                    batch = read_tiles(windows)
                    writer.write_tiles(batch)
                    # Release the previous batch before reading the next one.
                    del batch
                    completed += len(windows)
                    if reporter is not None:
                        reporter.update(completed)
            if reporter is not None:
                reporter.update(completed, 'Finalizing')
            writer.finish()
    except BaseException:
        if reporter is not None:
            reporter.finish(False)
        raise
    else:
        if reporter is not None:
            reporter.finish(True)
    finally:
        # Tracebacks can outlive the export, so release retained chunks explicitly.
        cache.clear()
    return path


@xr.register_dataarray_accessor('cog')
@xr.register_dataset_accessor('cog')
class _CogAccessor:
    """Expose COG export through an xarray object's .cog accessor.

    Args:
        obj: DataArray or Dataset associated with this accessor.
    """
    def __init__(self, obj: _Raster) -> None:
        self._obj = obj

    def to_cog(
        self,
        path: str | PathLike[str],
        *,
        x_dim: str | None = None,
        y_dim: str | None = None,
        band_dim: str | None = None,
        crs: object | None = None,
        nodata: _NoData | None = None,
        tile_size: int = 512,
        compression: Literal['deflate', 'none'] = 'deflate',
        resampling: Literal['nearest', 'average'] = 'nearest',
        overviews: Literal['auto', 'none'] | Iterable[int] = 'auto',
        overwrite: bool = False,
        memory_limit: int | float = 512,
        colorinterp: Literal['undefined', 'rgb'] = 'undefined',
        deflate_level: int = 7,
        progress: bool = False,
        num_threads: int | None = None,
    ) -> Path:
        """Write this xarray object to a local Cloud Optimized GeoTIFF.

        Args:
            path: Destination filesystem path.
            x_dim: Horizontal dimension name.
            y_dim: Vertical dimension name.
            band_dim: Optional DataArray band dimension name.
            crs: CRS override without reprojection.
            nodata: Output missing-value sentinel.
            tile_size: Output tile width and height in pixels.
            compression: Output compression mode.
            resampling: Overview resampling method.
            overviews: Automatic, disabled, or explicit overview factors.
            overwrite: Replace an existing destination only on success.
            memory_limit: Writer and retained-source-cache budget in MiB.
                Defaults to 512; accepts integers or floats.
            colorinterp: Numeric or RGB band interpretation.
            deflate_level: DEFLATE compression level.
            progress: Show tile and finalization progress on stderr.
            num_threads: Native compression worker ceiling, or automatic.

        Defaults and supported values match xarray_cog.write_cog.

        Returns:
            pathlib.Path: The destination path after successful export.

        Raises:
            TypeError: An export argument has an unsupported type.
            ValueError: Raster metadata or export options are invalid.
            pyproj.exceptions.CRSError: A CRS definition cannot be interpreted.
            FileExistsError: The destination exists and overwrite is False.
            OSError: Output or temporary-file I/O fails.

        Input-reader exceptions also propagate. See xarray_cog.write_cog for
        option defaults and validation details.
        """
        return write_cog(
            self._obj, path, x_dim=x_dim, y_dim=y_dim, band_dim=band_dim, crs=crs,
            nodata=nodata, tile_size=tile_size, compression=compression,
            resampling=resampling, overviews=overviews, overwrite=overwrite,
            memory_limit=memory_limit, colorinterp=colorinterp,
            deflate_level=deflate_level, progress=progress, num_threads=num_threads,
        )
