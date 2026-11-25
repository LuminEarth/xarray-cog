"""Bounded, export-local reuse of computed input chunks."""
from __future__ import annotations

from collections import OrderedDict
from collections.abc import Hashable, Sequence
from itertools import product
from typing import Any, cast

import numpy as np
import xarray as xr
from numpy.typing import NDArray


class _SourceCache:
    """Retain owned arrays within the budget supplied by the native writer.

    Cache entries include a small bookkeeping allowance. Reader and scheduler
    working allocations are separate from this retained-memory estimate.
    """

    def __init__(self) -> None:
        self.limit = 0
        self.bytes = 0
        self._entries: OrderedDict[Hashable, NDArray[Any]] = OrderedDict()

    def set_limit(self, limit: int) -> None:
        """Set available bytes after reserving the native writer's buffers.

        Args:
            limit: Nonnegative cache capacity in bytes.
        """
        self.limit = limit
        while self.bytes > limit:
            self._evict()

    def _evict(self) -> None:
        _, value = self._entries.popitem(last=False)
        self.bytes -= value.nbytes + 512

    def _get(self, key: Hashable) -> NDArray[Any] | None:
        value = self._entries.get(key)
        if value is not None:
            self._entries.move_to_end(key)
        return value

    def _put(self, key: Hashable, value: NDArray[Any]) -> None:
        # Object arrays retain Python payloads that nbytes cannot account for.
        if value.dtype.hasobject:
            return
        cost = value.nbytes + 512
        if cost > self.limit:
            return
        old = self._entries.pop(key, None)
        if old is not None:
            self.bytes -= old.nbytes + 512
        while self.bytes + cost > self.limit:
            self._evict()
        # Own the retained bytes: a small view could otherwise keep a much larger
        # parent alive, and a downstream Dask task could mutate its input.
        self._entries[key] = np.array(value, copy=True, order='C')
        self.bytes += cost

    def clear(self) -> None:
        """Release retained arrays, including on interrupted exports."""
        self._entries.clear()
        self.bytes = 0

    def read(self, array: xr.DataArray, row: int, col: int, size: int,
             band: int | None = None) -> NDArray[Any]:
        """Read a tile, reusing available source chunks across calls and bands.

        Args:
            array: DataArray in (y, x) or (band, y, x) output axis order.
            row: First output row.
            col: First output column.
            size: Output tile width and height.
            band: Optional band index for a three-dimensional array.

        Returns:
            A NumPy array containing the tile's current decoded values.

        Raises:
            ValueError: The configured Dask scheduler is distributed. Source
                exceptions propagate to the caller.
        """
        y_dim, x_dim = array.dims[-2:]
        height, width = array.shape[-2:]
        ys, xs = slice(row, min(row + size, height)), slice(col, min(col + size, width))
        selection: dict[Hashable, slice | int] = {y_dim: ys, x_dim: xs}
        if band is not None:
            selection[array.dims[0]] = band
        tile = array.isel(selection)
        if array.chunks is not None:
            return self.read_many(array, [(row, col)], size, band=band)[0]
        if not self.limit:
            return tile.to_numpy()

        chunks = array.encoding.get('preferred_chunks', {})
        cy, cx = chunks.get(y_dim), chunks.get(x_dim)
        if not isinstance(cy, int) or not isinstance(cx, int) or cy <= 0 or cx <= 0:
            return tile.to_numpy()
        cb = chunks.get(array.dims[0], 1) if band is not None else 1
        if not isinstance(cb, int) or cb <= 0:
            return tile.to_numpy()
        chunk_bytes = min(cy, height) * min(cx, width) * array.dtype.itemsize
        if band is not None:
            chunk_bytes *= min(cb, array.shape[0])
        if chunk_bytes + 512 > self.limit:
            # Do not expand a backend read to a full chunk we cannot retain.
            # The backend may still need its own decoding workspace.
            return tile.to_numpy()
        b0 = band // cb * cb if band is not None else None
        output = np.empty((ys.stop - row, xs.stop - col), dtype=array.dtype)
        for y in range(row // cy * cy, ys.stop, cy):
            for x in range(col // cx * cx, xs.stop, cx):
                key = (self, id(array), b0, y, x)
                values = self._get(key)
                if values is None:
                    selection = {y_dim: slice(y, y + cy), x_dim: slice(x, x + cx)}
                    if band is not None:
                        selection[array.dims[0]] = slice(b0, cast(int, b0) + cb)
                    values = array.isel(selection).to_numpy()
                    self._put(key, values)
                if band is not None:
                    values = values[band - cast(int, b0)]
                y0, y1 = max(row, y), min(ys.stop, y + cy)
                x0, x1 = max(col, x), min(xs.stop, x + cx)
                output[y0 - row:y1 - row, x0 - col:x1 - col] = values[y0 - y:y1 - y, x0 - x:x1 - x]
        return output

    def read_many(self, array: xr.DataArray, windows: Sequence[tuple[int, int]],
                  size: int, band: int | None = None) -> list[NDArray[Any]]:
        """Compute a batch's windows together while retaining reusable chunks.

        Args:
            array: Normalized DataArray in output axis order.
            windows: Row/column offsets for the requested tiles.
            size: Output tile width and height.
            band: Optional band index for a three-dimensional array.

        Returns:
            NumPy tiles in the same order as windows.

        Raises:
            ValueError: A distributed Dask scheduler is configured.
        """
        if array.chunks is None:
            return [self.read(array, row, col, size, band) for row, col in windows]
        import dask
        from dask.base import get_scheduler

        data = array.data
        scheduler = get_scheduler(collections=[data])
        if getattr(scheduler, '__module__', '').startswith('distributed'):
            raise ValueError('source chunk reuse requires a local Dask scheduler; '
                             'select scheduler="threads" or "synchronous"')
        # Work with declared chunks, including the full band extent of a chunk.
        # Dask owns graph optimization; only its final chunk results enter our LRU.
        bounds = [np.concatenate(([0], np.cumsum(axis))) for axis in data.chunks]
        requests = []
        needed = {}
        for row, col in windows:
            starts = ([band] if band is not None else []) + [row, col]
            stops = ([band + 1] if band is not None else []) + [
                min(row + size, array.shape[-2]), min(col + size, array.shape[-1])]
            indices = [range(int(np.searchsorted(edges, start, side='right') - 1),
                             int(np.searchsorted(edges, stop, side='left')))
                       for edges, start, stop in zip(bounds, starts, stops)]
            chunks = list(product(*indices))
            requests.append((starts, stops, chunks))
            for index in chunks:
                needed[index] = (data.name, *index)
        values = {index: self._get(key) for index, key in needed.items()}
        missing = [index for index, value in values.items() if value is None]
        if missing:
            # Compute a batch together so Dask can share dependencies within it.
            # Batch results remain live for assembly even if the LRU evicts them.
            computed = dask.compute(*(data.blocks[index] for index in missing),
                                    scheduler=scheduler)
            for index, value in zip(missing, computed):
                values[index] = value
                self._put(needed[index], value)
        outputs = []
        for starts, stops, chunks in requests:
            output = np.empty(tuple(stop - start for start, stop in zip(starts, stops)),
                              dtype=array.dtype)
            for index in chunks:
                origins = [int(edges[i]) for edges, i in zip(bounds, index)]
                ends = [int(edges[i + 1]) for edges, i in zip(bounds, index)]
                lo = [max(start, origin) for start, origin in zip(starts, origins)]
                hi = [min(stop, end) for stop, end in zip(stops, ends)]
                target = tuple(slice(a - start, b - start)
                               for a, b, start in zip(lo, hi, starts))
                source = tuple(slice(a - origin, b - origin)
                               for a, b, origin in zip(lo, hi, origins))
                output[target] = cast(NDArray[Any], values[index])[source]
            outputs.append(output[0] if band is not None else output)
        return outputs
