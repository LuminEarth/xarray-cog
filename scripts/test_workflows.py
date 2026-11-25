"""Run with python3 -m unittest discover -s scripts -p 'test_*.py'."""
import importlib.util
import json
from pathlib import Path
import re
import struct
import subprocess
import sys
import tempfile
import unittest
import zlib

ROOT = Path(__file__).resolve().parents[1]

spec = importlib.util.spec_from_file_location('benchmark', ROOT / 'scripts/benchmark.py')
assert spec is not None and spec.loader is not None
benchmark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark)


class SampleFixtureTests(unittest.TestCase):
    def test_sample_has_known_pixels_coordinates_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'sample.zarr'
            benchmark.create_sample(source, 16)
            metadata = json.loads((source / 'Band1/.zarray').read_text())
            self.assertEqual(metadata['shape'], [16, 16])
            values = struct.unpack('<256f', zlib.decompress((source / 'Band1/0.0').read_bytes()))
            self.assertEqual(values[1:4], (26.0, 33.0, 40.0))
            attrs = json.loads((source / 'Band1/.zattrs').read_text())
            self.assertEqual(attrs['_ARRAY_DIMENSIONS'], ['y', 'x'])
            self.assertEqual(attrs['epsg'], 4326)
            with self.assertRaises(FileExistsError):
                benchmark.create_sample(source, 16)
            self.assertEqual(json.loads((source / 'Band1/.zarray').read_text()), metadata)

    def test_sample_rejects_too_small_size(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'too-small.zarr'
            with self.assertRaises(ValueError):
                benchmark.create_sample(source, 1)
            self.assertFalse(source.exists())


class BenchmarkCliTests(unittest.TestCase):
    def test_invalid_size_and_existing_results_are_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / 'new-results'
            command = [sys.executable, str(ROOT / 'scripts/benchmark.py')]
            result = subprocess.run(command + [str(target), '--size', '513'], capture_output=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertFalse(target.exists())
            sentinel = root / 'keep.txt'
            sentinel.write_text('preserve')
            result = subprocess.run(command + [str(root)], capture_output=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(sentinel.read_text(), 'preserve')
            self.assertEqual(list(root.iterdir()), [sentinel])

    def test_help_lists_user_options(self):
        result = subprocess.run(
            [sys.executable, str(ROOT / 'scripts/benchmark.py'), '--help'],
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        for flag in ('--size', '--repeats', '--threads', '--tools', '--modes'):
            self.assertIn(flag, result.stdout)

    def test_format_summary_table(self):
        table = benchmark.format_summary_table([
            {
                'mode': 'numpy',
                'tool': 'xarray_cog',
                'median_export_seconds': 1.9824,
                'max_rss_mib': 1770.91,
                'output_mib': 11.0352,
            }
        ])
        lines = table.splitlines()
        self.assertEqual(len(lines), 3)
        self.assertIn('Mode', lines[0])
        self.assertIn('Peak RSS (MiB)', lines[0])
        self.assertIn('numpy', lines[2])
        self.assertIn('xarray_cog', lines[2])
        self.assertIn('1.982', lines[2])
        self.assertIn('1770.9', lines[2])
        self.assertIn('11.035', lines[2])

    @unittest.skipIf(sys.version_info < (3, 11), 'tomllib requires Python 3.11+')
    def test_benchmark_dependency_group_in_pyproject(self):
        import tomllib

        pyproject = tomllib.loads((ROOT / 'pyproject.toml').read_text())
        group = pyproject['dependency-groups']['benchmark']
        names = {re.split(r'[\s\[<>=!~;]', req, maxsplit=1)[0].lower() for req in group}
        expected = {
            'numpy', 'xarray', 'dask', 'zarr', 'h5netcdf',
            'h5py', 'rasterio', 'rioxarray', 'odc-geo', 'tifffile',
        }
        self.assertEqual(names, expected)


if __name__ == '__main__':
    unittest.main()
