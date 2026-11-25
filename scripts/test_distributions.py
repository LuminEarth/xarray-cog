"""Validate complete release sets without uploading packages."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location('check_distributions', Path(__file__).with_name('check_distributions.py'))
checks = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checks)


class DistributionTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        for python in ('310', '311', '312', '313', '314'):
            for platform in ('manylinux_2_28_x86_64', 'manylinux_2_28_aarch64', 'macosx_13_0_x86_64', 'macosx_13_0_arm64'):
                (self.root / f'xarray_cog-0.1.0-cp{python}-cp{python}-{platform}.whl').touch()
        (self.root / 'xarray_cog-0.1.0.tar.gz').touch()

    def test_accepts_complete_tagged_and_manual_builds(self):
        checks.validate_distributions(self.root, '0.1.0', 'v0.1.0')
        checks.validate_distributions(self.root, '0.1.0')

    def test_rejects_wrong_tag_or_version(self):
        for version, tag in [('0.1.0', 'v0.2.0'), ('0.2.0', '')]:
            with self.subTest(version=version, tag=tag), self.assertRaises(ValueError):
                checks.validate_distributions(self.root, version, tag)

    def test_rejects_incomplete_wheel_set(self):
        next(self.root.glob('*.whl')).unlink()
        with self.assertRaisesRegex(ValueError, 'Expected 20 wheels'):
            checks.validate_distributions(self.root, '0.1.0')

    def test_rejects_missing_or_wrong_source_archive(self):
        source = self.root / 'xarray_cog-0.1.0.tar.gz'
        source.rename(self.root / 'xarray_cog-0.2.0.tar.gz')
        with self.assertRaisesRegex(ValueError, 'Source archive'):
            checks.validate_distributions(self.root, '0.1.0')
        next(self.root.glob('*.tar.gz')).unlink()
        with self.assertRaisesRegex(ValueError, '1 source archive'):
            checks.validate_distributions(self.root, '0.1.0')

    def test_rejects_unexpected_upload_files(self):
        (self.root / 'unexpected.zip').touch()
        with self.assertRaisesRegex(ValueError, 'Unexpected files'):
            checks.validate_distributions(self.root, '0.1.0')


if __name__ == '__main__':
    unittest.main()
