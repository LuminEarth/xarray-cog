"""Check that a build produced the complete, consistently versioned release set."""
from pathlib import Path
import sys


def validate_distributions(directory: Path, version: str, tag: str = '') -> None:
    if tag and tag != f'v{version}':
        raise ValueError(f'Tag {tag} does not match package version {version}')
    wheels = list(directory.glob('*.whl'))
    sources = list(directory.glob('*.tar.gz'))
    if len(wheels) != 20 or len(sources) != 1:
        raise ValueError(f'Expected 20 wheels and 1 source archive; found {len(wheels)} and {len(sources)}')
    if any(not path.name.startswith(f'xarray_cog-{version}-') for path in wheels):
        raise ValueError('Wheel version or package name does not match')
    if sources[0].name != f'xarray_cog-{version}.tar.gz':
        raise ValueError('Source archive version or package name does not match')
    if len(list(directory.iterdir())) != 21:
        raise ValueError('Unexpected files in the release directory')


if __name__ == '__main__':
    validate_distributions(Path(sys.argv[1]), sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else '')
    print('Validated 20 wheels and 1 source archive')
