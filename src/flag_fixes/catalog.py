"""Read a pinned external catalog and in-repository fix descriptions."""
from __future__ import annotations
import hashlib
import json
import re
import subprocess
from pathlib import Path, PurePosixPath
import jsonschema
import yaml


class CatalogError(ValueError):
    pass


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def safe_path(base: Path, relative: str) -> Path:
    p = PurePosixPath(relative)
    if not relative or p.is_absolute() or '..' in p.parts or '\\' in relative or p.as_posix() != relative or relative == '.':
        raise CatalogError(f'Unsafe relative path: {relative!r}')
    current = base
    for part in p.parts:
        current = current / part
        if current.is_symlink():
            raise CatalogError(f'Symlink is not permitted: {current}')
    if not current.resolve().is_relative_to(base.resolve()):
        raise CatalogError(f'Path escapes root: {relative}')
    return current


def git(path: Path, *args: str) -> str:
    result = subprocess.run(['git', '-C', str(path), *args], text=True, capture_output=True)
    if result.returncode:
        raise CatalogError(f'Git check failed at {path}: {result.stderr.strip()}')
    return result.stdout.strip()


def load_catalog(root: Path) -> dict[str, dict]:
    schema = json.loads((root/'catalog/schemas/fix.schema.json').read_text())
    result = {}
    for file in sorted([*(root/'catalog/fixes').glob('*.yaml'), *(root/'fixes').glob('*/fix.yaml')]):
        if file.is_symlink() or file.parent.is_symlink():
            raise CatalogError(f'Symlink manifest: {file}')
        manifest = yaml.safe_load(file.read_text())
        try:
            jsonschema.Draft202012Validator(schema).validate(manifest)
        except jsonschema.ValidationError as error:
            raise CatalogError(f'{file.relative_to(root)}: {error.message}') from error
        ident = manifest['id']
        if ident in result:
            raise CatalogError(f'Duplicate fix ID: {ident}')
        source = manifest['source']
        safe_path(root, source['path'])
        expected = f'fixes/{ident}'
        if source['kind'] == 'local' and (source['path'] != expected or file != root/expected/'fix.yaml'):
            raise CatalogError(f'{ident}: local fix must be discovered at {expected}/fix.yaml')
        if source['kind'] == 'local' and source['commit'] is not None:
            raise CatalogError(f'{ident}: local provenance is the enclosing repository commit')
        if source['kind'] == 'submodule' and (not source['path'].startswith('third_party/') or not source['commit']):
            raise CatalogError(f'{ident}: external fix requires a pinned third_party submodule')
        paths = [p['path'] for p in manifest['targets']]
        if len(paths) != len(set(paths)):
            raise CatalogError(f'{ident}: duplicate target path')
        result[ident] = manifest
    for ident, manifest in result.items():
        for related in manifest['requires'] + manifest['conflicts']:
            if related not in result or related == ident:
                raise CatalogError(f'{ident}: invalid relationship {related}')
    return result


def verify_source(root: Path, manifest: dict) -> Path:
    source = manifest['source']
    directory = safe_path(root, source['path'])
    if not directory.is_dir():
        raise CatalogError(f'Missing source {directory}; clone with --recurse-submodules')
    if source['kind'] == 'submodule':
        # Check BOTH the parent gitlink and the checkout, including reviewed staged pin updates.
        entry = git(root, 'ls-files', '--stage', '--', source['path']).split()
        if len(entry) < 3 or entry[0] != '160000' or entry[1] != source['commit']:
            raise CatalogError(f'{manifest["id"]}: manifest pin differs from parent gitlink')
        if git(directory, 'rev-parse', 'HEAD') != source['commit']:
            raise CatalogError(f'{manifest["id"]}: submodule HEAD differs from manifest pin')
        if git(directory, 'status', '--porcelain', '--untracked-files=all'):
            raise CatalogError(f'{manifest["id"]}: submodule has local modifications')
    elif manifest['deployable']:
        # A deployable local fix belongs to an immutable enclosing repository commit.
        git(root, 'rev-parse', 'HEAD')
        tracked = git(root, 'ls-files', '--', source['path'] + '/fix.yaml')
        if not tracked or git(root, 'status', '--porcelain', '--untracked-files=all', '--', source['path']):
            raise CatalogError(f'{manifest["id"]}: commit the complete local fix before deployment')
    for patch in manifest['patches']:
        file = safe_path(directory, patch['path'])
        if not file.is_file() or sha256(file) != patch['sha256']:
            raise CatalogError(f'{manifest["id"]}: patch hash mismatch: {patch["path"]}')
    for target in manifest['targets']:
        safe_path(root, target['path'])
        for key, hashkey in [('before_source', 'before_sha256'), ('after_source', 'after_sha256')]:
            if key in target:
                file = safe_path(directory, target[key])
                if not file.is_file() or sha256(file) != target[hashkey]:
                    raise CatalogError(f'{manifest["id"]}: source snapshot mismatch: {target[key]}')
    for command in manifest['validation']['commands']:
        if not safe_path(directory, command['script']).is_file():
            raise CatalogError(f'{manifest["id"]}: missing verification script')
    return directory


def validate(root: Path) -> dict:
    catalog = load_catalog(root)
    for manifest in catalog.values():
        verify_source(root, manifest)
    return {'valid': True, 'fixes': len(catalog),
            'deployable': sum(m['deployable'] for m in catalog.values()),
            'hardware_verified': sum(m['validation']['hardware_status'] == 'passed' for m in catalog.values())}


def select(catalog: dict[str, dict], ids: list[str]) -> list[dict]:
    if len(set(ids)) != len(ids):
        raise CatalogError('Repeated fix IDs')
    for ident in ids:
        if ident not in catalog:
            raise CatalogError(f'Unknown fix: {ident}')
    chosen = set(ids)
    for ident in ids:
        m = catalog[ident]
        if not set(m['requires']).issubset(chosen):
            raise CatalogError(f'{ident}: include required fixes: {m["requires"]}')
        if chosen.intersection(m['conflicts']):
            raise CatalogError(f'{ident}: conflicting selection')
    # Ordered dependency traversal; no silent auto-installation.
    output, visiting, done = [], set(), set()
    def visit(ident):
        if ident in visiting:
            raise CatalogError('Dependency cycle')
        if ident in done:
            return
        visiting.add(ident)
        for dep in catalog[ident]['requires']:
            visit(dep)
        visiting.remove(ident)
        done.add(ident)
        output.append(catalog[ident])
    for ident in ids:
        visit(ident)
    return output
