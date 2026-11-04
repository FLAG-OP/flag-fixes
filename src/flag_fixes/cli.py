"""Command line interface. Target environments are always selected explicitly."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import yaml
from . import catalog, installer


def find_root(value=None):
    if value:
        root = Path(value).expanduser().resolve()
        if not (root/'catalog/schemas/fix.schema.json').is_file():
            raise catalog.CatalogError(f'Not a flag-fixes checkout: {root}')
        return root
    candidates = [Path.cwd(), *Path.cwd().parents, Path(__file__).resolve().parents[2]]
    for root in candidates:
        if (root/'catalog/schemas/fix.schema.json').is_file():
            return root
    raise catalog.CatalogError('Use --root /path/to/flag-fixes checkout')


def new_fix(root: Path, ident: str, title: str):
    if not re.fullmatch(r'[a-z][a-z0-9]*(?:-[a-z0-9]+)*', ident):
        raise catalog.CatalogError('Fix ID must use lowercase letters, numbers and hyphens')
    if ident in catalog.load_catalog(root):
        raise catalog.CatalogError(f'Fix ID already exists: {ident}')
    destination = catalog.safe_path(root, f'fixes/{ident}')
    if destination.exists():
        raise catalog.CatalogError(f'Destination already exists: {destination}')
    template = root/'templates/fix'
    if template.is_symlink() or any(p.is_symlink() for p in template.rglob('*')):
        raise catalog.CatalogError('Template contains a symlink')
    manifest = yaml.safe_load((template/'fix.yaml').read_text())
    manifest.update(id=ident, title=title)
    manifest['source']['path'] = f'fixes/{ident}'
    shutil.copytree(template, destination)
    for path in destination.rglob('*.md'):
        path.write_text(path.read_text().replace('FIX_ID', ident).replace('FIX_TITLE', title))
    (destination/'fix.yaml').write_text(yaml.safe_dump(manifest, allow_unicode=True, sort_keys=False))
    catalog.load_catalog(root)
    return {'created': f'fixes/{ident}', 'deployable': False}


def verify(root, target, manifest, python, timeout=None):
    source = catalog.verify_source(root, manifest)
    provenance = {'commit': manifest['source']['commit'], 'dirty': False}
    if manifest['source']['kind'] == 'local':
        try:
            provenance = {'commit': catalog.git(root, 'rev-parse', 'HEAD'),
                          'dirty': bool(catalog.git(root, 'status', '--porcelain', '--untracked-files=all',
                                                    '--', manifest['source']['path']))}
        except catalog.CatalogError:
            provenance = {'commit': None, 'dirty': True}
    commands = manifest['validation']['commands']
    if not commands:
        raise catalog.CatalogError('No reviewed verification commands; see source reports and complete the manifest')
    if not manifest['targets']:
        raise catalog.CatalogError('No target hashes; cannot verify deployed contents')
    for item in manifest['targets']:
        path = catalog.safe_path(target, item['path'])
        if not path.is_file() or catalog.sha256(path) != item['after_sha256']:
            raise catalog.CatalogError(f'Verification requires exact patched contents: {item["path"]}')
    executable = shutil.which(python)
    if not executable:
        raise catalog.CatalogError(f'Python interpreter not found: {python}')
    records = []
    for command in commands:
        script = catalog.safe_path(source, command['script'])
        env = os.environ.copy()
        for key, value in command.get('env', {}).items():
            env[key] = value.replace('{target}', str(target))
        env['PYTHONPATH'] = str(target/'src') + (os.pathsep+env['PYTHONPATH'] if env.get('PYTHONPATH') else '')
        argv = [executable, str(script), *command.get('args', [])]
        record = {'name': command['name'], 'level': command['level'], 'argv': argv}
        try:
            result = subprocess.run(argv, cwd=source, env=env, text=True, capture_output=True,
                                    timeout=timeout or command.get('timeout_seconds', 600))
            record.update(returncode=result.returncode, passed=result.returncode == 0,
                          stdout=result.stdout, stderr=result.stderr)
        except subprocess.TimeoutExpired:
            record.update(returncode=None, passed=False, error='timeout')
        records.append(record)
        if not record['passed']:
            break
    return {'fix': manifest['id'], 'passed': all(r['passed'] for r in records),
            'scope': 'declared commands only; does not promote catalog hardware status',
            'target': str(target), 'source_commit': provenance['commit'],
            'source_dirty': provenance['dirty'], 'results': records}


def parser():
    p = argparse.ArgumentParser(description='FLAG-OP pinned fix catalog and source-tree patch manager')
    p.add_argument('--root', help='flag-fixes repository root (required for standalone wheel use)')
    sub = p.add_subparsers(dest='command', required=True)
    sub.add_parser('list', help='List external and local fixes')
    show = sub.add_parser('show'); show.add_argument('id')
    sub.add_parser('validate', help='Check schemas, source pins, patch hashes and snapshots')
    new = sub.add_parser('new', help='Scaffold an in-repository fix')
    new.add_argument('id'); new.add_argument('--title', required=True)
    for name in ('check', 'plan', 'apply'):
        c = sub.add_parser(name); c.add_argument('ids', nargs='+'); c.add_argument('--target', required=True)
    for name in ('status', 'rollback'):
        c = sub.add_parser(name); c.add_argument('--target', required=True)
    v = sub.add_parser('verify'); v.add_argument('id'); v.add_argument('--target', required=True)
    v.add_argument('--python', required=True, help='Target environment Python executable')
    v.add_argument('--timeout', type=int, help='Positive timeout in seconds per command')
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        if args.command in ('status', 'rollback'):
            # Reject symlink target paths in installer; do not resolve them away here.
            result = getattr(installer, args.command)(Path(args.target).expanduser().absolute())
        else:
            root = find_root(args.root)
            manifests = catalog.load_catalog(root)
            if args.command == 'list':
                result = [{'id': m['id'], 'title': m['title'], 'source': m['source']['kind'],
                           'deployable': m['deployable'], 'hardware_status': m['validation']['hardware_status'],
                           'installation_status': m.get('installation_status', 'draft')} for m in manifests.values()]
            elif args.command == 'show':
                if args.id not in manifests:
                    raise catalog.CatalogError(f'Unknown fix: {args.id}')
                result = manifests[args.id]
            elif args.command == 'validate':
                result = catalog.validate(root)
            elif args.command == 'new':
                result = new_fix(root, args.id, args.title)
            elif args.command == 'verify':
                if args.timeout is not None and args.timeout <= 0:
                    raise catalog.CatalogError('Timeout must be positive')
                if args.id not in manifests:
                    raise catalog.CatalogError(f'Unknown fix: {args.id}')
                result = verify(root, Path(args.target).expanduser().resolve(),
                                manifests[args.id], args.python, args.timeout)
            else:
                chosen = catalog.select(manifests, args.ids)
                for m in chosen:
                    catalog.verify_source(root, m)
                operation = installer.apply if args.command == 'apply' else installer.plan
                result = operation(root, Path(args.target).expanduser().absolute(), chosen)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 1 if isinstance(result, dict) and result.get('passed') is False else 0
    except (ValueError, OSError, yaml.YAMLError) as error:
        print(f'error: {error}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
