"""Exact-content patch installation into an explicit source tree (Unix)."""
from __future__ import annotations

import contextlib
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
import uuid


class InstallError(ValueError):
    pass


def _hash(data):
    return hashlib.sha256(data).hexdigest()


def _path(base, name):
    p = Path(name)
    if not name or p.is_absolute() or '..' in p.parts or any(part in {'.git', '.flag-fixes-state'} for part in p.parts):
        raise InstallError(f'Unsafe path: {name}')
    base = Path(base).absolute()
    current = base
    if base.is_symlink() or base.resolve() != base:
        raise InstallError(f'Symlink root: {base}')
    for part in p.parts:
        current /= part
        if current.is_symlink():
            raise InstallError(f'Symlink path: {current}')
    if not current.is_relative_to(base):
        raise InstallError(f'Escaping path: {name}')
    return current


def _run(args, cwd):
    r = subprocess.run(args, cwd=cwd, capture_output=True, text=True)
    if r.returncode:
        raise InstallError(r.stderr.strip() or r.stdout.strip())
    return r.stdout.strip()


def _atomic(path, data):
    fd, tmp = tempfile.mkstemp(prefix='.flag-fixes-', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        if path.exists():
            os.chmod(tmp, path.stat().st_mode & 0o777)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def _clear_bytecode(path):
    if path.suffix != '.py':
        return
    cache = path.parent / '__pycache__'
    if cache.is_symlink():
        raise InstallError(f'Symlink cache directory: {cache}')
    if cache.is_dir():
        for entry in cache.glob(path.stem + '.*.pyc'):
            if entry.is_file() or entry.is_symlink():
                entry.unlink()


def _json(path, value):
    _atomic(path, (json.dumps(value, indent=2) + '\n').encode())


def _state(target, create=False):
    target = Path(target).absolute()
    if not target.is_dir() or target.is_symlink() or target.resolve() != target:
        raise InstallError('Target must be an existing non-symlink source root')
    state = target / '.flag-fixes-state'
    if state.is_symlink():
        raise InstallError('State directory cannot be a symlink')
    if create:
        state.mkdir(exist_ok=True, mode=0o700)
    if state.exists():
        for p in state.rglob('*'):
            if p.is_symlink():
                raise InstallError('State contains a symlink')
    return state


def _read(state):
    p = state / 'installed.json'
    return json.loads(p.read_text()) if p.exists() else []


@contextlib.contextmanager
def _lock(target):
    state = _state(target, True)
    with (state / 'lock').open('a') as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        yield state


def _recover(target, state):
    journal = state / 'journal.json'
    if not journal.exists():
        return
    j = json.loads(journal.read_text())
    # Preflight the whole recovery: never overwrite edits made after interruption.
    for item in j['files']:
        p = _path(target, item['path'])
        if _hash(p.read_bytes()) not in (item['before'], item['after']):
            raise InstallError(f'Recovery conflict: {item["path"]}')
    for item in j['files']:
        backup = _path(state, item['backup'])
        data = backup.read_bytes()
        if _hash(data) != item['before']:
            raise InstallError('Recovery backup checksum mismatch')
    for item in j['files']:
        restored = _path(target, item['path'])
        _atomic(restored, _path(state, item['backup']).read_bytes())
        _clear_bytecode(restored)
    _json(state / 'installed.json', j['previous'])
    journal.unlink()


def _manifest_key(manifests):
    return _hash(json.dumps(manifests, sort_keys=True).encode())


def _prepare(root, target, manifests, stage):
    if not manifests:
        raise InstallError('No fixes selected')
    seen, patches, files = set(), [], []
    ids = [m['id'] for m in manifests]
    if len(ids) != len(set(ids)):
        raise InstallError('Duplicate fix IDs')
    for m in manifests:
        source = _path(root, m['source']['path'])
        commit = m['source'].get('commit')
        if commit:
            if _run(['git', 'rev-parse', 'HEAD'], source) != commit:
                raise InstallError(f'Source commit differs: {m["id"]}')
            if _run(['git', 'status', '--porcelain'], source):
                raise InstallError(f'Source has local changes: {m["id"]}')
        declared = {t['path'] for t in m['targets']}
        if not declared or not m['patches']:
            raise InstallError('Fix must declare patches and targets')
        if len(declared) != len(m['targets']):
            raise InstallError('Duplicate target paths')
        if seen & declared:
            raise InstallError('Overlapping targets unsupported in v1')
        seen |= declared
        for t in m['targets']:
            p = _path(target, t['path'])
            if p.suffix == '.py' and (p.parent / '__pycache__').is_symlink():
                raise InstallError('Symlink cache directory')
            data = p.read_bytes()
            if _hash(data) != t['before_sha256']:
                raise InstallError(f'Baseline checksum differs: {t["path"]}')
            dest = _path(stage, t['path'])
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(data)
            files.append(dict(t))
        touched = set()
        for patch in m['patches']:
            pp = _path(source, patch['path'])
            data = pp.read_bytes()
            if _hash(data) != patch['sha256']:
                raise InstallError('Patch checksum differs')
            if commit and _run(['git', 'status', '--porcelain', '--', patch['path']], source):
                raise InstallError('Declared patch has local changes')
            text = data.decode('utf-8')
            if re.search(r'^(?:new file mode|deleted file mode|rename |copy |GIT binary patch|Binary files|old mode|new mode)', text, re.M):
                raise InstallError('Only textual edits to existing files supported')
            old = re.findall(r'^--- a/([^\n]+)$', text, re.M)
            new = re.findall(r'^\+\+\+ b/([^\n]+)$', text, re.M)
            if not old or old != new or not set(old) <= declared:
                raise InstallError('Patch changes unlisted or unsupported paths')
            for name in old:
                _path(target, name)
            touched.update(old)
            # A temporary patch snapshot avoids reading mutable input during apply.
            with tempfile.NamedTemporaryFile() as f:
                f.write(data)
                f.flush()
                _run(['git', 'apply', '--check', f.name], stage)
                _run(['git', 'apply', f.name], stage)
            patches.append(patch['path'])
        if touched != declared:
            raise InstallError('Targets must exactly match patch edits')
    for t in files:
        if _hash(_path(stage, t['path']).read_bytes()) != t['after_sha256']:
            raise InstallError(f'Patched checksum differs: {t["path"]}')
    return {'fixes': ids, 'targets': files, 'patches': patches,
            'deployable': all(m.get('deployable') is True for m in manifests),
            'hardware_status': [m.get('validation', {}).get('hardware_status', 'unverified') for m in manifests]}


def plan(root: Path, target: Path, manifests: list[dict]) -> dict:
    _state(target)
    with tempfile.TemporaryDirectory() as d:
        return _prepare(Path(root), Path(target), manifests, Path(d))


def status(target: Path) -> dict:
    state = _state(target)
    records = _read(state)
    drift = []
    for record in records:
        for t in record['files']:
            p = _path(target, t['path'])
            if not p.exists() or _hash(p.read_bytes()) != t['after']:
                drift.append(t['path'])
    return {'transactions': records, 'drift': drift,
            'recovery_required': (state / 'journal.json').exists()}


def apply(root: Path, target: Path, manifests: list[dict]) -> dict:
    target = Path(target).absolute()
    if not manifests or not all(m.get('deployable') is True for m in manifests):
        raise InstallError('Selected fixes are not approved for deployment')
    with _lock(target) as state:
        _recover(target, state)
        records = _read(state)
        key = _manifest_key(manifests)
        for record in records:
            if record['key'] == key:
                if status(target)['drift']:
                    raise InstallError('Installed files have changed')
                return {'state': 'already-installed', 'transaction': record['id']}
        ids = {m['id'] for m in manifests}
        if any(ids.intersection(r['fixes']) for r in records):
            raise InstallError('Partial or changed installation already exists')
        active = {t['path'] for r in records for t in r['files']}
        selected = {t['path'] for m in manifests for t in m['targets']}
        if active & selected:
            raise InstallError('Target overlaps an installed fix')
        for record in records:
            if ids & set(record.get('conflicts', [])):
                raise InstallError('Installed fix conflicts with selected fix')
        available = ids | {fix for record in records for fix in record['fixes']}
        for manifest in manifests:
            if not set(manifest.get('requires', [])) <= available:
                raise InstallError('Required fixes are missing')
            if set(manifest.get('conflicts', [])) & available:
                raise InstallError('Selected fix conflicts with another fix')
        with tempfile.TemporaryDirectory() as d:
            stage = Path(d)
            result = _prepare(Path(root), target, manifests, stage)
            tx = uuid.uuid4().hex
            backup_dir = state / tx
            backup_dir.mkdir()
            files = []
            for i, t in enumerate(result['targets']):
                p = _path(target, t['path'])
                data = p.read_bytes()
                if _hash(data) != t['before_sha256']:
                    raise InstallError('Target changed during planning')
                backup = backup_dir / str(i)
                backup.write_bytes(data)
                files.append({'path': t['path'], 'before': t['before_sha256'],
                              'after': t['after_sha256'], 'backup': f'{tx}/{i}'})
            record = {'id': tx, 'key': key, 'fixes': result['fixes'], 'files': files,
                      'verification': 'not-run',
                      'conflicts': sorted({item for m in manifests for item in m.get('conflicts', [])})}
            _json(state / 'journal.json', {'files': files, 'previous': records})
            try:
                for t in files:
                    destination = _path(target, t['path'])
                    _atomic(destination, _path(stage, t['path']).read_bytes())
                    _clear_bytecode(destination)
                _json(state / 'installed.json', records + [record])
                (state / 'journal.json').unlink()
            except BaseException:
                _recover(target, state)
                raise
    return {'state': 'installed', 'transaction': tx, 'verification': 'not-run'}


def rollback(target: Path) -> dict:
    target = Path(target).absolute()
    with _lock(target) as state:
        _recover(target, state)
        records = _read(state)
        if not records:
            raise InstallError('No installation to roll back')
        record = records[-1]
        for t in record['files']:
            if _hash(_path(target, t['path']).read_bytes()) != t['after']:
                raise InstallError(f'Rollback conflict: {t["path"]}')
            if _hash(_path(state, t['backup']).read_bytes()) != t['before']:
                raise InstallError('Backup checksum differs')
        # Recovery of interrupted rollback completes the same rollback.
        _json(state / 'journal.json', {'files': record['files'], 'previous': records[:-1]})
        _recover(target, state)
    return {'state': 'rolled-back', 'transaction': record['id']}
