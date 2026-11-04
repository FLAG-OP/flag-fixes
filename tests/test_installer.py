import hashlib
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from flag_fixes import installer as i


class InstallerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.source = self.root / 'fix'
        self.source.mkdir()
        self.target = self.root / 'target'
        self.target.mkdir()
        self.file = self.target / 'sample.py'
        self.file.write_text('old\n')
        self.patch = self.source / 'fix.patch'
        self.patch.write_text('diff --git a/sample.py b/sample.py\n--- a/sample.py\n+++ b/sample.py\n@@ -1 +1 @@\n-old\n+new\n')
        h = lambda b: hashlib.sha256(b).hexdigest()
        self.m = {'id': 'example', 'deployable': True,
                  'source': {'path': 'fix', 'commit': None},
                  'patches': [{'path': 'fix.patch', 'sha256': h(self.patch.read_bytes())}],
                  'targets': [{'path': 'sample.py', 'before_sha256': h(b'old\n'), 'after_sha256': h(b'new\n')}],
                  'validation': {'hardware_status': 'unverified'}}

    def test_install_idempotence_rollback(self):
        self.assertTrue(i.plan(self.root, self.target, [self.m])['deployable'])
        self.assertEqual(i.apply(self.root, self.target, [self.m])['state'], 'installed')
        self.assertEqual(self.file.read_text(), 'new\n')
        self.assertEqual(i.apply(self.root, self.target, [self.m])['state'], 'already-installed')
        self.assertEqual(i.status(self.target)['drift'], [])
        i.rollback(self.target)
        self.assertEqual(self.file.read_text(), 'old\n')
        self.assertFalse(i.status(self.target)['transactions'])

    def test_clears_only_modified_module_bytecode(self):
        cache = self.target / '__pycache__'
        cache.mkdir()
        stale = cache / 'sample.cpython-312.pyc'
        stale.write_bytes(b'stale')
        other = cache / 'other.cpython-312.pyc'
        other.write_bytes(b'keep')
        i.apply(self.root, self.target, [self.m])
        self.assertFalse(stale.exists())
        self.assertTrue(other.exists())

    def test_empty_targets_rejected(self):
        self.m['targets'] = []
        self.m['patches'] = []
        with self.assertRaises(i.InstallError):
            i.plan(self.root, self.target, [self.m])

    def test_duplicate_targets_rejected(self):
        self.m['targets'].append(dict(self.m['targets'][0]))
        with self.assertRaises(i.InstallError):
            i.plan(self.root, self.target, [self.m])

    def test_installed_one_sided_conflict(self):
        self.m['conflicts'] = ['second']
        i.apply(self.root, self.target, [self.m])
        second = copy.deepcopy(self.m)
        second.update(id='second', conflicts=[])
        second['patches'][0]['path'] = 'second.patch'
        second['targets'][0]['path'] = 'second.py'
        patchfile = self.source / 'second.patch'
        patchfile.write_text(self.patch.read_text().replace('sample.py', 'second.py'))
        second['patches'][0]['sha256'] = hashlib.sha256(patchfile.read_bytes()).hexdigest()
        (self.target / 'second.py').write_text('old\n')
        with self.assertRaisesRegex(i.InstallError, 'Installed fix conflicts'):
            i.apply(self.root, self.target, [second])
        self.assertEqual((self.target / 'second.py').read_text(), 'old\n')
        self.assertEqual(i.status(self.target)['transactions'][0]['conflicts'], ['second'])

    def test_control_directories_rejected(self):
        for name in ['.git/config', '.flag-fixes-state/installed.json', 'sub/.git/config']:
            with self.subTest(name=name):
                self.m['targets'][0]['path'] = name
                with self.assertRaises(i.InstallError):
                    i.plan(self.root, self.target, [self.m])

    def test_wrong_baseline(self):
        self.file.write_text('unexpected\n')
        with self.assertRaises(i.InstallError):
            i.apply(self.root, self.target, [self.m])
        self.assertEqual(self.file.read_text(), 'unexpected\n')

    def test_not_deployable(self):
        self.m['deployable'] = False
        self.assertFalse(i.plan(self.root, self.target, [self.m])['deployable'])
        with self.assertRaises(i.InstallError):
            i.apply(self.root, self.target, [self.m])

    def test_symlink(self):
        self.file.unlink()
        self.file.symlink_to(self.patch)
        with self.assertRaises(i.InstallError):
            i.plan(self.root, self.target, [self.m])

    def test_state_symlink(self):
        (self.target / '.flag-fixes-state').symlink_to(self.source, target_is_directory=True)
        with self.assertRaises(i.InstallError):
            i.apply(self.root, self.target, [self.m])

    def test_unlisted_patch(self):
        self.patch.write_text(self.patch.read_text().replace('sample.py', 'other.py'))
        self.m['patches'][0]['sha256'] = hashlib.sha256(self.patch.read_bytes()).hexdigest()
        with self.assertRaises(i.InstallError):
            i.plan(self.root, self.target, [self.m])

    def test_rollback_user_edit(self):
        i.apply(self.root, self.target, [self.m])
        self.file.write_text('user edit\n')
        with self.assertRaises(i.InstallError):
            i.rollback(self.target)
        self.assertEqual(self.file.read_text(), 'user edit\n')

    def test_interrupted_transaction_recovers_on_next_apply(self):
        i.apply(self.root, self.target, [self.m])
        state = self.target / '.flag-fixes-state'
        record = i.status(self.target)['transactions'][0]
        # Simulate an interrupted transaction before its installation record commit.
        i._json(state / 'installed.json', [])
        i._json(state / 'journal.json', {'files': record['files'], 'previous': []})
        self.assertTrue(i.status(self.target)['recovery_required'])
        i.apply(self.root, self.target, [self.m])
        self.assertFalse(i.status(self.target)['recovery_required'])
        i.rollback(self.target)
        self.assertEqual(self.file.read_text(), 'old\n')

    def test_patch_tamper(self):
        self.patch.write_text(self.patch.read_text() + '\n')
        with self.assertRaises(i.InstallError):
            i.plan(self.root, self.target, [self.m])

    def test_path_traversal(self):
        self.m['targets'][0]['path'] = '../sample.py'
        with self.assertRaises(i.InstallError):
            i.plan(self.root, self.target, [self.m])

    def test_failed_commit_recovers(self):
        original = i._atomic
        failed = []
        def fail_once(path, data):
            if path.name == 'installed.json' and not failed:
                failed.append(True)
                raise OSError('simulated disk failure')
            return original(path, data)
        with patch.object(i, '_atomic', side_effect=fail_once):
            with self.assertRaises(OSError):
                i.apply(self.root, self.target, [self.m])
        self.assertEqual(self.file.read_text(), 'old\n')
        self.assertFalse(i.status(self.target)['transactions'])
        self.assertFalse(i.status(self.target)['recovery_required'])


if __name__ == '__main__':
    unittest.main()
