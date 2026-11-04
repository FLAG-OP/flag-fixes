from pathlib import Path
import json
import shutil
import subprocess
import sys
import tempfile
import pytest
import yaml
from flag_fixes import catalog, cli, installer

ROOT = Path(__file__).resolve().parents[1]


def copy_skeleton(tmp_path):
    shutil.copytree(ROOT/'catalog/schemas', tmp_path/'catalog/schemas')
    (tmp_path/'catalog/fixes').mkdir()
    (tmp_path/'fixes').mkdir()
    shutil.copytree(ROOT/'templates', tmp_path/'templates')
    return tmp_path


def test_repository_catalog_and_pins():
    result = catalog.validate(ROOT)
    external = [m for m in catalog.load_catalog(ROOT).values() if m['source']['kind'] == 'submodule']
    assert result['fixes'] >= len(external)
    assert len(external) == 9
    assert len({m['source']['path'] for m in external}) == 8


def test_real_patch_audit(tmp_path):
    """Reproduce known imported mismatches; new unexpected drift must fail CI."""
    for m in catalog.load_catalog(ROOT).values():
        if m['source']['kind'] != 'submodule':
            continue
        if not m['targets']:
            assert m['id'] == 'cann-pow-promotion'
            assert m['installation_status'] == 'needs-adaptation'
            continue
        target = tmp_path/m['id']
        target.mkdir()
        for item in m['targets']:
            dest = target/item['path']
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT/m['source']['path']/item['before_source'], dest)
        if m['installation_status'] == 'source-checked':
            result = installer.plan(ROOT, target, [m])
            assert result['deployable'] is m['deployable']
        else:
            with pytest.raises(installer.InstallError, match='Patched checksum differs'):
                installer.plan(ROOT, target, [m])
        assert not (target/'.flag-fixes-state').exists()


def test_new_local_fix_is_discovered_and_title_is_yaml_safe(tmp_path):
    root = copy_skeleton(tmp_path)
    title = 'hello: "world"\nnext line'
    cli.new_fix(root, 'sample-fix', title)
    m = catalog.load_catalog(root)['sample-fix']
    assert m['title'] == title
    assert m['source']['path'] == 'fixes/sample-fix'
    assert m['deployable'] is False
    assert catalog.validate(root)['fixes'] == 1
    with pytest.raises(catalog.CatalogError, match='already exists'):
        cli.new_fix(root, 'sample-fix', 'another')
    with pytest.raises(catalog.CatalogError):
        cli.new_fix(root, '../escape', 'bad')


def test_template_cannot_be_promoted_without_evidence(tmp_path):
    root = copy_skeleton(tmp_path)
    cli.new_fix(root, 'sample-fix', 'sample')
    file = root/'fixes/sample-fix/fix.yaml'
    data = yaml.safe_load(file.read_text())
    data['deployable'] = True
    file.write_text(yaml.safe_dump(data))
    with pytest.raises(catalog.CatalogError):
        catalog.load_catalog(root)


def test_unsafe_paths_and_symlinks(tmp_path):
    for name in ['../escape', '/absolute', 'x/../escape', 'x\\escape']:
        with pytest.raises(catalog.CatalogError):
            catalog.safe_path(tmp_path, name)
    (tmp_path/'link').symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(catalog.CatalogError):
        catalog.safe_path(tmp_path, 'link/a')


def test_template_directory_is_not_catalog_entry(tmp_path):
    root = copy_skeleton(tmp_path)
    assert catalog.load_catalog(root) == {}


def test_cli_error_is_nonzero_and_readable(capsys):
    assert cli.main(['--root', str(ROOT), 'show', 'does-not-exist']) == 2
    assert 'Unknown fix' in capsys.readouterr().err


def test_dependency_selection_and_cycle():
    manifests = {'a': {'requires':['b'], 'conflicts':[]}, 'b': {'requires':[], 'conflicts':[]}}
    assert catalog.select(manifests, ['a','b']) == [manifests['b'], manifests['a']]
    with pytest.raises(catalog.CatalogError, match='required'):
        catalog.select(manifests, ['a'])
    manifests['b']['requires'] = ['a']
    with pytest.raises(catalog.CatalogError, match='cycle'):
        catalog.select(manifests, ['a','b'])


def test_show_does_not_require_selecting_dependencies(tmp_path, capsys):
    root = copy_skeleton(tmp_path)
    cli.new_fix(root, 'first', 'first')
    cli.new_fix(root, 'second', 'second')
    file = root/'fixes/first/fix.yaml'
    m = yaml.safe_load(file.read_text()); m['requires'] = ['second']; file.write_text(yaml.safe_dump(m))
    assert cli.main(['--root', str(root), 'show', 'first']) == 0
    assert json.loads(capsys.readouterr().out)['requires'] == ['second']


def test_verify_uses_selected_target_and_reports_script_failure(tmp_path, capsys):
    root = copy_skeleton(tmp_path/'manager')
    cli.new_fix(root, 'sample-fix', 'sample')
    source = root/'fixes/sample-fix'
    target = tmp_path/'target'
    (target/'src').mkdir(parents=True)
    file = target/'src/fixed_module.py'; file.write_text('fixed\n')
    (source/'tests/verify.py').write_text(
        'import os, sys\nfrom pathlib import Path\n'
        'assert (Path(os.environ["PYTHONPATH"].split(os.pathsep)[0])/"fixed_module.py").read_text() == "fixed\\n"\n'
        'print("selected target verified")\nsys.exit(7)\n')
    manifest_file = source/'fix.yaml'
    manifest = yaml.safe_load(manifest_file.read_text())
    manifest['targets'] = [{'path':'src/fixed_module.py', 'before_sha256':'0'*64, 'after_sha256':catalog.sha256(file)}]
    manifest['validation']['commands'] = [{'name':'failure', 'script':'tests/verify.py', 'level':'regression'}]
    manifest_file.write_text(yaml.safe_dump(manifest))
    code = cli.main(['--root',str(root),'verify','sample-fix','--target',str(target),'--python',sys.executable])
    assert code == 1
    result = json.loads(capsys.readouterr().out)
    assert result['passed'] is False
    assert result['results'][0]['returncode'] == 7
    assert result['results'][0]['stdout'].strip() == 'selected target verified'
    assert result['source_dirty'] is True
    assert not (target/'.flag-fixes-state').exists()
    assert yaml.safe_load(manifest_file.read_text())['validation']['hardware_status'] == 'unverified'
