import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET

meta = json.loads((Path(__file__).parent / 'case.json').read_text())
work = Path(sys.argv[1]).resolve()
evidence = Path(sys.argv[2]).resolve()
evidence.mkdir(parents=True, exist_ok=True)
python = sys.executable
runner = str(Path(__file__).parent / 'pytest_with_absl.py')
is_jax = meta['repo'].endswith('/jaxpruner')
source = work / meta['source']
fixed = source.read_bytes()
original = subprocess.check_output(['git', 'show', meta['base'] + ':' + meta['source']], cwd=work)
summary = {'head': meta['head'], 'source_sha256': hashlib.sha256(fixed).hexdigest()}


def run(label, command, expected=0, cwd=work, timeout=480):
  with (evidence / (label + '.log')).open('w') as log:
    proc = subprocess.run(command, cwd=cwd, env=os.environ.copy(),
                          stdout=log, stderr=subprocess.STDOUT, timeout=timeout)
  output = (evidence / (label + '.log')).read_text(errors='replace')
  print(label, proc.returncode, output[-550:], flush=True)
  if expected is not None:
    assert proc.returncode == expected, (label, proc.returncode, output[-7000:])
  return proc.returncode


def test(label, paths, expected=0, cov=False, cwd=work):
  args = [python, runner, *paths, '-q', '--tb=short',
          '--junitxml=' + str(evidence / (label + '.xml'))]
  if cov:
    package = 'jaxpruner' if is_jax else 'bleurt'
    args += ['--cov=' + package, '--cov-branch',
             '--cov-report=json:' + str(evidence / (label + '-coverage.json'))]
  code = run(label, args, expected, cwd=cwd)
  assert code in (0, 1), (label, code)
  doc = ET.parse(evidence / (label + '.xml'))
  suites = list(doc.getroot().iter('testsuite'))
  counts = [sum(int(s.attrib.get(k, 0)) for s in suites)
            for k in ('tests', 'failures', 'errors', 'skipped')]
  failures = {}
  for case in doc.getroot().iter('testcase'):
    error = case.find('failure')
    if error is None:
      error = case.find('error')
    if error is not None:
      key = case.attrib.get('classname', '') + '::' + case.attrib['name']
      failures[key] = error.attrib.get('message', '')
  value = {'counts': counts, 'failures': failures}
  summary[label] = value
  print(label, counts, list(failures), flush=True)
  return value


assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=work, text=True).strip() == meta['head']
run('dependencies', [python, '-m', 'pip', 'freeze'])
run('dependency-check', [python, '-m', 'pip', 'check'])
new_count, fail_count = meta['new_counts']
fixed_new = test('fixed-focused', [meta['test']], cov=True)
assert fixed_new['counts'] == [new_count, 0, 0, 0]
try:
  if is_jax:
    fixed_suite = test('fixed-core', meta['suite_files'] + [meta['test'], '-k', 'not LimitMemoryUsage'])
    assert fixed_suite['counts'] == [170, 0, 0, 0]
  else:
    fixed_suites = {}
    for path in meta['suite_files']:
      label = Path(path).stem
      value = test('fixed-' + label, [path], expected=None)
      assert sorted(value['failures']) == sorted(meta['baseline_failure_nodes'][path])
      fixed_suites[path] = value
  source.write_bytes(original)
  original_new = test('original-focused', [meta['test']], expected=1)
  assert original_new['counts'] == [new_count, fail_count, 0, 0]
  if is_jax:
    baseline = test('baseline-core', meta['suite_files'] + ['-k', 'not LimitMemoryUsage'])
    assert baseline['counts'] == [126, 0, 0, 0]
  else:
    for path in meta['suite_files']:
      value = test('baseline-' + Path(path).stem, [path], expected=None)
      assert value == fixed_suites[path], (path, value, fixed_suites[path])
finally:
  source.write_bytes(fixed)
assert source.read_bytes() == fixed
test('restored-focused', [meta['test']])
run('lint', [python, '-m', 'ruff', 'check', '--select', 'E9,F63,F7,F82', meta['source'], meta['test']])
run('new-test-lint', [python, '-m', 'ruff', 'check', '--select', 'F', meta['test']])
run('restored-source', ['git', 'diff', '--exit-code'])
run('patch-check', ['git', 'diff', '--check', meta['base']])
run('package-build', [python, '-m', 'build', '--outdir', str(evidence / 'dist')])
wheel, = (evidence / 'dist').glob('*.whl')
run('install-wheel', [python, '-m', 'pip', 'install', '--no-deps', '--force-reinstall', str(wheel)])
with tempfile.TemporaryDirectory() as directory:
  outside = Path(directory)
  copied = outside / Path(meta['test']).name
  shutil.copyfile(work / meta['test'], copied)
  package = 'jaxpruner' if is_jax else 'bleurt'
  code = (f'import {package}; from pathlib import Path; '
          f'p=Path({package}.__file__).resolve(); print(p); '
          f'assert not p.is_relative_to(Path({str(work)!r}))')
  run('installed-import', [python, '-c', code], cwd=outside)
  installed = test('installed-focused', [str(copied)], cwd=outside)
  assert installed['counts'] == [new_count, 0, 0, 0]
(evidence / 'summary.json').write_text(json.dumps(summary, indent=2))
print('VALIDATION COMPLETE', meta['head'], flush=True)
