import json, os, subprocess, sys
from pathlib import Path
backend = sys.argv[1]
folder = Path(__file__).resolve().parent
out = folder / 'record-refresh'
os.environ['JAX_PLATFORMS'] = 'cpu' if backend == 'cpu' else 'cuda,cpu'
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
if backend == 'gpu':
    os.environ['XLA_FLAGS'] = '--xla_gpu_enable_triton_gemm=false --xla_gpu_exclude_nondeterministic_ops=true'
names = ['options', 'rounding_options'] if backend == 'gpu' else ['compute_options', 'market_risk_calibrated']
rows = []
for name in names:
    destination = out / 'options-remaining' if name == 'options' else out
    destination.mkdir(exist_ok=True)
    command = [sys.executable, '-m', 'tests.support.precision_campaigns', name, '--out', str(destination)]
    if name == 'options':
        command += ['--paths', '4096', '16384']
    print('Refreshing', backend, name, flush=True)
    with (folder / f'refresh-final-{name}.log').open('w', encoding='utf-8') as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
    if name == 'options' and not result.returncode:
        from tests.support.precision_harness import Measurements
        from tests.support.precision_campaigns import write
        prefix = Measurements.load(out / 'options.json.gz')
        tail = Measurements.load(destination / 'options.json.gz')
        assert prefix.paths == (1024,), prefix.paths
        assert tail.paths == (4096, 16384), tail.paths
        write(prefix.merge(tail), out)
    rows.append({'campaign': name, 'exit_code': result.returncode})
    (folder / f'record-refresh-final-{backend}.json').write_text(json.dumps(rows, indent=2), encoding='utf-8')
    print(name, 'exit', result.returncode, flush=True)
raise SystemExit(int(any(r['exit_code'] for r in rows)))
