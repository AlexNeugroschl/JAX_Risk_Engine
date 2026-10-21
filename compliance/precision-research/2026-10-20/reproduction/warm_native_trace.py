"""Capture warm engine execution per policy, with optimized lowering beside the traces."""
import argparse
import functools
import hashlib
import json
import math
import os
from pathlib import Path
import sys

# This archived reproduction script runs directly from any working directory in the checkout.
root = next(p for p in Path(__file__).resolve().parents if (p / 'engine').is_dir())
sys.path.insert(0, str(root))
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--out', type=Path, required=True)
parser.add_argument('--platform', choices=('cpu', 'gpu'), default='gpu')
parser.add_argument('--expected-device', help='required device-kind substring, e.g. H100')
parser.add_argument('--paths', type=int, default=512)
parser.add_argument('--formats', nargs='+', default=['float16', 'float16_1x', 'float8_e4m3fn',
                    'float8_e4m3fn_1x', 'float8_e5m2', 'float8_e5m2_1x'])
args = parser.parse_args()
if args.paths < 128 or args.paths & (args.paths-1):
    parser.error('use a power-of-two count >=128')
args.out.mkdir(parents=True, exist_ok=True)
if any(args.out.iterdir()):
    parser.error('output directory must be empty')
os.environ['JAX_PLATFORMS'] = 'cpu' if args.platform == 'cpu' else 'cuda,cpu'
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
from engine.api.worker import deterministic_kernels_environment, gemm_environment
os.environ.update(gemm_environment('cublas', os.environ))
os.environ.update(deterministic_kernels_environment(os.environ))
os.environ['XLA_FLAGS'] += f' --xla_dump_to={args.out.resolve() / "hlo"} --xla_dump_hlo_as_text=true'
import jax
from demos.benchmark_reporting import atomic_json
from tests.support import precision_campaigns
from tests.support.precision_timing import _child, execution_environment

device = jax.devices()[0]
if device.platform != args.platform or (args.expected_device and args.expected_device.lower() not in device.device_kind.lower()):
    parser.error(f'unexpected device: {device}')
if jax.config.jax_disable_jit:
    parser.error('warm tracing requires JIT execution')
# Keep the same immutable workload objects for warm-up and trace; seed values stay traced data.
precision_campaigns.timing_campaign = functools.lru_cache(None)(precision_campaigns.timing_campaign)
requests = []
for campaign in ('comparison_single', 'comparison_portfolio'):
    labels = {c.label for c in precision_campaigns.timing_campaign(campaign).cases}
    for name in args.formats:
        label = 'float64' if name == 'float64' else f'pricing:{name}'
        if name != 'float64' and label not in labels:
            label = f'pricing:compute:{name}'
        if name != 'float64' and label not in labels:
            parser.error(f'unsupported engine control {name}')
        requests.append((campaign, name, label))
data = {'schema_version':1, 'environment':execution_environment(),
        'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'paths':args.paths,
        'scope':'execution-only traces; undefined financial figures are null with availability records; numerical acceptance and instruction attribution remain separate',
        'complete':False,'passes':False,'points':[]}
atomic_json(args.out / 'warm-traces.json', data)
with jax.default_device(device):
    for campaign, name, label in requests:
        destination = args.out / campaign / name
        destination.mkdir(parents=True)
        _child(campaign, label, args.paths, (1,2,3), None, cpu_inputs=True)
        with jax.profiler.trace(str(destination / 'trace')):
            timing = _child(campaign, label, args.paths, (4,5,6), None, cpu_inputs=True,
                            snapshot_directory=str(destination / 'snapshots'))
        unavailable = [name for name, values in timing['figures'].items()
                       if any(not math.isfinite(v) for seed in values for v in seed)]
        def finite_json(value):
            if isinstance(value, float) and not math.isfinite(value):
                return None
            if isinstance(value, dict):
                return {k:finite_json(v) for k,v in value.items()}
            if isinstance(value, (list,tuple)):
                return [finite_json(v) for v in value]
            return value
        timing = finite_json(timing)
        row = {'campaign':campaign,'format':name,'timing':timing,
               'unavailable_figures':unavailable, 'figures_available':not unavailable,
               'passes':timing['compiles']==0 and timing['warm_compiles']==0}
        if args.platform == 'gpu':
            import gzip
            traces = list((destination / 'trace').rglob('*.trace.json.gz'))
            events = json.loads(gzip.decompress(traces[0].read_bytes()))['traceEvents'] if traces else []
            pids = {e['pid'] for e in events if e.get('name')=='process_name'
                    and 'GPU' in e.get('args',{}).get('name','')}
            kernels = [e for e in events if e.get('pid') in pids and e.get('ph')=='X'
                       and e.get('args',{}).get('hlo_module')]
            row['device_kernel_events'] = len(kernels)
            row['passes'] &= bool(kernels)
        data['points'].append(row)
        atomic_json(args.out / 'warm-traces.json', data)
        print(campaign, name, 'PASS' if row['passes'] else 'FAILED',
              'compiles', timing['compiles'], 'device events', row.get('device_kernel_events'), flush=True)
data['complete'] = True
data['passes'] = all(r['passes'] for r in data['points'])
atomic_json(args.out / 'warm-traces.json', data)
raise SystemExit(int(not data['passes']))
