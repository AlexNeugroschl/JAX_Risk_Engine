"""Small local compiler regression, with all deviations from normal defaults explicit."""
import os
from pathlib import Path
import sys

root=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root))
mode=sys.argv[1]
os.environ['JAX_PLATFORMS']='cuda,cpu'
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']='false'
os.environ['JAX_ENABLE_COMPILATION_CACHE']='false'
os.environ.pop('JAX_RISK_EMULATE_PRODUCTS',None)
os.environ['XLA_FLAGS']='--xla_gpu_exclude_nondeterministic_ops=true'
if mode=='cublas':
    os.environ['XLA_FLAGS']+=' --xla_gpu_enable_triton_gemm=false'
if mode=='bfloat-workaround':
    os.environ['JAX_RISK_EMULATE_PRODUCTS']='bfloat16_3x'
from tests.support.precision_devices import qualify
from demos.benchmark_reporting import atomic_json
import jax
assert jax.devices()[0].platform=='gpu'
out=root/'.profile-out-native-defaults'/mode
out.mkdir(exist_ok=True)
if mode.startswith('bfloat'):
    formats=('bfloat16_3x',)
    shapes=((64,64,64),(512,64,1))
else:
    formats=('float8_e4m3fn','float8_e4m3fn_1x','float8_e5m2','float8_e5m2_1x')
    shapes=((33,13,7),(512,64,1))
result=qualify(jax.devices()[0],formats,out,shapes=shapes,patterns=('wide','cancellation'))
result['experiment']='local compiler regression; default failure retained, workarounds explicit'
atomic_json(out/'qualification.json',result)
print(mode,len(result['operations']),'operations;',sum(r['passes'] for r in result['operations']),
      'passed;',sum(r['observed_fp8_gemm'] for r in result['operations']),'observed FP8 calls',flush=True)
raise SystemExit(0 if result['passes'] else 1)
