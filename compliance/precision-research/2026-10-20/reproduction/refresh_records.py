import json,os,subprocess,sys
from pathlib import Path
backend=sys.argv[1]
folder=Path(__file__).resolve().parent
out=folder/'record-refresh'
out.mkdir(exist_ok=True)
os.environ['JAX_PLATFORMS']='cpu' if backend=='cpu' else 'cuda,cpu'
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']='false'
if backend=='gpu':
 os.environ['XLA_FLAGS']='--xla_gpu_enable_triton_gemm=false --xla_gpu_exclude_nondeterministic_ops=true'
names=(['linear','compute','market_risk','compute_market_risk','rounding','rounding_market_risk','replicated_storage'] if backend=='cpu' else ['options','compute_options','rounding_options','market_risk_calibrated'])
rows=[]
for name in names:
 print('Refreshing',backend,name,flush=True)
 with (folder/f'refresh-{name}.log').open('w',encoding='utf-8') as log:
  result=subprocess.run([sys.executable,'-m','tests.support.precision_campaigns',name,'--out',str(out)],stdout=log,stderr=subprocess.STDOUT)
 rows.append({'campaign':name,'exit_code':result.returncode})
 (folder/f'record-refresh-{backend}.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
 print(name,'exit',result.returncode,flush=True)
raise SystemExit(int(any(r['exit_code'] for r in rows)))
