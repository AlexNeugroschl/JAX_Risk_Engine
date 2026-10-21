import dataclasses,json,os
from pathlib import Path
os.environ['JAX_PLATFORMS']='cpu'
from tests.support.precision_campaigns import COMPARISON_CAMPAIGNS,write
from tests.support.precision_harness import measure
from tests.support.precision_evidence import Standard,generate
from tests.support.precision_timing import workload_settings,execution_environment
folder=Path(__file__).resolve().parent/'large-single'
folder.mkdir(exist_ok=True)
c=COMPARISON_CAMPAIGNS['comparison_single']()
labels={'pricing:compute:float32','pricing:compute:float16','pricing:compute:float8_e4m3fn'}
c=dataclasses.replace(c,name='single_large_count',seeds=tuple(range(1,33)),cases=tuple(x for x in c.cases if x.label in labels))
(folder/'workload.json').write_text(json.dumps({'workload':workload_settings(c),'environment':execution_environment()},indent=2),encoding='utf-8')
merged=None
for n in (65536,262144,1048576,4194304):
 result=measure(c,n,progress=lambda s:print(s,flush=True))
 point=folder/str(n);point.mkdir(exist_ok=True);write(result,point)
 merged=result if merged is None else merged.merge(result)
 write(merged,folder)
 markdown,evidence=generate([merged],Standard.load())
 (folder/'candidate-evidence.md').write_text(markdown,encoding='utf-8')
 (folder/'candidate-evidence.json').write_text(evidence,encoding='utf-8')
 print('Completed',n,'paths, 32 independent seeds, 3 compute policies plus FP64 reference',flush=True)
