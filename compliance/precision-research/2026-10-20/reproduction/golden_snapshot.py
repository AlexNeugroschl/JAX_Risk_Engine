import dataclasses,sys
import numpy as np
from engine.precision import Precision
from engine.market_simulation.config import HullWhiteConfig
from engine.pricing.cube import value_today
from tests.support.precision_campaigns import linear,options,market_risk
out={}
for factory in (linear,options):
 for model in ('LGM','HullWhite'):
  campaign=factory()
  if model=='HullWhite':
   ir={k:HullWhiteConfig(**{f.name:getattr(v,f.name) for f in dataclasses.fields(v)}) for k,v in campaign.simulation.ir.items()}
   campaign=dataclasses.replace(campaign,simulation=dataclasses.replace(campaign.simulation,ir=ir))
  for fmt in ('float64','float32'):
   p=Precision() if fmt=='float64' else Precision.throughout(fmt)
   result=campaign.run(campaign.scenarios(64,3),p)
   prefix=f'{fmt}/{campaign.name}/{model}/'
   out.update({prefix+k:np.asarray(v) for k,v in campaign.figures(result).items()})
   out[prefix+'cube']=np.asarray(result.npv_cube)
   out[prefix+'today']=np.asarray(value_today(campaign.trades,campaign.market,'USD',campaign.pricing))
for fmt in ('float64','float32'):
 campaign=market_risk()
 result=campaign.run(campaign.scenarios(128,3),Precision() if fmt=='float64' else Precision.throughout(fmt))
 out.update({f'{fmt}/market-risk/'+k:np.asarray(v) for k,v in campaign.snapshot(result).items()})
np.savez_compressed(sys.argv[1],**out)
print('Saved',len(out),'arrays',flush=True)
