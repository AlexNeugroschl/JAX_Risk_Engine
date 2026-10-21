import json
from pathlib import Path
from tests.support.precision_harness import Measurements
from tests.support.precision_evidence import Standard, rows_of

root = Path.cwd()
fresh = root / '.profile-out-fp8-implementation/record-refresh'
standard = Standard.load()
review = []
def normalized(value):
    if isinstance(value, dict):
        value = {k: normalized(v) for k, v in value.items()}
        if {'storage', 'compute', 'accumulate'} <= value.keys():
            value.setdefault('compute_rounding', 'nearest')
    return value
for source in sorted((root / 'compliance/precision-measurements').glob('*.json.gz')):
    path = fresh / source.name
    if not path.exists():
        continue
    old, new = Measurements.load(source), Measurements.load(path)
    complete = old.paths == new.paths
    item = {'campaign': old.campaign, 'complete': complete,
            'paths': new.paths, 'engine': new.environment['engine'],
            'backend_before': old.environment['backend'], 'backend_after': new.environment['backend']}
    if complete:
        for field in ('cases', 'seeds', 'settings', 'products', 'replicates', 'dates', 'kind'):
            assert normalized(getattr(old, field)) == normalized(getattr(new, field)), (old.campaign, field)
        assert all(r.recompiles == 0 for r in new.runs), old.campaign
        before = {(r.case, r.scope, r.figure): r.ceiling for r in rows_of(old, standard)}
        after = {(r.case, r.scope, r.figure): r.ceiling for r in rows_of(new, standard)}
        assert before.keys() == after.keys(), old.campaign
        item['changed_ceilings'] = [{'case': k[0], 'scope': k[1], 'figure': k[2],
                                    'before': before[k], 'after': after[k]}
                                   for k in before if before[k] != after[k]]
        item['figures'] = len(after)
    review.append(item)
    print(old.campaign, 'complete' if complete else 'partial',
          len(item.get('changed_ceilings', [])), 'changed ceilings', flush=True)
(fresh.parent / 'record-refresh-review.json').write_text(json.dumps(review, indent=2), encoding='utf-8')
