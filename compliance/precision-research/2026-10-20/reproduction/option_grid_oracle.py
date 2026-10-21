"""Independent ORE pricing and a matched-path FP64 rollback-grid diagnostic."""
import dataclasses, json, os
from pathlib import Path
os.environ['JAX_PLATFORMS'] = 'cpu'
import numpy as np
from benchmarks.qualification import build_campaign
from engine.pricing.cube import value_today
from tests.support.ore_npv_oracle import ore_npv_run
from tests.test_ore_xva_parity import _on_ores_paths, CUBE_RTOL

out = Path(__file__).resolve().parent / 'option-grid-oracle'
out.mkdir(exist_ok=True)
rows, cubes, numeraires = [], {}, {}
for grid in ('default', 'refined'):
    c = build_campaign('compute_options', fixture_seed=51, grid=grid)
    row = {'fixture_seed': 51, 'grid': grid}
    try:
        reference = ore_npv_run(c.market, c.trades, c.pricing)
        assert not reference.refused, reference.refused
        actual = value_today(c.trades, c.market, 'USD', c.pricing)
        today = [{'trade': t.trade_id, 'absolute_error': abs(v-reference.npv[t.trade_id]),
                  'tolerance': 2e-11 * abs(t.notional),
                  'passes': abs(v-reference.npv[t.trade_id]) <= 2e-11 * abs(t.notional)}
                 for t, v in zip(c.trades, actual)]
        ore, _, cube = _on_ores_paths(c.trades, 'LGM', c.pricing, c.market,
                                     dataclasses.replace(c.simulation, samples=32))
        errors = []
        for i, trade in enumerate(c.trades):
            error = float(np.max(np.abs(cube[:, :, i]-ore.cube[:, :, i])))
            tolerance = CUBE_RTOL * float(np.max(np.abs(ore.cube[:, :, i])))
            errors.append({'trade': trade.trade_id, 'absolute_error': error, 'tolerance': tolerance,
                           'passes': bool(np.isfinite(error) and error <= tolerance)})
        row.update(npv=today, cube=errors, passes=all(r['passes'] for r in today+errors))
        cubes[grid], numeraires[grid] = cube, ore.numeraire
        np.savez_compressed(out / f'{grid}.npz', engine_cube=cube, ore_cube=ore.cube, numeraire=ore.numeraire)
    except Exception as error:
        row.update(passes=False, reason=repr(error))
    rows.append(row)
    print(grid, 'PASS' if row['passes'] else row.get('reason','FAIL'), flush=True)
    (out / 'results.json').write_text(json.dumps({'rows':rows,'complete':False,'passes':False},indent=2))
diagnostic = None
if len(cubes) == 2:
    assert np.array_equal(numeraires['default'],numeraires['refined'])
    diagnostic = {'scope':'matched-path FP64 grid changes, not a convergence certificate',
                  'max_absolute_cube_change':float(np.max(np.abs(cubes['default']-cubes['refined']))),
                  'trades':[{'trade': t.trade_id,'max_absolute_cube_change':
                              float(np.max(np.abs(cubes['default'][:,:,i]-cubes['refined'][:,:,i])))}
                            for i,t in enumerate(c.trades)]}
(out / 'results.json').write_text(json.dumps({'rows':rows,'complete':True,
    'passes':all(r['passes'] for r in rows),'grid_diagnostic':diagnostic},indent=2))
raise SystemExit(int(not all(r['passes'] for r in rows)))
