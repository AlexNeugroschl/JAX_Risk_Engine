"""Independent ORE checks of the held-out linear pricing components.

EUR curves/trades are also priced through the existing USD-only ORE adapter after an
explicit currency relabelling. This is not joint multi-currency simulation qualification.
"""
import dataclasses, json, os
from pathlib import Path
os.environ['JAX_PLATFORMS'] = 'cpu'
import jax
import numpy as np
from benchmarks.qualification import build_campaign
from engine.pricing.cube import value_today
from tests.support.ore_npv_oracle import ore_npv_run
from tests.test_ore_xva_parity import _on_ores_paths, CUBE_RTOL

out = Path(__file__).resolve().parent / 'heldout-oracle'
out.mkdir(exist_ok=True)
rows = []
for seed in (40, 44, 45, 51):
    for model in ('LGM', 'HullWhite'):
        c = build_campaign('compute', fixture_seed=seed, model=model, multi_currency=True)
        for ccy in ('USD', 'EUR'):
            row = {'fixture_seed': seed, 'model': model, 'currency': ccy,
                   'scope': 'native-currency NPV and 32-path ORE cube; EUR relabelled to USD for the single-currency oracle'}
            try:
                native = tuple(t for t in c.trades if t.currency == ccy)
                trades = tuple(dataclasses.replace(t, currency='USD') for t in native)
                data = c.market.currency(ccy)
                data = dataclasses.replace(data, index_curves={k.replace(ccy + '-', 'USD-', 1): v
                                                               for k, v in data.index_curves.items()})
                market = dataclasses.replace(c.market, currencies={'USD': data}, fx_spots={})
                config = dataclasses.replace(c.simulation, samples=32, ir={'USD': c.simulation.ir[ccy]},
                                             base_currency='USD', fx_volatilities={}, correlations={})
                reference = ore_npv_run(market, trades, c.pricing)
                assert not reference.refused, reference.refused
                actual = value_today(native, c.market, ccy, c.pricing)
                npvs = []
                for cfg, value in zip(trades, actual):
                    scale = abs(float(getattr(cfg, 'notional', None) or cfg.face_amount))
                    error = abs(value - reference.npv[cfg.trade_id])
                    npvs.append({'trade': cfg.trade_id, 'absolute_error': error,
                                 'tolerance': 1e-12 * scale, 'passes': error <= 1e-12 * scale})
                ore, _, cube = _on_ores_paths(trades, model, c.pricing, market, config)
                cubes = []
                for i, trade in enumerate(trades):
                    error = float(np.max(np.abs(cube[:, :, i] - ore.cube[:, :, i])))
                    tolerance = CUBE_RTOL * float(np.max(np.abs(ore.cube[:, :, i])))
                    cubes.append({'trade': trade.trade_id, 'absolute_error': error,
                                  'tolerance': tolerance, 'passes': bool(np.isfinite(error) and error <= tolerance)})
                row.update(npv=npvs, cube=cubes, passes=all(r['passes'] for r in npvs + cubes))
            except Exception as error:
                row.update(passes=False, reason=repr(error))
            rows.append(row)
            (out / 'results.json').write_text(json.dumps({'rows': rows, 'complete': len(rows) == 16,
                'passes': len(rows) == 16 and all(r['passes'] for r in rows)}, indent=2), encoding='utf-8')
            print(seed, model, ccy, 'PASS' if row['passes'] else row.get('reason', 'FAIL'), flush=True)
            jax.clear_caches()
raise SystemExit(int(not all(r['passes'] for r in rows)))
