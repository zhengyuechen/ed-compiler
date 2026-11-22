#!/usr/bin/env python3
"""Scans of the lowest level over all sectors (edc_cli, `sector ... all` + `scan`): one solve per point against the
family scan, loose screening, concavity pruning, and the named decomposition, with and without skipping conjugate blocks.
Usage: test_scan_all.py"""
import numpy as np
import edc_cli

worst = 0.0
for lat, sz, scan, cpl in (('kagome a1=3,0 a2=0,3', 7, 'Delta=0.5:1.5:13', 'J1=1'), ('kagome a1=3,0 a2=0,3', 6, 'J2=-0.1:0.1:9', 'J1=1 Delta=0.9'),
                           ('triangular a1=1,4 a2=-4,5', 4, 'Delta=0.6:1.4:11', 'J1=1 J2=0.1'), ('honeycomb a1=2,2 a2=-2,4', 6, 'Delta=0.5:1.5:11', 'J1=1')):
    res = {}
    for tag, sector, extra, solve in (('points', 'all', ' method=points', ' pairs=0'), ('family', 'all', '', ''), ('screen', 'all', ' screen=1', ''),
                                      ('prune', 'all', ' prune=3', ''), ('named+prune', 'named', ' prune=3', '')):
        spec = edc_cli.parse(f'lattice {lat}\ncouplings {cpl}\nsector Sz={sz} {sector}\nscan {scan}{extra}\nsolve device=cpu{solve}\n')
        res[tag] = edc_cli.run(spec)
        assert res[tag]['sum_mult_dim'] == res[tag]['full_dim'] or tag.endswith('prune'), (lat, tag)
    E0 = np.array(res['points']['E'])
    d = {t: float(np.abs(np.array(r['E'])-E0).max()) for t, r in res.items() if t != 'points'}
    worst = max(worst, max(d.values()))
    print(f'{lat} Sz={sz} {scan} ({cpl}): products ' + ', '.join(f'{t} {r["products"]}' for t, r in res.items())
          + ' | largest difference to one solve per point ' + ', '.join(f'{t} {v:.1e}' for t, v in d.items()), flush=True)
assert worst < 1e-8, worst
print('scans over all sectors: all checks passed')
