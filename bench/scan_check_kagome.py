# Family scan (normal and light) against one solve per point, kagome 24, anisotropy and J2 scans in several S_z sectors.
import sys, os, json, numpy as np
sys.path.insert(0, os.getcwd()); import edc_cli
cases = [("Sz=0 k=0,0 irrep=X0 flip=+1", "Delta=0.4:1.6:41", "J1=1"), ("Sz=2 k=0,0 irrep=X0", "Delta=0.4:1.6:41", "J1=1"),
         ("Sz=4 k=0,0 irrep=X0", "Delta=0.4:1.6:41", "J1=1 J2=0.2"), ("Sz=3 k=0,0 irrep=X0", "J2=-0.3:0.3:41", "J1=1 Delta=0.8")]
for sec, scan, cpl in cases:
    res = {}
    for tag, extra in (('points', ' method=points'), ('family', ''), ('light', ' light=1')):
        spec = edc_cli.parse(f"lattice kagome a1=2,2 a2=-2,2\ncouplings {cpl}\nsector {sec}\nscan {scan}{extra}\nsolve device=cpu\n")
        res[tag] = edc_cli.run(spec)
    E = {k: np.array(v['E']) for k, v in res.items()}
    print('%-30s %-18s dim %7d | products: points %5d family %5d (basis %d) light %5d (basis %d) | max |dE| family %.1e light %.1e | E range %.4f..%.4f' % (
        sec, scan, res['points']['dim'], res['points']['products'], res['family']['products'], res['family']['basis'], res['light']['products'], res['light']['basis'],
        np.abs(E['family']-E['points']).max(), np.abs(E['light']-E['points']).max(), E['points'].min(), E['points'].max()), flush=True)
