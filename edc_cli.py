#!/usr/bin/env python3
"""A small model-description language for the ED compiler.

    # 40-site J1-J2, ground state and first excited level of one sector
    lattice square a1=6,2 a2=-2,6        # square | rect Lx= Ly= | chain L= | triangular | honeycomb | kagome (a1= a2=)
    couplings J1=1 J2=0.5 J3=0           # Heisenberg couplings by neighbour shell (square: nearest, diagonal, (2,0))
                                         # Delta=0.8: XXZ anisotropy, every bond J (SxSx + SySy + Delta SzSz); default 1
    bond 2,1 J=0.1                       # square lattices only: any extra Heisenberg bond vector
    sector Sz=0 k=0,0 irrep=A1 flip=+1   # k in units of pi; irrep of the little group; flip only for Sz = 0
                                         # `sector Sz=2 all`: every block; `sector Sz=2 named` (hexagonal lattices): every
                                         # irrep of every little group resolved in one pass (1D irreps of the full group,
                                         # 2D irreps through one rotation block; a level counts multiplicity times)
    solve levels=2 device=gpu tol=1e-12  # device gpu | cpu; engine sub (default) | lean (GPU, square) | k (CPU, any group)
                                         # rule=bound: the levels are accepted by an error bound from the Lanczos residuals
                                         # (edc_levels; tol then defaults to 1e-10); stored=partial: when the matrix
                                         # elements do not all fit in memory, store those of the rows that do (reserve=3:
                                         # vectors of the block left free on a GPU); stored=0 | 1: never | must fit

    # a parameter scan solved as one family: one connectivity, a few exact ground states, every other point from them
    lattice rect Lx=6 Ly=6
    couplings J1=1 J2=0.5
    sector Sz=0 k=0,0 irrep=A1 flip=+1   # or `all`: the lowest level of every sector along the scan
    scan J2=0:1:201                      # coupling (J1, J2, J3 or Delta) = first:last:points; method=family (default) | points
                                         # two or three couplings on one line (scan Delta=0.5:1.5:21 J2=-0.2:0.2:21) give a grid
                                         # prune=3 with `sector ... all` (one coupling): every sector solved at 3 scan points;
                                         # its lowest level is concave in the coupling, so the chords bound it from below, and
                                         # a sector whose chords stay above the lowest scanned sector is not scanned (exact);
                                         # a chord that dips below gets up to `solve refine=4` more exact points first
                                         # screen=1 with `sector ... all`: every sector first at a loose tolerance (1e-5), then
                                         # the full tolerance only for sectors that come within a margin of the lowest level
                                         # (one solve per point); light=1 keeps only the basis vectors (less memory)
    solve device=gpu                     # scan options on the scan line: tol=1e-10 start=random|mixed|guess audit=2
                                         # `sector ... all`: a block that is the complex conjugate of another is copied,
                                         # not solved (same spectrum for a real Hamiltonian); solve pairs=0 solves all

    # total spin 0 with both translations (ring-of-rows engine): rectangular torus, short side along x (Lx = 2, 4, 6)
    lattice rect Lx=4 Ly=8
    couplings J1=1
    sector S=0 k=0,0 px=+1 py=+1         # k in units of pi; mirror parities x -> -x and y -> -y
    solve engine=rows device=gpu         # device gpu | cpu;  steps=N (any spin engine): N recursion steps after the ground state
    measure S Q=1,1                      # spin structure factor S(Q) and M^2(Q) = S/(N+2) of the ground state, Q averaged
                                         # over its star; Q Cartesian in units of pi, or Qr=q1,q2 in units of b1, b2

    # any block of a spin model on the kagome, triangular or square lattice through layouts of the vector (edc_glayout):
    # the speed of stored connectivity with a fifth of its memory; levels=, measure and `sector ... all` as for engine=sub
    lattice kagome a1=2,2 a2=-2,4
    couplings J1=1 J2=0.12
    sector Sz=1 k=0,0 irrep=X0
    solve engine=layout device=gpu       # time_product=1 adds the time of one product; fold=table|fly|scatter, expand=table|fly
                                         # (GPU: chosen by the free device memory if not given; reserve=N vectors kept free)

    # kagome torus with inversion as its only point operation: one product through three layouts of the vector
    lattice kagome a1=3,0 a2=0,4
    couplings J1=1
    sector Sz=0 k=0,0 irrep=X0 flip=+1   # this sector only
    solve engine=layout3 device=gpu      # time_product=1 adds the time of one product

    # DMRG on a cylinder; the cylinder is cut into layers closed under the translation around it, a layer is one site
    # of the chain and every bond carries S_z and the momentum
    lattice cylinder Lx=8 Ly=6           # square: open along x, periodic along y; a ring of Ly spins is a layer
                                         # cell=kagome: YC cylinder, Ly unit cells around, Lx chain rings of 2 Ly spins
                                         # with layers of Ly spins between them; N = Ly (3 Lx - 1)
    couplings J1=1 J2=0.5                # J2: the diagonals between neighbouring rings (square); Delta=: XXZ
    sector Sz=0 k=0                      # any S_z; k around the cylinder in units of pi (k = 2 K / Ly, e.g. k=1/3)
    solve method=dmrg D=512              # sweeps=9 tol=3e-5 labels=sz,k | sz  (edc_dmrg.py; generated C kernel, CPU)

    # Fermi-Hubbard on a kagome cluster, every symmetry sector, ground state
    lattice kagome a1=2,0 a2=0,2
    model hubbard t=1 U=6 tp=0
    sector Nup=6 Ndn=6 all               # or: k=.. irrep=.. flip=..
    solve device=cpu

Hexagonal lattices (triangular, honeycomb, kagome) use lattice coordinates with a2 at 60 degrees, momenta as phases per
step along a1, a2 (units of pi) and irreps named X0, X1, ... (edc_lattices.LatticeCluster.irrep_table).
`devices=0,1,2,3` in `solve` spreads one sector over several GPUs of a node (sub engine; no measurements there).
Measurements: sub engine (CPU or one GPU) on every lattice, lean engine on square lattices (`measure M2 Q=..`, as before).
Usage: python edc_cli.py model.edc  ->  one JSON line with the results.
"""
import sys, json, math, time
import numpy as np
import edc, edc_lattice as el

HEX = ('triangular', 'honeycomb', 'kagome')


def parse(text):
    spec = dict(model=('heisenberg', {}), couplings={}, bonds=[], sector={}, sector_all=False, solve={}, measure=[], scan=None)
    for raw in text.splitlines():
        line = raw.split('#', 1)[0].strip()
        if not line:
            continue
        word, *rest = line.split()
        kv = dict(x.split('=', 1) for x in rest if '=' in x)
        if word == 'lattice':
            spec['lattice'] = (rest[0], kv)
        elif word == 'model':
            spec['model'] = (rest[0], kv)
        elif word == 'couplings':
            spec['couplings'].update({k: float(v) for k, v in kv.items()})
        elif word == 'bond':
            spec['bonds'].append((tuple(int(v) for v in rest[0].split(',')), float(kv['J'])))
        elif word == 'sector':
            spec['sector'] = kv; spec['sector_all'] = 'all' in rest or 'named' in rest; spec['sector_named'] = 'named' in rest
        elif word == 'scan':
            axes = []
            for k_, v_ in kv.items():
                if k_ not in ('method', 'tol', 'start', 'audit', 'light', 'basis', 'screen', 'screen_tol', 'prune'):
                    a, b, n = v_.split(':'); axes.append((k_, float(a), float(b), int(n)))
            name, a, b, n = axes[0]
            spec['scan'] = dict(axes=axes, name=name, first=float(a), last=float(b), points=int(n), method=kv.get('method', 'family'),
                                tol=float(kv.get('tol', 1e-10)), start=kv.get('start', 'random'), audit=int(kv.get('audit', 2)),
                                light=bool(int(kv.get('light', 0))), basis=int(kv.get('basis', 0)),
                                screen=bool(int(kv.get('screen', 0))), screen_tol=float(kv.get('screen_tol', 1e-5)),
                                prune=int(kv.get('prune', 0)))
        elif word == 'solve':
            spec['solve'] = kv
        elif word == 'measure':
            spec['measure'].append((rest[0], kv))
        else:
            raise ValueError(f'unknown statement: {line}')
    return spec


def cluster(spec):
    kind, kv = spec['lattice']
    vec = lambda s: tuple(int(v) for v in s.split(','))
    if kind in HEX:
        import edc_lattices as L
        return kind, L.LatticeCluster(kind, vec(kv['a1']), vec(kv['a2']))
    if kind == 'square':
        return kind, el.Cluster(vec(kv['a1']), vec(kv['a2']))
    if kind == 'rect':
        return kind, el.Cluster((int(kv['Lx']), 0), (0, int(kv['Ly'])))
    if kind == 'chain':
        return kind, el.Cluster((int(kv['L']), 0), (0, 1))
    raise ValueError(kind)


def spin_model(spec, kind, cl):
    c = spec['couplings']
    if kind in HEX:
        assert not spec['bonds'], 'bond statements: square lattices only'
        return _xxz(cl.heisenberg(c.get('J1', 1.0), c.get('J2', 0.0), c.get('J3', 0.0)), c.get('Delta', 1.0))
    vecs = []
    if c.get('J1'):
        vecs += [((1, 0), c['J1']), ((0, 1), c['J1'])] if kind != 'chain' else [((1, 0), c['J1'])]
    if c.get('J2'):
        vecs += [((1, 1), c['J2']), ((1, -1), c['J2'])] if kind != 'chain' else [((2, 0), c['J2'])]
    if c.get('J3'):
        vecs += [((2, 0), c['J3']), ((0, 2), c['J3'])] if kind != 'chain' else [((3, 0), c['J3'])]
    vecs += spec['bonds']
    m = edc.SpinHalfModel(cl.N)
    for s, (x, y) in enumerate(cl.coord):
        for (dx, dy), J in vecs:
            t = cl.site(x+dx, y+dy)
            if t != s:
                m.heisenberg(s, t, J)
    return _xxz(m, c.get('Delta', 1.0))


def _xxz(m, delta):
    """The same bonds with the S^z S^z part multiplied by delta."""
    if delta != 1.0:
        for key in m.bonds:
            m.bonds[key][1] *= delta
    return m


def scan_grid(sc):
    """The scan points: tuples of values, the first coupling of the scan line outermost."""
    import itertools
    return list(itertools.product(*[np.linspace(a[1], a[2], a[3]).tolist() for a in sc['axes']]))


def scan_models(spec, kind, cl):
    """The models of a scan and a model to build the engine with.  Every model has the same bonds (a coupling that passes
    through zero keeps its bonds); the build model has the scanned coupling at a value that coincides with no other."""
    sc = spec['scan']; base = dict(spec['couplings']); snames = [a[0] for a in sc['axes']]
    names = [n for n in ('J1', 'J2', 'J3') if base.get(n) or n in snames]
    assert all(n in ('J1', 'J2', 'J3', 'Delta') for n in snames), 'scan: couplings J1, J2, J3 or the anisotropy Delta'
    unit = {n: spin_model(dict(spec, couplings={q: (1.0 if q == n else 0.0) for q in ('J1', 'J2', 'J3')}, bonds=[]), kind, cl) for n in names}
    extra = spin_model(dict(spec, couplings={q: 0.0 for q in ('J1', 'J2', 'J3')}), kind, cl) if spec['bonds'] else None

    def build(val):                                               # val: scanned coupling -> value
        m = edc.SpinHalfModel(cl.N); m.keep_zero = True
        dl = val.get('Delta', base.get('Delta', 1.0))
        for n in names:
            c = val.get(n, base.get(n, 0.0))
            for key, (jxy, jz) in unit[n].bonds.items():
                i, j = unit[n].orient[key]; m.exchange(i, j, c*jxy); m.zz(i, j, dl*c*jz)
        if extra is not None:
            for key, (jxy, jz) in extra.bonds.items():
                i, j = extra.orient[key]; m.exchange(i, j, jxy); m.zz(i, j, dl*jz)
        return m
    gen = {a[0]: a[1]+(a[2]-a[1])*g+0.0137 for a, g in zip(sc['axes'], (0.3819660113, 0.2360679775, 0.1458980338))}
    return [build(dict(zip(snames, p))) for p in scan_grid(sc)], build(gen)


def _store_opts(spec):
    """`solve stored=partial|0|1 reserve=N` as SubEngine keywords (reserve: vectors of the block left free on a GPU; a family
    scan keeps its basis there, so give it room, e.g. reserve=12 with light=1)."""
    sol = spec['solve']
    return dict(stored={'partial': 'partial', '1': True, '0': False}.get(sol.get('stored'), None), reserve_vectors=int(sol.get('reserve', 3)))


def _spin_engine(spec, mb, nup, cl, k, ir, flip, device, pops):
    """The engine of one block for scans: `solve engine=layout` (edc_glayout) or the sub engine."""
    if spec['solve'].get('engine', 'sub') == 'layout':
        import edc_glayout
        return edc_glayout.LayoutEngine(mb, nup, cl, k, ir, flip, device=device, point_ops=pops, fold=spec['solve'].get('fold'), expand=spec['solve'].get('expand'),
                                        reserve_vectors=int(spec['solve'].get('reserve', 4)))
    import edc_sub
    return edc_sub.SubEngine(mb, nup, cl, k, ir, flip, device=device, point_ops=pops, **_store_opts(spec))


def scan_spin(spec, kind, cl, nup, k, ir, flip, device, pops=None):
    """One sector along the scan.  method family: edc_scan.reduced_basis_scan; points: one solve per point from a random vector."""
    import edc_sub, edc_scan
    sc = spec['scan']; pts = scan_grid(sc)
    models, mb = scan_models(spec, kind, cl)
    t0 = time.perf_counter()
    e = _spin_engine(spec, mb, nup, cl, k, ir, flip, device, pops)
    out = dict(dim=int(e.dim))
    if e.dim == 0:
        return dict(out, E=None)
    if sc['method'] == 'points' or e.dim <= 400:
        e.retune(models[0]); rec = e.scan(models, tol=float(spec['solve'].get('tol', 1e-12)), warm=False)
        out.update(E=[float(np.atleast_1d(r['E'])[0]) for r in rec], products=sum(int(r['iterations']) for r in rec), method='points')
    else:
        mb_ = sc.get('basis') or (64 if len(sc['axes']) == 1 else 200)   # largest basis: 64 on a line, 200 on a grid (basis=N on the scan line)
        r = edc_scan.reduced_basis_scan(e, models, tol=sc['tol'], start=sc['start'], audit=sc['audit'], light=sc.get('light', False), max_basis=mb_,
                                        solve_tol=sc.get('solve_tol', 1e-12))
        out['converged'] = bool(max(r['error_estimate']) < 10*sc['tol'])
        out.update(E=r['E'], products=r['products'], method='family', basis=r['basis'], exact_solves=r['solves'],
                   largest_error_estimate=float(max(r['error_estimate'])), audit=[float(d) for _, d in r['audit']],
                   trained=[(pts[i][0] if len(pts[i]) == 1 else list(pts[i])) for i in r['trained']])
    out['seconds'] = time.perf_counter()-t0
    return out


def sentinel_energies(spec, kind, cl, nup, k, ir, flip, device, pops, idxs, eng=None):
    """Exact lowest level of one sector at the scan points idxs (one engine, retuned).  Returns dict(dim, E, products, eng);
    eng: (engine, models) of an earlier call for the same sector, to go on with it."""
    import edc_sub
    if eng is None:
        models, mb = scan_models(spec, kind, cl)
        eng = (_spin_engine(spec, mb, nup, cl, k, ir, flip, device, pops), models)
    e, models = eng
    out = dict(dim=int(e.dim), E=[], products=0, eng=eng)
    for i in idxs:
        if e.dim == 0:
            out['E'].append(None); continue
        e.retune(models[i]); E, it, _ = e.ground_energy(tol=1e-10, etol=1e-9)   # a Ritz value: above the level by less than 1e-9 |E|
        out['E'].append(float(np.atleast_1d(E)[0])); out['products'] += int(it)
    return out


def solve_spin(spec, kind, cl, m, nup, k, ir, flip, nev, tol, device, engine, devices, pops=None):
    out = {}
    assert pops is None or engine in ('sub', 'layout'), 'restricted point groups (complete scans): engine=sub or layout'
    assert not (spec['measure'] and engine == 'k'), 'measurements: engine sub or lean'
    if engine == 'lean':
        import edc_gpu as eg, edc_gpu_lean as gl
        e = gl.GPULean(m, nup, cl, k, ir, flip=flip)
        E, it, t = e.ground_energy(tol=tol, nev=nev)
        if spec['measure']:
            psi, Er, res = eg.ground_state_vector(e)
            out['residual'] = res
            for name, kv in spec['measure']:
                Q = tuple(float(v)*math.pi for v in kv['Q'].split(','))
                orbit = sorted({tuple(round(c % (2*math.pi), 12) for c in np.array(el.D4[o]) @ Q) for o in cl.point_ops()})
                out[f'{name}({kv["Q"]})'] = gl.m2(e, psi, cl, orbit)
    elif engine == 'k':
        import edc_k
        assert flip == 0, 'engine k: no spin flip'
        perms, chars, _ = cl.group(k, ir)
        e = edc_k.KEngine(m, nup, perms, chars)
        E, it, t = e.ground_energy(tol=tol)
    else:
        import edc_sub
        sol = spec['solve']
        if engine == 'layout':                                   # the product through layouts of the vector (edc_glayout)
            import edc_glayout
            assert not devices, 'engine=layout: one device'
            e = edc_glayout.LayoutEngine(m, nup, cl, k, ir, flip, device=device, point_ops=pops, fold=sol.get('fold'), expand=sol.get('expand'), reserve_vectors=int(sol.get('reserve', 4)))
            out.update(parts=e.p, layout_entries=int(sum(e.expanded.values())), table_bytes=e.stored_bytes, fold=e.fold, expand=e.expand)
        else:
            e = edc_sub.SubEngine(m, nup, cl, k, ir, flip, device=device, devices=devices, point_ops=pops, **_store_opts(spec))
        if sol.get('rule') == 'bound' and e.dim > 400 and not devices:       # several levels, stopped by an error bound
            import edc_levels
            lev, it, conv, info = edc_levels.lowest_levels(e, nev, tol=float(sol.get('tol', 1e-10)))
            E = lev[0] if nev == 1 else lev
            out.update(converged=bool(conv), error_estimates=info['residuals'])
        else:
            E, it, t = e.ground_energy(tol=tol, nev=nev)
        if getattr(e, '_st', None) is not None and getattr(e, 'stored_rows', e.dim) < e.dim:
            out['rows_stored'] = int(e.stored_rows)
        if spec['measure'] and e.dim:
            import edc_hubbard_sym as hs, edc_measure as em
            assert not devices or len(devices) == 1, 'measurements: one device'
            psi, Er, res = hs.ground_state_vector(e)
            out['residual'] = res
            for name, kv in spec['measure']:
                if 'Qr' in kv:
                    b1, b2 = em.reciprocal(cl); q = [float(v) for v in kv['Qr'].split(',')]
                    Q, key = q[0]*b1+q[1]*b2, f'S(Qr={kv["Qr"]})'
                else:
                    Q, key = np.array([float(v)*math.pi for v in kv['Q'].split(',')]), f'S(Q={kv["Q"]})'
                out[key] = em.structure_factor(e, psi, cl, Q)
            del psi
    out.update(dim=e.dim, E=E, iterations=it)
    if 'time_product' in spec['solve'] and engine == 'layout' and e.dim:
        out['product_seconds'] = e.time_matvec()
    if 'steps' in spec['solve'] and engine == 'sub':
        e0 = float(E[0]) if isinstance(E, (list, tuple, np.ndarray)) else float(E); t_s = time.perf_counter()
        out['steps'] = int(spec['solve']['steps']); out['solve_seconds_before_steps'] = None
        out['recursion_seconds'] = recursion(lambda x, y: e.matvec(x, y), int(e.dim), device, 1.05*abs(e0), int(spec['solve']['steps']))
    return out


def recursion(matvec, dim, device, scale, steps):
    """`solve ... steps=N`: N steps of v_{n+1} = 2 (H/s) v_n - v_{n-1} after the ground state (one product and two vector
    updates per step: the work of a spectral function or of a Chebyshev time evolution).  Returns the seconds used."""
    if device == 'gpu':
        import cupy as xp
        sync = lambda: xp.cuda.Device().synchronize()
    else:
        xp = np; sync = lambda: None
    v0 = xp.asarray(np.random.default_rng(5).standard_normal(dim)); v0 /= xp.linalg.norm(v0)
    v1 = xp.zeros(dim); w = xp.zeros(dim); matvec(v0, v1); v1 *= 1.0/scale; sync(); t0 = time.perf_counter()
    for n in range(steps):
        matvec(v1, w); w *= 2.0/scale; w -= v0; v0, v1, w = v1, w, v0
    sync()
    return time.perf_counter()-t0


def run_rows(spec):
    """engine=rows: total spin 0 together with both translations (edc_rows).  Rectangular tori, rows along x."""
    import edc_rows
    kind, kv = spec['lattice']; sec, sol = spec['sector'], spec['solve']
    if kind != 'rect' or spec['model'][0] != 'heisenberg':
        raise ValueError('engine=rows: lattice rect Lx= Ly= (rows along x), Heisenberg model')
    cp = spec['couplings']
    if any(abs(v) > 0 for k, v in cp.items() if k != 'J1') or spec['bonds'] or spec['measure']:
        raise ValueError('engine=rows: nearest-neighbour coupling J1 only, no extra bonds or measurements yet')
    if int(sec.get('S', 0)) != 0:
        raise ValueError('engine=rows: total spin S = 0 only')
    Lx, Ly = int(kv['Lx']), int(kv['Ly'])
    if Lx % 2 or Lx > 6:
        raise ValueError('engine=rows: row length Lx must be 2, 4 or 6 (put the short side along x)')
    kx, ky = [float(v) for v in sec.get('k', '0,0').split(',')]
    Kx, Kyi = kx*Lx/2, ky*Ly/2
    if abs(Kx-round(Kx)) > 1e-9 or abs(Kyi-round(Kyi)) > 1e-9:
        raise ValueError('engine=rows: k is not a momentum of this torus')
    t0 = time.perf_counter(); device = sol.get('device', 'cpu')
    e = edc_rows.RowsEngine(Lx, Ly, 'square', int(round(Kx)) % Lx, int(round(Kyi)) % Ly, int(sec.get('px', 1)), int(sec.get('py', 1)),
                            device=device, rowbasis=sol.get('rows', 'dimer'), J=cp.get('J1', 1.0))
    en, products, t_solve, res = e.ground_energy(nev=int(sol.get('levels', 1)), tol=float(sol.get('tol', 1e-12)))
    out = dict(lattice=kind, N=Lx*Ly, model='heisenberg', device=device, engine='rows', S=0, k_over_pi=[kx, ky], px=int(sec.get('px', 1)),
               py=int(sec.get('py', 1)), dim=e.dim, nnz=e.nnz, real=e.real, stored_bytes=e.stored_bytes, energies=en, products=products,
               residual=res, setup_seconds=e.setup_seconds, solve_seconds=t_solve, group=e.info['group'])
    if 'time_product' in sol:
        out['product_seconds'] = e.time_product()
    if 'steps' in sol:
        out['steps'] = int(sol['steps']); out['recursion_seconds'] = recursion(lambda x, y: e.matvec(x, y), e.dim, device, 1.05*abs(en[0]), int(sol['steps']))
    out['seconds'] = time.perf_counter()-t0
    return out


def run_layout3(spec):
    """engine=layout3: one product through three layouts of the vector (edc_layout3).  Kagome clusters whose point group is
    inversion only, nearest-neighbour coupling, the sector S_z = 0, k = 0, trivial irrep, spin-flip even."""
    import edc_layout3
    kind, cl = cluster(spec); sec, sol, cp = spec['sector'], spec['solve'], spec['couplings']
    if kind != 'kagome' or spec['model'][0] != 'heisenberg':
        raise ValueError('engine=layout3: lattice kagome, Heisenberg model')
    if (any(abs(v) > 0 for k, v in cp.items() if k not in ('J1', 'Delta')) or abs(cp.get('Delta', 1.0)-1.0) > 1e-15
            or spec['bonds'] or spec['measure'] or spec['scan'] or spec['sector_all']):
        raise ValueError('engine=layout3: nearest-neighbour coupling J1 only; no anisotropy, extra bonds, scans or measurements yet')
    k = [float(v) for v in sec.get('k', '0,0').split(',')]
    if (abs(float(sec.get('Sz', 0))) > 1e-12 or any(abs(v) > 1e-12 for v in k) or sec.get('irrep', 'X0') != 'X0'
            or int(sec.get('flip', 0)) != 1 or int(sol.get('levels', 1)) != 1):
        raise ValueError('engine=layout3: sector Sz=0 k=0,0 irrep=X0 flip=+1, lowest level only')
    t0 = time.perf_counter(); device = sol.get('device', 'cpu')
    e = edc_layout3.Layout3Engine(cl, J=cp.get('J1', 1.0), device=device)
    E, it, t_solve = e.ground_energy(tol=float(sol.get('tol', 1e-12)))
    out = dict(lattice=kind, N=cl.N, model='heisenberg', device=device, engine='layout3', nup=cl.N//2, k_over_pi=k, irrep='X0', flip=1,
               dim=e.dim, expanded_entries=e.expanded, group_order=e.group_order, real=True, E=E, iterations=it,
               setup_seconds=e.setup_seconds, solve_seconds=t_solve)
    if 'time_product' in sol:
        out['product_seconds'] = e.time_product()
    out['seconds'] = time.perf_counter()-t0
    return out


def run(spec):
    if spec['solve'].get('method') == 'dmrg':
        import edc_dmrg
        return edc_dmrg.run_spec(spec)
    if spec['solve'].get('engine') == 'rows':
        return run_rows(spec)
    if spec['solve'].get('engine') == 'layout3':
        return run_layout3(spec)
    kind, cl = cluster(spec)
    mname, mkv = spec['model']
    sec, sol = spec['sector'], spec['solve']
    nev = int(sol.get('levels', 1)); tol = float(sol.get('tol', 1e-12)); device = sol.get('device', 'cpu')
    engine = sol.get('engine', 'sub'); devices = [int(v) for v in sol['devices'].split(',')] if 'devices' in sol else None
    assert not (spec['measure'] and mname == 'hubbard'), 'measurements: spin models only'
    t0 = time.perf_counter()
    out = dict(lattice=kind, N=cl.N, model=mname, device=device, engine=engine if mname == 'heisenberg' else 'hubbard_sym')
    import edc_hubbard_sym as hs
    if mname == 'hubbard':
        bonds = hs.hubbard_bonds(cl, float(mkv.get('t', 1.0)), float(mkv.get('tp', 0.0)))
        U = float(mkv.get('U', 0.0)); nu, nd = int(sec['Nup']), int(sec['Ndn'])
        def one(k, ir, fl, pops):
            e = hs.HubbardSym(cl, bonds, U, nu, nd, k, ir, fl, device=device, point_ops=pops)
            E, it, t = e.ground_energy(tol=tol, nev=nev) if e.dim else (None, 0, 0.0)
            return dict(dim=e.dim, E=E, iterations=it)
        out.update(U=U, Nup=nu, Ndn=nd)
    else:
        m = spin_model(spec, kind, cl)
        nu = cl.N//2+int(round(float(sec.get('Sz', 0))))
        def one(k, ir, fl, pops):
            if spec['scan']:
                assert engine in ('sub', 'layout') and not devices and nev == 1 and not spec['measure'], 'scan: engine sub or layout, one device, lowest level, no measurements'
                return scan_spin(spec, kind, cl, nu, k, ir, fl, device, pops)
            return solve_spin(spec, kind, cl, m, nu, k, ir, fl, nev, tol, device, engine, devices, pops)
        out.update(nup=nu)
        if spec['scan']:
            sc = spec['scan']; out['scan'] = dict(sc, values=np.linspace(sc['first'], sc['last'], sc['points']).tolist())
            if len(sc['axes']) > 1:
                out['scan']['grid'] = [list(p) for p in scan_grid(sc)]
    if spec['sector_all']:
        nd = cl.N-nu if mname != 'hubbard' else nd
        res, best, tot = [], None, 0
        named = None
        if spec.get('sector_named'):                             # every irrep of every little group resolved, nothing solved twice
            assert mname != 'hubbard' and hasattr(cl, 'irrep_table'), 'sector ... named: spin models on the hexagonal lattices'
            named = hs.sectors_named(cl, nu, nd)
            seclist = [(b['k'], b['irrep'], b['flip'], b['allowed'], b['star']*b['degeneracy']) for b in named]
        else:
            seclist = list(hs.sectors(cl, nu, nd))
        stage1 = None
        # a block whose characters are the complex conjugates of an earlier block's has the same spectrum (real H):
        # it is not solved, its record is a copy marked conjugate_of (solve pairs=0 solves every block)
        partner = hs.conjugate_partners(cl, seclist) if (int(sol.get('pairs', 1)) and named is None) else [None]*len(seclist)
        copy = lambda r, j: dict(r, conjugate_of=j, products=0, iterations=0, seconds=0.0)
        if spec['scan'] and spec['scan'].get('screen') and spec['scan']['method'] == 'family':
            # screening: all sectors at a loose tolerance; only those within a margin of the lowest level anywhere on the scan
            # are solved again at the full tolerance (the lowest level over the sectors then has the full accuracy)
            sc0 = spec['scan']; spec1 = dict(spec, scan=dict(sc0, tol=sc0['screen_tol'], solve_tol=1e-8, audit=1))
            stage1 = []
            for i1, (k, ir, fl, pops, mult) in enumerate(seclist):
                stage1.append(copy(stage1[partner[i1]], partner[i1]) if partner[i1] is not None
                              else scan_spin(spec1, kind, cl, nu, k, ir, fl, device, pops))
            E1 = np.array([r1['E'] for r1 in stage1 if r1['E'] is not None]); Emin = E1.min(axis=0)
            margin = max(1e-3, 100*sc0['screen_tol'])*np.maximum(1.0, np.abs(Emin))
            keepers = {i for i, r1 in enumerate(stage1) if r1['E'] is not None and np.any(np.array(r1['E'])-Emin < margin)}
            out['screen'] = dict(sectors=len(seclist), refined=len(keepers), products=int(sum(r1.get('products', 0) for r1 in stage1)))
        pruned = {}
        if spec['scan'] and spec['scan'].get('prune') and spec['scan']['method'] == 'family' and len(spec['scan']['axes']) == 1:
            # Pruning by concavity.  H is linear in the scanned coupling, so the lowest level of a sector is a concave
            # function of it: between two points where it is known exactly it lies on or above the chord.  The lowest
            # sector is scanned first (its energies bound the lowest level of all sectors from above); a sector whose
            # chords stay above that bound at every scan point cannot hold the lowest level anywhere and is not scanned.
            sc0 = spec['scan']; P_ = sc0['points']; ns_ = max(2, sc0['prune'])
            sidx = sorted({int(round(x)) for x in np.linspace(0, P_-1, ns_)}); xs = np.linspace(sc0['first'], sc0['last'], P_)
            sent = {}
            for i1, (k, ir, fl, pops, mult) in enumerate(seclist):
                if partner[i1] is None:
                    sent[i1] = sentinel_energies(spec, kind, cl, nu, k, ir, fl, device, pops, sidx); del sent[i1]['eng']
                    sent[i1]['idx'] = list(sidx)
            live = [i1 for i1 in sent if sent[i1]['dim'] > 0]
            chord = {i1: np.interp(xs, xs[sidx], np.array(sent[i1]['E'])) for i1 in live}
            order = sorted(live, key=lambda i1: float(np.min(chord[i1]-np.min([chord[j] for j in live], axis=0))))
            ub = None; scanned = {}; more = int(sol.get('refine', 4))
            for i1 in order:                                     # closest to the lowest level first: the bound tightens early
                k, ir, fl, pops, mult = seclist[i1]
                if ub is not None:
                    marg = 1e-7*np.maximum(1.0, np.abs(ub)); eng = None; extra = 0   # margin 100 times the sentinel error
                    while not np.all(chord[i1] > ub+marg):       # one more exact value where the chord is deepest below the bound
                        p_ = int(np.argmin(chord[i1]-ub))
                        if p_ in sent[i1]['idx'] or extra >= more:
                            break
                        r_ = sentinel_energies(spec, kind, cl, nu, k, ir, fl, device, pops, [p_], eng); eng = r_['eng']; extra += 1
                        sent[i1]['idx'].append(p_); sent[i1]['E'].append(r_['E'][0]); sent[i1]['products'] += r_['products']
                        o_ = np.argsort(sent[i1]['idx'])
                        chord[i1] = np.interp(xs, xs[np.array(sent[i1]['idx'])[o_]], np.array(sent[i1]['E'])[o_])
                    del eng
                    if np.all(chord[i1] > ub+marg):
                        pruned[i1] = dict(dim=sent[i1]['dim'], E=None, products=sent[i1]['products'], pruned=True,
                                          lower_bound=chord[i1].tolist(), exact_points=len(sent[i1]['idx']), method='chord')
                        continue
                r1 = scan_spin(spec, kind, cl, nu, k, ir, fl, device, pops); r1['products'] = r1.get('products', 0)+sent[i1]['products']
                scanned[i1] = r1; ub = np.array(r1['E']) if ub is None else np.minimum(ub, np.array(r1['E']))
            for i1 in sent:
                if i1 not in live:
                    pruned[i1] = dict(dim=0, E=None, products=0)
            pruned.update(scanned)
            out['prune'] = dict(sectors=len(seclist), solved=len(sent), scanned=len(scanned), sentinels=len(sidx),
                                sentinel_products=int(sum(v['products'] for v in sent.values())))
        for idx, (k, ir, fl, pops, mult) in enumerate(seclist):  # spins: flip sectors when N_up = N - N_up
            if partner[idx] is not None:
                r = copy(res[partner[idx]], partner[idx])
            elif idx in pruned:
                r = pruned[idx]
            elif stage1 is not None and idx not in keepers:
                r = dict(stage1[idx], screened=True)
            else:
                r = one(k, ir, fl, pops)
                if stage1 is not None:
                    r['products'] = r.get('products', 0)+stage1[idx].get('products', 0)
            r.update(k_over_pi=[v/math.pi for v in k], irrep=ir, flip=fl, multiplicity=mult); res.append(r)
            if named is not None:                                # star of k, degeneracy of a level, characters of the block
                b = named[idx]
                r.update(star=b['star'], degeneracy=b['degeneracy'], dimension_of_irrep=b['dimension_of_irrep'],
                         group='little group' if b['allowed'] is None else 'rotations',
                         chars={o: [round(c.real, 12), round(c.imag, 12)] for o, c in b['chars'].items()})
            tot += mult*r['dim']
            if spec['scan']:
                continue
            E0 = r['E'] if not isinstance(r['E'], list) else r['E'][0]
            if E0 is not None and (best is None or E0 < best[0]):
                best = (E0, r)
        if spec['scan']:                                         # lowest level over all sectors at every scan point
            have = [i for i, r in enumerate(res) if r['E'] is not None]
            Es = np.array([res[i]['E'] for i in have])
            out.update(E=Es.min(axis=0).tolist(), ground_sector=[int(have[i]) for i in Es.argmin(axis=0)],
                       products=sum(r.get('products', 0) for r in res))
        out.update(sectors=res, ground_state=best[1] if best else None, sum_mult_dim=tot,
                   full_dim=(math.comb(cl.N, nu)*math.comb(cl.N, nd) if mname == 'hubbard' else math.comb(cl.N, nu)))
    else:
        k = tuple(float(v)*math.pi for v in sec.get('k', '0,0').split(','))
        ir = sec.get('irrep', 'X0' if kind in HEX else 'A1'); flip = int(sec.get('flip', 0))
        out.update(k_over_pi=[v/math.pi for v in k], irrep=ir, flip=flip, **one(k, ir, flip, None))
    out['seconds'] = time.perf_counter()-t0
    return out


def main(argv=None):
    """Run a model file and print its result as JSON."""
    args = sys.argv[1:] if argv is None else argv
    if args in (['--help'], ['-h']):
        print(__doc__.strip())
        return 0
    if len(args) != 1:
        sys.exit('usage: edc model.edc')
    with open(args[0]) as model_file:
        print(json.dumps(run(parse(model_file.read()))))
    return 0


if __name__ == '__main__':
    sys.exit(main())
