#!/usr/bin/env python3
"""Zero temperature: the lowest levels of every symmetry block of the S_z sectors of a spin-1/2 cluster.
Model: H = sum_shells J_s sum_<ij> [ S^x_i S^x_j + S^y_i S^y_j + delta S^z_i S^z_j ]  (delta = 1: Heisenberg).
Blocks from edc_hubbard_sym.sectors (complete: one momentum per star, every 1D irrep of the little group or of its
rotation subgroup, spin flip at S_z = 0); engine edc_sub.SubEngine on one device.
  spectra:  --J 1,0.1 [--delta 1] --nev 6     the lowest nev distinct levels of every block (Lanczos, tolerance --tol)
  scan:     --scan J2=-0.3:0.3:31 [--J 1,0]   the lowest level of every block along the scan, each block as one family
            --scan delta=0:2:21               (edc_scan.reduced_basis_scan; blocks up to 400 states point by point)
One JSON line per block is appended to --out; blocks already there are skipped.  Every record carries the momentum, the
characters of the point operations and the multiplicity (size of the momentum star).

Usage: levels_run.py <kind> <A1x,A1y,A2x,A2y> --out file.jsonl [--nup a:b] [--part i/n] [--device gpu|cpu] [--max-seconds S] ..."""
import sys, os, json, time, math, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import edc, edc_lattices as L, edc_hubbard_sym as hs, edc_sub

ap = argparse.ArgumentParser()
ap.add_argument('kind'); ap.add_argument('vectors')
ap.add_argument('--out', required=True); ap.add_argument('--J', default='1'); ap.add_argument('--delta', type=float, default=1.0)
ap.add_argument('--nev', type=int, default=1); ap.add_argument('--tol', type=float, default=1e-10)
ap.add_argument('--maxit', type=int, default=2000); ap.add_argument('--scan', default='')
ap.add_argument('--scan-tol', type=float, default=1e-9)
ap.add_argument('--nup', default=''); ap.add_argument('--part', default='0/1'); ap.add_argument('--device', default='gpu')
ap.add_argument('--max-seconds', type=float, default=1e18); ap.add_argument('--max-dim', type=float, default=1e18)
ap.add_argument('--list', action='store_true')
ap.add_argument('--rule', default='hold', choices=('hold', 'bound'),
                help='stopping rule for several levels: hold = lanczos_levels below; bound = edc_levels.lowest_levels (error estimate)')
ap.add_argument('--stored', default='auto', choices=('auto', 'partial'),
                help='partial: when the connectivity does not fit on the card, store the rows that do (SubEngine stored=\'partial\')')
ap.add_argument('--mirror', action='store_true',
                help='blocks of an abelian subgroup that contains a mirror (and the twofold rotation where present) instead of the rotations')
args = ap.parse_args()
T_START = time.perf_counter()
a = [int(v) for v in args.vectors.split(',')]; J = [float(v) for v in args.J.split(',')]
part, nparts = (int(v) for v in args.part.split('/'))
cl = L.LatticeCluster(args.kind, (a[0], a[1]), (a[2], a[3])); N = cl.N
lo, hi = (int(v) for v in args.nup.split(':')) if args.nup else ((N+1)//2, N)
assert (N+1)//2 <= lo <= hi <= N


def make_model(Js, delta, shells):
    """Heisenberg/XXZ model on the given shells; a coupling that is zero keeps its bonds (same bonds along a scan)."""
    m = edc.SpinHalfModel(N); m.keep_zero = True
    for sh in shells:
        Jv = Js[sh-1] if sh-1 < len(Js) else 0.0
        for i, j in cl.bonds(sh):
            m.exchange(i, j, Jv); m.zz(i, j, Jv*delta)
    return m


scan = None
if args.scan:
    name, rng = args.scan.split('='); first, last, npts = rng.split(':')
    values = np.linspace(float(first), float(last), int(npts))
    assert name in ('J2', 'delta')
    Js0 = (J+[0.0, 0.0])[:2]
    shells = [1, 2] if (name == 'J2' or Js0[1]) else [1]
    if name == 'J2':
        models = [make_model([Js0[0], float(v)], args.delta, shells) for v in values]
        build = make_model([Js0[0], float(first)+(float(last)-float(first))*0.3819660113+0.0137], args.delta, shells)
    else:
        models = [make_model(Js0, float(v), shells) for v in values]
        build = make_model([Js0[0], Js0[1] if Js0[1] else 0.0], 0.7319+0.01*float(last), shells)
        if len(shells) == 2 and abs(Js0[0]-Js0[1]) < 1e-12:
            raise SystemExit('delta scan: J1 = J2 would merge the bond classes')
    scan = dict(name=name, values=values.tolist())
else:
    shells = [s for s in (1, 2, 3) if s-1 < len(J) and J[s-1]]
    model = make_model(J, args.delta, shells)

def mirror_sectors(nup):
    """Like hs.sectors, but where the little group is not abelian the blocks belong to {E, mirror} or
    {E, R3, mirror, R3 mirror} (R3 = twofold rotation) instead of the rotation subgroup: they resolve the mirror parity."""
    ops = cl.point_ops()
    key = lambda k: tuple(round((float(v) % (2*math.pi))/(2*math.pi)*4*cl.Nc) % (4*cl.Nc) for v in k)
    seen, out = set(), []
    for k in cl.momenta():
        if key(k) in seen:
            continue
        orbit = {key(cl.act_k(o, k)) for o in ops}
        seen |= orbit
        lg = cl.little_group(k)
        allowed = None
        if not all(cl._mul(x, y) == cl._mul(y, x) for x in lg for y in lg):
            s = sorted(o for o in lg if o.startswith('S'))[0]
            allowed = ['E', 'R3', s, cl._mul('R3', s)] if 'R3' in lg else ['E', s]
        for ir in cl.irrep_table(k, allowed):
            for fl in ((1, -1) if 2*nup == N else (0,)):
                out.append((tuple(float(x) for x in k), ir, fl, allowed, len(orbit)))
    return out


def lanczos_levels(e, nev, tol, maxit, seed=42, every=40, hold=5):
    """The lowest nev distinct levels of the engine's block by plain two-vector Lanczos (the recurrence of
    edc_hubbard_sym.lanczos).  Every `every` steps the lowest Ritz values of the tridiagonal matrix are computed
    (bisection).  A Ritz value that has not moved by more than tol since the previous check counts as converged: it is a
    level of the block (Lanczos without reorthogonalisation returns converged levels again; the copies are merged, and
    copies on their way down, which still move, are ignored).  The run stops when the lowest nev converged values have
    not changed over `hold` consecutive checks.  Returns (levels, steps, converged flag, number of moving Ritz values
    below the highest returned level).  Levels that are exactly degenerate inside the block are returned once."""
    from scipy.linalg import eigvalsh_tridiagonal
    xp = e.xp
    axpy, dot = hs._vec_ops(xp, e.real, False)
    dt = np.float64 if e.real else np.complex128
    v = hs._random_vector(xp, e.dim, e.real, seed, False)
    v *= 1.0/math.sqrt(dot(v, v).real)
    u = xp.zeros(e.dim, dt)
    A, B, hist, prev = [], [], [], None
    top = min(maxit, e.dim)
    lev, below = [], 0
    for it in range(1, top+1):
        e.matvec(v, u, accumulate=True)
        al = float(np.real(dot(v, u))); axpy(-al, v, u); be = math.sqrt(float(np.real(dot(u, u))))
        A.append(al); B.append(be)
        last = it == top or be < 1e-12
        if it % every == 0 or last:
            w = eigvalsh_tridiagonal(np.array(A), np.array(B[:-1]), select='i', select_range=(0, min(it-1, 10*nev+60)))
            scale = max(1.0, abs(w[0]))
            if prev is not None:
                fixed = np.array([np.abs(prev-x).min() < tol*scale for x in w])
                lev = []
                for x, ok in zip(w, fixed):
                    if not ok or (lev and abs(x-lev[-1]) < 1e-8*scale):
                        continue
                    if len(lev) == nev:
                        break
                    lev.append(float(x))
                below = int(sum(1 for x, ok in zip(w, fixed) if not ok and lev and x < lev[-1]))
                hist.append(lev)
                full = len(lev) == nev or it == e.dim
                stable = full and len(hist) > hold and all(len(h) == len(lev) and max(abs(a_-b_) for a_, b_ in zip(h, lev)) < 10*tol*scale
                                                        for h in hist[-hold-1:-1])
                if stable or it == e.dim:
                    return lev, it, True, below
            prev = w
        if last:
            break
        u *= 1.0/be; v *= -be
        v, u = u, v
    return lev, it, False, below


tasks = []
for nup in range(lo, hi+1):
    blocks = mirror_sectors(nup) if args.mirror else hs.sectors(cl, nup, N-nup)
    per_k = {}
    for b in blocks:
        per_k[b[0]] = per_k.get(b[0], 0)+1
    for bi, b in enumerate(blocks):
        tasks.append((math.comb(N, nup)/(cl.Nc*per_k[b[0]]), nup, bi, b))
tasks.sort(key=lambda t: (-t[0], t[1], t[2]))
mine = [t for i, t in enumerate(tasks) if i % nparts == part and t[0] <= args.max_dim]
done = set()
if os.path.exists(args.out):
    for line in open(args.out):
        try:
            r = json.loads(line); done.add((r['nup'], r['block']))
        except (ValueError, KeyError):
            pass
if args.list:
    for est, nup, bi, b in mine:
        print(nup, bi, b[1], b[2], b[4], f'{est:.3g}', 'done' if (nup, bi) in done else '')
    sys.exit(0)
if args.device == 'gpu':
    import cupy as cp
    sync = lambda: cp.cuda.Device().synchronize()
    freemem = lambda: cp.get_default_memory_pool().free_all_blocks()
else:
    sync = freemem = lambda: None

for est, nup, bi, (k, ir, fl, pops, mult) in mine:
    if (nup, bi) in done:
        continue
    if time.perf_counter()-T_START > args.max_seconds:
        print('time limit reached before', nup, bi, file=sys.stderr, flush=True)
        break
    t0 = time.perf_counter()
    chars = {o: [complex(c).real, complex(c).imag] for o, c in cl.irrep_table(k, pops)[ir].items()}
    rec = dict(nup=nup, block=bi, k_over_pi=[v/math.pi for v in k], irrep=ir, flip=fl, mult=mult, chars=chars, N=N,
               kind=args.kind, vectors=a, J=J, delta=args.delta, subgroup='mirror' if args.mirror else 'rotations')
    e = edc_sub.SubEngine(build if scan else model, nup, cl, k, ir, fl, device=args.device, point_ops=pops,
                          **(dict(stored='partial', reserve_vectors=4) if args.stored == 'partial' else {}))
    sync(); t1 = time.perf_counter()
    rec.update(dim=int(e.dim), setup_seconds=t1-t0)
    if e.dim == 0:
        pass
    elif scan is None:
        if e.dim <= 400:
            E, its, sec = e.ground_energy(tol=args.tol, maxit=args.maxit, nev=max(args.nev, 2) if e.dim > 1 else 1)
            E = [float(x) for x in np.atleast_1d(E)][:args.nev]; conv = True; rs = []
        elif args.rule == 'bound':
            import edc_levels
            E, its, conv, info = edc_levels.lowest_levels(e, args.nev, tol=args.tol, maxit=args.maxit); sync(); sec = time.perf_counter()-t1
            E = [float(x) for x in E]; rs = [float(x) for x in info['residuals']]; rec['unconverged_below'] = 0
        else:
            E, its, conv, below = lanczos_levels(e, args.nev, args.tol, args.maxit); sync(); sec = time.perf_counter()-t1
            rec['unconverged_below'] = below; rs = []
        has = getattr(e, '_st', None) is not None
        rec.update(E=E, iterations=int(its), converged=bool(conv), residuals=rs, solve_seconds=sec, real=bool(e.real),
                   stored=has, rule=args.rule, rows_stored=int(getattr(e, 'stored_rows', e.dim if has else 0)) if has else 0)
    else:
        rec['scan'] = scan
        if e.dim <= 400:
            e.retune(models[0]); out = e.scan(models, tol=1e-12, warm=False)
            rec.update(E=[float(np.atleast_1d(r['E'])[0]) for r in out], method='points')
        else:
            import edc_scan
            r = edc_scan.reduced_basis_scan(e, models, tol=args.scan_tol, start='random', audit=2)
            rec.update(E=r['E'], method='family', basis=r['basis'], products=r['products'], exact_solves=r['solves'],
                       largest_error_estimate=float(max(r['error_estimate'])), audit=[float(d) for _, d in r['audit']])
            del r
        rec.update(real=bool(e.real), stored=getattr(e, '_st', None) is not None, solve_seconds=time.perf_counter()-t1)
    with open(args.out, 'a') as f:
        f.write(json.dumps(rec)+'\n')
    print(f'nup {nup} block {bi} {ir} flip {fl} dim {e.dim} stored {rec.get("stored")} its {rec.get("iterations")} '
          f'{time.perf_counter()-t0:.1f} s', file=sys.stderr, flush=True)
    del e
    freemem()
print('part finished', file=sys.stderr, flush=True)
