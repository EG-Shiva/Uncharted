import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(np.max(g))

def _norm_rows(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return x / n

def _seed_to_vec(a0, amps, phases):
    # amps, phases length 6 for freq 1..6, j=0..12
    d = 13
    j = np.arange(d)
    v = np.full(d, float(a0))
    for f in range(1, 7):
        v = v + float(amps[f-1]) * np.cos(2*np.pi*f*j/d + float(phases[f-1]))
    nv = float(np.linalg.norm(v))
    if nv < 1e-12 or not np.isfinite(nv):
        v = np.ones(d)
        nv = float(np.linalg.norm(v))
    return v / nv

def _build_cyclic(seeds, extra_idx=None):
    # seeds: (K,13) already unit; use 7 full orbits + 4 of 8th
    d = 13
    K = seeds.shape[0]
    pts = []
    # assume K>=8; first 7 full
    for s in range(min(7, K)):
        v = seeds[s] / np.linalg.norm(seeds[s])
        for sh in range(13):
            pts.append(np.roll(v, sh))
    need = 95 - len(pts)
    if need > 0:
        # use seed 7 (index 7) partial orbit
        si = 7 if K > 7 else 0
        v = seeds[si] / np.linalg.norm(seeds[si])
        if extra_idx is None:
            extra_idx = [0, 3, 6, 9][:need]
        for sh in extra_idx[:need]:
            pts.append(np.roll(v, int(sh)))
    # if still need (K<8 edge), pad with rolls of seed0 different? already enough
    X = np.stack(pts[:95], axis=0)
    return _norm_rows(X)

def _random_seeds(rng):
    K = 8
    seeds = np.zeros((K, 13))
    for s in range(K):
        a0 = float(rng.standard_normal() * 0.5)
        amps = np.abs(rng.standard_normal(6)) * 1.0
        phases = rng.uniform(0, 2*np.pi, size=6)
        # occasional flat-spectrum (good autocorrelation)
        if rng.random() < 0.25:
            amps = np.ones(6)
            phases = rng.uniform(0, 2*np.pi, size=6)
            a0 = float(rng.standard_normal() * 0.2)
        seeds[s] = _seed_to_vec(a0, amps, phases)
    # random choice of 4-subset shifts for partial orbit
    perm = rng.permutation(13)[:4]
    return seeds, perm

def _annealed_refine(Xc, steps=800):
    n, d = Xc.shape
    Xw = Xc.copy()
    V = np.zeros_like(Xw)
    mu = 0.9
    best_X = Xw.copy()
    best_v = _max_cos(Xw)
    for s in range(steps):
        frac = s / float(max(1, steps - 1))
        beta = 30.0 * ((500.0 / 30.0) ** frac)
        step = 0.02 * (1.0 - 0.7 * frac)
        G = Xw @ Xw.T
        np.fill_diagonal(G, -np.inf)
        m = float(np.max(G))
        # avoid overflow: beta*(G-m) <=0
        E = np.exp(beta * (G - m))
        np.fill_diagonal(E, 0.0)
        grad = E @ Xw
        gdot = np.sum(grad * Xw, axis=1, keepdims=True)
        grad -= gdot * Xw
        gn = float(np.linalg.norm(grad, axis=1).max())
        if gn > 1e-12:
            grad = grad / gn
        V = mu * V + (1.0 - mu) * grad
        Xw = Xw - step * V
        Xw = _norm_rows(Xw)
        if (s + 1) % 100 == 0 or s == steps - 1:
            v = _max_cos(Xw)
            if v < best_v:
                best_v = v
                best_X = Xw.copy()
    return best_X, best_v

def _lp_polish(x, seconds):
    x = _norm_rows(x)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        if time.time() - start > seconds - 0.2:
            break
        c = (x @ x.T)[iu]
        m = float(c.max())
        sel = c > m - max(20 * radius, 1e-9)
        if int(np.sum(sel)) > 1500:
            idx = np.argpartition(c, -1500)[-1500:]
            mask = np.zeros_like(sel, dtype=bool)
            mask[idx] = True
            sel = mask
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        if p == 0:
            break
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate(
            [I[:, None] * d + np.arange(d),
             J[:, None] * d + np.arange(d),
             np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix(
            (x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))),
            shape=(n, nv)).tocsr()
        obj = np.zeros(nv)
        obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c[sel], A_eq=A_eq,
                          b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                          method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y = _norm_rows(y)
        v = _max_cos(y)
        if v < best:
            x, best, radius = y, v, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def solve():
    t0 = time.time()
    rng = np.random.default_rng(12345)
    # load record / best for fallback return
    rec = None
    rec_val = np.inf
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn).astype(float)
            if a.shape == (95, 13):
                a = _norm_rows(a)
                v = _max_cos(a)
                if v < rec_val:
                    rec_val = v
                    rec = a.copy()
        except Exception:
            continue
    if rec is None:
        z = rng.standard_normal((95, 13))
        rec = _norm_rows(z)
        rec_val = _max_cos(rec)

    # Phase 1: cyclic-orbit random search, at most ~50s
    t_phase1_end = t0 + 50.0
    elites = []  # list of (val, X, seeds)
    best_sym_val = np.inf
    best_sym_X = None
    n_eval = 0
    # seed the search with a few draws
    while time.time() < t_phase1_end:
        # 30% mutate best elite, else fresh
        if elites and rng.random() < 0.30:
            # pick one of top elites to perturb in Fourier domain
            ei = int(rng.integers(0, len(elites)))
            _, _, (s0, perm0) = elites[ei]
            # perturb: convert seeds back via small tangent noise then renormalize
            # plus occasionally resample phases of one seed
            seeds = s0 + 0.25 * rng.standard_normal(s0.shape)
            seeds = _norm_rows(seeds)
            perm = perm0
            if rng.random() < 0.3:
                perm = rng.permutation(13)[:4]
        else:
            seeds, perm = _random_seeds(rng)
        X = _build_cyclic(seeds, extra_idx=perm)
        v = _max_cos(X)
        n_eval += 1
        if v < best_sym_val:
            best_sym_val = v
            best_sym_X = X.copy()
        # maintain top-3 distinct
        elites.append((v, X.copy(), (seeds.copy(), perm.copy())))
        elites.sort(key=lambda e: e[0])
        elites = elites[:3]
        # safety: if many evals, briefly yield; loop is fast
        if n_eval > 30000:
            break
    if not elites:
        seeds, perm = _random_seeds(rng)
        X = _build_cyclic(seeds, extra_idx=perm)
        elites = [(_max_cos(X), X, (seeds, perm))]

    # Phase 2: symmetry breaking + annealed softmax Adam, no LP
    # take best 2-3 symmetric codes x noises 0.01,0.03,0.08
    sigmas = [0.01, 0.03, 0.08]
    cands = []
    for _, Xsym, _ in elites[:3]:
        for sig in sigmas:
            cands.append((Xsym, sig))
    # limit to time budget: leave 27s for final LP
    t_phase2_end = t0 + 148.0
    # steps per candidate adapted to count (9 cands)
    steps_each = 700
    # if little time, reduce
    refined_best = None
    refined_val = np.inf
    for (Xsym, sig) in cands:
        if time.time() > t_phase2_end:
            break
        noise = rng.standard_normal(Xsym.shape)
        # tangent projection
        dot = np.sum(noise * Xsym, axis=1, keepdims=True)
        noise = noise - dot * Xsym
        Xt = Xsym + sig * noise
        Xt = _norm_rows(Xt)
        Xr, vr = _annealed_refine(Xt, steps=steps_each)
        if vr < refined_val:
            refined_val = vr
            refined_best = Xr.copy()
    if refined_best is None:
        refined_best = elites[0][1].copy()
        refined_val = elites[0][0]

    # Phase 3: LP-polish only best refined for final ~25s
    remaining_total = 176.0 - (time.time() - t0)
    lp_time = min(25.0, max(2.0, remaining_total - 1.5))
    if lp_time > 1.0:
        try:
            polished = _lp_polish(refined_best, lp_time)
            pv = _max_cos(polished)
        except Exception:
            polished = refined_best
            pv = refined_val
    else:
        polished = refined_best
        pv = refined_val

    # return better of polished and loaded record
    if pv < rec_val:
        out = _norm_rows(polished)
    else:
        out = _norm_rows(rec)
    return out
