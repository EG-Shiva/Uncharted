import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _mc(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(np.max(g))

def _nm(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return x / n

def _lp_polish_deadline(x, deadline):
    x = _nm(x.copy())
    n, d = x.shape
    best = _mc(x)
    radius = 1e-3
    iu = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        G = x @ x.T
        c = G[iu]
        m = float(np.max(c))
        sel = c > m - max(20.0 * radius, 1e-9)
        I = iu[0][sel]; J = iu[1][sel]
        csel = c[sel]
        p = len(I)
        if p == 0:
            break
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d),
                                       np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq,
                          b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                          method="highs")
        except Exception:
            break
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y = _nm(y)
        v = _mc(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x, best

def solve():
    t0 = time.time()
    HARD_END = t0 + 170.0
    SEARCH_END = t0 + 132.0
    rng = np.random.RandomState(20261031)
    try:
        xb = np.load("best.npy").astype(float)
    except Exception:
        xb = rng.randn(168, 4)
    xb = _nm(np.asarray(xb, dtype=float))
    if xb.shape != (168, 4):
        xb = _nm(rng.randn(168, 4))
    try:
        xr = np.load("record.npy").astype(float)
        xr = _nm(xr)
        if xr.shape == (168, 4) and _mc(xr) < _mc(xb):
            xb = xr.copy()
    except Exception:
        pass
    n0_full, d = 168, 4
    best_val = _mc(xb)
    x_best = xb.copy()

    def refine(x_init, steps, beta0=30.0, beta1=600.0, lr0=0.015, lr1=0.0007, mu=0.88):
        x = x_init.copy()
        vel = np.zeros_like(x)
        lb0 = np.log(beta0); lb1 = np.log(beta1)
        for s in range(steps):
            if time.time() > SEARCH_END:
                break
            f = s / max(1, steps - 1)
            beta = float(np.exp(lb0 + (lb1 - lb0) * f))
            lr = lr0 * (1.0 - f) + lr1 * f
            G = x @ x.T
            np.fill_diagonal(G, -2.0)
            cmax = float(np.max(G))
            E = np.exp(np.clip(beta * (G - cmax), -25.0, 0.0))
            np.fill_diagonal(E, 0.0)
            rs = np.sum(E, axis=1, keepdims=True) + 1e-12
            grad = (E @ x) / rs
            dot = np.sum(grad * x, axis=1, keepdims=True)
            gt = grad - dot * x
            sc = np.mean(np.linalg.norm(gt, axis=1)) + 1e-12
            gt = gt / sc
            vel = mu * vel + (1.0 - mu) * gt
            x = x - lr * vel
            x = _nm(x)
        return x

    def refine_holes_vec(cand, Xkeep, iters=60, beta=80.0, lr=0.05):
        # cand (C,4), Xkeep (N,4) : minimize max dot
        C = cand.copy()
        for _ in range(iters):
            D = C @ Xkeep.T
            m = D.max(axis=1, keepdims=True)
            W = np.exp(np.clip(beta * (D - m), -20.0, 0.0))
            W /= (W.sum(axis=1, keepdims=True) + 1e-30)
            grad = W @ Xkeep
            dot = np.sum(grad * C, axis=1, keepdims=True)
            gt = grad - dot * C
            C = C - lr * gt
            C = _nm(C)
        return C

    def get_holes(Xkeep, K=8, n_sample=30000, n_pre=200):
        S = rng.randn(n_sample, 4)
        S = _nm(S)
        D = S @ Xkeep.T
        worst = D.max(axis=1)
        idx = np.argpartition(worst, n_pre)[:n_pre]
        cand = S[idx]
        cand = refine_holes_vec(cand, Xkeep, iters=60, beta=80.0, lr=0.05)
        Df = cand @ Xkeep.T
        wf = Df.max(axis=1)
        order = np.argsort(wf, kind="stable")
        cand = cand[order]; wf = wf[order]
        # diverse greedy pick
        picked = []
        for i in range(len(cand)):
            if len(picked) >= K:
                break
            c = cand[i]
            ok = True
            for pj in picked:
                if float(c @ cand[pj]) > 0.96:
                    ok = False
                    break
            if ok or len(picked) == 0:
                picked.append(i)
            elif len(cand) - i < (K - len(picked)) * 2:
                picked.append(i)
        # fill if short
        for i in range(len(cand)):
            if len(picked) >= K:
                break
            if i not in picked:
                picked.append(i)
        return cand[np.array(picked[:K])]

    # contact degrees of incumbent for pruning
    G0 = x_best @ x_best.T
    np.fill_diagonal(G0, -2.0)
    m0 = float(np.max(G0))
    deg0 = np.sum(G0 > m0 - 1e-7, axis=1).astype(float)
    stress0 = np.sum(np.exp(np.clip(300.0 * (G0 - m0), -20, 0.0)), axis=1)

    N_PRUNE = 160
    N_DROP = n0_full - N_PRUNE
    BEAM = 6
    K_PER = 8

    beam = []
    beam_vals = []
    # build diverse pruned cores
    for b in range(BEAM):
        if time.time() > SEARCH_END - 20.0:
            break
        jitter = rng.rand(n0_full) * 2.0
        score = deg0 + 0.3 * stress0 / (np.max(stress0) + 1e-12) + jitter
        # also occasionally prune low-contact defect to change topology
        if b == BEAM - 1:
            score = deg0[::-1].copy() + rng.rand(n0_full) * 1.0
        victims = np.argsort(-score, kind="stable")[:N_DROP]
        mask = np.ones(n0_full, dtype=bool)
        mask[victims] = False
        Xp = x_best[mask].copy()
        # small shake then full polish of smaller code (lower achievable max)
        Xp = _nm(Xp + 0.005 * rng.randn(*Xp.shape))
        Xp = refine(Xp, 500, beta0=25.0, beta1=500.0, lr0=0.014, lr1=0.0007)
        v = _mc(Xp)
        beam.append(Xp)
        beam_vals.append(v)
    if len(beam) == 0:
        beam = [x_best.copy()]
        beam_vals = [best_val]
    # order beam by value
    o = np.argsort(np.array(beam_vals))
    beam = [beam[i] for i in o]

    # beam-search reinsertion N_PRUNE+1 .. 168
    curN = N_PRUNE
    targetN = 168
    for N in range(curN + 1, targetN + 1):
        if time.time() > SEARCH_END:
            break
        pool = []
        for bi, Xb in enumerate(beam):
            if time.time() > SEARCH_END:
                break
            try:
                holes = get_holes(Xb, K=K_PER, n_sample=30000, n_pre=200)
            except Exception:
                continue
            for h in holes:
                if time.time() > SEARCH_END:
                    break
                Xn0 = np.vstack([Xb, h.reshape(1, 4)])
                Xn0 = _nm(Xn0)
                Xn = refine(Xn0, 200, beta0=30.0, beta1=600.0, lr0=0.015, lr1=0.0007)
                v = _mc(Xn)
                pool.append((v, Xn))
        if len(pool) == 0:
            break
        pool.sort(key=lambda t: t[0])
        # keep best BEAM
        beam = [p[1] for p in pool[:BEAM]]
        # if N==168 track global best (pre-LP)
        if N == 168:
            for v, Xn in pool[:BEAM]:
                if v < best_val:
                    best_val = v
                    x_best = Xn.copy()

    # final LP polish best few 168 completions + incumbent
    cands = []
    for Xb in beam:
        if Xb.shape[0] == 168:
            cands.append(Xb)
    cands.append(xb.copy())
    # dedup by value sort
    vals = []
    for c in cands:
        try:
            vals.append(_mc(_nm(c)))
        except Exception:
            vals.append(2.0)
    order = np.argsort(np.array(vals))
    cands_sorted = [cands[i] for i in order[:4]]
    # LP polish in order, time-boxed
    final_best = x_best.copy()
    final_val = _mc(_nm(final_best))
    # ensure incumbent included
    if _mc(_nm(xb)) < final_val:
        final_best, final_val = _nm(xb.copy()), _mc(_nm(xb))
    for c in cands_sorted:
        if time.time() > HARD_END - 2.0:
            break
        try:
            dl = min(time.time() + 6.0, HARD_END - 1.0)
            if dl <= time.time():
                break
            xs, vs = _lp_polish_deadline(_nm(c.copy()), dl)
            # vs is precomputed inside; recompute exact
            vs = _mc(xs)
            if vs < final_val:
                final_val = vs
                final_best = xs.copy()
            if vs < best_val:
                best_val = vs
                x_best = xs.copy()
        except Exception:
            continue
    # last squeeze on absolute best
    try:
        if time.time() < HARD_END - 1.0:
            xs, vs = _lp_polish_deadline(final_best, HARD_END - 0.8)
            vs = _mc(xs)
            if vs < final_val:
                final_best = xs
                final_val = vs
    except Exception:
        pass
    final_best = _nm(final_best)
    return final_best
