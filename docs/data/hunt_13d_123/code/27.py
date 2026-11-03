import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _normalize(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return x / n

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _lp_polish(x, seconds, t_deadline=1e30):
    x = _normalize(x)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        if time.time() > t_deadline:
            break
        c = (x @ x.T)[iu]
        m = float(c.max())
        sel = c > m - max(20.0 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        if p == 0:
            break
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))),
                              shape=(n, nv)).tocsr()
            obj = np.zeros(nv); obj[-1] = 1.0
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
        y = _normalize(y)
        v = _max_cos(y)
        if v < best:
            x, best, radius = y, v, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def _stress_metrics(x, eps=1e-6):
    n, d = x.shape
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    m = float(G.max())
    row_max = G.max(axis=1)
    # neighbours within eps
    neigh = [np.where(G[i] > m - eps)[0] for i in range(n)]
    degs = np.array([len(v) for v in neigh], dtype=float)
    # tight contact for rattler detection (1e-9)
    tight = [np.where(G[i] > m - 1e-9)[0] for i in range(n)]
    deg_tight = np.array([len(v) for v in tight], dtype=float)
    res_opt = np.zeros(n)
    wsum = np.zeros(n)
    for i in range(n):
        nb = neigh[i]
        k = len(nb)
        if k == 0:
            res_opt[i] = 0.0
            continue
        c = G[i, nb]
        T = x[nb] - c[:, None] * x[i][None, :]
        # Gram of tangent vectors
        Gm = T @ T.T
        # simplex-constrained min w^T Gm w via exponentiated gradient
        w = np.full(k, 1.0 / k)
        # scale step by trace
        tr = float(np.trace(Gm)) / max(k, 1) + 1e-12
        eta = 0.8 / max(tr, 1e-12)
        # cap eta
        eta = min(eta, 5.0)
        for _ in range(120):
            grad = 2.0 * (Gm @ w)
            w = w * np.exp(-eta * (grad - grad.mean()))
            s = w.sum()
            if s <= 0 or not np.isfinite(s):
                w = np.full(k, 1.0 / k)
                break
            w /= s
        val = float(w @ (Gm @ w))
        res_opt[i] = np.sqrt(max(val, 0.0))
        wsum[i] = float(k)
    return m, row_max, degs, deg_tight, res_opt, G

def _repack_patch(xb, patch, rng, budget_sec, t_deadline):
    n, d = xb.shape
    patch = np.array(sorted(set(int(v) for v in patch)), dtype=int)
    p = len(patch)
    all_idx = np.arange(n)
    frozen_idx = np.array([i for i in all_idx if i not in set(patch.tolist())], dtype=int)
    F = xb[frozen_idx]
    # frozen-frozen max (constant)
    Gff = F @ F.T
    np.fill_diagonal(Gff, -2.0)
    m_frozen = float(Gff.max()) if len(F) > 1 else -2.0
    # greedy hole re-init for patch points
    pool = _normalize(rng.randn(4000, d))
    V = np.empty((p, d))
    for k in range(p):
        d1 = pool @ F.T
        m1 = d1.max(axis=1)
        if k > 0:
            d2 = pool @ V[:k].T
            m2 = d2.max(axis=1)
            mtot = np.maximum(m1, m2)
        else:
            mtot = m1
        bi = int(np.argmin(mtot))
        V[k] = pool[bi]
    V = _normalize(V)
    # softmax anneal
    vel = np.zeros_like(V)
    bestV = V.copy()
    # current patch max
    def patch_max(Vc):
        a = Vc @ Vc.T
        np.fill_diagonal(a, -2.0)
        b = Vc @ F.T
        return float(max(float(a.max()), float(b.max())))
    best_pm = patch_max(V)
    start = time.time()
    # schedules
    beta0, beta1 = 40.0, 700.0
    lr0, lr1 = 0.5, 0.02
    mu = 0.85
    it = 0
    while True:
        el = time.time() - start
        if el >= budget_sec - 0.15:
            break
        if time.time() > t_deadline:
            break
        frac = el / max(budget_sec, 1e-9)
        frac = min(max(frac, 0.0), 1.0)
        beta = beta0 * (beta1 / beta0) ** frac
        lr = lr0 * (lr1 / lr0) ** frac
        # chunk
        for _ in range(200):
            Gpp = V @ V.T
            np.fill_diagonal(Gpp, -10.0)
            Gpf = V @ F.T
            cmax = float(max(float(Gpp.max()), float(Gpf.max())))
            Wpp = np.exp(beta * (Gpp - cmax))
            # zero diag approx
            # Gpp diag was -10 so exp tiny; force 0
            # set diag 0 via flat
            np.fill_diagonal(Wpp, 0.0)
            Wpf = np.exp(beta * (Gpf - cmax))
            S = float(Wpp.sum() / 2.0 + Wpf.sum()) + 1e-30
            Wppn = Wpp / S
            Wpfn = Wpf / S
            grad = Wppn @ V + Wpfn @ F
            dot = np.sum(grad * V, axis=1, keepdims=True)
            gt = grad - dot * V
            vel = mu * vel + gt
            V = V - lr * vel
            V = _normalize(V)
            it += 1
        pm = patch_max(V)
        if pm < best_pm:
            best_pm = pm
            bestV = V.copy()
        # early exit if already well below frozen and no progress needed? continue to refine
    xc = xb.copy()
    xc[patch] = bestV
    xc = _normalize(xc)
    return xc

def solve():
    t0 = time.time()
    DEAD = t0 + 172.0
    rng = np.random.RandomState(0)
    n, d = 123, 13
    try:
        xb = np.load("best.npy")
    except Exception:
        try:
            xb = np.load("record.npy")
        except Exception:
            xb = rng.randn(n, d)
    xb = np.array(xb, dtype=float)
    if xb.shape != (n, d):
        tmp = rng.randn(n, d)
        m = min(xb.size, n * d)
        tmp.ravel()[:m] = xb.ravel()[:m]
        xb = tmp
    xb = _normalize(xb)
    best_val = _max_cos(xb)
    x_best = xb.copy()
    # stress analysis
    try:
        m, row_max, degs, deg_tight, res_opt, G = _stress_metrics(x_best, eps=1e-6)
    except Exception:
        return x_best
    rattler = int(np.argmin(row_max))
    # orderings
    order_deg = np.argsort(degs)  # low stress first
    order_over = np.argsort(-degs)
    order_res = np.argsort(-res_opt)  # frustrated first
    # nearest to rattler
    near_r = np.argsort(-G[rattler])
    near_r = [int(v) for v in near_r if int(v) != rattler]
    patches = []
    # patch 0: rattler + nearest + lowest-stress
    p0 = [rattler] + near_r[:7]
    for v in order_deg:
        v = int(v)
        if v not in p0:
            p0.append(v)
        if len(p0) >= 14:
            break
    patches.append(p0[:14])
    # patch 1: overload hub neighbourhood + rattler
    hub = int(order_over[0])
    if hub == rattler:
        hub = int(order_over[1])
    Gr = G[hub]
    near_h = [int(v) for v in np.argsort(-Gr) if int(v) != hub][:7]
    p1 = [hub] + near_h
    if rattler not in p1:
        p1.append(rattler)
    for v in order_deg:
        v = int(v)
        if v not in p1:
            p1.append(v)
        if len(p1) >= 14:
            break
    patches.append(p1[:14])
    # patch 2: most frustrated + neighbours
    fr = int(order_res[0])
    near_f = [int(v) for v in np.argsort(-G[fr]) if int(v) != fr][:6]
    p2 = [fr] + near_f
    if rattler not in p2:
        p2.append(rattler)
    if hub not in p2:
        p2.append(hub)
    for v in order_deg:
        v = int(v)
        if v not in p2:
            p2.append(v)
        if len(p2) >= 15:
            break
    patches.append(p2[:15])
    # ensure 10-18 size and distinct
    uniq = []
    seen = set()
    for p in patches:
        p = sorted(set(p))
        # pad/truncate to 10-18
        if len(p) < 10:
            for v in order_deg:
                v = int(v)
                if v not in p:
                    p.append(v)
                if len(p) >= 12:
                    break
        p = p[:18]
        t = tuple(sorted(p))
        if t not in seen:
            seen.add(t)
            uniq.append(p)
        if len(uniq) >= 3:
            break
    patches = uniq
    for pi, patch in enumerate(patches):
        if time.time() > DEAD - 7:
            break
        remain = DEAD - time.time()
        if remain < 8:
            break
        # strictly capped 15s repack, 5s polish
        repack_budget = min(15.0, remain - 5.5)
        if repack_budget < 2:
            break
        try:
            xc = _repack_patch(x_best, patch, rng, repack_budget, DEAD - 5.0)
        except Exception:
            continue
        if time.time() > DEAD - 6:
            # no time for polish, check direct improvement
            try:
                v = _max_cos(xc)
            except Exception:
                continue
            if v < best_val:
                best_val = v
                x_best = xc
            continue
        polish_budget = min(5.0, DEAD - time.time() - 0.5)
        if polish_budget < 0.5:
            try:
                v = _max_cos(xc)
            except Exception:
                continue
            if v < best_val:
                best_val = v
                x_best = xc
            continue
        try:
            xp = _lp_polish(xc, polish_budget, DEAD)
        except Exception:
            continue
        try:
            v = _max_cos(xp)
        except Exception:
            continue
        if v < best_val:
            best_val = v
            x_best = xp.copy()
            # refresh stress for remaining patches? keep original plan but update G for next patch distinctness
            # quick update of G not needed
        # time checked after every stage (loop head checks)
    x_best = _normalize(x_best)
    return x_best
