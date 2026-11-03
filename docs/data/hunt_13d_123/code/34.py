import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _normalize(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return x / n

def _max_cos(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _proj_simplex(v):
    n = v.shape[0]
    if n == 0:
        return v
    u = np.sort(v)[::-1]
    css = np.cumsum(u) - 1.0
    ind = np.arange(1, n + 1)
    cond = u - css / ind > 0
    if not np.any(cond):
        return np.full(n, 1.0 / n)
    rho = ind[cond][-1]
    theta = css[cond][-1] / float(rho)
    w = v - theta
    w[w < 0] = 0
    s = w.sum()
    if s <= 0:
        return np.full(n, 1.0 / n)
    return w / s

def _stress_info(x, m, eps=1e-6):
    n, d = x.shape
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    row_max = G.max(axis=1)
    # adjacency within eps
    deg = np.zeros(n, dtype=int)
    neigh = []
    for i in range(n):
        nb = np.where(G[i] > m - eps)[0]
        nb = nb[nb != i]
        neigh.append(nb)
        deg[i] = len(nb)
    res_opt = np.zeros(n)
    res_uni = np.zeros(n)
    for i in range(n):
        k = deg[i]
        if k == 0:
            res_opt[i] = 1e9
            res_uni[i] = 1e9
            continue
        nb = neigh[i]
        c = G[i, nb]
        A = x[nb] - c[:, None] * x[i][None, :]
        mu = A.mean(axis=0)
        res_uni[i] = float(np.linalg.norm(mu))
        # simplex QP: min w^T Q w
        Q = A @ A.T
        w = np.full(k, 1.0 / k)
        # projected gradient, few iters
        step = 0.5 / (np.abs(Q).sum() + 1.0)
        for _ in range(120):
            g = 2.0 * (Q @ w)
            w = _proj_simplex(w - step * g)
        r = w @ A
        res_opt[i] = float(np.linalg.norm(r))
    return G, row_max, deg, neigh, res_opt, res_uni

def _holes_vs_frozen(Yf, p, rng):
    d = Yf.shape[1]
    S = rng.randn(20000, d)
    S = S / np.linalg.norm(S, axis=1, keepdims=True)
    dots = S @ Yf.T
    worst = dots.max(axis=1)
    order = np.argsort(worst)
    chosen = []
    for idx in order:
        cand = S[idx]
        ok = True
        for cv in chosen:
            if float(cand @ cv) > 0.90:
                ok = False
                break
        if ok:
            chosen.append(cand)
            if len(chosen) >= p:
                break
    if len(chosen) < p:
        for idx in order:
            cand = S[idx]
            chosen.append(cand)
            if len(chosen) >= p:
                break
    return np.array(chosen)

def _repack_patch(x0, patch, budget, rng, deadline):
    n, d = x0.shape
    p = len(patch)
    patch = np.array(patch, dtype=int)
    mask = np.ones(n, dtype=bool)
    mask[patch] = False
    Yf = x0[mask]
    V0 = _holes_vs_frozen(Yf, p, rng)
    V = _normalize(V0 + 1e-3 * rng.randn(p, d))
    X = x0.copy()
    X[patch] = V
    vel = np.zeros_like(V)
    start = time.time()
    # schedule
    STEPS = 2200
    beta0, beta1 = 30.0, 550.0
    lr0, lr1 = 0.02, 0.002
    mu = 0.8
    for s in range(STEPS):
        if time.time() - start > budget:
            break
        if time.time() > deadline:
            break
        f = s / max(STEPS - 1, 1)
        beta = beta0 * (beta1 / beta0) ** f
        lr = lr0 * (lr1 / lr0) ** f
        Xp = X  # full
        G = Xp @ Xp.T
        np.fill_diagonal(G, -10.0)
        # focus on pairs involving patch for gmax
        sub = G[np.ix_(patch, np.arange(n))]
        # exclude self columns: set those to -10
        # sub columns corresponding to patch rows themselves include internal; keep but diag already -10
        gmax = float(sub.max())
        W = np.exp(beta * (G - gmax))
        np.fill_diagonal(W, 0.0)
        # gradient for patch rows only
        Wp = W[patch, :]  # p x n
        grad = Wp @ Xp  # p x d
        Vcur = X[patch]
        dot = np.sum(grad * Vcur, axis=1, keepdims=True)
        row = Wp.sum(axis=1, keepdims=True) + 1e-12
        gt = (grad - dot * Vcur) / row
        # clip
        gn = np.linalg.norm(gt, axis=1, keepdims=True) + 1e-12
        gt = gt / np.maximum(gn, 1.0)
        vel = mu * vel + gt
        Vn = Vcur - lr * vel
        Vn = Vn / (np.linalg.norm(Vn, axis=1, keepdims=True) + 1e-16)
        X[patch] = Vn
    return _normalize(X)

def _lp_polish(x, seconds, deadline=None):
    x = _normalize(x)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    it = 0
    while time.time() - start < seconds:
        if deadline is not None and time.time() > deadline:
            break
        if radius < 1e-14:
            break
        it += 1
        if it > 25:
            break
        G = x @ x.T
        iuc = G[iu]
        m = float(iuc.max())
        sel = iuc > m - max(20.0 * radius, 1e-9)
        I = iu[0][sel]; J = iu[1][sel]
        c = iuc[sel]
        pp = len(I)
        if pp == 0:
            break
        nv = n * d + 1
        rows = np.repeat(np.arange(pp), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((pp, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((pp, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(pp, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))),
                              shape=(n, nv)).tocsr()
        except Exception:
            break
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c, A_eq=A_eq,
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
            x, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def solve():
    t0 = time.time()
    T_END = t0 + 168.0
    rng = np.random.RandomState(7)
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

    G0 = x_best @ x_best.T
    np.fill_diagonal(G0, -2.0)
    m0 = float(G0.max())
    _, row_max, deg, neigh, res_opt, res_uni = _stress_info(x_best, m0, eps=1e-6)

    rattler = int(np.argmin(row_max))
    # candidates sorted by degree / residual
    order_deg_asc = np.argsort(deg + row_max * 1e-6)
    order_res_desc = np.argsort(-res_opt)
    order_deg_desc = np.argsort(-deg)

    patches = []
    # Patch 0: rattler + nearest + lowest-stress hub neighbourhood
    low = None
    for c in order_deg_asc:
        if c != rattler:
            low = int(c)
            break
    # nearest to rattler by angle
    nn_r = np.argsort(-G0[rattler])[:7]
    nn_l = np.argsort(-G0[low])[:7]
    p0 = sorted(set([rattler] + nn_r.tolist() + [low] + nn_l[:5].tolist()))
    if len(p0) > 18:
        p0 = p0[:18]
    if len(p0) < 10:
        extra = [c for c in np.argsort(-G0[low]) if c not in p0][:10 - len(p0)]
        p0 = sorted(set(p0 + extra))
    patches.append(p0)
    # Patch 1: highest residual (frustrated) + neighbours
    c1 = int(order_res_desc[0])
    if c1 == rattler:
        c1 = int(order_res_desc[1])
    nn_c1 = np.argsort(-G0[c1])[:11]
    p1 = sorted(set([c1] + nn_c1.tolist() + [rattler]))
    if len(p1) > 18:
        p1 = p1[:18]
    if len(p1) < 10:
        extra = [c for c in np.argsort(-G0[c1]) if c not in p1][:10 - len(p1)]
        p1 = sorted(set(p1 + extra))
    patches.append(p1)
    # Patch 2: most overloaded hub + second lowest stress
    c2 = int(order_deg_desc[0])
    low2 = None
    for c in order_deg_asc:
        if c != rattler and c != low:
            low2 = int(c)
            break
    nn_c2 = np.argsort(-G0[c2])[:7]
    nn_l2 = np.argsort(-G0[low2])[:6]
    p2 = sorted(set([c2] + nn_c2.tolist() + [low2] + nn_l2.tolist()))
    if len(p2) > 18:
        p2 = p2[:18]
    if len(p2) < 10:
        extra = [c for c in np.argsort(-G0[c2]) if c not in p2][:10 - len(p2)]
        p2 = sorted(set(p2 + extra))
    patches.append(p2)

    for pi, patch in enumerate(patches[:3]):
        if time.time() > T_END - 8.0:
            break
        # strictly capped 15 s repack
        remain = T_END - time.time()
        if remain < 8.0:
            break
        repack_budget = min(15.0, remain - 7.0)
        if repack_budget <= 0.5:
            break
        x_rep = _repack_patch(x_best, patch, repack_budget, rng, T_END - 6.0)
        if time.time() > T_END - 6.5:
            # no time for polish; check raw
            v_raw = _max_cos(_normalize(x_rep))
            if v_raw < best_val:
                x_best, best_val = _normalize(x_rep), v_raw
            break
        # LP-polish whole config at most 5 s
        remain2 = T_END - time.time()
        pol = min(5.0, remain2 - 1.0)
        if pol < 0.5:
            v_raw = _max_cos(_normalize(x_rep))
            if v_raw < best_val:
                x_best, best_val = _normalize(x_rep), v_raw
            break
        x_pol = _lp_polish(x_rep, pol, deadline=T_END - 0.5)
        v_pol = _max_cos(x_pol)
        if v_pol < best_val:
            x_best, best_val = x_pol, v_pol
            # refresh stress for remaining patches (cheap, time-checked)
            if time.time() < T_END - 10.0 and pi < 2:
                G0 = x_best @ x_best.T
                np.fill_diagonal(G0, -2.0)
                m0 = float(G0.max())
        # else discard (accept only if improves)
        if time.time() > T_END - 1.0:
            break

    x_best = _normalize(x_best)
    return x_best
