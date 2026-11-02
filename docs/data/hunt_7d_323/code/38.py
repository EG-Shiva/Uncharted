import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _lp_polish_cap(x, deadline, max_iters=25, init_radius=1e-3):
    # trust-region LP with hard deadline and per-call time_limit
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = float(init_radius)
    iuI, iuJ = np.triu_indices(n, 1)
    it = 0
    while time.time() < deadline and radius > 1e-14 and it < max_iters:
        it += 1
        G = x @ x.T
        c = G[iuI, iuJ]
        m = float(c.max())
        thresh = m - max(20.0 * radius, 1e-9)
        sel = c > thresh
        cnt = int(sel.sum())
        if cnt == 0:
            break
        if cnt > 3000:
            idx = np.where(sel)[0]
            cidx = c[idx]
            part = np.argpartition(cidx, -3000)[-3000:]
            sel2 = np.zeros_like(sel, dtype=bool)
            sel2[idx[part]] = True
            sel = sel2
            cnt = 3000
        I = iuI[sel]; J = iuJ[sel]; csel = c[sel]
        p = cnt
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.empty(p * (2 * d + 1), dtype=np.int64)
        vals = np.empty(p * (2 * d + 1), dtype=np.float64)
        Cb = cols.reshape(p, 2 * d + 1)
        Vb = vals.reshape(p, 2 * d + 1)
        ar = np.arange(d)
        Cb[:, :d] = I[:, None] * d + ar
        Cb[:, d:2 * d] = J[:, None] * d + ar
        Cb[:, 2 * d] = n * d
        Vb[:, :d] = x[J]
        Vb[:, d:2 * d] = x[I]
        Vb[:, 2 * d] = -1.0
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv); obj[-1] = 1.0
        remain = deadline - time.time() - 0.15
        if remain < 0.3:
            break
        tl = min(4.0, remain)
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                          method="highs", options={"time_limit": tl})
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        yn = np.linalg.norm(y, axis=1, keepdims=True)
        yn[yn == 0] = 1.0
        y = y / yn
        v = _max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.02)
        else:
            radius *= 0.5
    return x, best

def _anneal(x, rng, steps, T0, T1, s0, s1):
    n, d = x.shape
    for s in range(steps):
        frac = s / max(steps - 1, 1)
        T = T0 * (T1 / T0) ** frac
        step = s0 * (s1 / s0) ** frac
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        # soft-max weights
        W = np.exp((G - m) / max(T, 1e-12))
        np.fill_diagonal(W, 0.0)
        # sparsify tiny weights for stability/speed
        # threshold at 1e-4
        # keep all (323x323 exp is cheap) but zero tiny
        W[W < 1e-4] = 0.0
        S = (W * G).sum(axis=1)
        F = S[:, None] * x - W @ x
        R = W.sum(axis=1, keepdims=True)
        x = x + (step / (R + 1.0)) * F
        xn = np.linalg.norm(x, axis=1, keepdims=True)
        xn[xn == 0] = 1.0
        x = x / xn
    return x

def solve():
    t0 = time.time()
    DEAD = t0 + 160.0
    rng = np.random.default_rng(12345)
    n, d = 323, 7
    best_x = None
    best_v = np.inf
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn).astype(np.float64)
            if a.shape == (n, d):
                a = a / np.linalg.norm(a, axis=1, keepdims=True)
                v = _max_cos(a)
                if v < best_v:
                    best_v = v
                    best_x = a.copy()
        except Exception:
            pass
    if best_x is None:
        best_x = rng.standard_normal(size=(n, d))
        best_x /= np.linalg.norm(best_x, axis=1, keepdims=True)
        best_v = _max_cos(best_x)
    # reserve: keep best incumbent safe
    incumbent_x = best_x.copy()
    incumbent_v = best_v
    trial = 0
    # precompute contact info updated each trial
    while time.time() < DEAD - 7.0:
        trial += 1
        if time.time() >= DEAD - 7.0:
            break
        x = incumbent_x.copy()
        # --- analyse contacts ---
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        cnt = np.sum(G > m - 1e-9, axis=1)
        worst = G.max(axis=1)
        order_cnt = np.argsort(cnt)       # low first (rattlers)
        order_worst = np.argsort(-worst)  # high worst first (hubs/bad)
        # choose surgery size 6..14 varying by trial
        ksize = 6 + (trial % 9)  # 6..14
        to_rep = set()
        # always include 2 lowest contact points
        to_rep.update(order_cnt[:2].tolist())
        # include up to 3 worst points
        for idx in order_worst[:4]:
            if len(to_rep) < ksize - 2:
                to_rep.add(int(idx))
        # include up to 2 highest contact hubs
        order_hub = np.argsort(-cnt)
        for idx in order_hub[:3]:
            if len(to_rep) < ksize:
                # avoid duplicating too much; add with probability
                if rng.random() < 0.7:
                    to_rep.add(int(idx))
        # fill rest randomly
        while len(to_rep) < ksize:
            to_rep.add(int(rng.integers(0, n)))
        to_rep = list(to_rep)
        # random kick scale for reinsertion candidates
        n_cand = 2500
        # greedy max-min reinsertion sequential
        # mask for fast dots: use full matrix but ignore self by recomputing
        for k in to_rep:
            if time.time() >= DEAD - 6.0:
                break
            cand = rng.standard_normal(size=(n_cand, d))
            cand /= np.linalg.norm(cand, axis=1, keepdims=True)
            # dots to all other points (use current x without k)
            others = np.delete(x, k, axis=0)  # (n-1,d)
            dots = cand @ others.T
            w = dots.max(axis=1)
            bi = int(np.argmin(w))
            x[k] = cand[bi]
        # small random tangential jitter on replaced points to break symmetry
        jitter = 0.02 + 0.02 * rng.random()
        for k in to_rep:
            v = rng.standard_normal(size=(d,)) * jitter
            # make tangential
            v = v - np.dot(v, x[k]) * x[k]
            x[k] = x[k] + v
            x[k] /= np.linalg.norm(x[k])
        # --- fast anneal ---
        # two phases: coarse then fine, total ~400-600 steps (fast numpy)
        steps1 = 250
        steps2 = 200
        # temperature schedule depends on trial parity for diversity
        if trial % 3 == 0:
            T0, T1 = 0.015, 0.0008
            s0, s1 = 0.08, 0.008
        elif trial % 3 == 1:
            T0, T1 = 0.008, 0.0005
            s0, s1 = 0.05, 0.005
        else:
            T0, T1 = 0.025, 0.002
            s0, s1 = 0.10, 0.01
        x = _anneal(x, rng, steps1, T0, T1, s0, s1)
        if time.time() >= DEAD - 6.0:
            # evaluate anyway
            v = _max_cos(x)
            if v < incumbent_v:
                incumbent_x, incumbent_v = x, v
            continue
        x = _anneal(x, rng, steps2, T1, T1 * 0.3, s1, s1 * 0.3)
        v = _max_cos(x)
        # if promising, LP snap with small budget
        # threshold: within 0.004 of incumbent or better
        if v < incumbent_v + 0.004:
            budget = min(5.0, DEAD - time.time() - 1.0)
            if budget >= 1.0:
                # short polish
                xp, vp = _lp_polish_cap(x, time.time() + budget, max_iters=20, init_radius=5e-3)
                if vp < v:
                    x, v = xp, vp
        if v < incumbent_v - 1e-12:
            incumbent_x, incumbent_v = x.copy(), v
            # immediate small extra polish if time allows and improvement
            if time.time() < DEAD - 10.0:
                budget2 = min(4.0, DEAD - time.time() - 1.0)
                if budget2 >= 1.0:
                    xp2, vp2 = _lp_polish_cap(incumbent_x, time.time() + budget2, max_iters=15, init_radius=1e-3)
                    if vp2 < incumbent_v:
                        incumbent_x, incumbent_v = xp2, vp2
        # occasional keep of near-miss with different basin? ignore
    # final short polish of incumbent if time remains
    if time.time() < DEAD - 2.0:
        budget = min(6.0, DEAD - time.time())
        try:
            xp, vp = _lp_polish_cap(incumbent_x, time.time() + budget, max_iters=25, init_radius=1e-3)
            if vp < incumbent_v:
                incumbent_x, incumbent_v = xp, vp
        except Exception:
            pass
    incumbent_x = incumbent_x / np.linalg.norm(incumbent_x, axis=1, keepdims=True)
    return incumbent_x
