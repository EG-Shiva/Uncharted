import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def lp_polish(x, seconds, radius_init=1e-3, cap=1500):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = max_cos(x)
    radius = float(radius_init)
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        if time.time() - start > seconds - 0.05:
            break
        c = (x @ x.T)[iu]
        m = float(c.max())
        thresh = m - max(20.0 * radius, 1e-9)
        sel = c > thresh
        if int(sel.sum()) > cap:
            idx = np.argpartition(c, -cap)[-cap:]
            mask = np.zeros_like(sel, dtype=bool)
            mask[idx] = True
            sel = mask
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        if p == 0:
            break
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c[sel], A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        v = max_cos(y)
        if v < best:
            x, best, radius = y, v, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def gd_refine(X, steps, T0, T1, lr0, lr1, mu=0.85, frozen=None):
    n, d = X.shape
    V = np.zeros_like(X)
    lT0, lT1 = np.log(T0), np.log(T1)
    ll0, ll1 = np.log(lr0), np.log(lr1)
    mask = np.ones(n, dtype=bool) if frozen is None else ~frozen
    for s in range(steps):
        f = s / max(1, steps - 1)
        T = float(np.exp(lT0 + (lT1 - lT0) * f))
        lr = float(np.exp(ll0 + (ll1 - ll0) * f))
        G = X @ X.T
        np.fill_diagonal(G, -10.0)
        mx = G.max()
        W = np.exp((G - mx) / T)
        np.fill_diagonal(W, 0.0)
        sw = W.sum()
        if not np.isfinite(sw) or sw <= 0:
            break
        W /= sw
        Grad = W @ X
        dot = np.sum(Grad * X, axis=1, keepdims=True)
        Gt = Grad - dot * X
        if frozen is not None:
            Gt[~mask] = 0.0
            V[~mask] = 0.0
        V = mu * V + Gt
        X = X - lr * V
        X /= np.linalg.norm(X, axis=1, keepdims=True)
    return X

def greedy_hole_init(X, idx_list, rng, n_cand=2000):
    # sequentially place each idx to minimize max dot to rest
    for idx in idx_list:
        others = np.delete(X, idx, axis=0)
        C = rng.standard_normal((n_cand, 13))
        C /= np.linalg.norm(C, axis=1, keepdims=True)
        dots = C @ others.T
        md = dots.max(axis=1)
        b = int(np.argmin(md))
        X[idx] = C[b]
    return X

def solve():
    t0 = time.time()
    BUDGET = 172.0
    EXPL = 138.0
    rng = np.random.default_rng(0)
    try:
        xb = np.load("best.npy").astype(float)
    except Exception:
        try:
            xb = np.load("record.npy").astype(float)
        except Exception:
            xb = rng.standard_normal((95, 13))
    xb = xb / np.linalg.norm(xb, axis=1, keepdims=True)
    try:
        xr = np.load("record.npy").astype(float)
        xr = xr / np.linalg.norm(xr, axis=1, keepdims=True)
    except Exception:
        xr = xb
    bv = max_cos(xb); rv = max_cos(xr)
    if rv < bv:
        best, best_val = xr.copy(), rv
    else:
        best, best_val = xb.copy(), bv
    n, d = best.shape
    cur = best.copy(); cur_val = best_val
    iu = np.triu_indices(n, 1)
    it = 0
    lp_count = 0
    while time.time() - t0 < EXPL:
        it += 1
        remain = EXPL - (time.time() - t0)
        if remain < 2.0:
            break
        u = rng.random()
        Xt = cur.copy()
        if u < 0.50:
            # patch ejection around overloaded / random seed
            G = Xt @ Xt.T
            np.fill_diagonal(G, -2.0)
            mc = G.max()
            cc = (G > mc - 0.01).sum(axis=1)
            if rng.random() < 0.7:
                order = np.argsort(-cc)
                # pick among top 12 randomly
                seed = int(order[rng.integers(0, min(12, n))])
            else:
                seed = int(rng.integers(0, n))
            k = int(rng.integers(4, 10))  # 4..9
            if rng.random() < 0.15:
                k = int(rng.integers(10, 15))
            sims = Xt @ Xt[seed]
            sims[seed] = 2.0
            # nearest = largest dot excluding seed; take k-1
            nn = np.argpartition(-sims, k-1)[:k-1]
            idx_list = [seed] + nn.tolist()
            # randomize them first then greedy hole
            R = rng.standard_normal((len(idx_list), d))
            R /= np.linalg.norm(R, axis=1, keepdims=True)
            for ii, jj in enumerate(idx_list):
                Xt[jj] = R[ii]
            Xt = greedy_hole_init(Xt, idx_list, rng, n_cand=1500)
            # joint GD: first move only patch, then all
            frozen = np.zeros(n, dtype=bool)
            frozen[:] = True
            frozen[idx_list] = False  # True means frozen? we use frozen mask inverted inside; build accordingly
            # gd_refine expects frozen array where True=frozen? we passed frozen=None logic: mask=~frozen. So pass inv:
            inv = np.ones(n, dtype=bool)  # True = frozen
            inv[idx_list] = False
            # Actually gd_refine frozen param: frozen is bool array where True means frozen? code: mask=~frozen if not None. So pass inv.
            Xt = gd_refine(Xt, 250, 0.008, 0.002, 0.25, 0.05, frozen=inv)
            Xt = gd_refine(Xt, 500, 0.006, 0.0005, 0.15, 0.02)
        elif u < 0.80:
            sigma = float(rng.choice([0.01, 0.02, 0.04, 0.08, 0.15]))
            if rng.random() < 0.3:
                ksub = int(rng.integers(3, 12))
                idx = rng.choice(n, size=ksub, replace=False)
                Xt[idx] += sigma * 2.0 * rng.standard_normal((ksub, d))
            else:
                Xt += sigma * rng.standard_normal((n, d))
            Xt /= np.linalg.norm(Xt, axis=1, keepdims=True)
            Xt = gd_refine(Xt, 600, 0.010, 0.0006, 0.12, 0.015)
        else:
            # rattler / low-contact reinsertion + global relax
            G = Xt @ Xt.T
            np.fill_diagonal(G, -2.0)
            mc = G.max()
            cc = (G > mc - 0.01).sum(axis=1)
            k = int(rng.integers(1, 4))
            # pick lowest contact points
            order = np.argsort(cc)
            idx_list = order[:k].tolist()
            if rng.random() < 0.4:
                # add a random one
                idx_list.append(int(rng.integers(0, n)))
                idx_list = list(dict.fromkeys(idx_list))
            R = rng.standard_normal((len(idx_list), d))
            R /= np.linalg.norm(R, axis=1, keepdims=True)
            for ii, jj in enumerate(idx_list):
                Xt[jj] = R[ii]
            Xt = greedy_hole_init(Xt, idx_list, rng, n_cand=2500)
            Xt = gd_refine(Xt, 600, 0.008, 0.0005, 0.15, 0.02)
        gv = max_cos(Xt)
        # screen: only LP if promising
        thresh_lp = best_val + 0.012
        if gv > thresh_lp and not (it % 25 == 0):
            # accept as cur move occasionally even if bad? no
            # small chance to move cur to diversify
            if gv < cur_val and rng.random() < 0.5:
                cur, cur_val = Xt, gv
            continue
        # short LP polish
        if time.time() - t0 > EXPL - 1.0:
            break
        Xt = lp_polish(Xt, min(1.2, EXPL - (time.time() - t0) - 0.2), radius_init=0.004)
        lp_count += 1
        v = max_cos(Xt)
        if v < best_val - 1e-12:
            best, best_val = Xt.copy(), v
            cur, cur_val = Xt.copy(), v
        elif v < cur_val:
            # greedy move of current basin with some tolerance
            cur, cur_val = Xt.copy(), v
        elif rng.random() < 0.05:
            # random restart of cur from best to avoid drift
            cur, cur_val = best.copy(), best_val
        # periodic reset cur to best
        if it % 40 == 0:
            cur, cur_val = best.copy(), best_val
    # final long polish of best
    remain = BUDGET - (time.time() - t0)
    if remain > 5:
        best = lp_polish(best, remain - 1.0, radius_init=0.003, cap=2000)
    best = best / np.linalg.norm(best, axis=1, keepdims=True)
    return best
