import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _contacts(x, tol=1e-7):
    n = x.shape[0]
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    m = float(G.max())
    sel = G > m - tol
    per = sel.sum(axis=1)
    iu = np.triu_indices(n, 1)
    c = (x @ x.T)[iu]
    return m, per, G

def _smooth(x, iters, k0, k1, lr0, lr1, mu, rng, noise=0.0):
    n, d = x.shape
    v = np.zeros_like(x)
    for t in range(iters):
        frac = t / max(1, iters - 1)
        k = k0 * (k1 / k0) ** frac if k1 != k0 else k0
        lr = lr0 * (lr1 / lr0) ** frac if lr1 != lr0 else lr0
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        moff = G.max()
        W = np.exp(k * (G - moff))
        np.fill_diagonal(W, 0.0)
        rs = W.sum(axis=1, keepdims=True) + 1e-12
        grad = (W @ x) / rs
        dot = np.sum(grad * x, axis=1, keepdims=True)
        gt = grad - dot * x
        v = mu * v + gt
        denom = (1.0 - mu ** (t + 1)) if mu > 0 else 1.0
        vc = v / denom if denom != 0 else v
        x = x - lr * vc
        if noise > 0 and frac < 0.6:
            amp = noise * (1.0 - frac / 0.6)
            pert = rng.standard_normal(x.shape)
            pd = np.sum(pert * x, axis=1, keepdims=True)
            pert -= pd * x
            x = x + amp * pert
        x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x

def _lp_polish(x, seconds):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    try:
        best = _max_cos(x)
    except Exception:
        return x
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        try:
            Ct = (x @ x.T)[iu]
        except Exception:
            break
        m = float(Ct.max())
        sel = Ct > m - max(20 * radius, 1e-9)
        I = iu[0][sel]; J = iu[1][sel]
        p = len(I)
        if p == 0:
            break
        nv = n * d + 1
        try:
            rows = np.repeat(np.arange(p), 2 * d + 1)
            cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
            vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv); obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-Ct[sel], A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        try:
            y = x + res.x[:-1].reshape(n, d)
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            vv = _max_cos(y)
        except Exception:
            radius *= 0.5
            continue
        if vv < best:
            x, best, radius = y, vv, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def _best_void_insert(Xkeep, rng, ncand=20000):
    m, d = Xkeep.shape
    best_c = None
    best_s = np.inf
    chunk = 5000
    done = 0
    while done < ncand:
        b = min(chunk, ncand - done)
        cand = rng.standard_normal((b, d))
        cand /= np.linalg.norm(cand, axis=1, keepdims=True)
        dots = cand @ Xkeep.T
        mx = dots.max(axis=1)
        j = int(np.argmin(mx))
        if float(mx[j]) < best_s:
            best_s = float(mx[j])
            best_c = cand[j].copy()
        done += b
    return best_c

def solve():
    t0 = time.time()
    rng_master = np.random.default_rng(12345)
    n, d = 153, 4
    base0 = None
    base0_val = np.inf
    for f in ("best.npy", "record.npy"):
        try:
            a = np.load(f)
            a = np.asarray(a, dtype=float)
            if a.shape == (n, d):
                a = a / np.linalg.norm(a, axis=1, keepdims=True)
                v = _max_cos(a)
                if v < base0_val:
                    base0_val = v
                    base0 = a.copy()
        except Exception:
            continue
    if base0 is None:
        r = rng_master.standard_normal((n, d))
        r /= np.linalg.norm(r, axis=1, keepdims=True)
        base0 = r
        base0_val = _max_cos(r)
    fallback = base0.copy()
    incumbent = base0.copy()
    inc_val = base0_val
    pool = [(inc_val, incumbent.copy())]

    SEARCH_END = t0 + 135.0
    FINAL_POLISH = 25.0
    branch_id = 0
    # pre-defined branch templates to cycle
    temps = [
        {"kind": "down", "low": 148, "pick": "max"},
        {"kind": "down", "low": 150, "pick": "randtop"},
        {"kind": "down", "low": 152, "pick": "max"},
        {"kind": "down", "low": 151, "pick": "randtop"},
        {"kind": "up", "high": 155},
        {"kind": "up", "high": 156},
        {"kind": "down", "low": 149, "pick": "max"},
        {"kind": "down", "low": 150, "pick": "max"},
    ]
    try:
        while time.time() < SEARCH_END and branch_id < 40:
            cfg = temps[branch_id % len(temps)]
            seed = 1000 + branch_id * 77
            rng = np.random.default_rng(seed)
            branch_id += 1
            # start from fallback best (published) to force different basin, sometimes from incumbent
            if branch_id % 3 == 0 and len(pool) > 0:
                # use incumbent for some branches
                start_x = incumbent.copy()
            else:
                start_x = fallback.copy()
            # small initial shake to diversify removal order evaluation (does not change topology much)
            # we keep shake tiny so LP after removal decides basin
            try:
                if cfg["kind"] == "down":
                    low = int(cfg["low"])
                    x = start_x.copy()
                    # iterative removal
                    ok = True
                    for step in range(n - low):
                        nn = x.shape[0]
                        if time.time() > SEARCH_END:
                            ok = False
                            break
                        _, per, _ = _contacts(x, tol=1e-7)
                        if cfg["pick"] == "max":
                            idx = int(np.argmax(per))
                        else:
                            # random among top-10 loaded
                            order = np.argsort(-per)
                            k = min(10, nn)
                            idx = int(order[int(rng.integers(k))])
                        mask = np.ones(nn, dtype=bool)
                        mask[idx] = False
                        x = x[mask]
                        # brief LP polish after each removal (0.5s)
                        x = _lp_polish(x, 0.5)
                        # tiny smooth to settle
                        x = _smooth(x, 120, 25.0, 60.0, 0.015, 0.005, 0.7, rng, noise=0.0)
                        x = _lp_polish(x, 0.4)
                    if not ok:
                        continue
                    # rebuild to 153
                    while x.shape[0] < 153:
                        if time.time() > SEARCH_END:
                            break
                        c = _best_void_insert(x, rng, ncand=20000)
                        x = np.vstack([x, c.reshape(1, d)])
                        x = _smooth(x, 250, 20.0, 80.0, 0.02, 0.004, 0.75, rng, noise=0.002)
                        x = _lp_polish(x, 0.6)
                    if x.shape[0] == 153:
                        x = x / np.linalg.norm(x, axis=1, keepdims=True)
                        v = _max_cos(x)
                        pool.append((v, x.copy()))
                        if v < inc_val:
                            inc_val = v
                            incumbent = x.copy()
                else:  # overshoot then prune
                    high = int(cfg["high"])
                    x = start_x.copy()
                    for _a in range(high - 153):
                        if time.time() > SEARCH_END:
                            break
                        c = _best_void_insert(x, rng, ncand=20000)
                        x = np.vstack([x, c.reshape(1, d)])
                        x = _smooth(x, 200, 20.0, 70.0, 0.02, 0.005, 0.75, rng, noise=0.002)
                        x = _lp_polish(x, 0.5)
                    # prune one by one to 153 choosing lowest polished max
                    while x.shape[0] > 153:
                        if time.time() > SEARCH_END:
                            break
                        nn = x.shape[0]
                        # quick score without polish: max after removal
                        G = x @ x.T
                        np.fill_diagonal(G, -2.0)
                        # candidate removals: points with largest row-max are likely best to remove
                        rowmax = G.max(axis=1)
                        order = np.argsort(-rowmax)  # worst first
                        cand_idx = order[:4]  # try top 4
                        best_v = np.inf
                        best_x = None
                        for ci in cand_idx:
                            mask = np.ones(nn, dtype=bool)
                            mask[int(ci)] = False
                            y = x[mask].copy()
                            y = _lp_polish(y, 0.35)
                            v = _max_cos(y)
                            if v < best_v:
                                best_v = v
                                best_x = y
                        if best_x is None:
                            break
                        x = best_x
                        x = _smooth(x, 100, 25.0, 60.0, 0.012, 0.005, 0.7, rng, noise=0.0)
                        x = _lp_polish(x, 0.35)
                    if x.shape[0] == 153:
                        x = x / np.linalg.norm(x, axis=1, keepdims=True)
                        v = _max_cos(x)
                        pool.append((v, x.copy()))
                        if v < inc_val:
                            inc_val = v
                            incumbent = x.copy()
            except Exception:
                continue
    except Exception:
        pass
    # retain every intermediate 153 result, pick best
    try:
        pool_sorted = sorted(pool, key=lambda t: t[0])
        best153 = pool_sorted[0][1].copy()
        best153_val = float(pool_sorted[0][0])
    except Exception:
        best153 = fallback.copy()
        best153_val = _max_cos(best153)
    # final reserve LP polish of best one
    try:
        remain = max(0.0, (t0 + 175.0) - time.time())
        fp = min(FINAL_POLISH + 5.0, max(5.0, remain))
        # if we already beat record, give full polish; otherwise still polish
        polished = _lp_polish(best153, fp)
        pv = _max_cos(polished)
        if pv <= best153_val + 1e-12:
            best153 = polished
            best153_val = pv
        # fallback to loaded record if worse (tolerance)
        if best153_val > base0_val + 2e-9:
            best153 = fallback.copy()
    except Exception:
        best153 = fallback.copy()
    best153 = best153 / np.linalg.norm(best153, axis=1, keepdims=True)
    # strict finite check
    try:
        if not np.all(np.isfinite(best153)):
            best153 = fallback.copy()
    except Exception:
        best153 = fallback.copy()
    return best153
