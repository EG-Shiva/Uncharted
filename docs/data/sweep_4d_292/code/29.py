import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _norm(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return x / n

def _max_cos(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _anneal(x, steps, beta0, beta1, lr0, lr1, mu):
    n, d = x.shape
    x = _norm(x)
    vel = np.zeros_like(x)
    lb = np.log(beta1 / beta0) if beta1 > beta0 else 0.0
    ll = np.log(lr1 / lr0) if lr1 != lr0 else 0.0
    for s in range(steps):
        frac = s / max(1, steps - 1)
        beta = beta0 * np.exp(lb * frac)
        lr = lr0 * np.exp(ll * frac)
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        gmax = G.max()
        W = np.exp(beta * (G - gmax))
        np.fill_diagonal(W, 0.0)
        r = W.sum(axis=1, keepdims=True)
        np.maximum(r, 1e-30, out=r)
        A = (W @ x) / r
        adot = np.sum(A * x, axis=1, keepdims=True)
        At = A - adot * x
        vel = mu * vel + (1.0 - mu) * At
        bias = 1.0 - mu ** (s + 1)
        vh = vel / max(bias, 1e-12)
        x = x - lr * vh
        x = _norm(x)
    return x

def _find_holes(Xkeep, k, rng, n_cand=16000, sep=0.86):
    cand = rng.standard_normal((n_cand, 4))
    cand = cand / np.linalg.norm(cand, axis=1, keepdims=True)
    # chunked max dot
    best = np.empty(n_cand)
    cs = 4000
    for a in range(0, n_cand, cs):
        b = min(a + cs, n_cand)
        D = cand[a:b] @ Xkeep.T
        best[a:b] = D.max(axis=1)
    order = np.argsort(best)
    sel = []
    for idx in order:
        c = cand[idx]
        ok = True
        for s in sel:
            if float(s @ c) > sep:
                ok = False
                break
        if ok:
            sel.append(c)
            if len(sel) >= k:
                break
    while len(sel) < k:
        sel.append(cand[order[len(sel)]])
    return np.array(sel)

def _lp_polish(x, seconds, r0=8e-4):
    x = _norm(np.asarray(x, dtype=float).copy())
    n, d = x.shape
    best = _max_cos(x)
    xb = x.copy()
    if seconds <= 0:
        return xb
    start = time.time()
    radius = r0
    iu0 = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        G = x @ x.T
        c = G[iu0]
        m = c.max()
        thresh = m - max(20.0 * radius, 1e-9)
        sel = c > thresh
        if not np.any(sel):
            break
        I = iu0[0][sel]
        J = iu0[1][sel]
        p = len(I)
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        ci = I[:, None] * d + np.arange(d)
        cj = J[:, None] * d + np.arange(d)
        cols = np.concatenate([ci, cj, np.full((p, 1), n * d)], axis=1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], axis=1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv)
            obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-c[sel], A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y = _norm(y)
        v = _max_cos(y)
        if v < best:
            best = v
            xb = y.copy()
            x = y
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return xb

def solve():
    t0 = time.time()
    rng = np.random.default_rng(20261031)
    try:
        x0 = np.load("best.npy")
    except Exception:
        try:
            x0 = np.load("record.npy")
        except Exception:
            z = rng.standard_normal((292, 4))
            x0 = z / np.linalg.norm(z, axis=1, keepdims=True)
    x0 = np.asarray(x0, dtype=float)
    if x0.shape != (292, 4) or not np.all(np.isfinite(x0)):
        z = rng.standard_normal((292, 4))
        x0 = z / np.linalg.norm(z, axis=1, keepdims=True)
    x0 = _norm(x0)
    try:
        r0 = np.load("record.npy")
        r0 = np.asarray(r0, dtype=float)
        if r0.shape == (292, 4) and np.all(np.isfinite(r0)):
            r0 = _norm(r0)
            if _max_cos(r0) < _max_cos(x0):
                x0 = r0
    except Exception:
        pass
    best = x0.copy()
    best_val = _max_cos(best)
    cur = best.copy()
    cur_val = best_val
    n, d = best.shape
    EXPLORE_END = 132.0
    T0, T1 = 0.0022, 2.0e-05
    it = 0
    # precompute nothing
    while time.time() - t0 < EXPLORE_END:
        it += 1
        now = time.time() - t0
        if now > 165:
            break
        frac = now / max(1e-9, EXPLORE_END)
        Temp = T0 * (T1 / T0) ** frac
        # per-point stats of cur (cheap)
        Gc = cur @ cur.T
        np.fill_diagonal(Gc, -2.0)
        m_cur = float(Gc.max())
        pm = Gc.max(axis=1)
        u = rng.random()
        xt = None
        if u < 0.34:
            sigma = float(rng.choice(np.array([0.02, 0.05, 0.10, 0.20, 0.32])))
            xt = cur + sigma * rng.standard_normal((n, d))
            xt = _norm(xt)
            xt = _anneal(xt, steps=550, beta0=18.0, beta1=420.0, lr0=0.040, lr1=0.0028, mu=0.85)
        elif u < 0.60:
            center = int(rng.integers(0, n))
            k = int(rng.integers(6, 19))
            dots = cur @ cur[center]
            S = np.argsort(-dots)[:k]
            xt = cur.copy()
            xt[S] += 0.38 * rng.standard_normal((k, d))
            xt = _norm(xt)
            xt = _anneal(xt, steps=500, beta0=20.0, beta1=450.0, lr0=0.040, lr1=0.0030, mu=0.86)
        elif u < 0.85:
            ns = int(rng.integers(2, 7))
            # worst by per-point max
            order = np.argsort(-pm)
            pool = order[:14]
            src = rng.choice(pool, size=ns, replace=False)
            mask = np.ones(n, dtype=bool)
            mask[src] = False
            Xkeep = cur[mask]
            holes = _find_holes(Xkeep, k=ns, rng=rng, n_cand=16000, sep=0.86)
            xt = cur.copy()
            for s, h in zip(src, holes):
                xt[int(s)] = h + 0.006 * rng.standard_normal(d)
            xt = _norm(xt)
            xt = _anneal(xt, steps=620, beta0=18.0, beta1=480.0, lr0=0.042, lr1=0.0028, mu=0.87)
        else:
            flat = int(np.argmax(Gc))
            wi, wj = divmod(flat, n)
            tgt = wi if pm[wi] >= pm[wj] else wj
            nbrs = np.where(Gc[tgt] > m_cur - 1e-3)[0]
            if len(nbrs) > 14:
                nbrs = nbrs[np.argsort(-Gc[tgt][nbrs])[:14]]
            xt = cur.copy()
            xt[tgt] += 0.30 * rng.standard_normal(d)
            if len(nbrs) > 0:
                xt[nbrs] += 0.08 * rng.standard_normal((len(nbrs), d))
            xt += 0.006 * rng.standard_normal((n, d))
            xt = _norm(xt)
            xt = _anneal(xt, steps=560, beta0=22.0, beta1=420.0, lr0=0.040, lr1=0.0030, mu=0.85)
        # fine squeeze
        xt = _anneal(xt, steps=180, beta0=150.0, beta1=1100.0, lr0=0.010, lr1=0.0012, mu=0.88)
        try:
            vt = _max_cos(xt)
        except Exception:
            continue
        if vt < best_val - 1e-12:
            remain = EXPLORE_END - (time.time() - t0)
            if remain > 1.5 and vt < best_val - 2e-07:
                xp = _lp_polish(xt, seconds=min(1.0, max(0.4, remain * 0.08)), r0=8e-4)
                try:
                    vp = _max_cos(xp)
                except Exception:
                    vp = vt
                if vp < vt:
                    xt, vt = xp, vp
            best = xt.copy()
            best_val = vt
            cur = xt.copy()
            cur_val = vt
        elif vt < cur_val or rng.random() < np.exp((cur_val - vt) / max(Temp, 1e-12)):
            cur = xt.copy()
            cur_val = vt
        # occasional restart to best if wandering too far
        if cur_val > best_val + 0.006 and (it % 10 == 0) and rng.random() < 0.5:
            cur = best.copy() + 0.015 * rng.standard_normal((n, d))
            cur = _norm(cur)
            cur_val = _max_cos(cur)
        if it % 25 == 0 and (time.time() - t0) > EXPLORE_END - 5:
            break
    # intensify around best with small moves
    while time.time() - t0 < 158.0:
        sigma = float(rng.choice(np.array([0.008, 0.018, 0.035])))
        xt = best + sigma * rng.standard_normal((n, d))
        xt = _norm(xt)
        xt = _anneal(xt, steps=500, beta0=30.0, beta1=700.0, lr0=0.030, lr1=0.002, mu=0.87)
        xt = _anneal(xt, steps=150, beta0=250.0, beta1=1400.0, lr0=0.008, lr1=0.001, mu=0.88)
        try:
            vt = _max_cos(xt)
        except Exception:
            break
        if vt < best_val - 1e-12:
            best = xt.copy()
            best_val = vt
        if time.time() - t0 > 158:
            break
    remain = min(18.0, max(0.0, 172.0 - (time.time() - t0) - 0.5))
    if remain > 1.0:
        best = _lp_polish(best, seconds=remain, r0=7e-4)
    best = _norm(best)
    return best
