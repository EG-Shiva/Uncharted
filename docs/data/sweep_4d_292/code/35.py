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
    lb = np.log(beta1 / beta0) if beta1 > beta0 and beta0 > 0 else 0.0
    ll = np.log(lr1 / lr0) if lr1 != lr0 and lr0 > 0 else 0.0
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

def _degrees(x, tol=1e-3):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    m = float(G.max())
    C = G > (m - tol)
    deg = C.sum(axis=1)
    return deg, m, G

def _find_holes(Xkeep, k, rng, n_cand=20000, sep=0.88):
    cand = rng.standard_normal((n_cand, 4))
    cand = cand / np.linalg.norm(cand, axis=1, keepdims=True)
    best = np.empty(n_cand)
    cs = 5000
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
        if len(sel) >= k:
            break
    j = 0
    while len(sel) < k and j < len(order):
        sel.append(cand[order[j]])
        j += 1
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
        if p > 4000:
            # keep tightest subset to bound LP size
            idx = np.argsort(-c[sel])[:4000]
            I = I[idx]; J = J[idx]
            ssel = np.where(sel)[0][idx]
            csub = c[ssel]
            p = len(I)
        else:
            csub = c[sel]
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
            res = linprog(obj, A_ub=A_ub, b_ub=-csub, A_eq=A_eq, b_eq=np.zeros(n),
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
    rng = np.random.default_rng(20261101)
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
    n, d = 292, 4
    EXPLORE_END = 130.0
    T0, T1 = 0.0030, 2.5e-05
    it = 0
    while time.time() - t0 < EXPLORE_END:
        it += 1
        now = time.time() - t0
        if now > 163:
            break
        frac = now / max(1e-9, EXPLORE_END)
        Temp = T0 * (T1 / T0) ** frac
        # degree stats of cur (throttled: every iter is ok, cheap)
        try:
            deg, m_cur, Gc = _degrees(cur, tol=1e-3)
        except Exception:
            deg = np.full(n, 6)
            m_cur = cur_val
            Gc = cur @ cur.T
            np.fill_diagonal(Gc, -2.0)
        pm = Gc.max(axis=1)
        u = rng.random()
        xt = None
        if u < 0.28:
            # global shake + balanced anneal (moderate beta to equalize)
            sigma = float(rng.choice(np.array([0.02, 0.05, 0.11, 0.22, 0.35])))
            xt = cur + sigma * rng.standard_normal((n, d))
            xt = _norm(xt)
            xt = _anneal(xt, steps=480, beta0=14.0, beta1=220.0, lr0=0.045, lr1=0.0035, mu=0.86)
        elif u < 0.55:
            # dense-cluster dispersal: pick overloaded center, blast its neighbourhood
            top = np.argsort(-deg)[:6]
            center = int(top[rng.integers(0, len(top))])
            dots = cur @ cur[center]
            k = int(rng.integers(7, 17))
            S = np.argsort(-dots)[:k]
            xt = cur.copy()
            # strong dispersal of cluster + slight global jitter to change topology
            xt[S] += 0.55 * rng.standard_normal((k, d))
            xt += 0.008 * rng.standard_normal((n, d))
            xt = _norm(xt)
            xt = _anneal(xt, steps=500, beta0=16.0, beta1=260.0, lr0=0.045, lr1=0.0032, mu=0.86)
        elif u < 0.82:
            # degree-balanced transplant: kill dense cluster, reinsert into global holes
            ns = int(rng.integers(3, 7))
            top = np.argsort(-deg)[:12]
            # choose center among top, then its nearest neighbours as victims
            center = int(rng.choice(top))
            dots = cur @ cur[center]
            src = np.argsort(-dots)[:ns]
            mask = np.ones(n, dtype=bool)
            mask[src] = False
            Xkeep = cur[mask]
            holes = _find_holes(Xkeep, k=ns, rng=rng, n_cand=20000, sep=0.86)
            xt = cur.copy()
            for s_, h in zip(src, holes):
                xt[int(s_)] = h + 0.008 * rng.standard_normal(d)
            xt = _norm(xt)
            xt = _anneal(xt, steps=560, beta0=15.0, beta1=300.0, lr0=0.045, lr1=0.0030, mu=0.87)
        else:
            # tight-pair split + rattler/low-degree nudge
            flat = int(np.argmax(Gc))
            wi, wj = divmod(flat, n)
            xt = cur.copy()
            # split the tightest pair apart along their difference
            dif = xt[wi] - xt[wj]
            nd = np.linalg.norm(dif)
            if nd < 1e-9:
                dif = rng.standard_normal(d)
                nd = np.linalg.norm(dif)
            dif = dif / nd
            xt[wi] += 0.12 * dif + 0.05 * rng.standard_normal(d)
            xt[wj] -= 0.12 * dif + 0.05 * rng.standard_normal(d)
            # nudge lowest-degree points to activate them
            low = np.argsort(deg)[:4]
            pick = low[rng.integers(0, len(low))]
            xt[int(pick)] += 0.20 * rng.standard_normal(d)
            # light global noise
            xt += 0.005 * rng.standard_normal((n, d))
            xt = _norm(xt)
            xt = _anneal(xt, steps=520, beta0=18.0, beta1=320.0, lr0=0.042, lr1=0.0030, mu=0.86)
        # fine max-focused squeeze
        xt = _anneal(xt, steps=160, beta0=160.0, beta1=1200.0, lr0=0.010, lr1=0.0012, mu=0.88)
        try:
            vt = _max_cos(xt)
        except Exception:
            continue
        if vt < best_val - 1e-12:
            remain = EXPLORE_END - (time.time() - t0)
            if remain > 1.5 and vt < best_val - 1.5e-07:
                xp = _lp_polish(xt, seconds=min(0.9, max(0.35, remain * 0.07)), r0=8e-4)
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
        if cur_val > best_val + 0.007 and (it % 8 == 0) and rng.random() < 0.6:
            cur = best.copy() + 0.02 * rng.standard_normal((n, d))
            cur = _norm(cur)
            cur_val = _max_cos(cur)
        if it % 25 == 0 and (time.time() - t0) > EXPLORE_END - 6:
            break
    while time.time() - t0 < 157.0:
        # intensify around best: small shakes with balanced + sharp polish
        sigma = float(rng.choice(np.array([0.007, 0.016, 0.032])))
        # bias shake toward high-degree points
        try:
            degb, _, _ = _degrees(best, tol=1e-3)
        except Exception:
            degb = np.full(n, 6)
        xt = best.copy()
        # shake all lightly, overloaded more
        w = 1.0 + 1.5 * (degb - degb.min()) / max(1.0, (degb.max() - degb.min())))
        noise = rng.standard_normal((n, d)) * (sigma * w[:, None])
        xt = xt + noise
        xt = _norm(xt)
        xt = _anneal(xt, steps=420, beta0=28.0, beta1=650.0, lr0=0.030, lr1=0.0022, mu=0.87)
        xt = _anneal(xt, steps=140, beta0=260.0, beta1=1500.0, lr0=0.008, lr1=0.0010, mu=0.88)
        try:
            vt = _max_cos(xt)
        except Exception:
            break
        if vt < best_val - 1e-12:
            best = xt.copy()
            best_val = vt
        if time.time() - t0 > 157:
            break
    remain = min(17.0, max(0.0, 172.0 - (time.time() - t0) - 0.5))
    if remain > 1.0:
        best = _lp_polish(best, seconds=remain, r0=7e-4)
    best = _norm(best)
    return best
