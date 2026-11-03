import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _norm(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[~np.isfinite(n)] = 1.0
    n[n < 1e-12] = 1.0
    return x / n

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _adam(x_init, steps, T0, T1, lr0, lr1, deadline, seed_off=0):
    X = _norm(x_init.copy())
    n, d = X.shape
    G0 = X @ X.T
    np.fill_diagonal(G0, -2.0)
    bv = float(G0.max())
    bx = X.copy()
    m1 = np.zeros_like(X)
    m2 = np.zeros_like(X)
    b1, b2, eps = 0.9, 0.999, 1e-8
    lt0, lt1 = np.log(T0), np.log(T1)
    ll0, ll1 = np.log(lr0), np.log(lr1)
    glob = seed_off
    for it in range(steps):
        if (it & 127) == 0 and time.time() > deadline:
            break
        frac = it / max(1, steps - 1)
        T = float(np.exp(lt0 + (lt1 - lt0) * frac))
        lr = float(np.exp(ll0 + (ll1 - ll0) * frac))
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        if m < bv:
            bv = m
            bx = X.copy()
        E = np.exp((G - m) / T)
        np.fill_diagonal(E, 0.0)
        s = float(E.sum() * 0.5)
        if not np.isfinite(s) or s <= 0:
            continue
        Grad = (E @ X) / s
        dot = np.sum(Grad * X, axis=1, keepdims=True)
        Gt = Grad - dot * X
        glob += 1
        m1 *= b1
        m1 += (1 - b1) * Gt
        m2 *= b2
        m2 += (1 - b2) * (Gt * Gt)
        m1h = m1 / (1 - b1 ** glob)
        m2h = m2 / (1 - b2 ** glob)
        step = m1h / (np.sqrt(m2h) + eps)
        # keep tangent
        dn = np.sum(step * X, axis=1, keepdims=True)
        step = step - dn * X
        # clip extreme
        nm = np.linalg.norm(step, axis=1, keepdims=True)
        # avoid huge steps
        sc = np.minimum(1.0, 2.0 / (nm + 1e-12))
        X = X - lr * step * sc
        X = _norm(X)
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    m = float(G.max())
    if m < bv:
        bv = m
        bx = X.copy()
    return bx, bv

def _lp(x, deadline, r0=8e-4):
    x = _norm(x.copy())
    n, d = x.shape
    best = _max_cos(x)
    bx = x.copy()
    iu = np.triu_indices(n, 1)
    radius = r0
    stall = 0
    while time.time() < deadline and radius > 1e-13:
        Gf = x @ x.T
        cf = Gf[iu]
        m = float(cf.max())
        thresh = m - max(18.0 * radius, 4e-7)
        sel = cf > thresh
        p = int(sel.sum())
        if p < 400:
            kk = min(900, cf.size)
            idx = np.argpartition(-cf, kk - 1)[:kk]
            sel = np.zeros_like(sel, dtype=bool)
            sel[idx] = True
            p = kk
        if p > 3800:
            idx = np.where(sel)[0]
            order = np.argsort(cf[idx])[-3800:]
            ns = np.zeros_like(sel, dtype=bool)
            ns[idx[order]] = True
            sel = ns
            p = 3800
        I = iu[0][sel]; J = iu[1][sel]
        cs = cf[sel]
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel() if False else np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, n * d + 1)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, n * d + 1)).tocsr()
            obj = np.zeros(n * d + 1); obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-cs, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            stall += 1
            if stall >= 7:
                rnd = np.random.randn(n, d) * 8e-7
                rnd -= (rnd * x).sum(axis=1, keepdims=True) * x
                x = _norm(x + rnd)
                stall = 0
            continue
        dx = res.x[:-1].reshape(n, d)
        improved = False
        for alpha in (1.0, 0.5, 0.25, 0.125):
            y = _norm(x + alpha * dx)
            v = _max_cos(y)
            if v < best - 1e-12:
                best = v
                bx = y.copy()
                x = y
                radius = min(radius * 1.35, 1.2e-3)
                improved = True
                stall = 0
                break
        if not improved:
            radius *= 0.5
            stall += 1
            if stall >= 8:
                rnd = np.random.randn(n, d) * (radius * 0.5 + 8e-8)
                rnd -= (rnd * x).sum(axis=1, keepdims=True) * x
                yt = _norm(x + rnd)
                vt = _max_cos(yt)
                if vt < best + 3e-7:
                    x = yt
                    if vt < best:
                        best = vt; bx = yt.copy()
                stall = 0
        if m < best:
            best = m; bx = x.copy()
    return bx, best

def _void(X, k, rng, n_cand=2500):
    n, d = X.shape
    Y = X.copy()
    G = Y @ Y.T
    np.fill_diagonal(G, -10.0)
    per = G.max(axis=1)
    worst = np.argsort(-per)[:k]
    for idx in worst:
        others = np.delete(Y, idx, axis=0)
        cand = rng.randn(n_cand, d)
        cand = cand / (np.linalg.norm(cand, axis=1, keepdims=True) + 1e-12)
        sc = cand @ others.T
        mx = sc.max(axis=1)
        v = cand[int(np.argmin(mx))].copy()
        for _ in range(120):
            dots = others @ v
            mm = float(dots.max())
            w = np.exp((dots - mm) / 0.004)
            ws = float(w.sum())
            if not np.isfinite(ws) or ws <= 0:
                break
            w /= ws
            g = w @ others
            g = g - float(g @ v) * v
            v = v - 0.08 * g
            nv = float(np.linalg.norm(v))
            if not np.isfinite(nv) or nv < 1e-12:
                break
            v /= nv
        Y[idx] = v
    return _norm(Y)

def solve():
    np.random.seed(0)
    t0 = time.time()
    BUDGET = 168.0
    deadline_final = t0 + BUDGET - 1.0
    try:
        x0 = np.load("best.npy")
    except Exception:
        try:
            x0 = np.load("record.npy")
        except Exception:
            x0 = np.random.RandomState(0).randn(182, 9)
    x0 = np.asarray(x0, dtype=np.float64)
    if x0.shape != (182, 9):
        rngf = np.random.RandomState(1)
        y = rngf.randn(182, 9)
        if x0.ndim == 2:
            mh = min(x0.shape[0], 182); mw = min(x0.shape[1], 9)
            y[:mh, :mw] = x0[:mh, :mw]
        x0 = y
    x_best = _norm(x0)
    best_val = _max_cos(x_best)
    n = 182
    iu0 = np.triu_indices(n, 1)
    # stage 1: broad anneal from best
    dl = min(t0 + 30.0, deadline_final - 60)
    if time.time() < dl - 5:
        xb, vb = _adam(x_best, steps=8000, T0=0.0035, T1=0.00030, lr0=0.0025, lr1=0.00022, deadline=dl, seed_off=0)
        if vb < best_val:
            best_val = vb; x_best = xb
    # stage 2: basin hops
    sigs = [0.004, 0.010, 0.017, 0.007, 0.024, 0.012, 0.005, 0.019]
    trial = 0
    off = 8000
    while time.time() < deadline_final - 38 and trial < 10:
        if time.time() > t0 + BUDGET - 45:
            break
        sig = sigs[trial % len(sigs)]
        rng = np.random.RandomState(7000 + trial * 131)
        if trial % 3 == 2:
            try:
                Xp = _void(x_best, 2 if (trial % 2 == 0) else 3, rng)
            except Exception:
                Xp = x_best.copy()
            Xp = Xp + (sig * 0.35) * rng.randn(n, 9)
            Xp = _norm(Xp)
        else:
            try:
                G = x_best @ x_best.T
                np.fill_diagonal(G, -10.0)
                flat = G[iu0]
                thr = np.sort(flat)[-700]
                msk = flat > thr - 1e-12
                cnt = np.zeros(n)
                A = iu0[0][msk]; B = iu0[1][msk]
                for a, b in zip(A, B):
                    cnt[a] += 1; cnt[b] += 1
                mx = cnt.max() if cnt.max() > 0 else 1.0
                cn = cnt / mx
            except Exception:
                cn = np.zeros(n)
            noise = rng.randn(n, 9)
            scale = (sig * (0.35 + cn))[:, None]
            Xp = x_best + scale * noise
            try:
                G2 = x_best @ x_best.T
                np.fill_diagonal(G2, -2.0)
                k2 = int(np.argmax(G2))
                a = k2 // n; b = k2 % n
                v = rng.randn(9); v /= (np.linalg.norm(v) + 1e-12)
                Xp[a] += sig * 2.0 * v
                Xp[b] -= sig * 2.0 * v
            except Exception:
                pass
            Xp = _norm(Xp)
        inner_dl = min(time.time() + 9.0, deadline_final - 35)
        xb, vb = _adam(Xp, steps=3200, T0=0.0018, T1=0.00022, lr0=0.0018, lr1=0.00018, deadline=inner_dl, seed_off=off)
        off += 3200
        if vb < best_val + 0.0025 and time.time() < deadline_final - 33:
            try:
                xlp, vlp = _lp(xb, min(time.time() + 3.0, deadline_final - 32), r0=7e-4)
            except Exception:
                xlp, vlp = xb, vb
            if vlp < best_val - 1e-12:
                best_val = vlp; x_best = xlp
            elif vb < best_val - 1e-12:
                best_val = vb; x_best = xb
        else:
            if vb < best_val - 1e-12:
                best_val = vb; x_best = xb
        trial += 1
    # stage 3: sharp equalize
    if time.time() < deadline_final - 30:
        xb, vb = _adam(x_best, steps=4500, T0=0.0005, T1=0.00012, lr0=0.0007, lr1=0.00012, deadline=min(time.time()+14.0, deadline_final-28), seed_off=off)
        if vb < best_val:
            best_val = vb; x_best = xb
    # stage 4: final jam
    rem = deadline_final - time.time()
    if rem > 5:
        try:
            x_best, best_val = _lp(x_best, deadline_final, r0=8e-4)
        except Exception:
            pass
    x_best = _norm(x_best)
    x_best[~np.isfinite(x_best)] = 0.1
    x_best = _norm(x_best)
    return x_best
