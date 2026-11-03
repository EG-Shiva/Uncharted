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

def _adam(x_init, steps, T0, T1, lr0, lr1, deadline, off=0):
    X = _norm(x_init.copy())
    n, d = X.shape
    G0 = X @ X.T
    np.fill_diagonal(G0, -2.0)
    bv = float(G0.max())
    bx = X.copy()
    m1 = np.zeros_like(X); m2 = np.zeros_like(X)
    b1, b2, eps = 0.9, 0.999, 1e-8
    lt0, lt1 = np.log(T0), np.log(T1)
    ll0, ll1 = np.log(lr0), np.log(lr1)
    for it in range(steps):
        if (it & 127) == 0 and time.time() > deadline:
            break
        f = it / max(1, steps - 1)
        T = float(np.exp(lt0 + (lt1 - lt0) * f))
        lr = float(np.exp(ll0 + (ll1 - ll0) * f))
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        if m < bv:
            bv = m; bx = X.copy()
        E = np.exp((G - m) / T)
        np.fill_diagonal(E, 0.0)
        s = float(E.sum() * 0.5)
        if not np.isfinite(s) or s <= 0:
            continue
        Grad = (E @ X) / s
        Gt = Grad - (np.sum(Grad * X, axis=1, keepdims=True)) * X
        gl = it + 1 + off
        m1 *= b1; m1 += (1 - b1) * Gt
        m2 *= b2; m2 += (1 - b2) * (Gt * Gt)
        m1h = m1 / (1 - b1 ** gl)
        m2h = m2 / (1 - b2 ** gl)
        st = m1h / (np.sqrt(m2h) + eps)
        X = X - lr * st
        X = _norm(X)
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    m = float(G.max())
    if m < bv:
        bv = m; bx = X.copy()
    return bx, bv

def _greedy_pair(x, deadline):
    x = _norm(x.copy())
    best = _max_cos(x)
    bx = x.copy()
    n = x.shape[0]
    step = 2e-4
    it = 0
    while time.time() < deadline:
        it += 1
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        k = int(np.argmax(G))
        i = k // n; j = k % n
        dot = float(G[i, j])
        # tangent directions that separate i,j
        vi = dot * x[i] - x[j]
        vj = dot * x[j] - x[i]
        ni = float(np.linalg.norm(vi)); nj = float(np.linalg.norm(vj))
        if ni < 1e-14 or nj < 1e-14:
            step *= 0.5
            if step < 1e-12:
                step = 1e-5
            if it > 2000:
                break
            continue
        vi /= ni; vj /= nj
        improved = False
        for s in (step, step * 0.5, step * 0.25):
            y = x.copy()
            y[i] = x[i] + s * vi
            y[j] = x[j] + s * vj
            y[i] /= np.linalg.norm(y[i]); y[j] /= np.linalg.norm(y[j])
            v = _max_cos(y)
            if v < best - 1e-15:
                x = y; best = v; bx = y.copy()
                step = min(step * 1.2, 3e-4)
                improved = True
                break
        if not improved:
            # try single sided
            for idx, vv in ((i, vi), (j, vj)):
                y = x.copy()
                y[idx] = x[idx] + step * 0.5 * vv
                y[idx] /= np.linalg.norm(y[idx])
                v = _max_cos(y)
                if v < best - 1e-15:
                    x = y; best = v; bx = y.copy()
                    improved = True
                    break
            if not improved:
                step *= 0.5
                if step < 5e-8:
                    # random kick on this pair
                    step = 1e-5
                    if it > 1500:
                        break
        if it > 4000:
            break
    return bx, best

def _void_reinsert(X, k, rng, n_cand=4000):
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
        for _ in range(80):
            dots = others @ v
            mm = float(dots.max())
            T = 0.003
            w = np.exp((dots - mm) / T)
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

def _lp_polish(x, best, deadline):
    n, d = x.shape
    iu = np.triu_indices(n, 1)
    radius = 1.5e-3
    stall = 0
    while time.time() < deadline and radius > 1e-14:
        Gf = x @ x.T
        cf = Gf[iu]
        m = float(cf.max())
        if m < best:
            best = m
        thr = m - max(10.0 * radius, 1.2e-6)
        sel = cf > thr
        p = int(sel.sum())
        if p < 500:
            kk = min(1200, cf.size)
            idx = np.argpartition(-cf, kk - 1)[:kk]
            sel = np.zeros_like(sel, dtype=bool)
            sel[idx] = True
            p = kk
        if p > 4000:
            idx = np.where(sel)[0]
            order = np.argsort(cf[idx])[-4000:]
            ns = np.zeros_like(sel, dtype=bool)
            ns[idx[order]] = True
            sel = ns
            p = 4000
        I = iu[0][sel]; J = iu[1][sel]
        cs = cf[sel]
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
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
                rnd -= (np.sum(rnd * x, axis=1, keepdims=True)) * x
                x = _norm(x + rnd)
                stall = 0
            continue
        dx = res.x[:-1].reshape(n, d)
        improved = False
        for alpha in (1.0, 0.6, 0.3, 0.15):
            y = _norm(x + alpha * dx)
            v = _max_cos(y)
            if v < best - 1e-13:
                x = y; best = v
                radius = min(radius * 1.35, 2e-3)
                improved = True
                stall = 0
                break
        if not improved:
            radius *= 0.5
            stall += 1
            if stall >= 8:
                rnd = np.random.randn(n, d) * (radius * 0.4 + 5e-8)
                rnd -= (np.sum(rnd * x, axis=1, keepdims=True)) * x
                yt = _norm(x + rnd)
                vt = _max_cos(yt)
                if vt < best + 1e-7:
                    x = yt
                    if vt < best:
                        best = vt
                stall = 0
    return x, best

def solve():
    np.random.seed(0)
    t0 = time.time()
    BUDGET = 170.0
    deadline_final = t0 + BUDGET
    try:
        x0 = np.load("best.npy")
    except Exception:
        try:
            x0 = np.load("record.npy")
        except Exception:
            x0 = np.random.RandomState(0).randn(182, 9)
    x0 = np.asarray(x0, dtype=np.float64)
    if x0.shape != (182, 9):
        r = np.random.RandomState(1)
        y = r.randn(182, 9)
        if x0.ndim == 2:
            mh = min(x0.shape[0], 182); mw = min(x0.shape[1], 9)
            y[:mh, :mw] = x0[:mh, :mw]
        x0 = y
    x0 = _norm(x0)
    best_val = _max_cos(x0)
    x_best = x0.copy()
    n = 182
    # Stage 1 broad anneal
    if time.time() < deadline_final - 30:
        xb, vb = _adam(x_best, steps=8000, T0=0.0035, T1=0.00028,
                       lr0=0.0025, lr1=0.00022,
                       deadline=min(time.time() + 28, deadline_final - 25), off=0)
        if vb < best_val:
            best_val = vb; x_best = xb
    # Stage 2 basin hops
    sigs = [0.005, 0.010, 0.018, 0.007, 0.013]
    for trial, sig in enumerate(sigs):
        if time.time() > deadline_final - 40:
            break
        rng = np.random.RandomState(1000 + trial * 131)
        if trial == 2 or trial == 3:
            try:
                Xp = _void_reinsert(x_best, 2 if trial == 3 else 3, rng)
            except Exception:
                Xp = x_best.copy()
            Xp = Xp + sig * 0.35 * rng.randn(n, 9)
            Xp = _norm(Xp)
        else:
            G = x_best @ x_best.T
            np.fill_diagonal(G, -10.0)
            flat = G[np.triu_indices(n, 1)]
            iu0, iu1 = np.triu_indices(n, 1)
            # per-point load
            per = G.max(axis=1)
            # normalize load to scale noise
            lo = per.min(); hi = per.max()
            w = (per - lo) / max(hi - lo, 1e-12)
            noise = rng.randn(n, 9)
            scale = (sig * (0.4 + w))[:, None]
            Xp = x_best + scale * noise
            try:
                k2 = int(np.argmax(flat))
                a, b = int(iu0[k2]), int(iu1[k2])
                v = rng.randn(9); v /= np.linalg.norm(v) + 1e-12
                Xp[a] += sig * 1.5 * v
                Xp[b] -= sig * 1.5 * v
            except Exception:
                pass
            Xp = _norm(Xp)
        xb, vb = _adam(Xp, steps=3200, T0=0.0016, T1=0.0002,
                       lr0=0.0016, lr1=0.00016,
                       deadline=min(time.time() + 8, deadline_final - 30),
                       off=8000 + trial * 3200)
        if vb < best_val + 0.002 and time.time() < deadline_final - 32:
            try:
                xlp, vlp = _lp_polish(xb, vb, min(time.time() + 3.0, deadline_final - 28))
            except Exception:
                xlp, vlp = xb, vb
            if vlp < best_val - 1e-12:
                best_val = vlp; x_best = xlp
            elif vb < best_val - 1e-12:
                best_val = vb; x_best = xb
        else:
            if vb < best_val - 1e-12:
                best_val = vb; x_best = xb
    # Stage 3 sharp low-T
    if time.time() < deadline_final - 28:
        xb, vb = _adam(x_best, steps=5000, T0=0.00045, T1=0.00010,
                       lr0=0.0006, lr1=0.00010,
                       deadline=min(time.time() + 14, deadline_final - 22), off=60000)
        if vb < best_val:
            best_val = vb; x_best = xb
    # Stage 4 greedy worst-pair split (shave single max)
    if time.time() < deadline_final - 20:
        try:
            xg, vg = _greedy_pair(x_best, min(time.time() + 7, deadline_final - 18))
            if vg < best_val - 1e-14:
                best_val = vg; x_best = xg
        except Exception:
            pass
    # Stage 5 intensive LP
    rem = deadline_final - time.time() - 1.5
    if rem > 4:
        try:
            x_best, best_val = _lp_polish(x_best, best_val, time.time() + min(rem, 60))
        except Exception:
            pass
    # final tiny greedy if time
    if time.time() < deadline_final - 2:
        try:
            xg, vg = _greedy_pair(x_best, deadline_final - 1.0)
            if vg < best_val:
                best_val = vg; x_best = xg
        except Exception:
            pass
    x_best = _norm(x_best)
    x_best[~np.isfinite(x_best)] = 0.1
    x_best = _norm(x_best)
    return x_best
