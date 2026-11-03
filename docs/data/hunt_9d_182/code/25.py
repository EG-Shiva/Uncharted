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

def _adam_lse(X0, steps, T0, T1, lr0, lr1, deadline, off=0):
    X = _norm(X0.copy())
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
    gl = off
    for it in range(steps):
        if (it & 63) == 0 and time.time() > deadline:
            break
        f = it / max(1, steps - 1)
        T = float(np.exp(lt0 + (lt1 - lt0) * f))
        lr = float(np.exp(ll0 + (ll1 - ll0) * f))
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
        Gt = Grad - np.sum(Grad * X, axis=1, keepdims=True) * X
        gl += 1
        m1 *= b1
        m1 += (1 - b1) * Gt
        m2 *= b2
        m2 += (1 - b2) * (Gt * Gt)
        m1h = m1 / (1 - b1 ** gl)
        m2h = m2 / (1 - b2 ** gl)
        st = m1h / (np.sqrt(m2h) + eps)
        dn = np.sum(st * X, axis=1, keepdims=True)
        st = st - dn * X
        nm = np.linalg.norm(st, axis=1, keepdims=True)
        sc = np.minimum(1.0, 1.5 / (nm + 1e-12))
        X = X - lr * st * sc
        X = _norm(X)
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    m = float(G.max())
    if m < bv:
        bv = m
        bx = X.copy()
    return bx, bv

def _mom_lse(X0, steps, T0, T1, lr0, lr1, mu, deadline):
    X = _norm(X0.copy())
    n, d = X.shape
    G0 = X @ X.T
    np.fill_diagonal(G0, -2.0)
    bv = float(G0.max())
    bx = X.copy()
    V = np.zeros_like(X)
    lt0, lt1 = np.log(T0), np.log(T1)
    ll0, ll1 = np.log(lr0), np.log(lr1)
    for it in range(steps):
        if (it & 63) == 0 and time.time() > deadline:
            break
        f = it / max(1, steps - 1)
        T = float(np.exp(lt0 + (lt1 - lt0) * f))
        lr = float(np.exp(ll0 + (ll1 - ll0) * f))
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
        Gt = Grad - np.sum(Grad * X, axis=1, keepdims=True) * X
        V = mu * V + (1.0 - mu) * Gt
        Vt = V - np.sum(V * X, axis=1, keepdims=True) * X
        rms = float(np.sqrt(np.mean(Vt * Vt))) + 1e-12
        step = Vt / rms
        # keep tangent
        step = step - np.sum(step * X, axis=1, keepdims=True) * X
        X = X - lr * step
        X = _norm(X)
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    m = float(G.max())
    if m < bv:
        bv = m
        bx = X.copy()
    return bx, bv

def _void_reinsert(X, k, rng):
    n, d = X.shape
    Y = X.copy()
    G = Y @ Y.T
    np.fill_diagonal(G, -10.0)
    per = G.max(axis=1)
    worst = np.argsort(-per)[:k]
    for idx in worst:
        others = np.delete(Y, idx, axis=0)
        cand = rng.randn(2000, d)
        cand /= (np.linalg.norm(cand, axis=1, keepdims=True) + 1e-12)
        sc = cand @ others.T
        mx = sc.max(axis=1)
        v = cand[int(np.argmin(mx))].copy()
        for _ in range(60):
            dots = others @ v
            mm = float(dots.max())
            w = np.exp((dots - mm) / 0.005)
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

def _lp_jam(x, deadline, r0=7e-4):
    x = _norm(x.copy())
    n, d = x.shape
    best = _max_cos(x)
    bx = x.copy()
    iu = np.triu_indices(n, 1)
    radius = r0
    stall = 0
    itn = 0
    while time.time() < deadline and radius > 1e-13 and itn < 40:
        itn += 1
        Gf = x @ x.T
        cf = Gf[iu]
        m = float(cf.max())
        if m < best - 1e-12:
            best = m
            bx = x.copy()
        thresh = m - max(12.0 * radius, 8e-7)
        sel = cf > thresh
        p = int(sel.sum())
        if p < 500:
            kk = min(1200, cf.size)
            idx = np.argpartition(-cf, kk - 1)[:kk]
            sel = np.zeros_like(sel, dtype=bool)
            sel[idx] = True
            p = kk
        if p > 3000:
            idx = np.where(sel)[0]
            order = np.argsort(cf[idx])[-3000:]
            ns = np.zeros_like(sel, dtype=bool)
            ns[idx[order]] = True
            sel = ns
            p = 3000
        I = iu[0][sel]
        J = iu[1][sel]
        cs = cf[sel]
        rows = np.repeat(np.arange(p), 2 * d + 1)
        c0 = np.concatenate([I[:, None] * d + np.arange(d),
                             J[:, None] * d + np.arange(d),
                             np.full((p, 1), n * d)], 1).ravel()
        vv = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vv, (rows, c0)), shape=(p, n * d + 1)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, n * d + 1)).tocsr()
            obj = np.zeros(n * d + 1)
            obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-cs, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            stall += 1
            if stall >= 6:
                rnd = np.random.randn(n, d) * 5e-7
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
                radius = min(radius * 1.3, 1.2e-3)
                improved = True
                stall = 0
                break
        if not improved:
            radius *= 0.5
            stall += 1
            if stall >= 7:
                rnd = np.random.randn(n, d) * (radius * 0.5 + 5e-8)
                rnd -= (rnd * x).sum(axis=1, keepdims=True) * x
                yt = _norm(x + rnd)
                vt = _max_cos(yt)
                if vt < best + 2e-7:
                    x = yt
                    if vt < best:
                        best = vt
                        bx = yt.copy()
                stall = 0
    return bx, best

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
            mh = min(x0.shape[0], 182)
            mw = min(x0.shape[1], 9)
            y[:mh, :mw] = x0[:mh, :mw]
        x0 = y
    n, d = 182, 9
    x_best = _norm(x0)
    best_val = _max_cos(x_best)
    iu0 = np.triu_indices(n, 1)
    off = 0
    # Stage 1: warm equalization with momentum to spread load
    dl = min(t0 + 18.0, deadline_final - 90)
    if time.time() < dl - 3:
        try:
            xb, vb = _mom_lse(x_best, steps=2500, T0=0.012, T1=0.0012, lr0=0.004, lr1=0.0012, mu=0.85, deadline=dl)
            if vb < best_val:
                best_val = vb
                x_best = xb
            off += 2500
        except Exception:
            pass
    # Stage 1b: cool adam
    dl2 = min(time.time() + 12.0, deadline_final - 80)
    if time.time() < dl2 - 3:
        try:
            xb, vb = _adam_lse(x_best, steps=2500, T0=0.0025, T1=0.00025, lr0=0.002, lr1=0.0002, deadline=dl2, off=off)
            off += 2500
            if vb < best_val:
                best_val = vb
                x_best = xb
        except Exception:
            pass
    sigs = [0.012, 0.005, 0.020, 0.008, 0.026, 0.014, 0.006, 0.018, 0.010, 0.022]
    trial = 0
    while time.time() < deadline_final - 42 and trial < 9:
        if time.time() > t0 + BUDGET - 50:
            break
        sig = sigs[trial % len(sigs)]
        rng = np.random.RandomState(9000 + trial * 277)
        mode = trial % 4
        try:
            if mode == 2:
                Xp = _void_reinsert(x_best, 3 if (trial % 2 == 0) else 2, rng)
                Xp = _norm(Xp + (sig * 0.3) * rng.randn(n, d))
            elif mode == 3:
                # subset coherent shift: move a random cluster opposite
                Xp = x_best.copy()
                perm = rng.permutation(n)
                ksub = 18 + int(rng.rand() * 14)
                sub = perm[:ksub]
                v = rng.randn(d)
                v /= (np.linalg.norm(v) + 1e-12)
                Xp[sub] += sig * 1.5 * v
                Xp[perm[ksub:ksub + ksub]] -= sig * 0.8 * v
                # contact-weighted jitter
                try:
                    G = x_best @ x_best.T
                    np.fill_diagonal(G, -10.0)
                    per = G.max(axis=1)
                    w = (per - per.min() + 1e-9)
                    w = w / w.max()
                    Xp += (sig * 0.5 * w[:, None]) * rng.randn(n, d)
                except Exception:
                    Xp += sig * 0.5 * rng.randn(n, d)
                Xp = _norm(Xp)
            else:
                try:
                    G = x_best @ x_best.T
                    np.fill_diagonal(G, -10.0)
                    flat = G[iu0]
                    thr = np.sort(flat)[-800]
                    cnt = np.zeros(n)
                    A = iu0[0][flat > thr]
                    B = iu0[1][flat > thr]
                    for a, b in zip(A, B):
                        cnt[a] += 1
                        cnt[b] += 1
                    mx = cnt.max() if cnt.max() > 0 else 1.0
                    cn = cnt / mx
                except Exception:
                    cn = np.zeros(n)
                scale = (sig * (0.35 + cn))[:, None]
                Xp = x_best + scale * rng.randn(n, d)
                try:
                    G2 = x_best @ x_best.T
                    np.fill_diagonal(G2, -2.0)
                    k2 = int(np.argmax(G2))
                    a = k2 // n
                    b = k2 % n
                    vv = rng.randn(d)
                    vv /= (np.linalg.norm(vv) + 1e-12)
                    Xp[a] += sig * 2.2 * vv
                    Xp[b] -= sig * 2.2 * vv
                except Exception:
                    pass
                Xp = _norm(Xp)
        except Exception:
            Xp = x_best.copy()
        inner_dl = min(time.time() + 8.0, deadline_final - 38)
        try:
            if trial % 2 == 0:
                xb, vb = _adam_lse(Xp, steps=2200, T0=0.0022, T1=0.00020, lr0=0.0018, lr1=0.00018, deadline=inner_dl, off=off)
            else:
                xb, vb = _mom_lse(Xp, steps=2200, T0=0.004, T1=0.00030, lr0=0.0022, lr1=0.0006, mu=0.88, deadline=inner_dl)
            off += 2200
        except Exception:
            trial += 1
            continue
        if vb < best_val - 1e-12:
            best_val = vb
            x_best = xb
        elif vb < best_val + 0.0015 and time.time() < deadline_final - 36:
            try:
                xlp, vlp = _lp_jam(xb, min(time.time() + 2.2, deadline_final - 34), r0=6e-4)
                if vlp < best_val - 1e-12:
                    best_val = vlp
                    x_best = xlp
            except Exception:
                pass
        trial += 1
    # sharp equalize
    if time.time() < deadline_final - 30:
        try:
            xb, vb = _adam_lse(x_best, steps=3500, T0=0.00055, T1=0.00010, lr0=0.0007, lr1=0.00010, deadline=min(time.time() + 12.0, deadline_final - 26), off=off)
            if vb < best_val:
                best_val = vb
                x_best = xb
        except Exception:
            pass
    rem = deadline_final - time.time()
    if rem > 5:
        try:
            x_best, best_val = _lp_jam(x_best, deadline_final, r0=7e-4)
        except Exception:
            pass
    x_best = _norm(x_best)
    x_best[~np.isfinite(x_best)] = 0.1
    x_best = _norm(x_best)
    return x_best
