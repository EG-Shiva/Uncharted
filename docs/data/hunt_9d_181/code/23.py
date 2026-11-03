import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def _contacts(X, m=None, tol=1e-9):
    n = X.shape[0]
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    if m is None:
        m = float(np.max(G))
    C = G >= m - tol
    deg = np.sum(C, axis=1)
    iu = np.triu_indices(n, 1)
    I = iu[0][G[iu] >= m - tol]
    J = iu[1][G[iu] >= m - tol]
    return m, deg, I, J, G

def _lp_polish(X, budget, init_radius=1e-3):
    t0 = time.time()
    x = X / np.linalg.norm(X, axis=1, keepdims=True)
    n, d = x.shape
    try:
        best = _max_cos(x)
    except Exception:
        return x
    radius = init_radius
    iu = np.triu_indices(n, 1)
    while time.time() - t0 < budget and radius > 1e-14:
        if time.time() - t0 + 0.15 > budget and radius < 5e-4:
            break
        Gfull = x @ x.T
        c = Gfull[iu]
        m = float(np.max(c))
        sel = c > m - max(20.0 * radius, 1e-9)
        cnt = int(np.sum(sel))
        if cnt == 0:
            radius *= 0.5
            continue
        if cnt > 2800:
            kth = np.partition(c, -2800)[-2800]
            sel = c >= kth
        I = iu[0][sel]; J = iu[1][sel]
        csel = c[sel]
        p = len(I)
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv); obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq, b_eq=np.zeros(n),
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
            v = _max_cos(y)
        except Exception:
            radius *= 0.5
            continue
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def _smooth(Y, steps, T0, T1, lr, mu=0.6):
    n, d = Y.shape
    V = np.zeros_like(Y)
    lT0 = np.log(T0); lT1 = np.log(T1)
    for s in range(steps):
        frac = s / max(steps - 1, 1)
        T = float(np.exp(lT0 + (lT1 - lT0) * frac))
        G = Y @ Y.T
        np.fill_diagonal(G, -1.0)
        m = float(np.max(G))
        W = (G - m) / T
        np.clip(W, -50.0, 0.0, out=W)
        np.exp(W, out=W)
        np.fill_diagonal(W, 0.0)
        rs = np.sum(W, axis=1)
        WY = W @ Y
        r = np.sum(W * G, axis=1)
        inv = np.zeros_like(rs)
        mk = rs > 1e-12
        inv[mk] = 1.0 / rs[mk]
        avgWY = WY * inv[:, None]
        avgR = (r * inv)[:, None]
        Dir = avgWY - avgR * Y
        Dir[~mk] = 0.0
        V *= mu
        V += Dir
        # normalize step by mean norm
        mn = float(np.mean(np.linalg.norm(V, axis=1))) + 1e-12
        Y = Y - (lr / mn) * V * 0.05
        Y /= np.linalg.norm(Y, axis=1, keepdims=True)
    return Y

def _best_void(Xkeep, rng, n_cand=3000):
    d = Xkeep.shape[1]
    cand = rng.standard_normal((n_cand, d))
    cand /= np.linalg.norm(cand, axis=1, keepdims=True)
    dots = cand @ Xkeep.T
    vals = np.max(dots, axis=1)
    bi = int(np.argmin(vals))
    v = cand[bi].copy()
    for _ in range(25):
        dv = Xkeep @ v
        m = float(np.max(dv))
        w = np.exp(np.clip((dv - m) / 0.02, -50, 0))
        g = w @ Xkeep
        g = g - np.dot(g, v) * v
        ng = float(np.linalg.norm(g))
        if ng < 1e-12:
            break
        g /= ng
        v = v - 0.02 * g
        v /= np.linalg.norm(v)
    return v / np.linalg.norm(v)

def solve():
    t0 = time.time()
    BUDGET = 174.0
    rng = np.random.default_rng(7)
    n, d = 181, 9
    cands = []
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=float)
            if a.shape == (n, d):
                a = a / np.linalg.norm(a, axis=1, keepdims=True)
                if np.all(np.isfinite(a)):
                    cands.append(a)
        except Exception:
            pass
    if not cands:
        a = rng.standard_normal((n, d))
        a /= np.linalg.norm(a, axis=1, keepdims=True)
        cands.append(a)
    vals = [_max_cos(c) for c in cands]
    bi = int(np.argmin(vals))
    xb = cands[bi].copy()
    bb = float(vals[bi])
    # quick baseline squeeze if time allows (short)
    if BUDGET - (time.time() - t0) > 20:
        xb = _lp_polish(xb, budget=3.0, init_radius=2e-3)
        bb = _max_cos(xb)
    xc = xb.copy(); bc = bb
    Tacc = 3e-05
    it = 0
    # hopping phase: reserve ~22s for final polish
    while True:
        rem = BUDGET - (time.time() - t0)
        if rem < 24.0:
            break
        it += 1
        y = xc.copy()
        # contact info of xc (cheap)
        try:
            Gtmp = xc @ xc.T
            np.fill_diagonal(Gtmp, -2.0)
            mtmp = float(np.max(Gtmp))
            deg = np.sum(Gtmp >= mtmp - 1e-9, axis=1)
        except Exception:
            deg = np.zeros(n)
            mtmp = bc
        mode = it % 6
        if mode == 0:
            # rattler void teleport: 2 lowest-degree points
            k = 2 if (it % 12 == 0) else 1
            idx = np.argsort(deg)[:k]
            keep_mask = np.ones(n, dtype=bool); keep_mask[idx] = False
            Xkeep = y[keep_mask]
            for j in idx:
                v = _best_void(Xkeep, rng, n_cand=3000)
                y[j] = v
                Xkeep = np.vstack([Xkeep, v[None, :]])
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            y = _smooth(y, steps=160, T0=0.012, T1=0.0025, lr=0.12)
        elif mode == 1:
            # overloaded hub shake
            k = 3
            idx = np.argsort(-deg)[:k]
            # include neighbors of worst hub
            worst = int(idx[0])
            try:
                row = (xc @ xc[worst])
                nbr = np.where(row >= mtmp - 0.03)[0]
            except Exception:
                nbr = idx
            tgt = np.unique(np.concatenate([idx, nbr[:8]]))
            y[tgt] += rng.normal(size=(len(tgt), d)) * 0.025
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            y = _smooth(y, steps=170, T0=0.015, T1=0.003, lr=0.14)
        elif mode == 2:
            # regional shake: random cap
            cdir = rng.standard_normal(d); cdir /= np.linalg.norm(cdir)
            proj = y @ cdir
            sel = np.where(proj > np.quantile(proj, 0.75))[0]
            y[sel] += rng.normal(size=(len(sel), d)) * 0.02
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            y = _smooth(y, steps=150, T0=0.010, T1=0.002, lr=0.12)
        elif mode == 3:
            y += rng.normal(size=y.shape) * 0.008
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            y = _smooth(y, steps=140, T0=0.008, T1=0.002, lr=0.10)
        elif mode == 4:
            y += rng.normal(size=y.shape) * 0.025
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            y = _smooth(y, steps=180, T0=0.018, T1=0.003, lr=0.15)
        else:
            # single-point hole fill for a random loose point
            idx = np.argsort(deg)[:4]
            j = int(rng.choice(idx))
            mask = np.ones(n, dtype=bool); mask[j] = False
            v = _best_void(y[mask], rng, n_cand=2500)
            # mix with old to keep some continuity
            y[j] = v
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            y = _smooth(y, steps=150, T0=0.010, T1=0.002, lr=0.12)
        try:
            v = _max_cos(y)
        except Exception:
            continue
        if v < bc or rng.random() < np.exp(-(v - bc) / Tacc):
            xc, bc = y, v
            if v < bb - 1e-12:
                # promising: short LP squeeze to verify true basin value
                rem2 = BUDGET - (time.time() - t0)
                if rem2 > 26.0:
                    ys = _lp_polish(y, budget=min(2.0, rem2 - 24.0), init_radius=0.004)
                    vs = _max_cos(ys)
                    if vs < bb:
                        xb, bb = ys, vs
                        xc, bc = ys.copy(), vs
                    elif vs < bc:
                        xc, bc = ys.copy(), vs
                    else:
                        xb, bb = y.copy(), v
                else:
                    xb, bb = y.copy(), v
        # slow cool
        Tacc *= 0.9995
        if Tacc < 8e-06:
            Tacc = 8e-06
    # final squeeze of best
    rem = BUDGET - (time.time() - t0)
    if rem > 2.0:
        xb = _lp_polish(xb, budget=max(2.0, rem - 1.0), init_radius=2e-3)
        bb = _max_cos(xb)
    xb = xb / np.linalg.norm(xb, axis=1, keepdims=True)
    # safety: if somehow worse than file, return file best clipped
    try:
        b0 = np.load("best.npy")
        b0 = b0 / np.linalg.norm(b0, axis=1, keepdims=True)
        if _max_cos(b0) < _max_cos(xb):
            return b0
    except Exception:
        pass
    return xb
