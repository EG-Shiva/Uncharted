import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _norm(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return X / n

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _adam_softmax(x0, beta, steps, lr):
    x = x0.copy()
    n, d = x.shape
    m = np.zeros_like(x)
    v = np.zeros_like(x)
    b1, b2, eps = 0.9, 0.999, 1e-8
    best_x = x.copy()
    g0 = x @ x.T
    np.fill_diagonal(g0, -2.0)
    best_max = float(g0.max())
    t = 0
    iu = np.triu_indices(n, 1)
    for it in range(steps):
        t += 1
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        M = float(G.max())
        E = np.exp(beta * (G - M))
        np.fill_diagonal(E, 0.0)
        Z = float(E[iu].sum())
        if Z == 0 or not np.isfinite(Z):
            Z = 1.0
        W = E / Z
        grad = W @ x
        dot = (grad * x).sum(axis=1, keepdims=True)
        gt = grad - dot * x
        m = b1 * m + (1 - b1) * gt
        v = b2 * v + (1 - b2) * (gt * gt)
        mhat = m / (1 - b1 ** t)
        vhat = v / (1 - b2 ** t)
        step = lr * mhat / (np.sqrt(vhat) + eps)
        x = x - step
        x = _norm(x)
        if (it % 50 == 0) or (it == steps - 1):
            c = _max_cos(x)
            if c < best_max:
                best_max = c
                best_x = x.copy()
    return best_x, best_max

def _lp_polish(x, seconds, deadline=1e18):
    x = _norm(x.copy())
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and time.time() < deadline:
        if radius < 1e-14:
            break
        Gfull = x @ x.T
        c_all = Gfull[iu]
        m = float(c_all.max())
        thresh = m - max(20 * radius, 1e-9)
        sel = c_all > thresh
        idx = np.where(sel)[0]
        if len(idx) > 2000:
            part = np.argpartition(c_all[idx], -2000)[-2000:]
            mask = np.zeros_like(sel, dtype=bool)
            mask[idx[part]] = True
            sel = mask
            idx = np.where(sel)[0]
        if len(idx) == 0:
            radius *= 0.5
            continue
        I, J = iu[0][sel], iu[1][sel]
        c = c_all[sel]
        p = len(I)
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv); obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-c, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y = _norm(y)
        vv = _max_cos(y)
        if vv < best:
            x, best, radius = y, vv, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def solve():
    np.random.seed(0)
    rng = np.random.RandomState(0)
    t0 = time.time()
    deadline = t0 + 175.0
    n, d = 93, 13
    # load baseline
    x_base = None
    base_val = 1e9
    for fn in ["best.npy", "record.npy"]:
        try:
            a = np.load(fn)
            if a.shape == (93, 13):
                a = _norm(a.astype(float))
                v = _max_cos(a)
                if v < base_val:
                    base_val = v
                    x_base = a.copy()
        except Exception:
            pass
    if x_base is None:
        x_base = _norm(rng.randn(n, d))
        base_val = _max_cos(x_base)
    best_x = x_base.copy()
    best_val = float(base_val)

    # ---- C93 single-orbit screening: exact LP over weights ----
    # cos table a=1..46, t=1..46
    aa = np.arange(1, 47)[:, None]
    tt = np.arange(1, 47)[None, :]
    cos_tab = np.cos(2.0 * np.pi * aa * tt / 93.0)  # 46x46
    c_obj = np.zeros(8); c_obj[-1] = 1.0
    A_eq = np.zeros((1, 8)); A_eq[0, :7] = 1.0
    b_eq = np.ones(1)
    bounds = [(0, None)] * 7 + [(None, None)]
    cands = []  # (val, freqs, w)
    NSAMP = 400
    # deterministic sampling
    for s in range(NSAMP):
        if time.time() > t0 + 22:
            break
        try:
            freq = np.sort(rng.choice(46, size=6, replace=False) + 1)  # 1..46
        except Exception:
            continue
        # build Aub: w_j*C +w6 -T <=0
        try:
            Cmat = cos_tab[freq - 1, :]  # 6x46
            Aub = np.zeros((46, 8))
            Aub[:, :6] = Cmat.T
            Aub[:, 6] = 1.0
            Aub[:, 7] = -1.0
            bub = np.zeros(46)
            res = linprog(c_obj, A_ub=Aub, b_ub=bub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
            if res.status != 0 or not np.isfinite(res.fun):
                continue
            val = float(res.fun)
            w = np.array(res.x[:7], dtype=float)
            w = np.clip(w, 0, None)
            sw = w.sum()
            if sw <= 0:
                continue
            w = w / sw
            cands.append((val, freq.copy(), w.copy()))
        except Exception:
            continue
    cands.sort(key=lambda z: z[0])
    top = cands[:4] if len(cands) >= 4 else cands

    # ---- build symmetric codes for top sets, break symmetry ----
    ks = np.arange(93)
    for rank, (sval, freq, w) in enumerate(top):
        if time.time() > deadline - 55:
            break
        try:
            r = np.sqrt(np.clip(w[:6], 0, 1))
            cax = float(np.sqrt(np.clip(w[6], 0, 1)))
            # two signs for cax give same Gram; keep positive
            Xsym = np.zeros((93, 13))
            for j in range(6):
                a = int(freq[j])
                ang = 2.0 * np.pi * a * ks / 93.0
                Xsym[:, 2 * j] = r[j] * np.cos(ang)
                Xsym[:, 2 * j + 1] = r[j] * np.sin(ang)
            Xsym[:, 12] = cax
            Xsym = _norm(Xsym)
            # break symmetry with small noise (deterministic per rank)
            rng2 = np.random.RandomState(1000 + rank)
            for trial in range(2):
                if time.time() > deadline - 50:
                    break
                sigma = 0.008 if trial == 0 else 0.02
                Xt = _norm(Xsym + sigma * rng2.randn(93, 13))
                # rapid annealed log-sum-exp at several temperatures
                for beta, steps, lr in [(1500, 900, 0.006), (6000, 900, 0.0025)]:
                    if time.time() > deadline - 50:
                        break
                    Xt_opt, v_opt = _adam_softmax(Xt, beta, steps, lr)
                    Xt = Xt_opt
                    if v_opt < best_val - 1e-12:
                        best_val = float(v_opt)
                        best_x = Xt.copy()
                # light LP snap
                if time.time() < deadline - 45:
                    try:
                        Xlp = _lp_polish(Xt, seconds=2.5, deadline=deadline - 40)
                        vlp = _max_cos(Xlp)
                        if vlp < best_val:
                            best_val = float(vlp)
                            best_x = Xlp.copy()
                        # continue annealing from snapped point once
                        Xt = Xlp
                    except Exception:
                        pass
        except Exception:
            continue

    # ---- final full-coordinate polish of global best ----
    # annealed Adam + LP with remaining time
    try:
        if time.time() < deadline - 20:
            xcur = best_x.copy()
            # medium anneal
            for beta, steps, lr in [(2500, 1200, 0.004), (8000, 1200, 0.0018), (20000, 800, 0.0009)]:
                if time.time() > deadline - 18:
                    break
                xo, vo = _adam_softmax(xcur, beta, steps, lr)
                if vo < best_val:
                    best_val = float(vo)
                    best_x = xo.copy()
                xcur = best_x.copy() if vo >= best_val - 1e-9 else xo.copy()
        if time.time() < deadline - 12:
            try:
                tl = min(10.0, deadline - time.time() - 0.8)
                if tl > 2:
                    xlp = _lp_polish(best_x, seconds=tl, deadline=deadline - 0.5)
                    vlp = _max_cos(xlp)
                    if vlp < best_val:
                        best_val = float(vlp)
                        best_x = xlp.copy()
            except Exception:
                pass
    except Exception:
        pass

    best_x = _norm(best_x)
    return best_x
