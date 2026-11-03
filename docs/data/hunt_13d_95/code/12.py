import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())


def lp_polish(x, seconds, radius_init=1e-3, max_c=2000):
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
        thresh = m - max(20 * radius, 1e-9)
        sel = c > thresh
        cnt = int(sel.sum())
        if cnt == 0:
            break
        if cnt > max_c:
            idx = np.argpartition(c, -max_c)[-max_c:]
            mask = np.zeros_like(sel, dtype=bool)
            mask[idx] = True
            sel = mask
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(),
                           (np.repeat(np.arange(n), d), np.arange(n * d))),
                          shape=(n, nv)).tocsr()
        obj = np.zeros(nv)
        obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c[sel], A_eq=A_eq,
                          b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                          method="highs")
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


def contact_counts(x, mc, tol=0.01):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return (g > mc - tol).sum(axis=1)


def smooth_refine(X, steps, T0, T1, lr0, lr1, mu=0.9, idx=None):
    n, d = X.shape
    if idx is None:
        idx_arr = np.arange(n)
    else:
        idx_arr = np.asarray(idx, dtype=int)
    k = len(idx_arr)
    V = np.zeros((k, d))
    lT0, lT1 = np.log(T0), np.log(T1)
    ll0, ll1 = np.log(lr0), np.log(lr1)
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
        if not np.isfinite(sw) or sw == 0:
            break
        W /= sw
        Grad = W[idx_arr] @ X
        dot = np.sum(Grad * X[idx_arr], axis=1, keepdims=True)
        Gt = Grad - dot * X[idx_arr]
        V = mu * V + Gt
        X[idx_arr] = X[idx_arr] - lr * V
        X[idx_arr] /= np.linalg.norm(X[idx_arr], axis=1, keepdims=True)
    return X


def refine_hole(others, x0, steps=250, lr0=0.15, lr1=0.005):
    x = x0 / np.linalg.norm(x0)
    best_x = x.copy()
    best_val = float((others @ x).max())
    l0, l1 = np.log(lr0), np.log(lr1)
    m0 = others.shape[0]
    for s in range(steps):
        f = s / max(1, steps - 1)
        lr = float(np.exp(l0 + (l1 - l0) * f))
        dots = others @ x
        m = float(dots.max())
        if m0 > 8:
            part = np.argpartition(-dots, 8)[:8]
            thr = m - 0.02
            act = part[dots[part] > thr]
            if len(act) == 0:
                act = part[:1]
        else:
            act = np.where(dots > m - 0.02)[0]
        g = others[act].mean(axis=0)
        gt = g - float(g @ x) * x
        if np.linalg.norm(gt) < 1e-12:
            break
        x = x - lr * gt
        x /= np.linalg.norm(x)
        m2 = float((others @ x).max())
        if m2 < best_val:
            best_val = m2
            best_x = x.copy()
    return best_x, best_val


def deep_hole(others, rng, n_samples=3000, n_ref=3):
    d = others.shape[1]
    C = rng.normal(size=(n_samples, d))
    C /= np.linalg.norm(C, axis=1, keepdims=True)
    md = (C @ others.T).max(axis=1)
    order = np.argpartition(md, n_ref)[:n_ref]
    bx = None
    bv = 1e9
    for j in order:
        rx, rv = refine_hole(others, C[j])
        if rv < bv:
            bv = rv
            bx = rx
    return bx


def solve():
    t0 = time.time()
    BUDGET = 168.0
    EXPL_T = 124.0
    rng = np.random.default_rng(0)

    try:
        xb = np.load("best.npy").astype(float)
    except Exception:
        xb = None
    try:
        xr = np.load("record.npy").astype(float)
    except Exception:
        xr = None
    cands = []
    if xb is not None and xb.shape == (95, 13):
        cands.append(xb / np.linalg.norm(xb, axis=1, keepdims=True))
    if xr is not None and xr.shape == (95, 13):
        cands.append(xr / np.linalg.norm(xr, axis=1, keepdims=True))
    if not cands:
        tmp = rng.normal(size=(95, 13))
        cands.append(tmp / np.linalg.norm(tmp, axis=1, keepdims=True))
    vals = [max_cos(c) for c in cands]
    bi = int(np.argmin(vals))
    x_best = cands[bi].copy()
    best_val = float(vals[bi])
    cur = x_best.copy()
    cur_val = best_val
    n, d = 95, 13

    it = 0
    while time.time() - t0 < EXPL_T:
        it += 1
        if time.time() - t0 > BUDGET - 35:
            break
        remain = EXPL_T - (time.time() - t0)
        if remain < 1.0:
            break
        r = rng.random()
        try:
            if r < 0.35:
                # single / double deep-hole reinsertion of overloaded points
                k = 1 if rng.random() < 0.6 else 2
                cc = contact_counts(cur, cur_val, tol=0.01)
                order = np.argsort(-cc)
                if rng.random() < 0.7:
                    idx = order[:k].copy()
                    if k > 1 and rng.random() < 0.4:
                        idx[-1] = int(rng.integers(0, n))
                elif rng.random() < 0.5:
                    idx = rng.choice(n, size=k, replace=False)
                else:
                    idx = np.argsort(cc)[:k]
                Xt = cur.copy()
                for j in idx:
                    mask = np.ones(n, dtype=bool)
                    mask[int(j)] = False
                    others = Xt[mask]
                    ns = 3500 if k == 1 else 2000
                    Xt[int(j)] = deep_hole(others, rng, n_samples=ns, n_ref=3)
                Xt /= np.linalg.norm(Xt, axis=1, keepdims=True)
                v0 = max_cos(Xt)
                if v0 > cur_val + 0.10:
                    continue
                Xp = lp_polish(Xt, seconds=min(1.4, max(0.7, remain * 0.12)), radius_init=0.01)
                v = max_cos(Xp)
                if v < best_val - 1e-12:
                    x_best = Xp.copy(); best_val = v
                    cur = Xp.copy(); cur_val = v
                elif v < cur_val:
                    cur = Xp.copy(); cur_val = v
                else:
                    if rng.random() < 0.05 and v < cur_val + 0.01:
                        cur = Xp.copy(); cur_val = v
                    if rng.random() < 0.10:
                        cur = x_best.copy(); cur_val = best_val
            elif r < 0.72:
                # block cluster rearrangement
                k = int(rng.integers(6, 14))
                cc = contact_counts(cur, cur_val, tol=0.01)
                top5 = np.argsort(-cc)[:5]
                seed = int(top5[rng.integers(0, 5)]) if rng.random() < 0.8 else int(rng.integers(0, n))
                ds = cur @ cur[seed]
                block = np.argsort(-ds)[:k]
                Xt = cur.copy()
                Xt[block] += 0.22 * rng.normal(size=(k, d))
                Xt[block] /= np.linalg.norm(Xt[block], axis=1, keepdims=True)
                Xt = smooth_refine(Xt, steps=550, T0=0.015, T1=0.001,
                                   lr0=0.30, lr1=0.015, mu=0.9, idx=block)
                v0 = max_cos(Xt)
                if v0 > cur_val + 0.08:
                    continue
                Xp = lp_polish(Xt, seconds=min(1.9, max(0.9, remain * 0.14)), radius_init=0.008)
                v = max_cos(Xp)
                if v < best_val - 1e-12:
                    x_best = Xp.copy(); best_val = v
                    cur = Xp.copy(); cur_val = v
                elif v < cur_val:
                    cur = Xp.copy(); cur_val = v
                else:
                    if rng.random() < 0.06 and v < cur_val + 0.008:
                        cur = Xp.copy(); cur_val = v
                    if rng.random() < 0.12:
                        cur = x_best.copy(); cur_val = best_val
            else:
                # global shake + smooth
                sig = float(rng.choice(np.array([0.02, 0.05, 0.12, 0.25])))
                Xt = cur + sig * rng.normal(size=(n, d))
                Xt /= np.linalg.norm(Xt, axis=1, keepdims=True)
                Xt = smooth_refine(Xt, steps=500, T0=0.014, T1=0.001,
                                   lr0=0.15, lr1=0.010, mu=0.9, idx=None)
                v0 = max_cos(Xt)
                if v0 > cur_val + 0.08:
                    continue
                Xp = lp_polish(Xt, seconds=min(1.6, max(0.8, remain * 0.13)), radius_init=0.006)
                v = max_cos(Xp)
                if v < best_val - 1e-12:
                    x_best = Xp.copy(); best_val = v
                    cur = Xp.copy(); cur_val = v
                elif v < cur_val:
                    cur = Xp.copy(); cur_val = v
                else:
                    if rng.random() < 0.05 and v < cur_val + 0.01:
                        cur = Xp.copy(); cur_val = v
                    if rng.random() < 0.12:
                        cur = x_best.copy(); cur_val = best_val
        except Exception:
            continue

    remain = BUDGET - (time.time() - t0)
    final_sec = float(np.clip(remain - 2.0, 5.0, 40.0))
    if final_sec > 1.0:
        try:
            x_best = lp_polish(x_best, seconds=final_sec * 0.6, radius_init=1e-3)
            rest = BUDGET - (time.time() - t0) - 1.0
            if rest > 2.0:
                x_best = lp_polish(x_best, seconds=min(rest, 18.0), radius_init=5e-4)
        except Exception:
            pass
    x_best /= np.linalg.norm(x_best, axis=1, keepdims=True)
    return x_best
