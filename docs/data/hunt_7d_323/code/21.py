import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(np.max(g))

def _worst(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return np.max(g, axis=1)

def _contacts(x, m, tol=1e-9):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return np.sum(g > m - tol, axis=1)

def _hole_reinsert(y, lst, rng, n_cand=4000):
    n, d = y.shape
    for k in lst:
        mask = np.ones(n, dtype=bool); mask[k] = False
        Xf = y[mask]
        cand = rng.standard_normal(size=(n_cand, d))
        cand /= np.linalg.norm(cand, axis=1, keepdims=True)
        w = np.max(cand @ Xf.T, axis=1)
        y[k] = cand[int(np.argmin(w))]
    # tiny local hill climb on reinserted
    for k in lst:
        oth = np.delete(y, k, axis=0)
        cur = y[k].copy()
        cw = float(np.max(oth @ cur))
        prop = cur[None,:] + rng.standard_normal(size=(200, d)) * 0.03
        prop /= np.linalg.norm(prop, axis=1, keepdims=True)
        dw = np.max(prop @ oth.T, axis=1)
        bi = int(np.argmin(dw))
        if dw[bi] < cw:
            y[k] = prop[bi]
    return y

def _anneal(x, steps, lr0=0.25, margin=0.02, cap=0.5):
    n, d = x.shape
    lr = float(lr0)
    for _ in range(steps):
        g = x @ x.T
        np.fill_diagonal(g, -2.0)
        m = float(np.max(g))
        th = m - margin
        C = g - th
        C[C < 0] = 0.0
        np.fill_diagonal(C, 0.0)
        Gr = C @ x
        dot = np.sum(Gr * x, axis=1, keepdims=True)
        Gt = Gr - dot * x
        nrm = np.linalg.norm(Gt, axis=1)
        mx = float(np.max(nrm)) if n > 0 else 0.0
        if mx > cap and mx > 0:
            Gt *= (cap / mx)
        x = x - lr * Gt
        x /= np.linalg.norm(x, axis=1, keepdims=True)
        lr *= 0.99
    return x

def _lp_polish(x, deadline, init_radius=6e-3, max_pairs=2000, max_it=60):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = float(init_radius)
    iuI, iuJ = np.triu_indices(n, 1)
    it = 0
    while it < max_it:
        it += 1
        if time.time() >= deadline - 0.25:
            break
        if radius < 1e-13:
            break
        G = x @ x.T
        c = G[iuI, iuJ]
        m = float(np.max(c))
        th = m - max(20.0 * radius, 1e-9)
        sel = c > th
        cnt = int(np.sum(sel))
        if cnt == 0:
            break
        if cnt > max_pairs:
            idx = np.where(sel)[0]
            cv = c[idx]
            part = np.argpartition(cv, -max_pairs)[-max_pairs:]
            sel = np.zeros_like(sel, dtype=bool)
            sel[idx[part]] = True
            cnt = max_pairs
        I = iuI[sel]; J = iuJ[sel]; cs = c[sel]
        p = cnt
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.empty(p * (2 * d + 1), dtype=np.int64)
        vals = np.empty(p * (2 * d + 1), dtype=np.float64)
        cr = cols.reshape(p, 2 * d + 1); vr = vals.reshape(p, 2 * d + 1)
        ar = np.arange(d)
        cr[:, :d] = I[:, None] * d + ar[None, :]
        cr[:, d:2*d] = J[:, None] * d + ar[None, :]
        cr[:, 2*d] = n * d
        vr[:, :d] = x[J]; vr[:, d:2*d] = x[I]; vr[:, 2*d] = -1.0
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n*d))), shape=(n, nv)).tocsr()
        except Exception:
            radius *= 0.5
            continue
        obj = np.zeros(nv); obj[-1] = 1.0
        tl = deadline - time.time() - 0.08
        if tl < 0.15:
            break
        tl = min(tl, 1.2)
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-cs, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)]*(n*d)+[(None, None)],
                          method="highs", options={"time_limit": tl})
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        yn = np.linalg.norm(y, axis=1, keepdims=True); yn[yn==0]=1.0
        y = y / yn
        v = _max_cos(y)
        if v < best:
            x = y; best = v
            radius = min(radius*1.4, 0.02)
        else:
            radius *= 0.5
    return x, best

def solve():
    t0 = time.time()
    HOP_END = t0 + 142.0
    FINAL_END = t0 + 174.0
    rng = np.random.default_rng(7)
    x0 = None; b0 = np.inf
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn).astype(np.float64)
            if a.shape == (323, 7):
                a = a / np.linalg.norm(a, axis=1, keepdims=True)
                v = _max_cos(a)
                if v < b0:
                    b0 = v; x0 = a.copy()
        except Exception:
            pass
    if x0 is None:
        a = rng.standard_normal(size=(323,7))
        a /= np.linalg.norm(a, axis=1, keepdims=True)
        x0 = a; b0 = _max_cos(a)
    x_best = x0.copy(); best = float(b0)
    n, d = x_best.shape
    it = 0; stall = 0
    while True:
        now = time.time()
        if now >= HOP_END - 1.0:
            break
        if HOP_END - now < 1.2:
            break
        s = it % 6
        x = x_best.copy()
        sm = 0
        if s == 0:
            sig = 0.015 + 0.002*min(stall,8)
            x = x + rng.standard_normal(size=x.shape)*sig
            x /= np.linalg.norm(x, axis=1, keepdims=True)
            budget = 2.2; init_r = 0.006
        elif s == 1:
            w = _worst(x)
            k = 30
            idx = np.argpartition(w, -k)[-k:]
            x[idx] += rng.standard_normal(size=(k,d))*0.08
            x[idx] /= np.linalg.norm(x[idx], axis=1, keepdims=True)
            sm = 35; budget = 2.5; init_r = 0.008
        elif s == 2:
            cc = _contacts(x, best)
            ratt = np.where(cc==0)[0].tolist()
            order = np.argsort(-cc)
            torep = [int(v) for v in ratt]
            for o in order:
                if len(torep) >= 5: break
                if int(o) not in torep: torep.append(int(o))
            torep = torep[:5]
            if len(torep)==0:
                torep=[int(v) for v in order[:5]]
            x = _hole_reinsert(x, torep, rng, n_cand=4000)
            x += rng.standard_normal(size=x.shape)*0.003
            x /= np.linalg.norm(x, axis=1, keepdims=True)
            sm = 60; budget = 2.8; init_r = 0.008
        elif s == 3:
            k = 20
            idx = rng.choice(n, size=k, replace=False)
            x[idx] += rng.standard_normal(size=(k,d))*0.07
            x[idx] /= np.linalg.norm(x[idx], axis=1, keepdims=True)
            budget = 2.2; init_r = 0.007
        elif s == 4:
            w = _worst(x)
            k = 10
            idx = np.argpartition(w,-k)[-k:].tolist()
            x = _hole_reinsert(x, [int(v) for v in idx], rng, n_cand=3500)
            x += rng.standard_normal(size=x.shape)*0.004
            x /= np.linalg.norm(x, axis=1, keepdims=True)
            sm = 60; budget = 2.8; init_r = 0.008
        else:
            cc = _contacts(x, best)
            k = 18
            idx = np.argsort(-cc)[:k]
            x[idx] += rng.standard_normal(size=(k,d))*0.11
            x[idx] /= np.linalg.norm(x[idx], axis=1, keepdims=True)
            sm = 30; budget = 2.5; init_r = 0.009
        budget = min(budget, HOP_END - time.time())
        if sm and budget > 0.6:
            try:
                x = _anneal(x, steps=sm, lr0=0.25, margin=0.02)
            except Exception:
                pass
        deadline = min(time.time()+budget, HOP_END)
        try:
            y, v = _lp_polish(x, deadline, init_radius=init_r)
        except Exception:
            it += 1; stall += 1
            continue
        if v < best - 1e-12:
            best = float(v); x_best = y; stall = 0
        else:
            stall += 1
        it += 1
    if FINAL_END - time.time() > 2.0:
        try:
            x_best, best = _lp_polish(x_best, FINAL_END, init_radius=1.2e-3, max_pairs=2000)
        except Exception:
            pass
    x_best = x_best / np.linalg.norm(x_best, axis=1, keepdims=True)
    return x_best
