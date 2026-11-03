import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(np.max(g))

def lp_polish(x, t_end, init_radius=0.01):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = max_cos(x)
    radius = init_radius
    iu0 = np.triu_indices(n, 1)
    iuI, iuJ = iu0
    while time.time() < t_end and radius > 1e-13:
        rem = t_end - time.time()
        if rem < 0.35:
            break
        G = x @ x.T
        c_all = G[iuI, iuJ]
        m = float(np.max(c_all))
        thresh = max(20.0 * radius, 1e-9)
        sel = c_all > m - thresh
        cnt = int(np.sum(sel))
        if cnt == 0:
            radius *= 0.5
            continue
        if cnt > 2500:
            kth = np.partition(c_all, -2500)[-2500]
            sel = c_all >= kth
            idxw = np.where(sel)[0]
            if len(idxw) > 2800:
                o = np.argsort(-c_all[idxw])[:2800]
                msk = np.zeros_like(sel)
                msk[idxw[o]] = True
                sel = msk
        I = iuI[sel]; J = iuJ[sel]; cs = c_all[sel]
        p = len(I)
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))),
                          shape=(n, nv)).tocsr()
        obj = np.zeros(nv); obj[-1] = 1.0
        bnds = [(-radius, radius)] * (n * d) + [(None, None)]
        try:
            tl = float(max(0.4, min(2.0, rem - 0.15)))
            res = linprog(obj, A_ub=A_ub, b_ub=-cs, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=bnds, method="highs", options={"time_limit": tl, "presolve": True})
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
            x, best = y, v
            radius = min(radius * 1.5, 0.04)
        else:
            radius *= 0.5
    return x

def smooth_rep(x, iters, p0, p1, lr):
    n, d = x.shape
    for it in range(iters):
        f = it / max(1, iters - 1)
        p = p0 + (p1 - p0) * f
        G = x @ x.T
        np.fill_diagonal(G, -1.0)
        Gp = np.where(G > 0, G, 0.0)
        m = float(np.max(Gp))
        if m <= 1e-12:
            break
        W = (Gp / m) ** (p - 1)
        np.fill_diagonal(W, 0.0)
        grad = W @ x
        dot = np.sum(grad * x, axis=1, keepdims=True)
        gt = grad - dot * x
        nr = float(np.mean(np.linalg.norm(gt, axis=1))) + 1e-12
        step = lr * (1.0 - 0.5 * f) / nr
        x = x - step * gt
        x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x

def deepest_hole(y, forbid, rng, nt=6000):
    n, d = y.shape
    R = rng.standard_normal((nt, d))
    R /= np.linalg.norm(R, axis=1, keepdims=True)
    S = R @ y.T
    if forbid is not None and len(forbid) > 0:
        S[:, list(forbid)] = -10.0
    sc = np.max(S, axis=1)
    return R[int(np.argmin(sc))]

def solve():
    t0 = time.time()
    BUDGET = 165.0
    t_end_all = t0 + BUDGET
    rng = np.random.default_rng(42)
    cand = []
    for fn in ["best.npy", "record.npy"]:
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=np.float64)
            if a.shape == (181, 9):
                a = a / np.linalg.norm(a, axis=1, keepdims=True)
                if np.all(np.isfinite(a)):
                    cand.append(a)
        except Exception:
            pass
    if not cand:
        z = rng.standard_normal((181, 9))
        z /= np.linalg.norm(z, axis=1, keepdims=True)
        cand.append(z)
    scores = [max_cos(a) for a in cand]
    bi = int(np.argmin(scores))
    xb = cand[bi].copy()
    bb = scores[bi]
    n, d = xb.shape
    # quick baseline polish
    if t_end_all - time.time() > 12:
        xb = lp_polish(xb, time.time() + 3.0, init_radius=0.002)
        bb = max_cos(xb)
    xc = xb.copy(); bc = bb
    T = 3e-5
    it = 0
    while time.time() < t_end_all - 3.0:
        it += 1
        rem = t_end_all - time.time()
        if rem < 3.0:
            break
        y = xc.copy()
        mode = it % 6
        G = y @ y.T
        np.fill_diagonal(G, -2.0)
        m0 = float(np.max(G))
        np.fill_diagonal(G, 1.0)
        contacts = np.sum(G > m0 - 1e-7, axis=1) - 1
        if mode == 0:
            y = y + rng.standard_normal(y.shape) * 0.008
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            init_r = 0.008; do_sm = False
        elif mode == 1:
            order = np.argsort(contacts)
            nr = 3
            for r in order[:nr]:
                try:
                    y[int(r)] = deepest_hole(y, [int(r)], rng, nt=6000)
                except Exception:
                    pass
            y = y + rng.standard_normal(y.shape) * 0.004
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            init_r = 0.015; do_sm = True
        elif mode == 2:
            top = int(np.argmax(contacts))
            sims = G[top].copy(); sims[top] = -10
            nn = np.argsort(-sims)[:8]
            grp = np.concatenate([[top], nn])
            y[grp] += rng.standard_normal((len(grp), d)) * 0.10
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            init_r = 0.022; do_sm = True
        elif mode == 3:
            k = 2
            idx = rng.choice(n, size=k, replace=False)
            R = rng.standard_normal((k, d))
            R /= np.linalg.norm(R, axis=1, keepdims=True)
            y[idx] = R
            init_r = 0.028; do_sm = True
        elif mode == 4:
            iu = np.triu_indices(n, 1)
            np.fill_diagonal(G, -2.0)
            c = G[iu]
            K = 40
            o = np.argsort(-c)[:K]
            pick = o[rng.integers(0, K)]
            i0 = int(iu[0][pick]); j0 = int(iu[1][pick])
            diff = y[i0] - y[j0]
            diff /= (np.linalg.norm(diff) + 1e-12)
            y[i0] = y[i0] + diff * 0.05 + rng.standard_normal(d) * 0.015
            y[j0] = y[j0] - diff * 0.05 + rng.standard_normal(d) * 0.015
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            init_r = 0.014; do_sm = True
        else:
            y = y + rng.standard_normal(y.shape) * 0.06
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            init_r = 0.025; do_sm = True
        if do_sm:
            if time.time() > t_end_all - 2.5:
                break
            y = smooth_rep(y, iters=220, p0=25, p1=70, lr=0.04)
        rem2 = t_end_all - time.time()
        if rem2 < 2.0:
            break
        budg = min(2.6, rem2 - 1.0)
        if budg < 0.8:
            budg = max(0.6, rem2 - 0.6)
        y2 = lp_polish(y, time.time() + float(budg), init_radius=init_r)
        v = max_cos(y2)
        if v < bb - 1e-15:
            xb = y2.copy(); bb = v
            xc = y2.copy(); bc = v
            rem3 = t_end_all - time.time()
            if rem3 > 9:
                xb2 = lp_polish(xb, time.time() + 1.2, init_radius=0.001)
                vv = max_cos(xb2)
                if vv < bb:
                    xb = xb2; bb = vv; xc = xb2.copy(); bc = vv
        else:
            if v < bc or rng.random() < np.exp((bc - v) / max(T, 1e-9)):
                xc = y2.copy(); bc = v
        T *= 0.9993
        if T < 7e-06:
            T = 7e-06
    remf = t_end_all - time.time()
    if remf > 1.5:
        try:
            yf = lp_polish(xb, time.time() + min(remf - 0.4, 4.0), init_radius=0.001)
            vf = max_cos(yf)
            if vf < bb:
                xb = yf; bb = vf
        except Exception:
            pass
    xb = xb / np.linalg.norm(xb, axis=1, keepdims=True)
    return xb
