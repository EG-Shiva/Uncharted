import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())


def per_point_max(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return g.max(axis=1)


def gd_refine(X, steps, T0, T1, lr0, lr1, mu=0.9):
    n, d = X.shape
    V = np.zeros_like(X)
    lT0 = np.log(T0)
    lT1 = np.log(T1)
    llr0 = np.log(lr0)
    llr1 = np.log(lr1)
    for s in range(steps):
        f = s / max(1, steps - 1)
        T = float(np.exp(lT0 + (lT1 - lT0) * f))
        lr = float(np.exp(llr0 + (llr1 - llr0) * f))
        G = X @ X.T
        np.fill_diagonal(G, -10.0)
        mx = float(G.max())
        W = np.exp((G - mx) / T)
        np.fill_diagonal(W, 0.0)
        ss = float(W.sum())
        if not np.isfinite(ss) or ss == 0:
            break
        W /= ss
        Grad = W @ X
        dot = np.sum(Grad * X, axis=1, keepdims=True)
        Gt = Grad - dot * X
        # clip huge steps
        V = mu * V + Gt
        X = X - lr * V
        X /= np.linalg.norm(X, axis=1, keepdims=True)
    return X


def find_hole(others, rng, n_sample=6000, refine=120):
    # others: (m,d)
    m, d = others.shape
    C = rng.standard_normal((n_sample, d))
    C /= np.linalg.norm(C, axis=1, keepdims=True)
    dots = C @ others.T
    md = dots.max(axis=1)
    bi = np.argpartition(md, 3)[:3]
    best_c = C[int(bi[0])].copy()
    best_m = float(md[int(bi[0])])
    for b in bi:
        c = C[int(b)].copy()
        for s in range(refine):
            dv = others @ c
            j = int(np.argmax(dv))
            mv = float(dv[j])
            if mv < best_m:
                best_m = mv
                best_c = c.copy()
            g = others[j] - mv * c
            step = 0.25 / (1.0 + s * 0.06)
            c = c - step * g
            nrm = np.linalg.norm(c)
            if nrm == 0 or not np.isfinite(nrm):
                break
            c /= nrm
    return best_c


def lp_polish(x, seconds, deadline=None):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = max_cos(x)
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    ar = np.arange(d)
    while time.time() - start < seconds:
        if deadline is not None and time.time() > deadline:
            break
        if radius < 1e-14:
            break
        if seconds - (time.time() - start) < 0.05:
            break
        c = (x @ x.T)[iu]
        m = float(c.max())
        sel = c > m - max(20 * radius, 1e-9)
        if int(sel.sum()) > 2000:
            idx = np.argpartition(c, -2000)[-2000:]
            mask = np.zeros_like(sel, dtype=bool)
            mask[idx] = True
            sel = mask
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        if p == 0:
            break
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + ar, J[:, None] * d + ar, np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv)
        obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c[sel], A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
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


def solve():
    t0 = time.time()
    HARD_END = t0 + 178.0
    EXPL_END = t0 + 138.0
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
        xb = xb / np.linalg.norm(xb, axis=1, keepdims=True)
        cands.append(xb)
    if xr is not None and xr.shape == (95, 13):
        xr = xr / np.linalg.norm(xr, axis=1, keepdims=True)
        cands.append(xr)
    if not cands:
        z = rng.standard_normal((95, 13))
        z /= np.linalg.norm(z, axis=1, keepdims=True)
        cands.append(z)
    vals = [max_cos(c) for c in cands]
    bi = int(np.argmin(vals))
    best = cands[bi].copy()
    best_val = float(vals[bi])
    cur = best.copy()
    cur_val = best_val
    n, d = best.shape

    sigmas = [0.005, 0.01, 0.02, 0.05, 0.10, 0.20, 0.35, 0.015, 0.06]

    it = 0
    Tb0 = 0.003
    while time.time() < EXPL_END and time.time() < HARD_END - 35:
        it += 1
        frac = (time.time() - t0) / max(1e-9, (EXPL_END - t0))
        Tb = Tb0 * (1.0 - 0.9 * min(1.0, max(0.0, frac)))
        Tb = max(Tb, 2e-4)
        u = rng.random()
        if u < 0.14 and it > 4:
            # fresh random restart + long anneal
            Xt = rng.standard_normal((n, d))
            Xt /= np.linalg.norm(Xt, axis=1, keepdims=True)
            Xt = gd_refine(Xt, 1100, 0.020, 0.0009, 0.18, 0.015, mu=0.9)
        elif u < 0.55:
            sigma = sigmas[int(rng.integers(0, len(sigmas)))]
            Xt = cur.copy()
            if rng.random() < 0.30:
                ksub = int(rng.integers(5, 16))
                idx = rng.choice(n, size=ksub, replace=False)
                Xt[idx] += sigma * 2.0 * rng.standard_normal((ksub, d))
            else:
                Xt += sigma * rng.standard_normal((n, d))
            Xt /= np.linalg.norm(Xt, axis=1, keepdims=True)
            # anneal params vary
            if sigma > 0.12:
                Xt = gd_refine(Xt, 700, 0.018, 0.0010, 0.16, 0.012, mu=0.9)
            elif sigma > 0.03:
                Xt = gd_refine(Xt, 550, 0.012, 0.0009, 0.12, 0.010, mu=0.9)
            else:
                Xt = gd_refine(Xt, 400, 0.008, 0.0008, 0.10, 0.010, mu=0.9)
        else:
            # hole re-insertion of worst points
            rr = rng.random()
            if rr < 0.40:
                k = 1
            elif rr < 0.70:
                k = 2
            else:
                k = int(rng.integers(3, 6))
            ppm = per_point_max(cur)
            order = np.argsort(ppm)
            idx = []
            for _ in range(k):
                if rng.random() < 0.70:
                    pool = [int(v) for v in order[-12:] if int(v) not in idx]
                    if not pool:
                        pool = [i for i in range(n) if i not in idx]
                    idx.append(pool[int(rng.integers(0, len(pool)))])
                else:
                    cand = int(rng.integers(0, n))
                    while cand in idx:
                        cand = int(rng.integers(0, n))
                    idx.append(cand)
            Xt = cur.copy()
            ns = 6000 if k <= 2 else 4000
            for j in idx:
                others = np.delete(Xt, j, axis=0)
                h = find_hole(others, rng, n_sample=ns, refine=110)
                Xt[j] = h
            Xt += 0.004 * rng.standard_normal((n, d))
            Xt /= np.linalg.norm(Xt, axis=1, keepdims=True)
            Xt = gd_refine(Xt, 450, 0.010, 0.0008, 0.12, 0.010, mu=0.9)

        tv = max_cos(Xt)
        if tv < best_val:
            best_val = tv
            best = Xt.copy()
        if tv < cur_val or rng.random() < np.exp((cur_val - tv) / Tb):
            cur = Xt.copy()
            cur_val = tv
        # periodically restart walker from best
        if it % 60 == 0:
            cur = best.copy()
            cur_val = best_val
        if best_val < 0.239:
            break

    # final polish of best with remaining budget
    remain = HARD_END - time.time() - 1.0
    if remain > 3:
        # short GD squeeze before LP to clean
        # use best as start for LP
        lp_time = max(5.0, remain * 0.85)
        best = lp_polish(best, lp_time, deadline=HARD_END - 2.0)
        best_val = max_cos(best)
        # tiny final anneal squeeze if time left
        left = HARD_END - time.time() - 0.5
        if left > 2.0:
            tmp = gd_refine(best.copy(), 300, 0.002, 0.0004, 0.03, 0.005, mu=0.9)
            tv = max_cos(tmp)
            if tv < best_val:
                best = tmp
                best_val = tv
                # re-LP very briefly
                l2 = HARD_END - time.time() - 0.5
                if l2 > 2.0:
                    best = lp_polish(best, min(l2, 6.0), deadline=HARD_END - 0.3)

    best = best / np.linalg.norm(best, axis=1, keepdims=True)
    return best
