import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix, csr_matrix, hstack


def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(np.max(g))


def anneal(y, steps, beta0, beta1, s0, s1):
    n, d = y.shape
    lb0 = np.log(beta0) if beta0 > 0 else 0.0
    lb1 = np.log(beta1) if beta1 > 0 else 0.0
    ls0 = np.log(s0) if s0 > 0 else 0.0
    ls1 = np.log(s1) if s1 > 0 else 0.0
    for s in range(steps):
        frac = s / max(1, steps - 1)
        beta = float(np.exp(lb0 + (lb1 - lb0) * frac))
        step = float(np.exp(ls0 + (ls1 - ls0) * frac))
        G = y @ y.T
        np.fill_diagonal(G, -np.inf)
        m = float(np.max(G))
        E = np.exp(beta * (G - m))
        np.fill_diagonal(E, 0.0)
        grad = E @ y
        dot = np.sum(grad * y, axis=1, keepdims=True)
        gt = grad - dot * y
        nrm = np.linalg.norm(gt, axis=1, keepdims=True) + 1e-12
        y = y - step * (gt / nrm)
        y = y / (np.linalg.norm(y, axis=1, keepdims=True) + 1e-30)
    return y


def lp_polish(x, seconds, deadline=1e30):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    try:
        best = max_cos(x)
    except Exception:
        return x
    radius = 1e-3
    start = time.time()
    iu0 = np.triu_indices(n, 1)
    ar_d = np.arange(d)
    while time.time() - start < seconds and time.time() < deadline:
        if radius < 1e-14:
            break
        G = x @ x.T
        c = G[iu0]
        m = float(np.max(c))
        thresh = m - max(20.0 * radius, 1e-9)
        sel = c > thresh
        cnt = int(np.sum(sel))
        if cnt == 0:
            break
        if cnt > 800:
            K = 800
            part = np.argpartition(-c, K)[:K]
            thr2 = float(np.min(c[part]))
            sel = c >= thr2
            # if ties blow up, truncate
            idx = np.flatnonzero(sel)
            if len(idx) > 900:
                # keep top 900
                o = np.argsort(-c[idx])[:900]
                idx = idx[o]
                sel = np.zeros_like(sel)
                sel[idx] = True
        I = iu0[0][sel]
        J = iu0[1][sel]
        p = len(I)
        b = -c[sel]
        nv = n * d + 1
        try:
            # sparse M (p x n*d): each row has xJ at I-block and xI at J-block
            rows = np.repeat(np.arange(p), 2 * d)
            cI = (I[:, None] * d + ar_d[None, :]).ravel()
            cJ = (J[:, None] * d + ar_d[None, :]).ravel()
            cols = np.concatenate([cI, cJ])
            # vals interleaved per row: need order matching rows
            vI = x[J].ravel()  # p*d in row-major: row k -> x[J[k]]
            vJ = x[I].ravel()
            # rows currently [row0*d..., row0*d..., row1...] if repeat grouped?
            # repeat gives row0 x2d, row1 x2d...; cols above is [all I-blocks then all J-blocks] -> mismatch.
            # Fix: interleave
            vals_I = x[J].reshape(p, d)
            vals_J = x[I].reshape(p, d)
            V = np.concatenate([vals_I, vals_J], axis=1).ravel()
            M = coo_matrix((V, (rows, cols)), shape=(n * d, p)).tocsc() if False else coo_matrix((V, (rows, cols)), shape=(p, n * d)).tocsr()
            # reorder cols to match rows: rows repeat 2d, cols must be row-major [Iblock,Jblock] per row
            # Build cols row-major directly:
            cols_rm = np.empty(p * 2 * d, dtype=int)
            cols_rm.reshape(p, 2 * d)[:, :d] = I[:, None] * d + ar_d[None, :]
            cols_rm.reshape(p, 2 * d)[:, d:] = J[:, None] * d + ar_d[None, :]
            M = coo_matrix((V, (rows, cols_rm)), shape=(p, n * d)).tocsr()
            A_ub = hstack([M, csr_matrix((-np.ones((p, 1))))], format="csr")
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv)
            obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=b, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                          method="highs", options={"time_limit": 2.0})
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        dx = res.x[:-1].reshape(n, d)
        y = x + dx
        y = y / (np.linalg.norm(y, axis=1, keepdims=True) + 1e-30)
        v = max_cos(y)
        if v < best:
            best = v
            x = y
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x


def best_hole(others, rng, n_samples=20000):
    d = others.shape[1]
    cand = rng.standard_normal((n_samples, d))
    cand /= (np.linalg.norm(cand, axis=1, keepdims=True) + 1e-30)
    dots = cand @ others.T
    vals = np.max(dots, axis=1)
    i = int(np.argmin(vals))
    return cand[i]


def refine_hole(h, others, steps=80, beta=400.0, step=0.01):
    for _ in range(steps):
        dd = others @ h
        mm = float(np.max(dd))
        w = np.exp(beta * (dd - mm))
        g = w @ others
        g = g - float(g @ h) * h
        n = float(np.linalg.norm(g)) + 1e-12
        h = h - step * g / n
        h = h / (float(np.linalg.norm(h)) + 1e-30)
    return h


def solve():
    t0 = time.time()
    hard_deadline = t0 + 170.0
    explore_end = t0 + 132.0
    rng = np.random.default_rng(20261101)
    try:
        x0 = np.load("best.npy").astype(float)
    except Exception:
        try:
            x0 = np.load("record.npy").astype(float)
        except Exception:
            x0 = rng.standard_normal((54, 21))
    n, d = 54, 21
    if x0.shape != (n, d):
        tmp = rng.standard_normal((n, d))
        m0 = min(x0.shape[0], n)
        m1 = min(x0.shape[1], d)
        tmp[:m0, :m1] = x0[:m0, :m1]
        x0 = tmp
    x0 = x0 / (np.linalg.norm(x0, axis=1, keepdims=True) + 1e-30)
    best = x0.copy()
    best_val = max_cos(best)
    no_improve = 0
    it = 0
    iu = np.triu_indices(n, 1)
    while time.time() < explore_end:
        it += 1
        mode = it % 8
        y = best.copy()
        if mode == 0:
            sigma = 0.007 if no_improve < 15 else 0.016
            y = y + sigma * rng.standard_normal(y.shape)
            y /= (np.linalg.norm(y, axis=1, keepdims=True) + 1e-30)
            y = anneal(y, steps=300, beta0=120.0, beta1=700.0, s0=0.004, s1=0.0004)
        elif mode == 1:
            sigma = 0.018 if no_improve < 15 else 0.035
            y = y + sigma * rng.standard_normal(y.shape)
            y /= (np.linalg.norm(y, axis=1, keepdims=True) + 1e-30)
            y = anneal(y, steps=350, beta0=150.0, beta1=900.0, s0=0.005, s1=0.0004)
        elif mode == 2:
            G = y @ y.T
            np.fill_diagonal(G, -2.0)
            pm = np.max(G, axis=1)
            k = 6
            idx = np.argsort(-pm)[:k]
            sigma = 0.05 if no_improve < 20 else 0.10
            y[idx] += sigma * rng.standard_normal((k, d))
            y /= (np.linalg.norm(y, axis=1, keepdims=True) + 1e-30)
            y = anneal(y, steps=300, beta0=120.0, beta1=800.0, s0=0.004, s1=0.0004)
        elif mode == 3:
            G = y @ y.T
            np.fill_diagonal(G, -2.0)
            pm = np.max(G, axis=1)
            w = int(np.argmax(pm))
            others = np.delete(y, w, axis=0)
            h = best_hole(others, rng, n_samples=25000)
            h = refine_hole(h, others, steps=100, beta=400.0, step=0.01)
            y[w] = h
            y = y + 0.002 * rng.standard_normal(y.shape)
            y /= (np.linalg.norm(y, axis=1, keepdims=True) + 1e-30)
            y = anneal(y, steps=280, beta0=150.0, beta1=850.0, s0=0.003, s1=0.0003)
        elif mode == 4:
            k = 3 if (it % 16 != 0) else 5
            idx = rng.choice(n, size=k, replace=False)
            y[idx] = rng.standard_normal((k, d))
            y /= (np.linalg.norm(y, axis=1, keepdims=True) + 1e-30)
            y = anneal(y, steps=400, beta0=100.0, beta1=800.0, s0=0.006, s1=0.0005)
        elif mode == 5:
            G = y @ y.T
            c = G[iu]
            K = 12
            top = np.argsort(-c)[:K]
            s = 0.012
            for t in top:
                i = int(iu[0][t]); j = int(iu[1][t])
                dij = float(y[i] @ y[j])
                vi = y[i] - dij * y[j]
                vj = y[j] - dij * y[i]
                ni = float(np.linalg.norm(vi)) + 1e-12
                nj = float(np.linalg.norm(vj)) + 1e-12
                y[i] = y[i] + s * vi / ni
                y[j] = y[j] + s * vj / nj
            y /= (np.linalg.norm(y, axis=1, keepdims=True) + 1e-30)
            y = anneal(y, steps=280, beta0=200.0, beta1=900.0, s0=0.003, s1=0.0003)
        elif mode == 6:
            k = 8
            idx = rng.choice(n, size=k, replace=False)
            y[idx] += 0.03 * rng.standard_normal((k, d))
            y /= (np.linalg.norm(y, axis=1, keepdims=True) + 1e-30)
            y = anneal(y, steps=300, beta0=120.0, beta1=750.0, s0=0.004, s1=0.0004)
        else:
            # double hole re-insertion
            G = y @ y.T
            np.fill_diagonal(G, -2.0)
            pm = np.max(G, axis=1)
            ws = np.argsort(-pm)[:2]
            for w in ws:
                others = np.delete(y, int(w), axis=0)
                h = best_hole(others, rng, n_samples=15000)
                h = refine_hole(h, others, steps=60, beta=350.0, step=0.012)
                y[int(w)] = h
            y = y + 0.002 * rng.standard_normal(y.shape)
            y /= (np.linalg.norm(y, axis=1, keepdims=True) + 1e-30)
            y = anneal(y, steps=320, beta0=130.0, beta1=800.0, s0=0.004, s1=0.0004)
        vs = max_cos(y)
        # short LP only if promising to save time
        if vs < best_val + 0.008:
            budget = 0.9
            remain = explore_end - time.time()
            if remain < 1.0:
                break
            budget = min(budget, remain - 0.1)
            if budget > 0.25:
                y = lp_polish(y, seconds=budget, deadline=explore_end)
        v = max_cos(y)
        if v < best_val - 1e-12:
            best_val = v
            best = y.copy()
            no_improve = 0
        else:
            no_improve += 1
        # safety: if very little time left, break
        if time.time() > explore_end:
            break
    # final squeeze of the champion
    fb = hard_deadline - time.time() - 1.0
    fb = min(max(0.0, fb), 30.0)
    if fb > 0.6:
        best = lp_polish(best, seconds=fb, deadline=hard_deadline)
    best = np.nan_to_num(best, nan=0.0, posinf=0.0, neginf=0.0)
    nn = np.linalg.norm(best, axis=1, keepdims=True)
    nn[nn == 0] = 1.0
    best = best / nn
    return best
