import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _per_point_max(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return G.max(axis=1), G

def _adam_rowwise(X, steps, k0, k1, lr0, lr1, deadline=None, b1=0.9, b2=0.999, eps=1e-8):
    n, d = X.shape
    m1 = np.zeros_like(X)
    v1 = np.zeros_like(X)
    for s in range(steps):
        if deadline is not None and (s % 100 == 0) and time.time() > deadline:
            break
        frac = s / max(1, steps - 1)
        k = k0 * (k1 / k0) ** frac if k0 != k1 else k0
        lr = lr0 * (lr1 / lr0) ** frac if lr0 != lr1 else lr0
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        mr = G.max(axis=1, keepdims=True)
        # shift for stability, cap exponent
        D = k * (G - mr)
        # clip to avoid overflow
        np.clip(D, -50.0, 0.0, out=D)
        W = np.exp(D)
        np.fill_diagonal(W, 0.0)
        rs = W.sum(axis=1, keepdims=True) + 1e-30
        Grad = (W @ X) / rs
        dot = (Grad * X).sum(axis=1, keepdims=True)
        Gt = Grad - dot * X
        m1 = b1 * m1 + (1 - b1) * Gt
        v1 = b2 * v1 + (1 - b2) * (Gt * Gt)
        bc1 = 1.0 - b1 ** (s + 1)
        bc2 = 1.0 - b2 ** (s + 1)
        mhat = m1 / bc1
        vhat = v1 / bc2
        upd = lr * mhat / (np.sqrt(vhat) + eps)
        # keep update tangential
        ud = (upd * X).sum(axis=1, keepdims=True)
        upd = upd - ud * X
        X = X - upd
        nr = np.linalg.norm(X, axis=1, keepdims=True)
        nr[nr == 0] = 1.0
        X = X / nr
        if not np.all(np.isfinite(X)):
            break
    return X

def _void_point(others, rng, n_cand=8000, k=60.0, steps=120, lr=0.2):
    d = others.shape[1]
    cand = rng.standard_normal(size=(n_cand, d))
    cand /= np.linalg.norm(cand, axis=1, keepdims=True)
    sc = (cand @ others.T).max(axis=1)
    c = cand[np.argmin(sc)].copy()
    # tangential soft-max descent on c alone
    for it in range(steps):
        dots = others @ c
        m = dots.max()
        DD = k * (dots - m)
        np.clip(DD, -50.0, 0.0, out=DD)
        w = np.exp(DD)
        s = w.sum()
        if s == 0 or not np.finite(s):
            break
        w /= s
        grad = w @ others
        gt = grad - (grad @ c) * c
        st = lr * (1.0 - 0.5 * it / steps)
        c = c - st * gt
        nrm = np.linalg.norm(c)
        if nrm == 0 or not np.finite(nrm):
            break
        c /= nrm
    return c

def _lp_polish(x, deadline):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    iu = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        cmat = (x @ x.T)[iu]
        m = cmat.max()
        sel = cmat > m - max(20 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p, nv = len(I), n * d + 1
        if p == 0:
            break
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        except Exception:
            break
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-cmat[sel], A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        v = _max_cos(y)
        if v < best:
            x, best, radius = y, v, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def solve():
    T0 = time.time()
    BUDGET = 176.0
    HOP_END = T0 + 128.0
    rng = np.random.default_rng(0)
    # load start
    best_val = 1e9
    best = None
    for fn in ["best.npy", "record.npy"]:
        try:
            a = np.load(fn).astype(float)
            if a.shape == (94, 13):
                a = a / np.linalg.norm(a, axis=1, keepdims=True)
                if np.all(np.isfinite(a)):
                    v = _max_cos(a)
                    if v < best_val:
                        best_val = v
                        best = a.copy()
        except Exception:
            pass
    if best is None:
        best = rng.standard_normal(size=(94, 13))
        best /= np.linalg.norm(best, axis=1, keepdims=True)
        best_val = _max_cos(best)
    n, d = best.shape
    cur = best.copy()
    cur_val = best_val
    it = 0
    sigmas = [0.004, 0.01, 0.025, 0.06, 0.12, 0.22]
    while time.time() < HOP_END:
        it += 1
        elapsed = time.time() - T0
        prog = elapsed / max(1e-9, (HOP_END - T0))
        Temp = 0.003 * (1 - prog) + 0.00015 * prog
        # restart source
        if it % 12 == 0:
            base = best.copy()
            base_val = best_val
        else:
            base = cur.copy()
            base_val = cur_val
        y = base.copy()
        r = rng.random()
        if r < 0.45:
            # global shake + refine
            sig = sigmas[rng.integers(0, len(sigmas))]
            y = y + sig * rng.standard_normal(size=y.shape)
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            k0 = float(rng.choice([12.0, 20.0, 30.0]))
            k1 = float(rng.choice([60.0, 90.0, 140.0]))
            lr0 = float(rng.choice([0.01, 0.02, 0.03]))
            lr1 = lr0 * float(rng.choice([0.3, 0.5, 1.0]))
            steps = int(rng.integers(250, 550))
            y = _adam_rowwise(y, steps, k0, k1, lr0, lr1, deadline=HOP_END)
        elif r < 0.70:
            # subset shake of worst / random points
            per, G = _per_point_max(y)
            order = np.argsort(-per)
            k = int(rng.integers(2, 9))
            victims = set()
            for j in range(k):
                if rng.random() < 0.7:
                    victims.add(int(order[rng.integers(0, 12)]))
                else:
                    victims.add(int(rng.integers(0, n)))
            victims = list(victims)
            sig = float(rng.choice([0.1, 0.25, 0.45]))
            y[victims] += sig * rng.standard_normal(size=(len(victims), d))
            y[victims] /= np.linalg.norm(y[victims], axis=1, keepdims=True)
            k0 = float(rng.choice([15.0, 25.0]))
            k1 = float(rng.choice([80.0, 130.0]))
            lr0 = 0.02
            steps = int(rng.integers(250, 500))
            y = _adam_rowwise(y, steps, k0, k1, lr0, lr0 * 0.5, deadline=HOP_END)
        else:
            # kill worst 1-3 and reinsert at voids
            per, G = _per_point_max(y)
            order = np.argsort(-per)
            m = int(rng.integers(1, 4))
            victims = [int(order[j]) for j in range(m)]
            # occasionally add random victim
            if rng.random() < 0.3:
                victims.append(int(rng.integers(0, n)))
                victims = list(set(victims))
            for v in victims:
                mask = np.ones(n, dtype=bool); mask[v] = False
                others = y[mask]
                c = _void_point(others, rng, n_cand=8000, k=float(rng.choice([40.0, 60.0, 90.0])), steps=100)
                y[v] = c
            # short joint relax, sharper
            k0 = 25.0; k1 = float(rng.choice([90.0, 150.0]))
            y = _adam_rowwise(y, int(rng.integers(200, 450)), k0, k1, 0.015, 0.007, deadline=HOP_END)
        if not np.all(np.isfinite(y)):
            continue
        # normalize + eval
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        v = _max_cos(y)
        if v < best_val:
            best_val = v
            best = y.copy()
            cur = y.copy(); cur_val = v
        elif v < cur_val:
            cur = y.copy(); cur_val = v
        else:
            if Temp > 0 and rng.random() < np.exp(-(v - cur_val) / Temp):
                cur = y.copy(); cur_val = v
            # occasional random restart from best with big kick if stuck
            if it % 40 == 0 and rng.random() < 0.25:
                cur = best.copy(); cur_val = best_val
        if time.time() > HOP_END:
            break
    # final sharpen + LP polish on best
    remain = T0 + BUDGET - time.time()
    if remain > 8:
        # high-k adam sharpen before LP
        try:
            best = _adam_rowwise(best, 400, 60.0, 180.0, 0.008, 0.003, deadline=T0 + BUDGET - 35.0)
            bv = _max_cos(best / np.linalg.norm(best, axis=1, keepdims=True))
            best = best / np.linalg.norm(best, axis=1, keepdims=True)
            best_val = bv
        except Exception:
            pass
    try:
        best = _lp_polish(best, T0 + BUDGET - 1.0)
    except Exception:
        pass
    best = best / np.linalg.norm(best, axis=1, keepdims=True)
    return best
