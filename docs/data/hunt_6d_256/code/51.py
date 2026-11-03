import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _adam_lse(x0, steps, K, lr, eval_every=200):
    x = x0.copy()
    n, d = x.shape
    m = np.zeros_like(x)
    v = np.zeros_like(x)
    b1, b2, eps = 0.9, 0.999, 1e-8
    b1p, b2p = 1.0, 1.0
    best_val = _max_cos(x)
    best_x = x.copy()
    for t in range(1, steps + 1):
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        gmax = float(G.max())
        # LSE weights
        W = np.exp(K * (G - gmax))
        np.fill_diagonal(W, 0.0)
        S = float(W.sum() * 0.5)
        if not np.isfinite(S) or S <= 0:
            continue
        grad = (W @ x) / S
        dot = np.sum(grad * x, axis=1, keepdims=True)
        gt = grad - dot * x
        b1p *= b1
        b2p *= b2
        m *= b1
        m += (1 - b1) * gt
        v *= b2
        v += (1 - b2) * (gt * gt)
        mhat = m / (1 - b1p)
        vhat = v / (1 - b2p)
        step = lr * mhat / (np.sqrt(vhat) + eps)
        sdot = np.sum(step * x, axis=1, keepdims=True)
        step = step - sdot * x
        x = x - step
        nr = np.linalg.norm(x, axis=1, keepdims=True)
        nr[nr == 0] = 1.0
        x = x / nr
        if (t % eval_every == 0) or (t == steps):
            val = _max_cos(x)
            if val < best_val:
                best_val = val
                best_x = x.copy()
    return best_x, best_val

def _adam_topk(x0, steps, topk, lr, eval_every=50):
    x = x0.copy()
    n, d = x.shape
    iuI, iuJ = np.triu_indices(n, 1)
    m = np.zeros_like(x)
    v = np.zeros_like(x)
    b1, b2, eps = 0.9, 0.999, 1e-8
    b1p, b2p = 1.0, 1.0
    best_val = _max_cos(x)
    best_x = x.copy()
    K = int(min(topk, len(iuI)))
    for t in range(1, steps + 1):
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        c = G[iuI, iuJ]
        # select top-K pairs uniformly weighted
        if K < len(c):
            part = np.argpartition(-c, K)[:K]
            I = iuI[part]
            J = iuJ[part]
        else:
            I = iuI
            J = iuJ
        W = np.zeros((n, n), dtype=float)
        W[I, J] = 1.0
        W[J, I] = 1.0
        S = float(K)
        grad = (W @ x) / S
        dot = np.sum(grad * x, axis=1, keepdims=True)
        gt = grad - dot * x
        b1p *= b1
        b2p *= b2
        m *= b1
        m += (1 - b1) * gt
        v *= b2
        v += (1 - b2) * (gt * gt)
        mhat = m / (1 - b1p)
        vhat = v / (1 - b2p)
        step = lr * mhat / (np.sqrt(vhat) + eps)
        sdot = np.sum(step * x, axis=1, keepdims=True)
        step = step - sdot * x
        x = x - step
        nr = np.linalg.norm(x, axis=1, keepdims=True)
        nr[nr == 0] = 1.0
        x = x / nr
        if (t % eval_every == 0) or (t == steps):
            val = _max_cos(x)
            if val < best_val:
                best_val = val
                best_x = x.copy()
    return best_x, best_val

def _tangent_bases(x):
    n, d = x.shape
    dd = d - 1
    Q = np.empty((n, d, dd))
    amax = np.argmax(np.abs(x), axis=1)
    E = np.eye(d)
    for i in range(n):
        k = int(amax[i])
        xi = x[i]
        cols = [l for l in range(d) if l != k]
        F = np.empty((d, dd))
        for ci, l in enumerate(cols):
            e = E[l]
            F[:, ci] = e - xi[l] * xi
        Qi, _ = np.linalg.qr(F)
        Q[i] = Qi
    return Q

def _lp_polish(x0, best_val, deadline):
    x = x0.copy()
    cur_max = best_val
    n, d = x.shape
    dd = d - 1
    NV = n * dd
    iuI, iuJ = np.triu_indices(n, 1)
    obj = np.zeros(NV + 1)
    obj[-1] = 1.0
    radii = [5e-4, 2e-4, 1e-4, 5e-5, 2e-4, 1e-4]
    ridx = 0
    while time.time() < deadline:
        rem = deadline - time.time()
        if rem < 2.5:
            break
        r = radii[ridx % len(radii)]
        ridx += 1
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        c_all = G[iuI, iuJ]
        thr = m - max(25.0 * r, 1.2e-4)
        sel = c_all > thr
        I = iuI[sel]
        J = iuJ[sel]
        csel = c_all[sel]
        p = len(I)
        if p < 200:
            k = min(800, len(c_all))
            part = np.argpartition(-c_all, k)[:k]
            I = iuI[part]; J = iuJ[part]; csel = c_all[part]
            p = k
        if p > 2200:
            part = np.argpartition(-csel, 2200)[:2200]
            I = I[part]; J = J[part]; csel = csel[part]
            p = 2200
        try:
            Q = _tangent_bases(x)
        except Exception:
            continue
        xI = x[I]
        xJ = x[J]
        QI = Q[I]
        QJ = Q[J]
        AI = np.sum(QI * xJ[:, :, None], axis=1)
        AJ = np.sum(QJ * xI[:, :, None], axis=1)
        rows = np.repeat(np.arange(p), 2 * dd + 1)
        cols = np.empty(p * (2 * dd + 1), dtype=int)
        vals = np.empty(p * (2 * dd + 1), dtype=float)
        for k in range(p):
            base = k * (2 * dd + 1)
            vals[base:base + dd] = AI[k]
            cols[base:base + dd] = int(I[k]) * dd + np.arange(dd)
            vals[base + dd:base + 2 * dd] = AJ[k]
            cols[base + dd:base + 2 * dd] = int(J[k]) * dd + np.arange(dd)
            vals[base + 2 * dd] = -1.0
            cols[base + 2 * dd] = NV
        bub = -csel
        try:
            Aub = coo_matrix((vals, (rows, cols)), shape=(p, NV + 1)).tocsr()
        except Exception:
            continue
        bnds = [(-r, r)] * NV + [(None, None)]
        try:
            tl = min(6.0, max(1.0, rem - 1.0))
            res = linprog(obj, A_ub=Aub, b_ub=bub, bounds=bnds, method="highs",
                          options={"time_limit": tl, "presolve": True})
        except Exception:
            continue
        if res is None or getattr(res, "status", 1) != 0 or res.x is None:
            continue
        try:
            y = np.asarray(res.x[:-1], dtype=float).reshape(n, dd)
            if not np.all(np.isfinite(y)):
                continue
            dx = np.einsum("ndk,nk->nd", Q, y)
            trial = x + dx
            nr = np.linalg.norm(trial, axis=1, keepdims=True)
            nr[nr == 0] = 1.0
            trial = trial / nr
            nm = _max_cos(trial)
            if nm < cur_max - 1e-12:
                x = trial
                cur_max = nm
        except Exception:
            continue
    return x, cur_max

def solve():
    t0 = time.time()
    T_HOP = t0 + 125.0
    T_LP = t0 + 168.0
    rng = np.random.default_rng(7)
    n, d = 256, 6
    # load incumbent
    best_x = None
    best_val = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=float)
            if a.shape == (256, 6) and np.all(np.isfinite(a)):
                nr = np.linalg.norm(a, axis=1, keepdims=True)
                nr[nr == 0] = 1.0
                a = a / nr
                v = _max_cos(a)
                if v < best_val:
                    best_val = v
                    best_x = a.copy()
        except Exception:
            continue
    if best_x is None:
        best_x = rng.normal(size=(n, d))
        best_x /= np.linalg.norm(best_x, axis=1, keepdims=True)
        best_val = _max_cos(best_x)
    global_x = best_x.copy()
    global_val = best_val
    Ks = [60.0, 100.0, 150.0, 220.0, 320.0, 500.0]
    it = 0
    while time.time() < T_HOP:
        it += 1
        rem = T_HOP - time.time()
        if rem < 1.0:
            break
        u = float(rng.random())
        if u < 0.55:
            sigma = float(rng.uniform(0.08, 0.38))
            noise = rng.normal(size=(n, d)) * sigma
            x0 = global_x + noise
        elif u < 0.80:
            x0 = rng.normal(size=(n, d))
        else:
            x0 = global_x.copy()
            k = int(rng.integers(16, 64))
            idx = rng.choice(n, k, replace=False)
            x0[idx] = rng.normal(size=(k, d))
            # also small shake to rest to change basin
            x0 += rng.normal(size=(n, d)) * 0.02
        nr = np.linalg.norm(x0, axis=1, keepdims=True)
        nr[nr == 0] = 1.0
        x0 = x0 / nr
        K = float(Ks[int(rng.integers(0, len(Ks)))])
        lr = float(rng.uniform(0.002, 0.008))
        # adapt steps to remaining time
        if rem > 60:
            steps = 900
        elif rem > 20:
            steps = 600
        else:
            steps = 400
        try:
            cand, val = _adam_lse(x0, steps, K, lr, eval_every=300)
        except Exception:
            continue
        # uniform top-k equalizer if promising (within 0.003 of best)
        if val < global_val + 0.003:
            try:
                cand2, val2 = _adam_topk(cand, 120, topk=2000, lr=0.004, eval_every=60)
                if val2 < val:
                    cand, val = cand2, val2
            except Exception:
                pass
        if val < global_val - 1e-12:
            global_val = val
            global_x = cand.copy()
    # final LP polish on global best if time remains
    if time.time() < T_LP - 3.0:
        try:
            global_x, global_val = _lp_polish(global_x, global_val, T_LP)
        except Exception:
            pass
    # ensure normalized
    nr = np.linalg.norm(global_x, axis=1, keepdims=True)
    nr[nr == 0] = 1.0
    global_x = global_x / nr
    return np.asarray(global_x, dtype=float)
