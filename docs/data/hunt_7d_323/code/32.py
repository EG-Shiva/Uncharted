import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _lp_polish(X, seconds, deadline):
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    n, d = X.shape
    best = _max_cos(X)
    radius = 5e-3
    start = time.time()
    iuI, iuJ = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        if time.time() > deadline - 1.0:
            break
        remain = seconds - (time.time() - start)
        if remain <= 0.05:
            break
        G = X @ X.T
        c = G[iuI, iuJ]
        m = float(c.max())
        thresh = m - max(20.0 * radius, 1e-9)
        sel = c > thresh
        cnt = int(sel.sum())
        if cnt == 0:
            break
        if cnt > 2200:
            idx = np.where(sel)[0]
            cidx = c[idx]
            part = np.argpartition(cidx, -2200)[-2200:]
            sel = np.zeros_like(sel, dtype=bool)
            sel[idx[part]] = True
            cnt = 2200
        I = iuI[sel]; J = iuJ[sel]; csel = c[sel]
        p = cnt
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.empty(p * (2 * d + 1), dtype=np.int64)
        vals = np.empty(p * (2 * d + 1), dtype=np.float64)
        Cb = cols.reshape(p, 2 * d + 1)
        Vb = vals.reshape(p, 2 * d + 1)
        ar = np.arange(d)
        Cb[:, :d] = I[:, None] * d + ar
        Cb[:, d:2 * d] = J[:, None] * d + ar
        Cb[:, 2 * d] = n * d
        Vb[:, :d] = X[J]
        Vb[:, d:2 * d] = X[I]
        Vb[:, 2 * d] = -1.0
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((X.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv); obj[-1] = 1.0
        tl = float(min(2.0, max(0.2, remain - 0.05)))
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                          method="highs", options={"time_limit": tl})
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = X + res.x[:-1].reshape(n, d)
        yn = np.linalg.norm(y, axis=1, keepdims=True)
        yn[yn == 0] = 1.0
        y = y / yn
        v = _max_cos(y)
        if v < best:
            X = y; best = v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return X, best

def _contact_counts(X, m, tol=1e-9):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return np.sum(G > m - tol, axis=1)

def _greedy_reinsert(Y, to_replace, rng, deadline, n_cand=3500):
    n, d = Y.shape
    for k in to_replace:
        if time.time() > deadline - 1.0:
            break
        mask = np.ones(n, dtype=bool); mask[k] = False
        Xf = Y[mask]
        cand = rng.standard_normal(size=(n_cand, d))
        cand /= np.linalg.norm(cand, axis=1, keepdims=True)
        dots = cand @ Xf.T
        worst = dots.max(axis=1)
        bi = int(np.argmin(worst))
        Y[k] = cand[bi]
    return Y

def _anneal(Y, rng, steps, step0, step1, T0, T1, deadline):
    n, d = Y.shape
    for s in range(steps):
        if (s & 31) == 0 and time.time() > deadline - 1.0:
            break
        frac = s / max(1, steps - 1)
        T = T0 * (T1 / T0) ** frac
        step = step0 * (step1 / step0) ** frac
        G = Y @ Y.T
        np.fill_diagonal(G, -1.0)
        m = float(G.max())
        cut = max(0.08, 6.0 * T)
        M = G > (m - cut)
        # weights
        W = np.exp((G - m) / max(T, 1e-12)) * M
        np.fill_diagonal(W, 0.0)
        ssum = (W * G).sum(axis=1)
        WY = W @ Y
        F = ssum[:, None] * Y - WY
        fn = np.linalg.norm(F, axis=1, keepdims=True) + 1e-12
        Y = Y + step * (F / fn)
        Y = Y / np.linalg.norm(Y, axis=1, keepdims=True)
    return Y

def solve():
    t0 = time.time()
    DEADLINE = t0 + 160.0
    HARD_STOP = t0 + 165.0
    rng = np.random.default_rng(7)
    best_x = None; best_v = np.inf
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn).astype(np.float64)
            if a.shape == (323, 7):
                a = a / np.linalg.norm(a, axis=1, keepdims=True)
                v = _max_cos(a)
                if v < best_v:
                    best_v = v; best_x = a.copy()
        except Exception:
            pass
    if best_x is None:
        a = rng.standard_normal(size=(323, 7))
        a /= np.linalg.norm(a, axis=1, keepdims=True)
        best_x = a; best_v = _max_cos(a)
    X0 = best_x.copy()
    # quick initial tiny polish to lock baseline within 3s
    try:
        Xp, vp = _lp_polish(X0, min(3.0, max(0.5, DEADLINE - time.time() - 20.0)), DEADLINE)
        if vp < best_v:
            best_v = vp; best_x = Xp
    except Exception:
        pass
    it = 0
    n, d = 323, 7
    while time.time() < DEADLINE - 12.0:
        it += 1
        if time.time() > HARD_STOP - 12.0:
            break
        Y = best_x.copy()
        m0 = _max_cos(Y)
        cnt = _contact_counts(Y, m0, tol=1e-9)
        order_hub = np.argsort(-cnt)
        order_low = np.argsort(cnt)
        mode = it % 4
        if mode == 0:
            k = int(rng.integers(6, 11))
            to_rep = order_hub[:k].tolist()
        elif mode == 1:
            k = int(rng.integers(6, 11))
            # rattlers/low + hubs mix
            to_rep = order_low[:max(2, k // 2)].tolist() + order_hub[:k - max(2, k // 2)].tolist()
        elif mode == 2:
            k = int(rng.integers(8, 14))
            rnd = rng.choice(n, size=k, replace=False).tolist()
            to_rep = rnd
        else:
            k = int(rng.integers(5, 10))
            to_rep = order_hub[:k].tolist()
            # add small global kick
            Y = Y + rng.standard_normal(size=Y.shape) * 0.01
            Y /= np.linalg.norm(Y, axis=1, keepdims=True)
        # de-duplicate
        to_rep = list(dict.fromkeys([int(v) for v in to_rep]))
        Y = _greedy_reinsert(Y, to_rep, rng, DEADLINE, n_cand=3500)
        if time.time() > DEADLINE - 11.0:
            break
        # anneal to change topology (cheap)
        steps = 260
        # vary aggression by mode
        if mode == 3:
            T0, T1 = 0.025, 0.002
            s0, s1 = 0.03, 0.004
        else:
            T0, T1 = 0.015, 0.0015
            s0, s1 = 0.025, 0.003
        Y = _anneal(Y, rng, steps, s0, s1, T0, T1, DEADLINE)
        v_ann = _max_cos(Y)
        # only LP-polish if promising (within 0.004 of best) to save time
        if v_ann > best_v + 0.004:
            continue
        polish_sec = 3.0 if v_ann > best_v else 4.5
        polish_sec = min(polish_sec, max(0.5, DEADLINE - 11.0 - time.time()))
        if polish_sec < 0.4:
            break
        try:
            Yp, vp = _lp_polish(Y, polish_sec, DEADLINE)
        except Exception:
            continue
        if vp < best_v - 1e-12:
            best_v = vp; best_x = Yp
            # early exit if big win
            if best_v < 0.6645:
                break
    # final snap on incumbent with remaining time
    try:
        remain = DEADLINE - time.time()
        if remain > 1.5:
            Xf, vf = _lp_polish(best_x, min(remain - 0.5, 9.0), HARD_STOP)
            if vf < best_v:
                best_v = vf; best_x = Xf
    except Exception:
        pass
    best_x = best_x / np.linalg.norm(best_x, axis=1, keepdims=True)
    return best_x
