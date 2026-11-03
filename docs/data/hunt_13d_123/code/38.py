import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())


def _active_descend(y, steps, lr0, lr1, w0, w1, rng=None):
    n, d = y.shape
    for s in range(steps):
        frac = s / max(steps - 1, 1)
        lr = lr0 + (lr1 - lr0) * frac
        win = w0 + (w1 - w0) * frac
        if win < 1e-12:
            win = 1e-12
        G = y @ y.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        thresh = m - win
        # weights: linear ramp above thresh
        W = (G - thresh) / win
        np.clip(W, 0.0, 1.0, out=W)
        # emphasize tightest: W = 0.2 + 0.8*W for active
        # keep zero for inactive
        np.fill_diagonal(W, 0.0)
        act = W > 0
        if not np.any(act):
            continue
        W[act] = 0.2 + 0.8 * W[act]
        S = (W * G).sum(axis=1)
        A = W @ y
        F = A - S[:, None] * y
        cnt = W.sum(axis=1, keepdims=True) + 1.0
        F /= cnt
        # clip force
        y = y - lr * F
        yn = np.linalg.norm(y, axis=1, keepdims=True)
        yn[yn == 0] = 1.0
        y /= yn
    return y


def _degrees(y, window=0.005):
    G = y @ y.T
    np.fill_diagonal(G, -2.0)
    m = float(G.max())
    deg = (G > m - window).sum(axis=1)
    return deg, m, G


def _deepest_hole(y, keep_idx, rng, n_cand=2000):
    # find vector minimizing max cosine to y[keep]
    n, d = y.shape
    mask = np.ones(n, dtype=bool)
    mask[keep_idx] = False
    Yr = y[mask]
    cand = rng.normal(size=(n_cand, d))
    cand /= np.linalg.norm(cand, axis=1, keepdims=True)
    scores = (cand @ Yr.T).max(axis=1)
    bi = int(np.argmin(scores))
    return cand[bi]


def _lp_polish(x, seconds, p_cap=1500):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    best_x = x.copy()
    radius = 1e-3
    start = time.time()
    iu0, iu1 = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        G = x @ x.T
        c_all = G[iu0, iu1]
        m = float(c_all.max())
        window = max(20.0 * radius, 1e-9)
        sel = c_all > m - window
        cnt = int(np.count_nonzero(sel))
        if cnt == 0:
            break
        if cnt > p_cap:
            idx = np.where(sel)[0]
            part = np.argpartition(-c_all[idx], p_cap - 1)[:p_cap]
            keep = np.zeros_like(sel)
            keep[idx[part]] = True
            sel = keep
            cnt = p_cap
        I = iu0[sel]
        J = iu1[sel]
        csel = c_all[sel]
        p = cnt
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([ (I[:, None] * d + np.arange(d)).reshape(p, d),
                                (J[:, None] * d + np.arange(d)).reshape(p, d),
                                np.full((p, 1), n * d)], axis=1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], axis=1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            erows = np.repeat(np.arange(n), d)
            ecols = np.arange(n * d)
            A_eq = coo_matrix((x.ravel(), (erows, ecols)), shape=(n, nv)).tocsr()
            obj = np.zeros(nv); obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        yn = np.linalg.norm(y, axis=1, keepdims=True)
        yn[yn == 0] = 1.0
        y /= yn
        v = _max_cos(y)
        if v < best:
            best = v
            best_x = y.copy()
            x = y
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
            if radius < 1e-7:
                x = best_x.copy()
        if time.time() - start >= seconds:
            break
    return best_x


def solve():
    t0 = time.time()
    HOP_END = 130.0
    HARD_END = 168.0
    n, d = 123, 13
    rng = np.random.default_rng(0)
    np.random.seed(0)

    best = None
    best_val = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=float)
            if a.shape == (n, d):
                a = a / np.linalg.norm(a, axis=1, keepdims=True)
                v = _max_cos(a)
                if v < best_val:
                    best_val = v
                    best = a.copy()
        except Exception:
            pass
    if best is None:
        best = rng.normal(size=(n, d))
        best /= np.linalg.norm(best, axis=1, keepdims=True)
        best_val = _max_cos(best)

    # degrees to know rattler/worst initially (diagnostic only)
    it = 0
    # hopping loop (elitist from best)
    while time.time() - t0 < HOP_END:
        it += 1
        if time.time() - t0 > HOP_END - 2:
            break
        y = best.copy()
        try:
            deg, m, G = _degrees(y, window=0.005)
        except Exception:
            deg = np.zeros(n)
            m = best_val
        r = int(np.argmin(deg))
        w = int(np.argmax(deg))
        mode = it % 7
        try:
            if mode == 0:
                # small global shake
                sig = 0.012
                y = y + sig * rng.normal(size=(n, d))
                y /= np.linalg.norm(y, axis=1, keepdims=True)
                y = _active_descend(y, steps=500, lr0=0.015, lr1=0.002, w0=0.03, w1=0.002)
            elif mode == 1:
                # cluster shake around worst hub
                order = np.argsort(-G[w])
                k = 15
                nb = order[1:1 + k]
                y[nb] += 0.18 * rng.normal(size=(k, d))
                y[w] += 0.10 * rng.normal(size=(d,))
                y /= np.linalg.norm(y, axis=1, keepdims=True)
                y = _active_descend(y, steps=600, lr0=0.018, lr1=0.002, w0=0.04, w1=0.002)
            elif mode == 2:
                # relocate rattler to deepest hole
                h = _deepest_hole(y, r, rng, n_cand=2500)
                y[r] = h
                # small shake neighbors of hole to make room
                y /= np.linalg.norm(y, axis=1, keepdims=True)
                y = _active_descend(y, steps=600, lr0=0.015, lr1=0.002, w0=0.03, w1=0.0015)
            elif mode == 3:
                # move worst hub into deepest hole (without worst)
                h = _deepest_hole(y, w, rng, n_cand=2500)
                y[w] = h
                y /= np.linalg.norm(y, axis=1, keepdims=True)
                y = _active_descend(y, steps=650, lr0=0.018, lr1=0.002, w0=0.04, w1=0.002)
            elif mode == 4:
                # move worst near rattler void to balance load
                y[w] = y[r] + 0.35 * rng.normal(size=(d,))
                yn = np.linalg.norm(y[w])
                if yn == 0:
                    y[w] = rng.normal(size=(d,))
                    yn = np.linalg.norm(y[w])
                y[w] /= yn
                # shake a few other hubs lightly
                order = np.argsort(-deg)
                hubs = order[:5]
                y[hubs] += 0.05 * rng.normal(size=(5, d))
                y /= np.linalg.norm(y, axis=1, keepdims=True)
                y = _active_descend(y, steps=600, lr0=0.015, lr1=0.002, w0=0.035, w1=0.002)
            elif mode == 5:
                # medium global shake
                sig = 0.035
                y = y + sig * rng.normal(size=(n, d))
                y /= np.linalg.norm(y, axis=1, keepdims=True)
                y = _active_descend(y, steps=550, lr0=0.016, lr1=0.002, w0=0.035, w1=0.002)
            else:
                # delete 3 worst cluster and refill holes greedily
                order = np.argsort(-deg)
                victims = order[:3]
                for v in victims:
                    h = _deepest_hole(y, v, rng, n_cand=1500)
                    y[v] = h
                y /= np.linalg.norm(y, axis=1, keepdims=True)
                y = _active_descend(y, steps=700, lr0=0.018, lr1=0.0015, w0=0.04, w1=0.0015)
        except Exception:
            continue
        try:
            v = _max_cos(y)
        except Exception:
            continue
        if v < best_val - 1e-12:
            best = y.copy()
            best_val = v
        # time guard
        if time.time() - t0 >= HOP_END:
            break

    # final snap: short active fine tune then LP
    try:
        # quick fine active descent on best
        if time.time() - t0 < HARD_END - 30:
            best = _active_descend(best, steps=400, lr0=0.006, lr1=0.001, w0=0.01, w1=0.001)
            nv = _max_cos(best)
            if nv < best_val:
                best_val = nv
            else:
                pass
    except Exception:
        pass

    remain = HARD_END - (time.time() - t0) - 3.0
    if remain > 6:
        try:
            cand = _lp_polish(best, seconds=min(remain, 28.0), p_cap=1500)
            v = _max_cos(cand)
            if v < best_val:
                best = cand
                best_val = v
        except Exception:
            pass

    best = best / np.linalg.norm(best, axis=1, keepdims=True)
    return best
