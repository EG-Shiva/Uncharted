import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))


def adam_lse_polish(x, steps, a0=50.0, a1=300.0, lr0=0.003, deadline=None):
    n, d = x.shape
    m = np.zeros_like(x)
    vv = np.zeros_like(x)
    b1, b2, eps = 0.9, 0.999, 1e-8
    for t in range(steps):
        if deadline is not None and (t & 63) == 0 and time.time() > deadline:
            break
        frac = t / max(steps - 1, 1)
        alpha = a0 * ((a1 / a0) ** frac)
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        M = float(np.max(G))
        E = np.exp(alpha * (G - M))
        np.fill_diagonal(E, 0.0)
        S = float(E.sum() * 0.5)
        if S <= 0 or not np.isfinite(S):
            break
        Grad = (E @ x) / S
        dot = np.sum(Grad * x, axis=1, keepdims=True)
        Gt = Grad - dot * x
        m = b1 * m + (1 - b1) * Gt
        vv = b2 * vv + (1 - b2) * (Gt * Gt)
        mhat = m / (1 - b1 ** (t + 1))
        vhat = vv / (1 - b2 ** (t + 1))
        step = mhat / (np.sqrt(vhat) + eps)
        np.clip(step, -3.0, 3.0, out=step)
        lrt = lr0 * (1.0 - 0.5 * frac)
        x = x - lrt * step
        nr = np.linalg.norm(x, axis=1, keepdims=True)
        nr[nr == 0] = 1.0
        x /= nr
        if not np.all(np.isfinite(x)):
            break
    nr = np.linalg.norm(x, axis=1, keepdims=True)
    nr[nr == 0] = 1.0
    return x / nr


def single_refine(y, Xf, steps=200, alpha=200.0, lr=0.01, deadline=None):
    y = y / np.linalg.norm(y)
    m = np.zeros(9)
    vv = np.zeros(9)
    b1, b2, eps = 0.9, 0.999, 1e-8
    for t in range(steps):
        if deadline is not None and (t & 63) == 0 and time.time() > deadline:
            break
        d = Xf @ y
        M = float(np.max(d))
        w = np.exp(alpha * (d - M))
        S = float(np.sum(w))
        if S <= 0 or not np.isfinite(S):
            break
        Grad = (w @ Xf) / S
        Gt = Grad - float(np.dot(Grad, y)) * y
        m = b1 * m + (1 - b1) * Gt
        vv = b2 * vv + (1 - b2) * (Gt * Gt)
        mhat = m / (1 - b1 ** (t + 1))
        vhat = vv / (1 - b2 ** (t + 1))
        step = mhat / (np.sqrt(vhat) + eps)
        step = np.clip(step, -3.0, 3.0)
        y = y - lr * step
        nrm = float(np.linalg.norm(y))
        if not np.isfinite(nrm) or nrm < 1e-12:
            break
        y /= nrm
    return y / np.linalg.norm(y)


def best_candidate(Xf, rng, n_cand=50000, batch=10000, deadline=None):
    d = Xf.shape[1]
    best_v = np.inf
    best_y = None
    nb = n_cand // batch
    for _ in range(nb):
        if deadline is not None and time.time() > deadline:
            break
        T = rng.standard_normal((batch, d))
        T /= np.linalg.norm(T, axis=1, keepdims=True)
        D = T @ Xf.T
        mm = D.max(axis=1)
        i = int(np.argmin(mm))
        if float(mm[i]) < best_v:
            best_v = float(mm[i])
            best_y = T[i].copy()
    if best_y is None:
        best_y = rng.standard_normal(d)
        best_y /= np.linalg.norm(best_y)
    return best_y


def lp_polish_deadline(x, deadline, r_init=0.005, cap=2500):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    try:
        best = max_cos(x)
    except Exception:
        return x
    radius = r_init
    iu0 = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        G = x @ x.T
        c_all = G[iu0]
        mval = float(c_all.max())
        thresh = mval - max(20.0 * radius, 1e-9)
        sel = c_all > thresh
        cnt = int(np.count_nonzero(sel))
        if cnt == 0:
            radius *= 0.5
            continue
        if cnt > cap:
            idx = np.where(sel)[0]
            order = np.argpartition(-c_all[idx], cap)[:cap]
            keep = np.zeros_like(sel)
            keep[idx[order]] = True
            sel = keep
        I = iu0[0][sel]; J = iu0[1][sel]
        c_sel = c_all[sel]
        p = len(I)
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.empty((p, 2 * d + 1), dtype=int)
        cols[:, :d] = I[:, None] * d + np.arange(d)
        cols[:, d:2 * d] = J[:, None] * d + np.arange(d)
        cols[:, 2 * d] = n * d
        cols = cols.ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], axis=1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        except Exception:
            radius *= 0.5
            continue
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c_sel, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        yn = np.linalg.norm(y, axis=1, keepdims=True)
        if not np.all(np.isfinite(yn)) or np.any(yn < 1e-12):
            radius *= 0.5
            continue
        y = y / yn
        v = max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.03)
        else:
            radius *= 0.5
    return x


def solve():
    t0 = time.time()
    LOOP_END = t0 + 120.0
    LP_END = t0 + 168.0
    n, d = 176, 9
    rng = np.random.default_rng(0)
    try:
        xb = np.load("best.npy")
    except Exception:
        try:
            xb = np.load("record.npy")
        except Exception:
            xb = rng.standard_normal((n, d))
    xb = np.asarray(xb, dtype=float)
    if xb.shape != (n, d) or not np.all(np.isfinite(xb)):
        xb = rng.standard_normal((n, d))
    xb = xb / np.linalg.norm(xb, axis=1, keepdims=True)
    best_v = max_cos(xb)
    thresh_keep = 0.4611825753
    # contact counts for hub
    try:
        G0 = xb @ xb.T
        np.fill_diagonal(G0, -2.0)
        m0 = float(G0.max())
        C = (G0 > m0 - 1e-9)
        contacts = C.sum(axis=1)
        iu0 = np.triu_indices(n, 1)
        c_all = G0[iu0]
    except Exception:
        contacts = np.zeros(n)
    hub = int(np.argmax(contacts))
    # neighbour order of hub
    try:
        dots_hub = xb @ xb[hub]
        dots_hub[hub] = -2.0
        nbr_order = np.argsort(-dots_hub)
        nbr_order = nbr_order[nbr_order != hub]
    except Exception:
        nbr_order = np.arange(n)
        nbr_order = nbr_order[nbr_order != hub]

    trials = []
    for k in (8, 12, 16):
        # hub + neighbourhood
        del_hub = [hub] + [int(v) for v in nbr_order[:k - 1]]
        trials.append((k, np.array(del_hub, dtype=int), True))
        # random
        rs = np.random.default_rng(1000 + k).choice(n, size=k, replace=False)
        trials.append((k, np.array(rs, dtype=int), False))

    best_X = xb.copy()
    improved = False

    for ti, (k, del_idx, is_hub) in enumerate(trials):
        if time.time() > LOOP_END:
            break
        keep = np.ones(n, dtype=bool)
        keep[del_idx] = False
        Xrem = xb[keep].copy()
        # Adam polish remainder 800 steps, alpha 50->300 lr 0.003
        Xrem = adam_lse_polish(Xrem, steps=800, a0=50.0, a1=300.0, lr0=0.003, deadline=LOOP_END)
        if time.time() > LOOP_END:
            break
        cur = Xrem.copy()
        # reinsert one at a time
        ok = True
        for s in range(k):
            if time.time() > LOOP_END:
                ok = False
                break
            y0 = best_candidate(cur, rng, n_cand=50000, batch=10000, deadline=LOOP_END)
            if time.time() > LOOP_END:
                ok = False
                break
            y0 = single_refine(y0, cur, steps=200, alpha=200.0, lr=0.01, deadline=LOOP_END)
            cur = np.vstack([cur, y0[None, :]])
            cur = adam_lse_polish(cur, steps=300, a0=50.0, a1=300.0, lr0=0.003, deadline=LOOP_END)
        if not ok:
            continue
        if cur.shape != (n, d):
            continue
        cur = cur / np.linalg.norm(cur, axis=1, keepdims=True)
        if not np.all(np.isfinite(cur)):
            continue
        v = max_cos(cur)
        if v < best_v and v < thresh_keep:
            best_v = v
            best_X = cur.copy()
            improved = True
            # keep searching for even better if time remains? continue
    if not improved:
        return xb
    # one deadline-checked LP polish, at most ~20s
    try:
        remain = LP_END - time.time()
        if remain > 5.0:
            lp_dead = time.time() + min(20.0, remain)
            polished = lp_polish_deadline(best_X, lp_dead)
            pv = max_cos(polished)
            if pv < best_v and np.all(np.isfinite(polished)):
                best_X = polished
                best_v = pv
    except Exception:
        pass
    best_X = best_X / np.linalg.norm(best_X, axis=1, keepdims=True)
    return best_X
