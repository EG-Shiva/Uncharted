import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def _max_cos(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())


def _permax(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return G.max(axis=1)


def _lp_polish(x, deadline, max_solves=100):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    iu = np.triu_indices(n, 1)
    solves = 0
    while time.time() < deadline and radius > 1e-14 and solves < max_solves:
        G = x @ x.T
        c = G[iu]
        m = c.max()
        sel = c > m - max(20 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        if p == 0:
            break
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d),
                                       np.arange(n * d))), shape=(n, nv)).tocsr()
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
        solves += 1
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        v = _max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x, best


def _anneal(x, idx, steps, T0, T1, lrfac, balance, rng=None):
    # idx: array of movable indices; if None move all
    n, d = x.shape
    if idx is None:
        movable = None
    else:
        movable = np.asarray(idx)
        mask = np.zeros(n, dtype=bool)
        mask[movable] = True
    for s in range(steps):
        frac = s / max(steps - 1, 1)
        T = T0 * (T1 / T0) ** frac
        G = x @ x.T
        np.fill_diagonal(G, -1.0)
        m = float(G.max())
        W = np.exp((G - m) / T)
        np.fill_diagonal(W, 0.0)
        if balance:
            cnt = (G > m - 3.0 * T).sum(axis=1)
            scale = 1.0 + 0.45 * (cnt - 6.0) / 6.0
            scale = np.clip(scale, 0.5, 2.0)
            W = W * scale[:, None]
        grad = W @ x
        dot = np.sum(grad * x, axis=1, keepdims=True)
        gt = grad - dot * x
        row = W.sum(axis=1, keepdims=True) + 1e-12
        gt = gt / row
        lr = lrfac * T
        if movable is None:
            x = x - lr * gt
            x /= np.linalg.norm(x, axis=1, keepdims=True)
        else:
            x[movable] = x[movable] - lr * gt[movable]
            x[movable] /= np.linalg.norm(x[movable], axis=1, keepdims=True)
    return x


def _greedy_cap_refill(y, cap, rng, ns=2500):
    n = y.shape[0]
    # background indices
    is_cap = np.zeros(n, dtype=bool)
    is_cap[cap] = True
    # order to fill
    order = rng.permutation(cap)
    # current positions: keep background, cap holes empty
    # we fill sequentially: candidate pool per slot
    for p in order:
        R = rng.standard_normal(size=(ns, 4))
        R /= np.linalg.norm(R, axis=1, keepdims=True)
        # dots to all points except other unfilled holes:
        # build matrix of present points: background + already refilled cap points
        # already refilled = cap points not in remaining? simpler: use y with
        # not-yet-filled rows ignored by masking: compute S = R @ y.T then
        # mask out columns corresponding to not-yet-filled (set -inf)
        S = R @ y.T
        # mask self and unfilled: find unfilled set
        # we track filled flag
        # Instead maintain a boolean present array
        # present = ~is_cap initially False for cap; as we fill, set True
        # Use closure variable present
        S[:, ~present] = -2.0
        # also mask self column p (if present false already masked)
        m = S.max(axis=1)
        bi = int(np.argmin(m))
        y[p] = R[bi]
        present[p] = True
    return y


def solve():
    t0 = time.time()
    deadline = t0 + 175.0
    explore_end = t0 + 138.0
    rng = np.random.default_rng(7)
    try:
        xb = np.load("best.npy").astype(float)
    except Exception:
        xb = None
    try:
        xr = np.load("record.npy").astype(float)
    except Exception:
        xr = None
    cands = []
    if xb is not None and xb.shape == (153, 4):
        xn = xb / np.linalg.norm(xb, axis=1, keepdims=True)
        if np.all(np.isfinite(xn)):
            cands.append(xn)
    if xr is not None and xr.shape == (153, 4):
        xn = xr / np.linalg.norm(xr, axis=1, keepdims=True)
        if np.all(np.isfinite(xn)):
            cands.append(xn)
    if not cands:
        R = rng.standard_normal(size=(153, 4))
        R /= np.linalg.norm(R, axis=1, keepdims=True)
        cands.append(R)
    vals = [_max_cos(c) for c in cands]
    bi = int(np.argmin(vals))
    best_x = cands[bi].copy()
    best_val = vals[bi]
    cur_x = best_x.copy()
    cur_val = best_val
    T_accept = 0.0012
    trial = 0
    n = 153
    # quick initial LP squeeze of loaded best is done at end; explore first
    while time.time() < explore_end:
        trial += 1
        if trial % 9 == 0:
            base = best_x
        else:
            base = cur_x
        if cur_val > best_val + 0.012 and trial % 5 == 0:
            base = best_x
        mode = trial % 7
        y = base.copy()
        if mode == 6:
            # global kick + global anneal (diversify)
            y = y + 0.06 * rng.standard_normal(y.shape)
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            y = _anneal(y, None, 320, 0.022, 0.0022, 1.2, True)
            y = _anneal(y, None, 100, 0.0035, 0.0012, 1.0, False)
            v = _max_cos(y)
        else:
            # LNS cap move
            per = _permax(base)
            r = trial % 3
            if r == 0:
                cvec = base[int(np.argmax(per))]
            elif r == 1:
                cvec = base[int(np.argmin(per))]
            else:
                cvec = rng.standard_normal(4)
                cvec /= np.linalg.norm(cvec)
            dots = base @ cvec
            k = int(rng.integers(8, 22))
            # occasionally larger cap
            if trial % 11 == 0:
                k = int(rng.integers(22, 34))
            cap = np.argsort(-dots)[:k]
            # greedy refill with void search
            # need present mask global for helper
            global present
            present = np.ones(n, dtype=bool)
            present[cap] = False
            # present as global for helper closure (use python global trick via mutable)
            # monkey: attach to function via closure variable; here use global
            y = _greedy_cap_refill(y, cap, rng, ns=2200 if k < 20 else 1500)
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            # small jitter
            y[cap] = y[cap] + 0.004 * rng.standard_normal((k, 4))
            y[cap] /= np.linalg.norm(y[cap], axis=1, keepdims=True)
            # local balanced anneal on cap, then light global tail
            y = _anneal(y, cap, 260, 0.016, 0.002, 1.25, True)
            y = _anneal(y, cap, 80, 0.004, 0.0012, 1.0, False)
            # occasional global smoothing
            if trial % 4 == 0:
                y = _anneal(y, None, 90, 0.004, 0.0012, 1.0, False)
            v = _max_cos(y)
        if v < best_val - 1e-12:
            best_x, best_val = y, v
            cur_x, cur_val = y.copy(), v
            # quick LP probe when promising and time allows
            if v < 0.8492 and time.time() < explore_end - 8.0:
                y2, v2 = _lp_polish(y, time.time() + 1.0, max_solves=12)
                if v2 < best_val:
                    best_x, best_val = y2, v2
                    cur_x, cur_val = y2.copy(), v2
            continue
        if v < cur_val:
            cur_x, cur_val = y, v
        else:
            if rng.random() < np.exp((cur_val - v) / T_accept):
                cur_x, cur_val = y, v
        if trial % 7 == 0 and time.time() < explore_end - 6.0 and v < best_val + 0.0025:
            y2, v2 = _lp_polish(y, time.time() + 0.9, max_solves=10)
            if v2 < best_val - 1e-12:
                best_x, best_val = y2, v2
                cur_x, cur_val = y2.copy(), v2
            elif v2 < cur_val:
                cur_x, cur_val = y2, v2
    if time.time() < deadline - 1.0:
        # final global smooth then squeeze
        try:
            best_x = _anneal(best_x, None, 120, 0.003, 0.0008, 1.0, False)
        except Exception:
            pass
        best_x, best_val = _lp_polish(best_x, deadline - 0.5, max_solves=200)
    best_x = best_x / np.linalg.norm(best_x, axis=1, keepdims=True)
    return best_x
