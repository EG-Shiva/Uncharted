import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(np.max(g))


def _descent(x, beta, steps, lr, mu):
    n, d = x.shape
    v = np.zeros_like(x)
    for it in range(steps):
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        m = np.max(G)
        W = np.exp(beta * (G - m))
        np.fill_diagonal(W, 0.0)
        rs = W.sum(axis=1, keepdims=True) + 1e-12
        mean = (W @ x) / rs
        dot = np.sum(mean * x, axis=1, keepdims=True)
        tang = mean - dot * x
        v = mu * v + tang
        cur_lr = lr * (1.0 - 0.5 * it / steps)
        x = x - cur_lr * v
        nrm = np.linalg.norm(x, axis=1, keepdims=True)
        nrm[nrm == 0] = 1.0
        x = x / nrm
    return x


def _polish_single(y, X, idx, beta=120.0, steps=200, lr=0.15):
    # minimize max_j!=idx <y, X[j]> over sphere
    for _ in range(steps):
        c = X @ y
        c[idx] = -2.0
        m = float(np.max(c))
        w = np.exp(beta * (c - m))
        w[idx] = 0.0
        rs = float(np.sum(w)) + 1e-12
        mean = (w @ X) / rs
        dot = float(mean @ y)
        tang = mean - dot * y
        y = y - lr * tang
        n = float(np.linalg.norm(y))
        if n == 0 or not np.isfinite(n):
            break
        y = y / n
        lr *= 0.995
    return y


def _lp_polish(x, deadline):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    iu = np.triu_indices(n, 1)
    while time.time() < deadline - 1.0 and radius > 1e-14:
        Gall = x @ x.T
        c_all = Gall[iu]
        m = float(np.max(c_all))
        thresh = m - max(20.0 * radius, 1e-9)
        sel = c_all > thresh
        cnt = int(np.sum(sel))
        if cnt == 0:
            radius *= 0.5
            continue
        if cnt > 3500:
            K = 3500
            top = np.argpartition(-c_all, K)[:K]
            msel = np.zeros_like(sel)
            msel[top] = True
            sel = msel
        I = iu[0][sel]
        J = iu[1][sel]
        c_sel = c_all[sel]
        p = len(I)
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols_I = (I[:, None] * d + np.arange(d)).ravel()
        cols_J = (J[:, None] * d + np.arange(d)).ravel()
        cols = np.empty(p * (2 * d + 1), dtype=np.int64)
        cols[0::2 * d + 1] = cols_I[0::d] if False else 0  # placeholder
        # build cols vectorised
        cols_blocks = np.concatenate(
            [I[:, None] * d + np.arange(d),
             J[:, None] * d + np.arange(d),
             np.full((p, 1), n * d)], axis=1
        ).ravel()
        cols = cols_blocks
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], axis=1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix(
                (x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))),
                shape=(n, nv)).tocsr()
            obj = np.zeros(nv)
            obj[-1] = 1.0
            bounds = [(-radius, radius)] * (n * d) + [(None, None)]
            res = linprog(obj, A_ub=A_ub, b_ub=-c_sel, A_eq=A_eq,
                          b_eq=np.zeros(n), bounds=bounds, method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        nrm = np.linalg.norm(y, axis=1, keepdims=True)
        nrm[nrm == 0] = 1.0
        y = y / nrm
        if not np.all(np.isfinite(y)):
            radius *= 0.5
            continue
        v = _max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x


def solve():
    t0 = time.time()
    HARD = t0 + 175.0
    rng = np.random.default_rng(0)
    cands = []
    try:
        a = np.load("best.npy")
        cands.append(np.asarray(a, dtype=float))
    except Exception:
        pass
    try:
        b = np.load("record.npy")
        cands.append(np.asarray(b, dtype=float))
    except Exception:
        pass
    if len(cands) == 0:
        cands.append(rng.standard_normal((250, 6)))

    best = None
    best_score = np.inf
    for c in cands:
        c = np.nan_to_num(np.asarray(c, dtype=float))
        if c.shape != (250, 6):
            continue
        nrm = np.linalg.norm(c, axis=1, keepdims=True)
        nrm[nrm == 0] = 1.0
        c = c / nrm
        s = _max_cos(c)
        if s < best_score:
            best_score = s
            best = c.copy()
    if best is None:
        best = rng.standard_normal((250, 6))
        best = best / np.linalg.norm(best, axis=1, keepdims=True)
        best_score = _max_cos(best)
    n, d = best.shape
    global_best = best.copy()

    # Phase 1: basin hopping with soft-max repulsion
    phase1_end = t0 + 108.0
    trials = 0
    betas = [45.0, 90.0, 160.0]
    while time.time() < min(phase1_end, HARD - 55.0):
        trials += 1
        if trials % 7 == 0:
            sigma = 0.14
        elif trials % 3 == 0:
            sigma = 0.05
        else:
            sigma = 0.018
        beta = betas[trials % 3]
        # find worst indices to kick harder
        G = global_best @ global_best.T
        np.fill_diagonal(G, -2.0)
        row_max = G.max(axis=1)
        worst = np.argsort(-row_max)[:12]
        noise = rng.standard_normal((n, d))
        x_try = global_best + sigma * noise
        extra = rng.standard_normal((len(worst), d))
        x_try[worst] += sigma * 1.5 * extra
        nrm = np.linalg.norm(x_try, axis=1, keepdims=True)
        nrm[nrm == 0] = 1.0
        x_try = x_try / nrm
        # two-stage descent: coarse then fine
        try:
            if beta < 100:
                x_try = _descent(x_try, beta, 320, 0.008, 0.85)
                x_try = _descent(x_try, 150.0, 220, 0.005, 0.88)
            else:
                x_try = _descent(x_try, beta, 450, 0.007, 0.86)
                x_try = _descent(x_try, 200.0, 150, 0.004, 0.88)
        except Exception:
            continue
        if time.time() > HARD - 50.0:
            break
        s = _max_cos(x_try)
        if s < best_score - 1e-12:
            global_best = x_try
            best_score = s
        if time.time() > min(phase1_end, HARD - 55.0):
            break

    # Phase 2: greedy worst-point reinsertion
    phase2_end = t0 + 142.0
    it2 = 0
    while time.time() < min(phase2_end, HARD - 35.0):
        it2 += 1
        G = global_best @ global_best.T
        np.fill_diagonal(G, -2.0)
        row_max = G.max(axis=1)
        order = np.argsort(-row_max)
        improved_this_round = False
        for k in range(min(6, n)):
            idx = int(order[k])
            cur_worst = float(row_max[idx])
            # other max excluding idx
            mask = np.ones(n, dtype=bool)
            mask[idx] = False
            other_max = float(np.max(G[mask][:, mask])) if n > 2 else -2.0
            C = 3500
            cand = rng.standard_normal((C, d))
            cand = cand / np.linalg.norm(cand, axis=1, keepdims=True)
            S = cand @ global_best.T
            S[:, idx] = -2.0
            srow = S.max(axis=1)
            jstar = int(np.argmin(srow))
            y0 = cand[jstar]
            # local polish of single point
            y0 = _polish_single(y0.copy(), global_best, idx, beta=130.0, steps=180, lr=0.12)
            cnew = global_best @ y0
            cnew[idx] = -2.0
            new_pt_max = float(np.max(cnew))
            new_global = float(max(other_max, new_pt_max))
            if new_global < best_score - 1e-12 and new_pt_max < cur_worst:
                global_best[idx] = y0 / np.linalg.norm(y0)
                best_score = new_global
                improved_this_round = True
                break
            if time.time() > min(phase2_end, HARD - 35.0):
                break
        if not improved_this_round:
            # small random kick to a rattler-like point and local descent
            if it2 % 3 == 0 and time.time() < min(phase2_end, HARD - 35.0) - 2.0:
                idx = int(order[rng.integers(0, n)])
                y = global_best[idx] + 0.08 * rng.standard_normal(d)
                y = y / np.linalg.norm(y)
                y = _polish_single(y, global_best, idx, beta=130.0, steps=150, lr=0.10)
                cnew = global_best @ y
                cnew[idx] = -2.0
                new_pt_max = float(np.max(cnew))
                mask = np.ones(n, dtype=bool)
                mask[idx] = False
                other_max = float(np.max((global_best @ global_best.T)[mask][:, mask]))
                new_global = float(max(other_max, new_pt_max))
                if new_global < best_score - 1e-12:
                    global_best[idx] = y / np.linalg.norm(y)
                    best_score = new_global
            else:
                # no improvement, do a short full descent from slight noise to change basin
                if time.time() < min(phase2_end, HARD - 35.0) - 3.0:
                    x_try = global_best + 0.012 * rng.standard_normal((n, d))
                    x_try = x_try / np.linalg.norm(x_try, axis=1, keepdims=True)
                    x_try = _descent(x_try, 110.0, 250, 0.006, 0.86)
                    s = _max_cos(x_try)
                    if s < best_score - 1e-12:
                        global_best = x_try
                        best_score = s
                        improved_this_round = True
            if not improved_this_round and it2 > 40:
                break
        if time.time() >= min(phase2_end, HARD - 35.0):
            break

    # Phase 3: LP polish with remaining time
    remain = HARD - time.time()
    if remain > 4.0:
        try:
            global_best = _lp_polish(global_best, HARD - 1.0)
        except Exception:
            pass

    global_best = global_best / np.linalg.norm(global_best, axis=1, keepdims=True)
    global_best = np.nan_to_num(global_best)
    return global_best
