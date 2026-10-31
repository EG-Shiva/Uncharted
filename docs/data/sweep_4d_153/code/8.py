import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(np.max(g))


def _smooth_descent(x, steps, mu=0.85):
    n, d = x.shape
    v = np.zeros_like(x)
    for t in range(steps):
        p = t / max(steps - 1, 1)
        K = 30.0 + (300.0 - 30.0) * (p * p)
        delta = 0.04 * (1 - p) + 0.003 * p
        lr = 0.015 * (1 - p) + 0.0015 * p
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        m = np.max(G)
        W = np.exp(K * (G - m))
        # sparsify far pairs
        W[G < (m - delta)] = 0.0
        np.fill_diagonal(W, 0.0)
        grad = W @ x
        dot = np.sum(grad * x, axis=1, keepdims=True)
        gt = grad - dot * x
        nrm = float(np.mean(np.sqrt(np.sum(gt * gt, axis=1))) + 1e-12)
        if nrm < 1e-15:
            continue
        gt = gt / nrm
        v = mu * v + (1.0 - mu) * gt
        x = x - lr * v
        nr = np.linalg.norm(x, axis=1, keepdims=True) + 1e-16
        x = x / nr
    return x


def _refine_holes(remaining, new_pts, steps=200, mu=0.85):
    k = new_pts.shape[0]
    if k == 0:
        return new_pts
    n_rem = remaining.shape[0]
    v = np.zeros_like(new_pts)
    for t in range(steps):
        p = t / max(steps - 1, 1)
        K = 30.0 + (200.0 - 30.0) * (p * p)
        lr = 0.01 * (1 - p) + 0.001 * p
        all_pts = np.vstack([remaining, new_pts])
        G = new_pts @ all_pts.T
        # mask self dot (1.0) -> ignore
        for i in range(k):
            G[i, n_rem + i] = -2.0
        mi = np.max(G, axis=1, keepdims=True)
        W = np.exp(K * (G - mi))
        W[W < 1e-4] = 0.0
        grad = W @ all_pts
        dot = np.sum(grad * new_pts, axis=1, keepdims=True)
        gt = grad - dot * new_pts
        nrm = float(np.mean(np.sqrt(np.sum(gt * gt, axis=1))) + 1e-12)
        if nrm < 1e-15:
            continue
        gt = gt / nrm
        v = mu * v + (1.0 - mu) * gt
        new_pts = new_pts - lr * v
        new_pts = new_pts / (np.linalg.norm(new_pts, axis=1, keepdims=True) + 1e-16)
    return new_pts


def _lp_polish(x, deadline):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    iu = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        G = x @ x.T
        c = G[iu]
        m = float(np.max(c))
        sel_thr = m - max(20.0 * radius, 1e-9)
        sel = c > sel_thr
        if np.count_nonzero(sel) == 0:
            radius *= 0.5
            continue
        I = iu[0][sel]
        J = iu[1][sel]
        p = len(I)
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
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
        y = y / (np.linalg.norm(y, axis=1, keepdims=True) + 1e-16)
        vv = _max_cos(y)
        if vv < best:
            x, best = y, vv
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x


def solve():
    t0 = time.time()
    deadline = t0 + 178.0
    search_end = t0 + 143.0
    rng = np.random.default_rng(0)

    # load starting point
    best = None
    best_val = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn).astype(float)
            if a.shape == (153, 4):
                a = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-16)
                v = _max_cos(a)
                if v < best_val:
                    best_val = v
                    best = a.copy()
        except Exception:
            pass
    if best is None:
        best = rng.normal(size=(153, 4))
        best = best / np.linalg.norm(best, axis=1, keepdims=True)
        best_val = _max_cos(best)

    sigmas = [0.015, 0.03, 0.06, 0.12, 0.20]
    trial_idx = 0
    # main basin hopping, always from current best (elitist)
    while time.time() < search_end and trial_idx < 250:
        trial_idx += 1
        if trial_idx % 6 == 0:
            # hole re-insertion: remove worst points, sample holes
            G = best @ best.T
            np.fill_diagonal(G, -2.0)
            row_max = np.max(G, axis=1)
            worst_k = 2 if (trial_idx % 12 == 0) else 1
            worst = np.argsort(row_max)[-worst_k:]
            mask = np.ones(153, dtype=bool)
            mask[worst] = False
            remaining = best[mask]
            S = 15000
            cand = rng.normal(size=(S, 4))
            cand = cand / (np.linalg.norm(cand, axis=1, keepdims=True) + 1e-16)
            s_max = cand @ remaining.T
            s_max = np.max(s_max, axis=1)
            order = np.argsort(s_max)
            chosen = []
            for oi in order:
                if len(chosen) >= worst_k:
                    break
                c = cand[oi]
                ok = True
                for cj in chosen:
                    if abs(float(c @ cand[cj])) > 0.92:
                        ok = False
                        break
                if ok:
                    chosen.append(int(oi))
            new_pts = cand[np.array(chosen, dtype=int)]
            new_pts = _refine_holes(remaining, new_pts, steps=200)
            trial = np.vstack([remaining, new_pts])
            # tiny jitter
            trial = trial + 1e-3 * rng.normal(size=trial.shape)
            trial = trial / (np.linalg.norm(trial, axis=1, keepdims=True) + 1e-16)
            steps = 600
        else:
            sigma = sigmas[trial_idx % len(sigmas)]
            if trial_idx % 3 == 2:
                trial = best.copy()
                idx = rng.choice(153, size=8, replace=False)
                trial[idx] = trial[idx] + (sigma * 2.0) * rng.normal(size=(8, 4))
                trial = trial / (np.linalg.norm(trial, axis=1, keepdims=True) + 1e-16)
            else:
                trial = best + sigma * rng.normal(size=best.shape)
                trial = trial / (np.linalg.norm(trial, axis=1, keepdims=True) + 1e-16)
            steps = 550 if sigma > 0.05 else 650

        trial_opt = _smooth_descent(trial, steps=steps)
        v = _max_cos(trial_opt)
        if v < best_val - 1e-13:
            best = trial_opt
            best_val = v
        # time guard
        if time.time() > search_end:
            break

    # final LP polish on best with remaining time
    try:
        best = _lp_polish(best, deadline - 1.0)
    except Exception:
        pass
    best = best / (np.linalg.norm(best, axis=1, keepdims=True) + 1e-16)
    # safety: remove NaN
    best = np.nan_to_num(best, nan=0.0, posinf=0.0, neginf=0.0)
    nr = np.linalg.norm(best, axis=1, keepdims=True) + 1e-16
    # avoid zero rows
    for i in range(best.shape[0]):
        if nr[i, 0] < 1e-12:
            vv = rng.normal(size=(4,))
            vv = vv / (np.linalg.norm(vv) + 1e-16)
            best[i] = vv
            nr[i, 0] = 1.0
    best = best / nr
    return best
