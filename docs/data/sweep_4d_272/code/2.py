import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _normalize(x):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    x = np.where(np.isfinite(x), x, 0.0)
    nrm = np.linalg.norm(x, axis=1, keepdims=True)
    bad = (nrm[:, 0] < 1e-12)
    if np.any(bad):
        x[bad] = np.array([1.0, 0.0, 0.0, 0.0])
        x = x / np.linalg.norm(x, axis=1, keepdims=True)
    return x

def _lp_polish(x, seconds):
    x = _normalize(x)
    n, d = x.shape
    best = _max_cos(x)
    radius, start = 1e-3, time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        xx = x @ x.T
        c = xx[iu]
        m = c.max()
        sel = c > m - max(20 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p, nv = len(I), n * d + 1
        if p == 0:
            break
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv); obj[-1] = 1.0
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
        y = _normalize(y)
        v = _max_cos(y)
        if v < best:
            x, best, radius = y, v, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def _soft_run(y, beta, steps, lr):
    n = y.shape[0]
    for _ in range(steps):
        G = y @ y.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        # weights focused on near-max pairs
        W = np.exp(beta * (G - m))
        np.fill_diagonal(W, 0.0)
        # sparsify tiny weights for speed/stability
        # (keep at least near contacts)
        W[W < 1e-9] = 0.0
        grad = W @ y
        proj = np.sum(grad * y, axis=1, keepdims=True)
        gt = grad - proj * y
        nrm = float(np.mean(np.linalg.norm(gt, axis=1)))
        if not np.isfinite(nrm) or nrm < 1e-12:
            break
        y = y - (lr / nrm) * gt
        y = _normalize(y)
    return y

def _best_void(others, rng, n_cand=4000):
    # find point on S^3 minimizing max dot to others
    cand = rng.standard_normal((n_cand, 4))
    cand = cand / np.linalg.norm(cand, axis=1, keepdims=True)
    s = cand @ others.T
    scores = s.max(axis=1)
    bi = int(np.argmin(scores))
    b = cand[bi].copy()
    bs = float(scores[bi])
    # local hill climb
    step = 0.05
    for _ in range(300):
        prop = b + step * rng.standard_normal((32, 4))
        prop = prop / np.linalg.norm(prop, axis=1, keepdims=True)
        sc = (prop @ others.T).max(axis=1)
        j = int(np.argmin(sc))
        if sc[j] < bs:
            b, bs = prop[j], float(sc[j])
            step *= 1.05
        else:
            step *= 0.97
        if step < 1e-5:
            break
    return b

def solve():
    t0 = time.time()
    TIME_LIMIT = 172.0
    rng = np.random.default_rng(7)
    try:
        x0 = np.load("best.npy")
    except Exception:
        try:
            x0 = np.load("record.npy")
        except Exception:
            x0 = rng.standard_normal((272, 4))
    x0 = _normalize(np.asarray(x0, dtype=float))
    if x0.shape != (272, 4):
        # fallback: tile/resize deterministically
        tmp = rng.standard_normal((272, 4))
        x0 = _normalize(tmp)
    best = x0.copy()
    best_val = _max_cos(best)
    cur = best.copy()
    cur_val = best_val

    # precompute personal worst helper
    def personal_worst(x):
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        return G.max(axis=1)

    it = 0
    # reserve ~25s for final LP
    while time.time() - t0 < TIME_LIMIT - 28.0:
        it += 1
        remain = TIME_LIMIT - 28.0 - (time.time() - t0)
        if remain <= 0:
            break
        mode = it % 4
        y = cur.copy()
        n = 272
        if mode == 0:
            # single worst-point void reinsertion (topology change)
            pw = personal_worst(y)
            # pick among 5 worst randomly to diversify
            order = np.argsort(-pw)
            cand_idx = order[:8]
            wi = int(cand_idx[rng.integers(0, len(cand_idx))])
            others = np.delete(y, wi, axis=0)
            b = _best_void(others, rng, n_cand=5000)
            y[wi] = b
            # short soft relax with increasing sharpness
            y = _soft_run(y, 50.0, 120, 0.008)
            y = _soft_run(y, 120.0, 120, 0.004)
        elif mode == 1:
            # shake a cluster of overloaded points
            pw = personal_worst(y)
            order = np.argsort(-pw)
            k = int(rng.integers(5, 13))
            idx = order[:k * 2]
            sel = rng.choice(idx, size=k, replace=False)
            sigma = float(rng.uniform(0.05, 0.16))
            y[sel] += sigma * rng.standard_normal((k, 4))
            y = _normalize(y)
            y = _soft_run(y, 40.0, 150, 0.010)
            y = _soft_run(y, 100.0, 150, 0.005)
        elif mode == 2:
            # global small shake + anneal
            sigma = float(rng.uniform(0.008, 0.03))
            y = y + sigma * rng.standard_normal(y.shape)
            y = _normalize(y)
            y = _soft_run(y, 60.0, 130, 0.007)
            y = _soft_run(y, 150.0, 130, 0.0035)
        else:
            # double reinsertion: remove worst + random low-contact
            G = y @ y.T
            np.fill_diagonal(G, -2.0)
            m = G.max()
            contacts = np.sum(G > m - 1e-9, axis=1)
            pw = G.max(axis=1)
            w1 = int(np.argmax(pw))
            # rattler or low-contact random
            low = np.where(contacts <= 4)[0]
            if len(low) > 0:
                w2 = int(low[rng.integers(0, len(low))])
            else:
                w2 = int(rng.integers(0, n))
            if w2 == w1:
                w2 = (w1 + 137) % n
            keep = [i for i in range(n) if i != w1 and i != w2]
            others = y[keep]
            b1 = _best_void(others, rng, n_cand=3000)
            # insert first, then second conditioned on first
            tmp = np.vstack([others, b1.reshape(1, 4)])
            b2 = _best_void(tmp, rng, n_cand=3000)
            y[w1] = b1
            y[w2] = b2
            y = _soft_run(y, 50.0, 120, 0.008)
            y = _soft_run(y, 130.0, 120, 0.004)

        v = _max_cos(y)
        # acceptance: improve best, or sideways walk
        if v < best_val:
            best_val = v
            best = y.copy()
            cur, cur_val = y.copy(), v
        elif v < cur_val + 0.004 - 0.003 * min(1.0, (time.time() - t0) / 140.0):
            # accept neutral/worse to explore new basin
            cur, cur_val = y.copy(), v
        # opportunistic quick LP if promising (within 0.002 of best)
        if v < best_val + 0.002 and (it % 7 == 0):
            if time.time() - t0 < TIME_LIMIT - 35.0:
                z = _lp_polish(y, 2.5)
                zv = _max_cos(z)
                if zv < best_val:
                    best_val = zv
                    best = z.copy()
                    cur, cur_val = z.copy(), zv
                elif zv < cur_val:
                    cur, cur_val = z.copy(), zv

    # final tightening of global best with remaining time
    rem = TIME_LIMIT - (time.time() - t0) - 2.0
    if rem > 4.0:
        rem = min(rem, 25.0)
        best = _lp_polish(best, rem)
        best_val = _max_cos(best)
    # final soft micro-polish to remove numerical slack
    best = _soft_run(best, 200.0, 60, 0.0015)
    best = _normalize(best)
    # safety: ensure strictly valid
    return best
