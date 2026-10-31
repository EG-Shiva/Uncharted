import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def _norm(X):
    X = np.asarray(X, dtype=np.float64)
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[~np.isfinite(n)] = 1.0
    n[n < 1e-12] = 1.0
    X = X / n
    bad = ~np.isfinite(X).all(axis=1)
    if np.any(bad):
        X[bad] = np.array([1.0, 0.0, 0.0, 0.0])
        X = X / np.linalg.norm(X, axis=1, keepdims=True)
    return X


def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))


def _contacts(X, tol=1e-9):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    m = float(np.max(G))
    cnt = np.sum(G >= m - tol, axis=1)
    pw = np.max(G, axis=1)
    return m, cnt, pw, G


def _adam_run(X, steps, tau0, tau1, lr0, lr1, mov=None):
    n, d = X.shape
    if mov is None:
        m1 = np.zeros_like(X)
        v1 = np.zeros_like(X)
    else:
        m1 = np.zeros((len(mov), d))
        v1 = np.zeros((len(mov), d))
    b1, b2, eps = 0.9, 0.999, 1e-8
    for s in range(steps):
        frac = s / max(steps - 1, 1)
        tau = tau0 * (tau1 / tau0) ** frac if tau1 != tau0 and tau0 > 0 else tau0
        lr = lr0 + (lr1 - lr0) * frac
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        gm = float(np.max(G))
        # exp weights, avoid overflow
        W = np.exp((G - gm) / max(tau, 1e-12))
        np.fill_diagonal(W, 0.0)
        # sparsify tiny
        W[W < 1e-14] = 0.0
        grad = W @ X
        dot = np.sum(grad * X, axis=1, keepdims=True)
        gt = grad - dot * X
        if mov is None:
            g = gt
            m1 = b1 * m1 + (1 - b1) * g
            v1 = b2 * v1 + (1 - b2) * (g * g)
            bc1 = 1.0 - b1 ** (s + 1)
            bc2 = 1.0 - b2 ** (s + 1)
            step = (m1 / bc1) / (np.sqrt(v1 / bc2) + eps)
            np.clip(step, -2.0, 2.0, out=step)
            X = X - lr * step
            X = _norm(X)
        else:
            g = gt[mov]
            m1 = b1 * m1 + (1 - b1) * g
            v1 = b2 * v1 + (1 - b2) * (g * g)
            bc1 = 1.0 - b1 ** (s + 1)
            bc2 = 1.0 - b2 ** (s + 1)
            step = (m1 / bc1) / (np.sqrt(v1 / bc2) + eps)
            np.clip(step, -2.0, 2.0, out=step)
            X[mov] = X[mov] - lr * step
            nn = np.linalg.norm(X[mov], axis=1, keepdims=True)
            nn[nn < 1e-12] = 1.0
            X[mov] = X[mov] / nn
    return X


def _refine_full(X0, fast=False):
    X = X0.copy()
    if fast:
        X = _adam_run(X, 180, 0.015, 0.004, 0.02, 0.01)
        X = _adam_run(X, 180, 0.004, 0.0012, 0.01, 0.005)
    else:
        X = _adam_run(X, 220, 0.02, 0.005, 0.025, 0.012)
        X = _adam_run(X, 220, 0.005, 0.0015, 0.012, 0.006)
        X = _adam_run(X, 160, 0.0015, 0.0006, 0.006, 0.003)
    return X


def _void_candidates(core, rng, n_cand=6000):
    cand = rng.standard_normal((n_cand, 4))
    cand = cand / np.linalg.norm(cand, axis=1, keepdims=True)
    s = cand @ core.T
    sc = np.max(s, axis=1)
    return cand, sc


def _greedy_refill(Xbase, patch, rng):
    n = Xbase.shape[0]
    X = Xbase.copy()
    mask = np.ones(n, dtype=bool)
    mask[patch] = False
    fixed = X[mask]
    # place sequentially farthest from fixed+placed
    placed = []
    for _ in patch:
        others = fixed if len(placed) == 0 else np.vstack([fixed] + placed)
        cand, sc = _void_candidates(others, rng, n_cand=4000)
        bi = int(np.argmin(sc))
        p = cand[bi] + 0.002 * rng.standard_normal(4)
        p = p / np.linalg.norm(p)
        placed.append(p.reshape(1, 4))
    for idx, p in zip(patch, placed):
        X[idx] = p[0]
    return _norm(X)


def _lp_polish(X, deadline):
    X = _norm(X)
    n, d = X.shape
    best = _max_cos(X)
    radius = 1e-3
    iu = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        if time.time() + 0.8 > deadline and radius < 5e-4:
            break
        cmat = X @ X.T
        c = cmat[iu]
        mcur = float(np.max(c))
        sel = c > mcur - max(20.0 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        if p == 0:
            break
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], axis=1).ravel()
        vals = np.concatenate([X[J], X[I], -np.ones((p, 1))], axis=1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((X.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
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
        Y = X + res.x[:-1].reshape(n, d)
        Y = _norm(Y)
        v = _max_cos(Y)
        if v < best:
            X, best = Y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return X


def solve():
    t0 = time.time()
    T_HARD = t0 + 172.0
    T_EXP = t0 + 132.0
    rng = np.random.default_rng(7)
    try:
        best = np.load("best.npy")
    except Exception:
        try:
            best = np.load("record.npy")
        except Exception:
            best = rng.standard_normal((272, 4))
    best = _norm(np.asarray(best, dtype=np.float64))
    if best.shape != (272, 4):
        best = _norm(rng.standard_normal((272, 4)))
    best_val = _max_cos(best)
    n = 272

    hop = 0
    while time.time() < T_EXP and hop < 500:
        if time.time() + 1.2 > T_EXP:
            break
        mode = hop % 7
        Xp = best.copy()
        m, cnt, pw, G = _contacts(best, tol=1e-7)
        order = np.argsort(-cnt, kind="stable")
        ratt = np.where(cnt == 0)[0]
        if mode == 0:
            Xp = _norm(Xp + 0.015 * rng.standard_normal(Xp.shape))
            Xp = _refine_full(Xp, fast=False)
        elif mode == 1:
            Xp = _norm(Xp + 0.045 * rng.standard_normal(Xp.shape))
            Xp = _refine_full(Xp, fast=True)
        elif mode == 2:
            idx = order[:20]
            Xp = Xp + 0.010 * rng.standard_normal(Xp.shape)
            Xp[idx] += 0.13 * rng.standard_normal((len(idx), 4))
            Xp = _norm(Xp)
            Xp = _refine_full(Xp, fast=True)
        elif mode == 3:
            # coordinated patch: hubs + neighbors + rattlers
            hubs = list(order[:3])
            # nearest neighbors of top hub
            sims = (best @ best[hubs[0]])
            sims[hubs[0]] = -2
            nn = np.argsort(-sims)[:10]
            patch = sorted(set(hubs) | set(nn.tolist()) | set(ratt.tolist()))
            # limit size 8..18
            if len(patch) > 18:
                # keep most extreme
                patch = patch[:18]
            if len(patch) < 6:
                extra = rng.choice(n, size=6 - len(patch), replace=False)
                patch = sorted(set(patch) | set(extra.tolist()))
            patch = np.array(patch, dtype=int)
            Xp = _greedy_refill(best, patch, rng)
            # patch-only anneal then full
            Xp = _adam_run(Xp, 250, 0.012, 0.002, 0.02, 0.008, mov=patch)
            Xp = _adam_run(Xp, 250, 0.004, 0.0008, 0.012, 0.004, mov=None)
        elif mode == 4:
            # rattler to hole + shake worst
            k = max(len(ratt), 3)
            worst = order[:k]
            S, sc = _void_candidates(best, rng, n_cand=6000)
            hole_idx = np.argsort(sc)[:k]
            Xp[worst] = S[hole_idx] + 0.003 * rng.standard_normal((k, 4))
            mask = np.ones(n, dtype=bool)
            mask[worst] = False
            Xp[mask] += 0.008 * rng.standard_normal((n - k, 4))
            Xp = _norm(Xp)
            Xp = _refine_full(Xp, fast=True)
        elif mode == 5:
            k = 7
            idx = rng.choice(n, size=k, replace=False)
            Xp[idx] = rng.standard_normal((k, 4))
            mask = np.ones(n, dtype=bool)
            mask[idx] = False
            Xp[mask] += 0.010 * rng.standard_normal((n - k, 4))
            Xp = _norm(Xp)
            Xp = _refine_full(Xp, fast=True)
        else:
            # balance-seeking: move points with large nearest dist variance
            # push rattlers randomly, shake hubs moderately
            Xp = Xp + 0.025 * rng.standard_normal(Xp.shape)
            if len(ratt) > 0:
                Xp[ratt] = rng.standard_normal((len(ratt), 4))
            Xp = _norm(Xp)
            Xp = _refine_full(Xp, fast=True)

        v = _max_cos(Xp)
        # snap promising basins with quick LP
        if v < best_val + 0.002 and time.time() + 3.0 < T_EXP:
            # short LP snap (2 sec)
            Xp = _lp_polish(Xp, min(time.time() + 1.5, T_EXP))
            v = _max_cos(Xp)
        if v < best_val:
            best = Xp
            best_val = v
        hop += 1

    if time.time() < T_HARD - 3.0:
        best = _lp_polish(best, T_HARD - 1.0)
    return _norm(best)
