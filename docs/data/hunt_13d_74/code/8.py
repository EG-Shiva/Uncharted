import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _normalize(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return X / n

def _lp_polish(x, seconds, t0, deadline):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and time.time() < deadline and radius > 1e-14:
        G = x @ x.T
        c = G[iu]
        m = c.max()
        sel = c > m - max(20 * radius, 1e-9)
        if np.count_nonzero(sel) > 2500:
            thr = np.sort(c)[-2500]
            sel = c >= thr
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        if p == 0:
            break
        nv = n * d + 1
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
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        v = _max_cos(y)
        if v < best:
            x, best, radius = y, v, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x, best

def solve():
    t0 = time.time()
    DEADLINE = t0 + 174.0
    rng = np.random.default_rng(0)
    cands = []
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=np.float64)
            if a.shape == (74, 13):
                cands.append(_normalize(a))
        except Exception:
            pass
    if not cands:
        cands.append(_normalize(rng.standard_normal((74, 13))))
    vals = [_max_cos(a) for a in cands]
    best = cands[int(np.argmin(vals))].copy()
    best_val = float(min(vals))
    n, d = 74, 13

    def refine_lse(X, schedule):
        m = np.zeros_like(X); v = np.zeros_like(X); cnt = 0
        for tau, steps, lr0 in schedule:
            for s in range(steps):
                if time.time() > DEADLINE - 6:
                    return X
                G = X @ X.T
                np.fill_diagonal(G, -10.0)
                gmax = float(G.max())
                E = np.exp((G - gmax) / tau)
                np.fill_diagonal(E, 0.0)
                S = float(E.sum())
                if not np.isfinite(S) or S <= 0:
                    continue
                Gr = (E @ X) / S
                dot = (Gr * X).sum(axis=1, keepdims=True)
                Gt = Gr - dot * X
                m[:] = 0.9 * m + 0.1 * Gt
                v[:] = 0.999 * v + 0.001 * (Gt * Gt)
                cnt += 1
                c1 = 1.0 - 0.9 ** cnt
                c2 = 1.0 - 0.999 ** cnt
                mhat = m / c1; vhat = v / c2
                lr = lr0 * (1.0 - 0.5 * s / max(1, steps))
                X -= lr * mhat / (np.sqrt(vhat) + 1e-8)
                nr = np.linalg.norm(X, axis=1, keepdims=True); nr[nr == 0] = 1.0
                X /= nr
        return X

    def greedy(X, steps, lr0):
        cur = _max_cos(X)
        lr = lr0
        for _ in range(steps):
            if time.time() > DEADLINE - 6:
                break
            G = X @ X.T
            np.fill_diagonal(G, -2.0)
            gmax = float(G.max())
            delta = max(3.0 * lr, 5e-7)
            mask = G > (gmax - delta)
            A = mask.astype(np.float64)
            Gr = A @ X
            dd = (Gr * X).sum(axis=1, keepdims=True)
            Gt = Gr - dd * X
            norms = np.linalg.norm(Gt, axis=1, keepdims=True)
            step = np.zeros_like(Gt)
            act = norms[:, 0] > 1e-14
            if not np.any(act):
                break
            step[act] = Gt[act] / (norms[act] + 1e-12) * lr
            Xn = _normalize(X - step)
            nv = _max_cos(Xn)
            if nv < cur:
                X[:] = Xn; cur = nv; lr *= 1.08
            else:
                lr *= 0.5
                if lr < 1e-11:
                    break
        return X, cur

    def worst_indices(X, k):
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        mx = G.max(axis=1)
        return np.argsort(-mx)[:k]

    def farthest_reinsert(X, idx):
        cand = _normalize(rng.standard_normal((2000, 13)))
        S = cand @ X.T
        S[:, idx] = -10.0
        sc = S.max(axis=1)
        j = int(np.argmin(sc))
        X[idx] = cand[j]
        return X

    full_sched = [(0.03, 300, 0.012), (0.01, 600, 0.008), (0.003, 800, 0.005), (0.0012, 600, 0.003)]
    fast_sched = [(0.01, 350, 0.008), (0.003, 500, 0.005), (0.0012, 350, 0.003)]
    tiny_sched = [(0.003, 250, 0.004), (0.001, 250, 0.002)]

    # basin 0: direct squeeze of incumbent
    try:
        X0 = best.copy()
        X0 = refine_lse(X0, full_sched)
        X0, v0 = greedy(X0, 400, 0.002)
        if v0 < best_val:
            best, best_val = X0.copy(), float(v0)
    except Exception:
        pass

    basin = 0
    sigmas = [0.015, 0.03, 0.06, 0.12, 0.008, 0.2, 0.04, 0.10]
    while time.time() < DEADLINE - 12:
        basin += 1
        remain = DEADLINE - 12 - (time.time() - t0)
        if remain < 12:
            sched = tiny_sched
        elif remain < 35:
            sched = fast_sched
        else:
            sched = full_sched if (basin % 3 != 0) else fast_sched
        Xcur = best.copy()
        mode = basin % 7
        try:
            if mode == 0:
                sigma = sigmas[basin % len(sigmas)]
                Xcur = _normalize(Xcur + sigma * rng.standard_normal(Xcur.shape))
            elif mode == 1:
                G = Xcur @ Xcur.T; np.fill_diagonal(G, -2.0)
                gmax = float(G.max())
                thr = gmax - 0.03
                cntc = (G > thr).sum(axis=1)
                hot = np.where(cntc > 0)[0]
                if len(hot) == 0:
                    hot = worst_indices(Xcur, 4)
                sigma = sigmas[basin % len(sigmas)] * 2.0
                noise = rng.standard_normal(Xcur.shape)
                Xcur = Xcur + 0.3 * sigma * rng.standard_normal(Xcur.shape)
                Xcur[hot] += sigma * noise[hot] * 2.0
                Xcur = _normalize(Xcur)
            elif mode == 2:
                # kill 1-2 worst points, farthest reinsert
                k = 1 if (basin % 2 == 0) else 2
                wi = worst_indices(Xcur, k)
                for idx in wi:
                    farthest_reinsert(Xcur, int(idx))
                # small shake of rest
                Xcur = _normalize(Xcur + 0.005 * rng.standard_normal(Xcur.shape))
            elif mode == 3:
                # direct worst-pair repulsion: push max pair apart
                G = Xcur @ Xcur.T; np.fill_diagonal(G, -2.0)
                ij = np.unravel_index(int(np.argmax(G)), G.shape)
                i, j = int(ij[0]), int(ij[1])
                eps = 0.02 + 0.02 * rng.random()
                # tangential components
                xi, xj = Xcur[i].copy(), Xcur[j].copy()
                ti = xj - (xj @ xi) * xi; tj = xi - (xi @ xj) * xj
                ti /= (np.linalg.norm(ti) + 1e-12); tj /= (np.linalg.norm(tj) + 1e-12)
                Xcur[i] = _normalize((xi - eps * ti)[None, :])[0]
                Xcur[j] = _normalize((xj - eps * tj)[None, :])[0]
                Xcur = _normalize(Xcur + 0.008 * rng.standard_normal(Xcur.shape))
            elif mode == 4:
                sigma = 0.05 + 0.05 * rng.random()
                k = int(rng.integers(5, 15))
                idx = rng.choice(n, size=k, replace=False)
                Xcur[idx] = _normalize(rng.standard_normal((k, d)) * 1.0)
                # keep rest, small shake
                mask = np.ones(n, bool); mask[idx] = False
                Xcur[mask] = _normalize(Xcur[mask] + 0.005 * rng.standard_normal(((n - k), d)))
            elif mode == 5:
                # coulomb-style random restart smoothed: heavy shake
                sigma = sigmas[basin % len(sigmas)]
                Xcur = _normalize(Xcur + sigma * rng.standard_normal(Xcur.shape))
                # extra shake of top-6 worst
                wi = worst_indices(Xcur, 6)
                Xcur[wi] = _normalize(Xcur[wi] + 0.05 * rng.standard_normal((len(wi), d)))
            else:
                Xcur = _normalize(rng.standard_normal((n, d)) * 0.3 + 0.7 * Xcur)
            Xcur = refine_lse(Xcur, sched)
            Xcur, vcur = greedy(Xcur, 300, 0.002)
            if vcur < best_val - 1e-12:
                best, best_val = Xcur.copy(), float(vcur)
        except Exception:
            continue

    # final LP polish of global best with remaining time
    try:
        rem = DEADLINE - time.time() - 1.0
        if rem > 3:
            best, best_val2 = _lp_polish(best, min(rem, 28.0), t0, DEADLINE - 0.5)
            best_val = float(min(best_val, best_val2))
            # one more greedy after LP to clean
            if time.time() < DEADLINE - 2:
                best, vv = greedy(best, 150, 0.0005)
                if vv < best_val:
                    best_val = float(vv)
    except Exception:
        pass
    best = _normalize(best)
    return best
