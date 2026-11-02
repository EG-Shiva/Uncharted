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

def _lp_polish(X, best, t0, deadline):
    n, d = X.shape
    iu = np.triu_indices(n, 1)
    radius = 1.2e-3
    while time.time() < deadline and radius > 1e-13:
        G = X @ X.T
        cfull = G[iu]
        m = float(cfull.max())
        sel = cfull > m - max(25.0 * radius, 2e-9)
        cnt = int(np.count_nonzero(sel))
        if cnt > 2200:
            thr = np.sort(cfull)[-2200]
            sel = cfull >= thr
            cnt = int(np.count_nonzero(sel))
        if cnt == 0:
            break
        I = iu[0][sel]; J = iu[1][sel]
        cc = cfull[sel]
        p = cnt
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([X[J], X[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((X.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        except Exception:
            radius *= 0.5
            continue
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-cc, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs",
                          options={"time_limit": 4.0})
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        Y = X + res.x[:-1].reshape(n, d)
        Y = _normalize(Y)
        v = _max_cos(Y)
        if v < best:
            X = Y
            best = v
            radius = min(radius * 1.6, 0.02)
        else:
            radius *= 0.5
    return X, best

def solve():
    t0 = time.time()
    LIM = 172.0
    DEAD = t0 + LIM
    rng = np.random.default_rng(12345)
    n, d = 74, 13

    cands = []
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=np.float64)
            if a.shape == (n, d):
                cands.append(_normalize(a))
        except Exception:
            pass
    if not cands:
        cands.append(_normalize(rng.standard_normal((n, d))))
    vals = [_max_cos(a) for a in cands]
    bi = int(np.argmin(vals))
    best = cands[bi].copy()
    best_val = float(vals[bi])

    def refine_lse(X, sched, m, v, cnt):
        for tau, steps, lr0 in sched:
            for s in range(steps):
                if time.time() > DEAD - 22:
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
                cnt[0] += 1
                c1 = 1.0 - 0.9 ** cnt[0]
                c2 = 1.0 - 0.999 ** cnt[0]
                mhat = m / c1
                vhat = v / c2
                lr = lr0 * (1.0 - 0.5 * s / max(1, steps))
                X -= lr * mhat / (np.sqrt(vhat) + 1e-8)
                nr = np.linalg.norm(X, axis=1, keepdims=True)
                nr[nr == 0] = 1.0
                X /= nr
        return X

    def greedy(X, steps, lr0):
        cur = _max_cos(X)
        lr = lr0
        for _ in range(steps):
            if time.time() > DEAD - 22:
                break
            G = X @ X.T
            np.fill_diagonal(G, -2.0)
            gmax = float(G.max())
            delta = max(3.0 * lr, 6e-7)
            mask = G > (gmax - delta)
            Gr = mask.astype(np.float64) @ X
            dd = (Gr * X).sum(axis=1, keepdims=True)
            Gt = Gr - dd * X
            norms = np.linalg.norm(Gt, axis=1, keepdims=True)
            act = norms[:, 0] > 1e-14
            if not np.any(act):
                break
            step = np.zeros_like(Gt)
            step[act] = Gt[act] / (norms[act] + 1e-12) * lr
            Xn = _normalize(X - step)
            nv = _max_cos(Xn)
            if nv < cur:
                X[:] = Xn
                cur = nv
                lr *= 1.08
            else:
                lr *= 0.5
                if lr < 1e-11:
                    break
        return X, cur

    def worst_order(X):
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        return np.argsort(-G.max(axis=1))

    def farthest_reinsert(X, idx):
        cand = _normalize(rng.standard_normal((2000, d)))
        S = cand @ X.T
        S[:, idx] = -10.0
        sc = S.max(axis=1)
        j = int(np.argmin(sc))
        X[idx] = cand[j]
        return X

    full_sched = [(0.02, 300, 0.011), (0.007, 500, 0.007), (0.0022, 600, 0.004), (0.0009, 400, 0.0025)]
    fast_sched = [(0.008, 350, 0.007), (0.0025, 500, 0.0045), (0.0009, 350, 0.0025)]
    tiny_sched = [(0.0025, 250, 0.004), (0.0009, 250, 0.002)]

    # initial squeeze of incumbent
    X0 = best.copy()
    m = np.zeros_like(X0); vv = np.zeros_like(X0); cc = [0]
    X0 = refine_lse(X0, full_sched, m, vv, cc)
    X0, v0 = greedy(X0, 350, 0.002)
    if v0 < best_val:
        best, best_val = X0.copy(), float(v0)

    basin = 0
    while time.time() < DEAD - 22:
        remain = DEAD - 22 - (time.time() - t0)
        if remain < 12:
            sched = tiny_sched
        elif remain < 35:
            sched = fast_sched
        else:
            sched = fast_sched if (basin % 3 == 0) else full_sched
        Xcur = best.copy()
        mode = basin % 7
        if mode == 0:
            Xcur = _normalize(Xcur + 0.016 * rng.standard_normal(Xcur.shape))
        elif mode == 1:
            wo = worst_order(Xcur)
            k = 6 + (basin % 5)
            idx = wo[:k]
            noise = 0.0035 * rng.standard_normal(Xcur.shape)
            noise[idx] += 0.028 * rng.standard_normal((len(idx), d))
            Xcur = _normalize(Xcur + noise)
        elif mode == 2:
            iw = int(worst_order(Xcur)[0])
            Xcur = farthest_reinsert(Xcur, iw)
            Xcur = _normalize(Xcur + 0.004 * rng.standard_normal(Xcur.shape))
        elif mode == 3:
            # directed worst-pair split + small noise
            G = Xcur @ Xcur.T
            np.fill_diagonal(G, -2.0)
            ij = int(np.argmax(G))
            ii, jj = divmod(ij, n)
            Xcur[ii] -= 0.012 * Xcur[jj]
            Xcur[jj] -= 0.012 * Xcur[ii]
            Xcur = _normalize(Xcur + 0.007 * rng.standard_normal(Xcur.shape))
        elif mode == 4:
            iw = int(worst_order(Xcur)[0])
            Xcur = farthest_reinsert(Xcur, iw)
            iw2 = int(worst_order(Xcur)[0])
            if iw2 != iw:
                Xcur = farthest_reinsert(Xcur, iw2)
            Xcur = _normalize(Xcur + 0.003 * rng.standard_normal(Xcur.shape))
        elif mode == 5:
            # random subset relocation to change topology
            k = 10
            idx = rng.choice(n, size=k, replace=False)
            Xcur[idx] = _normalize(rng.standard_normal((k, d)))
            Xcur = _normalize(Xcur + 0.002 * rng.standard_normal(Xcur.shape))
        else:
            Xcur = _normalize(Xcur + 0.010 * rng.standard_normal(Xcur.shape))

        m = np.zeros_like(Xcur); vv = np.zeros_like(Xcur); cc = [0]
        Xcur = refine_lse(Xcur, sched, m, vv, cc)
        Xcur, vcur = greedy(Xcur, 250, 0.0018)
        if vcur < best_val - 1e-12:
            best = Xcur.copy()
            best_val = float(vcur)
        basin += 1
        if basin > 300:
            break

    # final squeeze
    if time.time() < DEAD - 2:
        Xf = best.copy()
        Xf, vf = greedy(Xf, 250, 0.0007)
        if vf < best_val:
            best, best_val = Xf.copy(), float(vf)
    # trust-region LP polish with remaining time
    if time.time() < DEAD - 1:
        try:
            Xp, vp = _lp_polish(best.copy(), best_val, t0, DEAD - 0.5)
            if vp < best_val:
                best, best_val = Xp, vp
        except Exception:
            pass

    best = _normalize(best)
    return best
