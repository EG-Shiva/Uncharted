import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())


def _lp_polish(x, seconds):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    try:
        best = _max_cos(x)
    except Exception:
        return x
    radius = 1e-3
    deadline = time.time() + seconds
    iu = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        c = (x @ x.T)[iu]
        m = float(c.max())
        sel = c > m - max(20.0 * radius, 1e-9)
        I = iu[0][sel]
        J = iu[1][sel]
        p = len(I)
        if p == 0:
            break
        nv = n * d + 1
        cols = np.concatenate(
            [I[:, None] * d + np.arange(d),
             J[:, None] * d + np.arange(d),
             np.full((p, 1), n * d)], axis=1
        ).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], axis=1).ravel()
        rows = np.repeat(np.arange(p), 2 * d + 1)
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix(
                (x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))),
                shape=(n, nv)).tocsr()
            obj = np.zeros(nv)
            obj[-1] = 1.0
            res = linprog(
                obj, A_ub=A_ub, b_ub=-c[sel],
                A_eq=A_eq, b_eq=np.zeros(n),
                bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        nr = np.linalg.norm(y, axis=1, keepdims=True)
        nr[nr == 0] = 1.0
        y = y / nr
        v = _max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x


def _smooth_anneal(v0, steps, T0, T1, lr0, lr1, mu, rng, track_best=True):
    v = v0.copy()
    vel = np.zeros_like(v)
    best = v.copy()
    best_val = _max_cos(v)
    n = v.shape[0]
    for s in range(steps):
        frac = s / (steps - 1) if steps > 1 else 0.0
        T = T0 * (T1 / T0) ** frac
        lr = lr0 * (lr1 / lr0) ** frac
        G = v @ v.T
        np.fill_diagonal(G, -2.0)
        gmax = float(G.max())
        # truncated softmax: ignore pairs >5T below max for focus/speed
        thr = gmax - 5.0 * T
        M = G > thr
        E = np.exp((G - gmax) / T) * M
        np.fill_diagonal(E, 0.0)
        S = float(E.sum())
        if not np.isfinite(S) or S <= 0:
            break
        W = E / S
        grad = W @ v
        dot = np.sum(grad * v, axis=1, keepdims=True)
        tg = grad - dot * v
        vel = mu * vel + tg
        v = v - lr * vel
        nr = np.linalg.norm(v, axis=1, keepdims=True)
        nr[nr == 0] = 1.0
        v /= nr
        if track_best and ((s % 150 == 149) or s == steps - 1):
            cv = _max_cos(v)
            if cv < best_val:
                best_val = cv
                best = v.copy()
    return best if track_best else v


def _contact_refine(v0, iters, lr0, lr1, tol0, tol1, mu=0.85):
    v = v0.copy()
    vel = np.zeros_like(v)
    best = v.copy()
    best_val = _max_cos(v)
    for s in range(iters):
        frac = s / (iters - 1) if iters > 1 else 0.0
        lr = lr0 * (lr1 / lr0) ** frac
        tol = tol0 * (tol1 / tol0) ** frac
        G = v @ v.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        A = G - (m - tol)
        A[A < 0] = 0.0
        np.fill_diagonal(A, 0.0)
        # quadratic weighting focuses on worst pairs
        A = A * A
        S = float(A.sum())
        if S <= 0 or not np.isfinite(S):
            continue
        # scale to keep step size comparable
        A = A / (S / v.shape[0] + 1e-12)
        grad = A @ v
        dot = np.sum(grad * v, axis=1, keepdims=True)
        tg = grad - dot * v
        # clip extreme tangents
        nrm = np.linalg.norm(tg, axis=1, keepdims=True)
        cap = 2.0
        scale = np.ones_like(nrm)
        mask = nrm > cap
        scale[mask] = cap / (nrm[mask] + 1e-12)
        tg = tg * scale
        vel = mu * vel + tg
        v = v - lr * vel
        nr = np.linalg.norm(v, axis=1, keepdims=True)
        nr[nr == 0] = 1.0
        v /= nr
        if (s % 100 == 99) or s == iters - 1:
            cv = _max_cos(v)
            if cv < best_val:
                best_val = cv
                best = v.copy()
    return best


def solve():
    t_start = time.time()
    TIME_LIMIT = 172.0
    FINAL_POLISH = 18.0
    rng = np.random.default_rng(12345)
    n, d = 153, 4

    cands = []
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=float)
            if a.shape == (n, d):
                a = a / np.linalg.norm(a, axis=1, keepdims=True)
                if np.all(np.isfinite(a)):
                    cands.append(a)
        except Exception:
            pass
    if not cands:
        z = rng.standard_normal((n, d))
        z /= np.linalg.norm(z, axis=1, keepdims=True)
        cands.append(z)

    best = None
    best_val = 1e9
    for c in cands:
        v = _max_cos(c)
        if v < best_val:
            best_val = v
            best = c.copy()
    cur = best.copy()
    cur_val = best_val

    it = 0
    # quick initial LP to get true local value (short)
    # do not spend too much here
    best = _lp_polish(best, seconds=2.0)
    best_val = _max_cos(best)
    cur = best.copy()
    cur_val = best_val

    while time.time() - t_start < TIME_LIMIT - FINAL_POLISH - 1.0:
        it += 1
        if time.time() - t_start > TIME_LIMIT - FINAL_POLISH - 1.0:
            break
        # pick base
        if rng.random() < 0.82:
            base = best
        else:
            base = cur
        y = base.copy()
        mode = it % 7

        if mode == 0:
            sigma = 0.012 + 0.012 * float(rng.random())
            y = y + sigma * rng.standard_normal(y.shape)
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            T0, T1 = 0.018, 0.0022
            SM_STEPS, CR_IT = 650, 350
        elif mode == 1:
            k = int(8 + rng.integers(0, 8))
            idx = rng.choice(n, size=k, replace=False)
            sigma = 0.12 + 0.08 * float(rng.random())
            y[idx] += sigma * rng.standard_normal((k, d))
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            T0, T1 = 0.025, 0.0025
            SM_STEPS, CR_IT = 700, 350
        elif mode == 2:
            # overloaded cluster coherent shift
            G = y @ y.T
            np.fill_diagonal(G, -2.0)
            m = float(G.max())
            cnt = (G > m - 0.004).sum(axis=1)
            order = np.argsort(cnt)
            k = 10
            idx = order[-k:]
            # BFS expansion around worst node through contact graph
            adj = G > m - 0.003
            seed = int(idx[-1])
            comp = set([seed])
            frontier = [seed]
            while frontier and len(comp) < 16:
                u = frontier.pop()
                nbrs = np.where(adj[u])[0]
                for w in nbrs:
                    if w not in comp and len(comp) < 16:
                        comp.add(int(w))
                        frontier.append(int(w))
            idx2 = np.array(sorted(comp), dtype=int)
            common = rng.standard_normal((d,)) * (0.10 + 0.08 * float(rng.random()))
            y[idx2] += common + 0.03 * rng.standard_normal((len(idx2), d))
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            T0, T1 = 0.02, 0.002
            SM_STEPS, CR_IT = 700, 400
        elif mode == 3:
            # bottleneck pair splitting
            G = y @ y.T
            np.fill_diagonal(G, -2.0)
            # top few pairs
            iu = np.triu_indices(n, 1)
            flat = G[iu]
            k = 5
            top = np.argpartition(-flat, k)[:k]
            eps = 0.06
            for t in top:
                a = int(iu[0][t]); b = int(iu[1][t])
                diff = y[a] - y[b]
                diff = diff - 0.5 * (float(diff @ y[a]) * y[a] + float(diff @ y[b]) * y[b])
                nd = float(np.linalg.norm(diff))
                if nd < 1e-12:
                    diff = rng.standard_normal((d,))
                    nd = float(np.linalg.norm(diff))
                diff /= nd
                y[a] += eps * diff
                y[b] -= eps * diff
            y += 0.008 * rng.standard_normal(y.shape)
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            T0, T1 = 0.016, 0.0018
            SM_STEPS, CR_IT = 650, 400
        elif mode == 4:
            # rattler / void reinsertion
            G = y @ y.T
            np.fill_diagonal(G, -2.0)
            m = float(G.max())
            cnt_tight = (G > m - 1e-7).sum(axis=1)
            ratt = np.where(cnt_tight == 0)[0]
            if len(ratt) == 0:
                cnt2 = (G > m - 0.004).sum(axis=1)
                idx = np.argsort(cnt2)[-3:]
            else:
                nr = min(len(ratt), 1 + int(rng.integers(0, 2)))
                idx = rng.choice(ratt, size=nr, replace=False)
            for j in idx:
                cand = rng.standard_normal((600, d))
                cand /= np.linalg.norm(cand, axis=1, keepdims=True)
                mask = np.ones(n, dtype=bool)
                mask[j] = False
                others = y[mask]
                scores = (cand @ others.T).max(axis=1)
                y[j] = cand[np.argmin(scores)]
            y += 0.006 * rng.standard_normal(y.shape)
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            T0, T1 = 0.015, 0.0018
            SM_STEPS, CR_IT = 600, 350
        elif mode == 5:
            # large kick: randomize a block
            k = int(12 + rng.integers(0, 10))
            idx = rng.choice(n, size=k, replace=False)
            y[idx] = rng.standard_normal((k, d))
            y += 0.02 * rng.standard_normal(y.shape)
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            T0, T1 = 0.035, 0.003
            SM_STEPS, CR_IT = 800, 350
            if it % 14 == 0:
                # full-ish shake occasionally
                y = y + 0.25 * rng.standard_normal(y.shape)
                y /= np.linalg.norm(y, axis=1, keepdims=True)
                T0 = 0.045
        else:
            sigma = 0.03 + 0.03 * float(rng.random())
            y = y + sigma * rng.standard_normal(y.shape)
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            T0, T1 = 0.022, 0.0022
            SM_STEPS, CR_IT = 700, 350

        # anneal
        y2 = _smooth_anneal(y, steps=SM_STEPS, T0=T0, T1=T1,
                            lr0=0.55, lr1=0.05, mu=0.9, rng=rng, track_best=True)
        y2 = _contact_refine(y2, iters=CR_IT, lr0=0.015, lr1=0.0015,
                             tol0=0.004, tol1=0.0004, mu=0.85)
        v2 = _max_cos(y2)

        # selective LP: only polish promising basins
        if v2 < best_val + 0.0015:
            # check time
            if time.time() - t_start < TIME_LIMIT - FINAL_POLISH - 2.0:
                y2p = _lp_polish(y2, seconds=0.7)
                v2p = _max_cos(y2p)
                if v2p < v2:
                    y2, v2 = y2p, v2p

        if v2 < best_val - 1e-12:
            best = y2
            best_val = v2
            cur = y2
            cur_val = v2
        else:
            Temp = 0.0004
            if v2 < cur_val or float(rng.random()) < float(np.exp((cur_val - v2) / Temp)):
                cur = y2
                cur_val = v2

    remain = TIME_LIMIT - (time.time() - t_start) - 0.8
    if remain < 2.0:
        remain = 2.0
    remain = min(remain, 178.0 - (time.time() - t_start) - 0.8)
    if remain > 0.5:
        best = _lp_polish(best, seconds=float(remain))
    best = best / np.linalg.norm(best, axis=1, keepdims=True)
    return best
