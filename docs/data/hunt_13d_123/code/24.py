import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())


def _lp_polish(x, deadline, p_cap=3000):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    try:
        best = _max_cos(x)
    except Exception:
        return x
    best_x = x.copy()
    radius = 1e-3
    iu0, iu1 = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
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
        cols = np.concatenate(
            [(I[:, None] * d + np.arange(d)).reshape(p, d),
             (J[:, None] * d + np.arange(d)).reshape(p, d),
             np.full((p, 1), n * d)], axis=1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], axis=1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            erows = np.repeat(np.arange(n), d)
            ecols = np.arange(n * d)
            A_eq = coo_matrix((x.ravel(), (erows, ecols)), shape=(n, nv)).tocsr()
            obj = np.zeros(nv)
            obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq,
                          b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                          method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        yn = np.linalg.norm(y, axis=1, keepdims=True)
        yn[yn == 0] = 1.0
        y = y / yn
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
        if time.time() >= deadline:
            break
    return best_x


def _smooth_run(y, steps, T0, T1, lr0, lr1, deadline):
    n, d = y.shape
    for s in range(steps):
        if s % 100 == 0 and time.time() >= deadline:
            break
        frac = s / max(steps - 1, 1)
        T = T0 + (T1 - T0) * frac
        lr = lr0 + (lr1 - lr0) * frac
        G = y @ y.T
        np.fill_diagonal(G, -10.0)
        m = float(G.max())
        E = np.exp((G - m) / T)
        np.fill_diagonal(E, 0.0)
        row = E.sum(axis=1, keepdims=True) + 1e-20
        Avg = (E @ y) / row
        dot = (Avg * y).sum(axis=1, keepdims=True)
        Gt = Avg - dot * y
        # clip extreme grads
        y = y - lr * Gt
        yn = np.linalg.norm(y, axis=1, keepdims=True)
        yn[yn == 0] = 1.0
        y = y / yn
    return y


def _best_hole(Yrem, rng, n_cand=3000):
    d = Yrem.shape[1]
    C = rng.normal(size=(n_cand, d))
    C /= np.linalg.norm(C, axis=1, keepdims=True)
    S = C @ Yrem.T
    worst = S.max(axis=1)
    return C[int(np.argmin(worst))]


def solve():
    t0 = time.time()
    DEAD = t0 + 174.0
    rng = np.random.default_rng(0)
    n, d = 123, 13
    X0 = None
    best_val0 = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=float)
            if a.shape == (n, d):
                a = a / np.linalg.norm(a, axis=1, keepdims=True)
                if np.all(np.isfinite(a)):
                    v = _max_cos(a)
                    if v < best_val0:
                        best_val0 = v
                        X0 = a.copy()
        except Exception:
            pass
    if X0 is None:
        X0 = rng.normal(size=(n, d))
        X0 /= np.linalg.norm(X0, axis=1, keepdims=True)
        best_val0 = _max_cos(X0)
    best = X0.copy()
    best_val = float(best_val0)

    it = 0
    # main hopping loop, reserve ~14s for final polish
    while time.time() < DEAD - 14.0:
        it += 1
        if time.time() >= DEAD - 14.0:
            break
        mode = it % 5
        y = best.copy()
        try:
            G0 = best @ best.T
            np.fill_diagonal(G0, -2.0)
            m0 = float(G0.max())
            per_max = G0.max(axis=1)
            cnt = (G0 > m0 - 2e-3).sum(axis=1)
        except Exception:
            per_max = np.zeros(n)
            cnt = np.zeros(n)
        if mode == 0:
            # global small shake
            y = best + rng.normal(scale=0.022, size=(n, d))
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            steps, T0, T1, lr0, lr1 = 600, 0.018, 0.003, 0.18, 0.02
        elif mode == 1:
            # shake worst subset
            k = 8
            idx = np.argsort(-per_max)[:k]
            y[idx] = y[idx] + rng.normal(scale=0.13, size=(k, d))
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            steps, T0, T1, lr0, lr1 = 700, 0.015, 0.0025, 0.16, 0.015
        elif mode == 2:
            # single rattler hole re-insertion
            low = int(np.argmin(cnt))
            Yrem = np.delete(best, low, axis=0)
            h = _best_hole(Yrem, rng, n_cand=3000)
            y = Yrem.copy()
            # reinsert with small jitter around hole
            h = h + rng.normal(scale=0.01, size=d)
            h /= np.linalg.norm(h)
            y = np.vstack([y, h])
            # keep deterministic order: put hole at position low
            # move last row to low by swap
            tmp = y[low].copy()
            y[low] = y[-1]
            y[-1] = tmp
            steps, T0, T1, lr0, lr1 = 700, 0.014, 0.0025, 0.16, 0.015
        elif mode == 3:
            # overloaded cap shake: worst point + its closest neighbours
            w = int(np.argmax(per_max))
            try:
                Gw = (best @ best[w])
                Gw[w] = -2.0
                nn = np.argsort(-Gw)[:7]
            except Exception:
                nn = np.array([w])
            idx = np.unique(np.concatenate([[w], nn]))
            y[idx] = y[idx] + rng.normal(scale=0.10, size=(len(idx), d))
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            steps, T0, T1, lr0, lr1 = 700, 0.016, 0.0028, 0.17, 0.018
        else:
            # double hole + moderate global jitter
            o = np.argsort(cnt)[:2]
            keep = [i for i in range(n) if i not in set(o.tolist())]
            Yrem = best[keep]
            h1 = _best_hole(Yrem, rng, n_cand=2500)
            Yrem2 = np.vstack([Yrem, h1])
            h2 = _best_hole(Yrem2, rng, n_cand=2500)
            y = best.copy()
            y[o[0]] = h1 + rng.normal(scale=0.008, size=d)
            y[o[1]] = h2 + rng.normal(scale=0.008, size=d)
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            steps, T0, T1, lr0, lr1 = 800, 0.013, 0.0022, 0.15, 0.015

        # annealed repulsion descent (fast, numpy only)
        inner_dead = min(DEAD - 14.0, time.time() + 2.5)
        y = _smooth_run(y, steps, T0, T1, lr0, lr1, inner_dead)
        try:
            v = _max_cos(y)
        except Exception:
            continue
        # decide whether to spend LP time
        need_lp = False
        lp_time = 0.0
        if v < best_val - 1e-12:
            need_lp = True
            lp_time = 3.0
        elif v < best_val + 5e-04 and (it % 4 == 0):
            need_lp = True
            lp_time = 2.0
        elif v < best_val + 2e-03 and (it % 10 == 0):
            need_lp = True
            lp_time = 1.5
        if need_lp:
            if time.time() >= DEAD - 14.0:
                # no time for LP, just keep if better
                if v < best_val:
                    best, best_val = y, v
                continue
            lp_dead = min(DEAD - 14.0, time.time() + lp_time)
            try:
                z = _lp_polish(y, lp_dead)
                vz = _max_cos(z)
            except Exception:
                z, vz = y, v
            if vz < best_val:
                best, best_val = z, vz
            elif v < best_val:
                best, best_val = y, v
        else:
            if v < best_val:
                # better even without LP -> quick tighten next loop will catch it,
                # but update immediately
                best, best_val = y, v
        # opportunistic tiny guard to avoid infinite loop near deadline
        if best_val < 0.265:
            break

    # final tighten of global best
    try:
        if time.time() < DEAD - 1.0:
            best = _lp_polish(best, DEAD - 0.5)
    except Exception:
        pass
    best = best / np.linalg.norm(best, axis=1, keepdims=True)
    return best
