import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def _norm(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n = np.maximum(n, 1e-12)
    return x / n


def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(np.max(g))


def _contact_counts(x, thr=1e-3):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    m = float(np.max(g))
    return np.sum(g > m - thr, axis=1), m


def _quench(X0, steps, a0, a1, lr0, lr1, mu=0.85):
    X = X0.copy()
    vel = np.zeros_like(X)
    for t in range(1, steps + 1):
        frac = t / steps
        alpha = a0 * (a1 / a0) ** frac
        lr = lr0 * (lr1 / lr0) ** frac
        G = X @ X.T
        np.fill_diagonal(G, -1.0)
        M = float(np.max(G))
        # avoid overflow: exp(alpha*(G-M)), G-M<=0
        E = np.exp(alpha * (G - M))
        np.fill_diagonal(E, 0.0)
        S = float(np.sum(E))
        if not np.isfinite(S) or S <= 0:
            continue
        Gw = (E @ X) / S
        dot = np.sum(Gw * X, axis=1, keepdims=True)
        Gt = Gw - dot * X
        nr = np.sqrt(np.sum(Gt * Gt, axis=1, keepdims=True)) + 1e-12
        Dt = Gt / nr
        vel = mu * vel + Dt
        # bias correction-ish to keep step stable early
        X = X - lr * vel
        X /= np.linalg.norm(X, axis=1, keepdims=True) + 1e-16
    return X


def _hole(Y, rng, n_cand=3000):
    d = Y.shape[1]
    cand = rng.standard_normal((n_cand, d))
    cand = _norm(cand)
    dots = cand @ Y.T
    md = np.max(dots, axis=1)
    return cand[np.argmin(md)]


def _lp_polish(x, deadline):
    x = _norm(x)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1.5e-3
    iu = np.triu_indices(n, 1)
    it = 0
    while time.time() < deadline and radius > 1e-14 and it < 60:
        it += 1
        cfull = (x @ x.T)[iu]
        m = float(np.max(cfull))
        sel = cfull > m - max(20.0 * radius, 1e-9)
        cnt = int(np.count_nonzero(sel))
        if cnt == 0:
            break
        if cnt > 1300:
            idx = np.where(sel)[0]
            order = np.argsort(-cfull[idx])[:1300]
            nsel = np.zeros_like(sel)
            nsel[idx[order]] = True
            sel = nsel
        csel = cfull[sel]
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
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d),
                                           np.arange(n * d))), shape=(n, nv)).tocsr()
        except Exception:
            break
        obj = np.zeros(nv)
        obj[-1] = 1.0
        try:
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
        y = _norm(y)
        v = _max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x


def solve():
    t0 = time.time()
    T_SEARCH_END = t0 + 148.0
    T_END = t0 + 172.0
    rng = np.random.default_rng(0)
    n, d = 55, 21
    best = None
    best_score = 1e9
    for fn in ["best.npy", "record.npy"]:
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=np.float64)
            if a.shape == (n, d):
                a = _norm(a)
                s = _max_cos(a)
                if s < best_score:
                    best_score = s
                    best = a.copy()
        except Exception:
            pass
    if best is None:
        best = _norm(rng.standard_normal((n, d)))
        best_score = _max_cos(best)
    cur = best.copy()
    cur_score = best_score
    trial = 0
    while time.time() < T_SEARCH_END:
        trial += 1
        if trial % 40 == 0:
            cur = best.copy()
            cur_score = best_score
        # annealing temperature decaying with elapsed time
        el = (time.time() - t0) / max(T_SEARCH_END - t0, 1e-9)
        Temp = 0.003 * (1.0 - el) + 0.0002 * el
        r = rng.random()
        prop = cur.copy()
        if r < 0.30:
            # 1-2 worst-contact hole reinsertion
            cnt, mx = _contact_counts(prop, thr=1e-3)
            k = 1 if rng.random() < 0.6 else 2
            if rng.random() < 0.15:
                victims = rng.choice(n, size=k, replace=False)
            else:
                victims = np.argsort(-cnt)[:k]
            keep = np.ones(n, dtype=bool)
            keep[victims] = False
            Y = prop[keep]
            for vi in victims:
                h = _hole(Y, rng, n_cand=3000)
                prop[vi] = h
                Y = np.vstack([Y, h[None, :]])
            prop = _norm(prop + 0.005 * rng.standard_normal(prop.shape))
            steps, a0, a1, lr0, lr1 = 450, 50.0, 650.0, 0.008, 0.0008
        elif r < 0.60:
            # cluster hole reinsertion 3-6 hubs
            cnt, mx = _contact_counts(prop, thr=1e-3)
            k = 3 + int(rng.integers(0, 4))  # 3..6
            if rng.random() < 0.15:
                victims = rng.choice(n, size=k, replace=False)
            else:
                victims = np.argsort(-cnt)[:k]
            keep = np.ones(n, dtype=bool)
            keep[victims] = False
            Y = prop[keep]
            for vi in victims:
                h = _hole(Y, rng, n_cand=3000)
                prop[vi] = h
                Y = np.vstack([Y, h[None, :]])
            prop = _norm(prop + 0.010 * rng.standard_normal(prop.shape))
            steps, a0, a1, lr0, lr1 = 650, 40.0, 600.0, 0.010, 0.001
        elif r < 0.80:
            # global shake
            sigma = 0.06 + 0.16 * rng.random()
            prop = _norm(prop + sigma * rng.standard_normal(prop.shape))
            steps, a0, a1, lr0, lr1 = 550, 40.0, 550.0, 0.010, 0.001
        else:
            # subset strong kick
            s = 6 + int(rng.integers(0, 7))  # 6..12
            idx = rng.choice(n, size=s, replace=False)
            sigma = 0.25 + 0.30 * rng.random()
            prop[idx] = _norm(prop[idx] + sigma * rng.standard_normal((s, d)))
            steps, a0, a1, lr0, lr1 = 600, 40.0, 600.0, 0.010, 0.001
        prop = _quench(prop, steps, a0, a1, lr0, lr1)
        if time.time() > T_SEARCH_END:
            break
        sc = _max_cos(prop)
        if sc < best_score - 1e-12:
            best_score = sc
            best = prop.copy()
            cur = prop.copy()
            cur_score = sc
        elif sc < cur_score or rng.random() < np.exp((cur_score - sc) / max(Temp, 1e-12)):
            cur = prop.copy()
            cur_score = sc
    # final polish of global best with remaining budget
    try:
        if time.time() < T_END - 3:
            polished = _lp_polish(best, T_END - 1.0)
            sp = _max_cos(polished)
            if sp < best_score:
                best = polished
                best_score = sp
    except Exception:
        pass
    best = _norm(np.asarray(best, dtype=np.float64))
    return best
