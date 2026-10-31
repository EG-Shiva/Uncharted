import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(np.max(g))


def smooth_descent(x, steps=500, T0=0.025, T1=0.003, s0=0.015, s1=0.0012):
    x = x.copy()
    for it in range(steps):
        frac = it / max(1, steps - 1)
        T = T0 * (T1 / T0) ** frac
        s = s0 + (s1 - s0) * frac
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        cmax = float(np.max(G))
        W = np.exp((G - cmax) / T)
        np.fill_diagonal(W, 0.0)
        Grad = (W @ x) / T
        dot = np.sum(Grad * x, axis=1, keepdims=True)
        Gt = Grad - dot * x
        norms = np.linalg.norm(Gt, axis=1, keepdims=True) + 1e-12
        x = x - s * (Gt / norms)
        nrm = np.linalg.norm(x, axis=1, keepdims=True)
        nrm[nrm == 0] = 1.0
        x = x / nrm
    return x


def lp_polish(x, budget):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = max_cos(x)
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < budget and radius > 1e-14:
        cmat = x @ x.T
        c = cmat[iu]
        m = float(np.max(c))
        sel = c > m - max(20.0 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        if p == 0:
            break
        # cap constraints to tightest 2500 for speed
        if p > 2500:
            order = np.argsort(-c[sel])[:2500]
            I, J = I[order], J[order]
            p = 2500
            csel = c[sel][order]
        else:
            csel = c[sel]
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate(
            [I[:, None] * d + np.arange(d),
             J[:, None] * d + np.arange(d),
             np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix(
                (x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))),
                shape=(n, nv)).tocsr()
            obj = np.zeros(nv)
            obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-csel,
                          A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                          method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        try:
            y = x + res.x[:-1].reshape(n, d)
        except Exception:
            radius *= 0.5
            continue
        y = y / np.linalg.norm(y, axis=1, keepdims=True)
        v = max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x


def hole_candidate(others, rng, M=3000):
    cand = rng.normal(size=(M, 4))
    cand /= np.linalg.norm(cand, axis=1, keepdims=True)
    dots = cand @ others.T
    worst = np.max(dots, axis=1)
    return cand[np.argmin(worst)]


def contact_counts(x, tol=1e-7):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    m = float(np.max(G))
    C = G > m - tol
    cnt = np.sum(C, axis=1)
    return cnt, m, G


def solve():
    t0 = time.time()
    DEADLINE = t0 + 172.0
    EXPLORE_END = t0 + 138.0
    rng = np.random.default_rng(12345)
    n = 153

    cands = []
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
        except Exception:
            continue
        try:
            a = np.asarray(a, dtype=float)
            if a.shape == (153, 4) and np.all(np.isfinite(a)):
                nr = np.linalg.norm(a, axis=1, keepdims=True)
                if np.min(nr) > 1e-12:
                    a = a / nr
                    cands.append((max_cos(a), a))
        except Exception:
            continue
    if len(cands) == 0:
        x0 = rng.normal(size=(n, 4))
        x0 /= np.linalg.norm(x0, axis=1, keepdims=True)
        best_val = max_cos(x0)
        best = x0
    else:
        cands.sort(key=lambda t: t[0])
        best_val, best = cands[0][0], cands[0][1].copy()

    # quick initial LP squeeze if starting point is record
    cur = best.copy()
    cur_val = best_val
    Tb = 0.0022
    it = 0

    while time.time() < EXPLORE_END and time.time() < DEADLINE - 28:
        it += 1
        u = rng.random()
        y = None
        steps, T0, T1 = 500, 0.022, 0.0025

        if u < 0.22:
            sig = 0.007 + 0.010 * rng.random()
            y = cur + rng.normal(size=cur.shape) * sig
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            steps, T0, T1 = 450, 0.018, 0.0022
        elif u < 0.44:
            # contact-guided cluster surgery: replace dense neighbourhood
            try:
                cnt, m, G = contact_counts(cur, tol=5e-4)
                seed = int(np.argmax(cnt + rng.random(n) * 0.5))
                dots = G[seed].copy()
                dots[seed] = -2.0
                k = int(rng.integers(6, 15))
                neigh = np.argsort(-dots)[:k]
                victims = np.concatenate([[seed], neigh])
                victims = np.unique(victims)
                y = cur.copy()
                # replace one-by-one with holes w.r.t. survivors
                for v in victims:
                    others = np.delete(y, v, axis=0)
                    y[v] = hole_candidate(others, rng, M=2500)
                y += rng.normal(size=y.shape) * 0.0025
                y /= np.linalg.norm(y, axis=1, keepdims=True)
            except Exception:
                k = int(rng.integers(5, 14))
                idx = rng.choice(n, size=k, replace=False)
                y = cur.copy()
                y[idx] += rng.normal(size=(k, 4)) * 0.15
                y /= np.linalg.norm(y, axis=1, keepdims=True)
            steps, T0, T1 = 650, 0.026, 0.003
        elif u < 0.60:
            # rattler-targeted reinsertion: replace fewest-contact points
            try:
                cnt, m, _ = contact_counts(cur, tol=1e-7)
                order = np.argsort(cnt + rng.random(n) * 0.3)
                nr = 2 if rng.random() < 0.5 else 3
                victims = order[:nr]
                y = cur.copy()
                for v in victims:
                    others = np.delete(y, v, axis=0)
                    y[v] = hole_candidate(others, rng, M=3000)
                y += rng.normal(size=y.shape) * 0.002
                y /= np.linalg.norm(y, axis=1, keepdims=True)
            except Exception:
                y = cur.copy()
                ridx = rng.choice(n, size=2, replace=False)
                for r in ridx:
                    others = np.delete(y, r, axis=0)
                    y[r] = hole_candidate(others, rng, M=3000)
                y /= np.linalg.norm(y, axis=1, keepdims=True)
            steps, T0, T1 = 650, 0.024, 0.0025
        elif u < 0.74:
            k = int(rng.integers(5, 18))
            idx = rng.choice(n, size=k, replace=False)
            y = cur.copy()
            y[idx] += rng.normal(size=(k, 4)) * 0.12
            y += rng.normal(size=y.shape) * 0.003
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            steps, T0, T1 = 600, 0.025, 0.003
        elif u < 0.87:
            y = cur.copy()
            nr = 1 if rng.random() < 0.6 else 2
            ridx = rng.choice(n, size=nr, replace=False)
            for r in ridx:
                others = np.delete(y, r, axis=0)
                y[r] = hole_candidate(others, rng, M=3000)
            y += rng.normal(size=y.shape) * 0.002
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            steps, T0, T1 = 600, 0.022, 0.0025
        else:
            # fresh random basin for diversity
            y = rng.normal(size=(n, 4))
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            # slight bias toward best holes? just smooth
            steps, T0, T1 = 700, 0.030, 0.003

        try:
            ys = smooth_descent(y, steps=steps, T0=T0, T1=T1, s0=0.015, s1=0.0012)
        except Exception:
            continue
        try:
            v = max_cos(ys)
        except Exception:
            continue

        accept = False
        if v < cur_val:
            accept = True
        else:
            try:
                if rng.random() < np.exp((cur_val - v) / max(Tb, 1e-6)):
                    accept = True
            except Exception:
                accept = False
        if accept:
            cur, cur_val = ys, v

        if v < best_val + 0.0025 and accept:
            if time.time() < EXPLORE_END - 2.5:
                try:
                    z = lp_polish(ys, budget=0.9)
                    vz = max_cos(z)
                except Exception:
                    vz = v
                    z = ys
                if vz < best_val - 1e-12:
                    best, best_val = z, vz
                    cur, cur_val = z.copy(), vz
                elif vz < cur_val:
                    cur, cur_val = z, vz
            else:
                if v < best_val - 1e-12:
                    best, best_val = ys, v

        if cur_val > best_val + 0.012:
            try:
                cur = best.copy() + rng.normal(size=best.shape) * 0.008
                cur /= np.linalg.norm(cur, axis=1, keepdims=True)
                cur_val = max_cos(cur)
            except Exception:
                cur = best.copy()
                cur_val = best_val
            Tb = max(0.0010, Tb * 0.98)

    # final intensive polish: perturbed re-polish to squeeze ties
    remaining = DEADLINE - time.time() - 2.0
    if remaining > 4:
        try:
            best = lp_polish(best, budget=min(remaining * 0.5, 15.0))
            best_val = max_cos(best)
        except Exception:
            pass
        # a few shake + polish cycles
        t_end = DEADLINE - 2.0
        for _ in range(6):
            if time.time() > t_end - 3:
                break
            try:
                shake = best + rng.normal(size=best.shape) * 0.0015
                shake /= np.linalg.norm(shake, axis=1, keepdims=True)
                shake = smooth_descent(shake, steps=200, T0=0.008, T1=0.0015, s0=0.008, s1=0.001)
                z = lp_polish(shake, budget=1.5)
                vz = max_cos(z)
                if vz < best_val:
                    best, best_val = z, vz
            except Exception:
                continue
        try:
            best = lp_polish(best, budget=max(1.0, t_end - time.time()))
        except Exception:
            pass

    best = best / np.linalg.norm(best, axis=1, keepdims=True)
    return best
