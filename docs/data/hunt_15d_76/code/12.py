import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())


def smooth_descent(X, steps, T0, T1, lr0, lr1, mom=0.85, t_end=1e20):
    n, d = X.shape
    V = np.zeros_like(X)
    for s in range(steps):
        if s % 200 == 0 and time.time() > t_end:
            break
        frac = s / max(1, steps - 1)
        T = T0 + (T1 - T0) * frac
        lr = lr0 + (lr1 - lr0) * frac
        G = X @ X.T
        np.fill_diagonal(G, -10.0)
        m = G.max()
        E = np.exp((G - m) / T)
        np.fill_diagonal(E, 0.0)
        S = E.sum()
        if not np.isfinite(S) or S <= 0:
            break
        W = E / S
        Gr = W @ X
        dot = np.sum(Gr * X, axis=1, keepdims=True)
        Tg = Gr - dot * X
        V = mom * V + Tg
        Xn = X - lr * V
        Xn /= np.linalg.norm(Xn, axis=1, keepdims=True)
        X = Xn
    return X


def hard_polish(X, steps, lr0, lr1, tol0, tol1, t_end=1e20):
    n = X.shape[0]
    for s in range(steps):
        if s % 200 == 0 and time.time() > t_end:
            break
        frac = s / max(1, steps - 1)
        lr = lr0 + (lr1 - lr0) * frac
        tol = tol0 + (tol1 - tol0) * frac
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        m = G.max()
        M = G > (m - tol)
        c = float(M.sum() / 2)
        if c < 1:
            continue
        W = M.astype(float)
        np.fill_diagonal(W, 0.0)
        W /= W.sum()
        Gr = W @ X
        dot = np.sum(Gr * X, axis=1, keepdims=True)
        Tg = Gr - dot * X
        X = X - lr * Tg
        X /= np.linalg.norm(X, axis=1, keepdims=True)
    return X


def hole_for_index(X, keep_idx, rng, nsamp=8000):
    # largest hole w.r.t. points keep_idx
    d = X.shape[1]
    S = rng.normal(size=(nsamp, d))
    S /= np.linalg.norm(S, axis=1, keepdims=True)
    K = X[keep_idx]
    C = S @ K.T
    m = C.max(axis=1)
    return S[int(np.argmin(m))]


def lp_polish(x, seconds, t_deadline=1e20):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = max_cos(x)
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and time.time() < t_deadline:
        if radius < 1e-14:
            break
        c = (x @ x.T)[iu]
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
        obj = np.zeros(nv)
        obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c[sel], A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            break
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        v = max_cos(y)
        if v < best:
            x, best, radius = y, v, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x


def solve():
    t0 = time.time()
    HARD_DEADLINE = t0 + 165
    GRAD_DEADLINE = t0 + 115
    np.random.seed(0)
    rng = np.random.default_rng(0)
    try:
        Xb = np.load("best.npy").astype(float)
    except Exception:
        Xb = np.load("record.npy").astype(float)
    n, d = 76, 15
    if Xb.shape != (n, d):
        # fallback resize / random
        R = rng.normal(size=(n, d))
        R /= np.linalg.norm(R, axis=1, keepdims=True)
        Xb = R
    Xb = Xb / np.linalg.norm(Xb, axis=1, keepdims=True)
    best_val = max_cos(Xb)
    Xbest = Xb.copy()

    trial = 0
    # quick contact estimate for best to guide shakes (recomputed occasionally)
    while time.time() < GRAD_DEADLINE and trial < 80:
        trial += 1
        Xstart = Xbest.copy()
        mode = trial % 6
        if mode == 0:
            sigma = 0.10 + 0.02 * (trial % 3)
            Xstart = Xstart + rng.normal(0, sigma, size=Xstart.shape)
        elif mode == 1:
            k = 5 + (trial % 5)
            idx = rng.choice(n, size=k, replace=False)
            sigma = 0.22
            Xstart[idx] += rng.normal(0, sigma, size=(k, d))
        elif mode == 2:
            # rattler hole reinsertion: find loose points
            G = Xbest @ Xbest.T
            np.fill_diagonal(G, -2.0)
            m = G.max()
            cnt = (G > m - 0.015).sum(axis=1)
            loose = np.where(cnt <= 4)[0]
            if len(loose) == 0:
                loose = np.argsort(cnt)[:3]
            nr = min(len(loose), 2 + (trial % 2))
            victims = rng.choice(loose, size=nr, replace=False)
            keep = np.ones(n, dtype=bool)
            keep[victims] = False
            # sequential fill to avoid collisions
            for v in victims:
                keep_v = np.where(keep)[0]
                h = hole_for_index(Xstart, keep_v, rng, nsamp=6000)
                Xstart[v] = h
                keep[v] = True
            # small shake of rest
            Xstart += rng.normal(0, 0.03, size=Xstart.shape)
        elif mode == 3:
            k = 10
            idx = rng.choice(n, size=k, replace=False)
            Xstart[idx] += rng.normal(0, 0.15, size=(k, d))
        elif mode == 4:
            # single-point hole reinsert + medium shake
            G = Xbest @ Xbest.T
            np.fill_diagonal(G, -2.0)
            m = G.max()
            cnt = (G > m - 0.01).sum(axis=1)
            v = int(np.argmin(cnt))
            keep = np.ones(n, dtype=bool)
            keep[v] = False
            Xstart[v] = hole_for_index(Xstart, np.where(keep)[0], rng, nsamp=8000)
            Xstart += rng.normal(0, 0.05, size=Xstart.shape)
        else:
            sigma = 0.06
            Xstart = Xstart + rng.normal(0, sigma, size=Xstart.shape)

        Xstart /= np.linalg.norm(Xstart, axis=1, keepdims=True)

        # anneal params vary by trial
        if trial % 3 == 0:
            T0, T1 = 0.03, 0.003
            lr0, lr1 = 0.07, 0.008
            steps = 1800
        elif trial % 3 == 1:
            T0, T1 = 0.015, 0.002
            lr0, lr1 = 0.05, 0.005
            steps = 2200
        else:
            T0, T1 = 0.05, 0.005
            lr0, lr1 = 0.09, 0.01
            steps = 1500

        Xopt = smooth_descent(Xstart, steps, T0, T1, lr0, lr1, mom=0.85, t_end=GRAD_DEADLINE)
        if time.time() > GRAD_DEADLINE:
            v = max_cos(Xopt)
            if v < best_val:
                best_val = v
                Xbest = Xopt.copy()
            break
        Xopt = hard_polish(Xopt, steps=600, lr0=0.02, lr1=0.003, tol0=0.01, tol1=0.0008, t_end=GRAD_DEADLINE)
        v = max_cos(Xopt)
        if v < best_val:
            best_val = v
            Xbest = Xopt.copy()
            # intensify around new best with tiny hill climb immediately
            for _ in range(400):
                i = int(rng.integers(0, n))
                old = Xopt[i].copy()
                prop = old + rng.normal(0, 0.008, size=d)
                prop /= np.linalg.norm(prop)
                Xopt[i] = prop
                nv2 = max_cos(Xopt)
                if nv2 < best_val:
                    best_val = nv2
                    Xbest = Xopt.copy()
                else:
                    Xopt[i] = old
                    # occasional neutral move accepted to slide
            if time.time() > GRAD_DEADLINE:
                break

    # broader stochastic hill climbing on best
    hc_end = t0 + 132
    if time.time() < hc_end:
        Xw = Xbest.copy()
        cur = best_val
        it = 0
        sigmas = [0.01, 0.005, 0.002, 0.008]
        while time.time() < hc_end and it < 12000:
            it += 1
            sigma = sigmas[it % len(sigmas)]
            if it % 5 == 0:
                k = 2
                idx = rng.choice(n, size=k, replace=False)
                old = Xw[idx].copy()
                Xw[idx] += rng.normal(0, sigma, size=(k, d))
                Xw[idx] /= np.linalg.norm(Xw[idx], axis=1, keepdims=True)
                nv2 = max_cos(Xw)
                if nv2 < cur:
                    cur = nv2
                    if cur < best_val:
                        best_val = cur
                        Xbest = Xw.copy()
                else:
                    Xw[idx] = old
            else:
                i = int(rng.integers(0, n))
                old = Xw[i].copy()
                prop = old + rng.normal(0, sigma, size=d)
                prop /= np.linalg.norm(prop)
                Xw[i] = prop
                nv2 = max_cos(Xw)
                if nv2 < cur + 1e-12:
                    cur = nv2
                    if cur < best_val:
                        best_val = cur
                        Xbest = Xw.copy()
                        # keep the move and update cur exactly
                    # allow sideways drift if not worse by much
                    if nv2 > cur + 1e-14:
                        pass
                else:
                    Xw[i] = old
            if it % 2000 == 0:
                # re-anchor to best to avoid drift
                Xw = Xbest.copy()
                cur = best_val

    # final LP polish with remaining time
    rem = HARD_DEADLINE - time.time()
    if rem > 8:
        lp_time = min(rem - 2, 28)
        try:
            Xp = lp_polish(Xbest, seconds=lp_time, t_deadline=HARD_DEADLINE - 1)
            vp = max_cos(Xp)
            if vp < best_val and np.all(np.isfinite(Xp)):
                Xbest = Xp
                best_val = vp
        except Exception:
            pass

    Xbest = Xbest / np.linalg.norm(Xbest, axis=1, keepdims=True)
    return Xbest
