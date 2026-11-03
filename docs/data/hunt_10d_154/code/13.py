import time
import numpy as np
from scipy.optimize import linprog, minimize
from scipy.sparse import coo_matrix


def _normalize(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n = np.maximum(n, 1e-12)
    return x / n


def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())


def _lp_polish(x, seconds):
    x = _normalize(x)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        c = (x @ x.T)[iu]
        m = float(c.max())
        sel = c > m - max(20 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        if p == 0:
            break
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))),
                              shape=(n, nv)).tocsr()
            obj = np.zeros(nv)
            obj[-1] = 1.0
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
            x, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x


def _find_hole(X, victim, rng, n_cand=6000):
    n, d = X.shape
    R = rng.normal(size=(n_cand, d))
    R = _normalize(R)
    D = R @ X.T
    D[:, victim] = -2.0
    m = D.max(axis=1)
    # pick among few best for diversity
    K = 8
    idx = np.argpartition(m, K)[:K]
    if rng.random() < 0.6:
        pick = idx[np.argmin(m[idx])]
    else:
        pick = idx[rng.integers(K)]
    return R[pick]


def _adam_refine(X0, rng, steps, lr0, T0, T1, b1=0.9, b2=0.999, eps=1e-8):
    n, d = X0.shape
    x = X0.copy()
    m_ad = np.zeros_like(x)
    v_ad = np.zeros_like(x)
    b1p = 1.0
    b2p = 1.0
    if steps > 1:
        logr = np.log(max(T1, 1e-12) / max(T0, 1e-12)) / (steps - 1)
    else:
        logr = 0.0
    for t in range(1, steps + 1):
        T = T0 * np.exp(logr * (t - 1))
        # cosine decay on lr
        lr = lr0 * (0.15 + 0.85 * (0.5 + 0.5 * np.cos(np.pi * (t - 1) / max(steps - 1, 1))))
        G = x @ x.T
        np.fill_diagonal(G, -10.0)
        mv = float(G.max())
        E = np.exp((G - mv) / max(T, 1e-12))
        np.fill_diagonal(E, 0.0)
        S = float(E.sum() * 0.5)
        if not np.isfinite(S) or S <= 0:
            break
        Grad = (E @ x) / (S + 1e-300)
        dot = (Grad * x).sum(axis=1, keepdims=True)
        Gt = Grad - dot * x
        b1p *= b1
        b2p *= b2
        m_ad = b1 * m_ad + (1.0 - b1) * Gt
        v_ad = b2 * v_ad + (1.0 - b2) * (Gt * Gt)
        mhat = m_ad / (1.0 - b1p)
        vhat = v_ad / (1.0 - b2p)
        step = mhat / (np.sqrt(vhat) + eps)
        # project step to tangent
        sdot = (step * x).sum(axis=1, keepdims=True)
        step = step - sdot * x
        # clip extreme
        sn = np.linalg.norm(step, axis=1, keepdims=True)
        # avoid huge jumps
        clip = 3.0
        scale = np.minimum(1.0, clip / (sn + 1e-12))
        step = step * scale
        x = x - lr * step
        x = _normalize(x)
    return x


def _lbfgs_lse(X_init, T, maxiter, lam=1.0):
    n, d = X_init.shape
    iu0, iu1 = np.triu_indices(n, 1)
    Y0 = X_init.ravel().copy()

    def fun(Yf):
        Y = Yf.reshape(n, d)
        norms = np.linalg.norm(Y, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        X = Y / norms
        G = X @ X.T
        g = G[iu0, iu1]
        # shift = current max approx via T log-sum-exp stability
        sh = float(g.max())
        z = (g - sh) / T
        mz = float(z.max())
        e = np.exp(z - mz)
        s = float(e.sum())
        f = sh + T * (mz + np.log(s + 1e-300))
        w = e / (s + 1e-300)
        W = np.zeros((n, n))
        W[iu0, iu1] = w
        W = W + W.T
        GX = W @ X
        dot = (GX * X).sum(axis=1, keepdims=True)
        grad = (GX - dot * X) / norms
        diff = norms[:, 0] - 1.0
        f = f + 0.5 * lam * float(np.mean(diff ** 2))
        grad = grad + ((lam * diff / n)[:, None] * X)
        return f, grad.ravel()

    try:
        res = minimize(fun, Y0, jac=True, method="L-BFGS-B",
                       options={"maxiter": int(maxiter), "ftol": 1e-12, "gtol": 1e-6, "maxcor": 10})
        Y = res.x.reshape(n, d)
        return _normalize(Y)
    except Exception:
        return _normalize(X_init)


def solve():
    t0 = time.time()
    rng = np.random.RandomState(20261103)
    n, d = 154, 10
    try:
        xb = np.load("best.npy")
    except Exception:
        try:
            xb = np.load("record.npy")
        except Exception:
            xb = None
    if xb is None or np.asarray(xb).shape != (n, d):
        xb = _normalize(rng.normal(size=(n, d)))
    else:
        xb = _normalize(np.asarray(xb, dtype=float))

    bestX = xb.copy()
    best_score = _max_cos(bestX)
    iu0, iu1 = np.triu_indices(n, 1)

    trial = 0
    # rapid basin hopping with Adam
    while time.time() - t0 < 118:
        trial += 1
        if time.time() - t0 > 116:
            break
        G0 = bestX @ bestX.T
        np.fill_diagonal(G0, -2.0)
        m0 = float(G0.max())
        rmax = G0.max(axis=1)
        # contacts for weighting
        cnt = (G0 > m0 - 0.003).sum(axis=1).astype(float)
        probs = cnt + 0.4
        probs /= probs.sum()
        mode = trial % 5
        X0 = bestX.copy()
        if mode == 0:
            sigma = [0.02, 0.05, 0.09, 0.14][(trial // 5) % 4]
            noise = rng.normal(size=X0.shape) * sigma
            X0 = _normalize(X0 + noise)
            steps, lr0, T0, T1 = 650, 0.018, 0.012, 0.0018
        elif mode == 1:
            # shake jammed top points harder
            order = np.argsort(-rmax)
            k = 12
            idx = order[:k]
            noise = rng.normal(size=X0.shape) * 0.02
            noise[idx] *= 4.0
            X0 = _normalize(X0 + noise)
            steps, lr0, T0, T1 = 650, 0.015, 0.010, 0.0015
        elif mode == 2:
            nk = int(rng.choice([1, 2, 2, 3]))
            victims = rng.choice(n, size=nk, replace=False, p=probs)
            for vv in victims:
                h = _find_hole(X0, int(vv), rng, n_cand=6000)
                h = h + rng.normal(size=d) * 0.008
                h = h / (np.linalg.norm(h) + 1e-12)
                X0[int(vv)] = h
            noise = rng.normal(size=X0.shape) * 0.012
            for vv in victims:
                noise[int(vv)] = 0.0
            X0 = _normalize(X0 + noise)
            steps, lr0, T0, T1 = 750, 0.016, 0.011, 0.0012
        elif mode == 3:
            # worst-pair relocation
            k = int(np.argmax(G0[iu0, iu1]))
            a, b = int(iu0[k]), int(iu1[k])
            pick = a if rmax[a] >= rmax[b] else b
            if rng.random() < 0.3:
                pick = b if pick == a else a
            # occasionally relocate both
            picks = [pick]
            if rng.random() < 0.25:
                picks.append(b if pick == a else a)
            for pp in picks:
                h = _find_hole(X0, int(pp), rng, n_cand=6000)
                h = h + rng.normal(size=d) * 0.008
                h = h / (np.linalg.norm(h) + 1e-12)
                X0[int(pp)] = h
            noise = rng.normal(size=X0.shape) * 0.012
            for pp in picks:
                noise[int(pp)] = 0.0
            X0 = _normalize(X0 + noise)
            steps, lr0, T0, T1 = 750, 0.016, 0.011, 0.0012
        else:
            # cluster shake around random center + hole refill of worst
            c = int(rng.integers(n))
            dists = G0[c]
            neigh = np.argsort(-dists)[:8]
            noise = rng.normal(size=X0.shape) * 0.015
            noise[neigh] *= 3.0
            # also move one jammed to hole
            vv = int(np.argmax(rmax))
            if rng.random() < 0.7:
                h = _find_hole(X0, vv, rng, n_cand=5000)
                h = h + rng.normal(size=d) * 0.008
                h = h / (np.linalg.norm(h) + 1e-12)
                X0[vv] = h
                noise[vv] = 0.0
            X0 = _normalize(X0 + noise)
            steps, lr0, T0, T1 = 700, 0.017, 0.012, 0.0015

        # jitter hyperparams slightly
        lr0 = lr0 * (0.8 + 0.4 * rng.random())
        try:
            Xopt = _adam_refine(X0, rng, steps=int(steps), lr0=float(lr0),
                                T0=float(T0), T1=float(T1))
        except Exception:
            continue
        sc = _max_cos(Xopt)
        if np.isfinite(sc) and sc < best_score - 1e-12:
            best_score = sc
            bestX = Xopt.copy()

    # staged L-BFGS tightening on best
    try:
        if time.time() - t0 < 140:
            bestX = _lbfgs_lse(bestX, T=0.006, maxiter=60)
            if time.time() - t0 < 150:
                bestX = _lbfgs_lse(bestX, T=0.0025, maxiter=80)
    except Exception:
        pass
    try:
        sc = _max_cos(bestX)
        if np.isfinite(sc):
            best_score = sc
    except Exception:
        pass

    elapsed = time.time() - t0
    remain = 168.0 - elapsed
    if remain > 8:
        secs = min(34.0, remain - 2.0)
        try:
            bestX = _lp_polish(bestX, secs)
        except Exception:
            pass
    bestX = _normalize(bestX)
    norms = np.linalg.norm(bestX, axis=1)
    bad = ~(np.isfinite(norms) & (norms > 1e-8))
    if np.any(bad):
        R = rng.normal(size=(int(bad.sum()), d))
        bestX[bad] = _normalize(R)
    return bestX
