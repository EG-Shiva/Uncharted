import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(np.max(g))


def smooth_anneal(X, iters, T0, T1, lr):
    X = X.copy()
    n, d = X.shape
    m = np.zeros_like(X)
    v = np.zeros_like(X)
    b1 = 0.9
    b2 = 0.999
    eps = 1e-8
    log_ratio = np.log(T1 / T0)
    for k in range(1, iters + 1):
        T = T0 * np.exp(log_ratio * (k / iters))
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        M = np.max(G)
        E = np.exp((G - M) / T)
        np.fill_diagonal(E, 0.0)
        S = E.sum() * 0.5
        if S <= 0:
            break
        Grad = (E @ X) / S
        dot = np.sum(Grad * X, axis=1, keepdims=True)
        Gt = Grad - dot * X
        m = b1 * m + (1.0 - b1) * Gt
        v = b2 * v + (1.0 - b2) * (Gt * Gt)
        bc1 = 1.0 - b1 ** k
        bc2 = 1.0 - b2 ** k
        mhat = m / bc1
        vhat = v / bc2
        step = lr * mhat / (np.sqrt(vhat) + eps)
        X = X - step
        nr = np.linalg.norm(X, axis=1, keepdims=True)
        nr[nr == 0] = 1.0
        X = X / nr
    return X


def lp_polish(x, seconds, hard_deadline=None):
    x = np.asarray(x, dtype=np.float64)
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = max_cos(x)
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while True:
        now = time.time()
        if now - start >= seconds:
            break
        if hard_deadline is not None and now >= hard_deadline - 1.0:
            break
        if radius < 1e-14:
            break
        cfull = (x @ x.T)[iu]
        m = float(cfull.max())
        thresh = max(20.0 * radius, 1e-9)
        sel = cfull > m - thresh
        I = iu[0][sel].astype(np.int64)
        J = iu[1][sel].astype(np.int64)
        csel = cfull[sel]
        p = len(I)
        if p == 0:
            radius *= 0.5
            continue
        if p > 3500:
            # keep tightest 3500
            idx = np.argpartition(csel, -3500)[-3500:]
            I = I[idx]
            J = J[idx]
            csel = csel[idx]
            p = 3500
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols_I = I[:, None] * d + np.arange(d)
        cols_J = J[:, None] * d + np.arange(d)
        cols = np.concatenate([cols_I, cols_J, np.full((p, 1), n * d)], axis=1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], axis=1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        except Exception:
            radius *= 0.5
            continue
        obj = np.zeros(nv)
        obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
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
        vv = max_cos(y)
        if vv < best:
            x = y
            best = vv
            radius = min(radius * 1.5, 0.02)
        else:
            radius *= 0.5
    return x


def solve():
    t0 = time.time()
    HARD = t0 + 178.0
    rng = np.random.default_rng(0)
    cands = []
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=np.float64)
            if a.shape == (250, 6):
                cands.append(a)
        except Exception:
            pass
    best = None
    best_val = np.inf
    for c in cands:
        nrm = np.linalg.norm(c, axis=1, keepdims=True)
        nrm[nrm == 0] = 1.0
        c = c / nrm
        # guard NaN
        if not np.all(np.isfinite(c)):
            continue
        vv = max_cos(c)
        if vv < best_val:
            best_val = vv
            best = c.copy()
    if best is None:
        best = rng.normal(size=(250, 6))
        best = best / np.linalg.norm(best, axis=1, keepdims=True)
        best_val = max_cos(best)
    n, d = 250, 6

    # Stage 1: direct smooth anneal of incumbent (no noise)
    try:
        if time.time() < HARD - 30:
            X0 = smooth_anneal(best, iters=1200, T0=0.008, T1=0.0012, lr=0.008)
            v0 = max_cos(X0)
            if v0 < best_val:
                best = X0
                best_val = v0
    except Exception:
        pass

    # Iterated shakes + reinsertions
    loop_deadline = t0 + 148.0
    trial = 0
    sigmas = [0.004, 0.008, 0.015, 0.03, 0.06, 0.10]
    while time.time() < loop_deadline and time.time() < HARD - 25:
        trial += 1
        mode = trial % 4  # 0,1,2 full shake, 3 single-point reinit
        try:
            if mode == 3:
                # worst-point reinsertion: find points with largest individual max
                G = best @ best.T
                np.fill_diagonal(G, -2.0)
                rowmax = G.max(axis=1)
                k = 1 + (trial % 3)  # 1..3 points
                worst = np.argpartition(rowmax, -k)[-k:]
                Xt = best.copy()
                for w in worst:
                    v = rng.normal(size=(d,))
                    v = v / np.linalg.norm(v)
                    Xt[w] = v
                # small shake on rest
                Xt = Xt + 0.003 * rng.normal(size=Xt.shape)
                Xt = Xt / np.linalg.norm(Xt, axis=1, keepdims=True)
                Xs = smooth_anneal(Xt, iters=900, T0=0.007, T1=0.001, lr=0.009)
            else:
                sigma = sigmas[(trial - 1) % len(sigmas)]
                # occasionally use random sigma
                if trial % 7 == 0:
                    sigma = float(rng.uniform(0.005, 0.07))
                Xt = best + sigma * rng.normal(size=best.shape)
                Xt = Xt / np.linalg.norm(Xt, axis=1, keepdims=True)
                # adapt iters/lr to sigma
                if sigma < 0.02:
                    iters, lr, T0, T1 = 700, 0.007, 0.006, 0.001
                else:
                    iters, lr, T0, T1 = 1000, 0.010, 0.010, 0.0012
                Xs = smooth_anneal(Xt, iters=iters, T0=T0, T1=T1, lr=lr)
            vs = max_cos(Xs)
            # quick accept check: if promising, short LP
            if vs < best_val - 1e-12:
                # already better without LP
                best = Xs
                best_val = vs
                # refine a bit with very short LP if time allows
                if time.time() < loop_deadline - 5:
                    Xp = lp_polish(best, seconds=1.5, hard_deadline=HARD)
                    vp = max_cos(Xp)
                    if vp < best_val:
                        best = Xp
                        best_val = vp
            elif vs < best_val + 0.004:
                # close: try short LP to see if basin is better
                if time.time() < loop_deadline - 5:
                    Xp = lp_polish(Xs, seconds=1.2, hard_deadline=HARD)
                    vp = max_cos(Xp)
                    if vp < best_val:
                        best = Xp
                        best_val = vp
            # else discard
        except Exception:
            continue

    # Final polish with remaining time
    remain = HARD - time.time() - 1.5
    if remain > 2.0:
        remain = min(remain, 28.0)
        try:
            Xf = lp_polish(best, seconds=remain, hard_deadline=HARD)
            vf = max_cos(Xf)
            if vf < best_val and np.all(np.isfinite(Xf)):
                best = Xf
                best_val = vf
        except Exception:
            pass
    best = best / np.linalg.norm(best, axis=1, keepdims=True)
    return best
