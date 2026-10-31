import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _lp_polish(x, seconds, t_deadline=None):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds:
        if t_deadline is not None and time.time() > t_deadline:
            break
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
    return x

def _smooth_opt(Xinit, steps, a0, a1, lr0, lr1, mu, noise0, rng, track_best=True):
    n, d = Xinit.shape
    X = Xinit.copy()
    V = np.zeros_like(X)
    bestv = _max_cos(X)
    bestX = X.copy()
    lalpha0, lalpha1 = np.log(a0), np.log(a1)
    llr0, llr1 = np.log(lr0), np.log(lr1)
    for s in range(steps):
        frac = s / max(1, steps - 1)
        alpha = float(np.exp(lalpha0 + (lalpha1 - lalpha0) * frac))
        lr = float(np.exp(llr0 + (llr1 - llr0) * frac))
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        # weights, avoid overflow: exp(alpha*(G-m)), G<=-2 diag -> 0
        W = np.exp(alpha * (G - m))
        np.fill_diagonal(W, 0.0)
        grad = W @ X
        dot = np.sum(grad * X, axis=1, keepdims=True)
        Gt = grad - dot * X
        # clip extreme
        # momentum update: V = mu*V + Gt ; X -= lr*V
        V = mu * V + Gt
        X = X - lr * V
        if noise0 > 0:
            ns = noise0 * (1.0 - frac)
            if ns > 1e-12:
                N = rng.standard_normal(size=X.shape)
                Nd = np.sum(N * X, axis=1, keepdims=True)
                Nt = N - Nd * X
                X = X + ns * Nt
        X /= np.linalg.norm(X, axis=1, keepdims=True)
        if track_best and (s % 50 == 49 or s == steps - 1):
            v = _max_cos(X)
            if v < bestv:
                bestv = v
                bestX = X.copy()
    # final eval
    v = _max_cos(X)
    if v < bestv:
        bestv = v
        bestX = X.copy()
    return bestX, bestv

def _contacts(X, thresh):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    m = float(G.max())
    C = G > (m - thresh)
    cnt = C.sum(axis=1)
    return m, cnt, G

def _random_sphere(rng, k, d):
    A = rng.standard_normal(size=(k, d))
    A /= np.linalg.norm(A, axis=1, keepdims=True)
    return A

def solve():
    t0 = time.time()
    DEADLINE = t0 + 168.0
    rng = np.random.default_rng(12345)
    try:
        xb = np.load("best.npy")
    except Exception:
        xb = None
    try:
        xr = np.load("record.npy")
    except Exception:
        xr = None
    cands = []
    for c in (xb, xr):
        if c is not None:
            c = np.asarray(c, dtype=float)
            if c.shape == (168, 4):
                c = c / np.linalg.norm(c, axis=1, keepdims=True)
                cands.append(((_max_cos(c)), c))
    if len(cands) == 0:
        Xbest = _random_sphere(rng, 168, 4)
        best = _max_cos(Xbest)
    else:
        cands.sort(key=lambda t: t[0])
        best, Xbest = cands[0][0], cands[0][1].copy()
    n, d = 168, 4
    # quick contact analysis for targeting
    # main basin-hopping loop, reserve ~30s for final LP
    trial = 0
    while time.time() < DEADLINE - 32.0:
        trial += 1
        # choose perturbation type
        r = rng.random()
        X = Xbest.copy()
        m, cnt, _ = _contacts(X, 1e-7)
        order = np.argsort(-cnt)  # descending
        if r < 0.25:
            sigma = float(rng.choice([0.004, 0.008, 0.015, 0.03]))
            X = X + sigma * rng.standard_normal(size=X.shape)
            X /= np.linalg.norm(X, axis=1, keepdims=True)
        elif r < 0.5:
            sigma = float(rng.choice([0.02, 0.05, 0.08]))
            X = X + sigma * rng.standard_normal(size=X.shape)
            X /= np.linalg.norm(X, axis=1, keepdims=True)
        elif r < 0.8:
            # surgical: re-init k high-contact points
            k = int(rng.integers(2, 8))
            # pick mostly from top-60 overloaded
            pool = order[:60]
            idx = rng.choice(pool, size=min(k, len(pool)), replace=False)
            # sometimes add random point
            if rng.random() < 0.5:
                extra = int(rng.integers(0, n))
                idx = np.unique(np.append(idx, extra))
            X[idx] = _random_sphere(rng, len(idx), d)
            # small shake rest
            sigma = 0.008
            mask = np.ones(n, dtype=bool); mask[idx] = False
            X[mask] = X[mask] + sigma * rng.standard_normal(size=(mask.sum(), d))
            X /= np.linalg.norm(X, axis=1, keepdims=True)
        else:
            # big shake of random subset + rattler hunt
            k = int(rng.integers(4, 14))
            idx = rng.choice(n, size=k, replace=False)
            # bias to high contacts half the time
            if rng.random() < 0.5:
                idx[:k//2] = rng.choice(order[:40], size=k//2, replace=False)
            X[idx] = _random_sphere(rng, k, d)
            sigma = 0.015
            X /= np.linalg.norm(X, axis=1, keepdims=True)
            # extra global jitter
            X = X + 0.005 * rng.standard_normal(size=X.shape)
            X /= np.linalg.norm(X, axis=1, keepdims=True)
        # choose anneal params varied per trial
        a0 = float(rng.choice([20.0, 30.0, 50.0]))
        a1 = float(rng.choice([250.0, 400.0, 600.0]))
        lr0 = float(rng.choice([0.008, 0.015, 0.025]))
        lr1 = float(rng.choice([0.001, 0.002, 0.004]))
        mu = float(rng.choice([0.80, 0.85, 0.90]))
        noise0 = float(rng.choice([0.0, 0.0005, 0.0015]))
        steps = int(rng.choice([600, 900, 1200]))
        # cap steps if little time
        if time.time() > DEADLINE - 60:
            steps = min(steps, 600)
        Xopt, vopt = _smooth_opt(X, steps, a0, a1, lr0, lr1, mu, noise0, rng, track_best=True)
        # hard-tail refine with high alpha, small lr, no noise
        if vopt < best + 0.004:
            Xopt2, vopt2 = _smooth_opt(Xopt, 300, 400.0, 1500.0, 0.004, 0.0008, 0.85, 0.0, rng, track_best=True)
        else:
            Xopt2, vopt2 = Xopt, vopt
        if vopt2 < best - 2e-9:
            # verify with short LP squeeze (2s) to get true local optimum
            Xlp = _lp_polish(Xopt2, min(3.0, max(0.5, DEADLINE - 32.0 - time.time())), t_deadline=DEADLINE - 30.0)
            vlp = _max_cos(Xlp)
            # keep best of the two (LP sometimes worsens due to time cut)
            if vlp < vopt2:
                Xopt2, vopt2 = Xlp, vlp
            if vopt2 < best:
                Xbest, best = Xopt2, vopt2
                # early exit if big win? continue to improve further
        # also opportunistic: if we matched best closely, try LP rarely
        elif trial % 25 == 0 and time.time() < DEADLINE - 60:
            # occasional exploitation without improvement signal
            pass
        if time.time() >= DEADLINE - 32.0:
            break
    # final long polish of global best
    remain = DEADLINE - time.time() - 1.0
    if remain > 2.0:
        Xbest = _lp_polish(Xbest, remain, t_deadline=DEADLINE - 0.5)
    Xbest = Xbest / np.linalg.norm(Xbest, axis=1, keepdims=True)
    return Xbest
