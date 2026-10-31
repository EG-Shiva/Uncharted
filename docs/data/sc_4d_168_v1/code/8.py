import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(np.max(g))

def soft_anneal(Xc):
    # stages: (beta, steps, lr0, lr1)
    stages = [
        (40, 350, 0.03, 0.005),
        (120, 550, 0.02, 0.003),
        (300, 700, 0.012, 0.0015),
        (700, 700, 0.008, 0.0008),
        (1500, 500, 0.004, 0.0005),
    ]
    mu = 0.9
    for beta, steps, lr0, lr1 in stages:
        v = np.zeros_like(Xc)
        for t in range(steps):
            frac = t / (steps - 1) if steps > 1 else 0.0
            lr = lr0 + (lr1 - lr0) * frac
            G = Xc @ Xc.T
            np.fill_diagonal(G, -2.0)
            gm = np.max(G)
            # softmax weights, gm shift for stability (cancels in ratio)
            W = np.exp(beta * (G - gm))
            np.fill_diagonal(W, 0.0)
            rs = W.sum(axis=1, keepdims=True)
            A = W @ Xc
            An = A / np.maximum(rs, 1e-30)
            dot = np.sum(An * Xc, axis=1, keepdims=True)
            Gt = An - dot * Xc
            v = mu * v + Gt
            Xc = Xc - lr * v
            nrm = np.linalg.norm(Xc, axis=1, keepdims=True)
            nrm = np.maximum(nrm, 1e-12)
            Xc /= nrm
    return Xc

def lp_polish(x, seconds, t_deadline=None):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = max_cos(x)
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds:
        if t_deadline is not None and time.time() > t_deadline:
            break
        if radius < 1e-14:
            break
        Gfull = x @ x.T
        c = Gfull[iu]
        m = c.max()
        sel = c > m - max(20 * radius, 1e-9)
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
        v = max_cos(y)
        if v < best:
            x, best, radius = y, v, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def uniform_sphere(rng, n, d):
    y = rng.standard_normal((n, d))
    y /= np.linalg.norm(y, axis=1, keepdims=True)
    return y

def solve():
    t0 = time.time()
    DEADLINE = t0 + 168.0
    np.random.seed(0)
    rng = np.random.default_rng(12345)
    # load starting points
    cands = []
    try:
        b = np.load("best.npy")
        b = b / np.linalg.norm(b, axis=1, keepdims=True)
        cands.append(b)
    except Exception:
        pass
    try:
        r = np.load("record.npy")
        r = r / np.linalg.norm(r, axis=1, keepdims=True)
        cands.append(r)
    except Exception:
        pass
    n, d = 168, 4
    if len(cands) == 0:
        X_best = uniform_sphere(rng, n, d)
        best_val = max_cos(X_best)
    else:
        vals = [max_cos(c) for c in cands]
        bi = int(np.argmin(vals))
        X_best = cands[bi].copy()
        best_val = float(vals[bi])
    # pool of elites for polishing
    elites = [(best_val, X_best.copy())]

    trial = 0
    sigmas = [0.04, 0.12, 0.25, 0.06, 0.18]
    # phase 1: basin hopping until ~125s elapsed
    PHASE1_END = t0 + 118.0
    while time.time() < PHASE1_END and time.time() < DEADLINE - 30:
        typ = trial % 6
        sigma = sigmas[trial % len(sigmas)]
        X_init = X_best.copy()
        if typ in (0, 1):
            # gaussian shake all points
            if trial % 17 == 5:
                sigma = 0.35
            noise = rng.standard_normal((n, d)) * sigma
            # extra kick to rattler / isolated points sometimes
            X_init = X_init + noise
            X_init /= np.linalg.norm(X_init, axis=1, keepdims=True)
        elif typ == 2:
            k = 5
            idx = rng.choice(n, size=k, replace=False)
            X_init[idx] = uniform_sphere(rng, k, d)
        elif typ == 3:
            k = 12
            idx = rng.choice(n, size=k, replace=False)
            X_init[idx] = uniform_sphere(rng, k, d)
        elif typ == 4:
            # contact-guided: replace overloaded points
            G = X_init @ X_init.T
            np.fill_diagonal(G, -2.0)
            gm = G.max()
            thr = gm - 0.02
            cnt = (G > thr).sum(axis=1)
            k = 8
            idx = np.argsort(-cnt)[:k]
            # replace half with far-point heuristic, half random
            X_init[idx] = uniform_sphere(rng, k, d)
        else:
            # isolate rattlers: find points with smallest row-max, shake hardest
            G = X_init @ X_init.T
            np.fill_diagonal(G, -2.0)
            rowmax = G.max(axis=1)
            order = np.argsort(rowmax)
            # shake the most jammed (largest rowmax) a bit + random reinsert of one
            k = 3
            idx = order[-k:]
            X_init[idx] += rng.standard_normal((k, d)) * 0.2
            X_init /= np.linalg.norm(X_init, axis=1, keepdims=True)
            # also move the most isolated to a greedy far location
            iso = order[0]
            # try 200 random candidates, pick farthest from others
            cand = uniform_sphere(rng, 200, d)
            dots = cand @ X_init.T
            dots[:, iso] = -2.0
            md = dots.max(axis=1)
            best_c = np.argmin(md)
            X_init[iso] = cand[best_c]

        X_ref = soft_anneal(X_init)
        v = max_cos(X_ref)
        if v < best_val - 1e-12:
            best_val = v
            X_best = X_ref.copy()
            elites.append((v, X_ref.copy()))
            elites = sorted(elites, key=lambda e: e[0])[:4]
        elif v < best_val + 0.005 and len(elites) < 6:
            # keep near-best diverse candidates
            if all(np.linalg.norm(X_ref - e[1]) > 1e-3 for e in elites):
                elites.append((v, X_ref.copy()))
                elites = sorted(elites, key=lambda e: e[0])[:6]
        trial += 1

    # phase 2: LP polish elites
    elites = sorted(elites, key=lambda e: e[0])[:3]
    # polish best first with most time
    for ei, (ev, ex) in enumerate(list(elites)):
        if time.time() > DEADLINE - 8:
            break
        # budget: split remaining time
        remain = DEADLINE - time.time() - 2.0
        if remain < 3:
            break
        budget = remain / (len(elites) - ei)
        budget = min(max(budget, 3.0), 25.0)
        xp = lp_polish(ex, seconds=budget, t_deadline=DEADLINE - 2.0)
        vp = max_cos(xp)
        if vp < best_val:
            best_val = vp
            X_best = xp.copy()
    X_best = X_best / np.linalg.norm(X_best, axis=1, keepdims=True)
    # tiny shrink guard + NaN guard
    X_best = np.nan_to_num(X_best, nan=0.0, posinf=0.0, neginf=0.0)
    nrm = np.linalg.norm(X_best, axis=1, keepdims=True)
    nrm[nrm == 0] = 1.0
    X_best = X_best / nrm
    return X_best
