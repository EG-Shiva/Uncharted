import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(np.max(g))


def adam_smooth(X0, steps, s0, s1, lr, max_step=0.02):
    X = X0.copy()
    n, d = X.shape
    m = np.zeros_like(X)
    v = np.zeros_like(X)
    b1 = 0.9
    b2 = 0.999
    eps = 1e-8
    if steps <= 0:
        return X
    log_ratio = np.log(s1 / s0) / steps
    for t in range(1, steps + 1):
        s = s0 * np.exp(log_ratio * t)
        G = X @ X.T
        np.fill_diagonal(G, -1.0)
        M = np.max(G)
        E = np.exp(s * (G - M))
        np.fill_diagonal(E, 0.0)
        S = np.sum(E) * 0.5
        if S <= 0 or not np.isfinite(S):
            break
        E /= S
        Grad = E @ X
        dot = np.einsum('ij,ij->i', Grad, X)[:, None]
        Gt = Grad - dot * X
        m = b1 * m + (1.0 - b1) * Gt
        v = b2 * v + (1.0 - b2) * (Gt * Gt)
        bc1 = 1.0 - b1 ** t
        bc2 = 1.0 - b2 ** t
        mhat = m / bc1
        vhat = v / bc2
        step = mhat / (np.sqrt(vhat) + eps)
        step *= lr
        nrm = np.sqrt(np.sum(step * step, axis=1, keepdims=True)) + 1e-12
        # clip per-point step
        scale = np.ones_like(nrm)
        mask = (nrm[:, 0] > max_step)
        if np.any(mask):
            scale[mask, 0] = max_step / nrm[mask, 0]
            step *= scale
        X -= step
        nrmX = np.sqrt(np.sum(X * X, axis=1, keepdims=True)) + 1e-30
        X /= nrmX
    return X


def greedy_reinsert_single(others, rng, n_try=2000):
    # others: (m,d) unit rows, return best unit vector minimizing max dot
    d = others.shape[1]
    C = rng.normal(size=(n_try, d))
    C /= np.sqrt(np.sum(C * C, axis=1, keepdims=True)) + 1e-30
    D = C @ others.T
    M = np.max(D, axis=1)
    b = int(np.argmin(M))
    return C[b]


def lp_polish(x, seconds, deadline=1e18):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = max_cos(x)
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and time.time() < deadline and radius > 1e-14:
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


def solve():
    t0 = time.time()
    BUDGET = 175.0
    rng = np.random.default_rng(0)
    n, d = 94, 13
    # load starting points
    bestX = None
    best = 1e9
    for fn in ["best.npy", "record.npy"]:
        try:
            a = np.load(fn).astype(float)
            if a.shape == (94, 13):
                a = a / np.linalg.norm(a, axis=1, keepdims=True)
                v = max_cos(a)
                if v < best:
                    best = v
                    bestX = a.copy()
        except Exception:
            pass
    if bestX is None:
        bestX = rng.normal(size=(n, d))
        bestX /= np.linalg.norm(bestX, axis=1, keepdims=True)
        best = max_cos(bestX)
    # candidate archive (value, matrix)
    cands = [(best, bestX.copy())]

    def add_cand(val, mat):
        # keep top 5 unique
        cands.append((val, mat.copy()))
        cands.sort(key=lambda z: z[0])
        # dedup very close duplicates by value only keep few
        del cands[5:]

    trial = 0
    PHASE1_END = t0 + 112.0
    # precompute degree helper thresholds
    while time.time() < PHASE1_END and time.time() - t0 < BUDGET - 45:
        trial += 1
        r = rng.random()
        if r < 0.14:
            # from scratch
            Y = rng.normal(size=(n, d))
            Y /= np.linalg.norm(Y, axis=1, keepdims=True)
            steps = 2800
            s0 = 25.0
            s1 = float(rng.choice(np.array([300.0, 500.0, 800.0])))
            lr = float(rng.choice(np.array([0.004, 0.007, 0.012])))
            Yopt = adam_smooth(Y, steps, s0, s1, lr, max_step=0.03)
        elif r < 0.55:
            # gaussian shake of whole code
            if rng.random() < 0.7:
                base = bestX
            else:
                base = cands[int(rng.integers(len(cands)))][1]
            sigma = float(rng.choice(np.array([0.015, 0.04, 0.10, 0.22, 0.35])))
            Y = base + sigma * rng.normal(size=(n, d))
            Y /= np.linalg.norm(Y, axis=1, keepdims=True)
            steps = int(rng.choice(np.array([1500, 2000, 2600])))
            s0 = 30.0
            s1 = float(rng.choice(np.array([200.0, 400.0, 700.0])))
            lr = float(rng.choice(np.array([0.003, 0.006, 0.010])))
            Yopt = adam_smooth(Y, steps, s0, s1, lr, max_step=0.025)
        else:
            # kill and greedy reinsert
            if rng.random() < 0.6:
                base = bestX.copy()
            else:
                base = cands[int(rng.integers(len(cands)))][1].copy()
            # compute degrees to bias killing
            try:
                G = base @ base.T
                np.fill_diagonal(G, -2.0)
                M = float(np.max(G))
                thr = M - 5e-4
                deg = np.sum(G > thr, axis=1)
                order = np.argsort(deg)
            except Exception:
                order = np.arange(n)
            k = int(rng.integers(2, 9))
            kill_idx = []
            # mix low-degree and random
            nlow = min(k, int(rng.integers(1, k + 1)))
            perm = rng.permutation(n)
            # low-degree candidates
            for ii in range(nlow):
                kill_idx.append(int(order[ii % n]))
            # random rest
            for xxi in perm:
                if len(kill_idx) >= k:
                    break
                if int(xxi) not in kill_idx:
                    kill_idx.append(int(xxi))
            Y = base.copy()
            # randomise killed rows first
            for idx in kill_idx:
                # greedy best of many tries against current others
                others = np.delete(Y, idx, axis=0)
                # fewer tries for speed when k large
                nt = 800 if k > 4 else 1500
                Y[idx] = greedy_reinsert_single(others, rng, n_try=nt)
            steps = int(rng.choice(np.array([1500, 2200])))
            s0 = 30.0
            s1 = float(rng.choice(np.array([250.0, 500.0])))
            lr = float(rng.choice(np.array([0.004, 0.008])))
            Yopt = adam_smooth(Y, steps, s0, s1, lr, max_step=0.025)
        try:
            val = max_cos(Yopt)
        except Exception:
            continue
        if not np.isfinite(val):
            continue
        # keep if promising (within 0.01 of best or better than worst cand)
        if val < cands[-1][0] + 0.01 or val < best:
            add_cand(val, Yopt)
        if val < best - 1e-12:
            best = val
            bestX = Yopt.copy()
        # opportunistic: if we found clear improvement, update best immediately
        # time check for loop overhead
        if time.time() >= PHASE1_END:
            break

    # Phase 2: LP polish top candidates with remaining time
    # sort unique candidates
    cands.sort(key=lambda z: z[0])
    # build polish list: bestX + top 3 cands
    seen = []
    plist = []
    plist.append(bestX.copy())
    for v, m in cands:
        if len(plist) >= 4:
            break
        # skip near-duplicate of already listed (frobenius dist tiny)
        dup = False
        for q in plist:
            try:
                if np.linalg.norm(m - q) < 1e-9 or np.linalg.norm(m + q) < 1e-9:
                    dup = True
                    break
            except Exception:
                pass
        if not dup:
            plist.append(m.copy())
    remaining = BUDGET - (time.time() - t0) - 2.0
    if remaining < 5:
        remaining = 5.0
    # allocate more time to most promising
    # polish in order of promise
    # order plist by raw max_cos
    vals = []
    for m in plist:
        try:
            vals.append(max_cos(m))
        except Exception:
            vals.append(1e9)
    o = np.argsort(np.array(vals))
    plist = [plist[i] for i in o]
    per = remaining / max(1, len(plist))
    per = min(per, 18.0)
    for m in plist:
        if time.time() - t0 > BUDGET - 3.0:
            break
        try:
            mp = lp_polish(m, seconds=per, deadline=t0 + BUDGET - 2.0)
            v = max_cos(mp)
            if v < best:
                best = v
                bestX = mp.copy()
        except Exception:
            continue
    bestX = bestX / np.linalg.norm(bestX, axis=1, keepdims=True)
    # final safety: replace non-finite
    bestX = np.where(np.isfinite(bestX), bestX, 0.0)
    nr = np.linalg.norm(bestX, axis=1, keepdims=True)
    nr[nr == 0] = 1.0
    bestX = bestX / nr
    return bestX
