import time
import numpy as np
from scipy.optimize import linprog, minimize
from scipy.sparse import coo_matrix

def max_cos(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def per_point_max(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return G.max(axis=1)

def make_fg(n, d, IU0, beta):
    def fg(yf):
        Y = yf.reshape(n, d)
        nr = np.linalg.norm(Y, axis=1, keepdims=True)
        nr = np.maximum(nr, 1e-12)
        X = Y / nr
        G = X @ X.T
        g = G[IU0]
        m = float(g.max())
        e = np.exp(beta * (g - m))
        S = float(e.sum()) + 1e-300
        F = m + np.log(S) / beta
        w = e / S
        P = np.zeros((n, n))
        P[IU0] = w
        P = P + P.T
        GX = P @ X
        dot = np.sum(GX * X, axis=1, keepdims=True)
        GY = (GX - dot * X) / nr
        return F, GY.ravel()
    return fg

def run_lbfgs(x0, beta, maxiter):
    n, d = x0.shape
    IU0 = np.triu_indices(n, 1)
    fg = make_fg(n, d, IU0, beta)
    y0 = (x0 / np.linalg.norm(x0, axis=1, keepdims=True)).ravel().copy()
    try:
        r = minimize(fg, y0, jac=True, method='L-BFGS-B',
                     options={'maxiter': maxiter, 'ftol': 1e-16, 'gtol': 1e-8, 'maxfun': maxiter*2})
        Y = r.x.reshape(n, d)
        Y /= np.linalg.norm(Y, axis=1, keepdims=True)
        return Y
    except Exception:
        return x0

def lp_creep(x, seconds, r0=3e-6):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = max_cos(x)
    radius = r0
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-13:
        if time.time() - start > seconds - 0.2:
            break
        G = x @ x.T
        iu_c = G[iu]
        m = float(iu_c.max())
        # tiny window: only near-max pairs
        win = max(8.0 * radius, 2e-6)
        sel = iu_c > m - win
        if np.sum(sel) == 0:
            radius *= 0.5
            continue
        # cap number of constraints for speed: keep tightest 600
        if np.sum(sel) > 600:
            thr = np.sort(iu_c[sel])[-600]
            sel = iu_c >= thr
        I, J = iu[0][sel], iu[1][sel]
        c = iu_c[sel]
        p, nv = len(I), n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        except Exception:
            radius *= 0.5
            continue
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs",
                          options={"presolve": True})
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        v = max_cos(y)
        if v < best - 1e-15:
            x, best = y, v
            radius = min(radius * 1.3, 2e-5)
        else:
            radius *= 0.5
            if v < best:
                x, best = y, v
    return x

def best_hole(x_rest, rng, ncand=12000):
    d = x_rest.shape[1]
    cand = rng.standard_normal((ncand, d))
    cand /= np.linalg.norm(cand, axis=1, keepdims=True)
    scores = (cand @ x_rest.T).max(axis=1)
    return cand[int(np.argmin(scores))]

def solve():
    t0 = time.time()
    TLIM = 172.0
    try:
        xb = np.load("best.npy").astype(float)
    except Exception:
        xb = None
    try:
        xr = np.load("record.npy").astype(float)
    except Exception:
        xr = None
    cands = []
    if xb is not None:
        cands.append(xb)
    if xr is not None:
        cands.append(xr)
    if len(cands) == 0:
        rng0 = np.random.default_rng(0)
        x_init = rng0.standard_normal((68, 12))
        x_init /= np.linalg.norm(x_init, axis=1, keepdims=True)
    else:
        bv = 1e9
        x_init = cands[0]
        for c in cands:
            c = c / np.linalg.norm(c, axis=1, keepdims=True)
            v = max_cos(c)
            if v < bv:
                bv = v
                x_init = c.copy()
    x_init = x_init / np.linalg.norm(x_init, axis=1, keepdims=True)
    best_val = max_cos(x_init)
    x_best = x_init.copy()
    # initial creep to immediately inch down the isolated max
    if TLIM - (time.time() - t0) > 10:
        try:
            y = lp_creep(x_best, 3.0, r0=3e-6)
            v = max_cos(y)
            if v < best_val:
                x_best, best_val = y, v
        except Exception:
            pass
    n, d = x_best.shape
    rng = np.random.default_rng(777)
    explore_end = t0 + TLIM - 55.0
    trial = 0
    while time.time() < explore_end and trial < 400:
        trial += 1
        x = x_best.copy()
        pp = per_point_max(x)
        order = np.argsort(-pp)
        u = rng.random()
        if u < 0.45:
            # attack worst pair directly
            G = x @ x.T
            np.fill_diagonal(G, -2.0)
            ij = np.argmax(G)
            wi, wj = int(ij // n), int(ij % n)
            # reinsert one end into best hole or kick both
            if rng.random() < 0.5:
                mask = np.ones(n, dtype=bool); mask[wi] = False
                x[wi] = best_hole(x[mask], rng, ncand=12000)
                # shake the other end + neighbours
                nbrs = np.argsort(-(x @ x[wj]))[:5]
                x[nbrs] += 0.01 * rng.standard_normal((len(nbrs), d))
            else:
                x[wi] += float(rng.uniform(0.002, 0.05)) * rng.standard_normal(d)
                x[wj] += float(rng.uniform(0.002, 0.05)) * rng.standard_normal(d)
                # shake a few other worst
                k = int(rng.integers(1, 4))
                idx = order[:k]
                x[idx] += 0.01 * rng.standard_normal((len(idx), d))
            x += 0.0015 * rng.standard_normal(x.shape)
            x /= np.linalg.norm(x, axis=1, keepdims=True)
        elif u < 0.70:
            # kick top-k worst points
            k = int(rng.integers(2, 6))
            idx = order[:k]
            if rng.random() < 0.4:
                for q in idx:
                    vv = rng.standard_normal(d); vv /= np.linalg.norm(vv)
                    x[q] = vv
            else:
                sig = float(rng.uniform(0.01, 0.12))
                x[idx] += sig * rng.standard_normal((len(idx), d))
            x /= np.linalg.norm(x, axis=1, keepdims=True)
        elif u < 0.85:
            # small global shake
            sig = float(rng.choice([0.003, 0.008, 0.02, 0.05]))
            x = x + sig * rng.standard_normal(x.shape)
            x /= np.linalg.norm(x, axis=1, keepdims=True)
        else:
            # block re-init around random center
            c0 = int(rng.integers(0, n))
            dots = x @ x[c0]
            oo = np.argsort(-dots)
            k = int(rng.integers(3, 10))
            idx = oo[:k] if rng.random() > 0.3 else rng.choice(n, size=k, replace=False)
            for q in idx:
                vv = rng.standard_normal(d); vv /= np.linalg.norm(vv)
                x[q] = vv
            x += 0.003 * rng.standard_normal(x.shape)
            x /= np.linalg.norm(x, axis=1, keepdims=True)
        b1 = float(rng.uniform(60, 150))
        b2 = float(rng.uniform(300, 900))
        m1 = int(rng.integers(60, 140))
        m2 = int(rng.integers(60, 160))
        x = run_lbfgs(x, b1, m1)
        if time.time() > explore_end:
            v = max_cos(x)
            if v < best_val:
                x_best, best_val = x.copy(), v
            break
        x = run_lbfgs(x, b2, m2)
        v = max_cos(x)
        if v < best_val - 1e-12:
            # squeeze with creep if promising
            if v < best_val - 2e-7 and (explore_end - time.time()) > 3.0:
                x = lp_creep(x, min(2.5, explore_end - time.time() - 0.2), r0=3e-6)
                v = max_cos(x)
            if v < best_val:
                x_best, best_val = x.copy(), v
        elif v < best_val + 0.002:
            if (trial % 7 == 0) and (time.time() < explore_end - 3.0):
                y = lp_creep(x, 1.0, r0=2e-6)
                v2 = max_cos(y)
                if v2 < best_val:
                    x_best, best_val = y.copy(), v2
        if time.time() > explore_end:
            break
    # final intensive squeeze: high-sharp LBFGS + creep cycles
    remain = TLIM - (time.time() - t0)
    if remain > 4.0:
        try:
            # alternate sharpness
            for beta, it in [(800.0, 120), (2000.0, 120), (5000.0, 80)]:
                if TLIM - (time.time() - t0) < 6.0:
                    break
                x_best = run_lbfgs(x_best, beta, it)
                v = max_cos(x_best)
                if v < best_val:
                    best_val = v
                # short creep between sharp rounds
                rr = TLIM - (time.time() - t0)
                if rr > 8.0:
                    y = lp_creep(x_best, min(4.0, rr - 5.0), r0=2e-6)
                    vy = max_cos(y)
                    if vy <= best_val:
                        x_best, best_val = y, vy
        except Exception:
            pass
    remain = TLIM - (time.time() - t0)
    if remain > 3.0:
        try:
            y = lp_creep(x_best, max(0.5, remain - 1.0), r0=3e-6)
            vy = max_cos(y)
            if vy < best_val:
                x_best, best_val = y, vy
        except Exception:
            pass
    x_best = x_best / np.linalg.norm(x_best, axis=1, keepdims=True)
    return x_best
