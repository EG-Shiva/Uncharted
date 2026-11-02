import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def lp_polish(x, seconds):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = max_cos(x)
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        # time guard per iteration
        if time.time() - start > seconds - 1.0:
            break
        cfull = (x @ x.T)[iu]
        m = cfull.max()
        thr = m - max(20 * radius, 2e-9)
        sel = cfull > thr
        # cap LP size for speed
        idx = np.where(sel)[0]
        if len(idx) > 1300:
            # keep largest 1300
            part = np.argpartition(cfull[idx], len(idx) - 1300)[-1300:]
            idx2 = idx[part]
            sel = np.zeros_like(sel)
            sel[idx2] = True
        I, J = iu[0][sel], iu[1][sel]
        csel = cfull[sel]
        p = len(I)
        if p == 0:
            radius *= 0.5
            continue
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        # cols: I-block, J-block, t
        # build cols efficiently
        ar = np.arange(d)
        cols_I = (I[:, None] * d + ar).ravel()
        cols_J = (J[:, None] * d + ar).ravel()
        cols = np.empty(p * (2 * d + 1), dtype=np.int64)
        cols[0::(2*d+1)] = 0  # placeholder, will overwrite blockwise below
        # faster: construct via concatenate
        cols = np.concatenate([cols_I.reshape(p, d) + 0, cols_J.reshape(p, d), np.full((p, 1), n * d)], axis=1).ravel() if False else None
        # simpler construction:
        C1 = (I[:, None] * d + ar)  # p x d
        C2 = (J[:, None] * d + ar)
        cols = np.concatenate([C1, C2, np.full((p, 1), n * d)], axis=1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], axis=1).ravel()
        rows2 = np.repeat(np.arange(p), 2 * d + 1)
        A_ub = coo_matrix((vals, (rows2, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv); obj[-1] = 1.0
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
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        v = max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def quench_run(x, steps, tau0, tau1, lr0, lr1, mu, noise0, rng):
    v = np.zeros_like(x)
    ltau = np.log(tau0)
    ltau1 = np.log(tau1)
    llr0 = np.log(lr0); llr1 = np.log(lr1)
    for s in range(steps):
        frac = s / max(steps - 1, 1)
        tau = float(np.exp(ltau + (ltau1 - ltau) * frac))
        lr = float(np.exp(llr0 + (llr1 - llr0) * frac))
        G = x @ x.T
        np.fill_diagonal(G, -10.0)
        cmax = float(G.max())
        # softmax weights
        E = np.exp((G - cmax) / tau)
        np.fill_diagonal(E, 0.0)
        Sp = E.sum() * 0.5
        if Sp < 1e-12:
            Sp = 1e-12
        grad = (E @ x) / Sp
        dot = (grad * x).sum(axis=1, keepdims=True)
        gt = grad - dot * x
        v = mu * v + gt
        if noise0 > 0:
            nz = rng.standard_normal(x.shape)
            nz -= (nz * x).sum(axis=1, keepdims=True) * x
            amp = noise0 * (1.0 - frac)
            v = v + amp * nz
        x = x - lr * v
        x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x

def solve():
    rng = np.random.default_rng(0)
    t0 = time.time()
    TIME_BUDGET = 168.0
    POLISH_TIME = 55.0

    # load base
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
        # random fallback
        z = rng.standard_normal((58, 22))
        z /= np.linalg.norm(z, axis=1, keepdims=True)
        cands.append(z)
    best_x = None
    best_v = 1e9
    for c in cands:
        c = c / np.linalg.norm(c, axis=1, keepdims=True)
        # clean nan
        if not np.all(np.isfinite(c)):
            continue
        vv = max_cos(c)
        if vv < best_v:
            best_v = vv
            best_x = c.copy()
    base = best_x.copy()
    base_v = best_v

    # contact degrees of base for targeted kicks
    def degrees(x, tol=1e-4):
        G = x @ x.T
        np.fill_diagonal(G, -2)
        m = G.max()
        B = G > (m - tol)
        return B.sum(axis=1), m

    deg, cmax0 = degrees(base)

    trial = 0
    # time for search phase
    search_end = t0 + TIME_BUDGET - POLISH_TIME - 5.0
    # hyper sets cycling
    hypers = [
        (0.030, 0.004, 0.030, 0.003, 0.85, 0.004),
        (0.020, 0.003, 0.020, 0.002, 0.80, 0.002),
        (0.050, 0.005, 0.040, 0.004, 0.88, 0.006),
        (0.015, 0.0025, 0.015, 0.002, 0.75, 0.001),
    ]
    sigmas = [0.03, 0.06, 0.10, 0.15, 0.22, 0.05, 0.08, 0.12]

    n, d = base.shape
    while time.time() < search_end and trial < 40:
        hi = hypers[trial % len(hypers)]
        tau0, tau1, lr0, lr1, mu, noise0 = hi
        sigma = sigmas[trial % len(sigmas)]
        # perturb
        mode = trial % 3
        x0 = base.copy()
        if mode == 0:
            # global gaussian kick, degree-weighted
            w = 1.0 + 1.5 * (deg / max(deg.max(), 1))
            kick = rng.standard_normal(x0.shape) * (sigma * w[:, None] / np.sqrt(d))
            # make tangential-ish
            kick -= (kick * x0).sum(axis=1, keepdims=True) * x0
            x0 = x0 + kick
            x0 /= np.linalg.norm(x0, axis=1, keepdims=True)
        elif mode == 1:
            # global kick uniform + reinsert worst point randomly
            kick = rng.standard_normal(x0.shape) * (sigma / np.sqrt(d))
            kick -= (kick * x0).sum(axis=1, keepdims=True) * x0
            x0 = x0 + kick
            # worst point
            G = x0 @ x0.T
            np.fill_diagonal(G, -2)
            rowmax = G.max(axis=1)
            wi = int(np.argmax(rowmax))
            z = rng.standard_normal(d)
            z -= (z @ x0[wi]) * x0[wi]
            # orthogonalize a bit against its worst partner
            pj = int(np.argmax(G[wi]))
            z -= (z @ x0[pj]) * x0[pj]
            nz = np.linalg.norm(z)
            if nz < 1e-12:
                z = rng.standard_normal(d); nz = np.linalg.norm(z)
            x0[wi] = z / nz
            x0 /= np.linalg.norm(x0, axis=1, keepdims=True)
        else:
            # two-point hole: remove two worst overlapping points
            G = x0 @ x0.T
            np.fill_diagonal(G, -2)
            # find worst pair
            iu0 = np.unravel_index(int(np.argmax(G)), G.shape)
            for kk in list(iu0):
                z = rng.standard_normal(d)
                # try to be far from all
                # pick random and keep best of 5 tries (min max dot)
                bestz = None; bests = 1e9
                for _ in range(8):
                    zz = rng.standard_normal(d); zz /= np.linalg.norm(zz)
                    s = float(np.max(np.abs(x0 @ zz)))
                    if s < bests:
                        bests = s; bestz = zz
                x0[kk] = bestz
            kick = rng.standard_normal(x0.shape) * (sigma * 0.5 / np.sqrt(d))
            kick -= (kick * x0).sum(axis=1, keepdims=True) * x0
            x0 = x0 + kick
            x0 /= np.linalg.norm(x0, axis=1, keepdims=True)

        steps = 2500 if time.time() < search_end - 30 else 1500
        # slightly vary steps
        if trial % 4 == 3:
            steps = 3500
        x1 = quench_run(x0, steps, tau0, tau1, lr0, lr1, mu, noise0, rng)
        v1 = max_cos(x1)
        if v1 < best_v and np.all(np.isfinite(x1)):
            best_v = v1
            best_x = x1.copy()
            # update base to new best occasionally to intensify? keep base as best
            base = best_x.copy()
            deg, _ = degrees(base)
        trial += 1
        # opportunistic early exit if time low
        if time.time() >= search_end:
            break

    # final polish of best found
    remain = TIME_BUDGET - (time.time() - t0) - 2.0
    pt = min(POLISH_TIME, max(10.0, remain))
    if pt > 5:
        best_x = lp_polish(best_x, pt)
    # final normalize, guard
    best_x = best_x / np.linalg.norm(best_x, axis=1, keepdims=True)
    # tiny shrink not needed; ensure finite
    best_x = np.where(np.isfinite(best_x), best_x, base)
    best_x = best_x / np.linalg.norm(best_x, axis=1, keepdims=True)
    return best_x
