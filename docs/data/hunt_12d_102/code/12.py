import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def anneal_adam(x_init, steps, T0, T1, lr0, lr1, beta1=0.9, beta2=0.999, eps=1e-8, noise=0.0, rng=None):
    x = x_init.copy()
    n, d = x.shape
    m = np.zeros_like(x)
    v = np.zeros_like(x)
    # exponential temp schedule via log interp
    lt0 = np.log(T0); lt1 = np.log(T1)
    for s in range(steps):
        frac = s / max(steps - 1, 1)
        T = float(np.exp(lt0 + (lt1 - lt0) * frac))
        lrs = float(lr0 + (lr1 - lr0) * frac)
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        mval = float(G.max())
        E = np.exp((G - mval) / T)
        np.fill_diagonal(E, 0.0)
        S = float(E.sum())
        if S == 0 or not np.isfinite(S):
            break
        P = E / S
        Grad = P @ x
        # tangent projection
        dot = np.sum(Grad * x, axis=1, keepdims=True)
        Gt = Grad - dot * x
        m = beta1 * m + (1 - beta1) * Gt
        v = beta2 * v + (1 - beta2) * (Gt * Gt)
        bc1 = 1.0 - beta1 ** (s + 1)
        bc2 = 1.0 - beta2 ** (s + 1)
        mhat = m / bc1
        vhat = v / bc2
        step = lrs * mhat / (np.sqrt(vhat) + eps)
        # keep step tangential
        sn = np.sum(step * x, axis=1, keepdims=True)
        step = step - sn * x
        x = x - step
        if noise > 0 and rng is not None:
            nz = rng.normal(size=x.shape) * noise
            nz = nz - np.sum(nz * x, axis=1, keepdims=True) * x
            x = x + nz
        nrm = np.linalg.norm(x, axis=1, keepdims=True)
        nrm[nrm == 0] = 1.0
        x = x / nrm
    return x

def lp_polish(x, seconds):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = max_cos(x)
    radius, start = 2e-3, time.time()
    iu = np.triu_indices(n, 1)
    # cache gram triu to avoid recompute? recompute each iter
    while time.time() - start < seconds and radius > 1e-14:
        cfull = (x @ x.T)[iu]
        m = float(cfull.max())
        sel = cfull > m - max(25 * radius, 2e-9)
        I, J = iu[0][sel], iu[1][sel]
        p, nv = len(I), n * d + 1
        if p == 0:
            break
        # cap constraints to hardest 2500 to keep HiGHS fast
        if p > 2500:
            # keep largest
            idx = np.argpartition(-cfull[sel], 2500)[:2500]
            I, J = I[idx], J[idx]
            p = 2500
            csel = cfull[sel][idx]
        else:
            csel = cfull[sel]
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
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
        vv = max_cos(y)
        if vv < best:
            x, best, radius = y, vv, min(radius * 1.6, 0.05)
        else:
            radius *= 0.5
    return x

def refine_hole(q, Xref, steps=150, T=0.004, lr=0.05, rng=None):
    # minimize softmax max dot to fixed set Xref
    for _ in range(steps):
        dots = Xref @ q
        mm = float(dots.max())
        w = np.exp((dots - mm) / T)
        s = float(w.sum())
        if s == 0 or not np.isfinite(s):
            break
        w /= s
        g = w @ Xref
        g = g - float(g @ q) * q
        q = q - lr * g
        nq = float(np.linalg.norm(q))
        if nq == 0 or not np.isfinite(nq):
            break
        q /= nq
    return q

def solve():
    t0 = time.time()
    BUDGET = 174.0
    rng = np.random.default_rng(42)
    try:
        xb = np.load("best.npy").astype(float)
    except Exception:
        try:
            xb = np.load("record.npy").astype(float)
        except Exception:
            xb = rng.normal(size=(102, 12))
    xb = xb / np.linalg.norm(xb, axis=1, keepdims=True)
    if xb.shape != (102, 12):
        tmp = rng.normal(size=(102, 12))
        tmp /= np.linalg.norm(tmp, axis=1, keepdims=True)
        m0 = min(xb.shape[0], 102); m1 = min(xb.shape[1], 12)
        tmp[:m0, :m1] = xb[:m0, :m1]
        tmp /= np.linalg.norm(tmp, axis=1, keepdims=True)
        xb = tmp
    n, d = 102, 12
    best = max_cos(xb)
    x_best = xb.copy()

    HOP_END = t0 + 112.0
    trial = 0
    fail = 0
    # precompute candidate pool for holes (refresh periodically)
    cand_pool = rng.normal(size=(7000, d))
    cand_pool /= np.linalg.norm(cand_pool, axis=1, keepdims=True)

    while time.time() < HOP_END and (time.time() - t0) < BUDGET - 55:
        trial += 1
        # refresh pool occasionally
        if trial % 12 == 0:
            cand_pool = rng.normal(size=(7000, d))
            cand_pool /= np.linalg.norm(cand_pool, axis=1, keepdims=True)
        x_try = x_best.copy()
        u = rng.random()
        # adaptive magnitude grows with fail streak
        boost = min(1.6, 1.0 + fail / 25.0)
        if u < 0.32:
            sigma = float(rng.uniform(0.02, 0.11) * boost)
            x_try = x_try + rng.normal(size=x_try.shape) * sigma
            x_try /= np.linalg.norm(x_try, axis=1, keepdims=True)
        elif u < 0.62:
            G = x_best @ x_best.T
            np.fill_diagonal(G, -2.0)
            mval = float(G.max())
            cnt = (G > mval - 0.004).sum(axis=1)
            order = np.argsort(-cnt, kind="stable")
            k = int(rng.integers(6, 26))
            targets = order[:k]
            sig_small = float(rng.uniform(0.003, 0.010))
            sig_big = float(rng.uniform(0.07, 0.22) * boost)
            noise = rng.normal(size=x_try.shape) * sig_small
            noise[targets] += rng.normal(size=(k, d)) * sig_big
            x_try = x_try + noise
            x_try /= np.linalg.norm(x_try, axis=1, keepdims=True)
        else:
            # hole re-insertion of worst points
            G = x_best @ x_best.T
            np.fill_diagonal(G, -2.0)
            mval = float(G.max())
            # score per point: max + count
            mx = G.max(axis=1)
            cnt = (G > mval - 0.005).sum(axis=1)
            score = mx + 0.002 * cnt
            order = np.argsort(-score, kind="stable")
            nk = int(rng.integers(1, 6))  # 1..5
            worst = order[:nk].copy()
            mask = np.ones(n, dtype=bool); mask[worst] = False
            Xcur = x_best[mask]
            new_pts = []
            # greedy farthest from current set
            for _ in worst:
                sc = (cand_pool @ Xcur.T).max(axis=1)
                # pick among best few randomly for diversity
                bi = int(np.argmin(sc))
                # random choice among 8 best holes
                part = np.argpartition(sc, 8)[:8]
                bi = part[int(rng.integers(0, len(part)))]
                q = cand_pool[bi].copy()
                q = refine_hole(q, Xcur, steps=180, T=float(rng.uniform(0.003, 0.008)), lr=0.06)
                new_pts.append(q)
                Xcur = np.vstack([Xcur, q[None, :]])
            for wi, q in zip(worst, new_pts):
                x_try[wi] = q
            x_try = x_try + rng.normal(size=x_try.shape) * 0.004
            x_try /= np.linalg.norm(x_try, axis=1, keepdims=True)

        steps = int(rng.integers(800, 2000))
        # time guard
        t_rem_hop = HOP_END - time.time()
        est = steps * 0.00045
        if est > t_rem_hop - 0.3:
            steps = max(250, int((t_rem_hop - 0.3) / 0.00045))
            if steps < 250:
                break
        T0p = float(rng.uniform(0.010, 0.032))
        T1p = float(rng.uniform(0.0004, 0.0022))
        lr0 = float(rng.uniform(0.004, 0.012))
        lr1 = float(lr0 * rng.uniform(0.25, 0.7))
        x_pol = anneal_adam(x_try, steps, T0p, T1p, lr0, lr1)
        val = max_cos(x_pol)
        if val < best - 3e-9:
            # snap with short LP
            remain = HOP_END - time.time()
            snap_t = min(2.5, max(0.7, remain * 0.06))
            if snap_t > 0.6 and (time.time() - t0) < BUDGET - 55:
                try:
                    x_pol2 = lp_polish(x_pol, snap_t)
                    val2 = max_cos(x_pol2)
                    # keep snapped if not worse
                    if val2 <= val + 1e-12:
                        x_pol, val = x_pol2, val2
                except Exception:
                    pass
            if val < best:
                best = val
                x_best = x_pol.copy()
                fail = 0
            else:
                fail += 1
        else:
            fail += 1

    # middle intensification at low T
    try:
        if (time.time() - t0) < BUDGET - 50:
            x_best = anneal_adam(x_best, 900, 0.006, 0.0005, 0.005, 0.002)
            best = max_cos(x_best)
    except Exception:
        pass
    # final long LP polish with remaining time
    rem = BUDGET - (time.time() - t0) - 1.5
    if rem > 4:
        try:
            x_best = lp_polish(x_best, min(rem, 60.0))
        except Exception:
            pass
    else:
        # at least tiny polish
        try:
            x_best = lp_polish(x_best, max(1.0, rem))
        except Exception:
            pass
    x_best = x_best / np.linalg.norm(x_best, axis=1, keepdims=True)
    x_best = np.nan_to_num(x_best, nan=0.0, posinf=0.0, neginf=0.0)
    nrm = np.linalg.norm(x_best, axis=1, keepdims=True)
    nrm[nrm == 0] = 1.0
    x_best = x_best / nrm
    return x_best
