import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())


def adam_refine(x0, steps, a0, a1, l0, l1):
    x = x0.copy()
    n, d = x.shape
    m1 = np.zeros_like(x)
    m2 = np.zeros_like(x)
    b1, b2, eps = 0.9, 0.999, 1e-8
    log_a_ratio = np.log(a1 / a0)
    log_l_ratio = np.log(l1 / l0)
    for s in range(steps):
        frac = s / max(1, steps - 1)
        alpha = a0 * np.exp(log_a_ratio * frac)
        lr = l0 * np.exp(log_l_ratio * frac)
        G = x @ x.T
        np.fill_diagonal(G, -10.0)
        gmax = G.max()
        E = np.exp(alpha * (G - gmax))
        np.fill_diagonal(E, 0.0)
        ss = E.sum()
        if ss <= 0 or not np.isfinite(ss):
            break
        grad = (E @ x) / ss
        dot = np.sum(grad * x, axis=1, keepdims=True)
        gt = grad - dot * x
        m1 = b1 * m1 + (1.0 - b1) * gt
        m2 = b2 * m2 + (1.0 - b2) * (gt * gt)
        c1 = 1.0 - b1 ** (s + 1)
        c2 = 1.0 - b2 ** (s + 1)
        mh = m1 / c1
        vh = m2 / c2
        step = mh / (np.sqrt(vh) + eps)
        sdot = np.sum(step * x, axis=1, keepdims=True)
        stept = step - sdot * x
        x = x - lr * stept
        nr = np.linalg.norm(x, axis=1, keepdims=True)
        nr = np.maximum(nr, 1e-12)
        x = x / nr
    return x


def lp_polish(x, seconds, deadline=None):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = max_cos(x)
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds:
        if deadline is not None and time.time() > deadline:
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
    TIME_LIMIT = 172.0
    FINAL_LP = 22.0
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
    if xb is not None:
        cands.append(np.asarray(xb, dtype=float))
    if xr is not None:
        cands.append(np.asarray(xr, dtype=float))
    if len(cands) == 0:
        cands.append(rng.normal(size=(55, 21)))
    # normalize and pick best
    best_val = 1e9
    best_x = None
    for c in cands:
        c = c / np.linalg.norm(c, axis=1, keepdims=True)
        v = max_cos(c)
        if v < best_val:
            best_val = v
            best_x = c.copy()
    n, d = best_x.shape
    # quick initial Adam squeeze to ensure starting point is tight
    # (short, cheap)
    try:
        tmp = adam_refine(best_x, 800, 40.0, 1500.0, 0.008, 0.001)
        v = max_cos(tmp)
        if v < best_val:
            best_val = v
            best_x = tmp
    except Exception:
        pass

    trial = 0
    no_improve = 0
    # contact counts for targeted shakes (computed from best)
    while True:
        now = time.time()
        if now - t0 > TIME_LIMIT - FINAL_LP - 2.0:
            break
        trial += 1
        r = rng.random()
        cand = best_x.copy()
        if r < 0.55:
            # local shake of 1-3 points
            k = int(rng.integers(1, 4))
            idx = rng.choice(n, k, replace=False)
            sigma = 0.015 + 0.05 * rng.random()
            cand[idx] += rng.normal(0, sigma, size=(k, d))
            cand += rng.normal(0, 0.004, size=(n, d))
        elif r < 0.80:
            # medium shake 4-11 points, focus on overloaded points if known
            k = int(rng.integers(4, 12))
            # pick worst points by current max-row
            if rng.random() < 0.6:
                G = best_x @ best_x.T
                np.fill_diagonal(G, -2.0)
                # score per point = max cosine in its row
                s = G.max(axis=1)
                # take top 20 as candidates
                order = np.argsort(-s)
                pool = order[:20]
                idx = rng.choice(pool, min(k, len(pool)), replace=False)
                # pad if needed
                if len(idx) < k:
                    extra = rng.choice(n, k - len(idx), replace=False)
                    idx = np.concatenate([idx, extra])
            else:
                idx = rng.choice(n, k, replace=False)
            sigma = 0.07 + 0.08 * rng.random()
            cand[idx] += rng.normal(0, sigma, size=(len(idx), d))
            cand += rng.normal(0, 0.006, size=(n, d))
        elif r < 0.92:
            # global shake
            sigma = 0.18 + 0.15 * rng.random()
            cand += rng.normal(0, sigma, size=(n, d))
        else:
            # worst-pair break + reinsert one point randomly
            G = best_x @ best_x.T
            np.fill_diagonal(G, -2.0)
            ij = np.argmax(G)
            ii, jj = divmod(int(ij), n)
            # move ii away from jj
            vdir = best_x[ii] - best_x[jj] * float(G[ii, jj])
            vdir /= (np.linalg.norm(vdir) + 1e-12)
            cand[ii] = best_x[ii] + 0.25 * vdir + rng.normal(0, 0.05, size=(d,))
            # reinsert a random worst point far away
            if rng.random() < 0.5:
                s = G.max(axis=1)
                wi = int(np.argmax(s))
                # random vector orthogonal-ish to all: just random
                rv = rng.normal(size=(d,))
                rv /= np.linalg.norm(rv)
                cand[wi] = rv + 0.2 * rng.normal(size=(d,))
            cand += rng.normal(0, 0.01, size=(n, d))
        cand /= np.linalg.norm(cand, axis=1, keepdims=True)
        # choose polish length adaptively
        if r < 0.55:
            steps = 1200
            a0, a1 = 50.0, 2500.0
            l0, l1 = 0.008, 0.0008
        elif r < 0.80:
            steps = 1500
            a0, a1 = 35.0, 2000.0
            l0, l1 = 0.009, 0.001
        else:
            steps = 1800
            a0, a1 = 25.0, 1800.0
            l0, l1 = 0.010, 0.001
        try:
            polished = adam_refine(cand, steps, a0, a1, l0, l1)
        except Exception:
            no_improve += 1
            continue
        v = max_cos(polished)
        if v < best_val - 1e-12:
            best_val = v
            best_x = polished
            no_improve = 0
        else:
            no_improve += 1
        # occasional diversification: if stuck long, big random restart polish
        if no_improve >= 40:
            try:
                rc = rng.normal(size=(n, d))
                rc /= np.linalg.norm(rc, axis=1, keepdims=True)
                # mix with best to keep some structure
                alpha_mix = 0.7
                rc = alpha_mix * rc + (1 - alpha_mix) * best_x
                rc /= np.linalg.norm(rc, axis=1, keepdims=True)
                pol2 = adam_refine(rc, 1500, 25.0, 1800.0, 0.01, 0.001)
                v2 = max_cos(pol2)
                if v2 < best_val - 1e-12:
                    best_val = v2
                    best_x = pol2
                    no_improve = 0
                else:
                    no_improve = 0  # reset anyway to avoid loop
            except Exception:
                no_improve = 0
    # final long fine polish with high alpha
    try:
        if time.time() - t0 < TIME_LIMIT - FINAL_LP:
            fine = adam_refine(best_x, 2500, 80.0, 6000.0, 0.004, 0.0004)
            vf = max_cos(fine)
            if vf < best_val:
                best_val = vf
                best_x = fine
    except Exception:
        pass
    # final LP squeeze with remaining time
    remain = TIME_LIMIT - (time.time() - t0)
    if remain > 3.0:
        try:
            lp = lp_polish(best_x, min(remain - 0.5, FINAL_LP))
            vl = max_cos(lp)
            if vl <= best_val:
                best_val = vl
                best_x = lp
            else:
                # keep lp only if better; else keep best (lp may be equal)
                pass
        except Exception:
            pass
    best_x = best_x / np.linalg.norm(best_x, axis=1, keepdims=True)
    # safety: remove NaN
    best_x = np.nan_to_num(best_x)
    best_x = best_x / np.linalg.norm(best_x, axis=1, keepdims=True)
    return best_x
