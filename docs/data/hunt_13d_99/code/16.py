import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())


def soft_refine(v, schedule, rng=None):
    cur = v.copy()
    best = cur.copy()
    best_s = max_cos(cur)
    mom = np.zeros_like(cur)
    beta = 0.9
    step_cnt = 0
    n, d = cur.shape
    for (alpha, steps, lr0, lr1) in schedule:
        mom[:] = 0.0
        for s in range(steps):
            step_cnt += 1
            G = cur @ cur.T
            np.fill_diagonal(G, -10.0)
            m = G.max()
            # exp weights, shift for stability
            W = np.exp(alpha * (G - m))
            np.fill_diagonal(W, 0.0)
            Grad = W @ cur
            proj = np.sum(Grad * cur, axis=1, keepdims=True)
            Gt = Grad - proj * cur
            rs = W.sum(axis=1, keepdims=True) + 1e-12
            Gt = Gt / rs
            mom = beta * mom + (1.0 - beta) * Gt
            bc = 1.0 - beta ** (s + 1)
            direction = mom / bc
            # linear lr decay within stage
            frac = s / max(1, steps - 1)
            lr = lr0 + (lr1 - lr0) * frac
            # normalize direction per-point to keep step stable
            dn = np.linalg.norm(direction, axis=1, keepdims=True) + 1e-12
            # cap huge steps
            direction = direction / np.maximum(1.0, dn * 5.0)
            cur = cur - lr * direction
            cur /= np.linalg.norm(cur, axis=1, keepdims=True)
            if (s % 200 == 199) or (s == steps - 1):
                sc = max_cos(cur)
                if sc < best_s:
                    best_s = sc
                    best = cur.copy()
    return best, best_s


def hard_refine(v, steps, lr):
    cur = v.copy()
    best = cur.copy()
    best_s = max_cos(cur)
    n, d = cur.shape
    for s in range(steps):
        G = cur @ cur.T
        np.fill_diagonal(G, -2.0)
        m = G.max()
        # only near-max pairs push
        thr = m - 0.002
        mask = G >= thr
        # adjacency weights linear in excess
        W = np.maximum(0.0, G - thr)
        np.fill_diagonal(W, 0.0)
        rs = W.sum(axis=1, keepdims=True) + 1e-12
        Grad = W @ cur
        proj = np.sum(Grad * cur, axis=1, keepdims=True)
        Gt = (Grad - proj * cur) / rs
        cur = cur - lr * Gt
        cur /= np.linalg.norm(cur, axis=1, keepdims=True)
        if (s % 100 == 99):
            sc = max_cos(cur)
            if sc < best_s:
                best_s = sc
                best = cur.copy()
            # adapt threshold decay
    sc = max_cos(cur)
    if sc < best_s:
        best_s = sc
        best = cur.copy()
    return best, best_s


def far_reinsert(base, idx, rng, n_cand=2000):
    n, d = base.shape
    x0 = base.copy()
    for k in idx:
        cand = rng.standard_normal(size=(n_cand, d))
        cand /= np.linalg.norm(cand, axis=1, keepdims=True)
        others = np.delete(x0, k, axis=0)
        dots = cand @ others.T
        md = dots.max(axis=1)
        j = int(np.argmin(md))
        x0[k] = cand[j]
    return x0


def worst_indices(x, k):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    wm = G.max(axis=1)
    return np.argsort(-wm)[:k]


def contact_counts(x, thr_offset=1e-4):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    m = G.max()
    cnt = (G >= m - thr_offset).sum(axis=1)
    return cnt, m


def lp_polish(x, seconds):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = max_cos(x)
    best_x = x.copy()
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
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
            best = v
            best_x = y.copy()
            x = y
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return best_x


def solve():
    t0 = time.time()
    BUDGET = 172.0
    EXPLORE = 138.0
    rng = np.random.default_rng(20261103)
    try:
        xb = np.load("best.npy").astype(float)
    except Exception:
        try:
            xb = np.load("record.npy").astype(float)
        except Exception:
            xb = rng.standard_normal(size=(99, 13))
    xb = xb / np.linalg.norm(xb, axis=1, keepdims=True)
    n, d = xb.shape
    if (n, d) != (99, 13):
        tmp = rng.standard_normal(size=(99, 13))
        tmp /= np.linalg.norm(tmp, axis=1, keepdims=True)
        # copy what fits
        xb = tmp
    best_val = max_cos(xb)
    best_x = xb.copy()

    sigmas = [0.015, 0.03, 0.06, 0.10, 0.16, 0.25, 0.38]
    it = 0
    # precompute contact info of best for targeted surgery
    while time.time() - t0 < EXPLORE and it < 400:
        it += 1
        remain = EXPLORE - (time.time() - t0)
        if remain < 2:
            break
        # choose base (mostly best, occasional perturbed-best reuse)
        base = best_x
        mode = rng.random()
        x0 = base.copy()
        if mode < 0.50:
            # worst-point deletion + void reinsertion
            k = int(rng.integers(1, 5))  # 1..4
            # mix worst-max and high-contact selection
            if rng.random() < 0.6:
                idx = worst_indices(base, k)
            else:
                cnt, _ = contact_counts(base, thr_offset=5e-4)
                idx = np.argsort(-cnt)[:k]
                # add randomness: sometimes replace random ones too
                if rng.random() < 0.4:
                    idx = np.concatenate([idx[:max(1, k-1)], rng.integers(0, n, size=1)])
            x0 = far_reinsert(base, idx, rng, n_cand=2500)
            # small shake to the rest to break symmetry
            sigma = float(sigmas[rng.integers(0, len(sigmas))]) * 0.4
            x0 = x0 + rng.standard_normal(size=(n, d)) * sigma
            x0 /= np.linalg.norm(x0, axis=1, keepdims=True)
        elif mode < 0.80:
            sigma = float(sigmas[rng.integers(0, len(sigmas))])
            x0 = base + rng.standard_normal(size=(n, d)) * sigma
            x0 /= np.linalg.norm(x0, axis=1, keepdims=True)
        else:
            # subset strong shake: pick ~5-20 points
            m = int(rng.integers(5, 21))
            sel = rng.choice(n, size=m, replace=False)
            sigma = 0.25 + 0.35 * rng.random()
            x0[sel] = x0[sel] + rng.standard_normal(size=(m, d)) * sigma
            x0 /= np.linalg.norm(x0, axis=1, keepdims=True)

        # schedule depends on iteration / remaining time
        a_choice = rng.random()
        if a_choice < 0.35:
            sched = [(35.0, 350, 0.10, 0.05), (90.0, 350, 0.06, 0.03)]
        elif a_choice < 0.70:
            sched = [(50.0, 400, 0.12, 0.06), (130.0, 400, 0.07, 0.03)]
        else:
            sched = [(25.0, 300, 0.14, 0.07), (70.0, 300, 0.08, 0.04), (150.0, 250, 0.05, 0.025)]
        # shorten if low on time
        if remain < 20:
            sched = [(50.0, 250, 0.10, 0.05), (120.0, 250, 0.06, 0.03)]

        x1, s1 = soft_refine(x0, sched)
        # hard push on near-max pairs
        x2, s2 = hard_refine(x1, steps=150, lr=0.05)
        if s2 < s1:
            x1, s1 = x2, s2
        if s1 < best_val - 1e-12:
            best_val = s1
            best_x = x1.copy()
        # also accept near-equal with different topology occasionally to diversify?
        # no, keep strictly best as base; exploration comes from perturbations

    # final polish with remaining budget
    remaining = BUDGET - (time.time() - t0)
    remaining = max(5.0, min(34.0, remaining))
    best_x = lp_polish(best_x, remaining)
    best_x = best_x / np.linalg.norm(best_x, axis=1, keepdims=True)
    return best_x
