import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(np.max(g))

def biased_anneal(x, BI, BJ, boost, steps, a0, a1, lr0, lr1):
    n, d = x.shape
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    B = np.ones((n, n), dtype=float)
    if len(BI) > 0:
        B[BI, BJ] = boost
        B[BJ, BI] = boost
    for s in range(steps):
        frac = s / max(steps - 1, 1)
        alpha = a0 * (a1 / a0) ** frac
        lr = lr0 * (lr1 / lr0) ** frac
        G = x @ x.T
        np.fill_diagonal(G, -np.inf)
        m = float(np.max(G))
        W = np.exp(alpha * (G - m))
        np.fill_diagonal(W, 0.0)
        if len(BI) > 0:
            W *= B
        rs = W.sum(axis=1, keepdims=True)
        rs = np.maximum(rs, 1e-300)
        A = (W @ x) / rs
        dot = np.sum(A * x, axis=1, keepdims=True)
        T = A - dot * x
        # clip extreme steps
        x = x - lr * T
        x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x

def lp_polish_excl(x, seconds, excl_I=None, excl_J=None):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = max_cos(x)
    radius, start = 1e-3, time.time()
    iu = np.triu_indices(n, 1)
    excl = None
    if excl_I is not None and len(excl_I) > 0:
        excl = set((int(min(a, b)), int(max(a, b))) for a, b in zip(excl_I, excl_J))
    while time.time() - start < seconds and radius > 1e-14:
        Cfull = (x @ x.T)[iu]
        m = float(np.max(Cfull))
        sel = Cfull > m - max(20 * radius, 1e-9)
        if np.sum(sel) == 0:
            break
        I = iu[0][sel].copy()
        J = iu[1][sel].copy()
        Cs = Cfull[sel]
        if excl is not None and len(excl) > 0:
            keep = np.ones(len(I), dtype=bool)
            # exclude broken edges unless they rose above max-1e-6
            for k in range(len(I)):
                a = int(min(I[k], J[k])); b = int(max(I[k], J[k]))
                if (a, b) in excl and Cs[k] <= m - 1e-6:
                    keep[k] = False
            I = I[keep]; J = J[keep]; Cs = Cs[keep]
            if len(I) == 0:
                radius *= 0.5
                continue
        p, nv = len(I), n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv); obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-Cs, A_eq=A_eq, b_eq=np.zeros(n),
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
    DEADLINE = 172.0
    rng = np.random.default_rng(7)
    best = None
    best_val = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            y = np.load(fn)
            y = y / np.linalg.norm(y, axis=1, keepdims=True)
            if y.shape == (153, 4) and np.all(np.isfinite(y)):
                G = y @ y.T
                np.fill_diagonal(G, -2.0)
                v = float(np.max(G))
                if v < best_val:
                    best_val = v
                    best = y.copy()
        except Exception:
            pass
    if best is None:
        z = rng.standard_normal(size=(153, 4))
        z /= np.linalg.norm(z, axis=1, keepdims=True)
        best = z
        best_val = max_cos(best)
    n, d = best.shape
    cur = best.copy()
    cur_val = best_val
    tabu = np.zeros((n, n), dtype=float)
    last_broken = []
    deltas = [3e-4, 1e-3, 3e-3]
    iu = np.triu_indices(n, 1)
    it = 0
    MAX_IT = 200
    while it < MAX_IT and (time.time() - t0) < 148.0:
        it += 1
        # incumbent contact graph (slack < 1e-9)
        Gc = cur @ cur.T
        np.fill_diagonal(Gc, -2.0)
        m0 = float(np.max(Gc))
        Cfull = Gc[iu]
        cmask = Cfull > m0 - 1e-9
        CI = iu[0][cmask]; CJ = iu[1][cmask]
        if len(CI) < 15:
            # fallback: use 1e-7 threshold
            cmask = Cfull > m0 - 1e-7
            CI = iu[0][cmask]; CJ = iu[1][cmask]
        if len(CI) == 0:
            CI = iu[0][:30]; CJ = iu[1][:30]
        degs = np.zeros(n, dtype=float)
        np.add.at(degs, CI, 1)
        np.add.at(degs, CJ, 1)
        w = (degs[CI] + degs[CJ] + 2.0) / (1.0 + 0.5 * tabu[CI, CJ])
        ssum = float(np.sum(w))
        if not np.isfinite(ssum) or ssum <= 0:
            w = np.ones_like(w) / len(w)
        else:
            w = w / ssum
        k = int(rng.integers(15, 41))
        k = min(k, len(CI))
        # sample without replacement
        try:
            chosen = rng.choice(len(CI), size=k, replace=False, p=w)
        except Exception:
            chosen = rng.choice(len(CI), size=k, replace=False)
        BI = CI[chosen].copy(); BJ = CJ[chosen].copy()
        # avoid re-forming same graph: limit overlap with recent sets
        cur_set = set((int(min(a, b)), int(max(a, b))) for a, b in zip(BI, BJ))
        for _ in range(6):
            bad = False
            for prev in last_broken[-3:]:
                inter = len(cur_set & prev)
                if inter > int(0.5 * k):
                    bad = True
                    break
            if not bad:
                break
            try:
                chosen = rng.choice(len(CI), size=k, replace=False, p=w)
            except Exception:
                chosen = rng.choice(len(CI), size=k, replace=False)
            BI = CI[chosen].copy(); BJ = CJ[chosen].copy()
            cur_set = set((int(min(a, b)), int(max(a, b))) for a, b in zip(BI, BJ))
        delta = deltas[(it - 1) % len(deltas)]
        boost = float(rng.uniform(8.0, 30.0))
        # stronger boost for larger delta
        if delta >= 3e-3:
            boost *= 1.4
        y = cur.copy()
        # nudge rattlers so they can gain contacts
        ratt = np.where(degs == 0)[0]
        for r in ratt:
            y[r] = y[r] + 0.06 * rng.standard_normal(d)
        # small global jitter to escape LP vertex
        y = y + 0.002 * rng.standard_normal(size=(n, d))
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        steps = 500 if (it % 2 == 0) else 650
        a0 = 40.0 if (it % 3 != 0) else 25.0
        y2 = biased_anneal(y, BI, BJ, boost, steps, a0, 800.0, 0.03, 0.003)
        v2 = max_cos(y2)
        if v2 > best_val + 0.008:
            continue
        remain = DEADLINE - (time.time() - t0)
        if remain < 5:
            break
        y3 = lp_polish_excl(y2, seconds=1.35, excl_I=BI, excl_J=BJ)
        v3 = max_cos(y3)
        for a, b in zip(BI, BJ):
            tabu[int(a), int(b)] += 1.0
            tabu[int(b), int(a)] += 1.0
        last_broken.append(cur_set)
        if len(last_broken) > 12:
            last_broken.pop(0)
        if v3 < best_val - 1e-12 and np.all(np.isfinite(y3)):
            best = y3.copy(); best_val = v3
            cur = y3.copy(); cur_val = v3
        elif v3 <= best_val + 0.002 and np.all(np.isfinite(y3)):
            cur = y3.copy(); cur_val = v3
        # if cur drifted too far for long, pull back to best occasionally
        if it % 12 == 0 and cur_val > best_val + 0.0015:
            cur = best.copy(); cur_val = best_val
    remain = DEADLINE - (time.time() - t0)
    if remain > 4:
        try:
            best = lp_polish_excl(best, seconds=min(remain - 1.0, 16.0), excl_I=None, excl_J=None)
        except Exception:
            pass
    best = best / np.linalg.norm(best, axis=1, keepdims=True)
    best = np.where(np.isfinite(best), best, 0.0)
    norms = np.linalg.norm(best, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    best = best / norms
    return best
