import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def max_cos(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def per_point_max(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return G.max(axis=1)

def lse_balanced(v, alpha0, alpha1, steps, lr, gamma, beta=0.9, frozen=None):
    n, d = v.shape
    cur = v.copy()
    best = cur.copy()
    best_s = max_cos(cur)
    mom = np.zeros_like(cur)
    if frozen is None:
        frozen = np.zeros(n, dtype=bool)
    move = ~frozen
    for s in range(steps):
        frac = s / max(steps - 1, 1)
        alpha = alpha0 * ((alpha1 / max(alpha0, 1e-12)) ** frac)
        G = cur @ cur.T
        np.fill_diagonal(G, -10.0)
        m = float(np.max(G))
        # row maxima for per-point weighting
        rm = np.max(G, axis=1)
        W = np.exp(alpha * (G - m))
        np.fill_diagonal(W, 0.0)
        Grad = W @ cur
        proj = np.sum(Grad * cur, axis=1, keepdims=True)
        Gt = Grad - proj * cur
        rs = W.sum(axis=1, keepdims=True) + 1e-12
        Gt = Gt / rs
        # focus on worst points -> equioscillation
        if gamma > 0:
            w = np.exp(gamma * (rm - m))
            w = w / (np.mean(w) + 1e-12)
            w = np.clip(w, 0.2, 5.0)
            Gt = Gt * w[:, None]
        Gt[frozen] = 0.0
        mom = beta * mom + (1.0 - beta) * Gt
        bc = 1.0 - beta ** (s + 1)
        step = mom / bc
        cur_lr = lr * (1.0 - 0.5 * frac)
        cur[move] = cur[move] - cur_lr * step[move]
        nr = np.linalg.norm(cur, axis=1, keepdims=True)
        nr[nr == 0] = 1.0
        cur /= nr
        if (s % 200 == 199) or (s == steps - 1):
            sc = max_cos(cur)
            if sc < best_s:
                best_s = sc
                best = cur.copy()
    return best, best_s

def topk_balance(v, steps, lr0, topk, w_top=1.0):
    n, d = v.shape
    cur = v.copy()
    best = cur.copy()
    best_s = max_cos(cur)
    for s in range(steps):
        G = cur @ cur.T
        np.fill_diagonal(G, -2.0)
        # top-k per row balancing mask (symmetrized)
        idx = np.argpartition(-G, kth=topk - 1, axis=1)[:, :topk]
        Comb = np.zeros_like(G)
        for i in range(n):
            Comb[i, idx[i]] = 1.0
        Comb = np.maximum(Comb, Comb.T)
        np.fill_diagonal(Comb, 0.0)
        cnt = Comb.sum(axis=1, keepdims=True) + 1e-12
        Grad = Comb @ cur
        proj = np.sum(Grad * cur, axis=1, keepdims=True)
        Gt = (Grad - proj * cur) / cnt
        cur_lr = lr0 * (1.0 - 0.6 * s / max(steps, 1))
        cur = cur - cur_lr * Gt
        cur /= np.linalg.norm(cur, axis=1, keepdims=True)
        if (s % 100 == 99) or (s == steps - 1):
            sc = max_cos(cur)
            if sc < best_s:
                best_s = sc
                best = cur.copy()
    return best, best_s

def far_reinsert(base, idx, rng, n_cand=2000):
    x0 = base.copy()
    n, d = base.shape
    for k in idx:
        cand = rng.standard_normal(size=(n_cand, d))
        cand /= np.linalg.norm(cand, axis=1, keepdims=True)
        others = np.delete(x0, k, axis=0)
        dots = cand @ others.T
        md = dots.max(axis=1)
        x0[k] = cand[np.argmin(md)]
    return x0

def lp_wide(x, seconds, kmin=900, wmin=5e-7):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = max_cos(x)
    best_x = x.copy()
    radius = 5e-4
    start = time.time()
    iu = np.triu_indices(n, 1)
    ar = np.arange(d)
    while time.time() - start < seconds and radius > 1e-14:
        c = (x @ x.T)[iu]
        m = float(np.max(c))
        thresh = m - max(30.0 * radius, wmin, 1e-9)
        sel = c > thresh
        if np.sum(sel) < kmin:
            order = np.argsort(-c)
            sel2 = np.zeros_like(sel, dtype=bool)
            sel2[order[:kmin]] = True
            sel = sel | sel2
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        if p == 0:
            break
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + ar, J[:, None] * d + ar, np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv)
            obj[-1] = 1.0
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
        vv = max_cos(y)
        if vv < best:
            best = vv
            best_x = y.copy()
            x = y.copy()
            radius = min(radius * 1.5, 0.02)
        else:
            radius *= 0.5
    return best_x

def solve():
    t0 = time.time()
    BUDGET = 172.0
    EXPLORE = 132.0
    rng = np.random.default_rng(12345)
    try:
        xb = np.load("best.npy").astype(float)
    except Exception:
        try:
            xb = np.load("record.npy").astype(float)
        except Exception:
            xb = rng.standard_normal(size=(99, 13))
    xb = np.asarray(xb, dtype=float)
    if xb.shape != (99, 13):
        tmp = rng.standard_normal(size=(99, 13))
        try:
            if xb.ndim == 2:
                m0 = min(xb.shape[0], 99)
                m1 = min(xb.shape[1], 13)
                tmp[:m0, :m1] = xb[:m0, :m1]
        except Exception:
            pass
        xb = tmp
    xb /= np.linalg.norm(xb, axis=1, keepdims=True)
    try:
        xr = np.load("record.npy").astype(float)
        xr /= np.linalg.norm(xr, axis=1, keepdims=True)
        if max_cos(xr) < max_cos(xb):
            xb = xr.copy()
    except Exception:
        pass
    n, d = 99, 13
    best_val = max_cos(xb)
    best_x = xb.copy()
    # quick initial equilibration of starting point
    try:
        c1, s1 = lse_balanced(best_x, 12.0, 30.0, 400, 0.08, 20.0)
        if s1 < best_val:
            best_val = s1
            best_x = c1.copy()
        c2, s2 = topk_balance(best_x, 200, 0.02, 6)
        if s2 < best_val:
            best_val = s2
            best_x = c2.copy()
    except Exception:
        pass
    it = 0
    sigmas = [0.02, 0.05, 0.10, 0.18, 0.30]
    while time.time() - t0 < EXPLORE and it < 300:
        it += 1
        remain = EXPLORE - (time.time() - t0)
        if remain < 3:
            break
        r = rng.random()
        base = best_x
        if r < 0.30:
            sigma = float(sigmas[int(rng.integers(0, len(sigmas)))])
            x0 = base + rng.standard_normal(size=(n, d)) * sigma
            x0 /= np.linalg.norm(x0, axis=1, keepdims=True)
        elif r < 0.55:
            ppm = per_point_max(base)
            q = int(rng.integers(2, 7))
            idx = np.argsort(ppm)[-q:]
            # far reinsert half the time, jitter otherwise
            if rng.random() < 0.6:
                x0 = far_reinsert(base, idx, rng, n_cand=1500)
                x0 += rng.standard_normal(size=(n, d)) * 0.008
                x0 /= np.linalg.norm(x0, axis=1, keepdims=True)
            else:
                x0 = base.copy()
                sg = float(rng.uniform(0.12, 0.35))
                x0[idx] += rng.standard_normal(size=(len(idx), d)) * sg
                x0[idx] /= np.linalg.norm(x0[idx], axis=1, keepdims=True)
        elif r < 0.75:
            G = base @ base.T
            np.fill_diagonal(G, -2.0)
            thr = float(np.max(G)) - 1e-3
            cnt = (G > thr).sum(axis=1)
            q = int(rng.integers(3, 9))
            idx = np.argsort(cnt)[-q:]
            # also add random companions to break clumps
            extra = rng.choice(n, size=int(rng.integers(0, 4)), replace=False)
            idx = np.unique(np.concatenate([idx, extra]))
            x0 = base.copy()
            sg = float(rng.uniform(0.10, 0.35))
            x0[idx] += rng.standard_normal(size=(len(idx), d)) * sg
            x0[idx] /= np.linalg.norm(x0[idx], axis=1, keepdims=True)
        else:
            m = int(rng.integers(4, 14))
            idx = rng.choice(n, size=m, replace=False)
            sigma = float(rng.uniform(0.08, 0.30))
            x0 = base.copy()
            x0[idx] += rng.standard_normal(size=(m, d)) * sigma
            x0[idx] /= np.linalg.norm(x0[idx], axis=1, keepdims=True)
        # annealed low->mid with balancing weight
        try:
            a0 = float(rng.choice([8.0, 12.0, 16.0]))
            a1 = float(rng.choice([30.0, 45.0, 60.0]))
            gam = float(rng.choice([15.0, 25.0, 40.0]))
            lr = float(rng.uniform(0.05, 0.12))
            cand, sc = lse_balanced(x0, a0, a1, 550, lr, gam)
            # top-k equilibration
            tk = int(rng.integers(5, 9))
            cand2, sc2 = topk_balance(cand, 220, 0.018, tk)
            if sc2 < sc:
                cand, sc = cand2, sc2
            # sharpen high-alpha short burst
            cand3, sc3 = lse_balanced(cand, 60.0, 120.0, 300, 0.06, 30.0)
            if sc3 < sc:
                cand, sc = cand3, sc3
        except Exception:
            continue
        if sc < best_val - 1e-12:
            best_val = sc
            best_x = cand.copy()
            # opportunistic wide LP snap to lock balanced gain
            if it % 3 == 0 and (time.time() - t0) < EXPLORE - 5:
                try:
                    snapped = lp_wide(best_x, 1.5, kmin=900, wmin=3e-7)
                    sv = max_cos(snapped)
                    if sv < best_val:
                        best_val = sv
                        best_x = snapped.copy()
                except Exception:
                    pass
    rem = BUDGET - (time.time() - t0)
    rem = max(6.0, min(rem, 40.0))
    try:
        polished = lp_wide(best_x, rem - 1.0, kmin=1000, wmin=2e-7)
        pv = max_cos(polished)
        if pv <= best_val + 1e-12:
            # keep tighter if essentially equal (more contacts)
            if pv < best_val:
                best_val = pv
            best_x = polished.copy()
    except Exception:
        pass
    best_x /= np.linalg.norm(best_x, axis=1, keepdims=True)
    return best_x
