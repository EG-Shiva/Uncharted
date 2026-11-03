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
    return np.max(G, axis=1)

def riesz_refine(cur, steps, lr, q=5.0, beta=0.9, frozen=None):
    n, d = cur.shape
    if frozen is None:
        move = np.ones(n, dtype=bool)
    else:
        move = ~frozen
    mom = np.zeros_like(cur)
    for s in range(steps):
        G = cur @ cur.T
        np.fill_diagonal(G, -5.0)
        # weight = (1-G)^-q, focus on close pairs but all pairs contribute
        D = 1.0 - G
        D[D < 1e-6] = 1e-6
        # cap to avoid overflow
        W = D ** (-q)
        np.fill_diagonal(W, 0.0)
        # normalize rows to keep step stable
        rs = np.sum(W, axis=1, keepdims=True) + 1e-12
        Grad = W @ cur
        proj = np.sum(Grad * cur, axis=1, keepdims=True)
        Gt = (Grad - proj * cur) / rs
        Gt[~move] = 0.0 if np.any(~move) else 0.0
        mom = beta * mom + (1.0 - beta) * Gt
        bc = 1.0 - beta ** (s + 1)
        cur = cur - lr * (1.0 - 0.5 * s / max(steps, 1)) * (mom / bc)
        # project only moved? project all for stability
        nr = np.linalg.norm(cur, axis=1, keepdims=True)
        nr[nr == 0] = 1.0
        cur /= nr
    return cur

def exp_refine(cur, alpha0, alpha1, steps, lr, beta=0.9, frozen=None):
    n, d = cur.shape
    if frozen is None:
        frozen = np.zeros(n, dtype=bool)
    move = ~frozen
    mom = np.zeros_like(cur)
    best = cur.copy()
    best_s = max_cos(cur)
    for s in range(steps):
        frac = s / max(steps - 1, 1)
        alpha = alpha0 * ((alpha1 / max(alpha0, 1e-12)) ** frac)
        G = cur @ cur.T
        np.fill_diagonal(G, -10.0)
        m = float(np.max(G))
        W = np.exp(alpha * (G - m))
        np.fill_diagonal(W, 0.0)
        Grad = W @ cur
        proj = np.sum(Grad * cur, axis=1, keepdims=True)
        Gt = Grad - proj * cur
        rs = W.sum(axis=1, keepdims=True) + 1e-12
        Gt = Gt / rs
        Gt[frozen] = 0.0
        mom = beta * mom + (1.0 - beta) * Gt
        bc = 1.0 - beta ** (s + 1)
        cur_lr = lr * (1.0 - 0.5 * frac)
        cur[move] = cur[move] - cur_lr * (mom[move] / bc)
        nr = np.linalg.norm(cur, axis=1, keepdims=True)
        nr[nr == 0] = 1.0
        cur /= nr
        if (s % 200 == 199) or (s == steps - 1):
            sc = max_cos(cur)
            if sc < best_s:
                best_s = sc
                best = cur.copy()
    return best, best_s

def greedy_minimax(cur, steps, lr0, delta=0.004):
    n, d = cur.shape
    best = cur.copy()
    best_s = max_cos(cur)
    c = cur.copy()
    for s in range(steps):
        G = c @ c.T
        np.fill_diagonal(G, -2.0)
        m = float(np.max(G))
        Mask = (G > m - delta).astype(float)
        np.fill_diagonal(Mask, 0.0)
        cnt = Mask.sum(axis=1, keepdims=True) + 1e-12
        Grad = Mask @ c
        proj = np.sum(Grad * c, axis=1, keepdims=True)
        Gt = (Grad - proj * c) / cnt
        cur_lr = lr0 * (1.0 - 0.7 * s / max(steps, 1))
        c = c - cur_lr * Gt
        c /= np.linalg.norm(c, axis=1, keepdims=True)
        if s == steps // 2:
            delta = delta * 0.3
        if (s % 100 == 99) or (s == steps - 1):
            sc = max_cos(c)
            if sc < best_s:
                best_s = sc
                best = c.copy()
    return best, best_s

def far_reinsert_inplace(x0, idx, rng, n_cand=2000):
    n, d = x0.shape
    for k in idx:
        cand = rng.standard_normal(size=(n_cand, d))
        cand /= np.linalg.norm(cand, axis=1, keepdims=True)
        # max dot against all other points
        dots = cand @ x0.T
        # ignore self column k (dot= <cand,x0[k]>), set to -2
        dots[:, k] = -2.0
        md = dots.max(axis=1)
        x0[k] = cand[np.argmin(md)]
    return x0

def lp_polish(x, seconds):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = max_cos(x)
    best_x = x.copy()
    radius = 2e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    ar = np.arange(d)
    while time.time() - start < seconds and radius > 1e-14:
        c = (x @ x.T)[iu]
        m = float(np.max(c))
        # wider window early to balance many near-contacts
        thr = max(20 * radius, 5e-5 if radius > 1e-4 else 1e-9)
        sel = c > m - thr
        # cap to top 2500 to keep LP tractable, keep largest
        if np.sum(sel) > 2500:
            idx = np.argsort(c)[-2500:]
            sel2 = np.zeros_like(sel, dtype=bool)
            sel2[idx] = True
            sel = sel2
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
            obj = np.zeros(nv); obj[-1] = 1.0
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
            radius = min(radius * 1.5, 0.02)
        else:
            radius *= 0.5
    return best_x

def solve():
    t0 = time.time()
    BUDGET = 172.0
    EXPLORE = 138.0
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
        xb = tmp
    xb /= np.linalg.norm(xb, axis=1, keepdims=True)
    n, d = 99, 13
    best_val = max_cos(xb)
    best_x = xb.copy()
    cur_x = xb.copy()
    cur_val = best_val
    # elite pool
    elites = [(best_val, best_x.copy())]

    it = 0
    while time.time() - t0 < EXPLORE and it < 500:
        it += 1
        remain = EXPLORE - (time.time() - t0)
        if remain < 3:
            break
        # temperature for Metropolis wander
        frac = (time.time() - t0) / max(EXPLORE, 1e-9)
        Temp = 0.0015 * (1.0 - 0.85 * frac) + 1e-5
        r = rng.random()
        # pick base: mostly current (wander), sometimes best/elite
        if r < 0.55:
            base = cur_x
        elif r < 0.80:
            base = best_x
        else:
            # random elite
            ei = int(rng.integers(0, len(elites)))
            base = elites[ei][1]
        mode = rng.random()
        x0 = base.copy()
        if mode < 0.28:
            sigma = float(rng.choice([0.05, 0.12, 0.25, 0.45, 0.7]))
            x0 = base + rng.standard_normal(size=(n, d)) * sigma
            x0 /= np.linalg.norm(x0, axis=1, keepdims=True)
        elif mode < 0.55:
            # kill worst + random block, far reinsert
            ppm = per_point_max(base)
            q = int(rng.integers(4, 14))
            # half worst, half random
            qw = q // 2
            idx_w = np.argsort(ppm)[-max(qw, 1):] if qw > 0 else np.array([], dtype=int)
            idx_r = rng.choice(n, size=q - len(idx_w), replace=False)
            idx = np.unique(np.concatenate([idx_w, idx_r]))
            x0 = base.copy()
            far_reinsert_inplace(x0, idx, rng, n_cand=1500)
            x0 += rng.standard_normal(size=(n, d)) * 0.015
            x0 /= np.linalg.norm(x0, axis=1, keepdims=True)
        elif mode < 0.70:
            # cap surgery: pick random direction, resample cap
            u = rng.standard_normal(size=(d,))
            u /= np.linalg.norm(u)
            dots = base @ u
            thr = float(np.quantile(dots, rng.uniform(0.75, 0.92)))
            idx = np.where(dots >= thr)[0]
            if len(idx) < 2:
                idx = rng.choice(n, size=int(rng.integers(5, 12)), replace=False)
            x0 = base.copy()
            # random points in cap around u with spread
            m = len(idx)
            cand = u[None, :] + rng.standard_normal(size=(m, d)) * float(rng.uniform(0.3, 0.7))
            cand /= np.linalg.norm(cand, axis=1, keepdims=True)
            x0[idx] = cand
            # small global jitter
            x0 += rng.standard_normal(size=(n, d)) * 0.01
            x0 /= np.linalg.norm(x0, axis=1, keepdims=True)
        elif mode < 0.85:
            # subset shake
            m = int(rng.integers(8, 25))
            idx = rng.choice(n, size=m, replace=False)
            sigma = float(rng.uniform(0.2, 0.6))
            x0 = base.copy()
            x0[idx] += rng.standard_normal(size=(m, d)) * sigma
            x0[idx] /= np.linalg.norm(x0[idx], axis=1, keepdims=True)
        else:
            # fresh random restart (5-15%)
            if rng.random() < 0.4:
                x0 = rng.standard_normal(size=(n, d))
                x0 /= np.linalg.norm(x0, axis=1, keepdims=True)
            else:
                # interpolate two elites / best + random
                a = rng.random()
                x0 = a * base + (1 - a) * rng.standard_normal(size=(n, d)) * 0.8
                x0 /= np.linalg.norm(x0, axis=1, keepdims=True)
        # refinement pipeline
        try:
            # stage 1: Riesz spreading to break clumps (short)
            if rng.random() < 0.6:
                q = float(rng.choice([3.0, 5.0, 7.0]))
                lr_r = float(rng.uniform(0.02, 0.07))
                x0 = riesz_refine(x0, steps=int(rng.integers(120, 260)), lr=lr_r, q=q)
            # stage 2: annealed exp
            a0 = float(rng.choice([15.0, 25.0, 40.0]))
            a1 = float(rng.choice([80.0, 130.0, 200.0]))
            if a1 <= a0:
                a1 = a0 * 3.0
            steps = int(rng.integers(450, 800))
            lr = float(rng.uniform(0.04, 0.13))
            cand, sc = exp_refine(x0, a0, a1, steps, lr)
            # stage 3: greedy minimax snap
            if rng.random() < 0.7:
                cand2, sc2 = greedy_minimax(cand, steps=int(rng.integers(120, 260)), lr0=float(rng.uniform(0.008, 0.025)), delta=float(rng.choice([0.003, 0.006, 0.010])))
                if sc2 < sc:
                    cand, sc = cand2, sc2
        except Exception:
            continue
        # update best
        if sc < best_val - 1e-12:
            best_val = sc
            best_x = cand.copy()
            elites.append((sc, cand.copy()))
            # keep top 5
            elites = sorted(elites, key=lambda e: e[0])[:5]
        # Metropolis update of current (wander, not greedy)
        if sc < cur_val or rng.random() < np.exp(-(sc - cur_val) / max(Temp, 1e-9)):
            cur_x = cand.copy()
            cur_val = sc
        else:
            # occasionally reset current to best to avoid drifting too far
            if rng.random() < 0.05:
                cur_x = best_x.copy()
                cur_val = best_val
    rem = BUDGET - (time.time() - t0)
    rem = max(6.0, min(rem, 40.0))
    try:
        polished = lp_polish(best_x, rem - 1.0)
        pv = max_cos(polished)
        if pv <= best_val + 1e-12:
            # keep polished if not worse (tighter contacts help score ties)
            if pv < best_val:
                best_val = pv
            best_x = polished
    except Exception:
        pass
    best_x /= np.linalg.norm(best_x, axis=1, keepdims=True)
    return best_x
