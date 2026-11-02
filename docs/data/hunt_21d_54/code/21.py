import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _norm(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return x / n

def _max_cos(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def _gd_repel(x, steps, T0, T1, lr0, lr1):
    x = _norm(x)
    n, d = x.shape
    for k in range(steps):
        f = k / max(1, steps - 1)
        T = T0 * (1 - f) + T1 * f
        lr = lr0 * (1 - f) + lr1 * f
        G = x @ x.T
        np.fill_diagonal(G, -np.inf)
        m = float(np.max(G))
        # softmax weights, avoid overflow
        W = np.exp((G - m) / T)
        np.fill_diagonal(W, 0.0)
        grad = W @ x
        dot = np.sum(grad * x, axis=1, keepdims=True)
        gt = grad - dot * x
        nr = np.linalg.norm(gt, axis=1, keepdims=True) + 1e-12
        gt = gt / nr
        x = x - lr * gt
        # fast renormalize
        x = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-30)
    return _norm(x)

def _lp_polish(x, deadline, r0=2e-3):
    x = _norm(np.asarray(x, dtype=float))
    n, d = x.shape
    best = _max_cos(x)
    radius = r0
    iu = np.triu_indices(n, 1)
    it = 0
    while time.time() < deadline and radius > 1e-14 and it < 60:
        it += 1
        G = x @ x.T
        c = G[iu]
        m = float(c.max())
        thresh = m - max(20.0 * radius, 1e-9)
        sel = c > thresh
        cnt = int(np.count_nonzero(sel))
        if cnt == 0:
            break
        if cnt > 700:
            idx = np.argsort(c)[-700:]
            mask = np.zeros_like(sel, dtype=bool)
            mask[idx] = True
            sel = mask
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        Ccol = np.empty((p, 2 * d + 1), dtype=int)
        Ccol[:, :d] = I[:, None] * d + np.arange(d)
        Ccol[:, d:2 * d] = J[:, None] * d + np.arange(d)
        Ccol[:, 2 * d] = n * d
        cols = Ccol.ravel()
        Arow = np.empty((p, 2 * d + 1))
        Arow[:, :d] = x[J]
        Arow[:, d:2 * d] = x[I]
        Arow[:, 2 * d] = -1.0
        vals = Arow.ravel()
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
        y = _norm(y)
        v = _max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.02)
        else:
            radius *= 0.5
    return x

def solve():
    import time as _t
    t0 = _t.time()
    HARD = t0 + 168.0
    rng = np.random.RandomState(7)
    try:
        xb = np.load("best.npy")
    except Exception:
        try:
            xb = np.load("record.npy")
        except Exception:
            xb = rng.randn(54, 21)
    xb = _norm(np.asarray(xb, dtype=float))
    if xb.shape != (54, 21):
        xb = _norm(rng.randn(54, 21))
    best = xb.copy()
    best_val = _max_cos(best)
    # elite set (max 3)
    elites = [(best_val, best.copy())]
    # contact helper
    def worst_idx(x):
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        # count near contacts
        C = (G > m - 1e-7).sum(axis=1)
        return int(np.argmax(C)), m
    sigmas = [0.05, 0.025, 0.012, 0.08, 0.035, 0.10, 0.015, 0.06]
    trial = 0
    # Phase A: cheap GD-only basin hopping until HARD-30
    while _t.time() < HARD - 30:
        trial += 1
        # pick base from elites
        bidx = rng.randint(len(elites))
        base = elites[bidx][1]
        sigma = sigmas[(trial - 1) % len(sigmas)]
        mode = trial % 4
        cand = base.copy()
        if mode == 0:
            cand = cand + sigma * rng.randn(*cand.shape)
            cand = _norm(cand)
            T0 = float(rng.uniform(0.008, 0.025)); T1 = T0 * 0.15
            lr0 = float(rng.uniform(0.02, 0.05)); lr1 = lr0 * 0.12
            cand = _gd_repel(cand, steps=550, T0=T0, T1=T1, lr0=lr0, lr1=lr1)
        elif mode == 1:
            wi, _ = worst_idx(base)
            cand[wi] = cand[wi] + (sigma * 2.5) * rng.randn(21)
            cand = _norm(cand)
            cand = cand + (sigma * 0.3) * rng.randn(*cand.shape)
            cand = _norm(cand)
            cand = _gd_repel(cand, steps=550, T0=0.015, T1=0.002, lr0=0.03, lr1=0.004)
        elif mode == 2:
            # kill worst into largest hole
            wi, _ = worst_idx(base)
            keep = np.delete(base, wi, axis=0)
            S = _norm(rng.randn(3000, 21))
            dots = S @ keep.T
            md = dots.max(axis=1)
            hole = S[int(np.argmin(md))]
            cand[wi] = hole + 0.008 * rng.randn(21)
            cand = _norm(cand)
            # slight shake of neighbours
            cand = cand + (sigma * 0.15) * rng.randn(*cand.shape)
            cand = _norm(cand)
            cand = _gd_repel(cand, steps=500, T0=0.012, T1=0.002, lr0=0.025, lr1=0.003)
        else:
            # double kill: two worst points resampled
            G = base @ base.T
            np.fill_diagonal(G, -2.0)
            m = float(G.max())
            C = (G > m - 1e-7).sum(axis=1)
            order = np.argsort(-C)
            w1, w2 = int(order[0]), int(order[1])
            S = _norm(rng.randn(3000, 21))
            for w in (w1, w2):
                keep = np.delete(cand, w, axis=0)
                dots = S @ keep.T
                md = dots.max(axis=1)
                # pick diverse holes: best + noise rank
                hole = S[int(np.argmin(md))]
                cand[w] = hole + 0.005 * rng.randn(21)
            cand = _norm(cand)
            cand = _gd_repel(cand, steps=600, T0=0.018, T1=0.0025, lr0=0.035, lr1=0.004)
        v = _max_cos(cand)
        if v < best_val - 1e-12:
            best_val = v
            best = cand.copy()
            elites.append((v, cand.copy()))
            elites = sorted(elites, key=lambda e: e[0])[:3]
        elif v < elites[-1][0] and len(elites) < 3:
            elites.append((v, cand.copy()))
            elites = sorted(elites, key=lambda e: e[0])[:3]
        # keep elites diverse: if trial many without improvement, inject random
        if trial % 60 == 0 and _t.time() < HARD - 40:
            # random restart quench from scratch (structured random)
            r = _norm(rng.randn(54, 21))
            r = _gd_repel(r, steps=500, T0=0.02, T1=0.003, lr0=0.04, lr1=0.005)
            v2 = _max_cos(r)
            if v2 < elites[-1][0]:
                elites[-1] = (v2, r)
                elites = sorted(elites, key=lambda e: e[0])[:3]
            if v2 < best_val:
                best_val, best = v2, r.copy()
        if _t.time() >= HARD - 30:
            break
    # Phase B: final squeeze of the single best with LP
    if _t.time() < HARD - 2:
        try:
            best = _lp_polish(best, deadline=HARD - 1.0, r0=2e-3)
        except Exception:
            pass
    best = _norm(best)
    return best
