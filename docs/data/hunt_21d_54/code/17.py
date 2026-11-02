import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _proj(x):
    nrm = np.linalg.norm(x, axis=1, keepdims=True)
    nrm[nrm == 0] = 1.0
    return x / nrm

def _quench(y, steps, lr0, m0, m1, mu=0.85, seed_vel_zero=True):
    n, d = y.shape
    vel = np.zeros((n, d))
    for s in range(steps):
        frac = s / steps
        margin = m0 * (1 - frac) + m1 * frac
        T = margin / 3.0 + 1e-12
        G = y @ y.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        # soft weights focused near max
        # clip exponent to avoid underflow overhead
        E = (G - m) / T
        # for very negative, exp->0; clip at -20
        np.clip(E, -20, 0, out=E)
        W = np.exp(E)
        np.fill_diagonal(W, 0.0)
        rs = W.sum(axis=1, keepdims=True) + 1e-12
        grad = W @ y
        dot = np.einsum('ij,ij->i', grad, y)
        grad -= dot[:, None] * y
        grad /= rs
        vel = mu * vel + grad
        lr = lr0 * (1.0 - 0.7 * frac)
        y = y - lr * vel
        # re-normalize
        nrm = np.linalg.norm(y, axis=1, keepdims=True)
        nrm[nrm == 0] = 1.0
        y /= nrm
    return y

def _lp_polish(x, seconds):
    x = _proj(x)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        G = x @ x.T
        c = G[iu]
        m = float(c.max())
        sel = c > m - max(20 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        if p > 2500:
            idx = np.argpartition(-c[sel], 2500)[:2500]
            I, J = I[idx], J[idx]
            csv = c[sel][idx]
            p = 2500
        else:
            csv = c[sel]
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))),
                          shape=(n, nv)).tocsr()
        obj = np.zeros(nv)
        obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-csv, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                          method="highs",
                          options={"time_limit": 2.0})
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y = _proj(y)
        v = _max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.02)
        else:
            radius *= 0.5
    return x

def solve():
    T0 = time.time()
    LIMIT = 168.0
    rng = np.random.default_rng(42)
    try:
        b0 = np.load("best.npy")
    except Exception:
        b0 = np.load("record.npy")
    b0 = _proj(np.nan_to_num(b0).astype(float))
    # in case shape mismatch
    if b0.shape != (54, 21):
        # fallback random
        b0 = _proj(rng.normal(size=(54, 21)))
    try:
        r0 = _proj(np.load("record.npy").astype(float))
    except Exception:
        r0 = b0
    best = b0.copy()
    bv = _max_cos(best)
    rv = _max_cos(r0)
    if rv < bv:
        best, bv = r0.copy(), rv

    n, d = 54, 21

    def time_left():
        return LIMIT - (time.time() - T0)

    # quick medium quench of starting point to ensure local convergence
    if time_left() > 40:
        cand = _quench(best.copy(), steps=2500, lr0=0.35, m0=0.02, m1=0.0015, mu=0.85)
        v = _max_cos(cand)
        if v < bv:
            best, bv = cand, v

    it = 0
    # basin hopping
    strengths = [0.008, 0.015, 0.03, 0.06, 0.12, 0.20, 0.01, 0.04, 0.09]
    while time_left() > 28:
        it += 1
        # choose mode: every 4th do reinsert, else jitter
        do_reinsert = (it % 4 == 0)
        base = best
        if do_reinsert:
            G = base @ base.T
            np.fill_diagonal(G, -2.0)
            rowmax = G.max(axis=1)
            i_worst = int(np.argmax(rowmax))
            # second worst sometimes
            if it % 8 == 0:
                # pick random among top-5 worst for diversity
                order = np.argsort(-rowmax)
                i_worst = int(order[int(rng.integers(0, 5))])
            y = base.copy()
            # propose K random candidates, pick best insertion
            K = 1500
            C = rng.normal(size=(K, d))
            C /= np.linalg.norm(C, axis=1, keepdims=True) + 1e-12
            # dots with existing (excluding i_worst)
            D = C @ y.T  # K x n
            D[:, i_worst] = -10.0
            scores = D.max(axis=1)
            kbest = int(np.argmin(scores))
            y[i_worst] = C[kbest]
            # small tangent jitter to others to break symmetry
            sig = 0.005
            Nz = rng.normal(size=(n, d)) * sig
            Nz -= (Nz * y).sum(axis=1, keepdims=True) * y
            y = _proj(y + Nz)
            # quench: medium
            steps = 1200
            lr0 = 0.30
            m0_, m1_ = 0.015, 0.0012
            y = _quench(y, steps=steps, lr0=lr0, m0=m0_, m1=m1_, mu=0.85)
        else:
            sig = strengths[(it - 1) % len(strengths)]
            # bias kicks to high-contact points occasionally
            y = base.copy()
            Nz = rng.normal(size=(n, d))
            Nz -= (Nz * y).sum(axis=1, keepdims=True) * y
            nn = np.linalg.norm(Nz, axis=1, keepdims=True) + 1e-12
            Nz /= nn
            # random per-point amplitude
            amp = sig * (0.5 + rng.random(size=(n, 1)))
            # rotate approx: y*cos+Nz*sin
            ang = amp
            y = np.cos(ang) * y + np.sin(ang) * Nz
            y = _proj(y)
            steps = 1300
            # larger shake -> larger initial margin/lr
            if sig > 0.08:
                lr0, m0_, m1_ = 0.35, 0.025, 0.0015
            elif sig > 0.03:
                lr0, m0_, m1_ = 0.32, 0.02, 0.0012
            else:
                lr0, m0_, m1_ = 0.28, 0.012, 0.0008
            y = _quench(y, steps=steps, lr0=lr0, m0=m0_, m1=m1_, mu=0.85)
        v = _max_cos(y)
        if v < bv - 1e-12:
            best, bv = y, v
        # opportunistic: if close but different topology, try extra fine quench
        elif v < bv + 0.002 and time_left() > 60 and (it % 10 == 0):
            # fine squeeze attempt from this candidate (short)
            y2 = _quench(y.copy(), steps=800, lr0=0.12, m0=0.005, m1=0.0005, mu=0.9)
            v2 = _max_cos(y2)
            if v2 < bv - 1e-12:
                best, bv = y2, v2

    # final intensive squeeze on best
    if time_left() > 14:
        # long fine quench
        budget = time_left() - 12.0
        # estimate steps that fit: ~0.06ms per step? use fixed
        # do 6000 steps (should be <8s)
        try:
            best = _quench(best.copy(), steps=6000, lr0=0.15, m0=0.008, m1=0.0004, mu=0.9)
            bv = _max_cos(best)
        except Exception:
            pass
    # final LP squeeze if time remains
    if time_left() > 4:
        try:
            lp_budget = min(10.0, time_left() - 1.0)
            if lp_budget > 2:
                cand = _lp_polish(best.copy(), lp_budget)
                v = _max_cos(cand)
                if v < bv:
                    best, bv = cand, v
        except Exception:
            pass

    best = _proj(best)
    return best
