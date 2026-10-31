import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _norm(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return X / n

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _qmul(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    aw, ax, ay, az = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bw, bx, by, bz = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    w = aw * bw - ax * bx - ay * by - az * bz
    x = aw * bx + ax * bw + ay * bz - az * by
    y = aw * by - ax * bz + ay * bw + az * bx
    z = aw * bz + ax * by - ay * bx + az * bw
    return np.stack([w, x, y, z], axis=-1)

def _hurwitz():
    H = []
    for i in range(4):
        for s in (-1.0, 1.0):
            v = [0.0, 0.0, 0.0, 0.0]
            v[i] = s
            H.append(v)
    for a in (-0.5, 0.5):
        for b in (-0.5, 0.5):
            for c in (-0.5, 0.5):
                for d in (-0.5, 0.5):
                    H.append([a, b, c, d])
    return np.array(H, dtype=np.float64)

def _repulse(X0, steps, a0, a1, s0, s1, mu=0.85):
    X = X0.copy()
    n = X.shape[0]
    vel = np.zeros_like(X)
    G0 = X @ X.T
    np.fill_diagonal(G0, -2.0)
    bv = float(G0.max())
    bX = X.copy()
    for it in range(steps):
        frac = it / max(steps - 1, 1)
        alpha = a0 * (a1 / a0) ** frac if a0 != a1 else a0
        smax = s0 * (s1 / s0) ** frac if s0 != s1 else s0
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        if m < bv:
            bv = m
            bX = X.copy()
        W = np.exp(alpha * (G - m))
        np.fill_diagonal(W, 0.0)
        grad = W @ X
        dot = np.sum(grad * X, axis=1, keepdims=True)
        gt = grad - dot * X
        vel = mu * vel + gt
        nrm = float(np.linalg.norm(vel, axis=1).max())
        if nrm > 1e-14:
            X = X - (smax / nrm) * vel
            X = _norm(X)
    Gf = X @ X.T
    np.fill_diagonal(Gf, -2.0)
    fv = float(Gf.max())
    if fv < bv:
        return X, fv
    return bX, bv

def _lp_polish(x, deadline):
    x = _norm(x.copy())
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    iu0 = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        G = x @ x.T
        c_all = G[iu0]
        m = float(c_all.max())
        sel = c_all > m - max(20.0 * radius, 1e-9)
        cnt = int(np.count_nonzero(sel))
        if cnt == 0:
            radius *= 0.5
            continue
        if cnt > 6000:
            idx = np.where(sel)[0]
            order = np.argpartition(-c_all[idx], 6000 - 1)[:6000]
            keep = np.zeros_like(sel, dtype=bool)
            keep[idx[order]] = True
            sel = keep
        c = c_all[sel]
        I = iu0[0][sel]
        J = iu0[1][sel]
        p = len(I)
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        except Exception:
            radius *= 0.5
            continue
        obj = np.zeros(nv)
        obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c, A_eq=A_eq, b_eq=np.zeros(n),
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
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def solve():
    t0 = time.time()
    HARD_END = t0 + 175.0
    SYM_END = t0 + 48.0
    HOP_END = t0 + 143.0
    n, d, K = 272, 4, 11
    rng = np.random.RandomState(20261031)
    H = _hurwitz()
    cross0 = np.vstack([np.eye(4), -np.eye(4)]).astype(np.float64)

    def rand_unit(m):
        a = rng.randn(m, 4)
        a /= np.linalg.norm(a, axis=1, keepdims=True)
        return a

    best_overall = None
    best_val = 2.0
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=np.float64)
            if a.shape == (n, d):
                a = _norm(a)
                if np.all(np.isfinite(a)):
                    v = _max_cos(a)
                    if v < best_val:
                        best_val = v
                        best_overall = a.copy()
        except Exception:
            pass
    if best_overall is None:
        best_overall = rand_unit(n)
        best_val = _max_cos(best_overall)

    def build_sym():
        S = rand_unit(K)
        P = rand_unit(K)
        R = rand_unit(K)
        # random choice: identity rotations for some orbits to diversify
        if rng.rand() < 0.4:
            P[:] = np.array([1.0, 0.0, 0.0, 0.0])
            R[:] = np.array([1.0, 0.0, 0.0, 0.0])
        Xorb = np.empty((K * 24, 4))
        for k in range(K):
            Y = _qmul(H, S[k])
            T = _qmul(P[k], Y)
            Z = _qmul(T, R[k])
            Xorb[k * 24:(k + 1) * 24] = Z
        # cross optionally rotated by a general pair
        if rng.rand() < 0.5:
            p0 = rand_unit(1)[0]
            r0 = rand_unit(1)[0]
            C = _qmul(_qmul(p0, cross0), r0)
        else:
            C = cross0
        X = np.vstack([Xorb, C])
        return _norm(X)

    sym_best = None
    sym_val = 2.0
    # symmetric generation + short full annealing (breaks symmetry implicitly via GD numerics)
    while time.time() < SYM_END:
        X0 = build_sym()
        # single-orbit rotated mixture variant: rotate first orbit extra
        if rng.rand() < 0.3:
            pA = rand_unit(1)[0]
            rA = rand_unit(1)[0]
            blk = X0[:24].copy()
            blk = _qmul(_qmul(pA, blk), rA)
            blk = _norm(blk)
            X0[:24] = blk
        Xo, vo = _repulse(X0, 550, 60.0, 420.0, 0.02, 0.002)
        if vo < sym_val:
            sym_val = vo
            sym_best = Xo.copy()
        if vo < best_val:
            best_val = vo
            best_overall = Xo.copy()
        # small symmetry-break + re-anneal of the same start
        if time.time() < SYM_END - 4:
            Xn = Xo + rng.randn(n, d) * 0.012
            Xn = _norm(Xn)
            Xo2, vo2 = _repulse(Xn, 350, 120.0, 600.0, 0.008, 0.0015)
            if vo2 < sym_val:
                sym_val = vo2
                sym_best = Xo2.copy()
            if vo2 < best_val:
                best_val = vo2
                best_overall = Xo2.copy()
        else:
            break
    if sym_best is None:
        sym_best = best_overall.copy()
        sym_val = best_val

    # basin hopping mixing record basin and symmetric descendants
    cur = best_overall.copy()
    cur_val = best_val
    trial = 0
    sigmas = [0.006, 0.012, 0.025, 0.05, 0.09]
    while time.time() < HOP_END:
        trial += 1
        # choose base: mostly cur, sometimes sym_best, sometimes best_overall
        r = rng.rand()
        if r < 0.55:
            base = cur
        elif r < 0.8:
            base = best_overall
        else:
            base = sym_best
        sig = sigmas[trial % len(sigmas)]
        # targeted contact break: find worst pair, push apart
        Xt = base + rng.randn(n, d) * sig
        if rng.rand() < 0.5:
            try:
                G = base @ base.T
                np.fill_diagonal(G, -2.0)
                flat = np.argmax(G)
                ii, jj = divmod(int(flat), n)
                # push ii and jj apart along difference, perturb neighbours too
                diff = base[ii] - base[jj]
                diff /= (np.linalg.norm(diff) + 1e-12)
                Xt[ii] += diff * sig * 2.0
                Xt[jj] -= diff * sig * 2.0
            except Exception:
                pass
        Xt = _norm(Xt)
        steps = 450 if sig < 0.03 else 650
        Xo, vo = _repulse(Xt, steps, 55.0, 500.0, 0.012, 0.0018)
        if vo < cur_val:
            cur = Xo.copy()
            cur_val = vo
        if vo < best_val:
            best_val = vo
            best_overall = Xo.copy()
        # extra rattler-style kick: move points with smallest row-max
        if trial % 4 == 0 and time.time() < HOP_END - 6:
            try:
                G2 = Xo @ Xo.T
                np.fill_diagonal(G2, -2.0)
                rm = G2.max(axis=1)
                order = np.argsort(rm)[:8]
                Xh = Xo.copy()
                Xh[order] += rng.randn(8, 4) * 0.08
                Xh = _norm(Xh)
                Xh2, vh2 = _repulse(Xh, 350, 80.0, 500.0, 0.010, 0.0015)
                if vh2 < cur_val:
                    cur = Xh2.copy()
                    cur_val = vh2
                if vh2 < best_val:
                    best_val = vh2
                    best_overall = Xh2.copy()
            except Exception:
                pass

    # final LP polish on top two, pick winner
    cands = [(best_overall, best_val), (cur, cur_val)]
    # dedup by value
    polished_best = best_overall.copy()
    polished_val = best_val
    # polish best first with most time
    remain = HARD_END - time.time()
    if remain > 5:
        # give 70% to best, rest to cur if different
        t1 = time.time() + remain * 0.7
        polished = _lp_polish(best_overall, min(t1, HARD_END - 2))
        v = _max_cos(polished)
        if v < polished_val:
            polished_val = v
            polished_best = polished.copy()
        if time.time() < HARD_END - 4:
            # polish cur as alternative start
            try:
                if _max_cos(cur) < polished_val + 0.002:
                    pol2 = _lp_polish(cur, HARD_END - 2)
                    v2 = _max_cos(pol2)
                    if v2 < polished_val:
                        polished_val = v2
                        polished_best = pol2.copy()
            except Exception:
                pass
        # final touch: polish the winner again briefly
        if time.time() < HARD_END - 3:
            try:
                pol3 = _lp_polish(polished_best, HARD_END - 1.5)
                v3 = _max_cos(pol3)
                if v3 < polished_val:
                    polished_val = v3
                    polished_best = pol3.copy()
            except Exception:
                pass
    return _norm(polished_best)
