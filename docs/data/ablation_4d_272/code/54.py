import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _normr(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return x / n

def _max_cos(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _row_max(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return G.max(axis=1), float(G.max())

def _lp_polish(x, deadline, max_pairs=5000):
    x = _normr(x.copy())
    n, d = x.shape
    try:
        best = _max_cos(x)
    except Exception:
        return x, 2.0
    radius = 1e-3
    iu0 = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        G = x @ x.T
        c_all = G[iu0]
        m = float(c_all.max())
        thresh = m - max(20.0 * radius, 1e-9)
        sel = c_all > thresh
        cnt = int(np.count_nonzero(sel))
        if cnt == 0:
            radius *= 0.5
            continue
        if cnt > max_pairs:
            idx = np.where(sel)[0]
            order = np.argpartition(-c_all[idx], max_pairs - 1)[:max_pairs]
            keep = np.zeros_like(sel, dtype=bool)
            keep[idx[order]] = True
            sel = keep
        c = c_all[sel]
        I = iu0[0][sel]; J = iu0[1][sel]
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
        obj = np.zeros(nv); obj[-1] = 1.0
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
        y = _normr(y)
        try:
            v = _max_cos(y)
        except Exception:
            radius *= 0.5
            continue
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x, best

def _smooth_gd(x, steps, a0, a1, s0, s1, mu=0.85, deadline=None, balance=0.0):
    n, d = x.shape
    x = _normr(x.copy())
    vel = np.zeros_like(x)
    best_X = x.copy()
    G0 = x @ x.T
    np.fill_diagonal(G0, -2.0)
    bv = float(G0.max())
    for it in range(steps):
        if deadline is not None and time.time() > deadline:
            break
        frac = it / max(steps - 1, 1)
        alpha = a0 * (a1 / a0) ** frac if a0 != a1 else a0
        smax = s0 * (s1 / s0) ** frac if s0 != s1 else s0
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        if m < bv:
            bv = m
            best_X = x.copy()
        W = np.exp(np.clip(alpha * (G - m), -50, 0))
        np.fill_diagonal(W, 0.0)
        grad = W @ x
        if balance != 0.0:
            # degree equalization: boost high-contact rows
            cnt = np.count_nonzero(G > m - 0.002, axis=1).astype(np.float64)
            mean = float(cnt.mean()) + 1e-9
            w = 1.0 + balance * (cnt - mean) / mean
            grad = grad * w[:, None]
        dot = np.sum(grad * x, axis=1, keepdims=True)
        gt = grad - dot * x
        vel = mu * vel + gt
        nrm = float(np.linalg.norm(vel, axis=1).max())
        if nrm > 1e-14:
            x = x - (smax / nrm) * vel
            x = _normr(x)
    return best_X, bv

def _riesz_gd(x, steps, s_pow=6.0, s0=0.05, s1=0.005, mu=0.85, deadline=None):
    n, d = x.shape
    x = _normr(x.copy())
    vel = np.zeros_like(x)
    for it in range(steps):
        if deadline is not None and time.time() > deadline:
            break
        frac = it / max(steps - 1, 1)
        smax = s0 * (s1 / s0) ** frac if s0 != s1 else s0
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        R = 2.0 - 2.0 * G
        np.fill_diagonal(R, np.inf)
        # W = s * R^{-s/2 -1}, clip to avoid overflow
        with np.errstate(divide='ignore', over='ignore'):
            W = np.power(np.clip(R, 1e-6, 4.0), -s_pow / 2.0 - 1.0)
        np.fill_diagonal(W, 0.0)
        if not np.all(np.isfinite(W)):
            W[~np.isfinite(W)] = 0.0
        grad = W @ x
        dot = np.sum(grad * x, axis=1, keepdims=True)
        gt = grad - dot * x
        vel = mu * vel + gt
        nrm = float(np.linalg.norm(vel, axis=1).max())
        if nrm > 1e-14:
            x = x - (smax / nrm) * vel
            x = _normr(x)
    return x

def _largest_holes(X, k, rng, n_samples=4000):
    n, d = X.shape
    C = rng.standard_normal((n_samples, d))
    C = _normr(C)
    S = C @ X.T
    sm = S.max(axis=1)
    idx = np.argpartition(sm, k)[:k]
    return C[idx].copy(), sm[idx].copy()

def _rand_SO4(rng, mag=0.3):
    A = rng.standard_normal((4, 4)) * mag
    A = A - A.T
    # exp approx via QR of I+A (Cayley-like): Q from I+A
    M = np.eye(4) + A
    Q, R = np.linalg.qr(M)
    # fix signs to make close to I
    dv = np.sign(np.diag(R))
    dv[dv == 0] = 1.0
    Q = Q * dv[None, :]
    if np.linalg.det(Q) < 0:
        Q[:, 0] *= -1
    return Q

def _hopf_init(rng, Nt, Np_list, jitter=0.02):
    pts = []
    for ti in range(Nt):
        frac = (ti + 0.5) / Nt
        th = np.arcsin(np.sqrt(frac))
        ct, st = np.cos(th), np.sin(th)
        Np = Np_list[ti] if isinstance(Np_list, (list, np.ndarray)) else Np_list
        # 2D grid on torus: factor Np = a*b
        a = int(np.sqrt(Np))
        while a > 1 and Np % a != 0:
            a -= 1
        b = Np // a
        off1 = rng.random() * 2 * np.pi
        off2 = rng.random() * 2 * np.pi
        for i in range(a):
            for j in range(b):
                p1 = off1 + 2 * np.pi * (i / max(a, 1) + rng.random() * 0.1 / max(a, 1))
                p2 = off2 + 2 * np.pi * (j / max(b, 1) + (i * 0.618) % 1.0 * 0.2)
                pts.append([ct * np.cos(p1), ct * np.sin(p1), st * np.cos(p2), st * np.sin(p2)])
    pts = np.array(pts, dtype=np.float64)
    # adjust to exactly 272
    if len(pts) > 272:
        pts = pts[:272]
    while len(pts) < 272:
        v = rng.standard_normal((4,))
        v /= np.linalg.norm(v)
        pts = np.vstack([pts, v[None, :]])
    pts = pts + rng.standard_normal(pts.shape) * jitter
    return _normr(pts)


def solve():
    t0 = time.time()
    HARD = t0 + 172.0
    PHASE_A_END = t0 + 45.0
    PHASE_B_END = t0 + 142.0
    rng = np.random.default_rng(7)
    n, d = 272, 4

    cand = []
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=np.float64)
            if a.shape == (n, d):
                a = _normr(a)
                if np.all(np.isfinite(a)):
                    cand.append(a)
        except Exception:
            pass
    if not cand:
        cand.append(_normr(rng.standard_normal((n, d))))
    vals = [_max_cos(a) for a in cand]
    bi = int(np.argmin(vals))
    best = cand[bi].copy()
    best_val = float(vals[bi])

    def time_left():
        return HARD - time.time()

    # ---------- Phase A: diverse balanced starts ----------
    try:
        starts = []
        # random
        starts.append(_normr(rng.standard_normal((n, d))))
        starts.append(_normr(rng.standard_normal((n, d))))
        # hopf variants
        try:
            starts.append(_hopf_init(rng, 17, 16, jitter=0.03))
        except Exception:
            pass
        try:
            starts.append(_hopf_init(rng, 8, 34, jitter=0.03))
        except Exception:
            pass
        # large shake of record
        for sig in (0.22, 0.35):
            try:
                starts.append(_normr(best + rng.standard_normal((n, d)) * sig))
            except Exception:
                pass
        for si, s0 in enumerate(starts):
            if time.time() > PHASE_A_END - 1:
                break
            try:
                # Riesz spread to balance degrees, then sharpen
                Xr = _riesz_gd(s0, steps=350, s_pow=5.0, s0=0.06, s1=0.01, mu=0.85, deadline=PHASE_A_END)
                bal = 0.5 if si % 2 == 0 else 0.0
                Xb, vb = _smooth_gd(Xr, steps=550, a0=25.0, a1=380.0, s0=0.05, s1=0.004,
                                    mu=0.85, deadline=PHASE_A_END, balance=bal)
                # quick LP only if promising
                if vb < best_val + 0.002 and time.time() < PHASE_A_END - 6:
                    Xb2, v2 = _lp_polish(Xb, min(PHASE_A_END - 0.5, time.time() + 3.0))
                    if v2 < vb:
                        Xb, vb = Xb2, v2
                if vb < best_val:
                    best_val = float(vb)
                    best = _normr(Xb.copy())
            except Exception:
                continue
    except Exception:
        pass

    # ---------- Phase B: basin hopping with topology changes ----------
    cur = best.copy()
    cur_val = float(best_val)
    # precompute temperature schedule
    it_count = 0
    sigmas = [0.015, 0.04, 0.10, 0.22, 0.35]
    while time.time() < PHASE_B_END - 1.0:
        it_count += 1
        frac = max(0.0, min(1.0, (time.time() - t0) / max(PHASE_B_END - t0, 1e-9)))
        Temp = 3e-4 * (1 - frac) + 2e-6 * frac
        op = int(rng.integers(0, 5))
        try:
            Xt = cur.copy()
            if op == 0:
                sig = sigmas[int(rng.integers(0, len(sigmas)))]
                Xt = _normr(Xt + rng.standard_normal((n, d)) * sig)
            elif op == 1:
                # cluster blast around worst pair
                G = Xt @ Xt.T
                np.fill_diagonal(G, -2.0)
                ij = int(np.argmax(G.ravel()))
                ii, jj = ij // n, ij % n
                dist = Xt @ Xt[ii]
                # k nearest to ii (including jj region)
                k = int(rng.integers(12, 36))
                order = np.argpartition(-dist, k)[:k]
                sig = float(rng.uniform(0.15, 0.45))
                Xt[order] = Xt[order] + rng.standard_normal((k, d)) * sig
                Xt = _normr(Xt)
            elif op == 2:
                # cap rotation: rigid SO4 on a cap -> changes cross topology
                c = rng.standard_normal((4,)); c /= np.linalg.norm(c)
                dots = Xt @ c
                k = int(rng.integers(18, 45))
                thr_idx = np.argpartition(-dots, k)[:k]
                Q = _rand_SO4(rng, mag=float(rng.uniform(0.15, 0.55)))
                Xt[thr_idx] = Xt[thr_idx] @ Q.T
                Xt[thr_idx] = Xt[thr_idx] + rng.standard_normal((k, d)) * 0.02
                Xt = _normr(Xt)
            elif op == 3:
                # rattler-to-hole regrowth: move low-contact points to holes
                G = Xt @ Xt.T
                np.fill_diagonal(G, -2.0)
                m = float(G.max())
                cnt = np.count_nonzero(G > m - 0.002, axis=1)
                nh = int(rng.integers(3, 9))
                victims = np.argpartition(cnt, nh)[:nh]
                holes, _ = _largest_holes(Xt, nh, rng, n_samples=3500)
                Xt[victims] = holes + rng.standard_normal((nh, d)) * 0.02
                Xt = _normr(Xt)
            else:
                # delete worst cluster and regrow + global jitter
                G = Xt @ Xt.T
                np.fill_diagonal(G, -2.0)
                rm = G.max(axis=1)
                nh = int(rng.integers(6, 14))
                victims = np.argpartition(-rm, nh)[:nh]
                holes, _ = _largest_holes(Xt, nh, rng, n_samples=3500)
                Xt[victims] = holes + rng.standard_normal((nh, d)) * 0.03
                Xt = _normr(Xt + rng.standard_normal((n, d)) * 0.008)

            # anneal: short Riesz if big move else softmax
            if op in (1, 3, 4) and rng.random() < 0.35:
                Xt = _riesz_gd(Xt, steps=120, s_pow=6.0, s0=0.02, s1=0.008, mu=0.8, deadline=PHASE_B_END)
            bal = 0.6 if rng.random() < 0.5 else 0.0
            a0 = float(rng.uniform(20.0, 60.0))
            a1 = float(rng.uniform(250.0, 500.0))
            steps = int(rng.integers(280, 520))
            Xn, vn = _smooth_gd(Xt, steps=steps, a0=a0, a1=a1, s0=0.04, s1=0.003,
                                mu=0.85, deadline=PHASE_B_END, balance=bal)
            if vn < best_val:
                best_val = float(vn)
                best = _normr(Xn.copy())
                cur, cur_val = _normr(Xn.copy()), float(vn)
            elif vn < cur_val + Temp:
                cur, cur_val = _normr(Xn.copy()), float(vn)
            else:
                # occasional reset to best with medium kick to avoid stagnation
                if rng.random() < 0.06:
                    cur = _normr(best + rng.standard_normal((n, d)) * 0.05)
                    cur_val = _max_cos(cur)
            # quick LP on promising new basins
            if vn < best_val + 1e-5 and time.time() < PHASE_B_END - 4:
                Xp, vp = _lp_polish(Xn, min(PHASE_B_END - 0.5, time.time() + 2.0), max_pairs=4000)
                if vp < best_val:
                    best_val = float(vp)
                    best = _normr(Xp.copy())
                    cur, cur_val = _normr(Xp.copy()), float(vp)
        except Exception:
            continue
        # stuck escape: if cur far worse than best for long, reset
        if it_count % 25 == 0 and cur_val > best_val + 0.004:
            try:
                cur = _normr(best + rng.standard_normal((n, d)) * 0.08)
                cur_val = _max_cos(cur)
            except Exception:
                pass

    # ---------- Phase C: final intensive polish ----------
    try:
        # small ensemble around best: shake + gd + lp, keep best
        while time.time() < HARD - 1.0:
            # first do direct LP on current best with remaining budget
            try:
                Xp, vp = _lp_polish(best, HARD - 0.5, max_pairs=6000)
                if vp < best_val:
                    best_val, best = float(vp), _normr(Xp.copy())
            except Exception:
                pass
            if time.time() > HARD - 8:
                break
            try:
                sig = 0.006 if rng.random() < 0.7 else 0.018
                Xt = _normr(best + rng.standard_normal((n, d)) * sig)
                Xn, vn = _smooth_gd(Xt, steps=400, a0=40.0, a1=450.0, s0=0.02, s1=0.002,
                                    mu=0.85, deadline=HARD - 4, balance=0.3)
                if vn < best_val + 0.0004:
                    Xp2, vp2 = _lp_polish(Xn, min(HARD - 0.5, time.time() + 4.0), max_pairs=6000)
                    if vp2 < best_val:
                        best_val, best = float(vp2), _normr(Xp2.copy())
                    elif vp2 < best_val + 1e-9:
                        pass
                else:
                    if vn < best_val:
                        best_val, best = float(vn), _normr(Xn.copy())
            except Exception:
                continue
            # only 1-2 rounds fit; loop checks time
            if time.time() > HARD - 2:
                break
    except Exception:
        pass

    best = _normr(np.asarray(best, dtype=np.float64))
    return best
