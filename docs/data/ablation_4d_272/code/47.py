import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def solve():
    t0 = time.time()
    HARD_END = t0 + 172.0
    rng_global = np.random.default_rng(272031)

    def _normr(x):
        n = np.linalg.norm(x, axis=1, keepdims=True)
        n[n == 0] = 1.0
        return x / n

    def _max_cos(x):
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        return float(G.max())

    def _lp_polish(x, deadline, maxpairs=5000):
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
            if cnt > maxpairs:
                idx = np.where(sel)[0]
                order = np.argpartition(-c_all[idx], maxpairs - 1)[:maxpairs]
                keep = np.zeros_like(sel, dtype=bool)
                keep[idx[order]] = True
                sel = keep
            c = c_all[sel]
            I = iu0[0][sel]; J = iu0[1][sel]
            p = len(I)
            nv = n * d + 1
            rows = np.repeat(np.arange(p), 2 * d + 1)
            cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
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
            v = _max_cos(y)
            if v < best:
                x, best = y, v
                radius = min(radius * 1.5, 0.05)
            else:
                radius *= 0.5
        return x, best

    def _smooth_gd(x, steps, a0, a1, s0, s1, mu=0.85, deadline=None):
        n, d = x.shape
        vel = np.zeros_like(x)
        for it in range(steps):
            if deadline is not None and (it % 50 == 0) and time.time() > deadline:
                break
            frac = it / max(steps - 1, 1)
            alpha = a0 * (a1 / a0) ** frac if a0 != a1 else a0
            smax = s0 * (s1 / s0) ** frac if s0 != s1 else s0
            G = x @ x.T
            np.fill_diagonal(G, -2.0)
            m = float(G.max())
            W = np.exp(np.clip(alpha * (G - m), -50, 0))
            np.fill_diagonal(W, 0.0)
            grad = W @ x
            dot = np.sum(grad * x, axis=1, keepdims=True)
            gt = grad - dot * x
            vel = mu * vel + gt
            nrm = float(np.linalg.norm(vel, axis=1).max())
            if nrm > 1e-14:
                x = x - (smax / nrm) * vel
                x = _normr(x)
        return x

    def _refine_hole(Y, z0, alpha=60.0, steps=250, s0=0.05, mu=0.6):
        z = z0 / (np.linalg.norm(z0) + 1e-300)
        vel = np.zeros(4)
        s = s0
        for it in range(steps):
            dots = Y @ z
            m = float(dots.max())
            w = np.exp(np.clip(alpha * (dots - m), -50, 0))
            sw = float(w.sum()) + 1e-300
            g = (w @ Y) / sw
            gt = g - float(g @ z) * z
            nn = float(np.linalg.norm(gt))
            if nn < 1e-14:
                break
            gt = gt / nn
            vel = mu * vel + gt
            vn = float(np.linalg.norm(vel))
            if vn < 1e-14:
                continue
            step = s / (1.0 + it / 80.0)
            z = z - (step / vn) * vel
            nz = float(np.linalg.norm(z))
            if nz < 1e-12 or not np.isfinite(nz):
                z = z0 / (np.linalg.norm(z0) + 1e-300)
                vel *= 0.5
                continue
            z = z / nz
        return z / (np.linalg.norm(z) + 1e-300)

    def _find_hole(Y, rng, n_samples=50000):
        d = Y.shape[1]
        best_s = 2.0
        best_cs = []
        bs = 5000
        # coarse scan: keep top few
        cand = []
        for a in range(0, n_samples, bs):
            nb = min(bs, n_samples - a)
            Cb = rng.standard_normal((nb, d))
            Cb = Cb / np.linalg.norm(Cb, axis=1, keepdims=True)
            S = Cb @ Y.T
            sm = S.max(axis=1)
            # keep best 3 in this block
            k = min(3, nb)
            idx = np.argpartition(sm, k - 1)[:k]
            for ii in idx:
                cand.append((float(sm[ii]), Cb[ii].copy()))
        cand.sort(key=lambda t: t[0])
        cand = cand[:6]
        best_z = cand[0][1].copy()
        best_v = 2.0
        for _, c0 in cand:
            z = _refine_hole(Y, c0, alpha=60.0, steps=300, s0=0.06)
            v = float((Y @ z).max())
            if v < best_v:
                best_v = v
                best_z = z
            if time.time() > HARD_END - 2:
                break
        return best_z

    def _degrees_stress(X, m, alpha=300.0):
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        sel_thresh = m - 2e-3
        deg = np.count_nonzero(G > sel_thresh, axis=1)
        W = np.exp(np.clip(alpha * (G - m), -50, 0))
        np.fill_diagonal(W, 0.0)
        stress = W.sum(axis=1)
        return deg, stress, G

    # ---- load base ----
    base = None
    base_v = 2.0
    for fn in ("best.npy", "record.npy"):
        try:
            A = np.load(fn)
            A = np.asarray(A, dtype=np.float64)
            if A.shape != (272, 4):
                continue
            A = _normr(A)
            if not np.all(np.isfinite(A)):
                continue
            v = _max_cos(A)
            if v < base_v:
                base_v = v
                base = A.copy()
        except Exception:
            continue
    if base is None:
        r = np.random.default_rng(0)
        base = _normr(r.standard_normal((272, 4)))
        base_v = _max_cos(base)
    bestX = base.copy()
    bestV = base_v

    # quick polish of base (short)
    if time.time() < HARD_END - 10:
        try:
            Xp, Vp = _lp_polish(bestX, min(HARD_END - 8, time.time() + 7.0))
            if Vp < bestV and np.all(np.isfinite(Xp)):
                bestX, bestV = Xp, Vp
        except Exception:
            pass

    # ---- N-continuation trials ----
    # (Nd, mode, seed)
    trials = [
        (32, "degree", 101),
        (32, "cluster", 202),
        (28, "stress", 303),
        (36, "random", 404),
        (24, "degree", 505),
    ]
    for ti, (Nd, mode, seed) in enumerate(trials):
        if time.time() > HARD_END - 12:
            break
        trial_end = min(HARD_END - 2, time.time() + (30.0 if ti < 3 else 22.0))
        # time slice guard: leave room for remaining trials
        try:
            X0 = bestX.copy()
            n0 = X0.shape[0]
            G0 = X0 @ X0.T
            np.fill_diagonal(G0, -2.0)
            m0 = float(G0.max())
            deg, stress, _ = _degrees_stress(X0, m0)
            rng = np.random.default_rng(seed)
            if mode == "degree":
                order = np.argsort(-deg, kind="stable")
                # tie-break by stress
                del_idx = np.sort(order[:Nd])
            elif mode == "stress":
                order = np.argsort(-stress, kind="stable")
                del_idx = np.sort(order[:Nd])
            elif mode == "cluster":
                # start from highest-stress point, delete nearest neighbors
                s0i = int(np.argmax(stress + rng.standard_normal(n0) * 1e-9))
                sims = G0[s0i].copy()
                # largest cosine = nearest; include self
                order = np.argsort(-sims, kind="stable")
                del_idx = np.sort(order[:Nd])
            else:  # random
                del_idx = np.sort(rng.choice(n0, size=Nd, replace=False))
            keep = np.ones(n0, dtype=bool)
            keep[del_idx] = False
            Y = _normr(X0[keep].copy())
            # anneal thinned code hard until well below record
            # stage A: strong repulsion anneal
            Y = _smooth_gd(Y, 900, 15.0, 350.0, 0.03, 0.003, mu=0.85, deadline=trial_end - 8)
            if time.time() > trial_end - 6:
                # not enough time to regrow; try polish and compare anyway
                Yp, Vp = _lp_polish(Y, trial_end)
                continue
            Y = _smooth_gd(Y, 700, 80.0, 800.0, 0.012, 0.0015, mu=0.8, deadline=trial_end - 6)
            # brief LP on thinned to compactify
            Y, _ = _lp_polish(Y, min(trial_end - 5, time.time() + 3.0), maxpairs=4000)

            # regrow one at a time
            rng_h = np.random.default_rng(seed * 1000 + 7)
            for k in range(Nd):
                if time.time() > trial_end - 6:
                    break
                z = _find_hole(Y, rng_h, n_samples=50000)
                # freeze older points: refine new point only (already done), append
                Y = np.vstack([Y, z.reshape(1, 4)])
                # brief local settle of last point vs frozen background (extra 150 steps)
                last = Y[-1].copy()
                last = _refine_hole(Y[:-1], last, alpha=80.0, steps=150, s0=0.04)
                Y[-1] = last
                # fully re-anneal after every 6 insertions (and at end)
                if ((k + 1) % 6 == 0) or (k + 1 == Nd):
                    steps_full = 500 if (k + 1 != Nd) else 900
                    # alternate schedules to diversify basin
                    if ((k + 1) // 6) % 2 == 0:
                        Y = _smooth_gd(Y, steps_full, 20.0, 400.0, 0.02, 0.002, mu=0.85, deadline=trial_end - 4)
                    else:
                        Y = _smooth_gd(Y, steps_full, 60.0, 700.0, 0.014, 0.0015, mu=0.8, deadline=trial_end - 4)
            if Y.shape[0] != 272:
                continue
            # final full anneal + LP polish within trial budget
            if time.time() < trial_end - 3:
                Y = _smooth_gd(Y, 600, 30.0, 600.0, 0.012, 0.0012, mu=0.85, deadline=trial_end - 2)
            if time.time() < trial_end - 1:
                Y, Vy = _lp_polish(Y, trial_end, maxpairs=5000)
            else:
                Vy = _max_cos(Y)
            if Vy < bestV and np.all(np.isfinite(Y)):
                bestV = Vy
                bestX = _normr(Y.copy())
        except Exception:
            continue

    # final safety polish if time remains and best can be squeezed
    if time.time() < HARD_END - 4:
        try:
            Xf, Vf = _lp_polish(bestX, HARD_END - 1, maxpairs=6000)
            if Vf < bestV and np.all(np.isfinite(Xf)):
                bestX, bestV = Xf, Vf
        except Exception:
            pass
    bestX = _normr(np.asarray(bestX, dtype=np.float64))
    return bestX
