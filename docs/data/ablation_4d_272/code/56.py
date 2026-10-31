import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def solve():
    t0 = time.time()
    HARD_END = t0 + 172.0

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
        try:
            best = _max_cos(x)
        except Exception:
            return x, 2.0
        n, d = x.shape
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
            if deadline is not None and (it % 60 == 0) and time.time() > deadline:
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

    def _subset_gd(x, active, steps, alpha, smax, mu=0.5, deadline=None):
        active = np.asarray(active, dtype=int)
        k = len(active)
        if k == 0:
            return x
        vel = np.zeros((k, 4))
        for it in range(steps):
            if deadline is not None and time.time() > deadline:
                break
            Xa = x[active]
            G = Xa @ x.T
            # mask self
            for p_, g_ in enumerate(active):
                G[p_, int(g_)] = -2.0
            m = float(G.max())
            W = np.exp(np.clip(alpha * (G - m), -50, 0))
            grad = W @ x
            dot = np.sum(grad * Xa, axis=1, keepdims=True)
            gt = grad - dot * Xa
            vel = mu * vel + gt
            nrm = float(np.linalg.norm(vel, axis=1).max())
            if nrm < 1e-14:
                continue
            Xa2 = Xa - (smax / nrm) * vel
            Xa2 = _normr(Xa2)
            x[active] = Xa2
        return x

    def _refine_hole(Y, z0, alpha=60.0, steps=250, s0=0.05, mu=0.6):
        z = z0 / (np.linalg.norm(z0) + 1e-300)
        vel = np.zeros(4)
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
            step = s0 / (1.0 + it / 80.0)
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
        cand = []
        bs = 5000
        for a in range(0, n_samples, bs):
            nb = min(bs, n_samples - a)
            Cb = rng.standard_normal((nb, d))
            Cb = Cb / np.linalg.norm(Cb, axis=1, keepdims=True)
            S = Cb @ Y.T
            sm = S.max(axis=1)
            k = min(3, nb)
            idx = np.argpartition(sm, k - 1)[:k]
            for ii in idx:
                cand.append((float(sm[ii]), Cb[ii].copy()))
            if time.time() > HARD_END - 2:
                break
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

    # load base
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

    if time.time() < HARD_END - 12:
        try:
            Xp, Vp = _lp_polish(bestX, min(HARD_END - 10, time.time() + 6.0), maxpairs=5000)
            if Vp < bestV and np.all(np.isfinite(Xp)):
                bestX, bestV = Xp, Vp
        except Exception:
            pass

    # trial schedule: cycle through modes, varying Nd and seeds
    sched = [
        (32, "degree"), (32, "cluster"), (28, "stress"), (36, "random"),
        (24, "degree"), (40, "cluster"), (30, "random"), (26, "stress"),
    ]
    trial_no = 0
    while time.time() < HARD_END - 10:
        Nd, mode = sched[trial_no % len(sched)]
        seed = 1000 + trial_no * 977 + int(Nd * 13)
        # budget: share remaining time, cap ~28s first rounds
        remain = HARD_END - 6 - time.time()
        if remain < 9:
            break
        # leave room for at least 2 more trials if early
        trial_budget = min(30.0, max(10.0, remain / 3.0))
        trial_end = min(HARD_END - 4, time.time() + trial_budget)
        try:
            X0 = bestX.copy()
            n0 = X0.shape[0]
            G0 = X0 @ X0.T
            np.fill_diagonal(G0, -2.0)
            m0 = float(G0.max())
            # degrees and stress
            sel_thresh = m0 - 2e-3
            deg = np.count_nonzero(G0 > sel_thresh, axis=1)
            W0 = np.exp(np.clip(300.0 * (G0 - m0), -50, 0))
            np.fill_diagonal(W0, 0.0)
            stress = W0.sum(axis=1)
            rng = np.random.default_rng(seed)
            if mode == "degree":
                # stable sort by degree, tie-break stress with tiny noise for diversity
                jitter = rng.standard_normal(n0) * 1e-6
                order = np.argsort(-(deg.astype(float) * 1000 + stress + jitter), kind="stable")
                del_idx = np.sort(order[:Nd])
            elif mode == "stress":
                jitter = rng.standard_normal(n0) * 1e-9
                order = np.argsort(-(stress + jitter), kind="stable")
                del_idx = np.sort(order[:Nd])
            elif mode == "cluster":
                s0i = int(np.argmax(stress + rng.standard_normal(n0) * 1e-9))
                sims = G0[s0i].copy()
                order = np.argsort(-sims, kind="stable")
                del_idx = np.sort(order[:Nd])
            else:
                del_idx = np.sort(rng.choice(n0, size=Nd, replace=False))
            keep = np.ones(n0, dtype=bool)
            keep[del_idx] = False
            Y = _normr(X0[keep].copy())
            # Stage A: hard anneal thinned code
            Y = _smooth_gd(Y, 700, 15.0, 350.0, 0.03, 0.003, mu=0.85, deadline=trial_end - 8)
            # small random kick to leave basin
            if time.time() < trial_end - 7:
                amp = 0.01 if (trial_no % 2 == 0) else 0.02
                Y = _normr(Y + rng.standard_normal(Y.shape) * amp)
                Y = _smooth_gd(Y, 500, 80.0, 800.0, 0.012, 0.0015, mu=0.8, deadline=trial_end - 6)
            if time.time() > trial_end - 6:
                trial_no += 1
                continue
            Y, _ = _lp_polish(Y, min(trial_end - 5, time.time() + 2.5), maxpairs=4000)
            # regrow one at a time
            rng_h = np.random.default_rng(seed * 1000 + 77)
            # randomize re-anneal period 4..8 per trial for diversity
            period = 4 + (trial_no % 5)  # 4..8
            for k in range(Nd):
                if time.time() > trial_end - 6:
                    break
                z = _find_hole(Y, rng_h, n_samples=50000)
                Y = np.vstack([Y, z.reshape(1, 4)])
                # freeze older points: settle only newest point briefly
                last = Y[-1].copy()
                last = _refine_hole(Y[:-1], last, alpha=80.0, steps=150, s0=0.04)
                Y[-1] = last
                # light settle of last few points against frozen background
                if Y.shape[0] > 4:
                    active = list(range(Y.shape[0] - min(3, k + 1), Y.shape[0]))
                    Y = _subset_gd(Y, active, 60, 120.0, 0.01, mu=0.5, deadline=trial_end - 5)
                if ((k + 1) % period == 0) or (k + 1 == Nd):
                    steps_full = 450 if (k + 1 != Nd) else 800
                    if (((k + 1) // period) % 2 == 0):
                        Y = _smooth_gd(Y, steps_full, 20.0, 400.0, 0.02, 0.002, mu=0.85, deadline=trial_end - 4)
                    else:
                        Y = _smooth_gd(Y, steps_full, 60.0, 700.0, 0.014, 0.0015, mu=0.8, deadline=trial_end - 4)
            if Y.shape[0] != 272:
                trial_no += 1
                continue
            if time.time() < trial_end - 3:
                Y = _smooth_gd(Y, 500, 30.0, 600.0, 0.012, 0.0012, mu=0.85, deadline=trial_end - 2)
            if time.time() < trial_end - 0.5:
                Y, Vy = _lp_polish(Y, trial_end, maxpairs=5000)
            else:
                Vy = _max_cos(Y)
            if Vy < bestV and np.all(np.isfinite(Y)):
                bestV = Vy
                bestX = _normr(Y.copy())
        except Exception:
            pass
        trial_no += 1

    if time.time() < HARD_END - 3:
        try:
            Xf, Vf = _lp_polish(bestX, HARD_END - 1, maxpairs=6000)
            if Vf < bestV and np.all(np.isfinite(Xf)):
                bestX, bestV = Xf, Vf
        except Exception:
            pass
    bestX = _normr(np.asarray(bestX, dtype=np.float64))
    return bestX
