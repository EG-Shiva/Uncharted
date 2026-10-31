import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def solve():
    t0 = time.time()
    HARD_END = t0 + 172.0
    rngG = np.random.default_rng(272031)

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

    def _full_gd(x, active_all, steps, a0, a1, s0, s1, mu=0.85, deadline=None):
        n = x.shape[0]
        vel = np.zeros_like(x)
        for it in range(steps):
            if deadline is not None and (it % 40 == 0) and time.time() > deadline:
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

    def _sub_gd(x, active, steps, a0, a1, s0, s1, mu=0.7, deadline=None):
        active = np.asarray(active, dtype=int)
        P = len(active)
        # map global col -> position (for self mask)
        # self entries: row r corresponds to global g=active[r], col g
        vel = np.zeros((P, 4))
        # precompute col index of each active for masking: need position in x
        # build lookup global->col (identity since x rows indexed by global)
        for it in range(steps):
            if deadline is not None and (it % 40 == 0) and time.time() > deadline:
                break
            frac = it / max(steps - 1, 1)
            alpha = a0 * (a1 / a0) ** frac if a0 != a1 else a0
            smax = s0 * (s1 / s0) ** frac if s0 != s1 else s0
            Xa = x[active]
            Ga = Xa @ x.T  # P x N
            # mask self dot (should be 1) to -2
            Ga[np.arange(P), active] = -2.0
            m = float(Ga.max())
            W = np.exp(np.clip(alpha * (Ga - m), -50, 0))
            grad = W @ x  # P x 4
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

    def _refine_hole(Y, z0, alpha=60.0, steps=180, s0=0.05, mu=0.6):
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

    def _hole_vs(Y, rng, n_samples=20000):
        d = Y.shape[1]
        Cb = rng.standard_normal((n_samples, d))
        Cb = Cb / np.linalg.norm(Cb, axis=1, keepdims=True)
        S = Cb @ Y.T
        sm = S.max(axis=1)
        i = int(np.argmin(sm))
        z = _refine_hole(Y, Cb[i], alpha=60.0, steps=180, s0=0.06)
        return z

    def _contact_set(x, tol=1e-7):
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        iu = np.triu_indices(x.shape[0], 1)
        sel = iu[0][G[iu] > m - tol]
        sel2 = iu[1][G[iu] > m - tol]
        s = set(zip(sel.tolist(), sel2.tolist()))
        return s, m

    def _jaccard(s1, s2):
        if not s1 and not s2:
            return 1.0
        inter = len(s1.intersection(s2))
        union = len(s1.union(s2))
        return inter / max(union, 1)

    # ---- load ----
    base = None; base_v = 2.0
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
                base_v = v; base = A.copy()
        except Exception:
            continue
    if base is None:
        r = np.random.default_rng(0)
        base = _normr(r.standard_normal((272, 4)))
        base_v = _max_cos(base)
    bestX = base.copy(); bestV = base_v
    if time.time() < HARD_END - 10:
        try:
            Xp, Vp = _lp_polish(bestX, min(HARD_END - 8, time.time() + 6.0), maxpairs=5000)
            if Vp < bestV and np.all(np.isfinite(Xp)):
                bestX, bestV = Xp, Vp
        except Exception:
            pass
    base_contacts, _ = _contact_set(bestX, tol=1e-7)
    # degrees
    try:
        G0 = bestX @ bestX.T
        np.fill_diagonal(G0, -2.0)
        m0 = float(G0.max())
        deg = np.count_nonzero(G0 > m0 - 1e-7, axis=1)
        rmax, _ = _row_max(bestX)
        rattler = int(np.argmin(rmax))
    except Exception:
        deg = np.zeros(272, dtype=int)
        rattler = 0

    order_deg = np.argsort(-deg, kind="stable")
    # candidate centers: top degree + rattler + random
    centers = []
    for c in order_deg[:10]:
        centers.append(int(c))
    centers.append(int(rattler))
    # add rattler neighbor as center too
    try:
        nn_r = int(np.argmax((bestX @ bestX[rattler])) + 2.0 * (np.arange(272) == rattler))
                   ) if False else None
    except Exception:
        pass
    rng_c = np.random.default_rng(999)
    for _ in range(3):
        centers.append(int(rng_c.integers(0, 272)))
    # dedup preserve order
    seen = set(); ucenters = []
    for c in centers:
        if c not in seen:
            seen.add(c); ucenters.append(c)
    centers = ucenters

    patch_sizes = [14, 16, 18, 12, 20, 15, 16, 14, 18, 16, 14, 18, 16]
    archive = []  # list of (X, V, jacc)

    trial_i = 0
    for ci, center in enumerate(centers):
        if time.time() > HARD_END - 14:
            break
        # two patch sizes for first centers if time allows
        sizes_to_try = [patch_sizes[ci % len(patch_sizes)]]
        if ci < 4 and time.time() < HARD_END - 60:
            sizes_to_try.append(patch_sizes[(ci + 5) % len(patch_sizes)])
        for Ps in sizes_to_try:
            if time.time() > HARD_END - 12:
                break
            trial_end = min(HARD_END - 2, time.time() + 13.0)
            try:
                # use current best as exterior reference
                Xref = bestX.copy()
                Gc = Xref @ Xref.T
                np.fill_diagonal(Gc, -2.0)
                mc = float(Gc.max())
                sims = Gc[center].copy()
                # nearest Ps (including center itself: sims[center]=-2 so not included; add it)
                order = np.argsort(-sims, kind="stable")
                # order[0] is nearest neighbor; build del set
                del_idx = np.sort(np.concatenate([[center], order[:Ps-1]]))
                # boundary ring: next 25 nearest among remaining
                rem_order = order[Ps-1:Ps-1+25]
                bnd_idx = np.sort(rem_order)
                patch_and_bnd = np.sort(np.concatenate([del_idx, bnd_idx]))
                old_patch = Xref[del_idx].copy()
                Xfrozen = Xref.copy()
                # ---- refill cavity: multiple inits ----
                rng = np.random.default_rng(10000 + trial_i * 977 + center * 13 + Ps)
                n_init = 3 if trial_i < 4 else 2
                best_patch_X = None
                best_patch_score = 2.0
                # precompute frozen background for hole init
                keep_mask = np.ones(272, dtype=bool); keep_mask[del_idx] = False
                Yback_const = Xref[keep_mask]  # frozen exterior
                for ii in range(n_init):
                    if time.time() > trial_end - 7:
                        break
                    Xcur = Xref.copy()
                    if ii == 0:
                        # perturbed old positions
                        nz = rng.standard_normal(old_patch.shape) * 0.12
                        init = _normr(old_patch + nz)
                        Xcur[del_idx] = init
                    elif ii == 1:
                        # random around center direction + spread
                        cdir = Xref[center].copy()
                        init = rng.standard_normal((Ps, 4)) * 0.55 + cdir[None, :]
                        init = _normr(init)
                        Xcur[del_idx] = init
                    else:
                        # sequential hole-based init against frozen exterior
                        Yseq = Yback_const.copy()
                        seq = []
                        for k in range(Ps):
                            z = _hole_vs(Yseq, rng, n_samples=12000)
                            seq.append(z)
                            Yseq = np.vstack([Yseq, z.reshape(1, 4)])
                        Xcur[del_idx] = _normr(np.array(seq))
                    # stage 1: optimize patch vs frozen exterior at several temps
                    # low temp spread then high temp sharpen
                    Xcur = _sub_gd(Xcur, del_idx, 260, 18.0, 300.0, 0.035, 0.006, mu=0.75, deadline=trial_end-6)
                    Xcur = _sub_gd(Xcur, del_idx, 220, 80.0, 700.0, 0.016, 0.003, mu=0.7, deadline=trial_end-6)
                    # score = max dot involving patch
                    Xa = Xcur[del_idx]; Ga = Xa @ Xcur.T
                    Ga[np.arange(Ps), del_idx] = -2.0
                    sc = float(Ga.max())
                    if sc < best_patch_score:
                        best_patch_score = sc
                        best_patch_X = Xcur.copy()
                if best_patch_X is None:
                    trial_i += 1
                    continue
                Xw = best_patch_X
                # ---- unfreeze in stages ----
                # patch first extra
                if time.time() < trial_end - 6:
                    Xw = _sub_gd(Xw, del_idx, 200, 30.0, 500.0, 0.018, 0.0025, mu=0.75, deadline=trial_end-4)
                # boundary ring + patch
                if time.time() < trial_end - 5:
                    Xw = _sub_gd(Xw, patch_and_bnd, 260, 25.0, 550.0, 0.016, 0.002, mu=0.8, deadline=trial_end-3)
                # all points brief
                if time.time() < trial_end - 3.5:
                    Xw = _full_gd(Xw, None, 320, 25.0, 550.0, 0.012, 0.0015, mu=0.85, deadline=trial_end-2.5)
                # LP polish
                if time.time() < trial_end - 1:
                    Xw, Vw = _lp_polish(Xw, trial_end, maxpairs=5000)
                else:
                    Vw = _max_cos(Xw)
                # evaluate
                if not np.all(np.isfinite(Xw)) or Xw.shape != (272, 4):
                    trial_i += 1
                    continue
                cs, _ = _contact_set(Xw, tol=1e-7)
                jac = _jaccard(base_contacts, cs)
                if Vw < bestV - 1e-12:
                    bestV = Vw; bestX = _normr(Xw.copy())
                    base_contacts = cs
                    # update degrees/center? keep original centers
                elif Vw < bestV + 1e-5 and jac < 0.85:
                    # genuinely new basin: preserve
                    # avoid duplicates
                    dup = False
                    for (Xa_, Va_, ja_) in archive:
                        try:
                            c2, _ = _contact_set(Xa_, tol=1e-7)
                            if _jaccard(cs, c2) > 0.93:
                                dup = True; break
                        except Exception:
                            pass
                    if not dup and len(archive) < 4:
                        archive.append((_normr(Xw.copy()), Vw, jac))
                trial_i += 1
            except Exception:
                trial_i += 1
                continue

    # final polish of archive + best, pick lowest
    cands = [(bestX, bestV)]
    for (Xa, Va, _) in archive:
        cands.append((Xa, Va))
    # quick extra anneal+polish for diverse basins if time
    for k in range(len(cands)):
        if time.time() > HARD_END - 5:
            break
        try:
            Xc, Vc = cands[k]
            # brief anneal then polish only if promising (within 1e-5)
            if Vc < bestV + 1e-5:
                Xc2 = _full_gd(Xc.copy(), None, 250, 40.0, 600.0, 0.010, 0.0012, mu=0.85, deadline=HARD_END-3)
                Xc2, V2 = _lp_polish(Xc2, min(HARD_END-1, time.time()+3.0), maxpairs=5000)
                if V2 < bestV and np.all(np.isfinite(Xc2)):
                    bestV = V2; bestX = _normr(Xc2.copy())
        except Exception:
            continue
    if time.time() < HARD_END - 3:
        try:
            Xf, Vf = _lp_polish(bestX, HARD_END - 1, maxpairs=6000)
            if Vf < bestV and np.all(np.isfinite(Xf)):
                bestX, bestV = Xf, Vf
        except Exception:
            pass
    bestX = _normr(np.asarray(bestX, dtype=np.float64))
    return bestX
