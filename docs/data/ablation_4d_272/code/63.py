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

    def _row_max(x):
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        return G.max(axis=1), float(G.max())

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

    def _refine_hole(Y, z0, alpha=60.0, steps=200, s0=0.05, mu=0.6):
        z = np.asarray(z0, dtype=float).copy()
        nz0 = float(np.linalg.norm(z))
        if nz0 < 1e-12 or not np.isfinite(nz0):
            z = np.array([1.0, 0.0, 0.0, 0.0])
        else:
            z = z / nz0
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

    def _find_hole(Y, rng, n_samples=20000):
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
        cand = cand[:4]
        best_z = cand[0][1].copy()
        best_v = 2.0
        for _, c0 in cand:
            z = _refine_hole(Y, c0, alpha=60.0, steps=200, s0=0.06)
            v = float((Y @ z).max())
            if v < best_v:
                best_v = v
                best_z = z
            if time.time() > HARD_END - 2:
                break
        return best_z

    def _contact_set(x, tol=1e-6):
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        I, J = np.where(np.triu(G, 1) > m - tol)
        n = x.shape[0]
        s = set((int(i) * n + int(j)) for i, j in zip(I, J))
        return s, m

    def _jaccard(a, b):
        if len(a) == 0 and len(b) == 0:
            return 1.0
        inter = len(a & b)
        union = len(a | b)
        if union == 0:
            return 1.0
        return inter / union

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
            Xp, Vp = _lp_polish(bestX, min(HARD_END - 10, time.time() + 5.0), maxpairs=5000)
            if Vp < bestV and np.all(np.isfinite(Xp)):
                bestX, bestV = Xp, Vp
        except Exception:
            pass

    bestSet, _ = _contact_set(bestX, tol=1e-6)
    pool_X = [bestX.copy()]
    pool_V = [bestV]
    pool_S = [set(bestSet)]

    Kseq = [12, 15, 18, 20, 14, 16, 13, 19]
    trial_no = 0
    # precompute center cycling is done per trial from current base
    while time.time() < HARD_END - 11:
        remain = HARD_END - 5 - time.time()
        if remain < 10:
            break
        trial_budget = min(22.0, max(9.0, remain / 3.5))
        trial_end = min(HARD_END - 4, time.time() + trial_budget)
        try:
            # choose base: mostly best, every 4th from diverse pool
            if len(pool_X) > 1 and (trial_no % 4 == 3):
                # pick best diverse within 1e-5 that is not best
                cands = [(v, i) for i, v in enumerate(pool_V) if v < bestV + 1e-5]
                cands.sort()
                # round robin among them
                bi = cands[(trial_no // 4) % len(cands)][1]
                X0 = pool_X[bi].copy()
                v0 = pool_V[bi]
            else:
                X0 = bestX.copy()
                v0 = bestV
            n0 = X0.shape[0]
            G0 = X0 @ X0.T
            np.fill_diagonal(G0, -2.0)
            m0 = float(G0.max())
            rm, _ = _row_max(X0)
            deg = np.count_nonzero(G0 > m0 - 1e-3, axis=1)
            W0 = np.exp(np.clip(300.0 * (G0 - m0), -50, 0))
            np.fill_diagonal(W0, 0.0)
            stress = W0.sum(axis=1)
            rng = np.random.default_rng(7000 + trial_no * 1319)
            # choose center: high degree 8-10 preferred, sometimes rattler
            rattler = int(np.argmin(rm))
            if trial_no % 7 == 6:
                center = rattler
            else:
                # high degree candidates
                order_deg = np.argsort(-(deg.astype(float) * 1000 + stress))
                # filter deg 8..10 first
                hi = [i for i in order_deg if 8 <= deg[i] <= 10]
                if len(hi) == 0:
                    hi = list(order_deg[:12])
                # cycle with jitter
                center = int(hi[(trial_no * 3 + int(rng.integers(0, len(hi)))) % len(hi)])
            K = int(Kseq[trial_no % len(Kseq)])
            # patch = center + nearest K-1 by cosine
            sims = G0[center].copy()
            sims[center] = 2.0  # ensure center first? we want center included
            order = np.argsort(-sims, kind="stable")
            # ensure center is first
            patch = [center]
            for idx in order:
                if idx == center:
                    continue
                patch.append(int(idx))
                if len(patch) >= K:
                    break
            del_idx = np.sort(np.array(patch, dtype=int))
            keep_mask = np.ones(n0, dtype=bool)
            keep_mask[del_idx] = False
            keep_idx = np.where(keep_mask)[0]
            # multiple initializations
            n_init = 2
            best_trial_X = None
            best_trial_V = 2.0
            for ri in range(n_init):
                if time.time() > trial_end - 6:
                    break
                rng_h = np.random.default_rng(9000 + trial_no * 277 + ri * 1013)
                X = X0.copy()
                # hole-based refill for patch positions in random order
                fill_order = del_idx[rng_h.permutation(len(del_idx))]
                # fixed set grows
                placed = []
                for fi, pos in enumerate(fill_order):
                    # Y = keep points + already placed
                    if len(placed) == 0:
                        Y = X[keep_idx]
                    else:
                        Y = np.vstack([X[keep_idx], np.array(placed)])
                    if ri == 0:
                        z = _find_hole(Y, rng_h, n_samples=20000)
                    else:
                        # second init: mix random + hole
                        if fi % 2 == 0:
                            zr = rng_h.standard_normal(4)
                            zr = zr / np.linalg.norm(zr)
                            z = _refine_hole(Y, zr, alpha=50.0, steps=150, s0=0.07)
                        else:
                            z = _find_hole(Y, rng_h, n_samples=15000)
                    X[pos] = z
                    placed.append(z.copy())
                # stage: frozen-exterior softmax at several temperatures (patch only)
                temps = [40.0, 120.0, 350.0] if ri == 0 else [80.0, 250.0, 600.0]
                smaxs = [0.02, 0.012, 0.008]
                for ti, alpha in enumerate(temps):
                    if time.time() > trial_end - 5:
                        break
                    X = _subset_gd(X, del_idx, 160, alpha, smaxs[ti], mu=0.55, deadline=trial_end - 5)
                if time.time() > trial_end - 4.5:
                    # quick eval without full unfreeze
                    vq = _max_cos(X)
                    if vq < best_trial_V:
                        best_trial_V = vq
                        best_trial_X = X.copy()
                    continue
                # stage: unfreeze boundary ring
                # boundary = top 25 keep points closest to patch (max dot to any patch point)
                Pp = X[del_idx]
                S = X[keep_idx] @ Pp.T
                smax_keep = S.max(axis=1)
                bord_order = np.argsort(-smax_keep, kind="stable")[:25]
                bord_idx = keep_idx[bord_order]
                active2 = np.sort(np.concatenate([del_idx, bord_idx]))
                X = _subset_gd(X, active2, 160, 200.0, 0.010, mu=0.6, deadline=trial_end - 3.5)
                if time.time() < trial_end - 3:
                    X = _smooth_gd(X, 350, 25.0, 500.0, 0.016, 0.002, mu=0.85, deadline=trial_end - 2)
                if time.time() < trial_end - 1.2:
                    X, Vx = _lp_polish(X, trial_end - 0.6, maxpairs=4000)
                else:
                    Vx = _max_cos(X)
                if Vx < best_trial_V and np.all(np.isfinite(X)):
                    best_trial_V = Vx
                    best_trial_X = X.copy()
            if best_trial_X is None:
                trial_no += 1
                continue
            tSet, _ = _contact_set(best_trial_X, tol=1e-6)
            # acceptance
            if best_trial_V < bestV - 1e-12:
                bestX, bestV = best_trial_X.copy(), float(best_trial_V)
                bestSet = set(tSet)
                pool_X.append(bestX.copy()); pool_V.append(bestV); pool_S.append(set(bestSet))
                if len(pool_X) > 5:
                    # drop worst beyond best
                    worst = int(np.argmax(np.array(pool_V[1:])) + 1)
                    # never drop best (index 0 may not be best after update); keep smallest
                    # find index of best
                    bi_best = int(np.argmin(pool_V))
                    if worst != bi_best:
                        pool_X.pop(worst); pool_V.pop(worst); pool_S.pop(worst)
                    else:
                        # drop second worst
                        vs = sorted(range(len(pool_V)), key=lambda i: pool_V[i])
                        drop = vs[-1]
                        pool_X.pop(drop); pool_V.pop(drop); pool_S.pop(drop)
            elif best_trial_V < bestV + 1e-5:
                # diversity check vs all pool
                jmax = max(_jaccard(tSet, s) for s in pool_S) if pool_S else 1.0
                if jmax < 0.85:
                    # preserve genuinely new basin
                    pool_X.append(best_trial_X.copy()); pool_V.append(float(best_trial_V)); pool_S.append(set(tSet))
                    if len(pool_X) > 5:
                        # drop worst non-best that is most similar? just drop worst value not best
                        bi_best = int(np.argmin(pool_V))
                        # candidate worst among non-best
                        others = [i for i in range(len(pool_V)) if i != bi_best]
                        worst = max(others, key=lambda i: pool_V[i])
                        pool_X.pop(worst); pool_V.pop(worst); pool_S.pop(worst)
            trial_no += 1
        except Exception:
            trial_no += 1
            continue

    # final polish best
    try:
        if time.time() < HARD_END - 2:
            Xp, Vp = _lp_polish(bestX, HARD_END - 0.5, maxpairs=5000)
            if Vp < bestV and np.all(np.isfinite(Xp)):
                bestX, bestV = Xp, Vp
    except Exception:
        pass
    bestX = _normr(bestX)
    return bestX
