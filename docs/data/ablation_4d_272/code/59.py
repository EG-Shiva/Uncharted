import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def _normr(x):
    x = np.asarray(x, dtype=np.float64)
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
        if time.time() > deadline - 0.1:
            break
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
        y = _normr(y)
        v = _max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x, best


def _smooth_gd(x, steps, a0, a1, s0, s1, mu=0.85, deadline=None, freeze=None):
    x = _normr(x.copy())
    n = x.shape[0]
    vel = np.zeros_like(x)
    frozen = np.zeros(n, dtype=bool) if freeze is None else np.asarray(freeze, dtype=bool)
    for it in range(steps):
        if deadline is not None and (it % 40 == 0) and time.time() > deadline:
            break
        frac = it / max(steps - 1, 1)
        if a0 != a1:
            import math
            alpha = float(math.exp(math.log(a0) + (math.log(a1) - math.log(a0)) * frac))
        else:
            alpha = float(a0)
        smax = float(s0 + (s1 - s0) * frac)
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        W = np.exp(np.clip(alpha * (G - m), -50, 0))
        np.fill_diagonal(W, 0.0)
        if np.any(frozen):
            W[frozen, :] = 0.0
            W[:, frozen] = 0.0
        grad = W @ x
        dot = np.sum(grad * x, axis=1, keepdims=True)
        gt = grad - dot * x
        if np.any(frozen):
            gt[frozen] = 0.0
            vel[frozen] = 0.0
        vel = mu * vel + gt
        if np.any(frozen):
            vel[frozen] = 0.0
        nrm = float(np.linalg.norm(vel, axis=1).max())
        if nrm < 1e-14 or not np.isfinite(nrm):
            continue
        x = x - (smax / nrm) * vel
        x = _normr(x)
    return x


def _refine_hole(Y, z0, alpha=70.0, steps=220, s0=0.06, mu=0.6):
    z = np.asarray(z0, dtype=np.float64).copy()
    nz = float(np.linalg.norm(z))
    if nz < 1e-12 or not np.isfinite(nz):
        z = np.array([1.0, 0.0, 0.0, 0.0])
    else:
        z = z / nz
    vel = np.zeros(4)
    for it in range(steps):
        dots = Y @ z
        m = float(dots.max())
        w = np.exp(np.clip(alpha * (dots - m), -50, 0))
        sw = float(w.sum()) + 1e-300
        g = (w @ Y) / sw
        gt = g - float(g @ z) * z
        nn = float(np.linalg.norm(gt))
        if nn < 1e-14 or not np.isfinite(nn):
            break
        gt = gt / nn
        vel = mu * vel + gt
        vn = float(np.linalg.norm(vel))
        if vn < 1e-14 or not np.isfinite(vn):
            continue
        step = s0 / (1.0 + it / 70.0)
        z = z - (step / vn) * vel
        nz = float(np.linalg.norm(z))
        if nz < 1e-12 or not np.isfinite(nz):
            z = np.asarray(z0, dtype=np.float64)
            z = z / (float(np.linalg.norm(z)) + 1e-300)
            vel *= 0.5
            continue
        z = z / nz
    return z / (float(np.linalg.norm(z)) + 1e-300)


def _find_hole(Y, rng, n_samples=40000, refine=True):
    d = Y.shape[1]
    best_s = 2.0
    best_c = None
    bs = 5000
    for a in range(0, n_samples, bs):
        nb = min(bs, n_samples - a)
        Cb = rng.standard_normal((nb, d))
        Cb = Cb / np.linalg.norm(Cb, axis=1, keepdims=True)
        S = Cb @ Y.T
        sm = S.max(axis=1)
        k = int(np.argmin(sm))
        if float(sm[k]) < best_s:
            best_s = float(sm[k])
            best_c = Cb[k].copy()
    if best_c is None:
        best_c = rng.standard_normal(d)
        best_c = best_c / np.linalg.norm(best_c)
    if refine:
        best_c = _refine_hole(Y, best_c)
    return best_c


def solve():
    t0 = time.time()
    HARD_END = t0 + 172.0
    rng = np.random.default_rng(272031)

    X_best = None
    best_val = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn).astype(float)
            if a.shape == (272, 4):
                a = _normr(a)
                v = _max_cos(a)
                if v < best_val:
                    best_val = v
                    X_best = a.copy()
        except Exception:
            pass
    if X_best is None:
        X_best = _normr(rng.standard_normal((272, 4)))
        best_val = _max_cos(X_best)

    # initial fast LP squeeze of incumbent
    try:
        dl = min(t0 + 9.0, HARD_END - 60.0)
        if dl > time.time() + 1.0:
            Xp, vp = _lp_polish(X_best, dl, maxpairs=5000)
            if vp < best_val:
                best_val = vp
                X_best = Xp.copy()
    except Exception:
        pass

    X0 = X_best.copy()

    # delete-and-regrow topology loop
    it = 0
    k_cycle = [2, 3, 4, 6, 8, 3, 5, 2, 4, 7]
    while time.time() < HARD_END - 28.0:
        it += 1
        k = k_cycle[(it - 1) % len(k_cycle)]
        # victim selection: high row-max pool + occasional random
        try:
            rm, _ = _row_max(X_best)
        except Exception:
            rm = None
        n = 272
        if rm is not None:
            order = np.argsort(-rm)
            pool = order[:max(24, 4 * k)]
            # choose k victims: mostly worst, some random in pool
            nv_pick = min(k, len(pool))
            # deterministic-ish random choice
            perm = rng.permutation(len(pool))
            # bias: include the very worst with high prob
            victims = set()
            victims.add(int(order[0]))
            if k >= 2:
                victims.add(int(order[1]))
            idx_ptr = 0
            while len(victims) < k and idx_ptr < len(perm):
                victims.add(int(pool[perm[idx_ptr]]))
                idx_ptr += 1
            # occasionally also drop a rattler (min row max) to shake it
            if it % 5 == 0 and len(victims) > 0:
                ratt = int(np.argmin(rm))
                # replace one victim with rattler
                victims.pop()
                victims.add(ratt)
            victims = np.array(sorted(victims), dtype=int)
        else:
            victims = np.sort(rng.choice(n, size=k, replace=False))

        keep = np.ones(n, dtype=bool)
        keep[victims] = False
        Y = X_best[keep].copy()

        # relax thinned code
        try:
            Y = _smooth_gd(Y, steps=350, a0=10.0, a1=220.0, s0=0.020, s1=0.002, mu=0.85,
                           deadline=HARD_END - 25.0)
        except Exception:
            continue
        if time.time() > HARD_END - 25.0:
            break

        # regrow greedily at optimized holes
        Z = Y
        try:
            for j in range(k):
                ns = 30000 if j < 2 else 18000
                z = _find_hole(Z, rng, n_samples=ns, refine=True)
                Z = np.vstack([Z, z[None, :]])
                # light freeze-and-relax: relax only recent + neighbors? just relax new point a bit
                # quick local refine of the new point against frozen others
                zr = _refine_hole(Y if j == 0 else Z[:-1], z, alpha=90.0, steps=120, s0=0.03)
                Z[-1] = zr
            full = _normr(Z.copy())
        except Exception:
            continue

        # re-anneal full code
        try:
            full = _smooth_gd(full, steps=550, a0=12.0, a1=320.0, s0=0.018, s1=0.0015, mu=0.86,
                              deadline=HARD_END - 22.0)
            if time.time() < HARD_END - 22.0:
                full = _smooth_gd(full, steps=300, a0=80.0, a1=900.0, s0=0.008, s1=0.0008, mu=0.88,
                                  deadline=HARD_END - 20.0)
        except Exception:
            continue
        v = _max_cos(full)
        # short LP if promising
        if v < best_val + 0.004 and time.time() < HARD_END - 18.0:
            try:
                dl2 = min(time.time() + 2.2, HARD_END - 15.0)
                full2, v2 = _lp_polish(full, dl2, maxpairs=4500)
                if v2 < v:
                    full, v = full2, v2
            except Exception:
                pass
        if v < best_val:
            best_val = v
            X_best = full.copy()

        # interleaved pure kick + anneal on incumbent every few iters
        if it % 4 == 0 and time.time() < HARD_END - 28.0:
            try:
                sig = 0.006 if (it % 8 == 0) else 0.012
                cand = X_best + sig * rng.standard_normal(X_best.shape)
                cand = _normr(cand)
                cand = _smooth_gd(cand, steps=400, a0=14.0, a1=380.0, s0=0.014, s1=0.0012, mu=0.86,
                                  deadline=HARD_END - 22.0)
                vc = _max_cos(cand)
                if vc < best_val + 0.002 and time.time() < HARD_END - 18.0:
                    dl3 = min(time.time() + 1.8, HARD_END - 15.0)
                    cand2, vc2 = _lp_polish(cand, dl3, maxpairs=4000)
                    if vc2 < vc:
                        cand, vc = cand2, vc2
                if vc < best_val:
                    best_val = vc
                    X_best = cand.copy()
            except Exception:
                pass

    # final long LP polish
    try:
        dlF = HARD_END - 1.5
        if time.time() < dlF - 0.5:
            Xf, vf = _lp_polish(X_best, dlF, maxpairs=6000)
            if vf < best_val:
                best_val = vf
                X_best = Xf.copy()
    except Exception:
        pass

    # safety: if somehow worse than loaded incumbent, return loaded best polished
    try:
        if _max_cos(X_best) > _max_cos(_normr(X0)) + 1e-12:
            # keep the better one
            X_best = _normr(X0.copy())
    except Exception:
        pass

    X_best = _normr(X_best)
    # final NaN / zero guard
    X_best = np.where(np.isfinite(X_best), X_best, 0.0)
    nr = np.linalg.norm(X_best, axis=1, keepdims=True)
    nr[nr == 0] = 1.0
    X_best = X_best / nr
    # tiny deterministic tie-break shuffle guard: ensure shape
    assert X_best.shape == (272, 4)
    return X_best
