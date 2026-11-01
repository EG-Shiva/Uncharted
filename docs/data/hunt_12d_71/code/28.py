import time
import collections
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _norm_rows(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[~np.isfinite(n)] = 1.0
    n[n == 0] = 1.0
    return X / n

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _row_max(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return G.max(axis=1)

def _subset_quench(X_init, movable, steps, T0, T1, lr, mu, rng=None, noise=0.0):
    X = np.array(X_init, dtype=float, copy=True)
    n, d = X.shape
    mov = np.asarray(movable, dtype=int)
    V = np.zeros_like(X)
    lt0, lt1 = np.log(T0), np.log(T1)
    G0 = X @ X.T
    np.fill_diagonal(G0, -2.0)
    best_m = float(G0.max())
    best_X = X.copy()
    for s in range(steps):
        frac = s / max(1, steps - 1)
        T = float(np.exp(lt0 + (lt1 - lt0) * frac))
        G = X @ X.T
        np.fill_diagonal(G, -1e100)
        m = float(G.max())
        Q = (G - m) / T
        np.clip(Q, -50, 0, out=Q)
        E = np.exp(Q)
        np.fill_diagonal(E, 0.0)
        rs = E.sum(axis=1, keepdims=True)
        rs[rs == 0] = 1.0
        Grad = (E @ X) / rs
        dot = np.sum(Grad * X, axis=1, keepdims=True)
        Gt = Grad - dot * X
        # freeze
        # zero frozen rows
        if len(mov) < n:
            mask = np.ones(n, dtype=bool)
            mask[mov] = False
            Gt[mask] = 0.0
            V[mask] = 0.0
        gn = np.linalg.norm(Gt, axis=1, keepdims=True) + 1e-12
        bad = gn[:, 0] > 2.0
        if np.any(bad):
            Gt[bad] *= (2.0 / gn[bad])
        if rng is not None and noise > 0:
            N = rng.standard_normal((n, d)) * noise
            N -= X * np.sum(N * X, axis=1, keepdims=True)
            # only movable noise
            if len(mov) < n:
                N[mask] = 0.0
            Gt = Gt + N
        V = mu * V + Gt
        X[mov] = X[mov] - lr * V[mov]
        X[mov] = _norm_rows(X[mov])
        if (s % 80 == 0) or (s == steps - 1):
            Gc = X @ X.T
            np.fill_diagonal(Gc, -2.0)
            vc = float(Gc.max())
            if vc < best_m:
                best_m = vc
                best_X = X.copy()
    return best_X, best_m

def _full_quench_track(X_init, steps, T0, T1, lr, mu=0.9):
    X = np.array(X_init, dtype=float, copy=True)
    n, d = X.shape
    V = np.zeros_like(X)
    lt0, lt1 = np.log(T0), np.log(T1)
    G0 = X @ X.T
    np.fill_diagonal(G0, -2.0)
    best_m = float(G0.max())
    best_X = X.copy()
    for s in range(steps):
        frac = s / max(1, steps - 1)
        T = float(np.exp(lt0 + (lt1 - lt0) * frac))
        G = X @ X.T
        np.fill_diagonal(G, -1e100)
        m = float(G.max())
        Q = (G - m) / T
        np.clip(Q, -50, 0, out=Q)
        E = np.exp(Q)
        np.fill_diagonal(E, 0.0)
        rs = E.sum(axis=1, keepdims=True)
        rs[rs == 0] = 1.0
        Grad = (E @ X) / rs
        dot = np.sum(Grad * X, axis=1, keepdims=True)
        Gt = Grad - dot * X
        gn = np.linalg.norm(Gt, axis=1, keepdims=True) + 1e-12
        bad = gn[:, 0] > 2.0
        if np.any(bad):
            Gt[bad] *= (2.0 / gn[bad])
        V = mu * V + Gt
        X = X - lr * V
        X = _norm_rows(X)
        if (s % 100 == 0) or (s == steps - 1):
            Gc = X @ X.T
            np.fill_diagonal(Gc, -2.0)
            vc = float(Gc.max())
            if vc < best_m:
                best_m = vc
                best_X = X.copy()
    return best_X, best_m

def _lp_polish_wide(x, seconds, deadline=None, r0=3e-3):
    x = _norm_rows(np.array(x, dtype=float, copy=True))
    n, d = x.shape
    try:
        best = _max_cos(x)
    except Exception:
        return x
    radius = r0
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds:
        if deadline is not None and time.time() > deadline:
            break
        if radius < 1e-14:
            break
        cmat = (x @ x.T)[iu]
        m = float(cmat.max())
        # wide band: include pairs within max(30*radius, 3e-4)
        band = max(30.0 * radius, 3e-4)
        sel = cmat > m - band
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        if p == 0:
            break
        if p > 2000:
            order = np.argsort(-cmat[sel])[:2000]
            I, J = I[order], J[order]
            csel = cmat[sel][order]
            p = 2000
        else:
            csel = cmat[sel]
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cc = np.concatenate([I[:, None] * d + np.arange(d),
                             J[:, None] * d + np.arange(d),
                             np.full((p, 1), n * d)], 1).ravel()
        vv = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vv, (rows, cc)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        except Exception:
            break
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y = _norm_rows(y)
        v = _max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def _single_refine(s, R, steps=200, T=0.004, lr=0.12, mu=0.8):
    s = s / (np.linalg.norm(s) + 1e-12)
    v = np.zeros_like(s)
    for _ in range(steps):
        g = R @ s
        m = float(g.max())
        e = np.exp(np.clip((g - m) / T, -50, 0))
        den = e.sum() + 1e-30
        Grad = (R.T @ e) / den
        Gt = Grad - float(Grad @ s) * s
        v = mu * v + Gt
        s = s - lr * v
        s = s / (np.linalg.norm(s) + 1e-12)
    return s

def _find_hole(R, rng, n_sample=6000, n_cand=3):
    Z = rng.standard_normal(size=(n_sample, R.shape[1]))
    Z = _norm_rows(Z)
    S = Z @ R.T
    scores = S.max(axis=1)
    idx = np.argpartition(scores, n_cand)[:n_cand]
    best_s = None
    best_v = 1e100
    for k in idx:
        s1 = _single_refine(Z[k], R, steps=180, T=0.005, lr=0.12)
        s1 = _single_refine(s1, R, steps=180, T=0.0015, lr=0.07)
        v = float((R @ s1).max())
        if v < best_v:
            best_v = v
            best_s = s1
    return best_s / (np.linalg.norm(best_s) + 1e-12), best_v

def _fingerprint(X, k=24, scale=1e6):
    G = X @ X.T
    iu = np.triu_indices(X.shape[0], 1)
    c = G[iu]
    idx = np.argpartition(-c, k)[:k]
    top = np.sort(c[idx])
    return tuple((np.round(top * scale)).astype(int).tolist())

def solve():
    t0 = time.time()
    HARD_END = t0 + 173.0
    FINAL_POLISH = 18.0
    HOP_END = HARD_END - FINAL_POLISH - 1.0
    rng = np.random.default_rng(20261101)
    n, d = 71, 12
    cands = []
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.asarray(np.load(fn), dtype=float)
            if a.shape == (71, 12) and np.all(np.isfinite(a)):
                a = _norm_rows(a)
                cands.append(( _max_cos(a), a))
        except Exception:
            pass
    if not cands:
        a = _norm_rows(rng.standard_normal((71, 12)))
        cands.append((_max_cos(a), a))
    cands.sort(key=lambda t: t[0])
    best_score, best_X = cands[0][0], cands[0][1].copy()
    cur_X = best_X.copy()
    cur_score = best_score
    # tabu / archive
    tabu_sets = collections.deque(maxlen=80)
    tabu_set_lookup = set()
    archive_fp = set()
    archive_fp.add(_fingerprint(best_X))
    iu_all = np.triu_indices(n, 1)
    it = 0
    sigmas = [0.02, 0.05, 0.09, 0.15, 0.03, 0.07, 0.12]
    while time.time() < HOP_END:
        it += 1
        if time.time() > HOP_END - 2.0:
            break
        frac_time = (time.time() - t0) / max(HOP_END - t0, 1e-9)
        Tmet = 0.0009 * (1 - frac_time) + 0.00006 * frac_time
        # base: mostly cur, sometimes best
        base = cur_X if rng.random() < 0.7 else best_X
        Gb = base @ base.T
        np.fill_diagonal(Gb, -2.0)
        pmx = Gb.max(axis=1)
        mglob = float(Gb.max())
        # near-contact band counts (wide)
        band_c = 5e-4
        cnt = (Gb > mglob - band_c).sum(axis=1)
        # worst pair
        kflat = int(np.argmax(Gb[iu_all]))
        wi, wj = int(iu_all[0][kflat]), int(iu_all[1][kflat])
        r = rng.random()
        Xnew = None
        mov_for_tabu = None
        if r < 0.52:
            # coordinated rattler surgery: movable = max pair + 6-18 rattlers
            k_r = int(rng.integers(6, 19))  # 6..18
            # choose selection mode
            mode = rng.random()
            cand_idx = [i for i in range(n) if i != wi and i != wj]
            if mode < 0.55:
                # closest row-max to global max (most loaded)
                order = sorted(cand_idx, key=lambda i: (-pmx[i], -cnt[i]))
                chosen = order[:k_r]
            elif mode < 0.8:
                # most reduces near-contact band: highest cnt, tie by pmx
                order = sorted(cand_idx, key=lambda i: (-cnt[i], -pmx[i]))
                # take top with some random swap for diversity
                chosen = order[:k_r]
                if rng.random() < 0.5 and k_r >= 3:
                    # replace 1-2 with random low-load
                    lows = sorted(cand_idx, key=lambda i: (pmx[i]))[:20]
                    for _ in range(int(rng.integers(1, 3))):
                        chosen[rng.integers(0, k_r)] = lows[rng.integers(0, len(lows))]
            else:
                # mixed: half loaded, half random rattlers (low row-max)
                order_hi = sorted(cand_idx, key=lambda i: (-pmx[i]))[:30]
                order_lo = sorted(cand_idx, key=lambda i: (pmx[i]))[:30]
                h = k_r // 2
                chosen = list(rng.choice(order_hi, size=h, replace=False)) + list(rng.choice(order_lo, size=k_r - h, replace=False))
            movable = np.array([wi, wj] + list(map(int, chosen)), dtype=int)
            movable = np.unique(movable)
            # tabu: force unseen subset
            key = frozenset(movable.tolist())
            tries = 0
            while key in tabu_set_lookup and tries < 6:
                # resample 2 entries randomly
                pool = [i for i in range(n) if i not in movable]
                movable_arr = movable.copy()
                # replace two random non-pair entries
                nonpair = [v for v in movable_arr if v != wi and v != wj]
                if len(nonpair) >= 1:
                    out = rng.choice(nonpair, size=min(2, len(nonpair)), replace=False)
                    inn = rng.choice(pool, size=len(out), replace=False)
                    s = set(movable_arr.tolist())
                    for o, nn in zip(out, inn):
                        s.remove(int(o)); s.add(int(nn))
                    movable = np.array(sorted(s), dtype=int)
                    key = frozenset(movable.tolist())
                else:
                    break
                tries += 1
            mov_for_tabu = key
            # multi-start CMA-style random tangent proposals with true max objective
            nstarts = 3
            best_local_X = None
            best_local_s = 1e100
            for si in range(nstarts):
                sigma = sigmas[(it + si) % len(sigmas)]
                # larger sigma for later starts
                if si == 2:
                    sigma = float(rng.uniform(0.08, 0.20))
                X0 = base.copy()
                N = rng.standard_normal((len(movable), d)) * sigma
                # tangent project
                Xm = X0[movable]
                N -= Xm * np.sum(N * Xm, axis=1, keepdims=True)
                X0[movable] = _norm_rows(Xm + N)
                # anneal params depend on sigma
                steps = 650 if sigma < 0.08 else 800
                T0a = 0.014 if sigma < 0.08 else 0.020
                Xq, _ = _subset_quench(X0, movable, steps, T0a, 0.00035, 0.06 if sigma < 0.1 else 0.08, 0.88, None, 0.0)
                # exact score with true max
                s = _max_cos(Xq)
                if s < best_local_s:
                    best_local_s = s
                    best_local_X = Xq
            Xnew = best_local_X
        elif r < 0.68:
            # farthest-hole relocation of a chosen rattler
            # choose rattler: among low/mid row-max, or high-load to relieve
            if rng.random() < 0.6:
                # pick a rattler with high row-max (closest to max) to move
                order = np.argsort(-pmx)
                # avoid max pair sometimes
                pool = [i for i in order[:20] if i != wi and i != wj]
                if len(pool) == 0:
                    pool = list(order[:10])
                ri = int(rng.choice(pool))
            else:
                order = np.argsort(pmx)
                ri = int(order[int(rng.integers(0, 15))])
            mov_for_tabu = frozenset([ri])
            mask = np.ones(n, dtype=bool); mask[ri] = False
            Y = base[mask]
            try:
                hole, _ = _find_hole(Y, rng, n_sample=5000, n_cand=3)
            except Exception:
                hole = _norm_rows(rng.standard_normal((1, d))).ravel()
            Xnew = base.copy()
            Xnew[ri] = hole
            # short joint quench of ri + its new neighbours (top dots)
            dots = Y @ hole
            nb = np.argsort(-dots)[:6]
            movable2 = np.array([ri] + [int(x) if x < ri else int(x)+1 for x in nb[:4]], dtype=int)
            # map nb indices in Y to full indices: Y index -> full
            full_idx = np.where(mask)[0]
            movable2 = np.unique(np.concatenate([[ri], full_idx[nb[:4]]]))
            Xnew, _ = _subset_quench(Xnew, movable2, 500, 0.008, 0.0003, 0.05, 0.88, None, 0.0)
        elif r < 0.83:
            # symmetric pair-apart along tangent bisector + jitter rattlers
            mov_for_tabu = frozenset([wi, wj])
            Xnew = base.copy()
            xa = Xnew[wi].copy(); xb = Xnew[wj].copy()
            dot = float(xa @ xb)
            ta = xb - dot * xa
            tb = xa - dot * xb
            ta /= (np.linalg.norm(ta) + 1e-12)
            tb /= (np.linalg.norm(tb) + 1e-12)
            alpha = float(rng.uniform(0.04, 0.16))
            # move apart symmetrically
            xa2 = xa - alpha * ta
            xb2 = xb - alpha * tb
            # add small orthogonal noise
            na = rng.standard_normal(d) * 0.015
            nbv = rng.standard_normal(d) * 0.015
            na -= xa * float(na @ xa); nbv -= xb * float(nbv @ xb)
            Xnew[wi] = (xa2 + na) / (np.linalg.norm(xa2 + na) + 1e-12)
            Xnew[wj] = (xb2 + nbv) / (np.linalg.norm(xb2 + nbv) + 1e-12)
            # jitter 2-4 rattlers jointly
            lows = np.argsort(pmx)[:25]
            jr = rng.choice(lows, size=int(rng.integers(2, 5)), replace=False)
            Xnew[jr] += rng.standard_normal((len(jr), d)) * 0.03
            Xnew[jr] = _norm_rows(Xnew[jr])
            movable3 = np.unique(np.concatenate([[wi, wj], jr]))
            Xnew, _ = _subset_quench(Xnew, movable3, 550, 0.010, 0.0003, 0.055, 0.88, None, 0.0)
        else:
            # swap high-load and low-load positions + noise
            hi = int(np.argmax(pmx))
            lo = int(np.argmin(pmx))
            if hi == lo:
                lo = int(np.argsort(pmx)[1])
            Xnew = base.copy()
            tmp = Xnew[hi].copy()
            Xnew[hi] = Xnew[lo].copy()
            Xnew[lo] = tmp
            # perturb both to break permutation symmetry
            Xnew[[hi, lo]] += rng.standard_normal((2, d)) * 0.05
            Xnew[[hi, lo]] = _norm_rows(Xnew[[hi, lo]])
            mov_for_tabu = frozenset([hi, lo])
            movable4 = np.array([hi, lo] + [int(x) for x in np.argsort(-(Xnew @ Xnew[hi].T).ravel())[:3] if x != hi and x != lo][:2], dtype=int)
            movable4 = np.unique(movable4)
            Xnew, _ = _subset_quench(Xnew, movable4, 500, 0.010, 0.0003, 0.055, 0.88, None, 0.0)
        if Xnew is None:
            continue
        try:
            q0 = _max_cos(Xnew)
        except Exception:
            continue
        # short wide-band LP before exact accept (if promising)
        if q0 < best_score + 0.002 and time.time() < HOP_END - 1.5:
            try:
                Xnew = _lp_polish_wide(Xnew, 0.6, deadline=HOP_END)
                q0 = _max_cos(Xnew)
            except Exception:
                pass
        fp = None
        try:
            fp = _fingerprint(Xnew)
        except Exception:
            fp = None
        is_tabu_fp = (fp in archive_fp) if fp is not None else False
        # tabu update for subsets
        if mov_for_tabu is not None:
            if mov_for_tabu not in tabu_set_lookup:
                tabu_sets.append(mov_for_tabu)
                tabu_set_lookup.add(mov_for_tabu)
                if len(tabu_sets) > 80:
                    old = tabu_sets.popleft()
                    # keep lookup small: remove if not duplicated
                    try:
                        tabu_set_lookup.remove(old)
                    except KeyError:
                        pass
        if q0 < best_score - 1e-12:
            best_X, best_score = Xnew, q0
            cur_X, cur_score = Xnew, q0
            if fp is not None:
                archive_fp.add(fp)
        else:
            # Metropolis wander; penalise recently visited fingerprints
            eff = q0 + (0.00015 if is_tabu_fp else 0.0)
            if eff < cur_score or rng.random() < np.exp((cur_score - eff) / max(Tmet, 1e-9)):
                cur_X, cur_score = Xnew, q0
                if fp is not None:
                    archive_fp.add(fp)
            else:
                # occasionally force exploration from best with fresh subset
                if rng.random() < 0.05:
                    cur_X, cur_score = best_X, best_score
        # opportunistic tightening of best
        if rng.random() < 0.10 and time.time() < HOP_END - 4:
            try:
                Xf, sf = _full_quench_track(best_X, 350, 0.002, 0.00015, 0.02)
                if sf < best_score:
                    best_X, best_score = Xf, sf
                    cur_X, cur_score = Xf, sf
                    try:
                        archive_fp.add(_fingerprint(best_X))
                    except Exception:
                        pass
            except Exception:
                pass
    # final polish
    try:
        Xb, sb = _full_quench_track(best_X, 900, 0.004, 0.00012, 0.025)
        if sb < best_score:
            best_X, best_score = Xb, sb
    except Exception:
        pass
    try:
        best_X = _lp_polish_wide(best_X, min(9.0, max(2.0, HARD_END - time.time() - 0.5)), deadline=HARD_END - 0.3, r0=1.5e-3)
        best_score = _max_cos(best_X)
    except Exception:
        pass
    try:
        Xb2, sb2 = _full_quench_track(best_X, 500, 0.001, 0.0001, 0.015)
        if sb2 < best_score:
            best_X, best_score = Xb2, sb2
    except Exception:
        pass
    best_X = _norm_rows(best_X)
    # tiny shrink guard (no-op, keep unit rows)
    return best_X
