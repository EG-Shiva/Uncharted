import time
import numpy as np

def _norm_rows(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[~np.isfinite(n)] = 1.0
    n[n == 0] = 1.0
    return X / n

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _refine_hole(s, R, steps=260, T=0.004, lr=0.12, mu=0.85):
    s = np.array(s, dtype=float).copy()
    sn = float(np.linalg.norm(s))
    if not np.isfinite(sn) or sn == 0:
        s = R[0] * 0 + 1.0
        sn = float(np.linalg.norm(s))
    s = s / sn
    v = np.zeros_like(s)
    for _ in range(steps):
        g = R @ s
        m = float(g.max())
        e = np.exp(np.clip((g - m) / T, -50, 0))
        den = float(e.sum()) + 1e-30
        Grad = (R.T @ e) / den
        Gt = Grad - float(Grad @ s) * s
        gn = float(np.linalg.norm(Gt))
        if gn > 2.0:
            Gt *= (2.0 / gn)
        v = mu * v + Gt
        s = s - lr * v
        sn = float(np.linalg.norm(s))
        if not np.isfinite(sn) or sn < 1e-12:
            break
        s = s / sn
    sn = float(np.linalg.norm(s))
    if not np.isfinite(sn) or sn == 0:
        return s
    return s / sn

def _find_best_hole(R, rng, n_uniform=14000, n_combo=2000, n_refine=8):
    n, d = R.shape
    # uniform pool
    Z = rng.standard_normal(size=(n_uniform, d))
    Z = _norm_rows(Z)
    # antipodes + negative combos
    C = []
    # add exact antipodes
    C.append(-R)
    # negative combos of 2-3
    if n_combo > 0:
        Cc = np.empty((n_combo, d))
        for k in range(n_combo):
            sz = 2 + int(rng.integers(0, 2))  # 2 or 3
            idx = rng.choice(n, size=sz, replace=False)
            w = rng.standard_normal(size=sz)
            s = (w[:, None] * R[idx]).sum(axis=0)
            nn = float(np.linalg.norm(s))
            if nn < 1e-12 or not np.isfinite(nn):
                s = -R[int(rng.integers(0, n))]
                nn = float(np.linalg.norm(s))
            Cc[k] = -s / nn
        C.append(Cc)
    P = np.vstack([Z] + C)
    # scores
    S = P @ R.T
    scores = S.max(axis=1)
    k = min(n_refine, P.shape[0])
    idx = np.argpartition(scores, k - 1)[:k]
    # order by score
    o = np.argsort(scores[idx])
    idx = idx[o]
    best_s = None
    best_v = 1e100
    # refine each, two stages
    for ii in idx:
        s0 = P[ii]
        s1 = _refine_hole(s0, R, steps=220, T=0.006, lr=0.12, mu=0.85)
        s1 = _refine_hole(s1, R, steps=220, T=0.0015, lr=0.07, mu=0.8)
        v = float((R @ s1).max())
        if v < best_v:
            best_v = v
            best_s = s1
    # also consider unrefined best in case refine diverged
    raw_best = float(scores.min())
    if raw_best < best_v:
        # return raw
        j = int(np.argmin(scores))
        return P[j] / np.linalg.norm(P[j]), raw_best
    return best_s / np.linalg.norm(best_s), best_v

def _smooth_quench(X_init, steps=600, T0=0.005, T1=0.0005, lr=0.20, mu=0.88, deadline=None):
    X = _norm_rows(np.array(X_init, dtype=float, copy=True))
    n, d = X.shape
    V = np.zeros_like(X)
    G0 = X @ X.T
    np.fill_diagonal(G0, -2.0)
    best_m = float(G0.max())
    best_X = X.copy()
    lt0, lt1 = np.log(T0), np.log(T1)
    for s in range(steps):
        if deadline is not None and (s % 200 == 0) and time.time() > deadline:
            break
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
        if (s % 200 == 0) or (s == steps - 1):
            Gc = X @ X.T
            np.fill_diagonal(Gc, -2.0)
            vc = float(Gc.max())
            if vc < best_m:
                best_m = vc
                best_X = X.copy()
    return best_X

def solve():
    t0 = time.time()
    HARD_END = t0 + 172.0
    DEAD = HARD_END - 4.0
    rng = np.random.default_rng(12345)
    try:
        Xb = np.load("best.npy")
    except Exception:
        Xb = None
    try:
        Xr = np.load("record.npy")
    except Exception:
        Xr = None
    cands = []
    for A in (Xb, Xr):
        try:
            if A is None:
                continue
            A = np.asarray(A, dtype=np.float64)
            if A.shape == (71, 12):
                A = _norm_rows(A)
                if np.all(np.isfinite(A)):
                    cands.append(A)
        except Exception:
            pass
    if len(cands) == 0:
        X = _norm_rows(rng.standard_normal(size=(71, 12)))
        cur = _max_cos(X)
    else:
        vals = [_max_cos(A) for A in cands]
        X = cands[int(np.argmin(vals))].copy()
        cur = float(min(vals))
    bestX = X.copy()
    best = cur
    n, d = 71, 12

    # quick initial short quench to ensure local tightness (cheap)
    # main leave-k-out cycles
    outer = 0
    stall = 0
    while time.time() < DEAD:
        outer += 1
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        per_max = G.max(axis=1)
        cnt_near = (G > best - 1e-3).sum(axis=1) if np.isfinite(best) else (G > cur - 1e-3).sum(axis=1)
        # worst-first order (hubs), loosest last
        worst_order = np.argsort(-(per_max + 0.002 * cnt_near))
        loose_order = np.argsort(per_max + 0.002 * cnt_near)
        improved = False
        # ---- k=1 global reinsertion over worst 25 ----
        for ii in worst_order[:25]:
            if time.time() > DEAD - 2.0:
                break
            i = int(ii)
            keep = [k for k in range(n) if k != i]
            R = X[keep]
            GR = R @ R.T
            np.fill_diagonal(GR, -2.0)
            mR = float(GR.max())
            if mR >= cur - 1e-12:
                continue
            # adapt sample size: larger when close
            y, v = _find_best_hole(R, rng, n_uniform=12000, n_combo=2000, n_refine=7)
            new_max = mR if mR > v else v
            if new_max < cur - 1e-12:
                Xt = np.vstack([R, y])
                Xt = _smooth_quench(Xt, steps=500, T0=0.004, T1=0.0004, lr=0.18, deadline=DEAD)
                c2 = _max_cos(Xt)
                if c2 < cur - 1e-12:
                    X = Xt
                    cur = c2
                    if cur < best:
                        best = cur
                        bestX = X.copy()
                    improved = True
                    break
                else:
                    # accept raw without quench if quench failed but raw improved
                    if new_max < cur - 1e-12:
                        X = _norm_rows(np.vstack([R, y]))
                        cur = _max_cos(X)
                        if cur < best:
                            best = cur
                            bestX = X.copy()
                        improved = True
                        break
        if improved:
            stall = 0
            continue
        if time.time() > DEAD - 2.0:
            break
        # ---- k=2 greedy sequential over top hubs ----
        top = worst_order[:12]
        pairs = []
        for a in range(len(top)):
            for b in range(a + 1, len(top)):
                pairs.append((int(top[a]), int(top[b])))
        # shuffle deterministically via rng
        perm = rng.permutation(len(pairs))
        tried = 0
        for pi in perm:
            if tried >= 14:
                break
            if time.time() > DEAD - 2.0:
                break
            i, j = pairs[int(pi)]
            tried += 1
            keep = [k for k in range(n) if k != i and k != j]
            R = X[keep]
            GR = R @ R.T
            np.fill_diagonal(GR, -2.0)
            mR = float(GR.max())
            if mR >= cur - 1e-12:
                continue
            y1, v1 = _find_best_hole(R, rng, n_uniform=6000, n_combo=1200, n_refine=5)
            if max(mR, v1) >= cur - 1e-12:
                continue
            R2 = np.vstack([R, y1])
            y2, v2 = _find_best_hole(R2, rng, n_uniform=6000, n_combo=1200, n_refine=5)
            new_max = max(mR, v1, v2)
            # v2 already includes y1.y2, v1 still valid
            if new_max < cur - 1e-12:
                Xt = np.vstack([R, y1, y2])
                Xt = _smooth_quench(Xt, steps=500, T0=0.004, T1=0.0004, lr=0.18, deadline=DEAD)
                c2 = _max_cos(Xt)
                if c2 < cur - 1e-12:
                    X = Xt
                    cur = c2
                    if cur < best:
                        best = cur
                        bestX = X.copy()
                    improved = True
                    break
        if improved:
            stall = 0
            continue
        # ---- k=3 a few trials ----
        if stall >= 1 and time.time() < DEAD - 15.0:
            # pick 6 random triples among worst 14
            top3 = worst_order[:14]
            for _ in range(6):
                if time.time() > DEAD - 2.0:
                    break
                idx = rng.choice(len(top3), size=3, replace=False)
                rem = set(int(top3[k]) for k in idx)
                keep = [k for k in range(n) if k not in rem]
                R = X[keep]
                GR = R @ R.T
                np.fill_diagonal(GR, -2.0)
                mR = float(GR.max())
                if mR >= cur - 1e-12:
                    continue
                y1, v1 = _find_best_hole(R, rng, n_uniform=5000, n_combo=1000, n_refine=4)
                if max(mR, v1) >= cur - 1e-12:
                    continue
                R2 = np.vstack([R, y1])
                y2, v2 = _find_best_hole(R2, rng, n_uniform=5000, n_combo=1000, n_refine=4)
                if max(mR, v1, v2) >= cur - 1e-12:
                    continue
                R3 = np.vstack([R2, y2])
                y3, v3 = _find_best_hole(R3, rng, n_uniform=5000, n_combo=1000, n_refine=4)
                new_max = max(mR, v1, v2, v3)
                if new_max < cur - 1e-12:
                    Xt = np.vstack([R, y1, y2, y3])
                    Xt = _smooth_quench(Xt, steps=500, T0=0.004, T1=0.0004, lr=0.18, deadline=DEAD)
                    c2 = _max_cos(Xt)
                    if c2 < cur - 1e-12:
                        X = Xt
                        cur = c2
                        if cur < best:
                            best = cur
                            bestX = X.copy()
                        improved = True
                        break
            if improved:
                stall = 0
                continue
        # ---- sideways rattler exploitation (loosest 10-20) ----
        # relocate loosest points to their own optimal holes even if global max unchanged
        moved = False
        for ii in loose_order[:15]:
            if time.time() > DEAD - 2.0:
                break
            i = int(ii)
            old_row_max = float(per_max[i])
            keep = [k for k in range(n) if k != i]
            R = X[keep]
            # only bother if point is loose
            if old_row_max > cur - 5e-4:
                continue
            y, v = _find_best_hole(R, rng, n_uniform=4000, n_combo=800, n_refine=4)
            if v < old_row_max - 1e-6:
                # sideways accept if global not worsened
                GR = R @ R.T
                np.fill_diagonal(GR, -2.0)
                mR = float(GR.max())
                new_max = mR if mR > v else v
                if new_max <= cur + 1e-12:
                    Xt = np.vstack([R, y])
                    # short quench without LP, keep best tracking
                    Xtq = _smooth_quench(Xt, steps=300, T0=0.002, T1=0.0003, lr=0.15, deadline=DEAD)
                    c2 = _max_cos(Xtq)
                    if c2 <= cur + 1e-12:
                        # accept sideways; prefer quenched if not worse
                        if c2 <= new_max + 1e-12:
                            X = Xtq
                            cur = c2
                        else:
                            X = _norm_rows(Xt)
                            cur = new_max
                        if cur < best - 1e-15:
                            best = cur
                            bestX = X.copy()
                            improved = True
                            break
                        moved = True
        if improved:
            stall = 0
            continue
        if moved:
            # recompute and continue searching for global drop
            stall += 1
            # light global quench to settle
            if time.time() < DEAD - 8.0:
                Xq = _smooth_quench(X, steps=400, T0=0.003, T1=0.0003, lr=0.17, deadline=DEAD)
                cq = _max_cos(Xq)
                if cq <= cur + 1e-12:
                    X = Xq
                    cur = cq
                    if cur < best:
                        best = cur
                        bestX = X.copy()
            continue
        stall += 1
        # diversification kick if long stall: perturb a few rattlers
        if stall >= 2:
            if time.time() > DEAD - 10.0:
                break
            # random kick of 4 loosest + quench, keep elite
            kick_idx = [int(k) for k in loose_order[:4]]
            Xk = X.copy()
            for k in kick_idx:
                noise = rng.standard_normal(size=d) * 0.35
                # tangent-ish perturb
                Xk[k] = Xk[k] + noise
            Xk = _norm_rows(Xk)
            Xk = _smooth_quench(Xk, steps=700, T0=0.006, T1=0.0005, lr=0.20, deadline=DEAD)
            ck = _max_cos(Xk)
            if ck < cur - 1e-12:
                X = Xk
                cur = ck
                if cur < best:
                    best = cur
                    bestX = X.copy()
                stall = 0
            elif ck <= cur + 2e-4 and rng.random() < 0.5:
                # sideways drift to new basin
                X = Xk
                cur = ck
            else:
                # restore best and add tiny noise to escape exact plateau
                X = bestX.copy()
                cur = best
                # tiny jitter all points
                X = X + rng.standard_normal(size=X.shape) * 1e-4
                X = _norm_rows(X)
                X = _smooth_quench(X, steps=300, T0=0.002, T1=0.0004, lr=0.15, deadline=DEAD)
                cur = _max_cos(X)
                if cur < best:
                    best = cur
                    bestX = X.copy()
                    stall = 0
            if outer > 30:
                break
        if outer > 60:
            break
    # final: return best with tiny shrink guard (no scaling needed, just normalize)
    bestX = _norm_rows(np.asarray(bestX, dtype=np.float64))
    # ensure finite
    bestX[~np.isfinite(bestX)] = 0.0
    # if any zero row, replace
    nr = np.linalg.norm(bestX, axis=1)
    for i in range(bestX.shape[0]):
        if nr[i] < 1e-12 or not np.isfinite(nr[i]):
            v = rng.standard_normal(size=12)
            bestX[i] = v / np.linalg.norm(v)
    bestX = _norm_rows(bestX)
    return bestX
