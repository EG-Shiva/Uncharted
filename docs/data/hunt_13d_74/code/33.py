import time
import numpy as np
from scipy.optimize import minimize

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _normalize(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return X / n

def solve():
    t0 = time.time()
    LIM = 172.0
    rng = np.random.default_rng(12345)
    n, d = 74, 13
    Ppairs = n * (n - 1) // 2
    k_t = n / d
    norm1 = k_t * k_t * d  # = n^2/d

    cands = []
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=np.float64)
            if a.shape == (n, d):
                a = _normalize(a)
                cands.append(a)
        except Exception:
            pass
    if not cands:
        cands.append(_normalize(rng.standard_normal((n, d))))
    vals = [_max_cos(a) for a in cands]
    best = cands[int(np.argmin(vals))].copy()
    best_val = float(min(vals))

    IU = np.triu_indices(n, 1)
    I0, J0 = IU[0].astype(np.int64), IU[1].astype(np.int64)

    def lbfgs_polish(X0, tau, maxiter):
        if time.time() - t0 > LIM - 3:
            return X0
        y0 = X0.ravel().copy()
        def fun(yf):
            Y = yf.reshape(n, d)
            r2 = (Y * Y).sum(axis=1, keepdims=True)
            r = np.sqrt(np.maximum(r2, 1e-24))
            r = np.maximum(r, 1e-12)
            X = Y / r
            G = X @ X.T
            g = G[I0, J0]
            mmax = float(g.max())
            e = np.exp((g - mmax) / tau)
            S = float(e.sum())
            if not np.isfinite(S) or S <= 0:
                return float(mmax), np.zeros_like(yf)
            F = mmax + tau * np.log(S + 1e-300)
            w = e / (S + 1e-300)
            Wx = np.zeros_like(X)
            Wx[I0] += w[:, None] * X[J0]
            Wx[J0] += w[:, None] * X[I0]
            dot = (Wx * X).sum(axis=1, keepdims=True)
            gradY = (Wx - dot * X) / r
            return float(F), gradY.ravel()
        try:
            res = minimize(fun, y0, jac=True, method="L-BFGS-B",
                           options={"maxiter": int(maxiter), "ftol": 1e-20,
                                    "gtol": 1e-8, "maxfun": int(maxiter * 6)})
            Y = res.x.reshape(n, d)
        except Exception:
            return X0
        rn = np.linalg.norm(Y, axis=1, keepdims=True)
        rn[rn == 0] = 1.0
        X = Y / rn
        if not np.all(np.isfinite(X)):
            return X0
        return _normalize(X)

    def adam_iso(X0, s_pow, steps, lr, lam1_0, lam2_0):
        X = _normalize(X0.copy())
        m = np.zeros_like(X)
        v = np.zeros_like(X)
        b1, b2, eps = 0.9, 0.999, 1e-8
        idx = np.arange(n)
        for t in range(1, int(steps) + 1):
            G = X @ X.T
            D2 = 2.0 - 2.0 * G
            D2[idx, idx] = 2.0
            D2c = np.maximum(D2, 1e-3)
            W = s_pow * np.power(D2c, -s_pow / 2.0 - 1.0)
            W[idx, idx] = 0.0
            np.clip(W, 0.0, 1e4, out=W)
            gradR = (W @ X) / float(Ppairs)
            M = X.T @ X
            Diff = M - k_t * np.eye(d)
            gradP1 = (4.0 * (X @ Diff)) / norm1
            ssum = X.sum(axis=0, keepdims=True)  # 1xd
            mean = ssum / float(n)
            gradP2 = np.repeat((2.0 * mean / float(n)), n, axis=0)
            frac = 1.0 - float(t) / float(steps + 1)
            lam1 = lam1_0 * frac
            lam2 = lam2_0 * frac
            grad = gradR + lam1 * gradP1 + lam2 * gradP2
            # tangent project
            dot = (grad * X).sum(axis=1, keepdims=True)
            gt = grad - dot * X
            m = b1 * m + (1.0 - b1) * gt
            v = b2 * v + (1.0 - b2) * (gt * gt)
            mh = m / (1.0 - b1 ** t)
            vh = v / (1.0 - b2 ** t)
            step = lr * mh / (np.sqrt(vh) + eps)
            # mild clr decay
            X = X - step
            X = _normalize(X)
            if not np.all(np.isfinite(X)):
                X = _normalize(rng.standard_normal((n, d)))
                m[:] = 0; v[:] = 0
        return _normalize(X)

    def greedy_select(P):
        # P: (148,d) normalized, Procrustes aligned pool
        m2 = P.shape[0]
        sel = [0]
        selmask = np.zeros(m2, dtype=bool)
        selmask[0] = True
        scores = P @ P[0]
        scores[0] = np.inf
        for _ in range(1, n):
            j = int(np.argmin(scores))
            sel.append(j)
            selmask[j] = True
            scores[j] = np.inf
            # update
            dj = P @ P[j]
            # keep inf for selected
            np.maximum(scores, dj, out=dj)
            # restore inf
            dj[selmask] = np.inf
            scores = dj
        return np.array(sel, dtype=np.int64)

    def hole_replace(Xc):
        # one largest-hole repair: replace worst point if better hole exists
        G = Xc @ Xc.T
        np.fill_diagonal(G, -2.0)
        rowmax = G.max(axis=1)
        iw = int(np.argmax(rowmax))
        curmax = float(rowmax[iw])
        # candidate holes
        Nc = 3000
        C = _normalize(rng.standard_normal((Nc, d)))
        S = C @ Xc.T
        S[:, iw] = -10.0
        sc = S.max(axis=1)
        j = int(np.argmin(sc))
        if float(sc[j]) < curmax - 1e-12:
            Xc = Xc.copy()
            Xc[iw] = C[j]
        return _normalize(Xc)

    # ---- Phase A: diverse isotropic population ----
    s_list = [1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 2.5, 5.0]
    pop = []
    pop_val = []
    n_pop = 8
    for kk in range(n_pop):
        if time.time() - t0 > 55:
            break
        X0 = _normalize(rng.standard_normal((n, d)))
        lam1_0 = 1.5 + 0.4 * (kk % 4)
        lam2_0 = 1.5 + 0.5 * (kk % 3)
        Xa = adam_iso(X0, s_list[kk % len(s_list)], steps=550, lr=0.012,
                      lam1_0=lam1_0, lam2_0=lam2_0)
        # brief LSE polish
        Xp = lbfgs_polish(Xa, 0.002, 60)
        v = _max_cos(Xp)
        pop.append(Xp)
        pop_val.append(v)
        if v < best_val:
            best_val = float(v)
            best = Xp.copy()

    # diversity filter via Gram spectrum (eig of X^T X)
    if len(pop) > 0:
        order = np.argsort(np.array(pop_val))
        pop = [pop[i] for i in order]
        pop_val = [pop_val[i] for i in order]
        kept = []
        kept_val = []
        eigs = []
        for Xp, vv in zip(pop, pop_val):
            M = Xp.T @ Xp
            ev = np.linalg.eigvalsh(M)
            ev = np.sort(ev)
            ok = True
            for e2 in eigs:
                if float(np.linalg.norm(ev - e2)) < 0.03:
                    ok = False
                    break
            if ok or len(kept) < 3:
                kept.append(Xp); kept_val.append(vv); eigs.append(ev)
                if len(kept) >= 6:
                    break
            if len(kept) >= 6:
                break
        # if too few, fill with best
        if len(kept) < 3 and len(pop) >= 3:
            kept = pop[:min(6, len(pop))]
            kept_val = pop_val[:min(6, len(pop))]
        parents = kept
    else:
        parents = [best.copy()]

    # ---- Phase B: Procrustes crossover ----
    npar = len(parents)
    # all pairs, prioritize better parents
    pairs = []
    for i in range(npar):
        for j in range(i + 1, npar):
            pairs.append((i, j))
    # sort pairs by sum val
    pairs = sorted(pairs, key=lambda pr: kept_val[pr[0]] + kept_val[pr[1]] if len(kept_val) > pr[1] else 0)
    # limit count by time
    max_cross = 12
    pairs = pairs[:max_cross]
    children = []
    for (ia, ib) in pairs:
        if time.time() - t0 > LIM - 45:
            break
        Xa = parents[ia]; Xb = parents[ib]
        Cmat = Xb.T @ Xa
        try:
            U, _, Vt = np.linalg.svd(Cmat)
            R = U @ Vt
        except Exception:
            R = np.eye(d)
        Xb_al = Xb @ R
        Pool = np.vstack([Xa, Xb_al])
        Pool = _normalize(Pool)
        sel = greedy_select(Pool)
        Xc = Pool[sel]
        Xc = hole_replace(Xc)
        # quick polish
        Xc = lbfgs_polish(Xc, 5e-4, 60)
        vc = _max_cos(Xc)
        children.append((vc, Xc))
        if vc < best_val:
            best_val = float(vc)
            best = Xc.copy()
    # also try crossover of best with parents for extra topology
    if time.time() - t0 < LIM - 50 and len(parents) > 0:
        for ii in range(min(3, len(parents))):
            if time.time() - t0 > LIM - 45:
                break
            Xa = best.copy(); Xb = parents[ii]
            Cmat = Xb.T @ Xa
            try:
                U, _, Vt = np.linalg.svd(Cmat)
                R = U @ Vt
            except Exception:
                R = np.eye(d)
            Xb_al = Xb @ R
            Pool = np.vstack([Xa, Xb_al])
            Pool = _normalize(Pool)
            sel = greedy_select(Pool)
            Xc = Pool[sel]
            Xc = hole_replace(Xc)
            Xc = lbfgs_polish(Xc, 5e-4, 60)
            vc = _max_cos(Xc)
            if vc < best_val:
                best_val = float(vc)
                best = Xc.copy()

    # ---- Phase C: annealed deep polish of overall best ----
    sched = [(2e-3, 70), (5e-4, 90), (1.2e-4, 110), (3e-5, 110)]
    Xcur = best.copy()
    for tau, mi in sched:
        if time.time() - t0 > LIM - 4:
            break
        Xcur = lbfgs_polish(Xcur, tau, mi)
        v = _max_cos(Xcur)
        if v < best_val:
            best_val = float(v)
            best = Xcur.copy()
    # final tiny greedy descent if time remains
    if time.time() - t0 < LIM - 2:
        # simple active-set line search
        cur = _max_cos(best)
        lr = 1.5e-5
        X = best.copy()
        it = 0
        while time.time() - t0 < LIM - 1.0 and it < 300:
            it += 1
            G = X @ X.T
            np.fill_diagonal(G, -2.0)
            gmax = float(G.max())
            delta = max(3.0 * lr, 5e-7)
            mask = G > (gmax - delta)
            A = mask.astype(np.float64)
            Gr = A @ X
            dd = (Gr * X).sum(axis=1, keepdims=True)
            Gt = Gr - dd * X
            nm = np.linalg.norm(Gt, axis=1, keepdims=True)
            act = nm[:, 0] > 1e-14
            if not np.any(act):
                break
            step = np.zeros_like(Gt)
            step[act] = Gt[act] / (nm[act] + 1e-12) * lr
            Xn = _normalize(X - step)
            nv = _max_cos(Xn)
            if nv < cur:
                X[:] = Xn; cur = nv; lr *= 1.08
                if cur < best_val:
                    best_val = float(cur); best = X.copy()
            else:
                lr *= 0.5
                if lr < 1e-11:
                    break
    best = _normalize(best)
    return best
