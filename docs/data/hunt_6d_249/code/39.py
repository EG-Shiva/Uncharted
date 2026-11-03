import time
import numpy as np
from scipy.optimize import minimize

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def solve():
    t0 = time.time()
    T_DIVERSE_END = t0 + 148.0
    T_END = t0 + 177.0
    rng = np.random.default_rng(0)
    np.random.seed(0)
    n, d = 249, 6
    I0, J0 = np.triu_indices(n, 1)
    P = len(I0)

    cands = []
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=float)
            if a.shape == (n, d):
                nr = np.linalg.norm(a, axis=1, keepdims=True)
                nr[nr == 0] = 1.0
                a = a / nr
                if np.all(np.isfinite(a)):
                    cands.append(a)
        except Exception:
            pass
    if not cands:
        a = rng.standard_normal(size=(n, d))
        a /= np.linalg.norm(a, axis=1, keepdims=True)
        cands.append(a)
    vals = [_max_cos(a) for a in cands]
    bestX = cands[int(np.argmin(vals))].copy()
    bestVal = float(np.min(vals))

    def full_anneal(Xinit, taus, iters, creg=0.06):
        Xf = (Xinit / np.maximum(np.linalg.norm(Xinit, axis=1, keepdims=True), 1e-12)).ravel().copy()
        for tau, mi in zip(taus, iters):
            if time.time() > T_END - 2.0:
                break
            tau = float(tau)
            def fg(xf, tau=tau):
                X = xf.reshape(n, d)
                r = np.sqrt(np.sum(X * X, axis=1))
                r = np.maximum(r, 1e-12)
                Xn = X / r[:, None]
                G = Xn @ Xn.T
                g = G[I0, J0]
                gm = np.max(g)
                e = np.exp((g - gm) / tau)
                S = np.sum(e)
                Lsoft = gm + tau * np.log(S + 1e-300)
                w = e / (S + 1e-300)
                W = np.zeros((n, n), dtype=float)
                W[I0, J0] = w
                W = W + W.T
                A = W @ Xn
                proj = np.sum(A * Xn, axis=1)
                G0 = (A - proj[:, None] * Xn) / r[:, None]
                diff = r - 1.0
                L = Lsoft + creg * float(np.sum(diff * diff))
                Gr = G0 + (2.0 * creg * diff)[:, None] * (X / r[:, None])
                return float(L), Gr.ravel()
            try:
                res = minimize(fg, Xf, jac=True, method='L-BFGS-B',
                               options={'maxiter': int(mi), 'maxfun': int(mi * 5),
                                        'ftol': 1e-14, 'gtol': 1e-7, 'maxcor': 7})
                Xf = res.x
            except Exception:
                break
            Xm = Xf.reshape(n, d)
            rn = np.maximum(np.linalg.norm(Xm, axis=1, keepdims=True), 1e-12)
            Xf = (Xm / rn).ravel()
        Xo = Xf.reshape(n, d)
        Xo /= np.maximum(np.linalg.norm(Xo, axis=1, keepdims=True), 1e-12)
        return Xo

    def greedy_refill(Xkeep, m, Cnum=35000):
        cand = rng.standard_normal(size=(Cnum, d))
        cand /= np.maximum(np.linalg.norm(cand, axis=1, keepdims=True), 1e-12)
        D = cand @ Xkeep.T
        curmax = np.max(D, axis=1)
        used = np.zeros(Cnum, dtype=bool)
        new_pts = []
        for _ in range(m):
            curmax[used] = np.inf
            bi = int(np.argmin(curmax))
            used[bi] = True
            pt = cand[bi].copy()
            new_pts.append(pt)
            dd = cand @ pt
            np.maximum(curmax, dd, out=curmax)
            curmax[used] = np.inf
            if time.time() > T_DIVERSE_END:
                # fill rest randomly from remaining pool greedily still fast
                pass
        Ppts = np.stack(new_pts, axis=0)
        Ppts /= np.maximum(np.linalg.norm(Ppts, axis=1, keepdims=True), 1e-12)
        return Ppts

    def top_codes(X, K=1500):
        G = X @ X.T
        g = G[I0, J0]
        if K >= P:
            idx = np.arange(P)
        else:
            idx = np.argpartition(g, -K)[-K:]
        codes = I0[idx].astype(np.int64) * n + J0[idx].astype(np.int64)
        return set(codes.tolist())

    base_set = top_codes(bestX, 1500)
    archive = [(bestVal, bestX.copy())]

    def jacc_to_base(X):
        s = top_codes(X, 1500)
        inter = len(s.intersection(base_set))
        union = len(s.union(base_set))
        if union == 0:
            return 1.0
        return inter / union

    # --- diverse seeds: antipodal-symmetric + pure random, coarsely jammed ---
    try:
        if time.time() < T_DIVERSE_END:
            k = 124
            A = rng.standard_normal(size=(k, d))
            A /= np.linalg.norm(A, axis=1, keepdims=True)
            one = rng.standard_normal(size=(1, d))
            one /= np.linalg.norm(one)
            Xap = np.vstack([A, -A, one])
            # shuffle order deterministically
            perm = rng.permutation(n)
            Xap = Xap[perm]
            Xa1 = full_anneal(Xap, [0.02, 0.008, 0.003], [40, 40, 55])
            va1 = _max_cos(Xa1)
            if va1 < bestVal - 1e-12:
                bestVal = va1; bestX = Xa1.copy()
                base_set = top_codes(bestX, 1500)
            archive.append((va1, Xa1.copy()))
    except Exception:
        pass
    try:
        if time.time() < T_DIVERSE_END:
            Xr = rng.standard_normal(size=(n, d))
            Xr /= np.linalg.norm(Xr, axis=1, keepdims=True)
            Xr1 = full_anneal(Xr, [0.02, 0.008, 0.003], [40, 40, 55])
            vr1 = _max_cos(Xr1)
            if vr1 < bestVal - 1e-12:
                bestVal = vr1; bestX = Xr1.copy()
                base_set = top_codes(bestX, 1500)
            archive.append((vr1, Xr1.copy()))
            archive = sorted(archive, key=lambda t: t[0])[:6]
    except Exception:
        pass

    taus_short = [0.015, 0.006, 0.002, 0.0008]
    iters_short = [40, 45, 55, 60]

    it = 0
    while time.time() < T_DIVERSE_END:
        it += 1
        # source: mostly best, sometimes diverse archive
        if len(archive) > 1 and rng.random() < 0.30:
            ai = int(rng.integers(0, len(archive)))
            Xsrc = archive[ai][1]
        else:
            Xsrc = bestX
        op = rng.random()
        try:
            if op < 0.35:
                # large spherical cap deletion -> slack moat
                m = int(rng.integers(45, 76))
                u = rng.standard_normal(size=(d,))
                u /= (np.linalg.norm(u) + 1e-12)
                dots = Xsrc @ u
                free = np.argsort(-dots)[:m]
                keep_mask = np.ones(n, dtype=bool); keep_mask[free] = False
                Xkeep = Xsrc[keep_mask]
                Pp = greedy_refill(Xkeep, m, Cnum=35000)
                Xt0 = Xsrc.copy(); Xt0[free] = Pp
                Xt = full_anneal(Xt0, taus_short, iters_short)
            elif op < 0.55:
                # global shake -> different basin
                sigma = float(rng.uniform(0.02, 0.075))
                Xp = Xsrc + sigma * rng.standard_normal(size=(n, d))
                Xp /= np.maximum(np.linalg.norm(Xp, axis=1, keepdims=True), 1e-12)
                Xt = full_anneal(Xp, taus_short, iters_short)
            elif op < 0.78:
                # equatorial-band symmetrization: organized lower-dim patch
                m = int(rng.integers(35, 61))
                u = rng.standard_normal(size=(d,))
                u /= (np.linalg.norm(u) + 1e-12)
                adots = np.abs(Xsrc @ u)
                free = np.argsort(adots)[:m]
                Y = Xsrc[free].copy()
                # project to equator orthogonal to u
                coeff = Y @ u
                Y = Y - coeff[:, None] * u[None, :]
                nr = np.linalg.norm(Y, axis=1, keepdims=True)
                nr = np.maximum(nr, 1e-12)
                # guard against collapse (point parallel to u): re-randomize those
                bad = (np.linalg.norm(Xsrc[free] - coeff[:, None] * u[None, :], axis=1) < 1e-6)
                if np.any(bad):
                    rr = rng.standard_normal(size=(int(np.sum(bad)), d))
                    rr = rr - (rr @ u)[:, None] * u[None, :]
                    rr /= np.maximum(np.linalg.norm(rr, axis=1, keepdims=True), 1e-12)
                    Y[bad] = rr
                    nr = np.maximum(np.linalg.norm(Y, axis=1, keepdims=True), 1e-12)
                Y /= nr
                # small tangential jitter inside equator to break degeneracy
                Y = Y + 0.005 * rng.standard_normal(size=Y.shape)
                Y = Y - (Y @ u)[:, None] * u[None, :]
                Y /= np.maximum(np.linalg.norm(Y, axis=1, keepdims=True), 1e-12)
                Xt0 = Xsrc.copy(); Xt0[free] = Y
                Xt = full_anneal(Xt0, taus_short, iters_short)
            else:
                # hub neighbourhood deletion (high-contact cluster)
                m = int(rng.integers(18, 33))
                Gc = Xsrc @ Xsrc.T
                np.fill_diagonal(Gc, -2.0)
                mc = float(np.max(Gc))
                deg = np.sum(Gc > (mc - 5e-4), axis=1)
                hub = int(np.argmax(deg))
                dots = Xsrc @ Xsrc[hub]
                free = np.argsort(-dots)[:m]
                keep_mask = np.ones(n, dtype=bool); keep_mask[free] = False
                Xkeep = Xsrc[keep_mask]
                Pp = greedy_refill(Xkeep, m, Cnum=35000)
                Xt0 = Xsrc.copy(); Xt0[free] = Pp
                Xt = full_anneal(Xt0, taus_short, iters_short)
        except Exception:
            continue
        try:
            vt = _max_cos(Xt)
        except Exception:
            continue
        if vt < bestVal - 1e-12:
            bestVal = vt; bestX = Xt.copy()
            archive.append((vt, Xt.copy()))
            archive = sorted(archive, key=lambda t: t[0])[:6]
            base_set = top_codes(bestX, 1500)
        else:
            try:
                jc = jacc_to_base(Xt)
            except Exception:
                jc = 1.0
            if jc < 0.72 and vt < bestVal + 0.004:
                archive.append((vt, Xt.copy()))
                archive = sorted(archive, key=lambda t: t[0])[:6]
            elif vt < bestVal + 0.0004 and rng.random() < 0.08:
                archive.append((vt, Xt.copy()))
                archive = sorted(archive, key=lambda t: t[0])[:6]

    # final polish on best (and runner-up if time)
    archive = sorted(archive, key=lambda t: t[0])[:4]
    # polish best first with tiny taus
    try:
        taus_f = [0.006, 0.002, 0.0007, 0.00025]
        iters_f = [70, 80, 90, 110]
        Xp = full_anneal(bestX, taus_f, iters_f, creg=0.03)
        vp = _max_cos(Xp)
        if vp < bestVal:
            bestVal = vp; bestX = Xp
    except Exception:
        pass
    # if time remains, try runner-up polish and take better
    try:
        if time.time() < T_END - 8.0 and len(archive) > 1:
            for _, Xa in list(archive[1:2]):
                if time.time() > T_END - 8.0:
                    break
                Xq = full_anneal(Xa, [0.004, 0.0012, 0.00035], [60, 70, 90], creg=0.03)
                vq = _max_cos(Xq)
                if vq < bestVal:
                    bestVal = vq; bestX = Xq.copy()
    except Exception:
        pass

    bestX /= np.maximum(np.linalg.norm(bestX, axis=1, keepdims=True), 1e-12)
    # tiny shrink guard (no-op, keeps unit norm)
    return bestX
