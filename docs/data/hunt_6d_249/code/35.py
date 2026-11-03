import time
import numpy as np
from scipy.optimize import minimize, linprog
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def solve():
    t0 = time.time()
    T_END = t0 + 172.0
    FINAL_RESERVE = 22.0
    rng = np.random.default_rng(12345)
    n, d = 249, 6
    I0, J0 = np.triu_indices(n, 1)
    P = len(I0)

    cands = []
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=float)
            if a.shape == (n, d):
                a = a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-12)
                if np.all(np.isfinite(a)):
                    cands.append(a)
        except Exception:
            pass
    if not cands:
        a = rng.standard_normal(size=(n, d))
        a /= np.linalg.norm(a, axis=1, keepdims=True)
        cands.append(a)
    vals = [_max_cos(a) for a in cands]
    bi = int(np.argmin(vals))
    bestX = cands[bi].copy()
    bestVal = float(vals[bi])
    archive = [(bestVal, bestX.copy())]

    def top_set(X, K=1500):
        g = (X @ X.T)[I0, J0]
        if K >= P:
            idx = np.arange(P)
        else:
            idx = np.argpartition(g, -K)[-K:]
        codes = I0[idx].astype(np.int64) * n + J0[idx].astype(np.int64)
        return set(codes.tolist())

    inc_set = top_set(bestX, 1500)

    def jacc(a, b):
        if not a and not b:
            return 1.0
        u = len(a | b)
        if u == 0:
            return 1.0
        return len(a & b) / u

    def anneal(Xinit, taus, iters, creg=0.08):
        xf = Xinit.ravel().copy()
        for tau, mi in zip(taus, iters):
            if time.time() > T_END - FINAL_RESERVE:
                break
            tau = float(tau)
            def fg(xflat, tau=tau):
                X = xflat.reshape(n, d)
                r = np.sqrt(np.sum(X * X, axis=1))
                r = np.maximum(r, 1e-12)
                Xn = X / r[:, None]
                G = Xn @ Xn.T
                g = G[I0, J0]
                gm = np.max(g)
                e = np.exp((g - gm) / tau)
                S = np.sum(e)
                Lsoft = gm + tau * np.log(S)
                w = e / S
                W = np.zeros((n, n))
                W[I0, J0] = w
                W = W + W.T
                A = W @ Xn
                proj = np.sum(A * Xn, axis=1)
                G0 = (A - proj[:, None] * Xn) / r[:, None]
                diff = r - 1.0
                L = Lsoft + creg * np.sum(diff * diff)
                Gr = G0 + (2.0 * creg * diff)[:, None] * Xn
                return float(L), Gr.ravel()
            try:
                res = minimize(fg, xf, jac=True, method='L-BFGS-B',
                               options={'maxiter': int(mi), 'maxfun': int(mi * 4),
                                        'ftol': 1e-14, 'gtol': 1e-7, 'maxcor': 8})
                xf = res.x
            except Exception:
                break
            try:
                Xm = xf.reshape(n, d)
                rn = np.maximum(np.linalg.norm(Xm, axis=1, keepdims=True), 1e-12)
                xf = (Xm / rn).ravel()
            except Exception:
                pass
        Xo = xf.reshape(n, d)
        Xo /= np.maximum(np.linalg.norm(Xo, axis=1, keepdims=True), 1e-12)
        return Xo

    def greedy_refill(Xbase, del_idx, pool):
        keep_mask = np.ones(n, dtype=bool)
        keep_mask[del_idx] = False
        Xkeep = Xbase[keep_mask]
        D = pool @ Xkeep.T
        curmax = D.max(axis=1)
        C = pool.shape[0]
        used = np.zeros(C, dtype=bool)
        new_pts = []
        for _ in range(len(del_idx)):
            curmax[used] = np.inf
            b = int(np.argmin(curmax))
            used[b] = True
            pt = pool[b].copy()
            new_pts.append(pt)
            dd = pool @ pt
            np.maximum(curmax, dd, out=curmax)
            curmax[used] = np.inf
        new_pts = np.stack(new_pts, axis=0)
        # 1 hole-polish pass on worst new point
        try:
            allX = np.vstack([Xkeep, new_pts])
            worst = -1; worstv = -2
            base_nk = n - len(del_idx)
            for s in range(len(new_pts)):
                dots = allX @ new_pts[s]
                dots[base_nk + s] = -2
                mv = float(dots.max())
                if mv > worstv:
                    worstv = mv; worst = s
            others = np.delete(allX, base_nk + worst, axis=0)
            v0 = new_pts[worst].copy()
            tau_h = 0.004
            def fg1(yf):
                y = yf.reshape(6)
                rn = max(float(np.sqrt(np.sum(y*y))), 1e-12)
                yn = y / rn
                dots = others @ yn
                gm = float(dots.max())
                e = np.exp((dots - gm) / tau_h)
                S = float(e.sum())
                L = gm + tau_h * np.log(S) + 0.5 * (rn - 1.0) ** 2
                w = e / S
                A = w @ others
                proj = float(A @ yn)
                G0 = (A - proj * yn) / rn
                Gr = G0 + (rn - 1.0) * yn
                return float(L), Gr.reshape(6)
            res = minimize(fg1, v0.copy(), jac=True, method='L-BFGS-B',
                           options={'maxiter': 50, 'maxfun': 200, 'ftol': 1e-14, 'gtol': 1e-8})
            vo = res.x.reshape(6)
            vo /= max(float(np.linalg.norm(vo)), 1e-12)
            if float(np.max(others @ vo)) < float(np.max(others @ v0)):
                new_pts[worst] = vo
        except Exception:
            pass
        Xt = Xbase.copy()
        Xt[del_idx] = new_pts
        Xt /= np.maximum(np.linalg.norm(Xt, axis=1, keepdims=True), 1e-12)
        return Xt

    def random_twist_matrix(theta):
        # rotation in random 2-plane spanned by orthonormal u,v
        a = rng.standard_normal(6); b = rng.standard_normal(6)
        a /= max(np.linalg.norm(a), 1e-12)
        b = b - float(b @ a) * a
        bn = np.linalg.norm(b)
        if bn < 1e-9:
            b = rng.standard_normal(6); b /= max(np.linalg.norm(b), 1e-12)
        else:
            b /= bn
        u = a; v = b
        c = np.cos(theta); s = np.sin(theta)
        Q = np.eye(6) + (c - 1.0) * (np.outer(u, u) + np.outer(v, v)) + s * (np.outer(v, u) - np.outer(u, v))
        return Q

    taus_coarse = [0.02, 0.01, 0.005]
    iters_coarse = [25, 25, 35]
    taus_fast = [0.01, 0.005, 0.002, 0.001]
    iters_fast = [30, 35, 45, 55]
    taus_fine = [0.005, 0.002, 0.001, 0.0005, 0.00025]
    iters_fine = [60, 80, 100, 100, 80]

    # Phase 0: diverse random basins (2 quick ones) to avoid single-basin collapse
    for _ in range(2):
        if time.time() > T_END - FINAL_RESERVE - 100:
            break
        Xr = rng.standard_normal(size=(n, d))
        Xr /= np.linalg.norm(Xr, axis=1, keepdims=True)
        try:
            Xp = anneal(Xr, taus_coarse, iters_coarse)
            vp = _max_cos(Xp)
            archive.append((float(vp), Xp.copy()))
            archive = sorted(archive, key=lambda t: t[0])[:8]
            if vp < bestVal:
                bestVal = float(vp); bestX = Xp.copy()
                inc_set = top_set(bestX, 1500)
        except Exception:
            pass

    ncyc = 0
    while time.time() < T_END - FINAL_RESERVE:
        ncyc += 1
        rpick = rng.random()
        if len(archive) > 1 and rpick < 0.30:
            ai = int(rng.integers(len(archive)))
            Xbase = archive[ai][1]
        else:
            Xbase = bestX
        # choose operation
        op = rng.random()
        Gbase = Xbase @ Xbase.T
        np.fill_diagonal(Gbase, -2.0)
        mbase = float(Gbase.max())
        if op < 0.42:
            # large spherical cap deletion
            k = int(rng.integers(40, 81))
            direction = rng.standard_normal(6)
            direction /= max(np.linalg.norm(direction), 1e-12)
            # sometimes center on dense hub to force topology change
            if rng.random() < 0.4:
                thr = mbase - 5e-4
                deg = (Gbase > thr).sum(axis=1)
                hub = int(np.argmax(deg))
                direction = Xbase[hub] + 0.3 * rng.standard_normal(6)
                direction /= max(np.linalg.norm(direction), 1e-12)
            dots = Xbase @ direction + 1e-12 * rng.standard_normal(n)
            del_idx = np.argsort(dots)[-k:]
            # refill path
            pool = rng.standard_normal(size=(35000, 6))
            pool /= np.maximum(np.linalg.norm(pool, axis=1, keepdims=True), 1e-12)
            try:
                Xt = greedy_refill(Xbase, del_idx, pool)
            except Exception:
                continue
            try:
                Xj = anneal(Xt, taus_fast, iters_fast)
            except Exception:
                continue
        elif op < 0.62:
            # equatorial belt deletion (floppy ring)
            k = int(rng.integers(30, 61))
            direction = rng.standard_normal(6)
            direction /= max(np.linalg.norm(direction), 1e-12)
            dots = np.abs(Xbase @ direction) + 1e-12 * rng.standard_normal(n)
            del_idx = np.argsort(dots)[:k]
            pool = rng.standard_normal(size=(35000, 6))
            pool /= np.maximum(np.linalg.norm(pool, axis=1, keepdims=True), 1e-12)
            try:
                Xt = greedy_refill(Xbase, del_idx, pool)
            except Exception:
                continue
            try:
                Xj = anneal(Xt, taus_fast, iters_fast)
            except Exception:
                continue
        elif op < 0.82:
            # rigid twist of a cap (no deletion): creates shear at boundary
            k = int(rng.integers(30, 71))
            direction = rng.standard_normal(6)
            direction /= max(np.linalg.norm(direction), 1e-12)
            dots = Xbase @ direction
            cap = np.argsort(dots)[-k:]
            theta = float(rng.uniform(0.12, 0.40)) * (1.0 if rng.random() < 0.5 else -1.0)
            Q = random_twist_matrix(theta)
            Xt = Xbase.copy()
            Xt[cap] = (Q @ Xt[cap].T).T
            Xt /= np.maximum(np.linalg.norm(Xt, axis=1, keepdims=True), 1e-12)
            # small jitter inside cap to break symmetry
            jit = 0.02 * rng.standard_normal(size=(k, 6))
            Xt[cap] = Xt[cap] + jit
            Xt[cap] /= np.maximum(np.linalg.norm(Xt[cap], axis=1, keepdims=True), 1e-12)
            try:
                Xj = anneal(Xt, taus_fast, iters_fast)
            except Exception:
                continue
        else:
            # fresh random restart coarse (new basin)
            Xr = rng.standard_normal(size=(n, d))
            Xr /= np.linalg.norm(Xr, axis=1, keepdims=True)
            try:
                Xj = anneal(Xr, taus_coarse, iters_coarse)
            except Exception:
                continue
        vj = _max_cos(Xj)
        try:
            sj = top_set(Xj, 1500)
            js = jacc(sj, inc_set)
        except Exception:
            js = 1.0
        if vj < bestVal:
            bestVal = float(vj); bestX = Xj.copy()
            inc_set = top_set(bestX, 1500)
            archive.append((bestVal, bestX.copy()))
            archive = sorted(archive, key=lambda t: t[0])[:8]
        elif vj < bestVal + 3e-4 and js < 0.72:
            archive.append((float(vj), Xj.copy()))
            archive = sorted(archive, key=lambda t: t[0])[:8]
        elif vj < bestVal + 0.006 and js < 0.5:
            archive.append((float(vj), Xj.copy()))
            archive = sorted(archive, key=lambda t: t[0])[:8]

    # Final jamming: fine anneal top candidates then trust-region LP
    archive = sorted(archive, key=lambda t: t[0])[:3]
    cand_final = []
    for (vv, XX) in archive:
        if time.time() > T_END - FINAL_RESERVE + 6:
            break
        try:
            Xf = anneal(XX, taus_fine, iters_fine, creg=0.05)
            vf = _max_cos(Xf)
            cand_final.append((vf, Xf))
            if vf < bestVal:
                bestVal = vf; bestX = Xf.copy()
        except Exception:
            pass
    # trust-region LP jamming on best
    def lp_jam(Xinit, seconds):
        X = Xinit.copy()
        X /= np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
        cur = _max_cos(X)
        Nvars = n * d
        # objective: minimize t -> c = [0..0,1]
        c = np.zeros(Nvars + 1); c[-1] = 1.0
        radius = 2e-3
        start = time.time()
        iu = (I0, J0)
        while time.time() - start < seconds and radius > 1e-13:
            if time.time() > T_END - 2.0:
                break
            Gf = X @ X.T
            g = Gf[I0, J0]
            m = float(g.max())
            window = max(20.0 * radius, 3e-7)
            sel = g > m - window
            cnt = int(np.sum(sel))
            if cnt > 4000:
                idx = np.argpartition(g, -4000)[-4000:]
                sel = np.zeros_like(sel, dtype=bool)
                sel[idx] = True
                cnt = 4000
            if cnt < 10:
                radius *= 0.5
                continue
            si = I0[sel]; sj = J0[sel]
            rows = []; cols = []; data = []
            b = np.empty(cnt)
            for r_ in range(cnt):
                i = int(si[r_]); j = int(sj[r_])
                dot = float(X[i] @ X[j])
                # xj block at i, xi block at j, -1 at t
                for k_ in range(d):
                    rows.append(r_); cols.append(i * d + k_); data.append(float(X[j, k_]))
                    rows.append(r_); cols.append(j * d + k_); data.append(float(X[i, k_]))
                rows.append(r_); cols.append(Nvars); data.append(-1.0)
                b[r_] = -dot
            A_ub = coo_matrix((data, (rows, cols)), shape=(cnt, Nvars + 1)).tocsr()
            # bounds: dx in [-radius, radius], t in [m-0.01, m]
            bounds = [(-radius, radius)] * Nvars + [(m - 0.01, m)]
            # equality: Xi . dxi = 0
            A_eq = np.zeros((n, Nvars + 1))
            for i in range(n):
                A_eq[i, i * d:(i + 1) * d] = X[i]
            b_eq = np.zeros(n)
            try:
                res = linprog(c, A_ub=A_ub, b_ub=b, A_eq=A_eq, b_eq=b_eq,
                              bounds=bounds, method='highs',
                              options={'time_limit': max(1.0, seconds - (time.time() - start))})
            except Exception:
                radius *= 0.5
                continue
            if res.status != 0 or res.x is None:
                radius *= 0.5
                continue
            dx = res.x[:Nvars].reshape(n, d)
            t_pred = float(res.x[Nvars])
            # trust region step
            Xn = X + dx
            Xn /= np.maximum(np.linalg.norm(Xn, axis=1, keepdims=True), 1e-12)
            vnew = _max_cos(Xn)
            # accept if improves or predicted improvement
            if vnew < cur:
                X = Xn; cur = vnew
                # grow slightly if good
                if t_pred < m - 1e-12:
                    radius = min(radius * 1.4, 3e-3)
                else:
                    radius *= 0.9
            else:
                radius *= 0.5
            if radius < 1e-9 and vnew >= cur:
                # tiny final steps
                pass
        return X, cur

    try:
        secs_left = max(4.0, T_END - time.time() - 1.0)
        secs_left = min(secs_left, 18.0)
        Xlp, vlp = lp_jam(bestX, secs_left)
        if vlp < bestVal and np.all(np.isfinite(Xlp)):
            bestVal = vlp; bestX = Xlp
    except Exception:
        pass
    bestX /= np.maximum(np.linalg.norm(bestX, axis=1, keepdims=True), 1e-12)
    # tiny shrink guard (no-op, ensures strict unit)
    return bestX
