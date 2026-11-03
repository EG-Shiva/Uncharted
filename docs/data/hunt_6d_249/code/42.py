import time
import numpy as np
from scipy.optimize import minimize

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def solve():
    t0 = time.time()
    T_END = t0 + 172.0
    RESERVE = 22.0
    rng = np.random.default_rng(12345)
    np.random.seed(12345)
    n, d = 249, 6
    I0, J0 = np.triu_indices(n, 1)

    cands = []
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=float)
            if a.shape == (n, d):
                an = a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-12)
                if np.all(np.isfinite(an)):
                    cands.append(an)
        except Exception:
            pass
    if not cands:
        a = rng.standard_normal(size=(n, d))
        a /= np.linalg.norm(a, axis=1, keepdims=True)
        cands.append(a)
    vals = [_max_cos(a) for a in cands]
    bestX = cands[int(np.argmin(vals))].copy()
    bestVal = float(np.min(vals))
    cur = bestX.copy()
    curVal = bestVal

    def anneal(Xinit, taus, iters, creg=0.08):
        Xf = (Xinit / np.maximum(np.linalg.norm(Xinit, axis=1, keepdims=True), 1e-12)).ravel().copy()
        for tau, mi in zip(taus, iters):
            if time.time() > T_END - RESERVE:
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
                res = minimize(fg, Xf, jac=True, method='L-BFGS-B',
                               options={'maxiter': int(mi), 'maxfun': int(mi * 3),
                                        'ftol': 1e-14, 'gtol': 1e-7, 'maxcor': 8})
                Xf = res.x
            except Exception:
                break
            try:
                Xm = Xf.reshape(n, d)
                rn = np.maximum(np.linalg.norm(Xm, axis=1, keepdims=True), 1e-12)
                Xf = (Xm / rn).ravel()
            except Exception:
                break
        Xo = Xf.reshape(n, d)
        Xo /= np.maximum(np.linalg.norm(Xo, axis=1, keepdims=True), 1e-12)
        return Xo

    def subgrad_polish(Xinit, deadline, step0=1.5e-3):
        Xc = Xinit.copy()
        Xc /= np.linalg.norm(Xc, axis=1, keepdims=True)
        vc = _max_cos(Xc)
        step = step0
        while time.time() < deadline:
            if step < 1e-13:
                break
            G = Xc @ Xc.T
            np.fill_diagonal(G, -2.0)
            m = float(np.max(G))
            window = max(12.0 * step, 3e-5)
            mask = G > (m - window)
            F = mask.astype(float) @ Xc
            dot = np.sum(F * Xc, axis=1)
            Ft = F - dot[:, None] * Xc
            # normalize force scale
            fn = np.linalg.norm(Ft, axis=1, keepdims=True)
            # avoid huge steps
            trial = Xc - step * Ft
            trial /= np.maximum(np.linalg.norm(trial, axis=1, keepdims=True), 1e-12)
            v = _max_cos(trial)
            if v < vc:
                Xc = trial; vc = v
                step = min(step * 1.3, 0.008)
            else:
                step *= 0.5
                if step < 2e-7:
                    # restart with smaller window search around best
                    break
        return Xc, vc

    def greedy_refill(Xbase, del_idx, Cnum=22000):
        keep_mask = np.ones(n, dtype=bool)
        keep_mask[del_idx] = False
        Xkeep = Xbase[keep_mask]
        # candidate pool uniform on sphere
        Z = rng.standard_normal(size=(Cnum, d))
        Z /= np.linalg.norm(Z, axis=1, keepdims=True)
        # also add small mutations of keep points' antipodes of holes? add random jittered copies of farthest? keep simple
        D = Z @ Xkeep.T
        curmax = D.max(axis=1)
        used = np.zeros(Cnum, dtype=bool)
        new_pts = []
        # precompute for mutual update: we will update curmax with each chosen pt
        for _ in range(len(del_idx)):
            curmax[used] = np.inf
            bi = int(np.argmin(curmax))
            used[bi] = True
            pt = Z[bi].copy()
            new_pts.append(pt)
            dd = Z @ pt
            # mutual max: candidate's max vs chosen set grows
            np.maximum(curmax, dd, out=curmax)
            curmax[used] = np.inf
        new_pts = np.stack(new_pts, axis=0)
        # quick hole relaxation: 3 sweeps, each new pt moved away from its worst neighbor
        all_fixed = Xkeep
        for _ in range(6):
            for s in range(len(new_pts)):
                # dots vs keep + other new
                d1 = all_fixed @ new_pts[s]
                m1 = float(np.max(d1)) if d1.size else -2
                # vs other new pts
                others = np.delete(new_pts, s, axis=0)
                m2 = float(np.max(others @ new_pts[s])) if len(others) else -2
                if m1 >= m2:
                    # worst keeper
                    j = int(np.argmax(d1))
                    w = all_fixed[j]
                else:
                    # worst other new
                    dd2 = others @ new_pts[s]
                    j = int(np.argmax(dd2))
                    w = others[j]
                # tangential step away from w
                g = w - float(new_pts[s] @ w) * new_pts[s]
                gn = float(np.linalg.norm(g))
                if gn > 1e-12:
                    g /= gn
                    st = 0.02 * rng.uniform(0.5, 1.0)
                    p = new_pts[s] - st * g
                    p /= np.linalg.norm(p)
                    # accept if improves its own max
                    d1n = float(np.max(all_fixed @ p)) if all_fixed.size else -2
                    d2n = float(np.max(np.delete(new_pts, s, axis=0) @ p)) if len(new_pts) > 1 else -2
                    if max(d1n, d2n) < max(m1, m2):
                        new_pts[s] = p
        Xo = np.empty_like(Xbase)
        Xo[keep_mask] = Xkeep
        Xo[~keep_mask] = new_pts
        Xo /= np.linalg.norm(Xo, axis=1, keepdims=True)
        return Xo

    # exploration
    attempt = 0
    # pre-generate hub order helpers inline
    while time.time() < T_END - RESERVE:
        attempt += 1
        if time.time() > T_END - RESERVE:
            break
        try:
            mode = attempt % 5
            k = int(rng.integers(8, 26))
            if mode in (0, 1):
                # spherical cap deletion, random direction
                u = rng.standard_normal(size=(d,))
                u /= np.linalg.norm(u)
                s = cur @ u
                del_idx = np.argsort(s)[-k:]
            elif mode in (2, 3):
                # hub neighbourhood: worst point + nearest neighbors
                Gc = cur @ cur.T
                np.fill_diagonal(Gc, -2.0)
                rowm = np.max(Gc, axis=1)
                hub = int(np.argmax(rowm + 0.15 * rng.standard_normal(n) * 0.01))
                # nearest by dot to hub
                dots = cur @ cur[hub]
                dots[hub] = -10
                del_idx = np.argsort(dots)[-k:]
                # ensure hub included often
                if attempt % 2 == 0:
                    del_idx[-1] = hub
            else:
                # random scattered deletion (topology scramble)
                # bias to high-rowmax points
                Gc = cur @ cur.T
                np.fill_diagonal(Gc, -2.0)
                rowm = np.max(Gc, axis=1)
                prob = np.exp((rowm - rowm.max()) / 0.01)
                prob /= prob.sum()
                del_idx = rng.choice(n, size=k, replace=False, p=prob)
            Xt = greedy_refill(cur, del_idx, Cnum=18000 if attempt % 3 else 26000)
            # choose anneal depth by attempt
            if attempt % 4 == 0:
                Xp = anneal(Xt, taus=[0.006, 0.002, 0.0008], iters=[40, 50, 60])
            elif attempt % 4 == 1:
                Xp = anneal(Xt, taus=[0.004, 0.0012], iters=[45, 60])
            elif attempt % 4 == 2:
                Xp = anneal(Xt, taus=[0.008, 0.0025, 0.0007], iters=[35, 45, 55])
            else:
                # global shake + anneal
                Xs = Xt + 0.006 * rng.standard_normal(size=(n, d))
                Xs /= np.linalg.norm(Xs, axis=1, keepdims=True)
                Xp = anneal(Xs, taus=[0.005, 0.0015], iters=[45, 65])
            vp = _max_cos(Xp)
            if vp < bestVal:
                bestVal = vp
                bestX = Xp.copy()
            # non-monotone move: allow uphill to create new basin
            if vp < curVal + 0.004 and vp < 0.735:
                cur = Xp.copy()
                curVal = vp
            else:
                if curVal > bestVal + 0.005 or rng.random() < 0.08:
                    # restart from best with shake
                    cur = bestX + float(rng.uniform(0.002, 0.012)) * rng.standard_normal(size=(n, d))
                    cur /= np.linalg.norm(cur, axis=1, keepdims=True)
                    curVal = _max_cos(cur)
            if attempt > 200:
                break
        except Exception:
            continue

    # final intensive polish of best
    try:
        if time.time() < T_END - 18:
            t = anneal(bestX, taus=[0.003, 0.0012, 0.0005], iters=[90, 110, 130])
            v = _max_cos(t)
            if v < bestVal:
                bestVal = v
                bestX = t
    except Exception:
        pass
    try:
        if time.time() < T_END - 6:
            t2, v2 = subgrad_polish(bestX, T_END - 1.5, step0=1.0e-3)
            if v2 < bestVal:
                bestVal = v2
                bestX = t2
    except Exception:
        pass
    bestX /= np.linalg.norm(bestX, axis=1, keepdims=True)
    return bestX
