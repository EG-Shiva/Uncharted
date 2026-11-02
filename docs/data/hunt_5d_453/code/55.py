import time
import numpy as np
from scipy.optimize import minimize, linprog
from scipy.sparse import coo_matrix
from scipy.linalg import expm

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def _row_max(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return G.max(axis=1)

def _tangent_bases(X):
    n, d = X.shape
    B = np.empty((n, d, 4), dtype=np.float64)
    for i in range(n):
        x = X[i]
        k = int(np.argmin(np.abs(x)))
        T = np.empty((d, 4), dtype=np.float64)
        c = 0
        for j in range(d):
            if j == k:
                continue
            v = -x[j] * x
            v[j] += 1.0
            T[:, c] = v
            c += 1
        Q, _ = np.linalg.qr(T)
        B[i] = Q[:, :4]
    return B

def solve():
    t0 = time.time()
    HARD = t0 + 172.0
    CLUSTER_END = t0 + 128.0
    LP_CUTOFF = t0 + 150.0
    rng = np.random.default_rng(0)
    n, d = 453, 5
    # load starts
    cands = []
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            cands.append(np.asarray(a, dtype=np.float64))
        except Exception:
            pass
    bestX = None
    bestV = float("inf")
    for cc in cands:
        try:
            if cc.shape != (453, 5):
                continue
            nn = np.linalg.norm(cc, axis=1, keepdims=True)
            nn[nn == 0] = 1.0
            cc = cc / nn
            if not np.all(np.isfinite(cc)):
                continue
            vv = _max_cos(cc)
            if vv < bestV:
                bestV = vv
                bestX = cc.copy()
        except Exception:
            continue
    if bestX is None:
        bestX = rng.standard_normal((n, d))
        bestX /= np.linalg.norm(bestX, axis=1, keepdims=True)
        bestV = _max_cos(bestX)
    Xcur = bestX.copy()
    curV = bestV

    iu_r, iu_c = np.triu_indices(n, 1)

    def softmax_smooth(X, steps, tau0, tau1, lr0, lr1, mu=0.85, deadline=1e18):
        v = np.zeros_like(X)
        lt = np.log(tau1 / tau0) if tau0 != tau1 else 0.0
        ll = np.log(lr1 / lr0) if lr0 != lr1 else 0.0
        for it in range(steps):
            if time.time() > deadline:
                break
            frac = it / max(1, steps - 1)
            tau = tau0 * np.exp(lt * frac)
            lr = lr0 * np.exp(ll * frac)
            G = X @ X.T
            np.fill_diagonal(G, -1.0)
            m = float(G.max())
            E = np.exp((G - m) / tau)
            np.fill_diagonal(E, 0.0)
            row = E.sum(axis=1, keepdims=True) + 1e-30
            Gr = (E @ X) / row
            dot = np.sum(Gr * X, axis=1, keepdims=True)
            Gt = Gr - dot * X
            v = mu * v + (1.0 - mu) * Gt
            Xn = X - lr * v
            Xn /= np.linalg.norm(Xn, axis=1, keepdims=True)
            X = Xn
        return X

    def lp_round(X, cur_best_X, cur_best_V, r_init, max_solves, deadline):
        X = X.copy()
        bX = cur_best_X.copy()
        bV = cur_best_V
        radius = float(r_init)
        Nvar = n * 4
        c_obj = np.zeros(Nvar + 1)
        c_obj[-1] = 1.0
        ar4 = np.arange(4)
        for _ in range(max_solves):
            if time.time() > deadline or time.time() > LP_CUTOFF or time.time() > HARD - 6:
                break
            if radius < 1e-12:
                break
            G = X @ X.T
            c_all = G[iu_r, iu_c]
            m = float(np.max(c_all))
            if m < bV:
                bV = m
                bX = X.copy()
            slack = m - c_all
            tight = np.where(slack < 2e-5)[0]
            if tight.size == 0:
                tight = np.argsort(slack)[:500]
            if tight.size > 2500:
                part = np.argpartition(slack[tight], 2500)[:2500]
                tight = tight[part]
            I = iu_r[tight]; J = iu_c[tight]
            p = int(I.size)
            b_ub = slack[tight]
            B = _tangent_bases(X)
            aI = np.einsum('pdk,pd->pk', B[I], X[J])
            aJ = np.einsum('pdk,pd->pk', B[J], X[I])
            cols2 = np.empty((p, 9), dtype=np.int64)
            cols2[:, 0:4] = I[:, None] * 4 + ar4[None, :]
            cols2[:, 4:8] = J[:, None] * 4 + ar4[None, :]
            cols2[:, 8] = Nvar
            vals2 = np.empty((p, 9))
            vals2[:, 0:4] = aI
            vals2[:, 4:8] = aJ
            vals2[:, 8] = -1.0
            rows = np.repeat(np.arange(p), 9)
            try:
                A_ub = coo_matrix((vals2.ravel(), (rows, cols2.ravel())), shape=(p, Nvar + 1)).tocsr()
            except Exception:
                radius *= 0.5
                continue
            rem = min(deadline - time.time(), HARD - 6 - time.time(), 10.0)
            if rem < 0.5:
                break
            bounds = [(-radius, radius)] * Nvar + [(None, None)]
            try:
                res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs", options={"time_limit": max(0.5, rem)})
            except Exception:
                radius *= 0.5
                continue
            if res.status != 0:
                radius *= 0.5
                continue
            dy = res.x[:-1].reshape(n, 4)
            # retract: X + B*dy
            Y = X + np.einsum('ndk,nk->nd', B, dy)
            Y /= np.linalg.norm(Y, axis=1, keepdims=True)
            v = _max_cos(Y)
            if v < m - 1e-12:
                X = Y
                if v < bV:
                    bV = v
                    bX = Y.copy()
                radius = min(radius * 1.4, 0.02)
            else:
                radius *= 0.5
        return bX, bV

    # skew helper
    triu_idx = [(0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)]
    def vec2skew(p):
        S = np.zeros((5, 5))
        for k, (a, b) in enumerate(triu_idx):
            S[a, b] = p[k]
            S[b, a] = -p[k]
        return S

    def cluster_attempt(X, cur_max, hub, K, mode, reflect, Hvec, rng):
        dots = X @ X[hub]
        # geometric cap: top-K dots
        if K >= n:
            K = 20
        idx = np.argpartition(-dots, K)[:K]
        # ensure hub included
        if hub not in idx:
            idx[0] = hub
        mask = np.zeros(n, dtype=bool)
        mask[idx] = True
        Xc = X[mask]
        Xo = X[~mask]
        # outside-outside max (constant)
        Go = Xo @ Xo.T
        np.fill_diagonal(Go, -2.0)
        out_max = float(Go.max())
        # internal max for global mode (constant)
        if mode == "global":
            Gi = Xc @ Xc.T
            np.fill_diagonal(Gi, -2.0)
            int_max = float(Gi.max())
        else:
            int_max = -2.0
            c0 = Xc.mean(axis=0)
        # reflection matrix
        if reflect:
            u = Hvec / (np.linalg.norm(Hvec) + 1e-30)
            H = np.eye(5) - 2.0 * np.outer(u, u)
        else:
            H = None
        # closure for eval (only internal+cross for speed)
        def fobj(p):
            S = vec2skew(p)
            try:
                R = expm(S)
            except Exception:
                return 9.0
            if reflect:
                R = R @ H
            if mode == "global":
                Xcr = Xc @ R.T
                cr = Xcr @ Xo.T
                cm = float(cr.max())
                return cm if cm > out_max else out_max if out_max > int_max else int_max
            else:
                Y = Xc - c0
                Yr = Y @ R.T
                Xcr = c0 + Yr
                Xcr /= np.linalg.norm(Xcr, axis=1, keepdims=True)
                cr = Xcr @ Xo.T
                cm = float(cr.max())
                Gi2 = Xcr @ Xcr.T
                np.fill_diagonal(Gi2, -2.0)
                im = float(Gi2.max())
                tot = cm
                if im > tot:
                    tot = im
                if out_max > tot:
                    tot = out_max
                return tot
        # random coordinate search first
        best_p = np.zeros(10)
        best_f = fobj(best_p)
        # random samples
        sigmas = [0.015, 0.035, 0.06]
        for sg in sigmas:
            for _ in range(14):
                pr = rng.normal(scale=sg, size=10)
                # sparsify sometimes to rotate in subspace
                if rng.random() < 0.4:
                    z = rng.random(10) < 0.5
                    pr[z] = 0.0
                fv = fobj(pr)
                if fv < best_f:
                    best_f = fv
                    best_p = pr.copy()
        # Nelder-Mead from best_p
        try:
            eps = 0.012
            simp = [best_p]
            for i in range(10):
                e = best_p.copy()
                e[i] += eps
                simp.append(e)
            simp = np.array(simp)
            res = minimize(fobj, best_p, method="Nelder-Mead",
                           options={"maxiter": 70, "maxfev": 130, "xatol": 1e-4, "fatol": 1e-8,
                                    "initial_simplex": simp, "disp": False})
            if res.fun < best_f:
                best_f = float(res.fun)
                best_p = np.asarray(res.x)
        except Exception:
            pass
        return best_p, best_f, idx, mask, H

    # main collective loop
    outer = 0
    K_choices = [12, 20, 30, 40]
    while time.time() < CLUSTER_END and time.time() < HARD - 30:
        outer += 1
        rm = _row_max(Xcur)
        order = np.argsort(-rm)
        # top hubs: take ~14 worst, cycle offset by outer to diversify
        nhub = 14
        hubs = order[:nhub]
        # rotate hub order deterministically with outer
        hubs = np.roll(hubs, outer % nhub)
        improved_sweep = False
        for hi, hub in enumerate(hubs):
            if time.time() > CLUSTER_END or time.time() > HARD - 30:
                break
            K = K_choices[(outer + hi) % len(K_choices)]
            mode = "global" if ((outer + hi) % 3 != 1) else "local"
            reflect = ((outer + hi) % 5 == 4)
            Hvec = rng.standard_normal(5) if reflect else None
            # skip tiny hubs if already good? still try
            cur_max_hub = float(rm[hub])
            try:
                bp, bf, idx, mask, H = cluster_attempt(Xcur, curV, int(hub), int(K), mode, reflect, Hvec, rng)
            except Exception:
                continue
            # accept only if promising vs current global max
            # allow small tolerance that shrinks with outer
            tol = 2e-4 / (1.0 + outer * 0.3)
            if bf > curV + tol and bf > cur_max_hub + tol:
                continue
            # build rotated configuration
            S = vec2skew(bp)
            try:
                R = expm(S)
            except Exception:
                continue
            if reflect:
                u = Hvec / (np.linalg.norm(Hvec) + 1e-30)
                Hm = np.eye(5) - 2.0 * np.outer(u, u)
                R = R @ Hm
            Xt = Xcur.copy()
            if mode == "global":
                Xt[mask] = Xcur[mask] @ R.T
            else:
                c0 = Xcur[mask].mean(axis=0)
                Y = Xcur[mask] - c0
                Yr = Y @ R.T
                Nw = c0 + Yr
                Nw /= np.linalg.norm(Nw, axis=1, keepdims=True)
                Xt[mask] = Nw
            # counter-rotate neighbouring ring
            dots = Xcur @ Xcur[int(hub)]
            # ring = next closest outside cluster
            ord2 = np.argsort(-dots)
            ring = []
            for o in ord2:
                if not mask[o]:
                    ring.append(o)
                    if len(ring) >= 14:
                        break
            ring = np.array(ring, dtype=int)
            # try counter rotation with factor -0.35
            try:
                Rc = expm(-0.35 * S)
                if reflect:
                    # counter without reflection
                    pass
                Xt2 = Xt.copy()
                Xt2[ring] = Xcur[ring] @ Rc.T
                v1 = _max_cos(Xt)
                v2 = _max_cos(Xt2)
                if v2 < v1:
                    Xt = Xt2
                    v = v2
                else:
                    v = v1
            except Exception:
                v = _max_cos(Xt)
            # quick local softmax relax (short, only if promising)
            if v < curV + 0.002:
                dl = min(HARD - 25, time.time() + 4.0)
                Xt = softmax_smooth(Xt, 220, 0.003, 0.00035, 0.012, 0.002, 0.85, dl)
                v = _max_cos(Xt)
            if v < curV:
                Xcur = Xt
                curV = v
                if v < bestV:
                    bestV = v
                    bestX = Xt.copy()
                improved_sweep = True
                # update rm for remaining hubs in sweep
                rm = _row_max(Xcur)
            elif v < curV + 0.0005 and rng.random() < 0.08:
                # sideways move to different basin
                Xcur = Xt
                curV = v
        # periodic LP polish if sweep improved or every 2 sweeps
        if outer % 2 == 0 or improved_sweep:
            if time.time() < HARD - 25 and time.time() < LP_CUTOFF:
                try:
                    bX2, bV2 = lp_round(Xcur, bestX, bestV, 6e-4, 3, min(HARD - 20, time.time() + 12))
                    if bV2 < bestV:
                        bestV = bV2
                        bestX = bX2.copy()
                    # continue from best
                    Xcur = bestX.copy()
                    curV = bestV
                except Exception:
                    pass
        # small global jitter to avoid exact repeat
        if not improved_sweep and outer % 3 == 0:
            if time.time() < CLUSTER_END:
                Xcur = Xcur + rng.normal(scale=0.0012, size=Xcur.shape)
                Xcur /= np.linalg.norm(Xcur, axis=1, keepdims=True)
                dl = min(HARD - 25, time.time() + 3.0)
                Xcur = softmax_smooth(Xcur, 180, 0.004, 0.0004, 0.014, 0.002, 0.85, dl)
                curV = _max_cos(Xcur)
                if curV < bestV:
                    bestV = curV
                    bestX = Xcur.copy()
    # final fine tune
    if time.time() < HARD - 12:
        try:
            Xcur = softmax_smooth(bestX, 500, 0.0015, 0.0002, 0.008, 0.001, 0.9, HARD - 10)
            v = _max_cos(Xcur)
            if v < bestV:
                bestV = v
                bestX = Xcur.copy()
        except Exception:
            pass
    if time.time() < HARD - 3:
        try:
            bXf, bVf = lp_round(bestX, bestX, bestV, 4e-4, 8, HARD - 1.0)
            bestX = bXf
            bestV = bVf
        except Exception:
            pass
    bestX = bestX / np.linalg.norm(bestX, axis=1, keepdims=True)
    bestX = np.nan_to_num(bestX, nan=1.0 / np.sqrt(5.0))
    bestX = bestX / np.linalg.norm(bestX, axis=1, keepdims=True)
    return bestX
