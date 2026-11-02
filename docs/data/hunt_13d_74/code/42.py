import time
import numpy as np
from scipy.optimize import minimize, linprog
from scipy.sparse import coo_matrix

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
    I0 = IU[0].astype(np.int64); J0 = IU[1].astype(np.int64)

    def lse_polish(X0, tau, maxiter, freeze=None):
        # freeze: boolean array n, True means frozen
        if freeze is None:
            free_idx = np.arange(n)
        else:
            free_idx = np.where(~freeze)[0]
        if len(free_idx) == 0:
            return X0.copy()
        mf = len(free_idx)
        # map global -> free position
        g2f = np.full(n, -1, dtype=np.int64)
        g2f[free_idx] = np.arange(mf)
        Ybase = X0.copy()  # assumed normalized
        p0 = Ybase[free_idx].ravel().copy()
        def fun(p):
            Y = Ybase.copy()
            Y[free_idx] = p.reshape(mf, d)
            r = np.sqrt((Y*Y).sum(axis=1, keepdims=True))
            r = np.maximum(r, 1e-12)
            X = Y / r
            G = X @ X.T
            g = G[I0, J0]
            mmax = float(g.max())
            e = np.exp((g - mmax) / tau)
            S = float(e.sum())
            if not np.isfinite(S) or S <= 0:
                return float(mmax), np.zeros_like(p)
            F = mmax + tau*np.log(S + 1e-300)
            w = e / (S + 1e-300)
            W = np.zeros((n, n), dtype=np.float64)
            W[I0, J0] = w
            W[J0, I0] = w
            Wx = W @ X
            dot = (Wx*X).sum(axis=1, keepdims=True)
            gradY = (Wx - dot*X) / r
            gp = gradY[free_idx].ravel()
            if not np.all(np.isfinite(gp)):
                gp = np.zeros_like(p)
            return float(F), gp
        try:
            res = minimize(fun, p0, jac=True, method="L-BFGS-B",
                           options={"maxiter": int(maxiter), "ftol": 1e-20,
                                    "gtol": 1e-7, "maxfun": int(maxiter*8)})
            p = res.x
        except Exception:
            return X0.copy()
        Y = Ybase.copy()
        Y[free_idx] = p.reshape(mf, d)
        rn = np.linalg.norm(Y, axis=1, keepdims=True)
        rn[rn == 0] = 1.0
        X = Y / rn
        if not np.all(np.isfinite(X)):
            return X0.copy()
        return _normalize(X)

    def lp_polish(X, budget, radius_init=8e-4):
        start = time.time()
        cur = _max_cos(X)
        radius = float(radius_init)
        nv = n*d + 1
        iu = IU
        while time.time()-start < budget and time.time()-t0 < LIM-2:
            if radius < 1e-13:
                break
            Cfull = (X @ X.T)[iu]
            m = float(Cfull.max())
            sel = Cfull > m - max(20.0*radius, 1e-9)
            if int(np.count_nonzero(sel)) > 2200:
                thr = float(np.sort(Cfull)[-2200])
                sel = Cfull >= thr
            I = iu[0][sel].astype(np.int64); J = iu[1][sel].astype(np.int64)
            c = Cfull[sel]
            p = len(I)
            if p == 0:
                break
            rows = np.repeat(np.arange(p), 2*d+1)
            cols = np.concatenate([I[:,None]*d+np.arange(d), J[:,None]*d+np.arange(d), np.full((p,1), n*d)],1).ravel()
            valsA = np.concatenate([X[J], X[I], -np.ones((p,1))],1).ravel()
            try:
                A_ub = coo_matrix((valsA,(rows,cols)), shape=(p,nv)).tocsr()
                A_eq = coo_matrix((X.ravel(),(np.repeat(np.arange(n),d), np.arange(n*d))), shape=(n,nv)).tocsr()
            except Exception:
                radius *= 0.5
                continue
            obj = np.zeros(nv); obj[-1]=1.0
            try:
                res = linprog(obj, A_ub=A_ub, b_ub=-c, A_eq=A_eq, b_eq=np.zeros(n),
                              bounds=[(-radius,radius)]*(n*d)+[(None,None)], method="highs")
            except Exception:
                radius *= 0.5
                continue
            if res.status != 0:
                radius *= 0.5
                continue
            Y = X + res.x[:-1].reshape(n,d)
            Y = _normalize(Y)
            if not np.all(np.isfinite(Y)):
                radius *= 0.5
                continue
            v = _max_cos(Y)
            if v < cur:
                X[:] = Y; cur = v
                radius = min(radius*1.5, 0.02)
            else:
                radius *= 0.5
        return X, cur

    def contacts(X, eps=1e-7):
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        gmax = float(G.max())
        C = G >= (gmax - eps)
        cnt = C.sum(axis=1).astype(int)
        per_max = G.max(axis=1)
        return gmax, cnt, per_max, G

    def best_holes(fixed_X, k=2, M=20000):
        # fixed_X: (nf,d) normalized
        cand = rng.standard_normal((M, d))
        cand /= np.linalg.norm(cand, axis=1, keepdims=True)
        S = cand @ fixed_X.T
        sc = S.max(axis=1)
        order = np.argsort(sc)
        picks = []
        chosen = []
        for idx in order:
            c = cand[idx]
            ok = True
            for ch in chosen:
                if float(c @ ch) > 0.9:
                    ok = False
                    break
            if ok:
                picks.append(c)
                chosen.append(c)
                if len(picks) >= k:
                    break
        while len(picks) < k:
            picks.append(cand[order[len(picks)]])
        return picks

    def perturb_rad(v, rad):
        # random tangent perturbation of geodesic radius rad
        t = rng.standard_normal(d)
        t -= float(t @ v) * v
        nm = float(np.linalg.norm(t))
        if nm < 1e-12:
            t = np.zeros(d); t[0]=1.0
            t -= float(t @ v)*v
            nm = float(np.linalg.norm(t))
            if nm < 1e-12:
                return v.copy()
        t /= nm
        return float(np.cos(rad))*v + float(np.sin(rad))*t

    # light initial tighten of incumbent
    Xb = best.copy()
    Xb = lse_polish(Xb, 3e-4, 80, freeze=None)
    Xb = lse_polish(Xb, 8e-5, 80, freeze=None)
    Xb, vb = lp_polish(Xb, budget=1.5, radius_init=5e-4)
    if vb < best_val:
        best, best_val = Xb, vb
    # archive keyed by sorted degree sequence
    def deg_key(cnt):
        return tuple(sorted(cnt.tolist()))
    _, cnt0, _, _ = contacts(best, eps=1e-7)
    archive = set()
    archive.add(deg_key(cnt0))

    # enumerate rattler/hub choices
    gmax, cnt, per_max, G = contacts(best, eps=1e-7)
    rattlers = np.where(cnt == 0)[0].tolist()
    hubs = np.where(cnt >= 7)[0].tolist()
    if len(rattlers) == 0:
        rattlers = np.argsort(cnt)[:4].tolist()
    if len(hubs) == 0:
        hubs = np.argsort(-cnt)[:6].tolist()
    # build trial list: (kind, hub, rattler) kind 0=single,1=double+neighbour
    trials = []
    for h in hubs:
        for r in rattlers:
            trials.append((0, int(h), int(r)))
            trials.append((1, int(h), int(r)))
    # also hub-only without rattler pairing for diversity (use first rattler)
    # shuffle deterministically
    perm = rng.permutation(len(trials))
    trials = [trials[i] for i in perm]

    trial_deadline = t0 + 120.0
    # strict 120s trial budget from start? use min(trial_deadline, LIM-30)
    trial_end = min(t0+120.0, t0+LIM-35.0)
    # ensure we don't exceed: if already late, skip
    for (kind, h, r) in trials:
        if time.time() > trial_end or time.time()-t0 > LIM-35:
            break
        # recompute hubs/rattlers of current best periodically? use incumbent best's info
        # use stored h,r but validate indices
        Xtry = best.copy()
        # find neighbour of h (highest contact) for kind 1
        if kind == 0:
            moving = [h, r]  # hub + rattler move, fixed 72
            fixed_idx = [i for i in range(n) if i not in (h, r)]
            fixed_X = Xtry[fixed_idx]
            holes = best_holes(fixed_X, k=2, M=20000)
            rad1 = float(rng.uniform(0.05, 0.20))
            rad2 = float(rng.uniform(0.05, 0.20))
            # transplant hub to best hole, rattler to second best
            Xtry[h] = _normalize(perturb_rad(holes[0], rad1).reshape(1,-1))[0]
            Xtry[r] = _normalize(perturb_rad(holes[1], rad2).reshape(1,-1))[0]
            frozen = np.zeros(n, dtype=bool)
            frozen[h] = True  # freeze transplanted hub first
        else:
            # hub plus its highest-contact neighbour
            row = G[h].copy()
            row[h] = -10
            nb = int(np.argmax(row))
            if nb == r:
                # pick second best to keep distinct
                row[nb] = -10
                nb = int(np.argmax(row))
            moving = [h, nb]
            fixed_idx = [i for i in range(n) if i not in (h, nb)]
            fixed_X = Xtry[fixed_idx]
            holes = best_holes(fixed_X, k=2, M=20000)
            rad1 = float(rng.uniform(0.05, 0.20))
            rad2 = float(rng.uniform(0.05, 0.20))
            Xtry[h] = _normalize(perturb_rad(holes[0], rad1).reshape(1,-1))[0]
            Xtry[nb] = _normalize(perturb_rad(holes[1], rad2).reshape(1,-1))[0]
            frozen = np.zeros(n, dtype=bool)
            frozen[h] = True
            frozen[nb] = True
        # stage 1 frozen anneal at tau=1e-3 (~200 steps: use maxiter 120 frozen)
        # keep budget tight
        try:
            Xtry = lse_polish(Xtry, 1e-3, 100, freeze=frozen)
        except Exception:
            continue
        # unfreeze polish at 3e-4, 8e-5
        try:
            Xtry = lse_polish(Xtry, 3e-4, 60, freeze=None)
            Xtry = lse_polish(Xtry, 8e-5, 60, freeze=None)
        except Exception:
            continue
        vtry = _max_cos(Xtry)
        _, cnt_try, _, _ = contacts(Xtry, eps=1e-7)
        key = deg_key(cnt_try)
        is_new = (key not in archive)
        # only spend LP polish if degree seq differs or within 1e-6 of best
        if (not is_new) and (vtry > best_val + 1e-6):
            # record key anyway? don't add unless polished? add to avoid repeat
            continue
        # 1-2s LP polish
        Xtry, vtry = lp_polish(Xtry, budget=1.2, radius_init=5e-4)
        # one more short LSE to clean
        Xtry = lse_polish(Xtry, 8e-5, 40, freeze=None)
        vtry = _max_cos(Xtry)
        _, cnt_try2, _, _ = contacts(Xtry, eps=1e-7)
        key2 = deg_key(cnt_try2)
        archive.add(key)
        archive.add(key2)
        if vtry < best_val - 1e-14:
            best = Xtry.copy()
            best_val = float(vtry)
            # update G for neighbour lookups
            _, cnt, per_max, G = contacts(best, eps=1e-7)
        if time.time()-t0 > LIM-30:
            break

    # final intensive polish on best
    if time.time()-t0 < LIM-2:
        remain = LIM - (time.time()-t0)
        lp_b = min(6.0, max(1.0, remain-6.0))
        best = lse_polish(best, 3e-4, 100, freeze=None)
        if time.time()-t0 < LIM-8:
            best = lse_polish(best, 8e-5, 100, freeze=None)
        best, best_val = lp_polish(best, budget=lp_b, radius_init=5e-4)
        # final micro LSE
        if time.time()-t0 < LIM-2:
            best = lse_polish(best, 8e-5, 60, freeze=None)
            best_val = _max_cos(best)
    best = _normalize(best)
    return best
