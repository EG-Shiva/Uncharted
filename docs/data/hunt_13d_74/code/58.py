import time
import numpy as np
from scipy.optimize import minimize, linprog
from scipy.sparse import coo_matrix

def _norm(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return X / n

def _mc(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def solve():
    t0 = time.time()
    LIM = 172.0
    TRIAL_END = t0 + 120.0
    rng = np.random.default_rng(12345)
    n, d = 74, 13

    cands = []
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=np.float64)
            if a.shape == (n, d):
                cands.append(_norm(a))
        except Exception:
            pass
    if not cands:
        cands.append(_norm(rng.standard_normal((n, d))))
    vals = [_mc(a) for a in cands]
    best = cands[int(np.argmin(vals))].copy()
    best_val = float(min(vals))

    IU0 = np.triu_indices(n, 1)
    I0 = IU0[0].astype(np.int64)
    J0 = IU0[1].astype(np.int64)

    def lse_run(X0, tau, maxiter, frozen=None):
        # frozen: array/list of indices to keep fixed
        if frozen is None or len(frozen) == 0:
            y0 = X0.ravel().copy()
            def fun(yf):
                Y = yf.reshape(n, d)
                r = np.sqrt(np.maximum((Y*Y).sum(axis=1, keepdims=True), 1e-24))
                r = np.maximum(r, 1e-12)
                X = Y / r
                G = X @ X.T
                g = G[I0, J0]
                mmax = float(g.max())
                e = np.exp((g - mmax) / tau)
                S = float(e.sum())
                if not np.isfinite(S) or S <= 0:
                    return float(mmax), np.zeros_like(yf)
                F = mmax + tau*np.log(S + 1e-300)
                w = e / (S + 1e-300)
                Wx = np.zeros_like(X)
                np.add.at(Wx, I0, w[:, None]*X[J0])
                np.add.at(Wx, J0, w[:, None]*X[I0])
                dot = (Wx*X).sum(axis=1, keepdims=True)
                gradY = (Wx - dot*X)/r
                return float(F), gradY.ravel()
            try:
                res = minimize(fun, y0, jac=True, method="L-BFGS-B",
                    options={"maxiter": int(maxiter), "ftol": 1e-20, "gtol": 1e-7, "maxfun": int(maxiter*8)})
                Y = res.x.reshape(n, d)
            except Exception:
                return X0
            rn = np.linalg.norm(Y, axis=1, keepdims=True); rn[rn==0]=1.0
            X = Y/rn
            if not np.all(np.isfinite(X)):
                return X0
            return _norm(X)
        else:
            frozen = np.asarray(frozen, dtype=int)
            fzset = set(frozen.tolist())
            free = np.array([i for i in range(n) if i not in fzset], dtype=int)
            nf = len(free)
            if nf == 0:
                return X0
            Xfz = X0[frozen].copy()  # fixed, already unit
            y0 = X0[free].ravel().copy()
            def fun2(yf):
                Yf = yf.reshape(nf, d)
                rf = np.sqrt(np.maximum((Yf*Yf).sum(axis=1, keepdims=True), 1e-24))
                rf = np.maximum(rf, 1e-12)
                Xf = Yf / rf
                # assemble full X
                X = np.empty((n, d))
                X[free] = Xf
                X[frozen] = Xfz
                G = X @ X.T
                g = G[I0, J0]
                mmax = float(g.max())
                e = np.exp((g - mmax)/tau)
                S = float(e.sum())
                if not np.isfinite(S) or S <= 0:
                    return float(mmax), np.zeros_like(yf)
                F = mmax + tau*np.log(S+1e-300)
                w = e/(S+1e-300)
                Wx = np.zeros_like(X)
                np.add.at(Wx, I0, w[:, None]*X[J0])
                np.add.at(Wx, J0, w[:, None]*X[I0])
                dot = (Wx*X).sum(axis=1, keepdims=True)
                gradYfull = (Wx - dot*X)
                # only free rows, chain through normalization
                gradYf = gradYfull[free]/rf
                return float(F), gradYf.ravel()
            try:
                res = minimize(fun2, y0, jac=True, method="L-BFGS-B",
                    options={"maxiter": int(maxiter), "ftol": 1e-20, "gtol": 1e-7, "maxfun": int(maxiter*8)})
                Yf = res.x.reshape(nf, d)
            except Exception:
                return X0
            rn = np.linalg.norm(Yf, axis=1, keepdims=True); rn[rn==0]=1.0
            Xf = Yf/rn
            X = np.empty((n, d))
            X[free] = Xf
            X[frozen] = Xfz
            if not np.all(np.isfinite(X)):
                return X0
            return _norm(X)

    def lp_polish(X, seconds, radius_init=8e-4):
        X = _norm(X)
        cur = _mc(X)
        radius = float(radius_init)
        start = time.time()
        iu = np.triu_indices(n, 1)
        while time.time()-start < seconds and radius > 1e-14:
            if time.time()-t0 > LIM-4:
                break
            Cfull = (X @ X.T)[iu]
            m = float(Cfull.max())
            sel = Cfull > m - max(20.0*radius, 1e-9)
            if np.count_nonzero(sel) > 2500:
                thr = np.sort(Cfull)[-2500]
                sel = Cfull >= thr
            I = iu[0][sel]; J = iu[1][sel]
            c = Cfull[sel]
            p = len(I)
            if p == 0:
                break
            nv = n*d+1
            rows = np.repeat(np.arange(p), 2*d+1)
            cols = np.concatenate([I[:,None]*d+np.arange(d), J[:,None]*d+np.arange(d), np.full((p,1), n*d)],1).ravel()
            vals = np.concatenate([X[J], X[I], -np.ones((p,1))],1).ravel()
            try:
                A_ub = coo_matrix((vals,(rows,cols)), shape=(p,nv)).tocsr()
                A_eq = coo_matrix((X.ravel(),(np.repeat(np.arange(n),d),np.arange(n*d))), shape=(n,nv)).tocsr()
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
            Y = _norm(Y)
            v = _mc(Y)
            if v < cur:
                X, cur = Y, v
                radius = min(radius*1.5, 0.02)
            else:
                radius *= 0.5
        return X

    def contacts_deg(X, slack=1e-7):
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        gmax = float(G.max())
        deg = (G >= gmax - slack).sum(axis=1)
        return G, gmax, deg

    # archive keyed by sorted degree sequence
    archive = set()
    def deg_key(X, slack=1e-7):
        _, _, deg = contacts_deg(X, slack)
        s = tuple(sorted(deg.tolist()))
        return s

    archive.add(deg_key(best))
    incumbent_key = deg_key(best)

    # quick baseline squeeze if time allows
    try:
        if time.time()-t0 < TRIAL_END-10:
            Xb = lse_run(best, 3e-4, 40, frozen=None)
            vb = _mc(Xb)
            if vb < best_val:
                best, best_val = Xb, float(vb)
                incumbent_key = deg_key(best)
                archive.add(incumbent_key)
    except Exception:
        pass

    # build surgery trial list
    def build_trials(Xcur):
        G, gmax, deg = contacts_deg(Xcur, slack=1e-7)
        ratt = np.where(deg == 0)[0].tolist()
        hubs = np.where(deg >= 7)[0].tolist()
        if len(ratt) == 0:
            # lowest degree points as pseudo-rattlers
            order = np.argsort(deg)
            ratt = order[:max(4, 0)].tolist() if len(order)>0 else []
            if len(ratt)==0:
                ratt=[int(np.argmin(deg))]
        if len(hubs) == 0:
            order = np.argsort(-deg)
            hubs = order[:8].tolist()
        # highest-contact neighbour for each hub
        trials = []
        for h in hubs:
            # neighbour with max cosine to h
            row = G[h].copy()
            row[h] = -2.0
            nb = int(np.argmax(row))
            for r in ratt:
                trials.append((int(h), None, int(r)))
                trials.append((int(h), int(nb), int(r)))
        # shuffle deterministic
        perm = rng.permutation(len(trials))
        trials = [trials[i] for i in perm]
        return trials, G, gmax, deg

    trials, _, _, _ = build_trials(best)
    # strict trial budget: iterate
    ti = 0
    while ti < len(trials) and time.time() < TRIAL_END-6 and time.time()-t0 < LIM-12:
        h, nb, r = trials[ti]
        ti += 1
        # refresh trial list if best changed substantially? rebuild occasionally
        if ti % 12 == 0:
            try:
                trials2, _, _, _ = build_trials(best)
                # append new unseen order tail
                # keep remaining + new shuffled
                extra = [t for t in trials2 if t not in set(trials)]
                if extra:
                    perm2 = rng.permutation(len(extra))
                    trials.extend([extra[i] for i in perm2])
            except Exception:
                pass
        try:
            Xcur = best.copy()
            if nb is None:
                moving = [h]
            else:
                if nb == h:
                    moving = [h]
                else:
                    moving = [h, nb]
            moving = [m for m in moving if 0 <= m < n]
            fixed = [i for i in range(n) if i not in set(moving)]
            # include rattler exclusion? spec says score vs non-moving 72
            # for single move we exclude rattler too to get 72 (helps find hole ignoring rattler)
            score_fixed = fixed
            if nb is None and r in score_fixed:
                score_fixed = [i for i in score_fixed if i != r]
            Yf = Xcur[score_fixed]
            M = 20000
            # sample in chunks to limit memory
            best_c = None
            best_s = 1e9
            # we need top candidates for double; keep all? chunk approach: track best few
            # for double need two holes: collect best candidate pool
            # simpler: generate full C (20000x13 ~ 2MB) at once
            C = _norm(rng.standard_normal((M, d)))
            S = C @ Yf.T
            sc = S.max(axis=1)
            if len(moving) == 1:
                j = int(np.argmin(sc))
                c_best = C[j]
                theta = float(rng.uniform(0.05, 0.20))
                z = rng.standard_normal(d)
                z = z - float(z @ c_best)*c_best
                nz = float(np.linalg.norm(z))
                if nz < 1e-12:
                    z = rng.standard_normal(d); z -= float(z@c_best)*c_best
                    nz = float(np.linalg.norm(z))
                    if nz < 1e-12:
                        continue
                z /= nz
                c_new = float(np.cos(theta))*c_best + float(np.sin(theta))*z
                c_new /= np.linalg.norm(c_new)
                Xt = Xcur.copy()
                Xt[h] = c_new
                frozen_pts = moving  # freeze transplanted first
            else:
                # double: pick two diverse holes
                order = np.argsort(sc)  # ascending
                c1 = None; c2 = None
                # first
                c1 = C[int(order[0])]
                # second with mutual dot < 0.7
                for idx in order[1:]:
                    cc = C[int(idx)]
                    if float(cc @ c1) < 0.7:
                        c2 = cc
                        break
                if c2 is None:
                    c2 = C[int(order[1])]
                # perturb both
                cn = []
                for cb in (c1, c2):
                    theta = float(rng.uniform(0.05, 0.20))
                    z = rng.standard_normal(d)
                    z = z - float(z @ cb)*cb
                    nz = float(np.linalg.norm(z))
                    if nz < 1e-12:
                        cn.append(cb)
                        continue
                    z /= nz
                    c_new = float(np.cos(theta))*cb + float(np.sin(theta))*z
                    c_new /= np.linalg.norm(c_new)
                    cn.append(c_new)
                Xt = Xcur.copy()
                Xt[moving[0]] = cn[0]
                Xt[moving[1]] = cn[1]
                frozen_pts = moving
            # staged polish: freeze transplanted ~200 steps at tau=1e-3
            # approximate 200 Adam steps by L-BFGS maxiter 60 frozen
            Xt = lse_run(Xt, 1e-3, 60, frozen=frozen_pts)
            # unfreeze polish
            Xt = lse_run(Xt, 3e-4, 40, frozen=None)
            Xt = lse_run(Xt, 8e-5, 30, frozen=None)
            Xt = lp_polish(Xt, seconds=0.8, radius_init=8e-4)
            v = _mc(Xt)
            key = deg_key(Xt)
            is_new_key = (key not in archive)
            archive.add(key)
            cond_final = is_new_key or (v <= best_val + 1e-6)
            if cond_final and time.time() < TRIAL_END-2 and time.time()-t0 < LIM-10:
                # final polish only on differing degree seq or near-best
                Xt2 = lse_run(Xt, 8e-5, 40, frozen=None)
                Xt2 = lp_polish(Xt2, seconds=0.7, radius_init=5e-4)
                v2 = _mc(Xt2)
                if v2 < v:
                    Xt, v = Xt2, v2
                    key = deg_key(Xt)
                    archive.add(key)
            if v < best_val:
                best, best_val = Xt.copy(), float(v)
                incumbent_key = key
        except Exception:
            continue

    # final intensive polish on best with remaining time
    try:
        remain = LIM - (time.time()-t0) - 2.0
        if remain > 4:
            Xf = lse_run(best, 3e-4, 60, frozen=None)
            Xf = lse_run(Xf, 8e-5, 60, frozen=None)
            v = _mc(Xf)
            if v < best_val:
                best, best_val = Xf, float(v)
            lp_t = min(2.0, max(0.5, remain-3.0))
            Xf2 = lp_polish(best, seconds=lp_t, radius_init=6e-4)
            v2 = _mc(Xf2)
            if v2 < best_val:
                best, best_val = Xf2, float(v2)
    except Exception:
        pass

    return _norm(np.asarray(best, dtype=np.float64))
