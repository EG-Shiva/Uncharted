import time
import numpy as np
from scipy.optimize import minimize

def _norm_rows(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[~np.isfinite(n)] = 1.0
    n[n == 0] = 1.0
    return X / n

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _quench_mom(X, steps, T0, T1, lr, mu, deadline=None, track=True):
    n, d = X.shape
    V = np.zeros_like(X)
    lt0 = np.log(T0); lt1 = np.log(T1)
    if track:
        G0 = X @ X.T
        np.fill_diagonal(G0, -2.0)
        bv = float(G0.max()); bX = X.copy()
    else:
        bv = 1e100; bX = None
    for s in range(steps):
        if deadline is not None and (s % 250 == 0) and time.time() > deadline:
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
        if track and ((s % 120 == 0) or s == steps - 1):
            Gc = X @ X.T
            np.fill_diagonal(Gc, -2.0)
            vc = float(Gc.max())
            if vc < bv:
                bv = vc; bX = X.copy()
    if track:
        return bX, bv
    return X, _max_cos(X)

def _lbfgs_lse(X0, T, maxiter):
    n, d = X0.shape
    Y0 = X0.ravel().copy()
    def fun(Yflat):
        Y = Yflat.reshape(n, d)
        r = np.linalg.norm(Y, axis=1, keepdims=True)
        r = np.maximum(r, 1e-12)
        X = Y / r
        G = X @ X.T
        # max off diag
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        np.fill_diagonal(G, 1.0)
        # recompute G for exp? we overwrote diag; fix
        # G diag now 1.0, Q diag = (1-m)/T large -> will be clipped to 0, so zero it after
        Q = (G - m) / T
        np.clip(Q, -50, 0, out=Q)
        E = np.exp(Q)
        np.fill_diagonal(E, 0.0)
        S = float(E.sum() * 0.5)
        if not np.isfinite(S) or S <= 0:
            S = 1e-300
        f = m + T * np.log(S)
        GradX = (E @ X) / S
        dot = np.sum(GradX * X, axis=1, keepdims=True)
        Gt = GradX - dot * X
        GY = Gt / r
        return float(f), GY.ravel()
    try:
        res = minimize(fun, Y0, jac=True, method='L-BFGS-B',
                       options={'maxiter': maxiter, 'ftol': 1e-20, 'gtol': 1e-10, 'maxfun': maxiter*4})
        Yopt = res.x.reshape(n, d)
        Xopt = _norm_rows(Yopt)
        return Xopt, _max_cos(Xopt)
    except Exception:
        return X0, _max_cos(X0)

def _refine_single(s, R, steps=160, T=0.004, lr=0.12, mu=0.8):
    s = s / (np.linalg.norm(s) + 1e-12)
    v = np.zeros_like(s)
    for _ in range(steps):
        g = R @ s
        m = float(g.max())
        e = np.exp(np.clip((g - m) / T, -50, 0))
        den = float(e.sum()) + 1e-30
        Grad = (R.T @ e) / den
        Gt = Grad - float(Grad @ s) * s
        v = mu * v + Gt
        s = s - lr * v
        nn = float(np.linalg.norm(s))
        if not np.isfinite(nn) or nn == 0:
            break
        s = s / nn
    return s

def _find_hole(R, rng, n_sample=4000, n_cand=2):
    d = R.shape[1]
    Z = rng.standard_normal(size=(n_sample, d))
    Z = _norm_rows(Z)
    S = Z @ R.T
    scores = S.max(axis=1)
    idx = np.argpartition(scores, n_cand)[:n_cand]
    best_s = None; best_v = 1e100
    for k in idx:
        s0 = Z[k]
        s1 = _refine_single(s0, R, steps=140, T=0.005, lr=0.12)
        s1 = _refine_single(s1, R, steps=140, T=0.0015, lr=0.07)
        v = float((R @ s1).max())
        if v < best_v:
            best_v = v; best_s = s1
    return best_s / (np.linalg.norm(best_s) + 1e-12)

def _fingerprint(X, k=400):
    G = X @ X.T
    iu = np.triu_indices(X.shape[0], 1)
    v = G[iu]
    v = np.sort(v)[::-1]
    if len(v) > k:
        v = v[:k]
    return v

def _jam_equalize(X, steps, T, lr0, lr1, mu=0.9, deadline=None):
    n, d = X.shape
    V = np.zeros_like(X)
    G0 = X @ X.T
    np.fill_diagonal(G0, -2.0)
    bv = float(G0.max()); bX = X.copy()
    lt0 = np.log(lr0); lt1 = np.log(lr1)
    for s in range(steps):
        if deadline is not None and (s % 400 == 0) and time.time() > deadline:
            break
        frac = s / max(1, steps - 1)
        lr = float(np.exp(lt0 + (lt1 - lt0) * frac))
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
        if (s % 100 == 0) or s == steps - 1:
            Gc = X @ X.T
            np.fill_diagonal(Gc, -2.0)
            vc = float(Gc.max())
            if vc < bv:
                bv = vc; bX = X.copy()
    return bX, bv

def solve():
    t0 = time.time()
    HARD_END = t0 + 174.0
    HOP_END = HARD_END - 34.0
    rng = np.random.default_rng(20261101)
    n, d = 71, 12
    # load references for fallback comparison only
    ref_best = None; ref_score = 1e100
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.asarray(np.load(fn), dtype=np.float64)
            if a.shape == (n, d) and np.all(np.isfinite(a)):
                a = _norm_rows(a)
                s = _max_cos(a)
                if s < ref_score:
                    ref_score = s; ref_best = a.copy()
        except Exception:
            pass
    # ---- initial diverse pool (from scratch, not basin-biased) ----
    inits = []
    for _ in range(4):
        inits.append(_norm_rows(rng.standard_normal((n, d))))
    # lattice-like: vectors with two +-1
    try:
        vecs = []
        for i in range(d):
            for j in range(i+1, d):
                for si in (-1.0, 1.0):
                    for sj in (-1.0, 1.0):
                        v = np.zeros(d); v[i]=si; v[j]=sj
                        vecs.append(v)
        vecs = np.array(vecs)
        idx = rng.choice(len(vecs), size=n, replace=False)
        L = vecs[idx].astype(float) + rng.standard_normal((n,d))*0.35
        inits.append(_norm_rows(L))
    except Exception:
        inits.append(_norm_rows(rng.standard_normal((n,d))))
    # cross + random
    try:
        C = np.zeros((2*d, d))
        for i in range(d):
            C[2*i, i]=1.0; C[2*i+1, i]=-1.0
        R = _norm_rows(rng.standard_normal((n-2*d, d)))
        inits.append(_norm_rows(np.concatenate([C, R], axis=0) + rng.standard_normal((n,d))*0.2))
    except Exception:
        pass
    if ref_best is not None:
        inits.append(ref_best.copy())

    elites = []; escores = []; efps = []
    for X0 in inits:
        if time.time() > HOP_END - 50:
            break
        try:
            # quick anneal: LBFGS at two temps + short momentum
            Xa, _ = _lbfgs_lse(X0, 0.02, 60)
            Xb, _ = _lbfgs_lse(Xa, 0.006, 80)
            Xc, sc = _quench_mom(Xb, steps=700, T0=0.006, T1=0.0008, lr=0.18, mu=0.88, deadline=HOP_END)
        except Exception:
            continue
        elites.append(Xc); escores.append(float(sc)); efps.append(_fingerprint(Xc))
    if len(elites)==0:
        X0 = _norm_rows(rng.standard_normal((n,d)))
        elites=[X0]; escores=[_max_cos(X0)]; efps=[_fingerprint(X0)]
    # keep diverse top 6
    od = np.argsort(np.array(escores))
    elites=[elites[i] for i in od[:6]]; escores=[escores[i] for i in od[:6]]; efps=[efps[i] for i in od[:6]]
    best_X = elites[0].copy(); best_score=float(escores[0])
    if ref_best is not None and ref_score < best_score:
        best_X = ref_best.copy(); best_score = float(ref_score)
        # insert ref as elite but keep diversity
        elites.append(ref_best.copy()); escores.append(float(ref_score)); efps.append(_fingerprint(ref_best))
        od = np.argsort(np.array(escores))
        elites=[elites[i] for i in od[:6]]; escores=[escores[i] for i in od[:6]]; efps=[efps[i] for i in od[:6]]

    it = 0
    while time.time() < HOP_END:
        it += 1
        tleft = HOP_END - time.time()
        if tleft < 1.0:
            break
        # rank-weighted base selection (favor best but often others)
        r = rng.random()
        if r < 0.45:
            base = best_X
        else:
            bi = int(rng.integers(0, len(elites)))
            base = elites[bi]
        u = rng.random()
        X0 = base.copy()
        try:
            if u < 0.10:
                X0 = _norm_rows(rng.standard_normal((n,d)))
            elif u < 0.34:
                sigma = float(10 ** rng.uniform(-1.3, -0.1))  # 0.05..0.8 large shake
                X0 = _norm_rows(base + rng.standard_normal((n,d))*sigma)
            elif u < 0.68:
                # large cap kill 6..18
                k = int(rng.integers(6, 19))
                direction = rng.standard_normal(d)
                direction /= (np.linalg.norm(direction)+1e-12)
                if rng.random() < 0.5:
                    direction = base[int(rng.integers(0,n))].copy()+rng.standard_normal(d)*0.3
                    direction /= (np.linalg.norm(direction)+1e-12)
                proj = base @ direction
                idx = np.argpartition(-proj, k)[:k]
                R = np.delete(base, idx, axis=0)
                for j in idx:
                    if rng.random() < 0.25:
                        h = _norm_rows(rng.standard_normal((1,d)))[0]
                    else:
                        h = _find_hole(R, rng, n_sample=3500, n_cand=2)
                    X0[j]=h
                    R = np.concatenate([R, h[None,:]], axis=0)
                keep = np.ones(n,bool); keep[idx]=False
                nk = int(np.count_nonzero(keep))
                X0[keep] += rng.standard_normal((nk,d))*float(10**rng.uniform(-2.5,-1.0))
                X0 = _norm_rows(X0)
            else:
                # core-keep: keep 35..55, greedy refill rest
                keep_n = int(rng.integers(35, 56))
                perm = rng.permutation(n)
                keep_idx = np.sort(perm[:keep_n])
                kill_idx = np.sort(perm[keep_n:])
                R = base[keep_idx].copy()
                X0 = base.copy()
                # small noise on kept
                X0[keep_idx] += rng.standard_normal((keep_n,d))*float(10**rng.uniform(-2.5,-1.2))
                X0[keep_idx] = _norm_rows(X0[keep_idx])
                R = X0[keep_idx].copy()
                for j in kill_idx:
                    h = _find_hole(R, rng, n_sample=3500, n_cand=2)
                    X0[j]=h
                    R = np.concatenate([R, h[None,:]], axis=0)
                X0 = _norm_rows(X0)
        except Exception:
            continue
        # quench choice
        try:
            q = rng.random()
            if q < 0.45:
                T0 = float(10**rng.uniform(-2.4,-1.2)); T1 = float(10**rng.uniform(-4.2,-3.0))
                lr = float(10**rng.uniform(-1.3,-0.7)); mu=float(rng.uniform(0.82,0.94))
                steps = 1500 if tleft>25 else 900
                if rng.random()<0.15: steps=2600
                X1, s1 = _quench_mom(X0, steps=steps, T0=T0, T1=T1, lr=lr, mu=mu, deadline=HOP_END)
            else:
                # LBFGS anneal + momentum finish
                Tl = float(10**rng.uniform(-2.0,-1.4))
                Xl, _ = _lbfgs_lse(X0, Tl, 50)
                Tl2 = float(10**rng.uniform(-2.8,-2.2))
                Xl, _ = _lbfgs_lse(Xl, Tl2, 60)
                T0 = float(10**rng.uniform(-2.6,-1.9)); T1=float(10**rng.uniform(-4.0,-3.2))
                lr=float(10**rng.uniform(-1.3,-0.8)); mu=float(rng.uniform(0.82,0.93))
                steps = 1000 if tleft>25 else 700
                X1, s1 = _quench_mom(Xl, steps=steps, T0=T0, T1=T1, lr=lr, mu=mu, deadline=HOP_END)
        except Exception:
            continue
        if not np.isfinite(s1):
            continue
        if s1 < best_score - 1e-12:
            best_score=float(s1); best_X=X1.copy()
        # diversity tabu: fingerprint distance
        try:
            fp = _fingerprint(X1)
            dmin = min(float(np.linalg.norm(fp - e)) for e in efps) if len(efps)>0 else 1e10
        except Exception:
            dmin = 1e10
        # accept if better than worst or novel and decent
        worst = max(escores) if len(escores)>0 else 1e100
        if (s1 < worst - 2e-6) or (dmin > 2e-4 and s1 < worst + 0.004 and s1 < 0.26):
            # duplicate basin rejection: if very close and not better than best, skip
            if dmin < 5e-5 and s1 >= best_score - 1e-9:
                continue
            elites.append(X1.copy()); escores.append(float(s1)); efps.append(fp)
            od = np.argsort(np.array(escores))
            elites=[elites[i] for i in od[:6]]; escores=[escores[i] for i in od[:6]]; efps=[efps[i] for i in od[:6]]

    # ---- final high-precision jamming on top 2 ----
    order = np.argsort(np.array(escores))
    final_cands = [best_X]
    for i in order[:2]:
        if np.linalg.norm(elites[i]-best_X) > 1e-6:
            final_cands.append(elites[i])
            break
    cur_best = best_X.copy(); cur_score = best_score
    for cand in final_cands:
        if time.time() > HARD_END - 3:
            break
        X = cand.copy()
        try:
            X, _ = _lbfgs_lse(X, 0.006, 70)
            X, _ = _lbfgs_lse(X, 0.002, 90)
            if time.time() < HARD_END - 12:
                X, _ = _lbfgs_lse(X, 0.0009, 120)
            X, s = _jam_equalize(X, steps=2500, T=0.0006, lr0=0.05, lr1=0.004, mu=0.9, deadline=HARD_END-2)
            if s < cur_score:
                cur_score=s; cur_best=X.copy()
            # second jam at even lower T
            if time.time() < HARD_END - 8:
                X2, s2 = _jam_equalize(cur_best, steps=1500, T=0.00032, lr0=0.02, lr1=0.002, mu=0.92, deadline=HARD_END-2)
                if s2 < cur_score:
                    cur_score=s2; cur_best=X2.copy()
        except Exception:
            continue
    # never regress vs reference
    if ref_best is not None and ref_score < cur_score:
        cur_best = ref_best.copy(); cur_score = ref_score
    cur_best = _norm_rows(cur_best)
    # tiny shrink guard (strict feasibility, no zero rows)
    return cur_best
