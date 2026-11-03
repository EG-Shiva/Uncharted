import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _per_point_max(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return G.max(axis=1), G

def _lp_polish(X, budget, t_deadline=1e18):
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    n, d = X.shape
    best = _max_cos(X)
    bestX = X.copy()
    radius = 1e-3
    t0 = time.time()
    iu0 = np.triu_indices(n, 1)
    while time.time() - t0 < budget and time.time() < t_deadline:
        if radius < 1e-14:
            break
        G = X @ X.T
        c_full = G[iu0]
        m = c_full.max()
        thresh = m - max(20 * radius, 1e-9)
        sel = c_full > thresh
        I = iu0[0][sel]; J = iu0[1][sel]
        cc = c_full[sel]
        p = len(I)
        if p == 0:
            break
        if p > 2500:
            idx = np.argpartition(-cc, 2500)[:2500]
            I, J, cc = I[idx], J[idx], cc[idx]
            p = 2500
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        ci = (I[:, None] * d + np.arange(d)).ravel()
        cj = (J[:, None] * d + np.arange(d)).ravel()
        cols = np.empty(p * (2 * d + 1), dtype=np.int64)
        cols[0::2*d+1] = 0  # placeholder
        # build cols vectorized
        base = np.concatenate([I[:, None]*d + np.arange(d), J[:, None]*d + np.arange(d), np.full((p,1), n*d)], axis=1).ravel()
        cols = base
        vals = np.concatenate([X[J], X[I], -np.ones((p,1))], axis=1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((X.ravel(), (np.repeat(np.arange(n), d), np.arange(n*d))), shape=(n, nv)).tocsr()
        except Exception:
            radius *= 0.5
            continue
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-cc, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)]*(n*d)+[(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = X + res.x[:-1].reshape(n, d)
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        v = _max_cos(y)
        if v < best:
            best = v
            bestX = y.copy()
            X = y
            radius = min(radius*1.5, 0.05)
        else:
            radius *= 0.5
            # keep X unchanged, continue shrinking
    return bestX, best

def _smooth_opt(X, rng, steps, a0, a1, lr0, t_deadline):
    n, d = X.shape
    M = np.zeros_like(X)
    V = np.zeros_like(X)
    b1, b2, eps = 0.9, 0.999, 1e-8
    bestX = X.copy()
    # initial best
    G0 = X @ X.T
    np.fill_diagonal(G0, -2.0)
    best = float(G0.max())
    for t in range(1, steps+1):
        if t % 32 == 0 and time.time() > t_deadline:
            break
        frac = t / steps
        alpha = a0 * (a1/a0)**frac
        # cosine anneal lr
        lr = lr0 * 0.5 * (1.0 + np.cos(np.pi*frac))
        lr = max(lr, lr0*0.05)
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        gmax = float(G.max())
        if gmax < best:
            best = gmax
            bestX = X.copy()
        # weights
        # clip exponent for stability
        E = alpha*(G - gmax)
        # avoid underflow: exp(-30)~9e-14, cut
        W = np.exp(np.clip(E, -30, 0.0))
        np.fill_diagonal(W, 0.0)
        Grad = W @ X
        dot = np.sum(Grad*X, axis=1, keepdims=True)
        Gt = Grad - dot*X
        M = b1*M + (1-b1)*Gt
        V = b2*V + (1-b2)*(Gt*Gt)
        Mh = M/(1-b1**t)
        Vh = V/(1-b2**t)
        step = Mh/(np.sqrt(Vh)+eps)
        X = X - lr*step
        X /= np.linalg.norm(X, axis=1, keepdims=True)
    # final check
    v = _max_cos(X)
    if v < best:
        best = v
        bestX = X.copy()
    return bestX, best

def _hole_for_rattlers(X, ratt_idx, rng, n_cand=4000):
    n, d = X.shape
    if len(ratt_idx)==0:
        return X
    others = X.copy()
    # iteratively place each rattler into biggest hole of the rest
    for k in ratt_idx:
        # candidates
        C = rng.standard_normal((n_cand, d))
        C /= np.linalg.norm(C, axis=1, keepdims=True)
        # max dot to all points except k
        mask = np.ones(n, dtype=bool); mask[k]=False
        Xo = others[mask]
        dots = C @ Xo.T
        md = dots.max(axis=1)
        bi = int(np.argmin(md))
        best_c = C[bi].copy()
        # local refine: few repulsion steps for this single point
        x = best_c
        for _ in range(200):
            dvec = Xo @ x  # (n-1,)
            # softmax weights to nearest
            am = 40.0
            mx = dvec.max()
            w = np.exp(am*(dvec-mx))
            g = (w[:,None]*Xo).sum(axis=0)
            g = g - (g@x)*x
            gn = np.linalg.norm(g)
            if gn < 1e-12:
                break
            x = x - 0.02*g/(gn+1e-12)
            x /= np.linalg.norm(x)
        others[k] = x
    return others

def solve():
    t_start = time.time()
    T_END = t_start + 172.0
    T_EXPL = t_start + 148.0
    rng = np.random.default_rng(0)
    # load seeds
    cands = []
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=np.float64)
            if a.shape == (131, 9):
                a = a/np.linalg.norm(a, axis=1, keepdims=True)
                if np.all(np.isfinite(a)):
                    cands.append(a)
        except Exception:
            pass
    if len(cands)==0:
        X0 = rng.standard_normal((131,9))
        X0 /= np.linalg.norm(X0, axis=1, keepdims=True)
        cands.append(X0)
    bestX = None
    best = 1e9
    for a in cands:
        v = _max_cos(a)
        if v < best:
            best = v; bestX = a.copy()
    # quick initial LP squeeze if time allows (should already be polished)
    # exploration
    sigmas = [0.008, 0.025, 0.06, 0.012, 0.04, 0.10]
    trial = 0
    n, d = bestX.shape
    while time.time() < T_EXPL:
        sig = sigmas[trial % len(sigmas)]
        mode = trial % 6
        X = bestX.copy()
        if mode in (0,1,2,5):
            X = X + sig*rng.standard_normal(X.shape)
            X /= np.linalg.norm(X, axis=1, keepdims=True)
        elif mode == 3:
            # subset kick: 8 points hard
            X = X + 0.005*rng.standard_normal(X.shape)
            k = 8
            idx = rng.choice(n, size=k, replace=False)
            X[idx] += 0.30*rng.standard_normal((k,d))
            X /= np.linalg.norm(X, axis=1, keepdims=True)
        else:  # mode 4 rattler/core targeted
            ppm, G = _per_point_max(X)
            gmax = ppm.max()
            ratt = np.where(ppm < gmax - 1e-3)[0]
            # jammed: top contacts
            thr = gmax - 5e-3
            cnts = np.sum(G > thr, axis=1)
            order = np.argsort(-cnts)
            nj = min(12, n)
            jam = order[:nj]
            X[jam] += 0.05*rng.standard_normal((nj,d))
            X /= np.linalg.norm(X, axis=1, keepdims=True)
            # relocate rattlers to holes
            if len(ratt)>0:
                # keep up to all rattlers
                X = _hole_for_rattlers(X, ratt[:12], rng, n_cand=3000)
        # smooth anneal: params depend on sig
        if sig >= 0.05:
            steps, a0, a1, lr0 = 3500, 20.0, 220.0, 0.004
        elif sig >= 0.02:
            steps, a0, a1, lr0 = 3000, 30.0, 260.0, 0.003
        else:
            steps, a0, a1, lr0 = 2200, 40.0, 300.0, 0.0025
        # don't exceed exploration deadline
        Xs, vs = _smooth_opt(X, rng, steps, a0, a1, lr0, T_EXPL)
        # decide to LP polish: only if promising
        # threshold: within 8e-3 of best (or better)
        need_lp = (vs < best + 0.008)
        if need_lp:
            # budget 1.5-3s depending on promise
            if vs < best:
                b = 3.0
            else:
                b = 1.6
            # cap by remaining exploration time
            rem = T_EXPL - time.time()
            b = min(b, max(0.4, rem-0.2))
            if b > 0.35:
                Xp, vp = _lp_polish(Xs, b, T_END)
                if vp < best - 1e-12:
                    best = vp
                    bestX = Xp.copy()
                elif vs < best - 1e-12:
                    best = vs
                    bestX = Xs.copy()
            else:
                if vs < best - 1e-12:
                    best = vs
                    bestX = Xs.copy()
        else:
            # still track if surprisingly good
            if vs < best - 1e-12:
                best = vs
                bestX = Xs.copy()
        trial += 1
        if time.time() >= T_EXPL:
            break
    # final polish on best with remaining time
    rem = T_END - time.time()
    if rem > 1.0:
        # use up to 22s but leave 1s margin
        b = min(rem-0.8, 25.0)
        if b > 0.8:
            Xp, vp = _lp_polish(bestX, b, T_END)
            if vp < best:
                best = vp
                bestX = Xp.copy()
    bestX = bestX/np.linalg.norm(bestX, axis=1, keepdims=True)
    return bestX
