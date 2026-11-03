import time
import numpy as np
from scipy.optimize import linprog, minimize
from scipy.sparse import coo_matrix

def _norm(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n = np.maximum(n, 1e-12)
    return X / n

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _contacts(X, tol=1e-9):
    n = X.shape[0]
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    m = float(G.max())
    rowmax = G.max(axis=1)
    cnt = (G > m - tol).sum(axis=1)
    # rattler: zero contacts, else most isolated
    z = np.where(cnt == 0)[0]
    if len(z) > 0:
        rattler = int(z[np.argmin(rowmax[z])])
    else:
        rattler = int(np.argmin(rowmax))
    order_high = np.argsort(-cnt)
    tied = np.where(rowmax >= m - 1e-12)[0]
    return m, rowmax, cnt, rattler, order_high, tied, G

def _find_holes(Xf, k, rng, n_samp=50000, sep=0.70):
    d = Xf.shape[1]
    C = rng.randn(n_samp, d)
    C = _norm(C)
    S = C @ Xf.T
    scores = S.max(axis=1)
    order = np.argsort(scores)
    picked = []
    for idx in order:
        if len(picked) >= k:
            break
        v = C[idx]
        ok = True
        for q in picked:
            if float(v @ q) > sep:
                ok = False
                break
        if ok:
            picked.append(v.copy())
    while len(picked) < k:
        v = rng.randn(d)
        v /= np.linalg.norm(v)
        picked.append(v)
    Y = np.array(picked)
    # local minimisation of max cosine to frozen code
    tau = 0.005
    lr = 0.08
    m = Xf.shape[0]
    for it in range(80):
        D = Y @ Xf.T  # k x m
        dm = D.max(axis=1, keepdims=True)
        E = np.exp((D - dm) / tau)
        W = E / E.sum(axis=1, keepdims=True)
        Gr = W @ Xf  # k x d
        dot = (Gr * Y).sum(axis=1, keepdims=True)
        Gt = Gr - dot * Y
        Y = _norm(Y - lr * Gt)
        lr *= 0.995
    return Y

def _joint_lbfgs(Xf, Yinit, taus=(0.01, 0.003, 0.001), iters=150):
    k, d = Yinit.shape
    mf = Xf.shape[0]
    iu = np.triu_indices(k, 1) if k > 1 else (np.array([], dtype=int), np.array([], dtype=int))
    has_cc = k > 1
    Z0 = Yinit.copy().ravel()

    def make_fun(tau):
        def fun(zflat):
            Z = zflat.reshape(k, d)
            r = np.sqrt((Z * Z).sum(axis=1, keepdims=True))
            r = np.maximum(r, 1e-12)
            Y = Z / r
            Dff = Y @ Xf.T
            dmax = float(Dff.max())
            if has_cc:
                Dcc = (Y @ Y.T)[iu]
                if Dcc.size > 0:
                    dmax = max(dmax, float(Dcc.max()))
            # shifted exps
            Eff = np.exp((Dff - dmax) / tau)
            sumE = float(Eff.sum())
            if has_cc and iu[0].size > 0:
                Ecc = np.exp((Dcc - dmax) / tau)
                sumE += float(Ecc.sum())
            else:
                Ecc = None
            val = dmax + tau * np.log(max(sumE, 1e-300))
            Wff = Eff / sumE
            Gr = Wff @ Xf  # k x d
            if has_cc and iu[0].size > 0:
                Wcc = Ecc / sumE
                Ws = np.zeros((k, k))
                Ws[iu] = Wcc
                Ws = Ws + Ws.T
                Gr = Gr + Ws @ Y
            # chain to Z
            dot = (Gr * Y).sum(axis=1, keepdims=True)
            Gt = Gr - dot * Y
            Gz = Gt / r
            return val, Gz.ravel()
        return fun
    z = Z0.copy()
    for tau in taus:
        fun = make_fun(tau)
        try:
            res = minimize(fun, z, method='L-BFGS-B', jac=True,
                           options={'maxiter': iters, 'maxfun': iters * 4, 'ftol': 1e-14, 'gtol': 1e-10})
            z = res.x
        except Exception:
            break
    Z = z.reshape(k, d)
    Y = _norm(Z)
    return Y

def _smooth_refine(X, steps=250, tau=0.004, lr=0.015):
    n, d = X.shape
    for it in range(steps):
        G = X @ X.T
        np.fill_diagonal(G, -10.0)
        gm = float(G.max())
        E = np.exp((G - gm) / tau)
        np.fill_diagonal(E, 0.0)
        s = float(E.sum())
        if not np.isfinite(s) or s <= 0:
            break
        H = E @ X
        WG = (E * G).sum(axis=1)
        Grad = (H - WG[:, None] * X) / s
        # tangential already approx
        dot = (Grad * X).sum(axis=1, keepdims=True)
        Grad = Grad - dot * X
        X = _norm(X - lr * Grad)
    return X

def _lp_polish(X, seconds, t_deadline=None):
    X = _norm(np.array(X, dtype=float))
    n, d = X.shape
    best = _max_cos(X)
    radius = 1e-3
    start = time.time()
    iu0 = np.triu_indices(n, 1)
    while time.time() - start < seconds:
        if t_deadline is not None and time.time() > t_deadline:
            break
        if radius < 1e-14:
            break
        Gfull = X @ X.T
        cfull = Gfull[iu0]
        m = float(cfull.max())
        if m < best:
            best = m
        sel = cfull > m - max(20 * radius, 1e-9)
        I = iu0[0][sel]; J = iu0[1][sel]
        p = len(I)
        if p == 0:
            break
        if p > 3500:
            idx = np.argpartition(-cfull[sel], 3500)[:3500]
            I = I[idx]; J = J[idx]
            p = len(I)
            c = cfull[sel][idx]
        else:
            c = cfull[sel]
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([X[J], X[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((X.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))),
                              shape=(n, nv)).tocsr()
        except Exception:
            break
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                          method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = X + res.x[:-1].reshape(n, d)
        y = _norm(y)
        v = _max_cos(y)
        if v < best - 1e-15:
            X, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return X

def solve():
    t0 = time.time()
    deadline = t0 + 172.0
    cav_deadline = t0 + 150.0
    rng = np.random.RandomState(12345)
    Xb = None
    sb = 1e9
    for fn in ["best.npy", "record.npy"]:
        try:
            v = np.load(fn).astype(float)
            if v.shape == (162, 8):
                v = _norm(v)
                s = _max_cos(v)
                if s < sb:
                    sb = s
                    Xb = v.copy()
        except Exception:
            pass
    if Xb is None:
        Xb = _norm(rng.randn(162, 8))
        sb = _max_cos(Xb)
    X_best = Xb.copy()
    s_best = float(sb)
    n, d = 162, 8
    ks = [3, 4, 6, 8, 10, 6, 4, 8, 3, 10, 4, 6]
    trial = 0
    while time.time() < cav_deadline and time.time() < deadline - 18:
        trial += 1
        try:
            m, rowmax, cnt, rattler, order_high, tied, G = _contacts(X_best)
        except Exception:
            continue
        k = ks[(trial - 1) % len(ks)]
        mode = (trial - 1) % 3
        dels = set()
        dels.add(int(rattler))
        if mode == 0:
            # rattler + highest-contact points
            for idx in order_high:
                if len(dels) >= k:
                    break
                dels.add(int(idx))
        elif mode == 1:
            # one spatial cluster + rattler
            # seed among high-contact / tied
            pool = order_high[:20].tolist()
            seed = int(pool[rng.randint(len(pool))])
            if seed in dels:
                # pick another
                for c in pool:
                    if int(c) not in dels:
                        seed = int(c)
                        break
            sims = G[seed].copy()
            sims[seed] = -2.0
            # nearest by dot
            order_near = np.argsort(-sims)
            need = k - 1
            cl = []
            for c in order_near:
                if int(c) == rattler:
                    continue
                cl.append(int(c))
                if len(cl) >= need:
                    break
            for c in cl:
                dels.add(c)
            # fill if overlap
            ri = 0
            while len(dels) < k:
                c = int(order_high[ri % len(order_high)])
                dels.add(c)
                ri += 1
        else:
            # random mixture always including rattler, bias to high contacts
            # half random, half high-contact random
            while len(dels) < k:
                if rng.rand() < 0.5:
                    c = int(rng.randint(n))
                else:
                    c = int(order_high[rng.randint(min(30, n))])
                dels.add(c)
        dels = sorted(list(dels))[:k]
        if len(dels) < k:
            while len(dels) < k:
                c = int(rng.randint(n))
                if c not in dels:
                    dels.append(c)
        k = len(dels)
        keep_mask = np.ones(n, dtype=bool)
        keep_mask[dels] = False
        Xf = X_best[keep_mask]
        # hole init
        try:
            Y0 = _find_holes(Xf, k, rng, n_samp=50000, sep=0.65)
            Yopt = _joint_lbfgs(Xf, Y0, taus=(0.01, 0.003, 0.001), iters=150)
        except Exception:
            continue
        Xt = X_best.copy()
        Xt[dels] = _norm(Yopt)
        # short smooth refine unfrozen
        try:
            Xt = _smooth_refine(Xt, steps=220, tau=0.004, lr=0.015)
        except Exception:
            pass
        # LP polish 3-5 s
        remain = min(cav_deadline - time.time(), deadline - time.time() - 4)
        if remain < 2.5:
            # evaluate without LP
            try:
                s_try = _max_cos(Xt)
            except Exception:
                continue
            if s_try < s_best:
                X_best, s_best = _norm(Xt), s_try
            continue
        budget = min(4.0, max(3.0, remain / 3.0))
        try:
            Xp = _lp_polish(Xt, budget, t_deadline=deadline - 2)
            sp = _max_cos(Xp)
        except Exception:
            continue
        if sp < s_best and np.all(np.isfinite(Xp)):
            X_best, s_best = Xp, sp
        # also try immediate second polish if improved a lot? skip to save time
        if time.time() > cav_deadline:
            break
    # final polish of best
    if time.time() < deadline - 3:
        budget = min(12.0, deadline - time.time() - 1.5)
        if budget > 2:
            try:
                Xp = _lp_polish(X_best, budget, t_deadline=deadline - 1)
                sp = _max_cos(Xp)
                if sp < s_best:
                    X_best, s_best = Xp, sp
            except Exception:
                pass
    X_best = _norm(X_best)
    return X_best
