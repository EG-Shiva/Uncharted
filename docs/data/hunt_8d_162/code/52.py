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
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    m = float(G.max())
    mask = G > m - tol
    cnt = mask.sum(axis=1).astype(int)
    # each contact counted once per row (symmetric); mask includes both sides but diagonal excluded
    return m, G, cnt

def _lp_polish(X, seconds, t_deadline=None):
    X = _norm(np.array(X, dtype=float))
    n, d = X.shape
    best = _max_cos(X)
    radius = 5e-3
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
        sel = cfull > m - max(20.0 * radius, 1e-9)
        I = iu0[0][sel]; J = iu0[1][sel]
        p = len(I)
        if p == 0:
            radius *= 0.5
            continue
        if p > 4000:
            # keep tightest
            cands = cfull[sel]
            idx = np.argpartition(-cands, 4000)[:4000]
            I = I[idx]; J = J[idx]
            p = 4000
            c = cands[idx]
        else:
            c = cfull[sel]
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([X[J], X[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((X.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
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
        if v < best:
            X, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return X

def _find_holes(Xf, k, rng, n_samp=50000):
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
            if float(v @ q) > 0.55:
                ok = False
                break
        if ok:
            picked.append(v.copy())
    while len(picked) < k:
        v = rng.randn(d); v /= np.linalg.norm(v)
        picked.append(v)
    H = np.array(picked)
    # local minimisation of max cosine to frozen code
    # projected gradient on LSE with beta=200
    beta = 200.0
    lr = 0.08
    for it in range(200):
        D = H @ Xf.T  # k x Nf
        m = D.max(axis=1, keepdims=True)
        W = np.exp(beta * (D - m))
        Ws = W.sum(axis=1, keepdims=True)
        G = (W @ Xf) / np.maximum(Ws, 1e-300)  # k x d weighted mean
        dot = (G * H).sum(axis=1, keepdims=True)
        Gt = G - dot * H
        cur_lr = lr * (1.0 - 0.7 * it / 200.0)
        H = H - cur_lr * Gt
        H = _norm(H)
        if it == 100:
            beta = 400.0
    return H

def _joint_lbfgs(H0, Xf, temps=(0.01, 0.003, 0.001), iters=150):
    k, d = H0.shape
    Nf = Xf.shape[0]
    iu = np.triu_indices(k, 1)
    has_pair = len(iu[0]) > 0
    Y = H0.copy().ravel()
    for T in temps:
        beta = 1.0 / T
        def fun(Yflat):
            Ym = Yflat.reshape(k, d)
            nm = np.linalg.norm(Ym, axis=1, keepdims=True)
            nm = np.maximum(nm, 1e-12)
            Xn = Ym / nm
            Df = Xn @ Xf.T  # k x Nf
            if has_pair:
                Dp_full = Xn @ Xn.T
                Dp = Dp_full[iu]
                m1 = float(Df.max()) if Df.size else -2.0
                m2 = float(Dp.max()) if Dp.size else -2.0
                m = m1 if m1 > m2 else m2
                Ef = np.exp((Df - m) / T)
                Ep = np.exp((Dp - m) / T)
                S = float(Ef.sum() + Ep.sum())
                val = m + T * np.log(max(S, 1e-300))
                # grads
                Gf = (Ef @ Xf) / S  # k x d
                # symmetric for free-free
                Esym = np.zeros((k, k))
                Esym[iu] = Ep
                Esym = Esym + Esym.T
                Gp = (Esym @ Xn) / S
                G = (Gf + Gp)
            else:
                m = float(Df.max())
                Ef = np.exp((Df - m) / T)
                S = float(Ef.sum())
                val = m + T * np.log(max(S, 1e-300))
                G = (Ef @ Xf) / S
            # chain through normalization
            dot = (G * Xn).sum(axis=1, keepdims=True)
            Gt = (G - dot * Xn) / nm
            return val, Gt.ravel()
        try:
            res = minimize(fun, Y, jac=True, method='L-BFGS-B',
                           options={'maxiter': iters, 'ftol': 1e-14, 'gtol': 1e-10, 'maxcor': 10})
            Y = res.x
        except Exception:
            break
    Ym = Y.reshape(k, d)
    return _norm(Ym)

def _smooth_refine(X, steps=200, beta=400.0, lr=0.008):
    X = _norm(X.copy())
    n, d = X.shape
    for it in range(steps):
        G = X @ X.T
        np.fill_diagonal(G, -10.0)
        m = float(G.max())
        E = np.exp(beta * (G - m))
        np.fill_diagonal(E, 0.0)
        s = float(E.sum())
        if not np.isfinite(s) or s <= 0:
            break
        P = E / s
        Gr = 2.0 * (P @ X)
        dot = (Gr * X).sum(axis=1, keepdims=True)
        Gt = Gr - dot * X
        mx = float(np.linalg.norm(Gt, axis=1).max())
        if mx < 1e-14 or not np.isfinite(mx):
            break
        cur = lr * (1.0 - 0.5 * it / steps) / max(mx, 1e-12)
        # cap step
        X = X - Gt * cur
        X = _norm(X)
    return X

def solve():
    t0 = time.time()
    deadline = t0 + 172.0
    hard_end = t0 + 178.0
    rng = np.random.RandomState(12345)
    Xb = None; sb = 1e9
    for fn in ["best.npy", "record.npy"]:
        try:
            v = np.load(fn).astype(float)
            if v.shape == (162, 8):
                v = _norm(v)
                s = _max_cos(v)
                if s < sb:
                    sb = s; Xb = v.copy()
        except Exception:
            pass
    if Xb is None:
        Xb = _norm(rng.randn(162, 8)); sb = _max_cos(Xb)
    X_best = Xb.copy(); s_best = float(sb)
    n, d = X_best.shape
    ks = [3, 4, 6, 8, 10]
    trial = 0
    # precompute contacts
    m0, G0, cnt0 = _contacts(X_best)
    while time.time() < t0 + 150.0:
        if time.time() > hard_end - 12:
            break
        trial += 1
        # refresh contacts every 5 trials from best
        if trial % 5 == 1:
            try:
                _, _, cnt0 = _contacts(X_best)
                _, Gcur = _contacts(X_best)[:2]
            except Exception:
                pass
        # rattler = min contacts
        rattler = int(np.argmin(cnt0))
        # choose k
        k = ks[rng.randint(len(ks))]
        # order by contacts desc
        order_desc = np.argsort(-cnt0, kind='stable')
        mode = trial % 3
        delset = set([rattler])
        if mode == 0:
            # rattler plus highest-contact points
            for idx in order_desc:
                if len(delset) >= k:
                    break
                delset.add(int(idx))
        elif mode == 1:
            # one spatial/contact-graph cluster + rattler
            # random seed: prefer high-contact
            if rng.rand() < 0.6:
                # pick among top 20
                cand = order_desc[:20]
                seed = int(cand[rng.randint(len(cand))])
            else:
                seed = int(rng.randint(n))
            if seed == rattler:
                seed = int(rng.randint(n))
            dots = (X_best @ X_best[seed])
            dots[seed] = 2.0
            dots[rattler] = -2.0  # exclude rattler from ranking, will add anyway
            nn = np.argsort(-dots)
            for idx in nn:
                if len(delset) >= k:
                    break
                if int(idx) == seed or True:
                    delset.add(int(idx))
            delset.add(seed)
            # trim to k keeping rattler
            delset = set(list(delset)[:k]) if len(delset) > k else delset
            if rattler not in delset:
                # replace last
                lst = list(delset)
                lst[-1] = rattler
                delset = set(lst)
            while len(delset) < k:
                delset.add(int(rng.randint(n)))
        else:
            # random mixture biased to high contacts
            while len(delset) < k:
                if rng.rand() < 0.5:
                    cand = order_desc[:30]
                    delset.add(int(cand[rng.randint(len(cand))]))
                else:
                    delset.add(int(rng.randint(n)))
        del_idx = np.array(sorted(delset))[:k]
        if len(del_idx) < k:
            # fill
            extra = []
            for i in range(n):
                if i not in delset:
                    extra.append(i)
                if len(delset) + len(extra) >= k:
                    break
            del_idx = np.array(sorted(list(delset) + extra[:k-len(delset)]))
        keep_mask = np.ones(n, dtype=bool); keep_mask[del_idx] = False
        Xf = X_best[keep_mask]
        # holes + local minim
        try:
            H0 = _find_holes(Xf, k, rng, n_samp=50000)
        except Exception:
            continue
        # joint L-BFGS on free points
        try:
            Hopt = _joint_lbfgs(H0, Xf, temps=(0.01, 0.003, 0.001), iters=150)
        except Exception:
            continue
        Xt = np.vstack([Xf, Hopt])
        # short smooth refinement unfrozen
        try:
            Xt = _smooth_refine(Xt, steps=150, beta=350.0, lr=0.010)
        except Exception:
            pass
        # trust-region LP polish 3-4 s
        remain = hard_end - time.time()
        if remain < 5:
            break
        lp_sec = 3.0
        if k >= 8:
            lp_sec = 4.0
        lp_sec = min(lp_sec, remain - 2.0)
        try:
            Xp = _lp_polish(Xt, seconds=lp_sec, t_deadline=hard_end - 1.0)
        except Exception:
            continue
        try:
            sp = _max_cos(Xp)
        except Exception:
            continue
        if sp < s_best:
            X_best = Xp.copy(); s_best = float(sp)
            try:
                _, _, cnt0 = _contacts(X_best)
            except Exception:
                pass
        # time check for loop
        if time.time() > t0 + 150.0:
            break
    return _norm(X_best)
