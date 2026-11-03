import time
import numpy as np
from scipy.optimize import minimize, linprog
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _build_e8():
    pts = []
    # 112: two coords +-1, rest 0
    for i in range(8):
        for j in range(i+1, 8):
            for si in (1.0, -1.0):
                for sj in (1.0, -1.0):
                    v = np.zeros(8)
                    v[i] = si
                    v[j] = sj
                    pts.append(v)
    # 128: all +-0.5 with even number of minus
    for mask in range(256):
        v = np.ones(8)
        nm = 0
        for k in range(8):
            if (mask >> k) & 1:
                v[k] = -0.5
                nm += 1
            else:
                v[k] = 0.5
        if nm % 2 == 0:
            pts.append(v.copy())
    E = np.array(pts, float)  # (240,8)
    E /= np.linalg.norm(E, axis=1, keepdims=True)
    return E

def _lp_polish(X, seconds, t_start=None, budget=1e9):
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    n, d = X.shape
    best = _max_cos(X)
    radius = 1e-3
    if t_start is None:
        t_start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - t_start < seconds and time.time() - t_start < budget:
        if radius < 1e-14:
            break
        cfull = (X @ X.T)[iu]
        m = float(cfull.max())
        sel = cfull > m - max(20.0 * radius, 1e-9)
        cnt = int(sel.sum())
        if cnt == 0:
            radius *= 0.5
            continue
        if cnt > 3000:
            idx = np.argpartition(-cfull, 3000)[:3000]
            mask = np.zeros_like(sel)
            mask[idx] = True
            sel = mask
        I = iu[0][sel]; J = iu[1][sel]
        p = len(I)
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([X[J], X[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((X.ravel(), (np.repeat(np.arange(n), d),
                                       np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-cfull[sel],
                          A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                          method="highs")
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
            X, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return X

def solve():
    t_start = time.time()
    TIME_BUDGET = 175.0
    n, d = 162, 8
    rng = np.random.default_rng(8162001)

    # load incumbent
    try:
        xb = np.load("best.npy").astype(float)
    except Exception:
        xb = None
    try:
        xr = np.load("record.npy").astype(float)
    except Exception:
        xr = None
    cands0 = []
    if xb is not None and xb.shape == (162, 8):
        cands0.append(xb / np.linalg.norm(xb, axis=1, keepdims=True))
    if xr is not None and xr.shape == (162, 8):
        cands0.append(xr / np.linalg.norm(xr, axis=1, keepdims=True))
    if cands0:
        vals0 = [_max_cos(c) for c in cands0]
        bi = int(np.argmin(vals0))
        X_best = cands0[bi].copy()
        best_val = float(vals0[bi])
    else:
        a = rng.standard_normal(size=(162, 8))
        a /= np.linalg.norm(a, axis=1, keepdims=True)
        X_best = a
        best_val = _max_cos(a)

    E8 = _build_e8()  # (240,8)

    def greedy_delete(rng_local):
        X = E8.copy()
        # delete 78
        for _ in range(240 - 162):
            G = X @ X.T
            np.fill_diagonal(G, -2.0)
            per_max = G.max(axis=1)
            m = float(per_max.max())
            cand = np.where(per_max >= m - 1e-12)[0]
            if len(cand) == 1:
                choice = int(cand[0])
            else:
                counts = (G[cand] > m - 1e-9).sum(axis=1)
                maxc = counts.max()
                tied = cand[counts == maxc]
                r = rng_local.random()
                if r < 0.30 and len(cand) > 2:
                    order = np.argsort(-counts)
                    k = min(5, len(cand))
                    pool = cand[order[:k]]
                    choice = int(rng_local.choice(pool))
                elif len(tied) == 1:
                    # occasional random among top to diversify
                    if rng_local.random() < 0.15 and len(cand) > 1:
                        choice = int(rng_local.choice(cand))
                    else:
                        choice = int(tied[0])
                else:
                    choice = int(rng_local.choice(tied))
            X = np.delete(X, choice, axis=0)
        return X

    # build starts
    starts = []
    # 6 greedy runs
    for gi in range(6):
        rng_g = np.random.default_rng(9000 + gi)
        try:
            s = greedy_delete(rng_g)
        except Exception:
            continue
        starts.append(s)
        if time.time() - t_start > 20:
            break
    # 3 random subsets
    for ri in range(3):
        idx = rng.choice(240, 162, replace=False)
        starts.append(E8[idx].copy())

    SHIFT = 0.5

    def make_eg(alpha):
        def eg_fn(Yf):
            Y = Yf.reshape(n, d)
            nr = np.linalg.norm(Y, axis=1, keepdims=True)
            nr = np.maximum(nr, 1e-12)
            X = Y / nr
            G = X @ X.T
            np.fill_diagonal(G, -2.0)
            M = alpha * (G - SHIFT)
            np.clip(M, -700, 200, out=M)
            Emat = np.exp(M)
            E = float(Emat.sum() * 0.5)
            GradX = alpha * (Emat @ X)
            dot = np.sum(GradX * X, axis=1, keepdims=True)
            GradY = (GradX - dot * X) / nr
            return E, GradY.ravel()
        return eg_fn

    def staged_opt(X_init, alphas, iters):
        Yf = X_init.ravel().copy()
        for alpha, mi in zip(alphas, iters):
            if time.time() - t_start > TIME_BUDGET - 40:
                break
            fn = make_eg(float(alpha))
            try:
                res = minimize(fn, Yf, jac=True, method="L-BFGS-B",
                               options={"maxiter": int(mi), "ftol": 1e-12,
                                        "gtol": 1e-5, "maxfun": int(mi) * 20})
                Yf = res.x
            except Exception:
                break
        Y = Yf.reshape(n, d)
        Y /= np.linalg.norm(Y, axis=1, keepdims=True)
        return Y

    def greedy_replace(Xcur, idx_list, rng_local, nsamp=3000):
        Xnew = Xcur.copy()
        for w in idx_list:
            cand = rng_local.standard_normal(size=(nsamp, d))
            cand /= np.linalg.norm(cand, axis=1, keepdims=True)
            others = np.delete(Xnew, w, axis=0)
            scores = (cand @ others.T).max(axis=1)
            Xnew[w] = cand[np.argmin(scores)]
        return Xnew

    ALPHAS = [30.0, 100.0, 300.0, 900.0]
    ITERS = [30, 30, 40, 50]

    for si, s0 in enumerate(starts):
        if time.time() - t_start > 128:
            break
        # jitter to break exact symmetry
        X_init = s0 + 1e-4 * rng.standard_normal(size=s0.shape)
        X_init /= np.linalg.norm(X_init, axis=1, keepdims=True)
        X_opt = staged_opt(X_init, ALPHAS, ITERS)
        v = _max_cos(X_opt)
        # thin / swap if rattler / imbalance
        try:
            G = X_opt @ X_opt.T
            np.fill_diagonal(G, -2.0)
            m = float(G.max())
            thr = m - 1e-9
            cont = (G > thr).sum(axis=1)
            mn = int(cont.min()); mx = int(cont.max())
            need_swap = (mn == 0) or (mx > 24) or (mx - mn > 18)
        except Exception:
            need_swap = False
        if need_swap and time.time() - t_start < TIME_BUDGET - 45:
            # replace rattlers + most overloaded
            G2 = X_opt @ X_opt.T
            np.fill_diagonal(G2, -2.0)
            m2 = float(G2.max())
            cont2 = (G2 > m2 - 1e-9).sum(axis=1)
            ratt = np.where(cont2 == 0)[0].tolist()
            overload = [int(np.argmax(cont2))]
            torep = []
            torep.extend(ratt[:2])
            for o in overload:
                if o not in torep:
                    torep.append(o)
            torep = torep[:2]
            if torep:
                X_rep = greedy_replace(X_opt, torep, rng, nsamp=3000)
                X_opt2 = staged_opt(X_rep, [100.0, 300.0, 900.0], [25, 30, 35])
                v2 = _max_cos(X_opt2)
                if v2 < v:
                    X_opt, v = X_opt2, v2
        if v < best_val - 1e-12:
            X_best = X_opt
            best_val = v
            # quick lock-in polish if time allows
            if time.time() - t_start < 110:
                X_lp = _lp_polish(X_best, 2.0, t_start=t_start, budget=TIME_BUDGET - 5)
                v2 = _max_cos(X_lp)
                if v2 < best_val:
                    X_best, best_val = X_lp, v2
        # also if close, try short LP to see true potential (sparingly)
        elif v < best_val + 0.002 and (si % 3 == 0) and time.time() - t_start < 100:
            X_lp = _lp_polish(X_opt, 1.5, t_start=t_start, budget=TIME_BUDGET - 5)
            v2 = _max_cos(X_lp)
            if v2 < best_val - 1e-12:
                X_best, best_val = X_lp, v2

    # a few basin hops on current best to squeeze further if E8 did not beat
    trial = 0
    while time.time() - t_start < 138 and trial < 6:
        trial += 1
        if time.time() - t_start > TIME_BUDGET - 40:
            break
        X_try = X_best + (0.02 if trial % 2 == 1 else 0.06) * rng.standard_normal(size=(n, d))
        X_try /= np.linalg.norm(X_try, axis=1, keepdims=True)
        X_o = staged_opt(X_try, [45.0, 130.0, 350.0], [30, 30, 30])
        v = _max_cos(X_o)
        if v < best_val - 1e-12:
            X_best, best_val = X_o, v

    remain = TIME_BUDGET - (time.time() - t_start) - 2.0
    if remain > 4:
        # cap final polish to 35s
        remain = min(remain, 35.0)
        X_best = _lp_polish(X_best, remain, t_start=t_start, budget=TIME_BUDGET - 2)

    X_best /= np.linalg.norm(X_best, axis=1, keepdims=True)
    X_best = np.nan_to_num(X_best, nan=0.0, posinf=0.0, neginf=0.0)
    nr = np.linalg.norm(X_best, axis=1, keepdims=True)
    nr[nr == 0] = 1.0
    X_best = X_best / nr
    return X_best
