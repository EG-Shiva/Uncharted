import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _polish_trust(X, Rinit, max_iters, deadline, cap_p=2500, verbose=False):
    n, d = X.shape
    N = n * d
    iu0, iu1 = np.triu_indices(n, 1)
    cur = _max_cos(X)
    R = float(Rinit)
    it = 0
    # pre-allocate alphas for line search
    alphas = np.array([1.0, 0.6, 0.35, 0.18, 0.09, 0.04, 0.015, 0.005], dtype=float)
    while it < max_iters and time.time() < deadline:
        it += 1
        if R < 1e-12:
            break
        G = X @ X.T
        # extract upper
        c_all = G[iu0, iu1]
        m = float(c_all.max())
        # update cur if drifted (should equal m)
        cur = m
        margin = 12.0 * R + 2e-9
        # select active
        sel = c_all > (m - margin)
        cnt = int(sel.sum())
        if cnt < 200:
            # ensure at least 200 tightest
            k = min(200, c_all.size)
            idx = np.argpartition(-c_all, k - 1)[:k]
            sel = np.zeros_like(sel, dtype=bool)
            sel[idx] = True
            cnt = k
        if cnt > cap_p:
            # keep tightest cap_p
            # need indices of sel sorted
            # get all indices where sel, then pick top cap_p by c_all
            cand = np.flatnonzero(sel)
            cv = c_all[cand]
            part = np.argpartition(-cv, cap_p - 1)[:cap_p]
            nsel = np.zeros_like(sel, dtype=bool)
            nsel[cand[part]] = True
            sel = nsel
            cnt = cap_p
        I = iu0[sel]
        J = iu1[sel]
        csel = c_all[sel]
        p = cnt
        # build LP: min t s.t. xJ*dxI + xI*dxJ - t <= -c, xI*dxI==0, |dx|<=R
        try:
            rows = np.repeat(np.arange(p), 2 * d + 1)
            cols_I = (I[:, None] * d + np.arange(d)).ravel()
            cols_J = (J[:, None] * d + np.arange(d)).ravel()
            cols = np.empty(p * (2 * d + 1), dtype=np.int64)
            vals = np.empty(p * (2 * d + 1), dtype=float)
            # block I uses x[J]
            cols[0::(2*d+1)] = 0  # placeholder, we fill vectorized below
            # vectorized fill
            cI = np.concatenate([I[:, None] * d + np.arange(d)], axis=1).ravel() if False else None
            # simpler: build cols arrays
            col_block_I = (I[:, None] * d + np.arange(d))  # p x d
            col_block_J = (J[:, None] * d + np.arange(d))
            cols_arr = np.concatenate([col_block_I, col_block_J, np.full((p, 1), N)], axis=1).ravel()
            vals_arr = np.concatenate([X[J], X[I], -np.ones((p, 1))], axis=1).ravel()
            A_ub = coo_matrix((vals_arr, (rows, cols_arr)), shape=(p, N + 1)).tocsr()
            # A_eq
            erows = np.repeat(np.arange(n), d)
            ecols = np.arange(N)
            A_eq = coo_matrix((X.ravel(), (erows, ecols)), shape=(n, N + 1)).tocsr()
            obj = np.zeros(N + 1)
            obj[-1] = 1.0
            bnds = [(-R, R)] * N + [(None, None)]
            res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=bnds, method="highs")
        except Exception:
            R *= 0.5
            continue
        if res.status != 0:
            R *= 0.5
            continue
        dx = res.x[:-1].reshape(n, d)
        t_pred = float(res.x[-1])
        # predicted gain
        pred_gain = m - t_pred
        if pred_gain < 1e-13:
            # no descent direction for this R / active set -> shrink R to focus
            R *= 0.5
            # if R very small and still no gain, break (local opt)
            if R < 5e-8:
                # try expanding active set? break
                pass
            continue
        # line search along dx
        best_v = m
        best_a = 0.0
        # evaluate alphas
        # quick loop
        for a in alphas:
            Y = X + a * dx
            # project + normalize
            nr = np.linalg.norm(Y, axis=1, keepdims=True)
            nr = np.maximum(nr, 1e-12)
            Y = Y / nr
            G2 = Y @ Y.T
            np.fill_diagonal(G2, -2.0)
            v = float(G2.max())
            if v < best_v:
                best_v = v
                best_a = float(a)
                # early exit if good? continue to find smaller
        if best_v < m - 1e-15:
            # accept
            Y = X + best_a * dx
            Y /= np.linalg.norm(Y, axis=1, keepdims=True)
            X = Y
            cur = best_v
            # adapt R: enlarge if full step was best, else keep/shrink slightly
            if best_a >= 0.9:
                R = min(R * 1.4, 0.008)
            else:
                # step truncated by curvature, reduce R a bit
                R = max(R * 0.9, 1e-7)
        else:
            R *= 0.5
    return X, _max_cos(X)


def solve():
    t0 = time.time()
    LIMIT = 172.0
    deadline_final = t0 + LIMIT
    rng = np.random.RandomState(0)

    n, d = 219, 6
    bestX = None
    best_val = 1e9
    for fn in ["best.npy", "record.npy"]:
        try:
            v = np.load(fn)
            v = np.asarray(v, dtype=float)
            if v.shape == (n, d):
                v = v / np.linalg.norm(v, axis=1, keepdims=True)
                c = _max_cos(v)
                if c < best_val:
                    best_val = c
                    bestX = v.copy()
        except Exception:
            pass
    if bestX is None:
        bestX = rng.randn(n, d)
        bestX /= np.linalg.norm(bestX, axis=1, keepdims=True)
        best_val = _max_cos(bestX)

    X = bestX.copy()
    cur = best_val

    # Phase A: squeeze current basin
    try:
        Xp, vp = _polish_trust(X, Rinit=8e-4, max_iters=250, deadline=t0 + 45.0)
        if vp < cur:
            X, cur = Xp, vp
            bestX, best_val = X.copy(), cur
    except Exception:
        pass

    # Phase B: basin hopping with targeted kicks
    # compute time for hopping: until ~ LIMIT-25
    hop_deadline = t0 + LIMIT - 25.0
    trial = 0
    # sigmas and kick sizes
    while time.time() < hop_deadline:
        trial += 1
        if time.time() > hop_deadline:
            break
        # identify jammed points using tight threshold
        try:
            G = X @ X.T
            np.fill_diagonal(G, -1.0)
            # per-point worst
            worst = G.max(axis=1)
            order = np.argsort(-worst)
        except Exception:
            order = np.arange(n)
        # choose mode
        mode = trial % 5
        Y = bestX.copy()
        if mode == 0:
            # global small kick
            sigma = 0.008
            pert = rng.randn(n, d)
            proj = (pert * Y).sum(axis=1, keepdims=True) * Y
            pert -= proj
            Y = Y + sigma * pert
        elif mode == 1:
            sigma = 0.02
            k = 18
            idx = order[:k]
            pert = rng.randn(n, d) * 0.002
            pert2 = rng.randn(k, d)
            proj2 = (pert2 * Y[idx]).sum(axis=1, keepdims=True) * Y[idx]
            pert2 -= proj2
            pert[idx] += sigma * pert2 / max(1.0, np.linalg.norm(pert2, axis=1).max())
            # scale correctly: just add sigma*pert2 normalized per point?
            # redo: add sigma * unit tangent
            # already approx; re-do simple:
            # use fresh
            Y = bestX.copy()
            P = rng.randn(k, d)
            P -= (P * Y[idx]).sum(axis=1, keepdims=True) * Y[idx]
            Pn = np.linalg.norm(P, axis=1, keepdims=True) + 1e-12
            P /= Pn
            Y[idx] = Y[idx] + sigma * P
            # tiny noise elsewhere
            Q = rng.randn(n, d) * 0.001
            Q -= (Q * Y).sum(axis=1, keepdims=True) * Y
            Y = Y + Q
        elif mode == 2:
            sigma = 0.035
            k = 8
            idx = order[:k]
            Y = bestX.copy()
            P = rng.randn(k, d)
            P -= (P * Y[idx]).sum(axis=1, keepdims=True) * Y[idx]
            Pn = np.linalg.norm(P, axis=1, keepdims=True) + 1e-12
            P /= Pn
            Y[idx] = Y[idx] + sigma * P
        elif mode == 3:
            # relocate single worst point randomly
            idx = int(order[0])
            v = rng.randn(d)
            v /= np.linalg.norm(v)
            Y[idx] = v * 0.5 + Y[idx] * 0.5
        else:
            sigma = 0.012
            pert = rng.randn(n, d)
            proj = (pert * Y).sum(axis=1, keepdims=True) * Y
            pert -= proj
            Y = Y + sigma * pert
        Y /= np.linalg.norm(Y, axis=1, keepdims=True)
        # quick polish of hopped point (limited iters)
        # adapt Rinit by sigma
        Rinit = 6e-4 if trial % 2 == 0 else 1.2e-3
        # per-trial time cap ~6s
        per_deadline = min(hop_deadline, time.time() + 6.0)
        try:
            Yp, vp = _polish_trust(Y, Rinit=Rinit, max_iters=60, deadline=per_deadline)
        except Exception:
            continue
        if vp < best_val:
            best_val, bestX = vp, Yp.copy()
            X = Yp.copy()
            cur = vp
        elif vp < cur:
            X = Yp.copy()
            cur = vp
        # occasionally restart from best if X drifts worse
        if trial % 4 == 0:
            X = bestX.copy()
            cur = best_val
        if trial > 40:
            break

    # Phase C: final squeeze with remaining time
    remain_deadline = deadline_final - 2.0
    if time.time() < remain_deadline:
        try:
            Xf, vf = _polish_trust(bestX, Rinit=4e-4, max_iters=400, deadline=remain_deadline)
            if vf < best_val:
                bestX, best_val = Xf, vf
        except Exception:
            pass

    bestX /= np.linalg.norm(bestX, axis=1, keepdims=True)
    return bestX
