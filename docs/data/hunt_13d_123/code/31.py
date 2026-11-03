import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())


def _lp_polish(X, seconds, p_cap=3000):
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    n, d = X.shape
    best = _max_cos(X)
    best_x = X.copy()
    radius = 1e-3
    start = time.time()
    iu0, iu1 = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        G = X @ X.T
        c_all = G[iu0, iu1]
        m = c_all.max()
        window = max(20 * radius, 1e-9)
        sel = c_all > m - window
        cnt = int(np.count_nonzero(sel))
        if cnt == 0:
            break
        if cnt > p_cap:
            idx = np.where(sel)[0]
            part = np.argpartition(-c_all[idx], p_cap - 1)[:p_cap]
            keep = np.zeros_like(sel)
            keep[idx[part]] = True
            sel = keep
            cnt = p_cap
        I = iu0[sel]
        J = iu1[sel]
        csel = c_all[sel]
        p = cnt
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate(
            [(I[:, None] * d + np.arange(d)).ravel(),
             (J[:, None] * d + np.arange(d)).ravel(),
             np.full(p, n * d)], axis=0
        )
        # need per-row layout: [I-block, J-block, t]
        # rebuild correctly row-major
        cI = (I[:, None] * d + np.arange(d))
        cJ = (J[:, None] * d + np.arange(d))
        cols2 = np.empty((p, 2 * d + 1), dtype=np.int64)
        cols2[:, :d] = cI
        cols2[:, d:2 * d] = cJ
        cols2[:, 2 * d] = n * d
        cols = cols2.ravel()
        vals2 = np.empty((p, 2 * d + 1))
        vals2[:, :d] = X[J]
        vals2[:, d:2 * d] = X[I]
        vals2[:, 2 * d] = -1.0
        vals = vals2.ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            erows = np.repeat(np.arange(n), d)
            ecols = np.arange(n * d)
            A_eq = coo_matrix((X.ravel(), (erows, ecols)), shape=(n, nv)).tocsr()
            obj = np.zeros(nv)
            obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = X + res.x[:-1].reshape(n, d)
        yn = np.linalg.norm(y, axis=1, keepdims=True)
        yn[yn == 0] = 1.0
        y = y / yn
        v = _max_cos(y)
        if v < best:
            best = v
            best_x = y.copy()
            X = y
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
        if time.time() - start >= seconds:
            break
    return best_x


def _stress_residuals(X, thresh=1e-6):
    n, d = X.shape
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    m = G.max()
    I, J = np.where(np.triu(G >= m - thresh, 1))
    neigh = [[] for _ in range(n)]
    for a, b in zip(I.tolist(), J.tolist()):
        neigh[a].append(b)
        neigh[b].append(a)
    deg = np.array([len(v) for v in neigh], dtype=float)
    res = np.zeros(n)
    for i in range(n):
        nb = neigh[i]
        k = len(nb)
        if k == 0:
            res[i] = 1e9
            continue
        V = np.empty((k, d))
        for t, j in enumerate(nb):
            c = float(np.dot(X[i], X[j]))
            V[t] = X[j] - c * X[i]
        # Gilbert algorithm for dist(0, conv(V))
        norms = np.einsum('ij,ij->i', V, V)
        s = V[int(np.argmin(norms))].copy()
        for _ in range(300):
            dots = V @ s
            jstar = int(np.argmin(dots))
            vs = V[jstar]
            sn2 = float(s @ s)
            if float(vs @ s) >= sn2 - 1e-14:
                break
            dvec = s - vs
            dn2 = float(dvec @ dvec)
            if dn2 < 1e-30:
                break
            t = float((s @ dvec)) / dn2
            if t < 0:
                t = 0.0
            elif t > 1:
                t = 1.0
            s = (1 - t) * s + t * vs
            if t == 0.0:
                break
        res[i] = float(np.linalg.norm(s))
    return res, deg, m


def _repack_patch(Y0, F, budget, rng):
    # penalized softmax smooth-max descent against frozen boundary
    m = Y0.shape[0]
    Y = Y0.copy()
    start = time.time()
    best_Y = Y.copy()
    # initial true patch score
    def patch_score(Ye):
        a = Ye @ Ye.T
        np.fill_diagonal(a, -2.0)
        v1 = a.max() if m > 1 else -2.0
        v2 = (Ye @ F.T).max() if F.shape[0] > 0 else -2.0
        return max(float(v1), float(v2))
    best_s = patch_score(Y)
    T0, T1 = 0.02, 0.002
    lr0, lr1 = 0.35, 0.05
    it = 0
    # estimate steps: each step ~ O(m^2 d + m (n-m) d) ~ small, allow many
    while True:
        elapsed = time.time() - start
        if elapsed >= budget:
            break
        frac = min(elapsed / max(budget, 1e-9), 1.0)
        T = T0 + (T1 - T0) * frac
        lr = lr0 + (lr1 - lr0) * frac
        # a few inner steps per time check
        for _ in range(25):
            M1 = Y @ Y.T
            M2 = Y @ F.T
            # shift for stability
            cur_max = max(float(M1.max()) if m > 1 else -1e9, float(M2.max()) if F.shape[0] > 0 else -1e9)
            E1 = np.exp((M1 - cur_max) / T)
            np.fill_diagonal(E1, 0.0)
            E2 = np.exp((M2 - cur_max) / T)
            Z = E1.sum() / 2.0 + E2.sum() + 1e-30
            # euclidean grad of smooth max
            Gmat = (E1 @ Y + E2 @ F) / Z
            # tangential projection
            dot = (Gmat * Y).sum(axis=1, keepdims=True)
            Gt = Gmat - dot * Y
            # normalize step by grad norm to stabilize
            gn = np.linalg.norm(Gt) + 1e-12
            step = lr / (1.0 + gn)
            Y = Y - step * Gt
            Y /= np.linalg.norm(Y, axis=1, keepdims=True)
        s = patch_score(Y)
        if s < best_s:
            best_s = s
            best_Y = Y.copy()
        it += 1
        if time.time() - start >= budget:
            break
    return best_Y


def solve():
    t0 = time.time()
    LIMIT = 176.0
    np.random.seed(0)
    n, d = 123, 13
    X0 = None
    best0 = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=float)
            if a.shape == (n, d):
                a = a / np.linalg.norm(a, axis=1, keepdims=True)
                v = _max_cos(a)
                if v < best0:
                    best0 = v
                    X0 = a.copy()
        except Exception:
            pass
    if X0 is None:
        rng0 = np.random.default_rng(0)
        X0 = rng0.normal(size=(n, d))
        X0 /= np.linalg.norm(X0, axis=1, keepdims=True)
        best0 = _max_cos(X0)
    cur = X0.copy()
    cur_val = float(best0)
    # brief initial tighten (2s) if time allows
    if time.time() - t0 < LIMIT - 60:
        try:
            cand = _lp_polish(cur, min(2.0, LIMIT - (time.time() - t0) - 55))
            v = _max_cos(cand)
            if v < cur_val:
                cur = cand
                cur_val = v
        except Exception:
            pass
        if time.time() - t0 >= LIMIT - 55:
            return cur / np.linalg.norm(cur, axis=1, keepdims=True)
    # stress analysis on current best
    try:
        resid, deg, gmax = _stress_residuals(cur, thresh=1e-6)
    except Exception:
        return cur / np.linalg.norm(cur, axis=1, keepdims=True)
    Gfull = cur @ cur.T
    np.fill_diagonal(Gfull, -2.0)
    # rattler: minimal contacts at tight 1e-9 level
    tight = Gfull >= gmax - 1e-9
    contacts_tight = tight.sum(axis=1)
    rattler = int(np.argmin(contacts_tight))
    # nearest neighbours of rattler in angle (largest cosine)
    sim_r = (cur @ cur[rattler].copy())
    sim_r[rattler] = -2.0
    nn_rattler = np.argsort(-sim_r)
    # candidate seeds: high residual (unbalanced / low stress), then overload
    order_res = np.argsort(-resid)  # descending
    order_res = [i for i in order_res if i != rattler]
    order_deg = np.argsort(-deg)
    order_deg = [i for i in order_deg if i != rattler]
    seeds = []
    if len(order_res) > 0:
        seeds.append(order_res[0])
    if len(order_res) > 1:
        seeds.append(order_res[1])
    if len(order_deg) > 0:
        cand = order_deg[0]
        if cand not in seeds:
            seeds.append(cand)
        else:
            for c in order_deg[1:]:
                if c not in seeds:
                    seeds.append(c)
                    break
    seeds = seeds[:3]
    # ensure 3 seeds
    k = 0
    while len(seeds) < 3 and k < n:
        if k != rattler and k not in seeds:
            seeds.append(k)
        k += 1
    patch_size_target = 14
    for si, seed in enumerate(seeds[:3]):
        if time.time() - t0 > LIMIT - 8:
            break
        # build patch: rattler + 5 nearest of rattler + seed neighbourhood
        sim_s = cur @ cur[seed]
        sim_s[seed] = -2.0
        nn_seed = np.argsort(-sim_s)
        patch = set()
        patch.add(rattler)
        for j in nn_rattler[:5]:
            patch.add(int(j))
        patch.add(int(seed))
        for j in nn_seed[:8]:
            patch.add(int(j))
        # adjust size to 10-18
        if len(patch) > 18:
            patch = list(patch)[:18]
        else:
            patch = list(patch)
            # pad with lowest-stress points if too small
            for c in order_res:
                if len(patch) >= patch_size_target:
                    break
                if c not in patch:
                    patch.append(int(c))
        if len(patch) < 10:
            for c in range(n):
                if len(patch) >= 10:
                    break
                if c not in patch:
                    patch.append(int(c))
        if len(patch) > 18:
            patch = patch[:18]
        patch = sorted(set(int(v) for v in patch))
        m = len(patch)
        if m < 10 or m > 18:
            continue
        mask = np.zeros(n, dtype=bool)
        mask[patch] = True
        F = cur[~mask].copy()
        # deterministic re-init of patch: seeded random + one hole-biased try
        rng = np.random.default_rng(1000 + si)
        Y0 = rng.normal(size=(m, d))
        Y0 /= np.linalg.norm(Y0, axis=1, keepdims=True)
        # keep current positions as alternative start half the time: blend
        # choose start as random (delete) per spec
        remain = LIMIT - (time.time() - t0)
        # strictly capped 15s repack
        repack_budget = min(15.0, max(remain - 7.0, 1.0))
        if repack_budget < 1.0:
            break
        Ynew = _repack_patch(Y0, F, repack_budget, rng)
        if time.time() - t0 > LIMIT - 6:
            # still evaluate without polish
            Xt = cur.copy()
            Xt[mask] = Ynew / np.linalg.norm(Ynew, axis=1, keepdims=True)
            vt = _max_cos(Xt)
            if vt < cur_val - 1e-12:
                cur = Xt
                cur_val = vt
            break
        Xt = cur.copy()
        Xt[mask] = Ynew / np.linalg.norm(Ynew, axis=1, keepdims=True)
        # LP-polish whole config at most 5s
        remain2 = LIMIT - (time.time() - t0)
        polish_budget = min(5.0, max(remain2 - 1.0, 0.5))
        try:
            Xp = _lp_polish(Xt, polish_budget)
        except Exception:
            Xp = Xt
        vp = _max_cos(Xp)
        # accept only if true max improves
        if vp < cur_val - 1e-12:
            cur = Xp
            cur_val = vp
            # refresh stress for next patch
            try:
                resid, deg, gmax = _stress_residuals(cur, thresh=1e-6)
                Gfull = cur @ cur.T
                np.fill_diagonal(Gfull, -2.0)
                contacts_tight = (Gfull >= gmax - 1e-9).sum(axis=1)
                rattler = int(np.argmin(contacts_tight))
                sim_r = cur @ cur[rattler].copy()
                sim_r[rattler] = -2.0
                nn_rattler = np.argsort(-sim_r)
            except Exception:
                pass
        if time.time() - t0 > LIMIT - 3:
            break
    cur = cur / np.linalg.norm(cur, axis=1, keepdims=True)
    # tiny shrink guard
    v = _max_cos(cur)
    # return best
    return cur
