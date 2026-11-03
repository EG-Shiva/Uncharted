import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def _lp_polish(X, budget, init_radius=1e-3, cap=3000):
    t0 = time.time()
    x = X / np.linalg.norm(X, axis=1, keepdims=True)
    n, d = x.shape
    try:
        best = _max_cos(x)
    except Exception:
        return x
    radius = float(init_radius)
    iu = np.triu_indices(n, 1)
    while time.time() - t0 < budget and radius > 1e-14:
        if time.time() - t0 + 0.15 > budget and radius < 5e-4:
            break
        Gfull = x @ x.T
        c = Gfull[iu]
        m = float(np.max(c))
        thresh = max(20.0 * radius, 1e-9)
        sel = c > m - thresh
        cnt = int(np.sum(sel))
        if cnt == 0:
            radius *= 0.5
            continue
        if cnt > cap:
            kth = np.partition(c, -cap)[-cap]
            sel = c >= kth
            cnt = int(np.sum(sel))
            if cnt > cap + 500:
                idx = np.where(sel)[0]
                order = np.argsort(-c[idx])[:cap]
                mask = np.zeros_like(c, dtype=bool)
                mask[idx[order]] = True
                sel = mask
        I = iu[0][sel]; J = iu[1][sel]
        csel = c[sel]
        p = len(I)
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv); obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        try:
            y = x + res.x[:-1].reshape(n, d)
            y = y / np.linalg.norm(y, axis=1, keepdims=True)
            v = _max_cos(y)
        except Exception:
            radius *= 0.5
            continue
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def _dual_stress(X, m=None, thresh=1e-7):
    n, d = X.shape
    G = X @ X.T
    iu = np.triu_indices(n, 1)
    c = G[iu]
    if m is None:
        m = float(np.max(c))
    sel = c > m - max(thresh, 1e-9)
    I = iu[0][sel]; J = iu[1][sel]
    csel = c[sel]
    p = len(I)
    w = np.ones(p)
    # one small LP to get duals
    try:
        radius = 1e-3
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([X[J], X[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((X.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv); obj[-1] = 1.0
        res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq, b_eq=np.zeros(n),
                      bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        if res.status == 0:
            marg = None
            try:
                marg = np.asarray(res.ineqlin.marginals, dtype=float).ravel()
            except Exception:
                marg = None
            if marg is not None and marg.size == p:
                w = -marg
                w = np.where(np.isfinite(w), w, 0.0)
                w = np.clip(w, 0.0, None)
                if w.max() <= 0:
                    w = np.ones(p)
                else:
                    w = w / w.max() + 1e-3
            else:
                w = np.ones(p)
        else:
            w = np.ones(p)
    except Exception:
        w = np.ones(p)
    pt_stress = np.zeros(n)
    np.add.at(pt_stress, I, w)
    np.add.at(pt_stress, J, w)
    return pt_stress, w, I, J, m

def _optimize_insert(Y, rng, n_rand=200, n_hole_pool=50000, steps=140):
    m, d = Y.shape
    # deepest holes from pool (batched)
    batch = 10000
    best_holes = None
    best_hole_vals = None
    nb = n_hole_pool // batch
    all_cand_holes = []
    for _ in range(nb):
        C = rng.standard_normal((batch, d))
        C /= np.linalg.norm(C, axis=1, keepdims=True)
        D = C @ Y.T
        mx = D.max(axis=1)
        # keep top 3 per batch
        idx = np.argpartition(mx, 3)[:3]
        all_cand_holes.append(C[idx])
        # free
    Hb = np.concatenate(all_cand_holes, axis=0)  # (3*nb, d)
    D2 = Hb @ Y.T
    mx2 = D2.max(axis=1)
    o = np.argsort(mx2)[:5]
    holes = Hb[o]
    S0 = rng.standard_normal((n_rand, d))
    S0 /= np.linalg.norm(S0, axis=1, keepdims=True)
    S = np.concatenate([S0, holes], axis=0)  # (n_rand+5, d)
    S /= np.linalg.norm(S, axis=1, keepdims=True)
    # vectorized LSE projected gradient with T anneal
    T0, T1 = 0.03, 0.008
    lr = 0.6
    for s in range(steps):
        frac = s / max(1, steps - 1)
        T = T0 + (T1 - T0) * frac
        D = (S @ Y.T) / T  # (Snum, m)
        Dm = D.max(axis=1, keepdims=True)
        E = np.exp(np.clip(D - Dm, -50, 0))
        Es = E.sum(axis=1, keepdims=True) + 1e-300
        P = E / Es  # softmax
        Gdir = P @ Y  # (Snum, d)
        dot = np.sum(Gdir * S, axis=1, keepdims=True)
        Gt = Gdir - dot * S
        # normalize step by mean norm to keep stable
        S = S - lr * Gt
        S /= np.linalg.norm(S, axis=1, keepdims=True)
        lr *= 0.995
    # true objective pick
    Df = S @ Y.T
    mxf = Df.max(axis=1)
    bi = int(np.argmin(mxf))
    return S[bi] / np.linalg.norm(S[bi])

def _smooth_exp(X, steps=350, T0=0.05, T1=0.008, lr=0.25):
    n, d = X.shape
    Y = X.copy()
    l0 = np.log(T0); l1 = np.log(T1)
    for s in range(steps):
        frac = s / max(1, steps - 1)
        T = float(np.exp(l0 + (l1 - l0) * frac))
        G = Y @ Y.T
        np.fill_diagonal(G, -1.0)
        mx = G.max()
        W = np.exp(np.clip((G - mx) / T, -50, 0))
        np.fill_diagonal(W, 0.0)
        Gr = W @ Y
        dot = np.sum(Gr * Y, axis=1, keepdims=True)
        Gt = Gr - dot * Y
        nm = np.mean(np.linalg.norm(Gt, axis=1)) + 1e-12
        Y = Y - (lr / nm) * Gt
        Y /= np.linalg.norm(Y, axis=1, keepdims=True)
    return Y

def solve():
    t_start = time.time()
    DEADLINE = t_start + 155.0
    rng = np.random.default_rng(0)
    n, d = 181, 9
    cands = []
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=float)
            if a.shape == (n, d):
                a = a / np.linalg.norm(a, axis=1, keepdims=True)
                if np.all(np.isfinite(a)):
                    cands.append(a)
        except Exception:
            pass
    if not cands:
        a = rng.standard_normal((n, d))
        a /= np.linalg.norm(a, axis=1, keepdims=True)
        cands.append(a)
    vals = [_max_cos(c) for c in cands]
    bi = int(np.argmin(vals))
    x_best = cands[bi].copy()
    b_best = float(vals[bi])
    # one LP polish
    rem = DEADLINE - time.time()
    if rem > 8:
        x_best = _lp_polish(x_best, budget=min(6.0, rem - 4.0), init_radius=2e-3)
        b_best = _max_cos(x_best)
    incumb = x_best.copy()
    incumb_val = b_best
    # dual stress
    try:
        pt_stress, w, I, J, mval = _dual_stress(incumb, m=incumb_val, thresh=2e-4)
    except Exception:
        pt_stress = None
    if pt_stress is None:
        return incumb / np.linalg.norm(incumb, axis=1, keepdims=True)
    # contacts for graph (tight threshold)
    G = incumb @ incumb.T
    np.fill_diagonal(G, -2.0)
    m0 = float(G.max())
    np.fill_diagonal(G, 1.0)
    contacts = np.sum(G > m0 - 1e-7, axis=1) - 1
    rattlers = np.where(contacts <= 0)[0].tolist()
    # if fewer than 2, take lowest stress
    if len(rattlers) < 2:
        order = np.argsort(pt_stress)
        for q in order:
            if int(q) not in rattlers:
                rattlers.append(int(q))
            if len(rattlers) >= 2:
                break
    else:
        rattlers = rattlers[:2]
    # highest-stress connected cluster (exclude rattlers)
    K = 14
    cand_idx = [i for i in np.argsort(-pt_stress) if i not in rattlers]
    seed = int(cand_idx[0])
    # adjacency from near-max pairs
    adj = [set() for _ in range(n)]
    for a, b in zip(I.tolist(), J.tolist()):
        # only keep very tight for connectivity
        # I,J already within 2e-4; filter tighter using G
        if G[a, b] > m0 - 1e-6:
            adj[a].add(b); adj[b].add(a)
    cluster = set([seed])
    # grow by highest stress neighbor of cluster
    while len(cluster) < K:
        frontier = set()
        for u in list(cluster):
            for v in adj[u]:
                if v not in cluster and v not in rattlers:
                    frontier.add(v)
        if not frontier:
            # add next highest stress not yet taken
            for q in cand_idx:
                if q not in cluster:
                    frontier.add(int(q))
                    break
            if not frontier:
                break
        # pick highest stress in frontier
        fl = sorted(list(frontier), key=lambda z: -pt_stress[z])
        cluster.add(int(fl[0]))
        if len(cluster) >= K:
            break
    removed = sorted(set(rattlers) | cluster)
    # order removed by stress descending (rattlers last? put rattlers first to fill holes first)
    # spec: remove two rattlers + cluster; reinsert one at a time. Use descending stress.
    removed_sorted = sorted(removed, key=lambda z: -pt_stress[z] if z not in rattlers else 1e18)
    # frozen set
    keep_mask = np.ones(n, dtype=bool)
    keep_mask[removed] = False
    Ylist = incumb[keep_mask].copy()  # (n-len, d)
    # reinsert one at a time, frozen
    new_pts = []
    for k, ridx in enumerate(removed_sorted):
        if time.time() > DEADLINE - 25:
            break
        Ycur = np.concatenate([Ylist] + ([np.stack(new_pts)] if new_pts else []), axis=0) if (len(new_pts) > 0) else Ylist
        try:
            v = _optimize_insert(Ycur, rng, n_rand=200, n_hole_pool=50000, steps=120)
        except Exception:
            v = rng.standard_normal(d); v /= np.linalg.norm(v)
        new_pts.append(v)
        # early deadline check
        if time.time() > DEADLINE - 20:
            break
    # assemble
    if len(new_pts) == len(removed_sorted):
        X1 = np.zeros((n, d))
        X1[keep_mask] = Ylist
        for ridx, v in zip(removed_sorted, new_pts):
            X1[ridx] = v
        X1 /= np.linalg.norm(X1, axis=1, keepdims=True)
        v1 = _max_cos(X1)
        if v1 < incumb_val:
            incumb = X1; incumb_val = v1
        else:
            # still use X1 as working point for smoothing (different basin)
            incumb_w = X1
            # short smoothing then check
            if time.time() < DEADLINE - 20:
                try:
                    Xs = _smooth_exp(X1, steps=250, T0=0.04, T1=0.01, lr=0.3)
                    vs = _max_cos(Xs)
                    # keep working point even if worse for final LP (basin hop)
                    incumb_w = Xs
                except Exception:
                    incumb_w = X1
            else:
                incumb_w = X1
            incumb = incumb_w
            incumb_val = _max_cos(incumb)
            # if working worse than best, revert for final? keep working to allow LP to improve
            # but track best separately
            if incumb_val > b_best:
                # keep best as fallback
                pass
        # unfreeze smoothing
        if time.time() < DEADLINE - 15:
            try:
                incumb = _smooth_exp(incumb, steps=250, T0=0.03, T1=0.008, lr=0.3)
                incumb_val = _max_cos(incumb)
            except Exception:
                pass
    # final LP polish under deadline
    rem = DEADLINE - time.time() - 1.0
    if rem > 1.0:
        try:
            Xp = _lp_polish(incumb, budget=float(min(rem, 60.0)), init_radius=8e-3)
            vp = _max_cos(Xp)
            if vp < incumb_val:
                incumb, incumb_val = Xp, vp
        except Exception:
            pass
    # return best between incumb and original best
    if b_best < incumb_val:
        # original x_best was better; but check if incumb close - return best
        # reload x_best? we mutated? x_best preserved separately
        try:
            xb = x_best / np.linalg.norm(x_best, axis=1, keepdims=True)
            if _max_cos(xb) < incumb_val:
                return xb
        except Exception:
            pass
    incumb = incumb / np.linalg.norm(incumb, axis=1, keepdims=True)
    # safety: if NaN, return x_best
    if not np.all(np.isfinite(incumb)):
        return x_best / np.linalg.norm(x_best, axis=1, keepdims=True)
    return incumb
