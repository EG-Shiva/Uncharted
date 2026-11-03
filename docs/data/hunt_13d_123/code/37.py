import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _lp_polish(X, seconds, p_cap=2500):
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    n, d = X.shape
    best = _max_cos(X)
    best_x = X.copy()
    radius = 1e-3
    start = time.time()
    iu0, iu1 = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        # time guard: need at least 0.4s for another linprog
        if seconds - (time.time() - start) < 0.35:
            break
        G = X @ X.T
        c_all = G[iu0, iu1]
        m = c_all.max()
        window = max(20.0 * radius, 1e-9)
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
        I = iu0[sel]; J = iu1[sel]; csel = c_all[sel]
        p = cnt
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([(I[:, None] * d + np.arange(d)).reshape(p, d),
                               (J[:, None] * d + np.arange(d)).reshape(p, d),
                               np.full((p, 1), n * d)], axis=1).ravel()
        vals = np.concatenate([X[J], X[I], -np.ones((p, 1))], axis=1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            erows = np.repeat(np.arange(n), d)
            ecols = np.arange(n * d)
            A_eq = coo_matrix((X.ravel(), (erows, ecols)), shape=(n, nv)).tocsr()
            obj = np.zeros(nv); obj[-1] = 1.0
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

def _proj_simplex(v, s):
    n = v.shape[0]
    u = np.sort(v)[::-1]
    css = np.cumsum(u)
    thetas = (css - s) / (np.arange(n) + 1.0)
    mask = u - thetas > 0
    # at least index 0 is True since s>0
    idx = np.where(mask)[0]
    if len(idx) == 0:
        return np.full_like(v, s / n)
    rho = idx[-1]
    theta = thetas[rho]
    w = v - theta
    w[w < 0] = 0
    return w

def _stress(X, cgap=1e-6):
    n, d = X.shape
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    m = G.max()
    thr = m - cgap
    neigh = []
    for i in range(n):
        row = G[i]
        # exclude self ( -2 )
        nb = np.where(row > thr)[0]
        nb = nb[nb != i]
        neigh.append(nb)
    deg = np.array([len(nb) for nb in neigh], dtype=float)
    imb = np.zeros(n)
    over = np.zeros(n)
    for i in range(n):
        nb = neigh[i]
        k = len(nb)
        if k == 0:
            imb[i] = 0.0; over[i] = 0.0
            continue
        if k == 1:
            c = float(X[i] @ X[nb[0]])
            v = X[nb[0]] - c * X[i]
            imb[i] = float(np.linalg.norm(v))  # /1
            over[i] = 1.0
            continue
        c = X[i] @ X[nb].T  # (k,)
        V = X[nb].T - np.outer(X[i], c)  # d x k
        GtG = V.T @ V
        try:
            L = float(np.linalg.norm(GtG, ord=2))
        except Exception:
            L = float(np.trace(GtG) / k) + 1e-12
        if not np.isfinite(L) or L < 1e-14:
            imb[i] = 0.0; over[i] = 1.0
            continue
        alpha = 1.0 / (L + 1e-14)
        w = np.ones(k)
        for _ in range(120):
            grad = GtG @ w
            w = w - alpha * grad
            w = _proj_simplex(w, float(k))
        r = V @ w
        en = float(np.linalg.norm(r))
        imb[i] = en / float(k)
        over[i] = float(w.max())
    return m, neigh, deg, imb, over, G

def _repack_patch(X, patch, best_thr, budget, rng):
    t0 = time.time()
    n, d = X.shape
    m = len(patch)
    patch = np.asarray(patch, dtype=np.int64)
    mask = np.ones(n, dtype=bool); mask[patch] = False
    fidx = np.where(mask)[0]
    Xf = X[fidx]
    # current patch max
    Y0 = X[patch].copy()
    def patch_max(Y):
        Gpf = Y @ Xf.T
        Gpp = Y @ Y.T
        np.fill_diagonal(Gpp, -2.0)
        return float(max(Gpf.max(), Gpp.max()))
    best_pm = patch_max(Y0)
    best_Y = Y0.copy()
    # two attempts: random delete + perturbed
    inits = []
    # deterministic random matrices
    R1 = rng.standard_normal(size=(m, d))
    R1 /= np.linalg.norm(R1, axis=1, keepdims=True)
    R2n = rng.standard_normal(size=(m, d)) * 0.35
    Yp = Y0 + R2n
    Yp /= np.linalg.norm(Yp, axis=1, keepdims=True)
    inits = [R1, Yp]
    for ai, Yinit in enumerate(inits):
        if time.time() - t0 >= budget:
            break
        Y = Yinit.copy()
        # steps adapted to remaining budget
        rem = budget - (time.time() - t0)
        if rem <= 0.2:
            break
        # ~3000 steps per attempt max
        steps = 3000
        # anneal params
        T0a, T1a = 0.02, 0.0025
        lr0a, lr1a = 0.8, 0.12
        beta = 2.0
        thr = best_thr - 0.002
        for s in range(steps):
            if (s % 25) == 0 and time.time() - t0 >= budget:
                break
            frac = s / max(steps - 1, 1)
            T = T0a + (T1a - T0a) * frac
            lr = lr0a + (lr1a - lr0a) * frac
            Gpf = Y @ Xf.T
            Gpp = Y @ Y.T
            mv = Gpf.max()
            gppmax = Gpp.max()  # diag is 1.0, would dominate!
            # for softmax max we need off-diag; compute mv correctly
            # Gpp diag=1, so mv at least 1? must exclude diag for mv
            # use off-diag max for mv
            # quick: temporarily set diag low for mv
            # Instead compute mv as max of Gpf and off-diag Gpp
            # Gpp diag is 1.0, so correct mv:
            # recompute with diag -2 for mv only
            # cheap: mv = max(Gpf.max(), (Gpp - eye).max()) ; eye diag 2?
            # do it:
            # (avoid extra alloc most steps: approximate mv = max(Gpf.max(), Gpp.max() second?) )
            # Simplest: set diag to -2 in a copy for mv? do every step cheap for m<=18
            np.fill_diagonal(Gpp, -2.0)
            mv2 = Gpf.max()
            mv3 = Gpp.max()
            mv = mv2 if mv2 > mv3 else mv3
            # restore diag to 1 for exp? we need E_pp diag 0, so keep -2 and exp will be ~0
            Epf = np.exp((Gpf - mv) / T)
            Epp = np.exp((Gpp - mv) / T)
            np.fill_diagonal(Epp, 0.0)
            # restore Gpp diag for later? Gpp recomputed each iter anyway
            Z = Epf.sum() + Epp.sum() * 0.5 + 1e-300
            Wpf = Epf / Z
            Wpp = Epp / Z
            Grad_sm = Wpf @ Xf + Wpp @ Y
            # penalty
            Ppf = Gpf - thr
            Ppf[Ppf < 0] = 0
            Ppp = Gpp - thr
            Ppp[Ppp < 0] = 0
            np.fill_diagonal(Ppp, 0.0)
            if Ppf.max() > 0 or Ppp.max() > 0:
                Grad_pen = 2.0 * beta * (Ppf @ Xf + Ppp @ Y)
            else:
                Grad_pen = 0.0
            Grad = Grad_sm + Grad_pen
            # tangential
            dot = (Grad * Y).sum(axis=1, keepdims=True)
            Gt = Grad - dot * Y
            Y = Y - lr * Gt
            yn = np.linalg.norm(Y, axis=1, keepdims=True)
            yn[yn == 0] = 1.0
            Y = Y / yn
            if (s % 100) == 99 or s == steps - 1:
                pm = patch_max(Y)
                if pm < best_pm:
                    best_pm = pm
                    best_Y = Y.copy()
                # early exit if already well below thr and time low?
            if time.time() - t0 >= budget:
                break
        # end steps
    return best_Y

def solve():
    t_start = time.time()
    HARD = 170.0  # leave margin for 180s limit
    rng = np.random.default_rng(0)
    n, d = 123, 13
    X0 = None
    best0 = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=float)
            if a.shape == (n, d):
                a = a / np.linalg.norm(a, axis=1, keepdims=True)
                if np.all(np.isfinite(a)):
                    v = _max_cos(a)
                    if v < best0:
                        best0 = v
                        X0 = a.copy()
        except Exception:
            pass
    if X0 is None:
        R = rng.standard_normal(size=(n, d))
        R /= np.linalg.norm(R, axis=1, keepdims=True)
        X0 = R
        best0 = _max_cos(X0)
    cur = X0.copy()
    cur_val = float(best0)
    try:
        m0, neigh, deg, imb, over, Gfull = _stress(cur, cgap=1e-6)
    except Exception:
        return cur / np.linalg.norm(cur, axis=1, keepdims=True)
    # rattler = minimal degree
    rattler = int(np.argmin(deg))
    # nearest neighbours of rattler by angle
    cos_r = (cur[rattler] @ cur.T).copy()
    cos_r[rattler] = -2.0
    order_r = np.argsort(-cos_r)
    nearest5 = [int(i) for i in order_r[:5]]
    # hub candidates
    cand_hubs = []
    idxs = [i for i in range(n) if i != rattler]
    try:
        hub_low = min(idxs, key=lambda i: (deg[i], -imb[i]))
        hub_high = max(idxs, key=lambda i: (deg[i], over[i]))
        hub_imb = max(idxs, key=lambda i: imb[i])
    except Exception:
        hub_low = idxs[0]; hub_high = idxs[1]; hub_imb = idxs[2]
    for h in (hub_low, hub_high, hub_imb):
        if h not in cand_hubs:
            cand_hubs.append(h)
    cand_hubs = cand_hubs[:3]
    patches = []
    for h in cand_hubs:
        S = set()
        S.add(rattler)
        for q in nearest5:
            S.add(q)
        S.add(int(h))
        # contacts of hub within 1e-6
        for q in neigh[int(h)]:
            S.add(int(q))
        # adjust size to 10-18, target 14
        S = list(S)
        if len(S) > 18:
            # keep priority: rattler+nearest+hub first, then closest to hub
            base = set([rattler] + nearest5 + [int(h)])
            rest = [q for q in S if q not in base]
            # sort rest by cosine to hub descending
            rest_sorted = sorted(rest, key=lambda q: float(cur[int(h)] @ cur[q]), reverse=True)
            need = 18 - len(base)
            S = list(base) + rest_sorted[:need]
        if len(S) < 10:
            # add nearest to hub
            cos_h = (cur[int(h)] @ cur.T).copy()
            oh = np.argsort(-cos_h)
            for q in oh:
                q = int(q)
                if q not in S:
                    S.append(q)
                if len(S) >= 10:
                    break
        # final trim/pad to 10-18
        S = [int(q) for q in S[:18]]
        if len(S) >= 10:
            patches.append(S)
        if len(patches) >= 3:
            break
    # iterate patches, strictly capped
    for pi, patch in enumerate(patches):
        if time.time() - t_start > HARD - 12:
            break
        remaining = HARD - (time.time() - t_start)
        if remaining < 12:
            break
        repack_budget = min(15.0, remaining - 8.0)
        polish_budget = min(5.0, remaining - 3.0)
        if repack_budget < 1.0 or polish_budget < 0.8:
            break
        try:
            Yb = _repack_patch(cur, patch, cur_val, repack_budget, rng)
        except Exception:
            continue
        if time.time() - t_start > HARD - 6:
            break
        Xcand = cur.copy()
        Xcand[np.asarray(patch, dtype=np.int64)] = Yb
        # renormalize (already)
        try:
            Xpol = _lp_polish(Xcand, polish_budget)
        except Exception:
            continue
        try:
            v = _max_cos(Xpol)
        except Exception:
            continue
        if v < cur_val - 1e-12 and np.all(np.isfinite(Xpol)):
            cur = Xpol.copy()
            cur_val = v
            # recompute stress for remaining patches? keep original patches to save time,
            # but update best_thr implicitly via cur_val
        # time checked after every stage
        if time.time() - t_start > HARD - 2:
            break
    cur = cur / np.linalg.norm(cur, axis=1, keepdims=True)
    # shrink tiny margin not needed; ensure finite
    cur[~np.isfinite(cur)] = 0.0
    # if any zero row (should not), replace with random
    nr = np.linalg.norm(cur, axis=1)
    for i in range(n):
        if nr[i] < 1e-12 or not np.isfinite(nr[i]):
            v = rng.standard_normal(size=d)
            v /= np.linalg.norm(v)
            cur[i] = v
    cur = cur / np.linalg.norm(cur, axis=1, keepdims=True)
    return cur
