import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _lp_polish(X, budget, init_radius=2e-2, cap=2000):
    t0 = time.time()
    x = X / np.linalg.norm(X, axis=1, keepdims=True)
    n, d = x.shape
    try:
        best = _max_cos(x)
    except Exception:
        return x
    radius = init_radius
    iu0 = np.triu_indices(n, 1)
    ar = np.arange(d)
    it = 0
    while True:
        if time.time() - t0 >= budget:
            break
        if radius < 1e-14 or it > 40:
            break
        # leave time for solver
        if time.time() - t0 + 0.25 > budget and radius < 5e-4:
            break
        it += 1
        G = x @ x.T
        c = G[iu0]
        m = float(c.max())
        thresh = max(20.0 * radius, 1e-9)
        sel = c > m - thresh
        cnt = int(sel.sum())
        if cnt == 0:
            radius *= 0.5
            continue
        I = iu0[0][sel]; J = iu0[1][sel]
        cs = c[sel]
        if cnt > cap:
            idx = np.argpartition(-cs, cap)[:cap]
            I = I[idx]; J = J[idx]; cs = cs[idx]
            cnt = cap
        nv = n * d + 1
        cols_mat = np.empty((cnt, 2 * d + 1), dtype=np.int64)
        cols_mat[:, :d] = I[:, None] * d + ar
        cols_mat[:, d:2 * d] = J[:, None] * d + ar
        cols_mat[:, 2 * d] = n * d
        rows = np.repeat(np.arange(cnt), 2 * d + 1)
        cols = cols_mat.ravel()
        vals_mat = np.empty((cnt, 2 * d + 1))
        vals_mat[:, :d] = x[J]
        vals_mat[:, d:2 * d] = x[I]
        vals_mat[:, 2 * d] = -1.0
        vals = vals_mat.ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(cnt, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv); obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-cs, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        try:
            y = x + res.x[:-1].reshape(n, d)
            y /= np.linalg.norm(y, axis=1, keepdims=True)
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

def _smooth_exp(X, steps=450, k0=60.0, k1=300.0, lr0=0.04, lr1=0.004):
    n, d = X.shape
    for s in range(steps):
        frac = s / max(steps - 1, 1)
        k = k0 + (k1 - k0) * frac
        lr = lr0 + (lr1 - lr0) * frac
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        gmax = float(G.max())
        # exp weights, clipped
        W = np.exp(np.clip(k * (G - gmax), -50, 0.0))
        np.fill_diagonal(W, 0.0)
        Gr = W @ X
        dot = np.einsum('ij,ij->i', Gr, X)[:, None]
        Gt = Gr - dot * X
        nrm = np.linalg.norm(Gt, axis=1).max()
        if not np.isfinite(nrm) or nrm < 1e-12:
            break
        X = X - (lr / (nrm + 1e-12)) * Gt
        X /= np.linalg.norm(X, axis=1, keepdims=True)
    return X

def _single_opt(p0, Ka, iters=250, k0=30.0, k1=250.0):
    p = p0 / (np.linalg.norm(p0) + 1e-12)
    for s in range(iters):
        frac = s / max(iters - 1, 1)
        k = k0 + (k1 - k0) * frac
        lr = 0.25 * (1.0 - frac) + 0.02 * frac
        sc = Ka @ p
        m = float(sc.max())
        W = np.exp(np.clip(k * (sc - m), -50, 0.0))
        sw = W.sum()
        if not np.isfinite(sw) or sw <= 0:
            break
        g = (W @ Ka) / sw
        gt = g - float(g @ p) * p
        nrm = float(np.linalg.norm(gt))
        if not np.isfinite(nrm) or nrm < 1e-12:
            break
        p = p - (lr / nrm) * gt
        p /= (np.linalg.norm(p) + 1e-12)
    return p

def solve():
    t0 = time.time()
    BUDGET_LOOP = 148.0
    HARD = 168.0
    rng = np.random.default_rng(12345)
    cand = []
    for fn in ["best.npy", "record.npy"]:
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=np.float64)
            if a.shape == (181, 9):
                a = a / np.linalg.norm(a, axis=1, keepdims=True)
                if np.all(np.isfinite(a)):
                    cand.append(a)
        except Exception:
            pass
    if len(cand) == 0:
        tmp = rng.standard_normal(size=(181, 9))
        tmp /= np.linalg.norm(tmp, axis=1, keepdims=True)
        cand.append(tmp)
    scores = [_max_cos(a) for a in cand]
    bi = int(np.argmin(scores))
    xb = cand[bi].copy()
    bb = float(scores[bi])
    xc = xb.copy()
    bc = bb
    n, d = xb.shape
    T = 1e-4
    it = 0
    # quick baseline polish if time allows
    if HARD - (time.time() - t0) > 12:
        try:
            xp = _lp_polish(xb, budget=2.0, init_radius=2e-3)
            vp = _max_cos(xp)
            if vp < bb:
                xb = xp; bb = vp; xc = xp.copy(); bc = vp
        except Exception:
            pass
    while time.time() - t0 < BUDGET_LOOP:
        rem_time = HARD - (time.time() - t0)
        if rem_time < 6.0:
            break
        it += 1
        try:
            # contacts of current
            Gc = xc @ xc.T
            np.fill_diagonal(Gc, -2.0)
            mcur = float(Gc.max())
            np.fill_diagonal(Gc, 1.0)
            tol = 1e-7
            contacts = np.sum(Gc > mcur - tol, axis=1) - 1
            # rattlers
            order = np.argsort(contacts)
            rattlers = [int(i) for i in order if contacts[i] <= 0][:2]
            # hub with >=20 contacts, else max
            hubs = np.where(contacts >= 20)[0]
            if len(hubs) > 0:
                # cycle through hubs deterministically + random
                hub = int(hubs[rng.integers(0, len(hubs))])
            else:
                hub = int(np.argmax(contacts))
            # neighbours of hub sorted by similarity
            sims = (xc[hub] @ xc.T).copy()
            sims[hub] = -10.0
            # exclude rattlers from neighbour list to avoid dup
            nn_order = np.argsort(-sims)
            k_nb = int(10 + rng.integers(0, 8))  # 10-17
            nb = []
            for idx in nn_order:
                if int(idx) in rattlers or int(idx) == hub:
                    continue
                nb.append(int(idx))
                if len(nb) >= k_nb:
                    break
            rem_idx = [hub] + nb + [r for r in rattlers if r != hub and r not in nb]
            rem_idx = sorted(set(rem_idx))
            keep_mask = np.ones(n, dtype=bool)
            keep_mask[rem_idx] = False
            K = xc[keep_mask].copy()
            kdel = len(rem_idx)
            # sequential deepest-hole reinsertion, 20000 candidates each
            D = np.empty((kdel, d))
            for j in range(kdel):
                # build augmented blocker set
                if j == 0:
                    Ka = K
                else:
                    Ka = np.vstack([K, D[:j]])
                NC = 20000
                C = rng.standard_normal((NC, d))
                C /= np.linalg.norm(C, axis=1, keepdims=True)
                S = C @ Ka.T
                mx = S.max(axis=1)
                best_i = int(np.argmin(mx))
                # small diversity: occasionally pick among top-8
                if rng.random() < 0.25:
                    idx8 = np.argpartition(mx, 8)[:8]
                    best_i = int(idx8[rng.integers(0, 8)])
                p0 = C[best_i]
                p0 = _single_opt(p0, Ka, iters=220, k0=30.0, k1=250.0)
                D[j] = p0
            y = xc.copy()
            y[np.array(rem_idx)] = D
            # exponential-repulsion smoothing, alpha 60->300
            # scale steps to remaining time (3-5s worth ~ 400-500 steps)
            y = _smooth_exp(y, steps=420, k0=60.0, k1=300.0, lr0=0.035, lr1=0.004)
            # LP polish ~3s
            rem2 = HARD - (time.time() - t0)
            if rem2 < 4.0:
                # still evaluate y without LP
                v_nolp = _max_cos(y)
                if v_nolp < bb - 1e-14:
                    xb = y.copy(); bb = v_nolp
                if v_nolp < bc or rng.random() < np.exp((bc - v_nolp) / max(T, 1e-12)):
                    xc = y.copy(); bc = v_nolp
                break
            budg = min(3.0, rem2 - 1.5)
            budg = max(1.0, budg)
            y2 = _lp_polish(y, budget=float(budg), init_radius=2e-2)
            v = _max_cos(y2)
            if v < bb - 1e-14:
                xb = y2.copy(); bb = v
                xc = y2.copy(); bc = v
            else:
                if v < bc or rng.random() < np.exp((bc - v) / max(T, 1e-12)):
                    xc = y2.copy(); bc = v
            T = max(T * 0.88, 1e-6)
        except Exception:
            continue
    # final safety: return best ever seen
    xb = xb / np.linalg.norm(xb, axis=1, keepdims=True)
    return xb
