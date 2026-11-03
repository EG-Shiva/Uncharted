import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def fast_quench(X, steps=260, beta0=30.0, beta1=220.0, lr0=0.03, lr1=0.003):
    X = X.copy()
    n = X.shape[0]
    for s in range(steps):
        frac = s / max(1, steps - 1)
        beta = beta0 * (beta1 / beta0) ** frac
        lr = lr0 * (lr1 / lr0) ** frac
        G = X @ X.T
        np.fill_diagonal(G, -10.0)
        m = float(G.max())
        W = np.exp((G - m) * beta)
        np.fill_diagonal(W, 0.0)
        Grad = W @ X
        proj = np.sum(Grad * X, axis=1, keepdims=True)
        Gt = Grad - proj * X
        nrm = float(np.linalg.norm(Gt) / np.sqrt(n))
        if not np.isfinite(nrm) or nrm < 1e-16:
            continue
        X = X - (lr / nrm) * Gt
        nr = np.linalg.norm(X, axis=1, keepdims=True)
        nr[~np.isfinite(nr)] = 1.0
        nr[nr < 1e-12] = 1.0
        X = X / nr
    return X

def lp_polish_deadline(x, deadline, r_init=0.004, cap=2000):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    try:
        best = max_cos(x)
    except Exception:
        return x
    radius = r_init
    iu0 = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        G = x @ x.T
        c_all = G[iu0]
        m = float(c_all.max())
        thresh = m - max(20.0 * radius, 2e-7)
        sel = c_all > thresh
        cnt = int(np.count_nonzero(sel))
        if cnt == 0:
            radius *= 0.5
            continue
        if cnt > cap:
            idx = np.where(sel)[0]
            order = np.argpartition(-c_all[idx], cap)[:cap]
            ns = np.zeros_like(sel)
            ns[idx[order]] = True
            sel = ns
            cnt = cap
        I = iu0[0][sel]; J = iu0[1][sel]
        c_sel = c_all[sel]
        p = cnt
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.empty((p, 2 * d + 1), dtype=int)
        ar = np.arange(d)
        cols[:, :d] = I[:, None] * d + ar
        cols[:, d:2*d] = J[:, None] * d + ar
        cols[:, 2*d] = n * d
        cols = cols.ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], axis=1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        except Exception:
            radius *= 0.5
            continue
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c_sel, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        yn = np.linalg.norm(y, axis=1, keepdims=True)
        if not np.all(np.isfinite(yn)) or np.any(yn < 1e-12):
            radius *= 0.5
            continue
        y = y / yn
        v = max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.03)
        else:
            radius *= 0.5
    return x

def deepest_hole(X_fixed, rng, M=2000):
    d = X_fixed.shape[1]
    R = rng.standard_normal((M, d))
    R /= np.linalg.norm(R, axis=1, keepdims=True)
    dots = R @ X_fixed.T
    worst = dots.max(axis=1)
    pick = int(np.argmin(worst))
    return R[pick], float(worst[pick])

def solve():
    t0 = time.time()
    HARD_END = t0 + 175.0
    LOOP_END = t0 + 148.0
    rng = np.random.default_rng(0)
    n, d = 176, 9
    cands = []
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            if a.shape == (n, d) and np.all(np.isfinite(a)):
                nr = np.linalg.norm(a, axis=1, keepdims=True)
                if np.all(nr > 1e-12):
                    cands.append(a / nr)
        except Exception:
            pass
    if not cands:
        X0 = rng.standard_normal((n, d))
        X0 /= np.linalg.norm(X0, axis=1, keepdims=True)
        cands.append(X0)
    vals = [max_cos(c) for c in cands]
    bi = int(np.argmin(vals))
    best_X = cands[bi].copy()
    best_v = float(vals[bi])
    if time.time() < LOOP_END:
        try:
            best_X = lp_polish_deadline(best_X, min(time.time() + 3.0, LOOP_END), r_init=0.0012)
            best_v = max_cos(best_X)
        except Exception:
            pass
    curr_X = best_X.copy()
    curr_v = best_v
    T = 5e-4
    trial = 0
    while time.time() < LOOP_END:
        trial += 1
        if trial % 30 == 0:
            curr_X = best_X.copy(); curr_v = best_v
        mode = trial % 5
        Xp = curr_X.copy()
        try:
            Gc = curr_X @ curr_X.T
            np.fill_diagonal(Gc, -2.0)
            rowmax = Gc.max(axis=1)
            order = np.argsort(rowmax)
        except Exception:
            rowmax = np.zeros(n); order = np.arange(n)
        if mode == 0:
            # coordinated cluster shake of contact neighborhood
            seed = int(rng.integers(n))
            dots = curr_X @ curr_X[seed]
            K = int(rng.integers(12, 28))
            neigh = np.argpartition(-dots, K)[:K]
            sigma_c = 0.02 + 0.03 * rng.random()
            sigma_i = 0.004 + 0.008 * rng.random()
            common = sigma_c * rng.standard_normal(d)
            Xp[neigh] += common + sigma_i * rng.standard_normal((K, d))
            mask = np.ones(n, dtype=bool); mask[neigh] = False
            Xp[mask] += 0.0015 * rng.standard_normal((n - K, d))
            Xp /= np.linalg.norm(Xp, axis=1, keepdims=True)
            steps = 300
        elif mode == 1:
            # hub kill into deepest hole (1-2 hubs)
            k = 1 if (trial % 2 == 0) else 2
            idxs = order[-k:]
            # sequential reinsert excluding each target
            for ii, idx in enumerate(idxs):
                keep = np.ones(n, dtype=bool); keep[idx] = False
                # for second hub, also use already moved first? use current Xp fixed set
                fixed = Xp[keep]
                hole, _ = deepest_hole(fixed, rng, M=2000)
                Xp[idx] = hole
            Xp += 0.002 * rng.standard_normal((n, d))
            Xp /= np.linalg.norm(Xp, axis=1, keepdims=True)
            steps = 300
        elif mode == 2:
            # medium global shake
            sigma = 0.008 if (trial % 2 == 0) else 0.02
            Xp = Xp + sigma * rng.standard_normal((n, d))
            Xp /= np.linalg.norm(Xp, axis=1, keepdims=True)
            steps = 260
        elif mode == 3:
            # block shake of random subset + common shift
            K = 20
            idxs = rng.choice(n, size=K, replace=False)
            Xp[idxs] += 0.03 * rng.standard_normal((K, d))
            Xp += 0.002 * rng.standard_normal((n, d))
            Xp /= np.linalg.norm(Xp, axis=1, keepdims=True)
            steps = 260
        else:
            # worst-8 targeted jitter with one shared direction
            k = 8
            idxs = order[-k:]
            common = 0.015 * rng.standard_normal(d)
            Xp[idxs] += common + 0.01 * rng.standard_normal((k, d))
            Xp /= np.linalg.norm(Xp, axis=1, keepdims=True)
            steps = 280
        try:
            Xq = fast_quench(Xp, steps=steps)
            vq = max_cos(Xq)
        except Exception:
            continue
        if not np.isfinite(vq):
            continue
        # screening: skip expensive LP if quench is far above best
        margin = 0.008 if trial < 60 else 0.005
        if vq > best_v + margin and vq > curr_v + 0.003:
            # still allow Metropolis move on cheap level occasionally
            if vq < curr_v and rng.random() < 0.1:
                curr_X = Xq.copy(); curr_v = vq
            continue
        budget = 1.7
        deadline = min(time.time() + budget, LOOP_END, HARD_END - 12.0)
        if deadline <= time.time() + 0.15:
            break
        try:
            r0 = 0.006 if mode in (0, 1, 3) else 0.0035
            Xf = lp_polish_deadline(Xq, deadline, r_init=r0)
            v = max_cos(Xf)
        except Exception:
            continue
        if not np.isfinite(v):
            continue
        if v < best_v:
            best_v = v; best_X = Xf.copy()
            curr_X = Xf.copy(); curr_v = v
        else:
            if v < curr_v or rng.random() < np.exp((curr_v - v) / T):
                curr_X = Xf.copy(); curr_v = v
        if time.time() > LOOP_END - 1.0:
            break
    try:
        fd = HARD_END - 1.0
        if time.time() < fd:
            best_X = lp_polish_deadline(best_X, fd, r_init=0.0012)
    except Exception:
        pass
    best_X = best_X / np.linalg.norm(best_X, axis=1, keepdims=True)
    return best_X
