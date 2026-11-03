import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def per_max(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return g.max(axis=1)

def lp_polish(x, seconds):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = max_cos(x)
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        if time.time() - start >= seconds:
            break
        c = (x @ x.T)[iu]
        m = float(c.max())
        sel = c > m - max(20 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        if p == 0:
            break
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d),
                                       np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv)
        obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c[sel], A_eq=A_eq,
                          b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                          method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        v = max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def soft_anneal(x, steps, beta0, beta1, lr, rng, noise_amp=0.0, mu=0.8):
    n, d = x.shape
    vel = np.zeros_like(x)
    for k in range(steps):
        frac = k / max(1, steps - 1)
        beta = beta0 + (beta1 - beta0) * frac
        lr_k = lr * (1.0 - 0.5 * frac)
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        gmax = float(G.max())
        W = np.exp(beta * (G - gmax))
        np.fill_diagonal(W, 0.0)
        rs = W.sum(axis=1, keepdims=True) + 1e-12
        grad = (W @ x) / rs
        dot = np.sum(grad * x, axis=1, keepdims=True)
        gt = grad - dot * x
        vel = mu * vel + (1.0 - mu) * gt
        x = x - lr_k * vel
        if noise_amp > 0:
            nz = rng.standard_normal(x.shape)
            nd = np.sum(nz * x, axis=1, keepdims=True)
            nz -= nd * x
            x = x + noise_amp * (1.0 - frac) * nz
        x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x

def refine_void(v0, Xrem, beta=120.0, steps=120, lr=0.08):
    v = v0 / (np.linalg.norm(v0) + 1e-30)
    for t in range(steps):
        dots = Xrem @ v
        mv = float(dots.max())
        W = np.exp(beta * (dots - mv))
        S = float(W.sum()) + 1e-300
        g = (W @ Xrem) / S
        gt = g - float(np.dot(g, v)) * v
        st = lr / (1.0 + t * 0.02)
        v = v - st * gt
        v /= (np.linalg.norm(v) + 1e-30)
    return v

def best_void_for_set(Xrem, rng, n_sample=16000, beta=120.0):
    d = Xrem.shape[1]
    cand = rng.standard_normal((n_sample, d))
    cand /= np.linalg.norm(cand, axis=1, keepdims=True)
    smax = (cand @ Xrem.T).max(axis=1)
    bi = int(np.argmin(smax))
    v0 = cand[bi]
    try:
        v0 = refine_void(v0, Xrem, beta=beta, steps=120, lr=0.08)
    except Exception:
        pass
    return v0

# ---- symmetric group utilities ----
def perm_mat(p):
    R = np.zeros((8, 8))
    p = np.asarray(p, dtype=int)
    R[p, np.arange(8)] = 1.0
    return R

def close_group(gens, limit=512):
    mods = [np.eye(8)]
    seen = [np.eye(8)]
    # BFS
    stack = [np.eye(8)]
    for _ in range(limit * 4):
        if len(mods) >= limit:
            break
        grew = False
        cur_list = list(mods)
        for a in cur_list:
            for g in gens:
                for c in (a @ g, g @ a):
                    ok = True
                    for m in mods:
                        if np.max(np.abs(m - c)) < 1e-8:
                            ok = False
                            break
                    if ok:
                        mods.append(c)
                        grew = True
                        if len(mods) >= limit:
                            break
                if len(mods) >= limit:
                    break
            if len(mods) >= limit:
                break
        if not grew:
            break
    return np.stack(mods, axis=0)

def build_groups():
    p_shift = [(i + 1) % 8 for i in range(8)]
    p_rev = [(-i) % 8 for i in range(8)]
    S = perm_mat(p_shift)
    Rv = perm_mat(p_rev)
    N = -np.eye(8)
    pA = [1, 2, 3, 0, 4, 5, 6, 7]
    pB = [0, 1, 2, 3, 5, 6, 7, 4]
    pC = [4, 5, 6, 7, 0, 1, 2, 3]
    A = perm_mat(pA); B = perm_mat(pB); C = perm_mat(pC)
    out = {}
    out['cyc8'] = close_group([S])
    out['cyc8neg'] = close_group([S, N])
    out['dih8'] = close_group([S, Rv])
    out['block'] = close_group([A, B, C])
    return out

def expand_seeds(Y, Gm):
    # Y (ns,8) unit rows, Gm (G,8,8) row-action
    ns = Y.shape[0]
    G = Gm.shape[0]
    X = np.einsum('kd,gde->kge', Y, Gm)  # ns,G,8
    return X.reshape(ns * G, 8)

def symmetric_adam(Y0, Gm, steps, lr0, rng):
    Y = Y0.copy()
    Y /= np.linalg.norm(Y, axis=1, keepdims=True)
    ns = Y.shape[0]
    G = Gm.shape[0]
    GmT = np.transpose(Gm, (0, 2, 1))
    m1 = np.zeros_like(Y)
    m2 = np.zeros_like(Y)
    b1, b2, eps = 0.9, 0.999, 1e-8
    beta_lo, beta_hi = 30.0, 400.0
    for k in range(steps):
        frac = k / max(1, steps - 1)
        beta = beta_lo * (beta_hi / beta_lo) ** frac
        lr = lr0 * (1.0 - 0.4 * frac)
        X = expand_seeds(Y, Gm)
        # LSE weights
        Gr = X @ X.T
        np.fill_diagonal(Gr, -2.0)
        mv = float(Gr.max())
        W = np.exp(beta * (Gr - mv))
        np.fill_diagonal(W, 0.0)
        rs = W.sum(axis=1, keepdims=True) + 1e-300
        gradX = (W @ X) / rs
        # accumulate to seeds: sum_g grad_{k,g} @ R_g^T
        gradXr = gradX.reshape(ns, G, 8)
        gseed = np.zeros_like(Y)
        for gi in range(G):
            gseed += gradXr[:, gi, :] @ GmT[gi]
        # tangent project
        dot = np.sum(gseed * Y, axis=1, keepdims=True)
        gseed -= dot * Y
        # Adam
        m1 = b1 * m1 + (1 - b1) * gseed
        m2 = b2 * m2 + (1 - b2) * (gseed * gseed)
        c1 = 1 - b1 ** (k + 1)
        c2 = 1 - b2 ** (k + 1)
        mh = m1 / c1
        vh = m2 / c2
        step = lr * mh / (np.sqrt(vh) + eps)
        # limit step norm per row
        sn = np.linalg.norm(step, axis=1, keepdims=True) + 1e-12
        cap = 0.05
        scale = np.minimum(1.0, cap / (sn + 1e-12))
        step = step * scale
        Y = Y - step
        Y /= np.linalg.norm(Y, axis=1, keepdims=True)
        if not np.all(np.isfinite(Y)):
            Y = Y0 / np.linalg.norm(Y0, axis=1, keepdims=True)
            m1[:] = 0; m2[:] = 0
    return Y

def prune_to_138(X):
    X = X.copy()
    while X.shape[0] > 138:
        n = X.shape[0]
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        # find max pair
        ij = np.unravel_index(int(np.argmax(G)), G.shape)
        i, j = int(ij[0]), int(ij[1])
        pm = G.max(axis=1)
        # remove the worse of the two (larger per-max, tie larger index)
        if pm[i] > pm[j] + 1e-15:
            rem = i
        elif pm[j] > pm[i] + 1e-15:
            rem = j
        else:
            # remove one with larger sum of top-3
            si = float(np.sort(G[i])[-3:].sum())
            sj = float(np.sort(G[j])[-3:].sum())
            rem = i if si >= sj else j
        X = np.delete(X, rem, axis=0)
    return X

def solve():
    t0 = time.time()
    T_LIMIT = 170.0
    groups = build_groups()
    # load best fallback
    best = None; best_val = np.inf
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn).astype(float)
            if a.shape == (138, 8):
                a /= np.linalg.norm(a, axis=1, keepdims=True)
                if np.all(np.isfinite(a)):
                    v = max_cos(a)
                    if v < best_val:
                        best_val = v; best = a.copy()
        except Exception:
            pass
    if best is None:
        rng0 = np.random.default_rng(0)
        best = rng0.standard_normal((138, 8))
        best /= np.linalg.norm(best, axis=1, keepdims=True)
        best_val = max_cos(best)
    # symmetric configs: (group name, nseed, seed, adam steps, lr)
    configs = [
        ('dih8', 9, 11, 900, 0.02),
        ('cyc8neg', 9, 22, 900, 0.02),
        ('block', 5, 33, 800, 0.02),
        ('cyc8', 18, 44, 800, 0.025),
        ('dih8', 10, 55, 700, 0.02),
    ]
    for (gname, ns, sd, astep, alr) in configs:
        tl = T_LIMIT - (time.time() - t0)
        if tl < 30:
            break
        try:
            Gm = groups[gname]
        except Exception:
            continue
        rng = np.random.default_rng(sd)
        Y0 = rng.standard_normal((ns, 8))
        Y0 /= np.linalg.norm(Y0, axis=1, keepdims=True)
        # cap steps by time
        if tl < 60:
            astep = min(astep, 400)
        try:
            Yopt = symmetric_adam(Y0, Gm, astep, alr, rng)
            Xfull = expand_seeds(Yopt, Gm)
            Xfull /= np.linalg.norm(Xfull, axis=1, keepdims=True)
        except Exception:
            continue
        # prune or pad to 138
        try:
            if Xfull.shape[0] > 138:
                X = prune_to_138(Xfull)
            elif Xfull.shape[0] < 138:
                X = Xfull.copy()
                need = 138 - X.shape[0]
                for _ in range(need):
                    v0 = best_void_for_set(X, rng, n_sample=12000, beta=120.0)
                    X = np.vstack([X, v0.reshape(1, 8)])
            else:
                X = Xfull
            X /= np.linalg.norm(X, axis=1, keepdims=True)
        except Exception:
            continue
        # gradual symmetry breaking: tangent noise 1e-3,1e-2,3e-2 with anneals
        try:
            amps = [1e-3, 1e-2, 3e-2]
            betas = [(30.0, 150.0), (30.0, 220.0), (40.0, 260.0)]
            stps = [500, 600, 700]
            if tl < 60:
                stps = [300, 350, 400]
            for ai, amp in enumerate(amps):
                tl2 = T_LIMIT - (time.time() - t0)
                if tl2 < 18:
                    break
                nz = rng.standard_normal(X.shape)
                nd = np.sum(nz * X, axis=1, keepdims=True)
                nz -= nd * X
                nn = np.linalg.norm(nz, axis=1, keepdims=True) + 1e-12
                nz /= nn
                X = X + amp * nz
                X /= np.linalg.norm(X, axis=1, keepdims=True)
                b0, b1 = betas[ai]
                X = soft_anneal(X, stps[ai], b0, b1, 0.022, rng, noise_amp=0.0, mu=0.8)
            # LP snap
            tl3 = T_LIMIT - (time.time() - t0)
            if tl3 > 12:
                X = lp_polish(X, min(2.0, tl3 - 8.0))
            v = max_cos(X)
            if v < best_val - 1e-12:
                best_val = v; best = X.copy()
            elif v < best_val + 0.004:
                # opportunistic extra anneal from this basin with tiny kick
                try:
                    tl4 = T_LIMIT - (time.time() - t0)
                    if tl4 > 15:
                        X2 = X + 0.008 * rng.standard_normal(X.shape)
                        X2 /= np.linalg.norm(X2, axis=1, keepdims=True)
                        X2 = soft_anneal(X2, 400, 30.0, 250.0, 0.02, rng, noise_amp=0.0, mu=0.8)
                        X2 = lp_polish(X2, 1.0)
                        v2 = max_cos(X2)
                        if v2 < best_val:
                            best_val = v2; best = X2.copy()
                except Exception:
                    pass
        except Exception:
            continue
    # final polish of global best with remaining time
    try:
        tl = T_LIMIT - (time.time() - t0)
        if tl > 6:
            best = lp_polish(best, min(8.0, tl - 2.0))
            best_val = max_cos(best)
    except Exception:
        pass
    best /= np.linalg.norm(best, axis=1, keepdims=True)
    return best
