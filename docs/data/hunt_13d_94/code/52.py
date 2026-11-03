import time
import numpy as np
from scipy.optimize import minimize, linprog
from scipy.sparse import coo_matrix

IDX = (np.arange(13)[None, :] - np.arange(13)[:, None]) % 13
S_CUBE = np.ones((16, 13), dtype=float)
for e in range(16):
    for j in range(4):
        if (e >> j) & 1:
            S_CUBE[e, j] = -1.0
FIXED26 = np.vstack([np.eye(13), -np.eye(13)])

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _make_expand(config):
    nproto = sum(1 for t in config if t != 'fixed26')
    def expand(flat):
        P_list = []
        idx = 0
        Ys = []
        for typ in config:
            if typ == 'fixed26':
                Ys.append(FIXED26)
            else:
                p = flat[idx*13:(idx+1)*13]
                idx += 1
                if typ == 'full':
                    R = p[IDX]
                    Ys.append(R)
                    Ys.append(-R)
                elif typ == 'half':
                    R = p[IDX]
                    Ys.append(R)
                elif typ == 'cube':
                    Ys.append(p[None, :] * S_CUBE)
        if len(Ys) == 0:
            return np.zeros((0, 13))
        return np.vstack(Ys)
    return nproto, expand

def _sym_fg_factory(config, nproto, expand, kval=130.0, wpen=0.1):
    def fg(flat):
        Y = expand(flat)
        rn = np.linalg.norm(Y, axis=1, keepdims=True)
        rn = np.maximum(rn, 1e-12)
        X = Y / rn
        G = X @ X.T
        # max off diag
        # set diag to -2 for max
        # copy to avoid mutating G used later? use view
        m = float(np.max(G - np.eye(G.shape[0]) * 5.0))
        K = kval
        E = np.exp(K * (G - m))
        np.fill_diagonal(E, 0.0)
        S = float(E.sum() * 0.5)
        if S <= 0 or not np.isfinite(S):
            f = m
            g = np.zeros_like(flat)
            return f, g
        f_lse = m + np.log(S) / K
        Pmat = flat.reshape(nproto, 13) if nproto > 0 else np.zeros((0, 13))
        if nproto > 0:
            pn = np.linalg.norm(Pmat, axis=1)
            pen = 0.5 * float(np.sum((pn - 1.0) ** 2))
        else:
            pen = 0.0
            pn = np.array([])
        f_total = f_lse + wpen * pen
        Grad_X = (E @ X) / S
        dot = (Grad_X * X).sum(axis=1, keepdims=True)
        Grad_Y = (Grad_X - dot * X) / rn
        grad_flat = np.zeros_like(flat)
        ptr = 0
        pi = 0
        for typ in config:
            if typ == 'fixed26':
                ptr += 26
            elif typ == 'full':
                Gb_p = Grad_Y[ptr:ptr+13]
                Gb_m = Grad_Y[ptr+13:ptr+26]
                gp = np.zeros(13)
                for k in range(13):
                    gp += np.roll(Gb_p[k], -k)
                    gp -= np.roll(Gb_m[k], -k)
                grad_flat[pi*13:(pi+1)*13] = gp
                pi += 1
                ptr += 26
            elif typ == 'half':
                Gb = Grad_Y[ptr:ptr+13]
                gp = np.zeros(13)
                for k in range(13):
                    gp += np.roll(Gb[k], -k)
                grad_flat[pi*13:(pi+1)*13] = gp
                pi += 1
                ptr += 13
            elif typ == 'cube':
                Gb = Grad_Y[ptr:ptr+16]
                gp = (S_CUBE * Gb).sum(axis=0)
                grad_flat[pi*13:(pi+1)*13] = gp
                pi += 1
                ptr += 16
        if nproto > 0:
            pn_safe = np.maximum(pn, 1e-12)
            pg = (Pmat * ((pn - 1.0) / pn_safe)[:, None]) * wpen
            grad_flat += pg.ravel()
        return float(f_total), grad_flat
    return fg

def _tangent_perturb(X, sigma, rng):
    N = rng.standard_normal(size=X.shape)
    dot = (N * X).sum(axis=1, keepdims=True)
    Nt = N - dot * X
    Y = X + sigma * Nt
    Y /= np.linalg.norm(Y, axis=1, keepdims=True)
    return Y

def _adam_refine(X0, steps, k0, k1, lr0, lr1):
    X = X0.copy()
    m1 = np.zeros_like(X)
    v1 = np.zeros_like(X)
    b1, b2, eps = 0.9, 0.999, 1e-8
    n = X.shape[0]
    for s in range(steps):
        frac = s / max(1, steps - 1)
        kk = k0 * (k1 / k0) ** frac
        lr = lr0 * (lr1 / lr0) ** frac
        G = X @ X.T
        mval = float(np.max(G - np.eye(n) * 5.0))
        W = np.exp(kk * (G - mval))
        np.fill_diagonal(W, 0.0)
        S = float(W.sum() * 0.5)
        if S <= 0 or not np.isfinite(S):
            break
        Grad = (W @ X) / S
        dot = (Grad * X).sum(axis=1, keepdims=True)
        Gt = Grad - dot * X
        m1 = b1 * m1 + (1 - b1) * Gt
        v1 = b2 * v1 + (1 - b2) * (Gt * Gt)
        bc1 = 1.0 - b1 ** (s + 1)
        bc2 = 1.0 - b2 ** (s + 1)
        upd = lr * (m1 / bc1) / (np.sqrt(v1 / bc2) + eps)
        X = X - upd
        X /= np.linalg.norm(X, axis=1, keepdims=True)
        if not np.all(np.isfinite(X)):
            break
    return X

def _lp_polish_deadline(x, deadline):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    iu = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        cmat = (x @ x.T)[iu]
        m = float(cmat.max())
        sel = cmat > m - max(20 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        if p == 0:
            break
        if p > 2500:
            # keep tightest 2500
            order = np.argpartition(-cmat[sel], 2500)[:2500]
            I, J = I[order], J[order]
            p = 2500
            sel_vals = cmat[sel][order]
        else:
            sel_vals = cmat[sel]
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        except Exception:
            break
        obj = np.zeros(nv)
        obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-sel_vals, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        v = _max_cos(y)
        if v < best:
            x, best, radius = y, v, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x, best

def solve():
    T0 = time.time()
    BUDGET = 172.0
    rng = np.random.default_rng(0)
    best_overall = None
    best_val = 1e9
    for fn in ["best.npy", "record.npy"]:
        try:
            a = np.load(fn).astype(float)
            a = a / np.linalg.norm(a, axis=1, keepdims=True)
            if a.shape == (94, 13) and np.all(np.isfinite(a)):
                v = _max_cos(a)
                if v < best_val:
                    best_val = v
                    best_overall = a.copy()
        except Exception:
            pass
    if best_overall is None:
        best_overall = rng.standard_normal((94, 13))
        best_overall /= np.linalg.norm(best_overall, axis=1, keepdims=True)
        best_val = _max_cos(best_overall)
    incumbent = best_overall.copy()

    configs = [
        ['full', 'full', 'full', 'cube'],
        ['full', 'full', 'half', 'half', 'cube'],
        ['full', 'half', 'half', 'half', 'half', 'cube'],
        ['fixed26', 'full', 'full', 'cube'],
        ['fixed26', 'full', 'half', 'half', 'cube'],
    ]
    # symmetry search: at least 100 sets
    results = []
    n_trials = 100
    sym_deadline = T0 + 65.0
    for t in range(n_trials):
        if time.time() > sym_deadline:
            break
        config = configs[t % len(configs)]
        nproto, expand = _make_expand(config)
        # init
        flat0 = rng.standard_normal(nproto * 13)
        # axis involvement: every 4th trial pin first proto near axis
        if nproto > 0 and (t % 4 == 0):
            # set first proto to e_{t%13} + small noise
            p0 = rng.standard_normal(13) * 0.2
            p0[t % 13] += 2.0
            flat0[0:13] = p0
        fg = _sym_fg_factory(config, nproto, expand, kval=130.0, wpen=0.1)
        try:
            res = minimize(fg, flat0, method='L-BFGS-B', jac=True,
                           options={'maxiter': 80, 'maxfun': 5000, 'ftol': 1e-14, 'gtol': 1e-07, 'maxcor': 12})
            flat = res.x
        except Exception:
            flat = flat0
        try:
            Y = expand(flat)
            Y /= np.linalg.norm(Y, axis=1, keepdims=True)
            if Y.shape != (94, 13) or not np.all(np.isfinite(Y)):
                continue
            v = _max_cos(Y)
            results.append((v, Y.copy()))
        except Exception:
            continue
    results.sort(key=lambda x: x[0])
    # retain five lowest
    top5 = results[:5] if len(results) >= 5 else results[:]
    # if symmetry failed, use incumbent
    cands = []
    for v, Y in top5:
        cands.append((v, Y))
    # break symmetry + Adam
    sigmas = [0.005, 0.02, 0.05]
    adam_cands = []
    for v0, Y0 in top5:
        for sg in sigmas:
            try:
                Yp = _tangent_perturb(Y0, sg, rng)
                adam_cands.append(Yp)
            except Exception:
                pass
    # limit to 15 (5*3)
    # Adam polish each with short run
    polished = []
    adam_deadline = T0 + 135.0
    for Yp in adam_cands:
        if time.time() > adam_deadline:
            break
        try:
            Yr = _adam_refine(Yp, steps=550, k0=50.0, k1=300.0, lr0=0.008, lr1=0.0012)
            v = _max_cos(Yr)
            polished.append((v, Yr))
            if v < best_val:
                best_val = v
                best_overall = Yr.copy()
        except Exception:
            continue
    # also Adam on incumbent briefly to not regress
    if time.time() < T0 + 140.0:
        try:
            Yr = _adam_refine(incumbent, steps=500, k0=50.0, k1=300.0, lr0=0.008, lr1=0.0012)
            v = _max_cos(Yr)
            polished.append((v, Yr))
            if v < best_val:
                best_val = v
                best_overall = Yr.copy()
        except Exception:
            pass
    # pick best few for longer Adam + LP
    polished.sort(key=lambda x: x[0])
    # longer refine on top 3
    for k in range(min(3, len(polished))):
        if time.time() > T0 + 150.0:
            break
        try:
            v0, Y0 = polished[k]
            Yr = _adam_refine(Y0, steps=400, k0=80.0, k1=350.0, lr0=0.005, lr1=0.0008)
            v = _max_cos(Yr)
            if v < best_val:
                best_val = v
                best_overall = Yr.copy()
            polished[k] = (v, Yr if v < v0 else Y0)
        except Exception:
            pass
    # final LP polish on best
    deadline = T0 + BUDGET - 2.0
    try:
        # choose current best
        cur = best_overall.copy()
        cur, v2 = _lp_polish_deadline(cur, deadline)
        if v2 < best_val:
            best_val = v2
            best_overall = cur.copy()
    except Exception:
        pass
    best_overall /= np.linalg.norm(best_overall, axis=1, keepdims=True)
    return best_overall
