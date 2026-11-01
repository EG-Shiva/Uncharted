import time
import numpy as np
from scipy.optimize import linprog, minimize
from scipy.sparse import coo_matrix

def max_cos(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def lp_polish(x, seconds):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = max_cos(x)
    radius, start = 1e-3, time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        if time.time() - start > seconds - 0.3:
            break
        c = (x @ x.T)[iu]
        m = float(c.max())
        sel = c > m - max(20 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p, nv = len(I), n * d + 1
        if p == 0:
            radius *= 0.5
            continue
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv); obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-c[sel], A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
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
            x, best, radius = y, v, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def quench_momentum(x, steps, beta0, beta1, lr0, lr1, mu):
    for s in range(steps):
        t = s / max(steps - 1, 1)
        beta = beta0 * (beta1 / beta0) ** t if beta0 > 0 else beta1
        lr = lr0 + (lr1 - lr0) * t
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        m = G.max()
        W = np.exp(beta * (G - m))
        np.fill_diagonal(W, 0.0)
        Grad = W @ x
        proj = np.sum(Grad * x, axis=1, keepdims=True)
        Tg = Grad - proj * x
        rms = np.sqrt(np.mean(Tg * Tg)) + 1e-12
        Tg_n = Tg / rms
        # momentum without persistent buffer (heavy-ball approx via exponential avg)
        # use simple smoothed update: keep velocity in closure via static
        x = x - lr * Tg_n
        x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x

# ---------- symmetry family ----------
# 5 orbits of size 12 under signed 12-cycle, + 1 orbit of size 8 under signed 8-cycle on first 8 coords
def precompute_orbit(shift, D):
    d = len(D)
    K = d  # orbit length = d for full cycles
    idx = np.zeros((K, d), dtype=int)
    coeff = np.zeros((K, d), dtype=float)
    idx[0] = np.arange(d)
    coeff[0] = 1.0
    for k in range(1, K):
        prev_idx = idx[k-1]
        prev_c = coeff[k-1]
        for i in range(d):
            j = (i - shift) % d
            idx[k, i] = prev_idx[j]
            coeff[k, i] = D[i] * prev_c[j]
    return idx, coeff

def precompute_orbit8(shift8, D8):
    # 12-dim seed, first 8 cycled, last 4 fixed
    K = 8
    idx = np.zeros((K, 12), dtype=int)
    coeff = np.zeros((K, 12), dtype=float)
    idx[0] = np.arange(12)
    coeff[0] = 1.0
    for k in range(1, K):
        prev_idx = idx[k-1]
        prev_c = coeff[k-1]
        for i in range(8):
            j = (i - shift8) % 8
            idx[k, i] = prev_idx[j]
            coeff[k, i] = D8[i] * prev_c[j]
        for i in range(8, 12):
            idx[k, i] = prev_idx[i]
            coeff[k, i] = prev_c[i]
    return idx, coeff

def build_X_from_seeds(seeds_norm, idx12, coeff12, idx8, coeff8):
    # seeds_norm (6,12) already unit rows
    X = np.zeros((68, 12))
    p = 0
    for b in range(5):
        s = seeds_norm[b]
        for k in range(12):
            X[p] = coeff12[k] * s[idx12[k]]
            p += 1
    s = seeds_norm[5]
    for k in range(8):
        X[p] = coeff8[k] * s[idx8[k]]
        p += 1
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    return X

def sym_fg_factory(idx12, coeff12, idx8, coeff8, beta):
    iu = np.triu_indices(68, 1)
    def fg(yf):
        Y = yf.reshape(6, 12)
        nr = np.linalg.norm(Y, axis=1, keepdims=True)
        nr = np.maximum(nr, 1e-12)
        S = Y / nr
        X = build_X_from_seeds(S, idx12, coeff12, idx8, coeff8)
        G = X @ X.T
        g = G[iu]
        m = float(g.max())
        e = np.exp(beta * (g - m))
        Sm = float(e.sum()) + 1e-300
        F = m + np.log(Sm) / beta
        w = e / Sm
        P = np.zeros((68, 68))
        P[iu] = w
        P = P + P.T
        GX = P @ X
        dot = np.sum(GX * X, axis=1, keepdims=True)
        Gx_t = (GX - dot * X)
        # backprop to seeds
        GS = np.zeros((6, 12))
        p = 0
        for b in range(5):
            acc = np.zeros(12)
            for k in range(12):
                gv = Gx_t[p]
                # X[p,i]=c*s[idx]; so d/ds[j] += c*gv[i] where idx==j
                # accumulate via bincount
                ii = idx12[k]
                cc = coeff12[k]
                # vectorized
                np.add.at(acc, ii, cc * gv)
                p += 1
            GS[b] = acc
        acc = np.zeros(12)
        for k in range(8):
            gv = Gx_t[p]
            ii = idx8[k]
            cc = coeff8[k]
            np.add.at(acc, ii, cc * gv)
            p += 1
        GS[5] = acc
        # chain through normalization S=Y/nr
        GY = (GS - np.sum(GS * S, axis=1, keepdims=True) * S) / nr
        return F, GY.ravel()
    return fg

def run_sym_opt(y0, idx12, coeff12, idx8, coeff8, betas, maxiters):
    y = y0.copy()
    for beta, mi in zip(betas, maxiters):
        fg = sym_fg_factory(idx12, coeff12, idx8, coeff8, beta)
        try:
            r = minimize(fg, y, jac=True, method='L-BFGS-B',
                         options={'maxiter': mi, 'ftol': 1e-14, 'gtol': 1e-7, 'maxfun': mi*2})
            y = r.x.copy()
        except Exception:
            break
    return y

def make_fg_all(beta):
    IU = np.triu_indices(68, 1)
    def fg(yf):
        Y = yf.reshape(68, 12)
        nr = np.linalg.norm(Y, axis=1, keepdims=True)
        nr = np.maximum(nr, 1e-12)
        X = Y / nr
        G = X @ X.T
        g = G[IU]
        m = float(g.max())
        e = np.exp(beta * (g - m))
        S = float(e.sum()) + 1e-300
        F = m + np.log(S) / beta
        w = e / S
        P = np.zeros((68, 68))
        P[IU] = w
        P = P + P.T
        GX = P @ X
        dot = np.sum(GX * X, axis=1, keepdims=True)
        GY = (GX - dot * X) / nr
        return F, GY.ravel()
    return fg

def lbfgs_all(x0, beta, maxiter):
    fg = make_fg_all(beta)
    y0 = (x0 / np.linalg.norm(x0, axis=1, keepdims=True)).ravel().copy()
    try:
        r = minimize(fg, y0, jac=True, method='L-BFGS-B',
                     options={'maxiter': maxiter, 'ftol': 1e-14, 'gtol': 1e-7, 'maxfun': maxiter*2})
        Y = r.x.reshape(68, 12)
        Y /= np.linalg.norm(Y, axis=1, keepdims=True)
        return Y
    except Exception:
        return x0

def solve():
    t0 = time.time()
    TIME_LIMIT = 172.0
    rng = np.random.default_rng(20261031)
    # load incumbents
    cands = []
    try:
        xb = np.load("best.npy").astype(float)
        xb /= np.linalg.norm(xb, axis=1, keepdims=True)
        cands.append(xb)
    except Exception:
        pass
    try:
        xr = np.load("record.npy").astype(float)
        xr /= np.linalg.norm(xr, axis=1, keepdims=True)
        cands.append(xr)
    except Exception:
        pass
    if len(cands) == 0:
        x0 = rng.standard_normal((68, 12))
        x0 /= np.linalg.norm(x0, axis=1, keepdims=True)
        cands.append(x0)
    best_val = 1e9
    x_best = cands[0].copy()
    for c in cands:
        v = max_cos(c)
        if v < best_val:
            best_val = v
            x_best = c.copy()

    # ---- symmetry-constrained scan ----
    D0 = np.ones(12)
    D1 = np.ones(12); D1[0] = -1; D1[6] = -1
    D2 = np.ones(12); D2[0] = -1; D2[1] = -1
    D3 = np.ones(12); D3[0] = -1; D3[3] = -1; D3[6] = -1; D3[9] = -1
    D8_0 = np.ones(8)
    D8_1 = np.ones(8); D8_1[0] = -1; D8_1[4] = -1
    cfgs = []
    for D in (D0, D1, D2, D3):
        for D8 in (D8_0, D8_1):
            idx12, coeff12 = precompute_orbit(1, D)
            idx8, coeff8 = precompute_orbit8(1, D8)
            cfgs.append((idx12, coeff12, idx8, coeff8))
    # limit to 6 cfgs to save time
    cfgs = [cfgs[0], cfgs[1], cfgs[2], cfgs[4], cfgs[6], cfgs[7]]
    sym_best = None
    sym_val = 1e9
    sym_cfg_best = cfgs[0]
    sym_deadline = t0 + 48.0
    for ci, (idx12, coeff12, idx8, coeff8) in enumerate(cfgs):
        for rep in range(2):
            if time.time() > sym_deadline:
                break
            y0 = rng.standard_normal(6*12)
            # staged betas
            yopt = run_sym_opt(y0, idx12, coeff12, idx8, coeff8, betas=[80.0, 220.0], maxiters=[80, 100])
            Y = yopt.reshape(6, 12)
            Y /= np.linalg.norm(Y, axis=1, keepdims=True)
            X = build_X_from_seeds(Y, idx12, coeff12, idx8, coeff8)
            v = max_cos(X)
            if v < sym_val:
                sym_val = v
                sym_best = X.copy()
                sym_cfg_best = (idx12, coeff12, idx8, coeff8)
            # one more high-beta refinement of the promising one immediately
            if v < best_val + 0.005:
                yopt2 = run_sym_opt(yopt, idx12, coeff12, idx8, coeff8, betas=[450.0], maxiters=[80])
                Y2 = yopt2.reshape(6, 12)
                Y2 /= np.linalg.norm(Y2, axis=1, keepdims=True)
                X2 = build_X_from_seeds(Y2, idx12, coeff12, idx8, coeff8)
                v2 = max_cos(X2)
                if v2 < sym_val:
                    sym_val = v2
                    sym_best = X2.copy()
                    sym_cfg_best = (idx12, coeff12, idx8, coeff8)
        if time.time() > sym_deadline:
            break

    # release symmetry: full-DOF polish of symmetric best
    if sym_best is not None:
        Xrel = sym_best.copy()
        # L-BFGS release with increasing beta
        for beta, mi in [(150.0, 120), (400.0, 120)]:
            if time.time() > t0 + 75.0:
                break
            Xrel = lbfgs_all(Xrel, beta, mi)
        vrel = max_cos(Xrel)
        # short LP
        if time.time() < t0 + 85.0:
            Xrel = lp_polish(Xrel, min(4.0, max(1.0, t0 + 85.0 - time.time())))
            vrel = max_cos(Xrel)
        if vrel < best_val:
            best_val = vrel
            x_best = Xrel.copy()
        # keep sym descendant as alternate basin even if slightly worse: store second candidate
        sym_released = Xrel.copy()
        sym_released_val = vrel
    else:
        sym_released = None

    # ---- basin hopping from best + symmetric descendant ----
    explore_end = t0 + TIME_LIMIT - 28.0
    trial = 0
    pool = [x_best.copy()]
    if sym_best is not None:
        try:
            pool.append(sym_released.copy())
        except Exception:
            pass
    while time.time() < explore_end:
        trial += 1
        # pick base: mostly best, sometimes second
        if len(pool) > 1 and (trial % 4 == 0):
            base = pool[1]
        else:
            base = x_best
        x = base.copy()
        u = rng.random()
        if u < 0.45:
            sigma = float(rng.choice([0.006, 0.015, 0.035, 0.08]))
            x = x + sigma * rng.standard_normal(x.shape)
            x /= np.linalg.norm(x, axis=1, keepdims=True)
            beta0 = float(rng.uniform(40, 120))
            beta1 = float(rng.uniform(320, 700))
            lr0 = float(rng.uniform(0.02, 0.05))
            lr1 = float(rng.uniform(0.001, 0.005))
            x = quench_momentum(x, int(rng.integers(400, 750)), beta0, beta1, lr0, lr1, float(rng.uniform(0.0, 0.9)))
        elif u < 0.75:
            # cluster reshuffle of worst region
            G = x @ x.T
            np.fill_diagonal(G, -2.0)
            per = G.max(axis=1)
            seed = int(np.argmax(per)) if rng.random() < 0.7 else int(rng.choice(68, p=np.exp(60*(per-per.max()))/np.exp(60*(per-per.max())).sum()))
            k = int(rng.integers(4, 9))
            nbrs = np.argsort(-G[seed])[:k]
            cluster = np.unique(np.concatenate([[seed], nbrs]))
            if rng.random() < 0.5:
                for q in cluster:
                    vv = rng.standard_normal(12); vv /= np.linalg.norm(vv); x[q] = vv
            else:
                x[cluster] += float(rng.uniform(0.3, 0.7)) * rng.standard_normal((len(cluster), 12))
                x /= np.linalg.norm(x, axis=1, keepdims=True)
            x = quench_momentum(x, int(rng.integers(500, 800)), float(rng.uniform(50,120)), float(rng.uniform(350,700)), 0.03, 0.002, float(rng.uniform(0.2,0.9)))
        else:
            # L-BFGS kick
            sigma = float(rng.uniform(0.01, 0.05))
            x = x + sigma * rng.standard_normal(x.shape)
            x /= np.linalg.norm(x, axis=1, keepdims=True)
            x = lbfgs_all(x, float(rng.choice([120.0, 250.0, 400.0])), int(rng.integers(80, 160)))
        x = quench_momentum(x, 150, 450.0, 650.0, 0.008, 0.001, 0.6)
        v = max_cos(x)
        if v < best_val - 1e-12:
            remain = explore_end - time.time()
            if remain > 3.0 and v < best_val - 2e-7:
                x = lp_polish(x, min(3.0, remain - 0.5))
                v = max_cos(x)
            if v < best_val:
                x_best = x.copy()
                best_val = v
                pool[0] = x_best.copy()
        elif v < best_val + 0.002 and (trial % 4 == 0) and (time.time() < explore_end - 4.0):
            y = lp_polish(x, 1.2)
            v2 = max_cos(y)
            if v2 < best_val:
                x_best = y.copy(); best_val = v2; pool[0] = x_best.copy()
        if trial > 300:
            break

    remain = TIME_LIMIT - (time.time() - t0)
    if remain > 3.0:
        x_best = lp_polish(x_best, min(remain - 1.0, 25.0))
    x_best = x_best / np.linalg.norm(x_best, axis=1, keepdims=True)
    return x_best
