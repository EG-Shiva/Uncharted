import time
import numpy as np
from scipy.optimize import linprog, minimize
from scipy.sparse import coo_matrix

def max_cos(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def per_max(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return G.max(axis=1)

def smooth_gd(x, steps, beta0, beta1, lr0, lr1):
    n, d = x.shape
    for s in range(steps):
        t = s / max(steps - 1, 1)
        beta = beta0 * (beta1 / beta0) ** t if beta0 > 0 else beta1
        lr = lr0 + (lr1 - lr0) * t
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        W = np.exp(beta * (G - m))
        np.fill_diagonal(W, 0.0)
        Grad = W @ x
        proj = np.sum(Grad * x, axis=1, keepdims=True)
        Tg = Grad - proj * x
        rms = float(np.sqrt(np.mean(Tg * Tg))) + 1e-12
        x = x - (lr / rms) * Tg
        x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x

def make_fg_full(n, d, IU, beta, Xfix=None):
    # full optimization
    def fg(yf):
        Y = yf.reshape(n, d)
        nr = np.maximum(np.linalg.norm(Y, axis=1, keepdims=True), 1e-12)
        X = Y / nr
        G = X @ X.T
        g = G[IU]
        m = float(g.max())
        e = np.exp(beta * (g - m))
        S = float(e.sum()) + 1e-300
        F = m + np.log(S) / beta
        w = e / S
        P = np.zeros((n, n))
        P[IU] = w
        P = P + P.T
        GX = P @ X
        dot = np.sum(GX * X, axis=1, keepdims=True)
        GY = (GX - dot * X) / nr
        return F, GY.ravel()
    return fg

def run_full_lbfgs(x, beta, maxiter):
    n, d = x.shape
    IU = np.triu_indices(n, 1)
    fg = make_fg_full(n, d, IU, beta)
    y0 = x.ravel().copy()
    try:
        r = minimize(fg, y0, jac=True, method='L-BFGS-B',
                     options={'maxiter': maxiter, 'ftol': 1e-14, 'gtol': 1e-7, 'maxfun': maxiter*2})
        Y = r.x.reshape(n, d)
        Y /= np.linalg.norm(Y, axis=1, keepdims=True)
        return Y
    except Exception:
        return x

def run_subset_lbfgs(x, idx, beta, maxiter=200):
    n, d = x.shape
    k = len(idx)
    if k == 0 or k >= n:
        return run_full_lbfgs(x, beta, maxiter)
    idx = np.asarray(idx, dtype=int)
    Xf = x.copy()
    def fg(yf):
        Y = yf.reshape(k, d)
        nr = np.maximum(np.linalg.norm(Y, axis=1, keepdims=True), 1e-12)
        Xn = Y / nr
        X = Xf.copy()
        X[idx] = Xn
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        # smooth max over upper triangle
        IU = np.triu_indices(n, 1)
        g = G[IU]
        e = np.exp(beta * (g - m))
        S = float(e.sum()) + 1e-300
        F = m + np.log(S) / beta
        w = e / S
        P = np.zeros((n, n))
        P[IU] = w
        P = P + P.T
        GX = P @ X
        # gradient w.r.t X rows, then chain to Y for idx only
        # dF/dX_i = GX_i - projected? Actually full grad before projection:
        # For unit-normalized param, grad_Y = (GX_sub - dot*Xsub)/nr
        dot = np.sum(GX[idx] * Xn, axis=1, keepdims=True)
        GY = (GX[idx] - dot * Xn) / nr
        return F, GY.ravel()
    y0 = x[idx].ravel().copy()
    try:
        r = minimize(fg, y0, jac=True, method='L-BFGS-B',
                     options={'maxiter': maxiter, 'ftol': 1e-15, 'gtol': 1e-8, 'maxfun': maxiter*2})
        Y = r.x.reshape(k, d)
        Y /= np.linalg.norm(Y, axis=1, keepdims=True)
        X = x.copy()
        X[idx] = Y
        X /= np.linalg.norm(X, axis=1, keepdims=True)
        return X
    except Exception:
        return x

def core_indices(x, maxk=16, margin=3e-4, toppairs=120):
    n = x.shape[0]
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    m = float(G.max())
    iu = np.triu_indices(n, 1)
    c = G[iu]
    order = np.argsort(-c)
    take = min(toppairs, len(order))
    sel = order[:take]
    # keep those within margin or at least top 8 pairs
    keep = [s for s in sel if c[s] > m - margin]
    if len(keep) < 8:
        keep = sel[:8].tolist()
    I = iu[0][keep]; J = iu[1][keep]
    uniq = list(set(I.tolist() + J.tolist()))
    # rank by per_max
    pm = per_max(x)
    uniq.sort(key=lambda q: -pm[q])
    return uniq[:maxk], m

def lp_polish(x, seconds):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = max_cos(x)
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds - 0.25 and radius > 1e-14:
        c = (x @ x.T)[iu]
        m = float(c.max())
        sel = c > m - max(20 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        if p == 0:
            radius *= 0.5
            continue
        if p > 1600:
            o = np.argsort(-c[sel])[:1600]
            I, J = I[o], J[o]
            csel = c[sel][o]
            p = 1600
        else:
            csel = c[sel]
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
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
        y = x + res.x[:-1].reshape(n, d)
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        v = max_cos(y)
        if v < best:
            x, best, radius = y, v, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def best_hole(rest, rng, ncand=12000):
    d = rest.shape[1]
    cand = rng.standard_normal((ncand, d))
    cand /= np.linalg.norm(cand, axis=1, keepdims=True)
    s = (cand @ rest.T).max(axis=1)
    return cand[int(np.argmin(s))]

def solve():
    t0 = time.time()
    TLIM = 172.0
    rng = np.random.default_rng(7)
    cands = []
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn).astype(float)
            a /= np.linalg.norm(a, axis=1, keepdims=True)
            if a.shape == (68, 12) and np.all(np.isfinite(a)):
                cands.append(a)
        except Exception:
            pass
    if len(cands) == 0:
        r0 = np.random.default_rng(0)
        x_best = r0.standard_normal((68, 12))
        x_best /= np.linalg.norm(x_best, axis=1, keepdims=True)
    else:
        bv = 1e9; x_best = cands[0]
        for c in cands:
            v = max_cos(c)
            if v < bv:
                bv, x_best = v, c.copy()
    x_best = x_best / np.linalg.norm(x_best, axis=1, keepdims=True)
    best_val = max_cos(x_best)
    n, d = x_best.shape
    # Stage 0: direct attack on isolated core
    try:
        for _ in range(6):
            if time.time() - t0 > 20:
                break
            idx, m = core_indices(x_best, maxk=12, margin=5e-4, toppairs=120)
            x = run_subset_lbfgs(x_best, idx, beta=900.0, maxiter=250)
            v = max_cos(x)
            if v < best_val:
                x_best, best_val = x, v
            else:
                # try larger core / higher beta
                idx2, _ = core_indices(x_best, maxk=18, margin=1e-3, toppairs=200)
                x2 = run_subset_lbfgs(x_best, idx2, beta=1800.0, maxiter=200)
                v2 = max_cos(x2)
                if v2 < best_val:
                    x_best, best_val = x2, v2
                else:
                    break
        # short LP to lock contacts
        if time.time() - t0 < 30:
            y = lp_polish(x_best, min(4.0, TLIM - (time.time()-t0) - 100.0 if (TLIM-(time.time()-t0))>105 else 2.0))
            vy = max_cos(y)
            if vy < best_val:
                x_best, best_val = y, vy
    except Exception:
        pass
    explore_end = t0 + TLIM - 34.0
    trial = 0
    sigmas = [0.004, 0.012, 0.03, 0.07, 0.15, 0.30]
    while time.time() < explore_end and trial < 300:
        trial += 1
        x = x_best.copy()
        u = rng.random()
        pm = per_max(x)
        order = np.argsort(-pm)
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        # locate worst pair
        flat = int(np.argmax(G))
        wi, wj = flat // n, flat % n
        if u < 0.45:
            # cooperative kick of worst pair + neighbours
            neigh_i = np.argsort(-G[wi])[:6]
            neigh_j = np.argsort(-G[wj])[:6]
            kick_idx = list(set([wi, wj] + neigh_i.tolist() + neigh_j.tolist()))[:10]
            sigma = float(sigmas[int(rng.integers(0, len(sigmas)))])
            # larger sigma for the core pair itself
            x[kick_idx] += sigma * rng.standard_normal((len(kick_idx), d))
            # occasionally reinsert worst point into hole
            if rng.random() < 0.35:
                q = int(order[0])
                mask = np.ones(n, dtype=bool); mask[q] = False
                hole = best_hole(x[mask], rng, ncand=8000)
                alpha = float(rng.uniform(0.75, 1.0))
                v = alpha * hole + (1 - alpha) * rng.standard_normal(d)
                v /= (np.linalg.norm(v) + 1e-12)
                x[q] = v
            x /= np.linalg.norm(x, axis=1, keepdims=True)
        elif u < 0.65:
            k = int(rng.integers(1, 4))
            for q in order[:k]:
                mask = np.ones(n, dtype=bool); mask[q] = False
                hole = best_hole(x[mask], rng, ncand=6000)
                mix = float(rng.uniform(0.6, 1.0))
                v = mix * hole + (1 - mix) * x[q]
                v /= (np.linalg.norm(v) + 1e-12)
                x[q] = v
            x += 0.008 * rng.standard_normal(x.shape)
            x /= np.linalg.norm(x, axis=1, keepdims=True)
        elif u < 0.85:
            sigma = float(sigmas[int(rng.integers(0, len(sigmas)))])
            x = x + sigma * rng.standard_normal(x.shape)
            x /= np.linalg.norm(x, axis=1, keepdims=True)
        else:
            k = int(rng.integers(4, 11))
            idx = rng.choice(n, size=k, replace=False)
            for q in idx:
                v = rng.standard_normal(d); v /= np.linalg.norm(v)
                x[q] = v
            x += 0.003 * rng.standard_normal(x.shape)
            x /= np.linalg.norm(x, axis=1, keepdims=True)
        # fast smooth descent
        try:
            b0 = float(rng.uniform(70, 130))
            b1 = float(rng.uniform(350, 650))
            x = smooth_gd(x, steps=180, beta0=b0, beta1=b1, lr0=0.035, lr1=0.008)
        except Exception:
            continue
        v = max_cos(x)
        # gate: only expensive refine if promising
        if v < best_val + 0.0015 or rng.random() < 0.08:
            try:
                idxc, _ = core_indices(x, maxk=12, margin=4e-4, toppairs=120)
                # one subset blast with moderate beta
                x = run_subset_lbfgs(x, idxc, beta=700.0, maxiter=120)
            except Exception:
                pass
            v = max_cos(x)
            if v < best_val - 1e-12:
                try:
                    x = run_full_lbfgs(x, beta=500.0, maxiter=120)
                except Exception:
                    pass
                v = max_cos(x)
                if v < best_val - 5e-7 and (explore_end - time.time()) > 6.0:
                    y = lp_polish(x, min(2.5, explore_end - time.time() - 0.3))
                    v2 = max_cos(y)
                    if v2 < v:
                        x, v = y, v2
                if v < best_val:
                    x_best, best_val = x.copy(), v
            elif v < best_val + 0.0005 and (trial % 12 == 0) and (time.time() < explore_end - 5.0):
                try:
                    y = lp_polish(x, 1.0)
                    v2 = max_cos(y)
                    if v2 < best_val:
                        x_best, best_val = y.copy(), v2
                except Exception:
                    pass
    # final intensive polish
    try:
        remain = TLIM - (time.time() - t0)
        if remain > 6.0:
            for _ in range(2):
                if TLIM - (time.time() - t0) < 9.0:
                    break
                idxc, _ = core_indices(x_best, maxk=20, margin=8e-4, toppairs=200)
                x_best = run_subset_lbfgs(x_best, idxc, beta=1200.0, maxiter=250)
            x_best = run_full_lbfgs(x_best, beta=800.0, maxiter=180)
            x_best = run_full_lbfgs(x_best, beta=1800.0, maxiter=150)
            remain = TLIM - (time.time() - t0)
            if remain > 4.0:
                x_best = lp_polish(x_best, max(2.5, remain - 1.0))
    except Exception:
        pass
    x_best = x_best / np.linalg.norm(x_best, axis=1, keepdims=True)
    x_best = np.nan_to_num(x_best)
    nr = np.linalg.norm(x_best, axis=1, keepdims=True)
    nr[nr == 0] = 1.0
    x_best = x_best / nr
    return x_best
