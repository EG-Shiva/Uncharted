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

def make_fg(n, d, IU, beta):
    def fg(yf):
        Y = yf.reshape(n, d)
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
        P = np.zeros((n, n))
        P[IU] = w
        P = P + P.T
        GX = P @ X
        dot = np.sum(GX * X, axis=1, keepdims=True)
        GY = (GX - dot * X) / nr
        return F, GY.ravel()
    return fg

def run_lbfgs(x0, beta, maxiter):
    n, d = x0.shape
    IU = np.triu_indices(n, 1)
    fg = make_fg(n, d, IU, beta)
    y0 = (x0 / np.linalg.norm(x0, axis=1, keepdims=True)).ravel().copy()
    try:
        r = minimize(fg, y0, jac=True, method='L-BFGS-B',
                     options={'maxiter': maxiter, 'ftol': 1e-14, 'gtol': 1e-7, 'maxfun': maxiter*2, 'maxcor': 12})
        Y = r.x.reshape(n, d)
        nrm = np.linalg.norm(Y, axis=1, keepdims=True) + 1e-300
        Y = Y / nrm
        return Y
    except Exception:
        return x0

def lp_polish(x, seconds, deadline=None):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = max_cos(x)
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while True:
        if time.time() - start > seconds - 0.3:
            break
        if deadline is not None and time.time() > deadline - 0.6:
            break
        if radius < 1e-14:
            break
        c = (x @ x.T)[iu]
        m = float(c.max())
        sel = c > m - max(20.0 * radius, 1e-9)
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

def best_hole_for_rest(x_rest, rng, ncand):
    d = x_rest.shape[1]
    cand = rng.standard_normal((ncand, d))
    cand /= np.linalg.norm(cand, axis=1, keepdims=True)
    scores = (cand @ x_rest.T).max(axis=1)
    return cand[int(np.argmin(scores))]

def greedy_hole_sweep(x, rng, k=8, ncand=12000):
    n = x.shape[0]
    cur = max_cos(x)
    pm = per_max(x)
    order = np.argsort(-pm)[:k]
    for idx in order:
        mask = np.ones(n, dtype=bool); mask[idx] = False
        Xr = x[mask]
        # current value without idx
        G = Xr @ Xr.T
        np.fill_diagonal(G, -2.0)
        rest_max = float(G.max())
        hole = best_hole_for_rest(Xr, rng, ncand=ncand)
        hmax = float((Xr @ hole).max())
        new_max = hmax if hmax > rest_max else rest_max
        if new_max < cur - 1e-13:
            x[idx] = hole
            cur = new_max
    return x, cur

def random_orth(d, rng):
    A = rng.standard_normal((d, d))
    Q, R = np.linalg.qr(A)
    s = np.sign(np.diag(R))
    s[s == 0] = 1.0
    Q = Q * s[None, :]
    return Q

def solve():
    t0 = time.time()
    LIMIT = 172.0
    deadline = t0 + LIMIT
    try:
        xb = np.load("best.npy").astype(float)
    except Exception:
        xb = None
    try:
        xr = np.load("record.npy").astype(float)
    except Exception:
        xr = None
    cands = []
    if xb is not None:
        cands.append(xb)
    if xr is not None:
        cands.append(xr)
    if len(cands) == 0:
        rng0 = np.random.default_rng(0)
        x_init = rng0.standard_normal((68, 12))
        x_init /= np.linalg.norm(x_init, axis=1, keepdims=True)
        best_val = max_cos(x_init)
        x_best = x_init.copy()
    else:
        best_val = 1e9
        x_best = None
        for c in cands:
            c = np.asarray(c, dtype=float)
            c = c / np.linalg.norm(c, axis=1, keepdims=True)
            v = max_cos(c)
            if v < best_val:
                best_val = v
                x_best = c.copy()
    x_best = x_best / np.linalg.norm(x_best, axis=1, keepdims=True)
    n, d = x_best.shape
    rng = np.random.default_rng(20261031)
    # elite population for diversity
    elites = [(best_val, x_best.copy())]

    explore_end = deadline - 38.0
    trial = 0
    sigmas = [0.04, 0.10, 0.22, 0.40, 0.60]
    while time.time() < explore_end:
        trial += 1
        if time.time() > deadline - 40:
            break
        # pick base: mostly best, sometimes elite
        if len(elites) > 1 and rng.random() < 0.25:
            bi = int(rng.integers(0, len(elites)))
            base = elites[bi][1].copy()
        else:
            base = x_best.copy()
        x = base.copy()
        u = rng.random()
        pm = per_max(x)
        order = np.argsort(-pm)
        if u < 0.22:
            sigma = float(sigmas[int(rng.integers(0, len(sigmas)))])
            noise = rng.standard_normal(x.shape)
            # bias to overloaded top-10
            boost = np.ones((n, 1))
            boost[order[:10]] *= 1.8
            x = x + sigma * boost * noise
            x /= np.linalg.norm(x, axis=1, keepdims=True)
        elif u < 0.45:
            k = int(rng.integers(2, 11))
            idxset = set()
            nw = min(k, int(rng.integers(1, 4)))
            for j in range(nw):
                idxset.add(int(order[j]))
            while len(idxset) < k:
                idxset.add(int(rng.integers(0, n)))
            idxlist = list(idxset)
            for j in idxlist:
                if rng.random() < 0.65:
                    mask = np.ones(n, dtype=bool); mask[j] = False
                    Xr = x[mask]
                    x[j] = best_hole_for_rest(Xr, rng, ncand=8000)
                else:
                    v = rng.standard_normal(d); v /= np.linalg.norm(v)
                    x[j] = v
        elif u < 0.60:
            # cap shake
            vdir = rng.standard_normal(d); vdir /= np.linalg.norm(vdir)
            dots = x @ vdir
            thresh = np.quantile(dots, 0.78)
            sel = np.where(dots > thresh)[0]
            if len(sel) < 3:
                sel = order[:8]
            sigma = float(rng.uniform(0.15, 0.45))
            x[sel] = x[sel] + sigma * rng.standard_normal((len(sel), d))
            x /= np.linalg.norm(x, axis=1, keepdims=True)
        elif u < 0.72:
            # subset orthogonal rotation (preserves internal structure)
            k = int(rng.integers(8, 21))
            sel = rng.choice(n, size=k, replace=False)
            Q = random_orth(d, rng)
            x[sel] = x[sel] @ Q.T
            # small jitter after rotation
            sigma = float(rng.uniform(0.02, 0.12))
            x[sel] = x[sel] + sigma * rng.standard_normal((k, d))
            x /= np.linalg.norm(x, axis=1, keepdims=True)
        elif u < 0.80:
            # antipodal rebuild: force near-antipodal pairs
            perm = rng.permutation(n)
            h = n // 2
            A = perm[:h]
            B = perm[h:2*h] if n % 2 == 0 else perm[h:h+h]
            # keep A, set B = -A + noise
            sigma = float(rng.uniform(0.02, 0.20))
            x[B] = -x[A] + sigma * rng.standard_normal((h, d))
            x /= np.linalg.norm(x, axis=1, keepdims=True)
        else:
            # fresh block randomisation
            k = int(rng.integers(4, 14))
            sel = rng.choice(n, size=k, replace=False)
            for j in sel:
                v = rng.standard_normal(d); v /= np.linalg.norm(v)
                x[j] = v
            sigma = float(rng.uniform(0.03, 0.10))
            x = x + sigma * rng.standard_normal(x.shape)
            x /= np.linalg.norm(x, axis=1, keepdims=True)

        # L-BFGS ladder with randomized betas
        b1 = float(rng.choice(np.array([40.0, 70.0, 110.0, 160.0])))
        b2 = float(rng.choice(np.array([220.0, 350.0, 500.0, 700.0])))
        # adapt iters to remaining time
        time_left = explore_end - time.time()
        if time_left < 6:
            m1, m2 = 60, 60
        elif time_left < 20:
            m1, m2 = 100, 80
        else:
            m1, m2 = 150, 120
        x = run_lbfgs(x, b1, maxiter=m1)
        x = run_lbfgs(x, b2, maxiter=m2)
        v = max_cos(x)
        # cheap hole sweep if promising
        if v < best_val + 0.004:
            x, v = greedy_hole_sweep(x, rng, k=6, ncand=8000)
            # one more short high-beta snap
            if v < best_val + 0.002:
                x = run_lbfgs(x, 800.0, maxiter=60)
                v = max_cos(x)
        if v < best_val - 1e-12:
            best_val = v
            x_best = x.copy()
            elites.append((v, x.copy()))
            # keep elites small and diverse
            elites = sorted(elites, key=lambda e: e[0])[:6]
            if time.time() < explore_end - 3:
                x_lp = lp_polish(x_best, seconds=1.5, deadline=deadline)
                vlp = max_cos(x_lp)
                if vlp < best_val:
                    best_val = vlp
                    x_best = x_lp.copy()
                    elites[0] = (best_val, x_best.copy())
        else:
            # add diverse near-best to elites
            if v < best_val + 0.008 and len(elites) < 6:
                # check diversity: min dist
                ok = True
                for _, e in elites:
                    # chordal distance proxy: mean abs dot after best perm? use simple Frobenius after alignment via rotation? cheap: compare max_cos diff and fro
                    if np.linalg.norm(np.sort((x @ e.T).ravel())[-10:] - np.sort((e @ e.T).ravel())[-10:]) < 1e-6:
                        pass
                if ok and rng.random() < 0.2:
                    elites.append((v, x.copy()))
                    elites = sorted(elites, key=lambda e: e[0])[:6]

    # final intensive polish of global best
    # hole sweeps + high beta + LP with remaining time
    try:
        x_best, bv = greedy_hole_sweep(x_best, rng, k=12, ncand=20000)
        best_val = bv
    except Exception:
        pass
    try:
        if deadline - time.time() > 6:
            x_best = run_lbfgs(x_best, 500.0, maxiter=250)
            x_best = run_lbfgs(x_best, 900.0, maxiter=200)
            best_val = max_cos(x_best)
    except Exception:
        pass
    try:
        x_best, bv = greedy_hole_sweep(x_best, rng, k=16, ncand=20000)
        best_val = bv
    except Exception:
        pass
    rem = deadline - time.time() - 0.8
    if rem > 3:
        try:
            x_lp = lp_polish(x_best, seconds=rem, deadline=deadline)
            vlp = max_cos(x_lp)
            if vlp < best_val:
                best_val = vlp
                x_best = x_lp.copy()
        except Exception:
            pass
    x_best = np.asarray(x_best, dtype=float)
    x_best /= np.linalg.norm(x_best, axis=1, keepdims=True)
    return x_best
