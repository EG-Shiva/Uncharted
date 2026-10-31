import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _norm(x):
    x = np.asarray(x, dtype=float)
    x = np.where(np.isfinite(x), x, 0.0)
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n < 1e-12] = 1.0
    return x / n

def _max_cos(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _contacts(x, tol=1e-9):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    m = G.max()
    C = G > m - tol
    deg = C.sum(axis=1)
    pm = G.max(axis=1)
    return m, deg, pm, G

def _rand_orth(s, rng):
    M = np.eye(4) + s * rng.standard_normal((4, 4))
    U, _, Vt = np.linalg.svd(M)
    Q = U @ Vt
    if np.linalg.det(Q) < 0:
        U[:, -1] *= -1
        Q = U @ Vt
    return Q

def _best_void_single(others, rng, n_cand=4000):
    cand = rng.standard_normal((n_cand, 4))
    cand /= np.linalg.norm(cand, axis=1, keepdims=True)
    s = cand @ others.T
    sc = s.max(axis=1)
    bi = int(np.argmin(sc))
    b = cand[bi].copy()
    bs = float(sc[bi])
    step = 0.05
    for _ in range(200):
        prop = b + step * rng.standard_normal((32, 4))
        prop /= np.linalg.norm(prop, axis=1, keepdims=True)
        sc2 = (prop @ others.T).max(axis=1)
        j = int(np.argmin(sc2))
        if sc2[j] < bs:
            b, bs = prop[j], float(sc2[j])
            step *= 1.05
        else:
            step *= 0.97
        if step < 1e-5:
            break
    return b

def _balanced_anneal(x, steps, tau0, tau1, lr0, lr1, lam0, lam1, tau2, rng_seed=0):
    n, d = x.shape
    m = np.zeros_like(x)
    v = np.zeros_like(x)
    b1, b2, eps = 0.9, 0.999, 1e-8
    for k in range(steps):
        frac = k / (steps - 1) if steps > 1 else 0.0
        tau = tau0 * (tau1 / tau0) ** frac
        lr = lr0 + (lr1 - lr0) * frac
        lam = lam0 + (lam1 - lam0) * frac
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        gm = G.max()
        # pair soft-max weights
        E = np.exp((G - gm) / tau, out=G) if False else np.exp((G - gm) / tau)
        np.fill_diagonal(E, 0.0)
        S = E.sum()
        if not np.isfinite(S) or S == 0:
            break
        W = E / S
        grad_pair = W @ x
        # per-point soft-max for min balancing
        # row max
        G2 = x @ x.T
        np.fill_diagonal(G2, -2.0)
        rm = G2.max(axis=1, keepdims=True)
        V = np.exp((G2 - rm) / tau2)
        # row normalize (exclude diag which is ~0)
        rs = V.sum(axis=1, keepdims=True)
        rs[rs == 0] = 1.0
        V = V / rs
        # soft-min weights over rm
        rmin = rm.min()
        u = np.exp(-(rm[:, 0] - rmin) / tau2)
        u = u / u.sum()
        UV = u[:, None] * V
        grad_min = (UV + UV.T) @ x
        grad = grad_pair - lam * grad_min
        dot = (grad * x).sum(axis=1, keepdims=True)
        g = grad - dot * x
        m *= b1
        m += (1 - b1) * g
        v *= b2
        v += (1 - b2) * (g * g)
        bc1 = 1.0 - b1 ** (k + 1)
        bc2 = 1.0 - b2 ** (k + 1)
        mh = m / bc1
        vh = v / bc2
        step = mh / (np.sqrt(vh) + eps)
        np.clip(step, -2.0, 2.0, out=step)
        x = x - lr * step
        x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x

def _lp_polish(x, seconds, deadline):
    x = _norm(x)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and time.time() < deadline and radius > 1e-14:
        c = (x @ x.T)[iu]
        mcur = c.max()
        sel = c > mcur - max(20 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p, nv = len(I), n * d + 1
        if p == 0:
            break
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv)
        obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c[sel], A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y = _norm(y)
        vv = _max_cos(y)
        if vv < best:
            x, best, radius = y, vv, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def solve():
    rng = np.random.default_rng(11)
    t0 = time.time()
    deadline = t0 + 172.0
    try:
        x0 = np.load("best.npy")
    except Exception:
        x0 = np.load("record.npy")
    x0 = _norm(np.asarray(x0, dtype=float))
    if x0.shape != (272, 4):
        flat = x0.ravel()
        need = 272 * 4
        if flat.size < need:
            flat = np.concatenate([flat, rng.standard_normal((need - flat.size,))])
        x0 = _norm(flat[:need].reshape(272, 4))
    x_best = x0.copy()
    best_val = _max_cos(x_best)
    trial = 0
    # basin hopping with coordinated moves, reserve ~35s for final polish
    while time.time() < t0 + 122 and time.time() < deadline - 45:
        trial += 1
        mode = trial % 3
        if mode == 0:
            # block rotation: split by random hyperplane, rotate each block
            u = rng.standard_normal((4,))
            u /= np.linalg.norm(u)
            s = float(u @ x_best.T)
            mask = s >= np.median(s)
            sc = 0.12 + 0.10 * ((trial // 3) % 3) / 2.0
            Q1 = _rand_orth(sc, rng)
            Q2 = _rand_orth(sc, rng)
            x = x_best.copy()
            x[mask] = x[mask] @ Q1.T
            x[~mask] = x[~mask] @ Q2.T
            x = x + 0.006 * rng.standard_normal(x.shape)
            x = _norm(x)
            steps = 500
            tau0, tau1 = 0.012, 0.0012
            lr0, lr1 = 0.008, 0.0008
            lam0, lam1 = 0.25, 0.05
        elif mode == 1:
            # patch resample: hubs + rattlers -> greedy voids (fully tight)
            _, deg, pm, G = _contacts(x_best, tol=1e-7)
            order_hi = np.argsort(-deg)
            order_lo = np.argsort(pm)  # lowest max first = rattlers/isolated
            k = 8 + (trial % 3) * 2  # 8..12
            victims = set(order_lo[:3].tolist())
            # dense patch around top hub
            hub = int(order_hi[0])
            d2 = G[hub].copy()
            # nearest = largest dot
            nn = np.argsort(-d2)[:k]
            for ii in nn[:k - len(victims)]:
                victims.add(int(ii))
            victims = list(victims)[:k]
            keep_mask = np.ones(272, dtype=bool)
            keep_mask[victims] = False
            fixed = x_best[keep_mask].copy()
            # sequential greedy voids
            new_pts = []
            for _ in victims:
                others = np.vstack([fixed] + new_pts) if new_pts else fixed
                b = _best_void_single(others, rng, n_cand=4000)
                new_pts.append(b[None, :])
            x = x_best.copy()
            for vi, bp in zip(victims, new_pts):
                x[vi] = bp[0]
            x = _norm(x)
            steps = 550
            tau0, tau1 = 0.015, 0.001
            lr0, lr1 = 0.010, 0.0008
            lam0, lam1 = 0.30, 0.05
        else:
            # small Gaussian kick + balanced anneal at lower temp
            sigma = 0.018
            x = x_best + sigma * rng.standard_normal((272, 4))
            x = _norm(x)
            steps = 450
            tau0, tau1 = 0.007, 0.0007
            lr0, lr1 = 0.005, 0.0005
            lam0, lam1 = 0.15, 0.02
        tau2 = 0.004 + 0.002 * ((trial % 2))
        x = _balanced_anneal(x, steps, tau0, tau1, lr0, lr1, lam0, lam1, tau2)
        v = _max_cos(x)
        if v < best_val:
            best_val = v
            x_best = x.copy()
    # short low-temp refinement of best
    for _ in range(2):
        if time.time() > deadline - 42:
            break
        x = _balanced_anneal(x_best.copy(), 350, 0.005, 0.0006, 0.004, 0.0004, 0.10, 0.0, 0.004)
        v = _max_cos(x)
        if v < best_val:
            best_val = v
            x_best = x.copy()
    remain = deadline - time.time() - 2.0
    if remain > 5:
        xp = _lp_polish(x_best, min(remain, 36.0), deadline - 1.0)
        vp = _max_cos(xp)
        if vp <= best_val:
            x_best = xp
            best_val = vp
    if not np.all(np.isfinite(x_best)):
        x_best = x0.copy()
    x_best = _norm(x_best)
    return x_best
