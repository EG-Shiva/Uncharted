import time
import numpy as np
from scipy.optimize import linprog, minimize
from scipy.sparse import coo_matrix


def max_cos(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())


def lp_polish(x, seconds, t_deadline=None):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = max_cos(x)
    best_x = x.copy()
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds:
        if t_deadline is not None and time.time() > t_deadline:
            break
        if radius < 1e-14:
            break
        Gfull = x @ x.T
        c_full = Gfull[iu]
        m = c_full.max()
        sel = c_full > m - max(20 * radius, 1e-9)
        I = iu[0][sel]; J = iu[1][sel]
        c = c_full[sel]
        p = len(I)
        if p == 0:
            radius *= 0.5
            continue
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c, A_eq=A_eq, b_eq=np.zeros(n),
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
            best = v
            best_x = y.copy()
            x = y
            radius = min(radius * 1.5, 0.05)
        else:
            # keep x at best? stay at x, shrink
            radius *= 0.5
            # if stuck, jump to best_x
            if radius < 1e-7:
                x = best_x.copy()
                radius = 1e-3
                # avoid infinite loop
                if time.time() - start > seconds * 0.9:
                    break
    return best_x


def lbfgs_smooth(x0, k, maxiter=120):
    n, d = x0.shape
    iu0 = np.triu_indices(n, 1)
    y0 = x0.ravel().copy()

    def fun(y):
        Y = y.reshape(n, d)
        norms = np.sqrt(np.sum(Y * Y, axis=1, keepdims=True)) + 1e-16
        Xn = Y / norms
        G = Xn @ Xn.T
        dots = G[iu0]
        m = float(dots.max())
        e = np.exp(k * (dots - m))
        S = float(e.sum()) + 1e-300
        L = m + np.log(S) / k
        w = e / S
        W = np.zeros((n, n), dtype=float)
        W[iu0] = w
        W = W + W.T
        GX = W @ Xn
        dot = np.sum(GX * Xn, axis=1, keepdims=True)
        Gt = (GX - dot * Xn) / norms
        return L, Gt.ravel()

    try:
        res = minimize(fun, y0, method='L-BFGS-B', jac=True,
                       options={'maxiter': maxiter, 'ftol': 1e-12, 'gtol': 1e-7})
        Y = res.x.reshape(n, d)
        Y = Y / np.linalg.norm(Y, axis=1, keepdims=True)
        return Y
    except Exception:
        return x0


def fast_exp_relax(x, iters, k0, k1, lr0=0.15, lr1=0.02, mu=0.85, rng=None, noise=0.0):
    n, d = x.shape
    v = np.zeros_like(x)
    for t in range(iters):
        frac = t / max(1, iters - 1)
        k = k0 * (k1 / k0) ** frac if k0 != k1 else k0
        lr = lr0 * (lr1 / lr0) ** frac
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        moff = float(G.max())
        W = np.exp(k * (G - moff))
        np.fill_diagonal(W, 0.0)
        rs = W.sum(axis=1, keepdims=True) + 1e-12
        grad = (W @ x) / rs
        dot = np.sum(grad * x, axis=1, keepdims=True)
        gt = grad - dot * x
        # normalize gt scale
        # clip
        v = mu * v + gt
        denom = 1.0 - mu ** (t + 1) if mu > 0 else 1.0
        vc = v / denom
        x = x - lr * 0.05 * vc
        if noise > 0 and frac < 0.6 and rng is not None:
            amp = noise * (1.0 - frac / 0.6)
            pert = rng.standard_normal(x.shape)
            pd = np.sum(pert * x, axis=1, keepdims=True)
            pert -= pd * x
            x = x + amp * pert
        x = x / np.linalg.norm(x, axis=1, keepdims=True)
    return x


def void_reinsert(x, idx, rng, nc=200):
    n, d = x.shape
    keep_mask = np.ones(n, dtype=bool)
    keep_mask[np.asarray(idx)] = False
    Xkeep = x[keep_mask]
    x_new = x.copy()
    order = np.asarray(idx)
    rng.shuffle(order)
    for j in order:
        cand = rng.standard_normal((nc, d))
        cand /= np.linalg.norm(cand, axis=1, keepdims=True)
        dots = cand @ Xkeep.T
        worst = dots.max(axis=1)
        bi = int(np.argmin(worst))
        chosen = cand[bi].copy()
        # local refine chosen against Xkeep (fixed) : minimize LSE
        c = chosen.copy()
        kk = 80.0
        step = 0.1
        for _ in range(60):
            g = Xkeep @ c  # (m,)
            mm = g.max()
            w = np.exp(kk * (g - mm))
            w /= (w.sum() + 1e-16)
            grad = w @ Xkeep  # (d,)
            grad -= (grad @ c) * c
            c = c - step * grad
            c /= (np.linalg.norm(c) + 1e-16)
            kk *= 1.02
            step *= 0.99
        x_new[j] = c
        Xkeep = np.vstack([Xkeep, c[None, :]])
    return x_new


def perturb_current(cur, rng, cur_val=None):
    n, d = cur.shape
    r = rng.random()
    x = cur.copy()
    if r < 0.30:
        sig = float(rng.choice(np.array([0.02, 0.05, 0.09, 0.14, 0.22])))
        x = x + sig * rng.standard_normal(x.shape)
        x /= np.linalg.norm(x, axis=1, keepdims=True)
        return x
    elif r < 0.65:
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        maxper = G.max(axis=1)
        k = int(rng.integers(3, 11))
        idx = np.argsort(-maxper)[:k]
        # sometimes include random low-contact points to move rattlers
        if rng.random() < 0.3:
            # add a random index
            idx[-1] = int(rng.integers(n))
        nc = int(rng.choice(np.array([80, 150, 300])))
        x = void_reinsert(x, idx, rng, nc=nc)
        # small jitter after reinsert
        sig = 0.01
        x = x + sig * rng.standard_normal(x.shape)
        x /= np.linalg.norm(x, axis=1, keepdims=True)
        return x
    elif r < 0.85:
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        maxper = G.max(axis=1)
        # overloaded: top contacts count with threshold
        thr = float(G.max() - 1e-4)
        cnts = (G > thr).sum(axis=1)
        # pick overloaded + rattlers
        k1 = int(rng.integers(2, 6))
        idx_big = np.argsort(-cnts)[:k1]
        # rattlers: zero or small contacts
        idx_small = np.argsort(cnts)[:2]
        idx = np.unique(np.concatenate([idx_big, idx_small]))
        amt = float(rng.uniform(0.1, 0.35))
        for j in idx:
            pert = rng.standard_normal(d)
            pert -= (pert @ x[j]) * x[j]
            pert /= (np.linalg.norm(pert) + 1e-16)
            # random tangent move
            ang = amt * (0.5 + rng.random())
            x[j] = x[j] + ang * pert
        # also global small noise
        x = x + 0.015 * rng.standard_normal(x.shape)
        x /= np.linalg.norm(x, axis=1, keepdims=True)
        return x
    else:
        # cap rotation: pick random direction, take closest m points, rotate them together
        u = rng.standard_normal(d); u /= np.linalg.norm(u)
        proj = x @ u
        m = int(rng.integers(8, 30))
        idx = np.argsort(-proj)[:m]
        # random small rotation in 4D
        A = rng.standard_normal((d, d)) * 0.3
        A = A - A.T
        # exp approx: R = I + A
        R = np.eye(d) + A
        # orthonormalize via QR
        Q, _ = np.linalg.qr(R)
        # interpolate: apply partial rotation
        alpha = float(rng.uniform(0.2, 1.0))
        Rm = np.eye(d) * (1 - alpha) + Q * alpha
        # re-orthonormalize roughly
        Q2, _ = np.linalg.qr(Rm)
        x[idx] = x[idx] @ Q2.T
        x = x + 0.01 * rng.standard_normal(x.shape)
        x /= np.linalg.norm(x, axis=1, keepdims=True)
        return x


def solve():
    t0 = time.time()
    DEADLINE = t0 + 170.0
    SEARCH_END = t0 + 138.0
    rng = np.random.default_rng(12345)
    n, d = 153, 4
    try:
        best0 = np.load("best.npy")
    except Exception:
        try:
            best0 = np.load("record.npy")
        except Exception:
            best0 = rng.standard_normal((n, d))
    best0 = np.asarray(best0, dtype=float)
    if best0.shape != (n, d):
        best0 = rng.standard_normal((n, d))
    best0 = best0 / np.linalg.norm(best0, axis=1, keepdims=True)
    try:
        rec = np.load("record.npy")
        rec = np.asarray(rec, dtype=float)
        if rec.shape == (n, d):
            rec = rec / np.linalg.norm(rec, axis=1, keepdims=True)
            if max_cos(rec) < max_cos(best0):
                best0 = rec.copy()
    except Exception:
        pass

    incumbent = best0.copy()
    inc_val = max_cos(incumbent)
    cur = incumbent.copy()
    cur_val = inc_val

    trial = 0
    # temperature schedule
    while time.time() < SEARCH_END:
        if time.time() > DEADLINE - 30:
            break
        trial += 1
        # choose base: mostly cur, sometimes incumbent/best0
        rb = rng.random()
        if rb < 0.75:
            base = cur
        elif rb < 0.92:
            base = incumbent
        else:
            base = best0
        x = perturb_current(base, rng)

        # smoothing choice
        # alternate fast vs LBFGS depending on trial
        remaining = SEARCH_END - time.time()
        if remaining < 5:
            break
        # fast pre-relax
        if trial % 3 == 0:
            # fast only then evaluate
            k0 = float(rng.choice(np.array([30.0, 50.0, 80.0])))
            k1 = float(rng.choice(np.array([200.0, 350.0, 500.0])))
            it = int(rng.choice(np.array([400, 800, 1500])))
            x = fast_exp_relax(x, iters=it, k0=k0, k1=k1, lr0=0.15, lr1=0.02, mu=0.85, rng=rng, noise=0.0)
            # one LBFGS refinement with high k if promising
            v = max_cos(x)
            if v < cur_val + 0.003:
                x = lbfgs_smooth(x, k=250.0, maxiter=60)
        else:
            # two-stage LBFGS
            k_a = float(rng.choice(np.array([25.0, 40.0, 60.0])))
            k_b = float(rng.choice(np.array([150.0, 250.0, 400.0])))
            it_a = int(rng.choice(np.array([60, 100])))
            it_b = int(rng.choice(np.array([60, 100])))
            # quick fast warmup for large kicks
            if rb < 0.0:
                pass
            x = lbfgs_smooth(x, k=k_a, maxiter=it_a)
            if time.time() > SEARCH_END:
                v = max_cos(x)
            else:
                x = lbfgs_smooth(x, k=k_b, maxiter=it_b)
            v = max_cos(x)

        v = max_cos(x)
        if v < inc_val:
            incumbent = x.copy()
            inc_val = v
            cur = x.copy()
            cur_val = v
        elif v < cur_val:
            cur = x.copy()
            cur_val = v
        else:
            # annealed acceptance to keep wandering
            progress = (time.time() - t0) / max(1e-9, (SEARCH_END - t0))
            T = 0.0015 * (1.0 - progress) + 1e-5
            if rng.random() < np.exp(-(v - cur_val) / T):
                cur = x.copy()
                cur_val = v
            # occasional restart to incumbent if cur drifts too far
            if cur_val > inc_val + 0.004 and rng.random() < 0.3:
                cur = incumbent.copy()
                cur_val = inc_val

        # keep cur from exploding: if too bad, reset
        if cur_val > inc_val + 0.006:
            cur = incumbent.copy()
            cur_val = inc_val

    # final polish of incumbent with remaining time
    remain = DEADLINE - time.time()
    if remain > 5:
        # short fast relax before LP to clean
        try:
            incumbent = fast_exp_relax(incumbent, iters=500, k0=100, k1=500, lr0=0.08, lr1=0.01, mu=0.9, rng=rng, noise=0.0)
        except Exception:
            pass
        remain = DEADLINE - time.time()
        if remain > 4:
            incumbent = lp_polish(incumbent, seconds=remain - 1.0, t_deadline=DEADLINE - 1.0)
    incumbent = incumbent / np.linalg.norm(incumbent, axis=1, keepdims=True)
    # safety: if polished worse than best0 (numerics), return best of them
    try:
        if max_cos(incumbent) > max_cos(best0) + 1e-12:
            # check inc history: incumbent should be best; but fallback
            if inc_val < max_cos(best0):
                pass
            else:
                incumbent = best0.copy()
    except Exception:
        pass
    return incumbent
