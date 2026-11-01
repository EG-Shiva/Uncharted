import time
import numpy as np
from scipy.optimize import linprog, linear_sum_assignment
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _row_max(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return g.max(axis=1), float(g.max())

def _void_fill(rem, k, rng, n_cand=3000):
    d = rem.shape[1]
    cand = rng.standard_normal((n_cand, d))
    cand /= np.linalg.norm(cand, axis=1, keepdims=True)
    s_max = (cand @ rem.T).max(axis=1)
    chosen = []
    for _ in range(k):
        idx = int(np.argmin(s_max))
        pt = cand[idx].copy()
        chosen.append(pt)
        upd = cand @ pt
        np.maximum(s_max, upd, out=s_max)
        s_max[idx] = 2.0
    return np.stack(chosen, axis=0)

def _repel_bal(x, rng, iters, a0, a1, lr0, lr1, mu, deadline=None, noise=0.0):
    v = np.zeros_like(x)
    for s in range(int(iters)):
        if deadline is not None and (s & 31) == 0 and time.time() > deadline:
            break
        frac = s / max(1, int(iters) - 1)
        alpha = a0 * (a1 / max(a0, 1e-9)) ** frac
        lr = lr0 * (lr1 / max(lr0, 1e-12)) ** frac
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        W = np.exp(alpha * (G - m))
        np.fill_diagonal(W, 0.0)
        rs = W.sum(axis=1, keepdims=True) + 1e-300
        A = (W @ x) / rs
        dot = (A * x).sum(axis=1, keepdims=True)
        Ft = A - dot * x
        # load balancing: overloaded rows move more
        mi = G.max(axis=1)
        fi = np.exp(0.5 * alpha * (mi - m))  # (n,) 0..1
        scale = (1.0 + 2.0 * fi)[:, None]
        Ft = Ft * scale
        v = mu * v + (1.0 - mu) * Ft
        x = x - lr * v
        if noise > 0:
            x = x + noise * lr * rng.standard_normal(x.shape)
        x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x

def _lp_polish(x, seconds):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        c = (x @ x.T)[iu]
        m = float(c.max())
        sel = c > m - max(20 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        nv = n * d + 1
        if p == 0:
            break
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv)
        obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c[sel], A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            break
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        vv = _max_cos(y)
        if vv < best:
            x, best, radius = y, vv, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def _icp_child(A, B, rng, sigma=0.012):
    n, d = A.shape
    Bc = B.copy()
    try:
        for _ in range(3):
            D = A @ Bc.T
            _, col = linear_sum_assignment(-D)
            # col maps A-row i -> Bc-row col[i]; build permuted Bc
            Bp = Bc[col]
            H = Bp.T @ A  # d x d
            U, _, Vt = np.linalg.svd(H, full_matrices=False)
            R = Vt.T @ U.T
            if np.linalg.det(R) < 0:
                Vt2 = Vt.copy()
                Vt2[-1, :] *= -1
                R = Vt2.T @ U.T
            Bc = (R @ Bc.T).T
        D = A @ Bc.T
        _, col = linear_sum_assignment(-D)
        Bp = Bc[col]
        mask = rng.rand(n) < 0.5
        C = np.where(mask[:, None], A, Bp)
        C = C + sigma * rng.standard_normal(C.shape)
        C /= np.linalg.norm(C, axis=1, keepdims=True)
        return C
    except Exception:
        # fallback: random mix
        idx = rng.permutation(n)
        C = np.where(rng.rand(n)[:, None] < 0.5, A, B[idx])
        C = C + sigma * rng.standard_normal(C.shape)
        C /= np.linalg.norm(C, axis=1, keepdims=True)
        return C

def solve():
    t0 = time.time()
    deadline = t0 + 168.0
    deadline_search = deadline - 30.0
    rng = np.random.RandomState(12345)
    n, d = 173, 5

    try:
        xb = np.load("best.npy").astype(float)
    except Exception:
        xb = None
    try:
        xr = np.load("record.npy").astype(float)
    except Exception:
        xr = None
    inits = []
    for c in (xb, xr):
        if c is not None and c.shape == (n, d):
            c = c / np.linalg.norm(c, axis=1, keepdims=True)
            inits.append(c)
    if len(inits) == 0:
        tmp = rng.standard_normal((n, d))
        tmp /= np.linalg.norm(tmp, axis=1, keepdims=True)
        inits.append(tmp)

    base = inits[0].copy()
    base_val = _max_cos(base)
    for c in inits[1:]:
        v = _max_cos(c)
        if v < base_val:
            base, base_val = c.copy(), v

    # population
    P = 7
    pop_x = []
    pop_v = []
    pop_x.append(base.copy()); pop_v.append(base_val)
    # diversified members: perturb + short repel
    for i in range(1, P):
        x = base.copy()
        sig = 0.05 + 0.15 * rng.rand()
        if i % 3 == 0:
            # cap removal init
            u = rng.standard_normal(d); u /= np.linalg.norm(u)
            proj = x @ u
            k = int(10 + rng.randint(0, 12))
            sel = np.argsort(-proj)[:k]
            mask = np.ones(n, dtype=bool); mask[sel] = False
            rem = x[mask]
            try:
                x[sel] = _void_fill(rem, k, rng, n_cand=2000)
            except Exception:
                pass
        x = x + sig * rng.standard_normal(x.shape)
        x /= np.linalg.norm(x, axis=1, keepdims=True)
        x = _repel_bal(x, rng, iters=220, a0=25.0, a1=150.0, lr0=0.10, lr1=0.02, mu=0.65, deadline=deadline_search)
        v = _max_cos(x)
        pop_x.append(x); pop_v.append(v)
    pop_v = np.array(pop_v, dtype=float)
    order = np.argsort(pop_v)
    pop_x = [pop_x[i] for i in order]; pop_v = pop_v[order]
    best = pop_x[0].copy(); best_val = float(pop_v[0])

    # annealing temps
    T0, T1 = 0.0025, 0.00003
    it = 0
    # estimate total iterations
    while time.time() < deadline_search:
        it += 1
        frac = (time.time() - t0) / max(1e-9, (deadline_search - t0))
        frac = min(max(frac, 0.0), 1.0)
        Temp = T0 * (T1 / T0) ** frac
        r = rng.rand()
        # select parent(s)
        # rank-biased selection
        if rng.rand() < 0.7:
            pidx = int(np.argmin(pop_v + 0.0005 * rng.rand(P)))
            # actually prefer best half
            if rng.rand() < 0.6:
                pidx = int(np.argsort(pop_v)[0]) if rng.rand() < 0.5 else int(rng.choice(np.argsort(pop_v)[:3]))
            parent = pop_x[pidx]
        else:
            pidx = int(rng.randint(0, P))
            parent = pop_x[pidx]
        x_try = parent.copy()
        op = r
        if op < 0.38:
            # worst-point void reinsertion (balanced load attack)
            k = int(rng.randint(2, 9))
            G = x_try @ x_try.T
            np.fill_diagonal(G, -2.0)
            mi = G.max(axis=1)
            # pool of overloaded
            order_pts = np.argsort(-mi)
            poolsize = 30
            pool = order_pts[:poolsize]
            # also sometimes include random to free rattler neighbourhood
            if rng.rand() < 0.25:
                # include low-contact points to force movement elsewhere? pick overloaded still
                pass
            sel = rng.choice(pool, size=k, replace=False)
            mask = np.ones(n, dtype=bool); mask[sel] = False
            rem = x_try[mask]
            try:
                new_pts = _void_fill(rem, k, rng, n_cand=3000)
            except Exception:
                new_pts = rng.standard_normal((k, d))
                new_pts /= np.linalg.norm(new_pts, axis=1, keepdims=True)
            x_try[sel] = new_pts
            sigma = 0.004 + 0.014 * rng.rand()
            x_try = x_try + sigma * rng.standard_normal(x_try.shape)
            x_try /= np.linalg.norm(x_try, axis=1, keepdims=True)
        elif op < 0.58:
            # spherical cap removal (topology change)
            u = rng.standard_normal(d); u /= np.linalg.norm(u)
            proj = x_try @ u
            k = int(rng.randint(8, 24))
            sel = np.argsort(-proj)[:k]
            mask = np.ones(n, dtype=bool); mask[sel] = False
            rem = x_try[mask]
            try:
                new_pts = _void_fill(rem, k, rng, n_cand=3000)
            except Exception:
                new_pts = rng.standard_normal((k, d))
                new_pts /= np.linalg.norm(new_pts, axis=1, keepdims=True)
            x_try[sel] = new_pts
            sigma = 0.008 + 0.02 * rng.rand()
            x_try = x_try + sigma * rng.standard_normal(x_try.shape)
            x_try /= np.linalg.norm(x_try, axis=1, keepdims=True)
        elif op < 0.73:
            # global kick
            sigma = 0.05 + 0.22 * rng.rand()
            if rng.rand() < 0.12:
                sigma = 0.35 + 0.20 * rng.rand()
            x_try = x_try + sigma * rng.standard_normal(x_try.shape)
            x_try /= np.linalg.norm(x_try, axis=1, keepdims=True)
        else:
            # crossover
            a_idx, b_idx = rng.choice(P, size=2, replace=False)
            A = pop_x[a_idx]; B = pop_x[b_idx]
            sig = 0.006 + 0.014 * rng.rand()
            x_try = _icp_child(A, B, rng, sigma=sig)

        # repel schedule randomized
        a0 = 22.0 + rng.rand() * 22.0
        a1 = 150.0 + rng.rand() * 180.0
        iters = int(rng.choice([350, 450, 550, 650]))
        remain = deadline_search - time.time()
        if remain < 12:
            iters = 220
        lr0 = 0.06 + rng.rand() * 0.10
        lr1 = 0.008 + rng.rand() * 0.018
        mu = 0.5 + rng.rand() * 0.38
        nz = 0.0
        if rng.rand() < 0.2:
            nz = 0.0005
        x_opt = _repel_bal(x_try, rng, iters=iters, a0=a0, a1=a1, lr0=lr0, lr1=lr1, mu=mu, deadline=deadline_search, noise=nz)
        v = _max_cos(x_opt)
        if v < best_val - 1e-12:
            best = x_opt.copy(); best_val = v
        # population update (non-greedy)
        worst_idx = int(np.argmax(pop_v))
        worst_val = float(pop_v[worst_idx])
        if v < worst_val:
            pop_x[worst_idx] = x_opt; pop_v[worst_idx] = v
        else:
            # metropolis accept worse to keep diversity
            if rng.rand() < np.exp(-(v - worst_val) / max(Temp, 1e-9)):
                # replace random non-best
                ridx = int(rng.randint(0, P))
                if ridx != int(np.argmin(pop_v)):
                    pop_x[ridx] = x_opt; pop_v[ridx] = v
        if best_val < 0.74:
            break

    # final: polish top candidates briefly then best long
    try:
        order2 = np.argsort(pop_v)
        cand_best = pop_x[order2[0]]
        # if global best differs keep it
        if best_val < float(pop_v[order2[0]]) - 1e-15:
            cand_best = best.copy()
        remain = deadline - time.time() - 2.0
        if remain < 5.0:
            remain = 5.0
        if remain > 34.0:
            remain = 34.0
        # quick polish second best to try alternative basin (use 1/4 time)
        try:
            if P >= 3 and remain > 15:
                second = pop_x[order2[1]].copy()
                second = _lp_polish(second, min(6.0, remain * 0.25))
                sv = _max_cos(second)
                if sv < best_val:
                    best, best_val, cand_best = second, sv, second.copy()
        except Exception:
            pass
        try:
            cand_best = _lp_polish(cand_best, remain)
        except Exception:
            pass
        best = cand_best
    except Exception:
        pass
    best = best / np.linalg.norm(best, axis=1, keepdims=True)
    return best
