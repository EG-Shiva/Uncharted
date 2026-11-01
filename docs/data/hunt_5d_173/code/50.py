import time
import numpy as np
from scipy.optimize import linprog, linear_sum_assignment
from scipy.sparse import coo_matrix

_N = 173
_D = 5

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _void_fill(rem, k, rng, n_cand=5000):
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

def _repel(x, alphas, iters_list, lrs, mu, decay, t_deadline=None):
    v = np.zeros_like(x)
    for alpha, iters, lr0 in zip(alphas, iters_list, lrs):
        lr = float(lr0)
        for s in range(int(iters)):
            if t_deadline is not None and (s & 31) == 0 and time.time() > t_deadline:
                break
            G = x @ x.T
            W = np.exp(alpha * (G - 0.75))
            np.fill_diagonal(W, 0.0)
            rs = W.sum(axis=1, keepdims=True) + 1e-300
            A = (W @ x) / rs
            dot = (A * x).sum(axis=1, keepdims=True)
            Ft = A - dot * x
            v = mu * v + (1.0 - mu) * Ft
            x = x - lr * v
            x /= np.linalg.norm(x, axis=1, keepdims=True)
            lr *= decay
        if t_deadline is not None and time.time() > t_deadline:
            break
    return x

def _squeeze(x, iters=80, lr=0.003, window=0.008):
    n = x.shape[0]
    for it in range(int(iters)):
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        # active pairs near max
        I, J = np.where(G > m - window)
        mask = I < J
        I = I[mask]; J = J[mask]
        if len(I) == 0:
            break
        F = np.zeros_like(x)
        # push apart: for pair (i,j), grad on i is (xj - Gij xi)
        # move opposite: -lr * grad
        for a, b in zip(I, J):
            g = G[a, b] if False else float(x[a] @ x[b])
            F[a] += (x[b] - g * x[a])
            F[b] += (x[a] - g * x[b])
        # project already tangential by construction, step
        step = lr / (1.0 + len(I) / 200.0)
        x = x - step * F
        x /= np.linalg.norm(x, axis=1, keepdims=True)
        # shrink window / lr
        window *= 0.99
        lr *= 0.995
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

def _recombine(A, B, rng):
    # orthogonal alignment B -> A
    M = B.T @ A
    try:
        U, _, Vt = np.linalg.svd(M)
        Q = U @ Vt
    except Exception:
        Q = np.eye(A.shape[1])
    Br = B @ Q
    C = -(A @ Br.T)
    try:
        r, c = linear_sum_assignment(C)
        Bp = np.empty_like(Br)
        Bp[r] = Br[c]
    except Exception:
        Bp = Br
    t = 0.3 + 0.4 * rng.rand()
    X = (1.0 - t) * A + t * Bp
    # small noise to break symmetry
    X = X + 0.01 * rng.standard_normal(X.shape)
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    return X

def solve():
    t0 = time.time()
    deadline = t0 + 168.0
    deadline_search = deadline - 28.0
    rng = np.random.RandomState(1)
    n, d = 173, 5

    cands = []
    try:
        xb = np.load("best.npy").astype(float)
        if xb.shape == (n, d):
            xb /= np.linalg.norm(xb, axis=1, keepdims=True)
            cands.append(xb)
    except Exception:
        pass
    try:
        xr = np.load("record.npy").astype(float)
        if xr.shape == (n, d):
            xr /= np.linalg.norm(xr, axis=1, keepdims=True)
            cands.append(xr)
    except Exception:
        pass
    tmp = rng.standard_normal((n, d))
    tmp /= np.linalg.norm(tmp, axis=1, keepdims=True)
    cands.append(tmp)

    pop = []
    for c in cands:
        try:
            v = _max_cos(c)
            pop.append((v, c.copy()))
        except Exception:
            pass
    pop = sorted(pop, key=lambda z: z[0])
    # quick repel on random to seed diversity (short)
    try:
        r0 = pop[-1][1].copy()
        r0 = _repel(r0, [40.0, 120.0], [120, 150], [0.08, 0.03], 0.6, 0.998, deadline_search)
        pop.append((_max_cos(r0), r0))
    except Exception:
        pass
    pop = sorted(pop, key=lambda z: z[0])[:8]

    best_val, best = pop[0][0], pop[0][1].copy()

    def get_base():
        if rng.rand() < 0.5:
            return best
        else:
            idx = rng.randint(len(pop))
            return pop[idx][1]

    trial = 0
    while time.time() < deadline_search:
        trial += 1
        base = get_base()
        x_try = base.copy()
        op = rng.rand()
        try:
            if op < 0.30:
                # small worst-point void reinsertion
                k = int(rng.randint(2, 7))
                G = x_try @ x_try.T
                np.fill_diagonal(G, -2.0)
                mi = G.max(axis=1)
                order = np.argsort(-mi)
                pool = order[:25]
                sel = rng.choice(pool, size=k, replace=False)
                mask = np.ones(n, dtype=bool)
                mask[sel] = False
                rem = x_try[mask]
                new_pts = _void_fill(rem, k, rng, n_cand=4000)
                x_try[sel] = new_pts
                sigma = 0.005 + 0.015 * rng.rand()
                x_try = x_try + sigma * rng.standard_normal(x_try.shape)
                x_try /= np.linalg.norm(x_try, axis=1, keepdims=True)
            elif op < 0.50:
                # spherical cap deletion
                u = rng.standard_normal(d)
                u /= np.linalg.norm(u)
                proj = x_try @ u
                k = int(rng.randint(12, 29))
                sel = np.argsort(-proj)[:k]
                mask = np.ones(n, dtype=bool)
                mask[sel] = False
                rem = x_try[mask]
                new_pts = _void_fill(rem, k, rng, n_cand=4000)
                x_try[sel] = new_pts
                sigma = 0.01 + 0.02 * rng.rand()
                x_try = x_try + sigma * rng.standard_normal(x_try.shape)
                x_try /= np.linalg.norm(x_try, axis=1, keepdims=True)
            elif op < 0.70:
                # overload-targeted removal: jammed hub + neighbors
                G = x_try @ x_try.T
                np.fill_diagonal(G, -2.0)
                mi = G.max(axis=1)
                hub = int(np.argmax(mi))
                neigh = np.argsort(-G[hub])[:15]
                k = int(rng.randint(5, 13))
                # include hub + (k-1) of top neighbors sampled
                pool = np.concatenate([[hub], neigh])
                # unique
                _, ui = np.unique(pool, return_index=True)
                pool = pool[np.sort(ui)]
                if len(pool) < k:
                    order = np.argsort(-mi)
                    extra = [o for o in order if o not in set(pool.tolist())]
                    pool = np.concatenate([pool, extra[:k - len(pool)]])
                sel = rng.choice(pool, size=min(k, len(pool)), replace=False)
                # pad if needed
                if len(sel) < k:
                    remaining = np.setdiff1d(np.arange(n), sel)
                    add = rng.choice(remaining, size=k - len(sel), replace=False)
                    sel = np.concatenate([sel, add])
                mask = np.ones(n, dtype=bool)
                mask[sel] = False
                rem = x_try[mask]
                new_pts = _void_fill(rem, k, rng, n_cand=4000)
                x_try[sel] = new_pts
                sigma = 0.005 + 0.02 * rng.rand()
                x_try = x_try + sigma * rng.standard_normal(x_try.shape)
                x_try /= np.linalg.norm(x_try, axis=1, keepdims=True)
            elif op < 0.80:
                # large global shake
                sigma = 0.20 + 0.30 * rng.rand()
                if rng.rand() < 0.2:
                    sigma = 0.45 + 0.2 * rng.rand()
                x_try = x_try + sigma * rng.standard_normal(x_try.shape)
                x_try /= np.linalg.norm(x_try, axis=1, keepdims=True)
            elif op < 0.90:
                # recombination
                if len(pop) >= 2:
                    a_idx = rng.randint(len(pop))
                    b_idx = rng.randint(len(pop))
                    if a_idx == b_idx:
                        b_idx = (b_idx + 1) % len(pop)
                    A = pop[a_idx][1]
                    B = pop[b_idx][1]
                    x_try = _recombine(A, B, rng)
                else:
                    sigma = 0.1 + 0.2 * rng.rand()
                    x_try = x_try + sigma * rng.standard_normal(x_try.shape)
                    x_try /= np.linalg.norm(x_try, axis=1, keepdims=True)
            else:
                # drop-worst, optimize 172 alone, then regrow
                G = x_try @ x_try.T
                np.fill_diagonal(G, -2.0)
                mi = G.max(axis=1)
                worst = int(np.argmax(mi))
                mask = np.ones(n, dtype=bool)
                mask[worst] = False
                rem = x_try[mask].copy()
                rem = rem + 0.01 * rng.standard_normal(rem.shape)
                rem /= np.linalg.norm(rem, axis=1, keepdims=True)
                # short repel on 172
                a1 = 40.0 + rng.rand() * 20.0
                a2 = 120.0 + rng.rand() * 60.0
                rem = _repel(rem, [a1, a2], [120, 150], [0.08, 0.03], 0.6, 0.998, deadline_search)
                # void insert 1
                new_pt = _void_fill(rem, 1, rng, n_cand=5000)
                x_try = np.vstack([rem, new_pt])
                # shuffle rows to avoid positional bias
                perm = rng.permutation(n)
                x_try = x_try[perm]
        except Exception:
            continue

        # annealed repulsion params
        a1 = 30.0 + rng.rand() * 20.0
        a2 = 80.0 + rng.rand() * 60.0
        a3 = 180.0 + rng.rand() * 140.0
        remain = deadline_search - time.time()
        if remain < 12:
            iters_list = [100, 120, 150]
        else:
            iters_list = [180, 220, 300]
        l1 = 0.06 + rng.rand() * 0.10
        l2 = 0.03 + rng.rand() * 0.05
        l3 = 0.015 + rng.rand() * 0.025
        mu = 0.5 + rng.rand() * 0.35
        decay = 0.997 + rng.rand() * 0.002
        if decay > 0.9995:
            decay = 0.9995
        try:
            x_opt = _repel(x_try, [a1, a2, a3], iters_list, [l1, l2, l3], mu, decay, deadline_search)
            # cheap max-pair squeeze
            x_opt = _squeeze(x_opt, iters=60, lr=0.004, window=0.006)
        except Exception:
            continue
        try:
            v = _max_cos(x_opt)
        except Exception:
            continue
        if not np.isfinite(v):
            continue
        if v < best_val - 1e-12:
            best_val = v
            best = x_opt.copy()
            # insert into pop
            pop.append((v, x_opt.copy()))
            pop = sorted(pop, key=lambda z: z[0])[:8]
            if best_val < 0.74:
                break
        else:
            # population diversity: keep if good enough and not duplicate
            if len(pop) < 8:
                if v < 0.85:
                    pop.append((v, x_opt.copy()))
                    pop = sorted(pop, key=lambda z: z[0])[:8]
            else:
                worst_v = pop[-1][0]
                if v < worst_v - 1e-7 and v < 0.83:
                    # avoid exact duplicate of best basin
                    pop.append((v, x_opt.copy()))
                    pop = sorted(pop, key=lambda z: z[0])[:8]
        # occasional short LP to reveal true value of promising trial
        if v < best_val + 5e-4 and (trial % 25) == 0:
            if time.time() < deadline_search - 5:
                try:
                    cand = _lp_polish(x_opt, 1.0)
                    vv = _max_cos(cand)
                    if vv < best_val - 1e-12:
                        best_val = vv
                        best = cand.copy()
                        pop.append((vv, cand.copy()))
                        pop = sorted(pop, key=lambda z: z[0])[:8]
                except Exception:
                    pass

    remain = deadline - time.time() - 2.0
    if remain < 5.0:
        remain = 5.0
    if remain > 32.0:
        remain = 32.0
    try:
        best = _lp_polish(best, remain)
    except Exception:
        pass
    best = best / np.linalg.norm(best, axis=1, keepdims=True)
    return best
