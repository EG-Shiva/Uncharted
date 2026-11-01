import time
import numpy as np
from scipy.optimize import linprog, linear_sum_assignment
from scipy.sparse import coo_matrix

N = 173
D = 5

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _lp_polish(x, seconds):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    if seconds <= 0.05:
        return x
    radius = 1e-3
    start = time.time()
    iu0, iu1 = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        G = x @ x.T
        cfull = G[iu0, iu1]
        m = float(cfull.max())
        thresh = m - max(20.0 * radius, 1e-9)
        sel = cfull > thresh
        if int(sel.sum()) > 4000:
            idx = np.where(sel)[0]
            order = np.argsort(cfull[idx])[-4000:]
            mask = np.zeros_like(sel, dtype=bool)
            mask[idx[order]] = True
            sel = mask
        I = iu0[sel]; J = iu1[sel]
        c = cfull[sel]
        p = len(I)
        if p == 0:
            radius *= 0.5
            continue
        nv = n * d + 1
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], axis=1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], axis=1).ravel()
        rows = np.repeat(np.arange(p), 2 * d + 1)
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv); obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-c, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        yn = np.linalg.norm(y, axis=1, keepdims=True)
        yn[yn == 0] = 1.0
        y = y / yn
        if not np.all(np.isfinite(y)):
            radius *= 0.5
            continue
        v = _max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def _repel(x, alphas, iters_list, lrs, mu, decay, deadline=float('inf')):
    v = np.zeros_like(x)
    for alpha, iters, lr0 in zip(alphas, iters_list, lrs):
        lr = float(lr0)
        for it in range(int(iters)):
            if it % 50 == 0 and time.time() > deadline:
                return x
            G = x @ x.T
            # stable weights: exp(alpha*(G-1)) in (0,1]
            E = alpha * (G - 1.0)
            # E<=0 so safe
            W = np.exp(E)
            np.fill_diagonal(W, 0.0)
            rs = W.sum(axis=1, keepdims=True) + 1e-300
            A = (W @ x) / rs
            dot = (A * x).sum(axis=1, keepdims=True)
            Ft = A - dot * x
            v = mu * v + (1.0 - mu) * Ft
            x = x - lr * v
            xn = np.linalg.norm(x, axis=1, keepdims=True)
            xn[xn == 0] = 1.0
            x = x / xn
            lr *= decay
    return x

def _void_fill(rem, k, rng, n_cand=4000):
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

def _quick_repel(x_try, rng, deadline):
    a1 = 25.0 + rng.random() * 25.0
    a2 = 70.0 + rng.random() * 60.0
    a3 = 160.0 + rng.random() * 160.0
    it1 = 120 + int(rng.integers(0, 80))
    it2 = 150 + int(rng.integers(0, 100))
    it3 = 180 + int(rng.integers(0, 120))
    l1 = 0.05 + rng.random() * 0.10
    l2 = 0.025 + rng.random() * 0.05
    l3 = 0.012 + rng.random() * 0.028
    mu = 0.5 + rng.random() * 0.35
    decay = 0.996 + rng.random() * 0.003
    if decay > 0.9995:
        decay = 0.9995
    # occasionally sharper schedule
    if rng.random() < 0.2:
        a3 = 300.0 + rng.random() * 150.0
        it3 = 250
        l3 = 0.008 + rng.random() * 0.015
    return _repel(x_try, [a1, a2, a3], [it1, it2, it3], [l1, l2, l3], mu, decay, deadline)

def _procrustes_align(B, A, rng):
    # align B to A: 2 rounds of assignment + rotation
    n, d = A.shape
    Brot = B.copy()
    for _ in range(2):
        C = Brot @ A.T  # cosine
        # maximize total cosine => minimize -C
        try:
            r, c = linear_sum_assignment(-C)
        except Exception:
            break
        Bp = Brot[r]
        Ap = A[c]
        # optimal rotation R: min ||Ap - Bp R|| => R = U Vt from SVD(Bp.T Ap)
        M = Bp.T @ Ap
        try:
            U, _, Vt = np.linalg.svd(M)
            R = U @ Vt
        except Exception:
            break
        Brot = Brot @ R
    # final assignment to order points like A
    try:
        C = Brot @ A.T
        r, c = linear_sum_assignment(-C)
        # reorder Brot so that Brot_ordered[i] corresponds to A[i]
        inv = np.empty(n, dtype=int)
        inv[c] = r
        Brot_ordered = Brot[inv]
        return Brot_ordered
    except Exception:
        return Brot

def solve():
    t0 = time.time()
    BUDGET = 172.0
    deadline = t0 + BUDGET
    rng = np.random.default_rng(12345)
    n, d = N, D
    best = None
    best_val = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn).astype(np.float64)
            if a.shape == (n, d):
                an = a / np.linalg.norm(a, axis=1, keepdims=True)
                if np.all(np.isfinite(an)):
                    v = _max_cos(an)
                    if v < best_val:
                        best_val = v
                        best = an.copy()
        except Exception:
            pass
    if best is None:
        best = rng.standard_normal((n, d))
        best /= np.linalg.norm(best, axis=1, keepdims=True)
        best_val = _max_cos(best)
    # small population for recombination diversity
    pop = [best.copy()]
    pop_vals = [best_val]
    # add a couple of quickly repelled random codes to population
    for _ in range(2):
        if time.time() > deadline - 30:
            break
        y = rng.standard_normal((n, d))
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        y = _quick_repel(y, rng, deadline)
        y = _lp_polish(y, 0.8)
        v = _max_cos(y)
        pop.append(y.copy()); pop_vals.append(v)
        if v < best_val:
            best, best_val = y.copy(), v
    # sort pop
    order = np.argsort(pop_vals)
    pop = [pop[i] for i in order]
    pop_vals = [pop_vals[i] for i in order]
    # keep at most 5
    pop = pop[:5]; pop_vals = pop_vals[:5]

    trial = 0
    while True:
        rem = deadline - time.time()
        if rem < 8.0:
            break
        pt = 1.2
        if rem < 25:
            pt = max(0.6, rem - 7.0)
        pt = min(pt, rem - 6.5)
        if pt < 0.5:
            break
        trial += 1
        mode = trial % 7
        # pick base parent (usually best, sometimes 2nd best for diversity)
        if len(pop) > 1 and rng.random() < 0.25:
            base_idx = int(rng.integers(1, len(pop)))
            base = pop[base_idx]
        else:
            base = best
        y = base.copy()

        if mode == 0:
            # void re-insertion of worst + rattler
            G = y @ y.T
            np.fill_diagonal(G, -2.0)
            m = float(G.max())
            cnt = (G >= m - 1e-9).sum(axis=1)
            per = G.max(axis=1)
            order_desc = np.argsort(-per)
            rattlers = np.where(cnt == 0)[0]
            k = int(rng.integers(3, 9))  # 3..8
            pool = order_desc[:35]
            k1 = min(k, len(pool))
            sel = list(rng.choice(pool, size=k1, replace=False))
            if len(rattlers) > 0 and rng.random() < 0.9:
                # force rattler inclusion
                r0 = int(rattlers[int(rng.integers(len(rattlers)))] )
                if r0 not in sel:
                    sel[0] = r0
            # also force most overloaded point
            if int(order_desc[0]) not in sel:
                if rng.random() < 0.7:
                    sel[-1] = int(order_desc[0])
            sel = np.array(sorted(set(sel)), dtype=int)
            k = len(sel)
            mask = np.ones(n, dtype=bool); mask[sel] = False
            rem_pts = y[mask]
            try:
                new_pts = _void_fill(rem_pts, k, rng, n_cand=4000)
            except Exception:
                new_pts = rng.standard_normal((k, d))
                new_pts /= np.linalg.norm(new_pts, axis=1, keepdims=True)
            y[sel] = new_pts
            sigma = 0.004 + 0.014 * rng.random()
            y = y + sigma * rng.standard_normal(y.shape)
            y /= np.linalg.norm(y, axis=1, keepdims=True)
        elif mode == 1:
            # large global shake of best
            sigma = 0.18 + 0.25 * rng.random()
            if rng.random() < 0.2:
                sigma = 0.45
            y = y + sigma * rng.standard_normal(y.shape)
            y /= np.linalg.norm(y, axis=1, keepdims=True)
        elif mode == 2:
            # fresh random
            y = rng.standard_normal((n, d))
            y /= np.linalg.norm(y, axis=1, keepdims=True)
        elif mode == 3:
            # cap replacement
            u = rng.standard_normal(d); u /= np.linalg.norm(u)
            proj = y @ u
            cap = int(rng.integers(6, 16))
            victims = np.argsort(-proj)[:cap]
            mask = np.ones(n, dtype=bool); mask[victims] = False
            rem_pts = y[mask]
            try:
                new_pts = _void_fill(rem_pts, cap, rng, n_cand=4000)
            except Exception:
                new_pts = rng.standard_normal((cap, d))
                new_pts /= np.linalg.norm(new_pts, axis=1, keepdims=True)
            y[victims] = new_pts
            sigma = 0.01 + 0.03 * rng.random()
            y = y + sigma * rng.standard_normal(y.shape)
            y /= np.linalg.norm(y, axis=1, keepdims=True)
        elif mode == 4:
            # small shake of jammed neighbourhood + rattler kick
            G = y @ y.T
            np.fill_diagonal(G, -2.0)
            m = float(G.max())
            cnt = (G >= m - 1e-9).sum(axis=1)
            order_desc = np.argsort(-cnt)
            jam = order_desc[:4]
            sigma_j = 0.06 + 0.08 * rng.random()
            y[jam] += sigma_j * rng.standard_normal((len(jam), d))
            ratt = np.where(cnt == 0)[0]
            if len(ratt) > 0:
                # relocate rattler to largest void
                mask = np.ones(n, dtype=bool); mask[ratt[0]] = False
                rem_pts = y[mask]
                try:
                    nv = _void_fill(rem_pts, 1, rng, n_cand=4000)
                    y[ratt[0]] = nv[0]
                except Exception:
                    y[ratt[0]] = rng.standard_normal(d)
            else:
                y += 0.012 * rng.standard_normal(y.shape)
            y /= np.linalg.norm(y, axis=1, keepdims=True)
        elif mode == 5:
            # recombination of two population members
            if len(pop) >= 2:
                i1, i2 = rng.choice(len(pop), size=2, replace=False)
                A = pop[i1]; B = pop[i2]
                try:
                    Bal = _procrustes_align(B, A, rng)
                except Exception:
                    Bal = B
                # child: random mask mix
                frac = 0.3 + 0.4 * rng.random()
                take = rng.random(n) < frac
                if take.sum() == 0 or take.sum() == n:
                    take[:n//2] = True; take[n//2:] = False
                y = A.copy()
                y[take] = Bal[take]
                # void-repair worst 2-4 after crossover
                G = y @ y.T
                np.fill_diagonal(G, -2.0)
                per = G.max(axis=1)
                w = np.argsort(-per)[:int(rng.integers(2, 5))]
                mask = np.ones(n, dtype=bool); mask[w] = False
                try:
                    nv = _void_fill(y[mask], len(w), rng, n_cand=3000)
                    y[w] = nv
                except Exception:
                    pass
                y = y + 0.008 * rng.standard_normal(y.shape)
                y /= np.linalg.norm(y, axis=1, keepdims=True)
            else:
                y = y + 0.05 * rng.standard_normal(y.shape)
                y /= np.linalg.norm(y, axis=1, keepdims=True)
        else:
            # medium shake + contact-neighbour jitter
            sigma = 0.04 + 0.07 * rng.random()
            y = y + sigma * rng.standard_normal(y.shape)
            y /= np.linalg.norm(y, axis=1, keepdims=True)

        # repel then LP polish
        inner_deadline = deadline - 6.0
        try:
            y = _quick_repel(y, rng, inner_deadline)
        except Exception:
            pass
        try:
            y = _lp_polish(y, pt)
        except Exception:
            continue
        if not np.all(np.isfinite(y)):
            continue
        v = _max_cos(y)
        # maintain population of diverse good codes
        if v < max(pop_vals) + 1e-6:
            # avoid duplicates: check min distance to pop
            dup = False
            for q in pop:
                # cheap check: mean abs corr? skip detailed
                pass
            pop.append(y.copy()); pop_vals.append(v)
            o = np.argsort(pop_vals)
            pop = [pop[i] for i in o[:5]]
            pop_vals = [pop_vals[i] for i in o[:5]]
        if v < best_val - 1e-12:
            best, best_val = y.copy(), v
        elif v < best_val and v < best_val + 1e-12:
            # tie but different topology: keep the more balanced one (fewer contacts)
            try:
                G1 = best @ best.T; np.fill_diagonal(G1, -2.0)
                m1 = float(G1.max()); c1 = int((G1 >= m1 - 1e-9).sum() // 2)
                G2 = y @ y.T; np.fill_diagonal(G2, -2.0)
                m2 = float(G2.max()); c2 = int((G2 >= m2 - 1e-9).sum() // 2)
                if m2 < best_val or (abs(m2 - best_val) < 2e-12 and c2 < c1):
                    best, best_val = y.copy(), v
            except Exception:
                pass

    # final polish of best
    try:
        rem = deadline - time.time()
        if rem > 2.0:
            best = _lp_polish(best, min(rem - 1.0, 5.0))
            best_val = _max_cos(best)
    except Exception:
        pass
    best = best / np.linalg.norm(best, axis=1, keepdims=True)
    # tiny shrink guard not needed; just return
    return best
