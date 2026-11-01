import time
import numpy as np
from scipy.optimize import linprog, linear_sum_assignment
from scipy.sparse import coo_matrix

def solve():
    t0 = time.time()
    BUDGET = 172.0
    deadline = t0 + BUDGET
    FINAL_RESERVE = 8.0
    rng = np.random.default_rng(7)
    n, d = 173, 5
    iu0, iu1 = np.triu_indices(n, 1)
    npair = len(iu0)

    def _max_cos(x):
        g = x @ x.T
        np.fill_diagonal(g, -2.0)
        return float(g.max())

    def _lp_polish(x, seconds, radius_init=1e-3):
        x = x / np.linalg.norm(x, axis=1, keepdims=True)
        if seconds <= 0.02:
            return x
        best = _max_cos(x)
        radius = radius_init
        start = time.time()
        while True:
            if time.time() - start >= seconds:
                break
            if radius < 1e-14:
                break
            G = x @ x.T
            c_all = G[iu0, iu1]
            m = float(c_all.max())
            thresh = m - max(20.0 * radius, 1e-9)
            sel = c_all > thresh
            ns = int(sel.sum())
            if ns == 0:
                radius *= 0.5
                continue
            if ns > 3000:
                idx = np.where(sel)[0]
                order = np.argsort(c_all[idx])[-3000:]
                mask = np.zeros_like(sel)
                mask[idx[order]] = True
                sel = mask
            I = iu0[sel]; J = iu1[sel]; c = c_all[sel]
            p = len(I)
            nv = n * d + 1
            cols = np.concatenate([(I[:, None] * d + np.arange(d)),
                                   (J[:, None] * d + np.arange(d)),
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
            yn[yn == 0] = 1
            y = y / yn
            v = _max_cos(y)
            if v < best:
                x, best = y, v
                radius = min(radius * 1.5, 0.05)
            else:
                radius *= 0.5
        return x

    def _smooth_gd(x, taus=(0.02, 0.01, 0.005), iters=(80, 80, 80), lrs=(0.8, 0.5, 0.3), mu=0.7, decay=0.998):
        x = x / np.linalg.norm(x, axis=1, keepdims=True)
        v = np.zeros_like(x)
        for tau, itn, lr0 in zip(taus, iters, lrs):
            lr = float(lr0)
            for _ in range(int(itn)):
                G = x @ x.T
                np.fill_diagonal(G, -10.0)
                m = float(G.max())
                E = np.exp((G - m) / tau)
                np.fill_diagonal(E, 0.0)
                s = float(E.sum())
                if not np.isfinite(s) or s <= 0:
                    break
                W = E / s
                A = W @ x
                dot = (A * x).sum(axis=1, keepdims=True)
                Ft = A - dot * x
                v = mu * v + (1.0 - mu) * Ft
                x = x - lr * v
                xn = np.linalg.norm(x, axis=1, keepdims=True)
                xn[xn == 0] = 1
                x = x / xn
                lr *= decay
        return x

    def _void_fill(rem, k, rng_, n_cand=4000):
        dd = rem.shape[1]
        cand = rng_.standard_normal((n_cand, dd))
        cand /= np.linalg.norm(cand, axis=1, keepdims=True)
        s_max = (cand @ rem.T).max(axis=1)
        out = []
        for _ in range(k):
            idx = int(np.argmin(s_max))
            pt = cand[idx].copy()
            out.append(pt)
            upd = cand @ pt
            np.maximum(s_max, upd, out=s_max)
            s_max[idx] = 2.0
        return np.stack(out, axis=0)

    def _contact_edges(x, thresh=1e-7):
        G = x @ x.T
        c = G[iu0, iu1]
        m = float(c.max())
        return (m - c) < thresh, m

    def _recombine(A, B, rng_):
        try:
            M = B.T @ A
            U, S, Vt = np.linalg.svd(M)
            Q = U @ Vt
            Brot = B @ Q
        except Exception:
            Brot = B
        try:
            Gc = A @ Brot.T
            row, col = linear_sum_assignment(-Gc)
            perm = np.empty(n, dtype=int)
            perm[row] = col
        except Exception:
            perm = np.arange(n)
            Brot = B
        child = np.empty_like(A)
        r = rng_.random()
        if r < 0.6:
            u = rng_.standard_normal(d)
            u /= np.linalg.norm(u)
            proj = A @ u
            q = float(rng_.uniform(0.3, 0.7))
            thr = np.quantile(proj, q)
            mask = proj > thr
            if rng_.random() < 0.5:
                mask = ~mask
            child[mask] = A[mask]
            idx = np.where(~mask)[0]
            child[idx] = Brot[perm[idx]]
        elif r < 0.85:
            mask = rng_.random(n) < 0.5
            child[mask] = A[mask]
            idx = np.where(~mask)[0]
            child[idx] = Brot[perm[idx]]
        else:
            w = float(rng_.uniform(0.3, 0.7))
            for i in range(n):
                v = w * A[i] + (1.0 - w) * Brot[perm[i]]
                nn = float(np.linalg.norm(v))
                if nn == 0:
                    v = A[i]
                    nn = 1.0
                child[i] = v / nn
        sigma = 0.008 + 0.025 * rng_.random()
        child = child + sigma * rng_.standard_normal(child.shape)
        child /= np.linalg.norm(child, axis=1, keepdims=True)
        return child

    def _mutate(parent, rng_):
        x = parent.copy()
        r = rng_.random()
        if r < 0.35:
            sigma = 0.02 + 0.12 * rng_.random()
            if rng_.random() < 0.10:
                sigma = 0.25 + 0.20 * rng_.random()
            x = x + sigma * rng_.standard_normal(x.shape)
            x /= np.linalg.norm(x, axis=1, keepdims=True)
        elif r < 0.62:
            G = x @ x.T
            np.fill_diagonal(G, -2.0)
            mi = G.max(axis=1)
            order = np.argsort(-mi)
            k = int(rng_.integers(2, 9))
            if rng_.random() < 0.7:
                sel = order[:k]
            else:
                pool = order[:30]
                sel = rng_.choice(pool, size=k, replace=False)
            mask = np.ones(n, dtype=bool)
            mask[sel] = False
            rem = x[mask]
            try:
                new_pts = _void_fill(rem, k, rng_, n_cand=4000)
            except Exception:
                new_pts = rng_.standard_normal((k, d))
                new_pts /= np.linalg.norm(new_pts, axis=1, keepdims=True)
            x[sel] = new_pts
            x = x + 0.004 * rng_.standard_normal(x.shape)
            x /= np.linalg.norm(x, axis=1, keepdims=True)
        elif r < 0.82:
            u = rng_.standard_normal(d)
            u /= np.linalg.norm(u)
            proj = x @ u
            k = int(rng_.integers(5, 16))
            sel = np.argsort(-proj)[:k]
            mask = np.ones(n, dtype=bool)
            mask[sel] = False
            rem = x[mask]
            try:
                new_pts = _void_fill(rem, k, rng_, n_cand=4000)
            except Exception:
                new_pts = rng_.standard_normal((k, d))
                new_pts /= np.linalg.norm(new_pts, axis=1, keepdims=True)
            x[sel] = new_pts
            x = x + 0.005 * rng_.standard_normal(x.shape)
            x /= np.linalg.norm(x, axis=1, keepdims=True)
        else:
            k = int(rng_.integers(10, 26))
            sel = rng_.choice(n, size=k, replace=False)
            mask = np.ones(n, dtype=bool)
            mask[sel] = False
            rem = x[mask]
            try:
                new_pts = _void_fill(rem, k, rng_, n_cand=3500)
            except Exception:
                new_pts = rng_.standard_normal((k, d))
                new_pts /= np.linalg.norm(new_pts, axis=1, keepdims=True)
            x[sel] = new_pts
            x = x + 0.01 * rng_.standard_normal(x.shape)
            x /= np.linalg.norm(x, axis=1, keepdims=True)
        return x

    # load incumbent
    best = None
    best_val = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn).astype(np.float64)
            if a.shape == (n, d):
                an = a / np.linalg.norm(a, axis=1, keepdims=True)
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

    # initial light polish of incumbent
    try:
        best = _lp_polish(best, 1.2, radius_init=2e-3)
        best_val = _max_cos(best)
    except Exception:
        pass

    P = 12
    pool_x = []
    pool_v = []
    pool_e = []
    e0, m0 = _contact_edges(best)
    pool_x.append(best.copy()); pool_v.append(best_val); pool_e.append(e0)
    # build diverse initial pool
    sigmas = [0.05, 0.08, 0.12, 0.18, 0.25, 0.35, 0.07, 0.15, 0.22, 0.30, 0.45]
    for si in range(P - 1):
        if time.time() > deadline - FINAL_RESERVE - 20.0:
            break
        try:
            sig = sigmas[si % len(sigmas)]
            xt = best + sig * rng.standard_normal((n, d))
            xt /= np.linalg.norm(xt, axis=1, keepdims=True)
            # occasional rebuild part
            if si % 3 == 2:
                k = int(rng.integers(6, 16))
                sel = rng.choice(n, size=k, replace=False)
                mask = np.ones(n, dtype=bool); mask[sel] = False
                try:
                    xt[sel] = _void_fill(xt[mask], k, rng, n_cand=3000)
                    xt /= np.linalg.norm(xt, axis=1, keepdims=True)
                except Exception:
                    pass
            xt = _smooth_gd(xt, taus=(0.02, 0.01), iters=(60, 60), lrs=(0.7, 0.4))
            xt = _lp_polish(xt, 0.55)
            v = _max_cos(xt)
            e, _ = _contact_edges(xt)
            pool_x.append(xt); pool_v.append(v); pool_e.append(e)
            if v < best_val - 1e-12:
                best_val = v; best = xt.copy()
        except Exception:
            continue
    # pad if short
    while len(pool_x) < 4:
        xt = rng.standard_normal((n, d))
        xt /= np.linalg.norm(xt, axis=1, keepdims=True)
        e, _ = _contact_edges(xt)
        pool_x.append(xt); pool_v.append(_max_cos(xt)); pool_e.append(e)

    pool_v = np.array(pool_v, dtype=float)

    it = 0
    while time.time() < deadline - FINAL_RESERVE:
        it += 1
        rem_total = deadline - FINAL_RESERVE - time.time()
        if rem_total < 0.8:
            break
        elapsed = time.time() - t0
        total_span = BUDGET - FINAL_RESERVE
        prog = min(max(elapsed / max(total_span, 1e-9), 0.0), 1.0)
        T = 2e-4 * (0.005 ** prog)  # 2e-4 -> 1e-6
        if T < 1e-7:
            T = 1e-7
        try:
            # parent selection: uniform for diversity
            np_ = len(pool_x)
            if rng.random() < 0.35 and np_ >= 2:
                a_idx, b_idx = rng.choice(np_, size=2, replace=False)
                A = pool_x[int(a_idx)]; B = pool_x[int(b_idx)]
                xt = _recombine(A, B, rng)
            else:
                p_idx = int(rng.integers(0, np_))
                xt = _mutate(pool_x[p_idx], rng)
            # brief anneal
            if rng.random() < 0.7:
                xt = _smooth_gd(xt, taus=(0.015, 0.007), iters=(70, 70), lrs=(0.6, 0.35))
            else:
                xt = _smooth_gd(xt, taus=(0.025, 0.012, 0.006), iters=(50, 50, 50), lrs=(0.8, 0.5, 0.3))
            pt = 0.55
            if rem_total < 4.0:
                pt = 0.35
            xt = _lp_polish(xt, pt)
            v = _max_cos(xt)
            if v < best_val - 1e-12:
                best_val = v
                best = xt.copy() / np.linalg.norm(xt, axis=1, keepdims=True)
            # diversity check
            e, _ = _contact_edges(xt)
            # hamming to pool
            dmins = []
            for pe in pool_e:
                dmins.append(np.count_nonzero(e != pe))
            dmins = np.array(dmins)
            imin = int(np.argmin(dmins))
            dmin = int(dmins[imin])
            if v < best_val + 1e-15:
                # new record (or tie): always insert replacing worst
                iw = int(np.argmax(pool_v))
                # avoid replacing if identical graph and not better
                pool_x[iw] = xt.copy(); pool_v[iw] = v; pool_e[iw] = e
                continue
            if dmin < 20 and v >= pool_v[imin]:
                continue  # veto: too similar and not better than neighbour
            iw = int(np.argmax(pool_v))
            if v < pool_v[iw]:
                pool_x[iw] = xt.copy(); pool_v[iw] = v; pool_e[iw] = e
            else:
                prob = float(np.exp(-(v - best_val) / max(T, 1e-12)))
                # cap prob
                if prob > 1.0:
                    prob = 1.0
                if rng.random() < prob * 0.5:
                    # replace a random member among worst half (keep best)
                    order = np.argsort(pool_v)  # ascending
                    # candidates: upper half excluding best? choose from top half
                    half = max(1, len(order) // 2)
                    cand_idx = order[-half:]
                    # don't replace global best index
                    best_idx = int(np.argmin(pool_v))
                    cand_idx = [c for c in cand_idx if c != best_idx]
                    if len(cand_idx) > 0:
                        ri = int(rng.choice(cand_idx))
                        pool_x[ri] = xt.copy(); pool_v[ri] = v; pool_e[ri] = e
        except Exception:
            continue

    # final intensive polish of global best
    try:
        frem = deadline - time.time() - 0.4
        if frem > 1.0:
            best = _lp_polish(best, min(frem, 7.0), radius_init=2e-3)
            best_val = _max_cos(best)
    except Exception:
        pass
    best = best / np.linalg.norm(best, axis=1, keepdims=True)
    return best
