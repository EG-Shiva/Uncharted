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

def _lp_polish(x, seconds):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    if seconds <= 0.05:
        return x
    radius = 1e-3
    start = time.time()
    iu0, iu1 = np.triu_indices(n, 1)
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
        cnt_sel = int(sel.sum())
        if cnt_sel > 3000:
            idx = np.where(sel)[0]
            order = np.argsort(c_all[idx])[-3000:]
            mask = np.zeros_like(sel)
            mask[idx[order]] = True
            sel = mask
        I = iu0[sel]
        J = iu1[sel]
        c = c_all[sel]
        p = len(I)
        if p == 0:
            radius *= 0.5
            continue
        nv = n * d + 1
        cols = np.concatenate([
            (I[:, None] * d + np.arange(d)),
            (J[:, None] * d + np.arange(d)),
            np.full((p, 1), n * d)], axis=1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], axis=1).ravel()
        rows = np.repeat(np.arange(p), 2 * d + 1)
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        except Exception:
            radius *= 0.5
            continue
        obj = np.zeros(nv)
        obj[-1] = 1.0
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
        yn = np.linalg.norm(y, axis=1, keepdims=True)
        yn[yn == 0] = 1.0
        y = y / yn
        v = _max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def _contact_bool(x, iu0, iu1, thresh=1e-7):
    G = x @ x.T
    c = G[iu0, iu1]
    m = float(c.max())
    return (m - c) < thresh, float(m)

def _void_fill(rem, k, rng, n_cand=2500):
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

def solve():
    t0 = time.time()
    BUDGET = 172.0
    deadline = t0 + BUDGET
    rng = np.random.default_rng(42)
    n, d = _N, _D
    iu0, iu1 = np.triu_indices(n, 1)

    base = None
    base_val = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn).astype(np.float64)
            if a.shape == (n, d):
                an = a / np.linalg.norm(a, axis=1, keepdims=True)
                v = _max_cos(an)
                if v < base_val:
                    base_val = v
                    base = an.copy()
        except Exception:
            pass
    if base is None:
        base = rng.normal(size=(n, d))
        base = base / np.linalg.norm(base, axis=1, keepdims=True)
        base_val = _max_cos(base)

    POOL = 14
    DTHRESH = 20

    def make_edges(x):
        e, m = _contact_bool(x, iu0, iu1, thresh=1e-7)
        return e, m

    pool = []
    # seed pool: first member is base as-is
    e0, m0 = make_edges(base)
    pool.append({"x": base.copy(), "val": base_val, "edges": e0})
    # build diverse raw starts
    for pi in range(1, POOL):
        y = base.copy()
        r = rng.random()
        if pi < 4:
            sigma = 0.08 + 0.10 * rng.random()
            y = y + sigma * rng.normal(size=y.shape)
            y /= np.linalg.norm(y, axis=1, keepdims=True)
        elif pi < 8:
            k = int(rng.integers(4, 12))
            G = y @ y.T
            np.fill_diagonal(G, -2.0)
            mi = G.max(axis=1)
            order = np.argsort(-mi)
            sel = rng.choice(order[:40], size=k, replace=False)
            mask = np.ones(n, dtype=bool)
            mask[sel] = False
            rem = y[mask]
            try:
                new_pts = _void_fill(rem, k, rng)
            except Exception:
                new_pts = rng.normal(size=(k, d))
                new_pts /= np.linalg.norm(new_pts, axis=1, keepdims=True)
            y[sel] = new_pts
            y = y + 0.01 * rng.normal(size=y.shape)
            y /= np.linalg.norm(y, axis=1, keepdims=True)
        elif pi < 11:
            sigma = 0.25 + 0.25 * rng.random()
            y = y + sigma * rng.normal(size=y.shape)
            y /= np.linalg.norm(y, axis=1, keepdims=True)
        else:
            # cap restart
            u = rng.normal(size=(d,))
            u /= np.linalg.norm(u)
            proj = y @ u
            k = int(rng.integers(10, 25))
            order = np.argsort(-proj)
            sel = order[:k]
            y[sel] = rng.normal(size=(k, d))
            y[sel] /= np.linalg.norm(y[sel], axis=1, keepdims=True)
            y = y + 0.02 * rng.normal(size=y.shape)
            y /= np.linalg.norm(y, axis=1, keepdims=True)
        rem = deadline - time.time()
        pt = 0.8
        if rem < 30:
            pt = 0.4
        y = _lp_polish(y, pt)
        v = _max_cos(y)
        e, _ = make_edges(y)
        pool.append({"x": y, "val": v, "edges": e})
        if time.time() - t0 > 40:
            break
    # sort pool
    pool.sort(key=lambda z: z["val"])
    # trim to unique-ish? keep all
    # deduplicate exact same graph keeping best
    best_overall_x = pool[0]["x"].copy()
    best_overall_val = float(pool[0]["val"])

    def recombine(A, B):
        M = B.T @ A
        U, S, Vt = np.linalg.svd(M)
        R = U @ Vt
        Bal = B @ R
        G = A @ Bal.T
        G = np.clip(G, -1.0, 1.0)
        D = np.sqrt(np.maximum(0.0, 2.0 - 2.0 * G))
        row, col = linear_sum_assignment(D)
        Bperm = np.empty_like(A)
        Bperm[row] = Bal[col]
        u = rng.normal(size=(d,))
        u /= np.linalg.norm(u)
        proj = ((A + Bperm) * 0.5) @ u
        thr = np.median(proj)
        mask = proj > thr
        if rng.random() < 0.5:
            # randomly swap which side comes from which parent by flipping mask
            # (symmetric but changes noise pattern) - keep as is, just choose orientation
            pass
        # randomly choose orientation of mask assignment
        if rng.random() < 0.5:
            child = np.where(mask[:, None], A, Bperm)
        else:
            child = np.where(mask[:, None], Bperm, A)
        sigma = 0.008 + 0.025 * rng.random()
        child = child + sigma * rng.normal(size=child.shape)
        child /= np.linalg.norm(child, axis=1, keepdims=True)
        return child

    def mutate(parent_x):
        y = parent_x.copy()
        r = rng.random()
        G = y @ y.T
        np.fill_diagonal(G, -2.0)
        mi = G.max(axis=1)
        order_desc = np.argsort(-mi)
        order_asc = np.argsort(mi)
        if r < 0.30:
            # overload-targeted void reinsertion
            k = int(rng.integers(2, 9))
            pool_top = order_desc[:35]
            sel = rng.choice(pool_top, size=k, replace=False)
            mask = np.ones(n, dtype=bool)
            mask[sel] = False
            rem_pts = y[mask]
            try:
                new_pts = _void_fill(rem_pts, k, rng)
            except Exception:
                new_pts = rng.normal(size=(k, d))
                new_pts /= np.linalg.norm(new_pts, axis=1, keepdims=True)
            y[sel] = new_pts
            y = y + (0.005 + 0.015 * rng.random()) * rng.normal(size=y.shape)
            y /= np.linalg.norm(y, axis=1, keepdims=True)
        elif r < 0.50:
            # jammed + random kick medium
            k_rand = rng.choice(n, 5, replace=False)
            subset = np.unique(np.concatenate([order_desc[:3], k_rand]))
            sigma = 0.06 + 0.06 * rng.random()
            y[subset] += sigma * rng.normal(size=(len(subset), d))
            y /= np.linalg.norm(y, axis=1, keepdims=True)
        elif r < 0.68:
            # broad shake
            sigma = 0.04 + 0.18 * rng.random()
            if rng.random() < 0.12:
                sigma = 0.30 + 0.20 * rng.random()
            y = y + sigma * rng.normal(size=y.shape)
            y /= np.linalg.norm(y, axis=1, keepdims=True)
        elif r < 0.84:
            # cap replacement
            u = rng.normal(size=(d,))
            u /= np.linalg.norm(u)
            proj = y @ u
            k = int(rng.integers(6, 20))
            # random side
            if rng.random() < 0.5:
                sel = np.argsort(-proj)[:k]
            else:
                sel = np.argsort(proj)[:k]
            # replace with void-ish random far from remainder
            mask = np.ones(n, dtype=bool)
            mask[sel] = False
            rem_pts = y[mask]
            try:
                new_pts = _void_fill(rem_pts, k, rng, n_cand=2000)
            except Exception:
                new_pts = rng.normal(size=(k, d))
                new_pts /= np.linalg.norm(new_pts, axis=1, keepdims=True)
            y[sel] = new_pts
            y = y + 0.01 * rng.normal(size=y.shape)
            y /= np.linalg.norm(y, axis=1, keepdims=True)
        else:
            # small global + rattler kick
            sigma = 0.012 + 0.02 * rng.random()
            y = y + sigma * rng.normal(size=y.shape)
            # extra kick to least-contact points
            G2 = parent_x @ parent_x.T
            np.fill_diagonal(G2, -2.0)
            mcur = float(G2.max())
            cnt = (G2 >= mcur - 1e-7).sum(axis=1)
            low = np.where(cnt <= 2)[0]
            if len(low) > 0:
                k = min(len(low), int(rng.integers(1, 4)))
                sel = rng.choice(low, size=k, replace=False)
                y[sel] += 0.08 * rng.normal(size=(k, d))
            y /= np.linalg.norm(y, axis=1, keepdims=True)
        return y

    trial = 0
    while True:
        rem = deadline - time.time()
        if rem < 7.0:
            break
        elapsed = time.time() - t0
        prog = elapsed / BUDGET
        prog = min(max(prog, 0.0), 1.0)
        T = 2e-4 * (0.005 ** prog)  # 2e-4 -> 1e-6
        pt = 1.1
        if rem < 25:
            pt = max(0.5, rem - 6.0)
        pt = min(pt, rem - 6.0)
        if pt < 0.4:
            break
        # select parent(s)
        idx_a = int(rng.integers(len(pool)))
        if rng.random() < 0.28 and len(pool) >= 2:
            idx_b = int(rng.integers(len(pool)))
            while idx_b == idx_a:
                idx_b = int(rng.integers(len(pool)))
            try:
                cand_raw = recombine(pool[idx_a]["x"], pool[idx_b]["x"])
            except Exception:
                cand_raw = mutate(pool[idx_a]["x"])
        else:
            cand_raw = mutate(pool[idx_a]["x"])
        cand = _lp_polish(cand_raw, pt)
        v = _max_cos(cand)
        e, _ = make_edges(cand)
        if v < best_overall_val:
            best_overall_val = float(v)
            best_overall_x = cand.copy()
        # diversity insertion
        # compute hamming distances
        dmins = None
        jmin = -1
        dmin = 10 ** 9
        for j, mem in enumerate(pool):
            dh = int(np.count_nonzero(e ^ mem["edges"]))
            if dh < dmin:
                dmin = dh
                jmin = j
        if dmin <= DTHRESH:
            if v < pool[jmin]["val"] - 1e-12:
                pool[jmin] = {"x": cand, "val": v, "edges": e}
                pool.sort(key=lambda z: z["val"])
        else:
            # diverse graph
            worst_idx = int(np.argmax([m["val"] for m in pool]))
            if v < pool[worst_idx]["val"]:
                pool[worst_idx] = {"x": cand, "val": v, "edges": e}
                pool.sort(key=lambda z: z["val"])
            else:
                # Metropolis accept worse with prob exp(-(v-best)/T)
                bv = float(min(m["val"] for m in pool))
                try:
                    p = float(np.exp(-(v - bv) / max(T, 1e-12)))
                except Exception:
                    p = 0.0
                if rng.random() < p:
                    pool[worst_idx] = {"x": cand, "val": v, "edges": e}
                    pool.sort(key=lambda z: z["val"])
        trial += 1

    rem = deadline - time.time()
    if rem > 1.2:
        try:
            best_overall_x = _lp_polish(best_overall_x, rem - 0.6)
        except Exception:
            pass
    best_overall_x = best_overall_x / np.linalg.norm(best_overall_x, axis=1, keepdims=True)
    return best_overall_x
