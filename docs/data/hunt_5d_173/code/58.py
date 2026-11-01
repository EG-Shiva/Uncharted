import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _repel(x, alphas, iters_list, lrs, mu, decay):
    v = np.zeros_like(x)
    for alpha, iters, lr0 in zip(alphas, iters_list, lrs):
        lr = float(lr0)
        for _ in range(int(iters)):
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
    return x

def _lp_polish(x, seconds):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    try:
        best = _max_cos(x)
    except Exception:
        return x
    radius = 1e-3
    start = time.time()
    iu0, iu1 = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        c = (x @ x.T)[iu0, iu1]
        m = float(c.max())
        sel = c > m - max(20 * radius, 1e-9)
        I, J = iu0[sel], iu1[sel]
        p = len(I)
        nv = n * d + 1
        if p == 0:
            break
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv); obj[-1] = 1.0
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

def _contact_adj(x, thresh=1e-9):
    n = x.shape[0]
    G = x @ x.T
    iu0, iu1 = np.triu_indices(n, 1)
    c = G[iu0, iu1]
    m = float(c.max())
    mask = (m - c) < thresh
    A = np.zeros((n, n), dtype=float)
    A[iu0[mask], iu1[mask]] = 1.0
    A[iu1[mask], iu0[mask]] = 1.0
    return A, m

def _orbit_classes(x, thresh=1e-9):
    A, _ = _contact_adj(x, thresh)
    n = x.shape[0]
    deg = A.sum(axis=1).astype(int)
    # hash by (degree, sorted neighbour degrees)
    buckets = {}
    for i in range(n):
        nbrs = np.where(A[i] > 0)[0]
        nd = tuple(sorted(deg[nbrs].tolist())) if len(nbrs) else ()
        key = (int(deg[i]), nd)
        buckets.setdefault(key, []).append(i)
    classes = [np.array(v, dtype=int) for v in buckets.values()]
    # sort by size descending for cycling
    classes.sort(key=lambda a: -len(a))
    return classes, A, deg

def _modularity(comm, A, k, m2):
    if m2 == 0:
        return 0.0
    q = 0.0
    for c in np.unique(comm):
        idx = np.where(comm == c)[0]
        e = A[np.ix_(idx, idx)].sum()
        tot = k[idx].sum()
        q += e / m2 - (tot / m2) ** 2
    return q

def _greedy_communities(A, rng, max_pass=8):
    n = A.shape[0]
    k = A.sum(axis=1)
    m2 = k.sum()
    if m2 == 0:
        return [np.arange(n)]
    comm = np.arange(n)
    curQ = _modularity(comm, A, k, m2)
    for _ in range(max_pass):
        improved = False
        order = rng.permutation(n)
        for i in order:
            nbrs = np.where(A[i] > 0)[0]
            if len(nbrs) == 0:
                continue
            cands = np.unique(comm[nbrs])
            # also keep own
            best_c = comm[i]
            best_q = curQ
            old = comm[i]
            for c in cands:
                if c == old:
                    continue
                comm[i] = c
                q = _modularity(comm, A, k, m2)
                if q > best_q + 1e-10:
                    best_q = q
                    best_c = c
            comm[i] = old
            if best_c != old:
                comm[i] = best_c
                curQ = best_q
                improved = True
        if not improved:
            break
    # relabel and collect
    uniq = np.unique(comm)
    out = [np.where(comm == c)[0] for c in uniq]
    out.sort(key=lambda a: -len(a))
    return out

def _rand_2plane_rot(d, theta, rng):
    Q, _ = np.linalg.qr(rng.standard_normal((d, 2)))
    q1 = Q[:, 0]; q2 = Q[:, 1]
    c = np.cos(theta); s = np.sin(theta)
    R = np.eye(d) + (c - 1.0) * (np.outer(q1, q1) + np.outer(q2, q2)) + s * (np.outer(q2, q1) - np.outer(q1, q2))
    return R

def _coherent_rotate(x, idx, theta, rng):
    d = x.shape[1]
    R = _rand_2plane_rot(d, theta, rng)
    x[idx] = (R @ x[idx].T).T
    x[idx] /= np.linalg.norm(x[idx], axis=1, keepdims=True)
    return x

def _coherent_drift(x, idx, amp, rng):
    d = x.shape[1]
    u = rng.standard_normal(d); u /= np.linalg.norm(u)
    for i in idx:
        xi = x[i]
        t = u - float(u @ xi) * xi
        nrm = np.linalg.norm(t)
        if nrm < 1e-8:
            u2 = rng.standard_normal(d); u2 /= np.linalg.norm(u2)
            t = u2 - float(u2 @ xi) * xi
            nrm = np.linalg.norm(t)
            if nrm < 1e-8:
                continue
        t /= nrm
        x[i] = np.cos(amp) * xi + np.sin(amp) * t
    x[idx] /= np.linalg.norm(x[idx], axis=1, keepdims=True)
    return x

def solve():
    t0 = time.time()
    deadline = t0 + 168.0
    deadline_search = deadline - 32.0
    rng = np.random.RandomState(7)
    n, d = 173, 5
    # load
    cands = []
    for fn in ("best.npy", "record.npy"):
        try:
            c = np.load(fn).astype(float)
            if c.shape == (n, d):
                c = c / np.linalg.norm(c, axis=1, keepdims=True)
                if np.all(np.isfinite(c)):
                    cands.append(c)
        except Exception:
            pass
    if not cands:
        tmp = rng.standard_normal((n, d)); tmp /= np.linalg.norm(tmp, axis=1, keepdims=True)
        cands.append(tmp)
    best = None; best_val = 2.0
    for c in cands:
        v = _max_cos(c)
        if v < best_val:
            best_val = v; best = c.copy()
    fallback = best.copy(); fallback_val = best_val

    # initial classes / communities from record
    try:
        orbits, A0, deg0 = _orbit_classes(best, 1e-9)
    except Exception:
        orbits = [np.arange(n)]; A0 = np.zeros((n, n))
    try:
        comms = _greedy_communities(A0, rng)
    except Exception:
        comms = [np.arange(n)]
    # build pool: list of (kind, array)
    def rebuild_pool():
        pool = []
        for o in orbits:
            pool.append((0, o))
        for cm in comms:
            if len(cm) >= 3 and len(cm) <= 120:
                pool.append((1, cm))
        if not pool:
            pool.append((0, np.arange(n)))
        return pool
    pool = rebuild_pool()

    tabu = set()
    trial = 0
    order_ptr = 0
    # randomize pool order
    perm = rng.permutation(len(pool))
    pool = [pool[i] for i in perm]

    while time.time() < deadline_search:
        trial += 1
        # refresh classes periodically or after improvement
        if trial % 25 == 1 and trial > 1:
            try:
                orbits, A0, _ = _orbit_classes(best, 1e-9)
                comms = _greedy_communities(A0, rng)
                pool = rebuild_pool()
                perm = rng.permutation(len(pool))
                pool = [pool[i] for i in perm]
                tabu.clear()
                order_ptr = 0
            except Exception:
                pass
        # pick class cycling, skipping tabu
        chosen = None; chosen_kind = 0
        for _ in range(len(pool)):
            kind, idx = pool[order_ptr % len(pool)]
            order_ptr += 1
            key = (tuple(sorted(idx.tolist())), trial % 3)
            # use simpler tabu: hash of (first,last,len,move)
            tkey = (len(idx), int(idx[0]) if len(idx) else -1, int(idx[-1]) if len(idx) else -1, trial % 3)
            if tkey in tabu:
                continue
            chosen = idx; chosen_kind = kind
            break
        if chosen is None:
            tabu.clear()
            kind, chosen = pool[rng.randint(len(pool))]
            chosen_kind = kind
        move = rng.randint(3)  # 0 rot, 1 drift, 2 break
        amp = 0.02 + rng.rand() * 0.13  # 0.02..0.15
        # occasionally larger to change basin
        if rng.rand() < 0.08:
            amp = 0.15 + rng.rand() * 0.15
        x_try = best.copy()
        try:
            if move == 0:
                theta = amp * (1 if rng.rand() < 0.5 else -1)
                # preserve-as-unit: rotate whole class by same R
                _coherent_rotate(x_try, chosen, theta, rng)
                # tiny global jitter to allow relax
                x_try += 0.002 * rng.standard_normal(x_try.shape)
                x_try /= np.linalg.norm(x_try, axis=1, keepdims=True)
            elif move == 1:
                _coherent_drift(x_try, chosen, amp, rng)
                x_try += 0.002 * rng.standard_normal(x_try.shape)
                x_try /= np.linalg.norm(x_try, axis=1, keepdims=True)
            else:
                # break symmetry: individual kicks inside class
                sig = amp
                x_try[chosen] += sig * rng.standard_normal((len(chosen), d))
                x_try[chosen] /= np.linalg.norm(x_try[chosen], axis=1, keepdims=True)
                # keep rest almost fixed
        except Exception:
            continue
        # quick repel
        a1 = 30.0 + rng.rand() * 20.0
        a2 = 90.0 + rng.rand() * 60.0
        remain = deadline_search - time.time()
        if remain < 8:
            iters_list = [100, 120]
            alphas = [a1, a2]
            lrs = [0.06 + rng.rand() * 0.08, 0.02 + rng.rand() * 0.03]
            mu = 0.5 + rng.rand() * 0.35
            decay = 0.997 + rng.rand() * 0.002
        else:
            a3 = 180.0 + rng.rand() * 140.0
            alphas = [a1, a2, a3]
            iters_list = [150, 180, 220]
            lrs = [0.06 + rng.rand() * 0.10, 0.03 + rng.rand() * 0.05, 0.015 + rng.rand() * 0.025]
            mu = 0.5 + rng.rand() * 0.35
            decay = 0.997 + rng.rand() * 0.002
            if decay > 0.9995:
                decay = 0.9995
        try:
            x_opt = _repel(x_try, alphas, iters_list, lrs, mu, decay)
            v = _max_cos(x_opt)
        except Exception:
            continue
        # mark tabu
        tkey = (len(chosen), int(chosen[0]) if len(chosen) else -1, int(chosen[-1]) if len(chosen) else -1, move)
        tabu.add(tkey)
        if len(tabu) > 60:
            # drop arbitrary
            tabu.pop()
        # if promising, short LP polish to evaluate true basin value
        thresh = best_val + (0.005 if remain > 20 else 0.002)
        if v < thresh:
            try:
                x_pol = _lp_polish(x_opt, 1.8 if remain > 15 else 0.8)
                vp = _max_cos(x_pol)
            except Exception:
                vp = v; x_pol = x_opt
            if vp < best_val - 1e-12:
                best = x_pol; best_val = vp
                # recompute orbits/comms immediately
                try:
                    orbits, A0, _ = _orbit_classes(best, 1e-9)
                    comms = _greedy_communities(A0, rng)
                    pool = rebuild_pool()
                    tabu.clear()
                    order_ptr = 0
                except Exception:
                    pass
                if best_val < 0.74:
                    break
            elif v < best_val - 1e-9:
                # repel alone improved w/o polish? polish longer next time
                pass
        # safety: time check
        if time.time() >= deadline_search:
            break

    # fallback retain
    if best_val > fallback_val + 1e-12:
        best = fallback; best_val = fallback_val
    # final LP polish with remaining budget
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
    # strict finite check; if bad return fallback
    try:
        vv = _max_cos(best)
        if not np.isfinite(vv):
            best = fallback
    except Exception:
        best = fallback
    return best
