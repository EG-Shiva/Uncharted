import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

N_D = 5
N_N = 173

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _degrees(x, thresh=1e-7):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    m = float(G.max())
    deg = ((m - G) < thresh).sum(axis=1)
    mi = G.max(axis=1)
    return deg, mi, m, G

def _repel_global(x, rng, deadline):
    n, d = x.shape
    a1 = 30.0 + rng.rand() * 20.0
    a2 = 80.0 + rng.rand() * 60.0
    a3 = 180.0 + rng.rand() * 140.0
    alphas = [a1, a2, a3]
    rem = deadline - time.time()
    if rem < 12:
        iters_list = [120, 150, 180]
    else:
        iters_list = [200, 250, 350]
    l1 = 0.06 + rng.rand() * 0.10
    l2 = 0.03 + rng.rand() * 0.05
    l3 = 0.015 + rng.rand() * 0.025
    lrs = [l1, l2, l3]
    mu = 0.5 + rng.rand() * 0.35
    decay = 0.997 + rng.rand() * 0.002
    if decay > 0.9995:
        decay = 0.9995
    v = np.zeros_like(x)
    for alpha, iters, lr0 in zip(alphas, iters_list, lrs):
        lr = float(lr0)
        for s in range(int(iters)):
            if (s & 31) == 0 and time.time() > deadline:
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
        if time.time() > deadline:
            break
    return x

def _local_repel_patch(x, patch_idx, iters=220, alpha=80.0, lr0=0.06, decay=0.9985, mu=0.6):
    patch = np.asarray(sorted(set(int(v) for v in patch_idx)), dtype=int)
    if len(patch) == 0:
        return x
    v = np.zeros_like(x)
    lr = float(lr0)
    for s in range(int(iters)):
        G = x @ x.T
        W = np.exp(alpha * (G - 0.75))
        np.fill_diagonal(W, 0.0)
        rs = W.sum(axis=1, keepdims=True) + 1e-300
        A = (W @ x) / rs
        dot = (A * x).sum(axis=1, keepdims=True)
        Ft = A - dot * x
        v = mu * v + (1.0 - mu) * Ft
        x[patch] = x[patch] - lr * v[patch]
        x[patch] /= np.linalg.norm(x[patch], axis=1, keepdims=True)
        lr *= decay
    return x

def _lp_polish(x, seconds):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
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

def _pick_patch(x, used, mode, rng, G, deg, mi):
    n = x.shape[0]
    # candidate centre ordering per mode
    if mode == 0:
        # 13-contact / most overloaded first
        order = np.argsort(-deg - 0.001 * mi)
        cand_list = [int(v) for v in order[:30]]
    elif mode == 1:
        # rattler's nearest jammed cluster
        ratt = np.where(deg == 0)[0]
        if len(ratt) > 0:
            r = int(ratt[0])
        else:
            r = int(np.argmin(deg + 0.01 * mi))
        nbr = np.argsort(-G[r])
        nbr = [int(v) for v in nbr if v != r]
        centre = None
        for v in nbr[:25]:
            if deg[v] >= 7:
                centre = v
                break
        if centre is None:
            centre = nbr[0] if len(nbr) > 0 else r
        cand_list = [centre] + [int(v) for v in np.argsort(-deg)[:10]]
    else:
        pool = np.where(deg >= 9)[0]
        if len(pool) == 0:
            pool = np.where(deg >= 7)[0]
        if len(pool) == 0:
            pool = np.argsort(-deg)[:20]
        perm = rng.permutation(len(pool))
        cand_list = [int(pool[i]) for i in perm[:20]]
    for centre in cand_list:
        if centre in used:
            continue
        m = int(rng.randint(8, 15))  # 8..14 neighbours
        # nearest neighbours by cosine
        row = G[centre]
        nbrs = np.argsort(-row)
        nbrs = [int(v) for v in nbrs if v != centre][:m]
        if len(nbrs) < m:
            continue
        patch = [centre] + nbrs
        if any(v in used for v in patch):
            # try smaller patch to avoid overlap
            ok = [centre] + [v for v in nbrs if v not in used]
            # need at least 9 points and centre not used; if overlap, skip unless still large
            if len(ok) < 9:
                continue
            # only accept reduced patch if it still has >=8 points
            patch = ok
        return patch
    # fallback: random non-used centre
    avail = [i for i in range(n) if i not in used]
    if len(avail) == 0:
        return None
    centre = avail[int(rng.randint(0, len(avail)))]
    row = G[centre]
    nbrs = np.argsort(-row)
    nbrs = [int(v) for v in nbrs if v != centre and v not in used][:int(rng.randint(8, 15))]
    patch = [centre] + nbrs
    if len(patch) < 5:
        return None
    return patch

def _void_fill_patch(x_try, patch, rng, n_cand=50000):
    n, d = x_try.shape
    patch = list(patch)
    frozen_mask = np.ones(n, dtype=bool)
    frozen_mask[np.array(patch, dtype=int)] = False
    rem = x_try[frozen_mask]
    cand = rng.standard_normal((n_cand, d))
    cand /= np.linalg.norm(cand, axis=1, keepdims=True)
    # max dot to frozen, chunked
    s_max = np.empty(n_cand, dtype=float)
    s_max.fill(-2.0)
    chunk = 10000
    for s in range(0, n_cand, chunk):
        e = min(s + chunk, n_cand)
        s_max[s:e] = (cand[s:e] @ rem.T).max(axis=1)
    order = list(patch)
    # random assignment order for greediness diversity
    rng.shuffle(order)
    for t in order:
        idx = int(np.argmin(s_max))
        pt = cand[idx].copy()
        x_try[t] = pt
        upd = cand @ pt
        np.maximum(s_max, upd, out=s_max)
        s_max[idx] = 2.0
    return x_try

def solve():
    t0 = time.time()
    deadline = t0 + 168.0
    deadline_search = deadline - 32.0
    rng = np.random.RandomState(7)
    try:
        xb = np.load("best.npy").astype(float)
    except Exception:
        xb = None
    try:
        xr = np.load("record.npy").astype(float)
    except Exception:
        xr = None
    cands = []
    for c in (xb, xr):
        if c is not None and c.shape == (173, 5):
            c = c / np.linalg.norm(c, axis=1, keepdims=True)
            cands.append(c)
    if len(cands) == 0:
        tmp = rng.standard_normal((173, 5))
        tmp /= np.linalg.norm(tmp, axis=1, keepdims=True)
        cands.append(tmp)
    best = None
    best_val = 2.0
    for c in cands:
        v = _max_cos(c)
        if v < best_val:
            best_val = v
            best = c.copy()
    n, d = 173, 5
    trial = 0
    # cycle modes 0,1,2 for non-overlapping neighbourhoods
    while time.time() < deadline_search:
        trial += 1
        if time.time() > deadline_search:
            break
        x_try = best.copy()
        # number of non-overlapping neighbourhoods this trial: 2 or 3
        if rng.rand() < 0.55:
            n_patches = 2
        else:
            n_patches = 3
        # occasionally single large patch
        if rng.rand() < 0.12:
            n_patches = 1
        used = set()
        # recompute contacts each patch to keep non-overlap w.r.t. current state
        patches_union = []
        ok_build = True
        base_mode = trial % 3
        for pi in range(n_patches):
            deg, mi, mval, G = _degrees(x_try, thresh=1e-7)
            mode = (base_mode + pi) % 3
            # mode 0: 13-contact, 1: rattler cluster, 2: random >=9
            patch = _pick_patch(x_try, used, mode, rng, G, deg, mi)
            if patch is None:
                ok_build = False
                break
            for v in patch:
                used.add(int(v))
            patches_union.extend(patch)
            # rebuild this patch immediately against frozen + already-reinserted
            try:
                x_try = _void_fill_patch(x_try, patch, rng, n_cand=50000)
            except Exception:
                ok_build = False
                break
            if time.time() > deadline_search:
                ok_build = False
                break
        if not ok_build or len(patches_union) == 0:
            continue
        # locally repel patch points with rest frozen
        try:
            x_try = _local_repel_patch(x_try, patches_union, iters=220, alpha=80.0 + rng.rand() * 40.0,
                                       lr0=0.04 + rng.rand() * 0.04, decay=0.9985, mu=0.55 + rng.rand() * 0.25)
        except Exception:
            continue
        if time.time() > deadline_search:
            break
        # unfreeze: annealed global repulsion
        try:
            x_try = _repel_global(x_try, rng, deadline_search)
        except Exception:
            continue
        v = _max_cos(x_try)
        if v < best_val - 1e-12:
            # squeeze promising basin with short LP before accepting
            try:
                x_try = _lp_polish(x_try, 2.0)
                v2 = _max_cos(x_try)
            except Exception:
                v2 = v
            if v2 < best_val:
                best = x_try / np.linalg.norm(x_try, axis=1, keepdims=True)
                best_val = float(v2)
                if best_val < 0.74:
                    break
        # opportunistic LP on near-best to discover topology change value
        elif v < best_val + 3e-4 and (trial % 25 == 0):
            if time.time() < deadline_search - 5:
                try:
                    y = _lp_polish(x_try, 1.0)
                    vy = _max_cos(y)
                    if vy < best_val - 1e-12:
                        best = y / np.linalg.norm(y, axis=1, keepdims=True)
                        best_val = float(vy)
                except Exception:
                    pass
    # final polish
    remain = deadline - time.time() - 1.5
    if remain < 5.0:
        remain = 5.0
    if remain > 34.0:
        remain = 34.0
    try:
        best = _lp_polish(best, remain)
    except Exception:
        pass
    best = best / np.linalg.norm(best, axis=1, keepdims=True)
    return best
