import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _norm(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return x / n

def _smooth(x_init, steps, k0=120.0, k1=2500.0, lr0=0.02, lr1=0.002):
    x = x_init.copy()
    n, d = x.shape
    iu = np.triu_indices(n, 1)
    for s in range(steps):
        f = s / max(steps - 1, 1)
        k = k0 * (k1 / k0) ** f
        lr = lr0 * (lr1 / lr0) ** f
        G = x @ x.T
        m = float(G[iu].max())
        E = np.exp(np.clip(k * (G - m), -50.0, 0.0))
        np.fill_diagonal(E, 0.0)
        Gr = E @ x
        dot = np.sum(Gr * x, axis=1, keepdims=True)
        Gt = Gr - dot * x
        deg = np.sum(E, axis=1, keepdims=True) + 1.0
        x = x - lr * Gt / deg
        x = _norm(x)
    return x

def _infer_blocks(x, nb=6):
    n = x.shape[0]
    G = x @ x.T
    np.fill_diagonal(G, 10.0)
    un = set(range(n))
    labels = np.zeros(n, dtype=int)
    bs = n // nb
    for b in range(nb):
        if not un:
            break
        seed = next(iter(un))
        # pick bs-1 others with smallest dot to seed
        others = list(un - {seed})
        if len(others) == 0:
            labels[seed] = b
            un.remove(seed)
            break
        dots = G[seed, others]
        order = np.argsort(dots)
        take = [seed] + [others[i] for i in order[:bs-1]]
        # last block takes remainder
        if b == nb - 1:
            take = list(un)
        for j in take:
            labels[j] = b
            un.discard(j)
    return labels

def _contact_sig(x, v):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    thr = v - 0.002
    cnt = (G > thr).sum(axis=1)
    return np.sort(cnt)

def _sig_dist(a, b):
    return float(np.sum(np.abs(a.astype(float) - b.astype(float))))

def _procrustes_align(A, B):
    # find R minimizing ||A - B R||
    M = B.T @ A
    U, _, Vt = np.linalg.svd(M)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    return B @ R

def _greedy_from_union(U, nsel=54):
    m = U.shape[0]
    G = U @ U.T
    np.fill_diagonal(G, 10.0)
    # start from minimal pair
    iu = np.triu_indices(m, 1)
    vals = G[iu]
    k = int(np.argmin(vals))
    i0, j0 = int(iu[0][k]), int(iu[1][k])
    sel = [i0, j0]
    selset = {i0, j0}
    # max dot to selected set for each candidate
    curmax = np.maximum(G[:, i0], G[:, j0])
    curmax[i0] = 10; curmax[j0] = 10
    for _ in range(nsel - 2):
        # mask selected
        curmax[list(selset)] = 10
        nxt = int(np.argmin(curmax))
        sel.append(nxt)
        selset.add(nxt)
        curmax = np.minimum(curmax, 10.0)  # keep
        curmax = np.maximum(curmax - 10.0 + 10.0, curmax)  # no-op
        # update: curmax[i] = max(curmax[i], G[i,nxt])
        # but selected entries stay 10
        gcol = G[:, nxt]
        # only update unselected
        mask = np.ones(m, dtype=bool)
        mask[list(selset)] = False
        curmax[mask] = np.maximum(curmax[mask], gcol[mask])
        curmax[list(selset)] = 10
    return U[np.array(sel)]

def _rand_plane(d, rng):
    a = rng.randn(d); b = rng.randn(d)
    # orthonormalize
    a /= np.linalg.norm(a)
    b = b - (a @ b) * a
    n = np.linalg.norm(b)
    if n < 1e-12:
        b = rng.randn(d); b /= np.linalg.norm(b)
    else:
        b /= n
    return a, b

def _rotate_block(x, idx, theta, u, v):
    y = x.copy()
    cu, su = np.cos(theta), np.sin(theta)
    for i in idx:
        p = y[i]
        a = float(p @ u); b_ = float(p @ v)
        rem = p - a * u - b_ * v
        ap = cu * a - su * b_
        bp = su * a + cu * b_
        q = rem + ap * u + bp * v
        n = np.linalg.norm(q)
        if n > 1e-12:
            q /= n
        y[i] = q
    return y

def _hole_cand(kept, rng, n_cand=1500):
    d = kept.shape[1]
    C = rng.randn(n_cand, d)
    C = _norm(C)
    dots = C @ kept.T
    vals = dots.max(axis=1)
    return C[int(np.argmin(vals))]

def _mutate(x, rng):
    n, d = x.shape
    labels = _infer_blocks(x, nb=6)
    blocks = [np.where(labels == b)[0] for b in range(6)]
    # drop empty
    blocks = [b for b in blocks if len(b) > 0]
    mode = int(rng.randint(3))
    y = x.copy()
    if mode == 0:
        bi = int(rng.randint(len(blocks)))
        idx = blocks[bi]
        th = float(rng.uniform(0.05, 0.5)) * (1 if rng.rand() < 0.5 else -1)
        u, v = _rand_plane(d, rng)
        y = _rotate_block(y, idx, th, u, v)
    elif mode == 1:
        if len(blocks) >= 2:
            i1, i2 = rng.choice(len(blocks), 2, replace=False)
            th = float(rng.uniform(0.05, 0.5))
            u, v = _rand_plane(d, rng)
            y = _rotate_block(y, blocks[int(i1)], th, u, v)
            y = _rotate_block(y, blocks[int(i2)], -th, u, v)
        else:
            idx = blocks[0]
            th = float(rng.uniform(0.05, 0.5))
            u, v = _rand_plane(d, rng)
            y = _rotate_block(y, idx, th, u, v)
    else:
        bi = int(rng.randint(len(blocks)))
        idx = blocks[bi]
        keep_mask = np.ones(n, dtype=bool)
        keep_mask[idx] = False
        kept = y[keep_mask]
        # reinsert len(idx) points greedily
        newpts = []
        cur = kept.copy()
        for _ in range(len(idx)):
            hv = _hole_cand(cur, rng, n_cand=1200)
            hv = hv + rng.randn(d) * 0.005
            hv /= np.linalg.norm(hv)
            newpts.append(hv)
            cur = np.vstack([cur, hv[None, :]])
        y[idx] = np.array(newpts)
        y = _norm(y)
    return y

def _lp_polish(x, seconds, t0=None, budget=1e9):
    x = _norm(x.copy())
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    start = time.time()
    iu0 = np.triu_indices(n, 1)
    while time.time() - start < seconds:
        if t0 is not None and (time.time() - t0) > budget:
            break
        if radius < 1e-14:
            break
        Gfull = x @ x.T
        c_all = Gfull[iu0]
        m = c_all.max()
        sel = c_all > m - max(20 * radius, 1e-9)
        if np.count_nonzero(sel) > 650:
            idx = np.where(sel)[0]
            order = np.argsort(c_all[idx])[::-1][:650]
            nsel = np.zeros_like(sel, dtype=bool)
            nsel[idx[order]] = True
            sel = nsel
        I, J = iu0[0][sel], iu0[1][sel]
        c = c_all[sel]
        p = len(I)
        if p == 0:
            break
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate(
            [I[:, None] * d + np.arange(d),
             J[:, None] * d + np.arange(d),
             np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix(
                (x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))),
                shape=(n, nv)).tocsr()
            obj = np.zeros(nv)
            obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-c, A_eq=A_eq,
                          b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                          method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y = _norm(y)
        v = _max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def solve():
    t0 = time.time()
    HARD_END = 168.0
    EXPLORE_END = 112.0
    rng = np.random.RandomState(7)
    n, d = 54, 23
    try:
        xb = np.load("best.npy").astype(float)
    except Exception:
        xb = rng.randn(n, d)
    xb = _norm(xb)
    if xb.shape != (n, d):
        xb = _norm(rng.randn(n, d))
    try:
        xr = np.load("record.npy").astype(float)
        xr = _norm(xr)
        if xr.shape != (n, d):
            xr = xb.copy()
    except Exception:
        xr = xb.copy()
    vb = _max_cos(xb)
    vr = _max_cos(xr)
    if vr < vb:
        best_v, x_best = vr, xr.copy()
    else:
        best_v, x_best = vb, xb.copy()
    # population: list of (x, v, sig)
    pop = []
    def add_to_pop(x, v):
        pop.append({"x": x.copy(), "v": float(v), "sig": _contact_sig(x, float(v))})
    add_to_pop(x_best, best_v)
    # second member: other file if distinct
    other = xr if best_v == vb else xb
    ov = vr if best_v == vb else vb
    # only add if not identical
    if abs(ov - best_v) > 1e-12 or True:
        # lightly perturb+ smooth to diversify quickly
        try:
            xp = _norm(other + rng.randn(n, d) * 0.01)
            xp = _smooth(xp, steps=150)
            add_to_pop(xp, _max_cos(xp))
        except Exception:
            add_to_pop(other, ov)
    # third member: block-rotated mutant prescreened
    try:
        xm = _mutate(x_best, rng)
        xm = _smooth(_norm(xm), steps=150)
        add_to_pop(xm, _max_cos(xm))
    except Exception:
        pass
    # update global best from pop
    for e in pop:
        if e["v"] < best_v:
            best_v, x_best = e["v"], e["x"].copy()
    # checkpoint
    try:
        if best_v <= vb:
            np.save("best.npy", _norm(x_best))
    except Exception:
        pass

    it = 0
    while time.time() - t0 < EXPLORE_END:
        it += 1
        if time.time() - t0 > EXPLORE_END - 1.0:
            break
        if len(pop) < 2:
            break
        # pick parents
        a = int(rng.randint(len(pop)))
        b = int(rng.randint(len(pop)))
        if len(pop) > 1:
            while b == a:
                b = int(rng.randint(len(pop)))
        A = pop[a]["x"]; B = pop[b]["x"]
        try:
            Ba = _procrustes_align(A, B)
            U = np.vstack([A, Ba])
            # randomly shuffle union slightly? no
            child = _greedy_from_union(U, nsel=n)
            child = _mutate(child, rng)
            child = _norm(child)
        except Exception:
            continue
        # prescreen cheap softmax steps
        remain = EXPLORE_END - (time.time() - t0)
        if remain < 2:
            break
        steps = 250 if remain > 30 else 150
        try:
            child_s = _smooth(child, steps=steps)
        except Exception:
            continue
        v = _max_cos(child_s)
        # diversity check
        sig = _contact_sig(child_s, v)
        best_sig = _contact_sig(x_best, best_v)
        ddiv = _sig_dist(sig, best_sig)
        # also min dist to pop
        mind = min(_sig_dist(sig, e["sig"]) for e in pop) if pop else 1e9
        improved = v < best_v - 1e-12
        diverse = (v < best_v + 0.004) and (mind > 18)
        if improved:
            best_v = v
            x_best = child_s.copy()
            add_to_pop(child_s, v)
            try:
                np.save("best.npy", _norm(x_best))
            except Exception:
                pass
        elif diverse:
            add_to_pop(child_s, v)
        # keep pop small (<=5): drop worst non-best
        if len(pop) > 5:
            # find worst index whose v is max and not global best
            vs = [e["v"] for e in pop]
            wi = int(np.argmax(vs))
            # don't drop if it is global best (shouldn't be worst)
            pop.pop(wi)
        # occasional injection of best mutant to avoid stagnation
        if it % 60 == 0 and (time.time() - t0) < EXPLORE_END - 5:
            try:
                xm2 = _mutate(x_best, rng)
                xm2 = _smooth(_norm(xm2), steps=150)
                vm2 = _max_cos(xm2)
                if vm2 < best_v - 1e-12:
                    best_v = vm2; x_best = xm2.copy()
                    add_to_pop(xm2, vm2)
                elif vm2 < best_v + 0.003:
                    # replace worst if diverse enough
                    s2 = _contact_sig(xm2, vm2)
                    md = min(_sig_dist(s2, e["sig"]) for e in pop)
                    if md > 18 and len(pop) < 6:
                        add_to_pop(xm2, vm2)
            except Exception:
                pass

    # LP-polish best one or two
    try:
        # pick top-2 by value
        pop_sorted = sorted(pop, key=lambda e: e["v"])[:2]
    except Exception:
        pop_sorted = [{"x": x_best, "v": best_v}]
    x_fin = x_best.copy()
    v_fin = best_v
    # allocate remaining time: leave 2s margin
    for k, e in enumerate(pop_sorted):
        rem_total = HARD_END - (time.time() - t0)
        if rem_total < 3:
            break
        secs = min(12.0, rem_total - 2.0) if k == 0 else min(6.0, rem_total - 2.0)
        if secs < 1.0:
            break
        try:
            xp = _lp_polish(e["x"], seconds=secs, t0=t0, budget=HARD_END - 1.0)
            vp = _max_cos(xp)
            if vp < v_fin:
                v_fin, x_fin = vp, xp.copy()
        except Exception:
            continue
    x_fin = _norm(x_fin)
    try:
        if v_fin <= best_v:
            np.save("best.npy", x_fin)
    except Exception:
        pass
    return x_fin
