import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _contact_mask(x, m, tol=1e-9):
    g = x @ x.T
    iu0, iu1 = np.triu_indices(x.shape[0], 1)
    c = g[iu0, iu1]
    return iu0, iu1, c, (c > m - tol)

def _lp_polish(x, deadline, r0=1e-3):
    x = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
    n, d = x.shape
    try:
        best = _max_cos(x)
    except Exception:
        return x, 1.0
    radius = r0
    iu0, iu1 = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        try:
            G = x @ x.T
            c = G[iu0, iu1]
            m = float(c.max())
            sel = c > m - max(20.0 * radius, 1e-9)
            if np.count_nonzero(sel) == 0:
                sel[int(np.argmax(c))] = True
            I = iu0[sel]; J = iu1[sel]
            csel = c[sel]
            p = len(I)
            nv = n * d + 1
            rows = np.repeat(np.arange(p), 2 * d + 1)
            cols = np.empty((p, 2 * d + 1), dtype=int)
            ar = np.arange(d)
            cols[:, :d] = I[:, None] * d + ar
            cols[:, d:2 * d] = J[:, None] * d + ar
            cols[:, 2 * d] = n * d
            cols = cols.ravel()
            vals = np.empty((p, 2 * d + 1), dtype=float)
            vals[:, :d] = x[J]
            vals[:, d:2 * d] = x[I]
            vals[:, 2 * d] = -1.0
            vals = vals.ravel()
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv); obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
            if res.status != 0:
                radius *= 0.5
                continue
            y = x + res.x[:-1].reshape(n, d)
            y /= np.maximum(np.linalg.norm(y, axis=1, keepdims=True), 1e-12)
            v = _max_cos(y)
            if v < best:
                x, best = y, v
                radius = min(radius * 1.5, 0.05)
            else:
                radius *= 0.5
        except Exception:
            radius *= 0.5
            continue
    return x, best

def _anneal(x0, steps, T0, T1, lr0, mu, rng, active=None, taboo=None, thr=0.0, pen=25.0, noise0=0.0, deadline=None):
    x = x0.copy()
    n, d = x.shape
    if active is None:
        act = np.ones(n, dtype=bool)
    else:
        act = np.asarray(active, dtype=bool)
    vel = np.zeros_like(x)
    G0 = x @ x.T
    np.fill_diagonal(G0, -2.0)
    bv = float(G0.max())
    bx = x.copy()
    if T0 <= 0 or T1 <= 0:
        T0 = max(T0, 1e-12); T1 = max(T1, 1e-12)
    lratio = np.log(T1 / T0)
    for k in range(steps):
        if deadline is not None and (k & 127) == 0 and time.time() > deadline:
            break
        frac = k / max(1, steps - 1)
        T = T0 * np.exp(lratio * frac)
        if T < 5e-5:
            T = 5e-5
        lr = lr0 * (1.0 - 0.55 * frac)
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        E = np.exp((G - m) / T)
        np.fill_diagonal(E, 0.0)
        if taboo is not None:
            try:
                boost = taboo & (G > thr)
                if np.any(boost):
                    E[boost] *= pen
            except Exception:
                pass
        S = float(E.sum() * 0.5)
        if not np.isfinite(S) or S < 1e-300:
            S = 1e-300
        Grad = (E @ x) / S
        Grad -= (Grad * x).sum(axis=1, keepdims=True) * x
        if not np.all(act):
            Grad[~act] = 0.0
        vel = mu * vel + (1.0 - mu) * Grad
        if not np.all(act):
            vel[~act] = 0.0
        x[act] = x[act] - lr * vel[act]
        if noise0 > 0:
            amp = noise0 * (1.0 - frac)
            if amp > 0:
                na = int(act.sum())
                nz = rng.randn(na, d) * amp
                xa = x[act]
                nz -= (nz * xa).sum(axis=1, keepdims=True) * xa
                x[act] = xa + nz
        xn = np.linalg.norm(x[act], axis=1, keepdims=True)
        x[act] /= np.maximum(xn, 1e-12)
        if m < bv:
            bv = m
            bx = x.copy()
    return bx, bv

def _greedy_holes(x_fixed, k, rng, n_samples=12000):
    cur_list = [x_fixed.copy()]
    # stacked fixed for scoring
    stack = x_fixed.copy()
    out = np.zeros((k, 4))
    for s in range(k):
        Y = rng.randn(n_samples, 4)
        Y /= np.linalg.norm(Y, axis=1, keepdims=True)
        D = Y @ stack.T
        vals = D.max(axis=1)
        idx = int(np.argmin(vals))
        h = Y[idx] + rng.randn(4) * 0.008
        h /= max(np.linalg.norm(h), 1e-12)
        out[s] = h
        stack = np.vstack([stack, h[None, :]])
    return out

def _build_cluster(n, hub, neigh_list, deg_order_asc, G_row_dot, k_target, rng, mode):
    # neigh_list: contact neighbours of hub (list)
    C = [int(hub)]
    # add contact neighbours (shuffled somewhat by degree)
    nl = list(neigh_list)
    if len(nl) > 0:
        # prioritize high-degree neighbours to remove jam, but keep deterministic shuffle
        perm = rng.permutation(len(nl))
        nl = [nl[i] for i in perm]
        for v in nl:
            if len(C) >= k_target:
                break
            if v not in C:
                C.append(int(v))
    # optionally add one low-contact boundary point nearest to hub among low-degree
    if mode % 3 == 1 and len(C) < k_target:
        # pick lowest-contact point with highest dot to hub (spatially near but loose)
        for cand in deg_order_asc:
            cand = int(cand)
            if cand not in C:
                # prefer near hub: check dot
                C.append(cand)
                break
    # BFS second hop via contact graph if still short
    # build adjacency quickly outside? fallback to spatial nearest
    if len(C) < k_target:
        # spatial nearest to hub by dot
        order = np.argsort(-G_row_dot)  # descending dot
        for cand in order:
            cand = int(cand)
            if cand == hub or cand in C:
                continue
            C.append(cand)
            if len(C) >= k_target:
                break
    # if mode spatial: replace tail with spatially localised cap
    if mode % 3 == 2:
        # keep hub, take k_target-1 Euclidean nearest
        order = np.argsort(-G_row_dot)
        C2 = [int(hub)]
        for cand in order:
            cand = int(cand)
            if cand == hub:
                continue
            C2.append(cand)
            if len(C2) >= k_target:
                break
        C = C2
    return np.array(C[:k_target], dtype=int)

def solve():
    t0 = time.time()
    BUDGET = 172.0
    SEARCH_END = t0 + 132.0
    rng = np.random.RandomState(12345)
    try:
        bx = np.load("best.npy")
    except Exception:
        try:
            bx = np.load("record.npy")
        except Exception:
            bx = rng.randn(163, 4)
    bx = np.asarray(bx, dtype=float)
    if bx.shape != (163, 4) or not np.all(np.isfinite(bx)):
        bx = rng.randn(163, 4)
    bx = bx / np.maximum(np.linalg.norm(bx, axis=1, keepdims=True), 1e-12)
    n = 163
    best_x = bx.copy()
    best_val = _max_cos(best_x)
    # quick contact snapshot for incumbent
    G0 = best_x @ best_x.T
    iu0, iu1 = np.triu_indices(n, 1)
    c0 = G0[iu0, iu1]
    best_mask = (c0 > best_val - 1e-9)
    trial = 0
    k_cycle = [8, 11, 14, 17, 20, 12, 16, 9]
    while time.time() < SEARCH_END:
        remain = SEARCH_END - time.time()
        if remain < 2.0:
            break
        # recompute degrees of incumbent
        G = best_x @ best_x.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        # contact adjacency with 1e-9
        cadj = (G > m - 1e-9)
        deg = cadj.sum(axis=1)
        order_desc = np.argsort(deg)  # asc
        order_asc = order_desc  # low first
        # choose hub: alternate high hub / low centre / random spatial
        mode = trial % 3
        if mode == 0:
            # hub-centred: cycle through highest-degree
            hi = np.argsort(-deg)
            hub = int(hi[(trial // 3) % n])
        elif mode == 1:
            # low-contact-centred
            lo = np.argsort(deg)
            hub = int(lo[(trial // 3) % n])
        else:
            hub = int(rng.randint(0, n))
        k_target = int(k_cycle[trial % len(k_cycle)])
        # neighbours of hub
        neigh = list(np.where(cadj[hub])[0])
        # true dot row for spatial (use best_x dots, diagonal excluded)
        dotrow = (best_x @ best_x[hub])
        dotrow[hub] = -2.0
        C = _build_cluster(n, hub, neigh, order_asc, dotrow, k_target, rng, mode)
        act = np.zeros(n, dtype=bool)
        act[C] = True
        # taboo: old contacts involving C above m-1e-4
        Gold = best_x @ best_x.T
        thr_taboo = m - 1e-4
        taboo = (Gold > thr_taboo)
        # require at least one endpoint in C
        inC = np.zeros(n, dtype=bool); inC[C] = True
        taboo = taboo & (inC[:, None] | inC[None, :])
        np.fill_diagonal(taboo, False)
        # freeze remaining
        fixed_idx = np.where(~act)[0]
        x_fixed = best_x[fixed_idx].copy()
        # maximin hole placement for cluster
        try:
            holes = _greedy_holes(x_fixed, len(C), rng, n_samples=12000)
        except Exception:
            holes = rng.randn(len(C), 4)
            holes /= np.maximum(np.linalg.norm(holes, axis=1, keepdims=True), 1e-12)
        cur = best_x.copy()
        cur[C] = holes
        # cluster-only anneal with taboo penalty
        dl = min(SEARCH_END - 0.5, time.time() + 2.2)
        # vary temperature slightly by trial
        T0 = 0.012 + 0.004 * ((trial % 4) / 3.0)
        try:
            cur_c, _ = _anneal(cur, steps=2200, T0=T0, T1=0.0009, lr0=0.035, mu=0.85,
                               rng=rng, active=act, taboo=taboo, thr=thr_taboo, pen=25.0,
                               noise0=0.002, deadline=dl)
        except Exception:
            trial += 1
            continue
        # short joint annealing (all points, no taboo, low T)
        try:
            cur_j, _ = _anneal(cur_c, steps=1200, T0=0.004, T1=0.0006, lr0=0.022, mu=0.88,
                               rng=rng, active=None, taboo=None, noise0=0.0, deadline=dl)
        except Exception:
            cur_j = cur_c
        vj = _max_cos(cur_j)
        # only LP-polish if promising (within 3e-3 of best or better) to save time
        if vj < best_val + 0.003:
            lp_dead = min(time.time() + 1.6, SEARCH_END)
            try:
                pol, vp = _lp_polish(cur_j, deadline=lp_dead, r0=8e-4)
            except Exception:
                pol, vp = cur_j, vj
            # edit distance of contact graph
            try:
                Gp = pol @ pol.T
                cp = Gp[iu0, iu1]
                mask_p = (cp > vp - 1e-9)
                # hamming distance
                edit = int(np.count_nonzero(mask_p != best_mask))
            except Exception:
                edit = 999
            if vp < best_val - 1e-12 and edit >= 15:
                best_x, best_val = pol, vp
                best_mask = mask_p
            elif vp < best_val - 5e-7 and edit >= 8:
                best_x, best_val = pol, vp
                best_mask = mask_p
            # else discard to enforce topology change
        trial += 1
        if time.time() > t0 + BUDGET - 34.0:
            break
    # final polish with remaining time
    try:
        fin_dead = t0 + BUDGET - 1.5
        best_x, best_val = _lp_polish(best_x, deadline=fin_dead, r0=1e-3)
    except Exception:
        pass
    best_x = best_x / np.maximum(np.linalg.norm(best_x, axis=1, keepdims=True), 1e-12)
    return best_x
