import time
import numpy as np

def _max_cos(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def _norm(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return x / n

def _lp_polish(x, deadline, radius_init=1e-3, max_iters=200):
    from scipy.optimize import linprog
    from scipy.sparse import coo_matrix
    x = _norm(x)
    n, d = x.shape
    best = _max_cos(x)
    radius = radius_init
    iu0 = np.triu_indices(n, 1)
    it = 0
    while time.time() < deadline and radius > 1e-14 and it < max_iters:
        it += 1
        G = x @ x.T
        c = G[iu0]
        m = float(c.max())
        thresh = m - max(20.0 * radius, 1e-9)
        idx = np.where(c > thresh)[0]
        if idx.size == 0:
            break
        if idx.size > 2500:
            order = np.argpartition(c[idx], -2500)[-2500:]
            idx = idx[order]
        I = iu0[0][idx]; J = iu0[1][idx]
        csel = c[idx]
        p = idx.size
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cI = I[:, None] * d + np.arange(d)
        cJ = J[:, None] * d + np.arange(d)
        cols = np.concatenate([cI, cJ, np.full((p, 1), n * d)], axis=1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], axis=1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
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
            x = y; best = v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x, best

def _constrained_phase(x0, m0, target, banned, delta, radius, steps):
    from scipy.optimize import linprog
    from scipy.sparse import coo_matrix
    x = _norm(x0.copy())
    n, d = x.shape
    iu0 = np.triu_indices(n, 1)
    # map pair to flat index for exclusion: build dict via position? use set of (min,max)
    ta, tb = int(target[0]), int(target[1])
    ca, cb = int(banned[0]), int(banned[1])
    if ta > tb: ta, tb = tb, ta
    if ca > cb: ca, cb = cb, ca
    Ktop = 700
    for _ in range(steps):
        G = x @ x.T
        cfull = G[iu0]
        # indices sorted descending, take top Ktop
        if cfull.size > Ktop:
            part = np.argpartition(cfull, -Ktop)[-Ktop:]
            # sort part descending for determinism
            o = np.argsort(-cfull[part])
            top_idx = part[o]
        else:
            o = np.argsort(-cfull)
            top_idx = o
        I = iu0[0][top_idx]; J = iu0[1][top_idx]
        csel_full = cfull[top_idx]
        # exclude banned pair from t-set
        mask_keep = ~(((I == ca) & (J == cb)) | ((I == cb) & (J == ca)))
        # ensure target is in set (it should be within top K since slack small)
        # if target not in top, add it
        has_target = np.any((I[mask_keep] == ta) & (J[mask_keep] == tb)) or np.any((I[mask_keep] == tb) & (J[mask_keep] == ta))
        I2 = I[mask_keep]; J2 = J[mask_keep]; c2 = csel_full[mask_keep]
        if not has_target:
            # append target
            ct = float(x[ta] @ x[tb])
            I2 = np.append(I2, ta); J2 = np.append(J2, tb); c2 = np.append(c2, ct)
        p = I2.size
        # current c for target and banned
        c_tar = float(x[ta] @ x[tb])
        c_ban = float(x[ca] @ x[cb])
        # Build A_ub: p rows (t) + 2 rows (target LB, banned UB)
        # t-rows: xJ*dxI + xI*dxJ - t <= -c
        # target LB: -xTb*dxTa - xTa*dxTb <= c_tar - (m0-1e-7)
        # banned UB: xCb*dxCa + xCa*dxCb <= (m0-delta - c_ban)
        nv = n * d + 1
        P = p + 2
        # construct in coo lists via numpy
        # t rows part
        rows_t = np.repeat(np.arange(p), 2 * d + 1)
        cI = I2[:, None] * d + np.arange(d)
        cJ = J2[:, None] * d + np.arange(d)
        cols_t = np.concatenate([cI, cJ, np.full((p, 1), n * d)], axis=1).ravel()
        vals_t = np.concatenate([x[J2], x[I2], -np.ones((p, 1))], axis=1).ravel()
        b_t = -c2
        # extra rows
        # row p: target
        cols_e = []
        vals_e = []
        rows_e = []
        # target row index p
        for k in range(d):
            cols_e.append(ta * d + k); vals_e.append(-x[tb, k]); rows_e.append(p)
            cols_e.append(tb * d + k); vals_e.append(-x[ta, k]); rows_e.append(p)
        # t coeff 0 -> no entry
        b_e_tar = c_tar - (m0 - 1e-7)
        # row p+1: banned
        for k in range(d):
            cols_e.append(ca * d + k); vals_e.append(x[cb, k]); rows_e.append(p + 1)
            cols_e.append(cb * d + k); vals_e.append(x[ca, k]); rows_e.append(p + 1)
        b_e_ban = (m0 - delta - c_ban)
        rows_all = np.concatenate([rows_t, np.array(rows_e, dtype=np.int64)])
        cols_all = np.concatenate([cols_t, np.array(cols_e, dtype=np.int64)])
        vals_all = np.concatenate([vals_t, np.array(vals_e, dtype=float)])
        b_all = np.concatenate([b_t, np.array([b_e_tar, b_e_ban])])
        A_ub = coo_matrix((vals_all, (rows_all, cols_all)), shape=(P, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=b_all, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            break
        if res.status != 0:
            break
        dx = res.x[:-1].reshape(n, d)
        # limit just in case
        x = x + dx
        x = _norm(x)
    return x

def solve():
    import time as _time
    t0 = _time.time()
    HARD_END = t0 + 175.0
    n, d = 153, 4
    best_x = None; best_v = 1e9
    for fname in ("best.npy", "record.npy"):
        try:
            a = np.load(fname)
            if a.shape == (n, d):
                a = _norm(a)
                v = _max_cos(a)
                if v < best_v:
                    best_v = v; best_x = a.copy()
        except Exception:
            pass
    if best_x is None:
        rng0 = np.random.default_rng(0)
        best_x = rng0.standard_normal(size=(n, d)); best_x = _norm(best_x)
        best_v = _max_cos(best_x)
    m0 = best_v
    x0 = best_x.copy()
    # contact analysis
    G0 = x0 @ x0.T
    iu = np.triu_indices(n, 1)
    c0 = G0[iu]
    order_desc = np.argsort(-c0)
    c_sorted = c0[order_desc]
    # contacts: slack < 1e-7
    Nc = int(np.count_nonzero(c_sorted > m0 - 1e-7))
    if Nc < 100: Nc = 450
    if Nc > 900: Nc = 450
    contact_idx = order_desc[:Nc]
    contact_pairs = [(int(iu[0][k]), int(iu[1][k])) for k in contact_idx]
    contact_set = set()
    for a, b in contact_pairs:
        if a > b: a, b = b, a
        contact_set.add((a, b))
    deg = np.zeros(n, dtype=int)
    # adjacency for quick lookup
    adj = [set() for _ in range(n)]
    for a, b in contact_pairs:
        deg[a] += 1; deg[b] += 1
        adj[a].add(b); adj[b].add(a)
    # high-degree nodes (degree-8)
    hi_nodes = np.where(deg >= 7)[0]
    if hi_nodes.size == 0:
        hi_nodes = np.argsort(-deg)[:10]
    # candidate near-contacts ranked ~450-550 : take slice Nc:Nc+120
    start = Nc  # ~450
    cand_slice = order_desc[start:start + 130]
    candidates = []
    for k in cand_slice:
        a = int(iu[0][k]); b = int(iu[1][k])
        slack = m0 - float(c0[k])
        if slack < 1e-9: continue
        if slack > 0.01: continue
        candidates.append((a, b, slack))
    # sort by slack ascending (they already are)
    # Build swap list >=40
    swaps = []
    deltas = [1e-4, 3e-4, 1e-3]
    radii = [0.01, 0.02, 0.03]
    # precompute a high-degree edge pool
    hi_edges = []
    for h in hi_nodes:
        for nb in sorted(adj[h]):
            a2, b2 = (h, nb) if h < nb else (nb, h)
            hi_edges.append((a2, b2))
    if len(hi_edges) == 0 and len(contact_pairs) > 0:
        hi_edges = contact_pairs[:10]
    si = 0
    for ci, (a, b, sl) in enumerate(candidates):
        # banned sharing a
        found = False
        # contacts of a
        if len(adj[a]) > 0:
            # pick neighbor deterministically: smallest index
            nb = sorted(adj[a])[ci % len(adj[a])]
            ba, bb = (a, nb) if a < nb else (nb, a)
            swaps.append(((a, b), (ba, bb), deltas[si % 3], radii[si % 3]))
            si += 1; found = True
        if len(adj[b]) > 0 and len(swaps) < 80:
            nb2 = sorted(adj[b])[(ci + 1) % len(adj[b])]
            ba2, bb2 = (b, nb2) if b < nb2 else (nb2, b)
            # avoid duplicate of previous
            if not (ba2 == swaps[-1][1][0] and bb2 == swaps[-1][1][1] and found and len(adj[a])==1):
                # alternate: only add second edge for some candidates to keep count manageable
                if ci % 2 == 0:
                    swaps.append(((a, b), (ba2, bb2), deltas[si % 3], radii[(si + 1) % 3]))
                    si += 1
        # occasionally add hi-degree unrelated edge
        if ci % 5 == 4 and len(hi_edges) > 0:
            he = hi_edges[(ci) % len(hi_edges)]
            # ensure he != target and not sharing? allow it
            swaps.append(((a, b), he, deltas[si % 3], radii[si % 3]))
            si += 1
        if len(swaps) >= 60:
            break
    # ensure at least 40
    if len(swaps) < 40:
        # pad with hi edges
        for k in range(40 - len(swaps)):
            tgt = candidates[k % len(candidates)][:2]
            he = hi_edges[k % len(hi_edges)]
            swaps.append((tgt, he, deltas[k % 3], radii[k % 3]))
    # truncate to fit time: aim 40-55 swaps
    if len(swaps) > 55:
        swaps = swaps[:55]

    global_best_x = x0.copy(); global_best_v = m0
    # quick record of topology-changed candidates
    changed = []
    NCONST = 22
    for sidx, (tgt, ban, delta, rad) in enumerate(swaps):
        if _time.time() > HARD_END - 30.0:
            break
        # constrained phase
        try:
            xc = _constrained_phase(x0, m0, tgt, ban, delta, rad, NCONST)
        except Exception:
            continue
        # unconstrained polish short
        dl = min(_time.time() + 0.9, HARD_END - 25.0)
        if dl <= _time.time() + 0.15:
            break
        try:
            xp, vp = _lp_polish(xc, dl, radius_init=1e-3, max_iters=60)
        except Exception:
            continue
        # check topology criteria
        ta, tb = tgt; ca2, cb2 = ban
        c_tar_new = float(xp[ta] @ xp[tb])
        c_ban_new = float(xp[ca2] @ xp[cb2])
        slack_tar = vp - c_tar_new
        slack_ban = vp - c_ban_new
        kept = (slack_tar < 1e-6 + 1e-9) and (slack_ban > 1e-6)
        if kept:
            changed.append((vp, sidx))
        if vp < global_best_v:
            global_best_v = vp; global_best_x = xp.copy()
        # also if kept but not better, we could do a tiny extra polish later for best few
        # early exit if improved meaningfully
        if global_best_v < m0 - 1e-7:
            # found real improvement, continue to polish it but keep searching briefly
            pass
    # extra polishing of best changed ones if no improvement yet: pick lowest few vp among changed? we already tracked global best.
    # Final thorough polish of global best
    try:
        dl2 = HARD_END - 1.0
        if dl2 > _time.time() + 0.5:
            global_best_x, global_best_v = _lp_polish(global_best_x, dl2, radius_init=1e-3, max_iters=300)
    except Exception:
        pass
    # safety: if global best worse than original (shouldn't happen since we init with m0), revert
    if global_best_v > m0 + 1e-12:
        global_best_x = x0.copy()
    global_best_x = _norm(global_best_x)
    return global_best_x
