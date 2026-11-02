import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _norm(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return x / n

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(np.max(g))

def _kmeans_rows(Xn, k, rng, iters=60):
    n = Xn.shape[0]
    # kmeans++ like init: random choice
    idx = rng.choice(n, size=k, replace=False)
    C = Xn[idx].copy()
    labels = np.zeros(n, dtype=int)
    for _ in range(iters):
        d2 = ((Xn[:, None, :] - C[None, :, :]) ** 2).sum(axis=2)
        nl = np.argmin(d2, axis=1)
        if np.array_equal(nl, labels) and _ > 0:
            labels = nl
            break
        labels = nl
        for c in range(k):
            m = labels == c
            if np.any(m):
                C[c] = Xn[m].mean(axis=0)
            else:
                C[c] = Xn[rng.integers(0, n)]
    return labels

def _spectral_labels(A, k, seed):
    n = A.shape[0]
    rng = np.random.default_rng(seed)
    deg = A.sum(axis=1)
    di = 1.0 / np.sqrt(np.maximum(deg, 1e-12))
    # normalized adjacency -> take top k eigenvectors (community)
    An = di[:, None] * A * di[None, :]
    try:
        vals, vecs = np.linalg.eigh(An)
    except Exception:
        return rng.integers(0, k, size=n)
    # largest k
    Xemb = vecs[:, -k:]
    nr = np.linalg.norm(Xemb, axis=1, keepdims=True) + 1e-12
    Xn = Xemb / nr
    return _kmeans_rows(Xn, k, rng, iters=80)

def _greedy_clique_cover(A, k, blksz):
    # A: non-contact adjacency (1 if intra-candidate), symmetric, 0 diag
    n = A.shape[0]
    rng = np.random.default_rng(0)
    unassigned = set(range(n))
    labels = np.full(n, -1, dtype=int)
    # order seeds by degree
    for b in range(k):
        if not unassigned:
            break
        # seed: max degree within unassigned
        best_s = -1
        best_d = -1
        for s in list(unassigned):
            d = sum(1 for u in unassigned if u != s and A[s, u] > 0.5)
            if d > best_d:
                best_d = d
                best_s = s
        clique = [best_s]
        in_c = {best_s}
        # candidates connected to all in clique
        while len(clique) < blksz:
            cands = []
            for u in list(unassigned):
                if u in in_c:
                    continue
                ok = True
                for c in clique:
                    if A[u, c] < 0.5:
                        ok = False
                        break
                if ok:
                    # score by connections to unassigned
                    sc = sum(1 for w in unassigned if w != u and A[u, w] > 0.5)
                    cands.append((sc, u))
            if not cands:
                break
            cands.sort(reverse=True)
            # pick best (deterministic, small random tie)
            chosen = cands[0][1]
            clique.append(chosen)
            in_c.add(chosen)
        # if clique too small, fill with most compatible (most non-contacts to clique)
        while len(clique) < blksz and unassigned:
            best_u = -1
            best_sc = -1
            for u in list(unassigned):
                if u in in_c:
                    continue
                sc = sum(A[u, c] for c in clique)
                if sc > best_sc:
                    best_sc = sc
                    best_u = u
            if best_u < 0:
                break
            clique.append(best_u)
            in_c.add(best_u)
        for u in clique:
            if u in unassigned:
                labels[u] = b
                unassigned.discard(u)
        # if leftover and last blocks, continue
    # assign any leftover (if sizes unbalanced) to nearest block by affinity
    if np.any(labels < 0):
        for i in np.where(labels < 0)[0]:
            # assign to smallest block
            cnts = [np.sum(labels == b) for b in range(k)]
            b = int(np.argmin(cnts))
            labels[i] = b
    return labels

def _partition_score(G, gmax, labels, gap_thresh=1e-3):
    n = len(labels)
    iu0, iu1 = np.triu_indices(n, 1)
    same = labels[iu0] == labels[iu1]
    # contact indicator with slack gap_thresh
    is_contact = (G[iu0, iu1] > gmax - gap_thresh)
    # want same => non-contact, diff => contact
    inter = ~same
    intra = same
    inter_hit = np.mean(is_contact[inter]) if np.any(inter) else 0
    intra_hit = np.mean(is_contact[intra]) if np.any(intra) else 1
    # balance penalty
    _, counts = np.unique(labels, return_counts=True)
    bal = np.std(counts)
    # gap of means
    inter_mean = np.mean(G[iu0[inter], iu1[inter]]) if np.any(inter) else 0
    intra_mean = np.mean(G[iu0[intra], iu1[intra]]) if np.any(intra) else 0
    gap = inter_mean - intra_mean
    return (inter_hit - intra_hit, gap, -bal)

def _reassign_labels(G, labels, k, max_blksz=12):
    n = len(labels)
    # 2 sweeps: assign each point to block with minimal mean dot
    for _ in range(3):
        changed = False
        order = np.arange(n)  # deterministic
        for i in order:
            scores = np.zeros(k)
            for b in range(k):
                members = np.where((labels == b) & (np.arange(n) != i))[0]
                if len(members) == 0:
                    scores[b] = 1e9
                else:
                    scores[b] = np.mean(G[i, members])
            rank = np.argsort(scores)
            cur = labels[i]
            for b in rank:
                b = int(b)
                if b == cur:
                    break
                # size guard
                if np.sum(labels == b) >= max_blksz:
                    continue
                # must improve by margin
                members_cur = np.where((labels == cur) & (np.arange(n) != i))[0]
                cur_sc = np.mean(G[i, members_cur]) if len(members_cur) > 0 else 1e9
                if scores[b] < cur_sc - 1e-4:
                    labels[i] = b
                    changed = True
                break
        if not changed:
            break
    return labels

def _block_anneal(X0, labels0, steps, delta0, weq0, w1, w2, lr_x, lr_t, seed, reassign_every, track_best):
    # track_best: dict with 'val' and 'X'
    rng = np.random.default_rng(seed)
    n, d = X0.shape
    k = int(labels0.max() + 1)
    X = X0.copy()
    labels = labels0.copy()
    # init t as max inter
    Gtmp = X @ X.T
    iu0, iu1 = np.triu_indices(n, 1)
    same0 = labels[iu0] == labels[iu1]
    inter_max = float(np.max(Gtmp[iu0[~same0], iu1[~same0]])) if np.any(~same0) else float(np.max(Gtmp[iu0, iu1]))
    t = inter_max
    mX = np.zeros_like(X)
    vX = np.zeros_like(X)
    mt = 0.0
    vt = 0.0
    b1, b2, eps = 0.9, 0.999, 1e-8
    iu0_all, iu1_all = iu0, iu1
    N_inter_total = None
    for s in range(steps):
        f = s / max(steps - 1, 1)
        delta = delta0 * (1.0 - f)
        weq = weq0 * (1.0 - f)
        # periodic reassign
        if reassign_every > 0 and s > 0 and (s % reassign_every == 0):
            Gcur = X @ X.T
            labels = _reassign_labels(Gcur, labels.copy(), k, max_blksz=12)
        same = labels[iu0_all] == labels[iu1_all]
        Ninter = int(np.sum(~same))
        Nintra = int(np.sum(same))
        if Ninter == 0 or Nintra == 0:
            break
        G = X @ X.T
        # gather
        inter_vals = G[iu0_all[~same], iu1_all[~same]]
        intra_vals = G[iu0_all[same], iu1_all[same]]
        cur_max_all = float(np.max(G[iu0_all, iu1_all]))
        if cur_max_all < track_best['val']:
            track_best['val'] = cur_max_all
            track_best['X'] = X.copy()
        m_inter = float(np.mean(inter_vals))
        # violations
        ri = np.maximum(inter_vals - t, 0.0)
        rj = np.maximum(intra_vals - t + delta, 0.0)
        mean_ri = float(np.mean(ri)) if ri.size else 0.0
        mean_rj = float(np.mean(rj)) if rj.size else 0.0
        # S matrix
        # grad coeffs
        ci = (w1 * 2.0 * ri / Ninter) + (weq * 2.0 * (inter_vals - m_inter) / Ninter)
        cj = (w2 * 2.0 * rj / Nintra)
        S = np.zeros((n, n), dtype=float)
        S[iu0_all[~same], iu1_all[~same]] = ci
        S[iu1_all[~same], iu0_all[~same]] = ci
        S[iu0_all[same], iu1_all[same]] = cj
        S[iu1_all[same], iu0_all[same]] = cj
        grad_X = 2.0 * (S @ X)
        # tangent project
        dot = np.sum(grad_X * X, axis=1, keepdims=True)
        grad_Xt = grad_X - dot * X
        # Adam for X
        mX = b1 * mX + (1 - b1) * grad_Xt
        vX = b2 * vX + (1 - b2) * (grad_Xt * grad_Xt)
        bc1 = 1 - b1 ** (s + 1)
        bc2 = 1 - b2 ** (s + 1)
        mhat = mX / bc1
        vhat = vX / bc2
        # lr schedule decay
        lr = lr_x * (1.0 - 0.7 * f)
        X = X - lr * mhat / (np.sqrt(vhat) + eps)
        X = _norm(X)
        # t update
        gt = 1.0 - 2.0 * w1 * mean_ri - 2.0 * w2 * mean_rj
        # clip gt to avoid explosion
        gt = float(np.clip(gt, -5.0, 5.0))
        mt = b1 * mt + (1 - b1) * gt
        vt = b2 * vt + (1 - b2) * gt * gt
        mthat = mt / bc1
        vthat = vt / bc2
        t = t - lr_t * mthat / (np.sqrt(vthat) + eps)
        # keep t in sane range
        if t < -0.1:
            t = -0.1
        if t > 0.2:
            t = 0.2
    return labels

def _lp_polish(x, deadline, r0=0.015, rmax=0.05):
    n, d = x.shape
    x = _norm(x.copy())
    def mc(a):
        g = a @ a.T
        np.fill_diagonal(g, -2.0)
        return float(np.max(g))
    best = mc(x)
    best_x = x.copy()
    radius = r0
    iu0, iu1 = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-13:
        G = x @ x.T
        c_all = G[iu0, iu1]
        m = float(c_all.max())
        thresh = m - max(20.0 * radius, 1e-9)
        sel = np.where(c_all > thresh)[0]
        if sel.size > 1500:
            part = np.argpartition(c_all, -1500)[-1500:]
            sel = part
        c_sel = c_all[sel]
        I = iu0[sel]
        J = iu1[sel]
        p = sel.size
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.empty(p * (2 * d + 1), dtype=np.int64)
        vals = np.empty(p * (2 * d + 1), dtype=np.float64)
        vI = x[J]
        vJ = x[I]
        ar = np.arange(d)
        for kk in range(p):
            base = kk * (2 * d + 1)
            cols[base:base + d] = I[kk] * d + ar
            vals[base:base + d] = vI[kk]
            cols[base + d:base + 2 * d] = J[kk] * d + ar
            vals[base + d:base + 2 * d] = vJ[kk]
            cols[base + 2 * d] = n * d
            vals[base + 2 * d] = -1.0
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        erows = np.repeat(np.arange(n), d)
        ecols = np.arange(n * d)
        A_eq = coo_matrix((x.ravel(), (erows, ecols)), shape=(n, nv)).tocsr()
        obj = np.zeros(nv)
        obj[-1] = 1.0
        bounds = [(-radius, radius)] * (n * d) + [(None, None)]
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c_sel, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=bounds, method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y = _norm(y)
        v = mc(y)
        if v < best:
            best = v
            best_x = y.copy()
            x = y
            radius = min(radius * 1.5, rmax)
        else:
            radius *= 0.5
            if radius < 1e-7:
                x = best_x.copy()
        if time.time() >= deadline:
            break
    return best_x, best

def solve():
    t0 = time.time()
    T_HARD = t0 + 176.0
    T_BLOCK_END = t0 + 150.0
    T_LP = 19.0
    try:
        xb = np.load("best.npy").astype(float)
    except Exception:
        xb = np.load("record.npy").astype(float)
    try:
        xr = np.load("record.npy").astype(float)
        xr = _norm(xr)
        xb = _norm(xb)
        if _max_cos(xr) < _max_cos(xb):
            xb = xr.copy()
    except Exception:
        xb = _norm(xb)
    n, d = xb.shape
    gmax0 = _max_cos(xb)
    G0 = xb @ xb.T
    # --- extract block structure: try thresholds ---
    thresholds = [1e-3, 3e-3, 1e-2]
    cand_labels = []
    # spectral candidates
    for th in thresholds:
        A = np.zeros((n, n), dtype=float)
        mask = (G0 < gmax0 - th)
        np.fill_diagonal(mask, False)
        A[mask] = 1.0
        # symmetrize (already sym)
        for seed in [0, 1]:
            try:
                lab = _spectral_labels(A, 6, seed)
                cand_labels.append(lab)
            except Exception:
                pass
        try:
            lab2 = _greedy_clique_cover(A, 6, 9)
            cand_labels.append(lab2)
        except Exception:
            pass
    # also try complement of contact graph at tight threshold for clique
    # rank candidates
    best_key = None
    best_lab = None
    for lab in cand_labels:
        # fix empty clusters: if fewer than 6 unique, skip
        u = np.unique(lab)
        if len(u) < 5:
            continue
        # remap to 0..k-1
        mp = {v: i for i, v in enumerate(u)}
        lab2 = np.array([mp[v] for v in lab], dtype=int)
        k = len(u)
        # if k !=6, allow but prefer 6
        key = _partition_score(G0, gmax0, lab2)
        # key tuple: (inter_hit-intra_hit, gap, -bal); add bonus for k==6
        bonus = 1.0 if k == 6 else 0.0
        tkey = (key[0] + 0.1 * bonus, key[1], key[2])
        if best_key is None or tkey > best_key:
            best_key = tkey
            best_lab = lab2
    if best_lab is None:
        # fallback: random balanced
        best_lab = np.repeat(np.arange(6), 9)
        np.random.default_rng(0).shuffle(best_lab)
    # ensure 6 labels 0..5 (if k!=6, force kmeans 6 on G0 rows)
    if len(np.unique(best_lab)) != 6:
        # force spectral on middle thresh
        A = np.zeros((n, n), dtype=float)
        mask = (G0 < gmax0 - 3e-3)
        np.fill_diagonal(mask, False)
        A[mask] = 1.0
        best_lab = _spectral_labels(A, 6, 0)
    # rebalance to ~9 each via size-constrained reassign starting from best_lab
    # quick balance: while any block >11, move its worst point
    for _ in range(20):
        cnts = [np.sum(best_lab == b) for b in range(6)]
        bmax = int(np.argmax(cnts))
        if cnts[bmax] <= 10:
            break
        members = np.where(best_lab == bmax)[0]
        # point with highest mean to block -> most misplaced
        scores = []
        for i in members:
            oth = [j for j in members if j != i]
            scores.append(np.mean(G0[i, oth]) if oth else 0)
        worst = members[int(np.argmax(scores))]
        # move to smallest block where it fits best
        cnts2 = cnts
        bmin_cands = np.argsort(cnts2)
        moved = False
        for b in bmin_cands:
            if b == bmax:
                continue
            mem2 = np.where(best_lab == b)[0]
            sc = np.mean(G0[worst, mem2]) if len(mem2) else 0
            # move if beneficial (small dot)
            best_lab[worst] = int(b)
            moved = True
            break
        if not moved:
            break
    track = {'val': gmax0, 'X': xb.copy()}
    # --- block-anneal phase ---
    # create perturbed starts around xb to allow topology moves
    rng0 = np.random.default_rng(7)
    # run 1: from xb directly, delta 0.008
    # run 2: small noise, delta 0.004, etc., time permitting
    configs = [
        (0.0, 0.008, 1.0, 6000, 0.02, 0.002, 2000),
        (0.004, 0.004, 0.5, 5000, 0.015, 0.002, 1500),
        (0.008, 0.002, 0.3, 4000, 0.012, 0.001, 1000),
    ]
    lab_cur = best_lab.copy()
    for ci, (sig, d0, weq0, steps, lrx, lrt, revery) in enumerate(configs):
        if time.time() > T_BLOCK_END - 5:
            break
        if sig == 0.0:
            X0 = track['X'].copy()
        else:
            X0 = track['X'] + sig * rng0.standard_normal(size=(n, d))
            X0 = _norm(X0)
        # w1,w2
        w1 = 4000.0
        w2 = 4000.0
        # adapt steps to remaining time
        rem = T_BLOCK_END - time.time()
        # rough: ~2000 steps per second? actually much faster; keep as is
        if rem < 10:
            steps = min(steps, 1500)
        try:
            lab_cur = _block_anneal(X0, lab_cur, steps, d0, weq0, w1, w2, lrx, lrt, 100 + ci, revery, track)
        except Exception:
            continue
    x_best = track['X'].copy()
    best_val = track['val']
    # small unconstrained soft-max polish (fast gradient) to release ansatz, time permitting
    # simple tangential repulsion on top pairs for ~1000 steps
    try:
        if time.time() < T_BLOCK_END:
            X = x_best.copy()
            V = np.zeros_like(X)
            mu = 0.8
            steps2 = 800
            for s in range(steps2):
                if time.time() > T_BLOCK_END:
                    break
                f = s / max(steps2 - 1, 1)
                ksoft = 800.0 + f * 1200.0
                G = X @ X.T
                np.fill_diagonal(G, -2.0)
                m = float(np.max(G))
                D = G - m
                # window
                w = 0.01 - f * 0.006
                sel = D > -w
                W = np.zeros_like(G)
                W[sel] = np.exp(ksoft * D[sel])
                np.fill_diagonal(W, 0.0)
                F = W @ X
                dot = np.sum(F * X, axis=1, keepdims=True)
                Ft = F - dot * X
                an = float(np.sqrt(np.sum(Ft * Ft) / n)) + 1e-12
                V = mu * V + Ft / an
                vn = float(np.sqrt(np.sum(V * V) / n)) + 1e-12)
                lr = 0.004 * (1 - 0.7 * f)
                X = X - lr * V / max(vn, 1.0)
                X = _norm(X)
                if s % 50 == 0:
                    Gc = X @ X.T
                    np.fill_diagonal(Gc, -2.0)
                    v = float(np.max(Gc))
                    if v < track['val']:
                        track['val'] = v
                        track['X'] = X.copy()
            x_best = track['X'].copy()
    except Exception:
        x_best = track['X'].copy()
    # --- final trust-region LP (<=20s) ---
    rem_total = T_HARD - time.time()
    lp_budget = min(T_LP, max(rem_total - 1.0, 3.0))
    if lp_budget > 2.0:
        try:
            xf, vf = _lp_polish(x_best, time.time() + lp_budget, r0=0.008, rmax=0.03)
            if vf < _max_cos(x_best):
                x_best = xf
        except Exception:
            pass
    x_best = _norm(x_best)
    return x_best
