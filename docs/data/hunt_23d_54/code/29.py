import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _lp_polish(x, seconds, t0, budget):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    bx = x.copy()
    radius = 1e-3
    start = time.time()
    iu0 = np.triu_indices(n, 1)
    it = 0
    while time.time() - start < seconds:
        if (time.time() - t0) > budget:
            break
        if radius < 1e-14:
            break
        it += 1
        if it > 40:
            break
        Gfull = x @ x.T
        c_all = Gfull[iu0]
        m = float(c_all.max())
        sel = c_all > m - max(20 * radius, 1e-9)
        if np.count_nonzero(sel) > 800:
            idx = np.where(sel)[0]
            order = np.argsort(c_all[idx])[::-1][:800]
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
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        v = _max_cos(y)
        if v < best:
            best = v
            bx = y.copy()
            x = y
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
            x = bx.copy()
            if radius < 5e-4:
                break
    return bx

def _smooth(x_init, steps, k0, k1, lr0, lr1, d0, d1):
    x = x_init.copy()
    n, d = x.shape
    iu = np.triu_indices(n, 1)
    for s in range(steps):
        frac = s / max(steps - 1, 1)
        k = k0 * (k1 / k0) ** frac
        lr = lr0 * (lr1 / lr0) ** frac
        G = x @ x.T
        m = float(G[iu].max())
        thr = m - (d0 + (d1 - d0) * frac)
        W = np.zeros_like(G)
        mask = G > thr
        np.fill_diagonal(mask, False)
        if np.any(mask):
            W[mask] = np.exp(k * (G[mask] - m))
            Gr = W @ x
            dot = np.sum(Gr * x, axis=1, keepdims=True)
            Gt = Gr - dot * x
            deg = np.sum(W, axis=1, keepdims=True) + 1.0
            x = x - lr * Gt / deg
            nr = np.linalg.norm(x, axis=1, keepdims=True)
            nr[nr == 0] = 1.0
            x /= nr
    return x

def _build_pool(rng, d=23):
    parts = []
    # +- e_i
    E = np.eye(d)
    parts.append(E)
    parts.append(-E)
    # +- e_i +- e_j / sqrt2
    v = []
    for i in range(d):
        for j in range(i+1, d):
            a = np.zeros(d); a[i]=1/np.sqrt(2); a[j]=1/np.sqrt(2); v.append(a)
            a2 = np.zeros(d); a2[i]=1/np.sqrt(2); a2[j]=-1/np.sqrt(2); v.append(a2)
    parts.append(np.array(v))
    parts.append(-np.array(v))
    # cyclic binary Golay (23,12,7) via QR circulant span
    try:
        residues = [1,2,3,4,6,8,9,12,13,16,18]
        v0 = 0
        for r in residues:
            v0 |= (1 << (r % 23))
        MASK = (1 << 23) - 1
        rows = []
        for i in range(23):
            # rotate left by i
            ri = ((v0 << i) | (v0 >> (23 - i))) & MASK if i != 0 else v0
            rows.append(ri)
        s = {0}
        for r in rows:
            s = s | {x ^ r for x in s}
            if len(s) > 5000:
                pass
        codes = list(s)[:8192]
        G = np.empty((len(codes), d), dtype=float)
        sq = np.sqrt(23.0)
        for ii, cc in enumerate(codes):
            for j in range(d):
                bit = (cc >> j) & 1
                G[ii, j] = (1.0 - 2.0*bit) / sq
        parts.append(G)
        parts.append(-G)
    except Exception:
        pass
    # Hadamard-24 rows with one coordinate deleted
    try:
        from scipy.linalg import hadamard as _had
        H = np.array(_had(24), dtype=float)
        sq = np.sqrt(23.0)
        hv = []
        for k in range(24):
            cols = [c for c in range(24) if c != k]
            # take first 23 rows? take all 24 rows truncated to 23 cols? need dim 23.
            # Use columns deletion: each row -> 23-dim
            for r in range(24):
                vec = H[r, cols[:23]] / sq
                # cols has 23 entries, norm sqrt(23)
                hv.append(vec)
        hv = np.array(hv)
        hv /= np.linalg.norm(hv, axis=1, keepdims=True)
        parts.append(hv)
    except Exception:
        # fallback random +-1
        rr = rng.choice([-1.0, 1.0], size=(600, d)) / np.sqrt(float(d))
        parts.append(rr)
    # sparse supports 4-8
    Nsp = 36000
    Sp = np.zeros((Nsp, d), dtype=float)
    for t in range(Nsp):
        k = int(rng.randint(4, 9))
        supp = rng.choice(d, size=k, replace=False)
        sgn = rng.choice(np.array([-1.0, 1.0]), size=k)
        Sp[t, supp] = sgn / np.sqrt(float(k))
    parts.append(Sp)
    C = np.vstack(parts)
    # normalise
    C /= np.linalg.norm(C, axis=1, keepdims=True)
    # deduplicate
    try:
        Cr = np.round(C, 8)
        _, iu = np.unique(Cr, axis=0, return_index=True)
        iu = np.sort(iu)
        C = C[iu]
    except Exception:
        pass
    # cap size to ~60000
    if C.shape[0] > 60000:
        idx = rng.choice(C.shape[0], size=60000, replace=False)
        C = C[idx]
    return C

def solve():
    t0 = time.time()
    BUDGET = 172.0
    rng = np.random.RandomState(7)
    n, d = 54, 23
    try:
        xb = np.load("best.npy").astype(float)
    except Exception:
        xb = rng.randn(n, d)
    xb = xb / np.linalg.norm(xb, axis=1, keepdims=True)
    if xb.shape != (n, d):
        xb = rng.randn(n, d)
        xb /= np.linalg.norm(xb, axis=1, keepdims=True)
    best_backup = _max_cos(xb)
    x_best_global = xb.copy()
    best_global = best_backup

    # build discrete pool
    C = _build_pool(rng, d)
    m = C.shape[0]

    # greedy best-hole insertion
    sel = []
    sel_mask = np.zeros(m, dtype=bool)
    cur_max = np.full(m, -2.0, dtype=float)
    # start from first candidate (all +1 type if present) else 0
    first = 0
    # prefer a Golay-like dense vector: pick index with max sum? just 0
    sel.append(first)
    sel_mask[first] = True
    # update
    dot0 = C @ C[first]
    dot0[first] = -2.0
    cur_max = np.maximum(cur_max, dot0)
    # also mask selected as inf so not picked
    cur_max[first] = np.inf
    for s in range(1, n):
        # pick min cur_max among unselected
        bi = int(np.argmin(cur_max))
        sel.append(bi)
        sel_mask[bi] = True
        cur_max[bi] = np.inf
        if (time.time()-t0) > 30:
            break
        dn = C @ C[bi]
        # don't let selected affect
        cur_max = np.maximum(cur_max, dn)
        cur_max[bi] = np.inf
        for q in sel:
            cur_max[q] = np.inf
    sel_idx = np.array(sel, dtype=int)
    # if broke early, fill randomly
    if len(sel_idx) < n:
        need = n - len(sel_idx)
        cand = np.where(~sel_mask)[0]
        extra = rng.choice(cand, size=need, replace=False)
        sel_idx = np.concatenate([sel_idx, extra])
        for e in extra:
            sel_mask[e] = True
    S = C[sel_idx].copy()

    def set_max(Sm):
        G = Sm @ Sm.T
        np.fill_diagonal(G, -2.0)
        return float(G.max()), G

    cur_best, Gsel = set_max(S)
    THR = 0.0475

    # one-for-one lexicographic swaps (max, count>THR)
    swap_deadline = t0 + 55.0
    # precompute for speed
    it_swap = 0
    while time.time() < swap_deadline:
        it_swap += 1
        if it_swap > 120:
            break
        M = C @ S.T  # m x n
        G = S @ S.T
        np.fill_diagonal(G, -2.0)
        cur_best = float(G.max())
        tot_cnt = int(np.sum(G > THR) // 2)
        # rest quantities
        rest_max = np.empty(n)
        rest_cnt = np.empty(n, dtype=int)
        # per-point involvement count
        for i in range(n):
            Gr = G.copy()
            Gr[i, :] = -2.0; Gr[:, i] = -2.0
            rest_max[i] = float(Gr.max())
            rest_cnt[i] = int(np.sum(Gr > THR) // 2)
        row_sum = np.sum(M > THR, axis=1)
        # max1/max2 per candidate for fast excl (for max only, counts need row_sum)
        # compute row max excl per i on fly with vector ops
        best_gain = None  # (new_max, new_cnt, i, c)
        # evaluate each i
        for i in range(n):
            # row max excluding i
            Mi = M.copy()
            Mi[:, i] = -2.0
            rmax = Mi.max(axis=1)
            tmax = np.maximum(rest_max[i], rmax)
            # counts
            Bcol = (M[:, i] > THR).astype(int)
            cnt_new = row_sum - Bcol
            tcnt = rest_cnt[i] + cnt_new
            # mask already selected
            tmax[sel_mask] = np.inf
            # find best c for this i lexicographically
            # first min max
            mm = float(tmax.min())
            # among those within 1e-12 of mm, min count
            cand_ok = np.where(tmax <= mm + 1e-12)[0]
            if len(cand_ok) == 0:
                continue
            cc = cand_ok[int(np.argmin(tcnt[cand_ok]))]
            nmx = float(tmax[cc]); nc = int(tcnt[cc])
            if (nmx < cur_best - 1e-12) or (abs(nmx-cur_best) <= 1e-12 and nc < tot_cnt):
                if best_gain is None or (nmx < best_gain[0]-1e-12) or (abs(nmx-best_gain[0])<=1e-12 and nc < best_gain[1]):
                    best_gain = (nmx, nc, i, cc)
        if best_gain is None:
            # try random double-hole escape: remove 2 worst, greedy reinsert
            # do a few random trials
            improved = False
            Gtmp = S @ S.T
            np.fill_diagonal(Gtmp, -2.0)
            per = Gtmp.max(axis=1)
            worst = np.argsort(-per)[:4]
            for _ in range(30):
                if time.time() >= swap_deadline:
                    break
                i1, i2 = rng.choice(worst, size=2, replace=False)
                keep = [k for k in range(n) if k != i1 and k != i2]
                Sk = S[keep]
                # best first hole
                d1 = C @ Sk.T
                m1 = d1.max(axis=1)
                m1[sel_mask] = np.inf
                c1 = int(np.argmin(m1))
                # update with c1
                Sk2 = np.vstack([Sk, C[c1]])
                d2 = C @ Sk2.T
                m2 = d2.max(axis=1)
                # mask sel + c1
                tmp_mask = sel_mask.copy()
                tmp_mask[c1] = True
                m2[tmp_mask] = np.inf
                c2 = int(np.argmin(m2))
                Strial = np.vstack([Sk, C[c1], C[c2]])
                vm, _ = set_max(Strial)
                # count check
                if vm < cur_best - 1e-9:
                    # apply: replace i1,i2 with c1,c2
                    sel_mask[sel_idx[i1]] = False
                    sel_mask[sel_idx[i2]] = False
                    # avoid duplicate c1==c2
                    if c1 == c2:
                        continue
                    sel_mask[c1] = True
                    sel_mask[c2] = True
                    sel_idx[i1] = c1
                    sel_idx[i2] = c2
                    S[i1] = C[c1]; S[i2] = C[c2]
                    improved = True
                    break
            if not improved:
                break
        else:
            _, _, i, cc = best_gain
            old = int(sel_idx[i])
            sel_mask[old] = False
            sel_mask[cc] = True
            sel_idx[i] = cc
            S[i] = C[cc]
    # discrete best
    disc_best, _ = set_max(S)
    # continuous polish of discrete set
    x_disc = S.copy()
    # quick smooth then LP, time-aware
    remain = BUDGET - (time.time() - t0)
    if remain > 15:
        try:
            x_disc = _smooth(x_disc, steps=800, k0=200, k1=2000, lr0=0.03, lr1=0.004, d0=0.06, d1=0.006)
        except Exception:
            pass
    remain = BUDGET - (time.time() - t0)
    if remain > 20:
        lp_t = min(25.0, remain - 12.0)
        try:
            x_disc = _lp_polish(x_disc, seconds=lp_t, t0=t0, budget=BUDGET-8.0)
        except Exception:
            pass
        try:
            x_disc = _smooth(x_disc, steps=400, k0=400, k1=3000, lr0=0.015, lr1=0.003, d0=0.03, d1=0.005)
        except Exception:
            pass
    v_disc = _max_cos(x_disc)
    if v_disc < best_global:
        best_global = v_disc
        x_best_global = x_disc.copy()
    # final polish of global best with remaining time
    remain = BUDGET - (time.time() - t0)
    if remain > 3:
        try:
            ft = min(20.0, max(1.0, remain - 1.0))
            x_best_global = _lp_polish(x_best_global, seconds=ft, t0=t0, budget=BUDGET-1.0)
        except Exception:
            pass
    x_best_global /= np.linalg.norm(x_best_global, axis=1, keepdims=True)
    return x_best_global
