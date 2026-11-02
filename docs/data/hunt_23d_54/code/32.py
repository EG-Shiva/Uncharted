import time
import numpy as np

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(np.max(g))

def _anneal(x, steps, k0, k1, lr0, lr1, w0, w1, mu=0.85):
    n, d = x.shape
    v = np.zeros_like(x)
    lk0, lk1 = np.log(k0), np.log(k1)
    ll0, ll1 = np.log(lr0), np.log(lr1)
    for s in range(steps):
        f = s / max(steps - 1, 1)
        k = np.exp(lk0 + (lk1 - lk0) * f)
        lr = np.exp(ll0 + (ll1 - ll0) * f)
        w = w0 + (w1 - w0) * f
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        m = float(np.max(G))
        D = G - m
        W = np.zeros_like(G)
        sel = D > -w
        Ds = D[sel]
        W[sel] = np.exp(k * Ds)
        np.fill_diagonal(W, 0.0)
        F = W @ x
        dot = np.sum(F * x, axis=1, keepdims=True)
        Ft = F - dot * x
        an = float(np.sqrt(np.sum(Ft * Ft) / n)) + 1e-12
        v = mu * v + Ft / (an + 1e-12)
        vn = float(np.sqrt(np.sum(v * v) / n)) + 1e-12
        step = lr * v / max(vn, 1.0)
        x = x - step
        nr = np.linalg.norm(x, axis=1, keepdims=True)
        nr[nr == 0] = 1.0
        x /= nr
    return x

def _build_pool(rng):
    d = 23
    parts = []
    # 1) cyclic binary Golay (23,12,7), g = 1+x+x^5+x^6+x^7+x^9+x^11
    gcoef = np.zeros(23, dtype=int)
    for jj in [0, 1, 5, 6, 7, 9, 11]:
        gcoef[jj] = 1
    Gmat = np.zeros((12, 23), dtype=int)
    for i in range(12):
        Gmat[i] = np.roll(gcoef, i)
    Ncode = 4096
    bits = ((np.arange(Ncode)[:, None] >> np.arange(12)) & 1).astype(int)
    codes = (bits @ Gmat) % 2
    Gol = (1 - 2 * codes).astype(float) / np.sqrt(23.0)
    parts.append(Gol)
    parts.append(-Gol)
    # 2) Paley Q + Hadamard-24 rows
    res = set()
    for i in range(1, 23):
        res.add((i * i) % 23)
    res.discard(0)
    Q = np.zeros((23, 23), dtype=float)
    for i in range(23):
        for j in range(23):
            if i == j:
                Q[i, j] = 0.0
            else:
                Q[i, j] = 1.0 if ((j - i) % 23) in res else -1.0
    # rows of Q+I are +-1 vectors
    Qp = Q + np.eye(23)
    Qp = Qp / np.linalg.norm(Qp, axis=1, keepdims=True)
    parts.append(Qp)
    parts.append(-Qp)
    Qm = Q - np.eye(23)
    Qm = Qm / np.linalg.norm(Qm, axis=1, keepdims=True)
    parts.append(Qm)
    parts.append(-Qm)
    # Hadamard 24 attempt: border Paley
    H = np.ones((24, 24), dtype=float)
    H[1:, 0] = -1.0
    H[0, 1:] = 1.0
    H[1:, 1:] = Q + np.eye(23)
    # delete-one-coordinate variants -> length 23
    delvars = []
    for r in range(24):
        row = H[r]
        for dc in range(24):
            v = np.delete(row, dc)
            # v length 23
            nrm = np.linalg.norm(v)
            if nrm > 1e-9:
                delvars.append(v / nrm)
    parts.append(np.array(delvars))
    # 3) sparse supports 4..8
    for k in [4, 5, 6, 7, 8]:
        nk = 7000
        M = np.zeros((nk, d))
        signs = rng.randint(0, 2, size=(nk, k)) * 2 - 1
        for t in range(nk):
            supp = rng.choice(d, k, replace=False)
            M[t, supp] = signs[t] / np.sqrt(float(k))
        parts.append(M)
    # 4) full +-1 random
    nr = 7000
    R = (rng.randint(0, 2, size=(nr, d)) * 2 - 1).astype(float) / np.sqrt(float(d))
    parts.append(R)
    # 5) e_i
    E = np.zeros((46, d))
    for i in range(23):
        E[2 * i, i] = 1.0
        E[2 * i + 1, i] = -1.0
    parts.append(E)
    # 6) (e_i +/- e_j)/sqrt2
    E2 = []
    for i in range(23):
        for j in range(i + 1, 23):
            for s in (1.0, -1.0):
                v = np.zeros(d)
                v[i] = 1.0
                v[j] = s
                v /= np.sqrt(2.0)
                E2.append(v)
                E2.append(-v)
    parts.append(np.array(E2))
    C = np.concatenate(parts, axis=0)
    # normalize safety
    C = C / np.linalg.norm(C, axis=1, keepdims=True)
    # deduplicate via rounding
    Cr = np.round(C, 8)
    # use unique
    try:
        Cu = np.unique(Cr, axis=0)
    except Exception:
        Cu = Cr
    # if too big, subsample deterministically
    if Cu.shape[0] > 70000:
        idx = rng.choice(Cu.shape[0], 70000, replace=False)
        Cu = Cu[idx]
    # renormalize after rounding
    Cu = Cu / np.linalg.norm(Cu, axis=1, keepdims=True)
    return Cu

def _greedy(C, start_idx, rng=None):
    M, d = C.shape
    n = 54
    sel = [int(start_idx % M)]
    best_scores = C @ C[sel[0]]
    best_scores[sel[0]] = np.inf
    in_set = np.zeros(M, dtype=bool)
    in_set[sel[0]] = True
    for _ in range(1, n):
        # argmin
        j = int(np.argmin(best_scores))
        sel.append(j)
        in_set[j] = True
        dots = C @ C[j]
        # update max
        np.maximum(best_scores, dots, out=best_scores)
        best_scores[in_set] = np.inf
    S = C[np.array(sel)]
    return S, np.array(sel, dtype=int)

def solve():
    t0 = time.time()
    BUDGET = 172.0
    rng = np.random.RandomState(12345)
    n, d = 54, 23
    # fallback best
    try:
        xb = np.load("best.npy").astype(float)
    except Exception:
        xb = rng.randn(n, d)
    xb = xb / np.linalg.norm(xb, axis=1, keepdims=True)
    if xb.shape != (n, d):
        xb = rng.randn(n, d)
        xb /= np.linalg.norm(xb, axis=1, keepdims=True)
    try:
        xr = np.load("record.npy").astype(float)
        xr /= np.linalg.norm(xr, axis=1, keepdims=True)
        if xr.shape == (n, d) and _max_cos(xr) < _max_cos(xb):
            xb = xr.copy()
    except Exception:
        pass
    best_fallback = _max_cos(xb)
    x_best = xb.copy()
    best_overall = best_fallback

    # build pool
    C = _build_pool(rng)
    M = C.shape[0]

    # greedy from a few starts, keep best discrete
    bestS = None
    bestSel = None
    bestD = np.inf
    # deterministic starts: 0, M//3, M//2
    starts = [0, M // 3, M // 2]
    # time guard
    for si in starts:
        if time.time() - t0 > 40:
            break
        S, sel = _greedy(C, si)
        v = _max_cos(S)
        if v < bestD:
            bestD = v
            bestS = S.copy()
            bestSel = sel.copy()
    if bestS is None:
        # fallback: return polished fallback quickly
        x = _anneal(x_best.copy(), steps=400, k0=200, k1=3000, lr0=0.004, lr1=0.0004, w0=0.02, w1=0.004)
        if _max_cos(x) < best_overall:
            return x
        return x_best
    S = bestS
    sel_idx = bestSel
    cur_max = _max_cos(S)

    # one-for-one swaps, vectorized
    # limit time for swaps ~ 45s
    sel_set = set(sel_idx.tolist())
    # map candidate index -> position? for exclusion we use boolean
    is_sel = np.zeros(M, dtype=bool)
    is_sel[sel_idx] = True
    nswap = 0
    sweep = 0
    while time.time() - t0 < 75.0 and sweep < 6:
        sweep += 1
        improved_this_sweep = False
        # current gram without-i maxima
        G = S @ S.T
        np.fill_diagonal(G, -2.0)
        # D matrix
        D = C @ S.T  # M x 54
        for i in range(n):
            if time.time() - t0 > 75.0:
                break
            # best without i
            mask = np.ones(n, dtype=bool)
            mask[i] = False
            without = float(np.max(G[np.ix_(mask, mask)])) if n > 2 else -2.0
            # max over j!=i for each candidate
            # compute max excluding column i efficiently
            # Dmax_excl = max over columns != i
            Dmax = np.max(D[:, mask], axis=1)
            newmax = np.maximum(Dmax, without)
            # forbid selected (except i's own candidate which equals current)
            # candidates currently selected (other than possibly sel_idx[i]) are forbidden
            newmax[is_sel] = np.inf
            # allow staying (current value) : ensure at least current
            # find best candidate
            cstar = int(np.argmin(newmax))
            vstar = float(newmax[cstar])
            if vstar < cur_max - 1e-12:
                # accept
                old_c = int(sel_idx[i])
                is_sel[old_c] = False
                is_sel[cstar] = True
                sel_idx[i] = cstar
                sel_set.discard(old_c)
                sel_set.add(cstar)
                S[i] = C[cstar]
                cur_max = _max_cos(S)
                improved_this_sweep = True
                # update D column i and G row/col i incrementally
                D[:, i] = C @ S[i]
                G[i, :] = S[i] @ S.T
                G[:, i] = G[i, :]
                G[i, i] = -2.0
                nswap += 1
            elif abs(vstar - cur_max) <= 1e-12:
                # tie-count improvement: compare counts
                # current ties
                thr = cur_max - 1e-9
                cur_ties = int(np.sum(G > thr) // 2)
                # new ties if swapped
                # ties without i
                Gw = G[np.ix_(mask, mask)]
                ties_wo = int(np.sum(Gw > thr) // 2)
                add = int(np.sum(D[cstar, mask] > thr))
                new_ties = ties_wo + add
                if new_ties < cur_ties - 1:
                    old_c = int(sel_idx[i])
                    is_sel[old_c] = False
                    is_sel[cstar] = True
                    sel_idx[i] = cstar
                    S[i] = C[cstar]
                    # cur_max unchanged
                    D[:, i] = C @ S[i]
                    G[i, :] = S[i] @ S.T
                    G[:, i] = G[i, :]
                    G[i, i] = -2.0
                    improved_this_sweep = True
        if not improved_this_sweep:
            break

    # occasional two-for-two sampled swaps on worst points
    if time.time() - t0 < 85.0:
        G = S @ S.T
        np.fill_diagonal(G, -2.0)
        ppm = np.max(G, axis=1)
        worst = np.argsort(-ppm)[:6]
        # promising candidates ranked by max to all S
        Dfull = C @ S.T
        scr = np.max(Dfull, axis=1)
        scr[is_sel] = np.inf
        order = np.argsort(scr)
        Ktop = 1500
        top = order[:Ktop]
        Ct = C[top]
        # Gram among top for pair dots (compute on fly per sample to save time)
        import itertools
        pairs = list(itertools.combinations(worst.tolist(), 2))[:8]
        for (a, b) in pairs:
            if time.time() - t0 > 85.0:
                break
            keep = np.ones(n, dtype=bool)
            keep[a] = False
            keep[b] = False
            Sk = S[keep]  # 52 x d
            without = float(np.max(Sk @ Sk.T)) if Sk.shape[0] > 1 else -2.0
            # scores of top to kept
            Dk = Ct @ Sk.T  # 1500 x 52
            s1 = np.max(Dk, axis=1)  # 1500
            # sample random pairs
            nsamp = 4000
            r1 = rng.randint(0, Ktop, size=nsamp)
            r2 = rng.randint(0, Ktop, size=nsamp)
            # avoid same
            # compute pair dot quickly
            # vectorized in chunks
            best_v = cur_max
            best_p = None
            for ch in range(0, nsamp, 1000):
                i1 = r1[ch:ch + 1000]
                i2 = r2[ch:ch + 1000]
                # skip equal
                dots = np.sum(Ct[i1] * Ct[i2], axis=1)
                m = np.maximum(np.maximum(s1[i1], s1[i2]), dots)
                m = np.maximum(m, without)
                j = int(np.argmin(m))
                if float(m[j]) < best_v - 1e-12:
                    best_v = float(m[j])
                    best_p = (int(top[i1[j]]), int(top[i2[j]]))
            if best_p is not None and best_v < cur_max - 1e-12:
                # apply
                old_a, old_b = int(sel_idx[a]), int(sel_idx[b])
                is_sel[old_a] = False
                is_sel[old_b] = False
                is_sel[best_p[0]] = True
                is_sel[best_p[1]] = True
                sel_idx[a] = best_p[0]
                sel_idx[b] = best_p[1]
                S[a] = C[best_p[0]]
                S[b] = C[best_p[1]]
                cur_max = _max_cos(S)

    # continuous polish of discrete set
    disc = S.copy()
    disc_max_before = _max_cos(disc)
    remaining = BUDGET - (time.time() - t0)
    # reserve 8s for final fallback compare
    steps = 1200
    if remaining < 60:
        steps = 700
    if remaining < 30:
        steps = 400
    try:
        if remaining > 10:
            disc = _anneal(disc, steps=steps, k0=70, k1=2500, lr0=0.014, lr1=0.0008, w0=0.055, w1=0.005)
            # second fine stage
            rem2 = BUDGET - (time.time() - t0)
            if rem2 > 20:
                disc = _anneal(disc, steps=400, k0=300, k1=4000, lr0=0.003, lr1=0.0003, w0=0.015, w1=0.003)
    except Exception:
        pass
    disc_max = _max_cos(disc)
    if disc_max < best_overall:
        best_overall = disc_max
        x_best = disc.copy()
    else:
        # if discrete not better, try light polish of fallback if time left
        rem = BUDGET - (time.time() - t0)
        if rem > 12:
            try:
                xf = _anneal(x_best.copy(), steps=min(600, int(rem * 10)), k0=150, k1=3000, lr0=0.005, lr1=0.0004, w0=0.02, w1=0.004)
                vf = _max_cos(xf)
                if vf < best_overall:
                    best_overall = vf
                    x_best = xf.copy()
            except Exception:
                pass
    # also consider disc even if slightly worse? No, keep best for scoring.
    # ensure unit norm
    x_best = x_best / np.linalg.norm(x_best, axis=1, keepdims=True)
    return x_best
