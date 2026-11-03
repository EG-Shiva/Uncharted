import time
import numpy as np
from scipy.optimize import linprog, linear_sum_assignment
from scipy.sparse import coo_matrix

def _norm(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n = np.maximum(n, 1e-12)
    return X / n

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _row_max(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return G.max(axis=1), G

def _lse_smooth(X, iters, a0, a1, lr0, lr1, nz0, nz1, rng):
    n, d = X.shape
    for t in range(iters):
        f = t / max(1, iters - 1)
        a = a0 * ((a1 / a0) ** f) if a0 != a1 else a0
        lr = lr0 * ((lr1 / lr0) ** f) if lr0 != lr1 else lr0
        nm = nz0 * ((nz1 / nz0) ** f) if nz0 > 0 and nz1 > 0 else (nz0 * (1.0 - f) if nz0 > 0 else 0.0)
        G = X @ X.T
        np.fill_diagonal(G, -10.0)
        gm = float(G.max())
        W = np.exp(a * (G - gm))
        np.fill_diagonal(W, 0.0)
        S = float(W.sum())
        if not np.isfinite(S) or S <= 0:
            break
        H = W @ X
        WG = (W * G).sum(axis=1)
        Grad = (H - WG[:, None] * X) / S
        dot = (Grad * X).sum(axis=1, keepdims=True)
        Gt = Grad - dot * X
        mx = float(np.linalg.norm(Gt, axis=1).max())
        if mx < 1e-14:
            if nm <= 0:
                continue
            mx = 1.0
        else:
            X = X - (lr / mx) * Gt
        if nm > 0:
            N = rng.randn(n, d)
            nd = (N * X).sum(axis=1, keepdims=True)
            N = N - nd * X
            nn = np.linalg.norm(N, axis=1, keepdims=True) + 1e-12
            X = X + nm * (N / nn)
        X = _norm(X)
    return X

def _riesz_smooth(X, s, iters, lr0, lr1, rng, noise=0.0):
    n, d = X.shape
    expo = -(s + 2.0) / 2.0
    for t in range(iters):
        f = t / max(1, iters - 1)
        lr = lr0 * ((lr1 / lr0) ** f) if lr0 != lr1 else lr0
        G = X @ X.T
        np.fill_diagonal(G, 1.0 - 1e-9)
        D2 = np.clip(2.0 - 2.0 * G, 1e-9, 4.0)
        W = np.power(D2, expo)
        # cap to avoid blowup
        np.clip(W, 0, 1e4, out=W)
        np.fill_diagonal(W, 0.0)
        H = W @ X
        ds = W.sum(axis=1)
        Rep = ds[:, None] * X - H
        dot = (Rep * X).sum(axis=1, keepdims=True)
        Gt = Rep - dot * X
        mx = float(np.linalg.norm(Gt, axis=1).max())
        if mx < 1e-14:
            if noise <= 0:
                continue
            mx = 1.0
        else:
            X = X + (lr / mx) * Gt
        if noise > 0:
            N = rng.randn(n, d)
            nd = (N * X).sum(axis=1, keepdims=True)
            N = N - nd * X
            nn = np.linalg.norm(N, axis=1, keepdims=True) + 1e-12
            X = X + noise * (N / nn)
        X = _norm(X)
    return X

def _lp_polish(X, seconds, deadline=None):
    X = _norm(np.array(X, dtype=float))
    n, d = X.shape
    best = _max_cos(X)
    radius = 5e-3
    start = time.time()
    iu0 = np.triu_indices(n, 1)
    iuI, iuJ = iu0[0], iu0[1]
    while time.time() - start < seconds:
        if deadline is not None and time.time() > deadline:
            break
        if radius < 1e-14:
            break
        G = X @ X.T
        cf = G[iuI, iuJ]
        m = float(cf.max())
        if m < best:
            best = m
        thresh = m - max(20.0 * radius, 1e-9)
        sel_idx = np.where(cf > thresh)[0]
        if sel_idx.size == 0:
            radius *= 0.5
            continue
        if sel_idx.size > 2600:
            part = np.argpartition(-cf[sel_idx], 2600)[:2600]
            sel_idx = sel_idx[part]
        I = iuI[sel_idx]; J = iuJ[sel_idx]
        c = cf[sel_idx]
        p = len(I)
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([X[J], X[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((X.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv); obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-c, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = X + res.x[:-1].reshape(n, d)
        y = _norm(y)
        v = _max_cos(y)
        if v < best - 1e-15:
            X, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return X

def _largest_holes(Xkeep, k, rng, n_samp=8000):
    d = Xkeep.shape[1]
    C = _norm(rng.randn(n_samp, d))
    S = C @ Xkeep.T
    sc = S.max(axis=1)
    order = np.argsort(sc)
    picked = []
    for idx in order:
        if len(picked) >= k:
            break
        v = C[idx]
        ok = True
        for q in picked:
            if float(v @ q) > 0.6:
                ok = False
                break
        if ok:
            picked.append(v.copy())
    while len(picked) < k:
        v = rng.randn(d); v /= (np.linalg.norm(v) + 1e-12)
        picked.append(v)
    return np.array(picked)

def _make_E8():
    pts = []
    # D8 type
    for i in range(8):
        for j in range(i+1, 8):
            for si in (1.0, -1.0):
                for sj in (1.0, -1.0):
                    v = np.zeros(8); v[i]=si; v[j]=sj
                    pts.append(v)
    # half-integer type
    for mask in range(256):
        bits = [(mask>>k)&1 for k in range(8)]
        if sum(bits) % 2 == 1:
            continue
        v = np.array([0.5 if b==0 else -0.5 for b in bits])
        pts.append(v)
    E = np.array(pts, dtype=float)  # 240x8
    E = E / np.linalg.norm(E, axis=1, keepdims=True)
    return E

def _fingerprint(X):
    rm, _ = _row_max(X)
    return np.sort(rm)

def _fp_dist(f1, f2):
    return float(np.linalg.norm(f1 - f2))

def _icp_align(A, B):
    n, d = A.shape
    R = np.eye(d)
    perm = np.arange(n)
    for _ in range(3):
        Brot = B @ R
        D = A @ Brot.T
        try:
            r, c = linear_sum_assignment(-D)
        except Exception:
            break
        perm = np.empty(n, dtype=int)
        perm[r] = c
        Bp = B[perm]
        M = Bp.T @ A
        try:
            U, _, Vt = np.linalg.svd(M)
            R = U @ Vt
        except Exception:
            break
    # final match with final R
    try:
        Brot = B @ R
        D = A @ Brot.T
        r, c = linear_sum_assignment(-D)
        perm2 = np.empty(n, dtype=int)
        perm2[r] = c
        perm = perm2
        Bp = B[perm]
        M = Bp.T @ A
        U, _, Vt = np.linalg.svd(M)
        R = U @ Vt
    except Exception:
        pass
    Bal = B[perm] @ R
    return Bal, perm, R

def solve():
    t0 = time.time()
    deadline = t0 + 172.0
    rng = np.random.RandomState(12345)
    Xb = None; sb = 1e9
    for fn in ["best.npy", "record.npy"]:
        try:
            v = np.load(fn).astype(float)
            if v.shape == (162, 8):
                v = _norm(v)
                s = _max_cos(v)
                if s < sb:
                    sb = s; Xb = v.copy()
        except Exception:
            pass
    if Xb is None:
        Xb = _norm(rng.randn(162, 8)); sb = _max_cos(Xb)
    X_best = Xb.copy(); s_best = float(sb)
    n, d = 162, 8
    E8 = _make_E8()

    elites = []  # list of (score, X, fp)
    def try_add(X):
        global_s = _max_cos(X)
        fp = _fingerprint(X)
        for (s2, _, fp2) in elites:
            if _fp_dist(fp, fp2) < 5e-4 and abs(s2 - global_s) < 2e-4:
                # duplicate: keep better
                if global_s < s2 - 1e-12:
                    # replace that entry
                    idx = next(i for i,(ss,_,ff) in enumerate(elites) if ff is fp2)
                    elites[idx] = (global_s, X.copy(), fp)
                    elites.sort(key=lambda z: z[0])
                return False
        elites.append((global_s, X.copy(), fp))
        elites.sort(key=lambda z: z[0])
        while len(elites) > 8:
            elites.pop()
        return True

    try_add(X_best)

    # Phase 1: diverse optima
    n_elite_target = 8
    start_types = ['inc_small', 'random', 'e8', 'inc_large', 'random2', 'e8_2', 'inc_med', 'lowdisc']
    s_choices = [3.0, 6.0, 9.0, 4.0, 12.0, 5.0, 7.0, 3.5]
    t1_end = t0 + 55.0
    si = 0
    while len(elites) < n_elite_target and time.time() < t1_end:
        typ = start_types[si % len(start_types)]
        s_exp = s_choices[si % len(s_choices)]
        si += 1
        if typ.startswith('inc'):
            amp = {'inc_small': 0.05, 'inc_med': 0.18, 'inc_large': 0.45}[typ]
            X0 = _norm(X_best + rng.randn(n, d) * amp)
        elif typ == 'random' or typ == 'random2' or typ == 'lowdisc':
            # low-discrepancy: use uniform sphere via normalized gaussian with different seed stream + slight repulsion init
            X0 = _norm(rng.randn(n, d))
            if typ == 'lowdisc':
                # antithetic pairs for spread
                half = n // 2
                X0[half:] = -X0[:half] + rng.randn(n-half, d) * 0.3
                X0 = _norm(X0)
        elif typ.startswith('e8'):
            idx = rng.choice(len(E8), size=n, replace=False)
            X0 = E8[idx].copy()
            jit = 0.08 if typ == 'e8' else 0.25
            X0 = _norm(X0 + rng.randn(n, d) * jit)
            # if need more than 240 pool, fill rest random (n=162 <240 so fine)
        else:
            X0 = _norm(rng.randn(n, d))
        # Riesz annealing then LSE refinement
        X0 = _riesz_smooth(X0, s_exp, 160, 0.05, 0.008, rng, noise=0.001 if typ.startswith('inc') else 0.0)
        a0 = 30.0; a1 = 500.0 + 400.0 * rng.rand()
        X0 = _lse_smooth(X0, 320, a0, a1, 0.014, 0.0015, 0.001, 0.0, rng)
        sc = _max_cos(X0)
        if sc < s_best + 0.01 and time.time() < deadline - 40:
            X0 = _lp_polish(X0, 1.2, deadline - 30)
            sc = _max_cos(X0)
        try_add(X0)
        if sc < s_best - 1e-12:
            s_best = sc; X_best = X0.copy()
        if time.time() > t1_end:
            break
    # ensure elites non-empty and sorted
    if len(elites) == 0:
        elites.append((s_best, X_best.copy(), _fingerprint(X_best)))
    elites.sort(key=lambda z: z[0])
    if elites[0][0] < s_best:
        s_best = elites[0][0]; X_best = elites[0][1].copy()

    # Phase 2: recombination
    child_iter = 0
    while time.time() < deadline - 14.0:
        child_iter += 1
        if len(elites) < 2:
            # basin hop from best
            Xt = _norm(X_best + rng.randn(n, d) * 0.05)
            Xt = _lse_smooth(Xt, 300, 25, 600, 0.012, 0.001, 0.0, 0.0, rng)
            if _max_cos(Xt) < s_best + 0.005:
                Xt = _lp_polish(Xt, 1.5, deadline - 12)
            sc = _max_cos(Xt)
            try_add(Xt)
            if sc < s_best:
                s_best = sc; X_best = Xt.copy()
            continue
        # select parents: bias to good but ensure diversity; pick i random among top, j diverse
        m = len(elites)
        # tournament: pick two indices
        i = rng.randint(m)
        j = rng.randint(m)
        if i == j:
            j = (j + 1) % m
        A = elites[i][1]; sA = elites[i][0]
        B = elites[j][1]; sB = elites[j][0]
        # skip near-duplicates
        if _fp_dist(elites[i][2], elites[j][2]) < 1e-4:
            # perturb one
            A = _norm(A + rng.randn(n, d) * 0.02)
        try:
            Bal, perm, R = _icp_align(A, B)
        except Exception:
            Bal = B.copy()
        # local scores for best-per-point mixing
        rmA, _ = _row_max(A)
        rmB, _ = _row_max(B)
        # Bal corresponds to B reordered; need rm for reordered B
        # recompute row max for Bal config? Bal is rotated B, same Gram so same rm permuted
        rmBp = rmB[perm]
        mode = child_iter % 3
        if mode == 0:
            # complementary spatial half
            u = rng.randn(d); u /= (np.linalg.norm(u) + 1e-12)
            proj = A @ u
            med = np.median(proj)
            mask = proj > med
            # randomize side
            if rng.rand() < 0.5:
                mask = ~mask
            C = np.where(mask[:, None], A, Bal)
        elif mode == 1:
            mask = rng.rand(n) < 0.5
            # ensure both parents contribute
            if mask.sum() == 0 or mask.sum() == n:
                mask[rng.randint(n)] = ~mask[rng.randint(n)]
            C = np.where(mask[:, None], A, Bal)
        else:
            # individually better points
            pickA = rmA <= rmBp
            # add randomness: flip 15% randomly
            flip = rng.rand(n) < 0.15
            pickA = np.logical_xor(pickA, flip)
            C = np.where(pickA[:, None], A, Bal)
        C = _norm(C)
        target = min(sA, sB, s_best)
        # greedy hole repair for egregious clashes
        for _rep in range(4):
            rmC, GC = _row_max(C)
            mc = float(rmC.max())
            if mc < target + 0.004:
                break
            worst = int(np.argmax(rmC))
            keep = np.ones(n, dtype=bool); keep[worst] = False
            hole = _largest_holes(C[keep], 1, rng, n_samp=6000)[0]
            C[worst] = hole
            # slight local shake of neighbours
            nbrs = np.argsort(-(C @ C[worst]))[1:6]
            C[nbrs] = _norm(C[nbrs] + rng.randn(len(nbrs), d) * 0.01)
        # smooth repair
        # vary Riesz exponent per child
        s_exp = float(rng.choice([3.0, 6.0, 9.0]))
        C = _riesz_smooth(C, s_exp, 120, 0.03, 0.006, rng, noise=0.0)
        C = _lse_smooth(C, 320, 35, 900, 0.010, 0.001, 0.0, 0.0, rng)
        sc = _max_cos(C)
        if sc < min(sA, sB) + 0.004 and time.time() < deadline - 12:
            C = _lp_polish(C, 1.6, deadline - 11)
            sc = _max_cos(C)
        # insert
        added = try_add(C)
        if sc < s_best - 1e-15:
            s_best = sc; X_best = C.copy()
            # bonus polish on new global best if time
            if time.time() < deadline - 13:
                X_best = _lp_polish(X_best, 1.5, deadline - 11)
                s_best = _max_cos(X_best)
                # update elite entry
                try_add(X_best)
        # occasional direct improvement attempt of best via small shake (avoid stagnation)
        if child_iter % 7 == 0 and time.time() < deadline - 14:
            Xt = _norm(X_best + rng.randn(n, d) * 0.015)
            Xt = _lse_smooth(Xt, 200, 60, 800, 0.006, 0.0008, 0.0, 0.0, rng)
            sxt = _max_cos(Xt)
            if sxt < s_best + 0.002:
                Xt = _lp_polish(Xt, 1.0, deadline - 11)
                sxt = _max_cos(Xt)
            try_add(Xt)
            if sxt < s_best:
                s_best = sxt; X_best = Xt.copy()

    # final: pick best elite
    elites.sort(key=lambda z: z[0])
    if len(elites) > 0 and elites[0][0] < s_best:
        s_best = elites[0][0]; X_best = elites[0][1].copy()
    remain = deadline - time.time() - 0.8
    if remain > 2.0:
        X_best = _lp_polish(X_best, min(remain, 12.0), deadline - 0.4)
    X_best = _norm(X_best)
    return X_best
