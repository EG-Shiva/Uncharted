import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _norm_rows(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[~np.isfinite(n)] = 1.0
    n[n == 0] = 1.0
    return X / n

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _stats(X, band=5e-06):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    m = float(G.max())
    flat = int(np.argmax(G))
    n = X.shape[0]
    a = flat // n; b = flat % n
    row_max = G.max(axis=1)
    cnt = np.sum(G > (m - band), axis=1)
    return m, a, b, row_max, cnt, G

def _fingerprint(X, k=40):
    n = X.shape[0]
    iu = np.triu_indices(n, 1)
    G = X @ X.T
    c = G[iu]
    top = np.sort(c)[-k:][::-1]
    q = np.round(top * 1e6).astype(int)
    return tuple(q.tolist())

def _lp_short(X, seconds=0.35, band=8e-04, radius_init=1e-3, deadline=None):
    X = _norm_rows(X.copy())
    n, d = X.shape
    try:
        best = _max_cos(X)
    except Exception:
        return X
    radius = float(radius_init)
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds:
        if deadline is not None and time.time() > deadline:
            break
        if radius < 1e-14:
            break
        G = X @ X.T
        c = G[iu]
        m = float(c.max())
        thresh = m - max(band, 20.0 * radius)
        sel = c > thresh
        I = iu[0][sel]; J = iu[1][sel]
        cs = c[sel]
        p = len(I)
        if p == 0:
            break
        if p > 1400:
            kk = np.argpartition(-cs, 1400)[:1400]
            I = I[kk]; J = J[kk]; cs = cs[kk]
            p = 1400
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([X[J], X[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((X.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        except Exception:
            radius *= 0.5
            continue
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-cs, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = X + res.x[:-1].reshape(n, d)
        y = _norm_rows(y)
        if not np.all(np.isfinite(y)):
            radius *= 0.5
            continue
        v = _max_cos(y)
        if v < best:
            X, best = y, v
            radius = min(radius * 1.5, 0.02)
        else:
            radius *= 0.5
            if radius < 2e-5:
                break
    return X

def _quench_subset(X0, mov, steps, T0, T1, lr0, lr1, rng=None):
    X = _norm_rows(X0.copy())
    n, d = X.shape
    mov = np.asarray(mov, dtype=int)
    m1 = np.zeros_like(X)
    v1 = np.zeros_like(X)
    b1, b2, eps = 0.9, 0.999, 1e-8
    lT0, lT1 = np.log(T0), np.log(T1)
    ll0, ll1 = np.log(lr0), np.log(lr1)
    G0 = X @ X.T
    np.fill_diagonal(G0, -2.0)
    best_m = float(G0.max())
    best_X = X.copy()
    for s in range(1, steps + 1):
        frac = (s - 1) / max(steps - 1, 1)
        T = float(np.exp(lT0 * (1 - frac) + lT1 * frac))
        lr = float(np.exp(ll0 * (1 - frac) + ll1 * frac))
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        if m < best_m:
            best_m = m
            best_X = X.copy()
        D = (G - m) / T
        np.clip(D, -50, 0, out=D)
        E = np.exp(D)
        np.fill_diagonal(E, 0.0)
        S = float(E.sum() * 0.5)
        if not np.isfinite(S) or S <= 0:
            S = 1.0
        Grad = (E @ X) / S
        dot = np.sum(Grad * X, axis=1, keepdims=True)
        Gt = Grad - dot * X
        if not np.all(np.isfinite(Gt)):
            break
        m1 = b1 * m1 + (1 - b1) * Gt
        v1 = b2 * v1 + (1 - b2) * (Gt * Gt)
        c1 = 1.0 - b1 ** s
        c2 = 1.0 - b2 ** s
        step = (m1 / c1) / (np.sqrt(v1 / c2) + eps)
        # mask to movable only
        mask = np.zeros((n, 1))
        mask[mov, 0] = 1.0
        step = step * mask
        sn = float(np.linalg.norm(step))
        mxn = 5.0 * np.sqrt(len(mov) * d)
        if sn > mxn:
            step *= (mxn / sn)
        X = X - lr * step
        X = _norm_rows(X)
    Gf = best_X @ best_X.T
    np.fill_diagonal(Gf, -2.0)
    mf = float(Gf.max())
    Gc = X @ X.T
    np.fill_diagonal(Gc, -2.0)
    mc = float(Gc.max())
    if mc < mf:
        return _norm_rows(X)
    return _norm_rows(best_X)

def _hill_true_max(X0, mov, iters, sigma0, rng):
    X = _norm_rows(X0.copy())
    n, d = X.shape
    mov = np.asarray(mov, dtype=int)
    best = _max_cos(X)
    best_X = X.copy()
    cur = X.copy()
    cur_v = best
    sigma = float(sigma0)
    for it in range(iters):
        N = rng.standard_normal((len(mov), d)) * sigma
        Xm = cur[mov]
        N -= Xm * np.sum(N * Xm, axis=1, keepdims=True)
        cand = cur.copy()
        cand[mov] = _norm_rows(Xm + N)
        v = _max_cos(cand)
        if not np.isfinite(v):
            sigma *= 0.99
            continue
        if v < cur_v:
            cur = cand
            cur_v = v
            sigma *= 1.03
            if v < best:
                best = v
                best_X = cand.copy()
        else:
            # occasional sideways to explore
            if rng.random() < 0.005 and v < cur_v + 2e-06:
                cur = cand
                cur_v = v
            sigma *= 0.9985
            if sigma < 1e-05:
                sigma = 1e-05
            if sigma > 0.02:
                sigma = 0.02
    return best_X, best

def _refine_single(s0, R, steps=200, T=0.004, lr=0.12, mu=0.85):
    s = np.asarray(s0, dtype=float).copy()
    nn = float(np.linalg.norm(s))
    if not np.isfinite(nn) or nn == 0:
        return s0
    s = s / nn
    v = np.zeros_like(s)
    for _ in range(steps):
        g = R @ s
        m = float(g.max())
        e = np.exp(np.clip((g - m) / T, -50, 0))
        den = float(e.sum()) + 1e-30
        Grad = (R.T @ e) / den
        Gt = Grad - float(Grad @ s) * s
        v = mu * v + Gt
        s = s - lr * v
        n2 = float(np.linalg.norm(s))
        if not np.isfinite(n2) or n2 == 0:
            break
        s = s / n2
    return s

def _find_hole(R, rng, n_sample=4000):
    d = R.shape[1]
    Z = rng.standard_normal(size=(n_sample, d))
    Z = _norm_rows(Z)
    S = Z @ R.T
    scores = S.max(axis=1)
    k = min(3, n_sample)
    idx = np.argpartition(scores, k)[:k]
    best_s = None
    best_v = 1e100
    for kk in idx:
        s1 = _refine_single(Z[kk], R, steps=160, T=0.005, lr=0.15)
        s1 = _refine_single(s1, R, steps=160, T=0.0012, lr=0.07)
        v = float((R @ s1).max())
        if v < best_v:
            best_v = v
            best_s = s1
    best_s = best_s / (np.linalg.norm(best_s) + 1e-12)
    return best_s

def _pair_apart(X, a, b, step, rng):
    Y = X.copy()
    xa = X[a]; xb = X[b]
    mid = xa + xb
    nm = float(np.linalg.norm(mid))
    if nm < 1e-12:
        mid = rng.standard_normal(12); nm = float(np.linalg.norm(mid))
    mid = mid / nm
    da = xa - float(xa @ mid) * mid
    nd = float(np.linalg.norm(da))
    if nd < 1e-12:
        # random tangent
        da = rng.standard_normal(12)
        da -= float(da @ mid) * mid
        da -= float(da @ xa) * xa
        nd = float(np.linalg.norm(da))
        if nd < 1e-12:
            return Y
    e = da / nd
    # symmetric move
    ya = xa + step * e
    yb = xb - step * e
    # add tiny orthogonal jitter
    ja = rng.standard_normal(12) * (step * 0.2)
    jb = rng.standard_normal(12) * (step * 0.2)
    ja -= float(ja @ xa) * xa
    jb -= float(jb @ xb) * xb
    ya = ya + ja
    yb = yb + jb
    ya /= (np.linalg.norm(ya) + 1e-12)
    yb /= (np.linalg.norm(yb) + 1e-12)
    Y[a] = ya; Y[b] = yb
    return Y

def solve():
    t0 = time.time()
    HARD_END = t0 + 174.0
    HOP_END = HARD_END - 13.0
    np.random.seed(0)
    rng = np.random.default_rng(0)
    n, d = 71, 12
    best_X = None
    try:
        best_X = np.load("best.npy").astype(float)
    except Exception:
        best_X = None
    if best_X is None or best_X.shape != (71, 12) or not np.all(np.isfinite(best_X)):
        try:
            best_X = np.load("record.npy").astype(float)
        except Exception:
            best_X = rng.standard_normal((n, d))
    best_X = _norm_rows(np.asarray(best_X, dtype=float))
    best_score = _max_cos(best_X)
    # safety: also try record
    try:
        rec = _norm_rows(np.load("record.npy").astype(float))
        rs = _max_cos(rec)
        if rs < best_score:
            best_X, best_score = rec, rs
    except Exception:
        pass
    curX = best_X.copy()
    curScore = best_score
    tabu = set()
    try:
        tabu.add(_fingerprint(best_X))
    except Exception:
        pass
    mov_tabu = set()
    it = 0
    BAND_SEL = 8e-06
    while time.time() < HOP_END:
        it += 1
        tleft = HOP_END - time.time()
        if tleft < 1.0:
            break
        # choose base
        rp = rng.random()
        if rp < 0.70:
            base = curX
        else:
            base = best_X
        try:
            m, a, b, row_max, cnt, G = _stats(base, band=BAND_SEL)
        except Exception:
            a, b = 0, 1
            row_max = np.zeros(n); cnt = np.zeros(n)
        # choose movable set size 8..20
        k = int(rng.integers(8, 21))
        n_other = k - 2
        order_rm = np.argsort(-row_max + rng.random(n) * 1e-07)
        order_cnt = np.argsort(-cnt + rng.random(n) * 1e-07)
        chosen = []
        chosen_set = {int(a), int(b)}
        # interleave
        i1 = 0; i2 = 0
        # half from row_max, half from cnt
        n1 = (n_other + 1) // 2
        n2 = n_other - n1
        c1 = 0; c2 = 0
        # alternate but respect counts
        while len(chosen) < n_other:
            # decide source
            use_rm = True
            if c1 >= n1:
                use_rm = False
            elif c2 >= n2:
                use_rm = True
            else:
                use_rm = rng.random() < 0.5
            if use_rm:
                while i1 < n and (int(order_rm[i1]) in chosen_set):
                    i1 += 1
                if i1 < n:
                    v = int(order_rm[i1]); i1 += 1
                    chosen.append(v); chosen_set.add(v); c1 += 1
                else:
                    # fallback random
                    cand = int(rng.integers(0, n))
                    if cand not in chosen_set:
                        chosen.append(cand); chosen_set.add(cand); c1 += 1
                    else:
                        break
            else:
                while i2 < n and (int(order_cnt[i2]) in chosen_set):
                    i2 += 1
                if i2 < n:
                    v = int(order_cnt[i2]); i2 += 1
                    chosen.append(v); chosen_set.add(v); c2 += 1
                else:
                    cand = int(rng.integers(0, n))
                    if cand not in chosen_set:
                        chosen.append(cand); chosen_set.add(cand); c2 += 1
                    else:
                        break
        mov = np.array([int(a), int(b)] + chosen, dtype=int)
        # tabu on subsets: if recently visited, perturb one element
        key_mov = tuple(sorted(mov.tolist()))
        if key_mov in mov_tabu and rng.random() < 0.8:
            # swap one random non-pair element
            if len(mov) > 2:
                pos = int(rng.integers(2, len(mov)))
                cand = int(rng.integers(0, n))
                tries = 0
                while cand in set(mov.tolist()) and tries < 10:
                    cand = int(rng.integers(0, n)); tries += 1
                mov[pos] = cand
                key_mov = tuple(sorted(mov.tolist()))
        mov_tabu.add(key_mov)
        if len(mov_tabu) > 400:
            # drop arbitrary
            mov_tabu.pop()
        rtype = rng.random()
        Xcand = None
        try:
            if rtype < 0.45:
                # coordinated quench + true-max hill climb
                T0 = float(10 ** rng.uniform(-2.7, -1.9))
                T1 = float(10 ** rng.uniform(-4.2, -3.3))
                lr0 = float(10 ** rng.uniform(-2.9, -2.2))
                lr1 = lr0 * float(10 ** rng.uniform(-1.1, -0.4))
                steps = 700
                if tleft > 30 and rng.random() < 0.3:
                    steps = 1100
                elif tleft < 8:
                    steps = 400
                Xq = _quench_subset(base, mov, steps, T0, T1, lr0, lr1, rng)
                # multi-start hill climbing (2 starts)
                sig0 = float(10 ** rng.uniform(-3.3, -2.3))
                Xh, _ = _hill_true_max(Xq, mov, iters=1200 if tleft > 10 else 500, sigma0=sig0, rng=rng)
                # second start from perturbed
                if rng.random() < 0.4 and tleft > 6:
                    pert = Xq.copy()
                    pert[mov] = _norm_rows(pert[mov] + rng.standard_normal((len(mov), d)) * (sig0 * 0.7))
                    Xh2, vh2 = _hill_true_max(pert, mov, iters=600, sigma0=sig0 * 0.7, rng=rng)
                    if _max_cos(Xh2) < _max_cos(Xh):
                        Xh = Xh2
                Xcand = Xh
            elif rtype < 0.65:
                # rattler to farthest hole
                # pick rattler among mov excluding a,b: lowest cnt or lowest row_max? choose random weighted
                pool = mov[2:] if len(mov) > 2 else mov
                # prefer low-load for relocation? spec says chosen rattler; pick one with high row_max among rattlers? use random
                r_idx = int(pool[int(rng.integers(0, len(pool)))])
                keep = np.ones(n, dtype=bool); keep[r_idx] = False
                R = base[keep]
                h = _find_hole(R, rng, n_sample=3500)
                Xcand = base.copy()
                Xcand[r_idx] = h
                # coordinated jitter on other mov points
                others = [v for v in mov if v != r_idx]
                sig = float(10 ** rng.uniform(-3.0, -2.0))
                Nz = rng.standard_normal((len(others), d)) * sig
                Xo = Xcand[np.array(others)]
                Nz -= Xo * np.sum(Nz * Xo, axis=1, keepdims=True)
                Xcand[np.array(others)] = _norm_rows(Xo + Nz)
            elif rtype < 0.83:
                # pair apart symmetrically
                step = float(10 ** rng.uniform(-4.0, -2.3))
                Xcand = _pair_apart(base, int(a), int(b), step, rng)
                others = [v for v in mov if v not in (int(a), int(b))]
                if len(others) > 0:
                    sig = float(10 ** rng.uniform(-3.2, -2.1))
                    Nz = rng.standard_normal((len(others), d)) * sig
                    Xo = Xcand[np.array(others)]
                    Nz -= Xo * np.sum(Nz * Xo, axis=1, keepdims=True)
                    Xcand[np.array(others)] = _norm_rows(Xo + Nz)
            else:
                # swap high-load and low-load
                hi = int(np.argmax(row_max + rng.random(n) * 1e-09))
                lo = int(np.argmin(row_max + rng.random(n) * 1e-09))
                if hi == lo:
                    lo = int(rng.integers(0, n))
                Xcand = base.copy()
                tmp = Xcand[hi].copy()
                Xcand[hi] = Xcand[lo]
                Xcand[lo] = tmp
                # small tangent noise on swapped + other mov
                sig = float(10 ** rng.uniform(-3.2, -2.2))
                Nz = rng.standard_normal((len(mov), d)) * sig
                Xm = Xcand[mov]
                Nz -= Xm * np.sum(Nz * Xm, axis=1, keepdims=True)
                Xcand[mov] = _norm_rows(Xm + Nz)
        except Exception:
            continue
        if Xcand is None or not np.all(np.isfinite(Xcand)):
            continue
        Xcand = _norm_rows(Xcand)
        # short wide-band LP acceptance
        try:
            lp_sec = 0.35
            if tleft < 8:
                lp_sec = 0.18
            Xlp = _lp_short(Xcand, seconds=lp_sec, band=8e-04, radius_init=1e-3, deadline=HOP_END)
        except Exception:
            Xlp = Xcand
        try:
            s = _max_cos(Xlp)
            fp = _fingerprint(Xlp)
        except Exception:
            continue
        if not np.isfinite(s):
            continue
        # tabu check: force new topologies unless new best
        if fp in tabu and s >= best_score - 1e-12:
            # do not accept, but occasionally allow sideways for cur diversity
            if rng.random() < 0.02 and s < curScore + 5e-06:
                pass
            else:
                continue
        tabu.add(fp)
        if len(tabu) > 800:
            # trim: keep last 600 (convert to list, drop oldest arbitrary)
            # sets unordered; just clear half
            lst = list(tabu)
            tabu = set(lst[-600:])
        # acceptance by exact score
        if s < best_score:
            best_score = s
            best_X = _norm_rows(Xlp)
            curX = best_X.copy()
            curScore = s
        elif s < curScore:
            curX = _norm_rows(Xlp)
            curScore = s
        elif rng.random() < 0.04 and s < curScore + 8e-05:
            curX = _norm_rows(Xlp)
            curScore = s
        # occasional reset cur to best if drifted
        if curScore > best_score + 3e-04 and rng.random() < 0.3:
            curX = best_X.copy()
            curScore = best_score
    # final polish: quench full + LP
    try:
        # light full quench on best
        if time.time() < HARD_END - 6:
            T0, T1 = 8e-04, 1.2e-04
            # reuse subset quench with all indices for a short run
            all_idx = np.arange(n)
            Xp = _quench_subset(best_X, all_idx, 500, T0, T1, 0.003, 0.0005, rng)
            sp = _max_cos(Xp)
            if sp < best_score:
                best_X, best_score = _norm_rows(Xp), sp
    except Exception:
        pass
    try:
        sec_left = HARD_END - time.time() - 1.0
        if sec_left > 1.0:
            Xf = _lp_short(best_X, seconds=min(sec_left, 6.0), band=2e-04, radius_init=8e-04, deadline=HARD_END - 0.5)
            sf = _max_cos(Xf)
            if sf < best_score and np.all(np.isfinite(Xf)):
                best_X, best_score = _norm_rows(Xf), sf
    except Exception:
        pass
    best_X = _norm_rows(best_X)
    return best_X
