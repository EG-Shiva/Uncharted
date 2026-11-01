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

def _quench_subset(X0, mov, steps, T0, T1, lr0, lr1):
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
        mask = np.zeros((n, 1))
        mask[mov, 0] = 1.0
        step = step * mask
        sn = float(np.linalg.norm(step))
        mxn = 5.0 * np.sqrt(len(mov) * d)
        if sn > mxn:
            step *= (mxn / sn)
        X = X - lr * step
        X = _norm_rows(X)
    Gc = X @ X.T
    np.fill_diagonal(Gc, -2.0)
    mc = float(Gc.max())
    if mc < best_m:
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
        da = rng.standard_normal(12)
        da -= float(da @ mid) * mid
        da -= float(da @ xa) * xa
        nd = float(np.linalg.norm(da))
        if nd < 1e-12:
            return Y
    e = da / nd
    ya = xa + step * e
    yb = xb - step * e
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
        rp = rng.random()
        base = curX if rp < 0.70 else best_X
        try:
            m, a, b, row_max, cnt, G = _stats(base, band=BAND_SEL)
        except Exception:
            a, b = 0, 1
            row_max = np.zeros(n); cnt = np.zeros(n)
        k = int(rng.integers(8, 21))
        n_other = k - 2
        order_rm = np.argsort(-row_max + rng.random(n) * 1e-07)
        order_cnt = np.argsort(-cnt + rng.random(n) * 1e-07)
        chosen = []
        chosen_set = {int(a), int(b)}
        n1 = (n_other + 1) // 2
        n2 = n_other - n1
        c1 = 0; c2 = 0
        i1 = 0; i2 = 0
        guard = 0
        while len(chosen) < n_other and guard < 500:
            guard += 1
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
                    cand = int(rng.integers(0, n))
                    if cand not in chosen_set:
                        chosen.append(cand); chosen_set.add(cand); c1 += 1
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
        mov = np.array([int(a), int(b)] + chosen, dtype=int)
        key_mov = tuple(sorted(mov.tolist()))
        if key_mov in mov_tabu and rng.random() < 0.8 and len(mov) > 2:
            pos = int(rng.integers(2, len(mov)))
            for _ in range(20):
                cand = int(rng.integers(0, n))
                if cand not in set(mov.tolist()):
                    mov[pos] = cand
                    break
            key_mov = tuple(sorted(mov.tolist()))
        move_r = rng.random()
        Y0 = base.copy()
        mov_use = mov.copy()
        try:
            if move_r < 0.40:
                pass
            elif move_r < 0.62:
                # hole relocation: pick a rattler among chosen with high row_max
                cand_pool = mov_use[2:] if len(mov_use) > 2 else mov_use
                # choose highest row_max in pool
                pool_rm = row_max[cand_pool]
                r_idx = int(cand_pool[int(np.argmax(pool_rm))])
                mask = np.ones(n, dtype=bool); mask[r_idx] = False
                R = base[mask]
                hole = _find_hole(R, rng, n_sample=3500)
                Y0[r_idx] = hole
            elif move_r < 0.82:
                step = float(rng.uniform(0.015, 0.06))
                Y0 = _pair_apart(base, int(a), int(b), step, rng)
            else:
                h = int(np.argmax(row_max + rng.random(n) * 1e-09))
                l = int(np.argmin(row_max + rng.random(n) * 1e-09))
                if h != l:
                    Y0[h] = base[l]
                    Y0[l] = base[h]
                    jit = rng.standard_normal((2, d)) * 0.008
                    for kk, idx in enumerate([h, l]):
                        v = Y0[idx]
                        j = jit[kk] - float(jit[kk] @ v) * v
                        Y0[idx] = _norm_rows((v + j)[None, :])[0]
                    sset = set(mov_use.tolist())
                    sset.add(int(h)); sset.add(int(l))
                    mov_use = np.array(sorted(sset), dtype=int)
                    if len(mov_use) > 20:
                        # drop lowest row_max extras keeping a,b,h,l
                        keep = {int(a), int(b), int(h), int(l)}
                        extras = [x for x in mov_use if int(x) not in keep]
                        # sort extras by row_max desc keep first 16
                        extras_sorted = sorted(extras, key=lambda x: -row_max[int(x)])
                        mov_use = np.array(sorted(list(keep) + extras_sorted[:16]), dtype=int)
        except Exception:
            Y0 = base.copy()
        # multi-start subset quench params randomized
        T0 = float(np.exp(rng.uniform(np.log(0.003), np.log(0.012))))
        T1 = float(np.exp(rng.uniform(np.log(0.0002), np.log(0.0012))))
        lr0 = float(np.exp(rng.uniform(np.log(0.15), np.log(0.6))))
        lr1 = float(np.exp(rng.uniform(np.log(0.02), np.log(0.15))))
        steps = int(rng.integers(450, 800))
        try:
            Y1 = _quench_subset(Y0, mov_use, steps, T0, T1, lr0, lr1)
        except Exception:
            Y1 = Y0
        # second start with tangent noise sometimes
        if rng.random() < 0.35:
            try:
                Yn = Y0.copy()
                Nz = rng.standard_normal((len(mov_use), d)) * 0.012
                Xm = Yn[mov_use]
                Nz -= Xm * np.sum(Nz * Xm, axis=1, keepdims=True)
                Yn[mov_use] = _norm_rows(Xm + Nz)
                Y1b = _quench_subset(Yn, mov_use, max(300, steps // 2), T0 * 0.7, T1, lr0, lr1)
                if _max_cos(Y1b) < _max_cos(Y1):
                    Y1 = Y1b
            except Exception:
                pass
        # true-max hill climbing on movable subset
        try:
            hill_iters = int(rng.integers(350, 700))
            sigma0 = float(np.exp(rng.uniform(np.log(0.0015), np.log(0.008))))
            Y2, _ = _hill_true_max(Y1, mov_use, hill_iters, sigma0, rng)
        except Exception:
            Y2 = Y1
        # short wide-band LP acceptance polish
        try:
            lp_sec = 0.30 if tleft > 8 else 0.18
            Y3 = _lp_short(Y2, seconds=lp_sec, band=8e-04, radius_init=1e-3, deadline=HOP_END)
        except Exception:
            Y3 = Y2
        try:
            v = _max_cos(Y3)
        except Exception:
            continue
        try:
            fp = _fingerprint(Y3)
        except Exception:
            fp = None
        is_tabu = (fp is not None and fp in tabu)
        # archive mov
        mov_tabu.add(tuple(sorted(mov_use.tolist())))
        if len(mov_tabu) > 4000:
            # prune randomly (deterministic-ish)
            mov_tabu = set(list(mov_tabu)[-2000:])
        if fp is not None:
            if len(tabu) > 4000:
                tabu = set(list(tabu)[-2000:])
        # acceptance by exact score
        improved_cur = v < curScore - 1e-12
        improved_best = v < best_score - 1e-12
        if is_tabu and not improved_best:
            # force diversity: accept sideways rarely to escape
            if rng.random() < 0.02 and v < curScore + 5e-06:
                curX = Y3.copy(); curScore = v
                if fp is not None:
                    tabu.add(fp)
            continue
        if improved_best:
            best_X = Y3.copy(); best_score = v
            curX = Y3.copy(); curScore = v
            if fp is not None:
                tabu.add(fp)
        elif improved_cur or (rng.random() < 0.03 and v < curScore + 2e-06):
            curX = Y3.copy(); curScore = v
            if fp is not None:
                tabu.add(fp)
        # occasional restart from best if cur drifts
        if curScore > best_score + 3e-05 and rng.random() < 0.3:
            curX = best_X.copy(); curScore = best_score
    # final squeeze
    try:
        best_X = _lp_short(best_X, seconds=4.0, band=1e-04, radius_init=8e-04, deadline=HARD_END - 1.0)
        best_score = _max_cos(best_X)
    except Exception:
        pass
    # tiny shrink to guarantee strict feasibility margin
    try:
        best_X = _norm_rows(best_X)
        # re-normalize explicitly
        nn = np.linalg.norm(best_X, axis=1, keepdims=True)
        best_X = best_X / np.maximum(nn, 1e-12)
    except Exception:
        pass
    return np.asarray(best_X, dtype=float)
