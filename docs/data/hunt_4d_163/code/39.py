import time
import math
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _anneal(x0, steps, T0, T1, lr, mu, rng, deadline=None):
    x = x0.copy()
    n, d = x.shape
    vel = np.zeros_like(x)
    bv = _max_cos(x)
    bx = x.copy()
    if T0 <= 0 or T1 <= 0:
        T0 = max(T0, 1e-12); T1 = max(T1, 1e-12)
    lratio = math.log(T1 / T0) if T0 > 0 and T1 > 0 else 0.0
    for k in range(steps):
        if deadline is not None and (k & 63) == 0 and time.time() > deadline:
            break
        frac = k / max(1, steps - 1)
        T = T0 * math.exp(lratio * frac)
        if T < 5e-5:
            T = 5e-5
        lr_k = lr * (1.0 - 0.5 * frac)
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        E = np.exp((G - m) / T)
        np.fill_diagonal(E, 0.0)
        S = float(E.sum() * 0.5)
        if not np.isfinite(S) or S < 1e-300:
            S = 1e-300
        Grad = (E @ x) / S
        Grad -= (Grad * x).sum(axis=1, keepdims=True) * x
        vel = mu * vel + (1.0 - mu) * Grad
        x = x - lr_k * vel
        x /= np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
        if m < bv:
            bv = m
            bx = x.copy()
    return bx, bv

def _lp_polish(x, deadline, r0=0.004, cap=1300):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    try:
        best = _max_cos(x)
    except Exception:
        return x
    radius = r0
    iu0, iu1 = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        G = x @ x.T
        c = G[iu0, iu1]
        m = float(c.max())
        thresh = max(20.0 * radius, 1e-9)
        sel = c > m - thresh
        cnt = int(np.count_nonzero(sel))
        if cnt == 0:
            sel[int(np.argmax(c))] = True
            cnt = 1
        if cnt > cap:
            idx = np.argsort(-c)[:cap]
            sel2 = np.zeros_like(sel, dtype=bool)
            sel2[idx] = True
            sel = sel2
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
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv); obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        try:
            y = x + res.x[:-1].reshape(n, d)
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            v = _max_cos(y)
        except Exception:
            radius *= 0.5
            continue
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def _hopf_xyz(t, a, b):
    ct = math.cos(t); st = math.sin(t)
    return np.array([ct * math.cos(a), ct * math.sin(a), st * math.cos(b), st * math.sin(b)])

def _gen_torus_config(rng):
    n = 163
    K = int(rng.randint(3, 8))  # 3..7
    # torus positions: mix uniform-volume and spaced
    if rng.rand() < 0.5:
        u = np.sort(rng.rand(K))
        # avoid endpoints
        u = 0.03 + 0.94 * u
        ts = np.arcsin(np.sqrt(u))
    else:
        base = (np.arange(K) + 0.5) / K
        jit = (rng.rand(K) - 0.5) * (0.6 / K)
        u = np.clip(base + jit, 0.03, 0.97)
        ts = np.arcsin(np.sqrt(np.sort(u)))
    # allocation summing to 163
    raw = rng.rand(K) + 0.3
    raw = raw / raw.sum() * n
    ns = np.floor(raw).astype(int)
    ns = np.maximum(ns, 3)
    diff = n - int(ns.sum())
    # fix sum deterministically
    order = np.argsort(-(raw - ns))
    ii = 0
    while diff != 0:
        j = int(order[ii % K])
        if diff > 0:
            ns[j] += 1; diff -= 1
        else:
            if ns[j] > 3:
                ns[j] -= 1; diff += 1
        ii += 1
        if ii > 1000:
            break
    pts = np.zeros((n, 4))
    pos = 0
    for k in range(K):
        t = float(ts[k]); nk = int(ns[k])
        pat = int(rng.randint(0, 4))
        pa = float(rng.rand() * 2 * math.pi)
        pb = float(rng.rand() * 2 * math.pi)
        ct = math.cos(t); st = math.sin(t)
        if ct < 1e-3: ct = 1e-3
        if st < 1e-3: st = 1e-3
        A = np.zeros(nk); B = np.zeros(nk)
        if pat == 0 or pat == 1:
            cot = ct / max(st, 1e-6)
            p = int(round(math.sqrt(nk * cot))) if cot > 0 else int(round(math.sqrt(nk)))
            p = max(1, p); q = max(1, int(round(nk / max(p, 1))))
            # ensure p*q >= nk, grow if needed
            while p * q < nk:
                if p < q: p += 1
                else: q += 1
            aa = []; bb = []
            for i in range(p):
                for j in range(q):
                    a = 2 * math.pi * i / p + pa
                    b = 2 * math.pi * j / q + pb
                    if pat == 1:
                        a += (math.pi / p) * (j % 2)
                    aa.append(a); bb.append(b)
            aa = np.array(aa); bb = np.array(bb)
            if len(aa) > nk:
                idx = (np.linspace(0, len(aa) - 1, nk)).astype(int)
                aa = aa[idx]; bb = bb[idx]
            A = aa[:nk]; B = bb[:nk]
        elif pat == 2:
            # twisted line: steps coprime
            s = int(rng.randint(1, max(2, nk)))
            g = math.gcd(s, nk)
            while g != 1 and nk > 2:
                s = int(rng.randint(1, nk))
                g = math.gcd(s, nk)
                if s == 0:
                    s = 1; break
            if nk <= 1:
                s = 1
            for i in range(nk):
                A[i] = 2 * math.pi * i / nk + pa
                B[i] = 2 * math.pi * i * s / nk + pb
        else:
            A = rng.rand(nk) * 2 * math.pi
            B = rng.rand(nk) * 2 * math.pi
        for i in range(nk):
            pts[pos + i, 0] = ct * math.cos(A[i])
            pts[pos + i, 1] = ct * math.sin(A[i])
            pts[pos + i, 2] = st * math.cos(B[i])
            pts[pos + i, 3] = st * math.sin(B[i])
        pos += nk
    # shuffle deterministic to avoid ordering bias
    perm = rng.permutation(n)
    pts = pts[perm]
    return pts

def _build_hurwitz_pool(rng, L=5, n_rand=4000):
    seen = set()
    pool = []
    def add_vec(v):
        nv = float(np.linalg.norm(v))
        if nv == 0 or not np.isfinite(nv):
            return
        u = v / nv
        key = tuple(np.round(u, 8))
        if key not in seen:
            seen.add(key)
            pool.append(u)
    # integer quaternions
    rng_vals = range(-L, L + 1)
    for a in rng_vals:
        for b in rng_vals:
            for c in rng_vals:
                for d in rng_vals:
                    if a == 0 and b == 0 and c == 0 and d == 0:
                        continue
                    add_vec(np.array([a, b, c, d], dtype=float))
    # half-integer (Hurwitz)
    off = 0.5
    vals = [i + off for i in range(-L, L)]
    for a in vals:
        for b in vals:
            for c in vals:
                for d in vals:
                    add_vec(np.array([a, b, c, d], dtype=float))
    # random uniform filler
    R = rng.randn(n_rand, 4)
    R /= np.maximum(np.linalg.norm(R, axis=1, keepdims=True), 1e-12)
    for i in range(n_rand):
        # no dedup needed, but avoid exact dup
        pool.append(R[i])
    return np.array(pool, dtype=float)

def _greedy_select(pool, n, first=0):
    P = pool.shape[0]
    best_dot = np.full(P, -2.0, dtype=float)
    selected = []
    sel_mask = np.zeros(P, dtype=bool)
    cur = int(first % P)
    for _ in range(n):
        selected.append(cur)
        sel_mask[cur] = True
        best_dot[cur] = 2.0  # exclude
        # update best_dot with dots to cur
        d = pool @ pool[cur]
        # for already selected keep 2.0
        upd = d > best_dot
        upd[sel_mask] = False
        best_dot[upd] = d[upd]
        # also mark selected as excluded
        best_dot[sel_mask] = 2.0
        if len(selected) >= n:
            break
        # next: minimal best_dot among unselected
        # argmin over masked array
        tmp = best_dot.copy()
        tmp[sel_mask] = 2.0
        cur = int(np.argmin(tmp))
    return np.array(selected, dtype=int)

def _swap_improve(pool, sel, max_iter=30):
    sel = sel.copy()
    P = pool.shape[0]
    n = len(sel)
    in_set = np.zeros(P, dtype=bool)
    in_set[sel] = True
    X = pool[sel]
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    cur_max = float(G.max())
    for _ in range(max_iter):
        rowmax = G.max(axis=1)
        order = np.argsort(-rowmax)
        improved = False
        for w_pos in order[:8]:
            w = int(sel[w_pos])
            # max dot of each pool candidate to others excluding w_pos
            others = X[np.arange(n) != w_pos]
            dots = pool @ others.T
            md = dots.max(axis=1)
            md[in_set] = 2.0
            # also exclude current? in_set includes w, so w excluded; allow staying? if best >= cur_max no swap
            bi = int(np.argmin(md))
            bv = float(md[bi])
            if bv < cur_max - 1e-12:
                # swap
                in_set[w] = False
                in_set[bi] = True
                sel[w_pos] = bi
                X = pool[sel]
                G = X @ X.T
                np.fill_diagonal(G, -2.0)
                cur_max = float(G.max())
                improved = True
                break
        if not improved:
            break
    return sel, cur_max

def solve():
    t0 = time.time()
    HARD_END = t0 + 174.0
    rng = np.random.RandomState(0)
    n = 163
    # load incumbents
    cands = []
    try:
        xb = np.load("best.npy").astype(np.float64)
        xb = xb / np.linalg.norm(xb, axis=1, keepdims=True)
        if xb.shape == (163, 4):
            cands.append(xb)
    except Exception:
        pass
    try:
        xr = np.load("record.npy").astype(np.float64)
        xr = xr / np.linalg.norm(xr, axis=1, keepdims=True)
        if xr.shape == (163, 4):
            cands.append(xr)
    except Exception:
        pass
    if len(cands) == 0:
        tmp = rng.randn(n, 4)
        tmp /= np.linalg.norm(tmp, axis=1, keepdims=True)
        cands.append(tmp)
    best = min(cands, key=lambda z: _max_cos(z)).copy()
    best_val = _max_cos(best)
    # Phase A: structured torus search
    deadline_A = min(t0 + 38.0, HARD_END - 100.0)
    torus_raw_best = []
    try:
        gen_count = 0
        while time.time() < deadline_A - 4.0 and gen_count < 4000:
            for _ in range(50):
                cfg = _gen_torus_config(rng)
                v = _max_cos(cfg)
                torus_raw_best.append((v, cfg))
                gen_count += 1
            # keep only top 12 to bound memory
            torus_raw_best.sort(key=lambda t: t[0])
            torus_raw_best = torus_raw_best[:12]
            if time.time() > deadline_A - 4.0:
                break
    except Exception:
        pass
    # short anneal top torus configs
    torus_polished = []
    try:
        for v, cfg in list(torus_raw_best[:6]):
            if time.time() > deadline_A:
                break
            x2, v2 = _anneal(cfg, steps=220, T0=0.008, T1=0.0004, lr=0.25, mu=0.8, rng=rng, deadline=deadline_A)
            torus_polished.append((v2, x2))
            if v2 < best_val:
                best, best_val = x2.copy(), v2
    except Exception:
        pass
    # Phase B: Hurwitz pool + greedy + swaps
    deadline_B = min(t0 + 75.0, HARD_END - 65.0)
    hur_polished = []
    try:
        if time.time() < deadline_B - 5.0:
            pool = _build_hurwitz_pool(rng, L=5, n_rand=4000)
            # a few greedy starts
            starts = [0, pool.shape[0] // 3, pool.shape[0] // 2]
            for s in starts:
                if time.time() > deadline_B - 8.0:
                    break
                sel = _greedy_select(pool, n, first=s)
                sel2, vraw = _swap_improve(pool, sel, max_iter=25)
                X = pool[sel2]
                # short anneal
                Xa, va = _anneal(X, steps=220, T0=0.006, T1=0.0004, lr=0.25, mu=0.8, rng=rng, deadline=deadline_B)
                hur_polished.append((va, Xa))
                if va < best_val:
                    best, best_val = Xa.copy(), va
            # also try greedy on torus-polished union? add best points to pool for diversity
            del pool
    except Exception:
        pass
    # collect elites for final push
    elites = [(best_val, best.copy())]
    for v, x in torus_polished:
        elites.append((float(v), x.copy()))
    for v, x in hur_polished:
        elites.append((float(v), x.copy()))
    # dedup elites by value? keep 3 best unique
    elites.sort(key=lambda t: t[0])
    elites = elites[:3]
    cur = elites[0][1].copy()
    cur_val = float(elites[0][0])
    if best_val < cur_val:
        cur, cur_val = best.copy(), best_val
    # Phase C: anneal + LP on elites, round-robin
    POLISH_DEADLINE = HARD_END - 10.0
    it = 0
    while time.time() < POLISH_DEADLINE:
        it += 1
        # pick elite cyclically with perturbation
        base = elites[it % len(elites)][1]
        # perturb base or cur
        src = cur if rng.rand() < 0.6 else base
        sigma = 0.004 + rng.rand() * 0.02
        if it % 7 == 0:
            sigma = 0.05 + rng.rand() * 0.05
        x = src + rng.randn(n, 4) * sigma
        x /= np.linalg.norm(x, axis=1, keepdims=True)
        T0 = 0.004 + rng.rand() * 0.008
        T1 = 0.0003 + rng.rand() * 0.0007
        lr = 0.2 + rng.rand() * 0.2
        mu = 0.75 + rng.rand() * 0.15
        steps = 180 + int(rng.randint(0, 160))
        x, v_ann = _anneal(x, steps=steps, T0=T0, T1=T1, lr=lr, mu=mu, rng=rng, deadline=POLISH_DEADLINE)
        if time.time() > POLISH_DEADLINE:
            if v_ann < best_val:
                best, best_val = x.copy(), v_ann
            break
        # short LP
        x = _lp_polish(x, deadline=min(time.time() + 1.2, POLISH_DEADLINE), r0=0.004 if (it % 3) else 0.007)
        v = _max_cos(x)
        if v < best_val - 1e-12:
            best, best_val = x.copy(), v
            cur, cur_val = x.copy(), v
            elites.append((v, x.copy()))
            elites.sort(key=lambda t: t[0])
            elites = elites[:3]
        elif v < cur_val:
            cur, cur_val = x.copy(), v
        else:
            Temp = 0.0008 * (0.995 ** (it // 2)) + 5e-5
            if rng.rand() < math.exp(-(v - cur_val) / Temp) * 0.4:
                cur, cur_val = x.copy(), v
    # final LP refinement of best
    remain = HARD_END - time.time() - 0.6
    if remain > 2.0:
        best = _lp_polish(best, deadline=HARD_END - 0.5, r0=0.003)
        best_val = _max_cos(best)
        if remain > 6.0:
            best = _lp_polish(best, deadline=HARD_END - 0.3, r0=0.0008)
    best = best / np.linalg.norm(best, axis=1, keepdims=True)
    return best
