import time
import itertools
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _nm(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return x / n

def _mc(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(np.max(g))

def _rand_orth(rng):
    A = rng.normal(size=(4, 4))
    Q, R = np.linalg.qr(A)
    d = np.diag(R)
    s = np.sign(d)
    s[s == 0] = 1.0
    Q = Q * s[None, :]
    if np.linalg.det(Q) < 0:
        Q[:, 0] *= -1
    return Q

def _base_D4a():
    pts = []
    sq = 1.0 / np.sqrt(2.0)
    for i in range(4):
        for j in range(i + 1, 4):
            for si in (-1.0, 1.0):
                for sj in (-1.0, 1.0):
                    v = np.zeros(4)
                    v[i] = si * sq
                    v[j] = sj * sq
                    pts.append(v)
    return np.stack(pts, 0)  # 24

def _base_D4b():
    pts = []
    for i in range(4):
        for s in (-1.0, 1.0):
            v = np.zeros(4); v[i] = s
            pts.append(v)
    for b in range(16):
        v = np.ones(4) * 0.5
        for i in range(4):
            if (b >> i) & 1:
                v[i] = -0.5
        pts.append(v)
    return np.stack(pts, 0)  # 24

def _base_600():
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    a = phi / 2.0
    b = 0.5
    c = 1.0 / (2.0 * phi)
    pts = []
    for i in range(4):
        for s in (-1.0, 1.0):
            v = np.zeros(4); v[i] = s
            pts.append(v)
    for bits in range(16):
        v = np.ones(4) * 0.5
        for i in range(4):
            if (bits >> i) & 1:
                v[i] = -0.5
        pts.append(v)
    base = np.array([a, b, c, 0.0])
    # even permutations of 4 elements (12)
    for perm in itertools.permutations(range(4)):
        inv = 0
        for i in range(4):
            for j in range(i + 1, 4):
                if perm[i] > perm[j]:
                    inv += 1
        if inv % 2 == 1:
            continue
        # perm defines which base-entry goes to position p:
        # tuple t[p] = base[perm[p]]? need all distinct; use inverse mapping
        # simpler: t = base[list(perm)] gives 12 distinct even perms
        t = base[list(perm)]
        # find zero position
        zpos = int(np.argmin(np.abs(t)))
        # signs on non-zero entries
        nz = [i for i in range(4) if i != zpos]
        for sb in range(8):
            v = t.copy()
            for k, idx in enumerate(nz):
                if (sb >> k) & 1:
                    v[idx] = -v[idx]
            pts.append(v)
    pts = np.stack(pts, 0)
    # dedup (should be 120)
    # round dedup
    uniq = []
    seen = set()
    for v in pts:
        key = tuple(np.round(v, 8))
        if key not in seen:
            seen.add(key)
            uniq.append(v)
    U = np.stack(uniq, 0)
    # if not 120 due to parity convention, fallback pad/truncate by sampling
    if U.shape[0] != 120:
        # keep what we have, pad with random rotations of itself is handled outside
        pass
    return _nm(U)

def _lp_polish(x, deadline):
    x = _nm(x.copy())
    n, d = x.shape
    try:
        best = _mc(x)
    except Exception:
        return x
    rad = 1e-3
    iu = np.triu_indices(n, 1)
    while time.time() < deadline and rad > 1e-14:
        c = (x @ x.T)[iu]
        m = float(np.max(c))
        sel = c > m - max(20 * rad, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        if p == 0:
            break
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            Aub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            Aeq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv); obj[-1] = 1.0
            r = linprog(obj, A_ub=Aub, b_ub=-c[sel], A_eq=Aeq, b_eq=np.zeros(n),
                        bounds=[(-rad, rad)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            break
        if r.status != 0:
            rad *= 0.5
            continue
        y = _nm(x + r.x[:-1].reshape(n, d))
        v = _mc(y)
        if v < best:
            x, best, rad = y, v, min(rad * 1.5, 0.05)
        else:
            rad *= 0.5
    return x

def _adam_full(X, steps, b0=30., b1=420., lr0=0.02, lr1=0.002):
    n, d = X.shape
    m = np.zeros_like(X); v = np.zeros_like(X)
    b1a, b2a, eps = 0.9, 0.999, 1e-8
    lb0, lb1 = np.log(b0), np.log(b1)
    ll0, ll1 = np.log(lr0), np.log(lr1)
    for t in range(1, steps + 1):
        f = t / steps
        beta = float(np.exp(lb0 + (lb1 - lb0) * f))
        lr = float(np.exp(ll0 + (ll1 - ll0) * f))
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        gm = float(np.max(G))
        W = np.exp(np.clip(beta * (G - gm), -20, 0))
        np.fill_diagonal(W, 0.0)
        Gr = (W @ X) * beta
        Gt = Gr - np.sum(Gr * X, axis=1, keepdims=True) * X
        m = b1a * m + (1 - b1a) * Gt
        v = b2a * v + (1 - b2a) * (Gt * Gt)
        mh = m / (1 - b1a ** t); vh = v / (1 - b2a ** t)
        X = _nm(X - lr * mh / (np.sqrt(vh) + eps))
    return X

def solve():
    t0 = time.time()
    HARD = t0 + 170.0
    SEARCH = t0 + 120.0
    POLISH_END = t0 + 162.0
    rng = np.random.default_rng(12345)
    # incumbent
    cands = []
    for f in ("best.npy", "record.npy"):
        try:
            c = np.load(f).astype(float)
            if c.shape == (168, 4):
                c = _nm(c)
                if np.all(np.isfinite(c)):
                    cands.append(c)
        except Exception:
            pass
    if cands:
        vs = [_mc(c) for c in cands]
        xb = cands[int(np.argmin(vs))].copy()
        bv = float(min(vs))
    else:
        xb = _nm(rng.normal(size=(168, 4))); bv = _mc(xb)
    # ---- build algebraic pool ----
    try:
        D4a = _base_D4a()
        D4b = _base_D4b()
        C120 = _base_600()
    except Exception:
        return _nm(xb)
    # optimize one relative rotation for pair of 600-cells
    try:
        A = C120
        bestR = np.eye(4)
        best_cross = 10.0
        for _ in range(120):
            R = _rand_orth(rng)
            B = A @ R.T
            cr = float(np.max(A @ B.T))
            if cr < best_cross:
                best_cross = cr
                bestR = R.copy()
        Bopt = A @ bestR.T
    except Exception:
        Bopt = C120
        bestR = np.eye(4)
    pool_list = [D4a, D4b, C120, Bopt]
    n120, n24 = 13, 18
    try:
        for _ in range(n120):
            R = _rand_orth(rng)
            pool_list.append(C120 @ R.T)
        for _ in range(n24):
            R = _rand_orth(rng)
            pool_list.append(D4a @ R.T)
        for _ in range(n24):
            R = _rand_orth(rng)
            pool_list.append(D4b @ R.T)
    except Exception:
        pass
    try:
        P = _nm(np.vstack(pool_list))
    except Exception:
        return _nm(xb)
    # cap size to keep Gram tractable (~3200)
    N = P.shape[0]
    if N > 3200:
        idx = rng.choice(N, size=3200, replace=False)
        P = P[np.sort(idx)]
        N = 3200
    # dedup ultra-close points (cos>0.999999) to save space
    try:
        Gfull = P @ P.T
    except Exception:
        return _nm(xb)
    # thresholds binary-searched around 0.850-0.8593
    thresholds = [0.8593, 0.8580, 0.8560, 0.8540, 0.8520, 0.8500]
    # adjust top threshold to just below incumbent to force improvement attempt,
    # but also keep incumbent level for certification
    # keep as is; feasibility at 0.8593 ~ incumbent value
    best_disc = None
    best_disc_t = None
    best_disc_size = -1
    # precompute degrees order helper
    rng2 = np.random.default_rng(999)
    for ti, thr in enumerate(thresholds):
        if time.time() > SEARCH:
            break
        try:
            Adj = Gfull > thr
        except Exception:
            continue
        deg = np.sum(Adj, axis=1)
        # trials budget per threshold adapts to remaining time
        n_trials = 28 if N < 3000 else 22
        if ti >= 3:
            n_trials = 16
        for trial in range(n_trials):
            if time.time() > SEARCH:
                break
            # permutation: mostly random, sometimes degree-biased
            if trial == 0:
                perm = np.argsort(deg, kind="stable")
            elif trial % 7 == 1:
                noise = rng2.normal(size=N) * 0.5
                perm = np.argsort(deg + noise, kind="stable")
            else:
                perm = rng2.permutation(N)
            blocked = np.zeros(N, dtype=bool)
            sel = []
            sel_mask = np.zeros(N, dtype=bool)
            # greedy
            # local loop in python; use Adj rows
            for vv in perm:
                if not blocked[vv]:
                    sel.append(int(vv))
                    sel_mask[vv] = True
                    blocked |= Adj[vv]
                    if len(sel) >= 168:
                        break
                # early exit: even if we take all unblocked we can't reach 168?
                # skip for speed
            # 1-for-2 local improvement (only if we need growth and maximal)
            # run up to a few swaps
            nswap = 0
            for _pass in range(6):
                if time.time() > SEARCH:
                    break
                if len(sel) >= 168:
                    break
                s = len(sel)
                if s == 0:
                    break
                Sarr = np.array(sel, dtype=int)
                # sub matrix N x s
                try:
                    sub = Gfull[:, Sarr] > thr
                except Exception:
                    break
                cnt = np.sum(sub, axis=1)
                # maximality: if any cnt==0 outside, just add it
                free = np.where((cnt == 0) & (~sel_mask))[0]
                if free.size > 0:
                    for f in free[:8]:
                        sel.append(int(f)); sel_mask[int(f)] = True
                        blocked |= Adj[int(f)]
                        if len(sel) >= 168:
                            break
                    # recompute next pass
                    continue
                # try 1-for-2
                improved = False
                # order u by smallest candidate pool first (randomized)
                order = rng2.permutation(s)
                for jj in order:
                    u = int(Sarr[jj])
                    # candidates whose sole neighbor is u
                    # mask: cnt==1 and sub[:,jj]
                    cand = np.where((cnt == 1) & sub[:, jj] & (~sel_mask))[0]
                    if cand.size < 2:
                        continue
                    # cap candidate size for speed
                    if cand.size > 40:
                        cand = rng2.choice(cand, size=40, replace=False)
                    # find nonadjacent pair
                    found = None
                    # try greedy: shuffle cand
                    rng2.shuffle(cand)
                    for ai in range(len(cand)):
                        a = int(cand[ai])
                        # row of a vs other cands
                        # non-neighbors of a within cand
                        # use Gfull
                        row = Gfull[a, cand] > thr
                        # row includes self (True); need False entries
                        # find first b with False
                        for bi in range(ai + 1, len(cand)):
                            b = int(cand[bi])
                            if not (Gfull[a, b] > thr):
                                found = (a, b)
                                break
                        if found is not None:
                            break
                    if found is not None:
                        a, b = found
                        # swap: remove u, add a,b
                        sel_mask[u] = False
                        sel_mask[a] = True; sel_mask[b] = True
                        # rebuild sel list
                        sel = [x for x in sel if x != u] + [a, b]
                        nswap += 1
                        improved = True
                        break
                if not improved:
                    break
            # optional 2-for-3 attempt if close (size>=150) and still <168, one shot
            if 150 <= len(sel) < 168 and time.time() < SEARCH - 2:
                try:
                    Sarr = np.array(sel, dtype=int)
                    sub = Gfull[:, Sarr] > thr
                    cnt = np.sum(sub, axis=1)
                    # try a few random pairs
                    for _a in range(30):
                        j1, j2 = rng2.choice(len(Sarr), size=2, replace=False)
                        u1, u2 = int(Sarr[j1]), int(Sarr[j2])
                        # candidates whose neighbors subset of {u1,u2}
                        mask = (cnt <= 2) & (np.sum(sub[:, [j1, j2]], axis=1) == cnt) & (~sel_mask) & (cnt > 0)
                        # also include cnt==0? those would be free (already none)
                        cand = np.where(mask)[0]
                        if cand.size < 3:
                            continue
                        if cand.size > 30:
                            cand = rng2.choice(cand, size=30, replace=False)
                        # brute search triple mutually independent
                        found3 = None
                        for ia in range(len(cand)):
                            for ib in range(ia + 1, len(cand)):
                                if Gfull[int(cand[ia]), int(cand[ib])] > thr:
                                    continue
                                for ic in range(ib + 1, len(cand)):
                                    if Gfull[int(cand[ia]), int(cand[ic])] > thr:
                                        continue
                                    if Gfull[int(cand[ib]), int(cand[ic])] > thr:
                                        continue
                                    found3 = (int(cand[ia]), int(cand[ib]), int(cand[ic]))
                                    break
                                if found3 is not None:
                                    break
                            if found3 is not None:
                                break
                        if found3 is not None:
                            a, b, c_ = found3
                            sel_mask[u1] = False; sel_mask[u2] = False
                            sel_mask[a] = True; sel_mask[b] = True; sel_mask[c_] = True
                            sel = [x for x in sel if x != u1 and x != u2] + [a, b, c_]
                            break
                except Exception:
                    pass
            sz = len(sel)
            if sz > best_disc_size:
                best_disc_size = sz
                # store actual points for best threshold that gives max size?
                # prefer smaller thr if feasible; handle below
            if sz >= 168:
                # feasible at thr: keep smallest thr feasible (thr decreasing, so overwrite)
                # store truncated 168
                pts168 = P[np.array(sel[:168], dtype=int)]
                # quick certify
                if _mc(pts168) <= thr + 1e-12:
                    best_disc = pts168.copy()
                    best_disc_t = thr
                    # don't break: try smaller thr
                    break
        # if feasible at this thr, continue to smaller thr; else if not feasible
        # and thr is high, continue anyway (smaller thr harder) – actually if high
        # thr infeasible, smaller will also be infeasible, so we could stop?
        # But greedy is heuristic, so continue a bit. We stop only if time out.
        if best_disc is not None and best_disc_t is not None:
            # we found feasible at thr; next thr smaller is harder, keep trying
            pass
    # ---- complete / polish discrete candidates ----
    x_best = xb.copy(); v_best = bv
    # helper to complete a discrete set to 168 via deepest voids (continuous)
    def complete_and_polish(Psel_pts, deadline):
        nonlocal x_best, v_best
        cur = Psel_pts.copy()
        if cur.shape[0] > 168:
            cur = cur[:168]
        if cur.shape[0] < 168:
            need = 168 - cur.shape[0]
            # greedy deepest-void insertion from random samples
            for _ in range(need):
                if time.time() > deadline - 5:
                    break
                S = _nm(rng2.normal(size=(20000, 4)))
                D = S @ cur.T
                mx = D.max(axis=1)
                bi = int(np.argmin(mx))
                cur = np.vstack([cur, S[bi:bi + 1]])
            # local refine each inserted point once (single-point LSE vs rest)
            # skip for speed
        # LSE polish
        try:
            steps = 700 if time.time() < deadline - 25 else 300
            cur2 = _adam_full(_nm(cur), steps=steps)
            v2 = _mc(cur2)
            if v2 < v_best and time.time() < POLISH_END:
                # LP squeeze
                cur3 = _lp_polish(cur2, min(deadline, POLISH_END - 1))
                v3 = _mc(cur3)
                if v3 < v_best:
                    x_best = cur3.copy(); v_best = v3
                elif v2 < v_best:
                    x_best = cur2.copy(); v_best = v2
        except Exception:
            pass
    if best_disc is not None:
        try:
            v0 = _mc(best_disc)
            # already certifies <= thr; polish it
            complete_and_polish(best_disc, POLISH_END - 8)
        except Exception:
            pass
    # if discrete never reached 168, try completing largest independent set found
    # re-run one greedy at high thr to get largest set for completion (if time)
    if best_disc is None and time.time() < SEARCH + 10 and time.time() < POLISH_END - 30:
        try:
            thr = 0.8593
            Adj = Gfull > thr
            deg = np.sum(Adj, axis=1)
            best_sel = None; best_sz = -1
            for _ in range(12):
                perm = rng2.permutation(N)
                blocked = np.zeros(N, dtype=bool)
                sel = []
                for vv in perm:
                    if not blocked[vv]:
                        sel.append(int(vv))
                        blocked |= Adj[vv]
                        if len(sel) >= 168:
                            break
                if len(sel) > best_sz:
                    best_sz = len(sel); best_sel = sel
                if best_sz >= 168:
                    break
            if best_sel is not None:
                pts = P[np.array(best_sel, dtype=int)]
                if pts.shape[0] > 168:
                    pts = pts[:168]
                complete_and_polish(pts, POLISH_END - 8)
        except Exception:
            pass
    # fallback: a few basin kicks from incumbent if discrete didn't win (ensure we use remaining time)
    if time.time() < POLISH_END - 12:
        try:
            X0 = _nm(x_best + 0.02 * rng2.normal(size=(168, 4)))
            Xo = _adam_full(X0, steps=400)
            v = _mc(Xo)
            if v < v_best:
                v_best = v; x_best = Xo.copy()
        except Exception:
            pass
    # final LP squeeze on best
    if time.time() < HARD - 2:
        try:
            x_best = _lp_polish(x_best, HARD - 1.0)
            v_best = _mc(x_best)
        except Exception:
            pass
    # safety: return incumbent if worse
    if v_best > bv + 1e-12:
        return _nm(xb)
    return _nm(x_best)
