import time
import numpy as np

def _norm(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return x / n

def _max_cos(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def _degrees_rowmax(x, tol=1e-6):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    m = float(np.max(G))
    deg = np.sum(G > (m - tol), axis=1)
    rowmax = np.max(G, axis=1)
    return deg, rowmax, m, G

def _hash_key(x, tol=5e-7):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    m = float(np.max(G))
    deg = np.sum(G > (m - tol), axis=1)
    return (tuple(sorted(deg.tolist())), round(float(m) * 1e7))

def _lp_polish(x, deadline, cap=2500, r0=8e-4, maxit=150):
    from scipy.optimize import linprog
    from scipy.sparse import coo_matrix
    x = _norm(x)
    n, d = x.shape
    best = _max_cos(x)
    best_x = x.copy()
    radius = r0
    iu0 = np.triu_indices(n, 1)
    it = 0
    while time.time() < deadline and radius > 1e-14 and it < maxit:
        it += 1
        G = x @ x.T
        c = G[iu0]
        m = float(c.max())
        thresh = m - max(20.0 * radius, 1e-9)
        idx = np.where(c > thresh)[0]
        if idx.size == 0:
            radius = min(radius * 1.5, 0.05)
            continue
        if idx.size > cap:
            order = np.argpartition(c[idx], -cap)[-cap:]
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
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        except Exception:
            radius *= 0.5
            continue
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
            best = v; best_x = y.copy(); x = y
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
            if _max_cos(x) > best + 1e-15:
                x = best_x.copy()
    return best_x, best

def _refine_holes_vec(C0, Y, tau=0.01, steps=140, lr0=0.25, lr1=0.01, mom=0.85):
    # C0: (k,4) unit, Y: (m,4) unit; minimize smooth max tau*logsumexp(dots/tau)
    C = C0.copy()
    k = C.shape[0]
    vel = np.zeros_like(C)
    lt0, lt1 = np.log(lr0), np.log(lr1)
    for t in range(steps):
        f = t / max(1, steps - 1)
        lr = float(np.exp(lt0 + (lt1 - lt0) * f))
        D = C @ Y.T  # k x m
        mrow = np.max(D, axis=1, keepdims=True)
        E = np.exp((D - mrow) / tau)
        s = np.sum(E, axis=1, keepdims=True)
        s[s < 1e-14] = 1e-14
        W = E / s
        grad = W @ Y  # k x 4
        dot = np.sum(grad * C, axis=1, keepdims=True)
        grad = grad - dot * C
        # clip
        gn = np.linalg.norm(grad, axis=1, keepdims=True)
        scale = np.ones_like(gn)
        big = gn > 2.0
        scale[big] = 2.0 / np.maximum(gn[big], 1e-12)
        grad = grad * scale
        vel = mom * vel + (1.0 - mom) * grad
        bc = 1.0 - mom ** (t + 1)
        C = C - lr * (vel / bc)
        C = _norm(C)
    return C

def _greedy_insert_one(Y, Rem, rng, n_rand=30000, n_top=20):
    d = Y.shape[1]
    m = Y.shape[0]
    # random candidates
    Cr = rng.standard_normal(size=(n_rand, d))
    Cr = _norm(Cr)
    # jittered copies of removed points
    nj = 3000
    if Rem is not None and len(Rem) > 0:
        pick = rng.integers(0, len(Rem), size=nj)
        base = Rem[pick]
        sigmas = np.array([0.02, 0.06, 0.15], dtype=float)
        schoice = sigmas[rng.integers(0, 3, size=nj)]
        noise = rng.standard_normal(size=(nj, d)) * schoice[:, None]
        Cj = _norm(base + noise)
        # also include exact removed points (in case hole is old spot)
        Call = np.vstack([Cr, Cj, _norm(Rem.copy())])
    else:
        Call = Cr
    # score by max cosine to survivors, batched
    best_idx = None
    best_vals = None
    # process in batches to bound memory
    batch = 11000
    # we need top-20 overall; keep running top
    # first compute all sm in chunks and collect
    sms = np.empty(Call.shape[0], dtype=float)
    for s in range(0, Call.shape[0], batch):
        e = min(Call.shape[0], s + batch)
        Cb = Call[s:e]
        D = Cb @ Y.T
        sms[s:e] = np.max(D, axis=1)
    # top 20 smallest
    k = min(n_top, Call.shape[0])
    part = np.argpartition(sms, k - 1)[:k]
    # sort them by sms for stability
    part = part[np.argsort(sms[part])]
    Ctop = Call[part]
    # refine top-20 by smooth-max tangent descent
    try:
        Cref = _refine_holes_vec(Ctop, Y, tau=0.01, steps=140, lr0=0.25, lr1=0.01, mom=0.85)
    except Exception:
        Cref = Ctop
    # exact re-score after refinement
    D2 = Cref @ Y.T
    sm2 = np.max(D2, axis=1)
    bi = int(np.argmin(sm2))
    return Cref[bi].copy(), float(sm2[bi])

def _smooth_free(x, free_idx, steps=200, tau0=0.012, tau1=0.0015, lr0=0.10, lr1=0.006, mom=0.9):
    n = x.shape[0]
    vel = np.zeros_like(x)
    lt0, lt1 = np.log(tau0), np.log(tau1)
    ll0, ll1 = np.log(lr0), np.log(lr1)
    fmask = np.zeros(n, dtype=bool)
    fmask[np.asarray(free_idx, dtype=int)] = True
    for t in range(steps):
        f = t / max(1, steps - 1)
        tau = float(np.exp(lt0 + (lt1 - lt0) * f))
        lr = float(np.exp(ll0 + (ll1 - ll0) * f))
        G = x @ x.T
        np.fill_diagonal(G, -np.inf)
        m = float(np.max(G))
        W = np.exp((G - m) / tau)
        np.fill_diagonal(W, 0.0)
        row = np.sum(W, axis=1, keepdims=True)
        row[row < 1e-12] = 1e-12
        grad = (W @ x) / row
        dot = np.sum(grad * x, axis=1, keepdims=True)
        grad = grad - dot * x
        gn = np.linalg.norm(grad, axis=1, keepdims=True)
        big = gn > 2.0
        if np.any(big):
            grad = grad * np.where(big, 2.0 / np.maximum(gn, 1e-12), 1.0)
        grad[~fmask] = 0.0
        vel = mom * vel + (1.0 - mom) * grad
        bc = 1.0 - mom ** (t + 1)
        x = x - lr * (vel / bc)
        x = _norm(x)
    return x

def solve():
    t0 = time.time()
    HARD_END = t0 + 174.0
    EXPLORE_END = HARD_END - 28.0
    rng = np.random.default_rng(20261031)
    n, d = 153, 4
    best_x = None; best_v = 1e9
    for fname in ("best.npy", "record.npy"):
        try:
            a = np.load(fname)
            a = np.asarray(a, dtype=float)
            if a.shape == (n, d):
                a = _norm(a)
                if np.all(np.isfinite(a)):
                    v = _max_cos(a)
                    if v < best_v:
                        best_v = v; best_x = a.copy()
        except Exception:
            pass
    if best_x is None:
        best_x = _norm(rng.standard_normal(size=(n, d)))
        best_v = _max_cos(best_x)
    global_best_x = best_x.copy()
    global_best_v = best_v
    seen = set()
    seen.add(_hash_key(global_best_x))
    pool = [(global_best_v, global_best_x.copy())]
    r_choices = [1, 2, 3, 5]
    trial = 0
    while time.time() < EXPLORE_END:
        trial += 1
        if time.time() > EXPLORE_END - 3.0:
            break
        # choose base
        try:
            if len(pool) > 1 and rng.random() < 0.30:
                k = int(rng.integers(len(pool)))
                base = pool[k][1].copy()
            else:
                base = global_best_x.copy()
            deg, rowmax, mcur, _ = _degrees_rowmax(base, tol=1e-6)
            # low-degree set deg<=4
            low = np.where(deg <= 4)[0]
            if low.size == 0:
                low = np.argsort(deg)[:8]
            # find degree-2 (minimum) point
            mindeg = int(np.min(deg))
            deg2pts = np.where(deg == mindeg)[0]
            # choose r
            r = int(r_choices[int(rng.integers(len(r_choices)))])
            r = min(r, n - 10)
            # build delete set: sample r from low, bias to include deg-2 point
            del_set = set()
            if deg2pts.size > 0 and rng.random() < 0.7:
                del_set.add(int(deg2pts[int(rng.integers(len(deg2pts)))]))
            # order low by rowmax descending (attack worst) with randomness
            # random sample among low
            # if low larger than needed, sample weighted by rowmax
            need = r - len(del_set)
            if need > 0:
                cand = [int(v) for v in low if v not in del_set]
                if len(cand) < need:
                    # supplement with lowest-degree order
                    order = np.argsort(deg + rowmax * 0.1)
                    for v in order:
                        v = int(v)
                        if v not in del_set and v not in cand:
                            cand.append(v)
                        if len(cand) >= need:
                            break
                if len(cand) > need:
                    # weighted sample: prefer higher rowmax
                    rw = rowmax[np.array(cand)]
                    # softmax weighting
                    rw = rw - np.min(rw)
                    w = 1.0 + 5.0 * (rw / (np.max(rw) + 1e-12))
                    w = w / np.sum(w)
                    chosen = set()
                    # sample without replacement
                    probs = w.copy()
                    idxs = list(range(len(cand)))
                    for _ in range(need):
                        s = rng.random()
                        cum = np.cumsum(probs)
                        tot = cum[-1]
                        ii = int(np.searchsorted(cum, s * tot))
                        ii = min(ii, len(cand) - 1)
                        # map to actual remaining index (handle removed)
                        # simple: pick by probs over original, skip chosen
                        # retry if already chosen
                        tries = 0
                        while ii in chosen and tries < 20:
                            ii = int(rng.integers(len(cand)))
                            tries += 1
                        chosen.add(ii)
                        probs[ii] = 0.0
                        ps = probs.sum()
                        if ps > 0:
                            probs = probs / ps
                        else:
                            break
                    for ii in chosen:
                        del_set.add(int(cand[ii]))
                else:
                    for v in cand[:need]:
                        del_set.add(int(v))
            del_idx = np.array(sorted(del_set), dtype=int)
            if del_idx.size != r:
                # adjust
                if del_idx.size > r:
                    del_idx = del_idx[:r]
                else:
                    # fill random
                    while del_idx.size < r:
                        v = int(rng.integers(n))
                        if v not in del_set:
                            del_set.add(v)
                            del_idx = np.array(sorted(del_set), dtype=int)
            keep_mask = np.ones(n, dtype=bool)
            keep_mask[del_idx] = False
            Y = base[keep_mask].copy()
            Rem = base[del_idx].copy()
            # greedy reinsertion one at a time
            Yg = Y
            for _k in range(r):
                if time.time() > EXPLORE_END:
                    break
                cnew, _s = _greedy_insert_one(Yg, Rem, rng, n_rand=30000, n_top=20)
                Yg = np.vstack([Yg, cnew.reshape(1, -1)])
            if Yg.shape[0] != n:
                continue
            if time.time() > EXPLORE_END:
                break
            # light smooth on inserted points only
            free_idx = list(range(n - r, n))
            try:
                Yg = _smooth_free(Yg, free_idx, steps=180, tau0=0.012, tau1=0.0015, lr0=0.10, lr1=0.006, mom=0.9)
            except Exception:
                pass
            if time.time() > EXPLORE_END:
                break
            budget = 1.2
            dl = min(time.time() + budget, EXPLORE_END + 1.0)
            if dl <= time.time():
                continue
            try:
                Yp, vp = _lp_polish(Yg, dl, cap=2200, r0=8e-4, maxit=120)
            except Exception:
                continue
            hk = _hash_key(Yp)
            is_new = hk not in seen
            seen.add(hk)
            if vp < global_best_v - 1e-12:
                global_best_v = vp
                global_best_x = Yp.copy()
                pool.append((vp, Yp.copy()))
                pool = sorted(pool, key=lambda z: z[0])[:12]
            else:
                if is_new and vp < global_best_v + 2e-3:
                    pool.append((vp, Yp.copy()))
                    pool = sorted(pool, key=lambda z: z[0])[:12]
                    if len(seen) > 6000:
                        seen = set([_hash_key(p[1]) for p in pool])
        except Exception:
            continue
    try:
        dl = HARD_END - 1.0
        if time.time() < dl:
            global_best_x, global_best_v = _lp_polish(global_best_x, dl, cap=3000, r0=1e-3, maxit=300)
    except Exception:
        pass
    global_best_x = _norm(global_best_x)
    return global_best_x
