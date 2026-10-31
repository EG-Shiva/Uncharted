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

def _deg_info(x, tol=5e-7):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    m = float(np.max(G))
    B = G > (m - tol)
    deg = np.sum(B, axis=1).astype(int)
    return deg, m

def _deg_hash(x, tol=5e-7):
    deg, m = _deg_info(x, tol=tol)
    return (tuple(sorted(deg.tolist())), round(float(m) * 1e7))

def _lp_polish(x, deadline, r0=1e-3, cap=2500, maxit=200):
    from scipy.optimize import linprog
    from scipy.sparse import coo_matrix
    x = _norm(np.asarray(x, dtype=float))
    n, d = x.shape
    best = _max_cos(x)
    best_x = x.copy()
    radius = float(r0)
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
            o = np.argpartition(c[idx], -cap)[-cap:]
            idx = idx[o]
        I = iu0[0][idx]; J = iu0[1][idx]
        csel = c[idx]
        p = idx.size
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cI = I[:, None] * d + np.arange(d)
        cJ = J[:, None] * d + np.arange(d)
        try:
            cols = np.concatenate([cI, cJ, np.full((p, 1), n * d)], axis=1).ravel()
            vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], axis=1).ravel()
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
        try:
            y = _norm(x + res.x[:-1].reshape(n, d))
        except Exception:
            radius *= 0.5
            continue
        v = _max_cos(y)
        if v < best:
            x = y; best = v; best_x = y.copy()
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return best_x, best

def _refine_holes(H, S, p=100.0, steps=120, lr=0.12):
    # H: (Hn,4) holes, S: (m,4) survivors. tangent descent on LSE smooth max.
    H = _norm(H.copy())
    for _ in range(steps):
        D = H @ S.T  # Hn x m
        dm = np.max(D, axis=1, keepdims=True)
        E = np.exp(p * (D - dm))
        s = np.sum(E, axis=1, keepdims=True)
        s[s == 0] = 1.0
        W = E / s
        G = W @ S  # Hn x 4
        dot = np.sum(G * H, axis=1, keepdims=True)
        Gt = G - dot * H
        H = _norm(H - lr * Gt)
    return H

def _find_hole(S, rng, n_rand=30000, n_jit=2000, jit_src=None, jit_sigma=0.08, n_best=20):
    d = S.shape[1]
    C = rng.standard_normal(size=(n_rand, d))
    C = _norm(C)
    if jit_src is not None and len(jit_src) > 0:
        J = np.repeat(np.asarray(jit_src, dtype=float), (n_jit // max(1, len(jit_src))) + 1, axis=0)[:n_jit]
        J = J + jit_sigma * rng.standard_normal(size=J.shape)
        J = _norm(J)
        C = np.vstack([C, J])
    # score by max cosine to survivors (lower = better hole)
    # chunk to save memory
    K = C.shape[0]
    chunk = 8000
    scores = np.empty(K, dtype=float)
    for a in range(0, K, chunk):
        b = min(K, a + chunk)
        scores[a:b] = np.max(C[a:b] @ S.T, axis=1)
    if n_best >= K:
        bi = np.argsort(scores)
    else:
        bi = np.argpartition(scores, n_best)[:n_best]
        bi = bi[np.argsort(scores[bi])]
    H0 = C[bi]
    # refine
    try:
        Hr = _refine_holes(H0, S, p=100.0, steps=120, lr=0.12)
    except Exception:
        Hr = H0
    # pick best refined by true max
    try:
        rs = np.max(Hr @ S.T, axis=1)
        k = int(np.argmin(rs))
        return Hr[k]
    except Exception:
        return H0[0]

def solve():
    t0 = time.time()
    HARD_END = t0 + 172.0
    LOOP_END = t0 + 148.0
    n, d = 153, 4
    best_x = None; best_v = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
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
        rng0 = np.random.default_rng(0)
        best_x = _norm(rng0.standard_normal((n, d)))
        best_v = _max_cos(best_x)
    global_x = best_x.copy(); global_v = best_v
    seen = set()
    try:
        seen.add(_deg_hash(global_x))
    except Exception:
        pass
    # short initial polish to lock basin
    try:
        if time.time() < LOOP_END - 10:
            dl = min(time.time() + 6.0, LOOP_END - 5.0)
            if dl > time.time() + 1.0:
                global_x, global_v = _lp_polish(global_x, dl, r0=1e-3)
                try:
                    seen.add(_deg_hash(global_x))
                except Exception:
                    pass
    except Exception:
        pass

    def low_pool(x):
        try:
            deg, _ = _deg_info(x)
        except Exception:
            return np.arange(n)[:10]
        idx = np.argsort(deg, kind="stable")
        # all deg<=4 first
        low = [int(i) for i in idx if int(deg[i]) <= 4]
        if len(low) < 5:
            low = [int(i) for i in idx[:max(15, 5)]]
        return np.array(low, dtype=int), deg

    pool, deg = low_pool(global_x)
    alt_bases = []  # list of (x, v)
    rs_cycle = [1, 2, 3, 5, 1, 2, 3, 1, 5, 2]
    trial = -1
    while time.time() < LOOP_END:
        trial += 1
        if time.time() > HARD_END - 25:
            break
        r = rs_cycle[trial % len(rs_cycle)]
        # choose base: mostly global, sometimes alt for diversity
        if alt_bases and (trial % 6 == 5):
            try:
                bi = trial % len(alt_bases)
                base_x = alt_bases[bi][0].copy()
            except Exception:
                base_x = global_x.copy()
        else:
            base_x = global_x.copy()
        # recompute pool from base every few trials
        if trial % 8 == 0:
            try:
                pool_b, deg_b = low_pool(base_x)
            except Exception:
                pool_b, deg_b = pool, deg
        else:
            # pool w.r.t. base if base is alt? recompute cheap
            if alt_bases and (trial % 6 == 5):
                try:
                    pool_b, deg_b = low_pool(base_x)
                except Exception:
                    pool_b, deg_b = pool, deg
            else:
                pool_b, deg_b = pool, deg
        rng_sel = np.random.default_rng(1000 + trial)
        # choose deletion set
        try:
            if r == 1:
                # cycle through low-degree points deterministically
                didx = int(pool_b[trial % len(pool_b)])
                D = [didx]
            else:
                rr = min(r, len(pool_b))
                # bias to lowest: take random subset of pool
                # ensure the degree-2 (minimum) point often included
                D = rng_sel.choice(pool_b, size=rr, replace=False).tolist()
        except Exception:
            rng_f = np.random.default_rng(999 + trial)
            D = rng_f.choice(n, size=r, replace=False).tolist()
        D = [int(v) for v in D]
        keep = [i for i in range(n) if i not in set(D)]
        try:
            S = base_x[keep].copy()
            removed_pts = base_x[D].copy()
        except Exception:
            continue
        # greedy reinsertion one at a time
        curS = S.copy()
        ok = True
        for s in range(len(D)):
            if time.time() > LOOP_END:
                ok = False
                break
            rng_c = np.random.default_rng(200000 + trial * 37 + s * 101)
            try:
                h = _find_hole(curS, rng_c, n_rand=30000, n_jit=2000,
                               jit_src=removed_pts, jit_sigma=0.08, n_best=20)
            except Exception:
                ok = False
                break
            try:
                curS = np.vstack([curS, h.reshape(1, 4)])
                curS = _norm(curS)
            except Exception:
                ok = False
                break
        if not ok:
            continue
        if curS.shape != (n, d):
            continue
        y0 = curS.copy()
        # LP polish
        remain = LOOP_END - time.time()
        if remain < 2.0:
            break
        budget = 4.0 if r <= 2 else 5.0
        budget = min(budget, remain)
        # small extra randomization of polish radius by trial
        r0 = 2e-3 if (trial % 2 == 0) else 1e-3
        try:
            yp, vp = _lp_polish(y0, time.time() + budget, r0=r0, cap=2500, maxit=200)
        except Exception:
            continue
        try:
            hh = _deg_hash(yp)
        except Exception:
            hh = None
        if vp < global_v - 1e-12:
            global_x = yp.copy(); global_v = vp
            if hh is not None:
                seen.add(hh)
            try:
                pool, deg = low_pool(global_x)
            except Exception:
                pass
            # reset alt bases slightly (keep them but they are worse now)
        else:
            if hh is not None:
                if hh not in seen:
                    seen.add(hh)
                    # store diverse near-miss as alt base if competitive
                    if vp < global_v + 8e-5:
                        try:
                            alt_bases.append((yp.copy(), vp))
                            if len(alt_bases) > 6:
                                # keep best alts
                                alt_bases.sort(key=lambda t: t[1])
                                alt_bases = alt_bases[:6]
                        except Exception:
                            pass
        # opportunistic: if pool exhausted for r=1, refresh
        if trial % 12 == 11:
            try:
                pool, deg = low_pool(global_x)
            except Exception:
                pass
    # final polish of global best with remaining time
    try:
        if time.time() < HARD_END - 2:
            global_x, global_v = _lp_polish(global_x, HARD_END - 1.5, r0=8e-4, cap=3000, maxit=300)
    except Exception:
        pass
    global_x = _norm(global_x)
    return global_x
