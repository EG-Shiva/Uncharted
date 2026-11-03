import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix
from scipy.stats import norm


def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def true_max_batch(X):
    # X: (B,N,D)
    G = X @ np.transpose(X, (0,2,1))
    B,N,_ = G.shape
    for b in range(B):
        np.fill_diagonal(G[b], -2.0)
    return G.max(axis=(1,2))

def contact_counts_single(x, mc, tol=0.002):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return (g > mc - tol).sum(axis=1)

def fingerprint(x, mc):
    g = x @ x.T
    w = np.linalg.eigvalsh(g)
    w = np.sort(w)[::-1]
    cc = contact_counts_single(x, mc, tol=0.002)
    cc = np.sort(cc)
    return w, cc

def lp_polish(x, seconds, radius_init=1e-3):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = max_cos(x)
    radius = float(radius_init)
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        if time.time() - start > seconds - 0.05:
            break
        c = (x @ x.T)[iu]
        m = float(c.max())
        thresh = m - max(20 * radius, 1e-9)
        sel = c > thresh
        if int(sel.sum()) > 1500:
            idx = np.argpartition(c, -1500)[-1500:]
            mask = np.zeros_like(sel, dtype=bool)
            mask[idx] = True
            sel = mask
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        if p == 0:
            break
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv)
        obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c[sel], A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        v = max_cos(y)
        if v < best:
            x, best, radius = y, v, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def halton_normals(n_pts, dim, primes, rng_offset=1):
    # n_pts points in dim-D halton cube -> normals via ppf
    seq = np.zeros((n_pts, dim))
    for dd in range(dim):
        b = primes[dd]
        for i in range(n_pts):
            n = i + rng_offset
            f = 1.0
            r = 0.0
            while n > 0:
                f /= b
                r += f * (n % b)
                n //= b
            seq[i, dd] = r
    eps = 1e-12
    seq = np.clip(seq, eps, 1 - eps)
    z = norm.ppf(seq)
    return z

def random_orth(d, rng):
    A = rng.normal(size=(d, d))
    Q, R = np.linalg.qr(A)
    s = np.sign(np.diag(R))
    s[s == 0] = 1
    Q = Q * s[None, :]
    return Q

def solve():
    t0 = time.time()
    BUDGET = 172.0
    rng = np.random.default_rng(12345)
    N, D, B = 95, 13, 24
    # ---- init population (B,N,D): half Gaussian, half low-discrepancy ----
    X = np.zeros((B, N, D))
    # Gaussian part: B1=12
    B1 = 12
    for b in range(B1):
        g = rng.normal(size=(N, D))
        g /= np.linalg.norm(g, axis=1, keepdims=True)
        X[b] = g
    # low-discrepancy part
    primes = [2,3,5,7,11,13,17,19,23,29,31,37,41]
    n_ld = (B - B1) * N
    z = halton_normals(n_ld, D, primes, rng_offset=1)
    # random global rotations per config to diversify + scramble order
    for k in range(B - B1):
        block = z[k*N:(k+1)*N, :].copy()
        # normalize
        nr = np.linalg.norm(block, axis=1, keepdims=True)
        nr[nr == 0] = 1
        block /= nr
        Q = random_orth(D, rng)
        block = block @ Q.T
        # shuffle rows deterministically
        perm = rng.permutation(N)
        block = block[perm]
        X[B1+k] = block
    X /= np.linalg.norm(X, axis=2, keepdims=True)

    # ---- batched Adam on annealed soft max ----
    m = np.zeros_like(X)
    v = np.zeros_like(X)
    b1, b2, eps = 0.9, 0.999, 1e-8
    BATched_TIME = 68.0
    t_batch_start = time.time()
    t_batch_end = t_batch_start + BATched_TIME
    step = 0
    mask_up = np.triu(np.ones((N, N), dtype=bool), 1)
    last_record = t_batch_start
    history = []
    # precompute for speed: broadcast mask
    while time.time() < t_batch_end:
        elapsed = time.time() - t_batch_start
        prog = elapsed / BATched_TIME
        if prog < 0.25:
            beta = 20.0
        elif prog < 0.50:
            beta = 80.0
        elif prog < 0.75:
            beta = 250.0
        else:
            beta = 700.0
        lr = 0.01 * (0.2 ** prog)  # 0.01 -> 0.002
        step += 1
        # Gram
        G = X @ np.transpose(X, (0, 2, 1))  # (B,N,N)
        # max per batch (upper tri)
        # compute per-batch max efficiently
        Gmax = np.zeros(B)
        for b in range(B):
            # use masked view
            Gmax[b] = G[b][mask_up].max()
        # weights
        # exp(beta*(G-M))
        # to save memory compute in place
        E = np.exp(beta * (G - Gmax[:, None, None]))
        # zero diag and lower
        # apply mask: keep only upper, then symmetrize
        Wup = np.zeros_like(E)
        # vectorized masking
        Wup[:, mask_up] = E[:, mask_up]
        # zero diag already excluded; lower stays 0
        S = Wup[:, mask_up].sum(axis=1)  # (B,)
        S[S == 0] = 1.0
        # symmetrize normalized
        Wn = Wup / S[:, None, None]
        Wsym = Wn + np.transpose(Wn, (0, 2, 1))
        Grad = Wsym @ X  # (B,N,D)
        # tangent project
        dot = (Grad * X).sum(axis=2, keepdims=True)
        Gt = Grad - dot * X
        # Adam
        m = b1 * m + (1 - b1) * Gt
        v = b2 * v + (1 - b2) * (Gt * Gt)
        bc1 = 1 - b1 ** step
        bc2 = 1 - b2 ** step
        mhat = m / bc1
        vhat = v / bc2
        U = mhat / (np.sqrt(vhat) + eps)
        # project update to tangent
        ud = (U * X).sum(axis=2, keepdims=True)
        Ut = U - ud * X
        X = X - lr * Ut
        X /= np.linalg.norm(X, axis=2, keepdims=True)
        # every 10s record true max
        now = time.time()
        if now - last_record >= 10.0 or now >= t_batch_end - 0.2:
            last_record = now
            tm = true_max_batch(X)
            history.append(tm.copy())
        if time.time() >= t_batch_end:
            break
        # safety: if over budget approaching, break
        if time.time() - t0 > BUDGET - 60:
            break

    # evaluate population
    tm = true_max_batch(X)
    order = np.argsort(tm)
    Xs = X[order]
    tms = tm[order]
    # fingerprint prune duplicates
    kept = []
    kept_vals = []
    for i in range(B):
        x = Xs[i] / np.linalg.norm(Xs[i], axis=1, keepdims=True)
        mc = float(tms[i])
        w, cc = fingerprint(x, mc)
        dup = False
        for (wk, cck, vk) in kept:
            # eigenvalue distance (top 20) + contact vector equality + max closeness
            if abs(vk - mc) < 5e-4:
                de = np.linalg.norm(w[:20] - wk[:20])
                if de < 1e-2 and np.array_equal(cc, cck):
                    dup = True
                    break
        if not dup:
            kept.append((w, cc, mc))
            kept_vals.append((x, mc))
        if len(kept_vals) >= 10:
            # keep collecting only if distinct and good? continue to allow more but limit
            pass
    # sort kept by value
    kept_vals = sorted(kept_vals, key=lambda t: t[1])
    # take top distinct elites (up to 8)
    elites = [kv[0] for kv in kept_vals[:8]]
    elite_vals = [kv[1] for kv in kept_vals[:8]]
    if len(elites) == 0:
        elites = [Xs[0]]
        elite_vals = [float(tms[0])]

    # ---- crossover: transplant 15-25 lowest-contact points into voids ----
    children = []
    # need at least 2 elites
    nE = len(elites)
    if nE >= 2:
        # generate up to 10 children from best pairs
        pairs = []
        for i in range(min(nE, 5)):
            for j in range(min(nE, 5)):
                if i != j:
                    pairs.append((i, j))
        # limit
        pairs = pairs[:10]
        for (ri, di) in pairs:
            R = elites[ri].copy()
            Dn = elites[di].copy()
            mcR = max_cos(R)
            mcD = max_cos(Dn)
            ccR = contact_counts_single(R, mcR, tol=0.002)
            ccD = contact_counts_single(Dn, mcD, tol=0.002)
            k = int(rng.integers(15, 26))
            k = min(k, N - 1)
            donor_idx = np.argsort(ccD)[:k]
            recip_idx = np.argsort(-ccR)[:k]
            Rrem_mask = np.ones(N, dtype=bool)
            Rrem_mask[recip_idx] = False
            Rrem = R[Rrem_mask]
            Dsub = Dn[donor_idx]  # (k,D)
            # search best random rotation for Dsub to fit voids
            bestQ = None
            bestScore = 1e100
            # include identity
            for trial in range(9):
                if trial == 0:
                    Q = np.eye(D)
                else:
                    Q = random_orth(D, rng)
                Td = Dsub @ Q.T
                # cross max
                cross = Td @ Rrem.T
                sc = cross.max()
                # also internal max of Td (should be low)
                if k > 1:
                    ii = Td @ Td.T
                    np.fill_diagonal(ii, -2)
                    sc = max(sc, float(ii.max()))
                if sc < bestScore:
                    bestScore = sc
                    bestQ = Q
            Td = Dsub @ bestQ.T
            C = R.copy()
            # for each slot, choose donor vs sampled hole whichever gives smaller max-dot
            # pre-sample holes: 2000 candidates, greedy sequential
            # compute sequentially
            for t in range(k):
                slot = recip_idx[t]
                dvec = Td[t]
                # mask without slot
                tmp_mask = np.ones(N, dtype=bool)
                tmp_mask[slot] = False
                others = C[tmp_mask]
                # donor score
                s_d = float((others @ dvec).max())
                # hole sample
                cand = rng.normal(size=(2000, D))
                cand /= np.linalg.norm(cand, axis=1, keepdims=True)
                sc_h = cand @ others.T
                mh = sc_h.max(axis=1)
                bj = int(np.argmin(mh))
                hvec = cand[bj]
                s_h = float(mh[bj])
                if s_d <= s_h:
                    C[slot] = dvec
                else:
                    C[slot] = hvec
            C /= np.linalg.norm(C, axis=1, keepdims=True)
            children.append(C)
    # evaluate children + elites together
    pool = []
    for x, vv in zip(elites, elite_vals):
        pool.append((float(vv), x))
    for c in children:
        pool.append((max_cos(c), c))
    pool = sorted(pool, key=lambda t: t[0])
    # deduplicate pool by fingerprint again, keep top 6
    pool2 = []
    seen = []
    for vv, xx in pool:
        w, cc = fingerprint(xx, vv)
        dup = False
        for (wk, cck, vk) in seen:
            if abs(vk - vv) < 5e-4 and np.linalg.norm(w[:20]-wk[:20]) < 1e-2 and np.array_equal(cc, cck):
                dup = True
                break
        if not dup:
            pool2.append((vv, xx))
            seen.append((w, cc, vv))
        if len(pool2) >= 6:
            break
    if len(pool2) == 0:
        pool2 = pool[:3]
    # top three mutually distinct for LP polish ~15s each (adaptive to budget)
    remain = BUDGET - (time.time() - t0)
    # reserve ~25s for final long polish + 2s margin
    avail = max(5.0, remain - 27.0)
    # split among top3
    top3 = pool2[:3]
    per = min(15.0, avail / max(1, len(top3)))
    # if very little time, reduce
    if per < 2.0:
        per = max(1.0, avail / max(1, len(top3)))
    polished = []
    for vv, xx in top3:
        if time.time() - t0 > BUDGET - 28:
            polished.append((vv, xx))
            continue
        try:
            xp = lp_polish(xx, seconds=float(per), radius_init=8e-4)
            vp = max_cos(xp)
            polished.append((vp, xp))
        except Exception:
            polished.append((vv, xx))
    polished = sorted(polished, key=lambda t: t[0])
    winner_val, winner = polished[0]
    # long-polish winner with remaining time (leave 1.5s margin)
    remain2 = BUDGET - (time.time() - t0) - 1.5
    if remain2 > 3.0:
        try:
            # two passes
            s1 = min(remain2 * 0.6, 20.0)
            winner = lp_polish(winner, seconds=float(s1), radius_init=5e-4)
            winner_val = max_cos(winner)
            rest = BUDGET - (time.time() - t0) - 1.0
            if rest > 2.0:
                winner = lp_polish(winner, seconds=float(min(rest, 12.0)), radius_init=2e-4)
                winner_val = max_cos(winner)
        except Exception:
            pass
    winner /= np.linalg.norm(winner, axis=1, keepdims=True)
    # return only if beats best.npy, else fallback
    try:
        xb = np.load("best.npy").astype(float)
        xb = xb / np.linalg.norm(xb, axis=1, keepdims=True)
        vb = max_cos(xb)
    except Exception:
        xb = None
        vb = 1e100
    if xb is not None and winner_val > vb - 1e-12:
        return xb
    return winner
