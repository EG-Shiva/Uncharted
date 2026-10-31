import time
import numpy as np
from scipy.optimize import linprog, linear_sum_assignment
from scipy.sparse import coo_matrix

def _norm(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n = np.maximum(n, 1e-12)
    y = x / n
    y = np.where(np.isfinite(y), y, 0.0)
    # fix any bad row
    bad = ~np.isfinite(np.sum(y, axis=1)) | (np.linalg.norm(y, axis=1) < 1e-9)
    if np.any(bad):
        y[bad] = np.array([1.0, 0.0, 0.0, 0.0])
        y = y / np.linalg.norm(y, axis=1, keepdims=True)
    return y

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(np.max(g))

def _per_point_max(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return np.max(g, axis=1)

def _smooth(x_init, steps, T0, T1, lr0, lr1, ms0, ms1, deadline=None):
    x = _norm(x_init.copy())
    n, d = x.shape
    v = np.zeros_like(x)
    beta = 0.8
    lT0, lT1 = np.log(T0), np.log(T1)
    ll0, ll1 = np.log(lr0), np.log(lr1)
    lm0, lm1 = np.log(ms0), np.log(ms1)
    eye = np.eye(n)
    for s in range(steps):
        if deadline is not None and (s % 150 == 0) and time.time() > deadline:
            break
        f = s / max(1, steps - 1)
        T = float(np.exp(lT0 + f * (lT1 - lT0)))
        lr = float(np.exp(ll0 + f * (ll1 - ll0)))
        ms = float(np.exp(lm0 + f * (lm1 - lm0)))
        G = x @ x.T
        m = float(np.max(G - 2.0 * eye))
        W = np.exp((G - m) / T)
        np.fill_diagonal(W, 0.0)
        S = float(np.sum(W) * 0.5)
        if not np.isfinite(S) or S <= 0:
            break
        Grad = (W @ x) / S
        Gt = Grad - np.sum(Grad * x, axis=1, keepdims=True) * x
        v = beta * v + Gt
        st = lr * v
        sn = np.linalg.norm(st, axis=1, keepdims=True) + 1e-16
        st = st * np.minimum(1.0, ms / sn)
        x = _norm(x - st)
    return x

def _lp(x, seconds, deadline=None, cap=2500):
    x = _norm(x.copy())
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    start = time.time()
    iu0 = np.triu_indices(n, 1)
    while True:
        now = time.time()
        if now - start >= seconds:
            break
        if deadline is not None and now > deadline:
            break
        if radius < 1e-14:
            break
        G = x @ x.T
        c = G[iu0]
        m = float(np.max(c))
        thr = m - max(20 * radius, 1e-9)
        sel = c > thr
        I = iu0[0][sel]; J = iu0[1][sel]
        p = len(I)
        if p == 0:
            break
        if p > cap:
            # keep tightest cap pairs
            cs = c[sel]
            idx = np.argpartition(-cs, cap)[:cap]
            I, J = I[idx], J[idx]
            cs = cs[idx]
            csub = cs
            p = cap
        else:
            csub = c[sel]
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        col_block = np.concatenate([I[:, None] * d + np.arange(d),
                                    J[:, None] * d + np.arange(d),
                                    np.full((p, 1), n * d)], axis=1).ravel()
        vals_block = np.concatenate([x[J], x[I], -np.ones((p, 1))], axis=1).ravel()
        try:
            A_ub = coo_matrix((vals_block, (rows, col_block)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        except Exception:
            radius *= 0.5
            continue
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-csub, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = _norm(x + res.x[:-1].reshape(n, d))
        vv = _max_cos(y)
        if vv < best:
            x, best = y, vv
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def _best_void(others, rng, n_cand=3000):
    cand = rng.standard_normal((n_cand, 4))
    cand = _norm(cand)
    s = cand @ others.T
    sc = np.max(s, axis=1)
    bi = int(np.argmin(sc))
    return cand[bi].copy()

def _align(B, A, iters=4):
    # return B rotated to align to A (both n x 4)
    R = np.eye(4)
    n = A.shape[0]
    try:
        for _ in range(iters):
            Br = B @ R
            C = A @ Br.T
            _, perm = linear_sum_assignment(-C)
            Bp = B[perm]
            M = Bp.T @ A
            U, _, Vt = np.linalg.svd(M)
            R = U @ Vt
    except Exception:
        pass
    return B @ R

def solve():
    t0 = time.time()
    HARD = t0 + 172.0
    rng = np.random.default_rng(20260607)
    n, d = 272, 4
    try:
        xrec = np.load("best.npy")
    except Exception:
        try:
            xrec = np.load("record.npy")
        except Exception:
            xrec = rng.standard_normal((n, d))
    xrec = np.asarray(xrec, dtype=np.float64)
    if xrec.shape != (n, d):
        if xrec.size == n * d:
            xrec = xrec.reshape(n, d)
        else:
            xrec = rng.standard_normal((n, d))
    xrec = _norm(xrec)
    try:
        r0 = float(_max_cos(xrec))
    except Exception:
        r0 = 0.99
    best = xrec.copy()
    best_val = r0

    # ---- population of distinct optima (strong kicks, worse allowed) ----
    pops = [xrec.copy()]
    pop_vals = [r0]
    # 7 diverse parents
    kick_sigmas = [0.18, 0.25, 0.32, 0.22, 0.30, 0.38, 0.28]
    for pi in range(7):
        if time.time() > HARD - 125:
            break
        y = xrec.copy()
        sig = kick_sigmas[pi % len(kick_sigmas)]
        if pi < 3:
            y = _norm(y + sig * rng.standard_normal(y.shape))
        else:
            k = 16 + 8 * (pi - 3)
            pm = _per_point_max(y)
            worst = np.argsort(-pm)[:k]
            # replace worst with random
            y[worst] = _norm(rng.standard_normal((k, d)))
            y = _norm(y + 0.5 * sig * rng.standard_normal(y.shape))
        # anneal (short, hot to escape)
        dl = min(HARD - 120, time.time() + 5.0)
        y = _smooth(y, steps=550, T0=0.035, T1=0.005, lr0=0.35, lr1=0.03, ms0=0.07, ms1=0.006, deadline=dl)
        if time.time() > HARD - 120:
            v = _max_cos(_norm(y))
            pops.append(_norm(y)); pop_vals.append(v)
            if v < best_val:
                best, best_val = _norm(y), v
            break
        y2 = _lp(y, seconds=1.0, deadline=min(HARD - 119, time.time() + 1.6))
        v = _max_cos(y2)
        pops.append(y2); pop_vals.append(v)
        if v < best_val - 1e-12:
            best, best_val = y2, v
    # ensure diversity: drop near-duplicates of record by assignment score? keep all but limit size
    # keep record + up to 7 others (all are kept to preserve diversity even if 0.905-0.93)

    # ---- recombination: align pairs + splice along random great spheres ----
    # build pair list: (0,i) plus diverse-diverse
    pairs = []
    m = len(pops)
    for i in range(1, m):
        pairs.append((0, i))
    if m > 3:
        pairs.append((1, 2))
        pairs.append((1, 3) if m > 3 else (1, 1))
        pairs.append((2, 3) if m > 3 else (0, 1))
    # add second round variants with different cuts
    pairs = pairs[:10]
    # duplicate pairs with different cuts to get ~10-12 children
    ext_pairs = []
    for (a, b) in pairs:
        ext_pairs.append((a, b))
        if len(ext_pairs) < 12:
            ext_pairs.append((a, b))
    pairs = ext_pairs[:12]

    for pidx, (ai, bi) in enumerate(pairs):
        if time.time() > HARD - 18:
            break
        if ai >= len(pops) or bi >= len(pops) or ai == bi:
            continue
        A = pops[ai]
        B = pops[bi]
        try:
            Bb = _align(B, A, iters=4)
        except Exception:
            Bb = B.copy()
        # random great sphere normal
        u = rng.standard_normal(4); u = u / (np.linalg.norm(u) + 1e-12)
        off = int(rng.integers(-28, 29))
        k = n // 2 + off
        k = max(60, min(n - 60, k))
        pA = A @ u
        pB = Bb @ u
        oA = np.argsort(-pA)  # descending
        oB = np.argsort(pB)   # ascending (bottom)
        takeA = oA[:k]
        takeB = oB[:n - k]
        child = np.concatenate([A[takeA], Bb[takeB]], axis=0)
        # boundary repair: void reinsertion for worst cross-boundary points
        try:
            for _ in range(3):
                pm = _per_point_max(child)
                # worst point
                wi = int(np.argmax(pm))
                if float(pm[wi]) < 0.895:
                    break
                others = np.delete(child, wi, axis=0)
                # adaptive candidates
                child[wi] = _best_void(others, rng, n_cand=2500)
        except Exception:
            pass
        # anneal + polish child
        dl = min(HARD - 16, time.time() + 6.0)
        # vary temperature slightly per child
        T0 = 0.025 + 0.008 * ((pidx % 3))
        child_a = _smooth(child, steps=700, T0=T0, T1=0.0025, lr0=0.32, lr1=0.02, ms0=0.06, ms1=0.004, deadline=dl)
        if time.time() > HARD - 16:
            v = _max_cos(_norm(child_a))
            if v < best_val:
                best, best_val = _norm(child_a), v
            # store as new pop member for further iteration
            pops.append(_norm(child_a)); pop_vals.append(v)
            continue
        child_b = _lp(child_a, seconds=1.8, deadline=min(HARD - 15, time.time() + 2.4))
        v = _max_cos(child_b)
        pops.append(child_b); pop_vals.append(v)
        if v < best_val - 1e-12:
            best, best_val = child_b, v
        # iterate using best + diverse: occasionally splice best with random pop
        # (next loop iterations already cover since pops grew; limit growth)
        if len(pops) > 14:
            # prune to keep size bounded but retain diversity: keep best + record + a few with median scores
            order = np.argsort(np.array(pop_vals))
            keep = set([order[0], 0])
            # keep 6 others spread in value
            for j in order[1:]:
                if len(keep) >= 8:
                    break
                keep.add(int(j))
            keep = sorted(list(keep))
            pops = [pops[j] for j in keep]
            pop_vals = [pop_vals[j] for j in keep]

    # second quick generation: splice current best with each of 2 most diverse (worst vals)
    if time.time() < HARD - 30:
        try:
            order = np.argsort(np.array(pop_vals))
            diverse_idx = order[-2:]
            for di in diverse_idx:
                if time.time() > HARD - 20:
                    break
                A = best.copy()
                B = pops[int(di)]
                Bb = _align(B, A, iters=3)
                u = rng.standard_normal(4); u /= (np.linalg.norm(u) + 1e-12)
                k = n // 2 + int(rng.integers(-20, 21))
                pA = A @ u; pB = Bb @ u
                child = np.concatenate([A[np.argsort(-pA)[:k]], Bb[np.argsort(pB)[:n - k]]], axis=0)
                pm = _per_point_max(child)
                for _ in range(2):
                    wi = int(np.argmax(pm))
                    others = np.delete(child, wi, axis=0)
                    child[wi] = _best_void(others, rng, n_cand=2500)
                    G = child @ child.T; np.fill_diagonal(G, -2.0)
                    pm = np.max(G, axis=1)
                dl = min(HARD - 18, time.time() + 5.0)
                child_a = _smooth(child, steps=600, T0=0.02, T1=0.002, lr0=0.3, lr1=0.02, ms0=0.05, ms1=0.004, deadline=dl)
                child_b = _lp(child_a, seconds=1.5, deadline=min(HARD - 17, time.time() + 2.0))
                v = _max_cos(child_b)
                if v < best_val - 1e-12:
                    best, best_val = child_b, v
        except Exception:
            pass

    # final polish of best with remaining time (keep record as fallback)
    rem = HARD - time.time() - 1.0
    if rem > 2.0:
        try:
            best = _lp(best, seconds=min(rem, 10.0), deadline=HARD - 0.5)
            best_val = _max_cos(best)
        except Exception:
            pass
    # safety: if we somehow got worse than record, return record
    if best_val > r0 + 1e-9:
        best = xrec.copy()
    best = _norm(best)
    return best
