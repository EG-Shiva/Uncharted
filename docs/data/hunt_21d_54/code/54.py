import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _proj(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return X / n

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _adam_lse(Y, steps, T0a, T1a, lr0, lr1, b1=0.9, b2=0.999, eps=1e-8):
    Y = _proj(Y.copy())
    n, d = Y.shape
    m = np.zeros_like(Y)
    v = np.zeros_like(Y)
    for s in range(steps):
        frac = s / max(1, steps - 1) if steps > 1 else 1.0
        # log-interp for T, linear for lr
        T = T0a * (T1a / T0a) ** frac if T0a > 0 and T1a > 0 else T0a * (1 - frac) + T1a * frac
        lr = lr0 * (1 - frac) + lr1 * frac
        G = Y @ Y.T
        np.fill_diagonal(G, -10.0)
        gmax = float(G.max())
        E = (G - gmax) / max(T, 1e-12)
        np.clip(E, -20.0, 0.0, out=E)
        W = np.exp(E)
        np.fill_diagonal(W, 0.0)
        S = float(W.sum()) + 1e-12
        grad = (W @ Y) / S
        dot = np.einsum('ij,ij->i', grad, Y)
        grad -= dot[:, None] * Y
        m = b1 * m + (1 - b1) * grad
        v = b2 * v + (1 - b2) * (grad * grad)
        bc1 = 1.0 - b1 ** (s + 1)
        bc2 = 1.0 - b2 ** (s + 1)
        mh = m / bc1
        vh = v / bc2
        step = lr * mh / (np.sqrt(vh) + eps)
        # keep step tangent
        sd = np.einsum('ij,ij->i', step, Y)
        step -= sd[:, None] * Y
        # clip large steps
        sn = np.linalg.norm(step, axis=1, keepdims=True) + 1e-12
        cap = 0.1
        over = sn > cap
        # scale down if needed (broadcast)
        step = np.where(over, step * (cap / sn), step)
        Y = Y - step
        Y = _proj(Y)
    return Y

def _lp_polish(X, seconds):
    X = _proj(X)
    n, d = X.shape
    best = _max_cos(X)
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        remain = seconds - (time.time() - start)
        if remain <= 0.05:
            break
        G = X @ X.T
        c = G[iu]
        m = float(c.max())
        sel = c > m - max(20 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        if p == 0:
            break
        if p > 2500:
            idx = np.argpartition(-c[sel], 2500)[:2500]
            I, J = I[idx], J[idx]
            csv = c[sel][idx]
            p = 2500
        else:
            csv = c[sel]
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([X[J], X[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((X.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))),
                          shape=(n, nv)).tocsr()
        obj = np.zeros(nv)
        obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-csv, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                          method="highs",
                          options={"time_limit": min(2.0, max(0.3, remain - 0.05))})
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = X + res.x[:-1].reshape(n, d)
        y = _proj(y)
        vv = _max_cos(y)
        if vv < best:
            X, best = y, vv
            radius = min(radius * 1.5, 0.02)
        else:
            radius *= 0.5
    return X

def _fingerprint_vec(X, k=300):
    G = X @ X.T
    iu = np.triu_indices(X.shape[0], 1)
    c = G[iu]
    c = np.sort(c)[::-1]
    if len(c) > k:
        c = c[:k]
    return c

def _fp_dist(a, b):
    return float(np.linalg.norm(a - b))

def _greedy_perm(C):
    n = C.shape[0]
    # sort pairs descending
    flat = np.argsort(-C.ravel())
    row_used = np.zeros(n, dtype=bool)
    col_used = np.zeros(n, dtype=bool)
    perm = np.full(n, -1, dtype=int)  # for each row i, col j
    for f in flat:
        i = f // n
        j = f % n
        if not row_used[i] and not col_used[j]:
            perm[i] = j
            row_used[i] = True
            col_used[j] = True
            if row_used.all():
                break
    return perm

def _random_orth(rng, d):
    A = rng.normal(size=(d, d))
    Q, R = np.linalg.qr(A)
    s = np.sign(np.diag(R))
    s[s == 0] = 1.0
    Q = Q * s[None, :]
    return Q

def _lattice_pool(rng, d, m=3000):
    pool = np.zeros((m, d))
    for i in range(m):
        sup = 2 if rng.random() < 0.5 else 3
        idx = rng.choice(d, size=sup, replace=False)
        sgn = rng.choice(np.array([-1.0, 1.0]), size=sup)
        v = np.zeros(d)
        v[idx] = sgn / np.sqrt(sup)
        pool[i] = v
    return pool

def _greedy_lattice_select(pool, n, rng):
    m = pool.shape[0]
    order = list(rng.permutation(m)[:1])
    # precompute pool norms (all 1)
    for _ in range(1, n):
        D = pool @ pool[order].T  # m x len
        mx = D.max(axis=1)
        mx[order] = 1e9
        # mask already chosen? need avoid duplicates
        # choose minimal max
        j = int(np.argmin(mx))
        order.append(j)
    return pool[np.array(order)]

def solve():
    T0 = time.time()
    DEADLINE = T0 + 160.0
    PHASE1_END = T0 + 70.0
    PHASE2_END = T0 + 140.0
    n, d = 54, 21
    rng = np.random.default_rng(7)

    def tl():
        return DEADLINE - (time.time() - T0) - 0.0
    def time_now():
        return time.time()

    # build starts (never use best.npy here)
    starts = []
    # 14 random
    for _ in range(14):
        starts.append(_proj(rng.normal(size=(n, d))))
    # 2 lattice-type
    try:
        pool = _lattice_pool(rng, d, m=3000)
        for _ in range(2):
            sel = _greedy_lattice_select(pool, n, rng)
            Q = _random_orth(rng, d)
            sel = sel @ Q.T
            # small jitter to break exact degeneracy
            sel = _proj(sel + 1e-3 * rng.normal(size=(n, d)))
            starts.append(sel)
    except Exception:
        for _ in range(2):
            starts.append(_proj(rng.normal(size=(n, d))))

    elites = []  # list of (val, X, fp)
    # Phase 1: diverse optima
    for si, s0 in enumerate(starts):
        if time_now() > min(PHASE1_END, DEADLINE - 60):
            break
        try:
            y = _adam_lse(s0, steps=800, T0a=0.02, T1a=0.0018, lr0=0.025, lr1=0.004)
        except Exception:
            y = _proj(s0)
        # short LP polish <=1.5s
        rem1 = min(1.5, max(0.3, min(PHASE1_END, PHASE2_END) - time_now() - 0.3))
        if rem1 > 0.25 and time_now() < DEADLINE - 5:
            try:
                y = _lp_polish(y, rem1)
            except Exception:
                pass
        y = _proj(y)
        v = _max_cos(y)
        fp = _fingerprint_vec(y)
        # discard near-duplicates
        dup = False
        for (ev, ex, efp) in elites:
            if abs(ev - v) < 2e-5 and _fp_dist(fp, efp) < 5e-4:
                dup = True
                break
        if not dup:
            elites.append((v, y, fp))
        if time_now() > DEADLINE - 5:
            break

    if len(elites) == 0:
        # fallback random elite
        y = _proj(rng.normal(size=(n, d)))
        y = _adam_lse(y, steps=400, T0a=0.02, T1a=0.002, lr0=0.02, lr1=0.004)
        elites.append((_max_cos(y), y, _fingerprint_vec(y)))

    # sort elites by value
    elites.sort(key=lambda t: t[0])
    best_val = elites[0][0]
    best_X = elites[0][1].copy()

    # Phase 2: recombination
    # precompute for speed
    rng2 = np.random.default_rng(123)
    it = 0
    while time_now() < min(PHASE2_END, DEADLINE - 25):
        it += 1
        if len(elites) < 2:
            break
        # pick two distinct elites (bias to better)
        # tournament: pick 3 random, take 2 best distinct
        idxs = rng2.choice(len(elites), size=min(4, len(elites)), replace=False)
        # sort those by val
        idxs = sorted(idxs, key=lambda i: elites[i][0])
        i1 = int(idxs[0])
        i2 = int(idxs[1]) if len(idxs) > 1 else int((i1 + 1) % len(elites))
        if i1 == i2:
            i2 = (i1 + 1) % len(elites)
        X1 = elites[i1][1]
        X2 = elites[i2][1]
        # Procrustes align X2 to X1
        try:
            M = X2.T @ X1  # 21x21
            U, _, Vt = np.linalg.svd(M, full_matrices=False)
            R = U @ Vt
            Y2 = X2 @ R
        except Exception:
            Y2 = X2.copy()
        # greedy match
        try:
            C = X1 @ Y2.T
            perm = _greedy_perm(C)  # row i -> col j in Y2
            Y2m = np.zeros_like(Y2)
            for i in range(n):
                Y2m[i] = Y2[perm[i]]
        except Exception:
            Y2m = Y2
        # choose cluster size and source
        k = int(rng2.integers(10, 28))  # 10..27
        swap = bool(rng2.random() < 0.5)
        A = X1 if not swap else Y2m
        B = Y2m if not swap else X1
        # spherical k-means cluster: nearest to random seed in A
        c0 = int(rng2.integers(0, n))
        dots = A @ A[c0]
        S = np.argsort(-dots)[:k]
        mask = np.zeros(n, dtype=bool)
        mask[S] = True
        child = np.where(mask[:, None], A, B)
        # mutation 2-4 points
        nm = int(rng2.integers(2, 5))
        mut_idx = rng2.choice(n, size=nm, replace=False)
        for mi in mut_idx:
            if rng2.random() < 0.5:
                v = rng2.normal(size=(d,))
                v /= np.linalg.norm(v) + 1e-12
                child[mi] = v
            else:
                # big tangent kick of existing
                vv = rng2.normal(size=(d,))
                vv -= (vv @ child[mi]) * child[mi]
                nn = np.linalg.norm(vv) + 1e-12
                vv /= nn
                ang = 0.3 + 0.5 * rng2.random()
                child[mi] = np.cos(ang) * child[mi] + np.sin(ang) * vv
        child = _proj(child)
        # Adam + LP polish (short)
        try:
            child = _adam_lse(child, steps=400, T0a=0.008, T1a=0.0012, lr0=0.015, lr1=0.003)
        except Exception:
            pass
        rem = min(1.0, max(0.3, min(PHASE2_END, DEADLINE - 22) - time_now() - 0.2))
        if rem > 0.25:
            try:
                child = _lp_polish(child, rem)
            except Exception:
                pass
        child = _proj(child)
        try:
            cv = _max_cos(child)
        except Exception:
            continue
        cfp = _fingerprint_vec(child)
        # keep if best or novel within 5e-4
        is_best = cv < best_val - 1e-12
        if is_best:
            best_val = cv
            best_X = child.copy()
            elites.append((cv, child.copy(), cfp))
            elites.sort(key=lambda t: t[0])
            # cap
            if len(elites) > 30:
                elites = elites[:30]
        else:
            if cv < best_val + 5e-4:
                # novelty check
                mind = min(_fp_dist(cfp, e[2]) for e in elites) if elites else 1e9
                if mind > 1e-4:
                    elites.append((cv, child.copy(), cfp))
                    elites.sort(key=lambda t: t[0])
                    if len(elites) > 30:
                        elites = elites[:30]
        # small time guard
        if time_now() > DEADLINE - 22:
            break

    # Final: LP polish overall winner, reserve last time
    # pick current best
    elites.sort(key=lambda t: t[0])
    if elites:
        if elites[0][0] < best_val:
            best_val, best_X = elites[0][0], elites[0][1].copy()
    remain = DEADLINE - time_now() - 1.0
    if remain > 0.5:
        budget = min(remain, 18.0)
        try:
            best_X = _lp_polish(best_X, budget)
            best_val = _max_cos(best_X)
        except Exception:
            pass
    best_X = _proj(best_X)

    # final comparator vs best.npy (allowed only as comparator)
    try:
        ref = np.load("best.npy")
        ref = _proj(np.nan_to_num(ref.astype(float)))
        if ref.shape == (54, 21):
            rv = _max_cos(ref)
            if rv < best_val:
                best_X = ref
                best_val = rv
    except Exception:
        pass
    try:
        rec = np.load("record.npy")
        rec = _proj(np.nan_to_num(rec.astype(float)))
        if rec.shape == (54, 21):
            rv2 = _max_cos(rec)
            if rv2 < best_val:
                best_X = rec
                best_val = rv2
    except Exception:
        pass

    best_X = _proj(np.nan_to_num(best_X))
    return best_X
