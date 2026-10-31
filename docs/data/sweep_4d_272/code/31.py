import time
import numpy as np
from scipy.optimize import linprog, linear_sum_assignment
from scipy.sparse import coo_matrix

def _norm(x):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    return x

def _maxcos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(np.max(g))

def _lp_polish(x, seconds, deadline=None):
    x = _norm(x)
    n, d = x.shape
    best = _maxcos(x)
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while True:
        if time.time() - start >= seconds:
            break
        if deadline is not None and time.time() >= deadline:
            break
        if radius < 1e-14:
            break
        c = (x @ x.T)[iu]
        m = float(c.max())
        sel = c > m - max(20.0 * radius, 1e-9)
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
        obj = np.zeros(nv); obj[-1] = 1.0
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
        y = _norm(y)
        v = _maxcos(y)
        if v < best:
            x, best, radius = y, v, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def _anneal(X, steps, T0, T1, lr, noise0, max_step, rng):
    n, d = X.shape
    X = _norm(X)
    for s in range(steps):
        frac = s / max(1, steps - 1)
        T = T0 * (1.0 - frac) + T1 * frac
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        gmax = float(np.max(G))
        W = np.exp((G - gmax) / max(T, 1e-6))
        np.fill_diagonal(W, 0.0)
        S = float(np.sum(W) * 0.5)
        if S < 1e-30:
            gt = np.zeros_like(X)
        else:
            W /= S
            grad = W @ X
            gt = grad - np.sum(grad * X, axis=1, keepdims=True) * X
        step = lr * (1.0 - 0.5 * frac) * gt
        sn = np.linalg.norm(step, axis=1, keepdims=True) + 1e-12
        step = step * np.minimum(1.0, max_step / sn)
        X = X - step
        if noise0 > 0:
            amp = noise0 * (1.0 - frac)
            N = rng.standard_normal(size=X.shape)
            N = N - np.sum(N * X, axis=1, keepdims=True) * X
            X = X + amp * 0.5 * N
        X = _norm(X)
    return X

def _best_hole(others, rng, n_cand=4000, refine=60):
    d = others.shape[1]
    cand = rng.standard_normal(size=(n_cand, d))
    cand = _norm(cand)
    vals = np.max(cand @ others.T, axis=1)
    bi = int(np.argmin(vals))
    b = cand[bi].copy(); bv = float(vals[bi])
    step = 0.04
    for _ in range(refine):
        prop = b + step * rng.standard_normal(size=(24, d))
        prop = _norm(prop)
        sc = np.max(prop @ others.T, axis=1)
        j = int(np.argmin(sc))
        if float(sc[j]) < bv:
            b = prop[j].copy(); bv = float(sc[j]); step *= 1.06
        else:
            step *= 0.95
        if step < 5e-6:
            break
    return b

def _align(A, B):
    # alternating Hungarian + Procrustes, returns B rotated + permuted to match A, and Q, perm
    n, d = A.shape
    Q = np.eye(d)
    perm = np.arange(n)
    for _ in range(3):
        Bq = B @ Q
        S = A @ Bq.T  # similarity
        try:
            ri, ci = linear_sum_assignment(-S)
        except Exception:
            break
        # ri should be arange; build perm: A[i] <-> B[perm[i]]
        p = np.empty(n, dtype=int)
        p[ri] = ci
        perm = p
        Bp = B[p]
        M = Bp.T @ A  # 4x4
        try:
            U, _, Vt = np.linalg.svd(M, full_matrices=False)
            Q = U @ Vt
        except Exception:
            break
    B_aligned = B[perm] @ Q
    return B_aligned, Q, perm

def _splice(A, Bal, r, rng):
    n = A.shape[0]
    ar = A @ r
    br = Bal @ r
    selA = ar > 0
    selB = br <= 0
    tot = int(np.sum(selA) + np.sum(selB))
    # indices
    idxA_sel = np.where(selA)[0]
    idxA_uns = np.where(~selA)[0]
    idxB_sel = np.where(selB)[0]
    idxB_uns = np.where(~selB)[0]
    if tot > n:
        need_drop = tot - n
        # key = |dot| smallest first among selected
        ka = np.abs(ar[idxA_sel]) if len(idxA_sel) else np.array([])
        kb = np.abs(br[idxB_sel]) if len(idxB_sel) else np.array([])
        combined = [(float(ka[i]), 0, i) for i in range(len(ka))] + [(float(kb[i]), 1, i) for i in range(len(kb))]
        combined.sort(key=lambda t: t[0])
        dropA = set(); dropB = set()
        for k in range(need_drop):
            _, typ, ii = combined[k]
            if typ == 0:
                dropA.add(ii)
            else:
                dropB.add(ii)
        keepA = [idxA_sel[i] for i in range(len(idxA_sel)) if i not in dropA]
        keepB = [idxB_sel[i] for i in range(len(idxB_sel)) if i not in dropB]
        C = np.vstack([A[keepA], Bal[keepB]]) if (len(keepA)+len(keepB)>0) else A.copy()
    elif tot < n:
        need_add = n - tot
        ka = np.abs(ar[idxA_uns]) if len(idxA_uns) else np.array([])
        kb = np.abs(br[idxB_uns]) if len(idxB_uns) else np.array([])
        combined = [(float(ka[i]), 0, i) for i in range(len(ka))] + [(float(kb[i]), 1, i) for i in range(len(kb))]
        combined.sort(key=lambda t: t[0])
        addA = []; addB = []
        for k in range(min(need_add, len(combined))):
            _, typ, ii = combined[k]
            if typ == 0:
                addA.append(idxA_uns[ii])
            else:
                addB.append(idxB_uns[ii])
        keepA = list(idxA_sel) + addA
        keepB = list(idxB_sel) + addB
        # safety pad if still short (should not happen)
        C = np.vstack([A[keepA], Bal[keepB]])
        if C.shape[0] < n:
            extra = n - C.shape[0]
            cand = rng.standard_normal(size=(extra, 4)); cand = _norm(cand)
            C = np.vstack([C, cand])
        elif C.shape[0] > n:
            C = C[:n]
    else:
        C = np.vstack([A[idxA_sel], Bal[idxB_sel]])
    # shuffle to avoid ordering bias
    order = rng.permutation(C.shape[0])
    C = C[order]
    return C[:n]

def _repair(C, rng, rounds=2):
    n = C.shape[0]
    for _ in range(rounds):
        G = C @ C.T
        np.fill_diagonal(G, -2.0)
        m = float(np.max(G))
        if m < 0.925:
            break
        rowmax = np.max(G, axis=1)
        wi = int(np.argmax(rowmax))
        mask = np.ones(n, dtype=bool); mask[wi] = False
        h = _best_hole(C[mask], rng, n_cand=4000, refine=50)
        C[wi] = h
    return C

def _sig(x, k=600):
    g = x @ x.T
    iu = np.triu_indices(x.shape[0], 1)
    c = g[iu]
    c = np.sort(c)[::-1][:k]
    return c

def solve():
    t0 = time.time()
    HARD = t0 + 172.0
    rng = np.random.default_rng(272031)
    np.random.seed(7)
    X0 = None; bv0 = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=np.float64)
            if a.shape != (272, 4):
                continue
            a = _norm(a)
            v = _maxcos(a)
            if v < bv0:
                bv0 = v; X0 = a.copy()
        except Exception:
            continue
    if X0 is None:
        X0 = _norm(rng.standard_normal(size=(272, 4)))
        bv0 = _maxcos(X0)
    Xbest = X0.copy(); best_val = float(bv0)
    # quick polish of record to lock fallback
    if HARD - time.time() > 50:
        try:
            Xp = _lp_polish(Xbest, seconds=3.0, deadline=HARD - 1.0)
            vp = _maxcos(Xp)
            if vp < best_val:
                Xbest = Xp; best_val = vp
        except Exception:
            pass
    record_fallback = Xbest.copy(); record_val = best_val
    n = 272
    # --- population of distinct local optima ---
    pop = [Xbest.copy()]; popv = [best_val]
    sigs = [_sig(Xbest)]
    # strong kicks
    kick_scales = [0.35, 0.45, 0.55, 0.30, 0.50, 0.60, 0.40]
    for ki, ks in enumerate(kick_scales):
        if time.time() > HARD - 100:
            break
        X = X0.copy()
        # strong kick: subset + global
        sel = rng.choice(n, size=60, replace=False)
        X[sel] = X[sel] + ks * rng.standard_normal(size=(60, 4))
        X = X + 0.08 * rng.standard_normal(size=X.shape)
        X = _norm(X)
        # anneal even if score stays 0.905-0.93
        steps = 350
        T0 = float([0.018, 0.025, 0.012][ki % 3]); T1 = 0.003
        Xq = _anneal(X, steps=steps, T0=T0, T1=T1, lr=10.0, noise0=0.014, max_step=0.03, rng=rng)
        v = _maxcos(Xq)
        # short polish to make it a local optimum (cheap)
        if time.time() < HARD - 90:
            try:
                Xq2 = _lp_polish(Xq, seconds=0.9, deadline=HARD - 85)
                v2 = _maxcos(Xq2)
                if v2 < v + 0.002:
                    Xq = Xq2; v = v2
            except Exception:
                pass
        # diversity check vs existing: compare signatures
        s = _sig(Xq)
        dup = False
        for se in sigs:
            if float(np.linalg.norm(s - se)) < 0.01:
                dup = True
                break
        if not dup:
            pop.append(Xq); popv.append(v); sigs.append(s)
        else:
            # keep anyway if better than worst and not too close to best
            if v < max(popv) and float(np.linalg.norm(s - sigs[0])) > 0.003:
                pop.append(Xq); popv.append(v); sigs.append(s)
        if v < best_val:
            Xbest = Xq.copy(); best_val = v
    # ensure 6-10 members: pad with random annealed if needed
    while len(pop) < 6 and time.time() < HARD - 95:
        X = _norm(rng.standard_normal(size=(n, 4)))
        Xq = _anneal(X, steps=250, T0=0.02, T1=0.003, lr=10.0, noise0=0.02, max_step=0.03, rng=rng)
        v = _maxcos(Xq)
        pop.append(Xq); popv.append(v); sigs.append(_sig(Xq))
    # --- recombination ---
    # pairs: best with each other + random pairs
    order = np.argsort(np.array(popv))
    pop_sorted = [pop[i] for i in order]
    # limit parents to at most 8
    if len(pop_sorted) > 8:
        # keep best + most diverse (farthest sig from best)
        dists = [float(np.linalg.norm(sigs[order[i]] - sigs[order[0]])) for i in range(len(order))]
        idx = list(range(len(order)))
        idx.sort(key=lambda i: dists[i], reverse=True)
        keep = set([0, 1])
        for i in idx:
            if len(keep) >= 8:
                break
            keep.add(i)
        pop_sorted = [pop_sorted[i] for i in sorted(keep)]
    pairs = []
    L = len(pop_sorted)
    for i in range(L):
        for j in range(i + 1, L):
            pairs.append((i, j))
    rng.shuffle(pairs)
    # prioritize pairs containing best (index 0)
    pairs.sort(key=lambda t: 0 if (t[0] == 0 or t[1] == 0) else 1)
    child_count = 0
    for (pi, pj) in pairs:
        if time.time() > HARD - 45:
            break
        A = pop_sorted[pi]; B = pop_sorted[pj]
        try:
            Bal, _, _ = _align(A, B)
        except Exception:
            continue
        # several random great spheres per pair
        for rep in range(3):
            if time.time() > HARD - 45:
                break
            r = rng.standard_normal(4); r /= np.linalg.norm(r)
            try:
                C = _splice(A, Bal, r, rng)
            except Exception:
                continue
            C = _norm(C)
            # repair closest cross-boundary pairs
            try:
                C = _repair(C, rng, rounds=2)
            except Exception:
                pass
            # small jitter
            C = C + 0.008 * rng.standard_normal(size=C.shape)
            C = _norm(C)
            # anneal + short polish
            Cq = _anneal(C, steps=300, T0=0.012, T1=0.0018, lr=9.0, noise0=0.008, max_step=0.02, rng=rng)
            v = _maxcos(Cq)
            if v < best_val + 0.004 and time.time() < HARD - 42:
                try:
                    Cq2 = _lp_polish(Cq, seconds=1.0, deadline=HARD - 40)
                    v2 = _maxcos(Cq2)
                    if v2 <= v:
                        Cq = Cq2; v = v2
                except Exception:
                    pass
            child_count += 1
            if v < best_val:
                Xbest = Cq.copy(); best_val = v
                # insert into parents for further iteration
                pop_sorted.append(Cq.copy())
                L += 1
            else:
                # keep graph-diverse offspring for iteration even if worse (within 0.93)
                if v < 0.935 and len(pop_sorted) < 12:
                    s = _sig(Cq)
                    dmin = min(float(np.linalg.norm(s - _sig(p))) for p in pop_sorted)
                    if dmin > 0.008:
                        pop_sorted.append(Cq.copy())
            # second splice with reflected alignment (Bal already includes reflection if optimal); also try complement: flip r
            # (covered by next rep with new r)
    # final: keep original record as safe fallback
    if best_val > record_val and record_val < 1:
        # children failed: return polished record
        Xbest = record_fallback.copy(); best_val = record_val
    remain = HARD - time.time() - 1.0
    if remain > 4:
        try:
            Xp = _lp_polish(Xbest, seconds=min(remain, 38.0), deadline=HARD - 0.5)
            vp = _maxcos(Xp)
            if vp < best_val and np.all(np.isfinite(Xp)):
                Xbest = Xp; best_val = vp
        except Exception:
            pass
    Xbest = _norm(Xbest)
    bad = ~np.all(np.isfinite(Xbest), axis=1)
    if np.any(bad):
        for i in np.where(bad)[0]:
            vv = rng.standard_normal(4); vv /= np.linalg.norm(vv)
            Xbest[i] = vv
    return Xbest
