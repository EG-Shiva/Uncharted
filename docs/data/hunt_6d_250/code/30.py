import time
import numpy as np

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def _adam(X0, steps, b0, b1, lr0, lr1):
    X = X0.copy()
    m = np.zeros_like(X)
    v = np.zeros_like(X)
    b1m = 0.9
    b2m = 0.999
    eps = 1e-8
    t = 0
    lb0 = np.log(b0)
    lb1 = np.log(b1)
    for s in range(steps):
        frac = s / (steps - 1) if steps > 1 else 1.0
        beta = np.exp(lb0 + (lb1 - lb0) * frac)
        lr = lr0 + (lr1 - lr0) * frac
        G = X @ X.T
        np.fill_diagonal(G, -10.0)
        cm = float(np.max(G))
        G -= cm
        G *= beta
        np.clip(G, -30.0, 0.0, out=G)
        np.exp(G, out=G)
        np.fill_diagonal(G, 0.0)
        S = float(np.sum(G))
        if S < 1e-100:
            continue
        GX = G @ X
        GX *= (2.0 / S)
        dot = np.sum(GX * X, axis=1, keepdims=True)
        GX -= dot * X
        t += 1
        m *= b1m
        m += (1.0 - b1m) * GX
        v *= b2m
        v += (1.0 - b2m) * (GX * GX)
        c1 = 1.0 - (b1m ** t)
        c2 = 1.0 - (b2m ** t)
        denom = np.sqrt(v / c2) + eps
        step = (lr / c1) * m / denom
        sn = np.linalg.norm(step, axis=1, keepdims=True) + 1e-16
        cap = 5.0 * lr + 1e-12
        sc = np.minimum(1.0, cap / sn)
        step *= sc
        X -= step
        nrm = np.linalg.norm(X, axis=1, keepdims=True)
        nrm[nrm == 0] = 1.0
        X /= nrm
        md = np.sum(m * X, axis=1, keepdims=True)
        m -= md * X
    return X

def _build_pool():
    vecs = []
    for code in range(5**6):
        tmp = code
        v = np.zeros(6, dtype=float)
        nz = 0
        for k in range(6):
            dig = tmp % 5
            tmp //= 5
            c = dig - 2  # -2..2
            v[k] = float(c)
            if c != 0:
                nz += 1
        if nz == 0:
            continue
        nrm = float(np.sqrt(np.sum(v * v)))
        v /= nrm
        vecs.append(v)
    vecs = np.array(vecs)  # 15624 x 6
    # dedup identical directions (positive scaling duplicates like (1,1) vs (2,2))
    seen = {}
    keep = []
    for v in vecs:
        key = tuple(np.round(v, 8))
        if key not in seen:
            seen[key] = True
            keep.append(v)
    pool = np.array(keep, dtype=float)
    # canonical line dedup for antipodal variant
    seen2 = {}
    lines = []
    for v in pool:
        # canonical sign: first nonzero positive
        w = v.copy()
        for k in range(6):
            if abs(w[k]) > 1e-12:
                if w[k] < 0:
                    w = -w
                break
        key = tuple(np.round(w, 8))
        if key not in seen2:
            seen2[key] = True
            lines.append(w)
    lines = np.array(lines, dtype=float)
    return pool, lines

def _greedy_points(pool, rng, first_idx=None, jitter_scale=2e-4):
    P = pool.shape[0]
    n = 250
    jitter = rng.random(P) * jitter_scale
    curr = np.full(P, -2.0, dtype=float)
    sel = []
    for k in range(n):
        if k == 0:
            s = int(rng.integers(P)) if first_idx is None else int(first_idx) % P
        else:
            # argmin curr (+jitter), ignoring selected (inf)
            # add jitter only to finite entries
            best = -1
            # use argpartition-free argmin with jitter
            # compute scores on fly
            # to avoid O(P) python loop, numpy:
            scores = curr + jitter
            scores[curr == np.inf] = np.inf
            s = int(np.argmin(scores))
        sel.append(s)
        curr[s] = np.inf
        dots = pool @ pool[s]
        m = curr != np.inf
        curr[m] = np.maximum(curr[m], dots[m])
    return pool[np.array(sel)], sel

def _greedy_lines(lines, rng, first_idx=None, jitter_scale=2e-4):
    Q = lines.shape[0]
    jitter = rng.random(Q) * jitter_scale
    curr = np.full(Q, -2.0, dtype=float)  # max abs
    sel = []
    for k in range(125):
        if k == 0:
            s = int(rng.integers(Q)) if first_idx is None else int(first_idx) % Q
        else:
            scores = curr + jitter
            scores[curr == np.inf] = np.inf
            s = int(np.argmin(scores))
        sel.append(s)
        curr[s] = np.inf
        dots = np.abs(lines @ lines[s])
        m = curr != np.inf
        curr[m] = np.maximum(curr[m], dots[m])
    L = lines[np.array(sel)]
    X = np.concatenate([L, -L], axis=0)
    return X, sel

def _swap_improve(X, pool, rng, deadline, passes=2, K=50):
    n = X.shape[0]
    P = pool.shape[0]
    X = X.copy()
    for _p in range(passes):
        if time.time() > deadline:
            break
        D = pool @ X.T  # P x n
        arg1 = np.argmax(D, axis=1)
        max1 = D[np.arange(P), arg1]
        Dtmp = D.copy()
        Dtmp[np.arange(P), arg1] = -2.0
        max2 = np.max(Dtmp, axis=1)
        del Dtmp
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        M = float(np.max(G))
        order = rng.permutation(n)
        for i in order:
            if time.time() > deadline:
                break
            mask = np.ones(n, dtype=bool)
            mask[i] = False
            # other max excluding i
            other_max = float(np.max(G[np.ix_(mask, mask)]))
            scores = np.where(arg1 == i, max2, max1)  # P
            # best by max
            bi = int(np.argmin(scores))
            bs = float(scores[bi])
            newM_best = other_max if other_max > bs else bs
            if newM_best < M - 1e-9:
                # accept
                X[i] = pool[bi]
                # update G row/col
                drow = X @ X[i]
                G[i, :] = drow
                G[:, i] = drow
                G[i, i] = -2.0
                M = float(np.max(G))
                # update D column and arg/max
                D[:, i] = pool @ X[i]
                arg1 = np.argmax(D, axis=1)
                max1 = D[np.arange(P), arg1]
                Dtmp2 = D.copy()
                Dtmp2[np.arange(P), arg1] = -2.0
                max2 = np.max(Dtmp2, axis=1)
                del Dtmp2
                continue
            # try near-count reduction if i participates
            cnt_i = int(np.count_nonzero(G[i, mask] > M - 1e-5))
            if cnt_i == 0:
                continue
            # candidates with scores <= M (no max increase)
            # take K smallest
            if K >= P:
                cand = np.arange(P)
            else:
                cand = np.argpartition(scores, K)[:K]
            # filter scores <= M+1e-12
            filt = cand[scores[cand] <= M + 1e-12]
            if filt.size == 0:
                continue
            # counts for filt: count dots > M-1e-5 among mask
            thr = M - 1e-5
            # D[filt][:, mask] > thr
            sub = D[filt][:, mask]  # (f x 249)
            cnts = np.count_nonzero(sub > thr, axis=1)
            mj = int(np.argmin(cnts * 10.0 + scores[filt]))  # prefer low count then low score
            if int(cnts[mj]) < cnt_i:
                ci = int(filt[mj])
                X[i] = pool[ci]
                drow = X @ X[i]
                G[i, :] = drow
                G[:, i] = drow
                G[i, i] = -2.0
                # M unchanged (still other_max or M)
                M = float(np.max(G))
                D[:, i] = pool @ X[i]
                arg1 = np.argmax(D, axis=1)
                max1 = D[np.arange(P), arg1]
                Dtmp2 = D.copy()
                Dtmp2[np.arange(P), arg1] = -2.0
                max2 = np.max(Dtmp2, axis=1)
                del Dtmp2
    # normalize
    X /= np.linalg.norm(X, axis=1, keepdims=True) + 1e-16
    return X

def solve():
    t0 = time.time()
    HARD = t0 + 172.0
    STRUCT_END = t0 + 60.0
    rng = np.random.default_rng(0)
    n, d = 250, 6
    # fallback benchmark
    best_ref = None
    best_ref_val = np.inf
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.asarray(np.load(fn), dtype=float)
            if a.shape == (n, d):
                a = a / np.linalg.norm(a, axis=1, keepdims=True)
                if np.all(np.isfinite(a)):
                    v = _max_cos(a)
                    if v < best_ref_val:
                        best_ref_val = v
                        best_ref = a.copy()
        except Exception:
            pass
    if best_ref is None:
        best_ref = rng.standard_normal((n, d))
        best_ref /= np.linalg.norm(best_ref, axis=1, keepdims=True)
        best_ref_val = _max_cos(best_ref)

    pool, lines = _build_pool()

    cands = []
    # several greedy constructions
    try:
        # 3 standard + 3 antipodal with different seeds
        seeds = [11, 22, 33]
        for si, sd in enumerate(seeds):
            if time.time() > STRUCT_END - 3.0:
                break
            r2 = np.random.default_rng(1000 + sd)
            Xg, _ = _greedy_points(pool, r2, first_idx=int(r2.integers(len(pool))))
            Xg = _swap_improve(Xg, pool, r2, min(STRUCT_END, HARD - 90.0), passes=2, K=50)
            v = _max_cos(Xg)
            cands.append((v, Xg.copy()))
        for si, sd in enumerate([44, 55, 66]):
            if time.time() > STRUCT_END - 3.0:
                break
            r2 = np.random.default_rng(2000 + sd)
            Xa, _ = _greedy_lines(lines, r2, first_idx=int(r2.integers(len(lines))))
            # swap on full 250 set with pool
            Xa = _swap_improve(Xa, pool, r2, min(STRUCT_END, HARD - 90.0), passes=1, K=40)
            v = _max_cos(Xa)
            cands.append((v, Xa.copy()))
    except Exception:
        pass

    if len(cands) == 0:
        # fallback: random
        X = rng.standard_normal((n, d))
        X /= np.linalg.norm(X, axis=1, keepdims=True)
        cands.append((_max_cos(X), X))

    cands.sort(key=lambda t: t[0])
    top = cands[:2]

    polished = []
    for idx, (v0, X0) in enumerate(top):
        if time.time() > HARD - 25.0:
            polished.append((v0, X0))
            continue
        try:
            # annealed LogSumExp: two-stage
            if idx == 0:
                steps1 = 900
                steps2 = 700
            else:
                steps1 = 700
                steps2 = 500
            Y = _adam(X0, steps1, 300.0, 8000.0, 0.003, 0.001)
            if time.time() > HARD - 18.0:
                v = _max_cos(Y)
                polished.append((v, Y))
                continue
            Y = _adam(Y, steps2, 2000.0, 20000.0, 0.0012, 0.0004)
            v = _max_cos(Y)
            # keep better of polished vs original
            if v < v0:
                polished.append((v, Y))
            else:
                polished.append((v0, X0))
        except Exception:
            polished.append((v0, X0))

    polished.sort(key=lambda t: t[0])
    best_struct_val, best_struct = polished[0]

    if best_struct_val < best_ref_val:
        out = best_struct / np.linalg.norm(best_struct, axis=1, keepdims=True)
        return out
    else:
        out = best_ref / np.linalg.norm(best_ref, axis=1, keepdims=True)
        return out
