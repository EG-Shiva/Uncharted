import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _norm_batched(X):
    n = np.linalg.norm(X, axis=-1, keepdims=True) + 1e-30
    return X / n

def _norm1(x):
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-30)

def _maxcos1(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(np.max(g))

def _percode_maxcos(X):
    B = X.shape[0]
    G = np.matmul(X, np.transpose(X, (0, 2, 1)))
    N = X.shape[1]
    r = np.arange(N)
    G[:, r, r] = -2.0
    return np.max(G.reshape(B, -1), axis=1)

def _contact_sig(X, slack=1e-4):
    # returns sorted contact counts per code: (B,N)
    B, N, D = X.shape
    G = np.matmul(X, np.transpose(X, (0, 2, 1)))
    r = np.arange(N)
    G[:, r, r] = -2.0
    m = np.max(G.reshape(B, -1), axis=1)  # (B,)
    cnt = np.sum(G > (m[:, None, None] - slack), axis=2)  # (B,N)
    cnt_sorted = np.sort(cnt, axis=1)
    return cnt_sorted, m

def lp_polish_single(x, seconds, cap=3500, radius_init=1e-3):
    x = _norm1(np.asarray(x, dtype=float))
    n, d = x.shape
    best = _maxcos1(x)
    radius = radius_init
    start = time.time()
    iu0 = np.triu_indices(n, 1)
    # pre-store iu arrays
    iuI, iuJ = iu0
    while True:
        if time.time() - start >= seconds:
            break
        if radius < 1e-14:
            break
        c_full = (x @ x.T)[iuI, iuJ]
        m = float(np.max(c_full))
        tol = max(20.0 * radius, 1e-9)
        sel = c_full > m - tol
        nnz = int(np.count_nonzero(sel))
        if nnz == 0:
            break
        if nnz > cap:
            idx = np.where(sel)[0]
            # keep largest
            part = np.argpartition(-c_full[idx], cap)[:cap]
            order = idx[part]
            mask = np.zeros_like(sel, dtype=bool)
            mask[order] = True
            sel = mask
        I = iuI[sel]; J = iuJ[sel]; c = c_full[sel]
        p = len(I)
        nv = n * d + 1
        try:
            rows = np.repeat(np.arange(p), 2 * d + 1)
            # cols
            cI = (I[:, None] * d + np.arange(d)).ravel()
            cJ = (J[:, None] * d + np.arange(d)).ravel()
            cols = np.empty(p * (2 * d + 1), dtype=np.int64)
            cols[0::(2*d+1)] = 0  # placeholder, we build correctly below
            # build in order per row: I-block, J-block, t
            cols = np.concatenate([ (I[:, None]*d+np.arange(d)).ravel().__array__(), np.zeros(0)], axis=0) if False else None
            # simpler: construct cols array row-major
            cols2 = np.empty((p, 2*d+1), dtype=np.int64)
            cols2[:, :d] = I[:, None]*d + np.arange(d)
            cols2[:, d:2*d] = J[:, None]*d + np.arange(d)
            cols2[:, 2*d] = n*d
            cols = cols2.ravel()
            vals2 = np.empty((p, 2*d+1), dtype=float)
            vals2[:, :d] = x[J]
            vals2[:, d:2*d] = x[I]
            vals2[:, 2*d] = -1.0
            vals = vals2.ravel()
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n*d))), shape=(n, nv)).tocsr()
        except Exception:
            radius *= 0.5
            continue
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)]*(n*d)+[(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y = _norm1(y)
        v = _maxcos1(y)
        if v < best:
            x, best = y, v
            radius = min(radius*1.5, 0.03)
        else:
            radius *= 0.5
        if time.time() - start >= seconds:
            break
    return x, best

def build_inits(n=180, d=9):
    rng = np.random.RandomState(0)
    B = 24
    X = np.empty((B, n, d), dtype=np.float64)
    # 20 random Gaussian
    for b in range(20):
        a = rng.randn(n, d)
        a /= (np.linalg.norm(a, axis=1, keepdims=True)+1e-30)
        X[b] = a
    # simple construction 1: sparse ternary 1-sparse + 2-sparse
    S = np.zeros((n, d), dtype=float)
    idx = 0
    for i in range(d):
        for s in (1.0, -1.0):
            if idx >= n:
                break
            v = np.zeros(d); v[i] = s
            S[idx] = v; idx += 1
    # 2-sparse signed: C(9,2)=36 *4 =144
    for i in range(d):
        for j in range(i+1, d):
            for si in (1.0, -1.0):
                for sj in (1.0, -1.0):
                    if idx >= n:
                        break
                    v = np.zeros(d); v[i]=si; v[j]=sj
                    v /= np.linalg.norm(v)
                    S[idx]=v; idx+=1
                if idx>=n:
                    break
            if idx>=n:
                break
    # fill remainder (should be 18 left: 18+144=162) with 3-sparse deterministic
    r2 = np.random.RandomState(1)
    while idx < n:
        v = np.zeros(d)
        k = 3
        cols = r2.choice(d, k, replace=False)
        signs = r2.choice(np.array([-1.0,1.0]), k)
        v[cols]=signs
        v/=np.linalg.norm(v)
        # avoid duplicate
        S[idx]=v; idx+=1
    # two perm/sign variants
    for bi, seed in [(20, 10), (21, 11)]:
        r = np.random.RandomState(seed)
        perm = r.permutation(d)
        signs = r.choice(np.array([-1.0,1.0]), d)
        V = S[:, perm] * signs[None, :]
        # tiny break of symmetry + random rotation? just jitter
        V = V + 1e-3 * r.randn(n, d)
        V /= (np.linalg.norm(V, axis=1, keepdims=True)+1e-30)
        # random row shuffle
        V = V[r.permutation(n)]
        X[bi] = V
    # 2 noisy record copies
    rec = None
    try:
        rec = np.asarray(np.load("best.npy"), dtype=float)
    except Exception:
        rec = None
    rec2 = None
    try:
        rec2 = np.asarray(np.load("record.npy"), dtype=float)
    except Exception:
        rec2 = None
    cand = []
    for a in (rec, rec2):
        if a is not None and a.shape==(n,d):
            a = _norm1(a)
            cand.append(a)
    if len(cand)==0:
        for b in (22,23):
            a = rng.randn(n,d); a/=(np.linalg.norm(a,axis=1,keepdims=True)+1e-30)
            X[b]=a
    elif len(cand)==1:
        for b, sig in [(22,0.008),(23,0.02)]:
            r = np.random.RandomState(20+b)
            V = cand[0] + sig*r.randn(n,d)
            V/=(np.linalg.norm(V,axis=1,keepdims=True)+1e-30)
            X[b]=V
    else:
        for b, base, sig in [(22,cand[0],0.008),(23,cand[1],0.015)]:
            r = np.random.RandomState(20+b)
            V = base + sig*r.randn(n,d)
            V/=(np.linalg.norm(V,axis=1,keepdims=True)+1e-30)
            X[b]=V
    return X

def batched_adam(X, t0, time_budget=95.0, total_steps=2500):
    B, N, D = X.shape
    X = _norm_batched(X.copy())
    M = np.zeros_like(X)
    Vv = np.zeros_like(X)
    b1, b2, eps = 0.9, 0.999, 1e-8
    t = 0
    ridx = np.arange(N)
    a0, a1 = 20.0, 500.0
    lr0, lr1 = 0.01, 0.0012
    # exponential interpolation
    for step in range(total_steps):
        frac = step / max(1, total_steps-1)
        alpha = a0 * ((a1/a0) ** frac)
        lr = lr0 * ((lr1/lr0) ** frac)
        t += 1
        # Gram batched
        G = np.matmul(X, np.transpose(X, (0, 2, 1)))
        G[:, ridx, ridx] = -10.0
        gmax = np.max(G.reshape(B, -1), axis=1)  # (B,)
        # exp
        # alpha*(G-gmax)
        E = np.exp(alpha * (G - gmax[:, None, None]))
        E[:, ridx, ridx] = 0.0
        S = np.sum(E, axis=(1, 2))  # (B,)
        S = np.maximum(S, 1e-30)
        grad = np.matmul(E, X) * (2.0 / S[:, None, None])
        # tangential
        nd = np.sum(grad * X, axis=2, keepdims=True)
        gt = grad - nd * X
        M = b1 * M + (1.0 - b1) * gt
        Vv = b2 * Vv + (1.0 - b2) * gt * gt
        bc1 = 1.0 - b1 ** t
        bc2 = 1.0 - b2 ** t
        upd = lr * (M / bc1) / (np.sqrt(Vv / bc2) + eps)
        # remove normal comp
        dn = np.sum(upd * X, axis=2, keepdims=True)
        upd = upd - dn * X
        # clip per-point norm
        rn = np.linalg.norm(upd, axis=2, keepdims=True) + 1e-12
        maxstep = 0.03 if frac < 0.5 else 0.015
        sc = np.minimum(1.0, maxstep / rn)
        upd = upd * sc
        X = X - upd
        X = _norm_batched(X)
        if (step+1) % 25 == 0:
            el = time.time() - t0
            if el >= time_budget:
                break
            # also stop exploration by 150s absolute (directive); time_budget already <150
            if t0 is not None and el > 148.0:
                break
    return X

def solve():
    t0 = time.time()
    DEAD = t0 + 176.0
    n, d = 180, 9
    # incumbent
    best_inc = None
    best_inc_val = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.asarray(np.load(fn), dtype=float)
            if a.shape == (n, d):
                a = _norm1(a)
                v = _maxcos1(a)
                if v < best_inc_val:
                    best_inc_val = v
                    best_inc = a.copy()
        except Exception:
            pass
    if best_inc is None:
        rng0 = np.random.RandomState(12345)
        best_inc = _norm1(rng0.randn(n, d))
        best_inc_val = _maxcos1(best_inc)
    # build 24 tensor
    try:
        X0 = build_inits(n, d)
    except Exception:
        rng = np.random.RandomState(0)
        X0 = rng.randn(24, n, d)
        X0 = _norm_batched(X0)
    # batched anneal: reserve time for LP: allow ~ up to 95s or until 150s rule
    # check time every 25 steps inside
    remain_for_lp = 55.0
    # decide batched budget so that LP still fits before DEAD-5
    el = time.time() - t0
    batch_budget = max(10.0, min(95.0, (DEAD - 5.0 - remain_for_lp) - (time.time()-t0)))
    # if very little time, still run short
    if batch_budget < 5.0:
        batch_budget = 5.0
    try:
        Xf = batched_adam(X0, t0, time_budget=batch_budget, total_steps=2500)
    except Exception:
        Xf = X0
    # rank
    try:
        vals = _percode_maxcos(Xf)
    except Exception:
        vals = np.array([_maxcos1(Xf[b]) for b in range(Xf.shape[0])])
    order = np.argsort(vals)
    # diversity signatures
    try:
        sigs, _ = _contact_sig(Xf, slack=1e-4)
    except Exception:
        sigs = None
    # select best 3 mutually dissimilar
    B = Xf.shape[0]
    # candidate pool: top 10 by value
    pool = order[:min(10, B)].tolist()
    selected = []
    if sigs is not None:
        # greedy: first = best
        selected.append(int(pool[0]))
        while len(selected) < 3 and len(pool) > len(selected):
            best_c = -1
            best_d = -1
            for c in pool:
                if c in selected:
                    continue
                # min distance to selected
                dd = min(float(np.linalg.norm(sigs[c].astype(float)-sigs[s].astype(float))) for s in selected)
                # bias to quality: add small bonus for lower vals? use dd + quality term
                # prefer larger dd; break ties by lower val
                score = dd - 0.05*float(np.where(order==c)[0][0])
                if score > best_d:
                    best_d = score
                    best_c = c
            if best_c < 0:
                break
            selected.append(int(best_c))
    else:
        selected = [int(x) for x in pool[:3]]
    # ensure 3
    for c in pool:
        if len(selected) >= 3:
            break
        if c not in selected:
            selected.append(int(c))
    # LP polish each selected, at most 15s each but respect DEAD and reserve 25s for winner
    # stop all exploration by 150s: check elapsed
    polished = []
    polished_vals = []
    # time accounting: we want to reserve ~25s for final winner polish
    for si, b in enumerate(selected):
        el = time.time() - t0
        if el > 150.0:
            break
        if time.time() >= DEAD - 27.0:
            break
        # each at most 15s, but also cap so that remaining for final >=25 if possible
        # simple: give min(12, remaining-30) etc.
        remaining = DEAD - time.time() - 1.0
        if remaining < 3.0:
            break
        # reserve 25 for final + (2-remaining selects)*8 rough
        left = len(selected) - si - 1
        budget = min(15.0, max(2.0, remaining - 25.0 - left*6.0))
        if budget < 1.0:
            break
        # constraint-capped polish
        try:
            x0 = _norm1(Xf[b].copy())
            xp, vp = lp_polish_single(x0, seconds=budget, cap=3500, radius_init=8e-4)
            polished.append(xp)
            polished_vals.append(vp)
        except Exception:
            continue
    # determine overall winner among polished + batched unpolished + incumbent
    best = best_inc
    best_val = best_inc_val
    # consider unpolished batched best too (in case LP hurt due to time)
    for b in order[:3]:
        try:
            v = float(vals[b])
            if v < best_val:
                best_val = v
                best = _norm1(Xf[b].copy())
        except Exception:
            pass
    for xp, vp in zip(polished, polished_vals):
        if vp < best_val:
            best_val = vp
            best = xp
    # final 25s polish of overall winner
    try:
        remaining = DEAD - time.time() - 1.0
        if remaining > 2.0:
            fb = min(remaining, 25.0)
            # if best is incumbent and no polished won, still polish it
            xp, vp = lp_polish_single(best, seconds=fb, cap=4000, radius_init=6e-4)
            if vp < best_val + 1e-12:
                best, best_val = xp, vp
    except Exception:
        pass
    best = _norm1(np.asarray(best, dtype=float))
    return best
