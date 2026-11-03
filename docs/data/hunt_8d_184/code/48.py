import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _norm(x):
    n = np.linalg.norm(x, axis=1, keepdims=True) + 1e-300
    return x / n

def _max_cos(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def _build_E8():
    s2 = np.sqrt(2.0)
    vecs = []
    for i in range(8):
        for j in range(i+1, 8):
            for s1 in (1.0, -1.0):
                for s2v in (1.0, -1.0):
                    v = np.zeros(8)
                    v[i] = s1 / s2
                    v[j] = s2v / s2
                    vecs.append(v)
    # all sign vectors with even number of minus
    for mask in range(256):
        bits = [(mask >> k) & 1 for k in range(8)]
        nminus = sum(1 for b in bits if b == 1)
        if nminus % 2 == 1:
            continue
        v = np.empty(8)
        for k in range(8):
            v[k] = -0.5 / s2 if bits[k] else 0.5 / s2
        # norm check: 8*(0.5/sqrt2)^2 = 1
        vecs.append(v)
    X = np.stack(vecs, axis=0)
    # should be 112+128=240
    return _norm(X)

def _greedy_deletions(E8, nruns, ndel, rng):
    N = E8.shape[0]
    G = E8 @ E8.T
    # exact 0.5 contacts (allow tiny fp error)
    A = (G > 0.4999).astype(np.int32)
    np.fill_diagonal(A, 0)
    subsets = []
    for r in range(nruns):
        alive = np.ones(N, dtype=bool)
        # maintain list of alive indices
        for _ in range(ndel):
            idx = np.where(alive)[0]
            # degrees within alive set
            # submatrix sum
            sub = A[np.ix_(idx, idx)]
            deg = sub.sum(axis=1)
            mx = int(deg.max())
            cand = np.where(deg == mx)[0]
            if len(cand) > 1:
                pick = cand[rng.integers(len(cand))]
            else:
                pick = cand[0]
            alive[idx[pick]] = False
        subsets.append(np.where(alive)[0].copy())
    return subsets

def _anneal_lse(x, steps, a0, a1, lr0, lr1, deadline=None):
    n, d = x.shape
    la0, la1 = np.log(a0), np.log(a1)
    ll0, ll1 = np.log(lr0), np.log(lr1)
    for t in range(steps):
        if deadline is not None and t % 50 == 0 and time.time() > deadline:
            break
        f = t / max(steps - 1, 1)
        alpha = np.exp(la0 + (la1 - la0) * f)
        lr = np.exp(ll0 + (ll1 - ll0) * f)
        G = x @ x.T
        np.fill_diagonal(G, -np.inf)
        mx = float(np.max(G))
        # avoid overflow: exp(alpha*(G-mx)), G-mx<=0
        W = np.exp(alpha * (G - mx))
        np.fill_diagonal(W, 0.0)
        S = W.sum(axis=1, keepdims=True) + 1e-30
        grad = W @ x
        dot = np.sum(grad * x, axis=1, keepdims=True)
        gt = (grad - dot * x) / S
        x = x - lr * gt
        x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x

def _greedy_top(x, steps, m0, m1, lr, deadline=None):
    n, d = x.shape
    for t in range(steps):
        if deadline is not None and t % 40 == 0 and time.time() > deadline:
            break
        f = t / max(steps - 1, 1)
        margin = m0 + (m1 - m0) * f
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        m = float(np.max(G))
        M = (G > (m - margin)).astype(np.float64)
        np.fill_diagonal(M, 0.0)
        deg = M.sum(axis=1, keepdims=True)
        grad = M @ x
        dot = np.sum(grad * x, axis=1, keepdims=True)
        gt = grad - dot * x
        gt /= np.maximum(deg, 1.0)
        x = x - lr * gt
        x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x

def _lp_polish(x, budget, t0, deadline, r0=1.5e-3, cap=2600):
    x = _norm(x)
    n, d = x.shape
    best = _max_cos(x)
    xb = x.copy()
    radius = r0
    iu0 = np.triu_indices(n, 1)
    ar = np.arange(d)
    while time.time() - t0 < budget and time.time() < deadline:
        if radius < 1e-14:
            break
        G = x @ x.T
        cf = G[iu0]
        m = float(cf.max())
        thr = m - max(20.0 * radius, 2e-9)
        sel = cf > thr
        I = iu0[0][sel]; J = iu0[1][sel]; cs = cf[sel]
        p = len(I)
        if p == 0:
            radius *= 0.5
            continue
        if p > cap:
            idx = np.argpartition(-cs, cap)[:cap]
            I, J, cs = I[idx], J[idx], cs[idx]
            p = cap
        nv = n * d + 1
        rows = np.concatenate([np.repeat(np.arange(p), d), np.repeat(np.arange(p), d), np.arange(p)])
        cols = np.concatenate([np.repeat(I * d, d) + np.tile(ar, p),
                               np.repeat(J * d, d) + np.tile(ar, p),
                               np.full(p, n * d)])
        vals = np.concatenate([x[J].ravel(), x[I].ravel(), -np.ones(p)])
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        except Exception:
            radius *= 0.5
            continue
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-cs, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs",
                          options={"time_limit": 3.0})
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = _norm(x + res.x[:-1].reshape(n, d))
        v = _max_cos(y)
        if v < best:
            best = v; xb = y.copy(); x = y
            radius = min(radius * 1.5, 0.02)
        else:
            radius *= 0.5
            if radius < 5e-7:
                break
    return xb, best

def solve():
    t0 = time.time()
    DEADLINE = t0 + 170.0
    HOP_DEADLINE = t0 + 135.0
    rng = np.random.default_rng(12345)
    n, d = 184, 8
    # fallback incumbent
    try:
        fb = _norm(np.asarray(np.load("best.npy"), dtype=np.float64))
        fb_val = _max_cos(fb) if fb.shape == (n, d) else 9.0
    except Exception:
        fb = _norm(rng.standard_normal((n, d)))
        fb_val = _max_cos(fb)
    # build E8
    E8 = _build_E8()
    # greedy deletions: at least 20 runs
    NRUN = 24
    subsets = _greedy_deletions(E8, NRUN, 56, rng)
    # coarse relax each subset
    coarse_vals = []
    coarse_X = []
    for r, sub in enumerate(subsets):
        if time.time() > HOP_DEADLINE - 8:
            break
        x = E8[sub].copy()
        # symmetry-breaking jitter, different scale per run
        sig = 3e-4 * (1.0 + (r % 4))
        noise = rng.standard_normal(x.shape)
        # tangential part
        dot = np.sum(noise * x, axis=1, keepdims=True)
        noise = noise - dot * x
        x = x + sig * noise
        x = _norm(x)
        # anneal p=8 -> 96 coarse
        x = _anneal_lse(x, steps=200, a0=8.0, a1=96.0, lr0=0.045, lr1=0.008, deadline=HOP_DEADLINE)
        x = _greedy_top(x, steps=40, m0=0.006, m1=0.0008, lr=0.010, deadline=HOP_DEADLINE)
        v = _max_cos(x)
        coarse_vals.append(v)
        coarse_X.append(x)
        if time.time() > HOP_DEADLINE:
            break
    if len(coarse_vals) == 0:
        return fb
    order = np.argsort(np.array(coarse_vals))
    # refine top-4
    NREF = min(4, len(order))
    refined = []
    for k in range(NREF):
        if time.time() > HOP_DEADLINE:
            break
        idx = order[k]
        x = coarse_X[idx].copy()
        # longer anneal 8->128 as assigned, plus stronger finish
        # vary slightly per rank for diversity
        a1 = 128.0 if k == 0 else 110.0
        x = _anneal_lse(x, steps=380, a0=8.0, a1=a1, lr0=0.038, lr1=0.005, deadline=HOP_DEADLINE)
        x = _greedy_top(x, steps=90, m0=0.008, m1=0.0005, lr=0.009, deadline=HOP_DEADLINE)
        v = _max_cos(x)
        refined.append((v, x))
    if not refined:
        # use coarse best
        b = int(order[0])
        bv, bx = coarse_vals[b], coarse_X[b]
    else:
        refined.sort(key=lambda t: t[0])
        bv, bx = refined[0]
    # LP only for the best subset
    best_overall = bx.copy()
    best_val = bv
    # LP budget: remaining until DEADLINE-4, capped
    rem = DEADLINE - time.time() - 4.0
    if rem > 3.0:
        budget = min(rem, 26.0)
        Xp, vp = _lp_polish(best_overall, budget=budget, t0=time.time(), deadline=DEADLINE - 2.0)
        if vp < best_val:
            best_val, best_overall = vp, Xp
    # final choice vs fallback file
    if fb_val < best_val:
        best_overall = fb.copy()
    best_overall = _norm(best_overall)
    return best_overall
