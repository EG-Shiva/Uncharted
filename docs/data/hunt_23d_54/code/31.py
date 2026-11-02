import time
import numpy as np

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _anneal(X, steps, k0, k1, lr0, lr1, w0, w1, mu=0.85):
    n = X.shape[0]
    V = np.zeros_like(X)
    lk0, lk1 = np.log(k0), np.log(k1)
    ll0, ll1 = np.log(lr0), np.log(lr1)
    for s in range(steps):
        f = s / max(steps - 1, 1)
        k = np.exp(lk0 + (lk1 - lk0) * f)
        lr = np.exp(ll0 + (ll1 - ll0) * f)
        w = w0 + (w1 - w0) * f
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        sel = G > (m - w)
        W = np.zeros_like(G)
        W[sel] = np.exp(k * (G[sel] - m))
        np.fill_diagonal(W, 0.0)
        F = W @ X
        dot = np.sum(F * X, axis=1, keepdims=True)
        Ft = F - dot * X
        an = float(np.sqrt(np.sum(Ft * Ft) / n)) + 1e-12
        V = mu * V + Ft / an
        vn = float(np.sqrt(np.sum(V * V) / n)) + 1e-12
        step = lr * V / max(vn, 1.0)
        X = X - step
        nr = np.linalg.norm(X, axis=1, keepdims=True)
        nr[nr == 0] = 1.0
        X /= nr
    return X

def _golay_pool():
    # binary Golay G23 cyclic, g(x)=x^11+x^9+x^7+x^6+x^5+x+1
    g = np.array([1,1,0,0,0,1,1,1,0,1,0,1], dtype=np.int32)  # g0..g11
    N = 4096
    C = np.empty((N, 23), dtype=float)
    for m in range(N):
        mb = np.array([(m >> i) & 1 for i in range(12)], dtype=np.int32)
        c = np.zeros(23, dtype=np.int32)
        for k in range(23):
            s = 0
            # c_k = sum_i mb_i * g_{k-i}
            lo = max(0, k - 11)
            hi = min(11, k)
            for i in range(lo, hi + 1):
                j = k - i
                if j <= 11:
                    s ^= (mb[i] & g[j]) if False else (mb[i] * g[j])
                    # use parity
            # faster parity: sum mod 2
            c[k] = s & 1
        # map 0->+1, 1->-1
        C[m] = np.where(c == 0, 1.0, -1.0) / np.sqrt(23.0)
    return C

def _hadamard_pool():
    p = 23
    sq = set((i * i) % p for i in range(1, p))
    H = np.ones((24, 24), dtype=float)
    for i in range(1, 24):
        for j in range(1, 24):
            d = (j - i) % p
            if d == 0:
                H[i, j] = -1.0
            elif d in sq:
                H[i, j] = 1.0
            else:
                H[i, j] = -1.0
    # rows (and negations) with one coordinate deleted
    out = []
    for r in range(24):
        row = H[r]
        for k in range(24):
            v = np.delete(row, k) / np.sqrt(23.0)
            out.append(v)
            out.append(-v)
    return np.array(out)

def _build_pool(rng):
    parts = []
    # Golay
    try:
        parts.append(_golay_pool())
    except Exception:
        pass
    # Hadamard-24 minus one coord
    try:
        parts.append(_hadamard_pool())
    except Exception:
        pass
    d = 23
    # coordinate vectors +-e_i
    E = np.zeros((46, d))
    for i in range(d):
        E[2*i, i] = 1.0
        E[2*i+1, i] = -1.0
    parts.append(E)
    # normalized +-e_i +- e_j
    EJ = []
    for i in range(d):
        for j in range(i+1, d):
            for s1 in (1.0, -1.0):
                for s2 in (1.0, -1.0):
                    v = np.zeros(d)
                    v[i] = s1; v[j] = s2
                    v /= np.sqrt(2.0)
                    EJ.append(v)
    parts.append(np.array(EJ))
    # sparse supports sizes 4-8 with random signs/perms
    n_sparse = 16000
    S = np.zeros((n_sparse, d))
    for t in range(n_sparse):
        s = 4 + (t % 5)  # 4..8 cycle for determinism
        idx = rng.choice(d, size=s, replace=False)
        signs = np.where(rng.rand(s) < 0.5, 1.0, -1.0)
        v = np.zeros(d)
        v[idx] = signs / np.sqrt(float(s))
        S[t] = v
    parts.append(S)
    # random dense sphere
    n_dense = 10000
    R = rng.randn(n_dense, d)
    R /= np.linalg.norm(R, axis=1, keepdims=True)
    parts.append(R)
    # random +-1/sqrt23 (signed Hadamard-like)
    n_pm = 6000
    P = np.where(rng.rand(n_pm, d) < 0.5, 1.0, -1.0) / np.sqrt(23.0)
    parts.append(P)
    C = np.vstack(parts)
    # normalize (safety)
    C /= np.linalg.norm(C, axis=1, keepdims=True)
    # deduplicate exact
    try:
        C = np.unique(C, axis=0)
    except Exception:
        pass
    # cap size to ~40000 for speed (deterministic subsample if larger)
    if C.shape[0] > 40000:
        idx = rng.choice(C.shape[0], size=40000, replace=False)
        C = C[idx]
    return C

def _greedy_select(C, n_select, start_idx, selected_mask_global=None):
    M = C.shape[0]
    cur_max = np.full(M, -2.0)
    selected = []
    is_sel = np.zeros(M, dtype=bool)
    # start
    selected.append(int(start_idx))
    is_sel[start_idx] = True
    cur_max = np.maximum(cur_max, C @ C[start_idx])
    cur_max[start_idx] = 2.0  # exclude
    for _ in range(1, n_select):
        # argmin of cur_max among non-selected
        # use argmin on masked array
        best = np.argmin(np.where(is_sel, np.inf, cur_max))
        selected.append(int(best))
        is_sel[best] = True
        cur_max = np.maximum(cur_max, C @ C[best])
        cur_max[best] = 2.0
    return np.array(selected, dtype=int)

def _one_for_one(C, sel_idx, max_passes=4):
    M = C.shape[0]
    n = sel_idx.shape[0]
    X = C[sel_idx]
    in_set = np.zeros(M, dtype=bool)
    in_set[sel_idx] = True
    for _ in range(max_passes):
        Gps = C @ X.T  # M x n
        Gs = X @ X.T
        np.fill_diagonal(Gs, -2.0)
        overall = float(Gs.max())
        # internal max excluding p
        improved = False
        order = np.argsort(Gs.max(axis=1))[::-1]  # worst first
        for p in order:
            # score of each candidate vs others
            others = [q for q in range(n) if q != p]
            score = Gps[:, others].max(axis=1)
            score[in_set] = np.inf
            b = int(np.argmin(score))
            bs = float(score[b])
            internal_other = float(Gs[np.ix_(others, others)].max()) if len(others) > 1 else -2.0
            new_max = internal_other if internal_other > bs else bs
            if new_max < overall - 1e-12:
                # swap p <- b
                old = int(sel_idx[p])
                in_set[old] = False
                in_set[b] = True
                sel_idx[p] = b
                X[p] = C[b]
                # update Gs/overall incrementally (recompute next iteration cheaply)
                Gs = X @ X.T
                np.fill_diagonal(Gs, -2.0)
                overall = float(Gs.max())
                # refresh Gps column p
                Gps[:, p] = C @ X[p]
                improved = True
        if not improved:
            break
    return sel_idx

def solve():
    t0 = time.time()
    DEADLINE = t0 + 170.0
    rng = np.random.RandomState(0)
    n, d = 54, 23
    try:
        xb = np.load("best.npy").astype(float)
    except Exception:
        xb = rng.randn(n, d)
    xb = xb / np.linalg.norm(xb, axis=1, keepdims=True)
    if xb.shape != (n, d):
        xb = rng.randn(n, d)
        xb /= np.linalg.norm(xb, axis=1, keepdims=True)
    try:
        xr = np.load("record.npy").astype(float)
        xr /= np.linalg.norm(xr, axis=1, keepdims=True)
        if xr.shape == (n, d) and _max_cos(xr) < _max_cos(xb):
            xb = xr.copy()
    except Exception:
        pass
    best = _max_cos(xb)
    x_best = xb.copy()

    # discrete pool
    C = _build_pool(np.random.RandomState(0))
    # a few greedy starts
    starts = [0, C.shape[0] // 3, C.shape[0] // 2, C.shape[0] - 1]
    tried = []
    for si, st in enumerate(starts):
        if time.time() > DEADLINE - 60:
            break
        sel = _greedy_select(C, n, int(st) % C.shape[0])
        sel = _one_for_one(C, sel, max_passes=3)
        Xd = C[sel].copy()
        v = _max_cos(Xd)
        tried.append((v, Xd))
        # occasional two-for-two sampled improvement on worst pair
        if time.time() < DEADLINE - 55:
            G = Xd @ Xd.T
            np.fill_diagonal(G, -2.0)
            ij = np.unravel_index(int(np.argmax(G)), G.shape)
            p, q = int(ij[0]), int(ij[1])
            others = [k for k in range(n) if k != p and k != q]
            Xo = Xd[others]
            internal = float((Xo @ Xo.T).max()) if len(others) > 1 else -2.0
            # sample candidate pairs
            nsamp = 4000
            a_idx = rng.choice(C.shape[0], size=nsamp, replace=True)
            b_idx = rng.choice(C.shape[0], size=nsamp, replace=True)
            # quick vectorized approx: need dots to Xo and to each other
            # compute in chunks to save time
            best_pair = None
            best_val = v
            for ch in range(0, nsamp, 1000):
                A = C[a_idx[ch:ch+1000]]
                B = C[b_idx[ch:ch+1000]]
                da = A @ Xo.T
                db = B @ Xo.T
                dab = np.sum(A * B, axis=1)
                cand_max = np.maximum(np.maximum(da.max(axis=1), db.max(axis=1)), dab)
                cand_max = np.maximum(cand_max, internal)
                j = int(np.argmin(cand_max))
                if float(cand_max[j]) < best_val - 1e-12:
                    best_val = float(cand_max[j])
                    best_pair = (int(a_idx[ch+j]), int(b_idx[ch+j]))
            if best_pair is not None:
                # ensure not already selected / distinct
                a0, b0 = best_pair
                if a0 != b0:
                    sel2 = sel.copy()
                    sel2[p] = a0; sel2[q] = b0
                    # check uniqueness
                    if len(set(sel2.tolist())) == n:
                        Xd2 = C[sel2]
                        v2 = _max_cos(Xd2)
                        if v2 < v:
                            v, Xd = v2, Xd2.copy()
                            tried[-1] = (v, Xd)

    # polish discrete candidates with anneal (cheapest first)
    tried = sorted(tried, key=lambda z: z[0])[:3]
    for v, Xd in tried:
        if time.time() > DEADLINE - 35:
            break
        # short anneal
        Xs = _anneal(Xd.copy(), steps=700, k0=80, k1=2500, lr0=0.012, lr1=0.0007, w0=0.05, w1=0.005)
        vs = _max_cos(Xs)
        if vs < best:
            best, x_best = vs, Xs.copy()
        # sharper second stage if promising
        if vs < best + 5e-04 and time.time() < DEADLINE - 25:
            Xs2 = _anneal(Xs.copy(), steps=500, k0=300, k1=4000, lr0=0.005, lr1=0.0004, w0=0.02, w1=0.003, mu=0.88)
            vs2 = _max_cos(Xs2)
            if vs2 < best:
                best, x_best = vs2, Xs2.copy()

    # also polish incumbent briefly so we never regress and may gain
    if time.time() < DEADLINE - 12:
        try:
            Xc = _anneal(x_best.copy(), steps=400, k0=250, k1=3500, lr0=0.004, lr1=0.0004, w0=0.02, w1=0.004, mu=0.88)
            vc = _max_cos(Xc)
            if vc < best:
                best, x_best = vc, Xc.copy()
        except Exception:
            pass

    x_best = x_best / np.linalg.norm(x_best, axis=1, keepdims=True)
    # safety: never return worse than loaded best
    try:
        x0 = np.load("best.npy").astype(float)
        x0 /= np.linalg.norm(x0, axis=1, keepdims=True)
        if _max_cos(x0) < _max_cos(x_best):
            x_best = x0.copy()
    except Exception:
        pass
    return x_best
