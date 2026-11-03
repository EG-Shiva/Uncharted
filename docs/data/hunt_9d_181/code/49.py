import time
import numpy as np
from scipy.optimize import linprog, linear_sum_assignment
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _norm(X):
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-30)
    return X

def _exp_smooth(X, steps, k0=40.0, k1=220.0, lr=0.025, rng=None):
    X = X.copy()
    n = X.shape[0]
    for s in range(steps):
        frac = s / max(1, steps - 1)
        k = k0 + (k1 - k0) * frac
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        gmax = float(G.max())
        # normalised exponential weights
        E = np.clip(k * (G - gmax), -50.0, 0.0)
        W = np.exp(E)
        np.fill_diagonal(W, 0.0)
        Gr = W @ X
        dot = np.einsum('ij,ij->i', Gr, X)[:, None]
        Gt = Gr - dot * X
        nrm = float(np.linalg.norm(Gt, axis=1).max())
        if not np.finite(nrm) or nrm < 1e-12:
            break
        X = X - (lr / nrm) * Gt
        X = _norm(X)
    return X

def _contacts_info(X, tol=1e-7):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    m = float(G.max())
    C = G > (m - tol)
    cnt = C.sum(axis=1)
    # stress = sum excess
    S = np.where(C, G - (m - tol), 0.0).sum(axis=1)
    return G, m, cnt, S

def _deepest_holes(K, kneed, rng, n_cand=2500):
    d = K.shape[1]
    D = np.empty((kneed, d))
    cur = K
    # incremental to avoid O(n^2) rebuild each time heavy; simple loop
    for j in range(kneed):
        C = rng.standard_normal((n_cand, d))
        C /= np.linalg.norm(C, axis=1, keepdims=True)
        S = C @ cur.T
        mx = S.max(axis=1)
        if j > 0:
            S2 = C @ D[:j].T
            mx = np.maximum(mx, S2.max(axis=1))
        # pick best, occasionally 2nd-8th for diversity
        if rng.random() < 0.25:
            idx = np.argpartition(mx, 8)[:8]
            pick = idx[rng.integers(0, len(idx))]
        else:
            pick = int(np.argmin(mx))
        D[j] = C[pick]
        # extend cur for next iteration (view via concat reference)
        cur = np.vstack([cur, D[j:j+1]]) if cur.shape[0] > 0 else D[j:j+1]
    return D

def _delete_and_refill(X, del_idx, rng, n_cand=2500, smooth_steps=150):
    n, d = X.shape
    del_idx = np.unique(np.asarray(del_idx, dtype=int))
    del_idx = del_idx[(del_idx >= 0) & (del_idx < n)]
    k = len(del_idx)
    if k == 0:
        return _exp_smooth(X, smooth_steps)
    keep_mask = np.ones(n, dtype=bool)
    keep_mask[del_idx] = False
    K = X[keep_mask]
    D = _deepest_holes(K, k, rng, n_cand=n_cand)
    Y = np.empty_like(X)
    Y[keep_mask] = K
    Y[~keep_mask] = D
    Y = _norm(Y)
    Y = _exp_smooth(Y, smooth_steps)
    return Y

def _procrustes_align(A, B):
    # find Q (orthogonal, incl. reflections) minimizing ||A - B Q||
    M = B.T @ A  # 9x9
    try:
        U, _, Vt = np.linalg.svd(M)
        Q = U @ Vt
    except Exception:
        Q = np.eye(A.shape[1])
    return B @ Q

def _ensure_181(X, rng):
    # remove near-duplicates >0.999, trim/ refill to 181
    X = _norm(X)
    # dedup loop
    for _ in range(5):
        n = X.shape[0]
        if n <= 1:
            break
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        if m < 0.999:
            break
        # find first pair
        iu = np.triu_indices(n, 1)
        c = G[iu]
        k = int(np.argmax(c))
        j = int(iu[1][k])
        X = np.delete(X, j, axis=0)
    n = X.shape[0]
    if n > 181:
        # greedily drop one of closest pair until 181
        while X.shape[0] > 181:
            nn = X.shape[0]
            G = X @ X.T
            np.fill_diagonal(G, -2.0)
            iu = np.triu_indices(nn, 1)
            c = G[iu]
            k = int(np.argmax(c))
            j = int(iu[1][k])
            X = np.delete(X, j, axis=0)
    elif n < 181:
        D = _deepest_holes(X, 181 - n, rng, n_cand=2500)
        X = np.vstack([X, D])
    return _norm(X[:181] if X.shape[0] >= 181 else X)

def _edge_set(X, tol=1e-5):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    m = float(G.max())
    I, J = np.where(np.triu(G > (m - tol), 1))
    return set(zip(I.tolist(), J.tolist())), m

def _jaccard_dist(s1, s2):
    if len(s1) == 0 and len(s2) == 0:
        return 0.0
    inter = len(s1.intersection(s2))
    union = len(s1.union(s2))
    if union == 0:
        return 0.0
    return 1.0 - inter / union

def _lp_final(X, budget):
    t0 = time.time()
    x = _norm(X.copy())
    n, d = x.shape
    try:
        best = _max_cos(x)
    except Exception:
        return x
    radius = 1e-3
    iu0 = np.triu_indices(n, 1)
    ar = np.arange(d)
    while True:
        if time.time() - t0 >= budget:
            break
        if radius < 1e-14:
            break
        G = x @ x.T
        c = G[iu0]
        m = float(c.max())
        sel = c > m - max(20.0 * radius, 1e-9)
        I = iu0[0][sel]; J = iu0[1][sel]
        p = len(I)
        if p == 0:
            radius *= 0.5
            continue
        if p > 1500:
            csel = c[sel]
            idx = np.argpartition(-csel, 1500)[:1500]
            I = I[idx]; J = J[idx]
            csel2 = csel[idx]
        else:
            csel2 = c[sel]
        p = len(I)
        nv = n * d + 1
        cols_mat = np.empty((p, 2 * d + 1), dtype=np.int64)
        cols_mat[:, :d] = I[:, None] * d + ar
        cols_mat[:, d:2 * d] = J[:, None] * d + ar
        cols_mat[:, 2 * d] = n * d
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = cols_mat.ravel()
        vals_mat = np.empty((p, 2 * d + 1))
        vals_mat[:, :d] = x[J]
        vals_mat[:, d:2 * d] = x[I]
        vals_mat[:, 2 * d] = -1.0
        vals = vals_mat.ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv); obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-csel2, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        try:
            y = x + res.x[:-1].reshape(n, d)
            y = _norm(y)
            v = _max_cos(y)
        except Exception:
            radius *= 0.5
            continue
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.02)
        else:
            radius *= 0.5
    return x

def solve():
    t0 = time.time()
    HARD = 160.0  # watchdog
    rng = np.random.default_rng(0)
    # load seeds
    cands = []
    for fn in ["best.npy", "record.npy"]:
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=np.float64)
            if a.shape == (181, 9) and np.all(np.isfinite(a)):
                a = _norm(a)
                cands.append(a)
        except Exception:
            pass
    if not cands:
        a = rng.standard_normal((181, 9))
        cands.append(_norm(a))
    scores = [_max_cos(a) for a in cands]
    base = cands[int(np.argmin(scores))].copy()
    n, d = base.shape

    def time_left():
        return HARD - (time.time() - t0)

    # ---- build 4-6 variants with different deletions, cheap smoothing, no LP ----
    pop = []
    pop_scores = []
    pop_edges = []
    # variant 0: base smoothed lightly
    try:
        G0, m0, cnt0, S0 = _contacts_info(base, tol=1e-7)
    except Exception:
        cnt0 = np.ones(n); S0 = np.zeros(n); G0 = base @ base.T; m0 = 0.46

    # define deletion sets
    order_cnt_asc = np.argsort(cnt0)
    order_cnt_desc = np.argsort(-cnt0)
    order_stress = np.argsort(-(S0 + 0.1 * cnt0))
    top_hub = int(order_cnt_desc[0])
    sims = G0[top_hub].copy()
    sims[top_hub] = -10
    nn10 = np.argsort(-sims)[:11]

    del_sets = []
    del_sets.append([])  # v0 no deletion
    # v1 rattler-based: rattlers (0 contacts) + lowest contacts up to 8
    ratt = [i for i in range(n) if cnt0[i] == 0]
    need = 8
    v1 = list(ratt)
    for i in order_cnt_asc:
        if len(v1) >= need:
            break
        if int(i) not in v1:
            v1.append(int(i))
    del_sets.append(v1[:8])
    # v2 stress-based: top 10 stress
    del_sets.append([int(i) for i in order_stress[:10]])
    # v3 cluster-based: hub + neighbours (11 pts)
    del_sets.append([int(i) for i in ([top_hub] + nn10[:9].tolist())][:10])
    # v4 high-contact hubs: top 14 scattered? take every other of top 28
    top28 = [int(i) for i in order_cnt_desc[:28]]
    v4 = top28[::2][:14]
    del_sets.append(v4)
    # v5 random 8
    del_sets.append(rng.choice(n, size=8, replace=False).tolist())

    for vi, ds in enumerate(del_sets[:6]):
        if time_left() < 25:
            break
        if len(ds) == 0:
            Y = _exp_smooth(base, steps=120, k0=40.0, k1=200.0)
        else:
            # different smoothing lengths per variant for diversity
            ss = 150 if vi % 2 == 0 else 200
            Y = _delete_and_refill(base, ds, rng, n_cand=2000, smooth_steps=ss)
        v = _max_cos(Y)
        pop.append(Y); pop_scores.append(v)
        try:
            es, _ = _edge_set(Y, tol=1e-5)
        except Exception:
            es = set()
        pop_edges.append(es)
        if time_left() < 10:
            break

    # trim to 5 best-diverse? keep all (4-6)
    # track incumbent
    bi = int(np.argmin(pop_scores))
    Xbest = pop[bi].copy(); bbest = float(pop_scores[bi])

    # ---- crossover loop with Procrustes alignment ----
    it = 0
    # precompute: run until ~130s to leave time for final LP
    while time.time() - t0 < 132.0:
        if time_left() < 12:
            break
        it += 1
        if len(pop) < 2:
            break
        # pick pair
        a_idx, b_idx = rng.choice(len(pop), size=2, replace=False)
        A = pop[a_idx]; B = pop[b_idx]
        try:
            Bal = _procrustes_align(A, B)
        except Exception:
            Bal = B
        use_hyper = (it % 2 == 0)
        try:
            if use_hyper:
                vdir = rng.standard_normal(d)
                vdir /= (np.linalg.norm(vdir) + 1e-30)
                pa = A @ vdir
                pb = Bal @ vdir
                # median split to keep counts balanced
                ma = np.median(pa); mb = np.median(pb)
                # opposite sides: A above, Bal below (or random flip)
                if rng.random() < 0.5:
                    XA = A[pa > ma]
                    XB = Bal[pb <= mb]
                else:
                    XA = A[pa <= ma]
                    XB = Bal[pb > mb]
                Xc = np.vstack([XA, XB]) if (len(XA) + len(XB)) > 0 else A.copy()
                # small jitter to break exact duplicates
                Xc = Xc + rng.standard_normal(Xc.shape) * 3e-4
                Xc = _norm(Xc)
                Xc = _ensure_181(Xc, rng)
            else:
                # nearest-point bipartite matching between halves
                Gram = A @ Bal.T
                cost = -Gram
                # linear_sum_assignment on 181x181
                r, c_ = linear_sum_assignment(cost)
                # r should be 0..180; build reordered Bal
                Bord = np.empty_like(Bal)
                Bord[r] = Bal[c_]
                mask = rng.random(n) < 0.5
                Xc = np.where(mask[:, None], A, Bord)
                Xc = Xc + rng.standard_normal(Xc.shape) * 3e-4
                Xc = _norm(Xc)
                Xc = _ensure_181(Xc, rng)
        except Exception:
            continue
        # cheap intermediate relaxation, no LP
        try:
            Xc = _exp_smooth(Xc, steps=110, k0=50.0, k1=220.0)
        except Exception:
            continue
        try:
            vc = _max_cos(Xc)
        except Exception:
            continue
        try:
            ec, _ = _edge_set(Xc, tol=1e-5)
        except Exception:
            ec = set()
        # diversity check: Jaccard distance >0.15 from every member
        ok = True
        for es in pop_edges:
            if _jaccard_dist(ec, es) < 0.15:
                ok = False
                break
        if not ok:
            # still allow if it is a new global best (basin change worth keeping)
            if vc >= bbest - 1e-12:
                continue
        # accept: if pop<6 add, else replace worst if better than worst
        if len(pop) < 6:
            pop.append(Xc); pop_scores.append(vc); pop_edges.append(ec)
            if vc < bbest:
                Xbest = Xc.copy(); bbest = vc
        else:
            wi = int(np.argmax(pop_scores))
            if vc < pop_scores[wi]:
                pop[wi] = Xc; pop_scores[wi] = vc; pop_edges[wi] = ec
                if vc < bbest:
                    Xbest = Xc.copy(); bbest = vc
        # early exit if time low
        if time.time() - t0 > HARD - 18:
            break

    # ---- LP only on final incumbent ----
    rem = 170.0 - (time.time() - t0)
    if rem > 3.0:
        try:
            budg = min(14.0, max(2.0, rem - 1.5))
            Yf = _lp_final(Xbest, budget=budg)
            vf = _max_cos(Yf)
            if vf < bbest and np.all(np.isfinite(Yf)):
                Xbest = Yf; bbest = vf
        except Exception:
            pass
    Xbest = _norm(Xbest)
    return Xbest
