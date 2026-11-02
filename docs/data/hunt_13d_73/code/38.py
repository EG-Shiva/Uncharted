import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _row_max(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return G.max(axis=1)

def _lp_polish(X, seconds, t_end=1e30):
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    n, d = X.shape
    best = _max_cos(X)
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and time.time() < t_end and radius > 1e-14:
        c = (X @ X.T)[iu]
        m = c.max()
        sel = c > m - max(20 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        if p == 0:
            break
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([X[J], X[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((X.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv); obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-c[sel], A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = X + res.x[:-1].reshape(n, d)
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        v = _max_cos(y)
        if v < best:
            X, best, radius = y, v, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return X

def solve():
    t0 = time.time()
    DEADLINE = t0 + 172.0
    EXPLORE_END = t0 + 150.0
    rng = np.random.default_rng(17)
    n, d = 73, 13
    try:
        xb = np.load("best.npy").astype(float)
    except Exception:
        try:
            xb = np.load("record.npy").astype(float)
        except Exception:
            xb = rng.standard_normal((n, d))
    xb = np.asarray(xb, dtype=float)
    if xb.shape != (n, d):
        tmp = rng.standard_normal((n, d))
        m0 = min(xb.shape[0], n); m1 = min(xb.shape[1], d)
        tmp[:m0, :m1] = xb[:m0, :m1]
        xb = tmp
    xb /= np.linalg.norm(xb, axis=1, keepdims=True)
    best_val = _max_cos(xb)
    X_best = xb.copy()

    cap_sizes = [10, 12, 14, 16, 11, 13, 15, 10, 16, 12]

    def single_refine(C0, Obs, steps=200, s0=30.0, s1=150.0, lr0=0.02, lr1=0.002):
        # C0: (k,d), Obs: (p,d)
        C = C0.copy()
        k = C.shape[0]
        m1 = np.zeros_like(C); v1 = np.zeros_like(C)
        b1, b2, eps = 0.9, 0.999, 1e-8
        b1p, b2p = b1, b2
        ls0, ls1 = np.log(s0), np.log(s1)
        ll0, ll1 = np.log(lr0), np.log(lr1)
        for it in range(steps):
            frac = it / (steps - 1) if steps > 1 else 1.0
            s = np.exp(ls0 + (ls1 - ls0) * frac)
            lr = np.exp(ll0 + (ll1 - ll0) * frac)
            D = C @ Obs.T
            cm = D.max(axis=1, keepdims=True)
            E = np.exp(s * (D - cm))
            S = E.sum(axis=1, keepdims=True)
            S[~np.isfinite(S)] = 1.0
            S[S == 0] = 1.0
            W = E / S
            G = W @ Obs
            dot = (G * C).sum(axis=1, keepdims=True)
            Gt = G - dot * C
            m1 *= b1; m1 += (1 - b1) * Gt
            v1 *= b2; v1 += (1 - b2) * (Gt * Gt)
            mh = m1 / (1 - b1p); vh = v1 / (1 - b2p)
            b1p *= b1; b2p *= b2
            C -= lr * mh / (np.sqrt(vh) + eps)
            C /= np.linalg.norm(C, axis=1, keepdims=True)
        return C

    def block_refine(Y0, F, steps=300, s0=35.0, s1=350.0, lr0=0.012, lr1=0.0008):
        Y = Y0.copy()
        m = Y.shape[0]
        if m == 0:
            return Y
        m1 = np.zeros_like(Y); v1 = np.zeros_like(Y)
        b1, b2, eps = 0.9, 0.999, 1e-8
        b1p, b2p = b1, b2
        ls0, ls1 = np.log(s0), np.log(s1)
        ll0, ll1 = np.log(lr0), np.log(lr1)
        iu_m = np.triu_indices(m, 1) if m > 1 else (np.array([], dtype=int), np.array([], dtype=int))
        has_pair = m > 1
        for it in range(steps):
            frac = it / (steps - 1) if steps > 1 else 1.0
            s = np.exp(ls0 + (ls1 - ls0) * frac)
            lr = np.exp(ll0 + (ll1 - ll0) * frac)
            GYF = Y @ F.T
            cmax = GYF.max()
            if has_pair:
                GYY = Y @ Y.T
                # max over triu
                tv = GYY[iu_m]
                if tv.size > 0:
                    cmax = max(cmax, float(tv.max()))
            EYF = np.exp(s * (GYF - cmax))
            S = EYF.sum()
            if has_pair:
                GYY2 = Y @ Y.T
                EYYfull = np.exp(s * (GYY2 - cmax))
                # keep only upper triangle for sum, but build symmetric weight
                # zero diag
                np.fill_diagonal(EYYfull, 0.0)
                # each unordered pair counted twice in full; we want once -> halve
                S = S + EYYfull.sum() * 0.5
                if not np.isfinite(S) or S == 0:
                    S = 1.0
                WYF = EYF / S
                WYY = EYYfull / S * 0.5
                # symmetrize already symmetric; row sums count each pair once per row side
                Ggrad = WYF @ F + WYY @ Y
            else:
                if not np.isfinite(S) or S == 0:
                    S = 1.0
                WYF = EYF / S
                Ggrad = WYF @ F
            dot = (Ggrad * Y).sum(axis=1, keepdims=True)
            Gt = Ggrad - dot * Y
            m1 *= b1; m1 += (1 - b1) * Gt
            v1 *= b2; v1 += (1 - b2) * (Gt * Gt)
            mh = m1 / (1 - b1p); vh = v1 / (1 - b2p)
            b1p *= b1; b2p *= b2
            Y -= lr * mh / (np.sqrt(vh) + eps)
            Y /= np.linalg.norm(Y, axis=1, keepdims=True)
            if time.time() >= DEADLINE - 8.0:
                break
        return Y

    def global_relax(X0, budget=1.0, steps=700, s0=40.0, s1=450.0, lr0=0.010, lr1=0.0005):
        X = X0.copy()
        t_start = time.time()
        m1 = np.zeros_like(X); v1 = np.zeros_like(X)
        b1, b2, eps = 0.9, 0.999, 1e-8
        b1p, b2p = b1, b2
        ls0, ls1 = np.log(s0), np.log(s1)
        ll0, ll1 = np.log(lr0), np.log(lr1)
        I0, J0 = np.triu_indices(n, 1)
        for k in range(steps):
            if time.time() - t_start >= budget:
                break
            if time.time() >= DEADLINE - 8.0:
                break
            frac = k / (steps - 1) if steps > 1 else 1.0
            s = np.exp(ls0 + (ls1 - ls0) * frac)
            lr = np.exp(ll0 + (ll1 - ll0) * frac)
            G = X @ X.T
            np.fill_diagonal(G, -2.0)
            c = G[I0, J0]
            cmax = c.max()
            e = np.exp(s * (c - cmax))
            ss = e.sum()
            if not np.isfinite(ss) or ss == 0:
                break
            w = e / ss
            W = np.zeros((n, n))
            W[I0, J0] = w
            W[J0, I0] = w
            Gg = W @ X
            dot = (Gg * X).sum(axis=1, keepdims=True)
            Gt = Gg - dot * X
            m1 *= b1; m1 += (1 - b1) * Gt
            v1 *= b2; v1 += (1 - b2) * (Gt * Gt)
            mh = m1 / (1 - b1p); vh = v1 / (1 - b2p)
            b1p *= b1; b2p *= b2
            X -= lr * mh / (np.sqrt(vh) + eps)
            X /= np.linalg.norm(X, axis=1, keepdims=True)
        return X

    trial = -1
    N_CAND = 80000
    CHUNK = 20000
    K_TOP = 5
    while time.time() < EXPLORE_END and time.time() < DEADLINE - 10.0:
        trial += 1
        mode = trial % 4
        cap = cap_sizes[trial % len(cap_sizes)]
        base = X_best
        # --- cap center ---
        try:
            if mode == 0:
                # rattler direction
                G = base @ base.T
                np.fill_diagonal(G, -2.0)
                cm = G.max()
                cnts = (G >= cm - 1e-9).sum(axis=1)
                idx = int(np.argmin(cnts))
                center = base[idx].copy()
                if cnts[idx] > 0 and (cnts == 0).any():
                    idx = int(np.where(cnts == 0)[0][0])
                    center = base[idx].copy()
            elif mode == 1:
                idx = int(rng.integers(0, n))
                center = base[idx].copy()
            elif mode == 2:
                G = base @ base.T
                np.fill_diagonal(G, -2.0)
                cm = G.max()
                cnts = (G >= cm - 1e-3).sum(axis=1)
                idx = int(np.argmax(cnts))
                center = base[idx].copy()
            else:
                # PCA outlier
                Mu = base.mean(axis=0, keepdims=True)
                Z = base - Mu
                try:
                    U, S, Vt = np.linalg.svd(Z, full_matrices=False)
                    V2 = Vt[:2, :].T
                    P = Z @ V2
                except Exception:
                    P = Z[:, :2]
                dists = np.linalg.norm(P, axis=1)
                idx = int(np.argmax(dists))
                # most isolated outlier direction; also randomize slightly among top3
                if rng.random() < 0.3:
                    order = np.argsort(-dists)
                    idx = int(order[int(rng.integers(0, min(3, n)))])
                center = base[idx].copy()
        except Exception:
            continue
        # --- delete cap ---
        try:
            dots_c = base @ center
            order = np.argsort(-dots_c)
            victims = order[:cap]
            frozen_mask = np.ones(n, dtype=bool)
            frozen_mask[victims] = False
            F = base[frozen_mask]  # ((n-cap), d)
            # greedy rebuild
            rebuilt = []  # list of (victim_idx, vec)
            rebuilt_vecs = []
            for vi, v in enumerate(victims):
                if time.time() >= DEADLINE - 10.0 or time.time() >= EXPLORE_END:
                    break
                if len(rebuilt_vecs) == 0:
                    Obs = F
                else:
                    Obs = np.vstack([F] + rebuilt_vecs)
                # sample 80k, track top K_TOP
                cand_pool = []
                md_pool = []
                n_chunks = N_CAND // CHUNK
                for _ in range(n_chunks):
                    C = rng.standard_normal((CHUNK, d))
                    C /= np.linalg.norm(C, axis=1, keepdims=True)
                    D = C @ Obs.T
                    md = D.max(axis=1)
                    # smallest K_TOP in chunk
                    bi = np.argpartition(md, K_TOP - 1)[:K_TOP]
                    cand_pool.append(C[bi])
                    md_pool.append(md[bi])
                if len(cand_pool) == 0:
                    break
                Pall = np.vstack(cand_pool)
                Mall = np.concatenate(md_pool)
                bi2 = np.argpartition(Mall, K_TOP - 1)[:K_TOP]
                Ctop = Pall[bi2]
                # refine best few
                try:
                    Cref = single_refine(Ctop, Obs, steps=200)
                except Exception:
                    Cref = Ctop
                Dref = Cref @ Obs.T
                mdref = Dref.max(axis=1)
                winner = Cref[int(np.argmin(mdref))]
                rebuilt_vecs.append(winner.reshape(1, d).copy())
            if len(rebuilt_vecs) != cap:
                continue
            X_new = base.copy()
            for v, wv in zip(victims, rebuilt_vecs):
                X_new[int(v)] = wv[0]
            X_new /= np.linalg.norm(X_new, axis=1, keepdims=True)
            # block-only optimization
            try:
                Y0 = np.vstack(rebuilt_vecs)
                Yopt = block_refine(Y0, F, steps=300)
                for v, yv in zip(victims, Yopt):
                    X_new[int(v)] = yv
                X_new /= np.linalg.norm(X_new, axis=1, keepdims=True)
            except Exception:
                pass
            # unfreeze 1-sec smooth relaxation
            try:
                X_new = global_relax(X_new, budget=1.0, steps=700)
            except Exception:
                pass
            v_new = _max_cos(X_new)
            # LP polish only if within 0.001 of incumbent or better
            if v_new < best_val + 0.001:
                rem = DEADLINE - time.time()
                if rem > 9.0 and time.time() < EXPLORE_END + 4.0:
                    try:
                        X_pol = _lp_polish(X_new, min(6.0, rem - 2.0), t_end=DEADLINE - 1.5)
                        v_pol = _max_cos(X_pol)
                        # accept polished version
                        X_new, v_new = X_pol, v_pol
                    except Exception:
                        pass
            if v_new < best_val - 1e-15:
                X_best = X_new.copy()
                best_val = v_new
        except Exception:
            continue
    X_best /= np.linalg.norm(X_best, axis=1, keepdims=True)
    X_best = np.where(np.isfinite(X_best), X_best, rng.standard_normal((n, d)))
    X_best /= np.linalg.norm(X_best, axis=1, keepdims=True)
    return X_best
