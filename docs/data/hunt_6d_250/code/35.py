import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def _rowmax_G(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    rm = G.max(axis=1)
    return rm, G

def _single_refine(y, R, beta=350.0, steps=60, lr=0.14):
    # R: (r,d) frozen+already grown, y: (d,) unit
    cur_lr = lr
    for _ in range(steps):
        c = R @ y
        m = float(np.max(c))
        w = np.exp(beta * (c - m))
        s = float(np.sum(w)) + 1e-12
        mean = (w @ R) / s
        dot = float(mean @ y)
        tang = mean - dot * y
        y = y - cur_lr * tang
        n = float(np.linalg.norm(y))
        if n == 0 or not np.finite(n):
            break
        y = y / n
        cur_lr *= 0.995
    return y

def _block_refine(B, F, beta=900.0, steps=250, lr=0.008, mu=0.85):
    # B: (m,d) variable, F: (f,d) frozen
    m = B.shape[0]
    V = np.zeros_like(B)
    cur_lr = lr
    for _ in range(steps):
        Gbb = B @ B.T
        np.fill_diagonal(Gbb, -2.0)
        Gbf = B @ F.T
        cur_max = float(Gbb.max()) if m > 1 else -2.0
        fmax = float(Gbf.max()) if Gbf.size else -2.0
        if fmax > cur_max:
            cur_max = fmax
        Wbb = np.exp(np.clip(beta * (Gbb - cur_max), -30.0, 0.0))
        if m > 1:
            np.fill_diagonal(Wbb, 0.0)
        Wbf = np.exp(np.clip(beta * (Gbf - cur_max), -30.0, 0.0))
        rs = Wbb.sum(axis=1, keepdims=True) + Wbf.sum(axis=1, keepdims=True) + 1e-12
        mean = (Wbb @ B + Wbf @ F) / rs
        dot = np.sum(mean * B, axis=1, keepdims=True)
        tang = mean - dot * B
        V = mu * V + tang
        B = B - cur_lr * V
        nr = np.linalg.norm(B, axis=1, keepdims=True)
        nr[nr == 0] = 1.0
        B = B / nr
    return B

def _global_descent(X, beta=900.0, steps=300, lr=0.007, mu=0.86):
    V = np.zeros_like(X)
    for it in range(steps):
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        m = float(np.max(G))
        W = np.exp(np.clip(beta * (G - m), -30.0, 0.0))
        np.fill_diagonal(W, 0.0)
        rs = W.sum(axis=1, keepdims=True) + 1e-12
        mean = (W @ X) / rs
        dot = np.sum(mean * X, axis=1, keepdims=True)
        tang = mean - dot * X
        V = mu * V + tang
        cur_lr = lr * (1.0 - 0.4 * it / steps)
        X = X - cur_lr * V
        nr = np.linalg.norm(X, axis=1, keepdims=True)
        nr[nr == 0] = 1.0
        X = X / nr
    return X

def _lp_polish(x, deadline, k_top=5000, radius_init=2.0e-3):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = radius_init
    iu0, iu1 = np.triu_indices(n, 1)
    ntri = iu0.shape[0]
    K = min(k_top, ntri)
    while time.time() < deadline - 1.0 and radius > 1e-13:
        Gall = x @ x.T
        c_all = Gall[iu0, iu1]
        if K < ntri:
            top = np.argpartition(-c_all, K)[:K]
            I = iu0[top]; J = iu1[top]; c_sel = c_all[top]
        else:
            I = iu0; J = iu1; c_sel = c_all
        p = I.shape[0]
        nv = n * d + 1
        cols_blocks = np.concatenate(
            [I[:, None] * d + np.arange(d),
             J[:, None] * d + np.arange(d),
             np.full((p, 1), n * d)], axis=1).ravel()
        rows = np.repeat(np.arange(p), 2 * d + 1)
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], axis=1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols_blocks)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix(
                (x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))),
                shape=(n, nv)).tocsr()
            obj = np.zeros(nv); obj[-1] = 1.0
            bounds = [(-radius, radius)] * (n * d) + [(None, None)]
            res = linprog(obj, A_ub=A_ub, b_ub=-c_sel, A_eq=A_eq,
                          b_eq=np.zeros(n), bounds=bounds, method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        nr = np.linalg.norm(y, axis=1, keepdims=True)
        nr[nr == 0] = 1.0
        y = y / nr
        if not np.all(np.isfinite(y)):
            radius *= 0.5
            continue
        val = _max_cos(y)
        if val < best - 1e-12:
            x, best = y, val
            radius = min(radius * 1.5, 0.012)
        else:
            radius *= 0.5
        if time.time() > deadline - 1.5:
            break
    return x

def _component_of_worst(G, thresh, a):
    n = G.shape[0]
    adj = G > thresh
    visited = np.zeros(n, dtype=bool)
    visited[a] = True
    stack = [int(a)]
    while stack:
        u = stack.pop()
        nbrs = np.where(adj[u])[0]
        for w in nbrs:
            w = int(w)
            if not visited[w]:
                visited[w] = True
                stack.append(w)
    return np.where(visited)[0]

def solve():
    t0 = time.time()
    HARD = t0 + 175.0
    REGROW_END = t0 + 140.0
    n, d = 250, 6
    cands = []
    for f in ("best.npy", "record.npy"):
        try:
            a = np.asarray(np.load(f), dtype=float)
            if a.shape == (n, d):
                a = np.nan_to_num(a)
                a = a / np.linalg.norm(a, axis=1, keepdims=True)
                if np.all(np.isfinite(a)):
                    cands.append(a)
        except Exception:
            pass
    rng0 = np.random.default_rng(0)
    if not cands:
        z = rng0.standard_normal((n, d))
        z /= np.linalg.norm(z, axis=1, keepdims=True)
        cands.append(z)
    best = min(cands, key=_max_cos).copy()
    best_val = _max_cos(best)
    fallback = best.copy()
    fallback_val = best_val

    # configs: (excise_size, seed, beta_single)
    sizes_cycle = [25, 15, 40, 25, 30, 20, 35, 15, 40, 25]
    seed_base = 1000
    attempt = 0
    # precompute nothing; loop
    while time.time() < REGROW_END and attempt < 60:
        K = sizes_cycle[attempt % len(sizes_cycle)]
        seed = seed_base + attempt * 977 + K * 13
        attempt += 1
        try:
            rm, G = _rowmax_G(best)
            cur_max = float(np.max(G))
            # worst pair
            iu = np.triu_indices(n, 1)
            flat = G[iu]
            wi = int(np.argmax(flat))
            a_worst = int(iu[0][wi]); b_worst = int(iu[1][wi])
            thresh = cur_max - 2e-5
            comp = _component_of_worst(G, thresh, a_worst)
            # order component by rowmax
            order = comp[np.argsort(-rm[comp])]
            if len(order) >= K:
                exc = order[:K].tolist()
            else:
                exc = order.tolist()
                # fill with global worst outside
                glob = np.argsort(-rm)
                for g in glob:
                    g = int(g)
                    if g not in exc:
                        exc.append(g)
                    if len(exc) >= K:
                        break
            # ensure bottleneck included
            for must in (a_worst, b_worst):
                if must not in exc:
                    exc[-1] = must
            exc = np.array(exc, dtype=int)
            mask = np.ones(n, dtype=bool)
            mask[exc] = False
            frozen = best[mask]  # (n-K, d)
            # frozen internal max must be below best to have a chance
            Gf = frozen @ frozen.T
            np.fill_diagonal(Gf, -2.0)
            frozen_max = float(np.max(Gf))
            if frozen_max >= best_val - 1e-12:
                # need larger excision to break frozen bottleneck; skip this K
                # try to expand excision by adding the frozen worst pair endpoints
                # quick fix: extend by 5 more worst points and retry once
                if K < 40 and time.time() < REGROW_END - 2.0:
                    pass
                continue
            rng = np.random.default_rng(seed)
            block_list = []
            ok = True
            for pos in range(K):
                if time.time() > REGROW_END - 1.0:
                    ok = False
                    break
                if len(block_list) == 0:
                    R = frozen
                else:
                    R = np.vstack([frozen, np.stack(block_list)])
                # propose 8192 candidates
                cand = rng.standard_normal((8192, d))
                cand /= np.linalg.norm(cand, axis=1, keepdims=True)
                S = cand @ R.T
                mx = S.max(axis=1)
                bi = int(np.argmin(mx))
                y = cand[bi].copy()
                # refine 60 iterations single-point smooth-minimax
                y = _single_refine(y, R, beta=380.0, steps=60, lr=0.14)
                block_list.append(y)
                # after every 5 insertions jointly re-optimise block
                if (pos + 1) % 5 == 0:
                    B = np.stack(block_list)
                    # vary beta slightly per cycle for diversity
                    beta_b = 700.0 + (seed % 5) * 150.0
                    B = _block_refine(B, frozen, beta=beta_b, steps=220, lr=0.008, mu=0.85)
                    block_list = [B[i] for i in range(B.shape[0])]
            if not ok:
                continue
            B = np.stack(block_list)
            # final joint optimisation of inserted block
            B = _block_refine(B, frozen, beta=1100.0, steps=350, lr=0.007, mu=0.86)
            # reassemble
            Xnew = np.empty_like(best)
            Xnew[mask] = frozen
            Xnew[exc] = B
            Xnew /= np.linalg.norm(Xnew, axis=1, keepdims=True)
            v = _max_cos(Xnew)
            # short collective polish for fair comparison if close
            if v < best_val + 0.002 and time.time() < REGROW_END - 1.0:
                try:
                    Xp = _global_descent(Xnew.copy(), beta=1200.0, steps=250, lr=0.006, mu=0.86)
                    vp = _max_cos(Xp)
                    if vp < v:
                        Xnew, v = Xp, vp
                except Exception:
                    pass
            if v < best_val - 1e-12:
                best, best_val = Xnew, v
                # keep going to improve further; update fallback
                fallback = best.copy()
                fallback_val = best_val
        except Exception:
            continue

    # if no improvement, keep fallback (best.npy)
    if best_val > fallback_val:
        best, best_val = fallback, fallback_val
    # short collective polish time-boxed
    try:
        if time.time() < HARD - 12.0:
            # annealed global descent
            steps_left = int(min(600, max(150, (HARD - 12.0 - time.time()) * 60)))
            Xp = _global_descent(best.copy(), beta=800.0, steps=min(300, steps_left), lr=0.005, mu=0.86)
            vp = _max_cos(Xp)
            if vp < best_val:
                best, best_val = Xp, vp
            if time.time() < HARD - 9.0:
                Xp2 = _global_descent(best.copy(), beta=2500.0, steps=200, lr=0.003, mu=0.87)
                vp2 = _max_cos(Xp2)
                if vp2 < best_val:
                    best, best_val = Xp2, vp2
    except Exception:
        pass
    try:
        if time.time() < HARD - 4.0:
            best = _lp_polish(best, HARD - 1.2, k_top=5000, radius_init=2.0e-3)
    except Exception:
        pass
    best = best / np.linalg.norm(best, axis=1, keepdims=True)
    best = np.nan_to_num(best)
    # safety: if somehow worse than fallback due to LP issue, restore
    try:
        bv = _max_cos(best)
        if bv > fallback_val and fallback_val < 10:
            # keep best anyway if fallback file check? we want lowest; restore fallback
            if _max_cos(fallback) < bv:
                best = fallback
    except Exception:
        best = fallback
    return best
