import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _deg_sig(X, thr=1e-4):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    m = float(G.max())
    D = (G > (m - thr)).sum(axis=1)
    s = tuple(sorted(D.tolist()))
    return s

def _block_opt(B0, F, steps=600, b0=60.0, b1=450.0, lr0=0.06, lr1=0.006):
    B = B0.copy()
    B /= np.linalg.norm(B, axis=1, keepdims=True)
    lb0, lb1 = np.log(b0), np.log(b1)
    ll0, ll1 = np.log(lr0), np.log(lr1)
    for s in range(steps):
        frac = s / max(steps - 1, 1)
        beta = np.exp(lb0 + (lb1 - lb0) * frac)
        lr = np.exp(ll0 + (ll1 - ll0) * frac)
        Gbf = B @ F.T
        Gbb = B @ B.T
        # max over both
        m = Gbf.max() if Gbf.size else -2.0
        mb = Gbb.max() if Gbb.size else -2.0
        # exclude diagonal from max: diagonal is 1, must ignore
        # Gbb diag =1, so mb=1 always; correct m to ignore diag
        # compute m ignoring diag
        tmp = Gbb.copy()
        np.fill_diagonal(tmp, -2.0)
        mb2 = float(tmp.max())
        m = m if m >= mb2 else mb2
        Ebf = np.exp(beta * (Gbf - m))
        Ebb = np.exp(beta * (Gbb - m))
        np.fill_diagonal(Ebb, 0.0)
        S = float(Ebf.sum() + Ebb.sum())
        if not np.isfinite(S) or S == 0:
            continue
        Grad = (Ebf @ F + 2.0 * (Ebb @ B)) / S
        dot = (Grad * B).sum(axis=1, keepdims=True)
        Gt = Grad - dot * B
        # clip
        nn = np.sqrt((Gt * Gt).sum(axis=1, keepdims=True)) + 1e-12
        sc = np.minimum(1.0, 2.0 / (nn + 1e-12))
        Gt = Gt * sc
        B = B - lr * Gt
        B /= np.linalg.norm(B, axis=1, keepdims=True)
    return B

def _refine_holes(H, F, steps=70, beta=120.0, lr=0.05):
    H = H.copy()
    H /= np.linalg.norm(H, axis=1, keepdims=True)
    for _ in range(steps):
        G = H @ F.T
        m = float(G.max())
        E = np.exp(beta * (G - m))
        S = float(E.sum()) + 1e-30
        Grad = (E @ F) / S
        dot = (Grad * H).sum(axis=1, keepdims=True)
        Gt = Grad - dot * H
        H = H - lr * Gt
        H /= np.linalg.norm(H, axis=1, keepdims=True)
    return H

def _balanced_quench(X, steps, b0, b1, lr0, lr1, lam=0.12):
    n, d = X.shape
    X = X.copy()
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    lb0, lb1 = np.log(b0), np.log(b1)
    ll0, ll1 = np.log(lr0), np.log(lr1)
    for s in range(steps):
        frac = s / max(steps - 1, 1)
        beta = np.exp(lb0 + (lb1 - lb0) * frac)
        lr = np.exp(ll0 + (ll1 - ll0) * frac)
        G = X @ X.T
        np.fill_diagonal(G, -10.0)
        m = float(G.max())
        E = np.exp(beta * (G - m))
        np.fill_diagonal(E, 0.0)
        Sg = float(E.sum()) / max(n, 1) + 1e-30
        Grad_g = (E @ X) / Sg
        # local term
        bl = beta * 0.25
        if bl < 8.0:
            bl = 8.0
        mx = G.max(axis=1, keepdims=True)
        W = np.exp(bl * (G - mx))
        np.fill_diagonal(W, 0.0)
        Rs = W.sum(axis=1, keepdims=True) + 1e-30
        Grad_l = (W @ X) / Rs
        Grad = Grad_g + lam * Grad_l
        dot = (Grad * X).sum(axis=1, keepdims=True)
        Gt = Grad - dot * X
        nn = np.sqrt((Gt * Gt).sum(axis=1, keepdims=True)) + 1e-12
        sc = np.minimum(1.0, 3.0 / (nn + 1e-12))
        Gt = Gt * sc
        X = X - lr * Gt
        X /= np.linalg.norm(X, axis=1, keepdims=True)
    return X

def _hard_polish(X, steps=260, lr0=0.003, lr1=0.00012, tol0=2e-4, tol1=1e-6):
    X = X.copy()
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    n = X.shape[0]
    for s in range(steps):
        frac = s / max(steps - 1, 1)
        tol = tol0 + (tol1 - tol0) * frac
        lr = lr0 * (lr1 / lr0) ** frac
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        M = (G > (m - tol)).astype(float)
        np.fill_diagonal(M, 0.0)
        cnt = M.sum(axis=1, keepdims=True) + 1e-12
        active = (cnt[:, 0] > 0.5)
        if not np.any(active):
            continue
        Grad = (M @ X) / cnt
        dot = (Grad * X).sum(axis=1, keepdims=True)
        Gt = Grad - dot * X
        nn = np.sqrt((Gt * Gt).sum(axis=1, keepdims=True)) + 1e-12
        step = np.zeros_like(Gt)
        step[active] = Gt[active] / nn[active]
        X = X - lr * step
        X /= np.linalg.norm(X, axis=1, keepdims=True)
    return X

def _lp_polish(x, seconds, r0=1.0e-3):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    try:
        best = _max_cos(x)
    except Exception:
        return x
    radius = r0
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        c = (x @ x.T)[iu]
        m = float(c.max())
        sel = c > m - max(20 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        if p == 0:
            radius *= 0.5
            continue
        if p > 2500:
            idx = np.argsort(c[sel])[-2500:]
            I, J = I[idx], J[idx]
            cc = c[sel][idx]
        else:
            cc = c[sel]
            p = len(I)
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d),
                                           np.arange(n * d))), shape=(n, nv)).tocsr()
        except Exception:
            radius *= 0.5
            continue
        obj = np.zeros(nv)
        obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-cc, A_eq=A_eq,
                          b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                          method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        v = _max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def solve():
    np.random.seed(7)
    t0 = time.time()
    TOTAL = 172.0
    try:
        X_best = np.load("best.npy").astype(float)
    except Exception:
        try:
            X_best = np.load("record.npy").astype(float)
        except Exception:
            X_best = np.random.randn(168, 8)
    X_best = np.asarray(X_best, dtype=float).reshape(168, 8)
    X_best /= np.linalg.norm(X_best, axis=1, keepdims=True)
    best_val = _max_cos(X_best)
    X_cur = X_best.copy()
    cur_val = best_val
    arch = set()
    try:
        arch.add(_deg_sig(X_best))
    except Exception:
        pass
    Tsa = 8e-7
    it = 0
    while time.time() - t0 < TOTAL - 20.0:
        it += 1
        # choose direction: hub or random
        if np.random.rand() < 0.5:
            try:
                Gc = X_cur @ X_cur.T
                np.fill_diagonal(Gc, -2.0)
                mc = float(Gc.max())
                degs = (Gc > (mc - 1e-4)).sum(axis=1)
                # add tiny noise to break ties
                hub = int(np.argmax(degs + np.random.rand(168) * 0.3))
                direction = X_cur[hub].copy()
                nl = float(np.linalg.norm(direction))
                if nl == 0 or not np.isfinite(nl):
                    direction = np.random.randn(8)
                    direction /= np.linalg.norm(direction)
                else:
                    direction /= nl
            except Exception:
                direction = np.random.randn(8)
                direction /= np.linalg.norm(direction)
        else:
            direction = np.random.randn(8)
            direction /= np.linalg.norm(direction)
        k = int(np.random.choice(np.array([10, 16, 24])))
        try:
            dots = X_cur @ direction
        except Exception:
            continue
        keep_idx = np.argsort(dots)[:168 - k]
        F = X_cur[keep_idx].copy()
        # 20000 probes
        try:
            P = np.random.randn(20000, 8)
            P /= np.linalg.norm(P, axis=1, keepdims=True)
        except Exception:
            continue
        # score probes vs frozen
        m_vals = np.empty(20000, dtype=float)
        try:
            for s0 in range(0, 20000, 5000):
                ch = P[s0:s0 + 5000]
                Gp = ch @ F.T
                m_vals[s0:s0 + ch.shape[0]] = Gp.max(axis=1)
        except Exception:
            continue
        order = np.argsort(m_vals)
        # refined holes from deepest 12
        try:
            H0 = P[order[:12]].copy()
            Href = _refine_holes(H0, F, steps=60, beta=120.0, lr=0.05)
            # pool: top 2000 + refined
            pool = np.vstack([P[order[:2000]], Href])
            pool_m = np.empty(pool.shape[0], dtype=float)
            for s0 in range(0, pool.shape[0], 1000):
                ch = pool[s0:s0 + 1000]
                pool_m[s0:s0 + ch.shape[0]] = (ch @ F.T).max(axis=1)
            porder = np.argsort(pool_m)
            pool_sorted = pool[porder]
        except Exception:
            pool_sorted = P[order]
        # greedy mutually distant selection
        sel_list = []
        for thr in (0.5, 0.65, 0.8, 0.95, 2.0):
            sel_list = []
            for cand in pool_sorted:
                if len(sel_list) >= k:
                    break
                if len(sel_list) == 0:
                    sel_list.append(cand)
                else:
                    S = np.stack(sel_list)
                    md = float((S @ cand).max())
                    if md < thr:
                        sel_list.append(cand)
            if len(sel_list) >= k:
                break
        if len(sel_list) < k:
            continue
        B0 = np.stack(sel_list[:k])
        # joint block optimisation with high beta
        try:
            B = _block_opt(B0, F, steps=550, b0=60.0, b1=450.0, lr0=0.06, lr1=0.006)
        except Exception:
            continue
        X_new = np.vstack([F, B])
        # balanced quench unfreezed
        try:
            # vary quench length a bit
            if np.random.rand() < 0.3:
                X_new = _balanced_quench(X_new, steps=1100, b0=25.0, b1=550.0, lr0=0.045, lr1=0.0025, lam=0.12)
            else:
                X_new = _balanced_quench(X_new, steps=850, b0=30.0, b1=500.0, lr0=0.04, lr1=0.003, lam=0.10)
            X_new = _hard_polish(X_new, steps=220)
        except Exception:
            continue
        try:
            v = _max_cos(X_new)
        except Exception:
            continue
        # short LP if promising
        if v < best_val + 5e-6 and (time.time() - t0) < TOTAL - 22.0:
            try:
                X_new = _lp_polish(X_new, seconds=1.0, r0=1.0e-3)
                v = _max_cos(X_new)
            except Exception:
                pass
        # basin rejection at 1e-4 degree sequence
        try:
            sig = _deg_sig(X_new)
        except Exception:
            sig = None
        is_archived = (sig is not None) and (sig in arch)
        if is_archived and not (v < best_val):
            # reject: do not accept, archive stays, continue
            # still occasionally add noise to cur to escape? just skip
            continue
        if sig is not None:
            arch.add(sig)
            if len(arch) > 2000:
                # drop arbitrary element (convert to list pop)
                arch.pop()
        if v < best_val:
            best_val = v
            X_best = X_new.copy()
            X_cur = X_new.copy()
            cur_val = v
        elif v < best_val + 2e-6:
            # SA acceptance within band
            if v < cur_val or np.random.rand() < np.exp(-(v - cur_val) / max(Tsa, 1e-12)):
                X_cur = X_new.copy()
                cur_val = v
        # occasional random restart of cur to best to focus
        if it % 60 == 0 and np.random.rand() < 0.5:
            X_cur = X_best.copy()
            cur_val = best_val
        if time.time() - t0 > TOTAL - 21.0:
            break
    # final polish of best
    try:
        if time.time() - t0 < TOTAL - 12.0:
            X_best = _balanced_quench(X_best, steps=900, b0=40.0, b1=700.0, lr0=0.02, lr1=0.0015, lam=0.10)
            X_best = _hard_polish(X_best, steps=300)
    except Exception:
        pass
    remain = TOTAL - (time.time() - t0) - 1.0
    if remain > 3.0:
        try:
            X_best = _lp_polish(X_best, seconds=min(remain, 8.0), r0=8e-4)
        except Exception:
            pass
    X_best /= np.linalg.norm(X_best, axis=1, keepdims=True)
    return X_best
