import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _per_max(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return G.max(axis=1)

def _metrics(X, slack_core=2e-4, tol_frac=1e-4):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    M = float(G.max())
    mi = G.max(axis=1)
    frac = float((mi > M - tol_frac).mean())
    deg = (G > M - slack_core).sum(axis=1).astype(float)
    mn = float(deg.min())
    std = float(deg.std())
    # entropy of degree histogram
    try:
        vals, cnts = np.unique(deg, return_counts=True)
        p = cnts.astype(float) / cnts.sum()
        ent = float(-(p * np.log(p + 1e-30)).sum())
    except Exception:
        ent = 0.0
    return M, mi, frac, mn, std, ent, deg

def _anneal_balanced(X, steps, beta0=30.0, beta1=300.0, lr0=1e-2, lr1=1e-4, lam=0.15):
    n, d = X.shape
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    m_ad = np.zeros_like(X)
    v_ad = np.zeros_like(X)
    b1 = 0.9; b2 = 0.999; eps = 1e-8
    lb0 = np.log(beta0); lb1 = np.log(beta1)
    ll0 = np.log(lr0); ll1 = np.log(lr1)
    try:
        best_m = _max_cos(X)
    except Exception:
        best_m = 1e9
    best = X.copy()
    for t in range(1, steps + 1):
        frac = t / steps
        beta = np.exp(lb0 + (lb1 - lb0) * frac)
        lr = np.exp(ll0 + (ll1 - ll0) * frac)
        G = X @ X.T
        np.fill_diagonal(G, -10.0)
        gmax = float(G.max())
        if gmax < best_m:
            best_m = gmax
            best = X.copy()
        # global softmax
        W = np.exp(beta * (G - gmax))
        np.fill_diagonal(W, 0.0)
        Z = float(W.sum())
        if Z == 0 or not np.isfinite(Z):
            continue
        Gg = (W @ X) / Z
        # per-point (row) softmax
        bl = beta * 0.25
        if bl < 8.0:
            bl = 8.0
        mx = G.max(axis=1, keepdims=True)
        Wl = np.exp(bl * (G - mx))
        np.fill_diagonal(Wl, 0.0)
        Rs = Wl.sum(axis=1, keepdims=True) + 1e-30
        Gl = (Wl @ X) / Rs
        # tame local part row-wise
        nrm_l = np.sqrt((Gl * Gl).sum(axis=1, keepdims=True)) + 1e-12
        Gl = Gl / nrm_l * np.minimum(nrm_l, 3.0)
        Grad = Gg + lam * Gl
        dot = (Grad * X).sum(axis=1, keepdims=True)
        Gt = Grad - dot * X
        m_ad *= b1; m_ad += (1.0 - b1) * Gt
        v_ad *= b2; v_ad += (1.0 - b2) * (Gt * Gt)
        bc1 = 1.0 - b1 ** t; bc2 = 1.0 - b2 ** t
        step = (m_ad / bc1) / (np.sqrt(v_ad / bc2) + eps)
        X = X - lr * step
        nrm = np.sqrt((X * X).sum(axis=1, keepdims=True))
        nrm[nrm == 0] = 1.0
        X /= nrm
    try:
        vm = _max_cos(X)
        if vm < best_m:
            best = X.copy()
    except Exception:
        pass
    return best

def _lp_polish(x, deadline, r0=1.2e-3, cap=2800):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    try:
        best = _max_cos(x)
    except Exception:
        return x
    radius = r0
    iu = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        try:
            c = (x @ x.T)[iu]
        except Exception:
            break
        m = float(c.max())
        sel = c > m - max(20.0 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        if p == 0:
            radius *= 0.5
            continue
        if p > cap:
            idx = np.argsort(c[sel])[-cap:]
            I, J = I[idx], J[idx]
            cc = c[sel][idx]
        else:
            cc = c[sel]
        p = len(I)
        nv = n * d + 1
        try:
            rows = np.repeat(np.arange(p), 2 * d + 1)
            cols = np.concatenate([I[:, None] * d + np.arange(d),
                                   J[:, None] * d + np.arange(d),
                                   np.full((p, 1), n * d)], 1).ravel()
            vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d),
                                           np.arange(n * d))), shape=(n, nv)).tocsr()
        except Exception:
            radius *= 0.5
            continue
        obj = np.zeros(nv); obj[-1] = 1.0
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
        try:
            y = x + res.x[:-1].reshape(n, d)
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            vv = _max_cos(y)
        except Exception:
            radius *= 0.5
            continue
        if vv < best:
            x, best = y, vv
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def _refill_voids(Y, k, rng, n_probes=20000):
    # Y: (n-k,8) remaining, return (n,8) with k refined probes
    nrem = Y.shape[0]
    P = rng.randn(n_probes, 8)
    P /= np.linalg.norm(P, axis=1, keepdims=True)
    try:
        D = P @ Y.T
        s = D.max(axis=1)
    except Exception:
        s = P[:, 0]
    order_p = np.argsort(s)
    thresh = 0.55
    selected = []
    for idx in order_p:
        if len(selected) >= k:
            break
        pi = P[int(idx)]
        ok = True
        for sj in selected:
            if float(pi @ P[int(sj)]) > thresh:
                ok = False
                break
        if ok:
            selected.append(int(idx))
    if len(selected) < k:
        for idx in order_p:
            ii = int(idx)
            if ii not in selected:
                selected.append(ii)
            if len(selected) >= k:
                break
    Psel = P[np.array(selected[:k])]
    # softmax hole minimization against Y + mutual repulsion
    Ps = Psel.copy()
    beta_h = 50.0
    lr_h = 0.05
    for it in range(250):
        lr = lr_h * (0.3 + 0.7 * (1.0 - it / 250.0))
        try:
            D1 = Ps @ Y.T
            mx1 = D1.max(axis=1, keepdims=True)
            W1 = np.exp(beta_h * (D1 - mx1))
            S1 = W1.sum(axis=1, keepdims=True) + 1e-30
            Gy = (W1 @ Y) / S1
            Ds = Ps @ Ps.T
            np.fill_diagonal(Ds, -10.0)
            mx2 = Ds.max(axis=1, keepdims=True)
            W2 = np.exp(beta_h * (Ds - mx2))
            np.fill_diagonal(W2, 0.0)
            S2 = W2.sum(axis=1, keepdims=True) + 1e-30
            Gs = (W2 @ Ps) / S2
        except Exception:
            break
        Grad = Gy + 0.6 * Gs
        dot = (Grad * Ps).sum(axis=1, keepdims=True)
        Gt = Grad - dot * Ps
        Ps = Ps - lr * Gt
        nrm = np.sqrt((Ps * Ps).sum(axis=1, keepdims=True))
        nrm[nrm == 0] = 1.0
        Ps /= nrm
    Xnew = np.vstack([Y, Ps])
    # random permute rows
    try:
        perm = rng.permutation(Xnew.shape[0])
        Xnew = Xnew[perm]
    except Exception:
        pass
    return Xnew

def solve():
    t0 = time.time()
    BUDGET = 175.0
    END = t0 + BUDGET
    rng = np.random.RandomState(7)
    best = None
    best_val = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn).astype(np.float64)
            if a.shape != (168, 8):
                continue
            a /= np.linalg.norm(a, axis=1, keepdims=True)
            if not np.all(np.isfinite(a)):
                continue
            va = _max_cos(a)
            if va < best_val:
                best_val = va
                best = a.copy()
        except Exception:
            pass
    if best is None:
        best = rng.randn(168, 8)
        best /= np.linalg.norm(best, axis=1, keepdims=True)
        best_val = _max_cos(best)
    # base for surgery (may be slightly worse but balanced)
    base = best.copy()
    base_val = best_val
    try:
        _, _, base_frac, base_mn, base_std, base_ent, _ = _metrics(base)
    except Exception:
        base_frac, base_mn, base_std, base_ent = 0.0, 0.0, 1e9, 0.0

    def is_better_balance(frac, mn, std, ent):
        # substantially improves balance metrics
        if mn > base_mn + 0.5:
            return True
        if frac > base_frac + 0.02:
            return True
        if std < base_std - 0.3:
            return True
        if ent > base_ent + 0.15 and mn >= base_mn:
            return True
        return False

    # surgery schedule: several cluster sizes
    sizes = [12, 16, 20, 24, 14, 18, 22, 12]
    trial = 0
    for k in sizes:
        if time.time() > END - 12.0:
            break
        trial += 1
        try:
            Xb = base.copy()
            G = Xb @ Xb.T
            np.fill_diagonal(G, -2.0)
            M0 = float(G.max())
            slack = 2e-4
            Adj = G > (M0 - slack)
            np.fill_diagonal(Adj, False)
            deg = Adj.sum(axis=1)
            try:
                sumG = np.where(Adj, G, 0.0).sum(axis=1)
            except Exception:
                sumG = deg.astype(float)
            score_deg = deg.astype(float) * 10.0 + sumG
            order = np.argsort(-score_deg)
            # choose removal set: alternate top-k vs dense cluster
            if rng.rand() < 0.5:
                rem = order[:k]
            else:
                top5 = order[:min(5, len(order))]
                seed = int(top5[rng.randint(0, len(top5))])
                neigh = np.where(Adj[seed])[0]
                if len(neigh) > 0:
                    try:
                        ns = neigh[np.argsort(-score_deg[neigh])]
                    except Exception:
                        ns = neigh
                else:
                    ns = np.array([], dtype=int)
                cand = [seed] + [int(v) for v in ns]
                seen = set(cand)
                for idx in order:
                    if len(cand) >= k:
                        break
                    ii = int(idx)
                    if ii not in seen:
                        cand.append(ii)
                        seen.add(ii)
                rem = np.array(cand[:k], dtype=int)
            mask = np.ones(168, dtype=bool)
            mask[rem] = False
            Y = Xb[mask]
            Xinit = _refill_voids(Y, k, rng, n_probes=20000)
            Xinit /= np.linalg.norm(Xinit, axis=1, keepdims=True)
            # small jitter to break symmetry
            Xinit = Xinit + rng.randn(*Xinit.shape) * 1e-4
            Xinit /= np.linalg.norm(Xinit, axis=1, keepdims=True)
        except Exception:
            continue
        # re-jam: balanced global+per-point anneal beta 30->300
        try:
            # adapt steps to remaining time
            remain = END - time.time()
            if remain < 15.0:
                break
            steps = 700
            if remain < 40.0:
                steps = 400
            Xa = _anneal_balanced(Xinit, steps=steps, beta0=30.0, beta1=300.0,
                                  lr0=1e-2, lr1=1e-4, lam=0.15)
        except Exception:
            continue
        # LP polish with strict deadline
        try:
            remain = END - time.time()
            lp_sec = min(5.0, max(1.5, remain - 8.0))
            deadline = time.time() + lp_sec
            Xp = _lp_polish(Xa, deadline)
            Xp /= np.linalg.norm(Xp, axis=1, keepdims=True)
            Mp = _max_cos(Xp)
        except Exception:
            continue
        try:
            _, _, frac, mn, std, ent, _ = _metrics(Xp)
        except Exception:
            frac, mn, std, ent = 0.0, 0.0, 1e9, 0.0
        # safe return update
        if Mp + 1e-12 < best_val:
            best_val = float(Mp)
            best = Xp.copy()
            # also move base to new optimum basin
            base = Xp.copy(); base_val = float(Mp)
            base_frac, base_mn, base_std, base_ent = frac, mn, std, ent
        elif Mp < best_val + 2e-4 and is_better_balance(frac, mn, std, ent):
            # accept slightly worse but balanced basin for further surgery
            base = Xp.copy(); base_val = float(Mp)
            base_frac, base_mn, base_std, base_ent = frac, mn, std, ent
        # also if no improvement but time is plenty, keep exploring from base
        # (base unchanged otherwise)
    # final safety: ensure best is finite and polished a bit if time remains
    try:
        if time.time() < END - 3.0 and best is not None:
            deadline = min(time.time() + 2.5, END - 0.5)
            best = _lp_polish(best, deadline)
            best /= np.linalg.norm(best, axis=1, keepdims=True)
            best_val = _max_cos(best)
    except Exception:
        pass
    best /= np.linalg.norm(best, axis=1, keepdims=True)
    return best
