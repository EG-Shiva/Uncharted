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

def _deg_seq(X, thr=1e-4):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    m = float(G.max())
    B = G > (m - thr)
    deg = B.sum(axis=1).astype(int)
    return tuple(sorted(deg.tolist(), reverse=True)), m, deg

def _lp_polish(x, seconds, t_end=None):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while True:
        if time.time() - start > seconds:
            break
        if t_end is not None and time.time() > t_end:
            break
        if radius < 1e-14:
            break
        Gfull = x @ x.T
        cv = Gfull[iu]
        m = float(cv.max())
        sel = cv > m - max(20 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        if p == 0:
            break
        if p > 2500:
            idx = np.argsort(cv[sel])[-2500:]
            I, J = I[idx], J[idx]
            cc = cv[sel][idx]
        else:
            cc = cv[sel]
        p = len(I)
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        except Exception:
            radius *= 0.5
            continue
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-cc, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
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
            x, best, radius = y, v, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x, best

def _refine_hole(Y, p, iters=180, beta=110.0, lr0=0.12, lr1=0.004):
    p = p / np.linalg.norm(p)
    for it in range(iters):
        frac = it / max(iters - 1, 1)
        lr = lr0 * (lr1 / lr0) ** frac
        g = Y @ p
        gm = float(g.max())
        w = np.exp(beta * (g - gm))
        s = float(w.sum()) + 1e-18
        A = (w @ Y) / s
        At = A - float(A @ p) * p
        p = p - lr * At
        nrm = np.linalg.norm(p)
        if nrm == 0 or not np.isfinite(nrm):
            break
        p /= nrm
    return p

def _select_caps(F, k, rng, n_probe=20000):
    d = F.shape[1]
    P = rng.normal(size=(n_probe, d))
    P /= np.linalg.norm(P, axis=1, keepdims=True)
    Df = P @ F.T
    mdF = Df.max(axis=1)
    sel = []
    for t in range(k):
        if t == 0:
            scores = mdF
        else:
            S = np.array(sel)
            Ds = P @ S.T
            mdS = Ds.max(axis=1)
            scores = np.maximum(mdF, mdS)
        bi = int(np.argmin(scores))
        sel.append(P[bi].copy())
    S = np.array(sel)
    # refine each against frozen + other selected
    out = []
    for i in range(k):
        others = [F] if len(out) == 0 else [F, np.array(out)]
        # also include not-yet-refined peers to keep mutual distance
        Yext_parts = [F]
        if len(out) > 0:
            Yext_parts.append(np.array(out))
        # include remaining raw peers
        rem_idx = [j for j in range(k) if j != i]
        # use raw S for remaining
        Yext = np.vstack([F] + ([np.array(out)] if len(out) > 0 else []) + ([S[rem_idx]] if len(rem_idx) > 0 else []))
        pr = _refine_hole(Yext, S[i], iters=160, beta=100.0 + rng.random() * 60.0)
        # ensure mutual distance: if too close to already refined, push
        if len(out) > 0:
            Q = np.array(out)
            for _ in range(40):
                md = float(Q @ pr.max()) if False else float((Q @ pr).max())
                if md < 0.85:
                    break
                # random kick
                pr = pr + rng.normal(size=d) * 0.05
                pr /= np.linalg.norm(pr)
                pr = _refine_hole(Yext, pr, iters=60, beta=120.0)
        out.append(pr)
    return np.array(out)

def _block_opt(Z0, F, beta=280.0, steps=900, lr0=0.08, lr1=0.004, mu=0.85, rng=None):
    Z = Z0.copy()
    Z /= np.linalg.norm(Z, axis=1, keepdims=True)
    k = Z.shape[0]
    V = np.zeros_like(Z)
    ll0, ll1 = np.log(lr0), np.log(lr1)
    eye = np.eye(k, dtype=bool)
    for s in range(steps):
        frac = s / max(steps - 1, 1)
        lr = float(np.exp(ll0 + (ll1 - ll0) * frac))
        Df = Z @ F.T
        Ds = Z @ Z.T
        # max over relevant
        m = float(Df.max())
        if k > 1:
            Ds_off = Ds.copy()
            np.fill_diagonal(Ds_off, -10.0)
            m = max(m, float(Ds_off.max()))
        Wf = np.exp(beta * (Df - m))
        Ws = np.exp(beta * (Ds - m))
        np.fill_diagonal(Ws, 0.0)
        rs = Wf.sum(axis=1, keepdims=True) + Ws.sum(axis=1, keepdims=True) + 1e-18
        Grad = (Wf @ F + Ws @ Z) / rs
        dot = (Grad * Z).sum(axis=1, keepdims=True)
        Gt = Grad - dot * Z
        # tame
        nn = np.sqrt((Gt * Gt).sum(axis=1, keepdims=True)) + 1e-12
        scale = np.minimum(1.0, 2.0 / (nn + 1e-12))
        Gt = Gt * scale
        V = mu * V + (1.0 - mu) * Gt if False else mu * V + Gt * 0.15
        # use velocity form: V = mu V + Gt ; step lr*V with small factor
        Z = Z - lr * V
        Z /= np.linalg.norm(Z, axis=1, keepdims=True)
    return Z

def _balanced_quench(X, steps, b0, b1, lr0, lr1, mu=0.85, lam=0.10):
    n, d = X.shape
    X = X.copy()
    V = np.zeros_like(X)
    lb0, lb1 = np.log(b0), np.log(b1)
    ll0, ll1 = np.log(lr0), np.log(lr1)
    for s in range(steps):
        frac = s / max(steps - 1, 1)
        beta = float(np.exp(lb0 + (lb1 - lb0) * frac))
        lr = float(np.exp(ll0 + (ll1 - ll0) * frac))
        G = X @ X.T
        np.fill_diagonal(G, -10.0)
        m = float(G.max())
        E = np.exp(beta * (G - m))
        np.fill_diagonal(E, 0.0)
        rs = E.sum(axis=1, keepdims=True) + 1e-30
        Grad_g = (E @ X) / rs
        Gb = X @ X.T
        np.fill_diagonal(Gb, 0.0)
        np.maximum(Gb, 0.0, out=Gb)
        Grad_b = (Gb @ X) * (2.0 / max(n, 1))
        Grad = Grad_g + lam * Grad_b
        dot = (Grad * X).sum(axis=1, keepdims=True)
        Gt = Grad - dot * X
        nn = np.sqrt((Gt * Gt).sum(axis=1, keepdims=True)) + 1e-12
        scale = np.minimum(1.0, 3.0 / (nn + 1e-12))
        Gt = Gt * scale
        V = mu * V + (1.0 - mu) * Gt
        # effective step
        X = X - lr * V * 8.0
        X /= np.linalg.norm(X, axis=1, keepdims=True)
    return X

def _hard_polish(X, steps=260, lr0=0.004, lr1=0.00015, tol0=3e-4, tol1=2e-6):
    X = X.copy()
    for s in range(steps):
        frac = s / max(steps - 1, 1)
        tol = tol0 + (tol1 - tol0) * frac
        lr = lr0 * (lr1 / lr0) ** frac
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        M = (G > (m - tol)).astype(np.float64)
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

def solve():
    t0 = time.time()
    BUDGET = 172.0
    rng = np.random.default_rng(7)
    n, d = 168, 8
    try:
        best0 = np.load("best.npy").astype(float)
    except Exception:
        best0 = None
    try:
        rec0 = np.load("record.npy").astype(float)
    except Exception:
        rec0 = None
    cands = []
    if best0 is not None and best0.shape == (n, d):
        cands.append(best0 / np.linalg.norm(best0, axis=1, keepdims=True))
    if rec0 is not None and rec0.shape == (n, d):
        cands.append(rec0 / np.linalg.norm(rec0, axis=1, keepdims=True))
    if not cands:
        c = rng.normal(size=(n, d))
        c /= np.linalg.norm(c, axis=1, keepdims=True)
        cands.append(c)
    best_x = min(cands, key=_max_cos)
    best_x = best_x / np.linalg.norm(best_x, axis=1, keepdims=True)
    best_val = _max_cos(best_x)
    # quick baseline LP
    if time.time() - t0 < BUDGET - 60:
        try:
            bx, bv = _lp_polish(best_x, 1.5, t_end=t0 + BUDGET - 15)
            if bv < best_val:
                best_x, best_val = bx, bv
        except Exception:
            pass
    cur_x = best_x.copy()
    cur_val = best_val
    try:
        tup0, _, _ = _deg_seq(best_x, thr=1e-4)
    except Exception:
        tup0 = ()
    archived = set()
    archived.add(tup0)
    ks = [10, 16, 24]
    it = 0
    while time.time() - t0 < BUDGET - 12:
        it += 1
        remain = (t0 + BUDGET - 12) - time.time()
        if remain < 4:
            break
        k = ks[it % 3] if it % 4 != 0 else int(rng.choice(ks))
        # choose center: hub or random direction
        try:
            _, _, degs = _deg_seq(cur_x, thr=1e-4)
            hub = int(np.argmax(degs + rng.random(n) * 0.1))
        except Exception:
            hub = int(rng.integers(0, n))
        if it % 2 == 0:
            direction = cur_x[hub].copy()
        else:
            direction = rng.normal(size=d)
            direction /= np.linalg.norm(direction)
            # occasionally use hub neighborhood random tilt
            if rng.random() < 0.3:
                direction = cur_x[hub] + rng.normal(size=d) * 0.3
                direction /= np.linalg.norm(direction)
        dots = cur_x @ direction
        kill = np.argsort(-dots)[:k]
        mask = np.ones(n, dtype=bool)
        mask[kill] = False
        F = cur_x[mask]
        # reinsert candidates
        try:
            S0 = _select_caps(F, k, rng, n_probe=20000)
        except Exception:
            continue
        # joint block optimisation with high beta
        beta = float(rng.choice([220.0, 280.0, 350.0]))
        bsteps = 700 if remain > 30 else 400
        try:
            Z = _block_opt(S0, F, beta=beta, steps=bsteps, lr0=0.09, lr1=0.005, mu=0.86, rng=rng)
        except Exception:
            continue
        Xnew = np.vstack([F, Z])
        # balanced quench unfrozen
        qsteps = 1400
        if remain < 30:
            qsteps = 800
        elif remain < 60:
            qsteps = 1100
        b0 = float(rng.uniform(25, 45))
        b1 = float(rng.uniform(200, 320))
        lr0 = float(np.exp(rng.uniform(np.log(0.03), np.log(0.08))))
        lr1 = float(np.exp(rng.uniform(np.log(0.0015), np.log(0.006))))
        lam = float(rng.uniform(0.05, 0.18))
        mu = float(rng.uniform(0.82, 0.90))
        try:
            Xq = _balanced_quench(Xnew, steps=qsteps, b0=b0, b1=b1, lr0=lr0, lr1=lr1, mu=mu, lam=lam)
            Xh = _hard_polish(Xq, steps=220, lr0=0.004, lr1=0.00015)
        except Exception:
            continue
        # short LP
        lp_sec = 0.9 if remain > 25 else 0.5
        try:
            Xp, Vp = _lp_polish(Xh, lp_sec, t_end=t0 + BUDGET - 10)
        except Exception:
            Vp = _max_cos(Xh)
            Xp = Xh
        # degree sequence tabu at 1e-4
        try:
            tup, _, _ = _deg_seq(Xp, thr=1e-4)
        except Exception:
            tup = None
        if Vp < best_val - 1e-12:
            # improvement always accepted even if degree repeats
            best_x, best_val = Xp.copy(), float(Vp)
            cur_x, cur_val = Xp.copy(), float(Vp)
            if tup is not None:
                archived.add(tup)
        else:
            if tup is not None and tup in archived:
                continue
            if tup is not None:
                archived.add(tup)
            if Vp < best_val + 2e-6:
                if Vp < cur_val or rng.random() < 0.40:
                    cur_x, cur_val = Xp.copy(), float(Vp)
            # occasional random restart from best to avoid drift
            if it % 12 == 0 and rng.random() < 0.5:
                cur_x, cur_val = best_x.copy(), best_val
        # keep archive bounded
        if len(archived) > 5000:
            # keep most recent: reset to current + best
            try:
                t1, _, _ = _deg_seq(best_x, thr=1e-4)
                t2, _, _ = _deg_seq(cur_x, thr=1e-4)
                archived = set([t1, t2])
            except Exception:
                archived = set(list(archived)[-1000:])
    best_x = best_x / np.linalg.norm(best_x, axis=1, keepdims=True)
    # final safety: shrink not needed; ensure finite
    best_x = np.nan_to_num(best_x)
    best_x /= np.linalg.norm(best_x, axis=1, keepdims=True)
    return best_x
