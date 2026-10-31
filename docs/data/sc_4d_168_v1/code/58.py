import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _norm(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return X / n

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def _deg_var(X, tol=0.004):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    m = float(np.max(G))
    deg = np.sum(G > m - tol, axis=1).astype(float)
    return float(np.var(deg)), float(np.mean(deg)), deg, m

def _lp_polish_deadline(x, deadline):
    x = _norm(np.asarray(x, dtype=float).copy())
    n, d = x.shape
    try:
        best = _max_cos(x)
    except Exception:
        return x
    radius = 1e-3
    iu = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        G = x @ x.T
        c = G[iu]
        m = float(c.max())
        thresh = m - max(20.0 * radius, 1e-9)
        sel = c > thresh
        cnt = int(np.count_nonzero(sel))
        if cnt == 0:
            radius *= 0.5
            continue
        if cnt > 4500:
            idx = np.argsort(-c)[:4500]
            keep = np.zeros_like(sel)
            keep[idx] = True
            sel = sel & keep.astype(bool)
        I, J = iu[0][sel], iu[1][sel]
        csel = c[sel]
        p = len(I)
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv); obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            break
        if res.status != 0:
            radius *= 0.5
            continue
        dx = res.x[:-1].reshape(n, d)
        y = _norm(x + dx)
        v = _max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.02)
        else:
            radius *= 0.5
    return x

def _anneal(X, steps, b0, b1, lr0, lr1, rng, noise0=0.0, mu=0.88):
    X = _norm(X.copy())
    vel = np.zeros_like(X)
    lb0 = np.log(max(b0, 1e-9)); lb1 = np.log(max(b1, 1e-9))
    for s in range(steps):
        f = s / max(1, steps - 1)
        beta = float(np.exp(lb0 + (lb1 - lb0) * f))
        lr = float(lr0 * (1 - f) + lr1 * f)
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        cmax = float(np.max(G))
        E = np.exp(np.clip(beta * (G - cmax), -25.0, 0.0))
        np.fill_diagonal(E, 0.0)
        rs = np.sum(E, axis=1, keepdims=True) + 1e-12
        grad = (E @ X) / rs
        dot = np.sum(grad * X, axis=1, keepdims=True)
        gt = grad - dot * X
        sc = float(np.mean(np.linalg.norm(gt, axis=1))) + 1e-12
        gt = gt / sc
        if noise0 > 0:
            amp = noise0 * (1.0 - f)
            if amp > 0:
                Nz = rng.normal(size=X.shape)
                Nz -= np.sum(Nz * X, axis=1, keepdims=True) * X
                gt = gt + amp * Nz
        vel = mu * vel + (1.0 - mu) * gt
        X = X - lr * vel
        X = _norm(X)
    return X

def _refine_single(x0, Xref, steps=90, beta0=40.0, beta1=250.0, lr0=0.08, lr1=0.005):
    s = np.asarray(x0, dtype=float).copy()
    nrm = float(np.linalg.norm(s))
    if nrm < 1e-12 or not np.isfinite(nrm):
        s = np.random.normal(size=4); nrm = float(np.linalg.norm(s))
    s = s / nrm
    lb0 = np.log(beta0); lb1 = np.log(beta1)
    for it in range(steps):
        f = it / max(1, steps - 1)
        beta = float(np.exp(lb0 + (lb1 - lb0) * f))
        lr = float(lr0 * (1 - f) + lr1 * f)
        dots = Xref @ s
        cm = float(np.max(dots))
        w = np.exp(np.clip(beta * (dots - cm), -25.0, 0.0))
        w /= (float(np.sum(w)) + 1e-30)
        g = w @ Xref
        gt = g - float(np.dot(g, s)) * s
        gn = float(np.linalg.norm(gt)) + 1e-12
        gt = gt / gn
        s = s - lr * gt
        ns = float(np.linalg.norm(s))
        if ns < 1e-12 or not np.isfinite(ns):
            break
        s = s / ns
    return s / (float(np.linalg.norm(s)) + 1e-30)

def _void_best(Xref, rng, n_sample=8000):
    C = rng.normal(size=(n_sample, 4))
    C = _norm(C)
    D = C @ Xref.T
    m = D.max(axis=1)
    return C[int(np.argmin(m))]

def _reinsert(X, idx_remove, rng, n_sample=7000):
    n, d = X.shape
    idx_remove = list(map(int, np.asarray(idx_remove).tolist()))
    keep_mask = np.ones(n, dtype=bool)
    keep_mask[idx_remove] = False
    cur = X[keep_mask].copy()
    new_pts = []
    for _ in idx_remove:
        seed = _void_best(cur, rng, n_sample=n_sample)
        pt = _refine_single(seed, cur, steps=90)
        new_pts.append(pt)
        cur = np.vstack([cur, pt.reshape(1, d)])
    Y = X.copy()
    Y[idx_remove] = np.array(new_pts)
    return _norm(Y)

def _hopf_optimize(F, k, steps, beta0, beta1, lr, rng):
    N = F * k
    ru = rng.normal(scale=0.6, size=F)
    phi = rng.uniform(0, 2 * np.pi, size=F)
    off = rng.uniform(0, 4 * np.pi, size=F)
    delta = np.zeros((F, k))
    t_idx = np.tile(np.arange(k), F)
    fib_rep = np.repeat(np.arange(F), k)
    m_ru = np.zeros(F); v_ru = np.zeros(F)
    m_ph = np.zeros(F); v_ph = np.zeros(F)
    m_of = np.zeros(F); v_of = np.zeros(F)
    m_dl = np.zeros((F, k)); v_dl = np.zeros((F, k))
    b1 = 0.9; b2 = 0.999; eps = 1e-8
    lb0 = np.log(beta0); lb1 = np.log(beta1)
    for st in range(1, steps + 1):
        f = st / max(1, steps)
        beta = float(np.exp(lb0 + (lb1 - lb0) * f))
        u = np.clip(np.tanh(ru), -0.985, 0.985)
        c1 = np.sqrt((1 + u) / 2.0)
        s1 = np.sqrt((1 - u) / 2.0)
        c1 = np.maximum(c1, 1e-3); s1 = np.maximum(s1, 1e-3)
        c_rep = c1[fib_rep]; s_rep = s1[fib_rep]
        phi_rep = phi[fib_rep]; off_rep = off[fib_rep]
        dl_rep = delta[fib_rep, t_idx]
        psi = off_rep + (4 * np.pi * t_idx / max(k, 1)) + dl_rep
        a = (psi + phi_rep) * 0.5
        b = (psi - phi_rep) * 0.5
        ca = np.cos(a); sa = np.sin(a); cb = np.cos(b); sb = np.sin(b)
        X = np.stack([c_rep * ca, c_rep * sa, s_rep * cb, s_rep * sb], axis=1)
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        gmax = float(np.max(G))
        W = np.exp(np.clip(beta * (G - gmax), -25.0, 0.0))
        np.fill_diagonal(W, 0.0)
        S = float(np.sum(W)) + 1e-12
        Grad = (W @ X) / S
        dc = 1.0 / (4.0 * c1 + 1e-12)
        ds = -1.0 / (4.0 * s1 + 1e-12)
        dc_rep = dc[fib_rep]; ds_rep = ds[fib_rep]
        dx_du = np.stack([dc_rep * ca, dc_rep * sa, ds_rep * cb, ds_rep * sb], axis=1)
        dx_dp = np.stack([-0.5 * c_rep * sa, 0.5 * c_rep * ca, 0.5 * s_rep * sb, -0.5 * s_rep * cb], axis=1)
        dx_ds = np.stack([-0.5 * c_rep * sa, 0.5 * c_rep * ca, -0.5 * s_rep * sb, 0.5 * s_rep * cb], axis=1)
        g_du_pt = np.sum(Grad * dx_du, axis=1)
        g_dp_pt = np.sum(Grad * dx_dp, axis=1)
        g_ds_pt = np.sum(Grad * dx_ds, axis=1)
        g_u = np.zeros(F); g_p = np.zeros(F); g_o = np.zeros(F)
        for j in range(F):
            msk = (fib_rep == j)
            g_u[j] = float(np.sum(g_du_pt[msk]))
            g_p[j] = float(np.sum(g_dp_pt[msk]))
            g_o[j] = float(np.sum(g_ds_pt[msk]))
        g_ru = g_u * (1.0 - u * u)
        g_dl = np.zeros((F, k))
        for j in range(F):
            for t in range(k):
                idx = j * k + t
                g_dl[j, t] = float(g_ds_pt[idx])
        # Adam updates
        m_ru = b1 * m_ru + (1 - b1) * g_ru; v_ru = b2 * v_ru + (1 - b2) * (g_ru * g_ru)
        m_ph = b1 * m_ph + (1 - b1) * g_p; v_ph = b2 * v_ph + (1 - b2) * (g_p * g_p)
        m_of = b1 * m_of + (1 - b1) * g_o; v_of = b2 * v_of + (1 - b2) * (g_o * g_o)
        m_dl = b1 * m_dl + (1 - b1) * g_dl; v_dl = b2 * v_dl + (1 - b2) * (g_dl * g_dl)
        c1b = 1.0 - b1 ** st; c2b = 1.0 - b2 ** st
        ru -= lr * (m_ru / c1b) / (np.sqrt(v_ru / c2b) + eps)
        phi -= lr * (m_ph / c1b) / (np.sqrt(v_ph / c2b) + eps)
        off -= lr * (m_of / c1b) / (np.sqrt(v_of / c2b) + eps)
        delta -= lr * (m_dl / c1b) / (np.sqrt(v_dl / c2b) + eps)
        delta -= np.mean(delta, axis=1, keepdims=True)
        ru = np.clip(ru, -3.0, 3.0)
    # final map
    u = np.clip(np.tanh(ru), -0.985, 0.985)
    c1 = np.maximum(np.sqrt((1 + u) / 2.0), 1e-3); s1 = np.maximum(np.sqrt((1 - u) / 2.0), 1e-3)
    c_rep = c1[fib_rep]; s_rep = s1[fib_rep]
    phi_rep = phi[fib_rep]; off_rep = off[fib_rep]
    dl_rep = delta[fib_rep, t_idx]
    psi = off_rep + (4 * np.pi * t_idx / max(k, 1)) + dl_rep
    a = (psi + phi_rep) * 0.5; b = (psi - phi_rep) * 0.5
    X = np.stack([c_rep * np.cos(a), c_rep * np.sin(a), s_rep * np.cos(b), s_rep * np.sin(b)], axis=1)
    return _norm(X)

def solve():
    t0 = time.time()
    TIME_LIMIT = 172.0
    HOPF_END = 62.0
    SEARCH_END = 142.0
    n, d = 168, 4
    rng = np.random.default_rng(7)
    cands = []
    try:
        xb = np.load("best.npy").astype(float)
        if xb.shape == (n, d):
            xb = _norm(xb)
            if np.all(np.isfinite(xb)):
                cands.append(xb)
    except Exception:
        pass
    try:
        xr = np.load("record.npy").astype(float)
        if xr.shape == (n, d):
            xr = _norm(xr)
            if np.all(np.isfinite(xr)):
                cands.append(xr)
    except Exception:
        pass
    if len(cands) == 0:
        bestX = _norm(rng.normal(size=(n, d)))
        best = _max_cos(bestX)
    else:
        vals = [_max_cos(c) for c in cands]
        bestX = cands[int(np.argmin(vals))].copy()
        best = float(min(vals))
    try:
        inc_var, _, _, _ = _deg_var(bestX, tol=0.004)
    except Exception:
        inc_var = 1e9
    # ---- Hopf-fibration ansatz from scratch ----
    partitions = [(21, 8), (28, 6), (24, 7), (42, 4), (56, 3)]
    hopf_seed = 0
    for (F, k) in partitions:
        if time.time() - t0 > HOPF_END:
            break
        for rep in range(2):
            if time.time() - t0 > HOPF_END:
                break
            hopf_seed += 1
            hrng = np.random.default_rng(1000 + hopf_seed)
            try:
                Xs = _hopf_optimize(F, k, steps=1400, beta0=25.0, beta1=420.0, lr=0.03, rng=hrng)
            except Exception:
                continue
            try:
                v0 = _max_cos(Xs)
            except Exception:
                continue
            # release with tiny symmetry-breaking noise then anneal
            for noise_amp, asteps in [(0.004, 700), (0.010, 700)]:
                if time.time() - t0 > HOPF_END + 14.0:
                    break
                try:
                    Xb = _norm(Xs + noise_amp * hrng.normal(size=(n, d)))
                    Xo = _anneal(Xb, steps=asteps, b0=30.0, b1=480.0, lr0=0.012, lr1=0.0006, rng=hrng, noise0=0.0)
                    co = _max_cos(Xo)
                    var_o, _, _, _ = _deg_var(Xo, tol=0.004)
                    # accept only if more regular than incumbent (per brief)
                    if co < best and var_o < inc_var:
                        dl = min(1.5, max(0.4, HOPF_END + 14.0 - (time.time() - t0)))
                        Xp = _lp_polish_deadline(Xo, time.time() + dl)
                        vp = _max_cos(Xp)
                        var_p, _, _, _ = _deg_var(Xp, tol=0.004)
                        if vp < best and var_p <= inc_var + 1e-9:
                            best = vp; bestX = Xp.copy()
                            inc_var = min(inc_var, var_p)
                        elif co < best:
                            best = co; bestX = Xo.copy()
                            inc_var = min(inc_var, var_o)
                    elif co < best - 0.02:
                        # far from record but keep annealing longer to try topology change
                        Xb2 = _anneal(Xo, steps=900, b0=18.0, b1=380.0, lr0=0.014, lr1=0.0006, rng=hrng, noise0=0.001)
                        c2 = _max_cos(Xb2)
                        if c2 < best and _deg_var(Xb2, tol=0.004)[0] < inc_var:
                            best = c2; bestX = Xb2.copy()
                except Exception:
                    continue
    # ---- balanced surgery fallback to squeeze record basin ----
    it = 0
    while time.time() - t0 < SEARCH_END:
        it += 1
        if time.time() - t0 > TIME_LIMIT - 30.0:
            break
        mode = it % 8
        try:
            if mode == 0:
                Xtry = _norm(bestX + 0.008 * rng.normal(size=(n, d)))
                kw = dict(steps=700, b0=30.0, b1=450.0, lr0=0.010, lr1=0.0006, noise0=0.0)
            elif mode == 1:
                Xtry = _norm(bestX + 0.030 * rng.normal(size=(n, d)))
                kw = dict(steps=1000, b0=22.0, b1=420.0, lr0=0.012, lr1=0.0006, noise0=0.0)
            elif mode == 2:
                _, _, deg, _ = _deg_var(bestX, tol=0.004)
                victims = np.argsort(-deg)[:4].tolist()
                Xtry = _reinsert(bestX, victims, rng, n_sample=6000)
                Xtry = _norm(Xtry + 0.004 * rng.normal(size=(n, d)))
                kw = dict(steps=1200, b0=20.0, b1=450.0, lr0=0.013, lr1=0.0006, noise0=0.001)
            elif mode == 3:
                G = bestX @ bestX.T; np.fill_diagonal(G, -2.0)
                ij = np.unravel_index(int(np.argmax(G)), G.shape)
                neigh = set(np.argsort(-G[ij[0]])[:5].tolist()) | set(np.argsort(-G[ij[1]])[:5].tolist())
                sel = list(neigh)[:7]
                Xtry = _reinsert(bestX, sel, rng, n_sample=6000)
                kw = dict(steps=1300, b0=20.0, b1=480.0, lr0=0.012, lr1=0.0006, noise0=0.001)
            elif mode == 4:
                _, _, deg, _ = _deg_var(bestX, tol=1e-4)
                ratt = np.argsort(deg)[:2].tolist()
                over = np.argsort(-deg)[:5].tolist()
                sel = list(dict.fromkeys(ratt + over))[:6]
                Xtry = _reinsert(bestX, sel, rng, n_sample=7000)
                Xtry = _norm(Xtry + 0.006 * rng.normal(size=(n, d)))
                kw = dict(steps=1300, b0=18.0, b1=420.0, lr0=0.013, lr1=0.0006, noise0=0.002)
            elif mode == 5:
                Xtry = _norm(bestX + 0.060 * rng.normal(size=(n, d)))
                kw = dict(steps=1100, b0=16.0, b1=380.0, lr0=0.013, lr1=0.0007, noise0=0.002)
            elif mode == 6:
                _, _, deg, _ = _deg_var(bestX, tol=0.004)
                hi = np.argsort(-deg)[:20]
                Xtry = bestX.copy()
                Xtry[hi] = _norm(Xtry[hi] + 0.10 * rng.normal(size=(len(hi), d)))
                kw = dict(steps=1100, b0=25.0, b1=500.0, lr0=0.010, lr1=0.0005, noise0=0.0)
            else:
                Xtry = _norm(bestX + 0.040 * rng.normal(size=(n, d)))
                k2 = 3
                idx = rng.choice(n, size=k2, replace=False).tolist()
                Xtry = _reinsert(Xtry, idx, rng, n_sample=5000)
                kw = dict(steps=1100, b0=24.0, b1=440.0, lr0=0.011, lr1=0.0006, noise0=0.0)
        except Exception:
            continue
        try:
            Xopt = _anneal(Xtry, kw["steps"], kw["b0"], kw["b1"], kw["lr0"], kw["lr1"], rng, noise0=kw["noise0"])
            co = _max_cos(Xopt)
        except Exception:
            continue
        if co < best:
            best = co
            bestX = Xopt.copy()
            try:
                rem = TIME_LIMIT - (time.time() - t0) - 6.0
                if rem > 2.0:
                    Xp = _lp_polish_deadline(bestX, time.time() + min(2.5, rem))
                    vp = _max_cos(Xp)
                    if vp < best:
                        best = vp; bestX = Xp.copy()
            except Exception:
                pass
        elif co < best + 0.005 and (time.time() - t0) < SEARCH_END - 3.0:
            try:
                Xp = _lp_polish_deadline(Xopt, time.time() + 0.9)
                vp = _max_cos(Xp)
                if vp < best:
                    best = vp; bestX = Xp.copy()
            except Exception:
                pass
    try:
        rem = TIME_LIMIT - (time.time() - t0) - 1.0
        if rem > 2.0:
            Xf = _lp_polish_deadline(bestX, time.time() + min(24.0, rem))
            vf = _max_cos(Xf)
            if vf <= best and np.all(np.isfinite(Xf)):
                bestX = Xf; best = vf
    except Exception:
        pass
    bestX = _norm(bestX)
    return np.asarray(bestX, dtype=float)
