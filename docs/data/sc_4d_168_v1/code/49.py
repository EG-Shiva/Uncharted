import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def _norm(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return x / n

def _degrees_var(x, tol=1e-7):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    m = float(np.max(G))
    adj = G > (m - tol)
    degs = np.sum(adj, axis=1).astype(float)
    return float(np.var(degs)), float(np.mean(degs)), degs

def _lp_polish(x, deadline):
    x = _norm(x)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    iu = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        C = (x @ x.T)[iu]
        m = float(np.max(C))
        sel = C > m - max(20.0 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        if p == 0:
            break
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
            break
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-C[sel], A_eq=A_eq,
                          b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                          method="highs")
        except Exception:
            break
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y = _norm(y)
        v = _max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def _hopf_X(theta, phi, psi):
    # theta(F,), phi(F,), psi(F,K) -> (F*K,4)
    c = np.cos(theta * 0.5)
    s = np.sin(theta * 0.5)
    alpha = (psi + phi[:, None]) * 0.5
    gamma = (psi - phi[:, None]) * 0.5
    ca = np.cos(alpha); sa = np.sin(alpha)
    cg = np.cos(gamma); sg = np.sin(gamma)
    Xf0 = c[:, None] * ca
    Xf1 = c[:, None] * sa
    Xf2 = s[:, None] * cg
    Xf3 = s[:, None] * sg
    F, K = psi.shape
    X = np.empty((F * K, 4))
    X[:, 0] = Xf0.ravel()
    X[:, 1] = Xf1.ravel()
    X[:, 2] = Xf2.ravel()
    X[:, 3] = Xf3.ravel()
    return X

def solve():
    t0 = time.time()
    HARD_END = t0 + 172.0
    HOPF_END = t0 + 80.0
    ANNEAL_END = t0 + 148.0
    rng = np.random.RandomState(12345)
    n, d = 168, 4
    try:
        xb = np.load("best.npy").astype(float)
    except Exception:
        xb = rng.randn(n, d)
    try:
        xr = np.load("record.npy").astype(float)
    except Exception:
        xr = xb.copy()
    def _chk(v):
        v = np.asarray(v, dtype=float)
        if v.shape != (n, d):
            return None
        v = _norm(v)
        if not np.all(np.isfinite(v)):
            return None
        return v
    xb = _chk(xb)
    xr = _chk(xr)
    if xb is None:
        xb = _norm(rng.randn(n, d))
    if xr is None:
        xr = xb.copy()
    vb = _max_cos(xb); vr = _max_cos(xr)
    if vr < vb:
        x_best = xr.copy(); best_val = vr
    else:
        x_best = xb.copy(); best_val = vb
    var_inc, mean_inc, _ = _degrees_var(x_best, tol=1e-7)
    # fallback variance threshold: require strictly lower
    # Hopf search
    partitions = [(14, 12), (21, 8), (12, 14), (28, 6), (24, 7), (42, 4)]
    hopf_cands = []
    b1, b2, eps = 0.9, 0.999, 1e-8
    for (F, K) in partitions:
        if F * K != 168:
            continue
        nseeds = 3
        # fewer seeds for large F to save time
        if F >= 28:
            nseeds = 2
        for seed in range(nseeds):
            if time.time() > HOPF_END:
                break
            rs = np.random.RandomState(1000 + F * 100 + seed * 17)
            # base init: uniform on S2
            u = rs.rand(F) * 2.0 - 1.0
            # avoid exact poles
            u = np.clip(u, -0.999, 0.999)
            theta = np.arccos(u)
            phi = rs.rand(F) * 2.0 * np.pi
            # fiber angles: uniform spacing + offset + jitter
            off = rs.rand(F) * 2.0 * np.pi
            base_grid = (np.arange(K)[None, :] * (2.0 * np.pi / K))
            if seed == 1:
                # fully random fiber angles
                psi = rs.rand(F, K) * 2.0 * np.pi
            elif seed == 2:
                jitter = (rs.randn(F, K)) * 0.25
                psi = off[:, None] + base_grid + jitter
            else:
                jitter = (rs.randn(F, K)) * 0.08
                psi = off[:, None] + base_grid + jitter
            steps = 2600
            if F >= 24:
                steps = 2000
            beta0, beta1h = 30.0, 450.0
            lr0, lr1 = 0.035, 0.003
            m_th = np.zeros(F); v_th = np.zeros(F)
            m_ph = np.zeros(F); v_ph = np.zeros(F)
            m_ps = np.zeros((F, K)); v_ps = np.zeros((F, K))
            lb0, lb1 = np.log(beta0), np.log(beta1h)
            for s in range(steps):
                if time.time() > HOPF_END:
                    break
                frac = s / max(1, steps - 1)
                beta = float(np.exp(lb0 + (lb1 - lb0) * frac))
                lr = lr0 * (1.0 - frac) + lr1 * frac
                X = _hopf_X(theta, phi, psi)
                G = X @ X.T
                np.fill_diagonal(G, -2.0)
                cmax = float(np.max(G))
                E = np.exp(np.clip(beta * (G - cmax), -25.0, 0.0))
                np.fill_diagonal(E, 0.0)
                ssum = float(np.sum(E)) + 1e-30
                # grad wrt X of LSE
                GX = (E @ X) * (2.0 / ssum)  # (n,4)
                GFr = GX.reshape(F, K, 4)
                c = np.cos(theta * 0.5); si = np.sin(theta * 0.5)
                alpha = (psi + phi[:, None]) * 0.5
                gamma = (psi - phi[:, None]) * 0.5
                ca = np.cos(alpha); sa = np.sin(alpha)
                cg = np.cos(gamma); sg = np.sin(gamma)
                # partials
                dth0 = -0.5 * si[:, None] * ca
                dth1 = -0.5 * si[:, None] * sa
                dth2 = 0.5 * c[:, None] * cg
                dth3 = 0.5 * c[:, None] * sg
                dph0 = -0.5 * c[:, None] * sa
                dph1 = 0.5 * c[:, None] * ca
                dph2 = 0.5 * si[:, None] * sg
                dph3 = -0.5 * si[:, None] * cg
                dps0 = -0.5 * c[:, None] * sa
                dps1 = 0.5 * c[:, None] * ca
                dps2 = -0.5 * si[:, None] * sg
                dps3 = 0.5 * si[:, None] * cg
                g_th = (GFr[:, :, 0] * dth0 + GFr[:, :, 1] * dth1 +
                        GFr[:, :, 2] * dth2 + GFr[:, :, 3] * dth3).sum(axis=1)
                g_ph = (GFr[:, :, 0] * dph0 + GFr[:, :, 1] * dph1 +
                        GFr[:, :, 2] * dph2 + GFr[:, :, 3] * dph3).sum(axis=1)
                g_ps = (GFr[:, :, 0] * dps0 + GFr[:, :, 1] * dps1 +
                        GFr[:, :, 2] * dps2 + GFr[:, :, 3] * dps3)
                # Adam updates
                t = s + 1
                m_th = b1 * m_th + (1 - b1) * g_th
                v_th = b2 * v_th + (1 - b2) * (g_th * g_th)
                mhat = m_th / (1 - b1 ** t); vhat = v_th / (1 - b2 ** t)
                theta = theta - lr * mhat / (np.sqrt(vhat) + eps)
                m_ph = b1 * m_ph + (1 - b1) * g_ph
                v_ph = b2 * v_ph + (1 - b2) * (g_ph * g_ph)
                mhat = m_ph / (1 - b1 ** t); vhat = v_ph / (1 - b2 ** t)
                phi = phi - lr * mhat / (np.sqrt(vhat) + eps)
                m_ps = b1 * m_ps + (1 - b1) * g_ps
                v_ps = b2 * v_ps + (1 - b2) * (g_ps * g_ps)
                mhat = m_ps / (1 - b1 ** t); vhat = v_ps / (1 - b2 ** t)
                psi = psi - lr * mhat / (np.sqrt(vhat) + eps)
            Xf = _norm(_hopf_X(theta, phi, psi))
            vf = _max_cos(Xf)
            hopf_cands.append((vf, Xf.copy(), F, K))
            # keep list sorted, keep all (few)
        if time.time() > HOPF_END:
            break
    hopf_cands.sort(key=lambda e: e[0])
    # full-dim release + anneal
    def refine_full(x_init, steps, beta0=30.0, beta1=600.0, lr0=0.016, lr1=0.0008, mu=0.88):
        x = x_init.copy()
        vel = np.zeros_like(x)
        lb0 = np.log(beta0); lb1 = np.log(beta1)
        for s in range(steps):
            if time.time() > ANNEAL_END:
                break
            f = s / max(1, steps - 1)
            beta = float(np.exp(lb0 + (lb1 - lb0) * f))
            lr = lr0 * (1.0 - f) + lr1 * f
            G = x @ x.T
            np.fill_diagonal(G, -2.0)
            cmax = float(np.max(G))
            E = np.exp(np.clip(beta * (G - cmax), -25.0, 0.0))
            np.fill_diagonal(E, 0.0)
            rs_ = np.sum(E, axis=1, keepdims=True) + 1e-12
            grad = (E @ x) / rs_
            dot = np.sum(grad * x, axis=1, keepdims=True)
            gt = grad - dot * x
            sc = float(np.mean(np.linalg.norm(gt, axis=1))) + 1e-12
            gt = gt / sc
            vel = mu * vel + (1.0 - mu) * gt
            x = x - lr * vel
            x = _norm(x)
        return x
    # try top hopf candidates
    sigmas = [0.002, 0.006, 0.010, 0.0035]
    tried = 0
    for idx, (vf, Xh, F, K) in enumerate(hopf_cands[:6]):
        for si, sig in enumerate(sigmas):
            if time.time() > ANNEAL_END:
                break
            if tried > 10:
                break
            # only try promising: within 0.01 of best or best hopf
            if vf > best_val + 0.012 and idx > 1:
                continue
            y = Xh + sig * rng.randn(n, d)
            y = _norm(y)
            y = refine_full(y, steps=700)
            vref = _max_cos(y)
            var_y, _, _ = _degrees_var(y, tol=1e-7)
            # variance filter: require lower than incumbent
            if var_y >= var_inc:
                # still give one chance with smaller sigma? skip LP
                # to honor direction, do not accept this run
                # but keep refining? discard
                tried += 1
                continue
            # LP polish if close
            if vref < best_val + 0.002:
                y = _lp_polish(y, min(time.time() + 6.0, HARD_END - 4.0))
                vref = _max_cos(y)
                var_y, _, _ = _degrees_var(y, tol=1e-7)
                if var_y >= var_inc:
                    tried += 1
                    continue
            tried += 1
            if vref < best_val:
                best_val = float(vref)
                x_best = y.copy()
                # update var threshold? keep original incumbent var as reference
                # do not update var_inc so filter stays strict
            if time.time() > ANNEAL_END:
                break
        if time.time() > ANNEAL_END:
            break
    # final safety: if hopf failed to improve, ensure we return at least incumbent
    # opportunistic tiny polish of incumbent if time remains (does not change basin)
    if time.time() < HARD_END - 8.0:
        # one quick polish of best to squeeze
        x_best = _lp_polish(x_best, min(time.time() + 5.0, HARD_END - 2.0))
    x_best = _norm(x_best)
    return x_best
