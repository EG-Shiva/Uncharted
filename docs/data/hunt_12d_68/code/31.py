import time
import numpy as np
from scipy.optimize import linprog, minimize
from scipy.sparse import coo_matrix

def _max_cos(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _per_max(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return G.max(axis=1)

def _lp_polish(x, seconds, cap_p=2500):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        if time.time() - start > seconds - 0.4:
            break
        C = (x @ x.T)[iu]
        m = float(C.max())
        sel = C > m - max(20 * radius, 1e-9)
        I = iu[0][sel]; J = iu[1][sel]
        csel = C[sel]
        p = len(I)
        if p == 0:
            radius *= 0.5
            continue
        if p > cap_p:
            o = np.argsort(-csel)[:cap_p]
            I, J, csel = I[o], J[o], csel[o]
            p = cap_p
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv); obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq, b_eq=np.zeros(n),
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
    return x

def _anneal(x, steps, b0, b1, lr0, lr1, mu, rng, noise=0.0):
    n, d = x.shape
    vel = np.zeros_like(x)
    lb = np.log(b1 / b0) if b0 > 0 and b1 != b0 else 0.0
    for s in range(steps):
        t = s / max(steps - 1, 1)
        beta = b0 * np.exp(lb * t) if lb != 0.0 else b0
        lr = lr0 + (lr1 - lr0) * t
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        W = np.exp(beta * (G - m))
        np.fill_diagonal(W, 0.0)
        Grad = W @ x
        proj = np.sum(Grad * x, axis=1, keepdims=True)
        Tg = Grad - proj * x
        rms = np.sqrt(np.mean(Tg * Tg)) + 1e-12
        step = Tg * (lr / rms)
        if mu > 0:
            vel = mu * vel + step
            step = vel
        x = x - step
        if noise > 0:
            ns = noise * (1.0 - t)
            nz = rng.standard_normal(size=x.shape)
            nz -= np.sum(nz * x, axis=1, keepdims=True) * x
            nn = np.linalg.norm(nz, axis=1, keepdims=True) + 1e-12
            x = x + ns * (nz / nn)
        x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x

def _subgrad(x, iters, lr0=0.02, margin0=0.01, margin1=2e-4):
    n, d = x.shape
    iu = np.triu_indices(n, 1)
    for s in range(iters):
        t = s / max(iters - 1, 1)
        margin = margin0 * (margin1 / margin0) ** t if margin0 > 0 else margin1
        lr = lr0 * (1.0 - 0.85 * t)
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        sel = G[iu] > m - margin
        if not np.any(sel):
            continue
        I = iu[0][sel]; J = iu[1][sel]
        Grad = np.zeros_like(x)
        np.add.at(Grad, I, x[J])
        np.add.at(Grad, J, x[I])
        proj = np.sum(Grad * x, axis=1, keepdims=True)
        Tg = Grad - proj * x
        rms = np.sqrt(np.mean(Tg * Tg)) + 1e-12
        x = x - (lr / rms) * Tg
        x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x

def _best_hole(rest, rng, ncand=6000):
    d = rest.shape[1]
    cand = rng.standard_normal((ncand, d))
    cand /= np.linalg.norm(cand, axis=1, keepdims=True)
    scores = np.max(cand @ rest.T, axis=1)
    return cand[int(np.argmin(scores))]

def _pre12(shift, D):
    d = len(D)
    K = d
    idx = np.zeros((K, d), dtype=int)
    coeff = np.zeros((K, d))
    idx[0] = np.arange(d); coeff[0] = 1.0
    for k in range(1, K):
        pi = idx[k-1]; pc = coeff[k-1]
        for i in range(d):
            j = (i - shift) % d
            idx[k, i] = pi[j]
            coeff[k, i] = D[i] * pc[j]
    return idx, coeff

def _pre8(shift8, D8):
    K = 8
    idx = np.zeros((K, 12), dtype=int)
    coeff = np.zeros((K, 12))
    idx[0] = np.arange(12); coeff[0] = 1.0
    for k in range(1, K):
        pi = idx[k-1]; pc = coeff[k-1]
        for i in range(8):
            j = (i - shift8) % 8
            idx[k, i] = pi[j]
            coeff[k, i] = D8[i] * pc[j]
        for i in range(8, 12):
            idx[k, i] = pi[i]; coeff[k, i] = pc[i]
    return idx, coeff

def _build(S, idx12, coeff12, idx8, coeff8):
    X = np.zeros((68, 12))
    p = 0
    for b in range(5):
        s = S[b]
        for k in range(12):
            X[p] = coeff12[k] * s[idx12[k]]
            p += 1
    s = S[5]
    for k in range(8):
        X[p] = coeff8[k] * s[idx8[k]]
        p += 1
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    return X

def _sym_fg(idx12, coeff12, idx8, coeff8, beta):
    iu = np.triu_indices(68, 1)
    def fg(yf):
        Y = yf.reshape(6, 12)
        nr = np.maximum(np.linalg.norm(Y, axis=1, keepdims=True), 1e-12)
        S = Y / nr
        X = _build(S, idx12, coeff12, idx8, coeff8)
        G = X @ X.T
        g = G[iu]
        m = float(g.max())
        e = np.exp(beta * (g - m))
        Sm = float(e.sum()) + 1e-300
        F = m + np.log(Sm) / beta
        w = e / Sm
        P = np.zeros((68, 68)); P[iu] = w; P = P + P.T
        GX = P @ X
        Gxt = GX - np.sum(GX * X, axis=1, keepdims=True) * X
        GS = np.zeros((6, 12))
        p = 0
        for b in range(5):
            acc = np.zeros(12)
            for k in range(12):
                np.add.at(acc, idx12[k], coeff12[k] * Gxt[p])
                p += 1
            GS[b] = acc
        acc = np.zeros(12)
        for k in range(8):
            np.add.at(acc, idx8[k], coeff8[k] * Gxt[p])
            p += 1
        GS[5] = acc
        GY = (GS - np.sum(GS * S, axis=1, keepdims=True) * S) / nr
        return F, GY.ravel()
    return fg

def _run_sym(y0, idx12, coeff12, idx8, coeff8, betas, iters):
    y = y0.copy()
    for beta, mi in zip(betas, iters):
        fg = _sym_fg(idx12, coeff12, idx8, coeff8, beta)
        try:
            r = minimize(fg, y, jac=True, method='L-BFGS-B',
                         options={'maxiter': mi, 'ftol': 1e-14, 'gtol': 1e-7, 'maxfun': mi*2})
            y = r.x.copy()
        except Exception:
            break
    return y

def _fg_all(beta):
    IU = np.triu_indices(68, 1)
    def fg(yf):
        Y = yf.reshape(68, 12)
        nr = np.maximum(np.linalg.norm(Y, axis=1, keepdims=True), 1e-12)
        X = Y / nr
        G = X @ X.T
        g = G[IU]
        m = float(g.max())
        e = np.exp(beta * (g - m))
        Sm = float(e.sum()) + 1e-300
        F = m + np.log(Sm) / beta
        w = e / Sm
        P = np.zeros((68, 68)); P[IU] = w; P = P + P.T
        GX = P @ X
        GY = (GX - np.sum(GX * X, axis=1, keepdims=True) * X) / nr
        return F, GY.ravel()
    return fg

def _lbfgs_all(x0, beta, maxiter):
    fg = _fg_all(beta)
    y0 = (x0 / np.linalg.norm(x0, axis=1, keepdims=True)).ravel().copy()
    try:
        r = minimize(fg, y0, jac=True, method='L-BFGS-B',
                     options={'maxiter': maxiter, 'ftol': 1e-14, 'gtol': 1e-7, 'maxfun': maxiter*2})
        Y = r.x.reshape(68, 12)
        Y /= np.linalg.norm(Y, axis=1, keepdims=True)
        return Y
    except Exception:
        return x0

def solve():
    t0 = time.time()
    TLIM = 172.0
    rng = np.random.default_rng(987654321)
    n, d = 68, 12
    cands = []
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.asarray(np.load(fn), dtype=np.float64)
            if a.shape == (n, d):
                a /= np.linalg.norm(a, axis=1, keepdims=True)
                if np.all(np.isfinite(a)):
                    cands.append(a)
        except Exception:
            pass
    if not cands:
        a = rng.standard_normal((n, d)); a /= np.linalg.norm(a, axis=1, keepdims=True)
        cands.append(a)
    best_val = 1e9
    x_best = cands[0].copy()
    for c in cands:
        v = _max_cos(c)
        if v < best_val:
            best_val = v; x_best = c.copy()
    # ---- symmetry scan (random signed cycles) ----
    sym_best = None; sym_val = 1e9
    sym_deadline = t0 + 42.0
    ncfg = 0
    while time.time() < sym_deadline and ncfg < 10:
        ncfg += 1
        shift12 = int(rng.choice([1, 5, 7, 11]))
        shift8 = int(rng.choice([1, 3, 5]))
        D = np.ones(12)
        k = int(rng.choice([0, 2, 4]))
        if k > 0:
            pos = rng.choice(12, size=k, replace=False)
            D[pos] = -1.0
        D8 = np.ones(8)
        k8 = int(rng.choice([0, 2]))
        if k8 > 0:
            pos = rng.choice(8, size=k8, replace=False)
            D8[pos] = -1.0
        idx12, coeff12 = _pre12(shift12, D)
        idx8, coeff8 = _pre8(shift8, D8)
        y0 = rng.standard_normal(6 * 12)
        yopt = _run_sym(y0, idx12, coeff12, idx8, coeff8, betas=[80.0, 300.0], iters=[60, 70])
        Y = yopt.reshape(6, 12); Y /= np.linalg.norm(Y, axis=1, keepdims=True)
        X = _build(Y, idx12, coeff12, idx8, coeff8)
        v = _max_cos(X)
        if v < sym_val:
            sym_val = v; sym_best = X.copy()
            # extra high-beta push if promising
            if v < best_val + 0.01:
                yopt2 = _run_sym(yopt, idx12, coeff12, idx8, coeff8, betas=[550.0], iters=[60])
                Y2 = yopt2.reshape(6, 12); Y2 /= np.linalg.norm(Y2, axis=1, keepdims=True)
                X2 = _build(Y2, idx12, coeff12, idx8, coeff8)
                v2 = _max_cos(X2)
                if v2 < sym_val:
                    sym_val = v2; sym_best = X2.copy()
        if time.time() > sym_deadline:
            break
    sym_rel = None; sym_rel_val = 1e9
    if sym_best is not None:
        Xr = sym_best.copy()
        for beta, mi in [(160.0, 90), (420.0, 90)]:
            if time.time() > t0 + 62.0:
                break
            Xr = _lbfgs_all(Xr, beta, mi)
        if time.time() < t0 + 70.0:
            Xr = _lp_polish(Xr, max(0.5, min(4.0, t0 + 70.0 - time.time())))
        sym_rel = Xr.copy(); sym_rel_val = _max_cos(Xr)
        if sym_rel_val < best_val:
            best_val = sym_rel_val; x_best = Xr.copy()
    # light squeeze of incumbent to lock baseline
    if time.time() < t0 + 74.0:
        x_best = _lp_polish(x_best, max(0.5, min(3.0, t0 + 74.0 - time.time())))
        best_val = _max_cos(x_best)
    # ---- basin hopping with contact-guided surgery ----
    pool2 = sym_rel if sym_rel is not None else None
    explore_end = t0 + TLIM - 26.0
    trial = 0
    while time.time() < explore_end:
        trial += 1
        base = x_best
        if pool2 is not None and (trial % 5 == 0):
            base = pool2
        x = base.copy()
        u = rng.random()
        pm = _per_max(x)
        order = np.argsort(-pm)
        if u < 0.35:
            sig = float(rng.choice([0.006, 0.015, 0.035, 0.08]))
            x = x + sig * rng.standard_normal(x.shape)
            x /= np.linalg.norm(x, axis=1, keepdims=True)
            b0 = float(rng.uniform(40, 130)); b1 = float(rng.uniform(320, 750))
            lr0 = float(rng.uniform(0.02, 0.06)); lr1 = float(rng.uniform(0.001, 0.006))
            mu = float(rng.choice([0.0, 0.5, 0.85]))
            x = _anneal(x, int(rng.integers(350, 700)), b0, b1, lr0, lr1, mu, rng, noise=float(rng.uniform(0, 0.004)))
            x = _subgrad(x, int(rng.integers(40, 140)), lr0=float(rng.uniform(0.008, 0.025)), margin0=float(rng.uniform(0.003, 0.02)), margin1=2e-4)
        elif u < 0.60:
            # rattler / worst-point hole reinsertion (targets 6 rattlers + overloaded)
            k = int(rng.integers(1, 4))
            victims = order[:k].copy()
            # sometimes also random low-contact point
            if rng.random() < 0.4:
                victims = np.append(victims, rng.integers(0, n))
                victims = np.unique(victims)[:3]
            for vi in victims:
                mask = np.ones(n, bool); mask[vi] = False
                hole = _best_hole(x[mask], rng, ncand=int(rng.choice([3000, 8000])))
                # small jitter to avoid exact ties
                hole = hole + 0.002 * rng.standard_normal(d)
                hole /= np.linalg.norm(hole)
                x[vi] = hole
            b0 = float(rng.uniform(60, 150)); b1 = float(rng.uniform(350, 700))
            x = _anneal(x, int(rng.integers(350, 650)), b0, b1, 0.03, 0.003, float(rng.choice([0.0, 0.7])), rng)
            x = _subgrad(x, 80, lr0=0.015, margin0=0.008, margin1=2e-4)
        elif u < 0.80:
            # coherent cap kick
            vv = rng.standard_normal(d); vv /= np.linalg.norm(vv)
            dots = x @ vv
            thr = float(np.quantile(dots, rng.uniform(0.75, 0.92)))
            sel = np.where(dots >= thr)[0]
            if len(sel) < 2:
                sel = order[: max(2, len(sel))]
            ang = float(rng.uniform(0.05, 0.35))
            # rotate cap toward random orthogonal dir
            ww = rng.standard_normal(d); ww -= float(ww @ vv) * vv
            ww /= np.linalg.norm(ww) + 1e-12
            for ii in sel:
                x[ii] = np.cos(ang) * x[ii] + np.sin(ang) * (ww + 0.2 * rng.standard_normal(d))
            # add small global noise
            x = x + 0.008 * rng.standard_normal(x.shape)
            x /= np.linalg.norm(x, axis=1, keepdims=True)
            x = _anneal(x, int(rng.integers(400, 700)), 80.0, 550.0, 0.04, 0.003, 0.6, rng)
            x = _subgrad(x, 60, lr0=0.015, margin0=0.01, margin1=2e-4)
        else:
            # single-point minimax sweep on worst k then global anneal
            k = int(rng.integers(2, 6))
            for vi in order[:k]:
                mask = np.ones(n, bool); mask[vi] = False
                Xr_ = x[mask]
                hole = _best_hole(Xr_, rng, ncand=8000)
                # refine hole by single-point soft-max descent
                z = hole.copy()
                for s in range(150):
                    t = s / 149.0
                    beta = 100.0 * (500.0 / 100.0) ** t
                    sc = Xr_ @ z
                    m_ = float(sc.max())
                    e_ = np.exp(beta * (sc - m_))
                    w_ = e_ / (e_.sum() + 1e-300)
                    g_ = Xr_.T @ w_
                    g_ = g_ - float(g_ @ z) * z
                    z = z - (0.05 * (1 - 0.8 * t) / (np.linalg.norm(g_) + 1e-12)) * g_
                    z /= np.linalg.norm(z) + 1e-300
                cur = float((Xr_ @ x[vi]).max())
                new = float((Xr_ @ z).max())
                if new < cur:
                    x[vi] = z
            x = _anneal(x, 300, 120.0, 500.0, 0.03, 0.003, 0.5, rng)
        v = _max_cos(x)
        if v < best_val:
            # quick LP confirm for new best
            x2 = _lp_polish(x, min(3.0, max(0.8, explore_end - time.time() + 8.0)))
            v2 = _max_cos(x2)
            if v2 < v:
                x, v = x2, v2
            best_val = v; x_best = x.copy()
        elif pool2 is not None and v < sym_rel_val and rng.random() < 0.3:
            sym_rel = x.copy(); sym_rel_val = v
        elif v < best_val + 2e-4 and rng.random() < 0.05:
            # occasional squeeze of near-best to probe plateau
            xq = _lp_polish(x, 0.8)
            vq = _max_cos(xq)
            if vq < best_val:
                best_val = vq; x_best = xq.copy()
    # final squeeze
    remain = t0 + TLIM - time.time() - 1.0
    if remain > 1.0:
        x_best = _lp_polish(x_best, min(remain, 14.0))
        best_val = _max_cos(x_best)
    x_best /= np.linalg.norm(x_best, axis=1, keepdims=True)
    return x_best
