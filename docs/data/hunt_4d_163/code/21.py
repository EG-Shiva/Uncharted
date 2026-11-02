import time
import numpy as np
from scipy.optimize import linprog, minimize
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _lp_polish(x, deadline, r0=1e-3):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    try:
        best = _max_cos(x)
    except Exception:
        return x, 1.0
    radius = r0
    iu0, iu1 = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        try:
            G = x @ x.T
            c = G[iu0, iu1]
            m = float(c.max())
            sel = c > m - max(20.0 * radius, 1e-9)
            if np.count_nonzero(sel) == 0:
                sel[np.argmax(c)] = True
            I = iu0[sel]; J = iu1[sel]
            csel = c[sel]
            p = len(I)
            nv = n * d + 1
            rows = np.repeat(np.arange(p), 2 * d + 1)
            tmpc = np.empty((p, 2 * d + 1), dtype=int)
            ar = np.arange(d)
            tmpc[:, :d] = I[:, None] * d + ar
            tmpc[:, d:2 * d] = J[:, None] * d + ar
            tmpc[:, 2 * d] = n * d
            cols = tmpc.ravel()
            cc = np.empty((p, 2 * d + 1), dtype=float)
            cc[:, :d] = x[J]
            cc[:, d:2 * d] = x[I]
            cc[:, 2 * d] = -1.0
            vals = cc.ravel()
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv); obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
            if res.status != 0:
                radius *= 0.5
                continue
            y = x + res.x[:-1].reshape(n, d)
            y /= np.maximum(np.linalg.norm(y, axis=1, keepdims=True), 1e-12)
            v = _max_cos(y)
            if v < best:
                x, best = y, v
                radius = min(radius * 1.5, 0.05)
            else:
                radius *= 0.5
        except Exception:
            radius *= 0.5
            continue
    return x, best

def _gd(x0, steps, T0, T1, lr0, mu, rng, noise0=0.0, deadline=None):
    x = x0.copy()
    n, d = x.shape
    v = np.zeros_like(x)
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    best_val = float(G.max())
    best = x.copy()
    if T0 > 0 and T1 > 0:
        lratio = np.log(T1 / T0)
    else:
        lratio = 0.0
    for k in range(steps):
        if deadline is not None and (k & 255) == 0 and time.time() > deadline:
            break
        frac = k / max(1, steps - 1)
        T = T0 * np.exp(lratio * frac)
        if T < 1e-12:
            T = 1e-12
        lr = lr0 * (1.0 - 0.55 * frac)
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        E = np.exp((G - m) / T)
        np.fill_diagonal(E, 0.0)
        S = float(E.sum() * 0.5)
        if S < 1e-300 or not np.isfinite(S):
            S = 1e-300
        grad = (E @ x) / S
        grad -= (grad * x).sum(axis=1, keepdims=True) * x
        v = mu * v + (1.0 - mu) * grad
        x = x - lr * v
        if noise0 > 0:
            amp = noise0 * (1.0 - frac)
            if amp > 0:
                nz = rng.randn(n, d) * amp
                nz -= (nz * x).sum(axis=1, keepdims=True) * x
                x = x + nz
        x /= np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
        if m < best_val:
            best_val = m
            best = x.copy()
    return best, best_val

def _lbfgs_smooth(x_init, deadline, taus=(0.004, 0.0015, 0.0006), iters=(120, 120, 180)):
    x = x_init / np.maximum(np.linalg.norm(x_init, axis=1, keepdims=True), 1e-12)
    n, d = x.shape
    iu0, iu1 = np.triu_indices(n, 1)
    Np = len(iu0)
    y = x.ravel().copy()
    for tau, maxiter in zip(taus, iters):
        if time.time() > deadline - 0.3:
            break
        def fg(yflat, _tau=tau):
            Y = yflat.reshape(n, d)
            rn = np.linalg.norm(Y, axis=1, keepdims=True)
            rn = np.maximum(rn, 1e-12)
            X = Y / rn
            G = X @ X.T
            g = G[iu0, iu1]
            m = float(g.max())
            e = np.exp((g - m) / _tau)
            s = float(e.sum())
            if s <= 0 or not np.isfinite(s):
                return float(m), np.zeros_like(yflat)
            L = m + _tau * np.log(s / Np)
            w = e / s
            W = np.zeros((n, n))
            W[iu0, iu1] = w
            W = W + W.T
            GX = W @ X
            dot = (GX * X).sum(axis=1, keepdims=True)
            GY = (GX - dot * X) / rn
            return float(L), GY.ravel()
        try:
            res = minimize(fg, y, method="L-BFGS-B", jac=True,
                           options={"maxiter": int(maxiter), "maxfun": int(maxiter * 5),
                                    "ftol": 1e-14, "gtol": 1e-9})
            y = res.x
            Yt = y.reshape(n, d)
            Yt /= np.maximum(np.linalg.norm(Yt, axis=1, keepdims=True), 1e-12)
            y = Yt.ravel().copy()
        except Exception:
            break
    Xf = y.reshape(n, d)
    Xf /= np.maximum(np.linalg.norm(Xf, axis=1, keepdims=True), 1e-12)
    return Xf, _max_cos(Xf)

def _refine_hole(h0, Xf, steps=250, T0=0.02, T1=0.002, lr0=0.08):
    h = h0 / max(np.linalg.norm(h0), 1e-12)
    lratio = np.log(T1 / T0)
    for k in range(steps):
        frac = k / max(1, steps - 1)
        T = T0 * np.exp(lratio * frac)
        lr = lr0 * (1.0 - 0.5 * frac)
        dots = Xf @ h
        m = float(dots.max())
        e = np.exp((dots - m) / max(T, 1e-12))
        s = float(e.sum())
        if s <= 0 or not np.isfinite(s):
            break
        w = e / s
        grad = w @ Xf
        grad = grad - float(grad @ h) * h
        h = h - lr * grad
        nh = np.linalg.norm(h)
        if nh < 1e-12:
            break
        h /= nh
    return h

def _find_hole(Xf, rng, n_samples=20000):
    Y = rng.randn(n_samples, 4)
    Y /= np.maximum(np.linalg.norm(Y, axis=1, keepdims=True), 1e-12)
    D = Y @ Xf.T
    vals = D.max(axis=1)
    idx = np.argsort(vals)[:6]
    best_h = Y[idx[0]]
    best_v = float(vals[idx[0]])
    for ii in idx:
        h = _refine_hole(Y[ii], Xf)
        v = float((Xf @ h).max())
        if v < best_v:
            best_v = v
            best_h = h
    return best_h, best_v

def solve():
    t0 = time.time()
    BUDGET = 172.0
    SEARCH_END = t0 + 132.0
    rng = np.random.RandomState(7)
    n = 163
    try:
        bx = np.load("best.npy")
    except Exception:
        bx = None
    try:
        rec = np.load("record.npy")
    except Exception:
        rec = None
    cands = []
    if bx is not None:
        bx = np.asarray(bx, dtype=float)
        if bx.shape == (163, 4):
            cands.append(bx / np.maximum(np.linalg.norm(bx, axis=1, keepdims=True), 1e-12))
    if rec is not None:
        rec = np.asarray(rec, dtype=float)
        if rec.shape == (163, 4):
            cands.append(rec / np.maximum(np.linalg.norm(rec, axis=1, keepdims=True), 1e-12))
    if len(cands) == 0:
        c = rng.randn(n, 4)
        c /= np.linalg.norm(c, axis=1, keepdims=True)
        cands.append(c)
    vals = [_max_cos(c) for c in cands]
    bi = int(np.argmin(vals))
    best_x = cands[bi].copy()
    best_val = float(vals[bi])
    # quick confirm polish if time allows and best is record-like (avoid long at start)
    # main basin-hopping loop
    sigmas = [0.02, 0.06, 0.14, 0.008, 0.04, 0.10]
    trial = 0
    while time.time() < SEARCH_END:
        if time.time() > t0 + BUDGET - 32:
            break
        sigma = sigmas[trial % len(sigmas)]
        mode = trial % 6
        # occasional full restart
        if trial % 19 == 18:
            cur = rng.randn(n, 4)
            cur /= np.linalg.norm(cur, axis=1, keepdims=True)
            dl = min(time.time() + 4.0, SEARCH_END)
            cur_b, v = _gd(cur, steps=7000, T0=0.025, T1=0.0009, lr0=0.038, mu=0.88, rng=rng, noise0=0.002, deadline=dl)
            if v < best_val:
                cur_b2, v2 = _lbfgs_smooth(cur_b, deadline=time.time() + 2.0)
                if v2 < v:
                    cur_b, v = cur_b2, v2
                if v < best_val and time.time() < t0 + BUDGET - 35:
                    pol, vp = _lp_polish(cur_b, deadline=time.time() + 2.5)
                    if vp < best_val:
                        best_x, best_val = pol, vp
                    elif v < best_val:
                        # keep gd result even if polish fails to beat
                        if v < best_val:
                            best_x, best_val = cur_b, v
                elif v < best_val:
                    best_x, best_val = cur_b, v
            trial += 1
            continue
        cur = best_x.copy()
        if mode == 0:
            cur = cur + rng.randn(*cur.shape) * sigma
            cur /= np.maximum(np.linalg.norm(cur, axis=1, keepdims=True), 1e-12)
        elif mode == 1:
            G = cur @ cur.T
            np.fill_diagonal(G, -2.0)
            m = float(G.max())
            deg = (G > m - 0.002).sum(axis=1)
            rowm = G.max(axis=1)
            order = np.argsort(rowm + 0.0005 * deg)
            # shake worst 12 strongly, rest weakly
            worst = order[-12:]
            noise = rng.randn(*cur.shape) * (sigma * 0.25)
            noise[worst] += rng.randn(len(worst), 4) * (sigma * 2.2)
            cur = cur + noise
            cur /= np.maximum(np.linalg.norm(cur, axis=1, keepdims=True), 1e-12)
        elif mode == 2:
            # surgery: remove k worst to optimized holes
            G = cur @ cur.T
            np.fill_diagonal(G, -2.0)
            rowm = G.max(axis=1)
            m = float(G.max())
            deg = (G > m - 0.002).sum(axis=1)
            score = rowm + 0.0004 * deg + rng.randn(n) * 1e-6
            k = 2 + (trial % 2)  # 2 or 3
            worst = np.argsort(score)[-k:]
            keep_mask = np.ones(n, dtype=bool)
            keep_mask[worst] = False
            # sequentially reinsert
            Xf = cur[~np.zeros(n, dtype=bool)] if False else cur[keep_mask].copy()
            for w in worst:
                h, _ = _find_hole(Xf, rng, n_samples=16000)
                h = h + rng.randn(4) * 0.008
                h /= max(np.linalg.norm(h), 1e-12)
                cur[w] = h
                Xf = np.vstack([Xf, h[None, :]])
            cur = cur + rng.randn(*cur.shape) * (sigma * 0.2)
            cur /= np.maximum(np.linalg.norm(cur, axis=1, keepdims=True), 1e-12)
        elif mode == 3:
            # spherical cap shake
            u = rng.randn(4); u /= np.linalg.norm(u)
            thr = 0.45 + 0.35 * rng.rand()
            dots = cur @ u
            mask = dots > thr
            cnt = int(mask.sum())
            if cnt == 0 or cnt > n // 2:
                idx = rng.choice(n, 16, replace=False)
                cur[idx] += rng.randn(16, 4) * sigma
            else:
                # common tangent shift + individual
                vdir = rng.randn(4); vdir -= float(vdir @ u) * u
                nv = np.linalg.norm(vdir)
                if nv < 1e-9:
                    vdir = rng.randn(4)
                    nv = np.linalg.norm(vdir)
                vdir /= nv
                cur[mask] += vdir[None, :] * (sigma * 1.5) + rng.randn(cnt, 4) * (sigma * 0.6)
                # small global shake
                cur += rng.randn(*cur.shape) * (sigma * 0.15)
            cur /= np.maximum(np.linalg.norm(cur, axis=1, keepdims=True), 1e-12)
        elif mode == 4:
            # pair-breaking: push apart top contact pairs
            G = cur @ cur.T
            np.fill_diagonal(G, -2.0)
            iu0, iu1 = np.triu_indices(n, 1)
            c = G[iu0, iu1]
            top = np.argsort(c)[-8:]
            for tt in top[:5]:
                i = int(iu0[tt]); j = int(iu1[tt])
                if rng.rand() < 0.7:
                    d = cur[i] - cur[j]
                    d -= float(d @ cur[i]) * cur[i]
                    nd = np.linalg.norm(d)
                    e = cur[j] - cur[i]
                    e -= float(e @ cur[j]) * cur[j]
                    ne = np.linalg.norm(e)
                    s = sigma * (0.8 + rng.rand())
                    if nd > 1e-9:
                        cur[i] += (d / nd) * s
                    if ne > 1e-9:
                        cur[j] += (e / ne) * s
            cur += rng.randn(*cur.shape) * (sigma * 0.35)
            cur /= np.maximum(np.linalg.norm(cur, axis=1, keepdims=True), 1e-12)
        else:
            # single rattler-to-hole + shake
            G = cur @ cur.T
            np.fill_diagonal(G, -2.0)
            m = float(G.max())
            deg = (G > m - 0.001).sum(axis=1)
            w = int(np.argmin(deg + rng.randn(n) * 0.05))
            keep = np.delete(np.arange(n), w)
            Xf = cur[keep]
            h, _ = _find_hole(Xf, rng, n_samples=16000)
            h = h + rng.randn(4) * 0.005
            h /= max(np.linalg.norm(h), 1e-12)
            cur[w] = h
            cur += rng.randn(*cur.shape) * (sigma * 0.3)
            cur /= np.maximum(np.linalg.norm(cur, axis=1, keepdims=True), 1e-12)
        # anneal
        dl = min(time.time() + 3.0, SEARCH_END + 1.0)
        cur_b, v = _gd(cur, steps=4200, T0=0.016, T1=0.0011, lr0=0.032, mu=0.88, rng=rng, noise0=0.0, deadline=dl)
        # threshold to invest more
        if v < best_val + 0.0015:
            cur_b2, v2 = _lbfgs_smooth(cur_b, deadline=time.time() + 1.6, taus=(0.003, 0.0012, 0.0005), iters=(90, 90, 120))
            if v2 < v:
                cur_b, v = cur_b2, v2
        if v < best_val:
            if time.time() < t0 + BUDGET - 34:
                pol, vp = _lp_polish(cur_b, deadline=time.time() + 2.2)
                if vp < best_val:
                    best_x, best_val = pol, vp
                elif v < best_val:
                    # accept gd/lbfgs even if LP slightly worse but gd beats best
                    if v < best_val:
                        # quick check: polish may have failed, keep raw
                        best_x, best_val = cur_b, v
            else:
                best_x, best_val = cur_b, v
        trial += 1
    remain = BUDGET - (time.time() - t0) - 1.0
    if remain > 4.0:
        remain = min(remain, 38.0)
        best_x, best_val = _lp_polish(best_x, deadline=time.time() + remain, r0=1e-3)
    best_x = best_x / np.maximum(np.linalg.norm(best_x, axis=1, keepdims=True), 1e-12)
    return best_x
