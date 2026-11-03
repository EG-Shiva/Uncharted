import time
import numpy as np

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _normalize(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n = np.maximum(n, 1e-12)
    return X / n

def _adam_lse(X0, steps, beta0, beta1, lr0, lr1, t_end, track_best=True):
    X = X0.copy()
    n, d = X.shape
    m = np.zeros_like(X)
    v = np.zeros_like(X)
    b1 = 0.9; b2 = 0.999; eps = 1e-8
    best = _max_cos(X)
    bestX = X.copy()
    pow1 = b1; pow2 = b2
    lb0 = np.log(beta0); lb1 = np.log(beta1)
    ll0 = np.log(lr0); ll1 = np.log(lr1)
    den = max(steps - 1, 1)
    for t in range(1, steps + 1):
        if (t & 63) == 0 and time.time() > t_end:
            break
        frac = (t - 1) / den
        beta = float(np.exp(lb0 + (lb1 - lb0) * frac))
        lr = float(np.exp(ll0 + (ll1 - ll0) * frac))
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        gm = float(G.max())
        W = np.exp(beta * (G - gm))
        np.fill_diagonal(W, 0.0)
        rs = W.sum(axis=1, keepdims=True) + 1e-16
        D = (W @ X) / rs
        dot = (D * X).sum(axis=1, keepdims=True)
        Gt = D - dot * X
        m *= b1; m += (1.0 - b1) * Gt
        v *= b2; v += (1.0 - b2) * (Gt * Gt)
        mhat = m / (1.0 - pow1); vhat = v / (1.0 - pow2)
        pow1 *= b1; pow2 *= b2
        step = lr * mhat / (np.sqrt(vhat) + eps)
        rn = np.linalg.norm(step, axis=1, keepdims=True) + 1e-12
        scale = np.minimum(1.0, 0.05 / rn)
        X = X - step * scale
        X = _normalize(X)
        if track_best and (t % 25) == 0:
            cur = _max_cos(X)
            if cur < best:
                best = cur; bestX = X.copy()
    if track_best:
        cur = _max_cos(X)
        if cur < best:
            best = cur; bestX = X.copy()
        return bestX, best
    else:
        return X, _max_cos(X)

def _greedy_polish(X0, iters, step0, step1, t_end, rng=None):
    Xcur = X0.copy()
    best = _max_cos(Xcur)
    bestX = Xcur.copy()
    decay = (step1 / step0) ** (1.0 / max(iters - 1, 1))
    step = step0
    n = Xcur.shape[0]
    for t in range(iters):
        if (t & 63) == 0 and time.time() > t_end:
            break
        G = Xcur @ Xcur.T
        np.fill_diagonal(G, -2.0)
        mm = float(G.max())
        delta = max(0.004, 1.5 * step)
        thr = mm - delta
        Mask = G > thr
        W = np.where(Mask, (G - thr + 0.01), 0.0)
        np.fill_diagonal(W, 0.0)
        Gr = W @ Xcur
        dot = (Gr * Xcur).sum(axis=1, keepdims=True)
        Gt = Gr - dot * Xcur
        nrm = np.linalg.norm(Gt, axis=1, keepdims=True) + 1e-12
        has = (nrm[:, 0] > 1e-9)
        upd = np.zeros_like(Xcur)
        upd[has] = Gt[has] / nrm[has]
        if rng is not None and np.any(~has):
            idx = np.where(~has)[0]
            R = rng.standard_normal(size=(len(idx), Xcur.shape[1]))
            Xr = Xcur[idx]
            R = R - (R * Xr).sum(axis=1, keepdims=True) * Xr
            rn2 = np.linalg.norm(R, axis=1, keepdims=True) + 1e-12
            upd[idx] = (R / rn2) * 0.3
        Xcur = Xcur - step * upd
        Xcur = _normalize(Xcur)
        cur = _max_cos(Xcur)
        if cur < best:
            best = cur; bestX = Xcur.copy()
        step *= decay
    return bestX, best

def _void_center(X, rng, S=20000):
    d = X.shape[1]
    C = _normalize(rng.standard_normal(size=(S, d)))
    sc = C @ X.T
    mx = sc.max(axis=1)
    i = int(np.argmin(mx))
    c = C[i].copy()
    # refine cap center: minimise max dot
    s0, s1, steps = 0.05, 0.002, 200
    dec = (s1 / s0) ** (1.0 / max(steps - 1, 1))
    s = s0
    for _ in range(steps):
        dots = X @ c
        j = int(np.argmax(dots))
        g = X[j] - dots[j] * c
        gn = np.linalg.norm(g) + 1e-12
        if gn < 1e-12:
            break
        g = g / gn
        c = c - s * g
        c = c / max(np.linalg.norm(c), 1e-12)
        s *= dec
    return c

def _tangent_dir(xk, c, rng):
    t = c - float(c @ xk) * xk
    n = float(np.linalg.norm(t))
    if n < 1e-6:
        r = rng.standard_normal(size=xk.shape)
        r = r - float(r @ xk) * xk
        n2 = float(np.linalg.norm(r))
        if n2 < 1e-12:
            r = rng.standard_normal(size=xk.shape); r = r - float(r @ xk) * xk
            n2 = float(np.linalg.norm(r)) + 1e-12
        return r / n2
    return t / n

def _random_tangent(xk, rng):
    r = rng.standard_normal(size=xk.shape)
    r = r - float(r @ xk) * xk
    n = float(np.linalg.norm(r)) + 1e-12
    return r / n

def solve():
    t0 = time.time()
    BUDGET = 168.0
    t_end = t0 + BUDGET
    rng = np.random.default_rng(0)
    n, d = 59, 22

    def rand_X(nn=n, dd=d):
        return _normalize(rng.standard_normal(size=(nn, dd)))

    cands = []
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=float)
            if a.shape == (n, d) and np.all(np.isfinite(a)):
                a = _normalize(a)
                cands.append(a)
        except Exception:
            pass
    if not cands:
        cands.append(rand_X())
    vals = [_max_cos(c) for c in cands]
    gbestX = cands[int(np.argmin(vals))].copy()
    gbest = float(min(vals))

    # quick initial polish of incumbent (short)
    try:
        X1, v1 = _adam_lse(gbestX, 1200, 40.0, 260.0, 0.015, 0.006, t_end - 1.0)
        if v1 < gbest:
            gbest, gbestX = v1, X1.copy()
        Xg, vg = _greedy_polish(X1 if v1 <= gbest + 1e-12 else gbestX, 500, 0.005, 0.0003, t_end - 1.0, rng)
        if vg < gbest:
            gbest, gbestX = vg, Xg.copy()
    except Exception:
        pass

    # ---- contact analysis: victims = highest-contact points ----
    try:
        G0 = gbestX @ gbestX.T
        np.fill_diagonal(G0, -2.0)
        mm0 = float(G0.max())
        rowmax0 = G0.max(axis=1)
        cnt0 = (G0 > (mm0 - 0.006)).sum(axis=1)
        order_vict = np.argsort(-rowmax0 - 0.0005 * cnt0)
        del_list = [int(q) for q in order_vict[:10]]
    except Exception:
        del_list = list(range(10))

    # ---- Phase B: deletion screening on 58-point codes (time-boxed ~20s) ----
    best58 = None; best58v = 1e9; best_del = -1
    phaseB_end = min(t_end - 80.0, time.time() + 30.0)
    for di in del_list:
        if time.time() > phaseB_end:
            break
        mask = np.ones(n, dtype=bool); mask[di] = False
        Y58 = gbestX[mask].copy()
        # tiny shake to leave basin
        Y58 = _normalize(Y58 + 0.004 * rng.standard_normal(size=Y58.shape))
        lim = min(phaseB_end, time.time() + 3.5)
        Yp, vp = _adam_lse(Y58, 700, 30.0, 220.0, 0.015, 0.005, lim)
        if vp < best58v:
            best58v = vp; best58 = Yp.copy(); best_del = di
    # thorough optimisation of best 58-point problem
    if best58 is not None and time.time() < t_end - 70.0:
        lim = min(t_end - 65.0, time.time() + 12.0)
        Yp, vp = _adam_lse(best58, 2000, 40.0, 380.0, 0.012, 0.004, lim)
        if vp < best58v:
            best58v = vp; best58 = Yp.copy()
        lim2 = min(t_end - 62.0, time.time() + 4.0)
        Yq, vq = _greedy_polish(best58, 600, 0.005, 0.0002, lim2, rng)
        if vq < best58v:
            best58v = vq; best58 = Yq.copy()
    if best58 is None:
        mask = np.ones(n, dtype=bool); mask[del_list[0]] = False
        best58 = gbestX[mask].copy()

    # ---- Phase C: split restoration with staged LSE annealing ----
    # find void of best58
    try:
        cstar = _void_center(best58, rng, S=20000)
        dots_c = best58 @ cstar
        k_void = int(np.argmax(dots_c))
        dots_c2 = np.sort(dots_c)
        # second closest
        k_void2 = int(np.argsort(dots_c)[-2])
        rowmax58 = (best58 @ best58.T + 2.0*np.eye(58)).max(axis=1) if False else None
    except Exception:
        cstar = _normalize(rng.standard_normal(size=(d,)))[0:0] if False else None
        k_void = int(rng.integers(0, 58)); k_void2 = int(rng.integers(0, 58))
        cstar = None
    G58 = best58 @ best58.T; np.fill_diagonal(G58, -2.0)
    rm58 = G58.max(axis=1)
    k_loose = int(np.argmin(rm58))
    k_rand = int(rng.integers(0, 58))

    xk_list = []
    # build split configs: (k, mode, eps)
    cfgs = []
    cfgs.append((k_void, 'void', 0.08))
    cfgs.append((k_void, 'rand', 0.08))
    cfgs.append((k_void, 'void', 0.15))
    cfgs.append((k_void2, 'void', 0.08))
    cfgs.append((k_loose, 'rand', 0.05))
    cfgs.append((k_loose, 'void', 0.12))
    cfgs.append((k_rand, 'rand', 0.10))
    cfgs.append((k_void, 'void', 0.04))

    split_end = min(t_end - 35.0, time.time() + 45.0)
    for (kk, mode, eps) in cfgs:
        if time.time() > split_end:
            break
        kk = int(kk) % 58
        xk = best58[kk].copy()
        if mode == 'void' and cstar is not None:
            tt = _tangent_dir(xk, cstar, rng)
        else:
            tt = _random_tangent(xk, rng)
        a = _normalize((xk + eps * tt).reshape(1, -1))[0]
        b = _normalize((xk - eps * tt).reshape(1, -1))[0]
        Y59 = np.vstack([best58[:kk], a.reshape(1, -1), best58[kk+1:], b.reshape(1, -1)])
        # Y59 has 60 rows; need 59: we inserted b as extra and kept all 58 others + split into 2 = 59?
        # best58 has 58 rows; replacing row kk by a and appending b gives 59
        Y59 = np.vstack([best58.copy(), np.zeros((1, d))])
        Y59[kk] = a
        Y59[58] = b
        # staged annealing: low beta first to separate pair, then high beta
        rem = split_end - time.time()
        if rem < 2.0:
            break
        lim1 = min(split_end, time.time() + min(3.0, rem * 0.4))
        Z1, _ = _adam_lse(Y59, 800, 20.0, 90.0, 0.02, 0.008, lim1, track_best=False)
        # Z1 raw; now anneal up
        lim2 = min(split_end, time.time() + min(4.0, (split_end - time.time()) * 0.6))
        Z2, v2 = _adam_lse(Z1, 1300, 120.0, 550.0, 0.010, 0.003, lim2)
        lim3 = min(split_end, time.time() + 1.5)
        Z3, v3 = _greedy_polish(Z2, 500, 0.004, 0.0002, lim3, None)
        bv, bX = (v2, Z2) if v2 < v3 else (v3, Z3)
        if bv < gbest:
            gbest, gbestX = bv, bX.copy()

    # ---- Phase D: dimensional lift to R23, optimise, SVD-project back ----
    if time.time() < t_end - 25.0:
        for lift in range(2):
            if time.time() > t_end - 22.0:
                break
            sigma = 0.05 if lift == 0 else 0.10
            try:
                E = rng.standard_normal(size=(n, 1)) * sigma
                X23 = np.hstack([gbestX, E])
                X23 = X23 / np.maximum(np.linalg.norm(X23, axis=1, keepdims=True), 1e-12)
                lim = min(t_end - 18.0, time.time() + 7.0)
                Y23, _ = _adam_lse(X23, 1600, 35.0, 300.0, 0.014, 0.005, lim, track_best=False)
                # rotate away last dim via smallest singular vector
                U, S, Vt = np.linalg.svd(Y23 - Y23.mean(axis=0, keepdims=True), full_matrices=False)
                vmin = Vt[-1, :].copy()  # (23,)
                # Householder rotating vmin to e_z
                ez = np.zeros(23); ez[22] = 1.0
                vv = vmin - ez
                nvv = float(np.linalg.norm(vv))
                if nvv < 1e-12:
                    R = np.eye(23)
                else:
                    vv = vv / nvv
                    # H = I - 2 vv vv^T ; maps vmin to ez (up to sign)
                    # apply to rows: Y @ H
                    H = np.eye(23) - 2.0 * np.outer(vv, vv)
                    R = H
                    # fix sign so vmin maps to +ez
                    if float(vmin @ (R[:, 22])) < 0:
                        R[:, 22] *= -1.0
                Yr = Y23 @ R
                Xp = Yr[:, :22].copy()
                Xp = _normalize(Xp)
                lim2 = min(t_end - 14.0, time.time() + 5.0)
                Xp2, vp2 = _adam_lse(Xp, 1200, 80.0, 450.0, 0.009, 0.003, lim2)
                lim3 = min(t_end - 12.0, time.time() + 2.0)
                Xp3, vp3 = _greedy_polish(Xp2, 500, 0.003, 0.0002, lim3, None)
                bv, bX = (vp2, Xp2) if vp2 < vp3 else (vp3, Xp3)
                if bv < gbest:
                    gbest, gbestX = bv, bX.copy()
            except Exception:
                continue

    # ---- final ultra-fine polish of incumbent ----
    if time.time() < t_end - 0.8:
        try:
            Xf, vf = _adam_lse(gbestX, 900, 300.0, 900.0, 0.003, 0.0008, t_end - 0.4)
            if vf < gbest:
                gbest, gbestX = vf, Xf.copy()
            Xg, vg = _greedy_polish(gbestX, 700, 0.002, 0.00004, t_end - 0.12, None)
            if vg < gbest:
                gbest, gbestX = vg, Xg.copy()
        except Exception:
            pass
    gbestX = _normalize(gbestX)
    return gbestX
