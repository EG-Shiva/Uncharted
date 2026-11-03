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

def _adam(X0, steps, beta0, beta1, lr, rng, jitter=0.0):
    X = X0.copy()
    n, d = X.shape
    m = np.zeros_like(X)
    v = np.zeros_like(X)
    b1 = 0.9; b2 = 0.999; eps = 1e-8
    p1 = b1; p2 = b2
    if jitter > 0:
        J = rng.standard_normal((n, n)) * jitter
        J = (J + J.T) * 0.5
        np.fill_diagonal(J, 0.0)
    else:
        J = None
    best = _max_cos(X)
    bestX = X.copy()
    for t in range(1, steps + 1):
        beta = beta0 + (beta1 - beta0) * (t / steps)
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        Gw = G + J if J is not None else G
        gm = Gw.max()
        W = np.exp(beta * (Gw - gm))
        np.fill_diagonal(W, 0.0)
        rs = W.sum(axis=1, keepdims=True) + 1e-16
        D = (W @ X) / rs
        dot = (D * X).sum(axis=1, keepdims=True)
        Gt = D - dot * X
        m *= b1; m += (1.0 - b1) * Gt
        v *= b2; v += (1.0 - b2) * (Gt * Gt)
        mh = m / (1.0 - p1); vh = v / (1.0 - p2)
        p1 *= b1; p2 *= b2
        step = lr * mh / (np.sqrt(vh) + eps)
        rn = np.linalg.norm(step, axis=1, keepdims=True) + 1e-12
        scale = np.minimum(1.0, 0.05 / rn)
        X = X - step * scale
        X = _normalize(X)
        if (t % 25) == 0:
            cur = _max_cos(X)
            if cur < best:
                best = cur
                bestX = X.copy()
    cur = _max_cos(X)
    if cur < best:
        best = cur
        bestX = X.copy()
    return bestX, best

def _greedy(X0, iters, s0, s1):
    Xc = X0.copy()
    best = _max_cos(Xc)
    bestX = Xc.copy()
    dec = (s1 / s0) ** (1.0 / max(iters - 1, 1))
    st = s0
    for _ in range(iters):
        G = Xc @ Xc.T
        np.fill_diagonal(G, -2.0)
        mm = G.max()
        delta = max(0.004, 1.5 * st)
        thr = mm - delta
        W = np.where(G > thr, (G - thr + 0.01), 0.0)
        np.fill_diagonal(W, 0.0)
        Gr = W @ Xc
        dot = (Gr * Xc).sum(axis=1, keepdims=True)
        Gt = Gr - dot * Xc
        nrm = np.linalg.norm(Gt, axis=1, keepdims=True) + 1e-12
        Xc = Xc - st * (Gt / nrm)
        Xc = _normalize(Xc)
        cur = _max_cos(Xc)
        if cur < best:
            best = cur
            bestX = Xc.copy()
        st *= dec
    return bestX, best

def _find_hole(Y, rng, M=30000, refine_iters=70):
    d = Y.shape[1]
    best_s = 1e9
    best_v = None
    batch = 10000
    nb = max(1, M // batch)
    for _ in range(nb):
        V = rng.standard_normal((batch, d))
        V = _normalize(V)
        S = V @ Y.T
        sc = S.max(axis=1)
        i = int(np.argmin(sc))
        if float(sc[i]) < best_s:
            best_s = float(sc[i])
            best_v = V[i].copy()
    # single-point tangent refinement
    step = 0.05
    for _ in range(refine_iters):
        Gv = Y @ best_v
        mm = float(Gv.max())
        beta = 80.0
        w = np.exp(beta * (Gv - mm))
        w = w / (w.sum() + 1e-16)
        grad = w @ Y
        grad = grad - (grad @ best_v) * best_v
        gn = float(np.linalg.norm(grad))
        if gn < 1e-12:
            break
        trial = best_v - step * (grad / (gn + 1e-12))
        trial = trial / (np.linalg.norm(trial) + 1e-12)
        s = float((Y @ trial).max())
        if s < best_s:
            best_s = s
            best_v = trial
            step = min(step * 1.1, 0.05)
        else:
            step *= 0.5
            if step < 1e-6:
                break
    return best_v

def _contacts(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    mm = float(G.max())
    rowmax = G.max(axis=1)
    cnt = (G > mm - 1e-9).sum(axis=1)
    return G, mm, rowmax, cnt

def solve():
    t0 = time.time()
    t_end = t0 + 168.0
    rng = np.random.default_rng(0)
    n, d = 59, 22
    cands = []
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=float)
            if a.shape != (n, d):
                continue
            if not np.all(np.isfinite(a)):
                continue
            a = _normalize(a)
            cands.append(a)
        except Exception:
            pass
    if not cands:
        cands.append(_normalize(rng.standard_normal((n, d))))
    vals = [_max_cos(c) for c in cands]
    gbestX = cands[int(np.argmin(vals))].copy()
    gbest = float(min(vals))
    # quick squeeze of loaded best
    try:
        X1, v1 = _adam(gbestX, 1500, 40.0, 260.0, 0.011, rng, 0.0)
        if v1 < gbest:
            gbest, gbestX = v1, X1.copy()
        X2, v2 = _adam(X1, 700, 150.0, 320.0, 0.006, rng, 0.0)
        if v2 < gbest:
            gbest, gbestX = v2, X2.copy()
        X3, v3 = _greedy(gbestX, 500, 0.003, 0.0002)
        if v3 < gbest:
            gbest, gbestX = v3, X3.copy()
    except Exception:
        pass

    it = 0
    while time.time() < t_end - 4.0 and it < 300:
        rem = t_end - time.time()
        if rem < 5.0:
            break
        try:
            Gg, mmax, rowmax, cnt = _contacts(gbestX)
        except Exception:
            it += 1
            continue
        small = rem < 18.0
        try:
            if it % 2 == 0:
                # ---- deletion path 58 -> 59 ----
                order_cnt = np.argsort(cnt)
                order_worst = np.argsort(rowmax)
                # remove highest-contact point (mix worst/contact)
                if rng.random() < 0.75:
                    victim = int(order_cnt[-1])
                else:
                    victim = int(order_worst[-1])
                    if rng.random() < 0.3:
                        pool = order_worst[-8:].tolist()
                        victim = int(rng.choice(pool))
                Ykeep = np.delete(gbestX, [victim], axis=0)  # 58
                # polish 58-pt code until appreciably below 59 value
                s58 = 500 if small else 800
                Ya, va = _adam(Ykeep, s58, 30.0, 240.0, 0.012, rng, 0.0)
                Yb, vb = _adam(Ya, 400 if small else 600, 120.0, 300.0, 0.007, rng, 0.0)
                Y58 = Yb if vb < va else Ya
                # locate largest spherical hole with batched candidates + refinement
                Mh = 20000 if small else 35000
                hv = _find_hole(Y58, rng, M=Mh, refine_iters=60)
                X Trial = None
                Xtrial = np.vstack([Y58, hv[None, :]])
                Xtrial = _normalize(Xtrial)
                # jointly re-polish
                jit = 0.003 if (it % 4 == 0) else 0.0
                Xa, va2 = _adam(Xtrial, 500 if small else 800, 35.0, 230.0, 0.012, rng, jitter=jit)
                Xb2, vb2 = _adam(Xa, 300 if small else 600, 150.0, 320.0, 0.006, rng, 0.0)
                bt, bX = (vb2, Xb2) if vb2 < va2 else (va2, Xa)
                if bt < gbest:
                    Xc, vc = _adam(bX, 800 if rem > 30 else 300, 80.0, 360.0, 0.005, rng, 0.0)
                    if vc < bt:
                        bt, bX = vc, Xc
                    Xd, vd = _greedy(bX, 400 if rem > 25 else 150, 0.0025, 0.00012)
                    if vd < bt:
                        bt, bX = vd, Xd
                    if bt < gbest:
                        gbest, gbestX = bt, bX.copy()
            else:
                # ---- overshoot path 63 -> 59 ----
                # create 61-64 points: start from gbest, add hole candidates / splits
                target = int(rng.integers(62, 64))  # 62 or 63
                nadd = target - 59
                Ycur = gbestX.copy()
                # mix: mostly hole candidates, sometimes split highly contacted points
                for k in range(nadd):
                    if rng.random() < 0.7:
                        Mh2 = 12000 if small else 20000
                        hv2 = _find_hole(Ycur, rng, M=Mh2, refine_iters=40)
                        Ycur = np.vstack([Ycur, hv2[None, :]])
                    else:
                        _, _, _, cnt2 = _contacts(_normalize(Ycur))
                        c = int(np.argmax(cnt2)) if rng.random() < 0.8 else int(rng.integers(Ycur.shape[0]))
                        pert = rng.standard_normal((d,)) * 0.03
                        # tangent-ish perturbation
                        xv = Ycur[c] + pert
                        xv = xv / (np.linalg.norm(xv) + 1e-12)
                        Ycur = np.vstack([Ycur, xv[None, :]])
                    Ycur = _normalize(Ycur)
                # polish overcrowded system briefly
                Yo, vo = _adam(Ycur, 300 if small else 500, 30.0, 200.0, 0.012, rng, 0.0)
                Ycur = Yo
                # delete one by one until 59: remove point whose removal most reduces max
                while Ycur.shape[0] > 59:
                    if time.time() > t_end - 5.0:
                        break
                    mm_cur = Ycur.shape[0]
                    Gm = Ycur @ Ycur.T
                    np.fill_diagonal(Gm, -2.0)
                    best_rem = 1e9
                    best_k = 0
                    # estimate remaining max for each k
                    for k in range(mm_cur):
                        # max excluding row/col k
                        # fast: max of Gm with row k and col k masked
                        # copy-free: compute max over masked view
                        rmax = -1e9
                        # loop rows
                        # use masking via temp: set row/col to -2 on copy of max candidates?
                        # brute but mm~60 so fine
                        tmp = Gm.copy()
                        tmp[k, :] = -2.0
                        tmp[:, k] = -2.0
                        cur = float(tmp.max())
                        if cur < best_rem:
                            best_rem = cur
                            best_k = k
                    Ycur = np.delete(Ycur, [best_k], axis=0)
                    Ycur = _normalize(Ycur)
                    # re-polish after each deletion
                    Yr, vr = _adam(Ycur, 250 if small else 400, 60.0, 260.0, 0.008, rng, 0.0)
                    Ycur = Yr
                Xa2, va3 = _adam(Ycur, 400 if small else 700, 80.0, 320.0, 0.007, rng, 0.0)
                bt2 = va3; bX2 = Xa2
                if bt2 < gbest:
                    Xc2, vc2 = _adam(bX2, 700 if rem > 30 else 250, 100.0, 360.0, 0.005, rng, 0.0)
                    if vc2 < bt2:
                        bt2, bX2 = vc2, Xc2
                    Xd2, vd2 = _greedy(bX2, 350 if rem > 25 else 120, 0.002, 0.0001)
                    if vd2 < bt2:
                        bt2, bX2 = vd2, Xd2
                    if bt2 < gbest:
                        gbest, gbestX = bt2, bX2.copy()
        except Exception:
            pass
        it += 1

    try:
        rem = t_end - time.time()
        if rem > 1.2:
            Xf, vf = _adam(gbestX, 800 if rem > 8 else 300, 120.0, 380.0, 0.005, rng, 0.0)
            if vf < gbest:
                gbest, gbestX = vf, Xf.copy()
            rem2 = t_end - time.time()
            if rem2 > 0.8:
                Xg, vg = _greedy(gbestX, 400 if rem2 > 5 else 150, 0.0012, 0.00005)
                if vg < gbest:
                    gbest, gbestX = vg, Xg.copy()
    except Exception:
        pass
    gbestX = _normalize(gbestX)
    return gbestX
