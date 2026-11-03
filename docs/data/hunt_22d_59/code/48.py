import time
import numpy as np

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _norm(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n = np.maximum(n, 1e-12)
    return X / n

def _adam_lse(Y0, steps, beta0, beta1, lr0, lr1, deadline=None):
    Y = Y0.copy()
    n, d = Y.shape
    M = np.zeros_like(Y)
    V = np.zeros_like(Y)
    b1 = 0.9; b2 = 0.999; eps = 1e-8
    G0 = Y @ Y.T
    np.fill_diagonal(G0, -2.0)
    best_val = float(G0.max())
    best_Y = Y.copy()
    if steps <= 1:
        lb = 0.0; ll = 0.0
    else:
        lb = np.log(beta1 / beta0) / (steps - 1)
        ll = np.log(lr1 / lr0) / (steps - 1)
    lbo = np.log(beta0); llo = np.log(lr0)
    b1p = 1.0; b2p = 1.0
    eye_n = n
    for t in range(1, steps + 1):
        if deadline is not None and (t % 200 == 0) and time.time() > deadline:
            break
        beta = float(np.exp(lbo + lb * (t - 1)))
        lr = float(np.exp(llo + ll * (t - 1)))
        G = Y @ Y.T
        np.fill_diagonal(G, -10.0)
        gmax = float(G.max())
        if gmax < best_val:
            best_val = gmax
            best_Y = Y.copy()
        G -= gmax
        G *= beta
        # avoid overflow: G<=0 so exp<=1
        np.exp(G, out=G)
        np.fill_diagonal(G, 0.0)
        s = float(G.sum())
        if not np.isfinite(s) or s <= 0:
            continue
        G /= s
        Gr = G @ Y
        # symmetrize factor (G symmetric) -> 2x, fold into lr
        Gr = Gr * 2.0
        dot = np.sum(Gr * Y, axis=1, keepdims=True)
        Gr -= Y * dot
        b1p *= b1; b2p *= b2
        M *= b1; M += (1.0 - b1) * Gr
        V *= b2; V += (1.0 - b2) * (Gr * Gr)
        Mh = M / (1.0 - b1p)
        Vh = V / (1.0 - b2p)
        Y -= lr * Mh / (np.sqrt(Vh) + eps)
        rn = np.linalg.norm(Y, axis=1, keepdims=True)
        rn = np.maximum(rn, 1e-12)
        Y /= rn
    Gf = Y @ Y.T
    np.fill_diagonal(Gf, -2.0)
    vf = float(Gf.max())
    if vf < best_val:
        best_val = vf
        best_Y = Y.copy()
    return best_Y, best_val

def _greedy(Y0, steps, lr0, tol, deadline=None):
    Y = Y0.copy()
    n = Y.shape[0]
    Gc = Y @ Y.T
    np.fill_diagonal(Gc, -2.0)
    best_val = float(Gc.max())
    best_Y = Y.copy()
    lr = float(lr0)
    for s in range(steps):
        if deadline is not None and (s % 200 == 0) and time.time() > deadline:
            break
        G = Y @ Y.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        if m < best_val:
            best_val = m
            best_Y = Y.copy()
        A = G > (m - tol)
        np.fill_diagonal(A, False)
        Af = A.astype(np.float64)
        AY = Af @ Y
        deg = np.maximum(Af.sum(axis=1, keepdims=True), 1.0)
        # normalized direction
        Step = AY / deg
        dot = np.sum(Step * Y, axis=1, keepdims=True)
        Step -= Y * dot
        Yt = Y - lr * Step
        Yt = _norm(Yt)
        Gt = Yt @ Yt.T
        np.fill_diagonal(Gt, -2.0)
        mt = float(Gt.max())
        if mt < m + 1e-12:
            Y = Yt
            if (s + 1) % 400 == 0:
                lr *= 0.7
        else:
            lr *= 0.5
            if lr < 1e-7:
                lr = 1e-7
                Y = Yt
    return best_Y, best_val

def _find_hole(X, rng, nsamp=8000):
    n, d = X.shape
    H = rng.standard_normal((nsamp, d))
    H = _norm(H)
    S = H @ X.T
    rowmax = S.max(axis=1)
    i = int(np.argmin(rowmax))
    return H[i].copy(), float(rowmax[i])

def solve():
    t0 = time.time()
    deadline = t0 + 168.0
    inner_dead = deadline - 6.0
    rng = np.random.default_rng(7)
    n, d = 59, 22

    cands = []
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=np.float64)
            if a.shape == (n, d) and np.all(np.isfinite(a)):
                nr = np.linalg.norm(a, axis=1)
                if np.all(nr > 1e-9):
                    cands.append(a / nr[:, None])
        except Exception:
            pass
    if len(cands) == 0:
        Xrec = _norm(rng.standard_normal((n, d)))
    else:
        vals = [_max_cos(c) for c in cands]
        Xrec = cands[int(np.argmin(vals))].copy()
    best_X = Xrec.copy()
    best_val = _max_cos(best_X)

    # contact counts to pick victims
    try:
        G = best_X @ best_X.T
        np.fill_diagonal(G, -2.0)
        m0 = float(G.max())
        tol_c = 0.004
        cnt = (G > (m0 - tol_c)).sum(axis=1)
        order = np.argsort(-cnt)
    except Exception:
        order = np.arange(n)
    victims = list(order[:10])

    # ---- Phase B: deletion -> 58pt quick polish ----
    best58 = None
    best58_val = 1e9
    for vi, idx in enumerate(victims):
        if time.time() > inner_dead - 55:
            break
        Y58 = np.delete(best_X, idx, axis=0)
        # small shake to leave basin
        Y58 = _norm(Y58 + 0.004 * rng.standard_normal(Y58.shape))
        y, v = _adam_lse(Y58, steps=1800, beta0=60.0, beta1=450.0,
                         lr0=0.012, lr1=0.0015, deadline=inner_dead)
        # light greedy
        if time.time() < inner_dead:
            y2, v2 = _greedy(y, steps=400, lr0=0.004, tol=0.006, deadline=inner_dead)
            if v2 < v:
                y, v = y2, v2
        if v < best58_val:
            best58_val = v
            best58 = y.copy()

    # thorough polish of best58
    if best58 is not None and time.time() < inner_dead - 40:
        y, v = _adam_lse(best58, steps=5000, beta0=50.0, beta1=600.0,
                         lr0=0.010, lr1=0.001, deadline=inner_dead)
        if v < best58_val:
            best58_val = v
            best58 = y.copy()
        if time.time() < inner_dead - 35:
            y2, v2 = _greedy(best58, steps=800, lr0=0.003, tol=0.006, deadline=inner_dead)
            if v2 < best58_val:
                best58_val = v2
                best58 = y2.copy()

    # ---- Phase D: split back to 59 ----
    if best58 is not None:
        # find hole direction for best58
        try:
            hole, hole_val = _find_hole(best58, rng, nsamp=8000)
        except Exception:
            hole = None
        # choose k candidates: closest to hole + a few random
        ks = []
        if hole is not None:
            dots = best58 @ hole
            ks.append(int(np.argmax(dots)))
            # two more neighbours of hole
            o = np.argsort(-dots)
            ks.extend([int(o[1]), int(o[2])])
        ks.extend([int(rng.integers(0, 58)) for _ in range(2)])
        # dedup
        seen = set(); klist = []
        for k in ks:
            if k not in seen:
                seen.add(k); klist.append(k)
        eps_list = [0.04, 0.08, 0.14]
        # time-box splits: try up to ~9 variants
        variants = []
        for k in klist[:3]:
            for eps in eps_list:
                variants.append((k, eps))
        # limit by time: each ~3000 steps
        for (k, eps) in variants:
            if time.time() > inner_dead - 28:
                break
            xk = best58[k].copy()
            # tangent from hole if available else random
            if hole is not None:
                tt = hole - float(hole @ xk) * xk
                nn = float(np.linalg.norm(tt))
                if nn < 1e-9:
                    tt = rng.standard_normal(d); tt -= float(tt @ xk) * xk
                    nn = float(np.linalg.norm(tt))
                tt /= max(nn, 1e-12)
            else:
                tt = rng.standard_normal(d); tt -= float(tt @ xk) * xk
                tt /= max(float(np.linalg.norm(tt)), 1e-12)
            # random tangent alternative for second half of variants
            if rng.random() < 0.4:
                tr = rng.standard_normal(d); tr -= float(tr @ xk) * xk
                tr /= max(float(np.linalg.norm(tr)), 1e-12)
                tt = _norm((tt + tr).reshape(1, -1)).ravel()
                tt -= float(tt @ xk) * xk
                tt /= max(float(np.linalg.norm(tt)), 1e-12)
            xp = _norm((xk + eps * tt).reshape(1, -1)).ravel()
            xm = _norm((xk - eps * tt).reshape(1, -1)).ravel()
            Y59 = np.zeros((59, 22))
            # build: keep all 58, replace k with xp, append xm
            # to keep size 59: copy best58 then overwrite k and append
            tmp = best58.copy()
            tmp[k] = xp
            Y59[:58] = tmp
            Y59[58] = xm
            Y59 = _norm(Y59 + 0.002 * rng.standard_normal(Y59.shape))
            # staged annealing: low beta to separate, then high beta
            y, v = _adam_lse(Y59, steps=1500, beta0=40.0, beta1=250.0,
                             lr0=0.012, lr1=0.002, deadline=inner_dead)
            if time.time() > inner_dead:
                # evaluate anyway
                vv = _max_cos(y)
                if vv < best_val:
                    best_val = vv; best_X = y.copy()
                break
            y, v = _adam_lse(y, steps=1500, beta0=250.0, beta1=900.0,
                             lr0=0.006, lr1=0.001, deadline=inner_dead)
            if v < best_val - 1e-12:
                best_val = v
                best_X = y.copy()
            elif v < best_val + 0.002:
                # near-miss: extra greedy squeeze might win
                y2, v2 = _greedy(y, steps=600, lr0=0.002, tol=0.005, deadline=inner_dead)
                if v2 < best_val:
                    best_val = v2; best_X = y2.copy()

    # ---- Phase E: dimensional lift 22->23 ----
    if time.time() < inner_dead - 15:
        for rep in range(2):
            if time.time() > inner_dead - 12:
                break
            sig = 0.06 if rep == 0 else 0.12
            base = best_X if rep == 0 else Xrec
            E = sig * rng.standard_normal((59, 1))
            X23 = np.concatenate([base, E], axis=1)
            X23 = _norm(X23)
            # optimise in 23D with extra freedom
            y23, v23 = _adam_lse(X23, steps=2500, beta0=60.0, beta1=500.0,
                                 lr0=0.010, lr1=0.0012, deadline=inner_dead)
            # penalise last coordinate gradually: scale it down and renormalise row-wise in 23D then drop
            for alpha in [0.5, 0.2, 0.0]:
                y23[:, 22] *= alpha
                rn = np.linalg.norm(y23, axis=1, keepdims=True)
                rn = np.maximum(rn, 1e-12)
                y23 /= rn
                if alpha > 0:
                    y23, v23 = _adam_lse(y23, steps=600, beta0=200.0, beta1=600.0,
                                         lr0=0.005, lr1=0.001, deadline=inner_dead)
            Y22 = _norm(y23[:, :22].copy())
            # repolish in 22D
            y, v = _adam_lse(Y22, steps=1500, beta0=200.0, beta1=900.0,
                             lr0=0.006, lr1=0.001, deadline=inner_dead)
            if v < best_val:
                best_val = v; best_X = y.copy()

    # ---- final squeeze of incumbent ----
    try:
        if time.time() < deadline - 6:
            y, v = _adam_lse(best_X, steps=2000, beta0=400.0, beta1=1500.0,
                             lr0=0.004, lr1=0.0006, deadline=deadline - 2)
            if v < best_val:
                best_val = v; best_X = y.copy()
        if time.time() < deadline - 2:
            y2, v2 = _greedy(best_X, steps=800, lr0=0.0015, tol=0.004, deadline=deadline - 0.5)
            if v2 < best_val:
                best_val = v2; best_X = y2.copy()
    except Exception:
        pass

    best_X = _norm(best_X)
    best_X = np.nan_to_num(best_X, nan=0.0, posinf=0.0, neginf=0.0)
    nr = np.linalg.norm(best_X, axis=1, keepdims=True)
    nr = np.maximum(nr, 1e-12)
    best_X = best_X / nr
    return best_X
