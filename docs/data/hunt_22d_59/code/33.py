import time
import numpy as np

def _max_cos(Y):
    G = Y @ Y.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _norm(A):
    n = np.linalg.norm(A, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return A / n

def _adam(Y0, steps, beta0, beta1, lr0, lr1):
    Y = Y0.copy()
    n, d = Y.shape
    M = np.zeros_like(Y)
    V = np.zeros_like(Y)
    b1 = 0.9
    b2 = 0.999
    eps = 1e-8
    G0 = Y @ Y.T
    np.fill_diagonal(G0, -10.0)
    best_val = float(G0.max())
    best_Y = Y.copy()
    if steps <= 1:
        return best_Y, best_val
    lbb0 = np.log(beta0); lbb1 = np.log(beta1)
    llr0 = np.log(lr0); llr1 = np.log(lr1)
    b1t = 1.0; b2t = 1.0
    for t in range(1, steps + 1):
        f = (t - 1) / float(steps - 1)
        beta = float(np.exp(lbb0 + (lbb1 - lbb0) * f))
        lr = float(np.exp(llr0 + (llr1 - llr0) * f))
        G = Y @ Y.T
        np.fill_diagonal(G, -10.0)
        gmax = float(G.max())
        if gmax < best_val:
            best_val = gmax
            best_Y = Y.copy()
        G -= gmax
        G *= beta
        np.exp(G, out=G)
        np.fill_diagonal(G, 0.0)
        s = float(G.sum())
        if not np.isfinite(s) or s <= 0:
            continue
        G /= s
        Gr = G @ Y
        dot = np.sum(Gr * Y, axis=1, keepdims=True)
        Gr -= Y * dot
        b1t *= b1; b2t *= b2
        M *= b1; M += (1.0 - b1) * Gr
        V *= b2; V += (1.0 - b2) * (Gr * Gr)
        Mh = M / (1.0 - b1t)
        Vh = V / (1.0 - b2t)
        Y -= lr * Mh / (np.sqrt(Vh) + eps)
        rn = np.linalg.norm(Y, axis=1, keepdims=True)
        rn[rn == 0] = 1.0
        Y /= rn
    Gf = Y @ Y.T
    np.fill_diagonal(Gf, -10.0)
    vf = float(Gf.max())
    if vf < best_val:
        best_val = vf
        best_Y = Y.copy()
    return best_Y, best_val

def _contact_counts(Y, tol=0.002):
    n = Y.shape[0]
    G = Y @ Y.T
    np.fill_diagonal(G, -2.0)
    m = float(G.max())
    rowmax = G.max(axis=1)
    cnt = (G > (m - tol)).sum(axis=1)
    return m, rowmax, cnt, G

def _find_hole(Y58, rng, n_batch=20000, n_top=4, refinesteps=250):
    d = Y58.shape[1]
    C = rng.standard_normal((n_batch, d))
    C = _norm(C)
    S = C @ Y58.T
    rm = S.max(axis=1)
    idx = np.argpartition(rm, n_top)[:n_top]
    best_c = None
    best_v = 1e9
    for k in idx:
        c = C[k].copy()
        beta0, beta1 = 30.0, 300.0
        lr0, lr1 = 0.08, 0.005
        lbb0 = np.log(beta0); lbb1 = np.log(beta1)
        llr0 = np.log(lr0); llr1 = np.log(lr1)
        for t in range(refinesteps):
            f = t / float(max(1, refinesteps - 1))
            beta = float(np.exp(lbb0 + (lbb1 - lbb0) * f))
            lr = float(np.exp(llr0 + (llr1 - llr0) * f))
            dv = Y58 @ c
            m = float(dv.max())
            w = np.exp(beta * (dv - m))
            sw = float(w.sum())
            if sw > 0:
                w /= sw
            grad = w @ Y58
            gp = grad - float(grad @ c) * c
            c = c - lr * gp
            nn = float(np.linalg.norm(c))
            if nn == 0:
                c = C[k]
            else:
                c /= nn
        dv = Y58 @ c
        v = float(dv.max())
        if v < best_v:
            best_v = v
            best_c = c.copy()
    return best_c, best_v

def solve():
    rng = np.random.default_rng(0)
    t0 = time.time()
    deadline = t0 + 148.0
    def left(slack=0.0):
        return time.time() < deadline - slack

    x_best = None
    best_val = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=np.float64)
            if a.shape != (59, 22):
                continue
            if not np.all(np.isfinite(a)):
                continue
            a = _norm(a)
            v = _max_cos(a)
            if v < best_val:
                best_val = v
                x_best = a.copy()
        except Exception:
            continue
    if x_best is None:
        x_best = _norm(rng.standard_normal((59, 22)))
        best_val = _max_cos(x_best)

    def consider(Y, v):
        nonlocal x_best, best_val
        if v < best_val and Y.shape == (59, 22) and np.all(np.isfinite(Y)):
            best_val = v
            x_best = _norm(Y.copy())

    # quick direct polish of incumbent (short, to have tight baseline)
    if left(120.0):
        y, v = _adam(x_best, steps=2500, beta0=60.0, beta1=500.0, lr0=0.006, lr1=0.0008)
        consider(y, v)

    # ---- 58-point deletion -> hole insertion branch ----
    # try two different deletions (highest and second-highest load)
    try:
        m0, rm0, cnt0, _ = _contact_counts(x_best, tol=0.002)
        order = np.lexsort((rm0, cnt0))  # ascending; worst at end
        cand_del = [int(order[-1]), int(order[-2]) if len(order) > 1 else int(order[-1])]
    except Exception:
        cand_del = [0, 1]
    for di, del_idx in enumerate(cand_del):
        if not left(90.0):
            break
        try:
            Y58 = np.delete(x_best, del_idx, axis=0)
            # small shake then polish easier 58 geometry
            Y58 = _norm(Y58 + 0.005 * rng.standard_normal(Y58.shape))
            y58, v58 = _adam(Y58, steps=2200, beta0=40.0, beta1=400.0, lr0=0.006, lr1=0.0008)
            if not left(80.0):
                continue
            cstar, _ = _find_hole(y58, rng, n_batch=16000, n_top=4, refinesteps=220)
            Y59 = np.vstack([y58, cstar.reshape(1, -1)])
            Y59 = _norm(Y59)
            y59, v59 = _adam(Y59, steps=2500 if di == 0 else 2000, beta0=80.0, beta1=600.0, lr0=0.006, lr1=0.0008)
            consider(y59, v59)
            # one extra joint polish if promising (within 0.002 of best)
            if v59 < best_val + 0.002 and left(60.0):
                y59b, v59b = _adam(y59, steps=1500, beta0=200.0, beta1=900.0, lr0=0.003, lr1=0.0004)
                consider(y59b, v59b)
        except Exception:
            continue

    # ---- overcomplete 60-point branch ----
    for trial in range(2):
        if not left(55.0):
            break
        try:
            sigma = 0.015 if trial == 0 else 0.035
            extra = rng.standard_normal((1, 22))
            Y60 = np.vstack([x_best, _norm(extra)])
            Y60 = _norm(Y60 + sigma * rng.standard_normal(Y60.shape))
            y60, v60 = _adam(Y60, steps=1800, beta0=60.0, beta1=500.0, lr0=0.006, lr1=0.0008)
            # delete worst point down to 59
            m1, rm1, cnt1, _ = _contact_counts(y60, tol=0.002)
            # worst = largest rowmax, tie-break largest count
            worst = int(np.lexsort((cnt1, rm1))[-1])
            Y59 = np.delete(y60, worst, axis=0)
            Y59 = _norm(Y59)
            y59, v59 = _adam(Y59, steps=2200, beta0=80.0, beta1=650.0, lr0=0.005, lr1=0.0006)
            consider(y59, v59)
        except Exception:
            continue

    # ---- overcomplete 61-point branch ----
    if left(35.0):
        try:
            extra2 = _norm(rng.standard_normal((2, 22)))
            Y61 = np.vstack([x_best, extra2])
            Y61 = _norm(Y61 + 0.02 * rng.standard_normal(Y61.shape))
            y61, v61 = _adam(Y61, steps=1500, beta0=60.0, beta1=450.0, lr0=0.006, lr1=0.0008)
            cur = y61
            for _ in range(2):
                m2, rm2, cnt2, _ = _contact_counts(cur, tol=0.002)
                worst = int(np.lexsort((cnt2, rm2))[-1])
                cur = np.delete(cur, worst, axis=0)
                cur = _norm(cur)
                cur, _ = _adam(cur, steps=1500, beta0=80.0, beta1=600.0, lr0=0.005, lr1=0.0006)
                if not left(15.0):
                    break
            if cur.shape == (59, 22):
                consider(cur, _max_cos(cur))
        except Exception:
            pass

    # ---- final joint polish of best ----
    if left(12.0):
        try:
            y, v = _adam(x_best, steps=2000, beta0=150.0, beta1=1000.0, lr0=0.003, lr1=0.0003)
            consider(y, v)
        except Exception:
            pass

    x_best = _norm(x_best)
    return x_best
