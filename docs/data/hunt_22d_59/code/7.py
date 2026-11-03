import time
import numpy as np

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def _rowmax(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return np.max(G, axis=1), float(np.max(G))

def _adam_lse(Y0, steps, b0, b1, lr0, lr1, t0, deadline):
    Y = Y0.copy()
    n, d = Y.shape
    m = np.zeros_like(Y)
    v = np.zeros_like(Y)
    b1a = 0.9
    b2a = 0.999
    eps = 1e-8
    lb0 = np.log(b0)
    lb1 = np.log(b1)
    ll0 = np.log(lr0)
    ll1 = np.log(lr1)
    cur_val = _max_cos(Y)
    best = Y.copy()
    best_val = cur_val
    b1p = 1.0
    b2p = 1.0
    denom = max(steps - 1, 1)
    for t in range(1, steps + 1):
        if (t & 255) == 0 and time.time() > deadline:
            break
        frac = (t - 1) / denom
        beta = float(np.exp(lb0 + (lb1 - lb0) * frac))
        lr = float(np.exp(ll0 + (ll1 - ll0) * frac))
        G = Y @ Y.T
        np.fill_diagonal(G, -10.0)
        gmax = float(np.max(G))
        if gmax < best_val:
            best_val = gmax
            best = Y.copy()
        G -= gmax
        G *= beta
        np.exp(G, out=G)
        np.fill_diagonal(G, 0.0)
        s = float(np.sum(G))
        if not np.isfinite(s) or s <= 0:
            continue
        G /= s
        Gr = G @ Y
        dot = np.sum(Gr * Y, axis=1, keepdims=True)
        Gr -= dot * Y
        b1p *= b1a
        b2p *= b2a
        m *= b1a
        m += (1.0 - b1a) * Gr
        v *= b2a
        v += (1.0 - b2a) * (Gr * Gr)
        c1 = 1.0 - b1p
        c2 = 1.0 - b2p
        mh = m / c1
        vh = v / c2
        Y -= lr * mh / (np.sqrt(vh) + eps)
        rn = np.linalg.norm(Y, axis=1, keepdims=True)
        rn[rn == 0] = 1.0
        Y /= rn
    fv = _max_cos(Y)
    if fv < best_val:
        best_val = fv
        best = Y.copy()
    # also check best is normalized
    return best, best_val

def _hole_score(y, Xo):
    return float(np.max(Xo @ y))

def _find_hole(Xo, rng, nsamp=8000):
    d = Xo.shape[1]
    R = rng.standard_normal((nsamp, d))
    R /= np.linalg.norm(R, axis=1, keepdims=True) + 1e-300
    S = R @ Xo.T
    smax = np.max(S, axis=1)
    bi = int(np.argmin(smax))
    return R[bi].copy(), float(smax[bi])

def _refine_hole(Xo, y0, steps=400, beta=300.0, lr=0.02):
    y = y0.copy()
    y /= np.linalg.norm(y) + 1e-300
    m = np.zeros_like(y)
    v = np.zeros_like(y)
    b1a = 0.9; b2a = 0.999; eps = 1e-8
    b1p = 1.0; b2p = 1.0
    best = y.copy()
    best_val = _hole_score(y, Xo)
    for t in range(1, steps + 1):
        dots = Xo @ y
        dm = float(np.max(dots))
        w = np.exp(beta * (dots - dm))
        sw = float(np.sum(w))
        if not np.finite(sw) or sw <= 0:
            break
        w /= sw
        g = w @ Xo
        g = g - np.dot(g, y) * y
        b1p *= b1a; b2p *= b2a
        m = b1a * m + (1 - b1a) * g
        v = b2a * v + (1 - b2a) * (g * g)
        mh = m / (1 - b1p); vh = v / (1 - b2p)
        upd = mh / (np.sqrt(vh) + eps)
        # decay lr a bit
        y = y - lr * upd
        y /= np.linalg.norm(y) + 1e-300
        cv = _hole_score(y, Xo)
        if cv < best_val:
            best_val = cv
            best = y.copy()
    return best, best_val

def solve():
    t0 = time.time()
    BUDGET = 172.0
    deadline = t0 + BUDGET
    rng = np.random.default_rng(0)
    cands = []
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn).astype(np.float64)
            if a.shape == (59, 22) and np.all(np.isfinite(a)):
                a /= np.linalg.norm(a, axis=1, keepdims=True) + 1e-300
                cands.append(a)
        except Exception:
            pass
    if not cands:
        a = rng.standard_normal((59, 22))
        a /= np.linalg.norm(a, axis=1, keepdims=True)
        cands.append(a)
    vals = [_max_cos(c) for c in cands]
    best = cands[int(np.argmin(vals))].copy()
    best_val = float(np.min(vals))
    n, d = best.shape

    # stage 0: solid polish
    if time.time() < deadline - 5:
        best, best_val = _adam_lse(best, steps=7000, b0=60.0, b1=600.0,
                                   lr0=0.006, lr1=0.0006, t0=t0, deadline=deadline)
    if time.time() < deadline - 5:
        best, best_val = _adam_lse(best, steps=2500, b0=800.0, b1=2500.0,
                                   lr0=0.0012, lr1=0.0002, t0=t0, deadline=deadline)

    sigmas = [0.006, 0.015, 0.035, 0.008, 0.06, 0.12]
    it = 0
    while time.time() < deadline - 8:
        it += 1
        remain = deadline - time.time()
        if remain < 6:
            break
        if remain > 70:
            steps = 3800
        elif remain > 30:
            steps = 2800
        else:
            steps = 1800
        mode = it % 6
        Y = None
        if mode in (0, 1, 5):
            sigma = sigmas[it % len(sigmas)]
            rm, gm = _rowmax(best)
            # boost noise on worst points
            order = np.argsort(rm)  # ascending
            scale = np.full((n, 1), sigma)
            # top-8 worst get 2.5x
            worst = order[-8:]
            scale[worst] *= 2.5
            noise = rng.standard_normal((n, d))
            Y = best + scale * noise
            if (it % 7) == 0:
                k = int(rng.integers(n))
                Y[k] = rng.standard_normal(d)
            Y /= np.linalg.norm(Y, axis=1, keepdims=True) + 1e-300
            if sigma >= 0.03:
                a0, a1, l0, l1 = 60.0, 700.0, 0.005, 0.0004
            else:
                a0, a1, l0, l1 = 150.0, 1200.0, 0.003, 0.00025
        elif mode == 2 or mode == 3:
            # single hole reinsertion of a worst point
            rm, gm = _rowmax(best)
            # pick among 8 worst randomly for diversity
            order = np.argsort(rm)
            cand_idx = order[-8:]
            k = int(cand_idx[rng.integers(len(cand_idx))])
            Xo = np.delete(best, k, axis=0)
            y0, _ = _find_hole(Xo, rng, nsamp=6000)
            yh, _ = _refine_hole(Xo, y0, steps=300, beta=250.0, lr=0.03)
            Y = best.copy()
            # small jitter to others to break symmetry
            Y += 0.002 * rng.standard_normal((n, d))
            Y[k] = yh + 0.001 * rng.standard_normal(d)
            Y /= np.linalg.norm(Y, axis=1, keepdims=True) + 1e-300
            a0, a1, l0, l1 = 120.0, 1000.0, 0.004, 0.0003
        else:
            # mode 4: break the tightest pair apart + double hole
            G = best @ best.T
            np.fill_diagonal(G, -2.0)
            ij = int(np.argmax(G))
            a = ij // n; b = ij % n
            Y = best.copy() + 0.004 * rng.standard_normal((n, d))
            # push a,b apart
            va = best[a] - best[b]
            vb = -va
            Y[a] += 0.02 * va + 0.005 * rng.standard_normal(d)
            Y[b] += 0.02 * vb + 0.005 * rng.standard_normal(d)
            # also reinsert one random of the pair's neighbours? just renormalize
            Y /= np.linalg.norm(Y, axis=1, keepdims=True) + 1e-300
            a0, a1, l0, l1 = 100.0, 900.0, 0.004, 0.0003
        Yp, Yv = _adam_lse(Y, steps=steps, b0=a0, b1=a1, lr0=l0, lr1=l1,
                           t0=t0, deadline=deadline)
        if Yv < best_val - 1e-12:
            best = Yp
            best_val = Yv
        # opportunistic squeeze of incumbent every few iters
        if (it % 6) == 0 and time.time() < deadline - 10:
            best, best_val = _adam_lse(best, steps=1200, b0=1000.0, b1=3000.0,
                                       lr0=0.0008, lr1=0.00015, t0=t0, deadline=deadline)

    # final squeeze
    if time.time() < deadline - 4:
        best, best_val = _adam_lse(best, steps=3000, b0=900.0, b1=3500.0,
                                   lr0=0.001, lr1=0.00012, t0=t0, deadline=deadline)

    # greedy tiny hill climb on max (exact acceptance)
    # budget ~3 sec
    hc_deadline = deadline - 1.0
    # precompute gram
    G = best @ best.T
    np.fill_diagonal(G, -2.0)
    cur = float(np.max(G))
    # use small sigmas decaying
    for h in range(4000):
        if time.time() > hc_deadline:
            break
        sigma = 0.0006 * (0.9995 ** h) + 1e-5
        if (h % 2) == 0:
            # pick a point involved in max
            # find max pair quickly every 32 steps, else random point
            if (h % 32) == 0:
                G = best @ best.T
                np.fill_diagonal(G, -2.0)
                cur = float(np.max(G))
                ij = int(np.argmax(G))
                k = ij // n if (rng.random() < 0.5) else ij % n
            else:
                k = int(rng.integers(n))
            prop = best[k] + sigma * rng.standard_normal(d)
            # make tangent-ish: remove radial a bit (retract handles)
            prop /= np.linalg.norm(prop) + 1e-300
            # fast eval: max of prop vs others and max of rest
            dots = best @ prop
            dots[k] = -2.0
            nm = float(np.max(dots))
            if nm >= cur:
                continue
            # max of rest (excluding k): need max over pairs not involving k
            # compute quickly: we have G; max excluding row/col k
            # do full recompute only when promising (nm < cur)
            old = best[k].copy()
            best[k] = prop
            G2 = best @ best.T
            np.fill_diagonal(G2, -2.0)
            nv = float(np.max(G2))
            if nv < cur - 1e-14:
                cur = nv
                best_val = nv
                G = G2
            else:
                best[k] = old
        else:
            # try moving max pair apart symmetrically tiny
            if (h % 32) == 1:
                G = best @ best.T
                np.fill_diagonal(G, -2.0)
                cur = float(np.max(G))
            ij = int(np.argmax(G))
            a = ij // n; b = ij % n
            oa = best[a].copy(); ob = best[b].copy()
            best[a] = (best[a] + sigma * rng.standard_normal(d))
            best[a] /= np.linalg.norm(best[a]) + 1e-300
            best[b] = (best[b] + sigma * rng.standard_normal(d))
            best[b] /= np.linalg.norm(best[b]) + 1e-300
            G2 = best @ best.T
            np.fill_diagonal(G2, -2.0)
            nv = float(np.max(G2))
            if nv < cur - 1e-14:
                cur = nv
                best_val = nv
                G = G2
            else:
                best[a] = oa; best[b] = ob
    best /= np.linalg.norm(best, axis=1, keepdims=True) + 1e-300
    return best
