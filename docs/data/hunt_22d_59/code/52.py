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

def _adam_lse(X0, steps, beta0, beta1, lr, rng, jitter=0.0):
    X = X0.copy()
    n, d = X.shape
    m = np.zeros_like(X)
    v = np.zeros_like(X)
    b1 = 0.9; b2 = 0.999; eps = 1e-8
    pow1 = b1; pow2 = b2
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
        # true max for tracking done periodically; use G for gm
        if J is not None:
            Gw = G + J
        else:
            Gw = G
        gm = Gw.max()
        W = np.exp(beta * (Gw - gm))
        np.fill_diagonal(W, 0.0)
        rs = W.sum(axis=1, keepdims=True) + 1e-16
        D = (W @ X) / rs
        dot = (D * X).sum(axis=1, keepdims=True)
        Gt = D - dot * X
        m *= b1; m += (1.0 - b1) * Gt
        v *= b2; v += (1.0 - b2) * (Gt * Gt)
        mhat = m / (1.0 - pow1)
        vhat = v / (1.0 - pow2)
        pow1 *= b1; pow2 *= b2
        step = lr * mhat / (np.sqrt(vhat) + eps)
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

def _greedy_polish(X0, iters, step0, step1):
    Xcur = X0.copy()
    best = _max_cos(Xcur)
    bestX = Xcur.copy()
    decay = (step1 / step0) ** (1.0 / max(iters - 1, 1))
    step = step0
    n = Xcur.shape[0]
    for _ in range(iters):
        G = Xcur @ Xcur.T
        np.fill_diagonal(G, -2.0)
        mm = G.max()
        delta = max(0.004, 1.5 * step)
        thr = mm - delta
        W = np.where(G > thr, (G - thr + 0.01), 0.0)
        np.fill_diagonal(W, 0.0)
        Gr = W @ Xcur
        dot = (Gr * Xcur).sum(axis=1, keepdims=True)
        Gt = Gr - dot * Xcur
        nrm = np.linalg.norm(Gt, axis=1, keepdims=True) + 1e-12
        upd = Gt / nrm
        Xcur = Xcur - step * upd
        Xcur = _normalize(Xcur)
        cur = _max_cos(Xcur)
        if cur < best:
            best = cur
            bestX = Xcur.copy()
        step *= decay
    return bestX, best

def _hole_vector(Y, rng, M=4000):
    # find unit vector minimizing max inner product to Y
    m = Y.shape[0]
    d = Y.shape[1]
    V = rng.standard_normal((M, d))
    V = _normalize(V)
    S = V @ Y.T
    scores = S.max(axis=1)
    # pick among top-5 for diversity
    idx = np.argpartition(scores, 5)[:5]
    pick = idx[int(rng.integers(5))] if M >= 5 else int(np.argmin(scores))
    best_v = V[pick].copy()
    best_s = float(scores[pick])
    # local hill climb on the sphere to deepen hole
    step = 0.05
    for _ in range(60):
        Gv = Y @ best_v  # (m,)
        mm = Gv.max()
        # softmax weights toward worst
        beta = 80.0
        w = np.exp(beta * (Gv - mm))
        w = w / (w.sum() + 1e-16)
        grad = w @ Y  # (d,)
        grad = grad - (grad @ best_v) * best_v
        gn = np.linalg.norm(grad) + 1e-12
        if gn < 1e-12:
            break
        trial = best_v - step * (grad / gn)
        trial = trial / (np.linalg.norm(trial) + 1e-12)
        s = float((Y @ trial).max())
        if s < best_s:
            best_s = s
            best_v = trial
            step = min(step * 1.1, 0.05)
        else:
            step *= 0.5
            if step < 1e-5:
                break
    return best_v

def solve():
    t0 = time.time()
    t_end = t0 + 168.0
    rng = np.random.default_rng(0)
    n, d = 59, 22
    def rand_X():
        return _normalize(rng.standard_normal((n, d)))
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
            if np.all(np.isfinite(a)):
                cands.append(a)
        except Exception:
            pass
    if not cands:
        cands.append(rand_X())
    vals = [_max_cos(c) for c in cands]
    gbestX = cands[int(np.argmin(vals))].copy()
    gbest = float(min(vals))

    # initial squeeze of loaded best
    try:
        X1, v1 = _adam_lse(gbestX, 2000, 40.0, 250.0, 0.012, rng, 0.0)
        if v1 < gbest:
            gbest, gbestX = v1, X1.copy()
        X2, v2 = _adam_lse(X1 if v1 < gbest + 1 else gbestX, 1000, 150.0, 320.0, 0.007, rng, 0.0)
        if v2 < gbest:
            gbest, gbestX = v2, X2.copy()
        X3, v3 = _greedy_polish(gbestX, 800, 0.004, 0.0002)
        if v3 < gbest:
            gbest, gbestX = v3, X3.copy()
    except Exception:
        pass

    it = 0
    # precompute contact info updated each outer loop
    while time.time() < t_end - 3.0 and it < 250:
        rem = t_end - time.time()
        if rem < 4.0:
            break
        # refresh contact stats of gbest occasionally
        Gg = gbestX @ gbestX.T
        np.fill_diagonal(Gg, -2.0)
        mmax = float(Gg.max())
        rowmax = Gg.max(axis=1)
        contacts = (Gg > mmax - 1e-9).sum(axis=1)
        order_worst = np.argsort(rowmax)  # ascending, worst last
        order_cnt = np.argsort(contacts)

        mode = it % 6
        X_trial = None
        try:
            if mode in (0, 1, 3):
                # kill-and-fill: k=1,2,3
                k = {0: 1, 1: 2, 3: 3}[mode]
                # choose victims: mostly worst, sometimes random among top-12
                if rng.random() < 0.7:
                    victims = order_worst[-k:].tolist()
                else:
                    pool = order_worst[-12:].tolist()
                    victims = rng.choice(pool, size=k, replace=False).tolist()
                victims = sorted(victims)
                Ykeep = np.delete(gbestX, victims, axis=0)
                # sequentially fill holes
                Ycur = Ykeep.copy()
                for _ in range(k):
                    hv = _hole_vector(Ycur, rng, M=4000)
                    Ycur = np.vstack([Ycur, hv[None, :]])
                X_trial = _normalize(Ycur)
            elif mode in (2, 4):
                # overloaded cluster shake
                # center weighted by contacts
                probs = (contacts.astype(float) + 1.0)
                probs = probs / probs.sum()
                c = int(rng.choice(n, p=probs))
                grow = Gg[c].copy()
                neigh = np.argsort(-grow)
                Cn = int(rng.integers(8, 14))  # cluster size
                cluster = neigh[:Cn]
                amp_c = 0.45 + 0.15 * rng.random()
                amp_r = 0.015
                Noise = rng.standard_normal((n, d))
                X_trial = gbestX.copy()
                X_trial[cluster] = X_trial[cluster] + amp_c * Noise[cluster]
                mask = np.ones(n, dtype=bool); mask[cluster] = False
                X_trial[mask] = X_trial[mask] + amp_r * Noise[mask]
                X_trial = _normalize(X_trial)
            else:  # mode 5: single-point jump to hole (create rattler then rejam)
                # pick most overloaded point and teleport to deepest hole
                c = int(order_cnt[-1]) if rng.random() < 0.8 else int(rng.integers(n))
                Ykeep = np.delete(gbestX, [c], axis=0)
                hv = _hole_vector(Ykeep, rng, M=6000)
                X_trial = np.vstack([Ykeep, hv[None, :]])
                # shuffle row order to avoid bias (insert at random position)
                perm = rng.permutation(n)
                # need to map: X_trial currently Ykeep order + new at end; just permute
                # build full by permuting indices: simpler random permute rows
                X_trial = X_trial[rng.permutation(X_trial.shape[0])] if False else X_trial
                # actually keep shape (59,22): insert hv at position c
                Xf = gbestX.copy()
                Xf[c] = hv
                X_trial = _normalize(Xf)
        except Exception:
            it += 1
            continue

        # fast trial polish: jittered then clean
        try:
            small = rem < 15
            s1 = 600 if small else 900
            s2 = 400 if small else 700
            jit = 0.004 if (it % 2 == 0) else 0.0
            Xa, va = _adam_lse(X_trial, s1, 35.0, 220.0, 0.013, rng, jitter=jit)
            # track true best inside already; va is true
            Xb, vb = _adam_lse(Xa, s2, 150.0, 300.0, 0.007, rng, jitter=0.0)
            best_trial, bestXt = (vb, Xb) if vb < va else (va, Xa)
            if best_trial < gbest:
                # deep squeeze
                Xc, vc = _adam_lse(bestXt, 1200, 80.0, 350.0, 0.006, rng, 0.0)
                if vc < best_trial:
                    best_trial, bestXt = vc, Xc
                Xd, vd = _greedy_polish(bestXt, 700, 0.003, 0.00015)
                if vd < best_trial:
                    best_trial, bestXt = vd, Xd
                if best_trial < gbest:
                    gbest = best_trial
                    gbestX = bestXt.copy()
        except Exception:
            pass
        it += 1
        # occasional fresh random to avoid stagnation (every 25)
        if it % 25 == 0 and time.time() < t_end - 20:
            try:
                Xr = rand_X()
                Xt, vt = _adam_lse(Xr, 1200, 35.0, 220.0, 0.013, rng, 0.0)
                if vt < gbest:
                    Xd2, vd2 = _greedy_polish(Xt, 500, 0.003, 0.0002)
                    if vd2 < gbest:
                        gbest, gbestX = vd2, Xd2.copy()
                    elif vt < gbest:
                        gbest, gbestX = vt, Xt.copy()
            except Exception:
                pass

    # final ultra-fine polish with remaining time
    try:
        rem = t_end - time.time()
        if rem > 1.0:
            Xf, vf = _adam_lse(gbestX, 1200 if rem > 8 else 500, 120.0, 380.0, 0.005, rng, 0.0)
            if vf < gbest:
                gbest, gbestX = vf, Xf.copy()
            rem2 = t_end - time.time()
            if rem2 > 0.6:
                Xg, vg = _greedy_polish(gbestX, 600 if rem2 > 5 else 200, 0.0015, 0.00005)
                if vg < gbest:
                    gbest, gbestX = vg, Xg.copy()
    except Exception:
        pass
    gbestX = _normalize(gbestX)
    return gbestX
