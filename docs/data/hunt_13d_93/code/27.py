import time
import numpy as np

def _normalize(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return X / n

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _p_run(X0, ps, steps_list, lrs, shift=0.15):
    X = _normalize(X0.copy())
    n, d = X.shape
    # initial max
    G0 = X @ X.T
    np.fill_diagonal(G0, -2.0)
    best_val = float(G0.max())
    best_X = X.copy()
    b1, b2, eps = 0.9, 0.999, 1e-8
    for si, (p, steps, lr) in enumerate(zip(ps, steps_list, lrs)):
        m = np.zeros_like(X)
        v = np.zeros_like(X)
        t = 0
        invp = 1.0 / p
        for s in range(steps):
            G = X @ X.T
            # S = max(G-shift,0), diag 0
            # in-place for speed
            S = G - shift
            # diag of G is 1 -> S diag = 1-shift >0, must zero
            np.fill_diagonal(S, 0.0)
            # clip negatives
            # np.maximum with out
            np.maximum(S, 0.0, out=S)
            smax = float(S.max())
            if not np.isfinite(smax) or smax <= 1e-300:
                break
            # R = S/smax
            # compute Rp1 = R^(p-1) where R>0
            # use divide in place
            S /= smax  # now R
            # mask positive
            # to avoid pow on zeros, use where
            # R in S
            with np.errstate(over='ignore', under='ignore'):
                Rp1 = np.power(S, p - 1.0, out=np.zeros_like(S), where=(S > 1e-300))
            Rp = Rp1 * S  # R^p
            # sum over i<j : triu sum (matrix symmetric, diag 0, so half of full sum)
            full = float(Rp.sum()) * 0.5
            if not np.isfinite(full) or full <= 0:
                break
            den = full ** (1.0 - invp)
            if not np.isfinite(den) or den <= 0:
                break
            A = Rp1 * (1.0 / den)
            grad = A @ X
            dot = np.einsum('ij,ij->i', grad, X)[:, None]
            gt = grad - dot * X
            t += 1
            m = b1 * m + (1.0 - b1) * gt
            v = b2 * v + (1.0 - b2) * (gt * gt)
            mh = m / (1.0 - b1 ** t)
            vh = v / (1.0 - b2 ** t)
            upd = lr * mh / (np.sqrt(vh) + eps)
            # tangent projection of update
            d2 = np.einsum('ij,ij->i', upd, X)[:, None]
            upd = upd - d2 * X
            X = X - upd
            X = _normalize(X)
            # re-project momentum to new tangent
            dm = np.einsum('ij,ij->i', m, X)[:, None]
            m = m - dm * X
            if (s % 10 == 0) or (s == steps - 1):
                # current max = smax+shift (smax from before scaling)
                cur = smax + shift
                # for safety recompute exactly every 50 steps (drift due to in-place? no)
                if s % 50 == 0:
                    cur = _max_cos(X)
                if cur < best_val:
                    best_val = cur
                    best_X = X.copy()
        # continue to next p from current X (momentum reset)
    fv = _max_cos(X)
    if fv < best_val:
        best_val = fv
        best_X = X.copy()
    return best_X, best_val

def _smooth_adam(X, steps, T0, T1, lr0, lr1):
    n, d = X.shape
    m = np.zeros_like(X)
    v = np.zeros_like(X)
    b1, b2, eps = 0.9, 0.999, 1e-8
    lT = np.log(T1 / T0) if T0 != T1 and T0 > 0 and T1 > 0 else 0.0
    lL = np.log(lr1 / lr0) if lr0 != lr1 and lr0 > 0 and lr1 > 0 else 0.0
    t = 0
    for s in range(steps):
        f = s / max(1, steps - 1)
        T = T0 * np.exp(lT * f)
        lr = lr0 * np.exp(lL * f)
        if T < 1e-12:
            T = 1e-12
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        mg = float(G.max())
        W = np.exp((G - mg) / T)
        np.fill_diagonal(W, 0.0)
        grad = W @ X
        dot = np.einsum('ij,ij->i', grad, X)[:, None]
        gt = grad - dot * X
        t += 1
        m = b1 * m + (1.0 - b1) * gt
        v = b2 * v + (1.0 - b2) * (gt * gt)
        mh = m / (1.0 - b1 ** t)
        vh = v / (1.0 - b2 ** t)
        upd = lr * mh / (np.sqrt(vh) + eps)
        d2 = np.einsum('ij,ij->i', upd, X)[:, None]
        upd = upd - d2 * X
        X = X - upd
        X = _normalize(X)
        dm = np.einsum('ij,ij->i', m, X)[:, None]
        m = m - dm * X
    return X

def _hard_push(X, iters, margin=0.0015, step0=0.003):
    cur = _max_cos(X)
    step = step0
    for _ in range(iters):
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        cmax = float(G.max())
        mask = G > (cmax - margin)
        grad = mask.astype(np.float64) @ X
        dot = np.einsum('ij,ij->i', grad, X)[:, None]
        gt = grad - dot * X
        rn = np.linalg.norm(gt, axis=1, keepdims=True)
        nz = (rn[:, 0] > 1e-12)
        if not np.any(nz):
            break
        direc = np.zeros_like(gt)
        direc[nz] = gt[nz] / rn[nz]
        Xt = _normalize(X - step * direc)
        vtry = _max_cos(Xt)
        if vtry < cmax:
            X = Xt
            cur = vtry
            step = min(step * 1.25, 0.006)
        else:
            step *= 0.5
            if step < 1e-7:
                break
    return X

def _micro_greedy(X, seconds, rng, t0, budget):
    n, d = X.shape
    best = _max_cos(X)
    step = 2e-4
    it = 0
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    while time.time() - t0 < budget:
        it += 1
        if it % 60 == 1:
            G = X @ X.T
            np.fill_diagonal(G, -2.0)
            per = G.max(axis=1)
            order = np.argsort(-per)
            k = int(rng.integers(0, min(25, n)))
            i = int(order[k])
        else:
            i = int(rng.integers(0, n))
        r = rng.standard_normal(d)
        r -= float(r @ X[i]) * X[i]
        rn = float(np.linalg.norm(r))
        if rn < 1e-12:
            continue
        r /= rn
        improved = False
        for sgn in (1.0, -1.0):
            v = X[i] + sgn * step * r
            nv = float(np.linalg.norm(v))
            if nv < 1e-12:
                continue
            v /= nv
            y = X.copy()
            y[i] = v
            # full max (93 dim cheap)
            D = y @ y.T
            np.fill_diagonal(D, -2.0)
            vv = float(D.max())
            if vv < best:
                X = y
                best = vv
                step = min(step * 1.12, 5e-4)
                improved = True
                break
        if not improved and it % 30 == 0:
            step *= 0.92
            if step < 2e-6:
                step = 2e-6
        if it > 30000:
            break
        if time.time() - t0 > budget - 0.3:
            break
    return X

def solve():
    t0 = time.time()
    BUDGET = 172.0
    rng = np.random.default_rng(0)
    n, d = 93, 13
    best_x = None
    best_val = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn).astype(np.float64)
            if a.shape == (93, 13):
                a = _normalize(a)
                va = _max_cos(a)
                if va < best_val:
                    best_val = va
                    best_x = a.copy()
        except Exception:
            pass
    if best_x is None:
        best_x = _normalize(rng.standard_normal((n, d)))
        best_val = _max_cos(best_x)
    try:
        rec = _normalize(np.load("record.npy").astype(np.float64))
    except Exception:
        rec = best_x.copy()

    ps = [8, 16, 32, 64, 128, 256]
    steps_list = [350, 350, 350, 300, 300, 250]
    lrs = [3e-3, 2.2e-3, 1.6e-3, 1.0e-3, 6e-4, 3e-4]

    # build initial pool
    inits = []
    inits.append(("best", best_x.copy()))
    inits.append(("record", rec.copy()))
    # perturbed / random
    sigmas = [0.15, 0.30, 0.60, 0.15, 0.30]
    bases = [best_x, best_x, best_x, rec, rec]
    for sig, b in zip(sigmas, bases):
        nz = rng.standard_normal((n, d))
        inits.append((f"pert{sig}", _normalize(b + sig * nz)))
    # two pure random
    for _ in range(2):
        inits.append(("rand", _normalize(rng.standard_normal((n, d)))))

    cands = []
    for name, X0 in inits:
        if time.time() - t0 > 110:
            break
        bx, bv = _p_run(X0, ps, steps_list, lrs, shift=0.15)
        cands.append((bv, bx))
        if bv < best_val and np.all(np.isfinite(bx)):
            best_val = bv
            best_x = bx.copy()
        # quick light polish of this basin if promising (within 0.003)
        if bv < best_val + 0.003 and (time.time() - t0 < 125):
            xp = _smooth_adam(bx.copy(), 300, 0.006, 0.0005, 0.003, 0.0006)
            xp = _hard_push(xp, 150, margin=0.0015, step0=0.002)
            vp = _max_cos(xp)
            cands.append((vp, xp))
            if vp < best_val and np.all(np.isfinite(xp)):
                best_val = vp
                best_x = xp.copy()

    # promote best 3 to hard polish
    cands.sort(key=lambda z: z[0])
    for k in range(min(3, len(cands))):
        if time.time() - t0 > 145:
            break
        bv, bx = cands[k]
        # skip if far
        if bv > best_val + 0.002:
            continue
        xp = _smooth_adam(bx.copy(), 700, 0.005, 0.00025, 0.004, 0.0004)
        xp = _hard_push(xp, 300, margin=0.0012, step0=0.003)
        xp = _smooth_adam(xp, 400, 0.0015, 0.0002, 0.0015, 0.0003)
        xp = _hard_push(xp, 300, margin=0.0008, step0=0.0015)
        vp = _max_cos(xp)
        if vp < best_val and np.all(np.isfinite(xp)):
            best_val = vp
            best_x = xp.copy()

    # extra hops from current best if time remains
    extra_sigmas = [0.06, 0.12, 0.22, 0.35, 0.10, 0.18]
    ei = 0
    while time.time() - t0 < 150 and ei < 12:
        sig = extra_sigmas[ei % len(extra_sigmas)]
        ei += 1
        X0 = _normalize(best_x + sig * rng.standard_normal((n, d)))
        # shorter schedule for speed: fewer steps per stage
        short_steps = [250, 250, 250, 220, 220, 180]
        bx, bv = _p_run(X0, ps, short_steps, lrs, shift=0.15)
        if bv < best_val + 0.0015:
            xp = _smooth_adam(bx.copy(), 350, 0.004, 0.0003, 0.003, 0.0005)
            xp = _hard_push(xp, 180, margin=0.0012, step0=0.002)
            vp = _max_cos(xp)
            if vp < bv and np.all(np.isfinite(xp)):
                bx, bv = xp, vp
        if bv < best_val and np.all(np.isfinite(bx)):
            best_val = bv
            best_x = bx.copy()
            cands.append((bv, bx))

    # final intense polish on global best
    remain = BUDGET - (time.time() - t0) - 0.6
    if remain > 3:
        # allocate: smooth+hard then micro
        try:
            best_x = _smooth_adam(best_x, 600, 0.003, 0.0002, 0.002, 0.0003)
            best_x = _hard_push(best_x, 350, margin=0.0010, step0=0.002)
            best_val = _max_cos(best_x)
        except Exception:
            pass
        remain2 = BUDGET - (time.time() - t0) - 0.5
        if remain2 > 1.5:
            try:
                best_x = _micro_greedy(best_x, remain2, rng, time.time(), time.time() + min(remain2, 6.0))
                best_x = _normalize(best_x)
            except Exception:
                pass

    best_x = _normalize(best_x)
    # safety: remove NaN
    best_x = np.where(np.isfinite(best_x), best_x, 0.0)
    # if any zero row, replace random
    nr = np.linalg.norm(best_x, axis=1)
    for i in range(n):
        if nr[i] < 1e-12 or not np.isfinite(nr[i]):
            v = rng.standard_normal(d)
            v /= np.linalg.norm(v)
            best_x[i] = v
    best_x = _normalize(best_x)
    return best_x
