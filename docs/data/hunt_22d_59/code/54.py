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

def solve():
    import time as _time
    import numpy as _np
    t0 = _time.time()
    TIME_BUDGET = 168.0
    t_end = t0 + TIME_BUDGET
    rng = _np.random.default_rng(0)
    n, d = 59, 22

    def rand_X():
        Y = rng.standard_normal(size=(n, d))
        return _normalize(Y)

    def adam_lse(X0, steps, beta0, beta1, lr, t_lim):
        X = X0.copy()
        m = _np.zeros_like(X)
        v = _np.zeros_like(X)
        b1 = 0.9
        b2 = 0.999
        eps = 1e-8
        best = _max_cos(X)
        bestX = X.copy()
        pow1 = b1
        pow2 = b2
        for t in range(1, steps + 1):
            if (t & 63) == 0 and _time.time() > t_lim:
                break
            beta = beta0 + (beta1 - beta0) * (t / steps)
            G = X @ X.T
            _np.fill_diagonal(G, -2.0)
            gm = G.max()
            W = _np.exp(beta * (G - gm))
            _np.fill_diagonal(W, 0.0)
            rs = W.sum(axis=1, keepdims=True) + 1e-16
            D = (W @ X) / rs
            dot = (D * X).sum(axis=1, keepdims=True)
            Gt = D - dot * X
            m *= b1
            m += (1.0 - b1) * Gt
            v *= b2
            v += (1.0 - b2) * (Gt * Gt)
            mhat = m / (1.0 - pow1)
            vhat = v / (1.0 - pow2)
            pow1 *= b1
            pow2 *= b2
            step = lr * mhat / (_np.sqrt(vhat) + eps)
            rn = _np.linalg.norm(step, axis=1, keepdims=True) + 1e-12
            scale = _np.minimum(1.0, 0.05 / rn)
            X = X - step * scale
            X = _normalize(X)
            if (t % 25) == 0:
                cur = _max_cos(X)
                if cur < best:
                    best = cur
                    bestX = X.copy()
        return bestX, best

    def greedy_polish(X0, iters, step0, step1, t_lim):
        Xcur = X0.copy()
        best = _max_cos(Xcur)
        bestX = Xcur.copy()
        decay = (step1 / step0) ** (1.0 / max(iters - 1, 1))
        step = step0
        for t in range(iters):
            if (t & 63) == 0 and _time.time() > t_lim:
                break
            G = Xcur @ Xcur.T
            _np.fill_diagonal(G, -2.0)
            mval = G.max()
            delta = max(0.004, 1.5 * step)
            thr = mval - delta
            Mask = G > thr
            W = _np.where(Mask, (G - thr + 0.01), 0.0)
            _np.fill_diagonal(W, 0.0)
            Gr = W @ Xcur
            dot = (Gr * Xcur).sum(axis=1, keepdims=True)
            Gt = Gr - dot * Xcur
            nrm = _np.linalg.norm(Gt, axis=1, keepdims=True) + 1e-12
            has = (nrm[:, 0] > 1e-9)
            upd = _np.zeros_like(Xcur)
            upd[has] = Gt[has] / nrm[has]
            nhas = ~has
            if _np.any(nhas):
                R = rng.standard_normal(size=(int(nhas.sum()), d))
                Xr = Xcur[nhas]
                R = R - (R * Xr).sum(axis=1, keepdims=True) * Xr
                rn2 = _np.linalg.norm(R, axis=1, keepdims=True) + 1e-12
                R = R / rn2
                upd[nhas] = R * 0.3
            Xcur = Xcur - step * upd
            Xcur = _normalize(Xcur)
            cur = _max_cos(Xcur)
            if cur < best:
                best = cur
                bestX = Xcur.copy()
            step *= decay
        return bestX, best

    def hole_vector(Xkeep, n_cand=6000, refine=150, beta0=40.0, beta1=250.0, lr=0.05):
        # sample candidates
        C = rng.standard_normal(size=(n_cand, d))
        C = _normalize(C)
        S = C @ Xkeep.T
        rm = S.max(axis=1)
        bi = int(_np.argmin(rm))
        c = C[bi].copy()
        # local refine: minimize softmax max dot
        for t in range(1, refine + 1):
            s = Xkeep @ c
            mv = float(s.max())
            beta = beta0 + (beta1 - beta0) * (t / refine)
            w = _np.exp(beta * (s - mv))
            w = w / (w.sum() + 1e-16)
            g = w @ Xkeep  # (d,)
            # project to tangent at c
            g = g - float(g @ c) * c
            # decay lr
            st = lr * (1.0 - 0.7 * t / refine)
            c = c - st * g
            nc = _np.linalg.norm(c)
            if nc < 1e-12:
                c = C[bi]
            else:
                c = c / nc
            if _time.time() > t_end - 0.5:
                break
        return c

    def kill_fill(X, nk=1):
        Y = X.copy()
        G = Y @ Y.T
        _np.fill_diagonal(G, -2.0)
        rowmax = G.max(axis=1)
        # worst nk indices; add randomness among top-5 to diversify
        order = _np.argsort(-rowmax)
        # pick worst, but occasionally second worst first
        for k in range(nk):
            if k == 0:
                # mostly worst, sometimes among top 4
                if rng.random() < 0.7:
                    w = int(order[0])
                else:
                    w = int(order[int(rng.integers(0, min(5, n)))])
            else:
                # second distinct
                cand = [o for o in order if o != w]
                w2 = int(cand[0])
                w = w2
            Xkeep = _np.delete(Y, w, axis=0)
            c = hole_vector(Xkeep)
            Y[w] = c
        return _normalize(Y)

    def subset_shake(X, k, amp, clustered):
        Y = X.copy()
        if clustered:
            G = Y @ Y.T
            _np.fill_diagonal(G, -2.0)
            rowmax = G.max(axis=1)
            # seed among top-8 jammed to diversify
            order = _np.argsort(-rowmax)
            top = order[:8]
            seed = int(top[int(rng.integers(0, len(top)))])
            sim = G[seed].copy()
            # nearest neighbours (largest dot)
            nbr = _np.argsort(-sim)
            # exclude self (sim=-2 at seed? no, diag -2 so seed last; use full)
            # build list: seed + closest
            idx = [seed]
            for j in nbr:
                if j == seed:
                    continue
                idx.append(int(j))
                if len(idx) >= k:
                    break
            idx = _np.array(idx[:k], dtype=int)
        else:
            idx = rng.choice(n, size=k, replace=False)
        Y[idx] = Y[idx] + amp * rng.standard_normal(size=(k, d))
        return _normalize(Y)

    # load best
    cands = []
    for fn in ("best.npy", "record.npy"):
        try:
            a = _np.load(fn)
            if a.shape == (n, d):
                a = _np.array(a, dtype=float)
                if not _np.all(_np.isfinite(a)):
                    continue
                a = _normalize(a)
                if _np.all(_np.isfinite(a)):
                    cands.append(a)
        except Exception:
            pass
    if len(cands) == 0:
        cands.append(rand_X())
    vals = [_max_cos(c) for c in cands]
    bi = int(_np.argmin(vals))
    gbestX = cands[bi].copy()
    gbest = float(vals[bi])

    trial = 0
    while True:
        if _time.time() > t_end - 3.0:
            break
        rem = t_end - _time.time()
        if rem < 4.0:
            break
        trial += 1
        mod = trial % 7
        if mod == 1:
            X0 = subset_shake(gbestX, k=3, amp=0.45, clustered=True)
        elif mod == 2:
            X0 = subset_shake(gbestX, k=6, amp=0.65, clustered=False)
        elif mod == 3:
            try:
                X0 = kill_fill(gbestX, nk=1)
            except Exception:
                X0 = subset_shake(gbestX, k=4, amp=0.5, clustered=True)
        elif mod == 4:
            X0 = subset_shake(gbestX, k=8, amp=0.8, clustered=True)
        elif mod == 5:
            amp = 0.18
            X0 = _normalize(gbestX + amp * rng.standard_normal(size=(n, d)))
        elif mod == 6:
            # double kill occasionally, else medium random subset
            if trial % 14 == 0:
                try:
                    X0 = kill_fill(gbestX, nk=2)
                except Exception:
                    X0 = subset_shake(gbestX, k=5, amp=0.6, clustered=False)
            else:
                X0 = subset_shake(gbestX, k=5, amp=0.55, clustered=False)
        else:  # mod 0
            if trial % 21 == 0:
                X0 = rand_X()
            else:
                X0 = subset_shake(gbestX, k=4, amp=0.35, clustered=True)

        if rem < 20:
            s1 = 700
        elif rem < 60:
            s1 = 1100
        else:
            s1 = 1400
        t_lim1 = min(_time.time() + min(rem * 0.55, 7.0), t_end - 1.5)
        X1, v1 = adam_lse(X0, s1, 35.0, 230.0, 0.015, t_lim1)
        if v1 < gbest:
            gbest = v1
            gbestX = X1.copy()
        # prune far basins to save time
        if v1 > gbest + 0.008 and (trial % 3 != 0):
            # skip expensive polish for clearly bad basin
            continue

        rem2 = t_end - _time.time()
        if rem2 < 1.5:
            continue
        s2 = 500 if rem2 < 15 else 1000
        t_lim2 = min(_time.time() + min(rem2 * 0.4, 5.0), t_end - 1.0)
        X2, v2 = adam_lse(X1, s2, 150.0, 320.0, 0.008, t_lim2)
        if v2 < gbest:
            gbest = v2
            gbestX = X2.copy()
        base = X2 if v2 < v1 else X1
        bv = v2 if v2 < v1 else v1
        # only greedy if reasonably close
        if bv > gbest + 0.004:
            continue
        rem3 = t_end - _time.time()
        if rem3 < 1.0:
            continue
        itg = 600 if rem3 < 12 else 1200
        t_lim3 = t_end - 0.5
        X3, v3 = greedy_polish(base, itg, 0.007, 0.00012, t_lim3)
        if v3 < gbest:
            gbest = v3
            gbestX = X3.copy()
        if trial > 200:
            break

    if _time.time() < t_end - 0.8:
        try:
            Xf, vf = greedy_polish(gbestX, 700, 0.002, 0.00005, t_end - 0.2)
            if vf < gbest:
                gbest = vf
                gbestX = Xf.copy()
        except Exception:
            pass
    gbestX = _normalize(gbestX)
    return gbestX
