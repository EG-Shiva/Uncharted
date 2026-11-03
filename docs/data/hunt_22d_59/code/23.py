import time
import numpy as np

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def _norm(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return X / n

def solve():
    import time as _tm
    import numpy as _np
    np = _np
    t0 = _tm.time()
    DEADLINE = t0 + 168.0
    rng = np.random.default_rng(0)
    n, d = 59, 22

    cands = []
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn).astype(np.float64)
            if a.shape == (59, 22) and np.all(np.isfinite(a)):
                a = _norm(a)
                cands.append(a)
        except Exception:
            pass
    if not cands:
        cands.append(_norm(rng.standard_normal((59, 22))))
    vals = [_max_cos(c) for c in cands]
    best = cands[int(np.argmin(vals))].copy()
    best_val = float(min(vals))

    def adam_sm(X0, steps, b0, b1, lr0, lr1):
        X = X0.copy()
        m = np.zeros_like(X)
        v = np.zeros_like(X)
        b1m, b2m, eps = 0.9, 0.999, 1e-8
        lb0, lb1 = np.log(b0), np.log(b1)
        ll0, ll1 = np.log(lr0), np.log(lr1)
        cur_best = X.copy()
        cur_val = _max_cos(X)
        p1 = 1.0; p2 = 1.0
        for t in range(1, steps + 1):
            if (t & 127) == 0 and _tm.time() > DEADLINE - 3:
                break
            f = t / steps
            beta = float(np.exp(lb0 + (lb1 - lb0) * f))
            lr = float(np.exp(ll0 + (ll1 - ll0) * f))
            G = X @ X.T
            np.fill_diagonal(G, -2.0)
            M = float(np.max(G))
            G -= M
            G *= beta
            np.exp(G, out=G)
            np.fill_diagonal(G, 0.0)
            S = float(np.sum(G))
            if not np.isfinite(S) or S <= 0:
                continue
            Gr = (G @ X) / S
            dot = np.sum(Gr * X, axis=1, keepdims=True)
            Gr -= dot * X
            p1 *= b1m; p2 *= b2m
            m *= b1m; m += 0.1 * Gr
            v *= b2m; v += 0.001 * (Gr * Gr)
            mc = m / (1.0 - p1)
            vc = v / (1.0 - p2)
            X -= lr * mc / (np.sqrt(vc) + eps)
            nn = np.linalg.norm(X, axis=1, keepdims=True)
            nn[nn == 0] = 1.0
            X /= nn
            if (t % 300 == 0) or (t == steps):
                cv = _max_cos(X)
                if cv < cur_val:
                    cur_val = cv
                    cur_best = X.copy()
        fv = _max_cos(X)
        if fv < cur_val:
            cur_val = fv
            cur_best = X.copy()
        return cur_best, cur_val

    def contact_sq(X0, steps, step0, tol):
        Y = X0.copy()
        bv = _max_cos(Y)
        bY = Y.copy()
        step = float(step0)
        E = np.eye(n, dtype=np.float64)
        for it in range(steps):
            G = Y @ Y.T
            m = float(np.max(G - 2.0 * E))
            A = G > (m - tol)
            np.fill_diagonal(A, False)
            Af = A.astype(np.float64)
            AY = Af @ Y
            rw = np.sum(Af * G, axis=1, keepdims=True)
            F = -AY + Y * rw
            # normalize force scale
            fn_ = np.sqrt(np.sum(F * F)) + 1e-12
            # trial
            Yt = Y + step * F
            Yt = _norm(Yt)
            Gt = Yt @ Yt.T
            np.fill_diagonal(Gt, -2.0)
            mt = float(np.max(Gt))
            if mt < m - 1e-13:
                Y = Yt
                if mt < bv:
                    bv = mt
                    bY = Y.copy()
                step = min(step * 1.07, 0.02)
            else:
                step *= 0.5
                if step < 1e-13:
                    # randomize tol escape: take tiny step anyway every so often
                    if (it % 40) == 0:
                        Y = _norm(Y + 1e-5 * F / fn_)
                    step = 1e-6
            if (it & 63) == 0 and _tm.time() > DEADLINE - 3:
                break
        return bY, bv

    def hole_opt(X, kill_list, n_cand=3000):
        Y = X.copy()
        keep = [i for i in range(n) if i not in set(list(kill_list))]
        Xk = X[keep].copy()
        for k in list(kill_list):
            C = rng.standard_normal((n_cand, d))
            C /= np.linalg.norm(C, axis=1, keepdims=True)
            sc = C @ Xk.T
            mx = np.max(sc, axis=1)
            bi = int(np.argmin(mx))
            p = C[bi:bi+1].copy()
            # refine hole: minimize LSE of dots to fixed set
            mp = np.zeros_like(p)
            vp = np.zeros_like(p)
            p1 = 1.0; p2 = 1.0
            beta = 250.0
            for t in range(1, 251):
                s = (p @ Xk.T).ravel()
                sm = float(np.max(s))
                w = np.exp((s - sm) * beta)
                sw = float(np.sum(w))
                if sw <= 0 or not np.isfinite(sw):
                    break
                w /= sw
                g = (w @ Xk).reshape(1, d)
                # tangent proj at p
                dot = float(np.sum(g * p))
                gt = g - dot * p
                lr = 0.02 * (0.15 ** (t / 250.0))
                p1 *= 0.9; p2 *= 0.999
                mp *= 0.9; mp += 0.1 * gt
                vp *= 0.999; vp += 0.001 * (gt * gt)
                mc = mp / (1.0 - p1)
                vc = vp / (1.0 - p2)
                p -= lr * mc / (np.sqrt(vc) + 1e-8)
                p /= (np.linalg.norm(p) + 1e-16)
            Y[k] = p.ravel()
            Xk = np.vstack([Xk, Y[k:k+1]])
        return Y

    def quick_polish(X):
        Y, vy = adam_sm(X, steps=1400, b0=120, b1=850, lr0=0.004, lr1=0.0004)
        if _tm.time() > DEADLINE - 6:
            return Y, vy
        Y2, v2 = contact_sq(Y, steps=350, step0=0.004, tol=0.005)
        if v2 < vy:
            return Y2, v2
        return Y, vy

    def deep_polish(X):
        Y, vy = adam_sm(X, steps=3500, b0=90, b1=1100, lr0=0.005, lr1=0.0003)
        if _tm.time() > DEADLINE - 6:
            return Y, vy
        Y2, v2 = contact_sq(Y, steps=800, step0=0.004, tol=0.006)
        if v2 < vy:
            Y, vy = Y2, v2
        if _tm.time() > DEADLINE - 6:
            return Y, vy
        Y3, v3 = adam_sm(Y, steps=1500, b0=900, b1=2200, lr0=0.0012, lr1=0.00012)
        if v3 < vy:
            Y, vy = Y3, v3
        return Y, vy

    def lbfgs_polish(X0, beta, maxiter):
        try:
            from scipy.optimize import minimize
        except Exception:
            return X0.copy(), _max_cos(X0)
        Z0 = X0.copy().ravel()
        def fg(zf):
            Z = zf.reshape(n, d)
            rn = np.sqrt(np.sum(Z * Z, axis=1, keepdims=True))
            rn[rn == 0] = 1.0
            X_ = Z / rn
            G = X_ @ X_.T
            np.fill_diagonal(G, -10.0)
            gm = float(np.max(G))
            E = G - gm
            E *= beta
            np.exp(E, out=E)
            np.fill_diagonal(E, 0.0)
            S = float(np.sum(E))
            if S <= 0 or not np.isfinite(S):
                return 1e9, np.zeros_like(zf)
            L = gm + np.log(S) / beta
            E /= S
            GX = (E @ X_) * 2.0
            dot = np.sum(GX * X_, axis=1, keepdims=True)
            GXt = GX - dot * X_
            DZ = GXt / rn
            return float(L), DZ.ravel()
        try:
            res = minimize(fg, Z0, jac=True, method='L-BFGS-B',
                           options={'maxiter': int(maxiter), 'ftol': 1e-22, 'gtol': 1e-14, 'maxfun': 100000})
            Z = res.x.reshape(n, d)
            X_ = _norm(Z)
            return X_, _max_cos(X_)
        except Exception:
            return X0.copy(), _max_cos(X0)

    # initial squeeze of record/best
    best, best_val = adam_sm(best, steps=3500, b0=60, b1=700, lr0=0.006, lr1=0.0005)
    if _tm.time() < DEADLINE - 10:
        b2, v2 = contact_sq(best, steps=600, step0=0.004, tol=0.006)
        if v2 < best_val:
            best, best_val = b2, v2
    cur = best.copy()
    cur_val = best_val
    Temp = 0.0012

    it = 0
    while _tm.time() < DEADLINE - 10:
        it += 1
        remain = DEADLINE - _tm.time()
        if remain < 9:
            break
        # cool temperature
        Temp *= 0.9985
        Temp = max(Temp, 0.00008)
        G = cur @ cur.T
        np.fill_diagonal(G, -2.0)
        rowmax = np.max(G, axis=1)
        cnt = np.sum(G > (cur_val - 0.004), axis=1)
        ow = np.argsort(-rowmax)
        ol = np.argsort(-cnt)
        mode = it % 7
        if mode == 0:
            sig = 0.006 if (it % 2 == 0) else 0.016
            Nz = rng.standard_normal((n, d))
            # tangentize
            Nz -= cur * np.sum(Nz * cur, axis=1, keepdims=True)
            boost = np.ones((n, 1))
            boost[ow[:6]] = 3.0
            Y = cur + (sig * boost) * Nz
            Y = _norm(Y)
        elif mode == 1:
            k = 1 if (it % 2 == 0) else 2
            kill = ow[:k]
            Y = hole_opt(cur, kill, n_cand=3500)
            Y += 0.0015 * rng.standard_normal((n, d))
            Y = _norm(Y)
        elif mode == 2:
            kill = np.unique(np.concatenate([ow[:2], ol[:2]]))[:3]
            Y = hole_opt(cur, kill, n_cand=3500)
            Y = _norm(Y)
        elif mode == 3:
            # subset shake of jammed block
            sel = np.unique(np.concatenate([ow[:4], ol[:6]]))
            Y = cur.copy()
            Nz = rng.standard_normal((len(sel), d))
            Nz -= Y[sel] * np.sum(Nz * Y[sel], axis=1, keepdims=True)
            Y[sel] = _norm(Y[sel] + 0.03 * Nz)
        elif mode == 4:
            # random single big jump
            k = int(rng.integers(0, n))
            Y = cur.copy()
            if (it % 3) == 0:
                Y[k] = rng.standard_normal(d)
            else:
                Y = hole_opt(cur, [k], n_cand=5000)
            Y += 0.0025 * rng.standard_normal((n, d))
            Y = _norm(Y)
        elif mode == 5:
            sig = 0.035 if (it % 2 == 0) else 0.008
            Y = _norm(cur + sig * rng.standard_normal((n, d)))
        else:
            # 4-hole + global jitter
            kill = ow[:4]
            Y = hole_opt(cur, kill, n_cand=2500)
            Y += 0.002 * rng.standard_normal((n, d))
            Y = _norm(Y)
        Yp, Yv = quick_polish(Y)
        if Yv < best_val - 1e-12:
            best, best_val = Yp.copy(), Yv
            cur, cur_val = Yp.copy(), Yv
        elif Yv < cur_val or rng.random() < np.exp(-max(0.0, Yv - cur_val) / Temp):
            cur, cur_val = Yp.copy(), Yv
        if (it % 14) == 0 and _tm.time() < DEADLINE - 25:
            Yi, Vi = deep_polish(best.copy())
            if Vi < best_val:
                best, best_val = Yi, Vi
                cur, cur_val = Yi.copy(), Vi

    # final intensification
    if _tm.time() < DEADLINE - 2:
        Y, vy = adam_sm(best, steps=2200, b0=900, b1=2500, lr0=0.0012, lr1=0.0001)
        if vy < best_val:
            best, best_val = Y, vy
    if _tm.time() < DEADLINE - 1.5:
        Y2, v2 = contact_sq(best, steps=500, step0=0.002, tol=0.003)
        if v2 < best_val:
            best, best_val = Y2, v2
    if _tm.time() < DEADLINE - 1.0:
        for be in (1200, 2500):
            Y2, v2 = lbfgs_polish(best, beta=be, maxiter=120)
            if v2 < best_val:
                best, best_val = Y2, v2
            if _tm.time() > DEADLINE - 1.0:
                break
    best = _norm(best)
    return best
