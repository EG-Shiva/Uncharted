import time
import numpy as np
from scipy.optimize import linprog, minimize
from scipy.sparse import coo_matrix

def _norm_rows(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[~np.isfinite(n)] = 1.0
    n[n == 0] = 1.0
    return x / n

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _lp_polish(x, seconds, deadline=None):
    x = _norm_rows(x.copy())
    n, d = x.shape
    try:
        best = _max_cos(x)
    except Exception:
        return x
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds:
        if deadline is not None and time.time() > deadline:
            break
        if radius < 1e-14:
            break
        G = x @ x.T
        cv = G[iu]
        m = float(cv.max())
        thresh = m - max(20 * radius, 1e-9)
        sel = cv > thresh
        if np.count_nonzero(sel) == 0:
            break
        I = iu[0][sel]; J = iu[1][sel]
        csub = cv[sel]
        p = len(I)
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        except Exception:
            radius *= 0.5
            continue
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-csub, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y = _norm_rows(y)
        if not np.all(np.isfinite(y)):
            radius *= 0.5
            continue
        v = _max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.6, 0.05)
        else:
            radius *= 0.5
    return x

def _quench_hybrid(X_init, steps, T0, T1, lr0, lr1, alpha=0.6):
    X = _norm_rows(X_init.copy())
    n, d = X.shape
    m1 = np.zeros_like(X)
    v1 = np.zeros_like(X)
    b1, b2, eps = 0.9, 0.999, 1e-8
    lT0 = np.log(T0); lT1 = np.log(T1)
    ll0 = np.log(lr0); ll1 = np.log(lr1)
    G0 = X @ X.T
    np.fill_diagonal(G0, -2.0)
    best_m = float(G0.max())
    best_X = X.copy()
    for s in range(1, steps + 1):
        frac = (s - 1) / max(steps - 1, 1)
        T = float(np.exp(lT0 * (1 - frac) + lT1 * frac))
        lr = float(np.exp(ll0 * (1 - frac) + ll1 * frac))
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        if m < best_m:
            best_m = m
            best_X = X.copy()
        D = (G - m) / T
        np.clip(D, -50, 0, out=D)
        E = np.exp(D)
        np.fill_diagonal(E, 0.0)
        S = float(E.sum() * 0.5)
        if not np.isfinite(S) or S <= 0:
            S = 1.0
        Grad_g = (E @ X) / S
        rm = G.max(axis=1)
        Q = (G - rm[:, None]) / T
        np.clip(Q, -50, 0, out=Q)
        Er = np.exp(Q)
        np.fill_diagonal(Er, 0.0)
        rs = Er.sum(axis=1, keepdims=True)
        rs[rs == 0] = 1.0
        W = Er / rs
        Ws = 0.5 * (W + W.T)
        Grad_r = Ws @ X
        Grad = alpha * Grad_g + (1.0 - alpha) * Grad_r
        dot = np.sum(Grad * X, axis=1, keepdims=True)
        Gt = Grad - dot * X
        if not np.all(np.isfinite(Gt)):
            break
        m1 = b1 * m1 + (1 - b1) * Gt
        v1 = b2 * v1 + (1 - b2) * (Gt * Gt)
        c1 = 1 - b1 ** s
        c2 = 1 - b2 ** s
        step = (m1 / c1) / (np.sqrt(v1 / c2) + eps)
        sn = float(np.linalg.norm(step))
        maxn = 10.0 * np.sqrt(n * d)
        if sn > maxn:
            step *= (maxn / sn)
        X = X - lr * step
        X = _norm_rows(X)
    Gf = X @ X.T
    np.fill_diagonal(Gf, -2.0)
    mf = float(Gf.max())
    if mf < best_m:
        best_m = mf
        best_X = X.copy()
    return best_X

def _lbfgs_lse(X0, T, maxiter=120):
    n, d = X0.shape
    iu = np.triu_indices(n, 1)
    Y = _norm_rows(X0.copy())
    def fun(Yflat, beta=T):
        Ym = Yflat.reshape(n, d)
        rn = np.linalg.norm(Ym, axis=1, keepdims=True)
        rn = np.maximum(rn, 1e-12)
        X = Ym / rn
        G = X @ X.T
        gv = G[iu]
        gm = float(gv.max())
        q = (gv - gm) / beta
        np.clip(q, -50, 0, out=q)
        e = np.exp(q)
        S = float(e.sum())
        if S <= 0 or not np.isfinite(S):
            S = 1.0
        F = gm + beta * np.log(S)
        w = e / S
        W = np.zeros((n, n))
        W[iu] = w
        W = W + W.T
        gradX = W @ X
        dot = (gradX * X).sum(axis=1, keepdims=True)
        gradY = (gradX - dot * X) / rn
        r1 = rn.squeeze() - 1.0
        pen = float((r1 ** 2).mean())
        Ftot = float(F) + pen
        gradY = gradY + (2.0 / n) * r1[:, None] * X
        if not np.all(np.isfinite(gradY)):
            gradY = np.zeros_like(gradY)
        return Ftot, gradY.ravel()
    try:
        res = minimize(fun, Y.ravel(), method='L-BFGS-B', jac=True,
                       options={'maxiter': int(maxiter), 'ftol': 1e-12, 'gtol': 1e-7, 'maxcor': 10})
        Y = res.x.reshape(n, d)
    except Exception:
        pass
    Y = _norm_rows(Y)
    return Y

def _refine_single(s0, R, steps=250, T=0.004, lr=0.12, mu=0.85):
    s = s0 / (np.linalg.norm(s0) + 1e-12)
    v = np.zeros_like(s)
    for _ in range(steps):
        g = R @ s
        m = float(g.max())
        e = np.exp(np.clip((g - m) / T, -50, 0))
        den = float(e.sum()) + 1e-30
        Grad = (R.T @ e) / den
        Gt = Grad - float(Grad @ s) * s
        v = mu * v + Gt
        s = s - lr * v
        nn = float(np.linalg.norm(s))
        if not np.isfinite(nn) or nn == 0:
            break
        s = s / nn
    return s

def _find_hole(R, rng, n_sample=6000):
    n, d = R.shape
    Z = rng.standard_normal(size=(n_sample, d))
    Z = _norm_rows(Z)
    S = Z @ R.T
    scores = S.max(axis=1)
    k = min(4, n_sample)
    idx = np.argpartition(scores, k)[:k]
    best_s = None
    best_v = 1e100
    for kk in idx:
        s0 = Z[kk]
        s1 = _refine_single(s0, R, steps=200, T=0.005, lr=0.15)
        s1 = _refine_single(s1, R, steps=200, T=0.0012, lr=0.07)
        v = float((R @ s1).max())
        if v < best_v:
            best_v = v
            best_s = s1
    best_s = best_s / (np.linalg.norm(best_s) + 1e-12)
    return best_s

def _loads(X, band=0.005):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    m = float(G.max())
    row_max = G.max(axis=1)
    cnt = np.sum(G > (m - band), axis=1)
    return row_max, cnt, m

def solve():
    t0 = time.time()
    HARD_END = t0 + 174.0
    HOP_END = HARD_END - 31.0
    np.random.seed(1)
    rng = np.random.default_rng(12345)
    n, d = 71, 12
    best_X = None
    try:
        best_X = np.load("best.npy").astype(float)
    except Exception:
        try:
            best_X = np.load("record.npy").astype(float)
        except Exception:
            pass
    if best_X is None or best_X.shape != (71, 12) or not np.all(np.isfinite(best_X)):
        best_X = _norm_rows(rng.standard_normal((n, d)))
    best_X = _norm_rows(best_X)
    best_score = _max_cos(best_X)
    # quick squeeze
    try:
        Xq = _quench_hybrid(best_X, steps=1800, T0=0.001, T1=0.00012, lr0=0.004, lr1=0.0004, alpha=0.6)
        sq = _max_cos(Xq)
        if sq < best_score:
            best_X, best_score = _norm_rows(Xq), sq
    except Exception:
        pass
    try:
        if HOP_END - time.time() > 5:
            Xp = _lp_polish(best_X, seconds=1.5, deadline=HOP_END)
            sp = _max_cos(Xp)
            if sp < best_score:
                best_X, best_score = _norm_rows(Xp), sp
    except Exception:
        pass
    archive = [(best_score, best_X.copy())]
    curX = best_X.copy()
    curVal = best_score
    def push_archive(v, X):
        archive.append((v, X.copy()))
        archive.sort(key=lambda z: z[0])
        del archive[6:]
    it = 0
    while time.time() < HOP_END:
        it += 1
        tleft = HOP_END - time.time()
        if tleft < 0.8:
            break
        # pick base
        rpick = rng.random()
        if rpick < 0.65:
            base = best_X
        elif len(archive) > 1 and rpick < 0.85:
            base = archive[int(rng.integers(1, len(archive)))][1]
        else:
            base = curX
        u = rng.random()
        if it % 53 == 0:
            X0 = _norm_rows(rng.standard_normal((n, d)))
        elif u < 0.38:
            sigma = float(10 ** rng.uniform(-2.8, -0.7))
            # targeted: bigger noise on overloaded points
            try:
                row_max, cnt, _ = _loads(base, band=0.005)
                w = (row_max - row_max.min() + 1e-6)
                w = w / w.max()
                sig = sigma * (0.4 + 1.6 * w)[:, None]
                X0 = base + rng.standard_normal((n, d)) * sig
            except Exception:
                X0 = base + rng.standard_normal((n, d)) * sigma
            X0 = _norm_rows(X0)
        elif u < 0.62:
            k_rep = int(rng.integers(1, 4))
            try:
                row_max, cnt, _ = _loads(base, band=0.005)
                order = np.argsort(-row_max - 0.002 * cnt + rng.random(n) * 0.001)
                idx = order[:k_rep]
            except Exception:
                idx = rng.choice(n, size=k_rep, replace=False)
            X0 = base.copy()
            X0[idx] = _norm_rows(rng.standard_normal((k_rep, d)))
            sigma = float(10 ** rng.uniform(-2.6, -1.0))
            mask = np.ones(n, dtype=bool); mask[idx] = False
            X0[mask] += rng.standard_normal((n - k_rep, d)) * sigma * 0.5
            X0 = _norm_rows(X0)
        else:
            k_rep = 1 if rng.random() < 0.65 else 2
            if rng.random() < 0.12:
                k_rep = 3
            try:
                row_max, cnt, _ = _loads(base, band=0.005)
                order = np.argsort(-row_max - 0.002 * cnt + rng.random(n) * 0.0005)
                idx = order[:k_rep]
            except Exception:
                idx = rng.choice(n, size=k_rep, replace=False)
            keep = np.ones(n, dtype=bool); keep[idx] = False
            R = base[keep]
            X0 = base.copy()
            for j in idx:
                try:
                    h = _find_hole(R, rng, n_sample=5000)
                except Exception:
                    h = _norm_rows(rng.standard_normal((1, d)))[0]
                X0[j] = h
                R = np.concatenate([R, h[None, :]], axis=0)
            sigma = float(10 ** rng.uniform(-2.9, -1.2))
            X0[keep] += rng.standard_normal((n - k_rep, d)) * sigma
            X0 = _norm_rows(X0)
        T0 = float(10 ** rng.uniform(-2.6, -1.7))
        T1 = float(10 ** rng.uniform(-4.3, -3.2))
        lr0 = float(10 ** rng.uniform(-2.9, -2.1))
        lr1 = lr0 * float(10 ** rng.uniform(-1.2, -0.5))
        if tleft < 6:
            steps = 1000
        elif tleft < 25:
            steps = 1600
        else:
            steps = 2000
            if rng.random() < 0.18:
                steps = 3000
        try:
            X1 = _quench_hybrid(X0, steps, T0, T1, lr0, lr1, alpha=0.6)
            s1 = _max_cos(X1)
        except Exception:
            continue
        if not np.isfinite(s1):
            continue
        # opportunistic L-BFGS
        if s1 < best_score + 0.0025 and tleft > 3:
            try:
                Tb = float(10 ** rng.uniform(-3.0, -2.4))
                Xb = _lbfgs_lse(X1, T=Tb, maxiter=80)
                sb = _max_cos(Xb)
                if sb < s1:
                    X1, s1 = _norm_rows(Xb), sb
                # second colder stage occasionally
                if rng.random() < 0.35:
                    Xc = _lbfgs_lse(X1, T=max(Tb * 0.4, 0.0004), maxiter=60)
                    sc = _max_cos(Xc)
                    if sc < s1:
                        X1, s1 = _norm_rows(Xc), sc
            except Exception:
                pass
        if s1 < best_score + 0.0012:
            secs = 1.2
            if tleft < secs + 0.6:
                secs = max(0.4, tleft - 0.6)
            try:
                X2 = _lp_polish(X1, seconds=secs, deadline=HOP_END)
                s2 = _max_cos(X2)
            except Exception:
                continue
            if np.isfinite(s2):
                # archive / accept
                if s2 < best_score - 1e-12:
                    best_X = _norm_rows(X2)
                    best_score = float(s2)
                    push_archive(best_score, best_X)
                    curX, curVal = best_X.copy(), best_score
                else:
                    # Metropolis wander on cur
                    try:
                        Tact = 0.0006
                        if s2 < curVal or rng.random() < np.exp((curVal - s2) / Tact):
                            curX, curVal = _norm_rows(X2).copy(), float(s2)
                            push_archive(float(s2), curX)
                    except Exception:
                        pass
        elif s1 < curVal - 0.0005:
            curX, curVal = _norm_rows(X1).copy(), float(s1)
    # final intensive squeeze
    try:
        trem = HARD_END - time.time() - 0.6
        if trem > 2.0:
            qs = 4500 if trem > 14 else 2200
            try:
                Xf = _quench_hybrid(best_X, steps=qs, T0=0.00055, T1=0.00008, lr0=0.0022, lr1=0.00022, alpha=0.5)
                sf = _max_cos(Xf)
                if sf < best_score:
                    best_X, best_score = _norm_rows(Xf), sf
            except Exception:
                pass
            try:
                Xl = _lbfgs_lse(best_X, T=0.0009, maxiter=150)
                sl = _max_cos(Xl)
                if sl < best_score:
                    best_X, best_score = _norm_rows(Xl), sl
                Xl2 = _lbfgs_lse(best_X, T=0.0004, maxiter=150)
                sl2 = _max_cos(Xl2)
                if sl2 < best_score:
                    best_X, best_score = _norm_rows(Xl2), sl2
            except Exception:
                pass
            trem2 = HARD_END - time.time() - 0.5
            if trem2 > 1.0:
                best_X = _lp_polish(best_X, seconds=min(trem2, 30.0), deadline=HARD_END - 0.4)
    except Exception:
        pass
    best_X = _norm_rows(best_X)
    for i in range(n):
        if not np.all(np.isfinite(best_X[i])) or float(np.linalg.norm(best_X[i])) < 1e-12:
            v = rng.standard_normal(d)
            v = v / (np.linalg.norm(v) + 1e-12)
            best_X[i] = v
    best_X = _norm_rows(best_X)
    return best_X
