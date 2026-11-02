import time
import numpy as np
from scipy.optimize import linprog, minimize
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _lp_polish(x, deadline, r0=1e-3):
    x = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
    n, d = x.shape
    try:
        best = _max_cos(x)
    except Exception:
        return x, 1.0
    radius = r0
    iu0, iu1 = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        try:
            G = x @ x.T
            c = G[iu0, iu1]
            m = float(c.max())
            sel = c > m - max(20.0 * radius, 1e-9)
            if np.count_nonzero(sel) == 0:
                sel[np.argmax(c)] = True
            I = iu0[sel]; J = iu1[sel]
            csel = c[sel]
            p = len(I)
            nv = n * d + 1
            rows = np.repeat(np.arange(p), 2 * d + 1)
            tmpc = np.empty((p, 2 * d + 1), dtype=int)
            ar = np.arange(d)
            tmpc[:, :d] = I[:, None] * d + ar
            tmpc[:, d:2 * d] = J[:, None] * d + ar
            tmpc[:, 2 * d] = n * d
            cols = tmpc.ravel()
            cc = np.empty((p, 2 * d + 1), dtype=float)
            cc[:, :d] = x[J]
            cc[:, d:2 * d] = x[I]
            cc[:, 2 * d] = -1.0
            vals = cc.ravel()
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv); obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
            if res.status != 0:
                radius *= 0.5
                continue
            y = x + res.x[:-1].reshape(n, d)
            y /= np.maximum(np.linalg.norm(y, axis=1, keepdims=True), 1e-12)
            v = _max_cos(y)
            if v < best:
                x, best = y, v
                radius = min(radius * 1.5, 0.05)
            else:
                radius *= 0.5
        except Exception:
            radius *= 0.5
            continue
    return x, best

def _gd_softmax(x0, steps, T0, T1, lr0, mu, rng, noise0=0.0, deadline=None):
    x = x0.copy()
    n, d = x.shape
    v = np.zeros_like(x)
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    best_val = float(G.max())
    best = x.copy()
    lratio = np.log(T1 / T0) if T0 > 0 and T1 > 0 else 0.0
    for k in range(steps):
        if deadline is not None and (k & 255) == 0 and time.time() > deadline:
            break
        frac = k / max(1, steps - 1)
        T = T0 * np.exp(lratio * frac)
        if T < 1e-12:
            T = 1e-12
        lr = lr0 * (1.0 - 0.6 * frac)
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        E = np.exp((G - m) / T)
        np.fill_diagonal(E, 0.0)
        S = float(E.sum() * 0.5)
        if S < 1e-300 or not np.isfinite(S):
            S = 1e-300
        grad = (E @ x) / S
        grad -= (grad * x).sum(axis=1, keepdims=True) * x
        v = mu * v + (1.0 - mu) * grad
        x = x - lr * v
        if noise0 > 0:
            amp = noise0 * (1.0 - frac)
            if amp > 0:
                nz = rng.randn(n, d) * amp
                nz -= (nz * x).sum(axis=1, keepdims=True) * x
                x = x + nz
        x /= np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
        if m < best_val:
            best_val = m
            best = x.copy()
    return best, best_val

def _gd_riesz(x0, s0, s1, steps, lr0, mu, deadline=None):
    x = x0.copy()
    n, d = x.shape
    vel = np.zeros_like(x)
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    best_val = float(G.max())
    best = x.copy()
    for k in range(steps):
        if deadline is not None and (k & 255) == 0 and time.time() > deadline:
            break
        frac = k / max(1, steps - 1)
        s = s0 * (s1 / s0) ** frac if s0 > 0 and s1 > 0 else s0
        lr = lr0 * (1.0 - 0.6 * frac)
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        D = 2.0 - 2.0 * G
        np.fill_diagonal(D, 1.0)
        np.maximum(D, 1e-12, out=D)
        np.fill_diagonal(D, 1e100)
        try:
            dmin = float(D.min())
        except Exception:
            dmin = 1e-6
        np.fill_diagonal(D, 1.0)
        pw = s / 2.0 + 1.0
        with np.errstate(over='ignore', under='ignore', divide='ignore', invalid='ignore'):
            W = np.exp(pw * (np.log(max(dmin, 1e-18)) - np.log(D)))
        W[~np.isfinite(W)] = 0.0
        np.fill_diagonal(W, 0.0)
        Gx = W @ x
        grad = Gx - (Gx * x).sum(axis=1, keepdims=True) * x
        gn = float(np.sqrt((grad * grad).sum() / n))
        if gn > 1e-12:
            grad /= max(1.0, gn * 2.0)
        vel = mu * vel + (1.0 - mu) * grad
        x = x - lr * vel
        x /= np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
        if m < best_val:
            best_val = m
            best = x.copy()
    return best, best_val

def _refine_hole(x_fixed, y):
    y = y / max(np.linalg.norm(y), 1e-12)
    vy = float((x_fixed @ y).max())
    for _ in range(180):
        dots = x_fixed @ y
        j = int(np.argmax(dots))
        g = x_fixed[j] - dots[j] * y
        ng = float(np.linalg.norm(g))
        if ng < 1e-12:
            break
        g /= ng
        improved = False
        for step in (0.03, 0.01, 0.003):
            yn = y - step * g
            nyn = float(np.linalg.norm(yn))
            if nyn < 1e-12:
                continue
            yn /= nyn
            vn = float((x_fixed @ yn).max())
            if vn < vy:
                y, vy = yn, vn
                improved = True
                break
        if not improved:
            break
    return y, vy

def _greedy_holes(fixed, k, rng, n_samples=10000):
    outs = []
    cur = fixed.copy()
    for _ in range(k):
        Y = rng.randn(n_samples, 4)
        Y /= np.maximum(np.linalg.norm(Y, axis=1, keepdims=True), 1e-12)
        D = Y @ cur.T
        vals = D.max(axis=1)
        idx = int(np.argmin(vals))
        y = Y[idx].copy()
        y, _ = _refine_hole(cur, y)
        outs.append(y)
        cur = np.vstack([cur, y.reshape(1, 4)])
    return outs

def _lbfgs_smooth(x_init, deadline, taus=(0.003, 0.001, 0.00045), iters=(100, 120, 150)):
    x = x_init / np.maximum(np.linalg.norm(x_init, axis=1, keepdims=True), 1e-12)
    n, d = x.shape
    iu0, iu1 = np.triu_indices(n, 1)
    Np = len(iu0)
    y = x.ravel().copy()
    for tau, maxiter in zip(taus, iters):
        if time.time() > deadline - 0.4:
            break
        def fg(yflat, _tau=tau):
            Y = yflat.reshape(n, d)
            rn = np.linalg.norm(Y, axis=1, keepdims=True)
            rn = np.maximum(rn, 1e-12)
            X = Y / rn
            G = X @ X.T
            g = G[iu0, iu1]
            m = float(g.max())
            e = np.exp((g - m) / _tau)
            s = float(e.sum())
            if s <= 0 or not np.isfinite(s):
                return float(m), np.zeros_like(yflat)
            L = m + _tau * np.log(s / Np)
            w = e / s
            W = np.zeros((n, n))
            W[iu0, iu1] = w
            W = W + W.T
            GX = W @ X
            dot = (GX * X).sum(axis=1, keepdims=True)
            GY = (GX - dot * X) / rn
            return float(L), GY.ravel()
        try:
            res = minimize(fg, y, method="L-BFGS-B", jac=True,
                           options={"maxiter": int(maxiter), "maxfun": int(maxiter * 6),
                                    "ftol": 1e-14, "gtol": 1e-9})
            y = res.x
            Yt = y.reshape(n, d)
            nt = np.linalg.norm(Yt, axis=1, keepdims=True)
            nt[nt < 1e-12] = 1.0
            y = (Yt / nt).ravel()
        except Exception:
            break
    Xf = y.reshape(n, d)
    Xf /= np.maximum(np.linalg.norm(Xf, axis=1, keepdims=True), 1e-12)
    return Xf

def solve():
    t0 = time.time()
    DEADLINE = t0 + 172.0
    SEARCH_END = t0 + 126.0
    rng = np.random.RandomState(987654321)
    n = 163
    best_x = None
    best_val = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=np.float64)
            if a.shape != (n, 4) or not np.all(np.isfinite(a)):
                continue
            a /= np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-12)
            v = _max_cos(a)
            if v < best_val:
                best_val = v
                best_x = a.copy()
        except Exception:
            continue
    if best_x is None:
        best_x = rng.randn(n, 4)
        best_x /= np.linalg.norm(best_x, axis=1, keepdims=True)
        best_val = _max_cos(best_x)
    elite = [(best_val, best_x.copy())]
    trial = 0
    iu0, iu1 = np.triu_indices(n, 1)
    while time.time() < SEARCH_END:
        if time.time() > DEADLINE - 46:
            break
        trial += 1
        if len(elite) > 1 and rng.rand() < 0.30:
            base = elite[rng.randint(len(elite))][1]
        else:
            base = best_x
        op = rng.randint(7)
        try:
            if op == 0:
                sigma = [0.02, 0.05, 0.12][trial % 3]
                cur = base + rng.randn(n, 4) * sigma
                cur /= np.linalg.norm(cur, axis=1, keepdims=True)
                cur_b, cur_v = _gd_softmax(cur, steps=4200, T0=0.011, T1=0.00055, lr0=0.038, mu=0.86, rng=rng, noise0=0.002, deadline=SEARCH_END)
            elif op == 1:
                G = base @ base.T
                np.fill_diagonal(G, -2.0)
                m = float(G.max())
                deg = (G > m - 0.0025).sum(axis=1)
                krep = 2 + (trial % 5)
                worst = np.argsort(deg)[-krep:]
                keep = np.ones(n, dtype=bool); keep[worst] = False
                fixed = base[keep].copy()
                holes = _greedy_holes(fixed, krep, rng, n_samples=9000)
                cur = base.copy()
                for wi, h in zip(worst, holes):
                    cur[wi] = h
                cur = cur + rng.randn(n, 4) * 0.004
                cur /= np.linalg.norm(cur, axis=1, keepdims=True)
                mid, _ = _gd_riesz(cur, s0=10.0, s1=28.0, steps=1200, lr0=0.06, mu=0.80, deadline=SEARCH_END)
                cur_b, cur_v = _gd_softmax(mid, steps=3400, T0=0.009, T1=0.0005, lr0=0.034, mu=0.87, rng=rng, noise0=0.001, deadline=SEARCH_END)
            elif op == 2:
                u = rng.randn(4); u /= np.linalg.norm(u)
                dots = base @ u
                q = 0.82 + 0.10 * rng.rand()
                thresh = np.quantile(dots, q)
                idx = np.where(dots > thresh)[0]
                if len(idx) < 3:
                    idx = np.argsort(dots)[-15:]
                cur = base.copy()
                cur[idx] += rng.randn(len(idx), 4) * (0.20 + 0.15 * rng.rand())
                cur /= np.linalg.norm(cur, axis=1, keepdims=True)
                cur_b, cur_v = _gd_softmax(cur, steps=4000, T0=0.014, T1=0.0007, lr0=0.040, mu=0.84, rng=rng, noise0=0.003, deadline=SEARCH_END)
            elif op == 3:
                G = base @ base.T
                np.fill_diagonal(G, -2.0)
                c = G[iu0, iu1]
                top = np.argsort(c)[-24:]
                cur = base.copy()
                touched = set()
                for tt in top:
                    i = int(iu0[tt]); j = int(iu1[tt])
                    pick = i if rng.rand() < 0.5 else j
                    if pick in touched:
                        continue
                    touched.add(pick)
                    ci = float(c[np.where((iu0 == pick) | (iu1 == pick))[0]].max()) if False else 0.0
                    other = j if pick == i else i
                    # deterministic separation + random orthogonal
                    gvec = cur[other] - float(cur[pick] @ cur[other]) * cur[pick]
                    ng = float(np.linalg.norm(gvec))
                    if ng > 1e-12:
                        gvec /= ng
                        cur[pick] = cur[pick] - gvec * (0.06 + 0.06 * rng.rand())
                    dvec = rng.randn(4)
                    dvec -= (dvec @ cur[pick]) * cur[pick]
                    dvec /= max(np.linalg.norm(dvec), 1e-12)
                    cur[pick] = cur[pick] + dvec * (0.06 + 0.08 * rng.rand())
                    if len(touched) >= 10:
                        break
                cur /= np.linalg.norm(cur, axis=1, keepdims=True)
                cur_b, cur_v = _gd_softmax(cur, steps=4200, T0=0.010, T1=0.0005, lr0=0.036, mu=0.87, rng=rng, noise0=0.002, deadline=SEARCH_END)
            elif op == 4:
                G = base @ base.T
                np.fill_diagonal(G, -2.0)
                m = float(G.max())
                deg = (G > m - 1e-9).sum(axis=1)
                hub = np.argsort(deg)[-8:]
                cur = base.copy()
                cur[hub] += rng.randn(len(hub), 4) * 0.14
                ksub = rng.choice(n, size=16, replace=False)
                cur[ksub] += rng.randn(len(ksub), 4) * 0.05
                cur /= np.linalg.norm(cur, axis=1, keepdims=True)
                mid, _ = _gd_riesz(cur, s0=6.0, s1=22.0, steps=1000, lr0=0.05, mu=0.82, deadline=SEARCH_END)
                cur_b, cur_v = _gd_softmax(mid, steps=3400, T0=0.012, T1=0.0006, lr0=0.038, mu=0.85, rng=rng, noise0=0.002, deadline=SEARCH_END)
            elif op == 5:
                cur = base + rng.randn(n, 4) * 0.09
                cur /= np.linalg.norm(cur, axis=1, keepdims=True)
                mid, _ = _gd_riesz(cur, s0=14.0, s1=32.0, steps=1400, lr0=0.065, mu=0.80, deadline=SEARCH_END)
                cur_b, cur_v = _gd_softmax(mid, steps=3200, T0=0.008, T1=0.00045, lr0=0.032, mu=0.88, rng=rng, noise0=0.001, deadline=SEARCH_END)
            else:
                if rng.rand() < 0.5:
                    cur = rng.randn(n, 4)
                    cur /= np.linalg.norm(cur, axis=1, keepdims=True)
                    cur_b, cur_v = _gd_softmax(cur, steps=5000, T0=0.016, T1=0.0006, lr0=0.042, mu=0.85, rng=rng, noise0=0.003, deadline=SEARCH_END)
                else:
                    cur = base + rng.randn(n, 4) * 0.22
                    cur /= np.linalg.norm(cur, axis=1, keepdims=True)
                    cur_b, cur_v = _gd_softmax(cur, steps=4200, T0=0.015, T1=0.0007, lr0=0.041, mu=0.84, rng=rng, noise0=0.003, deadline=SEARCH_END)
        except Exception:
            continue
        if not np.all(np.isfinite(cur_b)):
            continue
        try:
            tmpn = np.linalg.norm(cur_b, axis=1)
            if np.any(tmpn < 1e-9) or not np.isfinite(cur_v):
                continue
        except Exception:
            continue
        if cur_v < best_val - 1e-12:
            # polish promising jumps
            if time.time() < DEADLINE - 42:
                try:
                    pol, pv = _lp_polish(cur_b, deadline=min(time.time() + 2.0, DEADLINE - 36))
                    if pv < cur_v:
                        cur_b, cur_v = pol, pv
                except Exception:
                    pass
            best_val = cur_v
            best_x = cur_b.copy()
            elite.append((best_val, best_x.copy()))
            elite = sorted(elite, key=lambda z: z[0])[:6]
        elif cur_v < best_val + 3e-05 and rng.rand() < 0.08 and time.time() < DEADLINE - 44:
            try:
                pol, pv = _lp_polish(cur_b, deadline=min(time.time() + 1.5, DEADLINE - 38))
                if pv < best_val - 1e-12:
                    best_val = pv
                    best_x = pol.copy()
                    elite.append((best_val, best_x.copy()))
                    elite = sorted(elite, key=lambda z: z[0])[:6]
                elif pv < cur_v and rng.rand() < 0.3:
                    elite.append((pv, pol.copy()))
                    elite = sorted(elite, key=lambda z: z[0])[:8]
                    elite = elite[:6]
            except Exception:
                pass
    # final intensive polish
    try:
        if time.time() < DEADLINE - 5:
            pol, pv = _lp_polish(best_x, deadline=min(time.time() + 12.0, DEADLINE - 14))
            if pv < best_val:
                best_val = pv
                best_x = pol
    except Exception:
        pass
    try:
        if time.time() < DEADLINE - 8:
            pol2 = _lbfgs_smooth(best_x, deadline=min(time.time() + 10.0, DEADLINE - 6))
            pv2 = _max_cos(pol2)
            if pv2 < best_val:
                best_val = pv2
                best_x = pol2
    except Exception:
        pass
    try:
        if time.time() < DEADLINE - 2:
            pol, pv = _lp_polish(best_x, deadline=DEADLINE - 1.0)
            if pv < best_val:
                best_val = pv
                best_x = pol
    except Exception:
        pass
    best_x = np.asarray(best_x, dtype=np.float64)
    best_x /= np.maximum(np.linalg.norm(best_x, axis=1, keepdims=True), 1e-12)
    return best_x
