import time
import numpy as np
from scipy.optimize import linprog, minimize
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _lp_polish(x, deadline, r0=1e-3):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
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

def _gd(x0, steps, T0, T1, lr0, mu, rng, noise0=0.0, deadline=None):
    x = x0.copy()
    n, d = x.shape
    v = np.zeros_like(x)
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    best_val = float(G.max())
    best = x.copy()
    if T0 > 0 and T1 > 0:
        lratio = np.log(T1 / T0)
    else:
        lratio = 0.0
    for k in range(steps):
        if deadline is not None and (k & 255) == 0 and time.time() > deadline:
            break
        frac = k / max(1, steps - 1)
        T = T0 * np.exp(lratio * frac)
        if T < 1e-12:
            T = 1e-12
        lr = lr0 * (1.0 - 0.55 * frac)
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

def _lbfgs_smooth(x_init, deadline, taus=(0.004, 0.0015, 0.0006), iters=(120, 120, 180)):
    x = x_init / np.maximum(np.linalg.norm(x_init, axis=1, keepdims=True), 1e-12)
    n, d = x.shape
    iu0, iu1 = np.triu_indices(n, 1)
    Np = len(iu0)
    y = x.ravel().copy()
    for tau, maxiter in zip(taus, iters):
        if time.time() > deadline - 0.3:
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
                           options={"maxiter": int(maxiter), "maxfun": int(maxiter * 5),
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

def _find_hole(x_fixed, rng, n_samples=9000):
    Y = rng.randn(n_samples, 4)
    Y /= np.maximum(np.linalg.norm(Y, axis=1, keepdims=True), 1e-12)
    D = Y @ x_fixed.T
    vals = D.max(axis=1)
    idx = int(np.argmin(vals))
    y = Y[idx].copy()
    vy = float(vals[idx])
    for _ in range(200):
        dots = x_fixed @ y
        j = int(np.argmax(dots))
        g = x_fixed[j] - dots[j] * y
        ng = float(np.linalg.norm(g))
        if ng < 1e-12:
            break
        g /= ng
        improved = False
        for step in (0.02, 0.005):
            yn = y - step * g
            yn /= np.linalg.norm(yn)
            vn = float((x_fixed @ yn).max())
            if vn < vy:
                y, vy = yn, vn
                improved = True
                break
        if not improved:
            break
    return y, vy

def solve():
    t0 = time.time()
    DEADLINE = t0 + 172.0
    SEARCH_END = t0 + 130.0
    rng = np.random.RandomState(20261102)
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
    trial = 0
    sig_cycle = [0.02, 0.06, 0.15, 0.30]
    while time.time() < SEARCH_END:
        if time.time() > DEADLINE - 38:
            break
        trial += 1
        mode = trial % 7
        if mode == 0:
            cur = rng.randn(n, 4)
            cur /= np.linalg.norm(cur, axis=1, keepdims=True)
            cur_b, cur_v = _gd(cur, steps=5500, T0=0.018, T1=0.0007, lr0=0.045, mu=0.85, rng=rng, noise0=0.004, deadline=SEARCH_END)
        elif mode == 1:
            sigma = sig_cycle[(trial // 7) % 4]
            cur = best_x + rng.randn(n, 4) * sigma
            cur /= np.linalg.norm(cur, axis=1, keepdims=True)
            cur_b, cur_v = _gd(cur, steps=4500, T0=0.010, T1=0.0006, lr0=0.035, mu=0.88, rng=rng, noise0=0.002, deadline=SEARCH_END)
        elif mode == 2:
            mrep = 2 + (trial % 5)
            G = best_x @ best_x.T
            np.fill_diagonal(G, -2.0)
            m = float(G.max())
            deg = (G > m - 0.0025).sum(axis=1)
            worst = np.argsort(deg)[-mrep:]
            keep = np.ones(n, dtype=bool); keep[worst] = False
            fixed = best_x[keep].copy()
            cur = best_x.copy()
            for w in worst:
                h, _ = _find_hole(fixed, rng, n_samples=8000)
                cur[w] = h
                fixed = np.vstack([fixed, h.reshape(1, 4)])
            cur = cur + rng.randn(n, 4) * 0.005
            cur /= np.linalg.norm(cur, axis=1, keepdims=True)
            cur_b, cur_v = _gd(cur, steps=5000, T0=0.012, T1=0.0006, lr0=0.040, mu=0.86, rng=rng, noise0=0.001, deadline=SEARCH_END)
        elif mode == 3:
            u = rng.randn(4); u /= np.linalg.norm(u)
            dots = best_x @ u
            thresh = np.quantile(dots, 0.88)
            idx = np.where(dots > thresh)[0]
            cur = best_x.copy()
            cur[idx] += rng.randn(len(idx), 4) * 0.25
            cur /= np.linalg.norm(cur, axis=1, keepdims=True)
            cur_b, cur_v = _gd(cur, steps=4000, T0=0.014, T1=0.0008, lr0=0.040, mu=0.84, rng=rng, noise0=0.003, deadline=SEARCH_END)
        elif mode == 4:
            G = best_x @ best_x.T
            np.fill_diagonal(G, -2.0)
            m = float(G.max())
            deg = (G > m - 1e-9).sum(axis=1)
            hub = np.argsort(deg)[-6:]
            cur = best_x.copy()
            rattle = np.where(deg == 0)[0]
            if len(rattle) > 0:
                fixed = np.delete(cur, rattle, axis=0)
                for w in rattle:
                    h, _ = _find_hole(fixed, rng, n_samples=8000)
                    cur[w] = h
                    fixed = np.vstack([fixed, h.reshape(1, 4)])
            cur[hub] += rng.randn(len(hub), 4) * 0.12
            ksub = rng.choice(n, size=min(20, n), replace=False)
            cur[ksub] += rng.randn(len(ksub), 4) * 0.05
            cur /= np.linalg.norm(cur, axis=1, keepdims=True)
            cur_b, cur_v = _gd(cur, steps=4500, T0=0.009, T1=0.0005, lr0=0.032, mu=0.88, rng=rng, noise0=0.0015, deadline=SEARCH_END)
        elif mode == 5:
            G = best_x @ best_x.T
            np.fill_diagonal(G, -2.0)
            iu0, iu1 = np.triu_indices(n, 1)
            c = G[iu0, iu1]
            top = np.argsort(c)[-20:]
            cur = best_x.copy()
            touched = set()
            for t in top:
                i = int(iu0[t]); j = int(iu1[t])
                pick = i if rng.rand() < 0.5 else j
                if pick in touched:
                    continue
                touched.add(pick)
                dvec = rng.randn(4)
                dvec -= (dvec @ cur[pick]) * cur[pick]
                dvec /= max(np.linalg.norm(dvec), 1e-12)
                cur[pick] = cur[pick] + dvec * (0.10 + 0.10 * rng.rand())
            cur /= np.linalg.norm(cur, axis=1, keepdims=True)
            cur_b, cur_v = _gd(cur, steps=4500, T0=0.011, T1=0.0006, lr0=0.038, mu=0.86, rng=rng, noise0=0.002, deadline=SEARCH_END)
        else:
            cur = best_x + rng.randn(n, 4) * 0.10
            cur /= np.linalg.norm(cur, axis=1, keepdims=True)
            A = rng.randn(4, 4) * 0.05
            A = A - A.T
            Q, _ = np.linalg.qr(np.eye(4) + A)
            cur = cur @ Q.T
            cur /= np.linalg.norm(cur, axis=1, keepdims=True)
            cur_b, cur_v = _gd(cur, steps=4000, T0=0.016, T1=0.0007, lr0=0.042, mu=0.85, rng=rng, noise0=0.003, deadline=SEARCH_END)
        if cur_v < best_val - 1e-12:
            if cur_v < best_val - 3e-5 and time.time() < DEADLINE - 40:
                try:
                    pol, pv = _lp_polish(cur_b, deadline=min(time.time() + 2.5, DEADLINE - 30))
                except Exception:
                    pol, pv = cur_b, cur_v
                try:
                    if pv < best_val and time.time() < DEADLINE - 38:
                        pol2 = _lbfgs_smooth(pol, deadline=min(time.time() + 3.0, DEADLINE - 32), taus=(0.002, 0.0007), iters=(120, 150))
                        pv2 = _max_cos(pol2)
                        if pv2 < pv:
                            pol, pv = pol2, pv2
                except Exception:
                    pass
                if pv < best_val:
                    best_x, best_val = pol, pv
                elif cur_v < best_val:
                    best_x, best_val = cur_b, cur_v
            else:
                best_x, best_val = cur_b, cur_v
        elif cur_v < best_val + 0.0006 and (trial % 4 == 0) and time.time() < SEARCH_END - 8:
            try:
                pol = _lbfgs_smooth(cur_b, deadline=time.time() + 2.0, taus=(0.002, 0.0007), iters=(80, 100))
                pv = _max_cos(pol)
                if pv < best_val - 1e-12:
                    try:
                        pol2, pv2 = _lp_polish(pol, deadline=min(time.time() + 2.0, DEADLINE - 30))
                        if pv2 < pv:
                            pol, pv = pol2, pv2
                    except Exception:
                        pass
                    best_x, best_val = pol, pv
            except Exception:
                pass
        if best_val < 0.84:
            break
    remain = DEADLINE - time.time() - 1.5
    if remain > 4:
        try:
            pol = _lbfgs_smooth(best_x, deadline=DEADLINE - 8, taus=(0.003, 0.0012, 0.0005), iters=(150, 150, 200))
            pv = _max_cos(pol)
            if pv < best_val:
                best_x, best_val = pol, pv
        except Exception:
            pass
    remain = DEADLINE - time.time() - 1.0
    if remain > 3:
        try:
            px, pv = _lp_polish(best_x, deadline=DEADLINE - 0.8)
            if pv < best_val:
                best_x, best_val = px, pv
            elif pv <= best_val + 1e-12:
                best_x = px
        except Exception:
            pass
    best_x = best_x / np.maximum(np.linalg.norm(best_x, axis=1, keepdims=True), 1e-12)
    return best_x
