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
                sel[int(np.argmax(c))] = True
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
    lratio = np.log(T1 / T0) if T0 > 0 and T1 > 0 else 0.0
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

def _lbfgs_smooth(x_init, deadline, taus=(0.004, 0.0012, 0.0005), iters=(100, 100, 150)):
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
            rn = np.maximum(np.linalg.norm(Y, axis=1, keepdims=True), 1e-12)
            X = Y / rn
            g = (X @ X.T)[iu0, iu1]
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
            GY = (GX - (GX * X).sum(axis=1, keepdims=True) * X) / rn
            return float(L), GY.ravel()
        try:
            res = minimize(fg, y, method="L-BFGS-B", jac=True,
                           options={"maxiter": int(maxiter), "maxfun": int(maxiter * 4),
                                    "ftol": 1e-14, "gtol": 1e-9})
            y = res.x
        except Exception:
            break
    Yf = y.reshape(n, d)
    Yf /= np.maximum(np.linalg.norm(Yf, axis=1, keepdims=True), 1e-12)
    return Yf, _max_cos(Yf)

def _find_hole(x_fixed, rng, n_samples=8000):
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
        for step in (0.02, 0.005):
            yn = y - step * g / ng
            yn /= np.linalg.norm(yn)
            vn = float((x_fixed @ yn).max())
            if vn < vy:
                y, vy = yn, vn
                break
        else:
            break
    return y, vy

def _rand_rot(rng, scale):
    A = rng.randn(4, 4) * scale
    A = A - A.T
    # exp via Cayley-ish: use QR of I+A
    Q, _ = np.linalg.qr(np.eye(4) + A)
    return Q

def solve():
    t0 = time.time()
    DEADLINE = t0 + 172.0
    SEARCH_END = t0 + 128.0
    rng = np.random.RandomState(987654321)
    n = 163
    try:
        bx = np.load("best.npy")
    except Exception:
        try:
            bx = np.load("record.npy")
        except Exception:
            bx = rng.randn(n, 4)
    bx = np.asarray(bx, dtype=float)
    if bx.shape != (n, 4) or not np.all(np.isfinite(bx)):
        bx = rng.randn(n, 4)
    bx = bx / np.maximum(np.linalg.norm(bx, axis=1, keepdims=True), 1e-12)
    best_x = bx.copy()
    best_val = _max_cos(best_x)
    archive = [(best_val, best_x.copy())]

    def base_pick(t):
        if len(archive) == 1 or (t % 5) != 4:
            return best_x
        # diverse pick: second best
        k = 1 + (t // 5) % (len(archive) - 1)
        return archive[k][1]

    trial = 0
    while time.time() < SEARCH_END:
        if time.time() > DEADLINE - 42:
            break
        trial += 1
        base = base_pick(trial)
        mode = trial % 7
        if mode == 0:
            cur = rng.randn(n, 4)
            cur /= np.linalg.norm(cur, axis=1, keepdims=True)
            cur_b, cur_v = _gd(cur, steps=5000, T0=0.018, T1=0.0007, lr0=0.045, mu=0.85, rng=rng, noise0=0.004, deadline=SEARCH_END)
        elif mode == 1:
            sigmas = [0.02, 0.06, 0.15, 0.30]
            sigma = sigmas[(trial // 7) % 4]
            cur = base + rng.randn(n, 4) * sigma
            cur /= np.linalg.norm(cur, axis=1, keepdims=True)
            cur_b, cur_v = _gd(cur, steps=4200, T0=0.010, T1=0.0006, lr0=0.035, mu=0.88, rng=rng, noise0=0.002, deadline=SEARCH_END)
        elif mode == 2:
            # rigid cap rotation: preserves intra-cap structure, breaks cross contacts
            u = rng.randn(4); u /= np.linalg.norm(u)
            dots = base @ u
            q = 0.82 + 0.10 * rng.rand()
            thresh = np.quantile(dots, q)
            idx = np.where(dots > thresh)[0]
            if len(idx) < 3:
                idx = np.argsort(dots)[-12:]
            R = _rand_rot(rng, 0.10 + 0.25 * rng.rand())
            cur = base.copy()
            cur[idx] = (R @ cur[idx].T).T
            # small jitter to allow relaxation
            cur += rng.randn(n, 4) * 0.004
            cur /= np.linalg.norm(cur, axis=1, keepdims=True)
            cur_b, cur_v = _gd(cur, steps=4500, T0=0.013, T1=0.0006, lr0=0.040, mu=0.86, rng=rng, noise0=0.0015, deadline=SEARCH_END)
        elif mode == 3:
            # avalanche: hub + its contact neighbours shaken together then pushed apart
            G = base @ base.T
            np.fill_diagonal(G, -2.0)
            m = float(G.max())
            deg = (G > m - 0.0025).sum(axis=1)
            hub = int(np.argmax(deg))
            nbrs = np.where(G[hub] > m - 0.008)[0]
            if len(nbrs) > 12:
                nbrs = nbrs[np.argsort(G[hub][nbrs])[-12:]]
            group = np.unique(np.concatenate([[hub], nbrs]))
            cur = base.copy()
            # pair-break: push hub away from its worst partner
            j = int(np.argmax(G[hub]))
            away = cur[hub] - cur[j]
            away -= (away @ cur[hub]) * cur[hub]
            na = np.linalg.norm(away)
            if na > 1e-12:
                cur[hub] += (away / na) * (0.10 + 0.10 * rng.rand())
            cur[group] += rng.randn(len(group), 4) * 0.10
            cur += rng.randn(n, 4) * 0.004
            cur /= np.linalg.norm(cur, axis=1, keepdims=True)
            cur_b, cur_v = _gd(cur, steps=4500, T0=0.011, T1=0.0005, lr0=0.038, mu=0.87, rng=rng, noise0=0.002, deadline=SEARCH_END)
        elif mode == 4:
            # surgery: remove worst-loaded cluster, reinsert into holes
            G = base @ base.T
            np.fill_diagonal(G, -2.0)
            m = float(G.max())
            deg = (G > m - 0.0025).sum(axis=1)
            mrep = 2 + (trial % 5)
            worst = np.argsort(deg)[-mrep:]
            keep = np.ones(n, dtype=bool); keep[worst] = False
            fixed = base[keep].copy()
            cur = base.copy()
            for w in worst:
                h, _ = _find_hole(fixed, rng, n_samples=7000)
                cur[w] = h
                fixed = np.vstack([fixed, h.reshape(1, 4)])
            cur += rng.randn(n, 4) * 0.004
            cur /= np.linalg.norm(cur, axis=1, keepdims=True)
            cur_b, cur_v = _gd(cur, steps=4800, T0=0.012, T1=0.0006, lr0=0.040, mu=0.86, rng=rng, noise0=0.001, deadline=SEARCH_END)
        elif mode == 5:
            # global small rotation + medium shake
            cur = base + rng.randn(n, 4) * 0.10
            cur /= np.linalg.norm(cur, axis=1, keepdims=True)
            R = _rand_rot(rng, 0.05)
            cur = (cur @ R.T)
            cur /= np.linalg.norm(cur, axis=1, keepdims=True)
            cur_b, cur_v = _gd(cur, steps=3800, T0=0.016, T1=0.0007, lr0=0.042, mu=0.85, rng=rng, noise0=0.003, deadline=SEARCH_END)
        else:
            # antipodal-band shake: pick random great circle band and twist
            u = rng.randn(4); u /= np.linalg.norm(u)
            dots = np.abs(base @ u)
            idx = np.where(dots < 0.25)[0]
            if len(idx) < 5:
                idx = np.argsort(dots)[:20]
            R = _rand_rot(rng, 0.15 + 0.20 * rng.rand())
            cur = base.copy()
            cur[idx] = (R @ cur[idx].T).T
            cur[idx] += rng.randn(len(idx), 4) * 0.05
            cur /= np.linalg.norm(cur, axis=1, keepdims=True)
            cur_b, cur_v = _gd(cur, steps=4200, T0=0.014, T1=0.0007, lr0=0.040, mu=0.84, rng=rng, noise0=0.0025, deadline=SEARCH_END)

        # refine promising candidates with L-BFGS on smooth surrogate
        if cur_v < best_val + 0.004 and time.time() < SEARCH_END - 1.0:
            try:
                lb_x, lb_v = _lbfgs_smooth(cur_b, deadline=min(SEARCH_END, time.time() + 4.0))
                if lb_v < cur_v:
                    cur_b, cur_v = lb_x, lb_v
            except Exception:
                pass
        # short LP confirm if near/below best
        if cur_v < best_val - 2e-6 and time.time() < DEADLINE - 40:
            try:
                pol, pv = _lp_polish(cur_b, deadline=time.time() + 2.0)
                if pv < cur_v:
                    cur_b, cur_v = pol, pv
            except Exception:
                pass
        if cur_v < best_val - 1e-12:
            best_x, best_val = cur_b, cur_v
            archive.append((best_val, best_x.copy()))
            archive = sorted(archive, key=lambda z: z[0])[:5]
        elif cur_v < best_val + 8e-4:
            # keep diverse near-best
            if len(archive) < 5:
                archive.append((cur_v, cur_b.copy()))
                archive = sorted(archive, key=lambda z: z[0])[:5]
            elif cur_v < archive[-1][0] and rng.rand() < 0.3:
                archive[-1] = (cur_v, cur_b.copy())
                archive = sorted(archive, key=lambda z: z[0])
        if best_val < 0.84:
            break

    remain = DEADLINE - time.time() - 1.5
    if remain > 4:
        try:
            lb_x, lb_v = _lbfgs_smooth(best_x, deadline=DEADLINE - remain * 0.5)
            if lb_v < best_val:
                best_x, best_val = lb_x, lb_v
        except Exception:
            pass
    remain = DEADLINE - time.time() - 1.0
    if remain > 3:
        try:
            px, pv = _lp_polish(best_x, deadline=DEADLINE - 0.8, r0=1e-3)
            if pv <= best_val + 1e-12:
                best_x = px
                if pv < best_val:
                    best_val = pv
        except Exception:
            pass
    best_x = best_x / np.maximum(np.linalg.norm(best_x, axis=1, keepdims=True), 1e-12)
    return best_x
