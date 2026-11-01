import time
import numpy as np
from scipy.optimize import linprog, minimize
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(np.max(g))

def _per_max(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return np.max(g, axis=1)

def _anneal_lse(x, steps, b0, b1, lr0, lr1, mu, rng, noise=0.0):
    n, d = x.shape
    vel = np.zeros_like(x)
    lb = np.log(b1 / b0) if b0 > 0 and b1 != b0 else 0.0
    for s in range(steps):
        t = s / max(steps - 1, 1)
        beta = b0 * np.exp(lb * t) if lb != 0.0 else b0
        lr = lr0 + (lr1 - lr0) * t
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        m = float(np.max(G))
        W = np.exp(beta * (G - m))
        np.fill_diagonal(W, 0.0)
        Grad = W @ x
        proj = np.sum(Grad * x, axis=1, keepdims=True)
        Tg = Grad - proj * x
        rms = np.sqrt(np.mean(Tg * Tg)) + 1e-12
        step = Tg * (lr / rms)
        if mu > 0:
            vel = mu * vel + step
            step = vel
        x = x - step
        if noise > 0:
            ns = noise * (1.0 - t)
            nz = rng.standard_normal(size=x.shape)
            nz -= np.sum(nz * x, axis=1, keepdims=True) * x
            nn = np.linalg.norm(nz, axis=1, keepdims=True) + 1e-12
            x = x + ns * (nz / nn)
        x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x

def _anneal_riesz(x, steps, p0, p1, lr0, lr1, mu, rng):
    n, d = x.shape
    vel = np.zeros_like(x)
    for s in range(steps):
        t = s / max(steps - 1, 1)
        p = p0 + (p1 - p0) * t
        lr = lr0 + (lr1 - lr0) * t
        G = x @ x.T
        np.clip(G, -1.0, 0.999999, out=G)
        np.fill_diagonal(G, 0.0)
        D = 2.0 - 2.0 * G
        np.fill_diagonal(D, 1.0)
        W = np.power(D, -p)
        np.fill_diagonal(W, 0.0)
        Grad = W @ x
        proj = np.sum(Grad * x, axis=1, keepdims=True)
        Tg = Grad - proj * x
        rms = np.sqrt(np.mean(Tg * Tg)) + 1e-12
        step = Tg * (lr / rms)
        if mu > 0:
            vel = mu * vel + step
            step = vel
        x = x - step
        x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x

def _best_hole(rest, rng, ncand=12000):
    d = rest.shape[1]
    cand = rng.standard_normal((ncand, d))
    cand /= np.linalg.norm(cand, axis=1, keepdims=True)
    scores = np.max(cand @ rest.T, axis=1)
    return cand[int(np.argmin(scores))]

def _sweep_worst(x, rng, k=4, iters=180, beta=300.0, lr=0.03):
    n, d = x.shape
    pm = _per_max(x)
    order = np.argsort(-pm)[:k]
    for idx in order:
        mask = np.ones(n, dtype=bool); mask[idx] = False
        Xr = x[mask]
        y = x[idx].copy()
        for s in range(iters):
            sc = Xr @ y
            m = float(np.max(sc))
            w = np.exp(beta * (sc - m))
            w = w / (np.sum(w) + 1e-300)
            gy = Xr.T @ w
            gy = gy - float(gy @ y) * y
            rn = np.linalg.norm(gy) + 1e-12
            y = y - (lr / rn) * gy if rn > 1e-12 else y
            y /= np.linalg.norm(y) + 1e-300
        # accept if improves global max
        old = x.copy()
        x[idx] = y
        # keep only if per-point improves (avoid hurting others too much)
        if _max_cos(x) > _max_cos(old) + 2e-4:
            x[idx] = old[idx]
    return x

def _cap_kick(X, rng):
    n, d = X.shape
    u = rng.standard_normal(d); u /= np.linalg.norm(u)
    thr = float(rng.uniform(0.35, 0.80))
    dots = X @ u
    cap = np.where(dots > thr)[0]
    if len(cap) < 2 or len(cap) > 20:
        # fallback: kick worst point neighbourhood
        pm = _per_max(X)
        i0 = int(np.argmax(pm))
        dists = X @ X[i0]
        cap = np.where(dists > float(rng.uniform(0.3, 0.6)))[0]
        if len(cap) < 2 or len(cap) > 20:
            cap = np.array([i0])
    v = rng.standard_normal(d)
    v = v - float(v @ u) * u
    v /= np.linalg.norm(v) + 1e-12
    alpha = float(rng.uniform(0.05, 0.25))
    sigma = float(rng.uniform(0.005, 0.04))
    X2 = X.copy()
    X2[cap] = X2[cap] + alpha * v + sigma * rng.standard_normal((len(cap), d))
    # counter-move rest slightly
    rest = np.ones(n, dtype=bool); rest[cap] = False
    X2[rest] = X2[rest] - 0.15 * alpha * v / max(np.sum(rest), 1)
    X2 /= np.linalg.norm(X2, axis=1, keepdims=True)
    return X2

def _lp_polish(x, deadline, cap_p=1600):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    iu = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        if time.time() > deadline - 0.4:
            break
        c = (x @ x.T)[iu]
        m = float(np.max(c))
        sel = c > m - max(20 * radius, 1e-9)
        I = iu[0][sel]; J = iu[1][sel]
        p = len(I)
        if p == 0:
            radius *= 0.5
            continue
        csel = c[sel]
        if p > cap_p:
            order = np.argsort(-csel)[:cap_p]
            I, J = I[order], J[order]
            csel = csel[order]
            p = cap_p
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        v = _max_cos(y)
        if v < best:
            x, best, radius = y, v, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def _lbfgs_lse(x0, beta, maxiter):
    n, d = x0.shape
    IU = np.triu_indices(n, 1)
    def fg(yf):
        Y = yf.reshape(n, d)
        nr = np.linalg.norm(Y, axis=1, keepdims=True)
        nr = np.maximum(nr, 1e-12)
        X = Y / nr
        G = X @ X.T
        g = G[IU]
        m = float(g.max())
        e = np.exp(beta * (g - m))
        S = float(e.sum()) + 1e-300
        F = m + np.log(S) / beta
        w = e / S
        P = np.zeros((n, n))
        P[IU] = w
        P = P + P.T
        GX = P @ X
        dot = np.sum(GX * X, axis=1, keepdims=True)
        GY = (GX - dot * X) / nr
        return F, GY.ravel()
    y0 = (x0 / np.linalg.norm(x0, axis=1, keepdims=True)).ravel().copy()
    try:
        r = minimize(fg, y0, jac=True, method='L-BFGS-B',
                     options={'maxiter': maxiter, 'ftol': 1e-14, 'gtol': 1e-7, 'maxfun': maxiter * 2})
        Y = r.x.reshape(n, d)
        Y /= np.linalg.norm(Y, axis=1, keepdims=True)
        return Y
    except Exception:
        return x0

def solve():
    t0 = time.time()
    T_END = t0 + 170.0
    rng = np.random.default_rng(0)
    n, d = 68, 12
    bestX = None
    best = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.asarray(np.load(fn), dtype=np.float64)
            if a.shape == (n, d):
                a = a / np.linalg.norm(a, axis=1, keepdims=True)
                s = _max_cos(a)
                if s < best and np.all(np.isfinite(a)):
                    best = s
                    bestX = a.copy()
        except Exception:
            pass
    if bestX is None:
        bestX = rng.standard_normal((n, d))
        bestX /= np.linalg.norm(bestX, axis=1, keepdims=True)
        best = _max_cos(bestX)
    secondX = bestX.copy()
    second = best

    # diverse screening: random + D12 lattice starts
    # build D12 minimal vectors
    Dvec = []
    for i in range(d):
        for j in range(i + 1, d):
            for si in (1.0, -1.0):
                for sj in (1.0, -1.0):
                    v = np.zeros(d); v[i] = si; v[j] = sj; v /= np.sqrt(2.0)
                    Dvec.append(v)
    Dvec = np.array(Dvec)

    screen_pool = [bestX.copy()]
    # 6 fresh random
    for _ in range(6):
        q = rng.standard_normal((n, d))
        q /= np.linalg.norm(q, axis=1, keepdims=True)
        screen_pool.append(q)
    # 2 lattice-sampled
    for _ in range(2):
        idx = rng.choice(len(Dvec), size=n, replace=False)
        q = Dvec[idx] + 0.08 * rng.standard_normal((n, d))
        q /= np.linalg.norm(q, axis=1, keepdims=True)
        screen_pool.append(q)

    screened = []
    for q in screen_pool:
        if time.time() > t0 + 30.0:
            break
        # fast anneal with random schedule to land in different basin
        b0 = float(rng.uniform(25, 60)); b1 = float(rng.uniform(250, 500))
        q2 = _anneal_lse(q.copy(), 500, b0, b1, 0.04, 0.004, 0.6, rng, noise=0.002)
        q2 = _anneal_riesz(q2, 150, 1.2, 3.0, 0.03, 0.004, 0.5, rng)
        v = _max_cos(q2)
        screened.append((v, q2))
    screened.sort(key=lambda t: t[0])
    # adopt if screening found better (unlikely) else keep diversity
    if screened and screened[0][0] < best:
        best = screened[0][0]; bestX = screened[0][1].copy()
    # second basin = best screened distinct from bestX
    for v, q in screened[1:]:
        # distinctness: mean abs corr low
        if v < second + 0.01 or second >= best - 1e-12:
            secondX = q.copy(); second = v
            break

    # basin hopping with surgery
    it = 0
    explore_end = T_END - 26.0
    while time.time() < explore_end:
        it += 1
        # choose base
        u = rng.random()
        if u < 0.65:
            base = bestX
        elif u < 0.85:
            base = secondX
        else:
            # fresh random restart occasionally
            base = rng.standard_normal((n, d))
            base /= np.linalg.norm(base, axis=1, keepdims=True)
        x = base.copy()
        mode = it % 5
        if mode == 0:
            # delete worst 1-3 and reinsert holes
            pm = _per_max(x)
            kdel = int(rng.integers(1, 4))
            order = np.argsort(-pm)[:kdel]
            keep = np.ones(n, dtype=bool); keep[order] = False
            rest = x[keep]
            newpts = []
            for _ in range(kdel):
                h = _best_hole(np.vstack([rest] + newpts) if newpts else rest, rng, ncand=12000)
                newpts.append(h[None, :])
            x[order] = np.vstack(newpts)
            x /= np.linalg.norm(x, axis=1, keepdims=True)
            x = _anneal_lse(x, int(rng.integers(450, 700)), 50, float(rng.uniform(350, 650)), 0.035, 0.003, 0.65, rng, noise=0.001)
        elif mode == 1:
            # coherent cap kick + riesz then lse
            x = _cap_kick(x, rng)
            x = _anneal_riesz(x, int(rng.integers(200, 350)), 1.0, 3.5, 0.03, 0.004, 0.6, rng)
            x = _anneal_lse(x, int(rng.integers(350, 550)), 60, float(rng.uniform(350, 600)), 0.03, 0.003, 0.6, rng)
        elif mode == 2:
            # heavy gaussian on rattlers / random subset + anneal
            pm = _per_max(x)
            m = float(np.max(pm))
            # rattler-like: low per_max gets bigger kick to explore; overloaded gets medium
            sig = float(rng.choice([0.01, 0.03, 0.07, 0.15]))
            x = x + sig * rng.standard_normal(x.shape)
            x /= np.linalg.norm(x, axis=1, keepdims=True)
            x = _anneal_lse(x, int(rng.integers(500, 750)), float(rng.uniform(30, 90)), float(rng.uniform(320, 650)), float(rng.uniform(0.02, 0.05)), float(rng.uniform(0.001, 0.005)), 0.7, rng, noise=0.001)
        elif mode == 3:
            # pure riesz escape then lse quench
            sig = float(rng.uniform(0.02, 0.09))
            x = x + sig * rng.standard_normal(x.shape)
            x /= np.linalg.norm(x, axis=1, keepdims=True)
            x = _anneal_riesz(x, int(rng.integers(300, 500)), 0.8, 2.5, 0.035, 0.004, 0.5, rng)
            x = _anneal_lse(x, 350, 80, 500, 0.03, 0.003, 0.6, rng)
        else:
            # targeted worst-point teleport to hole + short quench
            pm = _per_max(x)
            i0 = int(np.argmax(pm))
            mask = np.ones(n, dtype=bool); mask[i0] = False
            h = _best_hole(x[mask], rng, ncand=15000)
            # small jitter around hole
            h = h + 0.01 * rng.standard_normal(d)
            h /= np.linalg.norm(h)
            x[i0] = h
            x = _anneal_lse(x, int(rng.integers(400, 600)), 60, 500, 0.03, 0.003, 0.6, rng)

        # single-point sweep on worst few
        if rng.random() < 0.6:
            x = _sweep_worst(x, rng, k=int(rng.integers(2, 5)), iters=120, beta=float(rng.uniform(200, 450)), lr=0.03)
        v = _max_cos(x)
        if v < best - 1e-12:
            best = v; bestX = x.copy()
        elif v < second - 1e-12 and v > best + 5e-6:
            # keep diverse second basin (not too close to best)
            diff = float(np.mean((x - bestX) ** 2))
            if diff > 1e-5:
                second = v; secondX = x.copy()
        # sideways: occasionally replace second even if slightly worse to keep moving
        elif rng.random() < 0.05 and v < second + 0.002:
            secondX = x.copy(); second = v

    # final intensification: L-BFGS + LP on best two
    for cand, val in sorted([(bestX, best), (secondX, second)], key=lambda t: _max_cos(t[0])):
        if time.time() > T_END - 24.0:
            break
        try:
            y = _lbfgs_lse(cand.copy(), 220.0, 120)
            y = _lbfgs_lse(y, 550.0, 120)
        except Exception:
            y = cand.copy()
        vy = _max_cos(y)
        if vy < best:
            best = vy; bestX = y.copy()
        elif time.time() < T_END - 20.0:
            pass
        break

    # trust-region LP squeeze on champion
    try:
        bestX = _lp_polish(bestX, T_END - 1.0)
        best = _max_cos(bestX)
    except Exception:
        pass
    bestX = bestX / np.linalg.norm(bestX, axis=1, keepdims=True)
    return bestX
