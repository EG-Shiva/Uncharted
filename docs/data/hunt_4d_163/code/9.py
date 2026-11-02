import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def lp_polish(x, seconds):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = max_cos(x)
    radius, start = 2e-3, time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        c = (x @ x.T)[iu]
        m = float(c.max())
        sel = c > m - max(30.0 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p, nv = len(I), n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c[sel], A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        v = max_cos(y)
        if v < best:
            x, best, radius = y, v, min(radius * 1.6, 0.05)
        else:
            radius *= 0.5
    return x

def softmax_anneal(x0, steps, T0, T1, lr0, mu, noise_amp, rng):
    x = x0.copy()
    n, d = x.shape
    v = np.zeros_like(x)
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    best_val = float(G.max())
    best = x.copy()
    for k in range(steps):
        frac = k / max(1, steps - 1)
        T = T0 * (T1 / T0) ** frac
        lr = lr0 * (1.0 - 0.65 * frac)
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        E = np.exp((G - m) / T)
        np.fill_diagonal(E, 0.0)
        S = float(E.sum() * 0.5)
        if S < 1e-300:
            S = 1e-300
        grad = (E @ x) / S
        grad -= (np.sum(grad * x, axis=1, keepdims=True)) * x
        v = mu * v + (1.0 - mu) * grad
        x = x - lr * v
        if noise_amp > 0:
            N = rng.randn(n, d)
            N -= (np.sum(N * x, axis=1, keepdims=True)) * x
            # scale noise by temperature fraction
            x = x + noise_amp * (0.3 + 0.7 * (T / T0)) * N
        x /= np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
        if m < best_val:
            best_val = m
            best = x.copy()
    return best, best_val

def sample_hole(x_fixed, rng, n_samples=30000):
    Y = rng.randn(n_samples, 4)
    Y /= np.linalg.norm(Y, axis=1, keepdims=True)
    D = Y @ x_fixed.T
    vals = D.max(axis=1)
    idx = int(np.argmin(vals))
    return Y[idx], float(vals[idx])

def refine_free(Xf, Y0, steps, T0, T1, lr0, rng, noise=0.0):
    Y = Y0.copy()
    k = Y.shape[0]
    V = np.zeros_like(Y)
    mu = 0.85
    fixmax = None
    if Xf.shape[0] > 1:
        Gf = Xf @ Xf.T
        np.fill_diagonal(Gf, -2.0)
        fixmax = float(Gf.max())
    for s in range(steps):
        frac = s / max(1, steps - 1)
        T = T0 * (T1 / T0) ** frac
        lr = lr0 * (1.0 - 0.6 * frac)
        D = Y @ Xf.T  # k x m
        G = Y @ Y.T
        np.fill_diagonal(G, -2.0)
        my = float(max(D.max(), G.max()))
        # if fixed-fixed dominates overall, still minimise Y-part
        Ef = np.exp((D - my) / T)
        Eg = np.exp((G - my) / T)
        np.fill_diagonal(Eg, 0.0)
        S = float(Ef.sum() + Eg.sum() * 0.5)
        if S < 1e-300:
            S = 1e-300
        grad = (Ef @ Xf + Eg @ Y) / S
        grad -= (np.sum(grad * Y, axis=1, keepdims=True)) * Y
        V = mu * V + (1.0 - mu) * grad
        Y = Y - lr * V
        if noise > 0:
            N = rng.randn(k, 4)
            N -= (np.sum(N * Y, axis=1, keepdims=True)) * Y
            Y = Y + noise * N
        Y /= np.maximum(np.linalg.norm(Y, axis=1, keepdims=True), 1e-12)
    return Y

def solve():
    t0 = time.time()
    BUDGET = 172.0
    rng = np.random.RandomState(7)
    try:
        bx = np.load("best.npy")
    except Exception:
        try:
            bx = np.load("record.npy")
        except Exception:
            bx = rng.randn(163, 4)
    bx = np.asarray(bx, dtype=float)
    if bx.shape != (163, 4):
        bx = rng.randn(163, 4)
    bx = bx / np.linalg.norm(bx, axis=1, keepdims=True)
    best_x = bx.copy()
    best_val = max_cos(best_x)
    n = 163
    trial = 0
    # quick LP on start if time allows is done at end; keep searching
    while time.time() - t0 < 124.0:
        # choose move
        r = rng.rand()
        cur = best_x.copy()
        G = cur @ cur.T
        np.fill_diagonal(G, -2.0)
        mcur = float(G.max())
        deg = (G > mcur - 0.002).sum(axis=1)
        order = np.argsort(deg)
        if trial % 17 == 16:
            # full random restart, different temperature
            cur = rng.randn(n, 4)
            cur /= np.linalg.norm(cur, axis=1, keepdims=True)
            cur_r, v = softmax_anneal(cur, steps=7000, T0=0.022, T1=0.0009, lr0=0.04, mu=0.88, noise_amp=0.0015, rng=rng)
            if v < best_val and time.time() - t0 < 155:
                pol = lp_polish(cur_r, seconds=2.0)
                vp = max_cos(pol)
                if vp < best_val:
                    best_x, best_val = pol, vp
            elif v < best_val:
                best_x, best_val = cur_r, v
            trial += 1
            continue
        if r < 0.18:
            sigma = [0.02, 0.06, 0.13][trial % 3]
            cur = cur + rng.randn(*cur.shape) * sigma
            cur /= np.linalg.norm(cur, axis=1, keepdims=True)
            steps, T0, T1, lr0, nz = 3800, 0.014, 0.0012, 0.032, 0.0008
        elif r < 0.36:
            # targeted hub shake
            sigma = [0.05, 0.12, 0.20][trial % 3]
            idx = order[-10:]
            noise = rng.randn(*cur.shape) * (sigma * 0.15)
            noise[idx] += rng.randn(len(idx), 4) * (sigma * 1.6)
            cur = cur + noise
            cur /= np.linalg.norm(cur, axis=1, keepdims=True)
            steps, T0, T1, lr0, nz = 4200, 0.016, 0.0011, 0.034, 0.001
        elif r < 0.55:
            # single hole reinsertion
            w = int(order[0]) if rng.rand() < 0.8 else int(rng.randint(n))
            keep = np.delete(np.arange(n), w)
            xf = cur[keep]
            h, _ = sample_hole(xf, rng, n_samples=30000)
            # refine single
            h = refine_free(xf, h[None, :], steps=600, T0=0.02, T1=0.0015, lr0=0.05, rng=rng)[0]
            h = h + rng.randn(4) * 0.004
            h /= np.linalg.norm(h)
            cur[w] = h
            cur = cur + rng.randn(*cur.shape) * 0.008
            cur /= np.linalg.norm(cur, axis=1, keepdims=True)
            steps, T0, T1, lr0, nz = 3800, 0.012, 0.0010, 0.030, 0.0005
        elif r < 0.75:
            # joint multi-hole: remove 2-3 worst
            k = 2 + (trial % 2) + (1 if rng.rand() < 0.25 else 0)
            worst = order[-k:]
            keep = np.delete(np.arange(n), worst)
            xf = cur[keep]
            Y0 = []
            tmp = xf.copy()
            for _ in range(k):
                h, _ = sample_hole(tmp, rng, n_samples=25000)
                Y0.append(h)
                tmp = np.vstack([tmp, h[None, :]])
            Y0 = np.stack(Y0, 0)
            Y0 = refine_free(xf, Y0, steps=900, T0=0.025, T1=0.002, lr0=0.06, rng=rng, noise=0.001)
            for ii, w in enumerate(worst):
                cur[w] = Y0[ii]
            cur = cur + rng.randn(*cur.shape) * 0.006
            cur /= np.linalg.norm(cur, axis=1, keepdims=True)
            steps, T0, T1, lr0, nz = 4500, 0.014, 0.0009, 0.033, 0.0008
        elif r < 0.90:
            # contact-break tunneling: pick random tight pair and force apart
            iu0 = np.triu_indices(n, 1)
            c = (cur @ cur.T)[iu0]
            tight = np.where(c > mcur - 0.0005)[0]
            if len(tight) > 0:
                pick = tight[rng.choice(len(tight), size=min(3, len(tight)), replace=False)]
                for pp in pick:
                    i, j = int(iu0[0][pp]), int(iu0[1][pp])
                    delta = 0.03 + 0.03 * rng.rand()
                    di = cur[i] - cur[j]
                    di -= np.dot(di, cur[i]) * cur[i]
                    dn = np.linalg.norm(di)
                    if dn > 1e-12:
                        di /= dn
                        cur[i] = cur[i] + di * delta
                    dj = cur[j] - cur[i]
                    dj -= np.dot(dj, cur[j]) * cur[j]
                    dn2 = np.linalg.norm(dj)
                    if dn2 > 1e-12:
                        dj /= dn2
                        cur[j] = cur[j] + dj * delta
                cur /= np.linalg.norm(cur, axis=1, keepdims=True)
            else:
                cur = cur + rng.randn(*cur.shape) * 0.05
                cur /= np.linalg.norm(cur, axis=1, keepdims=True)
            steps, T0, T1, lr0, nz = 4500, 0.017, 0.0010, 0.035, 0.0012
        else:
            # cap coherent move
            u = rng.randn(4); u /= np.linalg.norm(u)
            dots = cur @ u
            cap = np.where(dots > 0.65)[0]
            if len(cap) < 4:
                cap = np.argsort(dots)[-12:]
            shift = rng.randn(4) * 0.06
            shift -= np.dot(shift, u) * u
            cur[cap] = cur[cap] + shift[None, :] + rng.randn(len(cap), 4) * 0.02
            cur /= np.linalg.norm(cur, axis=1, keepdims=True)
            steps, T0, T1, lr0, nz = 4000, 0.015, 0.0011, 0.032, 0.001
        cur_r, v = softmax_anneal(cur, steps=steps, T0=T0, T1=T1, lr0=lr0, mu=0.87, noise_amp=nz, rng=rng)
        # only LP-polish promising candidates to save time
        if v < best_val - 1e-7 or (v < best_val + 0.001 and rng.rand() < 0.08):
            if time.time() - t0 < 152:
                # short confirm polish, longer if clearly better
                sec = 2.5 if v < best_val else 1.2
                pol = lp_polish(cur_r, seconds=sec)
                vp = max_cos(pol)
                if vp < best_val:
                    best_x, best_val = pol, vp
            else:
                if v < best_val:
                    best_x, best_val = cur_r, v
        elif v < best_val:
            best_x, best_val = cur_r, v
        trial += 1
        if time.time() - t0 > BUDGET - 32:
            break
    remain = BUDGET - (time.time() - t0) - 1.0
    if remain > 4.0:
        remain = min(remain, 42.0)
        best_x = lp_polish(best_x, seconds=remain)
    best_x = best_x / np.linalg.norm(best_x, axis=1, keepdims=True)
    return best_x
