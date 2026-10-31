import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())


def anneal_descend(x, steps):
    for s in range(steps):
        frac = s / steps if steps > 1 else 1.0
        alpha = 12.0 + frac * 138.0
        window = 0.16 * (1.0 - frac) + 0.015
        step = 0.020 * (1.0 - frac) + 0.0015
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        cmax = float(G.max())
        D = G - cmax
        mask = D > -window
        W = np.zeros_like(G)
        # exp only on near-max pairs, D<=0 so no overflow
        W[mask] = np.exp(alpha * D[mask])
        Gr = W @ x
        dot = np.sum(Gr * x, axis=1, keepdims=True)
        Gt = Gr - dot * x
        avg = float(np.mean(np.sqrt(np.sum(Gt * Gt, axis=1))) + 1e-12)
        if avg < 1e-14:
            continue
        x = x - (step / avg) * Gt
        nrm = np.sqrt(np.sum(x * x, axis=1, keepdims=True))
        nrm[nrm == 0] = 1.0
        x /= nrm
    return x


def lp_polish(x, seconds):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    try:
        best = max_cos(x)
    except Exception:
        return x
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        if time.time() - start > seconds - 0.2:
            break
        Gfull = x @ x.T
        c = Gfull[iu]
        m = float(c.max())
        sel = c > m - max(20 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        if p == 0:
            break
        nv = n * d + 1
        try:
            rows = np.repeat(np.arange(p), 2 * d + 1)
            cols = np.concatenate(
                [I[:, None] * d + np.arange(d),
                 J[:, None] * d + np.arange(d),
                 np.full((p, 1), n * d)], 1).ravel()
            vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix(
                (x.ravel(),
                 (np.repeat(np.arange(n), d), np.arange(n * d))),
                shape=(n, nv)).tocsr()
            obj = np.zeros(nv)
            obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-c[sel], A_eq=A_eq,
                          b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                          method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        nrm = np.linalg.norm(y, axis=1, keepdims=True)
        nrm[nrm == 0] = 1.0
        y /= nrm
        v = max_cos(y)
        if v < best:
            x, best, radius = y, v, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x


def solve():
    t0 = time.time()
    rng = np.random.RandomState(0)
    cands = []
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            cands.append(np.asarray(a, dtype=float))
        except Exception:
            pass
    if not cands:
        a = rng.randn(153, 4)
        a /= np.linalg.norm(a, axis=1, keepdims=True)
        cands.append(a)

    def normed(a):
        a = np.asarray(a, dtype=float)
        if a.shape != (153, 4):
            try:
                a = a.reshape(153, 4)
            except Exception:
                a = rng.randn(153, 4)
        nrm = np.linalg.norm(a, axis=1, keepdims=True)
        nrm[nrm == 0] = 1.0
        return a / nrm

    cands = [normed(a) for a in cands]
    vals = [max_cos(a) for a in cands]
    bi = int(np.argmin(vals))
    x_best = cands[bi]
    best_val = float(vals[bi])

    sigmas = [0.012, 0.025, 0.05, 0.09, 0.16, 0.30,
              0.008, 0.04, 0.07, 0.12, 0.20, 0.45]
    it = 0
    # exploration phase, reserve time for final polish
    while time.time() - t0 < 125:
        sigma = float(sigmas[it % len(sigmas)])
        noise = rng.randn(*x_best.shape)
        if it % 5 == 4:
            G = x_best @ x_best.T
            np.fill_diagonal(G, -2.0)
            cmax = float(G.max())
            per = np.max(G, axis=1)
            worst = np.where(per > cmax - 0.025)[0]
            if len(worst) > 0:
                pert = np.zeros_like(x_best)
                pert[worst] = noise[worst] * 2.0
                pert += 0.15 * rng.randn(*x_best.shape)
            else:
                pert = noise
            x_try = x_best + sigma * pert
        else:
            x_try = x_best + sigma * noise
        nrm = np.sqrt(np.sum(x_try * x_try, axis=1, keepdims=True))
        nrm[nrm == 0] = 1.0
        x_try /= nrm
        steps = 700 + (it % 3) * 250
        if sigma > 0.25:
            steps = 1200
        # guard remaining time
        if time.time() - t0 > 122:
            break
        x_try = anneal_descend(x_try, steps)
        v = max_cos(x_try)
        if v < best_val - 1e-12:
            best_val = v
            x_best = x_try
            if time.time() - t0 < 110:
                x_pol = lp_polish(x_best, 3.0)
                vp = max_cos(x_pol)
                if vp < best_val:
                    best_val = vp
                    x_best = x_pol
        it += 1
        if it > 500:
            break

    rem = 175.0 - (time.time() - t0)
    if rem > 6:
        try:
            x_best = lp_polish(x_best, min(rem - 2.0, 38.0))
        except Exception:
            pass
    nrm = np.linalg.norm(x_best, axis=1, keepdims=True)
    nrm[nrm == 0] = 1.0
    x_best = x_best / nrm
    x_best = np.nan_to_num(x_best, nan=0.0, posinf=0.0, neginf=0.0)
    nrm = np.linalg.norm(x_best, axis=1, keepdims=True)
    nrm[nrm == 0] = 1.0
    x_best = x_best / nrm
    return x_best
