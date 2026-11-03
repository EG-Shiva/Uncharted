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
    radius, start = 1e-3, time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        c = (x @ x.T)[iu]
        m = c.max()
        sel = c > m - max(20 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p, nv = len(I), n * d + 1
        if p == 0:
            break
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
            x, best, radius = y, v, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def smooth_adam(x_init, steps, T0, T1, lr, beta1=0.9, beta2=0.999, eps=1e-8):
    x = x_init.copy()
    n, d = x.shape
    m = np.zeros_like(x)
    v = np.zeros_like(x)
    for s in range(steps):
        frac = s / max(steps - 1, 1)
        T = T0 * ((T1 / T0) ** frac)
        lrs = lr * (1.0 - 0.9 * frac)
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        mval = G.max()
        E = np.exp((G - mval) / T)
        np.fill_diagonal(E, 0.0)
        S = E.sum()
        if S == 0 or not np.isfinite(S):
            break
        P = E / S
        Grad = P @ x
        dot = np.sum(Grad * x, axis=1, keepdims=True)
        Gt = Grad - dot * x
        m = beta1 * m + (1 - beta1) * Gt
        vv = Gt * Gt
        v = beta2 * v + (1 - beta2) * vv
        bc1 = 1.0 - beta1 ** (s + 1)
        bc2 = 1.0 - beta2 ** (s + 1)
        mhat = m / bc1
        vhat = v / bc2
        step = lrs * mhat / (np.sqrt(vhat) + eps)
        sn = np.sum(step * x, axis=1, keepdims=True)
        step = step - sn * x
        x = x - step
        nrm = np.linalg.norm(x, axis=1, keepdims=True)
        nrm[nrm == 0] = 1.0
        x = x / nrm
    return x

def solve():
    t0 = time.time()
    TIME_BUDGET = 175.0
    rng = np.random.default_rng(1)
    try:
        xb = np.load("best.npy").astype(float)
    except Exception:
        try:
            xb = np.load("record.npy").astype(float)
        except Exception:
            xb = rng.normal(size=(102, 12))
    xb = xb / np.linalg.norm(xb, axis=1, keepdims=True)
    if xb.shape != (102, 12):
        # fix shape if needed by random init
        tmp = rng.normal(size=(102, 12))
        tmp /= np.linalg.norm(tmp, axis=1, keepdims=True)
        # copy overlap
        m = min(xb.shape[0], 102)
        tmp[:m, :min(xb.shape[1], 12)] = xb[:m, :min(xb.shape[1], 12)]
        tmp /= np.linalg.norm(tmp, axis=1, keepdims=True)
        xb = tmp
    best = max_cos(xb)
    x_best = xb.copy()

    n, d = 102, 12
    trial = 0
    # hopping phase until ~130s
    HOP_END = t0 + 132.0
    while time.time() < HOP_END and time.time() - t0 < TIME_BUDGET - 40:
        trial += 1
        x_try = x_best.copy()
        mode = trial % 4
        if mode == 3:
            # hole re-insertion of worst points
            G = x_best @ x_best.T
            np.fill_diagonal(G, -2.0)
            mval = G.max()
            cnt = (G > mval - 0.005).sum(axis=1)
            order = np.argsort(-cnt)
            nk = int(rng.integers(1, 4))  # 1..3
            worst = order[:nk].copy()
            mask = np.ones(n, dtype=bool)
            mask[worst] = False
            X_rem_list = [x_best[mask]]
            cand = rng.normal(size=(8000, d))
            cand /= np.linalg.norm(cand, axis=1, keepdims=True)
            new_pts = []
            X_cur = x_best[mask]
            for wi in worst:
                sc = (cand @ X_cur.T).max(axis=1)
                bi = int(np.argmin(sc))
                q = cand[bi].copy()
                # refine hole: minimize soft-max to X_cur
                for _ in range(200):
                    dots = X_cur @ q
                    mm = dots.max()
                    w = np.exp((dots - mm) / 0.005)
                    ws = w.sum()
                    if ws == 0 or not np.isfinite(ws):
                        break
                    w /= ws
                    g = w @ X_cur
                    g = g - (g @ q) * q
                    q = q - 0.05 * g
                    nq = np.linalg.norm(q)
                    if nq == 0 or not np.isfinite(nq):
                        break
                    q /= nq
                new_pts.append(q)
                X_cur = np.vstack([X_cur, q[None, :]])
            for wi, q in zip(worst, new_pts):
                x_try[wi] = q
            x_try = x_try + rng.normal(size=x_try.shape) * 0.005
            x_try /= np.linalg.norm(x_try, axis=1, keepdims=True)
        elif mode in (0, 1):
            # global shake
            sigma = float(rng.uniform(0.015, 0.055))
            x_try = x_try + rng.normal(size=x_try.shape) * sigma
            x_try /= np.linalg.norm(x_try, axis=1, keepdims=True)
        else:
            # targeted shake of overloaded hubs
            G = x_best @ x_best.T
            np.fill_diagonal(G, -2.0)
            mval = G.max()
            cnt = (G > mval - 0.004).sum(axis=1)
            order = np.argsort(-cnt)
            k = int(rng.integers(5, 20))
            targets = order[:k]
            sigma_small = float(rng.uniform(0.004, 0.012))
            sigma_big = float(rng.uniform(0.05, 0.16))
            noise = rng.normal(size=x_try.shape) * sigma_small
            noise[targets] += rng.normal(size=(k, d)) * sigma_big
            x_try = x_try + noise
            x_try /= np.linalg.norm(x_try, axis=1, keepdims=True)

        steps = int(rng.integers(600, 1600))
        T0p = float(rng.uniform(0.008, 0.025))
        T1p = float(rng.uniform(0.0008, 0.003))
        lrp = float(rng.uniform(0.003, 0.008))
        # reduce steps if little time left
        if time.time() + steps * 0.0006 > HOP_END:
            steps = max(300, int((HOP_END - time.time()) / 0.0006))
            if steps <= 100:
                break
        x_pol = smooth_adam(x_try, steps, T0p, T1p, lrp)
        val = max_cos(x_pol)
        if val < best - 5e-9:
            # snap promising candidate with short LP
            remain = HOP_END - time.time()
            snap_t = min(3.0, max(0.8, remain * 0.08))
            if snap_t > 0.5:
                try:
                    x_pol = lp_polish(x_pol, snap_t)
                    val = max_cos(x_pol)
                except Exception:
                    pass
            if val < best:
                best = val
                x_best = x_pol.copy()
        # opportunistic: if trial many without progress, continue
        if time.time() >= HOP_END:
            break

    # final polish of best with remaining time
    rem = TIME_BUDGET - (time.time() - t0) - 2.0
    if rem > 3:
        try:
            # one more short smooth anneal at low T to tighten
            x_best = smooth_adam(x_best, 800, 0.006, 0.0008, 0.004)
            best = max_cos(x_best)
        except Exception:
            pass
        rem2 = TIME_BUDGET - (time.time() - t0) - 1.0
        if rem2 > 2:
            try:
                x_best = lp_polish(x_best, min(rem2, 35.0))
            except Exception:
                pass
    x_best = x_best / np.linalg.norm(x_best, axis=1, keepdims=True)
    # sanitize
    x_best = np.nan_to_num(x_best, nan=0.0, posinf=0.0, neginf=0.0)
    nrm = np.linalg.norm(x_best, axis=1, keepdims=True)
    nrm[nrm == 0] = 1.0
    x_best = x_best / nrm
    return x_best
