import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(np.max(g))


def lp_polish(x, budget):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    try:
        best = max_cos(x)
    except Exception:
        return x
    best_x = x.copy()
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < budget and radius > 1e-14:
        G = x @ x.T
        c = G[iu]
        m = float(c.max())
        sel = c > m - max(20.0 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        if p == 0:
            radius *= 0.5
            continue
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel() if False else np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        except Exception:
            radius *= 0.5
            continue
        obj = np.zeros(nv)
        obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c[sel], A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        try:
            y = x + res.x[:-1].reshape(n, d)
            y = y / np.linalg.norm(y, axis=1, keepdims=True)
            v = max_cos(y)
        except Exception:
            radius *= 0.5
            continue
        if v < best:
            x, best, best_x = y, v, y.copy()
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return best_x


def _angles(M, a, b):
    k = np.arange(M)
    th1 = 2.0 * np.pi * float(a) * k / float(M)
    th2 = 2.0 * np.pi * float(b) * k / float(M)
    return np.cos(th1), np.sin(th1), np.cos(th2), np.sin(th2)


def _build_X(R, M, c1, s1, c2, s2):
    rx1 = R[:, 0:1]
    ry1 = R[:, 1:2]
    rx2 = R[:, 2:3]
    ry2 = R[:, 3:4]
    M0 = rx1 * c1[None, :] - ry1 * s1[None, :]
    M1 = rx1 * s1[None, :] + ry1 * c1[None, :]
    M2 = rx2 * c2[None, :] - ry2 * s2[None, :]
    M3 = rx2 * s2[None, :] + ry2 * c2[None, :]
    r = R.shape[0]
    X = np.empty((r * M, 4))
    X[:, 0] = M0.ravel()
    X[:, 1] = M1.ravel()
    X[:, 2] = M2.ravel()
    X[:, 3] = M3.ravel()
    return X


def _sym_adam(Rinit, M, c1, s1, c2, s2, steps, b0, b1, lr, t_end=1e30):
    R = Rinit.copy()
    mt = np.zeros_like(R)
    vt = np.zeros_like(R)
    b1_ = 0.9
    b2_ = 0.999
    eps = 1e-8
    for k in range(steps):
        f = k / (steps - 1) if steps > 1 else 0.0
        beta = b0 * ((b1 / b0) ** f)
        X = _build_X(R, M, c1, s1, c2, s2)
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        m = float(np.max(G))
        A = beta * (G - m)
        np.clip(A, -12.0, 0.0, out=A)
        np.exp(A, out=A)
        np.fill_diagonal(A, 0.0)
        GradX = A @ X
        Gb = GradX.reshape(R.shape[0], M, 4)
        G0 = Gb[:, :, 0]
        G1 = Gb[:, :, 1]
        G2 = Gb[:, :, 2]
        G3 = Gb[:, :, 3]
        GradR = np.empty_like(R)
        GradR[:, 0] = G0 @ c1 + G1 @ s1
        GradR[:, 1] = -G0 @ s1 + G1 @ c1
        GradR[:, 2] = G2 @ c2 + G3 @ s2
        GradR[:, 3] = -G2 @ s2 + G3 @ c2
        dot = np.sum(GradR * R, axis=1, keepdims=True)
        Gt = GradR - dot * R
        mt = b1_ * mt + (1.0 - b1_) * Gt
        vt = b2_ * vt + (1.0 - b2_) * Gt * Gt
        mhat = mt / (1.0 - b1_ ** (k + 1))
        vhat = vt / (1.0 - b2_ ** (k + 1))
        step = mhat / (np.sqrt(vhat) + eps)
        R = R - lr * step
        nrm = np.sqrt(np.sum(R * R, axis=1, keepdims=True))
        nrm = np.where(nrm < 1e-12, 1.0, nrm)
        R = R / nrm
        if (k & 127) == 0 and time.time() > t_end:
            break
    return R


def solve():
    t0 = time.time()
    HARD = 176.0
    n, d = 153, 4
    # fallback record
    fallback = None
    fallback_val = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            v = np.load(fn).astype(float)
            if v.shape == (n, d):
                v = v / np.linalg.norm(v, axis=1, keepdims=True)
                if np.all(np.isfinite(v)):
                    c = max_cos(v)
                    if c < fallback_val:
                        fallback_val = c
                        fallback = v.copy()
        except Exception:
            continue
    if fallback is None:
        rng0 = np.random.default_rng(0)
        fallback = rng0.standard_normal((n, d))
        fallback = fallback / np.linalg.norm(fallback, axis=1, keepdims=True)
        fallback_val = max_cos(fallback)

    best = fallback.copy()
    best_val = float(fallback_val)

    try:
        rng = np.random.default_rng(17)
        # configs: (M, a, b, r) with r*M==153
        configs = []
        for a in range(1, 9):
            configs.append((17, 1, a, 9))
        for a in range(1, 5):
            configs.append((9, 1, a, 17))
        # brief screening: 1 init each, short Adam
        screened = []
        for (M, a, b, r) in configs:
            if time.time() - t0 > 60.0:
                break
            try:
                c1, s1, c2, s2 = _angles(M, a, b)
                Rinit = rng.standard_normal((r, 4))
                Rinit = Rinit / np.linalg.norm(Rinit, axis=1, keepdims=True)
                Ropt = _sym_adam(Rinit, M, c1, s1, c2, s2, steps=900,
                                 b0=20.0, b1=220.0, lr=0.01,
                                 t_end=t0 + 65.0)
                X = _build_X(Ropt, M, c1, s1, c2, s2)
                X = X / np.linalg.norm(X, axis=1, keepdims=True)
                if X.shape != (153, 4) or not np.all(np.isfinite(X)):
                    continue
                v = max_cos(X)
                screened.append((v, Ropt.copy(), (M, a, b, r)))
            except Exception:
                continue
        if len(screened) == 0:
            raise RuntimeError("no screened")
        screened.sort(key=lambda t: t[0])
        # refine top 3 longer
        refined = []
        for idx in range(min(3, len(screened))):
            v0, R0, cfg = screened[idx]
            M, a, b, r = cfg
            try:
                c1, s1, c2, s2 = _angles(M, a, b)
                R2 = _sym_adam(R0, M, c1, s1, c2, s2, steps=2000,
                               b0=30.0, b1=400.0, lr=0.006,
                               t_end=t0 + 100.0)
                X2 = _build_X(R2, M, c1, s1, c2, s2)
                X2 = X2 / np.linalg.norm(X2, axis=1, keepdims=True)
                v2 = max_cos(X2)
                refined.append((v2, X2.copy()))
            except Exception:
                refined.append((v0, _build_X(R0, M, *_angles(M, a, b)[0:4]) if False else None))
                continue
        # also keep best screened expanded
        for (v0, R0, cfg) in screened[:2]:
            try:
                M, a, b, r = cfg
                c1, s1, c2, s2 = _angles(M, a, b)
                X0 = _build_X(R0, M, c1, s1, c2, s2)
                X0 = X0 / np.linalg.norm(X0, axis=1, keepdims=True)
                refined.append((max_cos(X0), X0))
            except Exception:
                continue
        refined.sort(key=lambda t: t[0])
        # break symmetry with 1e-3 tangential noise + LP polish
        cand_pool = refined[:3]
        for (sv, SX) in cand_pool:
            if time.time() - t0 > HARD - 35.0:
                break
            try:
                if SX.shape != (153, 4) or not np.all(np.isfinite(SX)):
                    continue
                SX = SX / np.linalg.norm(SX, axis=1, keepdims=True)
                # tangential noise 1e-3
                noise = rng.standard_normal((n, d))
                dot = np.sum(noise * SX, axis=1, keepdims=True)
                noise = noise - dot * SX
                nn = np.linalg.norm(noise, axis=1, keepdims=True) + 1e-12
                noise = noise / nn * 1e-3
                Xp = SX + noise
                Xp = Xp / np.linalg.norm(Xp, axis=1, keepdims=True)
                # light LP polish
                remain = HARD - (time.time() - t0) - 8.0
                if remain < 1.0:
                    v = max_cos(Xp)
                    if v < best_val:
                        best_val = v
                        best = Xp.copy()
                    continue
                budget = min(8.0, remain / max(1, len(cand_pool)))
                Xq = lp_polish(Xp, budget)
                Xq = Xq / np.linalg.norm(Xq, axis=1, keepdims=True)
                vq = max_cos(Xq)
                if vq < best_val - 1e-12:
                    best_val = vq
                    best = Xq.copy()
                else:
                    # also try un-noised version
                    remain2 = HARD - (time.time() - t0) - 8.0
                    if remain2 > 1.5:
                        Xq2 = lp_polish(SX, min(4.0, remain2))
                        Xq2 = Xq2 / np.linalg.norm(Xq2, axis=1, keepdims=True)
                        vq2 = max_cos(Xq2)
                        if vq2 < best_val - 1e-12:
                            best_val = vq2
                            best = Xq2.copy()
            except Exception:
                continue
        # final polish of global best if improved or time left
        try:
            remain = HARD - (time.time() - t0) - 1.0
            if remain > 2.0:
                Xf = lp_polish(best, min(remain, 15.0))
                Xf = Xf / np.linalg.norm(Xf, axis=1, keepdims=True)
                vf = max_cos(Xf)
                if vf <= best_val and np.all(np.isfinite(Xf)) and Xf.shape == (153, 4):
                    best = Xf
                    best_val = vf
        except Exception:
            pass
        # validate
        if best.shape != (153, 4) or not np.all(np.isfinite(best)):
            best = fallback.copy()
            best_val = fallback_val
        best = best / np.linalg.norm(best, axis=1, keepdims=True)
        if not np.all(np.isfinite(best)):
            best = fallback.copy()
        # never return worse than fallback? return best found (could be fallback)
        if max_cos(best) > fallback_val + 1e-12:
            best = fallback.copy()
        return best
    except Exception:
        try:
            if fallback is not None and fallback.shape == (153, 4) and np.all(np.isfinite(fallback)):
                fb = fallback / np.linalg.norm(fallback, axis=1, keepdims=True)
                return fb
        except Exception:
            pass
        rng = np.random.default_rng(0)
        X = rng.standard_normal((153, 4))
        X = X / np.linalg.norm(X, axis=1, keepdims=True)
        return X
