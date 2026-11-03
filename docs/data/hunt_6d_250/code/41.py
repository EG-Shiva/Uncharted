import time
import numpy as np
from scipy.optimize import minimize, linprog
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def _norm(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n = np.maximum(n, 1e-12)
    return X / n

def _smooth_opt(X0, beta, maxiter, maxcor=10):
    n, d = X0.shape
    Y0 = X0.copy()
    def fg(Yflat):
        Y = Yflat.reshape(n, d)
        nrm = np.linalg.norm(Y, axis=1, keepdims=True)
        nrm = np.maximum(nrm, 1e-12)
        X = Y / nrm
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        M = float(np.max(G))
        D = beta * (G - M)
        np.clip(D, -30.0, 0.0, out=D)
        E = np.empty_like(D)
        np.exp(D, out=E)
        np.fill_diagonal(E, 0.0)
        S = float(np.sum(E))
        if S <= 0 or not np.isfinite(S):
            S = 1e-300
        loss = M + np.log(S) / beta
        W = E / S
        GX = 2.0 * (W @ X)
        dot = np.sum(GX * X, axis=1, keepdims=True)
        GY = (GX - dot * X) / nrm
        return loss, GY.ravel()
    try:
        res = minimize(fg, Y0.ravel(), jac=True, method="L-BFGS-B",
                       options={"maxiter": maxiter, "ftol": 1e-20,
                                "gtol": 1e-12, "maxfun": maxiter * 5,
                                "maxcor": maxcor})
        Y = res.x.reshape(n, d)
    except Exception:
        Y = Y0
    return _norm(Y)

def _adam_lse(X0, steps, b0, b1, lr0, lr1):
    X = X0.copy()
    m = np.zeros_like(X)
    v = np.zeros_like(X)
    b1m, b2m, eps = 0.9, 0.999, 1e-8
    t = 0
    lb0, lb1 = np.log(b0), np.log(b1)
    for s in range(steps):
        frac = s / (steps - 1) if steps > 1 else 1.0
        beta = float(np.exp(lb0 + (lb1 - lb0) * frac))
        lr = lr0 + (lr1 - lr0) * frac
        G = X @ X.T
        np.fill_diagonal(G, -10.0)
        cm = float(np.max(G))
        G -= cm
        G *= beta
        np.clip(G, -30.0, 0.0, out=G)
        np.exp(G, out=G)
        np.fill_diagonal(G, 0.0)
        S = float(np.sum(G))
        if S < 1e-100 or not np.isfinite(S):
            continue
        GX = G @ X
        GX *= (2.0 / S)
        dot = np.sum(GX * X, axis=1, keepdims=True)
        GX -= dot * X
        t += 1
        m *= b1m; m += (1.0 - b1m) * GX
        v *= b2m; v += (1.0 - b2m) * (GX * GX)
        c1 = 1.0 - (b1m ** t); c2 = 1.0 - (b2m ** t)
        denom = np.sqrt(v / c2) + eps
        step = (lr / c1) * m / denom
        sn = np.linalg.norm(step, axis=1, keepdims=True) + 1e-16
        cap = 5.0 * lr + 1e-12
        sc = np.minimum(1.0, cap / sn)
        step *= sc
        X -= step
        X = _norm(X)
        md = np.sum(m * X, axis=1, keepdims=True)
        m -= md * X
    return X

def _adam_riesz(X0, steps, s_pow, lr):
    X = X0.copy()
    m = np.zeros_like(X); v = np.zeros_like(X)
    b1m, b2m, eps = 0.9, 0.999, 1e-8
    t = 0
    for it in range(steps):
        G = X @ X.T
        np.fill_diagonal(G, 1.0 - 1e-12)
        D2 = 2.0 - 2.0 * G
        np.clip(D2, 1e-8, 4.0, out=D2)
        W = np.power(D2, -s_pow / 2.0 - 1.0)
        W *= s_pow
        np.fill_diagonal(W, 0.0)
        # normalize scale to avoid huge steps
        sc = 1.0 / (X.shape[0] * 2.0)
        GX = (W @ X) * sc
        dot = np.sum(GX * X, axis=1, keepdims=True)
        GX -= dot * X
        # clip grad
        gn = np.linalg.norm(GX, axis=1, keepdims=True) + 1e-16
        GX *= np.minimum(1.0, 1.0 / gn)
        t += 1
        m *= b1m; m += (1.0 - b1m) * GX
        v *= b2m; v += (1.0 - b2m) * (GX * GX)
        c1 = 1.0 - (b1m ** t); c2 = 1.0 - (b2m ** t)
        step = (lr / c1) * m / (np.sqrt(v / c2) + eps)
        sn = np.linalg.norm(step, axis=1, keepdims=True) + 1e-16
        cap = 5.0 * lr + 1e-12
        step *= np.minimum(1.0, cap / sn)
        X -= step
        X = _norm(X)
        md = np.sum(m * X, axis=1, keepdims=True)
        m -= md * X
    return X

def _lp_polish(x, seconds, t0, r0=1.2e-3, maxpairs=3800):
    x = _norm(x.copy())
    n, d = x.shape
    best = _max_cos(x)
    radius = r0
    iu = np.triu_indices(n, 1)
    while time.time() - t0 < seconds and radius > 5e-15:
        cmat = x @ x.T
        iu_c = cmat[iu]
        m = float(iu_c.max())
        sel = iu_c > m - max(10.0 * radius, 4e-7)
        cnt = int(np.count_nonzero(sel))
        if cnt == 0:
            break
        if cnt > maxpairs:
            thr = np.sort(iu_c[sel])[-maxpairs]
            sel = iu_c >= thr
        I, J = iu[0][sel], iu[1][sel]
        p, nv = len(I), n * d + 1
        if p == 0:
            break
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d),
                                           np.arange(n * d))), shape=(n, nv)).tocsr()
        except Exception:
            break
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-iu_c[sel], A_eq=A_eq,
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
        y = _norm(y)
        vv = _max_cos(y)
        if vv < best:
            x, best = y, vv
            radius = min(radius * 1.4, 0.006)
        else:
            radius *= 0.5
    return x

def _greedy_hole(Y, w, rng, Nc=9000):
    d = Y.shape[1]
    cand = rng.standard_normal((Nc, d))
    cand = _norm(cand)
    sc = cand @ Y.T
    sc[:, w] = -2.0
    mx = sc.max(axis=1)
    bi = int(np.argmin(mx))
    return cand[bi]

def _shell_init(rng, n=250, d=6, layers=10):
    per = n // layers
    rem = n - per * layers
    # latitudes evenly spaced
    ts = np.linspace(-0.92, 0.92, layers)
    X = []
    for li, t in enumerate(ts):
        k = per + (1 if li < rem else 0)
        s = np.sqrt(max(1e-6, 1 - t * t))
        Z = rng.standard_normal((k, d - 1))
        Z = _norm(Z) * s
        col = np.full((k, 1), t)
        # random rotation in full space: place col at random position? keep first coord as latitude then random orthogonal mix
        Q, _ = np.linalg.qr(rng.standard_normal((d, d)))
        blk = np.concatenate([col, Z], axis=1)
        blk = blk @ Q.T
        X.append(blk)
    X = np.concatenate(X, axis=0)
    # shuffle
    idx = rng.permutation(n)
    return _norm(X[idx])

def solve():
    t_start = time.time()
    BUDGET = 168.0
    def rem():
        return BUDGET - (time.time() - t_start)
    rng = np.random.default_rng(20261103)
    n, d = 250, 6
    cands = []
    for fn in ("best.npy", "record.npy"):
        try:
            z = np.asarray(np.load(fn), dtype=np.float64)
            if z.shape == (n, d):
                z = _norm(z)
                if np.all(np.isfinite(z)):
                    cands.append(z)
        except Exception:
            pass
    if not cands:
        z = _norm(rng.standard_normal((n, d)))
        cands.append(z)
    elite = min(cands, key=_max_cos).copy()
    elite_val = _max_cos(elite)

    # --- Stage 1: ultra-high-beta equalisation of elite (tie the plateau) ---
    if rem() > 60:
        try:
            Y = _adam_lse(elite, 350, 20000.0, 150000.0, 0.0012, 0.00025)
            v = _max_cos(Y)
            if v < elite_val:
                elite, elite_val = Y, v
        except Exception:
            pass
    if rem() > 55:
        try:
            Y = _smooth_opt(elite, 120000.0, 90)
            v = _max_cos(Y)
            if v < elite_val:
                elite, elite_val = Y, v
        except Exception:
            pass
    if rem() > 50:
        try:
            Y = _adam_lse(elite, 300, 150000.0, 500000.0, 0.0006, 0.00012)
            v = _max_cos(Y)
            if v < elite_val:
                elite, elite_val = Y, v
        except Exception:
            pass
    if rem() > 45:
        try:
            sq = _lp_polish(elite, min(4.0, rem() - 40.0), time.time(), r0=6e-4)
            v = _max_cos(sq)
            if v < elite_val:
                elite, elite_val = sq, v
        except Exception:
            pass
    cur = elite.copy(); cur_val = elite_val

    # --- Stage 2: symmetry-scaffolded + fresh basins, then release to free ---
    # Build scaffold starts
    scaffolds = []
    try:
        # antipodal random scaffold
        H = _norm(rng.standard_normal((n // 2, d)))
        scaffolds.append(np.concatenate([H, -H], axis=0))
        # layered shell scaffold
        scaffolds.append(_shell_init(rng, n, d, layers=10))
        # pure random
        scaffolds.append(_norm(rng.standard_normal((n, d))))
        scaffolds.append(_norm(rng.standard_normal((n, d))))
    except Exception:
        pass

    for si, S0 in enumerate(scaffolds):
        if rem() < 38:
            break
        try:
            Y = S0.copy()
            # enforce antipodal for first scaffold during early phase
            sym = (si == 0)
            if sym:
                # Riesz with symmetry projection
                Hc = Y[:n // 2].copy()
                # quick symmetric adam-riesz via full with projection
                Xf = np.concatenate([Hc, -Hc], axis=0)
                Xf = _adam_riesz(Xf, 150, 4.0, 0.008)
                # re-symmetrize
                Hc = 0.5 * (Xf[:n // 2] - Xf[n // 2:])
                Hc = _norm(Hc)
                Y = np.concatenate([Hc, -Hc], axis=0)
                # symmetric LSE adam
                for _ in range(2):
                    G = Y @ Y.T; np.fill_diagonal(G, -10.0)
                    cm = float(np.max(G)); beta = 8000.0
                    D = (G - cm) * beta; np.clip(D, -30, 0, out=D); np.exp(D, out=D)
                    np.fill_diagonal(D, 0.0); S = float(np.sum(D))
                    GX = (D @ Y) * (2.0 / max(S, 1e-100))
                    GX -= np.sum(GX * Y, axis=1, keepdims=True) * Y
                    gH = GX[:n // 2] - GX[n // 2:]
                    # tangent project for H
                    Hc = Y[:n // 2]
                    gH -= np.sum(gH * Hc, axis=1, keepdims=True) * Hc
                    Hc = _norm(Hc - 0.004 * gH)
                    Y = np.concatenate([Hc, -Hc], axis=0)
                # release symmetry: small break then free polish
                Y += rng.standard_normal(Y.shape) * 0.002
                Y = _norm(Y)
            else:
                Y = _adam_riesz(Y, 180, 4.0, 0.008)
            Y = _smooth_opt(Y, 6000.0, 70)
            v = _max_cos(Y)
            if v < cur_val + 8e-4 and rem() > 30:
                Y2 = _adam_lse(Y, 220, 20000.0, 120000.0, 0.001, 0.0002)
                v2 = _max_cos(Y2)
                if v2 < v:
                    Y, v = Y2, v2
            if v < elite_val - 1e-12:
                elite, elite_val = Y.copy(), v
                cur, cur_val = Y.copy(), v
                try:
                    sq = _lp_polish(elite, min(1.5, rem() - 25.0), time.time(), r0=5e-4)
                    vs = _max_cos(sq)
                    if vs < elite_val:
                        elite, elite_val = sq, vs
                        cur, cur_val = sq.copy(), vs
                except Exception:
                    pass
            elif v < cur_val:
                cur, cur_val = Y.copy(), v
        except Exception:
            continue

    # --- Stage 3: void-reinsertion basin hopping with low->ultra beta re-anneal ---
    trial = 0
    while rem() > 22 and trial < 40:
        trial += 1
        try:
            base = cur if rng.random() < 0.7 else elite
            G = base @ base.T
            np.fill_diagonal(G, -2.0)
            rm = G.max(axis=1)
            order = np.argsort(-rm)
            iu = np.triu_indices(n, 1)
            gv = G[iu]
            k = int(np.argmax(gv))
            a, b = int(iu[0][k]), int(iu[1][k])
            Y = base.copy()
            mode = trial % 3
            if mode == 0:
                kblk = 5 + (trial % 4)
                victims = [int(v) for v in order[:kblk]]
                if rng.random() < 0.5:
                    victims[-1] = int(rng.integers(0, n))
                for w in victims:
                    Y[w] = _greedy_hole(Y, w, rng, Nc=9000)
                Y += rng.standard_normal(Y.shape) * 0.0008
                Y = _norm(Y)
                Y = _smooth_opt(Y, 5000.0, 60)
            elif mode == 1:
                # cluster shake around worst
                dots = Y @ Y[a]
                neigh = np.where(dots > elite_val - 0.04)[0]
                for vv in neigh[:22]:
                    if vv == a or vv == b:
                        continue
                    Y[vv] += rng.standard_normal(d) * 0.06
                eps = 0.008 + 0.015 * rng.random()
                na = Y[a] - eps * Y[b]; nb = Y[b] - eps * Y[a]
                Y[a] = na / np.linalg.norm(na); Y[b] = nb / np.linalg.norm(nb)
                Y = _norm(Y)
                Y = _adam_lse(Y, 180, 6000.0, 6000.0, 0.002, 0.0006)
            else:
                krs = 8 + (trial % 5)
                idx = rng.choice(n, krs, replace=False)
                Y[idx] = _norm(rng.standard_normal((krs, d)))
                Y += rng.standard_normal(Y.shape) * 0.0005
                Y = _norm(Y)
                Y = _smooth_opt(Y, 4000.0, 60)
            v = _max_cos(Y)
            if v < elite_val + 4e-4 and rem() > 24:
                Y2 = _adam_lse(Y, 200, 25000.0, 150000.0, 0.0009, 0.0002)
                v2 = _max_cos(Y2)
                if v2 < v:
                    Y, v = Y2, v2
            if v < elite_val - 1e-12:
                elite, elite_val = Y.copy(), v
                cur, cur_val = Y.copy(), v
            elif v < cur_val:
                cur, cur_val = Y.copy(), v
            elif v < cur_val + 2e-4 and rng.random() < 0.2:
                cur, cur_val = Y.copy(), v
            if cur_val > elite_val + 5e-4 and rng.random() < 0.3:
                cur, cur_val = elite.copy(), elite_val
        except Exception:
            continue

    # --- Stage 4: final collective squeeze ---
    if rem() > 8:
        try:
            Y = _adam_lse(elite, 260, 60000.0, 400000.0, 0.0006, 0.0001)
            v = _max_cos(Y)
            if v < elite_val:
                elite, elite_val = Y, v
        except Exception:
            pass
    if rem() > 5:
        try:
            Y = _smooth_opt(elite, 150000.0, 80)
            v = _max_cos(Y)
            if v < elite_val:
                elite, elite_val = Y, v
        except Exception:
            pass
    if rem() > 2.5:
        try:
            sq = _lp_polish(elite, rem() - 1.0, time.time(), r0=4e-4)
            v = _max_cos(sq)
            if v < elite_val:
                elite, elite_val = sq, v
        except Exception:
            pass
    elite = _norm(elite)
    elite = np.where(np.isfinite(elite), elite, 0.0)
    elite = _norm(elite)
    return elite
