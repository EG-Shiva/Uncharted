import time
import math
import numpy as np
from scipy.optimize import linprog, minimize
from scipy.sparse import coo_matrix

_N = 173
_D = 5

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _lp_polish(x, seconds):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    start = time.time()
    iu0, iu1 = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        c = (x @ x.T)[iu0, iu1]
        m = float(c.max())
        sel = c > m - max(20 * radius, 1e-9)
        I, J = iu0[sel], iu1[sel]
        p = len(I)
        if p == 0:
            break
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv)
        obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c[sel], A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            break
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        vv = _max_cos(y)
        if vv < best:
            x, best, radius = y, vv, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def _repel(x, alpha, iters, lr0, mu=0.6, decay=0.998):
    v = np.zeros_like(x)
    lr = float(lr0)
    for _ in range(int(iters)):
        G = x @ x.T
        W = np.exp(alpha * (G - 0.75))
        np.fill_diagonal(W, 0.0)
        rs = W.sum(axis=1, keepdims=True) + 1e-300
        A = (W @ x) / rs
        dot = (A * x).sum(axis=1, keepdims=True)
        Ft = A - dot * x
        v = mu * v + (1.0 - mu) * Ft
        x = x - lr * v
        x /= np.linalg.norm(x, axis=1, keepdims=True)
        lr *= decay
    return x

# ---------- symmetric orbit machinery ----------
def _build_full(theta, K, a, b, nG, nA, nB, poles):
    th = np.asarray(theta, dtype=float)
    idx = 0
    parts = []
    twopi = 2.0 * np.pi
    for _ in range(nG):
        al, be, d1, d2 = th[idx], th[idx+1], th[idx+2], th[idx+3]
        idx += 4
        r = math.sin(al); z = math.cos(al)
        r1 = r * math.cos(be); r2 = r * math.sin(be)
        ks = np.arange(K)
        ang1 = d1 + a * twopi * ks / K
        ang2 = d2 + b * twopi * ks / K
        pts = np.empty((K, 5))
        pts[:, 0] = r1 * np.cos(ang1); pts[:, 1] = r1 * np.sin(ang1)
        pts[:, 2] = r2 * np.cos(ang2); pts[:, 3] = r2 * np.sin(ang2)
        pts[:, 4] = z
        parts.append(pts)
    if nA > 0:
        Ka = K // math.gcd(K, b)
        for _ in range(nA):
            al, d2 = th[idx], th[idx+1]
            idx += 2
            r2 = math.sin(al); z = math.cos(al)
            ks = np.arange(Ka)
            ang2 = d2 + b * twopi * ks / K
            pts = np.empty((Ka, 5))
            pts[:, 0] = 0.0; pts[:, 1] = 0.0
            pts[:, 2] = r2 * np.cos(ang2); pts[:, 3] = r2 * np.sin(ang2)
            pts[:, 4] = z
            parts.append(pts)
    if nB > 0:
        Kb = K // math.gcd(K, a)
        for _ in range(nB):
            al, d1 = th[idx], th[idx+1]
            idx += 2
            r1 = math.sin(al); z = math.cos(al)
            ks = np.arange(Kb)
            ang1 = d1 + a * twopi * ks / K
            pts = np.empty((Kb, 5))
            pts[:, 0] = r1 * np.cos(ang1); pts[:, 1] = r1 * np.sin(ang1)
            pts[:, 2] = 0.0; pts[:, 3] = 0.0
            pts[:, 4] = z
            parts.append(pts)
    for s in poles:
        pt = np.zeros((1, 5)); pt[0, 4] = float(s)
        parts.append(pt)
    X = np.vstack(parts)
    # safety: if rounding gives wrong count, trim/pad (should not happen)
    return X

def _nparam(nG, nA, nB):
    return 4 * nG + 2 * nA + 2 * nB

def _bounds(nG, nA, nB):
    b = []
    pi = math.pi
    for _ in range(nG):
        b.append((0.08, pi - 0.08))
        b.append((0.06, pi/2 - 0.06))
        b.append((0.0, 2*math.pi))
        b.append((0.0, 2*math.pi))
    for _ in range(nA + nB):
        b.append((0.08, pi - 0.08))
        b.append((0.0, 2*math.pi))
    return b

def _sym_max(theta, K, a, b, nG, nA, nB, poles):
    try:
        X = _build_full(theta, K, a, b, nG, nA, nB, poles)
    except Exception:
        return 2.0
    return _max_cos(X)

def _sym_smooth(theta, K, a, b, nG, nA, nB, poles, tau=120.0):
    try:
        X = _build_full(theta, K, a, b, nG, nA, nB, poles)
    except Exception:
        return 2.0
    n = X.shape[0]
    iu0, iu1 = np.triu_indices(n, 1)
    c = (X @ X.T)[iu0, iu1]
    m = float(c.max())
    # log-sum-exp smooth max
    v = c - m
    # clip to avoid overflow: tau*v <= 0
    s = float(np.log(np.exp(tau * v).sum()) / tau + m)
    return s

def _random_theta(rng, nG, nA, nB):
    p = []
    pi = math.pi
    for _ in range(nG):
        # bias alpha away from equator? uniform in cos
        u = rng.rand()
        al = math.acos(1.0 - 2.0 * u)
        al = float(np.clip(al, 0.08, pi - 0.08))
        be = float(rng.rand() * (pi/2 - 0.12) + 0.06)
        d1 = float(rng.rand() * 2 * pi)
        d2 = float(rng.rand() * 2 * pi)
        p.extend([al, be, d1, d2])
    for _ in range(nA + nB):
        u = rng.rand()
        al = math.acos(1.0 - 2.0 * u)
        al = float(np.clip(al, 0.08, pi - 0.08))
        d = float(rng.rand() * 2 * pi)
        p.extend([al, d])
    return np.array(p, dtype=float)

def solve():
    t0 = time.time()
    deadline = t0 + 168.0
    rng = np.random.RandomState(12345)
    n, d = _N, _D

    # load incumbents
    try:
        xb = np.load("best.npy").astype(float)
        if xb.shape != (n, d):
            xb = None
        else:
            xb = xb / np.linalg.norm(xb, axis=1, keepdims=True)
    except Exception:
        xb = None
    try:
        xr = np.load("record.npy").astype(float)
        if xr.shape != (n, d):
            xr = None
        else:
            xr = xr / np.linalg.norm(xr, axis=1, keepdims=True)
    except Exception:
        xr = None

    if xb is not None:
        best_val = _max_cos(xb)
        X_best = xb.copy()
    elif xr is not None:
        best_val = _max_cos(xr)
        X_best = xr.copy()
    else:
        X_best = rng.standard_normal((n, d))
        X_best /= np.linalg.norm(X_best, axis=1, keepdims=True)
        best_val = _max_cos(X_best)
    if xr is not None:
        try:
            v = _max_cos(xr)
            if v < best_val:
                best_val = v; X_best = xr.copy()
        except Exception:
            pass

    # symmetric candidates: (K,a,b,nG,nA,nB,poles)
    configs = [
        dict(K=18, a=1, b=2, nG=9, nA=1, nB=0, poles=[1.0, -1.0]),
        dict(K=27, a=1, b=3, nG=6, nA=1, nB=0, poles=[1.0, -1.0]),
        dict(K=18, a=2, b=3, nG=8, nA=3, nB=1, poles=[1.0, -1.0]),
        dict(K=18, a=1, b=2, nG=8, nA=3, nB=0, poles=[1.0, -1.0]),
        dict(K=27, a=1, b=3, nG=5, nA=4, nB=0, poles=[1.0, -1.0]),
    ]

    # time for symmetric phase
    sym_deadline = t0 + 118.0
    for cfg in configs:
        if time.time() > sym_deadline:
            break
        K = cfg["K"]; a = cfg["a"]; b = cfg["b"]
        nG = cfg["nG"]; nA = cfg["nA"]; nB = cfg["nB"]
        poles = cfg["poles"]
        # sanity count
        Ka = K // math.gcd(K, b) if nA else 0
        Kb = K // math.gcd(K, a) if nB else 0
        tot = nG * K + nA * Ka + nB * Kb + len(poles)
        if tot != 173:
            continue
        bnds = _bounds(nG, nA, nB)
        # random sampling
        n_sample = 1200
        # reduce samples for large K to save time
        best_th = None
        best_s = 1e9
        for _ in range(n_sample):
            if time.time() > sym_deadline:
                break
            th = _random_theta(rng, nG, nA, nB)
            v = _sym_max(th, K, a, b, nG, nA, nB, poles)
            if v < best_s:
                best_s = v; best_th = th.copy()
        if best_th is None:
            continue
        # keep top few distinct seeds: best + 2 random good ones (re-sample small)
        seeds = [best_th]
        # add one more random best-of-200
        bs2 = 1e9; bt2 = None
        for _ in range(300):
            th = _random_theta(rng, nG, nA, nB)
            v = _sym_max(th, K, a, b, nG, nA, nB, poles)
            if v < bs2:
                bs2 = v; bt2 = th.copy()
        if bt2 is not None:
            seeds.append(bt2)
        for seed in seeds:
            if time.time() > sym_deadline:
                break
            # two tau stages
            for tau in (60.0, 150.0):
                if time.time() > sym_deadline:
                    break
                try:
                    res = minimize(lambda th: _sym_smooth(th, K, a, b, nG, nA, nB, poles, tau),
                                   seed, method="L-BFGS-B", bounds=bnds,
                                   options={"maxiter": 120, "maxfun": 6000, "ftol": 1e-12, "gtol": 1e-8})
                    seed = res.x
                except Exception:
                    break
            try:
                v_true = _sym_max(seed, K, a, b, nG, nA, nB, poles)
            except Exception:
                continue
            # expand
            try:
                Xsym = _build_full(seed, K, a, b, nG, nA, nB, poles)
                Xsym /= np.linalg.norm(Xsym, axis=1, keepdims=True)
            except Exception:
                continue
            # release symmetry with tiny jitter + polish
            # quick full evaluation
            # jitter levels to break symmetry into different basins
            for jsig in (0.004, 0.012):
                if time.time() > sym_deadline + 8.0:
                    break
                try:
                    Xrel = Xsym + jsig * rng.standard_normal(Xsym.shape)
                    Xrel /= np.linalg.norm(Xrel, axis=1, keepdims=True)
                    # short repel to move basin
                    Xrel = _repel(Xrel.copy(), 90.0, 120, 0.05, mu=0.6, decay=0.997)
                    Xrel = _lp_polish(Xrel, 2.2)
                    v = _max_cos(Xrel)
                    if v < best_val:
                        best_val = v
                        X_best = Xrel.copy()
                except Exception:
                    continue
            # also polish symmetric point directly (no jitter) to keep equilibrated topology
            try:
                if time.time() < sym_deadline:
                    Xp = _lp_polish(Xsym.copy(), 1.5)
                    v = _max_cos(Xp)
                    if v < best_val:
                        best_val = v
                        X_best = Xp.copy()
            except Exception:
                pass

    # final intensive polish of global best with remaining time
    remain = deadline - time.time() - 2.0
    if remain > 3.0:
        try:
            # a couple of repel+LP cycles on best
            X = X_best.copy()
            # small shake to escape rattler issues, keep if improves
            t_end = deadline - 1.0
            # first deep LP
            X = _lp_polish(X, min(8.0, max(2.0, remain * 0.5)))
            v = _max_cos(X)
            if v < best_val:
                best_val = v; X_best = X.copy()
            # try one mild repel + LP if time
            if time.time() < t_end - 6.0:
                rng2 = np.random.RandomState(999)
                Xt = X_best + 0.002 * rng2.standard_normal(X_best.shape)
                Xt /= np.linalg.norm(Xt, axis=1, keepdims=True)
                Xt = _repel(Xt, 120.0, 150, 0.03, mu=0.65, decay=0.998)
                Xt = _lp_polish(Xt, t_end - time.time())
                vt = _max_cos(Xt)
                if vt < best_val:
                    best_val = vt; X_best = Xt.copy()
        except Exception:
            pass

    X_best = X_best / np.linalg.norm(X_best, axis=1, keepdims=True)
    # shrink tiny margin not needed; ensure finite
    X_best = np.where(np.isfinite(X_best), X_best, 0.0)
    nr = np.linalg.norm(X_best, axis=1, keepdims=True)
    nr[nr == 0] = 1.0
    X_best = X_best / nr
    return X_best
