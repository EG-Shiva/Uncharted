import time
import numpy as np


def _max_cos(Y):
    G = Y @ Y.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())


def _normalize(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return X / n


def _adam_refine(Y0, steps, beta0, beta1, lr0, lr1):
    n, d = Y0.shape
    Y = Y0.copy()
    M = np.zeros_like(Y)
    V = np.zeros_like(Y)
    b1 = 0.9
    b2 = 0.999
    eps = 1e-8
    # initial best
    G0 = Y @ Y.T
    np.fill_diagonal(G0, -10.0)
    best_val = float(G0.max())
    best_Y = Y.copy()
    lb = np.log(beta1 / beta0) / max(steps - 1, 1) if steps > 1 else 0.0
    ll = np.log(lr1 / lr0) / max(steps - 1, 1) if steps > 1 else 0.0
    lbb0 = np.log(beta0)
    llr0 = np.log(lr0)
    b1t = 1.0
    b2t = 1.0
    for t in range(1, steps + 1):
        beta = float(np.exp(lbb0 + lb * (t - 1)))
        lr = float(np.exp(llr0 + ll * (t - 1)))
        G = Y @ Y.T
        np.fill_diagonal(G, -10.0)
        gmax = float(G.max())
        if gmax < best_val:
            best_val = gmax
            best_Y = Y.copy()
        # weights
        # E = exp(beta*(G-gmax)), diag 0
        np.subtract(G, gmax, out=G)
        np.multiply(G, beta, out=G)
        # clip to avoid underflow overhead (optional)
        np.exp(G, out=G)
        # zero diag (diag currently exp(beta*(-10-gmax)) tiny, set 0)
        np.fill_diagonal(G, 0.0)
        s = float(G.sum())
        if s <= 0 or not np.isfinite(s):
            continue
        G /= s
        Gr = G @ Y
        # riemannian projection
        dot = np.sum(Gr * Y, axis=1, keepdims=True)
        Gr -= Y * dot
        # adam
        b1t *= b1
        b2t *= b2
        M *= b1
        M += (1.0 - b1) * Gr
        V *= b2
        V += (1.0 - b2) * (Gr * Gr)
        Mh = M / (1.0 - b1t)
        Vh = V / (1.0 - b2t)
        Y -= lr * Mh / (np.sqrt(Vh) + eps)
        # retract
        rn = np.linalg.norm(Y, axis=1, keepdims=True)
        rn[rn == 0] = 1.0
        Y /= rn
    # final check
    Gf = Y @ Y.T
    np.fill_diagonal(Gf, -10.0)
    vf = float(Gf.max())
    if vf < best_val:
        best_val = vf
        best_Y = Y.copy()
    return best_Y, best_val


def _contact_refine(Y0, steps, step0, tol):
    Y = Y0.copy()
    Gc = Y @ Y.T
    np.fill_diagonal(Gc, -2.0)
    best_val = float(Gc.max())
    best_Y = Y.copy()
    step = float(step0)
    n = Y.shape[0]
    for _ in range(steps):
        Gfull = Y @ Y.T
        # max off diag
        # copy for max without destroying Gfull needed for forces
        m = float(np.max(Gfull - 2.0 * np.eye(n, dtype=Gfull.dtype)))
        # active mask
        A = Gfull > (m - tol)
        np.fill_diagonal(A, False)
        # if no active (should not happen), break
        # forces: F = -(A@Y) + Y * row_sums
        # use float
        Af = A.astype(np.float64)
        AY = Af @ Y
        row_w = np.sum(Af * Gfull, axis=1, keepdims=True)
        F = -AY + Y * row_w
        # trial
        Yt = Y + step * F
        rn = np.linalg.norm(Yt, axis=1, keepdims=True)
        rn[rn == 0] = 1.0
        Yt /= rn
        Gt = Yt @ Yt.T
        np.fill_diagonal(Gt, -2.0)
        mt = float(Gt.max())
        if mt < m - 1e-12:
            Y = Yt
            if mt < best_val:
                best_val = mt
                best_Y = Y.copy()
            step = min(step * 1.08, 0.02)
        else:
            step *= 0.5
            if step < 1e-12:
                step = 1e-12
                # still move a bit to avoid stall? keep Y
                pass
    return best_Y, best_val


def solve():
    rng = np.random.default_rng(0)
    t0 = time.time()
    deadline = t0 + 165.0

    X0 = None
    best_val0 = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=np.float64)
            if a.shape != (59, 22):
                continue
            if not np.all(np.isfinite(a)):
                continue
            a = _normalize(a)
            v = _max_cos(a)
            if v < best_val0:
                best_val0 = v
                X0 = a.copy()
        except Exception:
            continue
    if X0 is None:
        X0 = _normalize(rng.standard_normal((59, 22)))
        best_val0 = _max_cos(X0)

    x_best = X0.copy()
    best_val = best_val0

    # Stage 0: direct polish of record
    y, v = _adam_refine(x_best, steps=7000, beta0=50.0, beta1=400.0, lr0=0.008, lr1=0.001)
    if v < best_val:
        best_val = v
        x_best = y.copy()
    y, v = _contact_refine(y, steps=2500, step0=0.005, tol=0.008)
    if v < best_val:
        best_val = v
        x_best = y.copy()

    sigmas = [0.025, 0.06, 0.012, 0.09, 0.03, 0.15, 0.02, 0.045, 0.07, 0.015]
    for k, sigma in enumerate(sigmas):
        if time.time() > deadline - 25.0:
            break
        # find rattlers of current best
        G = x_best @ x_best.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        rowmax = G.max(axis=1)
        ratt = rowmax < (m - 0.008)
        noise = rng.standard_normal((59, 22))
        if np.any(ratt):
            noise[ratt] *= 3.0
        xs = x_best + sigma * noise
        xs = _normalize(xs)
        y, v = _adam_refine(xs, steps=4500, beta0=80.0, beta1=550.0, lr0=0.006, lr1=0.001)
        y2, v2 = _contact_refine(y, steps=1800, step0=0.004, tol=0.006)
        if v2 < best_val:
            best_val = v2
            x_best = y2.copy()
        elif v < best_val:
            best_val = v
            x_best = y.copy()

    if time.time() < deadline - 10.0:
        y, v = _adam_refine(x_best, steps=9000, beta0=150.0, beta1=1000.0, lr0=0.004, lr1=0.0004)
        if v < best_val:
            best_val = v
            x_best = y.copy()
        y, v = _contact_refine(y, steps=3000, step0=0.003, tol=0.004)
        if v < best_val:
            best_val = v
            x_best = y.copy()
        # ultra fine with tiny tol
        y, v = _contact_refine(x_best, steps=1500, step0=0.001, tol=0.0015)
        if v < best_val:
            best_val = v
            x_best = y.copy()

    x_best = _normalize(x_best)
    return x_best
