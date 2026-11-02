import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())


def _adam_refine(X0, steps, T0, T1, lr0, lr1, deadline=None):
    X = X0.copy()
    m = np.zeros_like(X)
    v = np.zeros_like(X)
    b1 = 0.9
    b2 = 0.999
    eps = 1e-8
    # init best
    G0 = X @ X.T
    np.fill_diagonal(G0, -2.0)
    best_val = float(G0.max())
    best_X = X.copy()
    lT0 = np.log(T0)
    lT1 = np.log(T1)
    ll0 = np.log(lr0)
    ll1 = np.log(lr1)
    for s in range(1, steps + 1):
        if deadline is not None and (s % 200 == 0) and time.time() > deadline:
            break
        frac = s / steps
        T = float(np.exp(lT0 + (lT1 - lT0) * frac))
        lr = float(np.exp(ll0 + (ll1 - ll0) * frac))
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        m0 = float(G.max())
        if m0 < best_val:
            best_val = m0
            best_X = X.copy()
        # E = exp((G-m0)/T), max=1, diag~0, no overflow
        E = np.exp((G - m0) / T)
        S = float(E.sum())
        if not np.isfinite(S) or S == 0:
            break
        Grad = (E @ X) / S
        dot = np.sum(Grad * X, axis=1, keepdims=True)
        Grad -= dot * X
        m *= b1
        m += (1.0 - b1) * Grad
        v *= b2
        v += (1.0 - b2) * (Grad * Grad)
        bc1 = 1.0 - b1 ** s
        bc2 = 1.0 - b2 ** s
        mhat = m / bc1
        vhat = v / bc2
        step = lr * mhat / (np.sqrt(vhat) + eps)
        X = X - step
        nr = np.linalg.norm(X, axis=1, keepdims=True)
        nr[nr == 0] = 1.0
        X /= nr
    return best_X


def _lp_polish(X_init, deadline, radius_init=2e-3):
    X = X_init.copy()
    n, d = X.shape
    best = _max_cos(X)
    best_X = X.copy()
    radius = radius_init
    iu0 = np.triu_indices(n, 1)
    # precompute for eq: row indices
    eq_row = np.repeat(np.arange(n), d)
    eq_col = np.arange(n * d)
    while time.time() < deadline and radius > 1e-13:
        Gfull = X @ X.T
        c_all = Gfull[iu0]
        m = float(c_all.max())
        thresh = max(20.0 * radius, 1e-9)
        slack = m - c_all
        sel = slack < thresh
        cnt = int(sel.sum())
        if cnt == 0:
            radius *= 0.5
            continue
        if cnt > 3500:
            # keep 3500 tightest
            idx = np.argpartition(slack, 3500)[:3500]
            sel = np.zeros_like(sel)
            sel[idx] = True
            cnt = 3500
        I = iu0[0][sel]
        J = iu0[1][sel]
        csel = c_all[sel]
        p = cnt
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        # cols: I*d .. , J*d .. , last
        ar = np.arange(d)
        cols_I = (I[:, None] * d + ar).ravel()
        cols_J = (J[:, None] * d + ar).ravel()
        cols = np.empty(p * (2 * d + 1), dtype=np.int64)
        cols[0::(2 * d + 1)] = cols_I[0::d]
        # easier: build with concatenate
        cols = np.concatenate(
            [np.repeat(I * d, d) + np.tile(ar, p),
             np.repeat(J * d, d) + np.tile(ar, p),
             np.full(p, n * d)], axis=0
        )
        # rows already: need ordering matching vals: for each k, 2d+1 entries
        # rebuild rows to match cols order above (I-block, J-block, t-block)
        rows = np.concatenate(
            [np.repeat(np.arange(p), d),
             np.repeat(np.arange(p), d),
             np.arange(p)], axis=0
        )
        vals = np.concatenate([X[J].ravel(), X[I].ravel(), -np.ones(p)], axis=0)
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((X.ravel(), (eq_row, eq_col)), shape=(n, nv)).tocsr()
        obj = np.zeros(nv)
        obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq,
                          b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                          method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = X + res.x[:-1].reshape(n, d)
        nr = np.linalg.norm(y, axis=1, keepdims=True)
        nr[nr == 0] = 1.0
        y /= nr
        v = _max_cos(y)
        if v < best:
            best = v
            best_X = y.copy()
            X = y
            radius = min(radius * 1.5, 0.02)
        else:
            radius *= 0.5
    return best_X


def solve():
    t0 = time.time()
    deadline = t0 + 168.0
    rng = np.random.default_rng(0)
    try:
        Xb = np.load("best.npy").astype(float)
    except Exception:
        try:
            Xb = np.load("record.npy").astype(float)
        except Exception:
            Xb = rng.standard_normal(size=(453, 5))
    if Xb.shape != (453, 5):
        # try to reshape / truncate / pad deterministically
        flat = Xb.reshape(-1, 5)
        if flat.shape[0] >= 453:
            Xb = flat[:453].copy()
        else:
            extra = rng.standard_normal(size=(453 - flat.shape[0], 5))
            Xb = np.vstack([flat, extra])
    nr = np.linalg.norm(Xb, axis=1, keepdims=True)
    nr[nr == 0] = 1.0
    Xb /= nr
    best = _max_cos(Xb)

    # Phase A: direct smoothing from record (unjammed -> quick gain)
    if time.time() < deadline - 40:
        Xc = _adam_refine(Xb, steps=3000, T0=0.006, T1=0.00020,
                          lr0=0.006, lr1=0.00030, deadline=deadline - 32)
        vc = _max_cos(Xc)
        if vc < best:
            Xb = Xc
            best = vc

    sigmas = [0.008, 0.003, 0.015, 0.005, 0.002, 0.010]
    T1s = [0.00025, 0.00015, 0.00035]
    lr1s = [0.00040, 0.00025, 0.00060]
    k = 0
    while time.time() < deadline - 35 and k < 30:
        sigma = sigmas[k % len(sigmas)]
        T1 = T1s[k % len(T1s)]
        lr1 = lr1s[k % len(lr1s)]
        noise = rng.standard_normal(size=Xb.shape)
        if k % 3 == 2:
            # boost high-degree (near-max) points
            G = Xb @ Xb.T
            np.fill_diagonal(G, -2.0)
            mask = G > (best - 0.002)
            deg = mask.sum(axis=1).astype(float)
            mx = deg.max()
            if mx > 0:
                w = (deg / mx)[:, None]
                noise *= (0.5 + 1.5 * w)
        X0 = Xb + sigma * noise
        nr = np.linalg.norm(X0, axis=1, keepdims=True)
        nr[nr == 0] = 1.0
        X0 /= nr
        # steps adapt to remaining time
        rem = (deadline - 35) - time.time()
        steps = 2200
        if rem < 6:
            steps = 800
        elif rem < 12:
            steps = 1400
        Xc = _adam_refine(X0, steps=steps, T0=0.006, T1=T1,
                          lr0=0.008, lr1=lr1, deadline=deadline - 32)
        vc = _max_cos(Xc)
        if vc < best - 1e-12:
            Xb = Xc
            best = vc
        k += 1

    # Phase fine-tune with very small temperature
    if time.time() < deadline - 28:
        Xc = _adam_refine(Xb, steps=3500, T0=0.0010, T1=0.00012,
                          lr0=0.0020, lr1=0.00020, deadline=deadline - 20)
        vc = _max_cos(Xc)
        if vc < best:
            Xb = Xc
            best = vc

    # Final LP polish
    if time.time() < deadline - 5:
        Xc = _lp_polish(Xb, deadline=deadline - 1.5)
        vc = _max_cos(Xc)
        if vc < best:
            Xb = Xc
            best = vc

    nr = np.linalg.norm(Xb, axis=1, keepdims=True)
    nr[nr == 0] = 1.0
    Xb /= nr
    return Xb
