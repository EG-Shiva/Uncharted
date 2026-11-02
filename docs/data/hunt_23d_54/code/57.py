import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())


def _norm(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return x / n


def _lp_polish(x, deadline, r0=0.012, rmax=0.05):
    n, d = x.shape
    x = _norm(x.copy())
    best = _max_cos(x)
    best_x = x.copy()
    radius = r0
    iu0, iu1 = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-13:
        G = x @ x.T
        c_all = G[iu0, iu1]
        m = float(c_all.max())
        thresh = m - max(20.0 * radius, 1e-9)
        sel = np.where(c_all > thresh)[0]
        if sel.size > 1500:
            part = np.argpartition(c_all, -1500)[-1500:]
            sel = part
        if sel.size == 0:
            break
        c_sel = c_all[sel]
        I = iu0[sel]
        J = iu1[sel]
        p = sel.size
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.empty(p * (2 * d + 1), dtype=np.int64)
        vals = np.empty(p * (2 * d + 1), dtype=np.float64)
        vI = x[J]
        vJ = x[I]
        ar = np.arange(d)
        for k in range(p):
            base = k * (2 * d + 1)
            cols[base:base + d] = I[k] * d + ar
            vals[base:base + d] = vI[k]
            cols[base + d:base + 2 * d] = J[k] * d + ar
            vals[base + d:base + 2 * d] = vJ[k]
            cols[base + 2 * d] = n * d
            vals[base + 2 * d] = -1.0
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv)
        obj[-1] = 1.0
        bounds = [(-radius, radius)] * (n * d) + [(None, None)]
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c_sel, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=bounds, method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y = _norm(y)
        v = _max_cos(y)
        if v < best:
            best = v
            best_x = y.copy()
            x = y
            radius = min(radius * 1.5, rmax)
        else:
            radius *= 0.5
            if radius < 1e-7:
                x = best_x.copy()
        if time.time() >= deadline:
            break
    return best_x, best


def _lse_screen(x_init, steps=350, k0=100.0, k1=600.0, lr=0.005):
    x = x_init.copy()
    n, d = x.shape
    m1 = np.zeros_like(x)
    m2 = np.zeros_like(x)
    b1, b2, eps = 0.9, 0.999, 1e-8
    b1t, b2t = 1.0, 1.0
    for s in range(steps):
        f = s / max(steps - 1, 1)
        k = k0 + (k1 - k0) * f
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        # stable weights
        E = np.exp(k * (G - m))
        np.fill_diagonal(E, 0.0)
        F = E @ x
        dot = np.sum(F * x, axis=1, keepdims=True)
        Gt = F - dot * x
        # Adam update (descent)
        b1t *= b1
        b2t *= b2
        m1 = b1 * m1 + (1.0 - b1) * Gt
        m2 = b2 * m2 + (1.0 - b2) * (Gt * Gt)
        mh1 = m1 / (1.0 - b1t)
        mh2 = m2 / (1.0 - b2t)
        step = lr * mh1 / (np.sqrt(mh2) + eps)
        # cap per-step to keep stable
        x = x - step
        x = _norm(x)
    return x


def _build_modes(x_best, rng, nmodes=30):
    n, d = x_best.shape
    dm = d - 1
    G = x_best @ x_best.T
    iu0, iu1 = np.triu_indices(n, 1)
    c = G[iu0, iu1]
    m = float(c.max())
    slack = m - c
    idx = np.where(slack < 1e-6)[0]
    # keep tightest ~400 for speed
    if idx.size > 420:
        part = np.argpartition(slack, 420)[:420]
        idx = part
    if idx.size < 10:
        part = np.argpartition(slack, min(200, slack.size))[:min(200, slack.size)]
        idx = part
    I = iu0[idx]
    J = iu1[idx]
    p = idx.size
    # tangent bases via SVD of each x_i
    Bs = np.empty((n, d, dm))
    for i in range(n):
        _, _, Vh = np.linalg.svd(x_best[i:i + 1, :], full_matrices=True)
        Bs[i] = Vh[1:, :].T
    # rigidity matrix
    A = np.zeros((p, n * dm))
    for k in range(p):
        i = int(I[k]); j = int(J[k])
        A[k, i * dm:(i + 1) * dm] = x_best[j] @ Bs[i]
        A[k, j * dm:(j + 1) * dm] = x_best[i] @ Bs[j]
    Gram = A @ A.T
    Gram = Gram + 1e-10 * np.eye(p)
    try:
        L = np.linalg.cholesky(Gram)
        def _solve(B):
            # solve Gram W = B via cholesky
            Y = np.linalg.solve(L, B)
            W = np.linalg.solve(L.T, Y)
            return W
    except Exception:
        def _solve(B):
            return np.linalg.solve(Gram, B)
    R = rng.standard_normal((n * dm, nmodes))
    S = A @ R
    W = _solve(S)
    Y = R - A.T @ W
    Q, _ = np.linalg.qr(Y)
    return Q, Bs, I, J


def solve():
    import time as _time
    import numpy as _np
    t0 = _time.time()
    T_HARD = t0 + 170.0
    T_EXPLORE = t0 + 140.0
    rng = _np.random.default_rng(0)
    n, d = 54, 23
    dm = d - 1
    try:
        x0 = _np.load("best.npy").astype(float)
    except Exception:
        x0 = _np.load("record.npy").astype(float)
    try:
        xr = _np.load("record.npy").astype(float)
    except Exception:
        xr = x0.copy()
    x0 = _norm(x0)
    xr = _norm(xr)
    v0 = _max_cos(x0)
    vr = _max_cos(xr)
    if vr < v0:
        x_best, best = xr, vr
    else:
        x_best, best = x0, v0
    if x_best.shape != (n, d):
        tmp = rng.standard_normal((n, d))
        x_best = _norm(tmp)
        best = _max_cos(x_best)

    # build soft modes
    try:
        Q, Bs, _, _ = _build_modes(x_best, rng, nmodes=30)
    except Exception:
        Q = None
        Bs = None
    # fallback bases if needed
    if Bs is None:
        Bs = _np.empty((n, d, dm))
        for i in range(n):
            _, _, Vh = _np.linalg.svd(x_best[i:i + 1, :], full_matrices=True)
            Bs[i] = Vh[1:, :].T
        R = rng.standard_normal((n * dm, 30))
        Q, _ = _np.linalg.qr(R)

    step_set = [0.03, 0.08, 0.15, 0.25]
    nmodes = Q.shape[1]
    trial = 0
    # exploration loop
    while _time.time() < T_EXPLORE and _time.time() < T_HARD - 27.0:
        trial += 1
        # choose steps and combo size cyclically + random
        step_target = step_set[trial % len(step_set)]
        ksize = 2 + (trial % 4)  # 2..5
        if ksize > nmodes:
            ksize = nmodes
        cols = rng.choice(nmodes, size=ksize, replace=False)
        signs = rng.choice(_np.array([-1.0, 1.0]), size=ksize)
        coeff = _np.zeros((n * dm,))
        for c_, s_ in zip(cols, signs):
            coeff += float(s_) * Q[:, int(c_)]
        M = coeff.reshape(n, dm)
        Dx = _np.empty((n, d))
        for i in range(n):
            Dx[i] = Bs[i] @ M[i]
        rms = float(_np.sqrt(_np.sum(Dx * Dx) / n)) + 1e-12
        Dx = Dx * (step_target / rms)
        y = _norm(x_best + Dx)
        # fast screen
        steps = 350
        # shrink steps if time low
        if _time.time() > T_EXPLORE - 20:
            steps = 300
        try:
            ys = _lse_screen(y, steps=steps, k0=100.0, k1=600.0, lr=0.005)
        except Exception:
            continue
        if _time.time() >= T_HARD - 27.0:
            break
        vs = _max_cos(ys)
        # LP-polish only promising
        if vs < best + 0.004:
            budget = 2.2
            deadline = min(_time.time() + budget, T_EXPLORE, T_HARD - 26.0)
            if deadline - _time.time() < 0.4:
                # no time for LP, just greedy check
                if vs < best:
                    x_best, best = ys, vs
                    try:
                        Q, Bs, _, _ = _build_modes(x_best, rng, nmodes=30)
                    except Exception:
                        pass
                continue
            try:
                yp, vp = _lp_polish(ys, deadline, r0=0.012, rmax=0.05)
            except Exception:
                continue
            if vp < best - 1e-12:
                x_best, best = yp, vp
                # rebuild modes in new basin if time remains
                if _time.time() < T_EXPLORE - 5 and _time.time() < T_HARD - 30:
                    try:
                        Q, Bs, _, _ = _build_modes(x_best, rng, nmodes=30)
                    except Exception:
                        pass
        else:
            # not promising, skip LP
            pass
        if trial > 60:
            break

    # final LP polish with remaining time (reserve up to 25s)
    rem_end = min(T_HARD, t0 + 176.0)
    now = _time.time()
    if now < rem_end - 1.0:
        final_deadline = rem_end - 0.5
        # cap final polish to 25s
        if final_deadline - now > 25.0:
            final_deadline = now + 25.0
        try:
            xb, vb = _lp_polish(x_best, final_deadline, r0=0.005, rmax=0.03)
            if vb < best:
                x_best, best = xb, vb
        except Exception:
            pass
    x_best = _norm(x_best)
    return x_best
