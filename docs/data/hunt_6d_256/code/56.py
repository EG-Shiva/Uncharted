import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _lp_polish(x, deadline, r0=2e-3, cap=3000):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    try:
        best = _max_cos(x)
    except Exception:
        return x
    radius = r0
    iuI, iuJ = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        G = x @ x.T
        c = G[iuI, iuJ]
        m = float(c.max())
        margin = max(20.0 * radius, 8e-5)
        sel = c > (m - margin)
        I = iuI[sel]; J = iuJ[sel]
        csel = c[sel]
        p = len(I)
        if p == 0:
            radius *= 0.5
            continue
        if p > cap:
            part = np.argpartition(-csel, cap)[:cap]
            I = I[part]; J = J[part]; csel = csel[part]
            p = cap
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, n * d + 1)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))),
                              shape=(n, n * d + 1)).tocsr()
        except Exception:
            break
        obj = np.zeros(n * d + 1)
        obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                          method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        val = _max_cos(y)
        if val < best:
            x, best = y, val
            radius = min(radius * 1.4, 0.02)
        else:
            radius *= 0.5
    return x

def _adam_lse_full(x0, steps, K0, K1, lr, deadline=1e18):
    x = x0.copy()
    n, d = x.shape
    m = np.zeros_like(x)
    v = np.zeros_like(x)
    b1, b2, eps = 0.9, 0.999, 1e-8
    b1p, b2p = 1.0, 1.0
    la0, la1 = np.log(K0), np.log(K1)
    best_val = _max_cos(x)
    best_x = x.copy()
    for t in range(1, steps + 1):
        if time.time() > deadline and t > 10:
            break
        K = float(np.exp(la0 + (la1 - la0) * t / steps))
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        gmax = float(G.max())
        W = np.exp(K * (G - gmax))
        np.fill_diagonal(W, 0.0)
        S = float(W.sum() * 0.5)
        if not np.isfinite(S) or S <= 0:
            continue
        grad = (W @ x) / S
        dot = np.sum(grad * x, axis=1, keepdims=True)
        gt = grad - dot * x
        b1p *= b1
        b2p *= b2
        m *= b1
        m += (1 - b1) * gt
        v *= b2
        v += (1 - b2) * (gt * gt)
        mhat = m / (1 - b1p)
        vhat = v / (1 - b2p)
        step = lr * mhat / (np.sqrt(vhat) + eps)
        sdot = np.sum(step * x, axis=1, keepdims=True)
        step = step - sdot * x
        x = x - step
        x /= np.linalg.norm(x, axis=1, keepdims=True)
        if t % 200 == 0 or t == steps:
            val = _max_cos(x)
            if val < best_val:
                best_val = val
                best_x = x.copy()
    return best_x, best_val

def solve():
    import time as _time
    import numpy as _np
    t0 = _time.time()
    T_END = t0 + 172.0
    SYM_END = t0 + 70.0
    rng = _np.random.default_rng(0)
    # incumbent
    x_inc = None
    v_inc = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            a = _np.load(fn)
            a = _np.asarray(a, dtype=float)
            if a.shape == (256, 6) and _np.all(_np.isfinite(a)):
                a = a / _np.linalg.norm(a, axis=1, keepdims=True)
                G = a @ a.T
                _np.fill_diagonal(G, -2.0)
                vv = float(G.max())
                if vv < v_inc:
                    v_inc = vv
                    x_inc = a.copy()
        except Exception:
            continue
    if x_inc is None:
        x_inc = rng.normal(size=(256, 6))
        x_inc /= _np.linalg.norm(x_inc, axis=1, keepdims=True)
        v_inc = _max_cos(x_inc)
    # base hypercube H (64,6) / sqrt6
    H = _np.empty((64, 6), dtype=float)
    for k in range(64):
        for j in range(6):
            H[k, j] = 1.0 if ((k >> j) & 1) == 0 else -1.0
    H /= _np.sqrt(6.0)
    # cross mask: 1 if different 64-blocks
    bid = _np.repeat(_np.arange(4), 64)
    mask = (bid[:, None] != bid[None, :]).astype(float)
    iu0, iu1 = _np.triu_indices(6, 1)  # 15 entries
    I6 = _np.eye(6)
    nrest = 10
    steps_sym = 900
    K0s, K1s = 200.0, 5000.0
    la0s, la1s = _np.log(K0s), _np.log(K1s)
    best_sym_val = 1e9
    best_sym_X = None
    # run restarts
    for rs in range(nrest):
        if _time.time() > SYM_END:
            break
        rs_rng = _np.random.default_rng(1000 + rs)
        # init 45 params: skew entries ~ N(0,0.55)
        p = rs_rng.normal(scale=0.55, size=45)
        m = _np.zeros(45); v = _np.zeros(45)
        b1, b2, eps = 0.9, 0.999, 1e-8
        b1p, b2p = 1.0, 1.0
        lr = 0.02
        best_p = p.copy()
        best_v = 1e9
        # quick eval of init to set best_v
        try:
            As = []
            Rs = []
            for c in range(3):
                vv = p[c*15:(c+1)*15]
                A = _np.zeros((6, 6))
                A[iu0, iu1] = vv
                A[iu1, iu0] = -vv
                As.append(A)
                Minv = _np.linalg.inv(I6 + A)
                R = (I6 - A) @ Minv
                Rs.append(R)
            X0 = _np.vstack([H, H @ Rs[0], H @ Rs[1], H @ Rs[2]])
            # X rows already unit (rotation preserves norm up to numerics)
            X0 /= _np.linalg.norm(X0, axis=1, keepdims=True)
            G0 = X0 @ X0.T
            _np.fill_diagonal(G0, -2.0)
            best_v = float(G0.max())
            best_p = p.copy()
            best_Xr = X0.copy()
        except Exception:
            continue
        for t in range(1, steps_sym + 1):
            if _time.time() > SYM_END and t > 50:
                break
            K = float(_np.exp(la0s + (la1s - la0s) * t / steps_sym))
            # forward
            As = []; Rs = []; Minvs = []
            ok = True
            for c in range(3):
                vv = p[c*15:(c+1)*15]
                A = _np.zeros((6, 6))
                A[iu0, iu1] = vv
                A[iu1, iu0] = -vv
                As.append(A)
                try:
                    Minv = _np.linalg.inv(I6 + A)
                except Exception:
                    ok = False
                    break
                Minvs.append(Minv)
                Rs.append((I6 - A) @ Minv)
            if not ok:
                continue
            X = _np.vstack([H, H @ Rs[0], H @ Rs[1], H @ Rs[2]])
            # renormalize (should be unit)
            X /= _np.linalg.norm(X, axis=1, keepdims=True)
            G = X @ X.T
            _np.fill_diagonal(G, -2.0)
            gmax = float(G.max())
            # track true max
            if gmax < best_v:
                best_v = gmax
                best_p = p.copy()
                best_Xr = X.copy()
            # weights cross only
            # exp may underflow -> fine
            W = _np.exp(K * (G - gmax)) * mask
            S = float(W.sum() * 0.5)
            if not _np.isfinite(S) or S <= 0:
                continue
            GradX = (W @ X) / S  # (256,6)
            # backprop to params
            gvec = _np.empty(45)
            for c in range(3):
                Gb = GradX[(c+1)*64:(c+2)*64, :]  # (64,6)
                GR = H.T @ Gb  # (6,6)
                R = Rs[c]; Minv = Minvs[c]
                # Gfull = -(I+R)^T GR Minv^T
                Gfull = -(I6 + R.T) @ GR @ Minv.T
                gvec[c*15:(c+1)*15] = Gfull[iu0, iu1] - Gfull[iu1, iu0]
            # Adam step (minimize LSE)
            b1p *= b1; b2p *= b2
            m *= b1; m += (1 - b1) * gvec
            v *= b2; v += (1 - b2) * (gvec * gvec)
            mh = m / (1 - b1p); vh = v / (1 - b2p)
            p = p - lr * mh / (_np.sqrt(vh) + eps)
        # end steps
        if best_v < best_sym_val:
            best_sym_val = best_v
            best_sym_X = best_Xr.copy()
    # release symmetry
    x_best = x_inc.copy()
    best = v_inc
    x_rel = None
    v_rel = 1e9
    if best_sym_X is not None:
        # perturb 1e-3 tangent
        prng = _np.random.default_rng(7)
        sig = 1e-3
        R = prng.normal(size=(256, 6))
        R -= _np.sum(R * best_sym_X, axis=1, keepdims=True) * best_sym_X
        x_rel = best_sym_X + sig * R
        x_rel /= _np.linalg.norm(x_rel, axis=1, keepdims=True)
        v_rel = _max_cos(x_rel)
        # full LSE-Adam on released
        rem = SYM_END - _time.time()
        # ensure we have time; run Adam with deadline
        try:
            dl = min(_time.time() + 60.0, T_END - 30.0)
            if _time.time() < dl:
                # two-stage: coarse then fine
                xb, vb = _adam_lse_full(x_rel, 900, 150.0, 3000.0, 0.006, deadline=dl)
                x_rel, v_rel = xb, vb
        except Exception:
            pass
        try:
            dl2 = min(_time.time() + 15.0, T_END - 12.0)
            if _time.time() < dl2:
                x_rel = _lp_polish(x_rel, dl2, r0=1.2e-3)
                v_rel = _max_cos(x_rel)
        except Exception:
            pass
        if v_rel < best:
            x_best = x_rel.copy()
            best = v_rel
    # also polish incumbent a bit with remaining time to squeeze
    try:
        if _time.time() < T_END - 10:
            dl = min(_time.time() + 20.0, T_END - 6.0)
            xb, vb = _adam_lse_full(x_best, 500, 400.0, 3000.0, 0.004, deadline=dl)
            if vb < best:
                x_best, best = xb, vb
    except Exception:
        pass
    try:
        if _time.time() < T_END - 3:
            x_best = _lp_polish(x_best, T_END - 1.0, r0=8e-4)
    except Exception:
        pass
    x_best = x_best / _np.linalg.norm(x_best, axis=1, keepdims=True)
    # sanitize
    x_best = _np.where(_np.isfinite(x_best), x_best, 0.0)
    nr = _np.linalg.norm(x_best, axis=1, keepdims=True)
    nr[nr == 0] = 1.0
    x_best = x_best / nr
    return x_best
