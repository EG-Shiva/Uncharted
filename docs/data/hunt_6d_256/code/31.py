import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _lse_adam(x0, steps, T0, T1, lr0, lr1, rng_seed_off=0):
    x = x0.copy()
    n, d = x.shape
    m = np.zeros_like(x)
    v = np.zeros_like(x)
    b1, b2, eps = 0.9, 0.999, 1e-8
    lT0, lT1 = np.log(T0), np.log(T1)
    ll0, ll1 = np.log(lr0), np.log(lr1)
    best_val = _max_cos(x)
    best_x = x.copy()
    b1p, b2p = 1.0, 1.0
    for s in range(1, steps + 1):
        f = s / max(steps, 1)
        T = float(np.exp(lT0 + (lT1 - lT0) * f))
        lr = float(np.exp(ll0 + (ll1 - ll0) * f))
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        gmax = float(G.max())
        E = np.exp((G - gmax) / T)
        np.fill_diagonal(E, 0.0)
        S = float(E.sum() * 0.5)
        if not np.isfinite(S) or S <= 0:
            continue
        Grad = (E @ x) / S
        dot = np.sum(Grad * x, axis=1, keepdims=True)
        Gt = Grad - dot * x
        b1p *= b1
        b2p *= b2
        m *= b1
        m += (1 - b1) * Gt
        v *= b2
        v += (1 - b2) * (Gt * Gt)
        mhat = m / (1 - b1p)
        vhat = v / (1 - b2p)
        step = lr * mhat / (np.sqrt(vhat) + eps)
        sdot = np.sum(step * x, axis=1, keepdims=True)
        step = step - sdot * x
        x = x - step
        x /= np.linalg.norm(x, axis=1, keepdims=True)
        if s % 150 == 0 or s == steps:
            val = _max_cos(x)
            if val < best_val:
                best_val = val
                best_x = x.copy()
    return best_x, best_val

def _riesz_adam(x0, steps, s_pow, lr):
    x = x0.copy()
    n, d = x.shape
    m = np.zeros_like(x)
    v = np.zeros_like(x)
    b1, b2, eps = 0.9, 0.999, 1e-8
    b1p, b2p = 1.0, 1.0
    best_val = _max_cos(x)
    best_x = x.copy()
    sp = float(s_pow)
    for t in range(1, steps + 1):
        G = x @ x.T
        # squared distance
        R2 = 2.0 - 2.0 * G
        np.fill_diagonal(R2, 2.0)
        np.clip(R2, 1e-9, 4.0, out=R2)
        # W = s * r^{-s-2} ; r^2 = R2
        # r^{-s-2} = R2^{-(s+2)/2}
        W = np.power(R2, -(sp + 2.0) * 0.5) * sp
        np.fill_diagonal(W, 0.0)
        # normalize scale to avoid huge steps: divide by median?
        # use mean scaling
        ms = float(np.mean(np.abs(W)) + 1e-12)
        # scale to O(1)
        W = W / ms
        T1 = W @ x
        wdot = (W * G).sum(axis=1, keepdims=True)
        Gt = T1 - wdot * x
        # clip grad
        gn = np.linalg.norm(Gt, axis=1, keepdims=True)
        # Adam
        b1p *= b1
        b2p *= b2
        m *= b1
        m += (1 - b1) * Gt
        v *= b2
        v += (1 - b2) * (Gt * Gt)
        mhat = m / (1 - b1p)
        vhat = v / (1 - b2p)
        step = lr * mhat / (np.sqrt(vhat) + eps)
        sdot = np.sum(step * x, axis=1, keepdims=True)
        step = step - sdot * x
        x = x - step
        x /= np.linalg.norm(x, axis=1, keepdims=True)
        if t % 150 == 0 or t == steps:
            val = _max_cos(x)
            if val < best_val:
                best_val = val
                best_x = x.copy()
    return best_x, best_val

def _lp_polish(x, deadline, r0=2e-3, cap=4000):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = r0
    iuI, iuJ = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        G = x @ x.T
        c = G[iuI, iuJ]
        m = float(c.max())
        margin = max(20.0 * radius, 5e-5)
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
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, n * d + 1)).tocsr()
        except Exception:
            break
        obj = np.zeros(n * d + 1)
        obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
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
            radius = min(radius * 1.4, 0.01)
        else:
            radius *= 0.5
    return x

def _cube_base():
    M = np.empty((64, 6), dtype=float)
    for i in range(64):
        for k in range(6):
            M[i, k] = 1.0 if ((i >> k) & 1) else -1.0
    M /= np.sqrt(6.0)
    return M

def _rand_orth(rng, d):
    A = rng.normal(size=(d, d))
    Q, R = np.linalg.qr(A)
    s = np.sign(np.diag(R))
    s[s == 0] = 1.0
    Q = Q * s[None, :]
    if np.linalg.det(Q) < 0:
        Q[:, 0] = -Q[:, 0]
    return Q

def _greedy_points(x, deadline, rng, batch=2048):
    n, d = x.shape
    best = _max_cos(x)
    while time.time() < deadline:
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        prow = G.max(axis=1)
        order = np.argsort(-prow)[:24]
        improved = False
        for idx_pos, ii in enumerate(order):
            if time.time() >= deadline:
                break
            Gm = G.copy()
            Gm[ii, :] = -2.0
            Gm[:, ii] = -2.0
            other_max = float(Gm.max())
            if other_max >= best - 1e-12:
                # still can improve if we get below best? no, global max >= other_max
                if other_max >= best:
                    continue
            cur_row_max = float(prow[ii])
            # candidate sigmas
            for sig in (0.03, 0.012, 0.005):
                R = rng.normal(size=(batch, d))
                R -= (R @ x[ii])[:, None] * x[ii][None, :]
                nr = np.linalg.norm(R, axis=1, keepdims=True)
                nr[nr == 0] = 1.0
                R /= nr
                C = x[ii][None, :] + sig * R
                C /= np.linalg.norm(C, axis=1, keepdims=True)
                D = C @ x.T
                D[:, ii] = -2.0
                lm = D.max(axis=1)
                k = int(np.argmin(lm))
                bv = float(lm[k])
                glob = bv if bv > other_max else other_max
                if glob < best - 1e-12:
                    x[ii] = C[k]
                    G = x @ x.T
                    np.fill_diagonal(G, -2.0)
                    prow = G.max(axis=1)
                    best = glob
                    improved = True
                    break
            if improved:
                break
        if not improved:
            break
    return x

def solve():
    import time as _time
    import numpy as _np
    t0 = _time.time()
    T_END = t0 + 171.0
    ANNEAL_END = t0 + 128.0
    LP_END = t0 + 162.0
    rng = _np.random.default_rng(12345)
    n, d = 256, 6
    x_best = None
    best = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            a = _np.load(fn)
            a = _np.asarray(a, dtype=float)
            if a.shape == (256, 6) and _np.all(_np.isfinite(a)):
                a = a / _np.linalg.norm(a, axis=1, keepdims=True)
                vv = _max_cos(a)
                if vv < best:
                    best = vv
                    x_best = a.copy()
        except Exception:
            continue
    if x_best is None:
        x_best = rng.normal(size=(n, d))
        x_best /= _np.linalg.norm(x_best, axis=1, keepdims=True)
        best = _max_cos(x_best)
    cube = _cube_base()
    # ---- Phase A: symmetric cube-union exploration ----
    # try several rotations; each cheap
    n_cube_tries = 7
    for trial in range(n_cube_tries):
        if _time.time() > ANNEAL_END - 8.0:
            break
        # 4 rotated cubes
        parts = []
        for k in range(4):
            Q = _rand_orth(rng, d)
            parts.append(cube @ Q.T)
        x0 = _np.vstack(parts)
        # shuffle to avoid ordering bias
        perm = rng.permutation(n)
        x0 = x0[perm]
        # Riesz spread (balanced)
        try:
            xr, _ = _riesz_adam(x0, 350, 4.0, 0.008)
        except Exception:
            xr = x0
        if _time.time() > ANNEAL_END - 5.0:
            break
        try:
            # moderate LSE to equalize
            xm, vm = _lse_adam(xr, 450, 0.03, 0.0012, 0.010, 0.004)
        except Exception:
            continue
        # sharp finish short
        try:
            xs, vs = _lse_adam(xm, 350, 0.002, 0.00035, 0.006, 0.003)
        except Exception:
            xs, vs = xm, vm
        if vs < best - 1e-12:
            best = vs
            x_best = xs.copy()
        # also try keeping moderate one if more balanced (count near-max)
        # balance metric: count pairs within 2e-4
        # if moderate has many contacts but slightly worse max, keep as seed for hopping?
        # we just keep best max; diversity comes from next trials
    # ---- Phase B: random + riesz diversity (2 tries) ----
    for trial in range(2):
        if _time.time() > ANNEAL_END - 4.0:
            break
        x0 = rng.normal(size=(n, d))
        x0 /= _np.linalg.norm(x0, axis=1, keepdims=True)
        try:
            xr, _ = _riesz_adam(x0, 300, 5.0, 0.010)
            xm, vm = _lse_adam(xr, 400, 0.02, 0.001, 0.010, 0.004)
            xs, vs = _lse_adam(xm, 300, 0.002, 0.0004, 0.006, 0.003)
        except Exception:
            continue
        if vs < best:
            best = vs
            x_best = xs.copy()
    # ---- Phase C: hopping around incumbent with two scales ----
    hop = 0
    while _time.time() < ANNEAL_END:
        hop += 1
        rem = ANNEAL_END - _time.time()
        if rem < 4.0:
            break
        x = x_best.copy()
        # alternate kick types
        if hop % 3 == 1:
            # global small jitter + one cap rotation to change topology
            x += 0.025 * rng.normal(size=(n, d))
            x /= _np.linalg.norm(x, axis=1, keepdims=True)
            steps, T0, T1, l0, l1 = 500, 0.015, 0.0006, 0.008, 0.003
        elif hop % 3 == 2:
            # stronger patch shatter on worst neighbourhood
            G = x @ x.T
            _np.fill_diagonal(G, -2.0)
            flat = int(_np.argmax(G))
            ai = flat // n
            nb = _np.argsort(-G[ai])[:14]
            x[nb] += 0.09 * rng.normal(size=(14, d))
            x += 0.008 * rng.normal(size=(n, d))
            x /= _np.linalg.norm(x, axis=1, keepdims=True)
            steps, T0, T1, l0, l1 = 500, 0.010, 0.0004, 0.009, 0.003
        else:
            # orthogonal remix of random subset (re-basin)
            idx = rng.choice(n, size=32, replace=False)
            Q = _rand_orth(rng, d)
            # rotate subset around its mean
            mu = x[idx].mean(axis=0)
            mu /= _np.linalg.norm(mu)
            # apply Q in tangent-ish way
            x[idx] = (x[idx] @ Q.T)
            x[idx] /= _np.linalg.norm(x[idx], axis=1, keepdims=True)
            x += 0.012 * rng.normal(size=(n, d))
            x /= _np.linalg.norm(x, axis=1, keepdims=True)
            steps, T0, T1, l0, l1 = 550, 0.020, 0.0007, 0.008, 0.003
        try:
            xn, vn = _lse_adam(x, steps, T0, T1, l0, l1)
        except Exception:
            continue
        if vn < best - 2e-9:
            best = vn
            x_best = xn.copy()
        elif vn < best + 5e-5 and hop % 3 == 0:
            # allow side-step to more balanced basin if contact count higher
            # count near contacts
            Gc = xn @ xn.T
            _np.fill_diagonal(Gc, -2.0)
            cnt_new = int(_np.sum(Gc > vn - 3e-4))
            Go = x_best @ x_best.T
            _np.fill_diagonal(Go, -2.0)
            cnt_old = int(_np.sum(Go > best - 3e-4))
            if cnt_new > cnt_old + 200 and vn < best + 1e-5:
                best = vn
                x_best = xn.copy()
    # ---- Phase D: final polish ----
    try:
        if _time.time() < LP_END:
            x_best = _lp_polish(x_best, min(LP_END - 6.0, _time.time() + 18.0), r0=2e-3, cap=4000)
            best = _max_cos(x_best)
    except Exception:
        pass
    try:
        if _time.time() < T_END - 3.0:
            x_best = _greedy_points(x_best, min(T_END - 2.0, _time.time() + 10.0), rng, batch=2048)
            best = _max_cos(x_best)
    except Exception:
        pass
    try:
        if _time.time() < T_END - 1.0:
            x_best = _lp_polish(x_best, T_END - 0.8, r0=6e-4, cap=5000)
    except Exception:
        pass
    x_best = x_best / _np.linalg.norm(x_best, axis=1, keepdims=True)
    return x_best
