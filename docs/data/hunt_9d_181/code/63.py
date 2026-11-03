import time
import numpy as np
from scipy.optimize import linprog, linear_sum_assignment
from scipy.linalg import orthogonal_procrustes

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _norm(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n = np.where(n < 1e-12, 1.0, n)
    return X / n

def _lse_adam(X, steps, k0=80.0, k1=400.0, lr0=0.02, lr1=0.002):
    X = _norm(X.copy())
    n, d = X.shape
    m = np.zeros_like(X)
    v = np.zeros_like(X)
    b1, b2, eps = 0.9, 0.999, 1e-8
    for s in range(steps):
        frac = s / max(steps - 1, 1)
        k = k0 + (k1 - k0) * frac
        lr = lr0 + (lr1 - lr0) * frac
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        try:
            gmax = float(G.max())
        except Exception:
            break
        W = np.exp(k * (G - gmax))
        np.fill_diagonal(W, 0.0)
        Gr = W @ X
        dot = np.einsum('ij,ij->i', Gr, X)[:, None]
        Gt = Gr - dot * X
        t = s + 1
        m = b1 * m + (1.0 - b1) * Gt
        v = b2 * v + (1.0 - b2) * (Gt * Gt)
        mh = m / (1.0 - b1 ** t)
        vh = v / (1.0 - b2 ** t)
        upd = mh / (np.sqrt(vh) + eps)
        X = X - lr * upd
        X = _norm(X)
        if not np.all(np.isfinite(X)):
            break
    return _norm(X)

def _gen_frames(rng):
    X = []
    while len(X) < 181:
        A = rng.standard_normal((9, 9))
        Q, _ = np.linalg.qr(A)
        signs = rng.choice(np.array([-1.0, 1.0]), size=9)
        Q = Q * signs[:, None]
        for i in range(9):
            X.append(Q[i])
            if len(X) >= 181:
                break
    return _norm(np.array(X))

def _gen_sparse(rng):
    pts = []
    seen = set()
    tries = 0
    while len(pts) < 181 and tries < 10000:
        tries += 1
        k = 2 if rng.random() < 0.5 else 3
        idx = tuple(sorted(rng.choice(9, size=k, replace=False).tolist()))
        sgn = tuple(rng.choice(np.array([-1, 1]), size=k).tolist())
        key = (k, idx, sgn)
        if key in seen:
            continue
        seen.add(key)
        v = np.zeros(9)
        for a, s in zip(idx, sgn):
            v[a] = float(s) / np.sqrt(k)
        v = v + rng.standard_normal(9) * 0.01
        v = v / np.linalg.norm(v)
        pts.append(v)
    while len(pts) < 181:
        v = rng.standard_normal(9)
        pts.append(v / np.linalg.norm(v))
    return _norm(np.array(pts))

def _e8_pool():
    pool = []
    for i in range(8):
        for j in range(i + 1, 8):
            for si in (-1.0, 1.0):
                for sj in (-1.0, 1.0):
                    v = np.zeros(8)
                    v[i] = si
                    v[j] = sj
                    pool.append(v)
    for mask in range(256):
        bits = [(mask >> b) & 1 for b in range(8)]
        if sum(bits) % 2 != 0:
            continue
        v = np.array([0.5 if b == 0 else -0.5 for b in bits], dtype=float)
        pool.append(v)
    P = np.array(pool, dtype=float)
    P = P / np.sqrt(2.0)
    return P

_E8 = None
def _gen_e8(rng):
    global _E8
    if _E8 is None:
        try:
            _E8 = _e8_pool()
        except Exception:
            _E8 = rng.standard_normal((240, 8)) / np.sqrt(8.0)
    P = _E8
    sel = rng.choice(P.shape[0], size=181, replace=False)
    Q8 = P[sel]
    ninth = rng.standard_normal(181) * 0.25
    X = np.zeros((181, 9))
    X[:, :8] = Q8
    X[:, 8] = ninth
    A = rng.standard_normal((9, 9))
    R, _ = np.linalg.qr(A)
    X = X @ R.T
    return _norm(X)

def _gen_antipodal(rng):
    m = 90
    A = rng.standard_normal((m, 9))
    A = _norm(A)
    X = np.zeros((181, 9))
    X[:m] = A
    X[m:2 * m] = -A
    v = rng.standard_normal(9)
    X[180] = v / np.linalg.norm(v)
    X = X + rng.standard_normal((181, 9)) * 0.005
    return _norm(X)

def _lp_polish(X, budget, t_dead=None):
    t0 = time.time()
    x = _norm(X.copy())
    n, d = x.shape
    try:
        best = _max_cos(x)
    except Exception:
        return x
    radius = 1e-3
    iu0 = np.triu_indices(n, 1)
    ar = np.arange(d)
    it = 0
    while True:
        if time.time() - t0 >= budget:
            break
        if t_dead is not None and time.time() >= t_dead:
            break
        if radius < 1e-14 or it > 25:
            break
        it += 1
        G = x @ x.T
        c = G[iu0]
        mm = float(c.max())
        sel = c > mm - max(20.0 * radius, 1e-9)
        I = iu0[0][sel]
        J = iu0[1][sel]
        p = len(I)
        if p == 0:
            radius *= 0.5
            continue
        if p > 1500:
            cs = c[sel]
            idx = np.argpartition(-cs, 1500)[:1500]
            I = I[idx]
            J = J[idx]
            cs2 = cs[idx]
        else:
            cs2 = c[sel]
            p = len(I)
        nv = n * d + 1
        try:
            from scipy.sparse import coo_matrix
            cols_mat = np.empty((p, 2 * d + 1), dtype=np.int64)
            cols_mat[:, :d] = I[:, None] * d + ar
            cols_mat[:, d:2 * d] = J[:, None] * d + ar
            cols_mat[:, 2 * d] = n * d
            rows = np.repeat(np.arange(p), 2 * d + 1)
            cols = cols_mat.ravel()
            vals_mat = np.empty((p, 2 * d + 1))
            vals_mat[:, :d] = x[J]
            vals_mat[:, d:2 * d] = x[I]
            vals_mat[:, 2 * d] = -1.0
            vals = vals_mat.ravel()
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv)
            obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-cs2, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        try:
            y = x + res.x[:-1].reshape(n, d)
            y = _norm(y)
            vv = _max_cos(y)
        except Exception:
            radius *= 0.5
            continue
        if vv < best:
            x, best = y, vv
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def solve():
    t0 = time.time()
    DEADLINE = 172.0
    rng = np.random.default_rng(20261103)
    best_x = None
    best_v = 1e9
    for fn in ["best.npy", "record.npy"]:
        try:
            y = np.load(fn)
            y = np.asarray(y, dtype=float)
            if y.shape != (181, 9):
                continue
            y = _norm(y)
            if not np.all(np.finite(y)):
                continue
            vv = _max_cos(y)
            if vv < best_v:
                best_v = vv
                best_x = y.copy()
        except Exception:
            continue
    if best_x is None:
        y = rng.standard_normal((181, 9))
        best_x = _norm(y)
        best_v = _max_cos(best_x)
    # build 12 structured seeds
    seeds = []
    try:
        for _ in range(3):
            seeds.append(_gen_frames(rng))
        for _ in range(3):
            seeds.append(_gen_sparse(rng))
        for _ in range(3):
            seeds.append(_gen_e8(rng))
        for _ in range(3):
            seeds.append(_gen_antipodal(rng))
    except Exception:
        pass
    # optimise each with LSE-Adam (ramp 80->400)
    outs = []
    vals = []
    STEPS = 700
    for s in seeds:
        if time.time() - t0 > 118.0:
            break
        try:
            o = _lse_adam(s, steps=STEPS, k0=80.0, k1=400.0, lr0=0.02, lr1=0.002)
            vv = _max_cos(o)
            outs.append(o)
            vals.append(vv)
            if vv < best_v:
                best_v = vv
                best_x = o.copy()
        except Exception:
            continue
    # recombination of two best distinct outcomes
    try:
        if len(outs) >= 2:
            order = np.argsort(np.array(vals))
            A = outs[int(order[0])]
            B = None
            for k in order[1:]:
                C = outs[int(k)]
                # distinctness: mean abs corr
                try:
                    Ga = A @ A.T
                    Gc = C @ C.T
                    diff = float(np.mean(np.abs(np.sort(Ga, axis=1) - np.sort(Gc, axis=1))))
                except Exception:
                    diff = 1.0
                if diff > 1e-6:
                    B = C
                    break
            if B is None:
                B = outs[int(order[1])]
            # Hungarian on signatures (sorted inner-product rows)
            GA = A @ A.T
            GB = B @ B.T
            SA = np.sort(GA, axis=1)
            SB = np.sort(GB, axis=1)
            # cost matrix
            diff = SA[:, None, :] - SB[None, :, :]
            Cost = np.einsum('ijk,ijk->ij', diff, diff)
            ri, ci = linear_sum_assignment(Cost)
            perm = np.empty(181, dtype=int)
            perm[ri] = ci
            Bm = B[perm]
            try:
                R, _ = orthogonal_procrustes(Bm, A)
            except Exception:
                R = np.eye(9)
            Bal = Bm @ R.T if R.shape == (9, 9) else Bm @ R
            # slerp children at t=0.25,0.5,0.75
            dots = np.einsum('ij,ij->i', A, Bal)
            dots = np.clip(dots, -1.0, 1.0)
            om = np.arccos(dots)
            for tt in (0.5, 0.25, 0.75):
                if time.time() - t0 > 135.0:
                    break
                Child = np.empty_like(A)
                so = np.sin(om)
                small = so < 1e-12
                # slerp
                w1 = np.sin((1.0 - tt) * om) / np.where(small, 1.0, so)
                w2 = np.sin(tt * om) / np.where(small, 1.0, so)
                Child = w1[:, None] * A + w2[:, None] * Bal
                # antipodal fallback: linear avg
                lin = (A + Bal) * 0.5
                ln = np.linalg.norm(lin, axis=1, keepdims=True)
                lin = lin / np.where(ln < 1e-12, 1.0, ln)
                Child[small] = lin[small]
                # opposite points (dot~-1): perturb
                opp = dots < -0.9999
                if np.any(opp):
                    Child[opp] = _norm(A[opp] + rng.standard_normal((int(np.sum(opp)), 9)) * 0.01)
                Child = _norm(Child)
                try:
                    Child = _lse_adam(Child, steps=500, k0=80.0, k1=400.0, lr0=0.015, lr1=0.002)
                    vv = _max_cos(Child)
                    if vv < best_v:
                        best_v = vv
                        best_x = Child.copy()
                except Exception:
                    continue
    except Exception:
        pass
    # final smooth + LP polish of current best
    try:
        remain = DEADLINE - (time.time() - t0) - 1.0
        if remain > 6.0 and best_x is not None:
            # quick extra smooth on best
            try:
                if remain > 20.0:
                    best_x = _lse_adam(best_x, steps=400, k0=150.0, k1=500.0, lr0=0.008, lr1=0.001)
                    nv = _max_cos(best_x)
                    if nv < best_v:
                        best_v = nv
                    remain = DEADLINE - (time.time() - t0) - 1.0
            except Exception:
                pass
            if remain > 4.0:
                try:
                    Xp = _lp_polish(best_x, budget=min(remain, 30.0), t_dead=t0 + DEADLINE - 0.5)
                    vp = _max_cos(Xp)
                    if vp < best_v and np.all(np.finite(Xp)):
                        best_x = Xp
                        best_v = vp
                except Exception:
                    pass
    except Exception:
        pass
    best_x = _norm(best_x)
    return best_x
