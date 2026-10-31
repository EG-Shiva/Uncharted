import time
import numpy as np

def _norm_rows(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return x / n

def _max_cos(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def _hurwitz_units():
    H = []
    for i in range(4):
        for s in (-1.0, 1.0):
            v = [0.0, 0.0, 0.0, 0.0]
            v[i] = s
            H.append(v)
    for s0 in (-0.5, 0.5):
        for s1 in (-0.5, 0.5):
            for s2 in (-0.5, 0.5):
                for s3 in (-0.5, 0.5):
                    H.append([s0, s1, s2, s3])
    return np.array(H, dtype=np.float64)

def _left_mats(H):
    Ms = np.zeros((H.shape[0], 4, 4), dtype=np.float64)
    for h_idx, h in enumerate(H):
        w, x, y, z = h
        Ms[h_idx] = np.array([
            [w, -x, -y, -z],
            [x,  w, -z,  y],
            [y,  z,  w, -x],
            [z, -y,  x,  w],
        ], dtype=np.float64)
    return Ms

def _cell24():
    pts = []
    import itertools
    for a, b in itertools.combinations(range(4), 2):
        for sa in (-1.0, 1.0):
            for sb in (-1.0, 1.0):
                v = [0.0, 0.0, 0.0, 0.0]
                v[a] = sa / np.sqrt(2.0)
                v[b] = sb / np.sqrt(2.0)
                pts.append(v)
    return np.array(pts, dtype=np.float64)

def _rand_rot(rng):
    A = rng.standard_normal((4, 4))
    Q, R = np.linalg.qr(A)
    d = np.diag(R)
    s = np.sign(d)
    s[s == 0] = 1.0
    Q = Q * s[None, :]
    if np.linalg.det(Q) < 0:
        Q[:, 0] = -Q[:, 0]
    return Q

def _orbit_points(S, R, Ms):
    k = S.shape[0]
    H = Ms.shape[0]
    N = k * H
    Y = np.zeros((N, 4), dtype=np.float64)
    for ki in range(k):
        s = S[ki]
        for hi in range(H):
            Y[ki * H + hi] = Ms[hi] @ s
    X = Y @ R.T
    return _norm_rows(X)

def _seed_optimize(S0, R0, Ms, steps, tau0, tau1, lr0, lr1, rng=None):
    S = _norm_rows(S0.copy())
    R = R0.copy()
    k = S.shape[0]
    H = Ms.shape[0]
    N = k * H
    lt0, lt1 = np.log(tau0), np.log(tau1)
    ll0, ll1 = np.log(lr0), np.log(lr1)
    for t in range(steps):
        f = t / max(1, steps - 1)
        tau = float(np.exp(lt0 + (lt1 - lt0) * f))
        lr = float(np.exp(ll0 + (ll1 - ll0) * f))
        # build full config
        Y = np.zeros((N, 4))
        for ki in range(k):
            s = S[ki]
            for hi in range(H):
                Y[ki * H + hi] = Ms[hi] @ s
        X = Y @ R.T
        # per-row softmax grad on X
        G = X @ X.T
        np.fill_diagonal(G, -np.inf)
        m = float(np.max(G))
        W = np.exp((G - m) / tau)
        np.fill_diagonal(W, 0.0)
        row = np.sum(W, axis=1, keepdims=True)
        row[row < 1e-12] = 1e-12
        gradX = (W @ X) / row
        dot = np.sum(gradX * X, axis=1, keepdims=True)
        gradX = gradX - dot * X
        # backprop to Y and S,R
        gradY = gradX @ R
        gradS = np.zeros_like(S)
        for ki in range(k):
            acc = np.zeros(4)
            for hi in range(H):
                acc += Ms[hi].T @ gradY[ki * H + hi]
            # tangent projection
            acc = acc - float(np.dot(acc, S[ki])) * S[ki]
            gradS[ki] = acc
        # normalize gradS scale
        sc = float(np.max(np.linalg.norm(gradS, axis=1)))
        if sc > 1e-12:
            gradS = gradS / sc
        S = S - lr * gradS
        S = _norm_rows(S)
        # rotation grad
        Gr = gradX.T @ Y  # 4x4 dL/dR
        # tangent projection on SO(4)
        A = R.T @ Gr
        skew = 0.5 * (A - A.T)
        sn = float(np.linalg.norm(skew))
        if sn > 1e-12:
            skew = skew / sn
            # step size scaled by lr
            Rn = R - lr * (R @ skew) * min(1.0, sn)
        else:
            Rn = R
        try:
            U, _, Vt = np.linalg.svd(Rn)
            Rn = U @ Vt
            if np.linalg.det(Rn) < 0:
                U[:, -1] = -U[:, -1]
                Rn = U @ Vt
        except Exception:
            Rn = R
        R = Rn
    return S, R

def _smooth_full(X, steps, tau0, tau1, lr0, lr1, mom=0.88):
    n, d = X.shape
    vel = np.zeros_like(X)
    lt0, lt1 = np.log(tau0), np.log(tau1)
    ll0, ll1 = np.log(lr0), np.log(lr1)
    for t in range(steps):
        f = t / max(1, steps - 1)
        tau = float(np.exp(lt0 + (lt1 - lt0) * f))
        lr = float(np.exp(ll0 + (ll1 - ll0) * f))
        G = X @ X.T
        np.fill_diagonal(G, -np.inf)
        m = float(np.max(G))
        W = np.exp((G - m) / tau)
        np.fill_diagonal(W, 0.0)
        row = np.sum(W, axis=1, keepdims=True)
        row[row < 1e-12] = 1e-12
        grad = (W @ X) / row
        dot = np.sum(grad * X, axis=1, keepdims=True)
        grad = grad - dot * X
        gn = np.linalg.norm(grad, axis=1, keepdims=True)
        big = gn > 2.0
        if np.any(big):
            grad = grad * np.where(big, 2.0 / np.maximum(gn, 1e-12), 1.0)
        vel = mom * vel + (1.0 - mom) * grad
        bc = 1.0 - mom ** (t + 1)
        step = vel / bc
        X = X - lr * step
        X = _norm_rows(X)
    return X

def _greedy_fill(X, need, rng, cand_per=4000):
    X = X.copy()
    for _ in range(need):
        cand = rng.standard_normal((cand_per, 4))
        cand = _norm_rows(cand)
        G = cand @ X.T  # cand_per x n
        worst = np.max(G, axis=1)  # max cos to set
        # also consider candidate-candidate? greedy maximin to set is enough
        bi = int(np.argmin(worst))
        X = np.vstack([X, cand[bi:bi+1]])
    return X

def _greedy_trim(X, target):
    X = X.copy()
    while X.shape[0] > target:
        n = X.shape[0]
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        m = float(np.max(G))
        # find worst pair
        ij = np.unravel_index(int(np.argmax(G)), G.shape)
        i, j = int(ij[0]), int(ij[1])
        # row maxima
        rm = np.max(G, axis=1)
        drop = i if rm[i] >= rm[j] else j
        # tie-breaker: higher second max? just drop
        mask = np.ones(n, dtype=bool)
        mask[drop] = False
        X = X[mask]
    return X

def _lp_polish(x, deadline, r0=0.002, cap=2500, maxit=80):
    from scipy.optimize import linprog
    from scipy.sparse import coo_matrix
    x = _norm_rows(x.copy())
    n, d = x.shape
    try:
        best = _max_cos(x)
    except Exception:
        return x, 1e9
    iu0 = np.triu_indices(n, 1)
    radius = float(r0)
    it = 0
    while time.time() < deadline and radius > 1e-14 and it < maxit:
        it += 1
        G = x @ x.T
        c = G[iu0]
        m = float(np.max(c))
        thr = m - max(20.0 * radius, 1e-9)
        idx = np.where(c > thr)[0]
        if idx.size == 0:
            radius *= 0.5
            continue
        if idx.size > cap:
            part = np.argpartition(c[idx], -cap)[-cap:]
            idx = idx[part]
        I = iu0[0][idx]; J = iu0[1][idx]
        csel = c[idx]
        p = idx.size
        nv = n * d + 1
        try:
            rows = np.repeat(np.arange(p), 2 * d + 1)
            cI = I[:, None] * d + np.arange(d)
            cJ = J[:, None] * d + np.arange(d)
            cols = np.concatenate([cI, cJ, np.full((p, 1), n * d)], axis=1).ravel()
            vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], axis=1).ravel()
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv); obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        try:
            dx = res.x[:-1].reshape(n, d)
        except Exception:
            radius *= 0.5
            continue
        y = _norm_rows(x + dx)
        try:
            v = _max_cos(y)
        except Exception:
            radius *= 0.5
            continue
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x, best

def solve():
    t0 = time.time()
    HARD_END = t0 + 172.0
    n, d = 153, 4
    rng = np.random.default_rng(20261031)
    H = _hurwitz_units()
    Ms = _left_mats(H)
    C24 = _cell24()
    best_x = None
    best_v = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=np.float64)
            if a.shape == (n, d):
                a = _norm_rows(a)
                if np.all(np.isfinite(a)):
                    v = _max_cos(a)
                    if v < best_v:
                        best_v = v
                        best_x = a.copy()
        except Exception:
            pass
    if best_x is None:
        best_x = _norm_rows(rng.standard_normal((n, d)))
        best_v = _max_cos(best_x)
    global_x = best_x.copy()
    global_v = best_v
    # trial plan: cycle over 4 structural types
    trial = 0
    # reserve ~25s for final polish
    while time.time() < HARD_END - 25.0 and trial < 14:
        trial += 1
        ttype = trial % 4
        try:
            if ttype == 1:
                k = 6
                S0 = _norm_rows(rng.standard_normal((k, 4)))
                R0 = _rand_rot(rng)
                S1, R1 = _seed_optimize(S0, R0, Ms, steps=220, tau0=0.04, tau1=0.004, lr0=0.06, lr1=0.008)
                X0 = _orbit_points(S1, R1, Ms)  # 144
                X1 = _greedy_fill(X0, n - X0.shape[0], rng, cand_per=3000)
            elif ttype == 2:
                k = 7
                S0 = _norm_rows(rng.standard_normal((k, 4)))
                R0 = _rand_rot(rng)
                S1, R1 = _seed_optimize(S0, R0, Ms, steps=200, tau0=0.04, tau1=0.005, lr0=0.06, lr1=0.008)
                X0 = _orbit_points(S1, R1, Ms)  # 168
                X1 = _greedy_trim(X0, n)
            elif ttype == 3:
                k = 5
                S0 = _norm_rows(rng.standard_normal((k, 4)))
                R0 = _rand_rot(rng)
                S1, R1 = _seed_optimize(S0, R0, Ms, steps=200, tau0=0.035, tau1=0.004, lr0=0.06, lr1=0.008)
                X0 = _orbit_points(S1, R1, Ms)  # 120
                Rr = _rand_rot(rng)
                Cr = (C24 @ Rr.T)
                Xc = np.vstack([X0, Cr])  # 144
                X1 = _greedy_fill(Xc, n - Xc.shape[0], rng, cand_per=3000)
            else:
                k = 6
                S0 = _norm_rows(rng.standard_normal((k, 4)))
                R0 = _rand_rot(rng)
                S1, R1 = _seed_optimize(S0, R0, Ms, steps=180, tau0=0.05, tau1=0.006, lr0=0.05, lr1=0.01)
                X0 = _orbit_points(S1, R1, Ms)  # 144
                Rr = _rand_rot(rng)
                Cr = (C24 @ Rr.T)
                Xc = np.vstack([X0, Cr])  # 168
                X1 = _greedy_trim(Xc, n)
            # optional coordinate permutation for diversity
            if (trial % 3) == 0:
                perm = rng.permutation(4)
                X1 = X1[:, perm]
            # unfreeze: annealed softmax
            X2 = _smooth_full(X1, steps=320, tau0=0.02, tau1=0.0015, lr0=0.25, lr1=0.02, mom=0.88)
            # quick LP polish with short deadline
            dl = min(time.time() + 7.0, HARD_END - 26.0)
            if dl <= time.time() + 1.0:
                break
            Xp, vp = _lp_polish(X2, dl, r0=0.002, maxit=60)
            if vp < global_v and np.all(np.isfinite(Xp)):
                global_x = Xp.copy()
                global_v = vp
        except Exception:
            continue
    # final polish of global best with remaining time
    try:
        rem = HARD_END - time.time() - 1.5
        if rem > 3.0:
            global_x, global_v = _lp_polish(global_x, time.time() + min(rem, 28.0), r0=0.002, maxit=150)
    except Exception:
        pass
    global_x = _norm_rows(global_x)
    return global_x
