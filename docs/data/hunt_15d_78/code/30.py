import time
import numpy as np
from scipy.optimize import minimize, linprog
from scipy.sparse import coo_matrix
from scipy.linalg import expm, svd


def max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())


def lp_polish(X, seconds):
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    n, d = X.shape
    best = max_cos(X)
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        Gfull = X @ X.T
        c = Gfull[iu]
        m = c.max()
        sel = c > m - max(20 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        if p == 0:
            break
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([X[J], X[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((X.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))),
                          shape=(n, nv)).tocsr()
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
        y = X + res.x[:-1].reshape(n, d)
        y = y / np.linalg.norm(y, axis=1, keepdims=True)
        # exact score
        G2 = y @ y.T
        np.fill_diagonal(G2, -2.0)
        v = float(G2.max())
        if v < best:
            X, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return X


def build_generators(XA, XB, rng, k_top=4, k_rand=2):
    # cross-covariance after centering
    ma = XA.mean(axis=0, keepdims=True)
    mb = XB.mean(axis=0, keepdims=True)
    Ac = XA - ma
    Bc = XB - mb
    M = Ac.T @ Bc  # 15x15
    try:
        U, S, Vt = svd(M, full_matrices=True)
    except Exception:
        U, _, _ = np.linalg.svd(M, full_matrices=True)
        Vt = U.T
    V = Vt.T
    pairs = []
    for k in range(min(k_top, 15)):
        p = U[:, k].copy()
        pn = np.linalg.norm(p)
        if pn < 1e-12:
            continue
        p /= pn
        q0 = V[:, k].copy()
        # orthogonalize q0 against p
        q = q0 - float(p @ q0) * p
        nq = np.linalg.norm(q)
        if nq < 1e-6:
            # pick vector orthogonal to p at random
            r = rng.standard_normal(15)
            r = r - float(p @ r) * p
            nq = np.linalg.norm(r)
            if nq < 1e-9:
                continue
            q = r / nq
        else:
            q = q / nq
        pairs.append((p, q))
        if len(pairs) >= k_top:
            break
    # random cross pairs: random planes in U basis (orthonormal by construction)
    tries = 0
    while len(pairs) < k_top + k_rand and tries < 50:
        tries += 1
        i, j = rng.integers(0, 15, size=2)
        if i == j:
            continue
        p = U[:, int(i)].copy()
        q = U[:, int(j)].copy()
        # already orthonormal (U orthogonal)
        pairs.append((p, q))
    P = np.stack([p for p, q in pairs], axis=0)  # Kx15
    Q = np.stack([q for p, q in pairs], axis=0)
    return P, Q


def theta_to_R(theta, P, Q):
    d = P.shape[1]
    Th = np.zeros((d, d), dtype=np.float64)
    for k, t in enumerate(theta):
        if t == 0.0:
            continue
        p = P[k]
        q = Q[k]
        Th += float(t) * (np.outer(p, q) - np.outer(q, p))
    # exp of skew is rotation
    try:
        R = expm(Th)
    except Exception:
        # first-order Cayley fallback
        R = np.eye(d) + Th
        # project to orthogonal via QR
        R, _ = np.linalg.qr(R)
    return R


def solve():
    t0 = time.time()
    LIMIT = 168.0
    rng = np.random.default_rng(0)
    # load start
    bestX = None
    bestV = 1e9
    for fn in ["best.npy", "record.npy"]:
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=np.float64)
            if a.shape != (78, 15):
                continue
            a = a / np.linalg.norm(a, axis=1, keepdims=True)
            v = max_cos(a)
            if v < bestV:
                bestV = v
                bestX = a.copy()
        except Exception:
            continue
    if bestX is None:
        a = rng.standard_normal((78, 15))
        a /= np.linalg.norm(a, axis=1, keepdims=True)
        bestX = a
        bestV = max_cos(a)
    n, d = 78, 15
    # fixed index split: A = 0..56, B = 57..77
    idxA = np.arange(57)
    idxB = np.arange(57, 78)

    curX = bestX.copy()
    curV = bestV

    # quick initial snap tiny (already polished)
    if time.time() - t0 < LIMIT - 150:
        curX = lp_polish(curX, min(2.0, max(0.5, LIMIT - (time.time() - t0) - 150)))
        curV = max_cos(curX)
        if curV < bestV:
            bestV = curV
            bestX = curX.copy()

    nrounds = 0
    # alternate B-rotation / A-rotation
    while time.time() - t0 < LIMIT - 22:
        nrounds += 1
        for move_B in (True, False):
            if time.time() - t0 > LIMIT - 22:
                break
            if move_B:
                XA = curX[idxA]
                XB = curX[idxB]
            else:
                XA = curX[idxB]  # fixed = B
                XB = curX[idxA]  # moving = A
                # for uniform code below, rename: fixed/move
            Xfix = XA
            Xmov = XB
            # fixed intra max + moving intra max (moving intra invariant)
            Gf = Xfix @ Xfix.T
            np.fill_diagonal(Gf, -2.0)
            fixed_max = float(Gf.max())
            Gm = Xmov @ Xmov.T
            np.fill_diagonal(Gm, -2.0)
            mov_intra = float(Gm.max())
            fix_all = max(fixed_max, mov_intra)
            cross0 = float((Xfix @ Xmov.T).max())

            P, Qv = build_generators(Xfix, Xmov, rng, k_top=4, k_rand=2)
            K = P.shape[0]
            if K == 0:
                continue

            # objective: max cross cosine after rotating moving block
            def obj(th):
                R = theta_to_R(np.asarray(th, dtype=np.float64), P, Qv)
                Xr = Xmov @ R
                # renormalize (R orthogonal, but numeric)
                # norms preserved; skip renorm for speed
                c = float((Xfix @ Xr.T).max())
                # small ridge to prefer small angles on ties
                c += 1e-6 * float(np.dot(th, th))
                return c

            x0 = np.zeros(K, dtype=np.float64)
            best_th = x0
            best_c = cross0
            try:
                res = minimize(obj, x0, method='Nelder-Mead',
                               options={'maxiter': 120, 'maxfev': 220,
                                        'xatol': 0.01, 'fatol': 1e-7,
                                        'disp': False})
                if res is not None and res.x is not None:
                    c1 = float(obj(res.x))
                    # remove ridge
                    c1_true = c1 - 1e-6 * float(np.dot(res.x, res.x))
                    if c1_true < best_c - 1e-12:
                        best_c = c1_true
                        best_th = np.asarray(res.x, dtype=np.float64)
                    # clip to [-0.35,0.35]
                    best_th = np.clip(best_th, -0.35, 0.35)
            except Exception:
                pass

            # if no progress, try a random rotation that keeps cross reasonable
            if float(np.abs(best_th).max()) < 1e-9:
                # random small angles
                for _ in range(8):
                    if time.time() - t0 > LIMIT - 22:
                        break
                    th_r = rng.uniform(-0.30, 0.30, size=K)
                    Rr = theta_to_R(th_r, P, Qv)
                    Xr = Xmov @ Rr
                    c_r = float((Xfix @ Xr.T).max())
                    # allow slight increase over cross0 to diversify, but not much over global
                    if c_r < max(cross0, fix_all) + 0.002:
                        best_th = th_r
                        best_c = c_r
                        break

            Rbest = theta_to_R(best_th, P, Qv)
            XmovR = Xmov @ Rbest
            XmovR = XmovR / np.linalg.norm(XmovR, axis=1, keepdims=True)

            Xrot = curX.copy()
            if move_B:
                Xrot[idxB] = XmovR
            else:
                Xrot[idxA] = XmovR
            v_rot = max_cos(Xrot)
            # short LP polish per accepted rotation (rotation never worsens overall
            # unless random fallback increased cross slightly; polish only if promising)
            # polish if rotation improved or stayed close
            if v_rot > curV + 0.003:
                # reject bad random rotation
                continue
            budget = 1.4
            left = (LIMIT - 22) - (time.time() - t0)
            if left < 2.0:
                break
            budget = min(budget, left - 0.5)
            if budget < 0.4:
                # no time for polish: just accept rotation
                if v_rot < curV:
                    curX, curV = Xrot, v_rot
                    if curV < bestV:
                        bestV, bestX = curV, curX.copy()
                continue
            try:
                Xpol = lp_polish(Xrot, budget)
            except Exception:
                Xpol = Xrot
            v_pol = max_cos(Xpol)
            # lp_polish is monotone vs Xrot; accept as new cur (monotone vs cur unless random)
            if v_pol <= curV + 1e-12:
                curX, curV = Xpol, v_pol
                if v_pol < bestV - 1e-14:
                    bestV = v_pol
                    bestX = Xpol.copy()
            else:
                # polish failed to beat cur but rotation itself may beat cur
                if v_rot < curV:
                    curX, curV = Xrot, v_rot
                    if curV < bestV:
                        bestV, bestX = curX.copy(), curV
            # early exit if time low
            if time.time() - t0 > LIMIT - 22:
                break
        # end for move
    # final intensive polish of champion
    try:
        remain = LIMIT - (time.time() - t0) - 1.0
        if remain > 2.0:
            Xf = lp_polish(bestX, min(remain, 14.0))
            vf = max_cos(Xf)
            if vf < bestV:
                bestX, bestV = Xf, vf
    except Exception:
        pass
    bestX = bestX / np.linalg.norm(bestX, axis=1, keepdims=True)
    bestX = np.where(np.isnan(bestX), 0.0, bestX)
    nr = np.linalg.norm(bestX, axis=1, keepdims=True)
    nr = np.maximum(nr, 1e-12)
    bestX = bestX / nr
    return bestX
