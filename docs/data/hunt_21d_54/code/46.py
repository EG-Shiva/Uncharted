import time
import math
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))


def proj(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return X / n


def lp_polish(X, seconds, deadline=1e30):
    X = proj(X)
    n, d = X.shape
    try:
        best = max_cos(X)
    except Exception:
        return X
    radius = 1e-3
    start = time.time()
    iu0 = np.triu_indices(n, 1)
    P = len(iu0[0])
    while time.time() - start < seconds and time.time() < deadline:
        if radius < 1e-14:
            break
        G = X @ X.T
        c = G[iu0]
        m = float(np.max(c))
        thresh = m - max(20.0 * radius, 1e-9)
        sel = c > thresh
        cnt = int(np.sum(sel))
        if cnt == 0:
            break
        if cnt > 1200:
            K = 1200
            part = np.argpartition(-c, K)[:K]
            thr2 = float(np.min(c[part]))
            sel = c >= thr2
            if int(np.sum(sel)) > 1500:
                sel = np.zeros(P, dtype=bool)
                sel[part] = True
        I = iu0[0][sel]
        J = iu0[1][sel]
        p = len(I)
        csv = c[sel]
        nv = n * d + 1
        try:
            crow = np.empty(p * (2 * d + 1), dtype=int)
            ccol = np.empty(p * (2 * d + 1), dtype=int)
            cval = np.empty(p * (2 * d + 1), dtype=float)
            for k in range(p):
                s = k * (2 * d + 1)
                crow[s:s + 2 * d + 1] = k
                ccol[s:s + d] = I[k] * d + np.arange(d)
                cval[s:s + d] = X[J[k]]
                ccol[s + d:s + 2 * d] = J[k] * d + np.arange(d)
                cval[s + d:s + 2 * d] = X[I[k]]
                ccol[s + 2 * d] = n * d
                cval[s + 2 * d] = -1.0
            A_ub = coo_matrix((cval, (crow, ccol)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix(
                (np.tile(X.ravel(), 1),
                 (np.repeat(np.arange(n), d), np.arange(n * d))),
                shape=(n, nv)).tocsr()
            obj = np.zeros(nv)
            obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-csv, A_eq=A_eq,
                          b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                          method="highs",
                          options={"time_limit": 2.0})
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        dx = res.x[:-1].reshape(n, d)
        y = proj(X + dx)
        v = max_cos(y)
        if v < best:
            best = v
            X = y
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return X


def build_tables(mults, m):
    m_ = int(m)
    T = np.arange(m_)
    C = np.cos(2.0 * np.pi * np.outer(T, np.asarray(mults, dtype=float)) / m_)
    S = np.sin(2.0 * np.pi * np.outer(T, np.asarray(mults, dtype=float)) / m_)
    return C, S


def expand_code(S_full, Ctab, Stab):
    # S_full (ns,21), Ctab/Stab (m,10) -> X (ns*m,21)
    ns = S_full.shape[0]
    m = Ctab.shape[0]
    Sp = S_full[:, :20].reshape(ns, 10, 2)
    z = S_full[:, 20]
    Xp = np.empty((ns, m, 10, 2))
    for k in range(10):
        x = Sp[:, k, 0][:, None]  # (ns,1)
        y = Sp[:, k, 1][:, None]
        c = Ctab[:, k][None, :]  # (1,m)
        s = Stab[:, k][None, :]
        Xp[:, :, k, 0] = x * c - y * s
        Xp[:, :, k, 1] = x * s + y * c
    X = np.empty((ns * m, 21))
    X[:, :20] = Xp.transpose(0, 1, 2, 3).reshape(ns * m, 20)
    X[:, 20] = np.repeat(z, m)
    return X


def optimize_symmetric(m, mode, mults, rng, steps=1300, lr=0.04):
    Ctab, Stab = build_tables(mults, m)
    if mode == 'free18':
        nf = 3
    elif mode == 'free27':
        nf = 2
    elif mode == 'anti27':
        nf = 1
    elif mode == 'anti18':
        nf = 2
    else:
        nf = 3
    U = rng.standard_normal((nf, 21))
    if mode == 'anti18':
        U[1, 20] = 0.0
    b1 = 0.9
    b2 = 0.999
    eps = 1e-8
    mt = np.zeros_like(U)
    vt = np.zeros_like(U)
    best_val = 1e30
    best_X = None
    K0, K1 = 80.0, 1000.0
    for it in range(steps):
        frac = it / max(1, steps - 1)
        K = K0 * (K1 / K0) ** frac
        norms = np.linalg.norm(U, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        F = U / norms  # (nf,21)
        if mode == 'anti18':
            F[:, 20:21] = F[:, 20:21].copy()
            # enforce second free seed axial 0 exactly
            # renormalize that row over first 20 dims
            # do correction: set axial 0 then renormalize
            F[1, 20] = 0.0
            n1 = float(np.linalg.norm(F[1, :20]))
            if n1 > 0:
                F[1, :20] /= n1
        # map to full seeds
        if mode == 'free18' or mode == 'free27':
            S_full = F
        elif mode == 'anti27':
            S_full = np.vstack([F[0], -F[0]])
        elif mode == 'anti18':
            S_full = np.vstack([F[0], -F[0], F[1]])
        X = expand_code(S_full, Ctab, Stab)
        # LSE gradient
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        gm = float(np.max(G))
        E = np.exp(np.clip(K * (G - gm), -30, 0))
        np.fill_diagonal(E, 0.0)
        Spairs = float(np.sum(E) * 0.5)
        if Spairs <= 0:
            Spairs = 1.0
        W = E / Spairs
        gradX = W @ X  # (N,21)
        N = X.shape[0]
        ns = S_full.shape[0]
        mm = int(m)
        gradXr = gradX.reshape(ns, mm, 21)
        gradS = np.zeros_like(S_full)
        # planes backprop: sum_t Rot(-t) grad
        Gp = gradXr[:, :, :20].reshape(ns, mm, 10, 2)
        gradSp = np.zeros((ns, 10, 2))
        for k in range(10):
            c = Ctab[:, k]  # (m,)
            s = Stab[:, k]
            gx = Gp[:, :, k, 0]  # (ns,m)
            gy = Gp[:, :, k, 1]
            # Rot^T: [gx*c+gy*s, -gx*s+gy*c]
            gradSp[:, k, 0] = np.sum(gx * c[None, :] + gy * s[None, :], axis=1)
            gradSp[:, k, 1] = np.sum(-gx * s[None, :] + gy * c[None, :], axis=1)
        gradS[:, :20] = gradSp.reshape(ns, 20)
        gradS[:, 20] = np.sum(gradXr[:, :, 20], axis=1)
        # to free
        if mode == 'free18' or mode == 'free27':
            gradF = gradS
        elif mode == 'anti27':
            gradF = (gradS[0] - gradS[1])[None, :]
        elif mode == 'anti18':
            g0 = gradS[0] - gradS[1]
            g1 = gradS[2].copy()
            g1[20] = 0.0
            gradF = np.vstack([g0, g1])
        # to U
        dots = np.sum(gradF * F, axis=1, keepdims=True)
        gradU = (gradF - dots * F) / norms
        if mode == 'anti18':
            gradU[1, 20] = 0.0
        # Adam
        mt = b1 * mt + (1 - b1) * gradU
        vt = b2 * vt + (1 - b2) * (gradU * gradU)
        mhat = mt / (1 - b1 ** (it + 1))
        vhat = vt / (1 - b2 ** (it + 1))
        U = U - lr * mhat / (np.sqrt(vhat) + eps)
        if mode == 'anti18':
            U[1, 20] = 0.0
        if (it % 150 == 0) or (it == steps - 1):
            v = max_cos(X)
            if v < best_val:
                best_val = v
                best_X = X.copy()
    return best_X, best_val


def sample_mults(m, rng, odd_only=False):
    for _ in range(50):
        if odd_only:
            odds = list(range(1, m, 2))
            a = [int(odds[int(rng.integers(0, len(odds)))]) for _ in range(10)]
        else:
            a = [int(rng.integers(1, m)) for _ in range(10)]
        g = 0
        for v in a:
            g = math.gcd(g, v)
        g = math.gcd(g, m)
        if g == 1:
            return np.array(a, dtype=int)
    return np.array(a, dtype=int)


def solve():
    t0 = time.time()
    deadline = t0 + 172.0
    internal_cap = t0 + 150.0
    rng = np.random.default_rng(20261101)
    try:
        x0 = np.load("best.npy").astype(float)
    except Exception:
        try:
            x0 = np.load("record.npy").astype(float)
        except Exception:
            x0 = rng.standard_normal((54, 21))
    if x0.shape != (54, 21):
        tmp = rng.standard_normal((54, 21))
        try:
            mm0 = min(x0.shape[0], 54)
            mm1 = min(x0.shape[1], 21)
            tmp[:mm0, :mm1] = x0[:mm0, :mm1]
        except Exception:
            pass
        x0 = tmp
    x0 = np.nan_to_num(x0, nan=0.0, posinf=0.0, neginf=0.0)
    x0 = proj(x0)
    try:
        bv0 = max_cos(x0)
    except Exception:
        bv0 = 1.0
    best = x0.copy()
    best_val = float(bv0)
    # threshold from lab best; keep only improvements
    thresh = 0.0666634002447
    # trial sequence
    configs = [('free18', 18, False), ('free27', 27, False),
               ('anti18', 18, True), ('anti27', 27, True)]
    trial = 0
    while time.time() < internal_cap - 4.0:
        trial += 1
        cfg = configs[(trial - 1) % len(configs)]
        mode, m, odd = cfg
        # vary steps a bit, more inits cheap
        steps = 1100 if (trial % 3 != 0) else 1700
        mults = sample_mults(m, rng, odd_only=odd)
        # occasionally structured mults 1..10
        if trial % 9 == 0:
            mults = np.array([(i % (m - 1)) + 1 for i in range(10)], dtype=int)
            if odd:
                mults = np.array([x if x % 2 == 1 else x + 1 for x in mults], dtype=int)
                mults[mults >= m] = 1
        try:
            Xs, vs = optimize_symmetric(m, mode, mults, rng, steps=steps, lr=0.04)
        except Exception:
            continue
        if Xs is None:
            continue
        # break symmetry with 1e-3 tangent noise
        try:
            Xn = Xs.copy()
            nz = rng.standard_normal(Xn.shape)
            dot = np.sum(nz * Xn, axis=1, keepdims=True)
            nz = nz - dot * Xn
            nn = np.linalg.norm(nz, axis=1, keepdims=True) + 1e-12
            Xn = proj(Xn + 1e-3 * nz / nn * np.sqrt(21.0) * 0.25)
            # short free polish
            budget = 1.1
            if time.time() + budget > internal_cap - 2.0:
                budget = max(0.3, internal_cap - 2.0 - time.time())
            if budget < 0.25:
                break
            Y = lp_polish(Xn, seconds=budget, deadline=internal_cap - 1.0)
            v = max_cos(Y)
            if v < best_val - 1e-12 and v < thresh + 1e-12:
                # accept only if beats lab best (allow tiny tol for first)
                best_val = v
                best = Y.copy()
                thresh = v
        except Exception:
            continue
        if best_val < 0.055:
            break
        if time.time() > internal_cap - 4.0:
            break
    # final polish on best with remaining time
    try:
        fb = max(0.5, deadline - time.time() - 1.0)
        fb = min(fb, 22.0)
        if fb > 0.5:
            best = lp_polish(best, seconds=fb, deadline=deadline)
            best_val = max_cos(best)
    except Exception:
        pass
    best = np.nan_to_num(best, nan=0.0, posinf=0.0, neginf=0.0)
    best = proj(best)
    return best
