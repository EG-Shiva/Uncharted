import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(np.max(g))

def sym_max(A):
    G = A @ A.T
    np.fill_diagonal(G, -2.0)
    m1 = float(np.max(G))
    G2 = - (A @ A.T)
    np.fill_diagonal(G2, -2.0)
    m2 = float(np.max(G2))
    return max(m1, m2)

def lp_polish(x, seconds, deadline=None):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = max_cos(x)
    radius = 1e-3
    t0 = time.time()
    if deadline is None:
        deadline = t0 + seconds
    iu = np.triu_indices(n, 1)
    while time.time() - t0 < seconds and time.time() < deadline and radius > 1e-14:
        G = x @ x.T
        c = G[iu]
        m = float(c.max())
        sel = c > m - max(20 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        csub = c[sel]
        p = len(I)
        if p == 0:
            break
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        except Exception:
            break
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-csub, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        v = max_cos(y)
        if v < best:
            x, best, radius = y, v, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def solve():
    t0 = time.time()
    hard_deadline = t0 + 178.0
    rng = np.random.RandomState(0)
    d = 9
    nh = 90
    B = 6
    steps = 2000
    # --- build batched starts (B,90,9): random + D9-root-seeded ---
    A = rng.randn(B, nh, d)
    # D9 base: e_i+e_j, e_i-e_j normalized
    base = []
    for i in range(d):
        for j in range(i + 1, d):
            v = np.zeros(d); v[i] = 1; v[j] = 1; v /= np.linalg.norm(v)
            base.append(v)
            w = np.zeros(d); w[i] = 1; w[j] = -1; w /= np.linalg.norm(w)
            base.append(w)
    base = np.array(base)  # (72,9)
    for b in range(3, B):
        try:
            Q, _ = np.linalg.qr(rng.randn(d, d))
        except Exception:
            Q = np.eye(d)
        rot = base @ Q.T
        Ab = np.empty((nh, d))
        Ab[:72] = rot
        Ab[72:] = rng.randn(nh - 72, d)
        # small perturbation on rotated roots
        Ab[:72] += 0.08 * rng.randn(72, d)
        Ab /= np.linalg.norm(Ab, axis=1, keepdims=True)
        A[b] = Ab
    for b in range(0, 3):
        A[b] /= np.linalg.norm(A[b], axis=2, keepdims=True)
    # normalize all
    An = np.linalg.norm(A, axis=2, keepdims=True)
    A = A / np.maximum(An, 1e-12)
    m1 = np.zeros_like(A)
    v1 = np.zeros_like(A)
    b1, b2, eps = 0.9, 0.999, 1e-8
    b1p = 1.0; b2p = 1.0
    # --- batched annealed LogSumExp Adam, beta 30 -> 500 ---
    for s in range(steps):
        if time.time() > t0 + 75.0:
            break
        frac = s / max(1, steps - 1)
        beta = 30.0 * (500.0 / 30.0) ** frac
        lr = 0.02 * (1.0 - 0.7 * frac)
        # Gram (B,90,90)
        G = np.matmul(A, np.transpose(A, (0, 2, 1)))
        # max |offdiag| per batch
        Ag = np.abs(G)
        # mask diag by setting to -1
        # use loop-free max
        for bb in range(B):
            np.fill_diagonal(Ag[bb], -1.0)
        gmax = np.max(Ag, axis=(1, 2))  # (B,)
        # stabilize
        Gm = G - gmax[:, None, None]
        Gmn = -G - gmax[:, None, None]
        # clip to avoid overflow (values <=0, but beta*val can be -large -> exp 0 fine)
        Eplus = np.exp(beta * Gm)
        Eminus = np.exp(beta * Gmn)
        for bb in range(B):
            np.fill_diagonal(Eplus[bb], 0.0)
            np.fill_diagonal(Eminus[bb], 0.0)
        den = np.sum(Eplus + Eminus, axis=(1, 2)) * 0.5 + 1e-30  # (B,)
        Dmat = Eplus - Eminus
        Grad = np.matmul(Dmat, A) / den[:, None, None]
        dot = np.sum(Grad * A, axis=2, keepdims=True)
        Gt = Grad - dot * A
        b1p *= b1; b2p *= b2
        m1 = b1 * m1 + (1 - b1) * Gt
        v1 = b2 * v1 + (1 - b2) * (Gt * Gt)
        mhat = m1 / (1 - b1p)
        vhat = v1 / (1 - b2p)
        step = mhat / (np.sqrt(vhat) + eps)
        # limit huge steps
        # per-point clip
        sn = np.linalg.norm(step, axis=2, keepdims=True) + 1e-12
        # allow up to norm 1.0
        step = step * np.minimum(1.0, 1.0 / sn)
        A = A - lr * step
        A = A / np.maximum(np.linalg.norm(A, axis=2, keepdims=True), 1e-12)
    # pick best symmetric batch
    best_b = 0
    best_sym = 1e9
    for bb in range(B):
        Ab = A[bb] / np.linalg.norm(A[bb], axis=1, keepdims=True)
        val = sym_max(Ab)
        if val < best_sym:
            best_sym = val
            best_b = bb
    Abest = A[best_b] / np.linalg.norm(A[best_b], axis=1, keepdims=True)
    # expand to 180 antipodal
    Xsym = np.vstack([Abest, -Abest])
    # break symmetry: small tangential perturbation of one member of each pair
    rng2 = np.random.RandomState(1)
    sig1 = 0.005
    sig2 = 0.015
    n1 = rng2.randn(90, d) * sig1
    n2 = rng2.randn(90, d) * sig2
    # tangential projection
    n1 -= np.sum(n1 * Abest, axis=1, keepdims=True) * Abest
    Aneg = -Abest
    n2 -= np.sum(n2 * Aneg, axis=1, keepdims=True) * Aneg
    A1 = Abest + n1
    A2 = Aneg + n2
    A1 /= np.linalg.norm(A1, axis=1, keepdims=True)
    A2 /= np.linalg.norm(A2, axis=1, keepdims=True)
    X = np.vstack([A1, A2])
    # LP-polish full code (keep self-contained search under ~120s)
    mid_deadline = t0 + 120.0
    # leave time for best comparison
    Xp = lp_polish(X, seconds=min(30.0, max(5.0, mid_deadline - time.time())), deadline=min(mid_deadline, hard_deadline - 25.0))
    vp = max_cos(Xp)
    # load best.npy and keep better after final polish
    try:
        xb = np.load("best.npy")
        xb = np.asarray(xb, dtype=float)
        if xb.shape != (180, 9):
            xb = None
        else:
            xb = xb / np.linalg.norm(xb, axis=1, keepdims=True)
    except Exception:
        xb = None
    if xb is None:
        return Xp
    vb = max_cos(xb)
    # if our symmetric-broken already beats it, light-polish both and return best
    # polish loaded best briefly with remaining time
    rem = hard_deadline - time.time() - 2.0
    if rem > 4.0:
        # polish whichever is currently better with a bit more time, and the other briefly
        if vp < vb:
            # polish ours a bit more, polish best briefly for comparison
            extra = min(12.0, rem - 4.0)
            Xp2 = lp_polish(Xp, seconds=extra, deadline=hard_deadline - 4.0)
            vp2 = max_cos(Xp2)
            if vp2 < vp:
                Xp, vp = Xp2, vp2
            rem2 = hard_deadline - time.time() - 1.0
            if rem2 > 3.0:
                xb2 = lp_polish(xb, seconds=min(6.0, rem2), deadline=hard_deadline - 1.0)
                vb2 = max_cos(xb2)
                if vb2 < vb:
                    xb, vb = xb2, vb2
        else:
            xb2 = lp_polish(xb, seconds=min(12.0, rem - 4.0), deadline=hard_deadline - 4.0)
            vb2 = max_cos(xb2)
            if vb2 < vb:
                xb, vb = xb2, vb2
            rem2 = hard_deadline - time.time() - 1.0
            if rem2 > 3.0 and vp < 0.49:
                Xp2 = lp_polish(Xp, seconds=min(6.0, rem2), deadline=hard_deadline - 1.0)
                vp2 = max_cos(Xp2)
                if vp2 < vp:
                    Xp, vp = Xp2, vp2
    if vp <= vb:
        return Xp
    else:
        return xb
