import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def lp_polish(x, budget):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    try:
        best = max_cos(x)
    except Exception:
        return x
    best_x = x.copy()
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < budget and radius > 1e-14:
        G = x @ x.T
        c = G[iu]
        m = c.max()
        sel = c > m - max(20 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        if p == 0:
            radius *= 0.5
            continue
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c[sel], A_eq=A_eq, b_eq=np.zeros(n),
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
            x, best, best_x = y, v, y.copy()
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
        if time.time() - start >= budget:
            break
    return best_x

def smooth_opt_full(X, steps, b0, b1, lr0, lr1, mu=0.9):
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    n = X.shape[0]
    V = np.zeros_like(X)
    eye2 = 2.0 * np.eye(n)
    for k in range(steps):
        f = k / (steps - 1) if steps > 1 else 0.0
        beta = b0 * ((b1 / b0) ** f) if b0 > 0 and b1 > 0 else b0 + (b1 - b0) * f
        lr = lr0 * ((lr1 / lr0) ** f) if lr0 > 0 and lr1 > 0 else lr0
        G = X @ X.T
        m = np.max(G - eye2)
        A = beta * (G - m)
        np.clip(A, -12, 0, out=A)
        np.fill_diagonal(A, -30.0)
        np.exp(A, out=A)
        Grad = A @ X
        d = np.sum(Grad * X, axis=1, keepdims=True)
        T = Grad - d * X
        s = float(np.mean(np.sqrt(np.sum(T * T, axis=1)))) + 1e-12
        T /= s
        V *= mu
        V += (1.0 - mu) * T
        X -= lr * V
        X /= np.linalg.norm(X, axis=1, keepdims=True)
    return X

def solve():
    t0 = time.time()
    HARD = 176.0
    n, d = 153, 4
    rng = np.random.default_rng(20240117)
    best = None
    best_val = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            v = np.load(fn).astype(float)
            if v.shape == (n, d):
                v /= np.linalg.norm(v, axis=1, keepdims=True)
                if np.all(np.isfinite(v)):
                    c = max_cos(v)
                    if c < best_val:
                        best_val = c
                        best = v.copy()
        except Exception:
            pass
    if best is None:
        best = rng.standard_normal((n, d))
        best /= np.linalg.norm(best, axis=1, keepdims=True)
        best_val = max_cos(best)

    # ---- symmetry blocks ----
    # each entry is list of (M,a,b,r)
    blocks_list = []
    for b in [0, 1, 2, 3, 4, 5, 6, 7, 8]:
        if b == 0:
            blocks_list.append([(17, 0, 1, 9)])
        else:
            blocks_list.append([(17, 1, b, 9)])
    blocks_list.append([(17, 1, 0, 9)])
    blocks_list.append([(51, 1, 2, 3)])
    blocks_list.append([(51, 1, 5, 3)])
    blocks_list.append([(51, 1, 8, 3)])
    blocks_list.append([(17, 1, 2, 5), (17, 1, 5, 4)])
    blocks_list.append([(51, 1, 5, 2), (17, 1, 3, 3)])

    def precompute(blocks):
        pre = []
        off = 0
        for (M, a, b, r) in blocks:
            k = np.arange(M)
            th1 = 2 * np.pi * a * k / M
            th2 = 2 * np.pi * b * k / M
            c1 = np.cos(th1); s1 = np.sin(th1)
            c2 = np.cos(th2); s2 = np.sin(th2)
            pre.append((M, r, off, c1, s1, c2, s2))
            off += r
        return pre, off

    def build_X(R, pre):
        parts0, parts1, parts2, parts3 = [], [], [], []
        for (M, r, off, c1, s1, c2, s2) in pre:
            Rb = R[off:off + r]
            rx1 = Rb[:, 0:1]; ry1 = Rb[:, 1:2]; rx2 = Rb[:, 2:3]; ry2 = Rb[:, 3:4]
            M0 = rx1 * c1[None, :] - ry1 * s1[None, :]
            M1 = rx1 * s1[None, :] + ry1 * c1[None, :]
            M2 = rx2 * c2[None, :] - ry2 * s2[None, :]
            M3 = rx2 * s2[None, :] + ry2 * c2[None, :]
            parts0.append(M0.ravel()); parts1.append(M1.ravel())
            parts2.append(M2.ravel()); parts3.append(M3.ravel())
        N = sum(len(p) for p in parts0)
        X = np.empty((N, 4))
        X[:, 0] = np.concatenate(parts0); X[:, 1] = np.concatenate(parts1)
        X[:, 2] = np.concatenate(parts2); X[:, 3] = np.concatenate(parts3)
        return X

    def sym_optimize(Rinit, pre, steps, b0, b1, lr0, lr1, mu):
        R = Rinit.copy()
        V = np.zeros_like(R)
        for k in range(steps):
            f = k / (steps - 1) if steps > 1 else 0.0
            beta = b0 * ((b1 / b0) ** f)
            lr = lr0 * ((lr1 / lr0) ** f)
            # build X
            parts = []
            # inline build to also keep intermediate? use build_X
            X = build_X(R, pre)
            G = X @ X.T
            np.fill_diagonal(G, -2.0)
            m = np.max(G)
            A = beta * (G - m)
            np.clip(A, -12, 0, out=A)
            np.fill_diagonal(A, -30.0)
            np.exp(A, out=A)
            np.fill_diagonal(A, 0.0)
            ssum = np.sum(A) + 1e-300
            GradX = (A @ X) / ssum
            # map back
            GradR = np.zeros_like(R)
            row0 = 0
            for (M, r, off, c1, s1, c2, s2) in pre:
                Gb = GradX[row0:row0 + r * M].reshape(r, M, 4)
                row0 += r * M
                G0 = Gb[:, :, 0]; G1 = Gb[:, :, 1]; G2 = Gb[:, :, 2]; G3 = Gb[:, :, 3]
                GradR[off:off + r, 0] = G0 @ c1 + G1 @ s1
                GradR[off:off + r, 1] = -G0 @ s1 + G1 @ c1
                GradR[off:off + r, 2] = G2 @ c2 + G3 @ s2
                GradR[off:off + r, 3] = -G2 @ s2 + G3 @ c2
            dot = np.sum(GradR * R, axis=1, keepdims=True)
            Gt = GradR - dot * R
            sc = float(np.mean(np.sqrt(np.sum(Gt * Gt, axis=1)))) + 1e-12
            Tn = Gt / sc
            V *= mu
            V += (1.0 - mu) * Tn
            R -= lr * V
            R /= np.sqrt(np.sum(R * R, axis=1, keepdims=True))
            if time.time() - t0 > HARD - 45.0:
                break
        return R

    # ---- screening ----
    cands = []  # (val, R, pre_idx, blocks)
    pres = [precompute(b) for b in blocks_list]
    n_init_map = []
    for bi, b in enumerate(blocks_list):
        if len(b) == 2:
            n_init_map.append(12)
        elif b[0][0] == 51:
            n_init_map.append(18)
        else:
            n_init_map.append(22)

    for bi, blocks in enumerate(blocks_list):
        pre, Rcount = pres[bi]
        n_init = n_init_map[bi]
        for ii in range(n_init):
            if time.time() - t0 > 70.0:
                break
            Rinit = rng.standard_normal((Rcount, 4))
            Rinit /= np.linalg.norm(Rinit, axis=1, keepdims=True)
            Ropt = sym_optimize(Rinit, pre, steps=650, b0=20.0, b1=180.0, lr0=0.08, lr1=0.006, mu=0.85)
            X = build_X(Ropt, pre)
            X /= np.linalg.norm(X, axis=1, keepdims=True)
            v = max_cos(X)
            cands.append((v, Ropt.copy(), bi))
            # early keep global sym best even if worse than record? store
        if time.time() - t0 > 70.0:
            break

    cands.sort(key=lambda t: t[0])
    # fully relax top few
    top = cands[:5] if len(cands) >= 5 else cands
    relaxed = []
    for (v, Ropt, bi) in top:
        if time.time() - t0 > 95.0:
            relaxed.append((v, Ropt, bi))
            continue
        pre, Rcount = pres[bi]
        R2 = sym_optimize(Ropt, pre, steps=2200, b0=30.0, b1=450.0, lr0=0.05, lr1=0.003, mu=0.88)
        X2 = build_X(R2, pre)
        X2 /= np.linalg.norm(X2, axis=1, keepdims=True)
        v2 = max_cos(X2)
        relaxed.append((v2, R2, bi))
    relaxed.sort(key=lambda t: t[0])

    # consider relaxed symmetries as starting points for breaking
    sym_Xs = []
    for (v, R2, bi) in relaxed[:4]:
        pre, _ = pres[bi]
        X = build_X(R2, pre)
        X /= np.linalg.norm(X, axis=1, keepdims=True)
        sym_Xs.append((v, X))

    # ---- break symmetry: perturb + smooth + light LP ----
    sigmas = [0.03, 0.06, 0.10, 0.14]
    # also include unperturbed smooth of best sym (sigma 0)
    tried = 0
    for (sv, SX) in sym_Xs:
        for sigma in ([0.0] + sigmas if sv == relaxed[0][0] else sigmas[:2]):
            if time.time() - t0 > HARD - 30.0 or tried > 14:
                break
            tried += 1
            if sigma == 0.0:
                Xp = SX.copy()
            else:
                Xp = SX + sigma * rng.standard_normal((n, d))
                Xp /= np.linalg.norm(Xp, axis=1, keepdims=True)
            # smooth
            steps = 1600 if sigma < 0.09 else 2000
            Xp = smooth_opt_full(Xp, steps, 30.0, 350.0, 0.02, 0.002, mu=0.9)
            v = max_cos(Xp)
            if v < best_val + 0.02:
                remain = HARD - (time.time() - t0) - 14.0
                if remain > 2.0:
                    budget = min(6.0, remain)
                    Xq = lp_polish(Xp, budget)
                    vq = max_cos(Xq)
                    if vq < v:
                        Xp, v = Xq, vq
            if v < best_val - 1e-12:
                best_val = v
                best = Xp.copy()
            if time.time() - t0 > HARD - 30.0:
                break
        if time.time() - t0 > HARD - 30.0:
            break

    # if symmetry never beat, still try generic perturbation of record best to ensure progress
    if best is not None and time.time() - t0 < HARD - 25.0:
        # one extra diversification from best with moderate sigma
        for sigma in [0.04, 0.08]:
            if time.time() - t0 > HARD - 28.0:
                break
            Xp = best + sigma * rng.standard_normal((n, d))
            Xp /= np.linalg.norm(Xp, axis=1, keepdims=True)
            Xp = smooth_opt_full(Xp, 1200, 30.0, 320.0, 0.02, 0.002, mu=0.9)
            v = max_cos(Xp)
            if v < best_val + 0.015:
                remain = HARD - (time.time() - t0) - 10.0
                if remain > 1.5:
                    Xq = lp_polish(Xp, min(5.0, remain))
                    vq = max_cos(Xq)
                    if vq < v:
                        Xp, v = Xq, vq
            if v < best_val - 1e-12:
                best_val = v
                best = Xp.copy()

    remain = HARD - (time.time() - t0) - 1.0
    if remain > 2.0 and best is not None:
        best = lp_polish(best, min(remain, 22.0))
        best /= np.linalg.norm(best, axis=1, keepdims=True)
    best /= np.linalg.norm(best, axis=1, keepdims=True)
    return best
