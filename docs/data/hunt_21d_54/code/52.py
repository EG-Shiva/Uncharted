import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _proj(x):
    nrm = np.linalg.norm(x, axis=1, keepdims=True)
    nrm[nrm == 0] = 1.0
    return x / nrm

def _lp_polish(x, seconds):
    x = _proj(x)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        G = x @ x.T
        c = G[iu]
        m = float(c.max())
        sel = c > m - max(20 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        if p > 2500:
            idx = np.argpartition(-c[sel], 2500)[:2500]
            I, J = I[idx], J[idx]
            csv = c[sel][idx]
            p = 2500
        else:
            csv = c[sel]
        if p == 0:
            break
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))),
                          shape=(n, nv)).tocsr()
        obj = np.zeros(nv)
        obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-csv, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                          method="highs",
                          options={"time_limit": 3.0})
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y = _proj(y)
        v = _max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.02)
        else:
            radius *= 0.5
    return x

def _quench(y, steps=350, lr0=0.35, mu=0.85):
    n, d = y.shape
    vel = np.zeros((n, d))
    for s in range(steps):
        frac = s / steps
        T = 0.015 * (1 - frac) + 0.0025 * frac
        G = y @ y.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        E = (G - m) / T
        np.clip(E, -20, 0, out=E)
        W = np.exp(E)
        np.fill_diagonal(W, 0.0)
        rs = W.sum(axis=1, keepdims=True) + 1e-12
        grad = W @ y
        dot = np.einsum('ij,ij->i', grad, y)
        grad -= dot[:, None] * y
        grad /= rs
        vel = mu * vel + grad
        lr = lr0 * (1.0 - 0.6 * frac)
        y = y - lr * vel
        y = _proj(y)
    return y

def solve():
    T0 = time.time()
    LIMIT = 170.0
    def left():
        return LIMIT - (time.time() - T0)
    rng = np.random.default_rng(7)
    try:
        b0 = np.load("best.npy")
    except Exception:
        b0 = np.load("record.npy")
    b0 = _proj(np.nan_to_num(np.asarray(b0, dtype=float)))
    if b0.shape != (54, 21):
        b0 = _proj(rng.normal(size=(54, 21)))
    try:
        r0 = _proj(np.asarray(np.load("record.npy"), dtype=float))
        if r0.shape == (54, 21):
            if _max_cos(r0) < _max_cos(b0):
                b0 = r0.copy()
    except Exception:
        pass
    n, d = 54, 21
    best = b0.copy()
    bv = _max_cos(best)

    # original contact set for Hamming distance
    iu = np.triu_indices(n, 1)
    G0 = best @ best.T
    m0 = float(np.max(G0 - np.eye(n) * 10))
    mask0 = (G0[iu] > m0 - 1e-9)

    # ---- build tangent-contact Jacobian and null modes ----
    # pairs with slack < 1e-4
    G = best @ best.T
    c_all = G[iu]
    m_all = float(c_all.max())
    slack = m_all - c_all
    sel = slack < 1e-4
    if int(sel.sum()) < 200:
        sel = slack < 1e-3
    if int(sel.sum()) > 1800:
        # keep tightest 1800
        idx = np.argpartition(slack, 1800)[:1800]
        I = iu[0][idx]; J = iu[1][idx]
    else:
        I = iu[0][sel]; J = iu[1][sel]
    p = len(I)
    # tangent bases Q_i : d x (d-1+1?) use 20 dim
    Rs = np.random.RandomState(0).randn(d, d - 1)  # d x 20
    Qs = []
    for i in range(n):
        xi = best[i]
        Z = Rs - np.outer(xi, xi @ Rs)
        Q, _ = np.linalg.qr(Z)
        Qs.append(Q)
    Qs = np.array(Qs)  # n x d x 20
    Mt = np.zeros((p, n * 20))
    # vectorized fill via loop over p (p~800, fine)
    for k in range(p):
        i, j = int(I[k]), int(J[k])
        Mt[k, i * 20:(i + 1) * 20] = best[j] @ Qs[i]
        Mt[k, j * 20:(j + 1) * 20] = best[i] @ Qs[j]
    K = 15
    try:
        # capped SVD: Mt is ~800x1080, fast
        U, S, Vt = np.linalg.svd(Mt, full_matrices=False)
        K = min(K, Vt.shape[0])
        # smallest K rows
        modes_red = Vt[-K:, :]  # K x 1080
    except Exception:
        modes_red = rng.normal(size=(K, n * 20))
    # map to full dx fields, normalized to Fro=1
    modes_full = np.zeros((K, n, d))
    for k in range(K):
        cr = modes_red[k]
        for i in range(n):
            modes_full[k, i, :] = Qs[i] @ cr[i * 20:(i + 1) * 20]
        fn = np.linalg.norm(modes_full[k]) + 1e-12
        modes_full[k] /= fn

    amps = [0.01, 0.03, 0.06, 0.12, 0.20, 0.30]
    lp_used = 0
    trial = 0
    # budget: leave ~18s for final polish
    while left() > 22 and trial < 70:
        trial += 1
        w = rng.normal(size=K)
        w /= (np.linalg.norm(w) + 1e-12)
        # combine
        dx = np.tensordot(w, modes_full, axes=(0, 0))  # n x d
        # ensure tangent (numerical)
        dx -= (np.einsum('ij,ij->i', dx, best)[:, None]) * best
        fn = np.linalg.norm(dx) + 1e-12
        dx /= fn
        A = amps[(trial - 1) % len(amps)] * (0.8 + 0.4 * rng.random())
        scale = A * np.sqrt(n)
        dxs = dx * scale
        ni = np.linalg.norm(dxs, axis=1)
        y = best.copy()
        for i in range(n):
            a = ni[i]
            if a < 1e-12:
                continue
            y[i] = np.cos(a) * best[i] + np.sin(a) * (dxs[i] / a)
        y = _proj(y)
        # exact score + Hamming
        Gy = y @ y.T
        my = float(np.max(Gy - np.eye(n) * 10))
        if my > 0.15:
            continue
        masky = (Gy[iu] > my - 1e-9)
        # Hamming vs original (different sizes ok via full-length vectors)
        ham = int(np.sum(masky != mask0))
        if ham < 20:
            continue
        # repair with annealed soft-max descent, no LP
        # adapt steps: smaller amp -> fewer steps
        steps = 300 if A < 0.1 else 450
        # cap steps if time low
        if left() < 40:
            steps = 250
        yr = _quench(y.copy(), steps=steps)
        vr = _max_cos(yr)
        if vr < bv - 1e-12:
            # promising: LP polish only if already below 0.07
            if vr < 0.07 and lp_used < 3 and left() > 25:
                yr2 = _lp_polish(yr, min(4.0, left() - 20))
                vr2 = _max_cos(yr2)
                lp_used += 1
                if vr2 < vr:
                    yr, vr = yr2, vr2
            if vr < bv:
                best, bv = yr, vr
                # update reference mask for further Hamming
                G0 = best @ best.T
                m0 = float(np.max(G0 - np.eye(n) * 10))
                mask0 = (G0[iu] > m0 - 1e-9)
                if bv < 0.0666634 - 1e-9:
                    pass
        # timeout guard
        if left() <= 22:
            break

    # final squeeze of best if time remains
    if left() > 8:
        # quick quench touch then short LP
        if left() > 20:
            cand = _quench(best.copy(), steps=200, lr0=0.25)
            if _max_cos(cand) < bv:
                best, bv = cand, _max_cos(cand)
        best = _lp_polish(best, min(12.0, left() - 2.0))
        bv = _max_cos(best)

    # tiny shrink for strict feasibility (avoid rounding to larger max)
    best = _proj(best)
    return best
