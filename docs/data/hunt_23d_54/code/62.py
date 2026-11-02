import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _lp_polish(x, seconds, t0=None, deadline=1e9):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    start = time.time()
    iu0 = np.triu_indices(n, 1)
    while time.time() - start < seconds:
        if t0 is not None and (time.time() - t0) > deadline:
            break
        if radius < 1e-14:
            break
        Gfull = x @ x.T
        c_all = Gfull[iu0]
        m = c_all.max()
        sel = c_all > m - max(20 * radius, 1e-9)
        cnt = int(np.count_nonzero(sel))
        if cnt > 800:
            idx = np.where(sel)[0]
            order = np.argsort(c_all[idx])[::-1][:800]
            nsel = np.zeros_like(sel, dtype=bool)
            nsel[idx[order]] = True
            sel = nsel
        I, J = iu0[0][sel], iu0[1][sel]
        c = c_all[sel]
        p = len(I)
        if p == 0:
            break
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate(
            [I[:, None] * d + np.arange(d),
             J[:, None] * d + np.arange(d),
             np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix(
                (x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))),
                shape=(n, nv)).tocsr()
            obj = np.zeros(nv)
            obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-c, A_eq=A_eq,
                          b_eq=np.zeros(n),
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
        v = _max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def _screen_adam(x_init, steps, k0, k1, lr, t0, deadline):
    x = x_init.copy()
    n, d = x.shape
    # Adam state (tangent param, full ambient then projected)
    mt = np.zeros_like(x)
    vt = np.zeros_like(x)
    b1, b2, eps = 0.9, 0.999, 1e-8
    lk0, lk1 = np.log(k0), np.log(k1)
    for s in range(steps):
        if (s % 50 == 0) and t0 is not None and (time.time() - t0) > deadline:
            break
        frac = s / max(steps - 1, 1)
        k = np.exp(lk0 + (lk1 - lk0) * frac)
        G = x @ x.T
        m = float(np.max(G - np.eye(n) * 10.0))
        # stable exp of k*(G-m), zero diagonal
        E = np.exp(k * (G - m))
        np.fill_diagonal(E, 0.0)
        Gr = E @ x
        dot = np.sum(Gr * x, axis=1, keepdims=True)
        Gt = Gr - dot * x
        deg = np.sum(E, axis=1, keepdims=True) + 1.0
        Gt = Gt / deg
        # Adam update on tangent grad
        mt = b1 * mt + (1 - b1) * Gt
        vt = b2 * vt + (1 - b2) * (Gt * Gt)
        mth = mt / (1 - b1 ** (s + 1))
        vth = vt / (1 - b2 ** (s + 1))
        step = lr * mth / (np.sqrt(vth) + eps)
        # project step to tangent (remove radial part) for stability
        sd = np.sum(step * x, axis=1, keepdims=True)
        step = step - sd * x
        # cap step norm
        x = x - step
        nr = np.linalg.norm(x, axis=1, keepdims=True)
        nr[nr == 0] = 1.0
        x /= nr
    return x

def solve():
    t0 = time.time()
    DEADLINE = 170.0
    EXPLORE_END = 140.0
    rng = np.random.RandomState(0)
    n, d = 54, 23
    try:
        xb = np.load("best.npy").astype(float)
    except Exception:
        xb = rng.randn(n, d)
    xb = xb / np.linalg.norm(xb, axis=1, keepdims=True)
    if xb.shape != (n, d):
        tmp = rng.randn(n, d)
        tmp /= np.linalg.norm(tmp, axis=1, keepdims=True)
        xb = tmp
    try:
        xr = np.load("record.npy").astype(float)
        xr /= np.linalg.norm(xr, axis=1, keepdims=True)
        if xr.shape == (n, d):
            if _max_cos(xr) < _max_cos(xb):
                xb = xr.copy()
    except Exception:
        pass
    best = _max_cos(xb)
    x_best = xb.copy()

    # --- tangent bases ---
    # T: (n,d,dm) orthonormal tangent basis per point
    dm = d - 1
    try:
        T = np.zeros((n, d, dm))
        for i in range(n):
            xi = x_best[i]
            k = int(np.argmin(np.abs(xi)))
            # d-1 vectors e_j - xi[j]*xi
            M = np.zeros((d, dm))
            col = 0
            for j in range(d):
                if j == k:
                    continue
                e = np.zeros(d)
                e[j] = 1.0
                v = e - xi[j] * xi
                M[:, col] = v
                col += 1
            Q, _ = np.linalg.qr(M)
            T[i] = Q
    except Exception:
        T = None

    # --- rigidity matrix ---
    A = None
    Lc = None
    N = n * dm
    try:
        if T is not None:
            G0 = x_best @ x_best.T
            np.fill_diagonal(G0, -2.0)
            m0 = float(G0.max())
            slack_thr = 1e-6
            iu = np.triu_indices(n, 1)
            c_all = G0[iu]
            # adapt threshold to keep p in [200,600]
            for thr in [1e-6, 3e-6, 1e-5, 3e-7, 1e-7]:
                sel = c_all > m0 - thr
                p = int(np.count_nonzero(sel))
                if 150 <= p <= 650:
                    slack_thr = thr
                    break
            sel = c_all > m0 - slack_thr
            # cap to tightest 500 if too many
            if int(np.count_nonzero(sel)) > 550:
                idx = np.where(sel)[0]
                order = np.argsort(c_all[idx])[::-1][:550]
                nsel = np.zeros_like(sel, dtype=bool)
                nsel[idx[order]] = True
                sel = nsel
            I, J = iu[0][sel], iu[1][sel]
            p = len(I)
            if p >= 50:
                A = np.zeros((p, N), dtype=float)
                # precompute x_j^T T_i blocks
                for r in range(p):
                    i, j = int(I[r]), int(J[r])
                    a = x_best[j] @ T[i]  # (dm,)
                    b = x_best[i] @ T[j]
                    A[r, i * dm:(i + 1) * dm] = a
                    A[r, j * dm:(j + 1) * dm] = b
                Gg = A @ A.T
                # regularize
                reg = 1e-9 * max(1.0, float(np.trace(Gg) / max(p, 1)))
                Gg = Gg + reg * np.eye(p)
                try:
                    Lc = np.linalg.cholesky(Gg)
                except Exception:
                    Lc = None
    except Exception:
        A = None
        Lc = None

    def project_null(r):
        # r: (N,) -> null-projected
        if A is None or Lc is None:
            return r
        try:
            a = A @ r
            # solve Gg y = a via cholesky
            z = np.linalg.solve(Lc, a)
            y = np.linalg.solve(Lc.T, z)
            return r - A.T @ y
        except Exception:
            return r

    # precompute K soft modes (orthonormalized null vectors)
    K = 32
    Modes = None
    try:
        if A is not None and Lc is not None:
            Mraw = np.zeros((N, K))
            for kk in range(K):
                r = rng.randn(N)
                rp = project_null(r)
                nn = float(np.linalg.norm(rp))
                if nn < 1e-12:
                    rp = r
                    nn = float(np.linalg.norm(rp))
                Mraw[:, kk] = rp / max(nn, 1e-12)
            # orthonormalize
            Qq, _ = np.linalg.qr(Mraw)
            Modes = Qq  # N x K
    except Exception:
        Modes = None

    step_sizes = [0.03, 0.08, 0.15, 0.25]
    # quick polish of incumbent first (short) if time allows
    # skip long polish now; reserve for end

    trial = 0
    # exploration loop
    while (time.time() - t0) < EXPLORE_END:
        trial += 1
        if (time.time() - t0) > DEADLINE - 30.0:
            break
        # build displacement in tangent coords
        tot = step_sizes[rng.randint(0, len(step_sizes))]
        nmode = int(rng.randint(2, 6))  # 2..5
        if Modes is not None:
            cols = rng.choice(Modes.shape[1], size=nmode, replace=False)
            signs = rng.choice(np.array([-1.0, 1.0]), size=nmode)
            w = rng.rand(nmode) + 0.5
            combo = np.zeros(N)
            for cc, ss, ww in zip(cols, signs, w):
                combo += ss * ww * Modes[:, cc]
            nn = float(np.linalg.norm(combo))
            if nn < 1e-12:
                combo = rng.randn(N)
                nn = float(np.linalg.norm(combo))
            combo = combo / nn * tot
        else:
            combo = rng.randn(N)
            combo = combo / float(np.linalg.norm(combo)) * tot
        # map to ambient displacements
        try:
            U = combo.reshape(n, dm)
            Delta = np.zeros((n, d))
            for i in range(n):
                Delta[i] = T[i] @ U[i] if T is not None else 0.0
        except Exception:
            continue
        x_try = x_best + Delta
        x_try /= np.linalg.norm(x_try, axis=1, keepdims=True)
        # screen with fast LogSumExp Adam
        steps = 350 if tot <= 0.08 else 500
        # anneal k 100->600, lr 0.005 fixed-ish
        try:
            xs = _screen_adam(x_try, steps=steps, k0=100.0, k1=600.0, lr=0.005, t0=t0, deadline=DEADLINE - 28.0)
        except Exception:
            continue
        vs = _max_cos(xs)
        if vs > best + 0.004:
            continue
        # LP-polish only promising
        lp_sec = 2.0
        # occasionally longer
        if trial % 12 == 0:
            lp_sec = 3.5
        try:
            xp = _lp_polish(xs, lp_sec, t0, DEADLINE - 25.0)
        except Exception:
            continue
        vp = _max_cos(xp)
        if vp < best - 1e-12:
            best = vp
            x_best = xp.copy()
            # rebuild T, A, Modes around new best if big improvement? rebuild once
            if vp < best + 1e-12:
                # update tangent bases & A incrementally every improvement
                try:
                    for i in range(n):
                        xi = x_best[i]
                        k = int(np.argmin(np.abs(xi)))
                        M = np.zeros((d, dm))
                        col = 0
                        for j in range(d):
                            if j == k:
                                continue
                            e = np.zeros(d)
                            e[j] = 1.0
                            M[:, col] = e - xi[j] * xi
                            col += 1
                        Q, _ = np.linalg.qr(M)
                        T[i] = Q
                    G0 = x_best @ x_best.T
                    np.fill_diagonal(G0, -2.0)
                    m0 = float(G0.max())
                    iu2 = np.triu_indices(n, 1)
                    c_all2 = G0[iu2]
                    sel2 = c_all2 > m0 - 1e-6
                    if int(np.count_nonzero(sel2)) > 550:
                        idx = np.where(sel2)[0]
                        order = np.argsort(c_all2[idx])[::-1][:550]
                        nsel = np.zeros_like(sel2, dtype=bool)
                        nsel[idx[order]] = True
                        sel2 = nsel
                    I2, J2 = iu2[0][sel2], iu2[1][sel2]
                    p2 = len(I2)
                    if p2 >= 50:
                        A2 = np.zeros((p2, N), dtype=float)
                        for r2 in range(p2):
                            ii, jj = int(I2[r2]), int(J2[r2])
                            A2[r2, ii * dm:(ii + 1) * dm] = x_best[jj] @ T[ii]
                            A2[r2, jj * dm:(jj + 1) * dm] = x_best[ii] @ T[jj]
                        Gg2 = A2 @ A2.T
                        reg2 = 1e-9 * max(1.0, float(np.trace(Gg2) / max(p2, 1)))
                        Gg2 = Gg2 + reg2 * np.eye(p2)
                        Lc2 = np.linalg.cholesky(Gg2)
                        A, Lc = A2, Lc2
                        Mraw2 = np.zeros((N, K))
                        for kk in range(K):
                            rr = rng.randn(N)
                            aa = A @ rr
                            zz = np.linalg.solve(Lc, aa)
                            yy = np.linalg.solve(Lc.T, zz)
                            rp = rr - A.T @ yy
                            nn2 = float(np.linalg.norm(rp))
                            if nn2 < 1e-12:
                                rp = rr
                                nn2 = float(np.linalg.norm(rp))
                            Mraw2[:, kk] = rp / max(nn2, 1e-12)
                        Qq2, _ = np.linalg.qr(Mraw2)
                        Modes = Qq2
                except Exception:
                    pass
        # stop if out of time
        if (time.time() - t0) > EXPLORE_END:
            break

    # final LP polish with remaining time (reserve ~0.5s)
    try:
        left = DEADLINE - (time.time() - t0)
        if left > 2.0:
            x_best = _lp_polish(x_best, min(left - 0.5, 25.0), t0, DEADLINE)
    except Exception:
        pass
    x_best /= np.linalg.norm(x_best, axis=1, keepdims=True)
    return x_best
