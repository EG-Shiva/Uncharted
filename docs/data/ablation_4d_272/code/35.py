import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())


def lp_polish(x, seconds, r0=3e-3):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = max_cos(x)
    radius = float(r0)
    start = time.time()
    iu = np.triu_indices(n, 1)
    ar = np.arange(d)
    while time.time() - start < seconds and radius > 1e-14:
        G = x @ x.T
        c_all = G[iu]
        m = float(c_all.max())
        thr = m - max(20.0 * radius, 1e-9)
        sel_idx = np.where(c_all > thr)[0]
        if sel_idx.size == 0:
            break
        if sel_idx.size > 3500:
            part = np.argpartition(c_all, -3500)[-3500:]
            sel_idx = part
        I = iu[0][sel_idx]
        J = iu[1][sel_idx]
        c_sel = c_all[sel_idx]
        p = sel_idx.size
        nv = n * d + 1
        cols_blocks = np.empty((p, 2 * d + 1), dtype=np.int64)
        cols_blocks[:, :d] = I[:, None] * d + ar[None, :]
        cols_blocks[:, d:2 * d] = J[:, None] * d + ar[None, :]
        cols_blocks[:, 2 * d] = n * d
        cols = cols_blocks.ravel()
        vals_blocks = np.empty((p, 2 * d + 1), dtype=np.float64)
        vals_blocks[:, :d] = x[J]
        vals_blocks[:, d:2 * d] = x[I]
        vals_blocks[:, 2 * d] = -1.0
        vals = vals_blocks.ravel()
        rows = np.repeat(np.arange(p), 2 * d + 1)
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix(
                (x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))),
                shape=(n, nv)).tocsr()
            obj = np.zeros(nv)
            obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-c_sel, A_eq=A_eq,
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
        v = max_cos(y)
        if v < best - 1e-15:
            x = y
            best = v
            radius = min(radius * 1.5, 0.02)
        else:
            radius *= 0.5
            if radius < 1e-12:
                break
    return x


def smooth_relax(cur, iters=600, step0=0.022, T0=0.02, T1=0.00125, mu=0.82):
    n, d = cur.shape
    cur = cur / np.linalg.norm(cur, axis=1, keepdims=True)
    V = np.zeros_like(cur)
    G0 = cur @ cur.T
    np.fill_diagonal(G0, -2.0)
    best = float(G0.max())
    best_x = cur.copy()
    step = float(step0)
    ratio = (T1 / T0) if T0 > 0 else 1.0
    for it in range(iters):
        frac = it / max(1, iters)
        T = T0 * (ratio ** frac) if T0 > 0 else T1
        if T < 1e-12:
            T = 1e-12
        G = cur @ cur.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        cutoff = m - 12.0 * T
        if cutoff < m - 0.08:
            cutoff = m - 0.08
        mask = G > cutoff
        W = np.zeros_like(G)
        Gm = G[mask]
        if Gm.size > 0:
            W[mask] = np.exp((Gm - m) / T)
        sumW = W.sum(axis=1, keepdims=True)
        zm = sumW[:, 0] < 1e-12
        if np.any(zm):
            zidx = np.where(zm)[0]
            amax = np.argmax(G[zidx], axis=1)
            W[zidx, amax] = 1.0
            sumW[zidx, 0] = 1.0
        S = W @ cur
        coeff = (W * G).sum(axis=1, keepdims=True)
        F = (-S + cur * coeff) / (sumW + 1e-12)
        V = mu * V + step * F
        nd = (V * cur).sum(axis=1, keepdims=True)
        V = V - nd * cur
        vn = np.sqrt((V * V).sum(axis=1, keepdims=True))
        scale = np.minimum(1.0, 0.025 / (vn + 1e-12))
        V = V * scale
        cur = cur + V
        cur /= np.linalg.norm(cur, axis=1, keepdims=True)
        step *= 0.9997
        if m < best:
            best = m
            best_x = cur.copy()
        if (it % 50 == 49) or (it == iters - 1):
            Gt = cur @ cur.T
            np.fill_diagonal(Gt, -2.0)
            vt = float(Gt.max())
            if vt < best:
                best = vt
                best_x = cur.copy()
    return cur, best_x, best


def build_tangent_bases(x):
    n, d = x.shape
    B = np.empty((n, d, 3), dtype=np.float64)
    e0 = np.array([1.0, 0.0, 0.0, 0.0])
    e1 = np.array([0.0, 1.0, 0.0, 0.0])
    I4 = np.eye(4)
    for i in range(n):
        xi = x[i]
        if abs(float(xi @ e0)) < 0.99:
            ref = e0
        else:
            ref = e1
        u = xi - ref
        nu = np.linalg.norm(u)
        if nu < 1e-12:
            # xi == ref, pick any orthogonal
            if abs(float(xi @ e0)) < 0.99:
                B[i] = np.eye(4)[:, 1:].copy() if False else np.zeros((4, 3))
                # fallback gram schmidt
                M = I4 - np.outer(xi, xi)
                q, _ = np.linalg.qr(M + 1e-9 * np.eye(4))
                # find 3 cols orthogonal to xi
                dots = np.abs(q.T @ xi)
                idx = np.argsort(dots)[:3]
                B[i] = q[:, idx]
            else:
                M = I4 - np.outer(xi, xi)
                q, _ = np.linalg.qr(M + 1e-9 * np.eye(4))
                dots = np.abs(q.T @ xi)
                idx = np.argsort(dots)[:3]
                B[i] = q[:, idx]
            continue
        u = u / nu
        H = I4 - 2.0 * np.outer(u, u)
        # first column should be xi (up to sign); ensure ordering
        # H maps xi to ref, so H[:,0] is not xi; instead H maps: H@xi=ref.
        # Tangent basis: columns of H^T restricted? We need 3 orthonormal vectors perp to xi.
        # Take ref-perp basis and map back: let E = [e1,e2,e3] perp to e0=ref (if ref=e0).
        # Then B = H @ E (since H symmetric, H@E perp to xi).
        if ref is e0:
            E = np.eye(4)[:, 1:4]
        else:
            E = np.eye(4)[:, [0, 2, 3]]
        Bi = H @ E
        # re-orthogonalize against xi numerically
        Bi = Bi - np.outer(xi, (xi @ Bi))
        # qr to orthonormalize
        q, _ = np.linalg.qr(Bi)
        B[i] = q
    return B


def build_rigidity(x, B, I, J):
    n = x.shape[0]
    p = I.shape[0]
    R = np.empty((p, n * 3), dtype=np.float64)
    for r in range(p):
        i = int(I[r]); j = int(J[r])
        R[r, i * 3:(i + 1) * 3] = B[i].T @ x[j]
        R[r, j * 3:(j + 1) * 3] = B[j].T @ x[i]
    return R


def simplex_proj(v):
    n = v.shape[0]
    u = np.sort(v)[::-1]
    css = np.cumsum(u)
    rho = -1
    for j in range(n):
        if u[j] + (1.0 - css[j]) / (j + 1) > 0:
            rho = j
    if rho < 0:
        return np.full_like(v, 1.0 / n)
    theta = (css[rho] - 1.0) / (rho + 1)
    w = v - theta
    w[w < 0] = 0
    s = w.sum()
    if s <= 0:
        return np.full_like(v, 1.0 / n)
    return w / s


def equilibrium_stress(R, iters=400):
    p = R.shape[0]
    w = np.full(p, 1.0 / p, dtype=np.float64)
    # estimate step: 1 / ||R R^T|| ~ 1 / smax^2
    try:
        smax = np.linalg.norm(R, ord=2)
    except Exception:
        smax = 5.0
    eta = 0.5 / (smax * smax + 1e-9)
    eta = min(max(eta, 1e-4), 0.2)
    for _ in range(iters):
        Rt_w = R.T @ w  # 3n
        grad = R @ Rt_w  # p
        w = w - eta * grad
        w = simplex_proj(w)
    return w


def rigidity_escape(x, rng, strength=0.03, nmodes_pool=20, mode_kind=None):
    n, d = x.shape
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    m = float(G.max())
    iu = np.triu_indices(n, 1)
    c_all = G[iu]
    # contacts within 1e-7 of max
    sel = np.where(c_all > m - 1e-7)[0]
    if sel.size < 30:
        sel = np.where(c_all > m - 1e-4)[0]
    if sel.size < 30:
        # fallback: top 300
        k = min(300, c_all.size)
        sel = np.argpartition(c_all, -k)[-k:]
    if sel.size > 3000:
        k = 3000
        part = np.argpartition(c_all[sel], -k)[-k:]
        sel = sel[part]
    I = iu[0][sel]
    J = iu[1][sel]
    p = sel.size
    try:
        B = build_tangent_bases(x)
        R = build_rigidity(x, B, I, J)
    except Exception:
        # isotropic fallback
        y = x + 0.01 * rng.standard_normal(x.shape)
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        return y
    try:
        U, S, Vt = np.linalg.svd(R, full_matrices=False)
    except Exception:
        y = x + 0.01 * rng.standard_normal(x.shape)
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        return y
    K = Vt.shape[0]  # = min(p,3n)
    # floppyest are last rows
    pool = min(nmodes_pool, K)
    Flop = Vt[K - pool:K, :]  # pool x 3n
    # stress for unloading weight
    try:
        w = equilibrium_stress(R, iters=250)
    except Exception:
        w = np.ones(p) / p
    # identify highest-stress contacts
    order = np.argsort(-w)
    topk = min(60, p)
    top_idx = order[:topk]
    w_top = w[top_idx]
    w_top = w_top / (w_top.sum() + 1e-12)
    Rtop = R[top_idx, :]
    g_full = Rtop.T @ w_top  # 3n, direction increasing stress
    # project onto floppy subspace
    # V_f basis rows; projection:
    coeff_full = Flop @ g_full  # pool
    g_flop = Flop.T @ coeff_full  # 3n in span
    nrm = np.linalg.norm(g_flop)
    if nrm < 1e-12:
        u_unload = -g_full / (np.linalg.norm(g_full) + 1e-12)
        # project to tangent? g_full already in tangent coords
    else:
        u_unload = -g_flop / nrm
    # choose direction
    if mode_kind is None:
        mode_kind = int(rng.integers(0, 6))
    u = None
    if mode_kind == 0:
        # single floppyest
        idx = int(rng.integers(0, min(5, pool)))
        u = Flop[pool - 1 - idx, :]
        if rng.random() < 0.5:
            u = -u
    elif mode_kind == 1:
        k = int(rng.integers(3, 6))  # 3..5 combo
        k = min(k, pool)
        selm = rng.choice(pool, size=k, replace=False)
        coef = rng.standard_normal(k)
        coef /= (np.linalg.norm(coef) + 1e-12)
        u = (coef[:, None] * Flop[selm, :]).sum(axis=0)
    elif mode_kind == 2:
        k = min(pool, int(rng.integers(8, 21)))
        selm = rng.choice(pool, size=k, replace=False)
        coef = rng.standard_normal(k)
        coef /= (np.linalg.norm(coef) + 1e-12)
        u = (coef[:, None] * Flop[selm, :]).sum(axis=0)
    elif mode_kind == 3:
        u = u_unload
        if rng.random() < 0.2:
            u = -u
    elif mode_kind == 4:
        # unload + floppy combo
        k = min(pool, int(rng.integers(4, 11)))
        selm = rng.choice(pool, size=k, replace=False)
        coef = rng.standard_normal(k)
        coef /= (np.linalg.norm(coef) + 1e-12)
        u_rand = (coef[:, None] * Flop[selm, :]).sum(axis=0)
        a = float(rng.uniform(0.4, 1.2))
        b = float(rng.uniform(0.4, 1.2))
        u = a * u_unload + b * u_rand
    else:
        # random full-space tangent gaussian projected to floppy + unload mix
        g = rng.standard_normal(3 * n)
        # project onto floppy for collective part
        cf = Flop @ g
        u = Flop.T @ cf
        if np.linalg.norm(u) < 1e-12:
            u = g
    nu = np.linalg.norm(u)
    if nu < 1e-12 or not np.all(np.finite(u)):
        y = x + 0.01 * rng.standard_normal(x.shape)
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        return y
    u = u / nu
    # convert to per-point displacements
    U3 = u.reshape(n, 3)
    D = np.empty_like(x)
    for i in range(n):
        D[i] = B[i] @ U3[i]
    # rescale so max per-point norm == strength (0.005..0.08)
    norms = np.linalg.norm(D, axis=1)
    mx = float(norms.max())
    if mx < 1e-12:
        y = x + 0.01 * rng.standard_normal(x.shape)
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        return y
    D = D * (strength / mx)
    y = x + D
    y /= np.linalg.norm(y, axis=1, keepdims=True)
    # small extra isotropic jitter to break exact symmetry (10% of strength)
    jit = 0.15 * strength * rng.standard_normal(x.shape)
    # project jitter to tangent to avoid radial waste
    dot = (jit * y).sum(axis=1, keepdims=True)
    jit = jit - dot * y
    y = y + jit
    y /= np.linalg.norm(y, axis=1, keepdims=True)
    return y


def solve():
    t0 = time.time()
    BUDGET = 171.0
    deadline = t0 + BUDGET
    rng = np.random.default_rng(12345)
    try:
        xb = np.load("best.npy").astype(np.float64)
    except Exception:
        xb = None
    try:
        xr = np.load("record.npy").astype(np.float64)
    except Exception:
        xr = None
    cands = []
    if xb is not None and xb.shape == (272, 4):
        xb = xb / np.linalg.norm(xb, axis=1, keepdims=True)
        cands.append(xb)
    if xr is not None and xr.shape == (272, 4):
        xr = xr / np.linalg.norm(xr, axis=1, keepdims=True)
        cands.append(xr)
    if len(cands) == 0:
        x0 = rng.standard_normal((272, 4))
        x0 /= np.linalg.norm(x0, axis=1, keepdims=True)
        cands.append(x0)
    best_val = 1e9
    x_best = cands[0].copy()
    for c in cands:
        v = max_cos(c)
        if v < best_val:
            best_val = v
            x_best = c.copy()
    n, d = 272, 4
    trial = 0
    # quick LP on start to ensure locally optimal baseline (short)
    # main loop: rigidity-aware escapes
    strengths = [0.005, 0.01, 0.02, 0.03, 0.045, 0.06, 0.08]
    while True:
        trial += 1
        if time.time() > deadline - 26:
            break
        # cycle mode kinds 0..5, strengths sampled
        mode_kind = (trial - 1) % 6
        # bias strengths: mostly medium
        si = int(rng.integers(0, len(strengths)))
        # occasionally large
        if trial % 9 == 0:
            si = len(strengths) - 1 - int(rng.integers(0, 3))
        strength = float(strengths[si] * rng.uniform(0.8, 1.25))
        strength = min(max(strength, 0.004), 0.09)
        pool = 20
        if mode_kind in (1,):
            pool = 10
        elif mode_kind == 0:
            pool = 8
        y = x_best.copy()
        use_rigidity = True
        if trial % 11 == 0:
            use_rigidity = False  # isotropic fallback to diversify
        if use_rigidity:
            try:
                y = rigidity_escape(x_best, rng, strength=strength,
                                   nmodes_pool=pool, mode_kind=mode_kind)
            except Exception:
                y = x_best + 0.01 * rng.standard_normal(x_best.shape)
                y /= np.linalg.norm(y, axis=1, keepdims=True)
        else:
            sig = float(rng.uniform(0.006, 0.02))
            y = x_best + sig * rng.standard_normal(x_best.shape)
            y /= np.linalg.norm(y, axis=1, keepdims=True)
        # brief annealed softmax phase beta 50->800
        iters = 550
        if trial % 3 == 0:
            iters = 700
        elif trial % 3 == 1:
            iters = 450
        s0 = float(rng.uniform(0.016, 0.026))
        T0 = 0.02  # beta 50
        T1 = 0.00125  # beta 800
        mu = float(rng.uniform(0.80, 0.86))
        try:
            cur, best_path, v_path = smooth_relax(y, iters=iters, step0=s0, T0=T0, T1=T1, mu=mu)
        except Exception:
            continue
        v_cur = max_cos(cur)
        if v_cur > best_val + 0.012 and v_path > best_val + 0.006:
            continue
        remain = deadline - 26 - time.time()
        if remain < 1.0:
            break
        secs = 1.6 if (trial % 3 != 0) else 2.4
        secs = min(secs, remain)
        polish_start = cur
        if v_path + 1e-9 < v_cur and v_path < best_val + 0.002 and rng.random() < 0.25:
            polish_start = best_path
        try:
            z = lp_polish(polish_start, seconds=secs, r0=3e-3)
        except Exception:
            continue
        v = max_cos(z)
        if v < best_val - 1e-12:
            x_best = z
            best_val = v
        if v_path < best_val - 1e-12:
            try:
                rem2 = deadline - 26 - time.time()
                if rem2 > 1.0:
                    z2 = lp_polish(best_path, seconds=min(1.1, rem2), r0=2e-3)
                    v2 = max_cos(z2)
                    if v2 < best_val - 1e-12:
                        x_best = z2
                        best_val = v2
            except Exception:
                pass
    rem = deadline - time.time() - 0.8
    if rem > 3:
        try:
            cur, best_path, v_path = smooth_relax(x_best, iters=350, step0=0.008, T0=0.001, T1=0.0002, mu=0.80)
            vc = max_cos(cur)
            cand_best = x_best
            cand_val = best_val
            if vc < cand_val:
                cand_best = cur
                cand_val = vc
            if v_path < cand_val:
                cand_best = best_path
                cand_val = v_path
            x_best = cand_best
            best_val = cand_val
        except Exception:
            pass
        rem2 = deadline - time.time() - 0.4
        if rem2 > 2:
            try:
                x_best = lp_polish(x_best, seconds=rem2, r0=4e-3)
            except Exception:
                pass
    x_best = x_best / np.linalg.norm(x_best, axis=1, keepdims=True)
    x_best = np.nan_to_num(x_best, nan=0.0, posinf=0.0, neginf=0.0)
    norms = np.linalg.norm(x_best, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    x_best = x_best / norms
    return x_best
