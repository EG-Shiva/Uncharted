import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(np.max(g))

def _norm(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return x / n

def _lse_adam(x, steps, k0, k1, lr0, lr1, w0, w1, deadline, b1=0.9, b2=0.999, eps=1e-8):
    n, d = x.shape
    m1 = np.zeros_like(x)
    m2 = np.zeros_like(x)
    lk0 = np.log(k0); lk1 = np.log(k1)
    ll0 = np.log(lr0); ll1 = np.log(lr1)
    t = 0
    for s in range(steps):
        if (s & 63) == 0 and time.time() > deadline:
            break
        f = s / max(steps - 1, 1)
        k = np.exp(lk0 + (lk1 - lk0) * f)
        lr = np.exp(ll0 + (ll1 - ll0) * f)
        w = w0 + (w1 - w0) * f
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        # max for shift
        cur = float(np.max(G))
        D = G - cur
        W = np.zeros_like(G)
        # window to keep k*D in range
        sel = D > -w
        Ds = D[sel]
        # clip to avoid overflow (Ds<=0)
        W[sel] = np.exp(k * Ds)
        np.fill_diagonal(W, 0.0)
        F = W @ x
        dot = np.sum(F * x, axis=1, keepdims=True)
        Ft = F - dot * x
        t += 1
        m1 = b1 * m1 + (1.0 - b1) * Ft
        m2 = b2 * m2 + (1.0 - b2) * (Ft * Ft)
        c1 = 1.0 - b1 ** t
        c2 = 1.0 - b2 ** t
        mh = m1 / c1
        vh = m2 / c2
        step = lr * mh / (np.sqrt(vh) + eps)
        # limit step RMS to lr*2 for stability
        # simple global clip
        rms = float(np.sqrt(np.sum(step * step) / n)) + 1e-12
        maxrms = lr * 3.0
        if rms > maxrms:
            step *= (maxrms / rms)
        x = x - step
        # renormalize
        nr = np.linalg.norm(x, axis=1, keepdims=True)
        nr[nr == 0] = 1.0
        x /= nr
    return x

def _lp_polish(x, deadline, r0=0.008, rmax=0.03, cap=700):
    n, d = x.shape
    x = _norm(x.copy())
    best = _max_cos(x)
    best_x = x.copy()
    radius = r0
    iu0, iu1 = np.triu_indices(n, 1)
    ar = np.arange(d)
    while time.time() < deadline and radius > 1e-13:
        G = x @ x.T
        c_all = G[iu0, iu1]
        m = float(c_all.max())
        thresh = m - max(20.0 * radius, 1e-9)
        # select
        idx = np.where(c_all > thresh)[0]
        if idx.size > cap:
            part = np.argpartition(c_all, -cap)[-cap:]
            idx = part
        if idx.size == 0:
            break
        c_sel = c_all[idx]
        I = iu0[idx]; J = iu1[idx]
        p = idx.size
        nv = n * d + 1
        Wc = 2 * d + 1
        cols = np.empty((p, Wc), dtype=np.int64)
        vals = np.empty((p, Wc), dtype=np.float64)
        cols[:, :d] = I[:, None] * d + ar[None, :]
        cols[:, d:2 * d] = J[:, None] * d + ar[None, :]
        cols[:, 2 * d] = n * d
        vals[:, :d] = x[J]
        vals[:, d:2 * d] = x[I]
        vals[:, 2 * d] = -1.0
        rows = np.repeat(np.arange(p), Wc)
        A_ub = coo_matrix((vals.ravel(), (rows, cols.ravel())), shape=(p, nv)).tocsr()
        erows = np.repeat(np.arange(n), d)
        ecols = np.arange(n * d)
        A_eq = coo_matrix((x.ravel(), (erows, ecols)), shape=(n, nv)).tocsr()
        obj = np.zeros(nv); obj[-1] = 1.0
        bounds = [(-radius, radius)] * (n * d) + [(None, None)]
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c_sel, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=bounds, method="highs",
                          options={"time_limit": 2.0})
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            if radius < 1e-7:
                x = best_x.copy()
            continue
        y = x + res.x[:-1].reshape(n, d)
        y = _norm(y)
        v = _max_cos(y)
        if v < best:
            best = v
            best_x = y.copy()
            x = y
            radius = min(radius * 1.5, rmax)
        else:
            radius *= 0.5
            if radius < 1e-7:
                x = best_x.copy()
        if time.time() >= deadline:
            break
    return best_x, best

def _contact_mask(x, m, tol=5e-6):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    iu0, iu1 = np.triu_indices(x.shape[0], 1)
    c = G[iu0, iu1]
    return (c > (m - tol)), iu0, iu1

def solve():
    t0 = time.time()
    T_HARD = t0 + 172.0
    T_EXP = t0 + 157.0  # reserve 15s final polish
    rng = np.random.default_rng(12345)
    n, d = 54, 23
    try:
        x0 = np.load("best.npy").astype(float)
    except Exception:
        x0 = None
    try:
        xr = np.load("record.npy").astype(float)
    except Exception:
        xr = None
    cands = []
    if x0 is not None and x0.shape == (n, d):
        cands.append(_norm(x0))
    if xr is not None and xr.shape == (n, d):
        cands.append(_norm(xr))
    if len(cands) == 0:
        cands.append(_norm(rng.standard_normal(size=(n, d))))
    best = 1e9
    x_best = cands[0].copy()
    for c in cands:
        v = _max_cos(c)
        if v < best:
            best = v
            x_best = c.copy()
    # tiny initial squeeze (2s) to ensure local optimum baseline
    try:
        if time.time() < T_EXP - 1.0:
            xb, vb = _lp_polish(x_best, min(time.time() + 2.0, T_EXP - 0.5), r0=0.004, rmax=0.02, cap=800)
            if vb < best:
                x_best, best = xb, vb
    except Exception:
        pass
    # contact-graph signature memory for clustering
    sig_masks = []
    # store incumbent signature
    try:
        m0, _, _ = _contact_mask(x_best, best, tol=1e-6)
        sig_masks.append(m0)
    except Exception:
        pass
    iu0_all, iu1_all = np.triu_indices(n, 1)
    lp_used = 0
    LP_MAX = 4
    screen_tol = 0.0015  # only LP candidates within best+screen_tol
    it = 0
    # amplitude range log-uniform 0.03..0.30
    logA0, logA1 = np.log(0.03), np.log(0.30)
    while time.time() < T_EXP:
        it += 1
        now = time.time()
        if now >= T_EXP or now > T_HARD - 16:
            break
        # --- build tangent-space collective modes via contact Laplacian ---
        Gc = x_best @ x_best.T
        np.fill_diagonal(Gc, -2.0)
        mcur = float(np.max(Gc))
        # rigidity pairs within 1e-6 of max
        Athr = mcur - 1e-6
        A = (Gc > Athr).astype(float)
        np.fill_diagonal(A, 0.0)
        # ensure at least ~200 edges; if too few widen
        if np.sum(A) < 400:
            Athr2 = mcur - 8e-6
            A = (Gc > Athr2).astype(float)
            np.fill_diagonal(A, 0.0)
        deg = np.sum(A, axis=1)
        # Laplacian
        L = np.diag(deg) - A
        try:
            ew, ev = np.linalg.eigh(L)
        except Exception:
            ev = None
        # pick five softest non-trivial modes (skip zero mode)
        if ev is not None and ev.shape[1] >= 6:
            U = ev[:, 1:6]  # 54x5
        else:
            # fallback random orthonormal
            U = rng.standard_normal(size=(n, 5))
            U -= U.mean(axis=0, keepdims=True)
        # approximate equilibrium stress: degree-normalized weights (for record only)
        # random combination coefficients
        coeff = rng.standard_normal(size=(5,))
        # random tangent directions per mode
        R = rng.standard_normal(size=(5, d))
        # project each R_k to tangent of each point
        # t_k,i = R_k - (R_k.x_i) x_i
        # compute dots: (n,5) = X @ R.T
        dots = x_best @ R.T  # n x 5
        # Build displacement: dx_i = sum_k coeff_k * U[i,k] * (R_k - dots[i,k]*x_i)
        # vectorized
        # Expand: term1 = sum_k coeff_k*U[i,k]*R_k  = (U * coeff) @ R
        Uc = U * coeff[None, :]  # n x5
        dx = Uc @ R  # n x d
        # term2 = sum_k coeff_k*U[i,k]*dots[i,k]*x_i = s_i * x_i
        s = np.sum(Uc * dots, axis=1)  # n
        dx -= s[:, None] * x_best
        # rescale to target amplitude (RMS norm)
        rms = float(np.sqrt(np.sum(dx * dx) / n)) + 1e-12
        A_target = float(np.exp(rng.uniform(logA0, logA1)))
        # occasionally try larger kick (10% chance up to 0.30 already max; keep)
        dx *= (A_target / rms)
        x = _norm(x_best + dx)
        # --- cheap LogSumExp Adam screening 300-600 steps ---
        steps = int(rng.integers(300, 601))
        # randomize anneal slightly
        k0 = float(rng.uniform(60, 110))
        k1 = float(rng.uniform(1800, 2800))
        lr0 = float(np.exp(rng.uniform(np.log(0.008), np.log(0.015))))
        lr1 = float(np.exp(rng.uniform(np.log(0.0006), np.log(0.0012))))
        w0 = float(rng.uniform(0.05, 0.07))
        w1 = float(rng.uniform(0.004, 0.007))
        # inner deadline: leave time for LP/final
        inner_dl = min(time.time() + 1.5, T_EXP, T_HARD - 16)
        try:
            xa = _lse_adam(x, steps=steps, k0=k0, k1=k1, lr0=lr0, lr1=lr1, w0=w0, w1=w1, deadline=inner_dl)
        except Exception:
            continue
        va = _max_cos(xa)
        # screen by true max
        if not np.isfinite(va):
            continue
        if va > best + 0.004:
            continue
        # --- cluster by thresholded contact graph ---
        try:
            Gxa = xa @ xa.T
            np.fill_diagonal(Gxa, -2.0)
            cvec = Gxa[iu0_all, iu1_all]
            mask = cvec > (va - 5e-6)
        except Exception:
            mask = None
        is_dup = False
        if mask is not None and len(sig_masks) > 0:
            # Jaccard vs stored
            best_j = 0.0
            for sm in sig_masks:
                inter = np.sum(mask & sm)
                union = np.sum(mask | sm) + 1e-12
                j = inter / union
                if j > best_j:
                    best_j = j
            if best_j > 0.92 and va >= best - 1e-12:
                # repeated return to incumbent basin -> discard
                continue
        # accept improvement immediately
        if va < best - 1e-12:
            x_best, best = xa, va
            if mask is not None:
                sig_masks.append(mask.copy())
                if len(sig_masks) > 60:
                    sig_masks.pop(0)
            continue
        # near-best: store signature to avoid re-visits, maybe LP
        if mask is not None:
            # only store if reasonably close (avoid flooding)
            if va < best + 0.002:
                sig_masks.append(mask.copy())
                if len(sig_masks) > 60:
                    sig_masks.pop(0)
        # rare LP: only if below screening threshold and budget left
        if lp_used >= LP_MAX:
            continue
        if va > best + screen_tol:
            continue
        if time.time() > T_HARD - 16 or time.time() > T_EXP:
            continue
        # at most handful of 2s LP calls
        dl = min(time.time() + 2.0, T_EXP, T_HARD - 15)
        if dl <= time.time() + 0.25:
            continue
        try:
            yp, vp = _lp_polish(xa, dl, r0=0.012, rmax=0.04, cap=700)
        except Exception:
            continue
        lp_used += 1
        if vp < best - 1e-12:
            x_best, best = yp, vp
            try:
                msk2, _, _ = _contact_mask(x_best, best, tol=1e-6)
                sig_masks.append(msk2)
                if len(sig_masks) > 60:
                    sig_masks.pop(0)
            except Exception:
                pass
    # --- reserve 15s final polish ---
    rem = T_HARD - time.time()
    if rem > 2.5:
        try:
            xb, vb = _lp_polish(x_best, time.time() + rem - 0.7, r0=0.004, rmax=0.025, cap=800)
            if vb < best and np.isfinite(vb):
                x_best, best = xb, vb
        except Exception:
            pass
    x_best = _norm(x_best)
    return x_best
