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

def _contact_set(x, thr=1e-4):
    n = x.shape[0]
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    m = float(np.max(G))
    iu0, iu1 = np.triu_indices(n, 1)
    c = G[iu0, iu1]
    sel = np.where(c > m - thr)[0]
    return set(sel.tolist()), m, G

def _soft_modes(x, m, delta=1e-6):
    # build rigidity matrix for pairs within delta of max, return 5 soft non-rotational modes
    n, d = x.shape
    N = n * d
    G = x @ x.T
    iu0, iu1 = np.triu_indices(n, 1)
    c = G[iu0, iu1]
    # adaptive threshold to get 200..800 pairs
    sel = np.where(c > m - delta)[0]
    if sel.size < 200:
        # expand to top 400
        k = min(400, c.size)
        part = np.argpartition(c, -k)[-k:]
        # keep only those > m-5e-4 to avoid far pairs
        part = part[c[part] > m - 5e-4]
        if part.size >= 100:
            sel = part
    if sel.size > 800:
        part = np.argpartition(c[sel], -800)[-800:]
        sel = sel[part]
    p = sel.size
    if p < 50:
        return None
    I = iu0[sel]
    J = iu1[sel]
    # Build A : p x N dense (p~400, N=1242 -> 0.5M floats fine)
    A = np.zeros((p, N), dtype=np.float64)
    # vectorized fill: for each k, block I gets x[J], block J gets x[I]
    # use loop over p but with numpy assignment (p<=800, fast enough)
    # faster: advanced indexing
    # row indices
    rr = np.arange(p)
    # we do block assignment via loops in numpy: for d-dim use broadcasting
    # A[rr, I*d + k] = x[J,k]
    for k in range(d):
        A[rr, I * d + k] = x[J, k]
        A[rr, J * d + k] = x[I, k]
    M = A.T @ A  # N x N
    # penalize normal motions: Nmat = block diag(x_i x_i^T)
    # K = M + lam * Nmat  (lam large forces tangent)
    lam = 1000.0
    # add lam*outer to each diagonal block efficiently
    for i in range(n):
        xi = x[i]
        blk = lam * np.outer(xi, xi)
        r = slice(i * d, (i + 1) * d)
        M[r, r] += blk
    # eigh
    try:
        w, V = np.linalg.eigh(M)
    except Exception:
        return None
    # first ~d*(d-1)/2=253 are global rotations (near zero)
    # find number of near-zero
    # threshold 1e-7
    nzero = int(np.sum(w < 1e-7))
    # clamp
    start = nzero
    # if jammed estimate wrong, force start ~253
    if start < 200:
        start = 200
    if start + 5 >= N:
        start = N - 6
    # take 5 modes after start
    modes = V[:, start:start + 5]  # N x 5
    # each column norm 1, reshape to (5,n,d)
    out = np.empty((5, n, d), dtype=np.float64)
    for k in range(5):
        out[k] = modes[:, k].reshape(n, d)
    return out

def _lse_adam(x0, steps, k0, k1, lr0, lr1, deadline):
    x = x0
    n, d = x.shape
    m1 = np.zeros_like(x)
    m2 = np.zeros_like(x)
    b1 = 0.9
    b2 = 0.999
    eps = 1e-8
    lk0 = np.log(k0)
    lk1 = np.log(k1)
    ll0 = np.log(lr0)
    ll1 = np.log(lr1)
    for s in range(steps):
        if (s & 63) == 0 and time.time() > deadline:
            break
        f = s / max(steps - 1, 1)
        k = np.exp(lk0 + (lk1 - lk0) * f)
        lr = np.exp(ll0 + (ll1 - ll0) * f)
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        mx = float(np.max(G))
        # weights
        # clip argument to avoid overflow (already <=0)
        W = np.exp(k * (G - mx))
        np.fill_diagonal(W, 0.0)
        S = float(np.sum(W) * 0.5) + 1e-12
        F = (W @ x) / S
        # tangent projection
        dot = np.sum(F * x, axis=1, keepdims=True)
        Ft = F - dot * x
        # adam
        m1 = b1 * m1 + (1.0 - b1) * Ft
        m2 = b2 * m2 + (1.0 - b2) * (Ft * Ft)
        c1 = 1.0 - b1 ** (s + 1)
        c2 = 1.0 - b2 ** (s + 1)
        m1h = m1 / c1
        m2h = m2 / c2
        step = lr * m1h / (np.sqrt(m2h) + eps)
        x = x - step
        # renormalize
        nr = np.linalg.norm(x, axis=1, keepdims=True)
        nr[nr == 0] = 1.0
        x = x / nr
    return x

def _lp_polish_fast(x, deadline, r0=0.008, rmax=0.03):
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
        m = float(np.max(c_all))
        thresh = m - max(20.0 * radius, 1e-9)
        sel = np.where(c_all > thresh)[0]
        if sel.size > 1500:
            part = np.argpartition(c_all[sel], -1500)[-1500:]
            sel = sel[part]
        if sel.size == 0:
            break
        c_sel = c_all[sel]
        I = iu0[sel]
        J = iu1[sel]
        p = sel.size
        nv = n * d + 1
        # build A_ub vectorized
        cols_mat = np.empty((p, 2 * d + 1), dtype=np.int64)
        cols_mat[:, :d] = I[:, None] * d + ar[None, :]
        cols_mat[:, d:2 * d] = J[:, None] * d + ar[None, :]
        cols_mat[:, 2 * d] = n * d
        vals_mat = np.empty((p, 2 * d + 1), dtype=np.float64)
        vals_mat[:, :d] = x[J]
        vals_mat[:, d:2 * d] = x[I]
        vals_mat[:, 2 * d] = -1.0
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = cols_mat.ravel()
        vals = vals_mat.ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            erows = np.repeat(np.arange(n), d)
            ecols = np.arange(n * d)
            A_eq = coo_matrix((x.ravel(), (erows, ecols)), shape=(n, nv)).tocsr()
            obj = np.zeros(nv)
            obj[-1] = 1.0
            bounds = [(-radius, radius)] * (n * d) + [(None, None)]
            res = linprog(obj, A_ub=A_ub, b_ub=-c_sel, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=bounds, method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
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

def solve():
    t0 = time.time()
    T_EXPLORE = t0 + 160.0
    T_HARD = t0 + 176.0
    rng = np.random.default_rng(0)
    try:
        x0 = np.load("best.npy").astype(float)
    except Exception:
        x0 = None
    try:
        xr = np.load("record.npy").astype(float)
    except Exception:
        xr = None
    cands = []
    if x0 is not None:
        cands.append(x0)
    if xr is not None:
        cands.append(xr)
    if len(cands) == 0:
        tmp = rng.standard_normal((54, 23))
        cands.append(tmp)
    best = 1e9
    x_best = None
    for c in cands:
        try:
            cc = _norm(np.array(c, dtype=float))
            if cc.shape != (54, 23):
                continue
            v = _max_cos(cc)
            if v < best:
                best = v
                x_best = cc.copy()
        except Exception:
            continue
    if x_best is None:
        tmp = rng.standard_normal((54, 23))
        x_best = _norm(tmp)
        best = _max_cos(x_best)
    n, d = (54, 23)
    # seen contact graphs for clustering
    seen = []
    s0, m0, _ = _contact_set(x_best, thr=1e-4)
    seen.append(s0)
    # screening threshold for LP: only if below best+0.002
    lp_used = 0
    MAX_LP = 5
    # quick tiny initial Adam touch (optional, fast)
    # main loop: modal kicks
    outer = 0
    while time.time() < T_EXPLORE - 1.0:
        outer += 1
        # recompute modes from current best
        try:
            modes = _soft_modes(x_best, best, delta=1e-6)
        except Exception:
            modes = None
        if modes is None:
            # fallback: random tangent jitter + anneal
            modes = rng.standard_normal((5, n, d)) * 0.1
            # project to tangent
            for k in range(5):
                dot = np.sum(modes[k] * x_best, axis=1, keepdims=True)
                modes[k] -= dot * x_best
                nn = np.linalg.norm(modes[k]) + 1e-12
                modes[k] /= nn
        # try several random combos per mode set to amortize eig cost
        ntry = 4
        for tt in range(ntry):
            if time.time() > T_EXPLORE - 1.0:
                break
            amp = 0.03 + 0.27 * float(rng.random())
            coeff = rng.standard_normal(5)
            cn = float(np.linalg.norm(coeff)) + 1e-12
            coeff = coeff / cn * amp
            dx = np.tensordot(coeff, modes, axes=(0, 0))  # n x d
            # dx is tangent approx; add to x_best
            y = x_best + dx
            y = _norm(y)
            # anneal 300-600 steps
            steps = int(300 + rng.integers(0, 301))  # 300..600
            # randomize anneal hyperparams slightly
            k0 = 80.0
            k1 = 2000.0 + 1000.0 * float(rng.random())
            lr0 = 0.003
            lr1 = 0.0003
            deadline_a = min(time.time() + 1.5, T_EXPLORE)
            try:
                xa = _lse_adam(y, steps, k0, k1, lr0, lr1, deadline_a)
            except Exception:
                continue
            try:
                va = _max_cos(xa)
            except Exception:
                continue
            # screen
            if va < best - 1e-12:
                # check novelty? accept anyway
                x_best = xa.copy()
                best = va
                s, _, _ = _contact_set(x_best, thr=1e-4)
                seen.append(s)
                # early continue to recompute modes
                break
            # clustering: compute graph distance to seen
            try:
                s_new, _, _ = _contact_set(xa, thr=1e-4)
            except Exception:
                continue
            # min symmetric difference
            mind = 10 ** 9
            for s_old in seen[-20:]:
                # |sym diff|
                # fast approx: |union|-|inter| ... use set ops (size ~ few hundred)
                inter = len(s_new.intersection(s_old))
                diff = len(s_new) + len(s_old) - 2 * inter
                if diff < mind:
                    mind = diff
                if mind < 20:
                    break
            if mind < 20:
                # return to incumbent basin, discard
                continue
            # novel but not better: consider LP only if below screening threshold
            if va < best + 0.002 and lp_used < MAX_LP and time.time() < T_EXPLORE - 2.5:
                # preset screening: must be reasonably good
                lp_used += 1
                deadline_lp = min(time.time() + 2.0, T_EXPLORE)
                try:
                    yp, vp = _lp_polish_fast(xa, deadline_lp, r0=0.008, rmax=0.03)
                except Exception:
                    seen.append(s_new)
                    continue
                # add to seen regardless
                try:
                    s2, _, _ = _contact_set(yp, thr=1e-4)
                    seen.append(s2)
                except Exception:
                    pass
                if vp < best - 1e-12:
                    x_best = yp.copy()
                    best = vp
                    break
                # else keep seen to avoid repeat
            else:
                # record to avoid repeat loops (limit size)
                if len(seen) < 200:
                    seen.append(s_new)
            # time guard
            if time.time() > T_HARD - 16:
                break
        # occasional time check for hard deadline
        if time.time() > T_HARD - 16:
            break
    # final polish reserve ~15s
    rem = T_HARD - time.time()
    if rem > 2.0:
        # use up to 14s for final polish
        fdead = time.time() + min(rem - 0.5, 14.0)
        try:
            xb, vb = _lp_polish_fast(x_best, fdead, r0=0.004, rmax=0.025)
            if vb < best:
                x_best = xb
                best = vb
        except Exception:
            pass
    x_best = _norm(x_best)
    return x_best
