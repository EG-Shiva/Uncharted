import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _proj(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return X / n

def _lp_polish(X, seconds, t_end=1e18):
    X = _proj(np.asarray(X, dtype=np.float64))
    n, d = X.shape
    best = _max_cos(X)
    radius = 1e-3
    start = time.time()
    iu0 = np.triu_indices(n, 1)
    while time.time() - start < seconds and time.time() < t_end:
        if radius < 1e-14:
            break
        G = X @ X.T
        cfull = G[iu0]
        m = float(cfull.max())
        sel_thr = m - max(20 * radius, 1e-9)
        sel = cfull > sel_thr
        I = iu0[0][sel]; J = iu0[1][sel]
        cs = cfull[sel]
        p = len(I)
        if p == 0:
            break
        if p > 1500:
            idx = np.argpartition(-cs, 1500)[:1500]
            I, J, cs = I[idx], J[idx], cs[idx]
            p = 1500
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([X[J], X[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((X.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv); obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-cs, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = X + res.x[:-1].reshape(n, d)
        y = _proj(y)
        v = _max_cos(y)
        if v < best:
            X, best = y, v
            radius = min(radius * 1.5, 0.02)
        else:
            radius *= 0.5
    return X

def _tangent_bases(X):
    n, d = X.shape
    Ts = []
    for i in range(n):
        x = X[i]
        k = int(np.argmin(np.abs(x)))
        e = np.zeros(d); e[k] = 1.0
        w = e - x
        nw = np.linalg.norm(w)
        if nw < 1e-12:
            # x approx e_k, use any orthogonal basis via QR fallback
            M = np.eye(d)
            M[:, k] = x
            Q, _ = np.linalg.qr(M)
            # find column parallel to x
            dots = np.abs(Q.T @ x)
            jc = int(np.argmax(dots))
            keep = [c for c in range(d) if c != jc]
            # fix sign so Q[:,jc] ~ x (not needed)
            Ts.append(Q[:, keep])
            continue
        w = w / nw
        Q = np.eye(d) - 2.0 * np.outer(w, w)
        keep = [c for c in range(d) if c != k]
        Ts.append(Q[:, keep])
    return Ts

def _build_modes(X, thresh=1e-3, k=20):
    n, d = X.shape
    X = _proj(X)
    G = X @ X.T
    iu = np.triu_indices(n, 1)
    c = G[iu]
    m = float(c.max())
    sel = c > m - thresh
    I = iu[0][sel]; J = iu[1][sel]
    cs = c[sel]
    p = len(I)
    if p < 10:
        # widen threshold
        sel = c > m - 5e-3
        I = iu[0][sel]; J = iu[1][sel]; cs = c[sel]; p = len(I)
    if p > 2000:
        idx = np.argpartition(-cs, 2000)[:2000]
        I, J = I[idx], J[idx]
        p = 2000
    if p < 5:
        return None, None, None, m, None
    Ts = _tangent_bases(X)
    dm = d - 1
    N = n * dm
    A = np.zeros((p, N), dtype=np.float64)
    # Ti^T x_j blocks
    for t in range(p):
        i = int(I[t]); j = int(J[t])
        Ti = Ts[i]; Tj = Ts[j]
        A[t, i * dm:(i + 1) * dm] = Ti.T @ X[j]
        A[t, j * dm:(j + 1) * dm] = Tj.T @ X[i]
    # dense SVD, smallest vectors are last rows of Vt
    try:
        _, S, Vt = np.linalg.svd(A, full_matrices=False)
    except Exception:
        return None, None, None, m, None
    kk = min(k, Vt.shape[0])
    Vsmall = Vt[-kk:, :]  # kk x N
    b0 = (c > m - 1e-3)
    return Vsmall, Ts, (I, J), m, b0

def _apply_u(X, Ts, u, theta):
    n, d = X.shape
    dm = d - 1
    D = np.zeros((n, d), dtype=np.float64)
    norms = np.zeros(n)
    for i in range(n):
        ui = u[i * dm:(i + 1) * dm]
        di = Ts[i] @ ui
        D[i] = di
        norms[i] = np.linalg.norm(di)
    mx = norms.max()
    if mx < 1e-14:
        return _proj(X)
    scale = np.tan(theta) / mx if mx > 1e-14 else 0.0
    # cap huge tan for safety (theta<=0.5 so tan<0.55)
    D = D * scale
    norms = norms * scale
    Y = np.empty_like(X)
    for i in range(n):
        ni = norms[i]
        if ni < 1e-14:
            Y[i] = X[i]
        else:
            Y[i] = np.cos(ni) * X[i] + (np.sin(ni) / ni) * D[i]
    return _proj(Y)

def solve():
    t0 = time.time()
    T_END = t0 + 175.0
    EXPLORE_END = t0 + 130.0
    FINAL_RESERVE = 20.0
    rng = np.random.RandomState(0)
    try:
        xb = np.load("best.npy")
    except Exception:
        try:
            xb = np.load("record.npy")
        except Exception:
            xb = rng.randn(54, 21)
    xb = np.asarray(xb, dtype=np.float64)
    if xb.shape != (54, 21):
        flat = xb.ravel()
        need = 54 * 21
        if flat.size < need:
            extra = rng.randn(need - flat.size)
            flat = np.concatenate([flat, extra])
        xb = flat[:need].reshape(54, 21)
    xb = _proj(xb)
    best = xb.copy()
    best_val = _max_cos(best)
    cur = best.copy()
    cur_val = best_val
    thetas = [0.05, 0.1, 0.2, 0.35, 0.5]
    outer = 0
    # quick initial mode computation; if it fails just do short polish
    while time.time() < EXPLORE_END and time.time() < T_END - FINAL_RESERVE - 5:
        outer += 1
        if time.time() > T_END - FINAL_RESERVE - 8:
            break
        res = _build_modes(cur, thresh=1e-3, k=20)
        Vsmall, Ts, IJ, m0, b0 = res
        if Vsmall is None or Ts is None:
            # random small shake to escape degeneracy
            cur = _proj(cur + 0.05 * rng.randn(*cur.shape))
            cur_val = _max_cos(cur)
            continue
        kk = Vsmall.shape[0]
        n, d = cur.shape
        iu = np.triu_indices(n, 1)
        # incumbent mask b0 length 1431
        # generate random combos
        ncombo = 36
        cand_X = []
        cand_max = []
        cand_ham = []
        for ci in range(ncombo):
            if time.time() > EXPLORE_END - 1:
                break
            k2 = int(rng.choice([2, 3, 4, 5]))
            k2 = min(k2, kk)
            idx = rng.choice(kk, size=k2, replace=False)
            coeff = rng.randn(k2)
            u = coeff @ Vsmall[idx, :]  # N
            # normalize global norm to 1 for stability (max-rot scaling handles rest)
            nu = np.linalg.norm(u)
            if nu > 1e-12:
                u = u / nu
            for th in thetas:
                Y = _apply_u(cur, Ts, u, th)
                G = Y @ Y.T
                cf = G[iu]
                vm = float(cf.max())
                # hamming vs incumbent thresholded graph
                b = (cf > vm - 1e-3)
                # b0 may be None? ensure
                ham = int(np.sum(b != b0)) if b0 is not None else 0
                cand_X.append(Y)
                cand_max.append(vm)
                cand_ham.append(ham)
        if len(cand_X) == 0:
            continue
        cand_max = np.array(cand_max)
        cand_ham = np.array(cand_ham)
        order = np.argsort(cand_max)
        # keep trials with max<0.12 and materially different topology
        picks = []
        for o in order:
            if cand_max[o] < 0.12 and cand_ham[o] > 60:
                picks.append(o)
            if len(picks) >= 12:
                break
        if len(picks) == 0:
            # take best few by max regardless for sideways move attempt
            for o in order[:4]:
                if cand_max[o] < 0.12:
                    picks.append(o)
        if len(picks) == 0:
            # all bad, shake cur slightly along a random mode
            u = rng.randn(Vsmall.shape[1]); u /= np.linalg.norm(u) + 1e-12
            cur = _apply_u(cur, Ts, u * 0.2, 0.2)
            cur_val = _max_cos(cur)
            continue
        # limit polishes per outer to save time (at most 4)
        # prioritize lowest max
        picks = picks[:4]
        improved = False
        best_pol_in_round = None
        best_pol_val = 1e9
        best_pol_ham = 0
        for pi in picks:
            if time.time() > EXPLORE_END and not improved:
                break
            if time.time() > T_END - FINAL_RESERVE - 5:
                break
            Y0 = cand_X[pi]
            # short LP polish 3s
            Yp = _lp_polish(Y0, 3.0, t_end=T_END - FINAL_RESERVE)
            vp = _max_cos(Yp)
            if vp < best_pol_val:
                best_pol_val = vp
                best_pol_in_round = Yp
            if vp < best_val - 1e-12:
                best = Yp
                best_val = vp
                cur = Yp.copy()
                cur_val = vp
                improved = True
                break
        if improved:
            continue
        # no global improvement: sideways move to most diverse good candidate
        if best_pol_in_round is not None and best_pol_val < 0.12:
            # move cur to best polished in round to change basin (even if worse than cur)
            # only move if it is topologically different from cur
            Gc = best_pol_in_round @ best_pol_in_round.T
            cfc = Gc[iu]
            mc = float(cfc.max())
            bc = (cfc > mc - 1e-3)
            ham_c = int(np.sum(bc != b0)) if b0 is not None else 0
            if ham_c > 60:
                cur = best_pol_in_round
                cur_val = best_pol_val
            else:
                # force a larger jump along random combo
                u = rng.randn(Vsmall.shape[1]); u /= np.linalg.norm(u) + 1e-12
                cur = _apply_u(cur, Ts, u, 0.35)
                cur_val = _max_cos(cur)
                # keep best intact
        else:
            # randomize cur a bit
            u = rng.randn(Vsmall.shape[1]); u /= np.linalg.norm(u) + 1e-12
            cur = _apply_u(cur, Ts, u, 0.3)
            cur_val = _max_cos(cur)
        # safety: if cur drifts too bad (>0.15) reset to best
        if cur_val > 0.15:
            cur = best.copy()
            cur_val = best_val
    # final LP polish of best with remaining time (reserve 2s buffer)
    remain = T_END - time.time() - 2.0
    if remain > 4:
        # use at most 22s for final to respect instruction (20s reserved)
        use = min(remain, 22.0)
        best = _lp_polish(best, use, t_end=T_END - 1.0)
    best = _proj(best)
    return best
