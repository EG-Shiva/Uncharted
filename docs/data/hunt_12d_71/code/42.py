import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _norm_rows(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[~np.isfinite(n)] = 1.0
    n[n == 0] = 1.0
    return X / n

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _topk(X, k=50):
    n = X.shape[0]
    iu = np.triu_indices(n, 1)
    G = X @ X.T
    c = G[iu]
    s = np.sort(c)[::-1]
    return s[:k].copy(), float(s[0])

def _lex_better(new_top, old_top, tol=1e-12):
    for a, b in zip(new_top, old_top):
        if a < b - tol:
            return True
        if a > b + tol:
            return False
    return False

def _lex_le(new_top, old_top, tol=1e-12):
    for a, b in zip(new_top, old_top):
        if a < b - tol:
            return True
        if a > b + tol:
            return False
    return True  # equal

def _lp_trust_eps(X0, eps_active, radius_init, max_it, deadline, cap_p=1500):
    X = _norm_rows(X0.copy())
    n, d = X.shape
    iu = np.triu_indices(n, 1)
    cuv_all = (X @ X.T)[iu]
    best = float(cuv_all.max())
    bestX = X.copy()
    radius = float(radius_init)
    nv = n * d + 1
    for _ in range(max_it):
        if time.time() > deadline:
            break
        if radius < 1e-13:
            break
        G = X @ X.T
        cuv = G[iu]
        m = float(cuv.max())
        sel = cuv > (m - eps_active)
        if np.count_nonzero(sel) == 0:
            break
        I = iu[0][sel]; J = iu[1][sel]
        cs = cuv[sel]
        p = len(I)
        if p > cap_p:
            kk = np.argpartition(-cs, cap_p)[:cap_p]
            I = I[kk]; J = J[kk]; cs = cs[kk]
            p = cap_p
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([X[J], X[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((X.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        except Exception:
            radius *= 0.5
            continue
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-cs, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        dx = res.x[:-1].reshape(n, d)
        # exact-max line search on segment X+alpha*dx projected to sphere
        best_a = None
        best_v = 1e100
        for alpha in (1.0, 0.6, 0.3, 0.15, 0.07):
            Y = _norm_rows(X + alpha * dx)
            if not np.all(np.isfinite(Y)):
                continue
            v = _max_cos(Y)
            if v < best_v:
                best_v = v
                best_a = alpha
        if best_a is None:
            radius *= 0.5
            continue
        Y = _norm_rows(X + best_a * dx)
        # accept if improves current linearisation point
        if best_v < m - 1e-14:
            X = Y
            if best_v < best:
                best = best_v
                bestX = Y.copy()
            radius = min(radius * 1.4, 0.02)
        else:
            # no progress at this radius
            radius *= 0.5
            if best_v < best - 1e-14:
                best = best_v
                bestX = Y.copy()
                X = Y
            # if stuck, break early for small eps
            if radius < 3e-6:
                break
    return bestX, best

def _rigidity_null(X, I, J, cap_p=1200):
    # returns B, Q0, Vt_full, S, rank_est, contacts_info
    n, d = X.shape
    N = n * d
    p = len(I)
    if p > cap_p:
        G = X @ X.T
        cs = G[I, J]
        kk = np.argpartition(-cs, cap_p)[:cap_p]
        I = I[kk]; J = J[kk]
        p = cap_p
    # tangent basis B: N x M, M=n*(d-1)
    M1 = d - 1
    M = n * M1
    B = np.zeros((N, M), dtype=float)
    for i in range(n):
        x = X[i]
        # vh[0] ~ x, rest is tangent basis
        try:
            _, _, Vh = np.linalg.svd(x[None, :], full_matrices=True)
        except Exception:
            # fallback random
            Q, _ = np.linalg.qr(np.random.randn(d, d))
            Vh = Q.T
        basis = Vh[1:, :].T  # 12x11
        B[i * d:(i + 1) * d, i * M1:(i + 1) * M1] = basis
    # rotation subspace R: N x 66
    pairs = []
    for a in range(d):
        for b in range(a + 1, d):
            pairs.append((a, b))
    K = len(pairs)
    R = np.zeros((N, K), dtype=float)
    for k, (a, b) in enumerate(pairs):
        for i in range(n):
            xi = X[i]
            v = np.zeros(d)
            v[a] = xi[b]
            v[b] = -xi[a]
            R[i * d:(i + 1) * d, k] = v
    # C = B^T R  (M x K)
    try:
        C = B.T @ R
        Qfull, _ = np.linalg.qr(C, mode='complete')
        r = min(K, M)
        Q0 = Qfull[:, r:]
    except Exception:
        return None
    # Jacobian Jmat p x N
    try:
        Jmat = np.zeros((p, N), dtype=float)
        # vectorised
        for k in range(p):
            i = int(I[k]); j = int(J[k])
            Jmat[k, i * d:(i + 1) * d] = X[j]
            Jmat[k, j * d:(j + 1) * d] = X[i]
        JB = Jmat @ B  # p x M
        Jred = JB @ Q0  # p x (M-K)
    except Exception:
        return None
    try:
        U, S, Vt = np.linalg.svd(Jred, full_matrices=True)
    except Exception:
        return None
    # rank with tol
    rank_est = int(np.sum(S > 1e-6))
    return {"B": B, "Q0": Q0, "Vt": Vt, "S": S, "rank": rank_est, "I": I, "J": J, "dim": Q0.shape[1]}

def _null_kick(X, rig, rng, n_combine=24, scale=3e-3):
    B = rig["B"]; Q0 = rig["Q0"]; Vt = rig["Vt"]
    dim = rig["dim"]
    # bottom indices: smallest singular values; if p<dim, tail of Vt are true null
    # Vt shape dim x dim (full)
    # order: S ascending -> bottom rows. S is ascending? svd returns descending.
    # So bottom = last n_combine rows
    K = min(n_combine, dim)
    idx = np.arange(dim - K, dim)
    coeff = rng.standard_normal(K)
    # weight by inverse singular? just uniform; smaller sigma gets slightly larger weight
    v = np.zeros(dim)
    for t, ii in enumerate(idx):
        v += coeff[t] * Vt[ii, :]
    y = Q0 @ v  # M
    dxf = B @ y  # N
    n, d = X.shape
    dx = dxf.reshape(n, d)
    # scale to max row norm = scale
    rn = np.linalg.norm(dx, axis=1).max() + 1e-30
    dx *= (scale / rn)
    return dx

def solve():
    t0 = time.time()
    HARD_END = t0 + 174.0
    HOP_END = HARD_END - 12.0
    rng = np.random.default_rng(0)
    n, d = 71, 12
    best_X = None
    try:
        best_X = np.load("best.npy").astype(float)
    except Exception:
        best_X = None
    if best_X is None or best_X.shape != (71, 12) or not np.all(np.isfinite(best_X)):
        try:
            best_X = np.load("record.npy").astype(float)
        except Exception:
            best_X = rng.standard_normal((n, d))
    best_X = _norm_rows(np.asarray(best_X, dtype=float))
    try:
        rec = _norm_rows(np.load("record.npy").astype(float))
        if _max_cos(rec) < _max_cos(best_X):
            best_X = rec.copy()
    except Exception:
        pass
    best_top, best_score = _topk(best_X, 50)
    best_top = best_top.copy()
    curX = best_X.copy()
    cur_top = best_top.copy()
    cur_score = best_score
    iu = np.triu_indices(n, 1)
    eps_sched = [1e-4, 1e-5, 1e-6, 5e-8]
    # initial squeeze
    for eps in eps_sched:
        if time.time() > HOP_END - 2.0:
            break
        curX, _ = _lp_trust_eps(curX, eps, 4e-3 if eps >= 1e-5 else 1e-3, 6, HOP_END)
        nt, ns = _topk(curX, 50)
        if _lex_better(nt, cur_top) or ns < cur_score - 1e-14:
            cur_top, cur_score = nt, ns
            if _lex_better(nt, best_top):
                best_top, best_score, best_X = nt.copy(), ns, curX.copy()
    it = 0
    while time.time() < HOP_END:
        it += 1
        tleft = HOP_END - time.time()
        if tleft < 3.0:
            break
        # --- rigidity analysis on plateau set slack<5e-8 (or adapt to get ~300-800 pairs)
        G = curX @ curX.T
        cuv = G[iu]
        m = float(cuv.max())
        # contacts at 5e-8
        sel5 = cuv > (m - 5e-8)
        ncontacts = int(np.count_nonzero(sel5))
        # choose active set for Jacobian: target ~400-900 pairs
        for thr in (5e-8, 3e-7, 1e-6, 5e-6, 1e-5, 5e-5):
            sel = cuv > (m - thr)
            if np.count_nonzero(sel) >= 300:
                break
        Iact = iu[0][sel]; Jact = iu[1][sel]
        rig = _rigidity_null(curX, Iact, Jact, cap_p=1100)
        # --- trust LP polish across schedule (exploit)
        tmpX = curX.copy()
        for eps in eps_sched:
            if time.time() > HOP_END:
                break
            tmpX, _ = _lp_trust_eps(tmpX, eps, 2e-3 if eps >= 1e-5 else 7e-4, 4, HOP_END)
        nt, ns = _topk(tmpX, 50)
        if _lex_better(nt, cur_top):
            curX, cur_top, cur_score = tmpX, nt, ns
            if _lex_better(nt, best_top):
                best_top, best_score, best_X = nt.copy(), ns, tmpX.copy()
            continue
        # --- null kicks (explore different basin)
        if rig is None:
            # random tangent kick fallback
            dx = rng.standard_normal((n, d)) * 2e-3
            dx -= curX * np.sum(dx * curX, axis=1, keepdims=True)
            cand = _norm_rows(curX + dx)
            cand, _ = _lp_trust_eps(cand, 1e-5, 1e-3, 3, HOP_END)
            nt2, ns2 = _topk(cand, 50)
            if _lex_better(nt2, cur_top):
                curX, cur_top, cur_score = cand, nt2, ns2
                if _lex_better(nt2, best_top):
                    best_top, best_score, best_X = nt2.copy(), ns2, cand.copy()
            continue
        # generate several kicks along smallest singular directions
        n_kicks = 5 if tleft > 40 else 3
        scales = [1.5e-3, 3e-3, 6e-3, 1e-2, 2e-3][:n_kicks]
        improved = False
        for kk in range(n_kicks):
            if time.time() > HOP_END:
                break
            # vary combine width
            nc = int(rng.integers(8, 32))
            dx = _null_kick(curX, rig, rng, n_combine=nc, scale=float(scales[kk]))
            cand = _norm_rows(curX + dx)
            # short re-polish: wide then tight
            cand, _ = _lp_trust_eps(cand, 1e-5, 1.2e-3, 3, HOP_END)
            cand, _ = _lp_trust_eps(cand, 5e-8 if ncontacts > 0 else 1e-6, 6e-4, 3, HOP_END)
            nt2, ns2 = _topk(cand, 50)
            if _lex_better(nt2, cur_top):
                curX, cur_top, cur_score = cand, nt2, ns2
                if _lex_better(nt2, best_top):
                    best_top, best_score, best_X = nt2.copy(), ns2, cand.copy()
                improved = True
                break
        if improved:
            continue
        # --- if no null progress, try pair-apart on current max pair + re-polish (topology change)
        flat = int(np.argmax(cuv))
        a = int(iu[0][flat]); b = int(iu[1][flat])
        # direction separating a,b in tangent
        xa = curX[a]; xb = curX[b]
        mid = xa + xb
        nm = float(np.linalg.norm(mid))
        if nm < 1e-12:
            mid = rng.standard_normal(d); nm = float(np.linalg.norm(mid))
        mid = mid / nm
        da = xa - float(xa @ mid) * mid
        nd = float(np.linalg.norm(da))
        if nd > 1e-12:
            e = da / nd
            step = float(rng.uniform(2e-3, 8e-3))
            Y = curX.copy()
            Y[a] = _norm_rows((xa + step * e)[None, :])[0]
            Y[b] = _norm_rows((xb - step * e)[None, :])[0]
            # small random tangent jitter on rattlers (low contact count)
            # find high row-max indices
            Gm = curX @ curX.T
            np.fill_diagonal(Gm, -2.0)
            rmax = Gm.max(axis=1)
            order = np.argsort(-rmax)
            jr = order[:6]
            jit = rng.standard_normal((len(jr), d)) * 1e-3
            for t2, idxr in enumerate(jr):
                v = jit[t2] - float(jit[t2] @ Y[idxr]) * Y[idxr]
                Y[idxr] = Y[idxr] + v
            Y = _norm_rows(Y)
            Y, _ = _lp_trust_eps(Y, 1e-5, 1.2e-3, 3, HOP_END)
            Y, _ = _lp_trust_eps(Y, 1e-6, 7e-4, 3, HOP_END)
            nt3, ns3 = _topk(Y, 50)
            if _lex_le(nt3, cur_top):
                # accept sideways even if equal-ish to change basin, but only if not worse
                # require strict lex or equal with different fingerprint to avoid loops
                if _lex_better(nt3, cur_top):
                    curX, cur_top, cur_score = Y, nt3, ns3
                    if _lex_better(nt3, best_top):
                        best_top, best_score, best_X = nt3.copy(), ns3, Y.copy()
                else:
                    # equal: accept with small prob to diversify
                    if rng.random() < 0.25:
                        curX, cur_top, cur_score = Y, nt3, ns3
        # occasional restart from best with null kick to avoid drift
        if it % 7 == 0 and rng.random() < 0.5:
            curX, cur_top, cur_score = best_X.copy(), best_top.copy(), best_score
    # final squeeze on best
    Xf = best_X.copy()
    while time.time() < HARD_END - 1.0:
        before = _max_cos(Xf)
        Xf, _ = _lp_trust_eps(Xf, 1e-6, 6e-4, 3, HARD_END - 0.5)
        Xf, _ = _lp_trust_eps(Xf, 5e-8, 3e-4, 3, HARD_END - 0.5)
        after = _max_cos(Xf)
        if after >= before - 1e-14:
            break
        nt, ns = _topk(Xf, 50)
        best_top, best_score = nt, ns
        best_X = Xf.copy()
        if time.time() > HARD_END - 2.0:
            break
    best_X = _norm_rows(best_X)
    return best_X
