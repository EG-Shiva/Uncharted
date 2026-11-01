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

def _lex_better(a, b, tol=1e-12):
    for x, y in zip(a, b):
        if x < y - tol:
            return True
        if x > y + tol:
            return False
    return False

def _lp_eps(X0, eps_active, radius_init, max_it, deadline, cap_p=1200):
    X = _norm_rows(X0.copy())
    n, d = X.shape
    iu = np.triu_indices(n, 1)
    G0 = X @ X.T
    iu0 = G0[iu]
    best = float(iu0.max())
    bestX = X.copy()
    radius = float(radius_init)
    nv = n * d + 1
    N = n * d
    for _ in range(max_it):
        if time.time() > deadline:
            break
        if radius < 1e-13:
            break
        G = X @ X.T
        cuv = G[iu]
        m = float(cuv.max())
        sel = cuv > (m - eps_active)
        cnt = int(np.count_nonzero(sel))
        if cnt == 0:
            break
        I = iu[0][sel]; J = iu[1][sel]; cs = cuv[sel]
        if cnt > cap_p:
            kk = np.argpartition(-cs, cap_p)[:cap_p]
            I = I[kk]; J = J[kk]; cs = cs[kk]
            cnt = cap_p
        p = cnt
        cols_I = (I[:, None] * d + np.arange(d)).ravel()
        cols_J = (J[:, None] * d + np.arange(d)).ravel()
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.empty(p * (2 * d + 1), dtype=np.int64)
        cols[0::(2*d+1)] = 0  # placeholder, will overwrite vectorised below
        # build cols vectorised
        cmat = np.empty((p, 2 * d + 1), dtype=np.int64)
        cmat[:, :d] = I[:, None] * d + np.arange(d)
        cmat[:, d:2*d] = J[:, None] * d + np.arange(d)
        cmat[:, 2*d] = N
        cols = cmat.ravel()
        vmat = np.empty((p, 2 * d + 1), dtype=float)
        vmat[:, :d] = X[J]
        vmat[:, d:2*d] = X[I]
        vmat[:, 2*d] = -1.0
        vals = vmat.ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((X.ravel(), (np.repeat(np.arange(n), d), np.arange(N))), shape=(n, nv)).tocsr()
        except Exception:
            radius *= 0.5
            continue
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-cs, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * N + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        dx = res.x[:-1].reshape(n, d)
        best_a = None; best_v = 1e100
        for alpha in (1.0, 0.6, 0.3, 0.15):
            Y = _norm_rows(X + alpha * dx)
            if not np.all(np.isfinite(Y)):
                continue
            v = _max_cos(Y)
            if v < best_v:
                best_v = v; best_a = alpha
        if best_a is None:
            radius *= 0.5
            continue
        Y = _norm_rows(X + best_a * dx)
        if best_v < m - 1e-14:
            X = Y
            if best_v < best:
                best = best_v; bestX = Y.copy()
            radius = min(radius * 1.4, 0.02)
        else:
            radius *= 0.5
            if best_v < best - 1e-14:
                best = best_v; bestX = Y.copy(); X = Y
            if radius < 3e-6:
                break
    return bestX, best

def _rigidity(X, I, J, cap_p=1000):
    n, d = X.shape
    N = n * d
    p = len(I)
    if p > cap_p:
        cs = (X @ X.T)[I, J]
        kk = np.argpartition(-cs, cap_p)[:cap_p]
        I = I[kk]; J = J[kk]; p = cap_p
    M1 = d - 1
    M = n * M1
    B = np.zeros((N, M), dtype=float)
    for i in range(n):
        try:
            _, _, Vh = np.linalg.svd(X[i][None, :], full_matrices=True)
        except Exception:
            Q, _ = np.linalg.qr(np.random.randn(d, d))
            Vh = Q.T
        B[i*d:(i+1)*d, i*M1:(i+1)*M1] = Vh[1:, :].T
    K = d * (d - 1) // 2
    R = np.zeros((N, K), dtype=float)
    k = 0
    for a in range(d):
        for b in range(a+1, d):
            for i in range(n):
                xi = X[i]
                v = np.zeros(d)
                v[a] = xi[b]; v[b] = -xi[a]
                R[i*d:(i+1)*d, k] = v
            k += 1
    try:
        C = B.T @ R
        Qf, _ = np.linalg.qr(C, mode='complete')
        Q0 = Qf[:, K:]
    except Exception:
        return None
    dim = Q0.shape[1]
    try:
        Jm = np.zeros((p, N), dtype=float)
        for t in range(p):
            ii = int(I[t]); jj = int(J[t])
            Jm[t, ii*d:(ii+1)*d] = X[jj]
            Jm[t, jj*d:(jj+1)*d] = X[ii]
        JB = Jm @ B
        Jr = JB @ Q0
    except Exception:
        return None
    try:
        if p >= dim:
            U, S, Vt = np.linalg.svd(Jr, full_matrices=False)
        else:
            U, S, Vt = np.linalg.svd(Jr, full_matrices=True)
            # Vt is dim x dim in full mode
    except Exception:
        return None
    rank = int(np.sum(S > 1e-6))
    return {"B": B, "Q0": Q0, "Vt": Vt, "S": S, "rank": rank, "dim": dim, "p": p, "I": I, "J": J}

def _null_kick(X, rig, rng, n_combine=20, scale=3e-3):
    B = rig["B"]; Q0 = rig["Q0"]; Vt = rig["Vt"]; dim = rig["dim"]
    K = min(int(n_combine), dim)
    idx = np.arange(dim - K, dim)
    coeff = rng.standard_normal(K)
    v = np.zeros(dim)
    for t, ii in enumerate(idx):
        v += coeff[t] * Vt[ii, :]
    dx = (B @ (Q0 @ v)).reshape(X.shape)
    rn = float(np.linalg.norm(dx, axis=1).max()) + 1e-30
    dx *= (scale / rn)
    return dx

def _quench(X_init, steps, T0, T1, lr, mu, deadline):
    X = X_init.copy()
    n, d = X.shape
    V = np.zeros_like(X)
    lt0 = np.log(max(T0, 1e-12)); lt1 = np.log(max(T1, 1e-12))
    G0 = X @ X.T; np.fill_diagonal(G0, -2.0)
    bm = float(G0.max()); bX = X.copy()
    for s in range(steps):
        if (s % 100 == 0) and time.time() > deadline:
            break
        frac = s / max(1, steps - 1)
        T = float(np.exp(lt0 + (lt1 - lt0) * frac))
        G = X @ X.T; np.fill_diagonal(G, -1e100)
        m = float(G.max())
        if m < bm:
            bm = m; bX = X.copy()
        Q = np.clip((G - m) / T, -50, 0)
        E = np.exp(Q); np.fill_diagonal(E, 0.0)
        rs = E.sum(axis=1, keepdims=True); rs[rs == 0] = 1.0
        Gr = (E @ X) / rs
        Gt = Gr - np.sum(Gr * X, axis=1, keepdims=True) * X
        gn = np.linalg.norm(Gt, axis=1, keepdims=True) + 1e-12
        bad = gn[:, 0] > 2.0
        if np.any(bad):
            Gt[bad] *= (2.0 / gn[bad])
        V = mu * V + Gt
        X = _norm_rows(X - lr * V)
    Gf = bX @ bX.T; np.fill_diagonal(Gf, -2.0)
    return bX

def solve():
    t0 = time.time()
    HARD = t0 + 174.0
    HOP = HARD - 12.0
    rng = np.random.default_rng(0)
    n, d = 71, 12
    try:
        bx = np.load("best.npy").astype(float)
    except Exception:
        bx = None
    if bx is None or bx.shape != (71, 12) or not np.all(np.isfinite(bx)):
        try:
            bx = np.load("record.npy").astype(float)
        except Exception:
            bx = rng.standard_normal((n, d))
    bx = _norm_rows(np.asarray(bx, dtype=float))
    try:
        rc = _norm_rows(np.load("record.npy").astype(float))
        if _max_cos(rc) < _max_cos(bx):
            bx = rc.copy()
    except Exception:
        pass
    iu = np.triu_indices(n, 1)
    best_top, best_score = _topk(bx, 50)
    bestX = bx.copy()
    curX = bx.copy()
    cur_top, cur_score = best_top.copy(), best_score
    eps_sched = [1e-4, 1e-5, 1e-6, 5e-8]
    for eps in eps_sched:
        if time.time() > HOP - 2:
            break
        curX, _ = _lp_eps(curX, eps, 4e-3 if eps >= 1e-5 else 1e-3, 5, HOP)
        nt, ns = _topk(curX, 50)
        if _lex_better(nt, cur_top) or ns < cur_score - 1e-14:
            cur_top, cur_score = nt, ns
            if _lex_better(nt, best_top):
                best_top, best_score, bestX = nt.copy(), ns, curX.copy()
    it = 0
    while time.time() < HOP:
        it += 1
        if HOP - time.time() < 3.0:
            break
        G = curX @ curX.T
        cuv = G[iu]
        m = float(cuv.max())
        sel = None
        for thr in (5e-8, 3e-7, 1e-6, 5e-6, 1e-5):
            s = cuv > (m - thr)
            if int(np.count_nonzero(s)) >= 300:
                sel = s
                break
        if sel is None:
            sel = cuv > (m - 5e-5)
        Iact = iu[0][sel]; Jact = iu[1][sel]
        rig = _rigidity(curX, Iact, Jact, cap_p=950)
        # exploit: short polish
        tmp = curX.copy()
        for eps in eps_sched:
            if time.time() > HOP:
                break
            tmp, _ = _lp_eps(tmp, eps, 2e-3 if eps >= 1e-5 else 7e-4, 3, HOP)
        nt, ns = _topk(tmp, 50)
        if _lex_better(nt, cur_top):
            curX, cur_top, cur_score = tmp, nt, ns
            if _lex_better(nt, best_top):
                best_top, best_score, bestX = nt.copy(), ns, tmp.copy()
            continue
        # explore: null kicks
        improved = False
        if rig is not None:
            nk = 4 if (HOP - time.time()) > 40 else 2
            scales = [2e-3, 5e-3, 9e-3, 1.5e-3][:nk]
            for kk in range(nk):
                if time.time() > HOP:
                    break
                nc = int(rng.integers(8, 30))
                dx = _null_kick(curX, rig, rng, n_combine=nc, scale=float(scales[kk]))
                cand = _norm_rows(curX + dx)
                # mix quench for larger kicks to change topology
                if kk == 2:
                    cand = _quench(cand, 250, 0.004, 0.0004, 0.12, 0.85, HOP)
                cand, _ = _lp_eps(cand, 1e-5, 1.2e-3, 2, HOP)
                cand, _ = _lp_eps(cand, 1e-6, 6e-4, 2, HOP)
                nt2, ns2 = _topk(cand, 50)
                if _lex_better(nt2, cur_top):
                    curX, cur_top, cur_score = cand, nt2, ns2
                    if _lex_better(nt2, best_top):
                        best_top, best_score, bestX = nt2.copy(), ns2, cand.copy()
                    improved = True
                    break
            if improved:
                continue
        # topology change: separate current max pair + jitter rattlers
        flat = int(np.argmax(cuv))
        a = int(iu[0][flat]); b = int(iu[1][flat])
        xa = curX[a]; xb = curX[b]
        mid = xa + xb
        nm = float(np.linalg.norm(mid))
        if nm < 1e-12:
            mid = rng.standard_normal(d); nm = float(np.linalg.norm(mid))
        mid = mid / nm
        da = xa - float(xa @ mid) * mid
        nd = float(np.linalg.norm(da))
        Y = curX.copy()
        if nd > 1e-12:
            e = da / nd
            step = float(rng.uniform(2e-3, 8e-3))
            Y[a] = _norm_rows((xa + step * e)[None, :])[0]
            Y[b] = _norm_rows((xb - step * e)[None, :])[0]
        # jitter rattlers (contacts at 5e-8)
        sel5 = cuv > (m - 5e-8)
        cntp = np.zeros(n, dtype=int)
        np.add.at(cntp, iu[0][sel5], 1)
        np.add.at(cntp, iu[1][sel5], 1)
        rats = np.where(cntp == 0)[0]
        if len(rats) > 0:
            jr = rats[rng.integers(0, len(rats), size=min(6, len(rats)))]
            jit = rng.standard_normal((len(jr), d)) * 4e-3
            jit -= np.sum(jit * Y[jr], axis=1, keepdims=True) * Y[jr]
            Y[jr] = _norm_rows(Y[jr] + jit)
        else:
            jr = rng.integers(0, n, size=4)
            jit = rng.standard_normal((4, d)) * 3e-3
            jit -= np.sum(jit * Y[jr], axis=1, keepdims=True) * Y[jr]
            Y[jr] = _norm_rows(Y[jr] + jit)
        Y, _ = _lp_eps(Y, 1e-5, 1e-3, 2, HOP)
        Y, _ = _lp_eps(Y, 5e-8, 5e-4, 2, HOP)
        nt3, ns3 = _topk(Y, 50)
        if _lex_better(nt3, cur_top):
            curX, cur_top, cur_score = Y, nt3, ns3
            if _lex_better(nt3, best_top):
                best_top, best_score, bestX = nt3.copy(), ns3, Y.copy()
            continue
        # hole reinsertion every few iters: replace worst point
        if it % 3 == 0 and time.time() < HOP - 5:
            Gm = curX @ curX.T
            np.fill_diagonal(Gm, -2.0)
            rm = Gm.max(axis=1)
            w = int(np.argmax(rm))
            R = np.delete(curX, w, axis=0)
            Z = _norm_rows(rng.standard_normal((4000, d)))
            sc = Z @ R.T
            mx = sc.max(axis=1)
            bi = int(np.argmin(mx))
            s0 = Z[bi]
            # refine hole
            s = s0 / np.linalg.norm(s0)
            v = np.zeros_like(s)
            for _ in range(150):
                g = R @ s
                mm = float(g.max())
                ee = np.exp(np.clip((g - mm) / 0.002, -50, 0))
                Gr2 = (R.T @ ee) / (ee.sum() + 1e-30)
                Gt2 = Gr2 - float(Gr2 @ s) * s
                v = 0.8 * v + Gt2
                s = s - 0.08 * v
                s = s / (np.linalg.norm(s) + 1e-30)
            Y2 = curX.copy(); Y2[w] = s
            Y2, _ = _lp_eps(Y2, 1e-5, 1e-3, 2, HOP)
            nt4, ns4 = _topk(Y2, 50)
            if _lex_better(nt4, cur_top):
                curX, cur_top, cur_score = Y2, nt4, ns4
                if _lex_better(nt4, best_top):
                    best_top, best_score, bestX = nt4.copy(), ns4, Y2.copy()
                continue
        # random restart perturbation if fully stuck
        if not improved and (it % 2 == 0):
            dx = rng.standard_normal((n, d)) * 2e-3
            dx -= curX * np.sum(dx * curX, axis=1, keepdims=True)
            cand = _norm_rows(curX + dx)
            cand = _quench(cand, 300, 0.005, 0.0005, 0.14, 0.88, HOP)
            cand, _ = _lp_eps(cand, 1e-5, 1e-3, 2, HOP)
            nt5, ns5 = _topk(cand, 50)
            if _lex_better(nt5, cur_top):
                curX, cur_top, cur_score = cand, nt5, ns5
                if _lex_better(nt5, best_top):
                    best_top, best_score, bestX = nt5.copy(), ns5, cand.copy()
    # final tight polish of best
    for eps in (1e-5, 1e-6, 5e-8):
        if time.time() > HARD - 2:
            break
        bestX, _ = _lp_eps(bestX, eps, 8e-4, 4, HARD - 1)
    bestX = _norm_rows(bestX)
    return bestX
