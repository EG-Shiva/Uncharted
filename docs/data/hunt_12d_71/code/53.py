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

def _lp_polish_eps(X0, eps_active, radius_init, max_it, deadline, cap_p=1200):
    X = _norm_rows(X0.copy())
    n, d = X.shape
    iu = np.triu_indices(n, 1)
    best = _max_cos(X)
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
        cnt = int(np.count_nonzero(sel))
        if cnt == 0:
            break
        I = iu[0][sel]; J = iu[1][sel]
        cs = cuv[sel]
        if cnt > cap_p:
            kk = np.argpartition(-cs, cap_p)[:cap_p]
            I = I[kk]; J = J[kk]; cs = cs[kk]
            cnt = cap_p
        rows = np.repeat(np.arange(cnt), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((cnt, 1), n * d)], 1).ravel()
        vals = np.concatenate([X[J], X[I], -np.ones((cnt, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(cnt, nv)).tocsr()
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

def _quench(X0, steps, T0, T1, lr, mu=0.9, deadline=None):
    X = _norm_rows(X0.copy())
    n, d = X.shape
    V = np.zeros_like(X)
    lt0 = np.log(max(T0, 1e-12)); lt1 = np.log(max(T1, 1e-12))
    G0 = X @ X.T; np.fill_diagonal(G0, -2.0)
    bv = float(G0.max()); bX = X.copy()
    for s in range(steps):
        if deadline is not None and (s % 200 == 0) and time.time() > deadline:
            break
        frac = s / max(1, steps - 1)
        T = float(np.exp(lt0 + (lt1 - lt0) * frac))
        G = X @ X.T
        np.fill_diagonal(G, -1e100)
        m = float(G.max())
        Q = (G - m) / T
        np.clip(Q, -50, 0, out=Q)
        E = np.exp(Q)
        np.fill_diagonal(E, 0.0)
        rs = E.sum(axis=1, keepdims=True); rs[rs == 0] = 1.0
        Grad = (E @ X) / rs
        Gt = Grad - np.sum(Grad * X, axis=1, keepdims=True) * X
        gn = np.linalg.norm(Gt, axis=1, keepdims=True) + 1e-12
        bad = (gn[:, 0] > 2.0)
        if np.any(bad):
            Gt[bad] *= (2.0 / gn[bad])
        V = mu * V + Gt
        X = _norm_rows(X - lr * V)
        if (s % 150 == 0) or s == steps - 1:
            Gc = X @ X.T; np.fill_diagonal(Gc, -2.0)
            vc = float(Gc.max())
            if vc < bv:
                bv = vc; bX = X.copy()
    return bX, bv

def _rigidity(X, I, J, cap_p=800):
    n, d = X.shape
    N = n * d
    p = len(I)
    if p > cap_p:
        cs = (X @ X.T)[I, J]
        kk = np.argpartition(-cs, cap_p)[:cap_p]
        I = I[kk]; J = J[kk]; p = cap_p
    M1 = d - 1
    M = n * M1
    B = np.zeros((N, M))
    # Householder tangent basis: H maps x -> +/-ek
    for i in range(n):
        x = X[i]
        k = int(np.argmax(np.abs(x)))
        s = 1.0 if x[k] >= 0 else -1.0
        u = x.copy(); u[k] += s
        den = float(u @ u)
        if den < 1e-24:
            # x == -s*ek, use any orthogonal basis
            for j in range(d):
                if j == k:
                    continue
                col = j if j < k else j - 1
                e = np.zeros(d); e[j] = 1.0
                B[i*d:(i+1)*d, i*M1+col] = e
            continue
        # columns of H except k: H e_j for j!=k
        for j in range(d):
            if j == k:
                continue
            col = j if j < k else j - 1
            uj = u[j]
            # H e_j = e_j - 2 u_j u / den
            v = -2.0 * uj * u / den
            v[j] += 1.0
            B[i*d:(i+1)*d, i*M1+col] = v
    pairs = [(a, b) for a in range(d) for b in range(a+1, d)]
    K = len(pairs)
    R = np.zeros((N, K))
    for kk2, (a, b) in enumerate(pairs):
        for i in range(n):
            xi = X[i]
            R[i*d+a, kk2] = xi[b]
            R[i*d+b, kk2] = -xi[a]
    try:
        C = B.T @ R
        Qf, _ = np.linalg.qr(C, mode='complete')
        r = min(K, M)
        Q0 = Qf[:, r:]
        Jmat = np.zeros((p, N))
        for t in range(p):
            ii = int(I[t]); jj = int(J[t])
            Jmat[t, ii*d:(ii+1)*d] = X[jj]
            Jmat[t, jj*d:(jj+1)*d] = X[ii]
        JB = Jmat @ B
        Jred = JB @ Q0
        U, S, Vt = np.linalg.svd(Jred, full_matrices=True)
        rank = int(np.sum(S > 1e-6))
        return {"B": B, "Q0": Q0, "Vt": Vt, "S": S, "rank": rank, "dim": Q0.shape[1], "I": I, "J": J}
    except Exception:
        return None

def _null_kick(X, rig, rng, n_combine=20, scale=3e-3):
    B = rig["B"]; Q0 = rig["Q0"]; Vt = rig["Vt"]
    dim = rig["dim"]
    K = min(n_combine, dim)
    idx = np.arange(dim - K, dim)
    coeff = rng.standard_normal(K)
    v = np.zeros(dim)
    for t, ii in enumerate(idx):
        v += coeff[t] * Vt[ii, :]
    dxf = B @ (Q0 @ v)
    n, d = X.shape
    dx = dxf.reshape(n, d)
    rn = float(np.linalg.norm(dx, axis=1).max()) + 1e-30
    dx *= (scale / rn)
    return dx

def solve():
    t0 = time.time()
    HARD_END = t0 + 174.0
    HOP_END = HARD_END - 10.0
    rng = np.random.default_rng(0)
    n, d = 71, 12
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
    iu = np.triu_indices(n, 1)
    best_top, best_score = _topk(best_X, 50)
    best_top = best_top.copy()
    curX = best_X.copy()
    cur_top, cur_score = best_top.copy(), best_score
    eps_sched = [1e-4, 1e-5, 1e-6, 5e-8]
    # initial squeeze (cheap)
    for eps in eps_sched:
        if time.time() > HOP_END - 2:
            break
        curX, _ = _lp_polish_eps(curX, eps, 4e-3 if eps >= 1e-5 else 1e-3, 4, HOP_END)
        nt, ns = _topk(curX, 50)
        if _lex_better(nt, cur_top) or ns < cur_score - 1e-14:
            cur_top, cur_score = nt, ns
            if _lex_better(nt, best_top):
                best_top, best_score, best_X = nt.copy(), ns, curX.copy()
    it = 0
    while time.time() < HOP_END:
        it += 1
        if time.time() > HOP_END - 3:
            break
        tleft = HOP_END - time.time()
        G = curX @ curX.T
        cuv = G[iu]
        m = float(cuv.max())
        # active set targeting ~300-800 pairs
        sel = None
        for thr in (5e-8, 3e-7, 1e-6, 5e-6, 1e-5):
            s = cuv > (m - thr)
            if int(np.count_nonzero(s)) >= 300:
                sel = s
                break
        if sel is None:
            sel = cuv > (m - 1e-5)
        Iact = iu[0][sel]; Jact = iu[1][sel]
        rig = _rigidity(curX, Iact, Jact, cap_p=750)
        # exploit: short LP across schedule
        tmpX = curX.copy()
        for eps in eps_sched:
            if time.time() > HOP_END:
                break
            tmpX, _ = _lp_polish_eps(tmpX, eps, 2e-3 if eps >= 1e-5 else 7e-4, 3, HOP_END)
        nt, ns = _topk(tmpX, 50)
        if _lex_better(nt, cur_top):
            curX, cur_top, cur_score = tmpX, nt, ns
            if _lex_better(nt, best_top):
                best_top, best_score, best_X = nt.copy(), ns, tmpX.copy()
            continue
        if rig is None:
            dx = rng.standard_normal((n, d)) * 2e-3
            dx -= curX * np.sum(dx * curX, axis=1, keepdims=True)
            cand = _norm_rows(curX + dx)
            cand, _ = _quench(cand, 400, 0.004, 0.0004, 0.15, deadline=HOP_END)
            cand, _ = _lp_polish_eps(cand, 1e-6, 8e-4, 2, HOP_END)
            nt2, ns2 = _topk(cand, 50)
            if _lex_better(nt2, cur_top):
                curX, cur_top, cur_score = cand, nt2, ns2
                if _lex_better(nt2, best_top):
                    best_top, best_score, best_X = nt2.copy(), ns2, cand.copy()
            continue
        n_kicks = 3 if tleft < 60 else 4
        scales = [2e-3, 5e-3, 1e-2, 8e-4][:n_kicks]
        improved = False
        for kk in range(n_kicks):
            if time.time() > HOP_END:
                break
            nc = int(rng.integers(8, 28))
            dx = _null_kick(curX, rig, rng, n_combine=nc, scale=float(scales[kk]))
            cand = _norm_rows(curX + dx)
            # cheap quench then tight LP
            cand, _ = _quench(cand, 350, 0.003, 0.0003, 0.12, deadline=HOP_END)
            cand, _ = _lp_polish_eps(cand, 1e-6, 7e-4, 2, HOP_END)
            nt2, ns2 = _topk(cand, 50)
            if _lex_better(nt2, cur_top):
                curX, cur_top, cur_score = cand, nt2, ns2
                if _lex_better(nt2, best_top):
                    best_top, best_score, best_X = nt2.copy(), ns2, cand.copy()
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
        if nd > 1e-12:
            e = da / nd
            step = float(rng.uniform(2e-3, 7e-3))
            Y = curX.copy()
            Y[a] = _norm_rows((xa + step * e)[None, :])[0]
            Y[b] = _norm_rows((xb - step * e)[None, :])[0]
            # jitter 4 random points slightly
            for _ in range(4):
                q = int(rng.integers(0, n))
                dz = rng.standard_normal(d) * 2e-3
                dz -= Y[q] * float(dz @ Y[q])
                Y[q] = _norm_rows((Y[q] + dz)[None, :])[0]
            Y, _ = _quench(Y, 300, 0.003, 0.0005, 0.12, deadline=HOP_END)
            Y, _ = _lp_polish_eps(Y, 1e-6, 7e-4, 2, HOP_END)
            nt2, ns2 = _topk(Y, 50)
            if _lex_better(nt2, cur_top):
                curX, cur_top, cur_score = Y, nt2, ns2
                if _lex_better(nt2, best_top):
                    best_top, best_score, best_X = nt2.copy(), ns2, Y.copy()
                continue
        # no progress: random restart-ish small shake to escape
        if it % 6 == 0:
            dx = rng.standard_normal((n, d)) * 4e-3
            dx -= curX * np.sum(dx * curX, axis=1, keepdims=True)
            curX = _norm_rows(curX + dx)
            curX, _ = _lp_polish_eps(curX, 1e-5, 1e-3, 2, HOP_END)
            cur_top, cur_score = _topk(curX, 50)
            # keep best_X untouched; allow wandering
    # final tight polish of best
    for eps in [1e-5, 1e-6, 5e-8]:
        if time.time() > HARD_END - 2:
            break
        best_X, _ = _lp_polish_eps(best_X, eps, 8e-4, 3, HARD_END - 1)
    best_X = _norm_rows(best_X)
    return best_X
