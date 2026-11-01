import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _norm_rows(X):
    N = np.linalg.norm(X, axis=1, keepdims=True)
    N[~np.isfinite(N)] = 1.0
    N[N == 0] = 1.0
    return X / N

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _topk_sorted(X, k=50):
    G = X @ X.T
    iu = np.triu_indices(X.shape[0], 1)
    v = G[iu]
    if len(v) <= k:
        return np.sort(v)[::-1].copy()
    idx = np.argpartition(-v, k)[:k]
    return np.sort(v[idx])[::-1].copy()

def _lex_less(a, b, tol=1e-12):
    m = min(len(a), len(b))
    for i in range(m):
        d = float(a[i]) - float(b[i])
        if d < -tol:
            return True
        if d > tol:
            return False
    return False

def _line_search_best(X, DX, best, n_grid=13):
    # search alpha in (0,1]
    alphas = np.concatenate([
        np.array([1.0, 0.7, 0.45, 0.25, 0.12, 0.06, 0.03]),
        np.linspace(0.85, 0.15, 6)
    ])
    alphas = np.unique(np.clip(alphas, 1e-4, 1.0))
    best_a = 0.0
    best_v = best
    best_Y = None
    for a in alphas:
        Y = _norm_rows(X + a * DX)
        v = _max_cos(Y)
        if v < best_v:
            best_v = v
            best_a = float(a)
            best_Y = Y
    if best_Y is None:
        return None, best, 0.0
    # local refine around best_a
    for _ in range(2):
        for f in (0.7, 1.3):
            a2 = float(np.clip(best_a * f, 1e-4, 1.0))
            Y2 = _norm_rows(X + a2 * DX)
            v2 = _max_cos(Y2)
            if v2 < best_v:
                best_v = v2
                best_a = a2
                best_Y = Y2
    return best_Y, best_v, best_a

def _lp_polish_budget(X0, budget_sec, deadline, eps_cycle, r0=2e-3):
    X = _norm_rows(np.array(X0, dtype=float, copy=True))
    n, d = X.shape
    best = _max_cos(X)
    radius = float(r0)
    iu = np.triu_indices(n, 1)
    cuv_all = None
    start = time.time()
    ei = 0
    while time.time() - start < budget_sec:
        if time.time() > deadline:
            break
        if radius < 1e-13:
            radius = 1e-4
            ei = (ei + 1) % len(eps_cycle)
            if time.time() - start > budget_sec * 0.9:
                break
        eps = float(eps_cycle[ei % len(eps_cycle)])
        G = X @ X.T
        cuv = G[iu]
        m = float(cuv.max())
        sel = cuv > m - eps
        cnt = int(np.count_nonzero(sel))
        if cnt == 0:
            ei += 1
            continue
        I = iu[0][sel]; J = iu[1][sel]
        csel = cuv[sel]
        if cnt > 2200:
            od = np.argsort(-csel)[:2200]
            I, J, csel = I[od], J[od], csel[od]
            cnt = 2200
        p = cnt
        nv = n * d + 1
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
            ei += 1
            continue
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            # cycle epsilon inward on repeated failure
            if radius < 5e-6:
                ei += 1
            continue
        DX = res.x[:-1].reshape(n, d)
        if not np.all(np.isfinite(DX)):
            radius *= 0.5
            continue
        Y, v, a = _line_search_best(X, DX, best)
        if Y is not None and v < best - 1e-13:
            X = Y
            best = v
            radius = min(radius * 1.4, 8e-3)
        else:
            radius *= 0.5
            ei += 1
            if ei % len(eps_cycle) == 0 and radius < 1e-6:
                radius = min(max(radius * 2.0, 2e-4), 2e-3)
    return X, best

def _tangent_bases(X):
    n, d = X.shape
    Ts = []
    for i in range(n):
        xi = X[i]
        _, _, Vt = np.linalg.svd(xi[None, :], full_matrices=True)
        Ti = Vt[1:, :].T.copy()  # d x (d-1)
        Ts.append(Ti)
    return Ts

def _rigidity_system(X, eps_plateau=1.2e-4, exclude_max=True):
    n, d = X.shape
    G = X @ X.T
    iu = np.triu_indices(n, 1)
    cuv = G[iu]
    m = float(cuv.max())
    sel = cuv > m - eps_plateau
    I0 = iu[0][sel]; J0 = iu[1][sel]
    c0 = cuv[sel]
    order = np.argsort(-c0)
    I0, J0, c0 = I0[order], J0[order], c0[order]
    # identify max pair
    amax, bmax = int(I0[0]), int(J0[0])
    if exclude_max and len(I0) > 1:
        I = I0[1:]; J = J0[1:]; c = c0[1:]
    else:
        I, J, c = I0, J0, c0
    Ts = _tangent_bases(X)
    dm = d - 1
    M = n * dm
    p = len(I)
    Jr = np.zeros((max(p, 1), M), dtype=float)
    for r in range(p):
        i, j = int(I[r]), int(J[r])
        # row: x_j^T Ti at block i, x_i^T Tj at block j
        Jr[r, i * dm:(i + 1) * dm] = X[j] @ Ts[i]
        Jr[r, j * dm:(j + 1) * dm] = X[i] @ Ts[j]
    # rotation basis in tangent coords
    R = np.zeros((M, d * (d - 1) // 2), dtype=float)
    col = 0
    for a in range(d):
        for b in range(a + 1, d):
            for i in range(n):
                xi = X[i]
                dx = np.zeros(d)
                dx[a] = xi[b]
                dx[b] = -xi[a]
                R[i * dm:(i + 1) * dm, col] = Ts[i].T @ dx
            col += 1
    # orthonormalize rotation space
    Qr, _ = np.linalg.qr(R)
    # SVD of Jr
    if p == 0:
        S = np.array([0.0])
        Vt = np.eye(M)
    else:
        try:
            _, S, Vt = np.linalg.svd(Jr, full_matrices=True)
        except Exception:
            S = np.ones(min(Jr.shape))
            Vt = np.eye(M)
    rank = int(np.sum(S > 1e-6))
    contacts = int(np.sum(c0 > m - 5e-8))
    return {
        "Jr": Jr, "S": S, "Vt": Vt, "Ts": Ts, "Qr": Qr,
        "rank": rank, "contacts": contacts, "m": m,
        "I0": I0, "J0": J0, "c0": c0, "amax": amax, "bmax": bmax,
        "M": M, "p": p, "dm": dm
    }

def _null_kick(X, info, rng, scale, nvec=40):
    n, d = X.shape
    Vt = info["Vt"]; Qr = info["Qr"]; Ts = info["Ts"]
    M = info["M"]; dm = info["dm"]
    K = int(min(nvec, M))
    Vsmall = Vt[M - K:, :].copy()  # K x M
    # project out rotations
    # Vsmall rows: remove Qr component
    C = Vsmall @ Qr  # K x 66
    Vsmall = Vsmall - (C @ Qr.T)
    # renormalize rows
    nr = np.linalg.norm(Vsmall, axis=1, keepdims=True) + 1e-30
    Vsmall = Vsmall / nr
    coeff = rng.standard_normal(K)
    coeff /= (np.linalg.norm(coeff) + 1e-30)
    v = coeff @ Vsmall  # M
    # remove residual rotation
    v = v - Qr @ (Qr.T @ v)
    vn = float(np.linalg.norm(v)) + 1e-30
    v /= vn
    DX = np.zeros((n, d))
    for i in range(n):
        DX[i] = Ts[i] @ v[i * dm:(i + 1) * dm]
    # orient to reduce max pair
    a, b = info["amax"], info["bmax"]
    g = float(DX[a] @ X[b] + X[a] @ DX[b])
    if g > 0:
        DX = -DX
    # scale so max row norm == scale
    rn = float(np.max(np.linalg.norm(DX, axis=1)) + 1e-30)
    DX *= (scale / rn)
    return DX

def solve():
    t0 = time.time()
    HARD_END = t0 + 174.0
    deadline = HARD_END - 2.0
    rng = np.random.default_rng(7)
    n, d = 71, 12
    cands = []
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.asarray(np.load(fn), dtype=np.float64)
            if a.shape == (n, d) and np.all(np.isfinite(a)):
                a = _norm_rows(a)
                cands.append(a)
        except Exception:
            pass
    if not cands:
        cands.append(_norm_rows(rng.standard_normal((n, d))))
    # pick starting point with lowest max
    scores = [_max_cos(a) for a in cands]
    best_X = cands[int(np.argmin(scores))].copy()
    best_score = float(min(scores))
    best_top = _topk_sorted(best_X, 50)
    EPS_CYCLE = [1e-4, 3e-5, 1e-5, 3e-6, 1e-6, 3e-7, 5e-8]
    # initial polish to nearest local optimum on plateau
    tleft = deadline - time.time()
    init_budget = min(14.0, max(4.0, tleft * 0.12))
    best_X, best_score = _lp_polish_budget(best_X, init_budget, deadline, EPS_CYCLE, r0=2e-3)
    best_top = _topk_sorted(best_X, 50)
    try:
        info0 = _rigidity_system(best_X, eps_plateau=1.2e-4, exclude_max=False)
        print(f"init max={best_score:.12f} contacts@5e-8={info0['contacts']} rank={info0['rank']} p_plateau={len(info0['I0'])} Smin={float(info0['S'].min()):.2e} Smax={float(info0['S'].max()):.2e}")
    except Exception as e:
        print("rigidity init failed", e)
        info0 = None
    it = 0
    stall = 0
    scales = [3e-4, 7e-4, 1.5e-3, 3e-3, 6e-3, 1.2e-2]
    while time.time() < deadline - 6.0:
        it += 1
        try:
            info = _rigidity_system(best_X, eps_plateau=1.2e-4, exclude_max=True)
        except Exception:
            break
        if it % 10 == 1:
            try:
                S = info["S"]
                print(f"it={it} max={best_score:.12f} contacts={info['contacts']} rank={info['rank']}/{info['M']-66} p={len(info['I0'])} S[0]={S[0]:.3f} Smed={np.median(S):.2e} Smin={S.min():.2e}")
            except Exception:
                pass
        improved = False
        # try several null kicks per outer iteration
        nkick = 3 if (deadline - time.time()) > 60 else 2
        for kk in range(nkick):
            if time.time() > deadline - 5.0:
                break
            # random scale, larger when stalled
            if stall > 12:
                sc = float(scales[int(rng.integers(2, len(scales)))])
            else:
                sc = float(scales[int(rng.integers(0, len(scales)))])
                if rng.random() < 0.3:
                    sc *= float(10 ** rng.uniform(-0.3, 0.3))
            try:
                DX = _null_kick(best_X, info, rng, scale=sc, nvec=48)
            except Exception:
                continue
            Xkick = _norm_rows(best_X + DX)
            # quick re-polish
            trem = deadline - time.time()
            if trem < 4.0:
                break
            budget = 2.2 if trem > 40 else (1.4 if trem > 15 else 0.8)
            try:
                Xp, sp = _lp_polish_budget(Xkick, budget, deadline, EPS_CYCLE, r0=2e-3)
            except Exception:
                continue
            if not np.all(np.isfinite(Xp)):
                continue
            top = _topk_sorted(Xp, 50)
            # accept only lexicographic top-50 improvement (or strict max improvement)
            if _lex_less(top, best_top) or sp < best_score - 1e-12:
                best_X, best_score, best_top = Xp, float(sp), top
                improved = True
                stall = 0
                print(f"  accept it={it} kick={kk} scale={sc:.1e} newmax={sp:.12f}")
                break
        if improved:
            continue
        stall += 1
        # occasional larger plateau-orthogonal shake to change topology
        if stall % 6 == 0:
            if time.time() > deadline - 8.0:
                break
            try:
                DX2 = _null_kick(best_X, info, rng, scale=2e-2, nvec=80)
                Xkick2 = _norm_rows(best_X + DX2 + rng.standard_normal((n, d)) * 2e-3)
                Xkick2 = _norm_rows(Xkick2)
                trem = deadline - time.time()
                budget = min(3.0, max(1.0, trem * 0.05))
                Xp2, sp2 = _lp_polish_budget(Xkick2, budget, deadline, EPS_CYCLE, r0=3e-3)
                top2 = _topk_sorted(Xp2, 50)
                if _lex_less(top2, best_top) or sp2 < best_score - 1e-12:
                    best_X, best_score, best_top = Xp2, float(sp2), top2
                    print(f"  big-accept it={it} newmax={sp2:.12f}")
                    stall = 0
            except Exception:
                pass
        # fallback: tiny random tangent noise polished (keeps diversity cheap)
        if stall % 4 == 0 and not improved:
            if time.time() > deadline - 8.0:
                break
            try:
                Xn = _norm_rows(best_X + rng.standard_normal((n, d)) * 1.5e-3)
                Xp3, sp3 = _lp_polish_budget(Xn, 1.0, deadline, [1e-5, 1e-6, 5e-8], r0=1.5e-3)
                top3 = _topk_sorted(Xp3, 50)
                if _lex_less(top3, best_top) or sp3 < best_score - 1e-12:
                    best_X, best_score, best_top = Xp3, float(sp3), top3
                    print(f"  noise-accept it={it} newmax={sp3:.12f}")
                    stall = 0
            except Exception:
                pass
    # final squeeze
    try:
        trem = deadline - time.time()
        if trem > 3.0:
            best_X, best_score = _lp_polish_budget(best_X, min(trem - 0.5, 8.0), deadline, [1e-4, 1e-5, 1e-6, 5e-8, 5e-8], r0=1.5e-3)
    except Exception:
        pass
    best_X = _norm_rows(best_X)
    try:
        infof = _rigidity_system(best_X, eps_plateau=1.2e-4, exclude_max=False)
        print(f"final max={_max_cos(best_X):.12f} contacts@5e-8={infof['contacts']} rank={infof['rank']}")
    except Exception:
        pass
    return np.array(best_X, dtype=np.float64)
