import time
import numpy as np
from scipy.optimize import linprog, minimize
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def solve():
    t0 = time.time()
    HARD_END = t0 + 172.0
    rng = np.random.default_rng(0)
    n, d = 249, 6

    # ---------- load incumbent ----------
    cands = []
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn).astype(float)
            if a.shape == (n, d):
                a = a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-12)
                if np.all(np.isfinite(a)):
                    cands.append(a)
        except Exception:
            pass
    if not cands:
        a = rng.standard_normal(size=(n, d))
        a /= np.linalg.norm(a, axis=1, keepdims=True)
        cands.append(a)
    vals = [_max_cos(a) for a in cands]
    bestX = cands[int(np.argmin(vals))].copy()
    bestVal = float(np.min(vals))

    # ---------- D6 group (order 12) on coordinates ----------
    # perm[j] = (s*j+k) mod 6, s=+1/-1
    perms = []
    for k in range(6):
        perms.append(np.array([(j + k) % 6 for j in range(6)], dtype=int))
    for k in range(6):
        perms.append(np.array([(-j + k) % 6 for j in range(6)], dtype=int))
    perms = np.stack(perms, axis=0)  # 12x6
    inv_perms = np.empty_like(perms)
    for g in range(12):
        inv = np.empty(6, dtype=int)
        inv[perms[g]] = np.arange(6)
        inv_perms[g] = inv
    rot_perms = np.stack([np.array([(j + k) % 6 for j in range(6)], dtype=int) for k in range(6)], axis=0)
    inv_rots = np.empty_like(rot_perms)
    for k in range(6):
        inv = np.empty(6, dtype=int)
        inv[rot_perms[k]] = np.arange(6)
        inv_rots[k] = inv
    M = np.array([[1, 0, 0, 0],
                  [0, 1, 0, 0],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1],
                  [0, 0, 1, 0],
                  [0, 1, 0, 0]], dtype=float)  # base6 = M@s
    Nmat = np.array([[1, 0], [0, 1], [1, 0], [0, 1], [1, 0], [0, 1]], dtype=float)
    fix_raw = np.ones(6, dtype=float)

    I0, J0 = np.triu_indices(n, 1)

    def build_X(p):
        v = p[0:120].reshape(20, 6)
        s = p[120:124]
        t2 = p[124:126]
        base6 = M @ s
        base2 = Nmat @ t2
        Ygen = v[:, perms]  # 20x12x6
        Ys6 = base6[rot_perms]  # 6x6
        Ys2 = base2[rot_perms[:2]]  # 2x6
        Y = np.empty((249, 6), dtype=float)
        Y[0:240] = Ygen.reshape(240, 6)
        Y[240:246] = Ys6
        Y[246:248] = Ys2
        Y[248] = fix_raw
        r = np.linalg.norm(Y, axis=1, keepdims=True)
        r = np.maximum(r, 1e-12)
        X = Y / r
        return Y, r, X

    def fg_factory(tau):
        def fg(p):
            Y, r, X = build_X(p)
            G = X @ X.T
            np.fill_diagonal(G, -2.0)
            m = float(np.max(G))
            # exp((G-m)/tau), diag 0
            E = (G - m) / tau
            # clip to avoid overflow (max 0)
            np.clip(E, -50.0, 0.0, out=E)
            np.exp(E, out=E)
            # zero diag (was exp(( -2-m)/tau) ~0 but force)
            np.fill_diagonal(E, 0.0)
            Shalf = float(np.sum(E) * 0.5)
            if Shalf <= 0 or not np.isfinite(Shalf):
                return 1e6, np.zeros_like(p)
            L = m + tau * np.log(Shalf)
            W = E / Shalf
            GX = W @ X  # dL/dX
            dot = np.sum(GX * X, axis=1, keepdims=True)
            GY = (GX - dot * X) / r
            Ggen = GY[0:240].reshape(20, 12, 6)
            Gs6 = GY[240:246]
            Gs2 = GY[246:248]
            # grad v
            gv = np.zeros((20, 6), dtype=float)
            for g in range(12):
                gv += Ggen[:, g, :][:, inv_perms[g]]
            # grad base6
            gb6 = np.zeros(6, dtype=float)
            for k in range(6):
                gb6 += Gs6[k][inv_rots[k]]
            gs = M.T @ gb6
            gb2 = np.zeros(6, dtype=float)
            for k in range(2):
                gb2 += Gs2[k][inv_rots[k]]
            gt = Nmat.T @ gb2
            g = np.empty_like(p)
            g[0:120] = gv.ravel()
            g[120:124] = gs
            g[124:126] = gt
            return float(L), g
        return fg

    # ---------- symmetric optimization ----------
    p0 = rng.standard_normal(size=126)
    # scale specials to reasonable size
    p_cur = p0.copy()
    taus = [0.03, 0.015, 0.008, 0.004, 0.002]
    iters = [120, 120, 160, 180, 220]
    for tau, mi in zip(taus, iters):
        if time.time() > HARD_END - 120:
            break
        fg = fg_factory(float(tau))
        try:
            res = minimize(fg, p_cur, jac=True, method='L-BFGS-B',
                           options={'maxiter': int(mi), 'maxfun': int(mi * 5),
                                    'ftol': 1e-14, 'gtol': 1e-7, 'maxcor': 10})
            p_cur = res.x
        except Exception:
            break
    try:
        _, _, Xsym = build_X(p_cur)
    except Exception:
        Xsym = None
    if Xsym is None or not np.all(np.isfinite(Xsym)):
        Xsym = bestX.copy()
    # normalize
    Xsym = Xsym / np.maximum(np.linalg.norm(Xsym, axis=1, keepdims=True), 1e-12)
    vsym = _max_cos(Xsym)

    # ---------- helpers for release ----------
    def tangent_bases(X):
        T = np.empty((n, 6, 5), dtype=float)
        for i in range(n):
            a = X[i]
            k = int(np.argmax(np.abs(a)))
            idx = [j for j in range(6) if j != k]
            V = np.zeros((6, 5), dtype=float)
            for l, j in enumerate(idx):
                V[j, l] = 1.0
                V[:, l] -= a[j] * a
            Q, _ = np.linalg.qr(V)
            T[i] = Q
        return T

    def lp_jam(Xinit, deadline, radius_init=2e-3):
        X = Xinit.copy()
        X /= np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
        mcur = _max_cos(X)
        radius = float(radius_init)
        Nv = n * 5
        it = 0
        fails = 0
        while time.time() < deadline and it < 80:
            it += 1
            if radius < 1e-12:
                break
            G = X @ X.T
            np.fill_diagonal(G, -2.0)
            m = float(np.max(G))
            gfull = G[I0, J0]
            slack = max(1e-3, 5.0 * radius)
            sel = np.where(gfull > m - slack)[0]
            if len(sel) == 0:
                sel = np.argsort(gfull)[-800:]
            if len(sel) > 4000:
                part = np.argpartition(gfull[sel], -4000)[-4000:]
                sel = sel[part]
            Ic = I0[sel]; Jc = J0[sel]; cc = gfull[sel]
            K = len(sel)
            T = tangent_bases(X)
            TI = T[Ic]; TJ = T[Jc]
            XJ = X[Jc]; XI = X[Ic]
            Ai = np.sum(TI * XJ[:, :, None], axis=1)
            Aj = np.sum(TJ * XI[:, :, None], axis=1)
            rows = np.concatenate([np.repeat(np.arange(K), 5), np.repeat(np.arange(K), 5), np.arange(K)])
            cols = np.concatenate([(Ic[:, None] * 5 + np.arange(5)).ravel(),
                                    (Jc[:, None] * 5 + np.arange(5)).ravel(),
                                    np.full(K, Nv)])
            data = np.concatenate([Ai.ravel(), Aj.ravel(), -np.ones(K)])
            try:
                Aub = coo_matrix((data, (rows, cols)), shape=(K, Nv + 1)).tocsr()
            except Exception:
                radius *= 0.5
                continue
            bub = -cc
            cvec = np.zeros(Nv + 1); cvec[Nv] = 1.0
            bnds = [(-radius, radius)] * Nv + [(None, None)]
            try:
                res = linprog(cvec, A_ub=Aub, b_ub=bub, bounds=bnds, method='highs',
                              options={'presolve': True})
            except Exception:
                radius *= 0.5
                fails += 1
                if fails > 6:
                    break
                continue
            if res.status != 0 or res.x is None:
                radius *= 0.5
                fails += 1
                if fails > 8:
                    break
                continue
            fails = 0
            y = res.x[:Nv].reshape(n, 5)
            delta = np.einsum('ijk,ik->ij', T, y)
            Yn = X + delta
            Yn /= np.maximum(np.linalg.norm(Yn, axis=1, keepdims=True), 1e-12)
            v = _max_cos(Yn)
            if v < m - 1e-12:
                X = Yn
                mcur = v if v < mcur else mcur
                radius = min(radius * 1.5, 0.05)
            else:
                radius *= 0.5
        return X

    def perturb_tangent(X, eps, rs):
        Nn = rs.standard_normal(size=X.shape)
        d = np.sum(Nn * X, axis=1, keepdims=True)
        Nn -= d * X
        nr = np.linalg.norm(Nn, axis=1, keepdims=True)
        nr[nr == 0] = 1.0
        Nn /= nr
        Y = X + eps * Nn
        Y /= np.maximum(np.linalg.norm(Y, axis=1, keepdims=True), 1e-12)
        return Y

    # ---------- release symmetry at 3 magnitudes ----------
    eps_list = [0.005, 0.02, 0.05]
    # budget releases leaving ~28s for final
    for ei, eps in enumerate(eps_list):
        if time.time() > HARD_END - 30:
            break
        rs = np.random.default_rng(100 + ei)
        Xp = perturb_tangent(Xsym, float(eps), rs)
        # give each release ~ (remaining-28)/remaining_count
        rem = HARD_END - time.time() - 28.0
        cnt = len(eps_list) - ei
        budget = max(8.0, rem / cnt) if cnt > 0 else 8.0
        budget = min(budget, 35.0)
        deadline = time.time() + budget
        if deadline > HARD_END - 28.0:
            deadline = HARD_END - 28.0
        Xj = lp_jam(Xp, deadline, radius_init=2e-3 if eps < 0.03 else 4e-3)
        vj = _max_cos(Xj)
        if vj < bestVal - 1e-12:
            bestVal = vj
            bestX = Xj.copy()
        if time.time() > HARD_END - 30:
            break

    # ---------- final jam on current best ----------
    rem = HARD_END - time.time()
    if rem > 9.0:
        deadline = HARD_END - 1.5
        # cap final polish to 40s
        if deadline - time.time() > 40.0:
            deadline = time.time() + 40.0
        bestX = lp_jam(bestX, deadline, radius_init=1.5e-3)
        bestX = bestX / np.maximum(np.linalg.norm(bestX, axis=1, keepdims=True), 1e-12)
    else:
        bestX = bestX / np.maximum(np.linalg.norm(bestX, axis=1, keepdims=True), 1e-12)
    return bestX
