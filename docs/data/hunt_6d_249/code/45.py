import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def _tangent_bases(X):
    n = X.shape[0]
    T = np.empty((n, 6, 5), dtype=float)
    E = np.eye(6)
    e1 = E[:, 0]
    for i in range(n):
        a = X[i]
        d = e1 - a
        nv = float(np.linalg.norm(d))
        if nv < 1e-12:
            T[i] = E[:, 1:]
        else:
            w = d / nv
            H = E - 2.0 * np.outer(w, w)
            T[i] = H[:, 1:]
    return T

def solve():
    t0 = time.time()
    T_END = t0 + 172.0
    rng = np.random.default_rng(7)
    np.random.seed(7)
    n, d = 249, 6
    I0, J0 = np.triu_indices(n, 1)

    cands = []
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=float)
            if a.shape == (n, d):
                an = a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-12)
                if np.all(np.isfinite(an)):
                    cands.append(an)
        except Exception:
            pass
    if not cands:
        a = rng.standard_normal(size=(n, d))
        a /= np.linalg.norm(a, axis=1, keepdims=True)
        cands.append(a)
    vals = [_max_cos(a) for a in cands]
    bi = int(np.argmin(vals))
    bestX = cands[bi].copy()
    bestX /= np.linalg.norm(bestX, axis=1, keepdims=True)
    bestVal = float(vals[bi])

    def lp_jam(X0, deadline, radius0=0.01, slack=2e-3, max_pairs=6000):
        cur = X0.copy()
        cur /= np.linalg.norm(cur, axis=1, keepdims=True)
        curVal = _max_cos(cur)
        radius = float(radius0)
        nvars = n * 5 + 1
        t_idx = n * 5
        obj = np.zeros(nvars)
        obj[t_idx] = 1.0
        it = 0
        while time.time() < deadline - 0.4:
            if radius < 1e-12:
                break
            it += 1
            if it > 300:
                break
            G = cur @ cur.T
            np.fill_diagonal(G, -2.0)
            m = float(np.max(G))
            g = G[I0, J0]
            sel = np.where(g > m - slack)[0]
            if sel.size == 0:
                break
            if sel.size > max_pairs:
                part = np.argpartition(g[sel], -max_pairs)[-max_pairs:]
                sel = sel[part]
            I = I0[sel]; J = J0[sel]; csv = g[sel]
            p = sel.size
            # tangent bases at cur
            T = _tangent_bases(cur)
            TI = T[I]  # p,6,5
            TJ = T[J]
            XJ = cur[J]
            XI = cur[I]
            coeff_I = np.einsum('pab,pa->pb', TI, XJ)
            coeff_J = np.einsum('pab,pa->pb', TJ, XI)
            rows_I = np.repeat(np.arange(p), 5)
            cols_I = (I[:, None] * 5 + np.arange(5)).ravel()
            data_I = coeff_I.ravel()
            rows_J = np.repeat(np.arange(p), 5)
            cols_J = (J[:, None] * 5 + np.arange(5)).ravel()
            data_J = coeff_J.ravel()
            rows_t = np.arange(p)
            cols_t = np.full(p, t_idx, dtype=int)
            data_t = np.full(p, -1.0)
            rows = np.concatenate([rows_I, rows_J, rows_t])
            cols = np.concatenate([cols_I, cols_J, cols_t])
            data = np.concatenate([data_I, data_J, data_t])
            try:
                A_ub = coo_matrix((data, (rows, cols)), shape=(p, nvars)).tocsr()
                b_ub = -csv
                bounds = [(-radius, radius)] * (n * 5) + [(-1.0, m)]
                res = linprog(obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
            except Exception:
                radius *= 0.5
                continue
            if res.status != 0:
                radius *= 0.5
                continue
            try:
                t_pred = float(res.x[t_idx])
                y = res.x[:n * 5]
            except Exception:
                radius *= 0.5
                continue
            if not np.all(np.isfinite(y)) or not np.isfinite(t_pred):
                radius *= 0.5
                continue
            if t_pred >= m - 1e-13:
                radius *= 0.5
                if radius < 5e-11:
                    break
                # if very small radius and no progress -> jammed
                if radius < 1e-9 and p > 100:
                    # check if truly jammed: try one more tiny step
                    pass
                continue
            Y = y.reshape(n, 5)
            DX = np.einsum('ijk,ik->ij', T, Y)
            Xt = cur + DX
            nr = np.linalg.norm(Xt, axis=1, keepdims=True)
            nr = np.maximum(nr, 1e-12)
            Xt /= nr
            v = _max_cos(Xt)
            if v < curVal - 1e-15 and v < curVal:
                cur = Xt
                curVal = v
                radius = min(radius * 1.5, 0.02)
            else:
                radius *= 0.5
        return cur, curVal

    def contact_degrees(X, tol):
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        m = float(np.max(G))
        adj = G > (m - tol)
        deg = np.sum(adj, axis=1).astype(int)
        return deg, m, G

    def greedy_hole_refill(Xbase, idx_list, n_cand=25000):
        X = Xbase.copy()
        keep_all = X.copy()
        for r in idx_list:
            # scores vs all except r
            mask = np.ones(n, dtype=bool)
            mask[r] = False
            Xkeep = X[mask]
            C = rng.standard_normal(size=(n_cand, d))
            C /= np.maximum(np.linalg.norm(C, axis=1, keepdims=True), 1e-12)
            D = C @ Xkeep.T
            cmax = D.max(axis=1)
            b = int(np.argmin(cmax))
            X[r] = C[b]
        # 2 passes of local hole polishing for replaced points: small random search around each
        for _ in range(1):
            for r in idx_list:
                if time.time() > T_END - 9:
                    break
                cur_pt = X[r].copy()
                mask = np.ones(n, dtype=bool); mask[r] = False
                Xkeep = X[mask]
                best_pt = cur_pt
                best_s = float(np.max(Xkeep @ cur_pt))
                # local candidates around cur_pt
                Lc = 4000
                Z = rng.standard_normal(size=(Lc, d)) * 0.15
                # project to tangent and add
                for zi in range(Lc // 500 + 1):
                    pass
                # vectorized local sampling
                Z = rng.standard_normal(size=(Lc, d)) * 0.2
                # make tangent
                dots = Z @ cur_pt
                Z = Z - dots[:, None] * cur_pt[None, :]
                Cl = cur_pt[None, :] + Z
                Cl /= np.maximum(np.linalg.norm(Cl, axis=1, keepdims=True), 1e-12)
                D2 = Cl @ Xkeep.T
                s2 = D2.max(axis=1)
                b2 = int(np.argmin(s2))
                if float(s2[b2]) < best_s:
                    X[r] = Cl[b2]
        X /= np.linalg.norm(X, axis=1, keepdims=True)
        return X

    # Phase A: exact jam of incumbent
    try:
        Xj, Vj = lp_jam(bestX, deadline=min(t0 + 38.0, T_END - 100.0), radius0=0.01, slack=2e-3)
        if Vj < bestVal:
            bestVal = Vj; bestX = Xj.copy()
    except Exception:
        pass

    # hopping over rattlers / low-degree points
    it = 0
    while time.time() < T_END - 9.0:
        it += 1
        # identify rattlers at tight-ish tolerance
        try:
            deg7, mcur, _ = contact_degrees(bestX, 1e-7)
            deg9, _, _ = contact_degrees(bestX, 1e-9)
        except Exception:
            deg7 = np.zeros(n, dtype=int); deg9 = np.zeros(n, dtype=int)
        ratt7 = np.where(deg7 == 0)[0]
        ratt9 = np.where(deg9 == 0)[0]
        # choose pool: prefer ratt7, else lowest degree
        if len(ratt7) >= 1:
            pool = ratt7
        elif len(ratt9) >= 1:
            pool = ratt9
        else:
            pool = np.argsort(deg7)[:20]
        # choose k
        u = rng.random()
        if u < 0.5:
            k = 1
        elif u < 0.8:
            k = 2
        elif u < 0.93:
            k = 3
        else:
            k = int(rng.integers(6, 13))
        k = min(k, len(pool), 12)
        if k <= 0:
            k = 1
            pool = np.argsort(deg7)[:20]
        chosen = rng.choice(pool, size=k, replace=False).tolist()
        # occasionally add a lowest-rowmax point (overloaded neighbor)
        if rng.random() < 0.25:
            try:
                Gc = bestX @ bestX.T
                np.fill_diagonal(Gc, -2.0)
                rm = np.max(Gc, axis=1)
                worst = int(np.argmax(rm))
                if worst not in chosen:
                    chosen.append(worst)
            except Exception:
                pass
        # build trial by hole refill
        try:
            Xtrial = greedy_hole_refill(bestX, chosen, n_cand=25000)
        except Exception:
            continue
        # short exact jam
        jam_deadline = min(time.time() + 6.0, T_END - 8.0)
        if jam_deadline - time.time() < 1.0:
            break
        try:
            Xp, Vp = lp_jam(Xtrial, deadline=jam_deadline, radius0=0.008, slack=2e-3)
        except Exception:
            continue
        if Vp < bestVal - 1e-15:
            bestVal = Vp; bestX = Xp.copy()
        else:
            # non-monotone second chance: bigger kick of same region + rejam once
            if rng.random() < 0.3 and time.time() < T_END - 20:
                try:
                    X2 = bestX.copy()
                    for r in chosen:
                        pert = rng.standard_normal(size=(d,)) * 0.05
                        pert = pert - np.dot(pert, X2[r]) * X2[r]
                        X2[r] = X2[r] + pert
                    X2 /= np.linalg.norm(X2, axis=1, keepdims=True)
                    X2p, V2p = lp_jam(X2, deadline=min(time.time() + 4.0, T_END - 8.0), radius0=0.012, slack=2e-3)
                    if V2p < bestVal - 1e-15:
                        bestVal = V2p; bestX = X2p.copy()
                except Exception:
                    pass

    bestX /= np.maximum(np.linalg.norm(bestX, axis=1, keepdims=True), 1e-12)
    return bestX
