import time
import numpy as np

def solve():
    t0 = time.time()
    LIMIT = 173.0
    n0, d = 93, 13
    rng = np.random.default_rng(0)

    def _norm(X):
        nr = np.linalg.norm(X, axis=1, keepdims=True)
        nr[nr == 0] = 1.0
        return X / nr

    def _maxcos(X):
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        return float(np.max(G))

    def _rowmax(X):
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        return G.max(axis=1)

    # load
    best93 = None
    best_val = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=float)
            if a.shape == (n0, d) and np.all(np.isfinite(a)):
                a = _norm(a)
                v = _maxcos(a)
                if v < best_val:
                    best_val = v
                    best93 = a.copy()
        except Exception:
            pass
    if best93 is None:
        best93 = _norm(rng.standard_normal((n0, d)))
        best_val = _maxcos(best93)
    cur_best = best93.copy()
    cur_best_val = float(best_val)

    def smooth_steps(X, steps, tau, lr, seed=0):
        X = X.copy()
        N = X.shape[0]
        iu0, iu1 = np.triu_indices(N, 1)
        m_ad = np.zeros_like(X)
        v_ad = np.zeros_like(X)
        b1, b2, eps = 0.9, 0.999, 1e-8
        for t in range(1, steps + 1):
            if time.time() - t0 > LIMIT - 4:
                break
            G = X @ X.T
            c = G[iu0, iu1]
            mx = float(c.max())
            E = np.exp((c - mx) / tau)
            S = float(E.sum())
            if not np.isfinite(S) or S <= 0:
                break
            W = np.zeros((N, N), dtype=np.float64)
            W[iu0, iu1] = E
            W[iu1, iu0] = E
            gradX = (W @ X) / S
            dot = np.einsum('ij,ij->i', gradX, X)[:, None]
            g = gradX - dot * X
            m_ad = b1 * m_ad + (1 - b1) * g
            v_ad = b2 * v_ad + (1 - b2) * g * g
            mhat = m_ad / (1 - b1 ** t)
            vhat = v_ad / (1 - b2 ** t)
            upd = mhat / (np.sqrt(vhat) + eps)
            # re-project to tangent
            d2 = np.einsum('ij,ij->i', upd, X)[:, None]
            upd_t = upd - d2 * X
            # decay
            lr_t = lr * (1.0 - 0.5 * t / max(steps, 1))
            X = X - lr_t * upd_t
            X = _norm(X)
        return X

    def hard_opt(X, total_steps, taus, lr0):
        Xc = X.copy()
        k = len(taus)
        per = max(50, total_steps // k)
        for tau in taus:
            if time.time() - t0 > LIMIT - 4:
                break
            Xc = smooth_steps(Xc, per, tau, lr0)
        return Xc

    def largest_void(X, n_sample=12000):
        N = X.shape[0]
        Z = rng.standard_normal((n_sample, d))
        Z = _norm(Z)
        D = Z @ X.T
        rm = D.max(axis=1)
        idx = int(np.argmin(rm))
        return Z[idx:idx+1, :]

    def add_voids(X, k):
        Xc = X.copy()
        for _ in range(k):
            v = largest_void(Xc)
            # refine void: push away from set with few smooth steps on single point
            # simple local search: try small perturbations
            best_v = v.copy()
            G0 = (Xc @ best_v[0]).max()
            for _ in range(60):
                cand = _norm(best_v + 0.02 * rng.standard_normal((1, d)))
                gv = float((Xc @ cand[0]).max())
                if gv < G0:
                    G0 = gv
                    best_v = cand
            Xc = np.vstack([Xc, best_v])
            Xc = _norm(Xc)
        return Xc

    # ---------- (a) overshoot then prune ----------
    try:
        if time.time() - t0 < 60:
            N_big = 98
            kadd = N_big - n0  # 5
            Xbig = add_voids(cur_best, kadd)
            # hard optimize N=98 with p-continuation (tau ~ 1/p)
            Xbig = hard_opt(Xbig, 1800, [0.02, 0.012, 0.007, 0.0035, 0.0018], 0.012)
            curX = Xbig
            # greedy delete back to 93
            while curX.shape[0] > n0:
                if time.time() - t0 > LIMIT - 45:
                    break
                Ncur = curX.shape[0]
                rm = _rowmax(curX)
                order = np.argsort(-rm)
                cand_idx = order[:min(15, Ncur)].tolist()
                best_c_val = 1e9
                best_c_X = None
                for ci in cand_idx:
                    if time.time() - t0 > LIMIT - 45:
                        break
                    Xdel = np.delete(curX, ci, axis=0)
                    # brief repolish
                    Xdel_p = smooth_steps(Xdel, 120, 0.003, 0.008)
                    Xdel_p = smooth_steps(Xdel_p, 60, 0.0012, 0.005)
                    v = _maxcos(Xdel_p)
                    if v < best_c_val:
                        best_c_val = v
                        best_c_X = Xdel_p.copy()
                if best_c_X is None:
                    break
                curX = best_c_X
                # if we hit 93, consider archive (accept even if slightly worse, but archive only if better)
                if curX.shape[0] == n0:
                    # polish a bit more
                    curX = smooth_steps(curX, 200, 0.002, 0.007)
                    v93 = _maxcos(curX)
                    if v93 < cur_best_val:
                        cur_best_val = v93
                        cur_best = curX.copy()
    except Exception:
        pass

    # ---------- (b) undershoot then grow ----------
    try:
        if time.time() - t0 < 115:
            # delete 8 loosest rattlers
            rm = _rowmax(cur_best)
            order_loose = np.argsort(rm)  # ascending: loosest first
            ndel = 8
            keep = np.ones(n0, dtype=bool)
            keep[order_loose[:ndel]] = False
            Xsmall = cur_best[keep].copy()  # 85
            Xsmall = hard_opt(Xsmall, 1600, [0.02, 0.01, 0.005, 0.002], 0.012)
            curS = Xsmall
            # regrow to 93 by void insertion (only after distinct optimum)
            while curS.shape[0] < n0:
                if time.time() - t0 > LIMIT - 35:
                    break
                v = largest_void(curS, n_sample=8000)
                curS = np.vstack([curS, v])
                curS = _norm(curS)
                # polish after each insertion
                curS = smooth_steps(curS, 250, 0.004, 0.009)
                curS = smooth_steps(curS, 80, 0.0015, 0.005)
            if curS.shape[0] == n0:
                v93 = _maxcos(curS)
                # extra polish
                if v93 < cur_best_val + 2e-4:
                    curS2 = smooth_steps(curS, 250, 0.0012, 0.005)
                    v93b = _maxcos(curS2)
                    if v93b < v93:
                        curS = curS2
                        v93 = v93b
                if v93 < cur_best_val:
                    cur_best_val = v93
                    cur_best = curS.copy()
    except Exception:
        pass

    # ---------- final LP polish on best 93 ----------
    try:
        from scipy.optimize import linprog
        from scipy.sparse import coo_matrix
        X = cur_best.copy()
        cur = _maxcos(X)
        m_t = d - 1
        nv = n0 * m_t + 1
        ar = np.arange(m_t)
        iu0, iu1 = np.triu_indices(n0, 1)
        P = len(iu0)
        radius = 2.5e-3
        while time.time() - t0 < LIMIT - 2.0:
            G = X @ X.T
            c_all = G[iu0, iu1]
            cmax = float(c_all.max())
            cur = cmax
            # active filter: slack < thresh + top 800
            thresh = 4e-4
            sel = c_all > cmax - thresh
            if int(sel.sum()) < 800:
                idx = np.argpartition(-c_all, 800)[:800]
                sel = np.zeros(P, dtype=bool)
                sel[idx] = True
            I = iu0[sel]; J = iu1[sel]
            csub = c_all[sel]
            p = len(I)
            # tangent bases via per-point QR
            T = np.empty((n0, d, m_t), dtype=np.float64)
            for i in range(n0):
                xi = X[i]
                k0 = int(np.argmin(np.abs(xi)))
                B = np.empty((d, m_t), dtype=np.float64)
                col = 0
                for j in range(d):
                    if j == k0:
                        continue
                    vv = -xi * xi[j]
                    vv[j] += 1.0
                    B[:, col] = vv
                    col += 1
                Q, _ = np.linalg.qr(B)
                T[i] = Q
            TI = T[I]; TJ = T[J]
            XI = X[I]; XJ = X[J]
            CI = np.einsum('pdm,pd->pm', TI, XJ)
            CJ = np.einsum('pdm,pd->pm', TJ, XI)
            nnz = p * (2 * m_t + 1)
            rows = np.empty(nnz, dtype=np.int64)
            cols = np.empty(nnz, dtype=np.int64)
            data = np.empty(nnz, dtype=np.float64)
            r1 = np.repeat(np.arange(p), m_t)
            c1 = (I[:, None] * m_t + ar[None, :]).ravel()
            c2 = (J[:, None] * m_t + ar[None, :]).ravel()
            rows[0:p*m_t] = r1; cols[0:p*m_t] = c1; data[0:p*m_t] = CI.ravel()
            rows[p*m_t:2*p*m_t] = r1; cols[p*m_t:2*p*m_t] = c2; data[p*m_t:2*p*m_t] = CJ.ravel()
            rows[2*p*m_t:2*p*m_t+p] = np.arange(p)
            cols[2*p*m_t:2*p*m_t+p] = nv - 1
            data[2*p*m_t:2*p*m_t+p] = -1.0
            A_ub = coo_matrix((data, (rows, cols)), shape=(p, nv)).tocsr()
            b_ub = -csub
            c_obj = np.zeros(nv); c_obj[-1] = 1.0
            bounds = [(-radius, radius)] * (n0 * m_t) + [(-1.0, cur)]
            try:
                res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs", options={"presolve": True})
            except Exception:
                radius *= 0.5
                continue
            if res.status != 0 or res.x is None or not np.all(np.isfinite(res.x)):
                radius *= 0.5
                if radius < 1e-11:
                    radius = 2e-6
                continue
            yv = res.x[:-1]; t_pred = float(res.x[-1])
            pred_red = cur - t_pred
            if pred_red <= 1e-13:
                radius *= 0.5
                if radius < 1e-10:
                    radius = 2e-6
                continue
            Y = yv.reshape(n0, m_t)
            Ddir = np.einsum('iab,ib->ia', T, Y)
            Xt = _norm(X + Ddir)
            actual = _maxcos(Xt)
            if actual < cur - 1e-15:
                rho = (cur - actual) / pred_red if pred_red > 1e-14 else 0.0
                X = Xt; cur = actual
                if actual < cur_best_val:
                    cur_best_val = actual
                    cur_best = Xt.copy()
                if rho > 0.5:
                    radius = min(radius * 1.25, 8e-3)
                elif rho < 0.1:
                    radius = max(radius * 0.9, 1e-12)
            else:
                found = False
                for alpha in (0.5, 0.25, 0.125):
                    Xb = _norm(X + alpha * Ddir)
                    vb = _maxcos(Xb)
                    if vb < cur - 1e-15:
                        X = Xb; cur = vb
                        if vb < cur_best_val:
                            cur_best_val = vb
                            cur_best = Xb.copy()
                        radius = max(radius * 0.7, 1e-12)
                        found = True
                        break
                if not found:
                    radius *= 0.5
                    if radius < 5e-11:
                        radius = 2e-6
    except Exception:
        pass

    cur_best = _norm(cur_best)
    return cur_best
