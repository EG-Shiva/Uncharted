import time
import numpy as np

def _norm(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return X / n

def _maxcos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def _rowmax(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return G.max(axis=1)

def solve():
    t0 = time.time()
    DEADLINE = 172.0
    EQUI_END = 118.0
    rng = np.random.default_rng(7)
    n, d = 93, 13
    iu0, iu1 = np.triu_indices(n, 1)
    M = len(iu0)

    best = None
    best_val = 1e9
    rec = None
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=np.float64)
            if a.shape != (93, 13):
                continue
            a = _norm(a)
            v = _maxcos(a)
            if fn == "record.npy":
                rec = a.copy()
            if v < best_val:
                best_val = v
                best = a.copy()
        except Exception:
            pass
    if best is None:
        best = _norm(rng.standard_normal((n, d)))
        best_val = _maxcos(best)
    if rec is None:
        rec = best.copy()

    # ---------- reweighted row-max equioscillation Adam ----------
    def equi_run(X0, stages):
        X = _norm(X0.copy())
        w = np.ones(n, dtype=np.float64)
        # diagnostics bests
        cur_max = _maxcos(X)
        rm = _rowmax(X)
        spread = float(rm.max() - rm.min())
        best_max_X = X.copy(); best_max_v = cur_max
        best_combo_X = X.copy(); best_combo_v = cur_max + 0.5 * spread
        best_combo_spread = spread
        best_combo_max = cur_max
        m = np.zeros_like(X); v = np.zeros_like(X)
        b1, b2, eps = 0.9, 0.999, 1e-8
        tt = 0
        for (p, tau, steps, lr, beta) in stages:
            m[:] = 0.0; v[:] = 0.0; tt = 0
            pm1 = float(p - 1)
            for s in range(steps):
                if time.time() - t0 > EQUI_END:
                    break
                G = X @ X.T
                np.fill_diagonal(G, -2.0)
                # per-row max for stability
                mrow = G.max(axis=1)  # (n,)
                # LSE row maxima
                # exp((G - mrow[:,None])/tau)
                E = np.exp((G - mrow[:, None]) / tau)
                np.fill_diagonal(E, 0.0)
                # row sums (diag zeroed, but exp of (-2-m)/tau ~0 anyway)
                S = E.sum(axis=1) + 1e-300
                r = mrow + tau * np.log(S)  # smoothed row max
                # Q rows
                Q = E / S[:, None]
                # a_i = w_i * r_i^{p-1}, clip r to positive
                rc = np.clip(r, 1e-3, None)
                # avoid overflow: scale by max
                # loga = logw + pm1*log rc ; subtract max for grad scale (Adam invariant-ish)
                loga = np.log(np.maximum(w, 1e-12)) + pm1 * np.log(rc)
                loga -= float(np.max(loga))
                a = np.exp(loga)
                # grad = (diag(a)Q + (diag(a)Q)^T?) careful:
                # term1 = a[:,None]*(Q@X), term2 = (diag(a)Q)^T @ X
                QX = Q @ X  # (n,d)
                term1 = a[:, None] * QX
                AQ = a[:, None] * Q  # diag(a)Q
                term2 = AQ.T @ X
                grad = term1 + term2
                # tangent project
                dot = np.einsum('ij,ij->i', grad, X)[:, None]
                gt = grad - dot * X
                if not np.all(np.isfinite(gt)):
                    continue
                tt += 1
                m[:] = b1 * m + (1.0 - b1) * gt
                v[:] = b2 * v + (1.0 - b2) * (gt * gt)
                mh = m / (1.0 - b1 ** tt)
                vh = v / (1.0 - b2 ** tt)
                step = lr * mh / (np.sqrt(vh) + eps)
                # project step to tangent
                sd = np.einsum('ij,ij->i', step, X)[:, None]
                step_t = step - sd * X
                X = _norm(X - step_t)
                if (tt % 40) == 0 or s == steps - 1:
                    cm = _maxcos(X)
                    rm2 = _rowmax(X)
                    sp2 = float(rm2.max() - rm2.min())
                    # count within 1e-6
                    # update best max
                    if cm < best_max_v:
                        best_max_v = cm
                        best_max_X = X.copy()
                    combo = cm + 0.5 * sp2
                    if combo < best_combo_v:
                        best_combo_v = combo
                        best_combo_X = X.copy()
                        best_combo_spread = sp2
                        best_combo_max = cm
            if time.time() - t0 > EQUI_END:
                break
            # multiplicative weight update using true row maxima
            try:
                rmT = _rowmax(X)
                rbar = float(rmT.mean())
                w = w * np.exp(beta * (rmT - rbar))
                # clip + renormalize to mean 1
                w = np.clip(w, 0.02, 20.0)
                w *= (n / (w.sum() + 1e-300))
            except Exception:
                pass
        return (best_max_X, best_max_v, best_combo_X, best_combo_max, best_combo_spread)

    stages = [
        (8, 0.01, 500, 3.0e-3, 20.0),
        (16, 0.005, 500, 2.0e-3, 30.0),
        (32, 0.002, 650, 1.2e-3, 40.0),
        (48, 0.001, 650, 7.0e-4, 60.0),
        (64, 0.0007, 650, 4.0e-4, 80.0),
    ]

    starts = []
    starts.append(best.copy())
    starts.append(rec.copy())
    starts.append(_norm(best + 0.25 * rng.standard_normal(best.shape)))
    starts.append(_norm(rec + 0.35 * rng.standard_normal(rec.shape)))
    starts.append(_norm(rng.standard_normal((n, d))))

    cands = []  # (val, X, spread)
    glob_best_X = best.copy(); glob_best_v = best_val
    for si, X0 in enumerate(starts):
        if time.time() - t0 > EQUI_END:
            break
        # shorten later starts
        if si >= 2:
            st2 = [(p, t, max(250, st // 2), lr, be) for (p, t, st, lr, be) in stages]
        else:
            st2 = stages
        try:
            bX, bV, cX, cM, cS = equi_run(X0, st2)
        except Exception:
            continue
        cands.append((bV, bX.copy(), 1e9))
        # equioscillated iterate: value cM with spread cS
        cands.append((cM, cX.copy(), cS))
        if bV < glob_best_v:
            glob_best_v = bV; glob_best_X = bX.copy()
        if cM < glob_best_v:
            glob_best_v = cM; glob_best_X = cX.copy()

    # keep diverse shortlist: lowest max + most equioscillated
    cands.sort(key=lambda t: t[0])
    short = []
    if len(cands) > 0:
        short.append(cands[0][1])
        # most equioscillated among near-best (within 0.002 of best)
        thr = cands[0][0] + 0.002
        best_sp = 1e9; best_spX = None
        for val, XX, sp in cands:
            if val <= thr and sp < best_sp:
                best_sp = sp; best_spX = XX
        if best_spX is not None and len(short) == 1:
            # check distinct
            short.append(best_spX)
        # add second best max if distinct
        for val, XX, sp in cands:
            if len(short) >= 3:
                break
            # avoid duplicates
            dup = False
            for Y in short:
                if np.linalg.norm(XX - Y) < 1e-6:
                    dup = True; break
            if not dup:
                short.append(XX)
                break
    if len(short) == 0:
        short = [glob_best_X]

    # ---------- top-K hard-max descent ----------
    def hard_topk(X, iters, step0, K):
        X = _norm(X.copy())
        bv = _maxcos(X)
        bestX = X.copy()
        step = float(step0)
        K = int(min(K, M))
        grad = np.empty_like(X)
        for it in range(iters):
            if time.time() - t0 > DEADLINE - 14:
                break
            G = X @ X.T
            c = G[iu0, iu1]
            if K < M:
                idx = np.argpartition(-c, K)[:K]
            else:
                idx = np.arange(M)
            I = iu0[idx]; J = iu1[idx]
            grad.fill(0.0)
            np.add.at(grad, I, X[J])
            np.add.at(grad, J, X[I])
            grad /= float(len(idx))
            dot = np.einsum('ij,ij->i', grad, X)[:, None]
            gt = grad - dot * X
            rn = np.linalg.norm(gt, axis=1, keepdims=True)
            nz = (rn[:, 0] > 1e-14)
            direc = np.zeros_like(gt)
            direc[nz] = gt[nz] / rn[nz, None]
            Xt = _norm(X - step * direc)
            vv = _maxcos(Xt)
            if vv < bv:
                bv = vv; bestX = Xt.copy(); X = Xt
                step = min(step * 1.25, 0.006)
            else:
                step *= 0.55
                if step < 1e-12:
                    step = float(step0) * 0.25
                    kick = rng.standard_normal(X.shape) * 1e-4
                    kick -= (kick * X).sum(axis=1, keepdims=True) * X
                    X = _norm(X + kick)
        return bestX, bv

    polished = []
    for XX in short:
        if time.time() - t0 > DEADLINE - 14:
            break
        try:
            Xh, vh = hard_topk(XX, iters=200, step0=0.0012, K=700)
            Xh2, vh2 = hard_topk(Xh, iters=200, step0=0.0007, K=120)
            if vh2 < vh:
                Xh, vh = Xh2, vh2
        except Exception:
            Xh, vh = _norm(XX.copy()), _maxcos(XX)
        polished.append((vh, Xh))
        if vh < glob_best_v:
            glob_best_v = vh; glob_best_X = Xh.copy()

    # ---------- LP trust-region polish (tangent SQP) ----------
    try:
        from scipy.optimize import linprog
        from scipy.sparse import coo_matrix
        dm = d - 1
        Nvar = n * dm + 1
        ar = np.arange(dm)

        def lp_polish(Xinit, budget_s):
            X = _norm(Xinit.copy())
            cur = _maxcos(X)
            bestL = X.copy(); bestLv = cur
            radius = 3e-3
            rmax = 1e-2
            t_start = time.time()
            it = 0
            c_obj = np.zeros(Nvar); c_obj[-1] = 1.0
            while True:
                if time.time() - t0 > DEADLINE - 3:
                    break
                if time.time() - t_start > budget_s:
                    break
                it += 1
                if radius < 1e-12:
                    break
                G = X @ X.T
                c_all = G[iu0, iu1]
                cmax = float(c_all.max())
                # active: slack<2e-3 or top 800
                order = np.argpartition(-c_all, min(800, M) - 1)[:min(800, M)]
                thr = cmax - 2e-3
                sel = c_all >= thr
                # union
                mask = np.zeros(M, dtype=bool); mask[order] = True; mask |= sel
                I = iu0[mask]; J = iu1[mask]; csub = c_all[mask]; p = len(I)
                if p == 0:
                    break
                # tangent bases via QR of (e_j - xi_j xi)
                T = np.empty((n, d, dm))
                for i in range(n):
                    xi = X[i]
                    k = int(np.argmin(np.abs(xi)))
                    B = np.empty((d, dm))
                    col = 0
                    for j in range(d):
                        if j == k:
                            continue
                        vv = -xi * xi[j]; vv[j] += 1.0
                        B[:, col] = vv; col += 1
                    Qq, _ = np.linalg.qr(B)
                    T[i] = Qq
                TI = T[I]; TJ = T[J]
                XI = X[I]; XJ = X[J]
                CI = np.einsum('pdm,pd->pm', TI, XJ)
                CJ = np.einsum('pdm,pd->pm', TJ, XI)
                nnz = p * (2 * dm + 1)
                rows = np.empty(nnz, dtype=np.int64)
                cols = np.empty(nnz, dtype=np.int64)
                data = np.empty(nnz, dtype=np.float64)
                r1 = np.repeat(np.arange(p), dm)
                c1 = (I[:, None] * dm + ar[None, :]).ravel()
                c2 = (J[:, None] * dm + ar[None, :]).ravel()
                rows[0:p*dm] = r1; cols[0:p*dm] = c1; data[0:p*dm] = CI.ravel()
                rows[p*dm:2*p*dm] = r1; cols[p*dm:2*p*dm] = c2; data[p*dm:2*p*dm] = CJ.ravel()
                rows[2*p*dm:2*p*dm+p] = np.arange(p); cols[2*p*dm:2*p*dm+p] = Nvar - 1
                data[2*p*dm:2*p*dm+p] = -1.0
                A_ub = coo_matrix((data, (rows, cols)), shape=(p, Nvar)).tocsr()
                b_ub = -csub
                bounds = [(-radius, radius)] * (n * dm) + [(-1.0, cmax)]
                try:
                    res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs',
                                 options={'presolve': True})
                except Exception:
                    radius *= 0.5
                    continue
                if not res.success:
                    radius *= 0.5
                    continue
                y = res.x[:n*dm].reshape(n, dm)
                tpred = float(res.x[-1])
                pred_red = cmax - tpred
                if pred_red < 1e-13:
                    radius *= 0.5
                    if radius < 5e-6:
                        break
                    continue
                # retract
                dy = np.einsum('idm,im->id', T, y)
                Xn = _norm(X + dy)
                nmax = _maxcos(Xn)
                ared = cmax - nmax
                rho = ared / (pred_red + 1e-300)
                if ared > 0:
                    X = Xn; cur = nmax
                    if nmax < bestLv:
                        bestLv = nmax; bestL = Xn.copy()
                    radius = min(rmax, radius * (1.5 if rho > 0.5 else 1.1))
                else:
                    radius *= 0.4
            return bestL, bestLv

        polished.sort(key=lambda t: t[0])
        for k in range(min(2, len(polished))):
            if time.time() - t0 > DEADLINE - 4:
                break
            Xc = polished[k][1]
            try:
                Xl, vl = lp_polish(Xc, budget_s=14.0)
            except Exception:
                continue
            if vl < glob_best_v:
                glob_best_v = vl; glob_best_X = Xl.copy()
    except Exception:
        pass

    # final safety
    glob_best_X = _norm(glob_best_X)
    return glob_best_X
