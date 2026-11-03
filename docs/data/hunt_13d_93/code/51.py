import time
import numpy as np

def solve():
    t0 = time.time()
    BUDGET = 168.0
    n, d = 93, 13
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
        return np.max(G, axis=1)

    # load baseline
    base = None
    base_val = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=np.float64)
            if a.shape != (n, d):
                continue
            a = _norm(a)
            v = _maxcos(a)
            if v < base_val:
                base_val = v
                base = a.copy()
        except Exception:
            pass
    if base is None:
        base = _norm(rng.standard_normal((n, d)))
        base_val = _maxcos(base)

    G0 = base @ base.T
    np.fill_diagonal(G0, -2.0)
    rowmax0 = np.max(G0, axis=1)
    # contacts at thresholds
    cmax0 = float(np.max(G0))
    slack0 = cmax0 - G0
    # per-point contacts at 1e-7 and 1e-9 (off-diag only)
    np.fill_diagonal(slack0, 1.0)
    cnt7 = np.sum(slack0 < 1e-7, axis=1)
    cnt9 = np.sum(slack0 < 1e-9, axis=1)
    order_loose = np.argsort(rowmax0)  # increasing: loosest first
    hub = int(np.argmax(cnt7 * 1000 + rowmax0))
    # if all zero, hub = argmax rowmax
    if np.all(cnt7 == 0):
        hub = int(np.argmax(rowmax0))

    g_best = base.copy()
    g_val = float(base_val)

    iu_all0, iu_all1 = np.triu_indices(n, 1)

    # ---------- helpers ----------
    def refine_void(v0, Y, steps=140):
        v = v0 / (np.linalg.norm(v0) + 1e-300)
        for s in range(steps):
            f = s / max(1, steps - 1)
            tau = 0.008 * ((0.001 / 0.008) ** f)
            lr = 0.08 * ((0.008 / 0.08) ** f)
            dd = Y @ v
            m0 = float(np.max(dd))
            e = np.exp((dd - m0) / tau)
            se = float(np.sum(e))
            if not np.isfinite(se) or se <= 0:
                break
            w = e / se
            g = w @ Y
            gt = g - float(g @ v) * v
            v = v - lr * gt
            nr = np.linalg.norm(v)
            if not np.isfinite(nr) or nr < 1e-12:
                break
            v = v / nr
            if time.time() - t0 > BUDGET - 4:
                break
        return v / (np.linalg.norm(v) + 1e-300)

    def adam_pnorm(X0, steps, p, shift, lr0, lr1):
        X = _norm(X0.copy())
        nn = X.shape[0]
        iu0_, iu1_ = np.triu_indices(nn, 1)
        m = np.zeros_like(X)
        v2 = np.zeros_like(X)
        b1, b2, eps = 0.9, 0.999, 1e-8
        tt = 0
        bv = _maxcos(X)
        bX = X.copy()
        for s in range(steps):
            if time.time() - t0 > BUDGET - 4:
                break
            f = s / max(1, steps - 1)
            lr = lr0 * ((lr1 / lr0) ** f) if lr0 != lr1 else lr0
            G = X @ X.T
            np.fill_diagonal(G, -10.0)
            cur = float(np.max(G))
            if cur < bv:
                bv = cur
                bX = X.copy()
            c = G[iu0_, iu1_]
            e = c - shift
            e[e < 0] = 0.0
            # weights e^(p-1)
            # avoid all-zero
            if np.all(e <= 0):
                # random tangent kick
                R = rng.standard_normal(X.shape) * 1e-4
                R -= np.einsum('ij,ij->i', R, X)[:, None] * X
                X = _norm(X + R)
                continue
            # scale to avoid underflow: divide by max
            emax = float(np.max(e))
            if emax <= 0:
                continue
            en = e / emax
            w = np.power(en, p - 1)
            S = float(np.sum(w))
            if not np.isfinite(S) or S <= 0:
                continue
            grad = np.zeros_like(X)
            np.add.at(grad, iu0_, (w[:, None] * X[iu1_]))
            np.add.at(grad, iu1_, (w[:, None] * X[iu0_]))
            grad /= S
            dot = np.einsum('ij,ij->i', grad, X)[:, None]
            gt = grad - dot * X
            tt += 1
            m = b1 * m + (1 - b1) * gt
            v2 = b2 * v2 + (1 - b2) * (gt * gt)
            mh = m / (1 - b1 ** tt)
            vh = v2 / (1 - b2 ** tt)
            step = lr * mh / (np.sqrt(vh) + eps)
            step -= np.einsum('ij,ij->i', step, X)[:, None] * X
            X = _norm(X - step)
        return bX, bv

    def hard_topk(X0, iters, K, step0):
        X = _norm(X0.copy())
        nn = X.shape[0]
        iu0_, iu1_ = np.triu_indices(nn, 1)
        P_ = len(iu0_)
        bv = _maxcos(X)
        bX = X.copy()
        step = float(step0)
        for it in range(iters):
            if time.time() - t0 > BUDGET - 4:
                break
            G = X @ X.T
            c = G[iu0_, iu1_]
            Kk = int(min(K, P_))
            if Kk < P_:
                idx = np.argpartition(-c, Kk)[:Kk]
            else:
                idx = np.arange(P_)
            I = iu0_[idx]; J = iu1_[idx]
            grad = np.zeros_like(X)
            np.add.at(grad, I, X[J])
            np.add.at(grad, J, X[I])
            grad /= float(len(idx))
            dot = np.einsum('ij,ij->i', grad, X)[:, None]
            gt = grad - dot * X
            ng = float(np.linalg.norm(gt))
            if not np.isfinite(ng) or ng < 1e-15:
                break
            direc = gt / ng
            Xt = _norm(X - step * direc * np.sqrt(nn))
            vv = _maxcos(Xt)
            if vv < bv:
                bv = vv
                bX = Xt.copy()
                X = Xt
                step = min(step * 1.15, 0.02)
            else:
                step *= 0.5
                if step < 1e-12:
                    step = float(step0) * 0.3
                    R = rng.standard_normal(X.shape) * 1e-4
                    R -= np.einsum('ij,ij->i', R, X)[:, None] * X
                    X = _norm(X + R)
                    if it > iters - 15:
                        break
        return bX, bv

    def lp_polish(X0, max_it=6):
        # short trust-region LP polish
        try:
            from scipy.optimize import linprog
            from scipy.sparse import coo_matrix
        except Exception:
            return X0, _maxcos(X0)
        X = _norm(X0.copy())
        nn, dd = X.shape
        dm = dd - 1
        Nvar = nn * dm
        iu0_, iu1_ = np.triu_indices(nn, 1)
        ar = np.arange(dm)
        D0 = np.zeros((dd, dm))
        for b in range(dm):
            D0[b + 1, b] = 1.0
        radius = 2e-3
        cur = _maxcos(X)
        bX = X.copy(); bv = cur
        c_obj = np.zeros(Nvar + 1); c_obj[-1] = 1.0
        for it in range(max_it):
            if time.time() - t0 > BUDGET - 3:
                break
            G = X @ X.T
            c_all = G[iu0_, iu1_]
            cmax = float(np.max(c_all))
            # active: slack<2e-3 or top 800
            slack = cmax - c_all
            sel = slack < 2e-3
            if int(np.sum(sel)) < 600:
                Kk = min(600, len(c_all))
                idx = np.argpartition(-c_all, Kk - 1)[:Kk]
                sel = np.zeros(len(c_all), dtype=bool)
                sel[idx] = True
            I = iu0_[sel]; J = iu1_[sel]
            csub = c_all[sel]
            p_ = len(I)
            if p_ == 0:
                break
            sgn = np.sign(X[:, 0]); sgn[sgn == 0] = 1.0
            U = X.copy(); U[:, 0] -= sgn
            denom = np.einsum('ij,ij->i', U, U)
            Barr = np.empty((nn, dd, dm))
            Barr[:] = D0[None, :, :]
            mask = denom > 1e-12
            if np.any(mask):
                Um = U[mask]; denm = denom[mask]; Uc = Um[:, 1:]
                prod = Um[:, :, None] * Uc[:, None, :]
                Barr[mask] -= 2.0 * prod / denm[:, None, None]
            BI = Barr[I]; BJ = Barr[J]
            XI = X[I]; XJ = X[J]
            CI = np.einsum('pab,pa->pb', BI, XJ)
            CJ = np.einsum('pab,pa->pb', BJ, XI)
            try:
                cols = np.empty((p_, 2 * dm + 1), dtype=np.int64)
                cols[:, :dm] = I[:, None] * dm + ar[None, :]
                cols[:, dm:2 * dm] = J[:, None] * dm + ar[None, :]
                cols[:, 2 * dm] = Nvar
                vals = np.empty((p_, 2 * dm + 1))
                vals[:, :dm] = CI; vals[:, dm:2 * dm] = CJ; vals[:, 2 * dm] = -1.0
                rows = np.repeat(np.arange(p_), 2 * dm + 1)
                A_ub = coo_matrix((vals.ravel(), (rows, cols.ravel())), shape=(p_, Nvar + 1)).tocsr()
                b_ub = -csub
                bounds = [(-radius, radius)] * Nvar + [(-1.0, cmax)]
                res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs',
                              options={'presolve': True})
            except Exception:
                break
            if not res.success:
                radius *= 0.5
                if radius < 1e-10:
                    break
                continue
            y = res.x[:Nvar].reshape(nn, dm)
            t_pred = float(res.x[-1])
            # retract: X + B y normalized
            dX = np.einsum('ndm,nm->nd', Barr, y)
            # tangent already, retraction
            Xn = _norm(X + dX)
            vn = _maxcos(Xn)
            # accept if improves or predicted improvement
            if vn < cur + 1e-12:
                X = Xn; cur = vn
                if vn < bv:
                    bv = vn; bX = Xn.copy()
                if t_pred < cur - 1e-9:
                    radius = min(radius * 1.5, 8e-3)
                else:
                    radius *= 0.7
            else:
                radius *= 0.5
                if radius < 5e-7:
                    break
        return bX, bv

    def void_insert_trial(Xkeep):
        mcur = Xkeep.copy()
        return mcur

    # build delete sets
    ks = [3, 5, 2, 8, 1, 13, 19]
    trials = []
    for k in ks:
        # variant A: k loosest
        delA = order_loose[:k].tolist()
        trials.append(delA)
        # variant B: hub + k-1 loosest (excluding hub)
        if k >= 1:
            loos_no_hub = [i for i in order_loose.tolist() if i != hub][:max(0, k - 1)]
            delB = [hub] + loos_no_hub
            # avoid duplicate
            if sorted(delB) != sorted(delA):
                trials.append(delB)
    # time per trial estimate
    for ti, del_idx in enumerate(trials):
        if time.time() - t0 > BUDGET - 12:
            break
        k = len(del_idx)
        # strict per-trial budget
        remain = BUDGET - 10 - (time.time() - t0)
        left_trials = len(trials) - ti
        per_trial = remain / max(1, left_trials)
        per_trial = max(3.0, min(14.0, per_trial))
        t_trial_start = time.time()
        try:
            mask = np.ones(n, dtype=bool)
            mask[np.array(del_idx, dtype=int)] = False
            Xcur = base[mask].copy()  # (93-k,13)
            # greedy reinsert
            for ins in range(k):
                if time.time() - t_trial_start > per_trial * 0.65:
                    break
                if time.time() - t0 > BUDGET - 8:
                    break
                m = Xcur.shape[0]
                # sample 60000
                S = rng.standard_normal((60000, d))
                S /= (np.linalg.norm(S, axis=1, keepdims=True) + 1e-300)
                scores = np.max(S @ Xcur.T, axis=1)
                # best 20
                bidx = np.argpartition(scores, 20)[:20]
                # refine each
                best_v = None; best_s = 1e9
                for bi in bidx:
                    vref = refine_void(S[bi], Xcur, steps=120)
                    s = float(np.max(Xcur @ vref))
                    if s < best_s:
                        best_s = s; best_v = vref
                Xcur = np.vstack([Xcur, best_v[None, :]])
                Xcur = _norm(Xcur)
                # shifted p-norm Adam few hundred steps
                Xcur, _ = adam_pnorm(Xcur, steps=120, p=64, shift=0.20, lr0=0.004, lr1=0.0008)
            # finish: p sweep
            if time.time() - t_trial_start > per_trial * 0.75:
                # quick eval
                vv = _maxcos(Xcur) if Xcur.shape[0] == n else 1e9
                if vv < g_val:
                    g_val = vv; g_best = _norm(Xcur.copy())
                continue
            # top-K hard-max + short LP
            # p-norm sweep 32,128 brief
            Xcur, _ = adam_pnorm(Xcur, steps=80, p=32, shift=0.20, lr0=0.003, lr1=0.0006)
            if time.time() - t_trial_start > per_trial * 0.85:
                vv = _maxcos(Xcur)
                if vv < g_val:
                    g_val = vv; g_best = Xcur.copy()
                continue
            Xcur, _ = adam_pnorm(Xcur, steps=80, p=128, shift=0.20, lr0=0.003, lr1=0.0006)
            Xcur, _ = hard_topk(Xcur, iters=70, K=150, step0=0.007)
            # LP polish (at most ~2-3s)
            Xcur, vv = lp_polish(Xcur, max_it=5)
            # also hard topk after LP
            Xcur2, vv2 = hard_topk(Xcur, iters=30, K=150, step0=0.004)
            if vv2 < vv:
                Xcur, vv = Xcur2, vv2
            if Xcur.shape[0] == n and vv < g_val and np.all(np.isfinite(Xcur)):
                g_val = float(vv); g_best = _norm(Xcur.copy())
        except Exception:
            continue

    # final safety: if no improvement, return original best
    g_best = _norm(g_best)
    # tiny shrink not needed; just normalize
    return g_best
