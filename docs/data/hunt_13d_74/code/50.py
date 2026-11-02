import time
import numpy as np
from scipy.optimize import minimize, linprog
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _normalize(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return X / n

def solve():
    t0 = time.time()
    LIM = 172.0
    rng = np.random.default_rng(7)
    n, d = 74, 13

    cands = []
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=np.float64)
            if a.shape == (n, d):
                a = _normalize(a)
                cands.append(a)
        except Exception:
            pass
    if not cands:
        cands.append(_normalize(rng.standard_normal((n, d))))
    vals = [_max_cos(a) for a in cands]
    bi = int(np.argmin(vals))
    best = cands[bi].copy()
    best_val = float(vals[bi])

    IU0, IU1 = np.triu_indices(n, 1)
    IU0 = IU0.astype(np.int64); IU1 = IU1.astype(np.int64)
    iu_cache = {}
    def _iu(nn):
        if nn not in iu_cache:
            iu = np.triu_indices(nn, 1)
            iu_cache[nn] = (iu[0].astype(np.int64), iu[1].astype(np.int64))
        return iu_cache[nn]

    def stats(X, eps=1e-9):
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        gmax = float(G.max())
        rowmax = G.max(axis=1)
        cnt = (G >= gmax - eps).sum(axis=1)
        spread = float(rowmax.max() - rowmax.min())
        ratt = int((cnt == 0).sum())
        return gmax, rowmax, cnt, spread, ratt

    def balanced_polish(X0, tau, tau2, a_w, maxiter):
        nn = X0.shape[0]
        dd = X0.shape[1]
        y0 = X0.ravel().copy()
        aw = a_w.copy()
        def fun(yf):
            Y = yf.reshape(nn, dd)
            r2 = (Y*Y).sum(axis=1, keepdims=True)
            r = np.sqrt(np.maximum(r2, 1e-24))
            r = np.maximum(r, 1e-12)
            X = Y / r
            G = X @ X.T
            np.fill_diagonal(G, -10.0)
            rowmax = G.max(axis=1)
            E = np.exp((G - rowmax[:, None]) / tau)
            np.fill_diagonal(E, 0.0)
            S = E.sum(axis=1)
            S = np.maximum(S, 1e-300)
            m = rowmax + tau * np.log(S)
            mmax = float(m.max())
            e2 = aw * np.exp((m - mmax) / tau2)
            S2 = float(e2.sum()) + 1e-300
            F = mmax + tau2 * np.log(S2)
            if not np.isfinite(F):
                return float(mmax), np.zeros_like(yf)
            P = e2 / S2
            Q = E / S[:, None]
            PQ = P[:, None] * Q
            W = PQ + PQ.T
            Wx = W @ X
            dot = (Wx * X).sum(axis=1, keepdims=True)
            gradY = (Wx - dot * X) / r
            return float(F), gradY.ravel()
        try:
            res = minimize(fun, y0, jac=True, method="L-BFGS-B",
                           options={"maxiter": int(maxiter), "ftol": 1e-20,
                                    "gtol": 1e-7, "maxfun": int(maxiter*8)})
            Y = res.x.reshape(nn, dd)
        except Exception:
            return X0, None
        rn = np.linalg.norm(Y, axis=1, keepdims=True)
        rn[rn == 0] = 1.0
        X = Y / rn
        if not np.all(np.isfinite(X)):
            return X0, None
        X = _normalize(X)
        # smoothed m for weight update
        try:
            G = X @ X.T
            np.fill_diagonal(G, -10.0)
            rowmax = G.max(axis=1)
            E = np.exp((G - rowmax[:, None]) / tau)
            np.fill_diagonal(E, 0.0)
            S = np.maximum(E.sum(axis=1), 1e-300)
            m = rowmax + tau * np.log(S)
        except Exception:
            m = None
        return X, m

    def lp_polish(X, budget, radius_init=6e-4):
        start = time.time()
        cur = _max_cos(X)
        radius = float(radius_init)
        nn = X.shape[0]
        iu0, iu1 = _iu(nn)
        nv = nn * d + 1
        while time.time() - start < budget and time.time() - t0 < LIM - 2:
            if radius < 1e-13:
                break
            Cfull = (X @ X.T)[(iu0, iu1)]
            m = float(Cfull.max())
            sel = Cfull > m - max(20.0*radius, 1e-9)
            if int(np.count_nonzero(sel)) > 2200:
                thr = float(np.sort(Cfull)[-2200])
                sel = Cfull >= thr
            I = iu0[sel].astype(np.int64); J = iu1[sel].astype(np.int64)
            c = Cfull[sel]
            p = len(I)
            if p == 0:
                break
            rows = np.repeat(np.arange(p), 2*d+1)
            cols = np.concatenate([I[:, None]*d + np.arange(d), J[:, None]*d + np.arange(d), np.full((p,1), nn*d)], 1).ravel()
            vals = np.concatenate([X[J], X[I], -np.ones((p,1))], 1).ravel()
            try:
                A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
                A_eq = coo_matrix((X.ravel(), (np.repeat(np.arange(nn), d), np.arange(nn*d))), shape=(nn, nv)).tocsr()
            except Exception:
                radius *= 0.5
                continue
            obj = np.zeros(nv); obj[-1] = 1.0
            try:
                res = linprog(obj, A_ub=A_ub, b_ub=-c, A_eq=A_eq, b_eq=np.zeros(nn),
                              bounds=[(-radius, radius)]*(nn*d)+[(None, None)], method="highs")
            except Exception:
                radius *= 0.5
                continue
            if res.status != 0:
                radius *= 0.5
                continue
            Y = _normalize(X + res.x[:-1].reshape(nn, d))
            if not np.all(np.isfinite(Y)):
                radius *= 0.5
                continue
            v = _max_cos(Y)
            if v < cur:
                X[:] = Y
                cur = v
                radius = min(radius*1.5, 0.02)
            else:
                radius *= 0.5
        return X, cur

    def greedy(X, steps, lr0):
        cur = _max_cos(X)
        lr = float(lr0)
        for it in range(int(steps)):
            if (it & 15) == 0 and time.time() - t0 > LIM - 2:
                break
            G = X @ X.T
            np.fill_diagonal(G, -2.0)
            gmax = float(G.max())
            delta = max(3.0*lr, 5e-7)
            mask = G > (gmax - delta)
            A = mask.astype(np.float64)
            Gr = A @ X
            dd = (Gr * X).sum(axis=1, keepdims=True)
            Gt = Gr - dd * X
            norms = np.linalg.norm(Gt, axis=1, keepdims=True)
            act = norms[:, 0] > 1e-14
            if not np.any(act):
                break
            step = np.zeros_like(Gt)
            step[act] = Gt[act] / (norms[act] + 1e-12) * lr
            Xn = _normalize(X - step)
            nv = _max_cos(Xn)
            if nv < cur:
                X[:] = Xn
                cur = nv
                lr *= 1.08
            else:
                lr *= 0.5
                if lr < 1e-11:
                    break
        return X, cur

    # initial stats
    g0, rm0, cnt0, spread0, ratt0 = stats(best)
    # balanced state tracking: accept equal max if more balanced
    Xcur = best.copy()
    a_w = np.full(n, 1.0/n)
    best_bal = best.copy()
    best_bal_spread = spread0
    best_bal_ratt = ratt0

    schedule = [(2e-3,1e-3),(1e-3,5e-4),(5e-4,2e-4),(2e-4,8e-5),(1e-4,4e-5),(5e-5,2e-5)]

    n_cycles = 4
    for cyc in range(n_cycles):
        if time.time() - t0 > LIM - 30:
            break
        # --- balanced phase ---
        for (tau, tau2) in schedule:
            if time.time() - t0 > LIM - 28:
                break
            Xcur, m_sm = balanced_polish(Xcur, tau, tau2, a_w, 50)
            # weight update every 50 steps
            if m_sm is not None and np.all(np.finite(m_sm)):
                mx = float(m_sm.max())
                a_w = a_w * np.exp(20.0*(m_sm - mx))
                a_w = np.maximum(a_w, 1e-8)
                s = float(a_w.sum())
                if s > 0 and np.isfinite(s):
                    a_w /= s
                else:
                    a_w = np.full(n, 1.0/n)
            g, rm, cnt, spread, ratt = stats(Xcur)
            if g < best_val - 1e-13:
                best = Xcur.copy(); best_val = g
            # accept balanced improvements even at equal global max
            better_bal = False
            if g <= best_val + 5e-7:
                if ratt < best_bal_ratt - 0 + (-1e-12):
                    better_bal = True
                elif ratt == best_bal_ratt and spread < best_bal_spread - 1e-12:
                    better_bal = True
                elif ratt < best_bal_ratt and spread < best_bal_spread + 0.02:
                    better_bal = True
            if better_bal:
                best_bal = Xcur.copy(); best_bal_spread = spread; best_bal_ratt = ratt
        # report implicitly via tracking
        # recruit rattlers with tiny kick before LP? keep Xcur
        # --- ordinary global-max LP polish ---
        if time.time() - t0 > LIM - 26:
            break
        Xlp = Xcur.copy()
        Xlp, vlp = lp_polish(Xlp, budget=1.6, radius_init=6e-4)
        Xlp, vlp = greedy(Xlp, 80, 0.0008)
        if vlp < best_val:
            best = Xlp.copy(); best_val = float(vlp)
        g2, rm2, cnt2, spread2, ratt2 = stats(Xlp)
        if g2 <= best_val + 5e-7:
            if ratt2 < best_bal_ratt or (ratt2 == best_bal_ratt and spread2 < best_bal_spread):
                best_bal = Xlp.copy(); best_bal_spread = spread2; best_bal_ratt = ratt2
        # choose start for next cycle: most balanced state, with hub-offload / rattler kick
        # accept intermediate that reduces rattlers even at equal max
        if ratt2 <= ratt + 1 and g2 <= g + 5e-7:
            Xcur = Xlp.copy()
        else:
            # stay on balanced trajectory but inject rattler recruitment noise
            Xcur = best_bal.copy()
        # explicit rattler recruitment kick: perturb zero-contact points slightly
        try:
            _, _, cntc, _, _ = stats(Xcur, eps=1e-7)
            ratt_idx = np.where(cntc == 0)[0]
            if len(ratt_idx) > 0:
                noise = rng.standard_normal((len(ratt_idx), d)) * 0.004
                Xcur[ratt_idx] += noise
                # relieve top hubs a little
                order = np.argsort(cntc)
                hubs = order[-3:]
                Xcur[hubs] += rng.standard_normal((len(hubs), d)) * 0.002
                Xcur = _normalize(Xcur)
        except Exception:
            pass
        # reset weights partially toward uniform to avoid collapse
        a_w = 0.5*a_w + 0.5*np.full(n, 1.0/n)
        a_w /= float(a_w.sum())

    # finish from most balanced state with standard LP squeeze
    Xf = best_bal.copy() if best_bal_ratt <= stats(best)[4] + 2 else best.copy()
    # also try best itself; pick better start
    cand_starts = [best.copy(), best_bal.copy()]
    # quick pick: polish both briefly? just polish best_bal then best
    remain = LIM - (time.time() - t0)
    if remain > 6:
        for st in cand_starts:
            if time.time() - t0 > LIM - 8:
                break
            # short egalitarian cool-down
            aw2 = np.full(n, 1.0/n)
            for (tau, tau2) in [(2e-4,8e-5),(8e-5,3e-5)]:
                if time.time() - t0 > LIM - 7:
                    break
                st, _ = balanced_polish(st, tau, tau2, aw2, 40)
            st, vs = lp_polish(st, budget=2.0, radius_init=5e-4)
            st, vs = greedy(st, 120, 0.0006)
            if vs < best_val:
                best = st.copy(); best_val = float(vs)
    # final LP squeeze on best
    if LIM - (time.time() - t0) > 4:
        Xf2 = best.copy()
        Xf2, vf2 = lp_polish(Xf2, budget=min(4.0, max(0.5, LIM-(time.time()-t0)-2.0)), radius_init=4e-4)
        Xf2, vf2 = greedy(Xf2, 100, 0.0004)
        if vf2 < best_val:
            best = Xf2.copy(); best_val = float(vf2)
    best = _normalize(best)
    return best
