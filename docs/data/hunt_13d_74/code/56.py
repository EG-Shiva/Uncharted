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
    TRIAL_BUDGET = 120.0
    trial_deadline = t0 + TRIAL_BUDGET
    rng = np.random.default_rng(20261102)
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
    best = cands[int(np.argmin(vals))].copy()
    best_val = float(min(vals))

    IU = np.triu_indices(n, 1)
    I0 = IU[0].astype(np.int64)
    J0 = IU[1].astype(np.int64)

    def contact_info(X, thr=1e-7):
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        gmax = float(G.max())
        cnt = (G > gmax - thr).sum(axis=1).astype(int)
        return cnt, gmax, G

    def lbfgs_polish(X0, tau, maxiter, freeze=None):
        # freeze: bool array (n,) True means fixed
        if freeze is not None and np.any(freeze):
            free_idx = np.where(~freeze)[0]
            if len(free_idx) == 0:
                return X0
            nf = len(free_idx)
            # map full->reduced not needed; compute full X each eval
            Xfull_const = X0.copy()
            y0 = X0[free_idx].ravel().copy()
            def fun(yf):
                Yf = yf.reshape(nf, d)
                Y = Xfull_const.copy()
                Y[free_idx] = Yf
                r2 = (Y * Y).sum(axis=1, keepdims=True)
                r = np.sqrt(np.maximum(r2, 1e-24))
                r = np.maximum(r, 1e-12)
                X = Y / r
                G = X @ X.T
                g = G[I0, J0]
                mmax = float(g.max())
                e = np.exp((g - mmax) / tau)
                S = float(e.sum())
                if not np.isfinite(S) or S <= 0:
                    return float(mmax), np.zeros_like(yf)
                F = mmax + tau * np.log(S + 1e-300)
                w = e / (S + 1e-300)
                Wx = np.zeros_like(X)
                np.add.at(Wx, I0, w[:, None] * X[J0])
                np.add.at(Wx, J0, w[:, None] * X[I0])
                dot = (Wx * X).sum(axis=1, keepdims=True)
                gradY = (Wx - dot * X) / r
                return float(F), gradY[free_idx].ravel()
            try:
                res = minimize(fun, y0, jac=True, method="L-BFGS-B",
                               options={"maxiter": int(maxiter), "ftol": 1e-20,
                                        "gtol": 1e-7, "maxfun": int(maxiter * 8)})
                Y = Xfull_const.copy()
                Y[free_idx] = res.x.reshape(nf, d)
            except Exception:
                return X0
            rn = np.linalg.norm(Y, axis=1, keepdims=True)
            rn[rn == 0] = 1.0
            X = Y / rn
            if not np.all(np.isfinite(X)):
                return X0
            return _normalize(X)
        else:
            y0 = X0.ravel().copy()
            def fun(yf):
                Y = yf.reshape(n, d)
                r2 = (Y * Y).sum(axis=1, keepdims=True)
                r = np.sqrt(np.maximum(r2, 1e-24))
                r = np.maximum(r, 1e-12)
                X = Y / r
                G = X @ X.T
                g = G[I0, J0]
                mmax = float(g.max())
                e = np.exp((g - mmax) / tau)
                S = float(e.sum())
                if not np.isfinite(S) or S <= 0:
                    return float(mmax), np.zeros_like(yf)
                F = mmax + tau * np.log(S + 1e-300)
                w = e / (S + 1e-300)
                Wx = np.zeros_like(X)
                np.add.at(Wx, I0, w[:, None] * X[J0])
                np.add.at(Wx, J0, w[:, None] * X[I0])
                dot = (Wx * X).sum(axis=1, keepdims=True)
                gradY = (Wx - dot * X) / r
                return float(F), gradY.ravel()
            try:
                res = minimize(fun, y0, jac=True, method="L-BFGS-B",
                               options={"maxiter": int(maxiter), "ftol": 1e-20,
                                        "gtol": 1e-7, "maxfun": int(maxiter * 8)})
                Y = res.x.reshape(n, d)
            except Exception:
                return X0
            rn = np.linalg.norm(Y, axis=1, keepdims=True)
            rn[rn == 0] = 1.0
            X = Y / rn
            if not np.all(np.isfinite(X)):
                return X0
            return _normalize(X)

    def adam_lse(Xinit, tau, steps=200, freeze=None, lr0=0.004, lr1=0.0004):
        X = Xinit.copy()
        m = np.zeros_like(X)
        v = np.zeros_like(X)
        b1, b2, eps = 0.9, 0.999, 1e-8
        fr = np.zeros(n, dtype=bool) if freeze is None else freeze.astype(bool)
        for t in range(1, int(steps) + 1):
            if time.time() - t0 > LIM - 4:
                break
            G = X @ X.T
            g = G[I0, J0]
            mmax = float(g.max())
            e = np.exp((g - mmax) / tau)
            S = float(e.sum())
            if not np.isfinite(S) or S <= 0:
                break
            w = e / (S + 1e-300)
            Wx = np.zeros_like(X)
            np.add.at(Wx, I0, w[:, None] * X[J0])
            np.add.at(Wx, J0, w[:, None] * X[I0])
            dot = (Wx * X).sum(axis=1, keepdims=True)
            Gt = Wx - dot * X
            Gt[fr] = 0.0
            # annealed lr linear in log
            frac = (t - 1) / max(1, steps - 1)
            lr = lr0 * (lr1 / lr0) ** frac
            m = b1 * m + (1 - b1) * Gt
            v = b2 * v + (1 - b2) * (Gt * Gt)
            mh = m / (1 - b1 ** t)
            vh = v / (1 - b2 ** t)
            step = lr * mh / (np.sqrt(vh) + eps)
            # limit step norm per point
            X = _normalize(X - step)
            if not np.all(np.isfinite(X)):
                return Xinit
        return X

    def lp_polish(Xin, seconds, radius_init=1e-3):
        X = _normalize(Xin.copy())
        cur = _max_cos(X)
        radius = float(radius_init)
        start = time.time()
        iu = np.triu_indices(n, 1)
        while time.time() - start < seconds and time.time() - t0 < LIM - 2:
            if radius < 1e-14:
                break
            Cf = (X @ X.T)[iu]
            m = float(Cf.max())
            sel = Cf > m - max(20.0 * radius, 1e-9)
            if np.count_nonzero(sel) > 2500:
                thr = np.sort(Cf)[-2500]
                sel = Cf >= thr
            I = iu[0][sel]; J = iu[1][sel]
            c = Cf[sel]
            p = len(I)
            if p == 0:
                break
            nv = n * d + 1
            rows = np.repeat(np.arange(p), 2 * d + 1)
            cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
            valsA = np.concatenate([X[J], X[I], -np.ones((p, 1))], 1).ravel()
            try:
                A_ub = coo_matrix((valsA, (rows, cols)), shape=(p, nv)).tocsr()
                A_eq = coo_matrix((X.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            except Exception:
                radius *= 0.5
                continue
            obj = np.zeros(nv); obj[-1] = 1.0
            try:
                res = linprog(obj, A_ub=A_ub, b_ub=-c, A_eq=A_eq, b_eq=np.zeros(n),
                              bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
            except Exception:
                radius *= 0.5
                continue
            if res.status != 0:
                radius *= 0.5
                continue
            Y = X + res.x[:-1].reshape(n, d)
            Y = _normalize(Y)
            vv = _max_cos(Y)
            if vv < cur:
                X, cur = Y, vv
                radius = min(radius * 1.5, 0.02)
            else:
                radius *= 0.5
        return X, cur

    def hole_candidates(Xkeep, n_cand=20000):
        cand = _normalize(rng.standard_normal((n_cand, d)))
        S = cand @ Xkeep.T
        sc = S.max(axis=1)
        i1 = int(np.argmin(sc))
        return cand, sc, i1

    def perturb_tangent(y, theta):
        # random tangent direction
        u = rng.standard_normal(d)
        u = u - float(u @ y) * y
        nu = float(np.linalg.norm(u))
        if nu < 1e-12 or not np.isfinite(nu):
            return y
        u /= nu
        yp = float(np.cos(theta)) * y + float(np.sin(theta)) * u
        nu2 = float(np.linalg.norm(yp))
        if nu2 == 0 or not np.isfinite(nu2):
            return y
        return yp / nu2

    # archive keyed by sorted degree sequence
    try:
        cnt0, _, _ = contact_info(best, thr=1e-7)
        seen = set([tuple(sorted(cnt0.tolist()))])
        incumbent_seq = tuple(sorted(cnt0.tolist()))
    except Exception:
        seen = set()
        incumbent_seq = None

    # build rattler/hub lists
    try:
        cntB, _, GB = contact_info(best, thr=1e-7)
    except Exception:
        cntB = np.zeros(n, dtype=int)
        GB = best @ best.T
    rattlers = np.where(cntB == 0)[0]
    hubs = np.where(cntB >= 7)[0]
    if len(hubs) == 0:
        hubs = np.argsort(-cntB)[:8]
    else:
        hubs = hubs[np.argsort(-cntB[hubs])]
    # hub + highest-contact neighbour pairs
    Gtmp = best @ best.T
    np.fill_diagonal(Gtmp, -2.0)
    trial_victims = []
    for h in hubs:
        trial_victims.append((int(h),))
    for h in hubs:
        nb = int(np.argmax(Gtmp[int(h)]))
        if nb != int(h):
            # avoid duplicate single
            trial_victims.append((int(h), int(nb)))
    if len(trial_victims) == 0:
        trial_victims.append((int(np.argmax(cntB)),))
    # shuffled order deterministic
    order = rng.permutation(len(trial_victims))
    trial_victims = [trial_victims[i] for i in order]

    # surgery trials under 120s budget
    t_idx = 0
    n_trials_done = 0
    while time.time() < trial_deadline and time.time() - t0 < LIM - 25:
        victims = trial_victims[t_idx % len(trial_victims)]
        t_idx += 1
        n_trials_done += 1
        # refresh hub info periodically from best
        if n_trials_done % 6 == 0:
            try:
                cntB, _, Gtmp2 = contact_info(best, thr=1e-7)
                Gtmp = Gtmp2.copy()
                np.fill_diagonal(Gtmp, -2.0)
                # rebuild occasionally but keep deterministic shuffle
                hubs2 = np.where(cntB >= 7)[0]
                if len(hubs2) == 0:
                    hubs2 = np.argsort(-cntB)[:8]
                # update neighbour for next cycles lightly (not rebuilding list to save time)
            except Exception:
                pass
        try:
            Xcur = best.copy()
            keep_mask = np.ones(n, dtype=bool)
            keep_mask[list(victims)] = False
            Xkeep = Xcur[keep_mask]
            # sample 20000 candidates scored vs non-moving points
            cand, sc, i1 = hole_candidates(Xkeep, n_cand=20000)
            y1 = cand[i1].copy()
            theta1 = float(rng.uniform(0.05, 0.20))
            y1p = perturb_tangent(y1, theta1)
            if len(victims) == 1:
                Xcur[victims[0]] = y1p
                freeze = np.zeros(n, dtype=bool)
                freeze[victims[0]] = True
            else:
                # second hole conditioned on first (keep + y1)
                # compute second best diverse from cand pool
                s_keep = sc  # max vs Xkeep
                s_y1 = cand @ y1
                s2 = np.maximum(s_keep, s_y1)
                # exclude candidates too close to y1 to enforce diversity
                # pick best among those with dot to y1 < 0.9, else global second
                mask_div = (cand @ y1) < 0.9
                if np.any(mask_div):
                    tmp = s2.copy()
                    tmp[~mask_div] = 1e9
                    i2 = int(np.argmin(tmp))
                else:
                    # second best overall different from i1
                    tmp = s2.copy()
                    tmp[i1] = 1e9
                    i2 = int(np.argmin(tmp))
                y2 = cand[i2].copy()
                theta2 = float(rng.uniform(0.05, 0.20))
                y2p = perturb_tangent(y2, theta2)
                Xcur[victims[0]] = y1p
                Xcur[victims[1]] = y2p
                freeze = np.zeros(n, dtype=bool)
                freeze[list(victims)] = True
            # frozen annealed phase at tau=1e-3 (~200 steps Adam)
            Xcur = adam_lse(Xcur, tau=1e-3, steps=200, freeze=freeze, lr0=0.004, lr1=0.0004)
            # unfreeze polish at 3e-4, 8e-5 (short L-BFGS)
            remain = LIM - (time.time() - t0)
            if remain < 12:
                it_a, it_b = 40, 40
            else:
                it_a, it_b = 70, 70
            Xcur = lbfgs_polish(Xcur, tau=3e-4, maxiter=it_a, freeze=None)
            Xcur = lbfgs_polish(Xcur, tau=8e-5, maxiter=it_b, freeze=None)
            vcur = _max_cos(Xcur)
            cnt_cur, _, _ = contact_info(Xcur, thr=1e-7)
            seq = tuple(sorted(cnt_cur.tolist()))
            is_new = (seq not in seen)
            # archive
            seen.add(seq)
            need_final = is_new or (vcur <= best_val + 1e-6)
            if need_final:
                # 1-2 s trust-region LP polish
                budget_lp = 1.0
                if vcur < best_val + 1e-7:
                    budget_lp = 2.0
                    if LIM - (time.time() - t0) < 15:
                        budget_lp = 1.0
                Xlp, vlp = lp_polish(Xcur, seconds=budget_lp, radius_init=1e-3)
                # adopt LP result if not worse (allow tiny tol)
                if vlp <= vcur + 1e-12:
                    Xcur, vcur = Xlp, vlp
                    try:
                        cnt_lp, _, _ = contact_info(Xcur, thr=1e-7)
                        seen.add(tuple(sorted(cnt_lp.tolist())))
                    except Exception:
                        pass
            if vcur < best_val - 1e-12:
                best = Xcur.copy()
                best_val = float(vcur)
                incumbent_seq = seq
        except Exception:
            continue
        if n_trials_done > 300:
            break

    # final quick squeeze of best if time remains
    if time.time() - t0 < LIM - 4:
        try:
            Xf, vf = lp_polish(best, seconds=min(3.0, max(0.5, LIM - (time.time() - t0) - 2.0)), radius_init=5e-4)
            if vf < best_val:
                best, best_val = Xf, vf
        except Exception:
            pass

    best = _normalize(best)
    best[~np.isfinite(best)] = 0.0
    nr = np.linalg.norm(best, axis=1)
    for i in range(n):
        if nr[i] < 1e-12 or not np.isfinite(nr[i]):
            best[i] = _normalize(rng.standard_normal((1, d)))[0]
    best = _normalize(best)
    return best
