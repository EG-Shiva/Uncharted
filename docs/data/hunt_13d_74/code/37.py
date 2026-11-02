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
    rng = np.random.default_rng(777)
    n0, d = 74, 13

    cands = []
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=np.float64)
            if a.shape == (n0, d):
                a = _normalize(a)
                cands.append(a)
        except Exception:
            pass
    if not cands:
        cands.append(_normalize(rng.standard_normal((n0, d))))
    vals = [_max_cos(a) for a in cands]
    best = cands[int(np.argmin(vals))].copy()
    best_val = float(min(vals))

    _iu_cache = {}
    def _iu(n):
        if n not in _iu_cache:
            iu = np.triu_indices(n, 1)
            _iu_cache[n] = (iu[0].astype(np.int64), iu[1].astype(np.int64))
        return _iu_cache[n]

    def lse_polish(X0, tau, maxiter):
        n = X0.shape[0]
        I0, J0 = _iu(n)
        y0 = X0.ravel().copy()
        def fun(yf):
            Y = yf.reshape(n, d)
            r = np.sqrt(np.maximum((Y*Y).sum(axis=1, keepdims=True), 1e-24))
            r = np.maximum(r, 1e-12)
            X = Y / r
            G = X @ X.T
            g = G[I0, J0]
            mmax = float(g.max())
            e = np.exp((g - mmax) / tau)
            S = float(e.sum())
            if not np.finite(S) or S <= 0:
                return float(mmax), np.zeros_like(yf)
            F = mmax + tau * np.log(S + 1e-300)
            w = e / (S + 1e-300)
            Wx = np.zeros_like(X)
            Wx[I0] += w[:, None] * X[J0]
            Wx[J0] += w[:, None] * X[I0]
            dot = (Wx * X).sum(axis=1, keepdims=True)
            gradY = (Wx - dot * X) / r
            return float(F), gradY.ravel()
        try:
            res = minimize(fun, y0, jac=True, method="L-BFGS-B",
                           options={"maxiter": int(maxiter), "ftol": 1e-20,
                                    "gtol": 1e-8, "maxfun": int(maxiter*6)})
            Y = res.x.reshape(n, d)
        except Exception:
            return X0
        rn = np.linalg.norm(Y, axis=1, keepdims=True)
        rn[rn == 0] = 1.0
        X = Y / rn
        if not np.all(np.isfinite(X)):
            return X0
        return _normalize(X)

    def greedy(X, steps, lr0):
        cur = _max_cos(X)
        lr = float(lr0)
        for _ in range(int(steps)):
            if time.time() - t0 > LIM - 2:
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

    def anneal(X, taus):
        for tau, it in taus:
            if time.time() - t0 > LIM - 4:
                break
            X[:] = lse_polish(X, tau, it)
        return X

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

    def refine_candidates(Y0, Xc, steps=70, lr0=0.03):
        Y = Y0.copy()
        m = Y.shape[0]
        S0 = Y @ Xc.T
        curF = S0.max(axis=1)
        bestY = Y.copy()
        bestF = curF.copy()
        lr = float(lr0)
        ar = np.arange(m)
        for _ in range(int(steps)):
            Sm = Y @ Xc.T
            J = Sm.argmax(axis=1)
            f = Sm[ar, J]
            Gt = Xc[J] - f[:, None] * Y
            nnrm = np.linalg.norm(Gt, axis=1)
            step = np.zeros_like(Gt)
            nz = nnrm > 1e-14
            step[nz] = Gt[nz] / (nnrm[nz, None] + 1e-12)
            Yn = Y - lr * step
            Yn = _normalize(Yn)
            Fn = (Yn @ Xc.T).max(axis=1)
            imp = Fn < f - 1e-15
            if np.any(imp):
                Y[imp] = Yn[imp]
            upd = Fn < bestF
            if np.any(upd):
                bestY[upd] = Yn[upd]
                bestF[upd] = Fn[upd]
            lr *= 0.985
            if lr < 2e-4:
                pass
        return bestY, bestF

    # quick initial anneal of incumbent
    Xb = best.copy()
    Xb = anneal(Xb, [(0.004, 40), (0.0012, 40)])
    Xb, vb = greedy(Xb, 100, 0.002)
    if vb < best_val:
        best, best_val = Xb.copy(), float(vb)

    totals = [6, 10, 14]
    cycle = 0
    while time.time() - t0 < LIM - 26:
        if LIM - (time.time() - t0) < 13:
            break
        total = int(totals[cycle % len(totals)])
        cycle += 1
        if cycle > 14:
            break
        Xref = best.copy()
        G = Xref @ Xref.T
        np.fill_diagonal(G, -2.0)
        gmax = float(G.max())
        cnt = (G >= gmax - 1e-7).sum(axis=1)
        mx = G.max(axis=1)
        ratt = np.where(cnt == 0)[0].tolist()
        order_hub = list(np.argsort(-cnt))
        # build delete set: rattlers first (worst mx first), then hubs
        del_set = []
        if len(ratt) > 0:
            ratt_sorted = sorted(ratt, key=lambda i: -float(mx[i]))
            take = min(len(ratt_sorted), total)
            # for total>=8 take all rattlers; for 6 take worst 6
            # vary which rattlers kept by cycle to diversify
            if len(ratt_sorted) > total:
                # rotate choice deterministically
                shift = (cycle * 3) % len(ratt_sorted)
                rot = ratt_sorted[shift:] + ratt_sorted[:shift]
                del_set.extend([int(x) for x in rot[:take]])
            else:
                del_set.extend([int(x) for x in ratt_sorted[:take]])
        for h in order_hub:
            if len(del_set) >= total:
                break
            h = int(h)
            if h not in del_set:
                # require hub degree>=9 for first picks, else allow
                if cnt[h] >= 9 or len(del_set) >= total - 1 or total <= 6:
                    del_set.append(h)
        if len(del_set) < total:
            order = list(np.argsort(-mx))
            for i in order:
                i = int(i)
                if i not in del_set:
                    del_set.append(i)
                    if len(del_set) >= total:
                        break
        # number of hubs actually removed: ensure at least 2 hubs removed
        # if del_set is all rattlers, swap in 2 hubs
        n_hub_in = sum(1 for i in del_set if cnt[i] >= 5)
        if n_hub_in < 2:
            # replace last ones with top hubs
            for h in order_hub:
                h = int(h)
                if h not in del_set:
                    # replace best (smallest mx) rattler in set
                    # find member with smallest mx
                    scores = [(float(mx[i]), i) for i in del_set]
                    scores.sort()
                    del_set.remove(int(scores[0][1]))
                    del_set.append(h)
                    if sum(1 for i in del_set if cnt[i] >= 5) >= 2:
                        break
        mask = np.ones(n0, dtype=bool)
        mask[np.array(del_set, dtype=int)] = False
        Xcur = Xref[mask].copy()
        # successive greedy reinsertion with frozen core
        ok = True
        for k in range(total):
            if time.time() - t0 > LIM - 24:
                ok = False
                break
            M = 20000
            cand = rng.standard_normal((M, d))
            cand = _normalize(cand)
            S = cand @ Xcur.T
            sc = S.max(axis=1)
            part = np.argpartition(sc, 20)[:20]
            Y0 = cand[part].copy()
            bestY, bestF = refine_candidates(Y0, Xcur, steps=70, lr0=0.03)
            jbest = int(np.argmin(bestF))
            Xcur = np.vstack([Xcur, bestY[jbest:jbest+1]])
        if not ok or Xcur.shape[0] != n0:
            continue
        Xcur = _normalize(Xcur)
        # unfreeze: annealed LogSumExp + active-set minimax
        Xcur = anneal(Xcur, [(0.004, 45), (0.0012, 45)])
        Xcur, vcur = greedy(Xcur, 90, 0.0018)
        Xcur = anneal(Xcur, [(0.0006, 35)])
        Xcur, vcur2 = greedy(Xcur, 80, 0.0009)
        vcur = float(min(vcur, vcur2))
        if vcur < best_val - 1e-13:
            # extra squeeze
            Xt = anneal(Xcur.copy(), [(0.00035, 30)])
            Xt, vt = greedy(Xt, 60, 0.0006)
            if vt < vcur:
                Xcur, vcur = Xt, float(vt)
            if vcur < best_val:
                best, best_val = Xcur.copy(), float(vcur)

    # final intensive polish
    Xf = best.copy()
    remain = LIM - (time.time() - t0)
    if remain > 8:
        Xf = anneal(Xf, [(0.002, 55), (0.0007, 55)])
        Xf, vf = greedy(Xf, 180, 0.0012)
        if vf < best_val:
            best, best_val = Xf.copy(), float(vf)
        else:
            Xf = best.copy()
            vf = best_val
        lbudget = min(8.0, max(1.0, LIM - (time.time() - t0) - 3.0))
        if lbudget > 0.5:
            Xf, vf = lp_polish(Xf, budget=lbudget, radius_init=6e-4)
            Xf, vf = greedy(Xf, 100, 0.0005)
            if vf < best_val:
                best, best_val = Xf.copy(), float(vf)
    best = _normalize(best)
    return best
