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
    rng = np.random.default_rng(42)
    d = 13
    # load incumbent
    cands = []
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=np.float64)
            if a.shape == (74, 13):
                a = _normalize(a)
                cands.append(a)
        except Exception:
            pass
    if not cands:
        cands.append(_normalize(rng.standard_normal((74, 13))))
    vals = [_max_cos(a) for a in cands]
    bi = int(np.argmin(vals))
    best = cands[bi].copy()
    best_val = float(vals[bi])

    iu_cache = {}
    def get_iu(n):
        if n not in iu_cache:
            iu = np.triu_indices(n, 1)
            iu_cache[n] = (iu[0].astype(np.int64), iu[1].astype(np.int64))
        return iu_cache[n]

    def lse_polish(X0, schedule):
        X0 = _normalize(np.asarray(X0, dtype=np.float64))
        n = X0.shape[0]
        I0, J0 = get_iu(n)
        Xcur = X0
        for tau, maxiter in schedule:
            if time.time() - t0 > LIM - 4:
                break
            y0 = Xcur.ravel().copy()
            tau_f = float(tau)
            def fun(yf):
                Y = yf.reshape(n, d)
                r = np.sqrt((Y * Y).sum(axis=1, keepdims=True))
                r = np.maximum(r, 1e-12)
                X = Y / r
                G = X @ X.T
                g = G[I0, J0]
                mmax = float(g.max())
                e = np.exp((g - mmax) / tau_f)
                S = float(e.sum())
                if not np.isfinite(S) or S <= 0:
                    return float(mmax), np.zeros_like(yf)
                F = mmax + tau_f * np.log(S + 1e-300)
                w = e / (S + 1e-300)
                Wx = np.zeros_like(X)
                Wx[I0] += w[:, None] * X[J0]
                Wx[J0] += w[:, None] * X[I0]
                dot = (Wx * X).sum(axis=1, keepdims=True)
                gradY = (Wx - dot * X) / r
                return float(F), gradY.ravel()
            try:
                res = minimize(fun, y0, jac=True, method="L-BFGS-B",
                               options={"maxiter": int(maxiter), "ftol": 1e-22,
                                        "gtol": 1e-7, "maxfun": int(maxiter * 8)})
                Y = res.x.reshape(n, d)
            except Exception:
                break
            rn = np.linalg.norm(Y, axis=1, keepdims=True)
            rn[rn == 0] = 1.0
            Xcur = Y / rn
            if not np.all(np.isfinite(Xcur)):
                Xcur = X0
                break
            Xcur = _normalize(Xcur)
        return _normalize(Xcur)

    def greedy(X, steps, lr0):
        cur = _max_cos(X)
        lr = float(lr0)
        for it in range(int(steps)):
            if time.time() - t0 > LIM - 3:
                break
            G = X @ X.T
            np.fill_diagonal(G, -2.0)
            gmax = float(G.max())
            delta = max(3.0 * lr, 5e-7)
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

    def insert_caps(X, n_add, M=4000):
        X = X.copy()
        for _ in range(int(n_add)):
            cand = _normalize(rng.standard_normal((int(M), d)))
            S = cand @ X.T
            sc = S.max(axis=1)
            j = int(np.argmin(sc))
            X = np.vstack([X, cand[j:j+1]])
        return X

    def prune_score_remove(X, target):
        X = X.copy()
        while X.shape[0] > int(target):
            ncur = X.shape[0]
            G = X @ X.T
            np.fill_diagonal(G, -2.0)
            gmax = float(G.max())
            mx = G.max(axis=1)
            cnt = (G > (gmax - 1e-7)).sum(axis=1).astype(np.float64)
            # sum of top-5 cosines per row
            srt = np.sort(G, axis=1)[:, -5:]
            s5 = srt.sum(axis=1)
            score = mx + 1e-4 * cnt + 1e-5 * s5
            worst = int(np.argmax(score))
            X = np.delete(X, worst, axis=0)
        return X

    def lp_polish(X, cur, seconds=4.0):
        start = time.time()
        n = X.shape[0]
        radius = 1e-3
        iu = np.triu_indices(n, 1)
        while time.time() - start < seconds and time.time() - t0 < LIM - 2:
            if radius < 1e-14:
                break
            Cf = (X @ X.T)[iu]
            m = float(Cf.max())
            sel = Cf > m - max(20.0 * radius, 1e-9)
            if np.count_nonzero(sel) > 2000:
                thr = np.sort(Cf)[-2000]
                sel = Cf >= thr
            I = iu[0][sel]; J = iu[1][sel]
            c = Cf[sel]
            p = len(I)
            if p == 0:
                break
            nv = n * d + 1
            base = np.empty((p, 2 * d + 1), dtype=np.int64)
            base[:, :d] = I[:, None] * d + np.arange(d)
            base[:, d:2 * d] = J[:, None] * d + np.arange(d)
            base[:, 2 * d] = n * d
            rows = np.repeat(np.arange(p), 2 * d + 1)
            cols = base.ravel()
            vls = np.empty((p, 2 * d + 1), dtype=np.float64)
            vls[:, :d] = X[J]
            vls[:, d:2 * d] = X[I]
            vls[:, 2 * d] = -1.0
            try:
                A_ub = coo_matrix((vls.ravel(), (rows, cols)), shape=(p, nv)).tocsr()
                A_eq = coo_matrix((X.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            except Exception:
                radius *= 0.5
                continue
            obj = np.zeros(nv); obj[-1] = 1.0
            try:
                res = linprog(obj, A_ub=A_ub, b_ub=-c, A_eq=A_eq, b_eq=np.zeros(n),
                              bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs",
                              options={"time_limit": 2.0})
            except Exception:
                radius *= 0.5
                continue
            if res.status != 0:
                radius *= 0.5
                continue
            Y = X + res.x[:-1].reshape(n, d)
            Y = _normalize(Y)
            v = _max_cos(Y)
            if v < cur:
                X, cur = Y, v
                radius = min(radius * 1.5, 0.02)
            else:
                radius *= 0.5
        return X, cur

    # initial quick polish of incumbent
    X0 = lse_polish(best, [(0.004, 150), (0.0015, 180), (0.0005, 200)])
    X0, v0 = greedy(X0, 200, 0.002)
    if v0 < best_val:
        best, best_val = X0, v0

    FULL = [(0.005, 100), (0.002, 130), (0.0008, 150)]
    QUICK = [(0.002, 110), (0.0007, 130)]
    TINY = [(0.0015, 90)]

    cycle = 0
    while time.time() - t0 < LIM - 12:
        remain = LIM - (time.time() - t0)
        if cycle % 2 == 0:
            # overshoot / prune
            if remain < 20:
                n_add = 2
            elif cycle % 6 == 0:
                n_add = 4
            elif cycle % 4 == 0:
                n_add = 3
            else:
                n_add = 2
            M = 4000 if remain > 30 else 2500
            try:
                Xbig = insert_caps(best, n_add, M=M)
            except Exception:
                cycle += 1
                continue
            Xbig = lse_polish(Xbig, FULL)
            if time.time() - t0 > LIM - 6:
                # still need to prune
                Xpr = prune_score_remove(Xbig, 74)
                vpr = _max_cos(Xpr)
                if vpr < best_val:
                    best, best_val = Xpr, vpr
                break
            Xpr = prune_score_remove(Xbig, 74)
            Xpr = lse_polish(Xpr, QUICK)
            Xpr, vpr = greedy(Xpr, 150, 0.002)
            if vpr < best_val - 1e-12:
                best, best_val = Xpr, vpr
            elif vpr < best_val + 2e-6 and rng.random() < 0.05:
                # occasional neutral move to change basin
                pass
        else:
            # undershoot / grow
            try:
                Xsmall = prune_score_remove(best, 70)
            except Exception:
                cycle += 1
                continue
            Xsmall = lse_polish(Xsmall, [(0.004, 100), (0.001, 120)])
            # staged reinsertion one at a time with polishing
            Xg = Xsmall
            for k in range(4):
                if time.time() - t0 > LIM - 8:
                    break
                Xg = insert_caps(Xg, 1, M=4000)
                Xg = lse_polish(Xg, TINY)
                Xg, _ = greedy(Xg, 60, 0.002)
            if Xg.shape[0] != 74:
                try:
                    if Xg.shape[0] > 74:
                        Xg = prune_score_remove(Xg, 74)
                    else:
                        Xg = insert_caps(Xg, 74 - Xg.shape[0], M=2500)
                except Exception:
                    cycle += 1
                    continue
            Xg = lse_polish(Xg, QUICK)
            Xg, vg = greedy(Xg, 150, 0.002)
            if vg < best_val - 1e-12:
                best, best_val = Xg, vg
        cycle += 1
        if cycle > 60:
            break

    # final tightening
    if time.time() - t0 < LIM - 6:
        best = lse_polish(best, [(0.0012, 200), (0.00045, 220)])
        best, best_val = greedy(best, 250, 0.0015)
    if time.time() - t0 < LIM - 3:
        try:
            best, best_val = lp_polish(best, best_val, seconds=3.0)
            best, best_val = greedy(best, 120, 0.001)
        except Exception:
            pass

    best = _normalize(best)
    # safety: ensure shape
    if best.shape != (74, 13):
        # fallback to incumbent padded/truncated (should not happen)
        b0 = _normalize(cands[bi])
        best = b0.copy()
    return best
