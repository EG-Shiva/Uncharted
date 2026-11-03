import time
import numpy as np
from scipy.optimize import minimize, linprog
from scipy.sparse import coo_matrix

def solve():
    t0 = time.time()
    BUDGET = 168.0
    def remain():
        return BUDGET - (time.time() - t0)
    rng = np.random.RandomState(0)

    def normed(X):
        n = np.linalg.norm(X, axis=1, keepdims=True)
        n[n == 0] = 1.0
        return X / n

    def max_cos(X):
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        return float(np.max(G))

    # ---- load ----
    cands = []
    for fn in ("best.npy", "record.npy"):
        try:
            C = np.load(fn).astype(float)
            if C.shape == (250, 6) and np.all(np.isfinite(C)):
                C = normed(C)
                cands.append((max_cos(C), C))
        except Exception:
            pass
    if not cands:
        Xb = normed(rng.randn(250, 6))
        best_val = max_cos(Xb)
        best = Xb
    else:
        cands.sort(key=lambda t: t[0])
        best_val, best = cands[0][0], cands[0][1].copy()

    n, d = 250, 6

    # ---- smooth L-BFGS ----
    def smooth_opt(X0, beta, maxiter):
        Y0 = X0.copy()
        def fg(Yf):
            Y = Yf.reshape(n, d)
            nr = np.linalg.norm(Y, axis=1, keepdims=True)
            nr = np.maximum(nr, 1e-12)
            X = Y / nr
            G = X @ X.T
            np.fill_diagonal(G, -2.0)
            M = float(G.max())
            D = beta * (G - M)
            # clip very negative for exp stability (exp(-50)~2e-22)
            # use where to avoid underflow warning cost; direct exp fine
            E = np.exp(D)
            np.fill_diagonal(E, 0.0)
            S = float(E.sum())
            if not np.isfinite(S) or S <= 0:
                S = 1e-300
            loss = M + np.log(S) / beta
            W = E / S
            GX = 2.0 * (W @ X)
            dot = np.sum(GX * X, axis=1, keepdims=True)
            GY = (GX - dot * X) / nr
            return loss, GY.ravel()
        try:
            res = minimize(fg, Y0.ravel(), jac=True, method="L-BFGS-B",
                           options={"maxiter": maxiter, "ftol": 1e-20,
                                    "gtol": 1e-10, "maxcor": 15,
                                    "maxfun": maxiter * 6})
            Y = res.x.reshape(n, d)
            Y = normed(Y)
            return Y
        except Exception:
            return X0

    # ---- LP trust region polish ----
    iu0 = np.triu_indices(n, 1)
    def lp_polish(x, deadline, r0=3e-3, maxK=4500):
        nonlocal_best = [x.copy(), max_cos(x)]
        xb, vb = nonlocal_best[0], nonlocal_best[1]
        radius = r0
        # preallocate equality structure indices (tangent)
        while time.time() < deadline and radius > 1e-13:
            if remain() < 3:
                break
            Cmat = xb @ xb.T
            vals = Cmat[iu0]
            m = float(vals.max())
            # include pairs within band, cap to maxK tightest
            band = max(25.0 * radius, 8e-7)
            sel = vals > (m - band)
            cnt = int(np.count_nonzero(sel))
            if cnt > maxK:
                thr = np.sort(vals[sel])[-maxK]
                sel = vals >= thr
                cnt = int(np.count_nonzero(sel))
            if cnt == 0:
                break
            I = iu0[0][sel]; J = iu0[1][sel]; c = vals[sel]
            p = cnt
            nv = n * d + 1
            rows = np.repeat(np.arange(p), 2 * d + 1)
            # cols: I*d+k, J*d+k, last
            base_I = (I[:, None] * d + np.arange(d)).ravel()
            base_J = (J[:, None] * d + np.arange(d)).ravel()
            cols_IJ = np.empty(p * (2 * d + 1), dtype=np.int64)
            # interleave per row: [I-block, J-block, t]
            # build via loop-free reshape
            tmp = np.empty((p, 2 * d + 1), dtype=np.int64)
            tmp[:, :d] = I[:, None] * d + np.arange(d)
            tmp[:, d:2*d] = J[:, None] * d + np.arange(d)
            tmp[:, 2*d] = n * d
            cols = tmp.ravel()
            vXJ = xb[J].ravel()  # x_J for dx_I coeffs? check: row for (I,J): xJ*dxI + xI*dxJ
            vXI = xb[I].ravel()
            vvals = np.empty((p, 2 * d + 1))
            vvals[:, :d] = xb[J]
            vvals[:, d:2*d] = xb[I]
            vvals[:, 2*d] = -1.0
            A_ub = coo_matrix((vvals.ravel(), (rows, cols)), shape=(p, nv)).tocsr()
            # equality: xb[i]*dx[i]=0
            er = np.repeat(np.arange(n), d)
            ec = np.arange(n * d)
            ev = xb.ravel()
            A_eq = coo_matrix((ev, (er, ec)), shape=(n, nv)).tocsr()
            obj = np.zeros(nv); obj[-1] = 1.0
            try:
                res = linprog(obj, A_ub=A_ub, b_ub=-c, A_eq=A_eq,
                              b_eq=np.zeros(n),
                              bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                              method="highs",
                              options={"time_limit": max(1.0, deadline - time.time() - 0.2)})
            except Exception:
                radius *= 0.5
                continue
            if res.status != 0:
                radius *= 0.5
                continue
            y = xb + res.x[:-1].reshape(n, d)
            y = normed(y)
            v = max_cos(y)
            if v < vb - 1e-15:
                xb, vb = y, v
                radius = min(radius * 1.6, 0.012)
            else:
                radius *= 0.5
        return xb, vb

    # initial fast LP polish to squeeze record
    if remain() > 10:
        nb, nv = lp_polish(best, t0 + BUDGET - max(remain() - 18, 5), r0=2.5e-3)
        if nv < best_val:
            best, best_val = nb, nv

    # annealed smoothing chain
    sched = [(800.0, 150), (4000.0, 180), (15000.0, 180)]
    cur = best.copy()
    for beta, iters in sched:
        if remain() < 30:
            break
        cur = smooth_opt(cur, beta, iters)
        v = max_cos(cur)
        if v < best_val:
            best_val, best = v, cur.copy()
        # quick LP squeeze after each stage
        if remain() > 25:
            nb, nv = lp_polish(cur if v < best_val + 1e-9 else best,
                               time.time() + min(7.0, remain() - 20), r0=1.5e-3)
            if nv < best_val:
                best_val, best = nv, nb.copy()
            cur = nb

    # ---- basin hopping: shakes + kills ----
    trial = 0
    # precompute for void search: random candidate pool reused
    while remain() > 12:
        trial += 1
        mode = trial % 4
        Xs = best.copy()
        if mode == 0:
            # small global tangent shake
            sigma = 0.004 if (trial % 8 == 0) else 0.0015
            Xs = Xs + sigma * rng.randn(n, d)
            Xs = normed(Xs)
            beta, iters = (6000.0, 120)
        elif mode == 1:
            # targeted shake of worst points
            G = best @ best.T
            np.fill_diagonal(G, -2.0)
            mx = G.max(axis=1)
            order = np.argsort(-mx)
            k = 12 if trial % 8 == 1 else 5
            worst = order[:k]
            sig = 0.008
            for p in worst:
                xp = Xs[p]
                rv = rng.randn(d)
                rv -= np.dot(rv, xp) * xp
                nn = np.linalg.norm(rv)
                if nn < 1e-12:
                    continue
                rv /= nn
                Xs[p] = normed((xp + sig * rng.randn() * rv).reshape(1, -1))[0]
            # slight global jitter too
            Xs = normed(Xs + 0.0008 * rng.randn(n, d))
            beta, iters = (8000.0, 130)
        elif mode == 2:
            # kill worst point(s) and reinsert at largest void
            G = best @ best.T
            np.fill_diagonal(G, -2.0)
            mx = G.max(axis=1)
            order = np.argsort(-mx)
            nk = 1 if trial % 8 != 2 else 2
            cand_pool = normed(rng.randn(4000, d))
            for kk in range(nk):
                p = order[kk]
                # score pool by max dot to others (excluding p)
                others = np.delete(Xs, p, axis=0)
                # chunk for memory
                scores = np.empty(4000)
                for s in range(0, 4000, 1000):
                    scores[s:s+1000] = (cand_pool[s:s+1000] @ others.T).max(axis=1)
                bi = int(np.argmin(scores))
                Xs[p] = cand_pool[bi]
                # local refine void: few tangent steps maximizing min gap
                for _ in range(25):
                    dots = others @ Xs[p]
                    j = int(np.argmax(dots))
                    g = Xs[p] * float(dots[j]) - others[j]
                    # move opposite: decrease max dot -> direction -proj tangent
                    g = g - np.dot(g, Xs[p]) * Xs[p]
                    nn = np.linalg.norm(g)
                    if nn < 1e-12:
                        break
                    g /= nn
                    step = 0.002
                    Xs[p] = normed((Xs[p] - step * g).reshape(1, -1))[0]
            beta, iters = (5000.0, 120)
        else:
            # medium global shake, lower beta to allow topology change
            Xs = normed(Xs + 0.006 * rng.randn(n, d))
            beta, iters = (1500.0, 110)

        if remain() < 12:
            break
        try:
            cur2 = smooth_opt(Xs, beta, iters)
        except Exception:
            continue
        v2 = max_cos(cur2)
        # LP polish this trial (short)
        if remain() > 8:
            cur2p, v2p = lp_polish(cur2, time.time() + min(6.0, remain() - 6), r0=1.8e-3, maxK=4000)
        else:
            cur2p, v2p = cur2, v2
        if v2p < best_val - 1e-14:
            best_val, best = v2p, cur2p.copy()
        elif v2 < best_val - 1e-14:
            best_val, best = v2, cur2.copy()

    # final intensive polish
    if remain() > 4:
        nb, nv = lp_polish(best, t0 + BUDGET - 1.0, r0=1.2e-3, maxK=5000)
        if nv < best_val and np.all(np.isfinite(nb)):
            best, best_val = nb, nv

    best = normed(best)
    return best
