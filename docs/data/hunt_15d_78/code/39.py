import time
import numpy as np
from scipy.optimize import minimize, linprog
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _lp_polish(X, seconds, max_it=60):
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    n, d = X.shape
    best = _max_cos(X)
    start = time.time()
    radius = 2e-3
    iu0, iu1 = np.triu_indices(n, 1)
    it = 0
    while time.time() - start < seconds and radius > 1e-13 and it < max_it:
        it += 1
        if time.time() - start > seconds - 0.3:
            break
        Gf = X @ X.T
        c = Gf[iu0, iu1]
        m = c.max()
        sel = c > m - max(20 * radius, 2e-9)
        I, J = iu0[sel], iu1[sel]
        p = len(I)
        if p == 0:
            break
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([X[J], X[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((X.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c[sel], A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = X + res.x[:-1].reshape(n, d)
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        v = _max_cos(y)
        if v < best:
            X, best = y, v
            radius = min(radius * 1.5, 0.02)
        else:
            radius *= 0.5
    return X

def solve():
    t0 = time.time()
    LIMIT = 172.0
    rng = np.random.default_rng(101)
    n, d = 78, 15
    iu0, iu1 = np.triu_indices(n, 1)

    try:
        rec = np.asarray(np.load("record.npy"), dtype=np.float64)
        rec = rec / np.linalg.norm(rec, axis=1, keepdims=True)
        thresh = _max_cos(rec)
    except Exception:
        thresh = 0.168274222547

    def gen_sym(S):
        norms = np.linalg.norm(S, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        X = np.empty((78, 15), dtype=np.float64)
        for k in range(6):
            s0 = S[k, :13]
            tail = S[k, 13:]
            nn = norms[k, 0]
            for j in range(13):
                X[k*13+j, :13] = np.roll(s0, j) / nn
                X[k*13+j, 13:] = tail / nn
        return X

    def make_sym_fun(beta):
        def fun(sf):
            S = sf.reshape(6, 15)
            norms = np.linalg.norm(S, axis=1)
            norms = np.maximum(norms, 1e-12)
            X = np.empty((78, 15))
            for k in range(6):
                s0 = S[k, :13]
                tail = S[k, 13:]
                nn = norms[k]
                for j in range(13):
                    X[k*13+j, :13] = np.roll(s0, j) / nn
                    X[k*13+j, 13:] = tail / nn
            G = X @ X.T
            gv = G[iu0, iu1]
            m = gv.max()
            e = np.exp(beta * (gv - m))
            Ssum = e.sum()
            L = m + np.log(Ssum) / beta
            w = e / Ssum
            W = np.zeros((78, 78))
            W[iu0, iu1] = w
            W = W + W.T
            Gr = W @ X
            dot = np.sum(Gr * X, axis=1)
            rep = np.repeat(norms, 13)
            Da = (Gr - dot[:, None] * X) / rep[:, None]
            DS = np.zeros((6, 15))
            for k in range(6):
                block = Da[k*13:(k+1)*13, :]
                DS[k, 13:] = block[:, 13:].sum(axis=0)
                acc = np.zeros(13)
                for j in range(13):
                    acc += np.roll(block[j, :13], -j)
                DS[k, :13] = acc
            return float(L), DS.ravel().astype(np.float64)
        return fun

    def make_full_fun(beta):
        def fun(yf):
            Y = yf.reshape(n, d)
            rn = np.linalg.norm(Y, axis=1, keepdims=True)
            rn = np.maximum(rn, 1e-12)
            X = Y / rn
            G = X @ X.T
            gv = G[iu0, iu1]
            m = gv.max()
            e = np.exp(beta * (gv - m))
            Ssum = e.sum()
            L = m + np.log(Ssum) / beta
            w = e / Ssum
            W = np.zeros((n, n))
            W[iu0, iu1] = w
            W = W + W.T
            Gr = W @ X
            dot = np.sum(Gr * X, axis=1, keepdims=True)
            GY = (Gr - dot * X) / rn
            return float(L), GY.ravel().astype(np.float64)
        return fun

    def run_lbfgs(fun, x0, maxiter):
        try:
            r = minimize(fun, x0, jac=True, method="L-BFGS-B",
                         options={"maxiter": maxiter, "ftol": 1e-14, "gtol": 1e-6, "maxfun": maxiter*5})
            return r.x, True
        except Exception:
            return x0, False

    # ---- Phase A: C13 symmetric screening (from scratch) ----
    Nscreen = 28
    Slist = [rng.standard_normal((6, 15)) for _ in range(Nscreen)]
    scored = []
    for S in Slist:
        X = gen_sym(S)
        scored.append(_max_cos(X))
    order = np.argsort(scored)
    K = 6
    topS = [Slist[i] for i in order[:K]]

    best_sym_S = None
    best_sym_v = 1e9
    # cheap low-beta screen: short opt at beta=50
    for si, S0 in enumerate(topS):
        if time.time() - t0 > LIMIT - 110:
            break
        x = S0.ravel().copy()
        for beta, mi in [(50.0, 120), (200.0, 120), (800.0, 100)]:
            if time.time() - t0 > LIMIT - 105:
                break
            x, _ = run_lbfgs(make_sym_fun(beta), x, mi)
        Sopt = x.reshape(6, 15)
        v = _max_cos(gen_sym(Sopt))
        if v < best_sym_v:
            best_sym_v = v
            best_sym_S = Sopt.copy()

    candidates = []
    if best_sym_S is not None:
        candidates.append(gen_sym(best_sym_S))
        # second best symmetric also as candidate if time
        # add slight perturbation of best as diversity
        if len(topS) > 1:
            try:
                S2 = topS[1].ravel().copy()
                S2, _ = run_lbfgs(make_sym_fun(200.0), S2, 80)
                candidates.append(gen_sym(S2.reshape(6, 15)))
            except Exception:
                pass

    # ---- Phase B: weight-3 signed discrete variant ----
    # discrete pool: random weight-3 +-1/sqrt3 vectors
    Nw = 24
    wpool = []
    for _ in range(Nw):
        Xw = np.zeros((78, 15))
        for i in range(78):
            idx = rng.choice(15, size=3, replace=False)
            sgn = rng.choice(np.array([-1.0, 1.0]), size=3)
            Xw[i, idx] = sgn / np.sqrt(3.0)
        wpool.append(Xw)
    wsc = [_max_cos(Xw) for Xw in wpool]
    worder = np.argsort(wsc)
    # keep 2 best discrete starts
    for k in range(min(2, len(worder))):
        candidates.append(wpool[worder[k]])

    # ---- Phase C: unconstrained release (full L-BFGS) ----
    # screen candidates cheaply at low beta with few iters to pick top for full polish
    # quick evaluation: short beta=200, 60 iters each
    refined = []
    for Xc in candidates:
        if time.time() - t0 > LIMIT - 70:
            refined.append((Xc, _max_cos(Xc)))
            continue
        try:
            xf = Xc.ravel().copy()
            xf, _ = run_lbfgs(make_full_fun(200.0), xf, 80)
            Xf = (xf.reshape(n, d))
            Xf /= np.linalg.norm(Xf, axis=1, keepdims=True)
            refined.append((Xf, _max_cos(Xf)))
        except Exception:
            refined.append((Xc, _max_cos(Xc)))
    refined.sort(key=lambda t: t[1])
    # full release on top 2 with rising beta
    final_cands = []
    for ci in range(min(2, len(refined))):
        Xc, _ = refined[ci]
        if time.time() - t0 > LIMIT - 45:
            final_cands.append(Xc)
            continue
        try:
            xf = Xc.ravel().copy()
            for beta, mi in [(200.0, 150), (800.0, 150)]:
                if time.time() - t0 > LIMIT - 42:
                    break
                xf, _ = run_lbfgs(make_full_fun(beta), xf, mi)
            Xf = xf.reshape(n, d)
            Xf /= np.linalg.norm(Xf, axis=1, keepdims=True)
            final_cands.append(Xf)
        except Exception:
            final_cands.append(Xc)
    if not final_cands:
        # emergency random
        Xf = rng.standard_normal((78, 15))
        Xf /= np.linalg.norm(Xf, axis=1, keepdims=True)
        final_cands = [Xf]
    final_cands.sort(key=_max_cos)
    bestX = final_cands[0]
    bestV = _max_cos(bestX)

    # ---- Phase D: LP polish best release ----
    remain = LIMIT - (time.time() - t0) - 8.0
    if remain > 3:
        try:
            bestX = _lp_polish(bestX, min(remain, 30.0))
            bestV = _max_cos(bestX)
        except Exception:
            pass

    # ---- safety: threshold check, fallback to lab best if we failed ----
    try:
        fb = None
        fbv = 1e9
        for fn in ["best.npy", "record.npy"]:
            try:
                a = np.load(fn)
                a = np.asarray(a, dtype=np.float64)
                if a.shape != (78, 15):
                    continue
                a = a / np.linalg.norm(a, axis=1, keepdims=True)
                v = _max_cos(a)
                if v < fbv:
                    fbv = v
                    fb = a.copy()
            except Exception:
                continue
        # if our from-scratch best does not beat threshold, return fallback polished
        if fb is not None and bestV > min(fbv, thresh) - 1e-12:
            rem2 = LIMIT - (time.time() - t0) - 1.5
            if rem2 > 2:
                try:
                    fb = _lp_polish(fb, min(rem2, 6.0))
                    fbv = _max_cos(fb)
                except Exception:
                    pass
            if fbv < bestV:
                bestX = fb
                bestV = fbv
    except Exception:
        pass

    bestX = bestX / np.linalg.norm(bestX, axis=1, keepdims=True)
    bestX = np.where(np.isnan(bestX), 0.0, bestX)
    nr = np.linalg.norm(bestX, axis=1, keepdims=True)
    nr = np.maximum(nr, 1e-12)
    bestX = bestX / nr
    return bestX
