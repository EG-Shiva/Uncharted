import time
import numpy as np
from scipy.optimize import minimize, linprog
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _lp_polish(X, seconds):
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    n, d = X.shape
    best = _max_cos(X)
    start = time.time()
    radius = 2e-3
    iu0, iu1 = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-13:
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
            tl = max(0.3, seconds - (time.time() - start))
            res = linprog(obj, A_ub=A_ub, b_ub=-c[sel], A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs",
                          options={"time_limit": tl})
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
    rng = np.random.default_rng(7)
    n, d = 78, 15
    iu0, iu1 = np.triu_indices(n, 1)
    try:
        rec = np.asarray(np.load("record.npy"), dtype=np.float64)
        rec = rec / np.linalg.norm(rec, axis=1, keepdims=True)
        thresh = _max_cos(rec)
    except Exception:
        thresh = 0.168274222547

    def gen_sym(S):
        S = np.asarray(S, dtype=np.float64).reshape(6, 15)
        norms = np.linalg.norm(S, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        X = np.empty((78, 15), dtype=np.float64)
        for k in range(6):
            s0 = S[k, :13]
            tail = S[k, 13:]
            nn = norms[k, 0]
            for j in range(13):
                X[k * 13 + j, :13] = np.roll(s0, j) / nn
                X[k * 13 + j, 13:] = tail / nn
        return X

    def sym_fun(beta):
        def fun(sf):
            S = sf.reshape(6, 15)
            norms = np.linalg.norm(S, axis=1)
            nn = np.maximum(norms, 1e-12)
            X = np.empty((78, 15), dtype=np.float64)
            for k in range(6):
                s0 = S[k, :13]
                tail = S[k, 13:]
                den = nn[k]
                for j in range(13):
                    X[k * 13 + j, :13] = np.roll(s0, j) / den
                    X[k * 13 + j, 13:] = tail / den
            G = X @ X.T
            gv = G[iu0, iu1]
            m = gv.max()
            e = np.exp(beta * (gv - m))
            Ss = e.sum()
            L = m + np.log(Ss) / beta
            w = e / Ss
            W = np.zeros((78, 78), dtype=np.float64)
            W[iu0, iu1] = w
            W = W + W.T
            Gr = W @ X
            dot = np.sum(Gr * X, axis=1)
            rep = np.repeat(nn, 13)
            Da = (Gr - dot[:, None] * X) / rep[:, None]
            DS = np.zeros((6, 15), dtype=np.float64)
            for k in range(6):
                blk = Da[k * 13:(k + 1) * 13, :]
                DS[k, 13:] = blk[:, 13:].sum(axis=0)
                acc = np.zeros(13, dtype=np.float64)
                for j in range(13):
                    acc += np.roll(blk[j, :13], -j)
                DS[k, :13] = acc
            return float(L), DS.ravel()
        return fun

    def full_fun(beta):
        def fun(yf):
            Y = yf.reshape(n, d)
            rn = np.linalg.norm(Y, axis=1, keepdims=True)
            rn = np.maximum(rn, 1e-12)
            X = Y / rn
            G = X @ X.T
            gv = G[iu0, iu1]
            m = gv.max()
            e = np.exp(beta * (gv - m))
            Ss = e.sum()
            L = m + np.log(Ss) / beta
            w = e / Ss
            W = np.zeros((n, n), dtype=np.float64)
            W[iu0, iu1] = w
            W = W + W.T
            Gr = W @ X
            dot = np.sum(Gr * X, axis=1, keepdims=True)
            GY = (Gr - dot * X) / rn
            return float(L), GY.ravel()
        return fun

    def run_lbfgs(fun, x0, maxiter):
        try:
            r = minimize(fun, x0, jac=True, method="L-BFGS-B",
                         options={"maxiter": maxiter, "ftol": 1e-14, "gtol": 1e-6, "maxfun": maxiter * 5})
            return r.x, True
        except Exception:
            return x0, False

    # Phase A: screen symmetric seeds at low beta
    NSCREEN = 28
    scored = []
    f50 = sym_fun(50.0)
    for s in range(NSCREEN):
        if time.time() - t0 > LIMIT - 60:
            break
        S0 = rng.standard_normal((6, 15))
        x0 = S0.ravel()
        xf, _ = run_lbfgs(f50, x0, 60)
        S = xf.reshape(6, 15)
        X = gen_sym(S)
        v = _max_cos(X)
        scored.append((v, xf.copy()))
    scored.sort(key=lambda t: t[0])
    if len(scored) == 0:
        S0 = rng.standard_normal((6, 15))
        scored = [(1.0, S0.ravel())]
    # keep top 4
    top = [x for _, x in scored[:4]]

    # Phase B: refine top seeds at beta 200, 800
    f200s = sym_fun(200.0)
    f800s = sym_fun(800.0)
    refined = []
    for xf in top:
        if time.time() - t0 > LIMIT - 50:
            refined.append(xf)
            continue
        x1, _ = run_lbfgs(f200s, xf, 120)
        if time.time() - t0 > LIMIT - 45:
            refined.append(x1)
            continue
        x2, _ = run_lbfgs(f800s, x1, 120)
        refined.append(x2)
    # generate released starting points
    cands = []
    for xf in refined:
        X = gen_sym(xf.reshape(6, 15))
        cands.append((_max_cos(X), X))

    # Phase C: discrete weight-3 signed-permutation starts (full, no symmetry)
    NW3 = 8
    f80 = full_fun(80.0)
    w3cands = []
    for s in range(NW3):
        if time.time() - t0 > LIMIT - 45:
            break
        X0 = np.zeros((78, 15), dtype=np.float64)
        for i in range(78):
            idx = rng.choice(15, size=3, replace=False)
            sgn = rng.choice(np.array([-1.0, 1.0]), size=3)
            X0[i, idx] = sgn / np.sqrt(3.0)
            # tiny jitter to break exact ties
            X0[i] += 1e-3 * rng.standard_normal(15)
            X0[i] /= np.linalg.norm(X0[i])
        y0 = X0.ravel()
        y1, _ = run_lbfgs(f80, y0, 60)
        Y = (y1.reshape(n, d))
        Y /= np.linalg.norm(Y, axis=1, keepdims=True)
        w3cands.append((_max_cos(Y), Y))
    # merge: keep best w3 for further refine
    w3cands.sort(key=lambda t: t[0])
    # Phase D: full release refine of best symmetric + best w3
    # pick up to 3 to refine fully
    pool = []
    cands.sort(key=lambda t: t[0])
    if len(cands) > 0:
        pool.extend([c for _, c in cands[:2]])
    if len(w3cands) > 0:
        pool.append(w3cands[0][1])
    if len(pool) == 0:
        pool = [rng.standard_normal((78, 15))]
        pool[0] /= np.linalg.norm(pool[0], axis=1, keepdims=True)
    f200 = full_fun(200.0)
    f800 = full_fun(800.0)
    released = []
    for X0 in pool:
        if time.time() - t0 > LIMIT - 35:
            Xn = X0 / np.linalg.norm(X0, axis=1, keepdims=True)
            released.append(((_max_cos(Xn)), Xn))
            continue
        y0 = (X0 / np.linalg.norm(X0, axis=1, keepdims=True)).ravel()
        y1, _ = run_lbfgs(f200, y0, 150)
        Y1 = y1.reshape(n, d)
        Y1 /= np.linalg.norm(Y1, axis=1, keepdims=True)
        if time.time() - t0 > LIMIT - 32:
            released.append((_max_cos(Y1), Y1))
            continue
        y2, _ = run_lbfgs(f800, y1, 150)
        Y2 = y2.reshape(n, d)
        Y2 /= np.linalg.norm(Y2, axis=1, keepdims=True)
        released.append((_max_cos(Y2), Y2))
    released.sort(key=lambda t: t[0])
    bestX = released[0][1]
    bestV = released[0][0]
    # also consider unreleased symmetric best in case release diverged
    for v, X in cands:
        if v < bestV:
            bestV, bestX = v, X
    # Phase E: LP polish remaining time
    tleft = LIMIT - (time.time() - t0) - 3.0
    if tleft > 4.0:
        budget = min(tleft, 35.0)
        bestX = _lp_polish(bestX, budget)
        bestV = _max_cos(bestX)
    bestX = bestX / np.linalg.norm(bestX, axis=1, keepdims=True)
    # safety: remove NaN
    bestX = np.nan_to_num(bestX, nan=0.0, posinf=0.0, neginf=0.0)
    nr = np.linalg.norm(bestX, axis=1, keepdims=True)
    nr = np.maximum(nr, 1e-12)
    bestX = bestX / nr
    return bestX
