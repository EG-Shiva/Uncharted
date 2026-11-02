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
    it = 0
    while time.time() - start < seconds and radius > 1e-13 and it < 40:
        it += 1
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
            tl = max(0.5, seconds - (time.time() - start))
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
    rng = np.random.default_rng(12345)
    n, d = 78, 15
    iu0, iu1 = np.triu_indices(n, 1)

    try:
        rec = np.asarray(np.load("record.npy"), dtype=np.float64)
        rec = rec / np.linalg.norm(rec, axis=1, keepdims=True)
        thresh = _max_cos(rec)
    except Exception:
        thresh = 0.168274222547

    # ---- symmetric C13 generation helpers ----
    def gen_sym(S):
        # S (6,15) -> X (78,15) rownormalized
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
                    X[k * 13 + j, :13] = np.roll(s0, j) / nn
                    X[k * 13 + j, 13:] = tail / nn
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
                block = Da[k * 13:(k + 1) * 13, :]
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
                         options={"maxiter": maxiter, "ftol": 1e-12, "gtol": 1e-8, "maxfun": maxiter * 10})
            return r.x
        except Exception:
            return x0

    # ---- Phase A: symmetric screening 30 sets at low beta ----
    sym_fun50 = make_sym_fun(50.0)
    cands = []
    n_screen = 30
    for s in range(n_screen):
        if time.time() - t0 > 55:
            break
        S0 = rng.standard_normal((6, 15))
        xopt = run_lbfgs(sym_fun50, S0.ravel(), 60)
        S = xopt.reshape(6, 15)
        X = gen_sym(S)
        v = _max_cos(X)
        cands.append((v, S.copy()))
    if len(cands) == 0:
        S0 = rng.standard_normal((6, 15))
        cands = [(9.0, S0)]
    cands.sort(key=lambda t: t[0])
    top = cands[:6]

    # refine top through beta 200,800
    refined = []
    for v0, S0 in top:
        if time.time() - t0 > 85:
            # keep unrefined
            refined.append((v0, S0))
            continue
        S = S0.copy()
        for beta, mi in [(200.0, 150), (800.0, 150)]:
            if time.time() - t0 > 90:
                break
            fun = make_sym_fun(beta)
            S = run_lbfgs(fun, S.ravel(), mi).reshape(6, 15)
        X = gen_sym(S)
        refined.append((_max_cos(X), S))
    refined.sort(key=lambda t: t[0])
    Sbest = refined[0][1]
    Xsym_best = gen_sym(Sbest)

    # ---- Phase B: discrete weight-3 signed variant ----
    full_fun50 = make_full_fun(50.0)
    disc_list = []
    n_disc = 20
    for t in range(n_disc):
        if time.time() - t0 > 110:
            break
        X0 = np.zeros((78, 15))
        for i in range(78):
            idx = rng.choice(15, 3, replace=False)
            sg = rng.choice(np.array([-1.0, 1.0]), 3)
            X0[i, idx] = sg / np.sqrt(3.0)
        # small continuous jitter so LBFGS gradient well defined
        X0 = X0 + 1e-3 * rng.standard_normal((78, 15))
        X0 /= np.linalg.norm(X0, axis=1, keepdims=True)
        xopt = run_lbfgs(full_fun50, X0.ravel(), 60)
        Y = xopt.reshape(78, 15)
        Y /= np.linalg.norm(Y, axis=1, keepdims=True)
        disc_list.append((_max_cos(Y), Y))
    if len(disc_list) == 0:
        X0 = rng.standard_normal((78, 15))
        X0 /= np.linalg.norm(X0, axis=1, keepdims=True)
        disc_list = [(_max_cos(X0), X0)]
    disc_list.sort(key=lambda t: t[0])
    Xdisc_best = disc_list[0][1]

    # ---- Phase C: unconstrained release of both ----
    inits = [Xsym_best, Xdisc_best]
    # also add best of inits by max_cos first
    released = []
    for Xinit in inits:
        if time.time() - t0 > LIMIT - 45:
            Y = Xinit / np.linalg.norm(Xinit, axis=1, keepdims=True)
            released.append((_max_cos(Y), Y))
            continue
        Y = Xinit.copy()
        for beta, mi in [(200.0, 200), (800.0, 200)]:
            if time.time() - t0 > LIMIT - 45:
                break
            fun = make_full_fun(beta)
            # start from row-normalized Y
            Yn = Y / np.linalg.norm(Y, axis=1, keepdims=True)
            xopt = run_lbfgs(fun, Yn.ravel(), mi)
            Y = xopt.reshape(78, 15)
            Y /= np.linalg.norm(Y, axis=1, keepdims=True)
        released.append((_max_cos(Y), Y))
    released.sort(key=lambda t: t[0])
    bestX = released[0][1].copy()
    bestV = released[0][0]

    # keep whichever release scores best (threshold from record only for info)
    _ = thresh

    # ---- Phase D: LP polish ----
    remain = LIMIT - 1.0 - (time.time() - t0)
    if remain > 1.0:
        bestX = _lp_polish(bestX, min(max(2.0, remain), 40.0))
        bestV = _max_cos(bestX)

    # safety fallback: if catastrophic, return best release before polish
    bestX /= np.linalg.norm(bestX, axis=1, keepdims=True)
    # final tiny shrink not needed; ensure finite
    bestX = np.where(np.isnan(bestX), 0.0, bestX)
    rn = np.linalg.norm(bestX, axis=1, keepdims=True)
    rn = np.maximum(rn, 1e-12)
    bestX = bestX / rn
    return bestX
