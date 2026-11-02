import time
import numpy as np
from scipy.optimize import minimize, linprog
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _gen_sym(S):
    S = np.asarray(S, dtype=np.float64).reshape(6, 15)
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
    # renormalize rows for safety
    rn = np.linalg.norm(X, axis=1, keepdims=True)
    rn = np.maximum(rn, 1e-12)
    return X / rn

def _lp_polish(X, deadline, r_init=0.008, r_max=0.03, cap=2500, max_it=50):
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    n, d = X.shape
    best = _max_cos(X)
    best_x = X.copy()
    radius = r_init
    iu0, iu1 = np.triu_indices(n, 1)
    nv = n*d+1
    obj = np.zeros(nv); obj[-1] = 1.0
    it = 0
    while time.time() + 0.4 < deadline and radius > 1e-13 and it < max_it:
        it += 1
        G = X @ X.T
        c_all = G[iu0, iu1]
        m = float(c_all.max())
        thresh = max(20.0*radius, 1e-9)
        sel = c_all > m - thresh
        nnz = int(np.count_nonzero(sel))
        if nnz == 0:
            radius *= 0.5
            continue
        if nnz > cap:
            idx = np.where(sel)[0]
            order = np.argsort(-c_all[idx])[:cap]
            mask = np.zeros_like(sel, dtype=bool)
            mask[idx[order]] = True
            sel = mask
        I = iu0[sel]; J = iu1[sel]
        p = len(I)
        c_sel = c_all[sel]
        cols = np.concatenate([I[:,None]*d + np.arange(d), J[:,None]*d + np.arange(d), np.full((p,1), n*d)], axis=1).ravel()
        rows = np.repeat(np.arange(p), 2*d+1)
        vals = np.concatenate([X[J], X[I], -np.ones((p,1))], axis=1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((X.ravel(), (np.repeat(np.arange(n), d), np.arange(n*d))), shape=(n, nv)).tocsr()
        bounds = [(-radius, radius)]*(n*d) + [(None, None)]
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c_sel, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=bounds, method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = X + res.x[:-1].reshape(n, d)
        yn = np.linalg.norm(y, axis=1, keepdims=True)
        if not np.all(np.isfinite(yn)) or np.any(yn < 1e-12):
            radius *= 0.5
            continue
        y = y / yn
        v = _max_cos(y)
        if v < best:
            X = y; best = v; best_x = y.copy()
            radius = min(radius*1.5, r_max)
        else:
            radius *= 0.5
            if radius < 1e-12:
                break
    return best_x, best

def solve():
    t0 = time.time()
    HARD_END = t0 + 172.0
    rng = np.random.default_rng(0)
    n, d = 78, 15
    iu0, iu1 = np.triu_indices(n, 1)

    try:
        rec = np.asarray(np.load("record.npy"), dtype=np.float64)
        rec = rec / np.linalg.norm(rec, axis=1, keepdims=True)
        thresh = _max_cos(rec)
    except Exception:
        thresh = 0.168274222547

    def make_sym_fun(beta):
        def fun(sf):
            S = sf.reshape(6, 15)
            norms = np.linalg.norm(S, axis=1)
            norms = np.maximum(norms, 1e-12)
            X = np.empty((78, 15), dtype=np.float64)
            for k in range(6):
                s0 = S[k, :13]; tail = S[k, 13:]; nn = norms[k]
                for j in range(13):
                    X[k*13+j, :13] = np.roll(s0, j) / nn
                    X[k*13+j, 13:] = tail / nn
            G = X @ X.T
            gv = G[iu0, iu1]
            m = float(gv.max())
            e = np.exp(beta*(gv - m))
            Ssum = float(e.sum())
            L = m + np.log(Ssum)/beta
            w = e / Ssum
            W = np.zeros((78, 78), dtype=np.float64)
            W[iu0, iu1] = w
            W = W + W.T
            Gr = W @ X
            dot = np.sum(Gr*X, axis=1)
            rep = np.repeat(norms, 13)
            Da = (Gr - dot[:, None]*X) / rep[:, None]
            DS = np.zeros((6, 15), dtype=np.float64)
            for k in range(6):
                block = Da[k*13:(k+1)*13, :]
                DS[k, 13:] = block[:, 13:].sum(axis=0)
                acc = np.zeros(13, dtype=np.float64)
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
            m = float(gv.max())
            e = np.exp(beta*(gv - m))
            Ssum = float(e.sum())
            L = m + np.log(Ssum)/beta
            w = e / Ssum
            W = np.zeros((n, n), dtype=np.float64)
            W[iu0, iu1] = w
            W = W + W.T
            Gr = W @ X
            dot = np.sum(Gr*X, axis=1, keepdims=True)
            GY = (Gr - dot*X) / rn
            return float(L), GY.ravel().astype(np.float64)
        return fun

    def run_lbfgs(fun, x0, maxiter):
        try:
            r = minimize(fun, x0, jac=True, method="L-BFGS-B",
                         options={"maxiter": maxiter, "ftol": 1e-14, "gtol": 1e-10, "maxfun": maxiter*20})
            return r.x, float(r.fun)
        except Exception:
            return x0, 1e9

    # ---- Phase 1: symmetric screening at low beta ----
    Nsym = 24
    sym_screen = []  # (maxcos, S_opt)
    f50 = make_sym_fun(50.0)
    for s in range(Nsym):
        if time.time() > HARD_END - 90:
            break
        S0 = rng.standard_normal((6, 15))
        xopt, _ = run_lbfgs(f50, S0.ravel(), 80)
        Sopt = xopt.reshape(6, 15)
        try:
            Xg = _gen_sym(Sopt)
            v = _max_cos(Xg)
        except Exception:
            continue
        sym_screen.append((v, Sopt.copy()))
    sym_screen.sort(key=lambda t: t[0])
    # keep top 4
    top_sym = sym_screen[:4] if len(sym_screen) >= 4 else sym_screen

    # ---- Phase 2: refine top sym seeds at beta 200,800 ----
    best_sym_S = None
    best_sym_v = 1e9
    if len(top_sym) == 0:
        # fallback random
        best_sym_S = rng.standard_normal((6, 15))
        best_sym_v = _max_cos(_gen_sym(best_sym_S))
    else:
        f200s = make_sym_fun(200.0)
        f800s = make_sym_fun(800.0)
        for (v0, S0) in top_sym:
            if time.time() > HARD_END - 70:
                break
            x1, _ = run_lbfgs(f200s, S0.ravel(), 150)
            x2, _ = run_lbfgs(f800s, x1, 150)
            S2 = x2.reshape(6, 15)
            try:
                v2 = _max_cos(_gen_sym(S2))
            except Exception:
                continue
            if v2 < best_sym_v:
                best_sym_v = v2
                best_sym_S = S2.copy()
        if best_sym_S is None:
            best_sym_v, best_sym_S = top_sym[0][0], top_sym[0][1]

    X_sym = _gen_sym(best_sym_S)

    # ---- Phase 3: weight-3 discrete variant screening (full space) ----
    Ndisc = 8
    disc_cands = []
    f50f = make_full_fun(50.0)
    for t in range(Ndisc):
        if time.time() > HARD_END - 55:
            break
        X0 = np.zeros((n, d), dtype=np.float64)
        for i in range(n):
            idx = rng.choice(15, 3, replace=False)
            signs = rng.choice(np.array([-1.0, 1.0]), 3)
            X0[i, idx] = signs / np.sqrt(3.0)
        # small jitter to break exact symmetries
        X0 = X0 + 1e-3 * rng.standard_normal((n, d))
        X0 = X0 / np.linalg.norm(X0, axis=1, keepdims=True)
        xopt, _ = run_lbfgs(f50f, X0.ravel(), 80)
        Y = xopt.reshape(n, d)
        Y = Y / np.linalg.norm(Y, axis=1, keepdims=True)
        try:
            v = _max_cos(Y)
        except Exception:
            continue
        disc_cands.append((v, Y.copy()))
    disc_cands.sort(key=lambda t: t[0])
    top_disc = disc_cands[:2] if len(disc_cands) >= 2 else disc_cands

    # refine discrete top at higher beta (full)
    best_disc_X = None
    best_disc_v = 1e9
    if len(top_disc) > 0:
        f200f = make_full_fun(200.0)
        f800f = make_full_fun(800.0)
        for (v0, Y0) in top_disc:
            if time.time() > HARD_END - 40:
                break
            x1, _ = run_lbfgs(f200f, Y0.ravel(), 150)
            Y1 = (x1.reshape(n, d))
            Y1 = Y1 / np.linalg.norm(Y1, axis=1, keepdims=True)
            # second stage only if time
            if time.time() < HARD_END - 45:
                x2, _ = run_lbfgs(f800f, Y1.ravel(), 150)
                Y1 = x2.reshape(n, d)
                Y1 = Y1 / np.linalg.norm(Y1, axis=1, keepdims=True)
            v1 = _max_cos(Y1)
            if v1 < best_disc_v:
                best_disc_v = v1
                best_disc_X = Y1.copy()
    # ---- Phase 4: release symmetric to full ----
    X_rel_sym = X_sym.copy()
    X_rel_sym_v = _max_cos(X_rel_sym)
    if time.time() < HARD_END - 30:
        try:
            f200f2 = make_full_fun(200.0)
            x1, _ = run_lbfgs(f200f2, X_rel_sym.ravel(), 200)
            Y1 = x1.reshape(n, d)
            Y1 = Y1 / np.linalg.norm(Y1, axis=1, keepdims=True)
            v1 = _max_cos(Y1)
            X_rel_sym, X_rel_sym_v = Y1, v1
            if time.time() < HARD_END - 30:
                f800f2 = make_full_fun(800.0)
                x2, _ = run_lbfgs(f800f2, X_rel_sym.ravel(), 200)
                Y2 = x2.reshape(n, d)
                Y2 = Y2 / np.linalg.norm(Y2, axis=1, keepdims=True)
                v2 = _max_cos(Y2)
                X_rel_sym, X_rel_sym_v = Y2, v2
        except Exception:
            pass

    # choose whichever release scores best
    if best_disc_X is not None and best_disc_v < X_rel_sym_v:
        X_cur = best_disc_X
        v_cur = best_disc_v
    else:
        X_cur = X_rel_sym
        v_cur = X_rel_sym_v

    # also consider unreleased sym if release failed badly? keep best
    # ---- Phase 5: final LP polish ----
    # polish best candidate; if time, also polish runner-up and keep better
    try:
        if time.time() < HARD_END - 2:
            budget = HARD_END - 0.8
            Xp, vp = _lp_polish(X_cur, budget, r_init=0.008, r_max=0.03, cap=2500, max_it=50)
            if vp < v_cur:
                X_cur, v_cur = Xp, vp
    except Exception:
        pass
    # extra fine polish with small radius
    try:
        if time.time() < HARD_END - 3:
            Xp2, vp2 = _lp_polish(X_cur, HARD_END - 0.4, r_init=0.002, r_max=0.008, cap=2500, max_it=30)
            if vp2 < v_cur:
                X_cur, v_cur = Xp2, vp2
    except Exception:
        pass

    X_cur = X_cur / np.linalg.norm(X_cur, axis=1, keepdims=True)
    # safety: replace NaN
    if not np.all(np.isfinite(X_cur)):
        X_cur = _gen_sym(best_sym_S)
    return X_cur
