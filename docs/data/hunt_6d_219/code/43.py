import time
import numpy as np
from scipy.optimize import minimize, linprog
from scipy.sparse import coo_matrix

def _norm(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n = np.maximum(n, 1e-12)
    return X / n

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def _row_max(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return G.max(axis=1), G

def solve():
    t0 = time.time()
    BUDGET = 168.0
    n, d = 219, 6
    iu0, iu1 = np.triu_indices(n, 1)
    rng = np.random.RandomState(7)

    x_best = None
    best_val = 1e9
    for fn in ["best.npy", "record.npy"]:
        try:
            v = np.load(fn)
            v = np.asarray(v, dtype=float)
            if v.shape == (n, d):
                v = _norm(v)
                c = _max_cos(v)
                if c < best_val:
                    best_val = c
                    x_best = v.copy()
        except Exception:
            pass
    if x_best is None:
        x_best = _norm(rng.randn(n, d))
        best_val = _max_cos(x_best)

    def run_free_lbfgs(x0, beta, maxiter):
        y0 = x0.ravel().copy()
        def fun(y):
            Y = y.reshape(n, d)
            rn = np.maximum(np.linalg.norm(Y, axis=1, keepdims=True), 1e-12)
            X = Y / rn
            G = X @ X.T
            g = G[iu0, iu1]
            m = float(np.max(g))
            e = np.exp(beta * (g - m))
            S = float(np.sum(e)) + 1e-300
            loss = m + np.log(S) / beta
            w = e / S
            W = np.zeros((n, n))
            W[iu0, iu1] = w
            W = W + W.T
            GX = W @ X
            dot = (GX * X).sum(axis=1, keepdims=True)
            GY = (GX - dot * X) / rn
            return float(loss), GY.ravel()
        try:
            res = minimize(fun, y0, jac=True, method="L-BFGS-B",
                           options={"maxiter": maxiter, "ftol": 1e-12, "gtol": 1e-7, "maxfun": 20000})
            Y = res.x.reshape(n, d)
            return _norm(Y)
        except Exception:
            return _norm(x0)

    def run_frozen_lbfgs(Xfull, r, neigh, delta, beta, maxiter):
        # freeze row r, optimise other 218 with extra target <xr,xi> <= t-delta
        Xr = Xfull[r].copy()
        mask = np.ones(n, dtype=bool); mask[r] = False
        Xc0 = Xfull[mask].copy()
        # map original neigh indices to core indices
        orig_idx = np.arange(n)[mask]
        pos_of = {int(o): k for k, o in enumerate(orig_idx)}
        nlist = [pos_of[int(o)] for o in neigh if int(o) in pos_of]
        nlist = np.array(nlist, dtype=int)
        m = len(orig_idx)  # 218
        cu0, cu1 = np.triu_indices(m, 1)
        y0 = Xc0.ravel().copy()
        def fun(y):
            Y = y.reshape(m, d)
            rn = np.maximum(np.linalg.norm(Y, axis=1, keepdims=True), 1e-12)
            Xc = Y / rn
            Gcc = Xc @ Xc.T
            gcc = Gcc[cu0, cu1]
            gr = Xc[nlist] @ Xr  # (L,)
            grs = gr + delta
            # joint softmax
            mall = float(max(float(np.max(gcc)), float(np.max(grs)) if len(grs) else -2.0))
            e_cc = np.exp(beta * (gcc - mall))
            e_r = np.exp(beta * (grs - mall)) if len(grs) else np.zeros(0)
            S = float(np.sum(e_cc) + np.sum(e_r)) + 1e-300
            loss = mall + np.log(S) / beta
            w_cc = e_cc / S
            w_r = e_r / S if len(grs) else e_r
            W = np.zeros((m, m))
            W[cu0, cu1] = w_cc
            W = W + W.T
            GX = W @ Xc
            # add rattler pull/push
            if len(nlist):
                # GX[k] += w_r[t] * Xr
                for t, k in enumerate(nlist):
                    GX[k] += w_r[t] * Xr
            dot = (GX * Xc).sum(axis=1, keepdims=True)
            GY = (GX - dot * Xc) / rn
            return float(loss), GY.ravel()
        try:
            res = minimize(fun, y0, jac=True, method="L-BFGS-B",
                           options={"maxiter": maxiter, "ftol": 1e-12, "gtol": 1e-7, "maxfun": 20000})
            Y = res.x.reshape(m, d)
            Xc = _norm(Y)
        except Exception:
            Xc = _norm(Xc0)
        Xnew = np.empty((n, d))
        Xnew[mask] = Xc
        Xnew[r] = Xr
        return _norm(Xnew)

    def largest_void(core, nsearch=20000, seed=0):
        # core: (m,d) unit rows; find v minimizing max_i <v,xi>
        rs = np.random.RandomState(seed)
        C = rs.randn(nsearch, d)
        C = C / np.linalg.norm(C, axis=1, keepdims=True)
        S = C @ core.T
        vals = S.max(axis=1)
        order = np.argsort(vals)
        # take top 5 candidates, refine each
        best_v = C[order[0]].copy()
        best_s = float(vals[order[0]])
        cands = [C[order[k]] for k in range(min(5, len(order)))]
        for v0 in cands:
            y0 = v0.copy()
            def fun(y):
                yy = y.reshape(1, d)
                rn = max(float(np.linalg.norm(yy)), 1e-12)
                x = (yy / rn).ravel()
                g = core @ x
                mm = float(np.max(g))
                beta = 120.0
                e = np.exp(beta * (g - mm))
                Sm = float(np.sum(e)) + 1e-300
                loss = mm + np.log(Sm) / beta
                w = e / Sm
                gx = w @ core  # (d,)
                dot = float(gx @ x)
                gy = ((gx - dot * x) / rn).ravel()
                return float(loss), gy
            try:
                res = minimize(fun, y0, jac=True, method="L-BFGS-B",
                               options={"maxiter": 60, "ftol": 1e-14, "gtol": 1e-8})
                v = res.x.ravel()
                v = v / max(np.linalg.norm(v), 1e-12)
            except Exception:
                v = v0 / max(np.linalg.norm(v0), 1e-12)
            s = float(np.max(core @ v))
            if s < best_s:
                best_s = s
                best_v = v.copy()
        return best_v / max(np.linalg.norm(best_v), 1e-12), best_s

    def core_max_and_contacts(X, r, tol=1e-7):
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        overall = float(np.max(G))
        # core max excluding r
        mask = np.ones(n, dtype=bool); mask[r] = False
        Gc = G[np.ix_(mask, mask)]
        cmax = float(np.max(Gc))
        # contacts of r within tol
        row = G[r].copy()
        cr = int(np.sum(row >= overall - tol))
        # max of r
        maxr = float(np.max(row))
        return overall, cmax, cr, maxr

    def lp_polish(x, deadline, r_init=8e-4, max_steps=8):
        x = _norm(x.copy())
        best = _max_cos(x)
        bx = x.copy()
        radius = r_init
        steps = 0
        while time.time() < deadline and steps < max_steps and time.time() - t0 < BUDGET - 1.5:
            steps += 1
            if radius < 1e-14:
                break
            Gf = x @ x.T
            c = Gf[iu0, iu1]
            m = float(c.max())
            thr = m - max(18.0 * radius, 1e-9)
            sel = c > thr
            I = iu0[sel]; J = iu1[sel]; csel = c[sel]
            p = len(I)
            if p == 0:
                break
            if p > 3500:
                k2 = 3500
                part = np.argpartition(-csel, k2 - 1)[:k2]
                I, J, csel = I[part], J[part], csel[part]
                p = k2
            rows = np.repeat(np.arange(p), 2 * d + 1)
            cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d),
                                   np.full((p, 1), n * d)], 1).ravel()
            vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
            try:
                A_ub = coo_matrix((vals, (rows, cols)), shape=(p, n * d + 1)).tocsr()
                A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))),
                                  shape=(n, n * d + 1)).tocsr()
                obj = np.zeros(n * d + 1); obj[-1] = 1.0
                res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq, b_eq=np.zeros(n),
                              bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
            except Exception:
                radius *= 0.5
                continue
            if res.status != 0:
                radius *= 0.5
                continue
            y = _norm(x + res.x[:-1].reshape(n, d))
            v = _max_cos(y)
            if v < best:
                x, best = y, v
                bx = y.copy()
                radius = min(radius * 1.5, 0.03)
            else:
                radius *= 0.5
        return bx, best

    # identify initial rattler
    rm, _ = _row_max(x_best)
    r_idx = int(np.argmin(rm))
    x_cur = x_best.copy()
    cur_val = best_val

    deltas = [1e-5, 2e-5, 5e-5, 1e-4]
    outer = 0
    # main rattler-first homotopy loop
    while time.time() - t0 < BUDGET - 30 and outer < 12:
        outer += 1
        # re-identify rattler of current working point (lowest row max)
        rm_c, Gc_full = _row_max(x_cur)
        r = int(np.argmin(rm_c))
        overall_c = float(np.max(Gc_full))
        # core for void search
        mask = np.ones(n, dtype=bool); mask[r] = False
        core = x_cur[mask].copy()
        # move r to refined largest void
        seed_v = 1000 + outer * 77
        try:
            vstar, vs = largest_void(core, nsearch=20000, seed=seed_v)
        except Exception:
            vstar = x_cur[r].copy()
            vs = 1e9
        X_void = x_cur.copy()
        X_void[r] = vstar
        X_void = _norm(X_void)
        # quick free relax of void-inserted point to screen
        # frozen steps + release, sweeping delta
        for di, delta in enumerate(deltas):
            if time.time() - t0 > BUDGET - 30:
                break
            # 8-12 nearest neighbours of r in X_void (or evolving X)
            # use current X_void evolving
            Gtmp = X_void @ X_void.T
            np.fill_diagonal(Gtmp, -2.0)
            row = Gtmp[r].copy()
            nn = np.argsort(-row)[:12]
            # use 8-12: alternate 8,10,12
            L = [8, 10, 12, 9][di % 4]
            neigh = nn[:L]
            # freeze r, forced-activation optimisation of core
            beta_f = 260.0
            Xf = run_frozen_lbfgs(X_void, r, neigh, delta, beta_f, maxiter=70)
            # release r: free optimisation
            beta_rel = 300.0 if (di % 2 == 0) else 180.0
            Xr = run_free_lbfgs(Xf, beta_rel, maxiter=60)
            ov, cmax, cr, maxr = core_max_and_contacts(Xr, r, tol=1e-7)
            # acceptance: r integrated (>=5 contacts) and core dropped vs working overall
            # core drop measured against overall_c (before void move) or cur best core
            if cr >= 5 and cmax < overall_c - 1e-9:
                # finish accepted topology with trust-region LP polish
                Xp, vp = lp_polish(Xr, t0 + BUDGET - 14, r_init=7e-4, max_steps=6)
                ovp, cmaxp, crp, _ = core_max_and_contacts(Xp, r, tol=1e-7)
                # keep working point on new topology even if slightly worse overall,
                # but only if overall not exploded
                if ovp < overall_c + 0.002 and crp >= 4:
                    x_cur = Xp.copy()
                    overall_c = ovp
                    X_void = Xp.copy()
                if vp < best_val:
                    best_val = vp
                    x_best = Xp.copy()
            else:
                # still allow void point to evolve: keep Xf released version as next void base if promising
                # if overall improved, update void base
                if ov < _max_cos(X_void):
                    X_void = Xr.copy()
            # also try direct void without forcing: if r still 0-contact, keep void move
            if time.time() - t0 > BUDGET - 30:
                break
        # if no acceptance, perturb working point mildly around void to try new basin
        # tangential jitter of core + keep best fallback
        if time.time() - t0 < BUDGET - 30:
            # small random tangential kick to escape, then free relax
            sig = 0.02
            pert = rng.randn(n, d)
            pert -= (np.sum(pert * x_cur, axis=1, keepdims=True)) * x_cur
            Xt = _norm(x_cur + sig * pert)
            Xt = run_free_lbfgs(Xt, 220.0, maxiter=40)
            vt = _max_cos(Xt)
            if vt < _max_cos(x_cur) + 0.001:
                # check if this new point has better rattler integration
                rmt, _ = _row_max(Xt)
                # prefer points where min row max is higher (less rattling)
                if float(np.min(rmt)) > float(np.min(rm_c)) + 1e-6 or vt < _max_cos(x_cur):
                    x_cur = Xt.copy()
            if vt < best_val:
                best_val = vt
                x_best = Xt.copy()

    # final sharpen of incumbent
    try:
        x_best, best_val2 = lp_polish(x_best, t0 + BUDGET - 1.0, r_init=9e-4, max_steps=10)
        best_val = min(best_val, best_val2)
        # extra free + polish cycle
        if time.time() - t0 < BUDGET - 6:
            Xt = run_free_lbfgs(x_best, 320.0, maxiter=50)
            Xt2, v2 = lp_polish(Xt, t0 + BUDGET - 0.5, r_init=6e-4, max_steps=6)
            if v2 < best_val:
                best_val = v2
                x_best = Xt2.copy()
    except Exception:
        pass
    x_best = _norm(x_best)
    return x_best
