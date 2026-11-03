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

def solve():
    t0 = time.time()
    BUDGET = 168.0
    n, d = 219, 6
    iu0, iu1 = np.triu_indices(n, 1)
    rng0 = np.random.RandomState(0)

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
        x_best = _norm(rng0.randn(n, d))
        best_val = _max_cos(x_best)

    def per_max(X):
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        pm = G.max(axis=1)
        return pm, float(pm.max()), G

    def find_rattler(X):
        pm, m, _ = per_max(X)
        # zero contacts within 1e-9
        deg = np.sum((X @ X.T - np.eye(n)*2.0) > m - 1e-9, axis=1)
        # rattler = min degree, tie by min pm
        idx = int(np.lexsort((pm, deg))[0])
        return idx, m, pm

    def run_free_lbfgs(x0, beta, maxiter):
        y0 = x0.ravel().copy()
        def fun(y):
            Y = y.reshape(n, d)
            rn = np.linalg.norm(Y, axis=1, keepdims=True)
            rn = np.maximum(rn, 1e-12)
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

    def run_frozen_lbfgs(X0, r_idx, delta, beta, maxiter):
        # freeze row r_idx at X0[r_idx], optimise other 218
        r_vec = _norm(X0[r_idx:r_idx+1])[0].copy()
        core_idx = np.array([i for i in range(n) if i != r_idx], dtype=int)
        nc = n - 1
        Xc0 = X0[core_idx].copy()
        cu0, cu1 = np.triu_indices(nc, 1)
        y0 = Xc0.ravel().copy()
        def fun(y):
            Y = y.reshape(nc, d)
            rn = np.linalg.norm(Y, axis=1, keepdims=True)
            rn = np.maximum(rn, 1e-12)
            Xc = Y / rn
            Gc = Xc @ Xc.T
            g_cc = Gc[cu0, cu1]
            g_rc = Xc @ r_vec + delta  # shifted to enforce <= t-delta
            # joint softmax
            m1 = float(np.max(g_cc)) if g_cc.size else -2.0
            m2 = float(np.max(g_rc)) if g_rc.size else -2.0
            m = m1 if m1 > m2 else m2
            e1 = np.exp(beta * (g_cc - m))
            e2 = np.exp(beta * (g_rc - m))
            S = float(np.sum(e1) + np.sum(e2)) + 1e-300
            loss = m + np.log(S) / beta
            w1 = e1 / S
            w2 = e2 / S
            W = np.zeros((nc, nc))
            W[cu0, cu1] = w1
            W = W + W.T
            GX = W @ Xc + w2[:, None] * r_vec[None, :]
            dot = (GX * Xc).sum(axis=1, keepdims=True)
            GY = (GX - dot * Xc) / rn
            return float(loss), GY.ravel()
        try:
            res = minimize(fun, y0, jac=True, method="L-BFGS-B",
                           options={"maxiter": maxiter, "ftol": 1e-12, "gtol": 1e-7, "maxfun": 20000})
            Y = res.x.reshape(nc, d)
            Xc = _norm(Y)
        except Exception:
            Xc = _norm(Xc0)
        Xfull = np.zeros((n, d))
        Xfull[r_idx] = r_vec
        Xfull[core_idx] = Xc
        return _norm(Xfull)

    def lp_polish(x, deadline, r_init=1.0e-3, max_steps=12):
        x = _norm(x.copy())
        best = _max_cos(x)
        bx = x.copy()
        radius = r_init
        steps = 0
        while time.time() < deadline and steps < max_steps:
            steps += 1
            if radius < 1e-14:
                break
            Gf = x @ x.T
            c = Gf[iu0, iu1]
            m = float(c.max())
            thr = m - max(20.0 * radius, 1e-9)
            sel = c > thr
            I = iu0[sel]; J = iu1[sel]; csel = c[sel]
            p = len(I)
            if p == 0:
                break
            if p > 3500:
                k2 = 3500
                part = np.argpartition(-csel, k2-1)[:k2]
                I, J, csel = I[part], J[part], csel[part]
                p = k2
            rows = np.repeat(np.arange(p), 2*d+1)
            cols = np.concatenate([I[:, None]*d+np.arange(d), J[:, None]*d+np.arange(d), np.full((p,1), n*d)], 1).ravel()
            vals = np.concatenate([x[J], x[I], -np.ones((p,1))], 1).ravel()
            try:
                A_ub = coo_matrix((vals, (rows, cols)), shape=(p, n*d+1)).tocsr()
                A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n*d))), shape=(n, n*d+1)).tocsr()
                obj = np.zeros(n*d+1); obj[-1] = 1.0
                res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq, b_eq=np.zeros(n),
                              bounds=[(-radius, radius)]*(n*d)+[(None, None)], method="highs")
            except Exception:
                radius *= 0.5
                continue
            if res.status != 0:
                radius *= 0.5
                continue
            try:
                y = _norm(x + res.x[:-1].reshape(n, d))
            except Exception:
                radius *= 0.5
                continue
            v = _max_cos(y)
            if v < best:
                x, best = y, v
                bx = y.copy()
                radius = min(radius*1.5, 0.02)
            else:
                radius *= 0.5
        return bx, best

    def find_void(Xcore, seed, n_rand=20000):
        # Xcore: (nc,6) normalized
        rng = np.random.RandomState(seed)
        try:
            C = _norm(rng.randn(n_rand, d))
            D = C @ Xcore.T
            rm = D.max(axis=1)
            order = np.argsort(rm)
            top = order[:5]
            best_v = C[top[0]].copy()
            best_s = float(rm[top[0]])
            # refine each of top 5
            for ti in top:
                v = C[ti].copy()
                # local tangential descent on softmax max-dot
                beta = 80.0
                lr = 0.08
                cur_s = float(rm[ti])
                for it in range(150):
                    if time.time() - t0 > BUDGET - 20:
                        break
                    dd = Xcore @ v
                    m = float(dd.max())
                    e = np.exp(beta * (dd - m))
                    w = e / (e.sum() + 1e-300)
                    g = Xcore.T @ w  # 6
                    # project orthogonal to v, move opposite (away from neighbours)
                    gt = g - (g @ v) * v
                    gn = float(np.linalg.norm(gt))
                    if gn < 1e-14:
                        break
                    v2 = v - (lr / max(gn, 1e-12)) * gt * min(gn, 1.0)
                    # normalize
                    v2 = v2 / max(np.linalg.norm(v2), 1e-12)
                    m2 = float(np.max(Xcore @ v2))
                    if m2 < m:
                        v = v2
                        cur_s = m2
                        lr = min(lr*1.05, 0.15)
                    else:
                        lr *= 0.5
                        if lr < 1e-4:
                            break
                if cur_s < best_s:
                    best_s = cur_s
                    best_v = v.copy()
            return best_v / max(np.linalg.norm(best_v), 1e-12)
        except Exception:
            # fallback random
            v = rng.randn(d)
            return v / max(np.linalg.norm(v), 1e-12)

    def core_max_excluding(X, r_idx):
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        mask = np.ones(n, dtype=bool); mask[r_idx] = False
        Gs = G[np.ix_(mask, mask)]
        return float(np.max(Gs))

    # main rattler-first homotopy
    try:
        r_idx, m0, pm0 = find_rattler(x_best)
    except Exception:
        r_idx = int(np.argmin((x_best @ x_best.T).max(axis=1)))
    try:
        cur_core_max = core_max_excluding(x_best, r_idx)
    except Exception:
        cur_core_max = best_val

    deltas = [1e-4, 5e-5, 2e-5, 1e-5, 0.0]
    n_cycles = 4
    for cyc in range(n_cycles):
        if time.time() - t0 > BUDGET - 30:
            break
        try:
            # core for void search (exclude rattler)
            mask = np.ones(n, dtype=bool); mask[r_idx] = False
            Xcore = x_best[mask]
            void = find_void(Xcore, seed=20000 + cyc*101)
            Xtrial = x_best.copy()
            Xtrial[r_idx] = void
            Xtrial = _norm(Xtrial)
            # iterate deltas: freeze -> release
            for di, delta in enumerate(deltas):
                if time.time() - t0 > BUDGET - 25:
                    break
                # frozen-core barrier optimisation
                try:
                    Xfro = run_frozen_lbfgs(Xtrial, r_idx, delta, beta=300.0, maxiter=30)
                except Exception:
                    Xfro = Xtrial
                # release r: short free optimisation
                try:
                    beta_free = 250.0 if di < 4 else 400.0
                    Xrel = run_free_lbfgs(Xfro, beta_free, maxiter=20)
                except Exception:
                    Xrel = Xfro
                try:
                    mrel = _max_cos(Xrel)
                    pm, _, Grel = per_max(Xrel)
                    # contacts of r within 1e-7
                    r_cont = int(np.sum(Grel[r_idx] > mrel - 1e-7) - 0)  # diag is -2 so fine
                    # core max (excluding r)
                    cm = core_max_excluding(Xrel, r_idx)
                except Exception:
                    Xtrial = Xrel
                    continue
                # acceptance: need >=5 contacts and core max dropped, plus global improvement
                # always track global best regardless to never regress
                improved = mrel < best_val
                topology_ok = (r_cont >= 5) and (cm < cur_core_max - 1e-12)
                if improved and topology_ok:
                    # finish topology with LP polish
                    try:
                        Xp, vp = lp_polish(Xrel, deadline=time.time()+3.0, max_steps=8)
                    except Exception:
                        Xp, vp = Xrel, mrel
                    if vp < best_val:
                        x_best = _norm(Xp)
                        best_val = float(vp)
                        try:
                            cur_core_max = core_max_excluding(x_best, r_idx)
                        except Exception:
                            pass
                        Xtrial = x_best.copy()
                    elif mrel < best_val:
                        x_best = _norm(Xrel)
                        best_val = float(mrel)
                        cur_core_max = float(cm)
                        Xtrial = x_best.copy()
                    else:
                        Xtrial = Xrel
                elif improved:
                    # keep as fallback incumbent but don't count as topology success;
                    # still update to allow further progress, but require LP to confirm
                    try:
                        Xp, vp = lp_polish(Xrel, deadline=time.time()+2.0, max_steps=6)
                    except Exception:
                        Xp, vp = Xrel, mrel
                    if vp < best_val:
                        x_best = _norm(Xp)
                        best_val = float(vp)
                        try:
                            cur_core_max = core_max_excluding(x_best, r_idx)
                        except Exception:
                            pass
                        Xtrial = x_best.copy()
                    else:
                        # keep Xrel for next delta even if not global best (exploration)
                        Xtrial = Xrel
                else:
                    Xtrial = Xrel
                # after release, optionally move r to refined largest void of current core
                # (alternate step per assignment) if r still weak
                try:
                    pm2, _, _ = per_max(Xtrial)
                    # if r has <5 contacts, jump it to largest void before next delta
                    G2 = Xtrial @ Xtrial.T
                    np.fill_diagonal(G2, -2.0)
                    m2 = float(G2.max())
                    rc2 = int(np.sum(G2[r_idx] > m2 - 1e-7))
                    if rc2 < 5 and di < len(deltas)-1:
                        mask2 = np.ones(n, dtype=bool); mask2[r_idx] = False
                        void2 = find_void(_norm(Xtrial[mask2]), seed=30000 + cyc*1000 + di)
                        Xtrial[r_idx] = void2
                        Xtrial = _norm(Xtrial)
                except Exception:
                    pass
            # re-identify rattler for next cycle (weakest point)
            try:
                pmn, _, _ = per_max(x_best)
                # choose point with smallest row-max as new rattler target
                r_idx = int(np.argmin(pmn))
                cur_core_max = core_max_excluding(x_best, r_idx)
            except Exception:
                pass
        except Exception:
            continue

    # final squeeze on incumbent
    if time.time() - t0 < BUDGET - 8:
        try:
            Xf = run_free_lbfgs(x_best, 350.0, maxiter=30)
            vf = _max_cos(Xf)
            if vf < best_val:
                x_best, best_val = _norm(Xf), vf
        except Exception:
            pass
    if time.time() - t0 < BUDGET - 2:
        try:
            Xp, vp = lp_polish(x_best, deadline=t0+BUDGET-1.0, max_steps=20)
            if vp < best_val:
                x_best, best_val = Xp, vp
        except Exception:
            pass
    x_best = _norm(x_best)
    return x_best
