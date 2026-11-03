import time
import numpy as np
from scipy.optimize import minimize, linprog
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _lp_polish(x, seconds, t0, budget_end):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    iu = np.triu_indices(n, 1)
    while time.time() - t0 < seconds and time.time() < budget_end and radius > 1e-14:
        cmat = (x @ x.T)[iu]
        m = cmat.max()
        sel = cmat > m - max(20 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p, nv = len(I), n * d + 1
        if p == 0:
            break
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        except Exception:
            break
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-cmat[sel], A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        v = _max_cos(y)
        if v < best:
            x, best, radius = y, v, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x, best

def solve():
    T0 = time.time()
    BUDGET = 168.0
    END = T0 + BUDGET
    rng = np.random.default_rng(0)

    # load starts
    try:
        x_best = np.load("best.npy").astype(float)
    except Exception:
        try:
            x_best = np.load("record.npy").astype(float)
        except Exception:
            x_best = rng.standard_normal(size=(94, 13))
    try:
        x_rec = np.load("record.npy").astype(float)
        x_rec /= np.linalg.norm(x_rec, axis=1, keepdims=True)
        xb_n = x_best / np.linalg.norm(x_best, axis=1, keepdims=True)
        if _max_cos(x_rec) < _max_cos(xb_n):
            x_best = x_rec.copy()
    except Exception:
        pass
    x_best = x_best / np.linalg.norm(x_best, axis=1, keepdims=True)
    best_val = _max_cos(x_best)
    # chain base for feasibility (start from record/best)
    try:
        base_rec = np.load("record.npy").astype(float)
        base_rec /= np.linalg.norm(base_rec, axis=1, keepdims=True)
    except Exception:
        base_rec = x_best.copy()
    chain_x = base_rec.copy()
    chain_val = _max_cos(chain_x)
    if best_val < chain_val:
        # keep chain at record per spec, but best tracks global
        pass

    n, d = x_best.shape
    # global best tracking (list for closure)
    state = {"xb": x_best.copy(), "vb": best_val}

    def try_update(Xc):
        v = _max_cos(Xc)
        if v < state["vb"] - 1e-15 and np.all(np.isfinite(Xc)):
            state["xb"] = Xc.copy()
            state["vb"] = v
        return v

    targets = [0.2439, 0.2437, 0.2435, 0.2432]
    sigmas = [0.01, 0.03, 0.08]
    w2 = 0.02
    lam = 0.5

    # hinge run helper
    def hinge_run(x_start, t, maxiter):
        t2 = t - 0.005
        y0 = x_start.ravel().copy()
        # closure for best tracking inside
        itcount = [0]
        def fg(flat):
            Y = flat.reshape(n, d)
            rn = np.linalg.norm(Y, axis=1)
            rn = np.maximum(rn, 1e-12)
            X = Y / rn[:, None]
            G = X @ X.T
            D1 = G - t
            D2 = G - t2
            # masks excluding diag
            # use where; diag of G is 1 > t so must zero
            S1 = np.where(D1 > 0, D1, 0.0)
            S2 = np.where(D2 > 0, D2, 0.0)
            np.fill_diagonal(S1, 0.0)
            np.fill_diagonal(S2, 0.0)
            f1 = 0.5 * float(np.sum(S1 * S1))
            f2 = 0.5 * float(np.sum(S2 * S2))
            f = f1 + w2 * f2
            # penalty on norms
            pen = lam * float(np.sum((rn - 1.0) ** 2))
            ftot = f + pen
            # grad
            Seff = S1 + w2 * S2
            gradX = 2.0 * (Seff @ X)
            dot = np.sum(gradX * X, axis=1, keepdims=True)
            gradY = (gradX - dot * X) / rn[:, None]
            # penalty grad: 2 lam (rn-1) * X
            gradY = gradY + (2.0 * lam * (rn - 1.0))[:, None] * X
            return ftot, gradY.ravel()
        def cb(flat):
            itcount[0] += 1
            if itcount[0] % 25 == 0:
                try:
                    Y = flat.reshape(n, d)
                    rn = np.linalg.norm(Y, axis=1, keepdims=True)
                    rn[rn == 0] = 1.0
                    Xc = Y / rn
                    try_update(Xc)
                except Exception:
                    pass
                # early abort if time up? can't abort easily; just continue
        try:
            res = minimize(fg, y0, jac=True, method="L-BFGS-B",
                           options={"maxiter": maxiter, "ftol": 1e-14, "gtol": 1e-7, "maxfun": maxiter * 4},
                           callback=cb)
            Yf = res.x.reshape(n, d)
            rn = np.linalg.norm(Yf, axis=1, keepdims=True)
            rn[rn == 0] = 1.0
            Xf = Yf / rn
            return Xf
        except Exception:
            return None

    # allocate hinge phase until END-35s
    HINGE_END = END - 35.0
    # order: for each target, for each sigma, 2 repeats
    run_idx = 0
    for ti, t in enumerate(targets):
        if time.time() > HINGE_END:
            break
        for si, sigma in enumerate(sigmas):
            if time.time() > HINGE_END:
                break
            for rep in range(2):
                if time.time() > HINGE_END:
                    break
                # time per run adaptive
                remain = HINGE_END - time.time()
                runs_left = (len(targets) - ti) * len(sigmas) * 2 - (si * 2 + rep)
                runs_left = max(runs_left, 1)
                # give at most ~8s per run, at least use fair share
                maxiter = 350
                if remain / runs_left < 3.0:
                    maxiter = 200
                elif remain / runs_left > 8.0:
                    maxiter = 450
                # perturb chain_x tangentially
                # deterministic noise via rng
                R = rng.standard_normal(size=(n, d))
                # tangent component
                dot = np.sum(R * chain_x, axis=1, keepdims=True)
                Rt = R - dot * chain_x
                # normalize tangent scale: Rt has variance ~ (d-1); scale to unit-ish by /sqrt(d-1)? just use sigma directly
                x0 = chain_x + sigma * Rt / np.sqrt(d - 1) * np.sqrt(d) * 0.9
                # for larger sigma variation second rep uses different scaling / base
                if rep == 1:
                    # second repeat: start from global best instead, or add extra isotropic
                    if rng.random() < 0.5:
                        Rb = rng.standard_normal(size=(n, d))
                        dotb = np.sum(Rb * state["xb"], axis=1, keepdims=True)
                        Rbt = Rb - dotb * state["xb"]
                        x0 = state["xb"] + sigma * Rbt / np.sqrt(d - 1) * np.sqrt(d) * 0.9
                    else:
                        # bigger kick on same chain
                        R2 = rng.standard_normal(size=(n, d))
                        dot2 = np.sum(R2 * chain_x, axis=1, keepdims=True)
                        R2t = R2 - dot2 * chain_x
                        x0 = chain_x + (sigma * 1.4) * R2t / np.sqrt(d - 1) * np.sqrt(d) * 0.9
                x0 /= np.linalg.norm(x0, axis=1, keepdims=True)
                Xf = hinge_run(x0, t, maxiter)
                run_idx += 1
                if Xf is None:
                    continue
                v = try_update(Xf)
                # if feasible for this t, update chain to continue lowering
                if v <= t:
                    # keep the best feasible for this t as new chain (lowest max)
                    if v < chain_val or chain_val > t:
                        chain_x = Xf.copy()
                        chain_val = v
                    # if we beat target comfortably, we could keep searching lower t;
                    # don't break, continue to exploit other sigmas at same t for even better
                    pass
        # after finishing sigmas for this t, if chain became feasible, it will seed next lower t
        # also if global best improved below t, sync chain
        if state["vb"] <= t and state["vb"] < chain_val:
            chain_x = state["xb"].copy()
            chain_val = state["vb"]

    # LP polish phase on best distinct candidate(s)
    # choose candidate: global best; also chain if feasible and distinct
    cand = state["xb"].copy()
    # polish global best with remaining time
    remain = END - time.time() - 2.0
    if remain > 4:
        # reserve a bit: polish in two chunks (chain then best) if chain feasible
        polished = cand
        try:
            # quick hinge tightening at best_val target to clean? skip, go LP
            polished, pv = _lp_polish(cand, seconds=min(remain * 0.85, remain - 2), t0=time.time(), budget_end=END - 1.0)
            pv2 = _max_cos(polished)
            if pv2 < state["vb"] and np.all(np.isfinite(polished)):
                state["xb"] = polished.copy()
                state["vb"] = pv2
        except Exception:
            pass
        # if chain is distinct and feasible-ish and time left, polish it too
        try:
            if time.time() < END - 6 and chain_val < best_val + 1e-6:
                diff = float(np.linalg.norm(chain_x - state["xb"]))
                if diff > 1e-4:
                    secs = END - time.time() - 2.0
                    if secs > 4:
                        pc, pvc = _lp_polish(chain_x, seconds=min(8.0, secs), t0=time.time(), budget_end=END - 1.0)
                        vv = _max_cos(pc)
                        if vv < state["vb"]:
                            state["xb"] = pc.copy()
                            state["vb"] = vv
        except Exception:
            pass

    x_out = state["xb"] / np.linalg.norm(state["xb"], axis=1, keepdims=True)
    # safety: never return worse than input best.npy
    try:
        x0 = np.load("best.npy").astype(float)
        x0 /= np.linalg.norm(x0, axis=1, keepdims=True)
        if _max_cos(x_out) > _max_cos(x0) + 1e-12:
            # if our search failed, return original (still deterministic)
            # but keep ours if essentially equal? return better
            x_out = x0
    except Exception:
        pass
    return x_out
