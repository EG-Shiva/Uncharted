import time
import numpy as np
from scipy.optimize import minimize, linprog
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _viol_stats(X, t):
    G = X @ X.T
    iu = np.triu_indices(X.shape[0], 1)
    v = G[iu] - t
    pos = v[v > 0]
    cnt = int(pos.shape[0])
    ssum = float(np.sum(pos * pos)) if cnt else 0.0
    mx = float(G[iu].max())
    return cnt, ssum, mx

def _tangent_perturb(X, sigma, rng):
    N = rng.standard_normal(size=X.shape)
    dot = (N * X).sum(axis=1, keepdims=True)
    Nt = N - dot * X
    Y = X + sigma * Nt
    Y /= np.linalg.norm(Y, axis=1, keepdims=True)
    return Y

def _hinge_run(x_start, t, maxiter, rng_seed_off=0):
    n, d = x_start.shape
    w2 = 0.02
    t2 = t - 0.005
    lam = 0.5
    y0 = x_start.ravel().copy()
    def fg(flat):
        Y = flat.reshape(n, d)
        rn = np.linalg.norm(Y, axis=1)
        rn = np.maximum(rn, 1e-12)
        X = Y / rn[:, None]
        G = X @ X.T
        D1 = G - t
        D2 = G - t2
        S1 = np.where(D1 > 0, D1, 0.0)
        S2 = np.where(D2 > 0, D2, 0.0)
        np.fill_diagonal(S1, 0.0)
        np.fill_diagonal(S2, 0.0)
        f1 = 0.5 * float(np.sum(S1 * S1))
        f2 = 0.5 * float(np.sum(S2 * S2))
        f = f1 + w2 * f2
        pen = lam * float(np.sum((rn - 1.0) ** 2))
        ftot = f + pen
        Seff = S1 + w2 * S2
        gradX = 2.0 * (Seff @ X)
        dot = np.sum(gradX * X, axis=1, keepdims=True)
        gradY = (gradX - dot * X) / rn[:, None]
        gradY = gradY + (2.0 * lam * (rn - 1.0))[:, None] * X
        return ftot, gradY.ravel()
    try:
        res = minimize(fg, y0, method='L-BFGS-B', jac=True,
                       options={'maxiter': maxiter, 'maxfun': 15000,
                                'ftol': 1e-22, 'gtol': 1e-9, 'maxcor': 20})
        Y = res.x.reshape(n, d)
    except Exception:
        Y = y0.reshape(n, d)
    rn = np.linalg.norm(Y, axis=1, keepdims=True)
    rn[rn == 0] = 1.0
    X = Y / rn
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    return X

def _lp_polish(x, deadline):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    iu = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        cmat = (x @ x.T)[iu]
        m = float(cmat.max())
        sel = cmat > m - max(20.0 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        if p == 0:
            break
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        except Exception:
            break
        obj = np.zeros(nv)
        obj[-1] = 1.0
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
    BUDGET = 176.0
    HINGE_END = T0 + 132.0
    FINAL_END = T0 + BUDGET - 2.0
    rng = np.random.default_rng(0)
    # load starts
    rec = None
    bst = None
    for fn in ["record.npy", "best.npy"]:
        try:
            a = np.load(fn).astype(float)
            a = a / np.linalg.norm(a, axis=1, keepdims=True)
            if a.shape == (94, 13) and np.all(np.isfinite(a)):
                if fn == "record.npy":
                    rec = a.copy()
                if fn == "best.npy":
                    bst = a.copy()
        except Exception:
            pass
    if rec is None and bst is None:
        r = rng.standard_normal(size=(94, 13))
        r /= np.linalg.norm(r, axis=1, keepdims=True)
        rec = r.copy()
        bst = r.copy()
    if rec is None:
        rec = bst.copy()
    if bst is None:
        bst = rec.copy()
    rec_val = _max_cos(rec)
    bst_val = _max_cos(bst)
    if bst_val < rec_val:
        global_best = bst.copy()
        global_val = bst_val
    else:
        global_best = rec.copy()
        global_val = rec_val
    base_record = rec.copy()

    targets = [0.2439, 0.2437, 0.2435, 0.2432]
    sigmas = [0.01, 0.03, 0.08]
    # track least violating for fallback + feasible chain
    least_viol = None
    least_key = (10**18, 10**18)
    feas_chain = None  # feasible config for next lower t
    feas_list = []
    maxiter_map = {0.2439: 300, 0.2437: 350, 0.2435: 400, 0.2432: 450}

    def note_candidate(X, t):
        nonlocal global_best, global_val, least_viol, least_key
        v = _max_cos(X)
        if v < global_val and np.all(np.isfinite(X)):
            global_val = v
            global_best = X.copy()
        cnt, ssum, mx = _viol_stats(X, t)
        key = (cnt, ssum)
        if key < least_key:
            least_key = key
            least_viol = X.copy()
        return cnt, ssum, v

    # initial notes
    note_candidate(global_best.copy(), targets[0])

    run_id = 0
    for ti, t in enumerate(targets):
        if time.time() > HINGE_END:
            break
        maxiter = maxiter_map.get(t, 350)
        # reps: cycle sigmas, alternate base record / global best
        reps = 5 if ti < 2 else 4
        found_feas_this_t = False
        for rep in range(reps):
            if time.time() > HINGE_END - 2.0:
                break
            # chain continuation first rep if available
            if rep == 0 and feas_chain is not None:
                x_start = feas_chain.copy()
                # tiny tangent jitter to escape plateau
                x_start = _tangent_perturb(x_start, 0.005, rng)
            else:
                sigma = sigmas[(rep + ti) % len(sigmas)]
                # alternate base
                if (rep % 2) == 0:
                    base = base_record
                else:
                    base = global_best
                if sigma == 0:
                    x_start = base.copy()
                else:
                    x_start = _tangent_perturb(base, sigma, rng)
            run_id += 1
            try:
                Xout = _hinge_run(x_start, t, maxiter=maxiter)
            except Exception:
                continue
            cnt, ssum, v = note_candidate(Xout, t)
            if cnt == 0:
                feas_chain = Xout.copy()
                feas_list.append((v, Xout.copy()))
                found_feas_this_t = True
                # lower t and continue: break rep loop to go to next target
                break
            # if close to feasible, try continuing from least-violating with small noise next rep
            # also try a quick second-stage hinge from Xout with same t but more iters if time allows and viol small
            if cnt < 30 and time.time() < HINGE_END - 10.0:
                try:
                    Xout2 = _hinge_run(Xout, t, maxiter=200)
                    cnt2, ssum2, v2 = note_candidate(Xout2, t)
                    if cnt2 == 0:
                        feas_chain = Xout2.copy()
                        feas_list.append((v2, Xout2.copy()))
                        found_feas_this_t = True
                        break
                    if cnt2 < cnt:
                        # keep chain-ish for lower t attempt anyway
                        pass
                except Exception:
                    pass
        # if found feasible, continue to next (lower) t with feas_chain
        # else keep least_viol as potential start? do not reset feas_chain
        if found_feas_this_t:
            continue
    # build polish candidates: global best, feasible list, least violating
    cands = []
    cands.append((global_val, global_best.copy()))
    for vv, XX in feas_list:
        cands.append((vv, XX.copy()))
    if least_viol is not None:
        cands.append((_max_cos(least_viol), least_viol.copy()))
    # deduplicate by value and sort
    cands.sort(key=lambda tt: tt[0])
    keep = []
    for vv, XX in cands:
        dup = False
        for _, YY in keep:
            if np.linalg.norm(XX - YY) < 1e-6 or np.linalg.norm(XX + YY) < 1e-6:
                dup = True
                break
        if not dup:
            keep.append((vv, XX))
        if len(keep) >= 4:
            break
    remaining = FINAL_END - time.time()
    if remaining < 4.0:
        final_best = global_best / np.linalg.norm(global_best, axis=1, keepdims=True)
        return final_best
    per = remaining / max(1, len(keep))
    per = min(max(per, 5.0), 30.0)
    final_best = global_best.copy()
    final_val = global_val
    for vv, XX in keep:
        if time.time() > FINAL_END - 2.0:
            break
        dl = min(time.time() + per, FINAL_END)
        try:
            P, pv = _lp_polish(XX.copy(), dl)
        except Exception:
            continue
        if pv < final_val:
            final_val = pv
            final_best = P.copy()
    final_best = final_best / np.linalg.norm(final_best, axis=1, keepdims=True)
    return final_best
