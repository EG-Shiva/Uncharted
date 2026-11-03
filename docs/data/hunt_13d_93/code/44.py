import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _norm(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return X / n

def _maxcos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _rowmax(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return G.max(axis=1)

def _refine_void(v0, Others, steps=140, tau0=0.008, tau1=0.001, lr0=0.05, lr1=0.008):
    v = v0 / (np.linalg.norm(v0) + 1e-16)
    ltau = np.log(tau1 / tau0)
    llr = np.log(lr1 / lr0)
    for s in range(steps):
        f = s / max(1, steps - 1)
        tau = tau0 * np.exp(ltau * f)
        lr = lr0 * np.exp(llr * f)
        dots = Others @ v
        m0 = float(dots.max())
        w = np.exp((dots - m0) / max(tau, 1e-12))
        sw = float(w.sum()) + 1e-16
        w /= sw
        g = w @ Others
        gv = float(g @ v)
        gt = g - gv * v
        nrm = float(np.linalg.norm(gt))
        if not np.isfinite(nrm) or nrm < 1e-14:
            break
        # normalized descent on max-dot
        v = v - lr * (gt / nrm)
        nv = float(np.linalg.norm(v))
        if not np.isfinite(nv) or nv < 1e-12:
            break
        v /= nv
    return v

def _pnorm_adam(X, steps, p, shift, lr):
    n, d = X.shape
    m = np.zeros_like(X)
    v = np.zeros_like(X)
    b1, b2, eps = 0.9, 0.999, 1e-8
    t = 0
    for s in range(steps):
        G = X @ X.T
        np.fill_diagonal(G, -10.0)
        A = G - shift
        np.maximum(A, 0.0, out=A)
        # to avoid overflow for large p, scale: use power
        # w = p * A^{p-1}
        with np.errstate(over='ignore', under='ignore'):
            W = np.power(A, p - 1.0)
        W[G <= -5.0] = 0.0
        W *= float(p)
        # gradient
        grad = W @ X
        # tangent projection
        dot = np.einsum('ij,ij->i', grad, X)[:, None]
        gt = grad - dot * X
        # scale down to keep Adam stable
        # normalize by mean active weight
        sc = float(np.abs(W).sum() / max(1, X.shape[0]))
        if sc > 1e-12 and np.isfinite(sc):
            gt = gt / (sc + 1e-12)
        t += 1
        m = b1 * m + (1.0 - b1) * gt
        v = b2 * v + (1.0 - b2) * (gt * gt)
        mh = m / (1.0 - b1 ** t)
        vh = v / (1.0 - b2 ** t)
        step = lr * mh / (np.sqrt(vh) + eps)
        # project step to tangent
        d2 = np.einsum('ij,ij->i', step, X)[:, None]
        step_t = step - d2 * X
        X = X - step_t
        X = _norm(X)
        if not np.all(np.isfinite(X)):
            break
    return X

def _hardK(X, iters, step0, K, iu0, iu1):
    best = _maxcos(X)
    step = float(step0)
    mtot = len(iu0)
    K = int(min(K, mtot))
    grad = np.empty_like(X)
    for _ in range(iters):
        G = X @ X.T
        c = G[iu0, iu1]
        if K < mtot:
            idx = np.argpartition(-c, K)[:K]
        else:
            idx = np.arange(mtot)
        I = iu0[idx]; J = iu1[idx]
        grad.fill(0.0)
        np.add.at(grad, I, X[J])
        np.add.at(grad, J, X[I])
        grad /= float(len(idx))
        dot = (grad * X).sum(axis=1, keepdims=True)
        gt = grad - dot * X
        ng = float(np.linalg.norm(gt))
        if ng == 0 or not np.isfinite(ng):
            step *= 0.5
            if step < 1e-13:
                break
            continue
        Xt = X - step * gt
        Xt = _norm(Xt)
        vv = _maxcos(Xt)
        if vv < best:
            X = Xt
            best = vv
            step = min(step * 1.25, 0.02)
        else:
            step *= 0.5
            if step < 1e-13:
                break
    return X, best

def _lp_polish(X, seconds, r_init=2.5e-4, t_end=1e18):
    X = _norm(X.copy())
    n, d = X.shape
    best = _maxcos(X)
    radius = float(r_init)
    start = time.time()
    iu0, iu1 = np.triu_indices(n, 1)
    ar = np.arange(d)
    nv = n * d + 1
    obj = np.zeros(nv); obj[-1] = 1.0
    while time.time() - start < seconds:
        if time.time() > t_end - 1.0:
            break
        Gfull = X @ X.T
        c_all = Gfull[iu0, iu1]
        cmax = float(c_all.max())
        thresh = max(25.0 * radius, 5e-8)
        sel = c_all > cmax - thresh
        cnt = int(sel.sum())
        if cnt == 0:
            radius *= 0.5
            if radius < 1e-12:
                break
            continue
        if cnt > 2500:
            idx = np.argpartition(-c_all, 2500)[:2500]
            mask = np.zeros_like(sel); mask[idx] = True
            sel = mask
        I = iu0[sel]; J = iu1[sel]
        p = len(I)
        csub = c_all[sel]
        try:
            rows = np.repeat(np.arange(p), 2 * d + 1)
            cI = (I[:, None] * d + ar[None, :]).ravel()
            cJ = (J[:, None] * d + ar[None, :]).ravel()
            cols = np.empty(p * (2 * d + 1), dtype=np.int64)
            cols[0:p*d] = cI
            cols[p*d:2*p*d] = cJ
            cols[2*p*d:] = n * d
            vals = np.empty(p * (2 * d + 1), dtype=np.float64)
            vals[0:p*d] = X[J].ravel()
            vals[p*d:2*p*d] = X[I].ravel()
            vals[2*p*d:] = -1.0
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            b_ub = -csub
            # equality: dx_i . x_i = 0
            er = np.repeat(np.arange(n), d)
            ec = (np.repeat(np.arange(n), d) * d + np.tile(ar, n))
            ev = X.ravel()
            A_eq = coo_matrix((ev, (er, ec)), shape=(n, nv)).tocsr()
            b_eq = np.zeros(n)
            bounds = [(-radius, radius)] * (n * d) + [(-1.0, cmax)]
            res = linprog(obj, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                          bounds=bounds, method="highs",
                          options={"presolve": True})
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0 or res.x is None or not np.all(np.isfinite(res.x)):
            radius *= 0.5
            if radius < 1e-12:
                break
            continue
        dx = res.x[:n*d].reshape(n, d)
        t_pred = float(res.x[-1])
        pred = cmax - t_pred
        if not np.isfinite(pred) or pred <= 1e-13:
            radius *= 0.5
            if radius < 5e-11:
                break
            continue
        Xt = _norm(X + dx)
        av = _maxcos(Xt)
        if av < cmax - 1e-15:
            X = Xt
            if av < best:
                best = av
            # adapt
            maxabs = float(np.max(np.abs(dx)))
            if maxabs > 0.9 * radius:
                radius = min(radius * 1.3, 5e-3)
        else:
            radius *= 0.5
            if radius < 5e-11:
                break
    return _norm(X), best

def solve():
    np.random.seed(0)
    t0 = time.time()
    LIMIT = 172.0
    n, d = 93, 13
    bestX = None
    best_val = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=float)
            if a.shape == (n, d) and np.all(np.isfinite(a)):
                a = _norm(a)
                v = _maxcos(a)
                if v < best_val:
                    best_val = v
                    bestX = a.copy()
        except Exception:
            pass
    if bestX is None:
        rng0 = np.random.RandomState(0)
        bestX = _norm(rng0.randn(n, d))
        best_val = _maxcos(bestX)
    X0 = bestX.copy()
    v0 = float(best_val)
    iu0, iu1 = np.triu_indices(n, 1)

    # --- diagnose rattlers / hub on X0 ---
    G0 = X0 @ X0.T
    np.fill_diagonal(G0, -2.0)
    cmax0 = float(G0.max())
    slack0 = cmax0 - G0
    rowmax0 = _rowmax(X0)
    # contact counts at 1e-9
    try:
        c_all0 = G0[iu0, iu1]
        is_c = (cmax0 - c_all0) < 1e-9
        cnt_per = np.zeros(n, dtype=int)
        np.add.at(cnt_per, iu0[is_c], 1)
        np.add.at(cnt_per, iu1[is_c], 1)
    except Exception:
        cnt_per = np.zeros(n, dtype=int)
    rattlers = np.where(cnt_per == 0)[0]
    # sort rattlers by increasing row-max (loosest first)
    if len(rattlers) > 0:
        order = np.argsort(rowmax0[rattlers])
        ratt_sorted = rattlers[order]
    else:
        # fallback: loosest points by rowmax
        ratt_sorted = np.argsort(rowmax0)[:19]
    hub = int(np.argmax(cnt_per))
    # also thresholds info (not strictly needed)
    # trial plan
    k_list = [1, 2, 3, 5, 8, 13, 19]
    # cap k to available
    trial_id = 0
    # global best
    gX = X0.copy(); gV = v0

    def time_left():
        return LIMIT - (time.time() - t0)

    for k in k_list:
        if time_left() < 15.0:
            break
        kk = int(min(k, n - 5))
        # build two deletion variants
        variants = []
        # variant A: k loosest rattlers (or loosest overall if not enough)
        if len(ratt_sorted) >= kk:
            delA = ratt_sorted[:kk].tolist()
        else:
            need = kk - len(ratt_sorted)
            rest = [i for i in np.argsort(rowmax0) if i not in set(ratt_sorted.tolist())]
            delA = ratt_sorted.tolist() + rest[:need]
        variants.append(("A", delA))
        # variant B: hub swap (replace last deleted with hub if hub not already in)
        if kk >= 1:
            if hub not in delA:
                delB = delA[:-1] + [hub] if kk > 1 else [hub]
                variants.append(("B", delB))
        for vname, del_idx in variants:
            if time_left() < 12.0:
                break
            trial_id += 1
            # per-trial budget
            t_trial_start = time.time()
            trial_budget = min(14.0, time_left() - 4.0)
            if trial_budget < 3.0:
                break
            t_trial_end = t_trial_start + trial_budget
            # remaining set
            mask = np.ones(n, dtype=bool); mask[np.array(del_idx, dtype=int)] = False
            cur = X0[mask].copy()  # (n-kk, d)
            ok = True
            for s in range(kk):
                if time.time() > t_trial_end - 2.5:
                    ok = False
                    break
                mcur = cur.shape[0]
                rng = np.random.RandomState(10000 + trial_id * 1000 + s * 17)
                # sample 60000 candidates
                NC = 60000
                # batch to limit memory spike
                best_worst = 1e9
                best_cands = None
                # generate in two batches of 30000
                allC = rng.standard_normal((NC, d))
                allC /= np.linalg.norm(allC, axis=1, keepdims=True) + 1e-16
                D = allC @ cur.T
                worst = D.max(axis=1)
                # best 20
                K20 = 20
                idx20 = np.argpartition(worst, K20 - 1)[:K20]
                # refine each
                refined = []
                for bi in idx20:
                    v00 = allC[bi]
                    vr = _refine_void(v00, cur, steps=120)
                    wv = float((cur @ vr).max())
                    refined.append((wv, vr))
                refined.sort(key=lambda z: z[0])
                chosen = refined[0][1]
                # append
                cur = np.vstack([cur, chosen[None, :]])
                cur = _norm(cur)
                # shifted p-norm Adam few hundred steps: p=32,64,128
                # scale steps by size: fewer steps for large trials
                if kk <= 3:
                    s32, s64, s128 = 110, 70, 50
                elif kk <= 8:
                    s32, s64, s128 = 80, 50, 40
                else:
                    s32, s64, s128 = 50, 35, 25
                # check time
                if time.time() > t_trial_end - 2.5:
                    ok = False
                    break
                cur = _pnorm_adam(cur, s32, 32, 0.20, 0.006)
                if time.time() > t_trial_end - 2.0:
                    ok = False
                    break
                cur = _pnorm_adam(cur, s64, 64, 0.20, 0.004)
                if time.time() > t_trial_end - 1.8:
                    ok = False
                    break
                cur = _pnorm_adam(cur, s128, 128, 0.20, 0.003)
            if not ok:
                # still try to finish if cur has right size
                if cur.shape[0] != n:
                    continue
            if cur.shape[0] != n:
                continue
            # top-K hard-max descent
            remain = t_trial_end - time.time()
            if remain < 1.5:
                # evaluate anyway
                vv = _maxcos(cur)
                if vv < gV:
                    gV = vv; gX = _norm(cur.copy())
                continue
            # iters scaled
            iters = 250 if kk <= 5 else 180
            cur, vv = _hardK(cur, iters, 0.006, 150, iu0, iu1)
            if vv < gV - 1e-12:
                gV = vv; gX = _norm(cur.copy())
            # short LP polish with remaining time (max ~2.5s)
            remain2 = t_trial_end - time.time()
            lp_sec = min(2.5, max(0.6, remain2 - 0.4))
            if lp_sec > 0.4:
                try:
                    cur2, vv2 = _lp_polish(cur, lp_sec, 2.5e-4, t0 + LIMIT)
                    if vv2 < gV:
                        gV = vv2; gX = _norm(cur2.copy())
                    cur = cur2; vv = vv2
                except Exception:
                    pass
            else:
                if vv < gV:
                    gV = vv; gX = _norm(cur.copy())
            # early exit if big improvement? continue to try more
    # final safety: if no improvement, return original
    gX = _norm(gX)
    # final quick check
    try:
        fv = _maxcos(gX)
    except Exception:
        return _norm(X0)
    if not np.all(np.isfinite(gX)) or fv > v0 + 1e-12:
        return _norm(X0)
    return _norm(gX)
