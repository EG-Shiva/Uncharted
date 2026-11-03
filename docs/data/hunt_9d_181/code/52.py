import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _adam_run(X0, tau, steps, lr, alpha=140.0, w_lse=0.08, rng=None):
    n, d = X0.shape
    X = X0.copy()
    m = np.zeros_like(X)
    v = np.zeros_like(X)
    b1, b2, eps = 0.9, 0.999, 1e-8
    for t in range(1, steps + 1):
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        D = G - tau
        # hinge weights p=4 -> h'=4*D^3
        Mk = D > 0.0
        if np.any(Mk):
            Wd = np.where(Mk, D * D * D, 0.0)
            Wd *= 4.0
            np.fill_diagonal(Wd, 0.0)
            Gh = Wd @ X
            dot = np.sum(Gh * X, axis=1, keepdims=True)
            Gt_h = Gh - dot * X
            nh = float(np.mean(np.linalg.norm(Gt_h, axis=1))) + 1e-12
            Gt_h /= nh
        else:
            Gt_h = 0.0
        # soft repulsion for centering
        gmax = float(G.max())
        # clip exponent for stability
        E = np.exp(np.clip(alpha * (G - gmax), -50.0, 0.0))
        np.fill_diagonal(E, 0.0)
        Gs = E @ X
        dot2 = np.sum(Gs * X, axis=1, keepdims=True)
        Gt_s = Gs - dot2 * X
        ns = float(np.mean(np.linalg.norm(Gt_s, axis=1))) + 1e-12
        if ns < 1e-12 or not np.isfinite(ns):
            Gt = Gt_h if not isinstance(Gt_h, float) else 0.0
        else:
            Gt_s /= ns
            if isinstance(Gt_h, float):
                Gt = w_lse * Gt_s
            else:
                Gt = Gt_h + w_lse * Gt_s
            # Gt is array
        if isinstance(Gt, float):
            break
        if not np.all(np.isfinite(Gt)):
            break
        m = b1 * m + (1.0 - b1) * Gt
        v = b2 * v + (1.0 - b2) * (Gt * Gt)
        mh = m / (1.0 - b1 ** t)
        vh = v / (1.0 - b2 ** t)
        step = mh / (np.sqrt(vh) + eps)
        # limit step magnitude
        X = X - lr * step
        # retract
        nr = np.linalg.norm(X, axis=1, keepdims=True)
        nr[~np.isfinite(nr)] = 1.0
        nr[nr < 1e-12] = 1.0
        X /= nr
    return X

def _break_contacts(X, maxv, rng, k_pairs=40, sigma=0.08, top_tol=0.003):
    n, d = X.shape
    try:
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        # candidate pairs near max
        iu0 = np.triu_indices(n, 1)
        c = G[iu0]
        thr = maxv - top_tol
        sel = np.where(c >= thr)[0]
        if sel.size == 0:
            idx = rng.choice(n, size=min(n, 12), replace=False)
        else:
            if sel.size > 2000:
                # take worst 2000
                part = np.argpartition(-c[sel], min(2000, sel.size - 1))[:2000]
                sel = sel[part]
            take = min(k_pairs, sel.size)
            pick = rng.choice(sel, size=take, replace=False)
            I = iu0[0][pick]; J = iu0[1][pick]
            uniq = np.unique(np.concatenate([I, J]))
            # add some random indices for diversity
            if uniq.size < 8:
                extra = rng.choice(n, size=8 - uniq.size, replace=False)
                uniq = np.unique(np.concatenate([uniq, extra]))
            idx = uniq
        Y = X.copy()
        Y[idx] += rng.standard_normal((idx.size, d)) * sigma
        Y /= np.linalg.norm(Y, axis=1, keepdims=True)
        return Y
    except Exception:
        Y = X + rng.standard_normal(X.shape) * sigma * 0.5
        Y /= np.linalg.norm(Y, axis=1, keepdims=True)
        return Y

def _lp_polish_final(x, seconds):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    ar = np.arange(d)
    while time.time() - start < seconds and radius > 1e-14:
        if time.time() - start + 0.4 > seconds and radius < 6e-4:
            break
        Gfull = x @ x.T
        ca = Gfull[iu]
        m = float(ca.max())
        sel = ca > m - max(20 * radius, 1e-9)
        cnt = int(sel.sum())
        if cnt == 0:
            radius *= 0.5
            continue
        if cnt > 2200:
            kth = np.partition(ca, -2200)[-2200]
            sel = ca >= kth
            if int(sel.sum()) > 2600:
                idxw = np.where(sel)[0]
                order = np.argsort(-ca[idxw])[:2600]
                m2 = np.zeros_like(ca, dtype=bool)
                m2[idxw[order]] = True
                sel = m2
        I = iu[0][sel]; J = iu[1][sel]
        csel = ca[sel]
        p = len(I); nv = n * d + 1
        try:
            rows = np.repeat(np.arange(p), 2 * d + 1)
            cols_mat = np.empty((p, 2 * d + 1), dtype=np.int64)
            cols_mat[:, :d] = I[:, None] * d + ar
            cols_mat[:, d:2 * d] = J[:, None] * d + ar
            cols_mat[:, 2 * d] = n * d
            cols = cols_mat.ravel()
            vals_mat = np.empty((p, 2 * d + 1))
            vals_mat[:, :d] = x[J]
            vals_mat[:, d:2 * d] = x[I]
            vals_mat[:, 2 * d] = -1.0
            vals = vals_mat.ravel()
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv); obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq, b_eq=np.zeros(n),
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
            x, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def solve():
    t0 = time.time()
    BUDGET = 168.0
    FINAL = 15.0
    rng = np.random.default_rng(0)
    base = None
    for fn in ["best.npy", "record.npy"]:
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=np.float64)
            if a.shape == (181, 9):
                a = a / np.linalg.norm(a, axis=1, keepdims=True)
                if np.all(np.isfinite(a)):
                    if base is None:
                        base = a.copy()
                    break
        except Exception:
            pass
    if base is None:
        base = rng.standard_normal((181, 9))
        base /= np.linalg.norm(base, axis=1, keepdims=True)
    try:
        b = np.load("best.npy")
        b = np.asarray(b, dtype=np.float64)
        if b.shape == (181, 9):
            b = b / np.linalg.norm(b, axis=1, keepdims=True)
            if _max_cos(b) < _max_cos(base):
                base = b
    except Exception:
        pass
    try:
        r = np.load("record.npy")
        r = np.asarray(r, dtype=np.float64)
        if r.shape == (181, 9):
            r = r / np.linalg.norm(r, axis=1, keepdims=True)
    except Exception:
        r = base
    vb0 = _max_cos(base)
    TAU0 = 0.4615
    P = 6
    pool_X = []
    pool_tau = []
    # diverse starts: record, perturbed, contact-broken
    sigmas = [0.0, 0.012, 0.035, 0.07, 0.11, 0.05]
    for p in range(P):
        s = sigmas[p % len(sigmas)]
        X = base.copy()
        if s > 0:
            # deliberately break different subsets: sometimes global, sometimes contact-biased
            if p % 2 == 0:
                X = X + rng.standard_normal(X.shape) * s
                X /= np.linalg.norm(X, axis=1, keepdims=True)
            else:
                mv = _max_cos(X)
                X = _break_contacts(X, mv, rng, k_pairs=int(rng.integers(20, 70)), sigma=s + 0.02)
        pool_X.append(X)
        pool_tau.append(TAU0)
    # also try starting from record perturbation if different
    try:
        if r is not base and _max_cos(r) < 0.47:
            pool_X[0] = r.copy()
    except Exception:
        pass
    best_x = base.copy()
    best_v = vb0
    # quick eval of pool
    for X in pool_X:
        v = _max_cos(X)
        if v < best_v:
            best_v = v; best_x = X.copy()
    cur_max = [_max_cos(X) for X in pool_X]
    sweep = 0
    # main continuation loop
    while True:
        rem = BUDGET - (time.time() - t0)
        if rem < FINAL + 3.0:
            break
        sweep += 1
        # lr schedule
        lr = 0.006 * (0.992 ** sweep)
        lr = max(lr, 0.0012)
        steps = 450 if sweep < 30 else 350
        # adapt w_lse slightly
        w_lse = 0.06 if sweep % 2 == 0 else 0.12
        alpha = 120.0 + (sweep % 5) * 30.0
        for p in range(P):
            if BUDGET - (time.time() - t0) < FINAL + 2.0:
                break
            Xc = pool_X[p]
            tau = pool_tau[p]
            # Adam optimize toward tau-feasibility
            try:
                Xn = _adam_run(Xc, tau, steps=steps, lr=lr, alpha=alpha, w_lse=w_lse)
            except Exception:
                continue
            try:
                vn = _max_cos(Xn)
            except Exception:
                continue
            if vn < best_v:
                best_v = vn; best_x = Xn.copy()
            # feasibility check: max <= tau (+tiny)
            if vn <= tau + 2e-9:
                # success: lower target, keep without shake
                pool_X[p] = Xn
                pool_tau[p] = tau - 2e-4
                cur_max[p] = vn
                # also push global: if feasible well below, keep tau low
                continue
            else:
                # retain best few by true max: accept if improved for this slot
                if vn < cur_max[p]:
                    pool_X[p] = Xn
                    cur_max[p] = vn
                else:
                    # occasional acceptance of worse to change basin
                    if rng.random() < 0.35:
                        pool_X[p] = Xn
                        cur_max[p] = vn
                    else:
                        # restart from global best with fresh break
                        if rng.random() < 0.5:
                            Y = best_x.copy()
                            mvb = best_v
                            sig = float(rng.uniform(0.03, 0.12))
                            kp = int(rng.integers(15, 80))
                            Y = _break_contacts(Y, mvb, rng, k_pairs=kp, sigma=sig)
                            pool_X[p] = Y
                            # keep tau but not lower; sometimes reset tau toward best
                            # if tau far below best, relax slightly
                            if tau < best_v - 0.0015:
                                pool_tau[p] = best_v - 0.0008
                            cur_max[p] = _max_cos(Y)
                        else:
                            # shake current
                            sig = float(rng.uniform(0.04, 0.13))
                            kp = int(rng.integers(20, 90))
                            Y = _break_contacts(Xn, vn, rng, k_pairs=kp, sigma=sig)
                            pool_X[p] = Y
                            cur_max[p] = _max_cos(Y)
                # if slot tau is too optimistic (far below achievable), relax a bit occasionally
                if sweep % 12 == 0 and p == 0:
                    # align taus toward best
                    for q in range(P):
                        if pool_tau[q] < best_v - 0.0012:
                            pool_tau[q] = best_v - 0.0006
        # occasional injection of pure random perturbation of best to test new basin
        if sweep % 7 == 0:
            q = int(rng.integers(0, P))
            Y = best_x + rng.standard_normal(best_x.shape) * float(rng.uniform(0.02, 0.09))
            Y /= np.linalg.norm(Y, axis=1, keepdims=True)
            # replace worst slot
            wi = int(np.argmax(cur_max))
            pool_X[wi] = Y
            cur_max[wi] = _max_cos(Y)
            # keep its tau near best
            pool_tau[wi] = min(pool_tau[wi], best_v - 0.0002)
    # final LP polish only last seconds
    rem = BUDGET - (time.time() - t0)
    ft = max(2.0, min(FINAL, rem - 0.8))
    if ft > 2.0:
        try:
            best_x = _lp_polish_final(best_x, seconds=ft)
            best_v = _max_cos(best_x)
        except Exception:
            pass
    best_x /= np.linalg.norm(best_x, axis=1, keepdims=True)
    return best_x
