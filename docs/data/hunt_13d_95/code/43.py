import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def lp_polish(x, seconds, t_end=None):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = max_cos(x)
    radius = 1e-3
    start = time.time()
    if t_end is None:
        t_end = start + seconds
    else:
        t_end = min(t_end, start + seconds)
    iu = np.triu_indices(n, 1)
    while time.time() < t_end and radius > 1e-14:
        c = (x @ x.T)[iu]
        m = c.max()
        sel = c > m - max(20 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p, nv = len(I), n * d + 1
        if p == 0:
            break
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
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
        y = x + res.x[:-1].reshape(n, d)
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        v = max_cos(y)
        if v < best:
            x, best, radius = y, v, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def contact_duals(x, I, J, radius=5e-4):
    n, d = x.shape
    p = len(I)
    if p == 0:
        return None
    try:
        c = np.sum(x[I] * x[J], axis=1)
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv); obj[-1] = 1.0
        res = linprog(obj, A_ub=A_ub, b_ub=-c, A_eq=A_eq, b_eq=np.zeros(n),
                      bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        if res.status != 0:
            return None
        marg = None
        try:
            il = res.ineqlin
            if hasattr(il, 'marginals'):
                marg = np.asarray(il.marginals).ravel()
            elif isinstance(il, dict) and 'marginals' in il:
                marg = np.asarray(il['marginals']).ravel()
            else:
                try:
                    marg = np.asarray(il['marginals']).ravel()
                except Exception:
                    marg = None
        except Exception:
            marg = None
        if marg is None or marg.size != p:
            # try alternative location
            try:
                marg2 = np.asarray(res.marginals).ravel()
                if marg2.size == p:
                    marg = marg2
            except Exception:
                pass
        if marg is None or marg.size != p:
            return None
        return np.abs(marg)
    except Exception:
        return None

def adam_feas(X, P, targets=(0.2438, 0.2433, 0.2425), steps_list=(600, 600, 600), lr=0.003, tau=1e-4):
    n, d = X.shape
    P = np.asarray(P, dtype=int)
    k = len(P)
    if k == 0:
        return X
    Xp = X[P].copy()
    X_all = X.copy()
    m1 = np.zeros_like(Xp)
    m2 = np.zeros_like(Xp)
    tglob = 0
    for ti, t in enumerate(targets):
        steps = steps_list[ti] if ti < len(steps_list) else 600
        for it in range(steps):
            X_all[P] = Xp
            D = Xp @ X_all.T
            D[np.arange(k), P] = -2.0
            Z = (D - t) / tau
            np.clip(Z, -50.0, 50.0, out=Z)
            W = 1.0 / (1.0 + np.exp(-Z))
            grad = W @ X_all
            try:
                Wp = W[:, P]
                extra = Wp.T @ Xp
                grad = grad + extra
            except Exception:
                pass
            dot = np.sum(grad * Xp, axis=1, keepdims=True)
            gt = grad - dot * Xp
            tglob += 1
            m1 = 0.9 * m1 + 0.1 * gt
            m2 = 0.999 * m2 + 0.001 * (gt * gt)
            m1h = m1 / (1.0 - 0.9 ** tglob)
            m2h = m2 / (1.0 - 0.999 ** tglob)
            step = lr * m1h / (np.sqrt(m2h) + 1e-8)
            sn = np.linalg.norm(step, axis=1, keepdims=True) + 1e-12
            clip = np.minimum(1.0, 0.02 / sn)
            step = step * clip
            sdot = np.sum(step * Xp, axis=1, keepdims=True)
            step = step - sdot * Xp
            Xp = Xp - step
            Xp /= np.maximum(np.linalg.norm(Xp, axis=1, keepdims=True), 1e-12)
    X_new = X_all.copy()
    X_new[P] = Xp
    X_new /= np.linalg.norm(X_new, axis=1, keepdims=True)
    return X_new

def solve():
    t0 = time.time()
    search_deadline = t0 + 150.0
    hard_deadline = t0 + 172.0
    rng = np.random.default_rng(0)
    np.random.seed(0)
    try:
        xb = np.load("best.npy").astype(float)
    except Exception:
        xb = None
    try:
        xr = np.load("record.npy").astype(float)
    except Exception:
        xr = None
    cands = []
    if xb is not None:
        try:
            xb = xb / np.linalg.norm(xb, axis=1, keepdims=True)
            cands.append(xb)
        except Exception:
            pass
    if xr is not None:
        try:
            xr = xr / np.linalg.norm(xr, axis=1, keepdims=True)
            cands.append(xr)
        except Exception:
            pass
    if len(cands) == 0:
        tmp = rng.standard_normal((95, 13))
        tmp /= np.linalg.norm(tmp, axis=1, keepdims=True)
        cands.append(tmp)
    best_val = 1e9
    best_x = cands[0].copy()
    for c in cands:
        try:
            v = max_cos(c)
            if v < best_val:
                best_val = v
                best_x = c.copy()
        except Exception:
            pass
    record_thresh = 0.244097856927
    xb0 = best_x.copy()
    # one LP polish to reach nearest local optimum
    try:
        remain = min(4.0, max(1.0, search_deadline - time.time() - 100.0))
        if remain > 0.5:
            best_x = lp_polish(best_x, seconds=remain, t_end=search_deadline - 100.0)
            best_val = max_cos(best_x)
    except Exception:
        pass
    n, d = best_x.shape
    iu = np.triu_indices(n, 1)
    it = 0
    # surgery loop
    while time.time() < search_deadline - 4.0 and time.time() < hard_deadline - 8.0:
        it += 1
        if time.time() > search_deadline - 4.0:
            break
        try:
            G = best_x @ best_x.T
            np.fill_diagonal(G, -2.0)
            m = float(G.max())
            cfull = G[iu]
            selmask = cfull > m - 1e-9
            I = iu[0][selmask]
            J = iu[1][selmask]
            p = len(I)
            if p < 10:
                # fallback: widen threshold
                selmask = cfull > m - 1e-7
                I = iu[0][selmask]
                J = iu[1][selmask]
                p = len(I)
            if p < 10 or p > 5000:
                continue
            stresses = contact_duals(best_x, I, J, radius=5e-4)
            if stresses is None or stresses.size != p or not np.all(np.isfinite(stresses)):
                order = np.argsort(cfull[selmask])  # larger slack = lower stress proxy (ascending? slack small first)
                # cfull sorted ascending means tighter first; low stress assumed larger slack -> reverse
                # use random among all as low-stress proxy
                order = rng.permutation(p)
            else:
                order = np.argsort(stresses)  # ascending: low stress first
            poolsize = min(80, p)
            pool = order[:poolsize]
            L_target = int(rng.integers(10, 31))
            L_target = min(L_target, poolsize)
            # clustered growth: seed + touch-expansion in stress order
            seed = pool[int(rng.integers(len(pool)))]
            Pset = set([int(I[seed]), int(J[seed])])
            chosen = [int(seed)]
            used = set([int(seed)])
            # first pass: touching in stress order
            for cand in pool:
                cand = int(cand)
                if cand in used:
                    continue
                if len(chosen) >= 30:
                    break
                a = int(I[cand]); b = int(J[cand])
                if a not in Pset and b not in Pset:
                    continue
                ns = len(Pset | {a, b})
                if ns > 20:
                    continue
                Pset.add(a); Pset.add(b)
                chosen.append(cand); used.add(cand)
                if len(chosen) >= L_target and len(Pset) >= 8:
                    break
            # second pass: allow non-touching to fill to 10 contacts / 8 points
            if len(chosen) < 10 or len(Pset) < 8:
                for cand in pool:
                    cand = int(cand)
                    if cand in used:
                        continue
                    if len(chosen) >= 30:
                        break
                    a = int(I[cand]); b = int(J[cand])
                    ns = len(Pset | {a, b})
                    if ns > 20:
                        continue
                    Pset.add(a); Pset.add(b)
                    chosen.append(cand); used.add(cand)
                    if len(chosen) >= L_target and len(Pset) >= 8:
                        break
            if not (10 <= len(chosen) <= 30):
                continue
            if not (8 <= len(Pset) <= 20):
                continue
            Plist = sorted(list(Pset))
            # feasibility with frozen remainder, staged targets
            X_new = adam_feas(best_x, Plist, targets=(0.2438, 0.2433, 0.2425), steps_list=(550, 550, 550), lr=0.003, tau=1e-4)
            # quick check: if wildly bad skip LP? still LP per spec
            # unfreeze + LP polish
            budget = min(2.2, max(0.6, (search_deadline - 4.0) - time.time()))
            if budget < 0.4:
                break
            X_pol = lp_polish(X_new, seconds=budget, t_end=search_deadline - 1.0)
            v = max_cos(X_pol)
            if v < best_val - 1e-12 and v < record_thresh + 1e-12:
                # keep only if beats record (and improves)
                if v < record_thresh - 1e-12 or v < best_val - 1e-12:
                    # strictly require beating record to keep per spec, else allow equal-progress?
                    if v < record_thresh:
                        best_val = v
                        best_x = X_pol
                    elif best_val >= record_thresh and v < best_val - 1e-12:
                        best_val = v
                        best_x = X_pol
            # also accept tiny improvement below record only
            # if we already beat record, accept any improvement
            # (handled above)
        except Exception:
            continue
        if best_val < 0.2424:
            break
    # final safety: ensure fallback to record if not beating
    try:
        if xr is not None:
            vr = max_cos(xr / np.linalg.norm(xr, axis=1, keepdims=True))
            if not np.isfinite(best_val) or best_val >= vr:
                # if we did not beat record, return the better of xb0/xr polished briefly?
                # always fall back to record side
                if vr <= best_val:
                    best_x = xr / np.linalg.norm(xr, axis=1, keepdims=True)
                    best_val = vr
    except Exception:
        pass
    # final short polish if time remains and we beat record
    try:
        if time.time() < hard_deadline - 1.0 and best_val < record_thresh:
            rem = hard_deadline - time.time() - 0.5
            if rem > 1.0:
                best_x = lp_polish(best_x, seconds=min(rem, 6.0), t_end=hard_deadline - 0.5)
    except Exception:
        pass
    best_x = best_x / np.linalg.norm(best_x, axis=1, keepdims=True)
    # sanitize
    best_x = np.where(np.isfinite(best_x), best_x, 0.0)
    nr = np.linalg.norm(best_x, axis=1, keepdims=True)
    nr = np.where(nr < 1e-12, 1.0, nr)
    best_x = best_x / nr
    return best_x
