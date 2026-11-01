import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

_N = 173
_D = 5

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _pair_max_r(X, r):
    G = X @ X.T
    # max over pairs involving r and over pairs not involving r
    n = X.shape[0]
    col = G[r, :].copy()
    col[r] = -2.0
    mr = float(col.max())
    # non-r max: mask out row/col r by setting to -2 then max
    # compute via triu without r for exactness
    G2 = G.copy()
    G2[r, :] = -2.0
    G2[:, r] = -2.0
    np.fill_diagonal(G2, -2.0)
    mn = float(G2.max())
    return mn, mr, float(max(mn, mr))

def _contacts(X, tol=1e-9):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    m = float(G.max())
    cnt = (G >= m - tol).sum(axis=1)
    return m, cnt, G

def _frozen_step(X, r, T0, Tr, radius, iu0, iu1):
    n, d = X.shape
    G = X @ X.T
    # triu cos
    c_all = G[iu0, iu1]
    is_r = (iu0 == r) | (iu1 == r)
    thr_n = T0 - max(20.0 * radius, 1e-9)
    thr_r = Tr - max(20.0 * radius, 1e-9)
    sel_n = (c_all > thr_n) & (~is_r)
    sel_r = (c_all > thr_r) & (is_r)
    # ensure at least top 250 non-r pairs to bound t
    if int(sel_n.sum()) < 250:
        idx_n = np.where(~is_r)[0]
        # pick 250 largest
        order = np.argsort(c_all[idx_n])[-250:]
        sel_n[idx_n[order]] = True
    # cap non-r to 2500 tightest
    if int(sel_n.sum()) > 2500:
        idx = np.where(sel_n)[0]
        order = np.argsort(c_all[idx])[-2500:]
        tmp = np.zeros_like(sel_n)
        tmp[idx[order]] = True
        sel_n = tmp
    I_n = iu0[sel_n]; J_n = iu1[sel_n]; c_n = c_all[sel_n]
    I_r = iu0[sel_r]; J_r = iu1[sel_r]; c_r = c_all[sel_r]
    p1 = len(I_n); p2 = len(I_r)
    if p1 == 0 and p2 == 0:
        return None, None, None
    nv = n * d + 1
    tcol = n * d
    rows_list = []
    cols_list = []
    vals_list = []
    b_list = []
    # p1 rows: <dx_I,x_J>+<dx_J,x_I> - t <= -c
    if p1 > 0:
        for k in range(p1):
            i = int(I_n[k]); j = int(J_n[k])
            rn = k
            # I block
            for a in range(d):
                rows_list.append(rn); cols_list.append(i * d + a); vals_list.append(float(X[j, a]))
            for a in range(d):
                rows_list.append(rn); cols_list.append(j * d + a); vals_list.append(float(X[i, a]))
            rows_list.append(rn); cols_list.append(tcol); vals_list.append(-1.0)
            b_list.append(float(-c_n[k]))
    # p2 rows: single-sided, r frozen => only free endpoint
    # row index offset p1
    for k in range(p2):
        i = int(I_r[k]); j = int(J_r[k])
        rn = p1 + k
        if i == r:
            # free is j, coeff X[r]
            for a in range(d):
                rows_list.append(rn); cols_list.append(j * d + a); vals_list.append(float(X[i, a]))
            b_list.append(float(Tr - c_r[k]))
        elif j == r:
            for a in range(d):
                rows_list.append(rn); cols_list.append(i * d + a); vals_list.append(float(X[j, a]))
            b_list.append(float(Tr - c_r[k]))
        else:
            # should not happen
            pass
    p = p1 + p2
    rows = np.array(rows_list, dtype=np.int64)
    cols = np.array(cols_list, dtype=np.int64)
    vals = np.array(vals_list, dtype=np.float64)
    b_ub = np.array(b_list, dtype=np.float64)
    A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
    # equality: X_i . dx_i = 0
    er = np.repeat(np.arange(n), d)
    ec = np.arange(n * d)
    ev = X.ravel().copy()
    A_eq = coo_matrix((ev, (er, ec)), shape=(n, nv)).tocsr()
    b_eq = np.zeros(n)
    obj = np.zeros(nv); obj[-1] = 1.0
    bounds = [(-radius, radius)] * (n * d) + [(None, None)]
    # freeze r: bounds zero
    for a in range(d):
        bounds[r * d + a] = (0.0, 0.0)
    try:
        res = linprog(obj, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                      bounds=bounds, method="highs")
    except Exception:
        return None, None, None
    if res.status != 0:
        return None, None, None
    dx = res.x[:-1].reshape(n, d)
    # enforce frozen exactly
    dx[r, :] = 0.0
    t_pred = float(res.x[-1])
    return dx, t_pred, (p1, p2)

def _unfreeze_polish(X, r, seconds, deadline):
    # custom polish: pairs within 5e-3 of max plus 15 nearest r pairs always included
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    n, d = X.shape
    best = _max_cos(X)
    radius = 1e-3
    start = time.time()
    iu0, iu1 = np.triu_indices(n, 1)
    while time.time() - start < seconds and time.time() < deadline:
        if radius < 1e-14:
            break
        G = X @ X.T
        c_all = G[iu0, iu1]
        m = float(c_all.max())
        sel = (m - c_all) < 5e-3
        # force 15 nearest r pairs
        is_r = (iu0 == r) | (iu1 == r)
        idx_r = np.where(is_r)[0]
        if len(idx_r) > 0:
            order = np.argsort(c_all[idx_r])[-15:]
            sel[idx_r[order]] = True
        # cap to 3000 tightest if too many
        if int(sel.sum()) > 3000:
            idx = np.where(sel)[0]
            # keep: all forced r pairs + tightest others to fill 3000
            # ensure forced kept
            forced = set(idx_r[np.argsort(c_all[idx_r])[-15:]].tolist())
            # sort by c
            order = np.argsort(c_all[idx])
            # take from top, but ensure forced included: simple take top 3000 then add forced
            top = set(idx[order[-3000:]].tolist()) | forced
            tmp = np.zeros_like(sel)
            tmp[list(top)] = True
            sel = tmp
        I = iu0[sel]; J = iu1[sel]; c = c_all[sel]
        p = len(I)
        if p == 0:
            break
        # also adapt: if radius large, 5e-3 window may be smaller than 20*radius; expand
        # (already covered since 5e-3 ~ fixed)
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([X[J], X[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((X.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = X + res.x[:-1].reshape(n, d)
        yn = np.linalg.norm(y, axis=1, keepdims=True)
        yn[yn == 0] = 1.0
        y = y / yn
        v = _max_cos(y)
        if v < best:
            X, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return X

def _std_polish(X, seconds, deadline):
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    n, d = X.shape
    best = _max_cos(X)
    radius = 1e-3
    start = time.time()
    iu0, iu1 = np.triu_indices(n, 1)
    while time.time() - start < seconds and time.time() < deadline:
        if radius < 1e-14:
            break
        G = X @ X.T
        c_all = G[iu0, iu1]
        m = float(c_all.max())
        sel = c_all > m - max(20.0 * radius, 1e-9)
        if int(sel.sum()) > 3000:
            idx = np.where(sel)[0]
            order = np.argsort(c_all[idx])[-3000:]
            tmp = np.zeros_like(sel)
            tmp[idx[order]] = True
            sel = tmp
        I = iu0[sel]; J = iu1[sel]; c = c_all[sel]
        p = len(I)
        if p == 0:
            radius *= 0.5
            continue
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([X[J], X[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((X.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
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
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return X

def solve():
    t0 = time.time()
    BUDGET = 172.0
    deadline = t0 + BUDGET
    rng = np.random.default_rng(0)
    iu0, iu1 = np.triu_indices(173, 1)

    # load best
    best = None
    best_val = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn).astype(np.float64)
            if a.shape == (173, 5):
                an = a / np.linalg.norm(a, axis=1, keepdims=True)
                v = _max_cos(an)
                if v < best_val:
                    best_val = v
                    best = an.copy()
        except Exception:
            pass
    if best is None:
        b = rng.normal(size=(173, 5))
        best = b / np.linalg.norm(b, axis=1, keepdims=True)
        best_val = _max_cos(best)
    T0_record = float(best_val)
    global_best = best.copy()
    global_val = float(best_val)

    # identify rattler
    _, cnt0, _ = _contacts(global_best, tol=1e-9)
    order = np.argsort(cnt0)
    r = int(order[0])
    # if no zero-contact, pick min max point
    if int(cnt0[r]) != 0:
        G = global_best @ global_best.T
        np.fill_diagonal(G, -2.0)
        per = G.max(axis=1)
        r = int(np.argmin(per))

    eps_cycle = [1e-4, 3e-4, 1e-3, 3e-3]
    # outer attempts: start from global best (and later perturbed)
    attempt = 0
    active_best = None
    active_val = 1e9
    while time.time() < deadline - 12.0:
        # build start point
        if attempt == 0:
            x_cur = global_best.copy()
        else:
            # perturbed restart around global best to get new basin
            sig = [0.008, 0.02, 0.05, 0.012, 0.03][(attempt - 1) % 5]
            x_cur = global_best.copy() + sig * rng.normal(size=global_best.shape)
            x_cur /= np.linalg.norm(x_cur, axis=1, keepdims=True)
            # quick std polish to get to nearby optimum (short)
            rem = deadline - time.time()
            if rem < 15:
                break
            x_cur = _std_polish(x_cur, min(2.0, rem - 10.0), deadline)
            vq = _max_cos(x_cur)
            # re-identify r for this branch? keep original r index for freezing
            # but if rattler moved, update r to current zero-contact if exists
            _, cntq, _ = _contacts(x_cur, tol=1e-9)
            if (cntq == 0).any():
                r_cur = int(np.where(cntq == 0)[0][0])
            else:
                r_cur = r
            # use r_cur for this attempt
            r_use = r_cur
            # fall through to frozen phase with r_use
            # to avoid code dup, set r_attempt and continue to frozen
            # we handle by assigning r_save
            r_attempt = r_use
            # frozen phase below uses r_attempt
            # jump to frozen (duplicate code path) - handle via variable
            # set flag
            use_r = r_attempt
            # do frozen cycles
            T0 = min(global_val, vq)  # cap target at record
            # if vq much worse than record, reset to global_best neighbourhood? keep
            if vq > T0 + 5e-3:
                x_cur = global_best.copy()
                use_r = r
                T0 = global_val
            # run frozen+unfreeze (shared code) -> use helper inline
            # we will not duplicate: set x_start and go to shared block via exec pattern
            # simplest: continue to shared block by storing
            x_start = x_cur
            T0_start = T0
            r_start = use_r
            # ---- frozen cycles ----
            radius_f = 5e-3
            # cycle eps twice
            for cyc in range(2):
                for eps in eps_cycle:
                    if time.time() > deadline - 12.0:
                        break
                    Tr = T0_start - eps
                    for it in range(10):
                        if time.time() > deadline - 12.0:
                            break
                        old_n, old_rv, _ = _pair_max_r(x_start, r_start)
                        pen_old = max(0.0, old_n - T0_start) + max(0.0, old_rv - Tr)
                        out = _frozen_step(x_start, r_start, T0_start, Tr, radius_f, iu0, iu1)
                        if out[0] is None:
                            radius_f *= 0.5
                            if radius_f < 1e-12:
                                radius_f = 1e-4
                                break
                            continue
                        dx, t_pred, _ = out
                        y = x_start + dx
                        y /= np.linalg.norm(y, axis=1, keepdims=True)
                        new_n, new_rv, _ = _pair_max_r(y, r_start)
                        pen_new = max(0.0, new_n - T0_start) + max(0.0, new_rv - Tr)
                        if pen_new < pen_old - 1e-12:
                            x_start = y
                            radius_f = min(radius_f * 1.3, 0.02)
                            if pen_new <= 1e-12:
                                break
                        else:
                            radius_f *= 0.5
                            if radius_f < 1e-7:
                                break
                    # end it
                # end eps
            # ---- unfreeze ----
            rem2 = deadline - time.time()
            if rem2 > 10:
                x_pol = _unfreeze_polish(x_start, r_start, min(4.0, rem2 - 6.0), deadline)
                # further std polish
                rem3 = deadline - time.time()
                if rem3 > 8:
                    x_pol = _std_polish(x_pol, min(3.0, rem3 - 5.0), deadline)
                vp = _max_cos(x_pol)
                _, cntp, _ = _contacts(x_pol, tol=1e-9)
                # acceptance: rattler acquires >=3 contacts
                # find count of original r_start
                cr = int(cntp[r_start])
                # also check if any point still rattler? we care about r_start
                if cr >= 3:
                    if vp < active_val:
                        active_val = float(vp)
                        active_best = x_pol.copy()
                    # continue search from this balanced point even if worse
                    # if it is promising (within 5e-3 of record) keep exploring from it
                    if vp < global_val and vp < T0_start + 5e-3:
                        pass
                    if vp < global_val - 1e-12:
                        global_best = x_pol.copy()
                        global_val = float(vp)
                        T0_record = float(vp)
                        # update rattler
                        _, cnt0, _ = _contacts(global_best, tol=1e-9)
                        r = int(np.argsort(cnt0)[0])
                else:
                    if vp < global_val - 1e-12:
                        global_best = x_pol.copy()
                        global_val = float(vp)
                        _, cnt0, _ = _contacts(global_best, tol=1e-9)
                        r = int(np.argsort(cnt0)[0])
            attempt += 1
            continue

        # attempt==0 path (exact record start)
        x_start = x_cur
        T0_start = T0_record
        r_start = r
        radius_f = 5e-3
        for cyc in range(3):
            for eps in eps_cycle:
                if time.time() > deadline - 12.0:
                    break
                Tr = T0_start - eps
                for it in range(12):
                    if time.time() > deadline - 12.0:
                        break
                    old_n, old_rv, _ = _pair_max_r(x_start, r_start)
                    pen_old = max(0.0, old_n - T0_start) + max(0.0, old_rv - Tr)
                    out = _frozen_step(x_start, r_start, T0_start, Tr, radius_f, iu0, iu1)
                    if out[0] is None:
                        radius_f *= 0.5
                        if radius_f < 1e-12:
                            radius_f = 5e-4
                            break
                        continue
                    dx, t_pred, _ = out
                    y = x_start + dx
                    y /= np.linalg.norm(y, axis=1, keepdims=True)
                    new_n, new_rv, _ = _pair_max_r(y, r_start)
                    pen_new = max(0.0, new_n - T0_start) + max(0.0, new_rv - Tr)
                    if pen_new < pen_old - 1e-12:
                        x_start = y
                        radius_f = min(radius_f * 1.3, 0.02)
                        if pen_new <= 1e-12:
                            break
                    else:
                        radius_f *= 0.5
                        if radius_f < 1e-7:
                            break
        rem2 = deadline - time.time()
        if rem2 > 8:
            x_pol = _unfreeze_polish(x_start, r_start, min(5.0, rem2 - 5.0), deadline)
            rem3 = deadline - time.time()
            if rem3 > 7:
                x_pol = _std_polish(x_pol, min(4.0, rem3 - 4.0), deadline)
            vp = _max_cos(x_pol)
            _, cntp, _ = _contacts(x_pol, tol=1e-9)
            cr = int(cntp[r_start])
            if cr >= 3 and vp < active_val:
                active_val = float(vp)
                active_best = x_pol.copy()
            if vp < global_val - 1e-12:
                global_best = x_pol.copy()
                global_val = float(vp)
                _, cnt0, _ = _contacts(global_best, tol=1e-9)
                r = int(np.argsort(cnt0)[0])
        attempt += 1

    # final safety polish of global best
    rem = deadline - time.time()
    if rem > 6:
        global_best = _std_polish(global_best, min(rem - 2.0, 6.0), deadline)
        gv = _max_cos(global_best)
        global_val = float(gv)
    # If active balanced solution is within 1e-4 of global and has >=3 contacts, prefer it?
    # Scoring prefers lower max, so only prefer if it beats global.
    # Return best max found (never worse than start).
    global_best /= np.linalg.norm(global_best, axis=1, keepdims=True)
    return global_best
