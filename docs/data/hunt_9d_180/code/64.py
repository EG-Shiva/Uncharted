import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def lp_polish_capped(x, seconds, cap=3000, radius_init=1e-3):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    try:
        best = max_cos(x)
    except Exception:
        return x
    radius = radius_init
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        try:
            G = x @ x.T
            c_full = G[iu]
            m = float(c_full.max())
        except Exception:
            break
        # tightest cap
        try:
            if c_full.size > cap:
                order = np.argpartition(-c_full, cap)[:cap]
                # sort those for stability (optional)
                o2 = np.argsort(-c_full[order])
                order = order[o2]
            else:
                order = np.argsort(-c_full)
        except Exception:
            break
        c = c_full[order]
        # further filter by trust region width to reduce constraints
        thr = m - max(20.0 * radius, 1e-9)
        keep = c > thr
        if np.count_nonzero(keep) == 0:
            keep = np.zeros_like(c, dtype=bool)
            keep[:min(len(c), 500)] = True
        # if still too many, keep tightest cap (already)
        sel_order = order[keep]
        I = iu[0][sel_order]
        J = iu[1][sel_order]
        cc = c_full[sel_order]
        p = len(I)
        if p == 0:
            radius *= 0.5
            continue
        nv = n * d + 1
        try:
            rows = np.repeat(np.arange(p), 2 * d + 1)
            cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
            vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        except Exception:
            radius *= 0.5
            continue
        obj = np.zeros(nv)
        obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-cc, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        try:
            y = x + res.x[:-1].reshape(n, d)
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            v = max_cos(y)
        except Exception:
            radius *= 0.5
            continue
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
        if time.time() - start >= seconds:
            break
    return x

def get_dual_weights(x, cap=3000, radius=1e-3, slack=1e-7):
    n, d = x.shape
    iu = np.triu_indices(n, 1)
    try:
        G = x @ x.T
        c_full = G[iu]
        m = float(c_full.max())
    except Exception:
        return None, None, None
    # contacts per point with 1e-9 threshold (picture)
    try:
        thr_c = m - 1e-9
        sel_c = c_full > thr_c
        Ic = iu[0][sel_c]; Jc = iu[1][sel_c]
        contacts = np.zeros(n, dtype=int)
        np.add.at(contacts, Ic, 1)
        np.add.at(contacts, Jc, 1)
    except Exception:
        contacts = np.ones(n, dtype=int)
    rattlers = np.where(contacts == 0)[0]
    # build LP for duals: tightest cap
    try:
        if c_full.size > cap:
            order = np.argpartition(-c_full, cap)[:cap]
            o2 = np.argsort(-c_full[order])
            order = order[o2]
        else:
            order = np.argsort(-c_full)
        I = iu[0][order]; J = iu[1][order]; c = c_full[order]
        p = len(order)
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
            return None, contacts, rattlers
        # extract marginals
        marg = None
        try:
            if hasattr(res, "ineqlin") and res.ineqlin is not None:
                try:
                    marg = np.asarray(res.ineqlin.marginals, dtype=float)
                except Exception:
                    try:
                        marg = np.asarray(res.ineqlin["marginals"], dtype=float)
                    except Exception:
                        marg = None
        except Exception:
            marg = None
        if marg is None or marg.size != p:
            return None, contacts, rattlers
        w = np.zeros(n)
        # aggregate for slack < slack pairs
        for k in range(p):
            if m - float(c[k]) < slack:
                a = float(abs(marg[k]))
                if np.isfinite(a) and a != 0.0:
                    w[I[k]] += a
                    w[J[k]] += a
        return w, contacts, rattlers
    except Exception:
        return None, contacts, rattlers

def refine_vs(Vs, Xf, steps=300):
    # Vs (K,9), Xf (M,9), annealed softmax projected gradient
    K = Vs.shape[0]
    for s in range(steps):
        frac = s / max(1, steps - 1)
        alpha = 25.0 + (350.0 - 25.0) * frac
        lr = 0.04 * (1.0 - frac) + 0.004 * frac
        try:
            D = Vs @ Xf.T  # K x M
            dm = D.max(axis=1, keepdims=True)
            E = np.exp(alpha * (D - dm))
            S = E.sum(axis=1, keepdims=True) + 1e-30
            W = E / S
            G = W @ Xf  # K x 9
            nd = (G * Vs).sum(axis=1, keepdims=True)
            Gt = G - nd * Vs
            Vs = Vs - lr * Gt
            Vs /= np.linalg.norm(Vs, axis=1, keepdims=True)
        except Exception:
            break
    # clean non-finite
    Vs[~np.isfinite(Vs)] = 0.0
    nrm = np.linalg.norm(Vs, axis=1, keepdims=True)
    nrm[nrm == 0] = 1.0
    Vs /= nrm
    return Vs

def solve():
    np.random.seed(0)
    t0 = time.time()
    deadline = t0 + 176.0
    explore_deadline = t0 + 140.0
    best = None
    best_val = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=float)
            if a.shape == (180, 9):
                a = a / np.linalg.norm(a, axis=1, keepdims=True)
                if np.all(np.isfinite(a)):
                    v = max_cos(a)
                    if v < best_val:
                        best_val = v
                        best = a.copy()
        except Exception:
            pass
    if best is None:
        rng0 = np.random.RandomState(0)
        best = rng0.randn(180, 9)
        best /= np.linalg.norm(best, axis=1, keepdims=True)
        best_val = max_cos(best)
    n, d = 180, 9
    # once polish with cap 3000 (short) to get jammed optimum
    try:
        remain = explore_deadline - time.time()
        if remain > 10:
            b2 = lp_polish_capped(best, min(2.0, remain - 8.0), cap=3000, radius_init=1e-3)
            v2 = max_cos(b2)
            if v2 < best_val and np.all(np.isfinite(b2)):
                best = b2
                best_val = v2
    except Exception:
        pass
    # get dual weights baseline
    base_w = None
    base_contacts = None
    base_rattlers = None
    try:
        base_w, base_contacts, base_rattlers = get_dual_weights(best, cap=3000, radius=1e-3, slack=1e-7)
    except Exception:
        pass
    if base_w is None:
        # fallback to per-point max
        try:
            G = best @ best.T
            np.fill_diagonal(G, -2.0)
            ppm = G.max(axis=1)
            base_w = ppm - ppm.min() + 1e-12
        except Exception:
            base_w = np.ones(n)
    if base_rattlers is None or len(base_rattlers) == 0:
        try:
            G = best @ best.T
            iu = np.triu_indices(n, 1)
            c_full = G[iu]
            m = float(c_full.max())
            thr_c = m - 1e-9
            sel_c = c_full > thr_c
            Ic = iu[0][sel_c]; Jc = iu[1][sel_c]
            contacts = np.zeros(n, dtype=int)
            np.add.at(contacts, Ic, 1)
            np.add.at(contacts, Jc, 1)
            base_rattlers = np.where(contacts == 0)[0]
            base_contacts = contacts
        except Exception:
            base_rattlers = np.array([], dtype=int)
    # ensure at most 2 rattlers (take lowest contacts if more)
    try:
        if len(base_rattlers) > 2:
            # keep 2 with smallest dual? just first 2
            base_rattlers = base_rattlers[:2]
    except Exception:
        pass
    # per-point max for tie-break
    try:
        G0 = best @ best.T
        np.fill_diagonal(G0, -2.0)
        ppm0 = G0.max(axis=1)
    except Exception:
        ppm0 = base_w

    configs = [(4, 1e-7), (3, 1e-7), (5, 1e-7), (7, 1e-7), (4, 1e-9), (5, 1e-8), (3, 1e-9), (7, 1e-9), (4, 5e-8), (5, 1e-9)]
    rng = np.random.RandomState(1)
    trial = 0
    for (hub_count, slack) in configs:
        if time.time() >= explore_deadline:
            break
        trial += 1
        # recompute dual weights for this slack if different
        if slack != 1e-7:
            try:
                w_s, c_s, r_s = get_dual_weights(best, cap=3000, radius=1e-3, slack=slack)
                if w_s is not None and np.any(w_s > 0):
                    w_use = w_s
                    ratt_use = r_s
                else:
                    w_use = base_w
                    ratt_use = base_rattlers
            except Exception:
                w_use = base_w
                ratt_use = base_rattlers
        else:
            w_use = base_w
            ratt_use = base_rattlers
        try:
            if ratt_use is None or len(ratt_use) == 0:
                ratt_list = []
            else:
                ratt_list = list(np.asarray(ratt_use).ravel()[:2])
            # hubs: highest stress excluding rattlers
            order = np.argsort(-np.asarray(w_use, dtype=float), kind="stable")
            # tie-break by ppm0 for stability
            hubs = []
            for idx in order:
                if idx in ratt_list:
                    continue
                hubs.append(int(idx))
                if len(hubs) >= hub_count:
                    break
            remove_idx = sorted(set(hubs + [int(r) for r in ratt_list]))
            n_add = len(remove_idx)
            if n_add == 0:
                continue
            X_keep = np.delete(best, remove_idx, axis=0)
            X_keep /= np.linalg.norm(X_keep, axis=1, keepdims=True)
            Xf = X_keep.copy()
            # reinsert one at a time
            ok = True
            for a in range(n_add):
                if time.time() >= explore_deadline:
                    ok = False
                    break
                # 60000 random unit vectors vectorially
                try:
                    C_all = rng.randn(60000, 9)
                    C_all /= np.linalg.norm(C_all, axis=1, keepdims=True)
                except Exception:
                    ok = False
                    break
                # score in chunks
                try:
                    Dmax_all = np.empty(60000, dtype=float)
                    bs = 15000
                    for s in range(0, 60000, bs):
                        e = min(60000, s + bs)
                        Dmax_all[s:e] = (C_all[s:e] @ Xf.T).max(axis=1)
                    # best 32
                    if np.any(~np.isfinite(Dmax_all)):
                        Dmax_all[~np.isfinite(Dmax_all)] = 1e9
                    idx32 = np.argpartition(Dmax_all, 32)[:32]
                    Vs = C_all[idx32].copy()
                except Exception:
                    ok = False
                    break
                try:
                    Vs = refine_vs(Vs, Xf, steps=300)
                except Exception:
                    pass
                try:
                    Tm = (Vs @ Xf.T).max(axis=1)
                    bi = int(np.argmin(Tm))
                    v_best = Vs[bi].copy()
                    v_best /= np.linalg.norm(v_best)
                    if not np.all(np.isfinite(v_best)):
                        ok = False
                        break
                    Xf = np.vstack([Xf, v_best.reshape(1, 9)])
                except Exception:
                    ok = False
                    break
            if not ok:
                continue
            # Xf should be 180 x 9
            if Xf.shape != (180, 9):
                continue
            Xf /= np.linalg.norm(Xf, axis=1, keepdims=True)
            # capped LP polish at most 8 seconds
            remain = explore_deadline - time.time()
            if remain < 1.0:
                break
            secs = min(8.0, max(1.0, remain - 1.0))
            # if last configs and little time, use smaller polish
            try:
                Xp = lp_polish_capped(Xf, secs, cap=3000, radius_init=4e-3)
                vp = max_cos(Xp)
                # accept only true improvement
                if vp < best_val and np.all(np.isfinite(Xp)):
                    best = Xp / np.linalg.norm(Xp, axis=1, keepdims=True)
                    best_val = vp
                    # refresh dual baseline for next trials
                    try:
                        nw, nc, nr = get_dual_weights(best, cap=3000, radius=1e-3, slack=1e-7)
                        if nw is not None:
                            base_w = nw
                            base_rattlers = nr
                            base_contacts = nc
                            if base_rattlers is not None and len(base_rattlers) > 2:
                                base_rattlers = base_rattlers[:2]
                            G0 = best @ best.T
                            np.fill_diagonal(G0, -2.0)
                            ppm0 = G0.max(axis=1)
                    except Exception:
                        pass
            except Exception:
                continue
        except Exception:
            continue
        if time.time() >= explore_deadline:
            break
    # final safety: ensure best finite
    try:
        best = best / np.linalg.norm(best, axis=1, keepdims=True)
    except Exception:
        pass
    return best
