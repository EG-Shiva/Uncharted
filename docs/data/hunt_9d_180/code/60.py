import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(np.max(g))


def lp_polish_capped(x, seconds, cap=3000, deadline=None):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = max_cos(x)
    radius = 1e-3
    t_start = time.time()
    if deadline is None:
        deadline = t_start + seconds
    end = min(t_start + seconds, deadline)
    iu = np.triu_indices(n, 1)
    while time.time() < end and radius > 1e-14:
        Gfull = x @ x.T
        cv = Gfull[iu]
        m = float(np.max(cv))
        thr = m - max(20 * radius, 1e-9)
        sel_idx = np.where(cv > thr)[0]
        if sel_idx.size == 0:
            break
        if sel_idx.size > cap:
            # keep tightest cap
            part = sel_idx[np.argpartition(-cv[sel_idx], cap)[:cap]]
            sel_idx = part[np.argsort(-cv[part])]
        I = iu[0][sel_idx]
        J = iu[1][sel_idx]
        csub = cv[sel_idx]
        p = len(I)
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
            res = linprog(obj, A_ub=A_ub, b_ub=-csub, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        try:
            y = x + res.x[:-1].reshape(n, d)
        except Exception:
            radius *= 0.5
            continue
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        v = max_cos(y)
        if v < best:
            x, best, radius = y, v, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
        if time.time() >= end:
            break
    return x


def dual_stress(x, cap=3000, slack_thresh=1e-7, radius=1e-3):
    n, d = x.shape
    iu = np.triu_indices(n, 1)
    G = x @ x.T
    cv = G[iu]
    m = float(np.max(cv))
    order = np.argsort(-cv)
    if order.size > cap:
        order = order[:cap]
    I = iu[0][order]
    J = iu[1][order]
    csub = cv[order]
    slack = m - csub
    # contacts for rattler detection (1e-9 threshold on full matrix)
    contact_counts = np.zeros(n, dtype=int)
    sel_c = cv > m - 1e-9
    Ic = iu[0][sel_c]
    Jc = iu[1][sel_c]
    for a, b in zip(Ic, Jc):
        contact_counts[a] += 1
        contact_counts[b] += 1
    # build LP once
    p = len(I)
    nv = n * d + 1
    rows = np.repeat(np.arange(p), 2 * d + 1)
    cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
    vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
    w = np.zeros(n)
    try:
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv)
        obj[-1] = 1.0
        res = linprog(obj, A_ub=A_ub, b_ub=-csub, A_eq=A_eq, b_eq=np.zeros(n),
                      bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        marg = None
        try:
            if hasattr(res, "ineqlin") and res.ineqlin is not None:
                marg = np.asarray(res.ineqlin.marginals, dtype=float)
        except Exception:
            marg = None
        if marg is None:
            try:
                marg = np.asarray(res.ineqlin["marginals"], dtype=float)
            except Exception:
                marg = None
        if marg is not None and marg.size == p and res.status == 0:
            am = np.abs(marg)
            mask = slack < slack_thresh
            # if too few, relax to top quantile
            if np.sum(mask) < 50:
                # take top 500 by slack
                q = np.argsort(slack)[:500]
                mask = np.zeros(p, dtype=bool)
                mask[q] = True
            for k in np.where(mask)[0]:
                a = int(I[k]); b = int(J[k]); v = float(am[k])
                w[a] += v
                w[b] += v
            if np.sum(w) <= 0:
                w = contact_counts.astype(float)
        else:
            w = contact_counts.astype(float)
    except Exception:
        w = contact_counts.astype(float)
    return w, contact_counts


def refine_block(C, others, steps=300, T0=0.02, T1=0.0004, lr=0.06):
    K = C.shape[0]
    C = C / np.linalg.norm(C, axis=1, keepdims=True)
    for s in range(steps):
        frac = s / max(1, steps - 1)
        T = T0 * (T1 / T0) ** frac
        D = C @ others.T
        dmax = np.max(D, axis=1, keepdims=True)
        E = np.exp((D - dmax) / T)
        W = E / (np.sum(E, axis=1, keepdims=True) + 1e-30)
        Gr = W @ others
        dot = np.sum(Gr * C, axis=1, keepdims=True)
        Gt = Gr - dot * C
        C = C - lr * Gt
        C /= np.linalg.norm(C, axis=1, keepdims=True) + 1e-30
    return C


def insert_one(others, rng):
    d = others.shape[1]
    NTRY = 60000
    K = 32
    cand = rng.randn(NTRY, d)
    cand /= np.linalg.norm(cand, axis=1, keepdims=True)
    dots = cand @ others.T
    m = np.max(dots, axis=1)
    # best K
    idx = np.argpartition(m, K)[:K]
    C = cand[idx].copy()
    C = refine_block(C, others, steps=300, T0=0.02, T1=0.0004, lr=0.06)
    Df = C @ others.T
    mf = np.max(Df, axis=1)
    bi = int(np.argmin(mf))
    return C[bi].copy(), float(mf[bi])


def solve():
    t0 = time.time()
    deadline = t0 + 178.0
    explore_deadline = t0 + 140.0
    np.random.seed(0)
    rng = np.random.RandomState(0)
    try:
        xb = np.load("best.npy")
    except Exception:
        xb = None
    try:
        xr = np.load("record.npy")
    except Exception:
        xr = None
    cands = []
    for c in (xb, xr):
        try:
            if c is None:
                continue
            c = np.asarray(c, dtype=float)
            if c.shape == (180, 9):
                c = c / np.linalg.norm(c, axis=1, keepdims=True)
                if np.all(np.isfinite(c)):
                    cands.append((max_cos(c), c))
        except Exception:
            pass
    if len(cands) == 0:
        x_best = rng.randn(180, 9)
        x_best /= np.linalg.norm(x_best, axis=1, keepdims=True)
        b_best = max_cos(x_best)
    else:
        cands.sort(key=lambda t: t[0])
        b_best, x_best = cands[0][0], cands[0][1].copy()
    n, d = 180, 9

    # polish best.npy once with constraints capped at 3000 tightest pairs
    try:
        rem = explore_deadline - time.time()
        if rem > 6:
            xp = lp_polish_capped(x_best, seconds=min(5.0, rem - 2.0), cap=3000, deadline=deadline - 1.0)
            vp = max_cos(xp)
            if vp < b_best:
                x_best = xp
                b_best = vp
    except Exception:
        pass

    # surgery configs: hub counts 3,4,5,7 with dual-quantile choices
    configs = [
        (4, 3000, 1e-7),
        (3, 3000, 1e-7),
        (5, 3000, 1e-7),
        (7, 3000, 1e-7),
        (4, 2000, 1e-6),
        (5, 2000, 1e-7),
        (3, 1500, 5e-8),
        (7, 2500, 5e-7),
    ]
    ci = 0
    # hard 140s exploration deadline
    while time.time() < explore_deadline - 9.0 and time.time() < deadline - 12.0:
        hub_k, cap, slack_thresh = configs[ci % len(configs)]
        ci += 1
        try:
            # extract HiGHS duals
            w, contacts = dual_stress(x_best, cap=cap, slack_thresh=slack_thresh, radius=1e-3)
        except Exception:
            continue
        # rattlers: 0 contacts
        rattlers = list(np.where(contacts == 0)[0])
        # keep both rattlers (up to 2); if fewer, supplement with lowest-contact points
        if len(rattlers) > 2:
            rattlers = rattlers[:2]
        if len(rattlers) < 2:
            order_c = np.argsort(contacts)
            for o in order_c:
                if int(o) not in rattlers:
                    rattlers.append(int(o))
                if len(rattlers) >= 2:
                    break
        # hubs: highest stress
        order_w = np.argsort(-w)
        hubs = []
        for o in order_w:
            o = int(o)
            if o in rattlers or o in hubs:
                continue
            hubs.append(o)
            if len(hubs) >= hub_k:
                break
        removal = list(hubs) + list(rattlers)
        # ensure 6ish unique; if hub_k=4 -> 6 points
        removal = sorted(set(removal))
        if len(removal) == 0:
            continue
        # frozen remaining
        mask = np.ones(n, dtype=bool)
        mask[removal] = False
        base_remaining = x_best[mask].copy()
        placed = []
        # reinsert one at a time against frozen remaining + already placed
        for _ in removal:
            if time.time() >= explore_deadline - 8.5:
                break
            if len(placed) == 0:
                others = base_remaining
            else:
                others = np.vstack([base_remaining] + placed)
            try:
                y, _ = insert_one(others, rng)
            except Exception:
                y = rng.randn(d)
                y /= np.linalg.norm(y)
            placed.append(y.reshape(1, -1))
        if len(placed) != len(removal):
            continue
        x_try = x_best.copy()
        for idx_rm, pv in zip(removal, placed):
            x_try[idx_rm] = pv[0]
        x_try /= np.linalg.norm(x_try, axis=1, keepdims=True)
        # at most 8 seconds capped LP polish
        try:
            rem_exp = explore_deadline - time.time()
            rem_dl = deadline - time.time() - 1.0
            sec = min(8.0, rem_exp, rem_dl)
            if sec < 1.0:
                break
            x_pol = lp_polish_capped(x_try, seconds=sec, cap=3000, deadline=deadline - 0.5)
            v_pol = max_cos(x_pol)
            if v_pol < b_best - 1e-12 and np.all(np.isfinite(x_pol)):
                x_best = x_pol / np.linalg.norm(x_pol, axis=1, keepdims=True)
                b_best = float(v_pol)
        except Exception:
            continue
    x_best = x_best / np.linalg.norm(x_best, axis=1, keepdims=True)
    return x_best
