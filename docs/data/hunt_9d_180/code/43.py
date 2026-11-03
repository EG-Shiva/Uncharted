import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(np.max(g))

def contact_counts(x, thresh=1e-7):
    n = x.shape[0]
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    m = float(np.max(G))
    # contacts: within thresh of max
    C = G > (m - thresh)
    cnt = np.sum(C, axis=1)
    return cnt, m, G

def relocate_to_void(x, idx, rng, n_samples=200000, n_climb=1500):
    n, d = x.shape
    mask = np.ones(n, dtype=bool)
    mask[idx] = False
    others = x[mask]  # (n-1,d)
    best_y = None
    best_v = np.inf
    batch = 50000
    nb = n_samples // batch
    for _ in range(nb):
        cand = rng.randn(batch, d)
        cand /= np.linalg.norm(cand, axis=1, keepdims=True)
        dots = cand @ others.T
        m = np.max(dots, axis=1)
        bi = int(np.argmin(m))
        if float(m[bi]) < best_v:
            best_v = float(m[bi])
            best_y = cand[bi].copy()
    y = best_y
    cur = best_v
    # tangential hill climbing with geometric sigma decay
    s0, s1 = 0.05, 0.0004
    for k in range(n_climb):
        frac = k / max(1, n_climb - 1)
        sig = s0 * (s1 / s0) ** frac
        prop = y + sig * rng.randn(d)
        nrm = np.linalg.norm(prop)
        if nrm == 0:
            continue
        prop /= nrm
        pm = float(np.max(prop @ others.T))
        if pm < cur:
            y = prop
            cur = pm
    return y, cur

def subset_lp_polish(x, S, seconds, deadline):
    # optimize only points in S, core frozen; include frozen-frozen bounds on t
    n, d = x.shape
    S = np.asarray(S, dtype=int)
    k = len(S)
    if k == 0:
        return x
    is_in = np.zeros(n, dtype=bool)
    is_in[S] = True
    pos = np.full(n, -1, dtype=int)
    pos[S] = np.arange(k)
    nv = k * d + 1
    best = max_cos(x)
    radius = 1e-3
    iu = np.triu_indices(n, 1)
    start = time.time()
    # equality matrix sparsity pattern (fixed)
    # build A_eq once structure (values change as x changes -> rebuild each iter)
    while time.time() - start < seconds and time.time() < deadline and radius > 1e-14:
        Gfull = x @ x.T
        iuc = Gfull[iu]
        m = float(np.max(iuc))
        sel = iuc > m - max(20 * radius, 1e-9)
        I = iu[0][sel]; J = iu[1][sel]; cs = iuc[sel]
        p = len(I)
        if p == 0:
            break
        # build A_ub dense then to csr (p x nv), p~1000 nv~300
        try:
            # sparse via coo lists
            rows_l = []
            cols_l = []
            vals_l = []
            # t column
            rows_l.append(np.arange(p))
            cols_l.append(np.full(p, nv - 1))
            vals_l.append(-np.ones(p))
            # for rows involving S, add blocks; loop only over those rows
            # to keep it fast, find involved rows
            mi = is_in[I]
            mj = is_in[J]
            # rows where I in S
            idx_i = np.where(mi)[0]
            if len(idx_i) > 0:
                # x[J] coeffs
                Xr = x[J[idx_i]]  # (q,d)
                Pr = pos[I[idx_i]]  # (q,)
                # expand to coo
                rr = np.repeat(idx_i, d)
                cc = (Pr[:, None] * d + np.arange(d)).ravel()
                vv = Xr.ravel()
                rows_l.append(rr); cols_l.append(cc); vals_l.append(vv)
            idx_j = np.where(mj)[0]
            if len(idx_j) > 0:
                Xr2 = x[I[idx_j]]
                Pr2 = pos[J[idx_j]]
                rr2 = np.repeat(idx_j, d)
                cc2 = (Pr2[:, None] * d + np.arange(d)).ravel()
                vv2 = Xr2.ravel()
                rows_l.append(rr2); cols_l.append(cc2); vals_l.append(vv2)
            rows = np.concatenate(rows_l)
            cols = np.concatenate(cols_l)
            vals = np.concatenate(vals_l)
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            # A_eq k x nv
            er = np.repeat(np.arange(k), d)
            ec = np.arange(k * d)
            ev = x[S].ravel()
            A_eq = coo_matrix((ev, (er, ec)), shape=(k, nv)).tocsr()
        except Exception:
            break
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-cs, A_eq=A_eq, b_eq=np.zeros(k),
                          bounds=[(-radius, radius)] * (k * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        dx = res.x[:-1].reshape(k, d)
        y = x.copy()
        y[S] = x[S] + dx
        y[S] /= np.linalg.norm(y[S], axis=1, keepdims=True)
        v = max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.02)
        else:
            radius *= 0.5
        # early exit if time low
        if time.time() >= deadline:
            break
    return x

def full_lp_polish(x, seconds, deadline):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = max_cos(x)
    radius = 1e-3
    iu = np.triu_indices(n, 1)
    start = time.time()
    while time.time() - start < seconds and time.time() < deadline and radius > 1e-14:
        Gfull = x @ x.T
        iuc = Gfull[iu]
        m = float(np.max(iuc))
        sel = iuc > m - max(20 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        csub = iuc[sel]
        p = len(I)
        if p == 0:
            break
        nv = n * d + 1
        try:
            rows = np.repeat(np.arange(p), 2 * d + 1)
            cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
            vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        except Exception:
            break
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-csub, A_eq=A_eq, b_eq=np.zeros(n),
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

def solve():
    t0 = time.time()
    deadline = t0 + 178.0
    hard = t0 + 160.0
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
        x0 = rng.randn(180, 9)
        x0 /= np.linalg.norm(x0, axis=1, keepdims=True)
        x_best = x0
        b_best = max_cos(x0)
    else:
        cands.sort(key=lambda t: t[0])
        b_best, x_best = cands[0][0], cands[0][1].copy()
    n, d = 180, 9
    # identify free points: <=3 contacts at 1e-7
    try:
        cnt, _, _ = contact_counts(x_best, thresh=1e-7)
        order = np.argsort(cnt * 1000 + np.arange(n) * 1e-6)
        free = [int(i) for i in range(n) if cnt[i] <= 3]
        if len(free) == 0:
            free = [int(i) for i in order[:5]]
        # cap to 12 lowest to respect time
        if len(free) > 12:
            # keep lowest counts
            free_sorted = sorted(free, key=lambda i: (cnt[i], i))
            free = sorted(free_sorted[:12])
    except Exception:
        free = [int(i) for i in np.argsort([0]*n)[:4]]
    # cycle free points a few times
    ncyc = 3
    for cyc in range(ncyc):
        if time.time() >= hard - 2.0 or time.time() >= deadline - 45:
            break
        # refresh free set from current best at start of cycle (except first)
        if cyc > 0:
            try:
                cnt2, _, _ = contact_counts(x_best, thresh=1e-7)
                free2 = [int(i) for i in range(n) if cnt2[i] <= 3]
                if len(free2) > 0 and len(free2) <= 12:
                    free = sorted(free2)
            except Exception:
                pass
        for idx in list(free):
            if time.time() >= hard - 2.0 or time.time() >= deadline - 45:
                break
            try:
                y, _ = relocate_to_void(x_best, idx, rng, n_samples=200000, n_climb=1200)
            except Exception:
                continue
            if y is None or not np.all(np.isfinite(y)):
                continue
            x_try = x_best.copy()
            x_try[idx] = y / np.linalg.norm(y)
            # neighbourhood: 30-40 nearest neighbours by cosine
            try:
                dots = x_try @ x_try[idx]
                dots[idx] = -1e9
                nn = np.argsort(-dots)[:35]
                S = np.concatenate([[idx], nn])
                S = np.unique(S)
            except Exception:
                S = np.array([idx])
            # subset LP on S while core frozen
            rem_hard = hard - time.time()
            rem_dead = (deadline - 42.0) - time.time()
            budget = min(2.5, rem_hard - 0.5, rem_dead)
            if budget < 0.4:
                # still accept relocation if it improves without LP
                v0 = max_cos(x_try)
                if v0 < b_best - 1e-12:
                    x_best = x_try / np.linalg.norm(x_try, axis=1, keepdims=True)
                    b_best = v0
                continue
            try:
                x_pol = subset_lp_polish(x_try, S, seconds=budget, deadline=min(hard - 0.5, deadline - 42.0))
            except Exception:
                x_pol = x_try
            try:
                v = max_cos(x_pol)
            except Exception:
                continue
            if v < b_best - 1e-12 and np.all(np.isfinite(x_pol)):
                x_best = x_pol / np.linalg.norm(x_pol, axis=1, keepdims=True)
                b_best = v
            if time.time() >= hard or time.time() >= deadline - 40:
                break
    # one full LP polish
    try:
        rem = (deadline - 1.0) - time.time()
        if rem > 3:
            x_best = full_lp_polish(x_best, seconds=min(rem, 45.0), deadline=deadline - 0.5)
    except Exception:
        pass
    x_best = x_best / np.linalg.norm(x_best, axis=1, keepdims=True)
    return x_best
