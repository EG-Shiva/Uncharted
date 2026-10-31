import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def _norm(x):
    x = np.asarray(x, dtype=float)
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return x / n


def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())


def contact_counts(x, tol=1e-9):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    m = g.max()
    cnt = np.sum(g > m - tol, axis=1)
    mx = g.max(axis=1)
    return cnt, mx, m


def lp_polish(x, seconds):
    x = _norm(x)
    if seconds <= 0.02:
        return x
    n, d = x.shape
    best = max_cos(x)
    best_x = x.copy()
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        # time guard per iteration
        if time.time() - start > seconds - 0.02:
            break
        Gfull = x @ x.T
        c = Gfull[iu]
        m = c.max()
        sel = c > m - max(20 * radius, 1e-9)
        if np.count_nonzero(sel) == 0:
            break
        I = iu[0][sel]
        J = iu[1][sel]
        p = len(I)
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        # cols
        cols_I = (I[:, None] * d + np.arange(d)).ravel()
        cols_J = (J[:, None] * d + np.arange(d)).ravel()
        cols_t = np.full(p, n * d)
        cols = np.empty(p * (2 * d + 1), dtype=int)
        cols[0::2 * d + 1] = cols_I[0::d]  # placeholder, rebuild correctly below
        # build correctly via concatenate (simpler)
        cols = np.concatenate([ (I[:, None] * d + np.arange(d)),
                                (J[:, None] * d + np.arange(d)),
                                np.full((p, 1), n * d)], axis=1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], axis=1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv)
        obj[-1] = 1.0
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
        y = _norm(y)
        v = max_cos(y)
        if v < best:
            best = v
            best_x = y.copy()
            x = y
            radius = min(radius * 1.5, 0.05)
        else:
            # accept sideways move of x? no, just shrink
            # if very close, still move to explore
            radius *= 0.5
            # keep x at best? x stays
    return _norm(best_x)


def smooth_relax(x, iters, rng, k0=6.0, k1=50.0, lr0=0.06, lr1=0.004, mu=0.75, noise=0.0):
    x = _norm(x)
    n, d = x.shape
    v = np.zeros_like(x)
    for t in range(iters):
        frac = t / max(1, iters - 1)
        k = k0 * (k1 / k0) ** frac
        lr = lr0 * (lr1 / lr0) ** frac
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        moff = G.max()
        W = np.exp(k * (G - moff))
        np.fill_diagonal(W, 0.0)
        rs = W.sum(axis=1, keepdims=True) + 1e-12
        grad = (W @ x) / rs
        dot = np.sum(grad * x, axis=1, keepdims=True)
        gt = grad - dot * x
        v = mu * v + gt
        # bias correction
        bc = 1.0 - mu ** (t + 1) if mu > 0 else 1.0
        vc = v / bc if bc != 0 else v
        x = x - lr * vc
        if noise > 0 and frac < 0.6:
            amp = noise * (1.0 - frac / 0.6)
            pert = rng.standard_normal(x.shape)
            pd = np.sum(pert * x, axis=1, keepdims=True)
            pert -= pd * x
            x = x + amp * pert
        x = _norm(x)
    return x


def farthest_insertion(Xkeep, rng, ncand=20000):
    d = Xkeep.shape[1]
    cand = rng.standard_normal((ncand, d))
    cand = _norm(cand)
    dots = cand @ Xkeep.T
    md = dots.max(axis=1)
    idx = int(np.argmin(md))
    return cand[idx]


def rebuild_to_153(x, rng, t0, deadline, lp_short=0.6, smooth_iters=150, ncand=20000):
    # x has N<153, grow to 153
    x = _norm(x)
    while x.shape[0] < 153:
        if time.time() > deadline:
            break
        Xkeep = x
        try:
            v = farthest_insertion(Xkeep, rng, ncand=ncand)
        except Exception:
            # fallback small sample
            v = farthest_insertion(Xkeep, rng, ncand=2000)
        x = np.vstack([Xkeep, v.reshape(1, -1)])
        x = _norm(x)
        # short smooth relaxation
        try:
            x = smooth_relax(x, smooth_iters, rng)
        except Exception:
            pass
        # short LP
        try:
            # use smaller budget as N grows
            x = lp_polish(x, lp_short)
        except Exception:
            pass
    return _norm(x)


def solve():
    t0 = time.time()
    n, d = 153, 4
    rng_master = np.random.default_rng(12345)
    # load record
    loaded = None
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=float)
            if a.shape == (n, d):
                loaded = _norm(a)
                break
        except Exception:
            continue
    if loaded is None:
        tmp = rng_master.standard_normal((n, d))
        loaded = _norm(tmp)
    incumbent = loaded.copy()
    try:
        inc_val = max_cos(incumbent)
    except Exception:
        inc_val = 1.0

    # time plan: search until t0+~125, reserve 30s final polish, hard cap 170
    SEARCH_END = t0 + 125.0
    FINAL_RESERVE = 28.0

    def time_ok():
        return time.time() < SEARCH_END

    branch_id = -1
    # pre-defined low targets
    low_cycle = [152, 151, 150, 149, 148, 152, 150]
    try:
        nb = 0
        while time_ok() and nb < 30:
            nb += 1
            branch_id += 1
            seed = 1000 + branch_id * 17
            rng = np.random.default_rng(seed)
            mode = branch_id % 4  # 0: deterministic down, 1: random high-contact, 2: random high-contact other low, 3: overshoot
            if mode in (0, 1, 2):
                low = low_cycle[branch_id % len(low_cycle)]
                # start from incumbent or loaded
                if branch_id % 3 == 0:
                    base = incumbent.copy()
                elif branch_id % 3 == 1:
                    base = loaded.copy()
                else:
                    base = incumbent.copy()
                    # small kick to diversify before down
                    try:
                        sigma = 0.01
                        base = base + sigma * rng.standard_normal(base.shape)
                        base = _norm(base)
                    except Exception:
                        pass
                x = base.copy()
                # descend to low
                ok = True
                cur = x.shape[0]
                # need cur==153 generally
                while x.shape[0] > low:
                    if not time_ok():
                        ok = False
                        break
                    try:
                        cnt, mx, m = contact_counts(_norm(x))
                    except Exception:
                        ok = False
                        break
                    rm = -1
                    try:
                        if mode == 0:
                            # deterministic highest-contact
                            order = np.lexsort((-mx, -cnt))
                            rm = int(order[0])
                        else:
                            # random among high-contact
                            k = 10 if mode == 1 else 6
                            order = np.lexsort((-mx, -cnt))
                            cand_idx = order[:min(k, len(order))]
                            # weighted toward higher contacts
                            w = (cnt[cand_idx].astype(float) + 1.0)
                            w = w / w.sum()
                            rm = int(rng.choice(cand_idx, p=w))
                    except Exception:
                        rm = int(rng.integers(x.shape[0]))
                    try:
                        mask = np.ones(x.shape[0], dtype=bool)
                        mask[rm] = False
                        x = _norm(x[mask])
                    except Exception:
                        ok = False
                        break
                    # brief LP polish after each removal
                    try:
                        x = lp_polish(x, 0.7)
                    except Exception:
                        pass
                    # tiny smooth to settle
                    if x.shape[0] > 140:
                        try:
                            x = smooth_relax(x, 60, rng)
                            x = lp_polish(x, 0.3)
                        except Exception:
                            pass
                if not ok:
                    continue
                # rebuild to 153
                try:
                    x = rebuild_to_153(x, rng, t0, SEARCH_END, lp_short=0.6, smooth_iters=140, ncand=20000)
                except Exception:
                    continue
                if x.shape[0] != 153:
                    continue
                try:
                    v = max_cos(_norm(x))
                except Exception:
                    continue
                if v < inc_val - 1e-12:
                    inc_val = v
                    incumbent = _norm(x).copy()
                elif v < inc_val + 2e-4 and rng.random() < 0.1:
                    # keep diverse basin occasionally for further continuation? store as loaded alt
                    # we keep incumbent best only, but allow next branch from this x
                    loaded_alt = _norm(x).copy()
                    # use it as base once
                    try:
                        alt_v = v
                        # quick extra polish attempt
                        x2 = lp_polish(loaded_alt, 0.8)
                        v2 = max_cos(x2)
                        if v2 < inc_val:
                            inc_val = v2
                            incumbent = x2.copy()
                    except Exception:
                        pass
            else:
                # overshoot to 154-156 then prune
                target = int(rng.choice(np.array([154, 155, 156])))
                base = incumbent.copy()
                x = base.copy()
                # grow
                grew = True
                while x.shape[0] < target:
                    if not time_ok():
                        grew = False
                        break
                    try:
                        v = farthest_insertion(_norm(x), rng, ncand=20000)
                    except Exception:
                        grew = False
                        break
                    x = _norm(np.vstack([x, v.reshape(1, -1)]))
                    try:
                        x = smooth_relax(x, 80, rng)
                    except Exception:
                        pass
                if not grew:
                    continue
                try:
                    x = lp_polish(x, 0.8)
                except Exception:
                    pass
                # prune down to 153 choosing best removal
                while x.shape[0] > 153:
                    if not time_ok():
                        break
                    Xn = _norm(x)
                    m_cur = max_cos(Xn)
                    # evaluate removal score without polish (fast)
                    try:
                        G = Xn @ Xn.T
                        np.fill_diagonal(G, -2.0)
                        # for each i, max over pairs not involving i
                        # brute loop (156 * 156^2) fine
                        Ncur = Xn.shape[0]
                        best_rm = -1
                        best_score = 1e9
                        for i in range(Ncur):
                            # max of G excluding row/col i
                            # mask: copy is heavy; use max of two parts
                            # simple: compute max over all, if argmax involves i need second pass
                            # do direct masked max via temporary
                            Gi = G.copy()
                            Gi[i, :] = -2.0
                            Gi[:, i] = -2.0
                            sc = Gi.max()
                            if sc < best_score:
                                best_score = sc
                                best_rm = i
                        # consider top-5 candidates by this proxy for polished selection
                        scores = np.empty(Ncur)
                        for i in range(Ncur):
                            Gi = G.copy()
                            Gi[i, :] = -2.0
                            Gi[:, i] = -2.0
                            scores[i] = Gi.max()
                        order = np.argsort(scores)[:5]
                    except Exception:
                        order = [int(rng.integers(x.shape[0]))]
                        best_rm = int(order[0])
                    # polish each candidate pruned config briefly and pick best
                    best_x = None
                    best_v = 1e9
                    for cand_rm in order:
                        if time.time() > SEARCH_END:
                            break
                        try:
                            mask = np.ones(Xn.shape[0], dtype=bool)
                            mask[int(cand_rm)] = False
                            Xp = _norm(Xn[mask])
                            Xp = lp_polish(Xp, 0.5)
                            vv = max_cos(Xp)
                            if vv < best_v:
                                best_v = vv
                                best_x = Xp
                        except Exception:
                            continue
                    if best_x is None:
                        # fallback single removal
                        try:
                            mask = np.ones(Xn.shape[0], dtype=bool)
                            mask[int(best_rm)] = False
                            best_x = _norm(Xn[mask])
                            best_v = max_cos(best_x)
                        except Exception:
                            break
                    x = _norm(best_x)
                if x.shape[0] != 153:
                    continue
                try:
                    v = max_cos(_norm(x))
                except Exception:
                    continue
                if v < inc_val:
                    inc_val = v
                    incumbent = _norm(x).copy()
    except Exception:
        pass

    # final polish of best, reserve time
    try:
        elapsed = time.time() - t0
        # leave at least 5s margin under 175s
        remaining = min(FINAL_RESERVE, max(2.0, 170.0 - elapsed))
        # ensure we don't exceed: cap by SEARCH logic
        polished = lp_polish(_norm(incumbent), remaining)
        pv = max_cos(polished)
        if pv <= inc_val + 1e-12:
            # keep polished if not worse (allow tiny tol)
            if pv < inc_val:
                inc_val = pv
            incumbent = _norm(polished)
        # if polish somehow worse, keep previous incumbent
    except Exception:
        pass

    # fallback safety
    try:
        out = _norm(incumbent)
        # validate
        if out.shape != (n, d) or not np.all(np.isfinite(out)):
            out = _norm(loaded)
        # final renormalize
        out = _norm(out)
        # ensure no zero rows
        return out
    except Exception:
        return _norm(loaded)
