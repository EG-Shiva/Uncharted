import time
import numpy as np
from scipy.optimize import minimize, linprog
from scipy.sparse import coo_matrix

def _norm(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return X / n

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _max_cnt(X, gap=1e-7):
    n = X.shape[0]
    iu = np.triu_indices(n, 1)
    c = (X @ X.T)[iu]
    m = c.max()
    cnt = int(np.count_nonzero(c > m - gap))
    return m, cnt, c, m

def _lbfgs_run(X0, fun, maxiter):
    Y0 = _norm(X0).ravel()
    try:
        res = minimize(fun, Y0, jac=True, method='L-BFGS-B',
                       options={'maxiter': int(maxiter), 'ftol': 1e-14,
                                'gtol': 1e-7, 'maxcor': 10, 'maxls': 20})
        Y = _norm(res.x.reshape(X0.shape))
        return Y
    except Exception:
        return _norm(X0)

def _make_lse(beta, iu):
    def fun(Yf):
        n, d = 138, 8
        Ym = Yf.reshape(n, d)
        nr = np.sqrt(np.sum(Ym * Ym, axis=1, keepdims=True)) + 1e-30
        X = Ym / nr
        G = X @ X.T
        c = G[iu]
        mv = c.max()
        E = np.exp(beta * (c - mv))
        S = E.sum() + 1e-300
        f = float(mv + np.log(S) / beta)
        W = np.zeros((n, n))
        W[iu] = E / S
        W = W + W.T
        GX = W @ X
        dot = np.sum(GX * X, axis=1, keepdims=True)
        Gt = GX - dot * X
        GY = Gt / nr
        return f, GY.ravel()
    return fun

def _make_topk(K, iu):
    def fun(Yf):
        n, d = 138, 8
        Ym = Yf.reshape(n, d)
        nr = np.sqrt(np.sum(Ym * Ym, axis=1, keepdims=True)) + 1e-30
        X = Ym / nr
        G = X @ X.T
        c = G[iu]
        # indices of largest K
        if K >= c.size:
            sel = np.ones_like(c, dtype=bool)
        else:
            part = np.argpartition(-c, K - 1)[:K]
            sel = np.zeros_like(c, dtype=bool)
            sel[part] = True
        f = float(c[sel].sum() / K)
        W = np.zeros((n, n))
        Wij = np.zeros_like(c)
        Wij[sel] = 1.0 / K
        W[iu] = Wij
        W = W + W.T
        GX = W @ X
        dot = np.sum(GX * X, axis=1, keepdims=True)
        Gt = GX - dot * X
        GY = Gt / nr
        return f, GY.ravel()
    return fun

def _make_smoothcount(tau, sharp, iu):
    def fun(Yf):
        n, d = 138, 8
        Ym = Yf.reshape(n, d)
        nr = np.sqrt(np.sum(Ym * Ym, axis=1, keepdims=True)) + 1e-30
        X = Ym / nr
        G = X @ X.T
        c = G[iu]
        z = sharp * (c - tau)
        z = np.clip(z, -50.0, 50.0)
        s = 1.0 / (1.0 + np.exp(-z))
        f = float(s.sum())
        w = s * (1.0 - s) * sharp
        W = np.zeros((n, n))
        W[iu] = w
        W = W + W.T
        GX = W @ X
        dot = np.sum(GX * X, axis=1, keepdims=True)
        Gt = GX - dot * X
        # scale gradient to O(1): divide by sharp
        GY = Gt / (nr * sharp)
        f_scaled = f / sharp
        return f_scaled, GY.ravel()
    return fun

def _lp_polish(x, deadline, r0=1e-2, cap=3000, active_gap=1e-3):
    x = _norm(x)
    n, d = x.shape
    try:
        best = _max_cos(x)
    except Exception:
        return x
    radius = r0
    iu = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        Gfull = x @ x.T
        c = Gfull[iu]
        m = c.max()
        thr = m - active_gap
        sel = c > thr
        cnt = int(np.count_nonzero(sel))
        if cnt == 0:
            break
        if cnt > cap:
            idx = np.where(sel)[0]
            order = np.argsort(-c[idx])[:cap]
            mask = np.zeros_like(sel, dtype=bool)
            mask[idx[order]] = True
            sel = mask
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate(
            [I[:, None] * d + np.arange(d),
             J[:, None] * d + np.arange(d),
             np.full((p, 1), n * d)], 1
        ).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix(
                (x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))),
                shape=(n, nv)).tocsr()
            obj = np.zeros(nv)
            obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-c[sel], A_eq=A_eq,
                          b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                          method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y = _norm(y)
        v = _max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.05)
            if radius < 1e-3:
                radius = 1e-3
        else:
            radius *= 0.5
    return x

def solve():
    import time as _tm
    t0 = _tm.time()
    deadline = t0 + 168.0
    n, d = 138, 8
    rng = np.random.default_rng(0)
    best = None
    best_val = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            v = np.load(fn).astype(float)
            if v.shape == (n, d):
                v = _norm(v)
                c = _max_cos(v)
                if c < best_val:
                    best_val = c
                    best = v.copy()
        except Exception:
            pass
    if best is None:
        best = _norm(rng.standard_normal(size=(n, d)))
        best_val = _max_cos(best)
    m_ref = float(best_val)
    iu = np.triu_indices(n, 1)

    # current de-jam point
    Xcur = best.copy()
    m_cur, cnt_cur, _, _ = _max_cnt(Xcur, 1e-7)
    best_cnt = cnt_cur

    # build stage list
    stages = []
    stages.append((_make_lse(200, iu), 60))
    stages.append((_make_lse(600, iu), 60))
    stages.append((_make_lse(1500, iu), 80))
    stages.append((_make_topk(50, iu), 50))
    stages.append((_make_topk(200, iu), 50))
    stages.append((_make_topk(500, iu), 50))
    # smooth counts use tau relative to m_ref
    for delta, sharp in ((1e-5, 2e5), (1e-6, 1e6), (1e-7, 3e6)):
        tau = m_ref - delta
        stages.append((_make_smoothcount(tau, sharp, iu), 50))

    # first chain from incumbent
    for si, (fun, mit) in enumerate(stages):
        if _tm.time() > deadline - 55:
            break
        try:
            Xnew = _lbfgs_run(Xcur, fun, mit)
        except Exception:
            continue
        try:
            m_new, cnt_new, _, _ = _max_cnt(Xnew, 1e-7)
        except Exception:
            continue
        # accept if reduces count within 1e-7, allowing max rise 1e-5
        if m_new <= m_ref + 1e-5 + 1e-12:
            if (cnt_new < best_cnt) or (cnt_new == best_cnt and m_new < m_cur - 1e-13):
                Xcur = Xnew
                m_cur, best_cnt = m_new, cnt_new
                if m_new < best_val - 1e-12:
                    best_val = m_new
                    best = Xnew.copy()
        # also track global best even if count not improved
        try:
            if m_new < best_val - 1e-12 and m_new <= m_ref + 1e-5:
                pass
        except Exception:
            pass
        # update tau for remaining smooth stages if m_ref improved? keep fixed
    # keep de-jammed candidate
    dejam1 = Xcur.copy()
    m_d1, c_d1 = m_cur, best_cnt

    # second chain from small perturbation if time allows
    dejam2 = None
    try:
        if _tm.time() < deadline - 80:
            Xp = best.copy() + 0.004 * rng.standard_normal(size=(n, d))
            Xp = _norm(Xp)
            Xc2 = Xp
            m_c2, cnt_c2, _, _ = _max_cnt(Xc2, 1e-7)
            # short chain: LSE 200,600 + top200 + smooth 1e-6
            short = [(_make_lse(200, iu), 40), (_make_lse(600, iu), 40),
                     (_make_topk(200, iu), 30), (_make_smoothcount(m_ref - 1e-6, 1e6, iu), 30)]
            for fun, mit in short:
                if _tm.time() > deadline - 55:
                    break
                Xn = _lbfgs_run(Xc2, fun, mit)
                mn, cn, _, _ = _max_cnt(Xn, 1e-7)
                if mn <= m_ref + 1e-5 + 1e-12:
                    if (cn < cnt_c2) or (cn == cnt_c2 and mn < m_c2):
                        Xc2, m_c2, cnt_c2 = Xn, mn, cn
            dejam2 = Xc2.copy()
    except Exception:
        dejam2 = None

    # collect candidates to LP-polish: prefer low count, then low max
    cands = [(m_d1, c_d1, dejam1)]
    if dejam2 is not None:
        try:
            md2, cd2, _, _ = _max_cnt(dejam2, 1e-7)
            cands.append((md2, cd2, dejam2))
        except Exception:
            pass
    cands.append((best_val, 10**9, best.copy()))
    # sort by count then max? we want to polish low-count ones first
    # put lowest count first
    try:
        cands_sorted = sorted(cands, key=lambda t: (t[1], t[0]))
    except Exception:
        cands_sorted = cands
    # polish up to 3 candidates with time split
    remaining = deadline - _tm.time()
    # reserve 8s for final polish of global best
    for ci, (mm, cc, Xc) in enumerate(cands_sorted[:3]):
        if _tm.time() > deadline - 10:
            break
        budget = 6.0 if ci < 2 else 4.0
        # don't exceed remaining
        if _tm.time() + budget > deadline - 8:
            budget = max(1.0, deadline - 8 - _tm.time())
        try:
            Xp = _lp_polish(Xc, min(deadline - 8, _tm.time() + budget), r0=1e-2)
            v = _max_cos(Xp)
            if v < best_val - 1e-13:
                best_val = v
                best = Xp.copy()
        except Exception:
            continue

    # final polish of incumbent
    try:
        if _tm.time() < deadline - 1.0:
            best = _lp_polish(best, deadline - 0.3, r0=1e-2)
    except Exception:
        pass
    best = _norm(best)
    return best
