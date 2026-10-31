import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())


def _lp_polish_deadline(x, deadline):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    try:
        best = _max_cos(x)
    except Exception:
        return x
    radius = 1e-3
    iu = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        try:
            c = (x @ x.T)[iu]
        except Exception:
            break
        m = float(c.max())
        thr = max(20.0 * radius, 1e-9)
        sel = c > m - thr
        if int(sel.sum()) == 0:
            break
        I = iu[0][sel]
        J = iu[1][sel]
        csel = c[sel]
        p = len(I)
        nv = n * d + 1
        try:
            rows = np.repeat(np.arange(p), 2 * d + 1)
            cols = np.concatenate(
                [I[:, None] * d + np.arange(d),
                 J[:, None] * d + np.arange(d),
                 np.full((p, 1), n * d)], axis=1).ravel()
            vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], axis=1).ravel()
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix(
                (x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))),
                shape=(n, nv)).tocsr()
            obj = np.zeros(nv)
            obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq,
                          b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                          method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        try:
            y = x + res.x[:-1].reshape(n, d)
            yn = np.linalg.norm(y, axis=1, keepdims=True)
            yn[yn == 0] = 1.0
            y = y / yn
            v = _max_cos(y)
        except Exception:
            radius *= 0.5
            continue
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x


def _smooth_relax(x, iters, k0=6.0, k1=50.0, lr0=0.02, lr1=0.002, mu=0.8):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    v = np.zeros_like(x)
    for t in range(iters):
        frac = t / max(1, iters - 1)
        k = k0 * (k1 / k0) ** frac
        lr = lr0 * (lr1 / lr0) ** frac
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        moff = float(G.max())
        # cap exponent to avoid overflow
        E = k * (G - moff)
        # E <= 0 so exp safe
        W = np.exp(E)
        np.fill_diagonal(W, 0.0)
        rs = W.sum(axis=1, keepdims=True) + 1e-12
        grad = (W @ x) / rs
        dot = np.sum(grad * x, axis=1, keepdims=True)
        gt = grad - dot * x
        v = mu * v + gt
        denom = 1.0 - mu ** (t + 1)
        vc = v / denom if denom != 0 else v
        x = x - lr * vc
        nrm = np.linalg.norm(x, axis=1, keepdims=True)
        nrm[nrm == 0] = 1.0
        x = x / nrm
    return x


def _farthest_point(Xkeep, n_cand, rng):
    d = Xkeep.shape[1]
    cand = rng.standard_normal((n_cand, d))
    nrm = np.linalg.norm(cand, axis=1, keepdims=True)
    nrm[nrm == 0] = 1.0
    cand = cand / nrm
    # chunk to bound memory
    best_score = np.inf
    best_pt = cand[0]
    chunk = 5000
    for s in range(0, n_cand, chunk):
        e = min(s + chunk, n_cand)
        dots = cand[s:e] @ Xkeep.T
        scores = dots.max(axis=1)
        j = int(np.argmin(scores))
        if float(scores[j]) < best_score:
            best_score = float(scores[j])
            best_pt = cand[s + j]
    return best_pt.copy()


def _contact_counts(x, tol=1e-7):
    G = x @ x.T
    m = float(np.max(G - np.eye(G.shape[0]) * 2.0))
    C = (G > m - tol).sum(axis=1) - 1
    return C, m, G


def solve():
    t0 = time.time()
    HARD_END = t0 + 172.0
    SEARCH_END = t0 + 138.0
    rng = np.random.default_rng(12345)
    n, d = 153, 4

    try:
        init = np.load("best.npy")
    except Exception:
        try:
            init = np.load("record.npy")
        except Exception:
            init = rng.standard_normal((n, d))
    init = np.asarray(init, dtype=float)
    if init.shape != (n, d):
        tmp = rng.standard_normal((n, d))
        # try to salvage
        try:
            init = np.asarray(init, dtype=float).reshape(n, d)
        except Exception:
            init = tmp
    nrm = np.linalg.norm(init, axis=1, keepdims=True)
    nrm[nrm == 0] = 1.0
    init = init / nrm

    try:
        init_val = _max_cos(init)
    except Exception:
        init = rng.standard_normal((n, d))
        init = init / np.linalg.norm(init, axis=1, keepdims=True)
        init_val = _max_cos(init)

    best153 = init.copy()
    best_val = float(init_val)

    branch = 0
    # main variable-N continuation loop
    while time.time() < SEARCH_END and time.time() < HARD_END - 32:
        try:
            mode = branch % 4
            if mode in (0, 1):
                randomize = (mode == 1)
                Ntarget = 148 + ((branch // 4) % 5)  # 148..152
                cur = best153.copy()
                # removal chain down to Ntarget
                ok = True
                for _step in range(n - Ntarget):
                    if time.time() > SEARCH_END or time.time() > HARD_END - 32:
                        ok = False
                        break
                    counts, _, _ = _contact_counts(cur, tol=1e-7)
                    if not randomize:
                        mx = counts.max()
                        top = np.where(counts == mx)[0]
                        idx = int(rng.choice(top)) if len(top) > 1 else int(np.argmax(counts))
                    else:
                        order = np.argsort(-counts, kind="stable")
                        kk = min(10, len(order))
                        idx = int(rng.choice(order[:kk]))
                    cur = np.delete(cur, idx, axis=0)
                    dl = min(time.time() + 1.0, SEARCH_END, HARD_END - 32)
                    if dl - time.time() > 0.05:
                        cur = _lp_polish_deadline(cur, dl)
                if not ok:
                    branch += 1
                    continue
                # rebuild to 153
                while cur.shape[0] < n:
                    if time.time() > SEARCH_END or time.time() > HARD_END - 32:
                        break
                    pt = _farthest_point(cur, 20000, rng)
                    cur = np.vstack([cur, pt.reshape(1, d)])
                    cur = _smooth_relax(cur, 220)
                    dl = min(time.time() + 0.8, SEARCH_END, HARD_END - 32)
                    if dl - time.time() > 0.05:
                        cur = _lp_polish_deadline(cur, dl)
                if cur.shape[0] == n:
                    v = _max_cos(cur)
                    if v < best_val:
                        best_val = float(v)
                        best153 = cur.copy()
            else:
                # overshoot to 154-156 then prune
                Nextra = 1 + (branch % 3)  # 1..3
                Nextra = min(Nextra, 3)
                cur = best153.copy()
                for _ in range(Nextra):
                    if time.time() > SEARCH_END or time.time() > HARD_END - 32:
                        break
                    pt = _farthest_point(cur, 20000, rng)
                    cur = np.vstack([cur, pt.reshape(1, d)])
                    cur = _smooth_relax(cur, 180)
                    dl = min(time.time() + 0.6, SEARCH_END, HARD_END - 32)
                    if dl - time.time() > 0.05:
                        cur = _lp_polish_deadline(cur, dl)
                # prune back to 153
                while cur.shape[0] > n:
                    if time.time() > SEARCH_END or time.time() > HARD_END - 32:
                        break
                    Nc = cur.shape[0]
                    G = cur @ cur.T
                    np.fill_diagonal(G, -2.0)
                    rem = np.empty(Nc)
                    for i in range(Nc):
                        mask = np.ones(Nc, dtype=bool)
                        mask[i] = False
                        rem[i] = float(G[np.ix_(mask, mask)].max())
                    order = np.argsort(rem, kind="stable")
                    Kcand = min(5, Nc)
                    cand_idx = order[:Kcand]
                    best_c = None
                    best_cv = np.inf
                    for ci in cand_idx:
                        if time.time() > SEARCH_END or time.time() > HARD_END - 32:
                            break
                        trial = np.delete(cur, int(ci), axis=0)
                        dl = min(time.time() + 0.6, SEARCH_END, HARD_END - 32)
                        if dl - time.time() > 0.05:
                            trial = _lp_polish_deadline(trial, dl)
                        v = _max_cos(trial)
                        if v < best_cv:
                            best_cv = float(v)
                            best_c = trial.copy()
                    if best_c is None:
                        break
                    cur = best_c
                if cur.shape[0] == n:
                    v = _max_cos(cur)
                    if v < best_val:
                        best_val = float(v)
                        best153 = cur.copy()
        except Exception:
            pass
        branch += 1

    # final reserve polish of best
    try:
        tleft = HARD_END - time.time() - 1.5
        tleft = float(min(max(tleft, 5.0), 30.0))
        if tleft > 0.5:
            best153 = _lp_polish_deadline(best153, time.time() + tleft)
            try:
                nv = _max_cos(best153)
                if nv < best_val:
                    best_val = float(nv)
                # if polish somehow worsened (should not), keep it anyway
                # but guard against NaN
                if not np.all(np.isfinite(best153)):
                    raise ValueError("nonfinite")
            except Exception:
                pass
    except Exception:
        pass

    # strict fallback
    try:
        if best153.shape != (n, d) or not np.all(np.isfinite(best153)):
            best153 = init.copy()
            best_val = float(init_val)
        nrm = np.linalg.norm(best153, axis=1, keepdims=True)
        nrm[nrm == 0] = 1.0
        best153 = best153 / nrm
        # if our search is worse than init (numerical noise), return init-polished best
        # keep the better of the two
        try:
            bv = _max_cos(best153)
            if bv > init_val + 1e-12:
                # revert to init if we degraded
                best153 = init.copy()
        except Exception:
            best153 = init.copy()
    except Exception:
        best153 = init.copy()

    return best153
