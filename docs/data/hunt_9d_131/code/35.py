import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(np.max(g))

def _norm(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return x / n

def _lp_polish_deadline(x, deadline):
    x = _norm(x)
    n, d = x.shape
    try:
        best = _max_cos(x)
    except Exception:
        return x
    radius = 1e-3
    iu0 = np.triu_indices(n, 1)
    while time.time() < deadline:
        if radius < 1e-14:
            break
        try:
            G = x @ x.T
            c_all = G[iu0]
        except Exception:
            break
        m = float(c_all.max())
        thresh = m - max(20.0 * radius, 1e-9)
        sel = c_all > thresh
        cnt = int(np.count_nonzero(sel))
        if cnt == 0:
            break
        if cnt > 3500:
            idx = np.where(sel)[0]
            order = np.argsort(-c_all[idx])[:3500]
            mask = np.zeros_like(sel)
            mask[idx[order]] = True
            sel = mask
        I = iu0[0][sel]
        J = iu0[1][sel]
        c = c_all[sel]
        p = len(I)
        nv = n * d + 1
        try:
            rows = np.repeat(np.arange(p), 2 * d + 1)
            cols = np.concatenate(
                [I[:, None] * d + np.arange(d),
                 J[:, None] * d + np.arange(d),
                 np.full((p, 1), n * d)], 1).ravel()
            vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix(
                (x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))),
                shape=(n, nv)).tocsr()
            obj = np.zeros(nv)
            obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-c, A_eq=A_eq, b_eq=np.zeros(n),
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
            y = _norm(y)
            if not np.all(np.isfinite(y)):
                radius *= 0.5
                continue
            v = _max_cos(y)
        except Exception:
            radius *= 0.5
            continue
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
        if deadline - time.time() < 0.25:
            break
    return x

def _smooth(x, steps, T0, T1, lr0, lr1, mom=0.85):
    x = _norm(x)
    v = np.zeros_like(x)
    n = x.shape[0]
    for k in range(steps):
        frac = k / max(1, steps - 1)
        T = T0 + (T1 - T0) * frac
        lr = lr0 + (lr1 - lr0) * frac
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        W = np.exp((G - m) / max(T, 1e-6))
        np.fill_diagonal(W, 0.0)
        grad = W @ x
        dot = np.sum(grad * x, axis=1, keepdims=True)
        grad = grad - dot * x
        s = float(W.sum()) / n + 1e-12
        grad = grad / s
        if not np.all(np.isfinite(grad)):
            break
        v = mom * v + (1.0 - mom) * grad
        x = x - lr * v
        x = _norm(x)
    return x

def _hole_for(Xk, rng, n_samp=3000):
    d = Xk.shape[1]
    Z = rng.standard_normal((n_samp, d))
    Z = _norm(Z)
    D = Z @ Xk.T
    worst = D.max(axis=1)
    i = int(np.argmin(worst))
    z = Z[i].copy()
    for _ in range(60):
        dots = Xk @ z
        m = float(dots.max())
        W = np.exp((dots - m) / 0.01)
        W = W / (W.sum() + 1e-12)
        g = W @ Xk
        g = g - float(g @ z) * z
        ng = np.linalg.norm(g)
        if ng < 1e-12:
            break
        z = z - 0.05 * g
        z = z / (np.linalg.norm(z) + 1e-12)
    return z

def _build_ternary():
    N = 3 ** 9
    powers = 3 ** np.arange(9, dtype=np.int64)
    ar = np.arange(N, dtype=np.int64)[:, None]
    vecs = ((ar // powers) % 3) - 1  # -1,0,1
    vecs = vecs.astype(float)
    # remove zero row
    mask = np.any(vecs != 0, axis=1)
    vecs = vecs[mask]
    nr = np.linalg.norm(vecs, axis=1, keepdims=True)
    vecs = vecs / nr
    return vecs

def _greedy_select(pool, k, rng, topk=1, start=None):
    P = pool.shape[0]
    if start is None:
        start = int(rng.integers(P))
    sel_idx = np.empty(k, dtype=int)
    sel_idx[0] = start
    cur = pool @ pool[start]
    cur[start] = np.inf
    for t in range(1, k):
        if topk <= 1:
            nxt = int(np.argmin(cur))
        else:
            tk = min(topk, P - t)
            part = np.argpartition(cur, tk - 1)[:tk]
            nxt = int(part[rng.integers(tk)])
        sel_idx[t] = nxt
        d = pool @ pool[nxt]
        np.maximum(cur, d, out=cur)
        cur[nxt] = np.inf
        # keep previous selected inf
        # (maximum keeps inf)
    return sel_idx

def _swap_improve(pool, sel_idx, deadline, rounds=2, topW=15):
    k = len(sel_idx)
    X = pool[sel_idx].copy()
    sel_set = set(sel_idx.tolist())
    def cur_max(X_):
        G = X_ @ X_.T
        np.fill_diagonal(G, -2.0)
        return float(G.max())
    try:
        m = cur_max(X)
    except Exception:
        return sel_idx, X
    P = pool.shape[0]
    for r in range(rounds):
        if time.time() > deadline:
            break
        try:
            G = X @ X.T
            np.fill_diagonal(G, -2.0)
            rowmax = G.max(axis=1)
            order = np.argsort(-rowmax)[:topW]
        except Exception:
            break
        improved = False
        for i in order:
            if time.time() > deadline:
                break
            try:
                mask = np.ones(k, dtype=bool); mask[i] = False
                Xr = X[mask]
                Gr = Xr @ Xr.T
                np.fill_diagonal(Gr, -2.0)
                rest_max = float(Gr.max())
                dots = pool @ Xr.T
                cmax = dots.max(axis=1)
                total = np.maximum(cmax, rest_max)
                other = sel_idx[mask]
                total[other] = np.inf
                best_j = int(np.argmin(total))
                best_tot = float(total[best_j])
                if best_tot < m - 1e-12:
                    old = int(sel_idx[i])
                    sel_set.discard(old); sel_set.add(best_j)
                    sel_idx[i] = best_j
                    X[i] = pool[best_j]
                    m = best_tot
                    improved = True
                elif best_tot <= m + 1e-12:
                    if float(cmax[best_j]) < float(rowmax[i]) - 1e-9:
                        old = int(sel_idx[i])
                        sel_set.discard(old); sel_set.add(best_j)
                        sel_idx[i] = best_j
                        X[i] = pool[best_j]
                        improved = True
            except Exception:
                continue
        if not improved:
            break
    return sel_idx, X

def solve():
    t0 = time.time()
    DISC_END = t0 + 55.0
    HOP_END = t0 + 150.0
    HARD_END = t0 + 174.0
    rng = np.random.default_rng(7)
    n, d = 131, 9
    xb = None
    best_val = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            v = np.load(fn)
            v = np.asarray(v, dtype=float)
            if v.shape == (n, d) and np.all(np.isfinite(v)):
                v = _norm(v)
                s = _max_cos(v)
                if s < best_val:
                    best_val = s
                    xb = v.copy()
        except Exception:
            continue
    if xb is None:
        xb = _norm(rng.standard_normal((n, d)))
        best_val = _max_cos(xb)

    def contact_edges(x, tol=1e-7):
        try:
            G = x @ x.T
            np.fill_diagonal(G, -2.0)
            m = float(G.max())
            iu = np.triu_indices(n, 1)
            c = G[iu]
            sel = c > m - tol
            I = iu[0][sel]; J = iu[1][sel]
            s = set((int(a) * n + int(b)) for a, b in zip(I, J))
            # canonical by sorted pair not needed since I<J
            rowmax = G.max(axis=1)
            deg = np.sum(G > m - 1e-9, axis=1)
            ratt = int(np.sum(rowmax < m - 1e-7))
            return m, s, deg, ratt, rowmax
        except Exception:
            return 1e9, set(), np.zeros(n), n, np.zeros(n)

    inc_m, inc_edges, inc_deg, inc_ratt, _ = contact_edges(xb)

    # build pools
    try:
        pool_tri = _build_ternary()
    except Exception:
        pool_tri = None
    try:
        samp = rng.integers(-2, 3, size=(15000, 9)).astype(float)
        nz = np.linalg.norm(samp, axis=1)
        keep = nz > 1e-9
        samp = samp[keep]
        samp = samp / np.linalg.norm(samp, axis=1, keepdims=True)
        # concat for mixed pool
        if pool_tri is not None:
            pool_mix = np.vstack([pool_tri, samp])
        else:
            pool_mix = samp
    except Exception:
        samp = None
        pool_mix = pool_tri

    bases = []
    if pool_tri is not None and time.time() < DISC_END:
        configs = [
            (pool_tri, 1, None),
            (pool_tri, 3, None),
            (pool_mix if pool_mix is not None else pool_tri, 3, None),
            (pool_tri, 5, None),
        ]
        for bi, (pl, topk, st) in enumerate(configs):
            if time.time() > DISC_END - 8.0:
                break
            try:
                s0 = int(rng.integers(pl.shape[0]))
                sel = _greedy_select(pl, n, rng, topk=topk, start=s0)
                # swap improve with short deadline
                sel, Xd = _swap_improve(pl, sel.copy(), min(DISC_END - 1.0, time.time() + 4.0), rounds=2, topW=12)
                bases.append(Xd.copy())
            except Exception:
                continue

    # candidates: bases + rotated versions
    cands = []
    for Xd in bases:
        cands.append(Xd)
        if len(cands) >= 8:
            break
        try:
            A = rng.standard_normal((d, d))
            Q, R = np.linalg.qr(A)
            Xr = Xd @ Q.T
            Xr = _norm(Xr)
            cands.append(Xr)
        except Exception:
            continue
        if len(cands) >= 8:
            break

    # polish discrete candidates 40-60s total
    for ci, C0 in enumerate(cands):
        if time.time() > DISC_END:
            break
        try:
            y = _norm(C0 + 0.002 * rng.standard_normal((n, d)))
        except Exception:
            continue
        try:
            y = _smooth(y, 320, 0.020, 0.003, 0.014, 0.002, mom=0.85)
            y = _smooth(y, 120, 0.004, 0.0012, 0.004, 0.001, mom=0.80)
            vs = _max_cos(y)
        except Exception:
            continue
        # diversity metrics
        try:
            m2, edges2, deg2, ratt2, _ = contact_edges(y, tol=1e-7)
            inter = len(edges2.intersection(inc_edges)) if len(inc_edges) > 0 else 0
            overlap = inter / max(1, len(inc_edges))
            mindeg = int(np.min(deg2)) if deg2.size else 0
            diverse = (mindeg > 0) and (overlap < 0.70)
        except Exception:
            diverse = False
            overlap = 1.0
            mindeg = 0
        # decide LP polish
        allow_lp = False
        if vs < best_val + 0.015:
            allow_lp = True
        if diverse:
            allow_lp = True
        # always polish first two to get foothold
        if ci < 2:
            allow_lp = True
        if not allow_lp:
            # still accept if smooth alone beats best
            if vs < best_val - 1e-12:
                xb = _norm(y); best_val = vs
                inc_m, inc_edges, inc_deg, inc_ratt, _ = contact_edges(xb)
            continue
        budget = min(4.0, DISC_END - time.time() - 0.2)
        if budget < 0.6:
            if vs < best_val - 1e-12:
                xb = _norm(y); best_val = vs
                inc_m, inc_edges, inc_deg, inc_ratt, _ = contact_edges(xb)
            continue
        try:
            y2 = _lp_polish_deadline(y, time.time() + budget)
            v2 = _max_cos(y2)
        except Exception:
            continue
        if v2 < best_val - 1e-12 and np.all(np.isfinite(y2)):
            xb = _norm(y2); best_val = v2
            inc_m, inc_edges, inc_deg, inc_ratt, _ = contact_edges(xb)
        elif diverse and v2 < best_val + 0.005:
            # keep diverse seed for hopping start even if slightly worse? store as alt start
            # do extra smooth+LP short to try to push it
            try:
                if time.time() < DISC_END - 2.0:
                    y3 = _smooth(y2, 200, 0.010, 0.002, 0.010, 0.002, mom=0.85)
                    y3 = _lp_polish_deadline(y3, time.time() + min(2.0, DISC_END - time.time()))
                    v3 = _max_cos(y3)
                    if v3 < best_val - 1e-12:
                        xb = _norm(y3); best_val = v3
                        inc_m, inc_edges, inc_deg, inc_ratt, _ = contact_edges(xb)
            except Exception:
                pass

    # hopping phase from current best (record or discrete breakthrough)
    try:
        G = xb @ xb.T
        np.fill_diagonal(G, -2.0)
        rowmax_tmp = G.max(axis=1)
        m_tmp = float(G.max())
        deg_tmp = np.sum(G > m_tmp - 1e-9, axis=1)
        iu_tmp = np.triu_indices(n, 1)
        c_tmp = G[iu_tmp]
        sel_tmp = c_tmp > m_tmp - 1e-7
        cI = iu_tmp[0][sel_tmp]; cJ = iu_tmp[1][sel_tmp]
        rattlers = np.where(rowmax_tmp < m_tmp - 1e-7)[0]
    except Exception:
        deg_tmp = np.zeros(n); cI = np.array([], dtype=int); cJ = np.array([], dtype=int); rattlers = np.array([], dtype=int)

    trial = 0
    while time.time() < HOP_END:
        if time.time() > HOP_END - 1.0:
            break
        mod = trial % 10
        y = xb.copy()
        try:
            if mod == 0:
                y = y + 0.006 * rng.standard_normal((n, d))
            elif mod == 1:
                y = y + 0.016 * rng.standard_normal((n, d))
            elif mod == 2:
                y = y + 0.003 * rng.standard_normal((n, d))
                k = 10
                idx = rng.choice(n, size=k, replace=False)
                y[idx] += 0.07 * rng.standard_normal((k, d))
            elif mod == 3:
                y = y + 0.004 * rng.standard_normal((n, d))
                if len(cI) > 0:
                    e = rng.integers(0, len(cI))
                    i, j = int(cI[e]), int(cJ[e])
                    diff = y[i] - y[j]
                    diff = diff / (np.linalg.norm(diff) + 1e-12)
                    push = 0.05 + 0.03 * rng.random()
                    y[i] += push * diff
                    y[j] -= push * diff
                else:
                    y += 0.012 * rng.standard_normal((n, d))
            elif mod in (4, 5, 6):
                y = y + 0.004 * rng.standard_normal((n, d))
                if mod == 4 and len(rattlers) > 0:
                    k = min(2, len(rattlers))
                    idx = rng.choice(rattlers, size=k, replace=False)
                else:
                    order = np.argsort(-deg_tmp)
                    k = 2 if mod == 4 else (1 if mod == 5 else 3)
                    idx = order[:k]
                    if k == 3:
                        cand = order[:15]
                        idx = np.concatenate([[order[0]], rng.choice(cand[1:], size=2, replace=False)])
                keep = np.ones(n, dtype=bool); keep[idx] = False
                Xk = y[keep]
                for ii in idx:
                    z = _hole_for(Xk, rng, n_samp=3000)
                    if rng.random() < 0.2:
                        z = z + 0.08 * rng.standard_normal(d)
                        z = z / (np.linalg.norm(z) + 1e-12)
                    y[ii] = z
                    Xk = np.vstack([Xk, z[None, :]])
            elif mod == 7:
                y = y + 0.06 * rng.standard_normal((n, d))
            elif mod == 8:
                y = y + 0.025 * rng.standard_normal((n, d))
                ii = rng.integers(0, n)
                r = rng.standard_normal(d); r /= (np.linalg.norm(r) + 1e-12)
                y[ii] = r if rng.random() < 0.5 else y[ii] + 0.3 * r
            else:
                sig = 0.18 if (trial % 30 == 9) else 0.09
                y = y + sig * rng.standard_normal((n, d))
            y = _norm(y)
        except Exception:
            trial += 1
            continue
        try:
            if mod in (4, 5, 6):
                y = _smooth(y, 260, 0.018, 0.004, 0.014, 0.0025, mom=0.85)
                y = _smooth(y, 120, 0.004, 0.0015, 0.004, 0.0012, mom=0.80)
            elif mod in (3, 9):
                y = _smooth(y, 260, 0.015, 0.003, 0.016, 0.002, mom=0.86)
            else:
                y = _smooth(y, 200, 0.016, 0.004, 0.012, 0.002, mom=0.85)
        except Exception:
            pass
        try:
            vs = _max_cos(y)
        except Exception:
            trial += 1
            continue
        allow = best_val + (0.012 if mod in (7, 9) else 0.008)
        if vs > allow and rng.random() > 0.05:
            trial += 1
            if vs < best_val - 1e-12:
                xb = _norm(y); best_val = vs
                try:
                    G2 = xb @ xb.T; np.fill_diagonal(G2, -2.0)
                    m2 = float(G2.max()); deg_tmp = np.sum(G2 > m2 - 1e-9, axis=1)
                    rowmax2 = G2.max(axis=1); rattlers = np.where(rowmax2 < m2 - 1e-7)[0]
                    c2 = G2[np.triu_indices(n, 1)]; sel2 = c2 > m2 - 1e-7
                    cI = np.triu_indices(n, 1)[0][sel2]; cJ = np.triu_indices(n, 1)[1][sel2]
                except Exception:
                    pass
            continue
        budget = min(1.4, HOP_END - time.time() - 0.05)
        if budget < 0.35:
            trial += 1
            continue
        try:
            y = _lp_polish_deadline(y, time.time() + budget)
            v = _max_cos(y)
        except Exception:
            trial += 1
            continue
        if v < best_val - 1e-12 and np.all(np.isfinite(y)):
            xb = _norm(y); best_val = v
            try:
                G2 = xb @ xb.T; np.fill_diagonal(G2, -2.0)
                m2 = float(G2.max()); deg_tmp = np.sum(G2 > m2 - 1e-9, axis=1)
                rowmax2 = G2.max(axis=1); rattlers = np.where(rowmax2 < m2 - 1e-7)[0]
                c2 = G2[np.triu_indices(n, 1)]; sel2 = c2 > m2 - 1e-7
                cI = np.triu_indices(n, 1)[0][sel2]; cJ = np.triu_indices(n, 1)[1][sel2]
            except Exception:
                pass
        trial += 1

    try:
        remain = HARD_END - time.time() - 0.5
        if remain > 1.0:
            xb = _lp_polish_deadline(xb, time.time() + remain)
    except Exception:
        pass
    xb = _norm(xb)
    return xb
