import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _norm(x):
    x = np.asarray(x, dtype=np.float64)
    x /= np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
    return x

def _contact_mask(x, tol=2e-7):
    n = x.shape[0]
    iu0, iu1 = np.triu_indices(n, 1)
    g = (x @ x.T)[iu0, iu1]
    m = float(g.max())
    return ((g > m - tol), m, iu0, iu1, g)

def _jaccard(m1, m2):
    inter = np.count_nonzero(m1 & m2)
    union = np.count_nonzero(m1 | m2)
    if union == 0:
        return 0.0
    return 1.0 - inter / union

def _gd(x0, steps, T0, T1, lr0, mu, rng, noise0=0.0, active=None, deadline=None):
    x = x0.copy()
    n, d = x.shape
    if active is None:
        act = np.ones(n, dtype=bool)
    else:
        act = np.asarray(active, dtype=bool)
    v = np.zeros_like(x)
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    bv = float(G.max())
    bx = x.copy()
    if T0 <= 0 or T1 <= 0:
        T0 = max(T0, 1e-12); T1 = max(T1, 1e-12)
    lratio = np.log(T1 / T0)
    for k in range(steps):
        if deadline is not None and (k & 127) == 0 and time.time() > deadline:
            break
        frac = k / max(1, steps - 1)
        T = T0 * np.exp(lratio * frac)
        if T < 5e-5:
            T = 5e-5
        lr = lr0 * (1.0 - 0.55 * frac)
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        E = np.exp((G - m) / T)
        np.fill_diagonal(E, 0.0)
        S = float(E.sum() * 0.5)
        if not np.isfinite(S) or S < 1e-300:
            S = 1e-300
        grad = (E @ x) / S
        grad -= (grad * x).sum(axis=1, keepdims=True) * x
        if int(act.sum()) != n:
            grad[~act] = 0.0
            v[~act] = 0.0
        v = mu * v + (1.0 - mu) * grad
        x[act] = x[act] - lr * v[act]
        if noise0 > 0:
            amp = noise0 * (1.0 - frac)
            if amp > 0:
                nz = rng.randn(n, d) * amp
                nz -= (nz * x).sum(axis=1, keepdims=True) * x
                nz[~act] = 0.0
                x[act] = x[act] + nz[act]
        x[act] /= np.maximum(np.linalg.norm(x[act], axis=1, keepdims=True), 1e-12)
        if m < bv:
            bv = m
            bx = x.copy()
    return bx, bv

def _lp_polish(x, deadline, r0=0.004, cap=1500):
    x = _norm(x)
    n, d = x.shape
    try:
        best = _max_cos(x)
    except Exception:
        return x, 1.0
    radius = r0
    iu0, iu1 = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        try:
            G = x @ x.T
            c = G[iu0, iu1]
            m = float(c.max())
            sel = c > m - max(20.0 * radius, 1e-9)
            if int(np.count_nonzero(sel)) == 0:
                sel[int(np.argmax(c))] = True
            if int(np.count_nonzero(sel)) > cap:
                idx = np.argsort(-c)[:cap]
                s2 = np.zeros_like(sel, dtype=bool)
                s2[idx] = True
                sel = s2
            I = iu0[sel]; J = iu1[sel]
            csel = c[sel]
            p = len(I)
            nv = n * d + 1
            rows = np.repeat(np.arange(p), 2 * d + 1)
            cols = np.empty((p, 2 * d + 1), dtype=int)
            ar = np.arange(d)
            cols[:, :d] = I[:, None] * d + ar
            cols[:, d:2 * d] = J[:, None] * d + ar
            cols[:, 2 * d] = n * d
            cols = cols.ravel()
            vals = np.empty((p, 2 * d + 1), dtype=float)
            vals[:, :d] = x[J]
            vals[:, d:2 * d] = x[I]
            vals[:, 2 * d] = -1.0
            vals = vals.ravel()
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv); obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
            if res.status != 0:
                radius *= 0.5
                continue
            y = x + res.x[:-1].reshape(n, d)
            y = _norm(y)
            vv = _max_cos(y)
            if vv < best:
                x, best = y, vv
                radius = min(radius * 1.5, 0.05)
            else:
                radius *= 0.5
        except Exception:
            radius *= 0.5
            continue
    return x, best

def _stress_delete_set(x, k, rng):
    n = x.shape[0]
    # degrees from tight contacts
    try:
        mask, m, iu0, iu1, g = _contact_mask(x, tol=1e-9)
        # if too few, widen
        if int(np.count_nonzero(mask)) < 200:
            mask, m, iu0, iu1, g = _contact_mask(x, tol=2e-7)
        I = iu0[mask]; J = iu1[mask]
        deg = np.zeros(n, dtype=int)
        for a, b in zip(I, J):
            deg[a] += 1; deg[b] += 1
    except Exception:
        deg = np.zeros(n, dtype=int)
    # try LP dual multipliers for contact stress
    point_stress = deg.astype(float)
    contact_dual = None
    try:
        # build small LP and read marginals
        iu0f, iu1f = np.triu_indices(n, 1)
        G = x @ x.T
        c = G[iu0f, iu1f]
        m = float(c.max())
        sel = c > m - 2e-7
        if int(np.count_nonzero(sel)) > 1500:
            idx = np.argsort(-c)[:1500]
            s2 = np.zeros_like(sel, dtype=bool); s2[idx] = True; sel = s2
        I2 = iu0f[sel]; J2 = iu1f[sel]; csel = c[sel]
        p = len(I2)
        if p > 0 and p < 1600:
            nv = n * 4 + 1
            rows = np.repeat(np.arange(p), 2 * 4 + 1)
            cols = np.empty((p, 9), dtype=int)
            ar = np.arange(4)
            cols[:, :4] = I2[:, None] * 4 + ar
            cols[:, 4:8] = J2[:, None] * 4 + ar
            cols[:, 8] = n * 4
            cols = cols.ravel()
            vals = np.empty((p, 9), dtype=float)
            vals[:, :4] = x[J2]
            vals[:, 4:8] = x[I2]
            vals[:, 8] = -1.0
            vals = vals.ravel()
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), 4), np.arange(n * 4))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv); obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-0.002, 0.002)] * (n * 4) + [(None, None)], method="highs")
            marg = None
            try:
                if hasattr(res, "ineqlin") and hasattr(res.ineqlin, "marginals"):
                    marg = np.asarray(res.ineqlin.marginals, dtype=float)
                elif isinstance(res.get("ineqlin", None), dict):
                    marg = np.asarray(res["ineqlin"].get("marginals", None), dtype=float)
            except Exception:
                marg = None
            if marg is not None and marg.shape[0] == p and np.all(np.isfinite(marg)):
                dual = np.abs(marg)
                if dual.max() > 0:
                    dual /= dual.max()
                    ps = np.zeros(n)
                    for a, b, w in zip(I2, J2, dual):
                        ps[a] += w; ps[b] += w
                    # combine with degree
                    point_stress = ps + 0.15 * deg.astype(float)
                    contact_dual = (I2, J2, dual)
    except Exception:
        pass
    # pick top-k stressed; bias to 8-9 contact hubs first
    order = np.argsort(-point_stress, kind="stable")
    # small random shuffle among ties
    # add tiny noise to break ties deterministically via rng
    jitter = rng.rand(n) * 1e-6
    order = np.argsort(-(point_stress + jitter), kind="stable")
    doomed = list(order[:k])
    return sorted([int(v) for v in doomed]), deg

def _refine_hole(y0, Xf, iters=140):
    y = y0 / max(float(np.linalg.norm(y0)), 1e-12)
    dots = Xf @ y
    vy = float(dots.max())
    by = y.copy()
    step = 0.06
    for it in range(iters):
        dots = Xf @ y
        j = int(np.argmax(dots))
        dj = float(dots[j])
        g = Xf[j] - dj * y
        ng = float(np.linalg.norm(g))
        if ng < 1e-13:
            break
        g /= ng
        improved = False
        for s in (step, step * 0.35):
            yn = y - s * g
            nyn = float(np.linalg.norm(yn))
            if nyn < 1e-12:
                continue
            yn /= nyn
            v = float((Xf @ yn).max())
            if v < vy:
                y = yn; vy = v; by = yn.copy()
                improved = True
                break
        if not improved:
            step *= 0.82
            if step < 1e-5:
                break
    return by, vy

def solve():
    t0 = time.time()
    DEADLINE = t0 + 172.0
    SCREEN_END = t0 + 122.0
    n = 163
    d = 4
    rng0 = np.random.RandomState(20261102)
    best_x = None
    best_val = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=np.float64)
            if a.shape != (n, d) or not np.all(np.isfinite(a)):
                continue
            a = _norm(a)
            v = _max_cos(a)
            if v < best_val:
                best_val = v
                best_x = a.copy()
        except Exception:
            continue
    if best_x is None:
        best_x = _norm(rng0.randn(n, d))
        best_val = _max_cos(best_x)
    base_best = best_x.copy()
    base_val = float(best_val)
    try:
        inc_mask, _, _, _, _ = _contact_mask(best_x, tol=2e-7)
        inc_mask = inc_mask.copy()
    except Exception:
        inc_mask = None
    elite = [(base_val, base_best.copy())]
    ks = [10, 14, 6, 18]
    rep = 0
    # quick initial LP on incumbent if time allows later; focus on surgery
    while time.time() < SCREEN_END:
        k = ks[rep % len(ks)]
        rng = np.random.RandomState(9000 + rep * 7919)
        # stress surgery on current best (occasionally on elite variant)
        src = best_x
        if rep % 5 == 4 and len(elite) > 1:
            src = elite[rng.randint(len(elite))][1]
        try:
            doomed, deg = _stress_delete_set(src, k, rng)
        except Exception:
            doomed = sorted(rng.choice(n, k, replace=False).tolist())
        mask = np.ones(n, dtype=bool)
        mask[np.array(doomed, dtype=int)] = False
        R = src[mask].copy()
        nr = R.shape[0]
        # re-polish remainder with annealed softmax (under-press)
        try:
            R2, _ = _gd(R, steps=900, T0=0.004, T1=0.00025, lr0=0.5, mu=0.82, rng=rng,
                        noise0=0.004, deadline=SCREEN_END)
            # short LP to settle remainder
            R2, vr = _lp_polish(R2, deadline=min(time.time() + 1.6, SCREEN_END), r0=0.004, cap=1200)
        except Exception:
            R2 = R
        # sequential greedy reinsertion; restart order via different pool when stalls
        C = R2.copy()
        # random global rotation of remainder to change basin slightly every other rep
        if rep % 2 == 1:
            try:
                A = rng.randn(4, 4)
                Q, RR = np.linalg.qr(A)
                Q = Q * np.sign(np.diag(RR) + 1e-12)
                if np.linalg.det(Q) < 0:
                    Q[:, 0] = -Q[:, 0]
                C = (C @ Q.T)
                C = _norm(C)
            except Exception:
                pass
        inserted = []
        stall = 0
        ok = True
        for s in range(k):
            if time.time() > SCREEN_END - 1.0:
                ok = False
                break
            try:
                P = rng.randn(20000, 4)
                P /= np.maximum(np.linalg.norm(P, axis=1, keepdims=True), 1e-12)
                D = P @ C.T
                cur = D.max(axis=1)
                # best 20 holes
                idx20 = np.argsort(cur)[:20]
                best_y = None
                best_v = 1e9
                for jj in idx20:
                    y0 = P[jj]
                    yr, vr_ = _refine_hole(y0, C, iters=140)
                    if vr_ < best_v:
                        best_v = vr_
                        best_y = yr
                if best_y is None:
                    best_y = P[idx20[0]] / np.linalg.norm(P[idx20[0]])
                    best_v = float((C @ best_y).max())
                # stall detection: hole worse than incumbent target
                if best_v > base_val - 1e-6:
                    stall += 1
                C = np.vstack([C, best_y[None, :]])
                # local relax: inserted + neighbours cos>0.75
                try:
                    new_idx = C.shape[0] - 1
                    dots = (C[:new_idx] @ C[new_idx])
                    nbr = np.where(dots > 0.75)[0].tolist()
                    act = np.zeros(C.shape[0], dtype=bool)
                    act[new_idx] = True
                    if len(nbr) > 0:
                        act[np.array(nbr, dtype=int)] = True
                    # cap active size
                    if int(act.sum()) > 40:
                        # keep closest 39 + new
                        order = np.argsort(-dots)
                        keep = order[:39]
                        act[:] = False
                        act[new_idx] = True
                        act[keep] = True
                    C, _ = _gd(C, steps=260, T0=0.0012, T1=0.00015, lr0=0.35, mu=0.8, rng=rng,
                               noise0=0.001, active=act, deadline=SCREEN_END)
                except Exception:
                    pass
            except Exception:
                ok = False
                break
        if not ok or C.shape[0] != n:
            rep += 1
            continue
        C = _norm(C)
        try:
            v0 = _max_cos(C)
        except Exception:
            rep += 1
            continue
        # light anneal to settle without collapsing topology
        try:
            C2, v2 = _gd(C, steps=350, T0=0.0015, T1=0.0002, lr0=0.4, mu=0.85, rng=rng,
                         noise0=0.0, deadline=SCREEN_END)
        except Exception:
            C2, v2 = C, v0
        # Jaccard gating before expensive full LP polish
        do_full = True
        try:
            m2, _, _, _, _ = _contact_mask(C2, tol=2e-7)
            if inc_mask is not None and m2.shape == inc_mask.shape:
                jd = _jaccard(m2, inc_mask)
                # polish only if topology differs measurably OR already better
                if jd < 0.1 and v2 >= best_val - 2e-7:
                    do_full = False
            # keep elite archive of distinct reinsertions
            if len(elite) < 6:
                elite.append((v2, C2.copy()))
            else:
                # replace worst if distinct and better
                elite.sort(key=lambda e: e[0])
                if v2 < elite[-1][0]:
                    # check distinctness
                    distinct = True
                    for _, ex in elite:
                        try:
                            me, _, _, _, _ = _contact_mask(ex, tol=2e-7)
                            if _jaccard(m2, me) < 0.05:
                                distinct = False
                                break
                        except Exception:
                            pass
                    if distinct:
                        elite[-1] = (v2, C2.copy())
        except Exception:
            do_full = True
        if do_full:
            try:
                # frozen-subset then full: first relax worst points, then LP
                C3 = C2
                v3 = v2
                # short full anneal
                C3, v3 = _gd(C3, steps=500, T0=0.002, T1=0.00012, lr0=0.45, mu=0.85, rng=rng,
                             noise0=0.0, deadline=SCREEN_END)
                C4, v4 = _lp_polish(C3, deadline=min(time.time() + 3.5, SCREEN_END + 6), r0=0.004, cap=1500)
                if v4 < best_val:
                    best_val = float(v4)
                    best_x = _norm(C4)
                    try:
                        inc_mask, _, _, _, _ = _contact_mask(best_x, tol=2e-7)
                        inc_mask = inc_mask.copy()
                    except Exception:
                        pass
                    elite.append((best_val, best_x.copy()))
                    elite.sort(key=lambda e: e[0])
                    elite = elite[:6]
                elif v4 < elite[-1][0] + 1e-9:
                    elite.append((v4, _norm(C4)))
                    elite.sort(key=lambda e: e[0])
                    elite = elite[:6]
            except Exception:
                # accept light result if better
                if v2 < best_val:
                    best_val = float(v2)
                    best_x = _norm(C2)
                    try:
                        inc_mask, _, _, _, _ = _contact_mask(best_x, tol=2e-7)
                        inc_mask = inc_mask.copy()
                    except Exception:
                        pass
        else:
            if v2 < best_val:
                best_val = float(v2)
                best_x = _norm(C2)
                try:
                    inc_mask, _, _, _, _ = _contact_mask(best_x, tol=2e-7)
                    inc_mask = inc_mask.copy()
                except Exception:
                    pass
        rep += 1
    # final full polish of best with remaining time
    try:
        POLISH_END = DEADLINE - 1.0
        bx, bv = _gd(best_x, steps=700, T0=0.0018, T1=0.0001, lr0=0.4, mu=0.86, rng=rng0,
                     noise0=0.0, deadline=POLISH_END)
        if _max_cos(bx) < best_val:
            best_x, best_val = _norm(bx), _max_cos(bx)
        best_x, best_val2 = _lp_polish(best_x, deadline=POLISH_END, r0=0.003, cap=1500)
        if best_val2 < best_val:
            best_val = best_val2
            best_x = _norm(best_x)
    except Exception:
        pass
    best_x = _norm(best_x)
    return best_x
