import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _contact_mask(x, tol=2e-7):
    n = x.shape[0]
    iu0, iu1 = np.triu_indices(n, 1)
    g = (x @ x.T)[iu0, iu1]
    m = float(g.max())
    return (g > m - tol), m

def _jaccard(m1, m2):
    inter = np.count_nonzero(m1 & m2)
    union = np.count_nonzero(m1 | m2)
    if union == 0:
        return 0.0
    return 1.0 - inter / union

def _lp_polish(x, deadline, r0=0.004, cap=1200):
    x = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
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
            if np.count_nonzero(sel) == 0:
                sel[int(np.argmax(c))] = True
            cnt = int(np.count_nonzero(sel))
            if cnt > cap:
                idx = np.argsort(-c)[:cap]
                sel2 = np.zeros_like(sel, dtype=bool)
                sel2[idx] = True
                sel = sel2
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
            y /= np.maximum(np.linalg.norm(y, axis=1, keepdims=True), 1e-12)
            v = _max_cos(y)
            if v < best:
                x, best = y, v
                radius = min(radius * 1.5, 0.05)
            else:
                radius *= 0.5
        except Exception:
            radius *= 0.5
            continue
    return x, best

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
    best_val = float(G.max())
    best = x.copy()
    if T0 <= 0:
        T0 = 1e-12
    if T1 <= 0:
        T1 = 1e-12
    lratio = np.log(T1 / T0)
    for k in range(steps):
        if deadline is not None and (k & 127) == 0 and time.time() > deadline:
            break
        frac = k / max(1, steps - 1)
        T = T0 * np.exp(lratio * frac)
        if T < 1e-12:
            T = 1e-12
        lr = lr0 * (1.0 - 0.55 * frac)
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        E = np.exp((G - m) / T)
        np.fill_diagonal(E, 0.0)
        S = float(E.sum() * 0.5)
        if S < 1e-300 or not np.isfinite(S):
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
        if m < best_val:
            best_val = m
            best = x.copy()
    return best, best_val

def _refine_hole(y0, Xf, iters=140):
    y = y0 / max(float(np.linalg.norm(y0)), 1e-12)
    dots = Xf @ y
    vy = float(dots.max())
    step = 0.05
    for _ in range(iters):
        dots = Xf @ y
        j = int(np.argmax(dots))
        dj = float(dots[j])
        g = Xf[j] - dj * y
        ng = float(np.linalg.norm(g))
        if ng < 1e-13:
            break
        g /= ng
        improved = False
        for s in (step, step * 0.3):
            yn = y - s * g
            yn /= max(float(np.linalg.norm(yn)), 1e-12)
            vn = float((Xf @ yn).max())
            if vn < vy:
                y, vy = yn, vn
                improved = True
                break
        if not improved:
            step *= 0.92
            if step < 1e-5:
                break
        else:
            step = min(step * 1.05, 0.05)
    return y, vy

def _small_rot(rng, eps):
    M = np.eye(4) + rng.randn(4, 4) * eps
    Q, R = np.linalg.qr(M)
    d = np.sign(np.diag(R) + 1e-12)
    Q = Q * d
    if np.linalg.det(Q) < 0:
        Q[:, 0] = -Q[:, 0]
    return Q

def solve():
    t0 = time.time()
    DEADLINE = t0 + 172.0
    SEARCH_END = t0 + 128.0
    rng = np.random.RandomState(987654321)
    n = 163
    try:
        bx = np.load("best.npy")
    except Exception:
        try:
            bx = np.load("record.npy")
        except Exception:
            bx = rng.randn(n, 4)
    bx = np.asarray(bx, dtype=float)
    if bx.shape != (n, 4) or not np.all(np.isfinite(bx)):
        bx = rng.randn(n, 4)
    bx = bx / np.maximum(np.linalg.norm(bx, axis=1, keepdims=True), 1e-12)
    best_x = bx.copy()
    best_val = _max_cos(best_x)
    try:
        best_mask, _ = _contact_mask(best_x)
    except Exception:
        best_mask = np.ones(13286, dtype=bool)
    cur_x = best_x.copy()
    cur_val = best_val
    try:
        cur_mask = best_mask.copy()
    except Exception:
        cur_mask, _ = _contact_mask(cur_x)
    elite = [(best_val, best_x.copy())]
    basin_T = 0.0006
    trial = 0
    ks = [6, 9, 12, 16, 8, 13]
    while time.time() < SEARCH_END:
        if time.time() > DEADLINE - 44:
            break
        trial += 1
        op = trial % 5  # 0 kick,1 surgery,2 cap,3 kick-big,4 surgery-var
        cand = None
        cand_v = 1e9
        try:
            if op == 0 or op == 3:
                # Gaussian kick + anneal
                amp = 0.012 if op == 0 else 0.035
                if trial % 17 == 0:
                    amp = 0.07
                y = cur_x + rng.randn(n, 4) * amp
                y /= np.maximum(np.linalg.norm(y, axis=1, keepdims=True), 1e-12)
                T0 = 0.010 if (trial % 2 == 0) else 0.006
                steps = 1600 if op == 0 else 2000
                cand, cand_v = _gd(y, steps=steps, T0=T0, T1=0.00035, lr0=0.032, mu=0.87,
                                   rng=rng, noise0=0.001, deadline=SEARCH_END)
            elif op == 1 or op == 4:
                k = ks[(trial // 5) % len(ks)]
                if op == 4 and (trial % 7 == 0):
                    k = int(rng.randint(18, 26))
                # stressed order of cur
                Gf = cur_x @ cur_x.T
                np.fill_diagonal(Gf, -2.0)
                mf = float(Gf.max())
                deg = (Gf > mf - 2e-7).sum(axis=1)
                order = np.argsort(deg + 1e-6 * rng.rand(n))
                del_idx = order[-k:].copy()
                if (trial % 3) == 0 and k >= 6:
                    mid = order[n // 2:n - k]
                    if len(mid) >= 2:
                        pk = rng.choice(len(mid), 2, replace=False)
                        del_idx[:2] = mid[pk]
                keep = np.ones(n, dtype=bool)
                keep[del_idx] = False
                x_rem = cur_x[keep].copy()
                # under-press remainder
                x_rem_p, _ = _gd(x_rem, steps=900, T0=0.007, T1=0.0004, lr0=0.032, mu=0.86,
                                 rng=rng, noise0=0.0008, deadline=SEARCH_END)
                if time.time() < SEARCH_END - 2:
                    # cheap short LP only if time allows (skip often)
                    if (trial % 4) == 0:
                        try:
                            x_rem_p, _ = _lp_polish(x_rem_p, deadline=min(SEARCH_END, time.time() + 0.8), r0=0.004)
                        except Exception:
                            pass
                cur_fixed = x_rem_p.copy()
                ok = True
                for ins in range(k):
                    if time.time() > SEARCH_END - 1.0:
                        ok = False
                        break
                    Y = rng.randn(16000, 4)
                    Y /= np.maximum(np.linalg.norm(Y, axis=1, keepdims=True), 1e-12)
                    D = Y @ cur_fixed.T
                    vals = D.max(axis=1)
                    cidx = np.argpartition(vals, 16)[:16] if len(vals) > 16 else np.arange(len(vals))
                    by = None
                    bvy = 1e9
                    for ci in cidx:
                        yr, vr = _refine_hole(Y[ci], cur_fixed, iters=130)
                        if vr < bvy:
                            bvy = vr
                            by = yr
                    if by is None:
                        ok = False
                        break
                    cur_fixed = np.vstack([cur_fixed, by.reshape(1, 4)])
                    # local relax inserted + neighbours
                    try:
                        dots_ins = cur_fixed[:-1] @ by
                        nbr = np.where(dots_ins > 0.75)[0]
                        act = np.zeros(cur_fixed.shape[0], dtype=bool)
                        act[cur_fixed.shape[0] - 1] = True
                        act[nbr] = True
                        if int(act.sum()) < 3:
                            worst = np.argsort(-dots_ins)[:3]
                            act[worst] = True
                        cur_fixed, _ = _gd(cur_fixed, steps=220, T0=0.0035, T1=0.0004, lr0=0.02, mu=0.85,
                                           rng=rng, noise0=0.0004, active=act, deadline=SEARCH_END)
                    except Exception:
                        pass
                if not ok:
                    continue
                cur_fixed /= np.maximum(np.linalg.norm(cur_fixed, axis=1, keepdims=True), 1e-12)
                T0g = 0.009 if (trial % 2) else 0.013
                cand, cand_v = _gd(cur_fixed, steps=1800, T0=T0g, T1=0.00035, lr0=0.033, mu=0.87,
                                   rng=rng, noise0=0.001, deadline=SEARCH_END)
            else:  # op == 2 cap rotation + anneal
                u = rng.randn(4)
                u /= max(float(np.linalg.norm(u)), 1e-12)
                dots = cur_x @ u
                # choose cap size to get 8-28 points
                thr = float(np.sort(dots)[n - int(rng.randint(8, 26))])
                cap_idx = np.where(dots >= thr)[0]
                if len(cap_idx) < 4:
                    cap_idx = np.argsort(-dots)[:10]
                Q = _small_rot(rng, 0.03 + rng.rand() * 0.05)
                # rotate around mean? just apply Q to cap points
                y = cur_x.copy()
                # centre rotation: rotate cap points in place by Q
                y[cap_idx] = y[cap_idx] @ Q.T
                y /= np.maximum(np.linalg.norm(y, axis=1, keepdims=True), 1e-12)
                # anneal cap + neighbours harder, rest fixed first then all
                act = np.zeros(n, dtype=bool)
                act[cap_idx] = True
                # add nearest neighbours of cap (cos>0.7 to any cap point)
                try:
                    Gm = y @ y[cap_idx].T
                    nbr2 = np.where((Gm > 0.70).any(axis=1))[0]
                    act[nbr2] = True
                except Exception:
                    pass
                y2, v2 = _gd(y, steps=900, T0=0.006, T1=0.0004, lr0=0.030, mu=0.86,
                             rng=rng, noise0=0.001, active=act, deadline=SEARCH_END)
                cand, cand_v = _gd(y2, steps=1200, T0=0.007, T1=0.0003, lr0=0.030, mu=0.87,
                                   rng=rng, noise0=0.0007, deadline=SEARCH_END)
        except Exception:
            continue
        # evaluate topology distance
        try:
            cmask, _ = _contact_mask(cand)
            jd_best = _jaccard(cmask, best_mask)
            jd_cur = _jaccard(cmask, cur_mask)
        except Exception:
            jd_best = 1.0
            jd_cur = 1.0
        # gated LP polish
        do_lp = False
        if cand_v < best_val:
            do_lp = True
        elif cand_v < best_val + 0.0025 and jd_best > 0.08 and cand_v < cur_val + 0.001:
            # only polish distinct promising basins, limit frequency
            if (trial % 2 == 0) or cand_v < best_val + 0.0005:
                do_lp = True
        if do_lp and time.time() < DEADLINE - 10:
            try:
                lp_x, lp_v = _lp_polish(cand, deadline=min(DEADLINE - 2.0, time.time() + 3.5), r0=0.003)
                if lp_v < cand_v:
                    cand, cand_v = lp_x, lp_v
                    try:
                        cmask, _ = _contact_mask(cand)
                        jd_best = _jaccard(cmask, best_mask)
                    except Exception:
                        pass
            except Exception:
                pass
        # elite archive
        if cand_v < best_val + 0.004:
            try:
                distinct = True
                for ev, ex in elite:
                    try:
                        em, _ = _contact_mask(ex)
                        if _jaccard(cmask, em) < 0.04 and abs(cand_v - ev) < 3e-7:
                            distinct = False
                            break
                    except Exception:
                        continue
                if distinct:
                    elite.append((cand_v, cand.copy()))
                    elite.sort(key=lambda z: z[0])
                    elite = elite[:7]
            except Exception:
                pass
        # update best
        if cand_v < best_val - 1e-12:
            best_val = cand_v
            best_x = cand.copy()
            best_mask = cmask.copy() if 'cmask' in locals() else best_mask
            cur_x = cand.copy()
            cur_val = cand_v
            try:
                cur_mask = cmask.copy()
            except Exception:
                pass
            basin_T = max(basin_T * 0.97, 0.00015)
        else:
            # Metropolis move for cur to wander
            if cand_v < cur_val or rng.rand() < np.exp((cur_val - cand_v) / basin_T):
                cur_x = cand.copy()
                cur_val = cand_v
                try:
                    cur_mask = cmask.copy()
                except Exception:
                    pass
            # occasionally jump cur to elite or best to refocus
            if (trial % 25) == 0 and len(elite) > 1:
                # restart from random elite with kick
                ei = int(rng.randint(len(elite)))
                cur_x = elite[ei][1].copy()
                cur_val = elite[ei][0]
                try:
                    cur_mask, _ = _contact_mask(cur_x)
                except Exception:
                    pass
            if (trial % 40) == 0:
                cur_x = best_x.copy()
                cur_val = best_val
                try:
                    cur_mask = best_mask.copy()
                except Exception:
                    pass
    # final intensive polish of best and top elites
    try:
        # polish best
        if time.time() < DEADLINE - 1.0:
            best_x, best_val = _lp_polish(best_x, deadline=DEADLINE - 0.6, r0=0.0025)
    except Exception:
        pass
    try:
        # check elites, keep absolute best
        for ev, ex in list(elite):
            if time.time() > DEADLINE - 0.5:
                break
            if ev < best_val - 1e-12:
                best_val = ev
                best_x = ex.copy()
        # one last short anneal+LP if time remains
        if time.time() < DEADLINE - 6:
            try:
                yb, yv = _gd(best_x, steps=2500, T0=0.004, T1=0.0002, lr0=0.022, mu=0.88,
                             rng=rng, noise0=0.0, deadline=DEADLINE - 3.0)
                if yv < best_val:
                    best_x, best_val = yb, yv
                    best_x, best_val = _lp_polish(best_x, deadline=DEADLINE - 0.4, r0=0.002)
            except Exception:
                pass
    except Exception:
        pass
    best_x = best_x / np.maximum(np.linalg.norm(best_x, axis=1, keepdims=True), 1e-12)
    return best_x
