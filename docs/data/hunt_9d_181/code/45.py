import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def _lp_polish_with_duals(X, seconds, init_radius=1e-3, cap=2500):
    t0 = time.time()
    x = X / np.linalg.norm(X, axis=1, keepdims=True)
    n, d = x.shape
    try:
        best = _max_cos(x)
    except Exception:
        return x, np.inf, None, None, None
    radius = init_radius
    iu = np.triu_indices(n, 1)
    lastI = None; lastJ = None; lastW = None
    while time.time() - t0 < seconds and radius > 1e-14:
        if time.time() - t0 + 0.25 > seconds and radius < 8e-4:
            break
        Gfull = x @ x.T
        c_all = Gfull[iu]
        m = float(c_all.max())
        thresh = max(20.0 * radius, 1e-9)
        sel = c_all > m - thresh
        cnt = int(sel.sum())
        if cnt == 0:
            radius *= 0.5
            continue
        if cnt > cap:
            kth = np.partition(c_all, -cap)[-cap]
            sel = c_all >= kth
            # if still too many, subsample tightest
            cnt2 = int(sel.sum())
            if cnt2 > cap + 500:
                idx = np.where(sel)[0]
                order = np.argsort(-c_all[idx])[:cap]
                mask = np.zeros_like(sel)
                mask[idx[order]] = True
                sel = mask
        I = iu[0][sel]; J = iu[1][sel]
        csel = c_all[sel]
        p = len(I)
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv); obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        # extract duals
        try:
            duals = None
            if hasattr(res, "ineqlin") and res.ineqlin is not None:
                ine = res.ineqlin
                if hasattr(ine, "marginals"):
                    duals = np.asarray(ine.marginals, dtype=float)
                elif isinstance(ine, dict) and "marginals" in ine:
                    duals = np.asarray(ine["marginals"], dtype=float)
            if duals is not None and duals.size == p:
                lastI = I.copy(); lastJ = J.copy(); lastW = np.abs(duals)
            else:
                lastI = I.copy(); lastJ = J.copy(); lastW = np.ones(p)
        except Exception:
            lastI = I.copy(); lastJ = J.copy(); lastW = np.ones(p)
        try:
            y = x + res.x[:-1].reshape(n, d)
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            v = _max_cos(y)
        except Exception:
            radius *= 0.5
            continue
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x, best, lastI, lastJ, lastW

def _smooth_exp(X, steps, T0, T1, lr, t_deadline=None, t0=None):
    n, d = X.shape
    Y = X.copy()
    lT0 = np.log(T0); lT1 = np.log(T1)
    for s in range(steps):
        if t_deadline is not None and t0 is not None:
            if time.time() - t0 > t_deadline:
                break
        frac = s / max(1, steps - 1)
        T = float(np.exp(lT0 + (lT1 - lT0) * frac))
        G = Y @ Y.T
        np.fill_diagonal(G, -1.0)
        m = float(np.max(G))
        W = (G - m) / T
        np.clip(W, -50.0, 0.0, out=W)
        np.exp(W, out=W)
        np.fill_diagonal(W, 0.0)
        grad = W @ Y
        dot = np.sum(grad * Y, axis=1, keepdims=True)
        grad_t = grad - dot * Y
        mn = float(np.mean(np.linalg.norm(grad_t, axis=1))) + 1e-12
        step = lr / mn
        Y = Y - step * grad_t
        Y /= np.linalg.norm(Y, axis=1, keepdims=True)
    return Y

def _refine_vec(v, M, T0=0.03, T1=0.008, steps=220, lr=0.15):
    v = v / np.linalg.norm(v)
    for s in range(steps):
        frac = s / max(1, steps - 1)
        T = T0 + (T1 - T0) * frac
        dots = M @ v
        m = float(dots.max())
        w = np.exp(np.clip((dots - m) / T, -50, 0))
        sw = float(w.sum()) + 1e-300
        w /= sw
        g = w @ M
        g = g - float(np.dot(g, v)) * v
        v = v - lr * g
        nv = float(np.linalg.norm(v))
        if not np.isfinite(nv) or nv == 0:
            break
        v /= nv
    return v / np.linalg.norm(v)

def _insert_one(M, rng, t0, deadline):
    d = M.shape[1]
    # 50k hole samples + 200 random starts
    try:
        big = rng.standard_normal((50000, d))
        big /= np.linalg.norm(big, axis=1, keepdims=True)
        dots_big = big @ M.T
        mx_big = dots_big.max(axis=1)
        hole_idx = np.argsort(mx_big)[:5]
        holes = big[hole_idx]
    except Exception:
        holes = None
    starts = rng.standard_normal((200, d))
    starts /= np.linalg.norm(starts, axis=1, keepdims=True)
    if holes is not None:
        cand = np.vstack([starts, holes])
    else:
        cand = starts
    # score
    try:
        sc = (cand @ M.T).max(axis=1)
    except Exception:
        sc = np.zeros(len(cand))
    order = np.argsort(sc)[:8]
    best_v = cand[int(order[0])].copy()
    try:
        best_s = float((M @ best_v).max())
    except Exception:
        best_s = 1.0
    for k in order:
        if time.time() - t0 > deadline:
            break
        v0 = cand[int(k)].copy()
        v = _refine_vec(v0, M)
        s = float((M @ v).max())
        if s < best_s:
            best_s = s
            best_v = v
    return best_v / np.linalg.norm(best_v)

def solve():
    t0 = time.time()
    DEADLINE = 155.0
    rng = np.random.default_rng(0)
    n, d = 181, 9
    cands = []
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=float)
            if a.shape == (n, d):
                a = a / np.linalg.norm(a, axis=1, keepdims=True)
                if np.all(np.isfinite(a)):
                    cands.append(a)
        except Exception:
            pass
    if not cands:
        a = rng.standard_normal((n, d))
        a /= np.linalg.norm(a, axis=1, keepdims=True)
        cands.append(a)
    vals = [_max_cos(c) for c in cands]
    bi = int(np.argmin(vals))
    X0 = cands[bi].copy()
    inc = X0.copy()
    inc_val = float(vals[bi])
    # 1) one LP polish
    rem = DEADLINE - (time.time() - t0)
    if rem > 12:
        Xp, vp, LI, LJ, LW, = _lp_polish_with_duals(inc, seconds=min(7.0, rem - 5.0), init_radius=2e-3)
        if vp < inc_val and np.all(np.isfinite(Xp)):
            inc, inc_val = Xp, vp
            X0 = Xp.copy()
        else:
            # keep polished even if not better for stress info? use best
            if np.all(np.isfinite(Xp)):
                X0 = Xp.copy()
                # keep duals from this polish
            else:
                LI, LJ, LW = None, None, None
    else:
        LI, LJ, LW = None, None, None
        vp = inc_val
    if time.time() - t0 > DEADLINE:
        return inc / np.linalg.norm(inc, axis=1, keepdims=True)
    # 2) stress per point + contacts
    try:
        G = X0 @ X0.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        # contacts at 1e-9
        thr = 1e-9
        iu = np.triu_indices(n, 1)
        c_all = (X0 @ X0.T)[iu]
        is_contact = c_all >= m - thr
        deg = np.zeros(n, dtype=int)
        np.add.at(deg, iu[0][is_contact], 1)
        np.add.at(deg, iu[1][is_contact], 1)
    except Exception:
        return inc / np.linalg.norm(inc, axis=1, keepdims=True)
    # point stress from duals if available
    stress = np.zeros(n)
    if LI is not None and LJ is not None and LW is not None and len(LI) == len(LW):
        try:
            for a, b, w in zip(LI, LJ, LW):
                stress[int(a)] += float(w)
                stress[int(b)] += float(w)
        except Exception:
            stress = deg.astype(float)
    else:
        stress = deg.astype(float)
    # rattlers: two lowest degree
    order_deg = np.argsort(deg, kind="stable")
    rattlers = list(order_deg[:2])
    # build adjacency from tight pairs (use contact mask, expand if too few)
    adj = [set() for _ in range(n)]
    try:
        sel_idx = np.where(is_contact)[0]
        if len(sel_idx) < 500:
            # expand threshold to top 1200
            kk = min(1200, len(c_all))
            kth = np.partition(c_all, -kk)[-kk]
            is2 = c_all >= kth
            sel_idx = np.where(is2)[0]
        for k in sel_idx:
            i = int(iu[0][k]); j = int(iu[1][k])
            adj[i].add(j); adj[j].add(i)
    except Exception:
        pass
    # highest-stress connected cluster of size 12 (10-18 range)
    Kcore = 12
    seed = int(np.argmax(stress))
    # avoid seed being rattler
    for s in np.argsort(-stress):
        if int(s) not in rattlers:
            seed = int(s)
            break
    visited = set([seed])
    # grow by best stress neighbor of cluster
    while len(visited) < Kcore:
        best_c = -1; best_s = -1e100
        for v in list(visited):
            for nb in adj[v]:
                if nb in visited or nb in rattlers:
                    continue
                if stress[nb] > best_s:
                    best_s = stress[nb]; best_c = nb
        if best_c == -1:
            # disconnected: add highest stress not yet taken
            for s in np.argsort(-stress):
                si = int(s)
                if si not in visited and si not in rattlers:
                    best_c = si
                    break
            if best_c == -1:
                break
        visited.add(int(best_c))
        if time.time() - t0 > DEADLINE - 20:
            break
    core = sorted(list(visited))[:Kcore]
    R = sorted(set(rattlers) | set(core))
    F = [i for i in range(n) if i not in set(R)]
    if len(R) < 4 or len(F) < 10:
        # fallback: just return incumbent polished
        rem2 = DEADLINE - (time.time() - t0)
        if rem2 > 3:
            Xf, vf, _, _, _ = _lp_polish_with_duals(inc, seconds=min(rem2 - 1.0, 20.0))
            if vf < inc_val:
                inc, inc_val = Xf, vf
        return inc / np.linalg.norm(inc, axis=1, keepdims=True)
    Xfrozen = X0[F].copy()
    # 3) reinsert one at a time against frozen set
    cur = Xfrozen.copy()  # grows
    new_pts = []
    inner_deadline = DEADLINE - 25.0  # leave time for smoothing+polish
    for _ in R:
        if time.time() - t0 > inner_deadline:
            # fill remaining with deepest holes quickly
            break
        v = _insert_one(cur, rng, t0, inner_deadline)
        new_pts.append(v)
        cur = np.vstack([cur, v.reshape(1, -1)])
    # if interrupted, fill rest with random/hole best-effort
    while len(new_pts) < len(R):
        try:
            v = rng.standard_normal(d)
            v /= np.linalg.norm(v)
        except Exception:
            v = new_pts[0].copy() if new_pts else X0[R[0]]
        new_pts.append(v)
    Xnew = np.empty_like(X0)
    Xnew[F] = Xfrozen
    for idx, v in zip(R, new_pts):
        Xnew[int(idx)] = v
    Xnew /= np.linalg.norm(Xnew, axis=1, keepdims=True)
    if time.time() - t0 > DEADLINE - 2:
        # return incumbent immediately
        vnew = _max_cos(Xnew)
        if vnew < inc_val:
            return Xnew
        return inc
    # 4) short exponential-repulsion smoothing
    rem = DEADLINE - (time.time() - t0)
    if rem > 8:
        steps = 150 if rem > 30 else 80
        Xnew = _smooth_exp(Xnew, steps=steps, T0=0.02, T1=0.005, lr=0.04, t_deadline=DEADLINE - 6, t0=t0)
        Xnew /= np.linalg.norm(Xnew, axis=1, keepdims=True)
    if time.time() - t0 > DEADLINE - 2:
        vnew = _max_cos(Xnew)
        if vnew < inc_val:
            inc, inc_val = Xnew, vnew
        return inc / np.linalg.norm(inc, axis=1, keepdims=True)
    # 5) final LP polish under strict deadline
    rem = DEADLINE - (time.time() - t0)
    if rem > 3:
        Xp2, vp2, _, _, _ = _lp_polish_with_duals(Xnew, seconds=max(1.0, rem - 1.0), init_radius=2e-3)
        try:
            if np.all(np.isfinite(Xp2)) and vp2 < _max_cos(Xnew) + 1e-12:
                Xnew = Xp2
        except Exception:
            pass
    try:
        vnew = _max_cos(Xnew)
    except Exception:
        return inc
    if vnew < inc_val:
        inc, inc_val = Xnew, vnew
    inc /= np.linalg.norm(inc, axis=1, keepdims=True)
    return inc
