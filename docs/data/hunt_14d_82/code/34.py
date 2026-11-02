import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _norm(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return X / n

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _lp_polish(X, seconds, radius_init=2e-4, cap=1200, max_it=12):
    t_end = time.time() + seconds
    X = _norm(X.copy())
    n, d = X.shape
    best = _max_cos(X)
    radius = float(radius_init)
    iu = np.triu_indices(n, 1)
    it = 0
    while time.time() < t_end and radius > 1e-13 and it < max_it:
        it += 1
        G = X @ X.T
        c_all = G[iu]
        m = float(c_all.max())
        thr = m - max(20.0 * radius, 5e-8)
        sel = c_all > thr
        cnt = int(sel.sum())
        if cnt == 0:
            break
        if cnt > cap:
            idx = np.where(sel)[0]
            vals = c_all[idx]
            part = np.argpartition(-vals, cap - 1)[:cap]
            m2 = np.zeros_like(sel, dtype=bool)
            m2[idx[part]] = True
            sel = m2
        I = iu[0][sel]; J = iu[1][sel]
        c = c_all[sel]
        p = len(I); nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cd = np.arange(d)
        cols = np.concatenate([I[:, None] * d + cd, J[:, None] * d + cd, np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([X[J], X[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((X.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv); obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-c, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        Y = X + res.x[:-1].reshape(n, d)
        Y = _norm(Y)
        v = _max_cos(Y)
        if v < best:
            X, best = Y, v
            radius = min(radius * 1.5, 5e-3)
        else:
            radius *= 0.5
    return X

def solve():
    t0 = time.time()
    BUDGET = 168.0
    def remain():
        return BUDGET - (time.time() - t0)
    rng = np.random.default_rng(0)
    n, d = 82, 14
    nd = n * d
    # load start
    xb = None
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn).astype(float)
            if a.shape == (n, d):
                a = _norm(a)
                if np.all(np.isfinite(a)):
                    xb = a.copy()
                    break
        except Exception:
            continue
    if xb is None:
        xb = _norm(rng.standard_normal((n, d)))
    best_val = _max_cos(xb)
    X_best = xb.copy()
    iu0 = np.triu_indices(n, 1)
    I0 = iu0[0]; J0 = iu0[1]
    npair = len(I0)

    # ---------- rigidity surgery ----------
    X = X_best.copy()
    cur_m = best_val
    # preallocate
    SLACK = 5e-6
    TOL_DEG = 1e-5
    n_iter = 0
    max_iters = 400
    surgery_deadline = t0 + 115.0
    # for AA^T solve regularizer
    while n_iter < max_iters and time.time() < surgery_deadline and remain() > 25:
        n_iter += 1
        G = X @ X.T
        # upper values
        c_all = G[I0, J0]
        m = float(c_all.max())
        # track global best
        if m < best_val - 1e-12:
            best_val = m
            X_best = X.copy()
        slack = m - c_all
        # active for rigidity (slack < 5e-6)
        act_mask = slack < SLACK
        # degree at 1e-5
        deg_mask = slack < TOL_DEG
        deg = np.zeros(n, dtype=int)
        # fast bincount
        ai = I0[deg_mask]; aj = J0[deg_mask]
        # count
        for k in range(len(ai)):
            deg[ai[k]] += 1
            deg[aj[k]] += 1
        # identify max pairs (slack < 1e-12)
        max_mask = slack < 1e-12
        nmax = int(max_mask.sum())
        if nmax == 0:
            # numerical: take top 1
            kTop = int(np.argmax(c_all))
            max_mask = np.zeros_like(act_mask)
            max_mask[kTop] = True
            nmax = 1
        # build lists
        act_idx = np.where(act_mask)[0]
        max_idx = np.where(max_mask)[0]
        max_set = set(max_idx.tolist())
        sub_idx = act_idx[~np.isin(act_idx, max_idx)] if len(act_idx) > 0 else np.array([], dtype=int)
        # cap sub to tightest 450 to bound cost
        if len(sub_idx) > 450:
            # keep smallest slack
            ssub = slack[sub_idx]
            part = np.argpartition(ssub, 450)[:450]
            sub_idx = sub_idx[part]
        p_sub = len(sub_idx)
        # descent direction for max pairs (negative grad of max dot)
        D = np.zeros((n, d))
        for k in max_idx[:4]:
            i = int(I0[k]); j = int(J0[k])
            dot = float(G[i, j])
            #clip
            if dot > 1: dot = 1
            D[i] += -(X[j] - dot * X[i])
            D[j] += -(X[i] - dot * X[j])
        # project to tangent (remove normal comp)
        dn = np.sum(D * X, axis=1, keepdims=True)
        D = D - dn * X
        d_flat = D.ravel()
        nd0 = float(np.linalg.norm(d_flat))
        # build constraint matrix A: (p_sub + n) x nd
        q = p_sub + n
        if p_sub > 0 and nd0 > 1e-18:
            try:
                A = np.zeros((q, nd), dtype=np.float64)
                # fill pair rows
                # loop in python but p_sub<=450
                for r, k in enumerate(sub_idx):
                    i = int(I0[k]); j = int(J0[k])
                    A[r, i*d:(i+1)*d] = X[j]
                    A[r, j*d:(j+1)*d] = X[i]
                # tangent rows
                for i in range(n):
                    A[p_sub+i, i*d:(i+1)*d] = X[i]
                Ad = A @ d_flat
                # solve M y = Ad
                M = A @ A.T
                # regularize diagonal slightly
                M.flat[::q+1] += 1e-12
                # use lstsq for stability (fast via solve if well cond)
                try:
                    y = np.linalg.solve(M, Ad)
                except Exception:
                    y = np.linalg.lstsq(M, Ad, rcond=None)[0]
                v_flat = d_flat - A.T @ y
            except Exception:
                # fallback: tangent only
                v_flat = d_flat.copy()
        else:
            v_flat = d_flat.copy()
        nv = float(np.linalg.norm(v_flat))
        V = v_flat.reshape(n, d)
        # if floppy-projected direction vanished -> jammed w.r.t sub set: kick along smallest mode
        # detect: nv << nd0
        jammed = (nv < 0.05 * nd0) or (nv < 1e-14)
        step_accepted = False
        if not jammed and nv > 1e-18:
            # normalize to rms step scale
            Vrms = nv / np.sqrt(float(n))
            Vn = V / (Vrms + 1e-30)  # rms=1
            # event-driven line search: try increasing steps until new contact or improvement
            # base scale: rms displacement s
            for s in (2e-4, 5e-4, 1e-3, 2e-3, 4e-3):
                Xt = X + s * Vn
                Xt = _norm(Xt)
                Gt = Xt @ Xt.T
                mt = float(Gt[I0, J0].max())
                if mt < m - 2e-10:
                    X = Xt; cur_m = mt
                    step_accepted = True
                    break
            if not step_accepted:
                # try smaller steps that still decrease
                for s in (5e-5, 1e-4):
                    Xt = X + s * Vn
                    Xt = _norm(Xt)
                    mt = float((Xt @ Xt.T)[I0, J0].max())
                    if mt < m - 1e-12:
                        X = Xt; cur_m = mt
                        step_accepted = True
                        break
        if not step_accepted:
            # targeted rattler relocation along descent (lowest degree)
            order = np.argsort(deg)
            # pick 1-2 lowest; move along repulsive descent of its top neighbors
            moved = False
            for cand in order[:2]:
                i = int(cand)
                row = G[i].copy()
                row[i] = -2
                # top 8 neighbors
                nn_idx = np.argpartition(-row, 8)[:8]
                gi = np.zeros(d)
                for j in nn_idx:
                    w = row[j]
                    # weight by closeness
                    gi += X[int(j)]
                # tangent project
                gi = gi - float(gi @ X[i]) * X[i]
                ng = float(np.linalg.norm(gi))
                if ng < 1e-14:
                    gi = rng.standard_normal(d)
                    gi = gi - float(gi @ X[i]) * X[i]
                    ng = float(np.linalg.norm(gi))
                    if ng < 1e-14:
                        continue
                di = -gi / ng
                # line search for this point only
                best_m_local = m
                best_X_local = None
                for s in (1e-3, 3e-3, 8e-3, 2e-2, 5e-4):
                    Xt = X.copy()
                    Xt[i] = Xt[i] + s * di
                    Xt[i] /= np.linalg.norm(Xt[i])
                    mt = float((Xt @ Xt.T)[I0, J0].max())
                    if mt < best_m_local - 5e-10:
                        best_m_local = mt
                        best_X_local = Xt
                        break
                if best_X_local is not None:
                    X = best_X_local; cur_m = best_m_local
                    moved = True
                    break
            if not moved:
                # small global jitter along floppy + random to change topology
                sig = 3e-4 if (n_iter % 10 != 0) else 1.5e-3
                Xt = X + sig * rng.standard_normal((n, d))
                # keep tangent approx: reproject not needed, just normalize
                Xt = _norm(Xt)
                # accept if not much worse (allow hill climb to new basin)
                mt = float((Xt @ Xt.T)[I0, J0].max())
                if mt < m + 2e-4:
                    X = Xt; cur_m = mt
                else:
                    # force tiny move
                    Xt2 = _norm(X + 1e-4 * rng.standard_normal((n, d)))
                    X = Xt2; cur_m = float((X @ X.T)[I0, J0].max())
        # periodic polish of current if improved notably, and update best
        if n_iter % 40 == 0 and remain() > 30:
            if cur_m < best_val + 5e-5:
                Xp = _lp_polish(X, min(1.5, remain() - 20), radius_init=3e-4, cap=1000, max_it=6)
                vp = _max_cos(Xp)
                if vp < best_val:
                    best_val = vp; X_best = Xp.copy()
                # continue from polished if better than current
                if vp < cur_m:
                    X = Xp; cur_m = vp
        # early exit if highly rigid signature reached and improved
        if n_iter % 20 == 0:
            # check avg degree at 1e-5
            avgd = float(deg.mean())
            # if avg high and best improved, can polish harder
            if avgd > 18 and best_val < _max_cos(xb) - 1e-6:
                pass

    # final polish best + current
    if remain() > 6:
        # polish best
        try:
            Xp = _lp_polish(X_best, min(6.0, remain() * 0.4), radius_init=2e-4, cap=1500, max_it=12)
            vp = _max_cos(Xp)
            if vp < best_val:
                best_val = vp; X_best = Xp.copy()
        except Exception:
            pass
    if remain() > 4:
        try:
            # also try polishing current if different
            if cur_m < best_val + 1e-4:
                Xp2 = _lp_polish(X, min(4.0, remain() - 1.0), radius_init=1.5e-4, cap=1500, max_it=10)
                vp2 = _max_cos(Xp2)
                if vp2 < best_val:
                    best_val = vp2; X_best = Xp2.copy()
        except Exception:
            pass
    # safety: ensure finite, normalized, shrink margin
    X_best = _norm(np.asarray(X_best, dtype=float))
    # tiny shrink: pull max pair apart by epsilon to guarantee strict feasibility margin
    # do 2 micro gradient steps on max pair
    try:
        G = X_best @ X_best.T
        np.fill_diagonal(G, -2)
        ij = np.unravel_index(int(np.argmax(G)), G.shape)
        i, j = int(ij[0]), int(ij[1])
        # move them apart by 1e-9 tangent
        dot = float(X_best[i] @ X_best[j])
        di = -(X_best[j] - dot * X_best[i])
        dj = -(X_best[i] - dot * X_best[j])
        X_best[i] = _norm(X_best[i:i+1] + 1e-7 * di[None, :])[0]
        X_best[j] = _norm(X_best[j:j+1] + 1e-7 * dj[None, :])[0]
        X_best = _norm(X_best)
        vcheck = _max_cos(X_best)
        if vcheck > best_val + 5e-9:
            # revert micro move if harmful (recompute) -> just renormalize previous
            pass
    except Exception:
        pass
    X_best = _norm(X_best)
    return X_best
