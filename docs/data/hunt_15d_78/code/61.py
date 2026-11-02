import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _personal_max(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return G.max(axis=1), G

def _lp_polish(X, deadline, r_init=0.01, r_max=0.05, cap=2500):
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    n, d = X.shape
    best = _max_cos(X)
    best_x = X.copy()
    radius = r_init
    iu0, iu1 = np.triu_indices(n, 1)
    nv = n * d + 1
    obj = np.zeros(nv); obj[-1] = 1.0
    while time.time() + 0.45 < deadline and radius > 1e-13:
        G = X @ X.T
        c_all = G[iu0, iu1]
        m = float(c_all.max())
        thresh = max(20.0 * radius, 1e-9)
        sel = c_all > m - thresh
        nnz = int(np.count_nonzero(sel))
        if nnz == 0:
            radius *= 0.5
            continue
        if nnz > cap:
            idx = np.where(sel)[0]
            order = np.argsort(-c_all[idx])[:cap]
            mask = np.zeros_like(sel, dtype=bool)
            mask[idx[order]] = True
            sel = mask
        I = iu0[sel]; J = iu1[sel]
        p = len(I)
        c_sel = c_all[sel]
        cols = np.concatenate([I[:, None]*d + np.arange(d), J[:, None]*d + np.arange(d), np.full((p,1), n*d)], axis=1).ravel()
        rows = np.repeat(np.arange(p), 2*d+1)
        vals = np.concatenate([X[J], X[I], -np.ones((p,1))], axis=1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((X.ravel(), (np.repeat(np.arange(n), d), np.arange(n*d))), shape=(n, nv)).tocsr()
            bounds = [(-radius, radius)]*(n*d) + [(None, None)]
            res = linprog(obj, A_ub=A_ub, b_ub=-c_sel, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=bounds, method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = X + res.x[:-1].reshape(n, d)
        yn = np.linalg.norm(y, axis=1, keepdims=True)
        if not np.all(np.isfinite(yn)) or np.any(yn < 1e-12):
            radius *= 0.5
            continue
        y = y / yn
        v = _max_cos(y)
        if v < best:
            X = y; best = v; best_x = y.copy()
            radius = min(radius*1.5, r_max)
        else:
            radius *= 0.5
    return best_x, best

def _adam_polish(X, deadline, lr=0.004, beta0=250.0, beta1=900.0):
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    n, d = X.shape
    iu0, iu1 = np.triu_indices(n, 1)
    best = _max_cos(X)
    best_x = X.copy()
    m1 = np.zeros_like(X); m2 = np.zeros_like(X)
    t = 0
    while time.time() + 0.15 < deadline:
        t += 1
        frac = 1.0  # simple: ramp beta with t
        beta = beta0 + (beta1-beta0)*min(1.0, t/3000.0)
        G = X @ X.T
        gv = G[iu0, iu1]
        m = float(gv.max())
        e = np.exp(beta*(gv - m))
        s = float(e.sum())
        if not np.finite(s) or s == 0:
            break
        w = e / s
        Gr = np.zeros_like(X)
        # accumulate W @ X
        # w corresponds to pairs (iu0,iu1)
        Wx_J = w[:, None] * X[iu1]
        Wx_I = w[:, None] * X[iu0]
        np.add.at(Gr, iu0, Wx_J)
        np.add.at(Gr, iu1, Wx_I)
        Gr *= 2.0
        dot = np.sum(Gr*X, axis=1, keepdims=True)
        Gt = Gr - dot*X
        # Adam
        m1 = 0.9*m1 + 0.1*Gt
        m2 = 0.999*m2 + 0.001*(Gt*Gt)
        c1 = 1.0 - 0.9**t
        c2 = 1.0 - 0.999**t
        step = (m1/c1) / (np.sqrt(m2/c2)+1e-8)
        # clip step norm
        X = X - lr*step
        X = X / np.linalg.norm(X, axis=1, keepdims=True)
        if t % 120 == 0:
            v = _max_cos(X)
            if v < best:
                best = v; best_x = X.copy()
        if t > 8000:
            break
    v = _max_cos(X)
    if v < best:
        best = v; best_x = X.copy()
    return best_x, best

def _refine_hole(v0, Xf, iters=250, lr0=0.08):
    v = v0 / np.linalg.norm(v0)
    best_v = v.copy()
    dots = Xf @ v
    best_m = float(dots.max())
    lr = lr0
    for it in range(iters):
        dots = Xf @ v
        m = float(dots.max())
        act = dots > m - 5e-3
        if not np.any(act):
            break
        g = Xf[act].sum(axis=0)
        g = g - float(g @ v)*v
        gn = np.linalg.norm(g)
        if gn > 1e-12:
            g = g/gn
        v = v - lr*g
        nrm = np.linalg.norm(v)
        if not np.finite(nrm) or nrm < 1e-12:
            break
        v = v/nrm
        nm = float(Xf @ v .max())
        if nm < best_m:
            best_m = nm; best_v = v.copy()
        lr *= 0.995
    # count near contacts at best
    cnt = int(np.sum((Xf @ best_v) > best_m - 1e-6))
    return best_v, best_m, cnt

def _sample_hole(Xf, rng, n_sample, n_keep, refine_iters=200):
    d = Xf.shape[1]
    S = rng.standard_normal((n_sample, d))
    S /= np.linalg.norm(S, axis=1, keepdims=True)
    D = S @ Xf.T
    mc = D.max(axis=1)
    # near count per candidate (dots > mc-1e-6)
    # compute quickly for all? use loop over candidates via vectorization chunk
    # approximate cnt via (D > mc[:,None]-1e-6).sum(axis=1) -- 30k x 77 bool ~2.3M, ok
    cnt = np.sum(D > mc[:, None] - 1e-6, axis=1)
    order = np.lexsort((cnt, mc))[:n_keep]
    best_v = None; best_m = 1e9; best_c = 1e9
    for idx in order:
        v0 = S[idx]
        vr, mr, cr = _refine_hole(v0, Xf, iters=refine_iters)
        if (mr < best_m - 1e-12) or (abs(mr-best_m) <= 1e-12 and cr < best_c):
            best_m = mr; best_c = cr; best_v = vr
    return best_v, best_m, best_c

def solve():
    t0 = time.time()
    HARD_END = t0 + 172.0
    rng = np.random.default_rng(0)
    n, d = 78, 15
    cand = []
    try:
        b = np.load("best.npy")
        if b.shape == (n, d):
            cand.append(b.astype(float))
    except Exception:
        pass
    try:
        r = np.load("record.npy")
        if r.shape == (n, d):
            cand.append(r.astype(float))
    except Exception:
        pass
    if len(cand) == 0:
        x0 = rng.standard_normal((n, d)); x0 /= np.linalg.norm(x0, axis=1, keepdims=True)
        cand.append(x0)
    best_x = None; best_v = 1e9
    for c in cand:
        c = c/np.linalg.norm(c, axis=1, keepdims=True)
        v = _max_cos(c)
        if v < best_v:
            best_v = v; best_x = c.copy()
    x_best = best_x.copy(); v_best = best_v

    # ---- vertex-cover deletion ----
    try:
        G0 = x_best @ x_best.T
        np.fill_diagonal(G0, -2.0)
        m0 = float(G0.max())
        tight = (G0 > m0 - 1e-7)
        # greedy cover ~16 vertices
        uncovered = tight.copy()
        np.fill_diagonal(uncovered, False)
        cover = []
        in_cover = np.zeros(n, dtype=bool)
        for _ in range(16):
            deg = uncovered.sum(axis=1)
            deg[in_cover] = -1
            i = int(np.argmax(deg))
            if deg[i] <= 0:
                # add highest-degree in tight to reach size
                deg2 = tight.sum(axis=1); deg2[in_cover] = -1
                i = int(np.argmax(deg2))
                if deg2[i] <= 0:
                    break
            cover.append(i); in_cover[i] = True
            # remove edges incident to i
            uncovered[i, :] = False; uncovered[:, i] = False
            if len(cover) >= 14:
                # coverage fraction
                tot = int(tight.sum()//2)
                cov = tot - int(uncovered.sum()//2)
                if cov >= 0.88*tot or len(cover) >= 16:
                    break
        # ensure 14-18
        while len(cover) < 14:
            deg2 = tight.sum(axis=1); deg2[in_cover] = -1
            i = int(np.argmax(deg2))
            if deg2[i] <= 0: break
            cover.append(i); in_cover[i] = True
        keep_mask = ~in_cover
        keep_idx = np.where(keep_mask)[0]
        if len(keep_idx) < 60:
            # fallback keep 62
            keep_idx = np.argsort(tight.sum(axis=1))[:62]
            in_cover = np.ones(n, dtype=bool); in_cover[keep_idx] = False
            cover = np.where(in_cover)[0].tolist()
        Xrem = x_best[keep_idx].copy()
        # polish remaining: softmax adam + LP
        t_rem_adam = min(time.time()+6.0, HARD_END-60)
        if time.time() < t_rem_adam:
            Xrem, _ = _adam_polish(Xrem, t_rem_adam, lr=0.005)
        t_rem_lp = min(time.time()+6.0, HARD_END-55)
        if time.time() < t_rem_lp:
            Xrem, v_rem = _lp_polish(Xrem, t_rem_lp, r_init=0.01, r_max=0.05)
        else:
            v_rem = _max_cos(Xrem)
        # ---- reinsert one at a time ----
        Xcur = Xrem.copy()
        # internal max
        for s in range(len(cover)):
            if time.time() > HARD_END - 35:
                # fast fill remaining
                need = n - Xcur.shape[0]
                if need > 0:
                    R = rng.standard_normal((need, d)); R /= np.linalg.norm(R, axis=1, keepdims=True)
                    Xcur = np.vstack([Xcur, R])
                break
            # sample at least 30000
            bv, bm, bc = _sample_hole(Xcur, rng, 30000, 20, refine_iters=220)
            Xcur = np.vstack([Xcur, bv[None, :]])
        # Xcur should be 78 now
        if Xcur.shape[0] == n:
            Xcur = Xcur/np.linalg.norm(Xcur, axis=1, keepdims=True)
            v_cur = _max_cos(Xcur)
            # joint smoothing attempt
            if time.time() < HARD_END - 30:
                Xcur, v_cur = _adam_polish(Xcur, min(time.time()+5.0, HARD_END-28), lr=0.003)
            if time.time() < HARD_END - 25:
                Xcur, v_cur = _lp_polish(Xcur, min(time.time()+5.0, HARD_END-22), r_init=0.006, r_max=0.03)
            if v_cur < v_best - 1e-13:
                x_best = Xcur.copy(); v_best = v_cur
            else:
                # keep surgery result as alternate base for further cycles if close
                # choose better of the two for subsequent relocation cycles
                if v_cur < v_best + 5e-4:
                    x_alt = Xcur.copy(); v_alt = v_cur
                else:
                    x_alt = None
        else:
            x_alt = None
    except Exception:
        x_alt = None

    # ---- alternating best-hole relocation + smoothing ----
    base = x_best.copy()
    # if surgery produced competitive alt, start from it to stay in new basin
    try:
        if 'x_alt' in locals() and x_alt is not None and 'v_alt' in locals():
            # prefer alt if within 1e-4 (new basin) to try to push below
            if v_alt < v_best + 2e-4:
                base = x_alt.copy()
    except Exception:
        pass
    bv = _max_cos(base)
    if bv < v_best:
        x_best = base.copy(); v_best = bv
    # cycles until reserve
    while time.time() < HARD_END - 20.5:
        # find worst points
        pm, _ = _personal_max(base)
        mcur = float(pm.max())
        order = np.argsort(-pm)[:8]
        improved = False
        for ii in order:
            if time.time() > HARD_END - 21:
                break
            if pm[ii] < mcur - 1e-7:
                continue
            Xf = np.delete(base, ii, axis=0)
            # quick hole search with 6000 samples
            try:
                bvec, bm, bc = _sample_hole(Xf, rng, 6000, 6, refine_iters=150)
            except Exception:
                continue
            Xtry = Xf.copy()
            Xtry = np.vstack([Xtry, bvec[None, :]])
            # new global max
            vtry = _max_cos(Xtry)
            # count pairs above vtry-1e-6 for tie-break
            if vtry < _max_cos(base) - 1e-13:
                base = Xtry; improved = True
                break
            elif vtry <= _max_cos(base) + 1e-12:
                # compare counts
                G1 = base @ base.T; np.fill_diagonal(G1, -2)
                c1 = int(np.sum(G1 > vtry - 1e-6)//2)
                G2 = Xtry @ Xtry.T; np.fill_diagonal(G2, -2)
                c2 = int(np.sum(G2 > vtry - 1e-6)//2)
                if c2 < c1:
                    base = Xtry; improved = True
                    break
        # joint smoothing short burst
        if time.time() < HARD_END - 20.5:
            nb, vb = _adam_polish(base, min(time.time()+1.2, HARD_END-20.5), lr=0.002, beta0=400.0, beta1=1200.0)
            # accept if not worse by much
            if vb <= _max_cos(base) + 1e-12:
                base = nb
        if _max_cos(base) < v_best - 1e-13:
            x_best = base.copy(); v_best = _max_cos(base)
        if not improved and time.time() < HARD_END - 20.5:
            # small random kick to worst point to escape
            pass
        if time.time() >= HARD_END - 20.5:
            break
        # avoid infinite loop: at most a few cycles; break if time low
        # continue

    # ---- final LP reserve 20s ----
    if time.time() < HARD_END - 0.4:
        # start from best between x_best and base
        if _max_cos(base) < v_best:
            x_best = base.copy(); v_best = _max_cos(base)
        x_best, v_best = _lp_polish(x_best, HARD_END - 0.3, r_init=0.003, r_max=0.02)
        if time.time() < HARD_END - 0.3:
            x_best, v_best = _lp_polish(x_best, HARD_END - 0.2, r_init=0.0006, r_max=0.006)

    x_best = x_best/np.linalg.norm(x_best, axis=1, keepdims=True)
    if not np.all(np.isfinite(x_best)):
        x_best = best_x/np.linalg.norm(best_x, axis=1, keepdims=True)
    return x_best
