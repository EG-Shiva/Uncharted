import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _norm(X):
    n = np.linalg.norm(X, axis=1, keepdims=True) + 1e-30
    return X / n

def _lp_polish_big(x, deadline, r_init=1e-2, gap=1e-3, cap=3000):
    x = _norm(x.copy())
    n, d = x.shape
    try:
        best = _max_cos(x)
    except Exception:
        return x, 1e9
    radius = r_init
    iu = np.triu_indices(n, 1)
    ar = np.arange(d)
    while time.time() < deadline and radius > 1e-14:
        Gfull = x @ x.T
        c = Gfull[iu]
        m = float(c.max())
        thr = m - gap
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
        cols = np.concatenate([I[:, None] * d + ar[None, :], J[:, None] * d + ar[None, :], np.full((p, 1), n * d)], axis=1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], axis=1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv); obj[-1] = 1.0
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
        vv = _max_cos(y)
        if vv < best:
            x, best = y, vv
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x, best

def _anneal_lse(x0, steps, beta0, beta1, lr0, lr1, forbid=None, forbid_target=None, forbid_w=5.0):
    x = _norm(x0.copy())
    n, d = x.shape
    iu_forbid = None
    if forbid is not None:
        fi, fj = forbid
    for s in range(steps):
        f = s / max(steps - 1, 1)
        beta = beta0 + (beta1 - beta0) * f
        lr = lr0 + (lr1 - lr0) * f
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        E = np.exp(beta * (G - m))
        np.fill_diagonal(E, 0.0)
        if forbid is not None:
            E[fi, fj] = 0.0
            E[fj, fi] = 0.0
        S = float(E.sum()) + 1e-300
        Grad = (E @ x) / S
        if forbid is not None and forbid_target is not None:
            c = float(x[fi] @ x[fj])
            if c > forbid_target:
                # push apart along tangent
                gi = x[fj] - c * x[fi]
                gj = x[fi] - c * x[fj]
                Grad[fi] += forbid_w * (c - forbid_target) * gi
                Grad[fj] += forbid_w * (c - forbid_target) * gj
        dot = np.sum(Grad * x, axis=1, keepdims=True)
        Tan = Grad - dot * x
        # clip
        tn = np.linalg.norm(Tan)
        if tn > 10.0:
            Tan *= 10.0 / tn
        x = x - lr * Tan
        x = _norm(x)
    return x

def _tangent_project(D, X):
    dot = np.sum(D * X, axis=1, keepdims=True)
    return D - dot * X

def solve():
    t0 = time.time()
    deadline = t0 + 168.0
    n, d = 138, 8
    rng = np.random.default_rng(0)
    best = None; best_val = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            v = np.load(fn).astype(float)
            if v.shape == (n, d):
                v = _norm(v)
                c = _max_cos(v)
                if c < best_val:
                    best_val = c; best = v.copy()
        except Exception:
            pass
    if best is None:
        best = _norm(rng.standard_normal((n, d)))
        best_val = _max_cos(best)
    x0 = best.copy(); m0 = best_val

    # baseline big-radius polish of incumbent (short)
    try:
        xb, vb = _lp_polish_big(x0, min(deadline - 60.0, t0 + 12.0), r_init=1e-2, gap=1e-3, cap=3000)
        if vb < best_val - 1e-15:
            best_val, best = vb, xb.copy()
            x0, m0 = xb.copy(), vb
    except Exception:
        pass

    # ---- gather tight pairs ----
    G0 = x0 @ x0.T
    iu = np.triu_indices(n, 1)
    c0 = G0[iu]
    m_cur = float(c0.max())
    order = np.argsort(-c0)
    tight_pairs = [(int(iu[0][k]), int(iu[1][k]), float(c0[k])) for k in order[:40]]

    # ---- floppy modes via Jacobian SVD at several tolerances ----
    candidates = []  # (score_along_line, X_t, desc)
    tols = [1e-7, 1e-5, 1e-4]
    # log spaced steps
    steps_line = np.logspace(-4, -0.5, 14)  # 1e-4 .. 0.316
    dirs = []
    for tol in tols:
        if time.time() > deadline - 55.0:
            break
        sel_mask = c0 > (m_cur - tol)
        # ensure at least some pairs
        if int(np.count_nonzero(sel_mask)) < 5:
            # take top 50
            sel_mask = np.zeros_like(sel_mask, dtype=bool)
            sel_mask[order[:50]] = True
        I = iu[0][sel_mask]; J = iu[1][sel_mask]
        p = len(I)
        if p == 0:
            continue
        if p > 1500:
            # keep tightest 1500
            sub = np.argsort(-c0[sel_mask])[:1500]
            I, J = I[sub], J[sub]
            p = len(I)
        # Build Jacobian p x (n*d)
        Nd = n * d
        Jmat = np.zeros((p, Nd), dtype=float)
        cc = np.sum(x0[I] * x0[J], axis=1)
        Ri = x0[J] - cc[:, None] * x0[I]
        Rj = x0[I] - cc[:, None] * x0[J]
        # fill
        for r in range(p):
            Jmat[r, I[r]*d:(I[r]+1)*d] = Ri[r]
            Jmat[r, J[r]*d:(J[r]+1)*d] = Rj[r]
        try:
            # SVD: J = U S Vt ; null dirs = last rows of Vt
            _, Svals, Vt = np.linalg.svd(Jmat, full_matrices=False)
        except Exception:
            continue
        # pick smallest singular vectors: last 4 + a couple mid
        kpick = min(4, Vt.shape[0])
        for k in range(kpick):
            v = Vt[-(k+1), :].copy()
            D = v.reshape(n, d)
            D = _tangent_project(D, x0)
            fn_ = np.linalg.norm(D) + 1e-30
            D = D / fn_
            # both signs
            for sgn in (+1.0, -1.0):
                dirs.append((D * sgn, tol, Svals[-1] if len(Svals)>0 else 0.0))
        # also one random combo of null space for diversity (deterministic)
        if Vt.shape[0] >= 8:
            combo = Vt[-8:, :].T @ rng.standard_normal(8)
            D = combo.reshape(n, d)
            D = _tangent_project(D, x0)
            fn_ = np.linalg.norm(D) + 1e-30
            if fn_ > 1e-12:
                D = D / fn_
                dirs.append((D, tol, 0.0))
                dirs.append((-D, tol, 0.0))

    # line search over dirs: evaluate quickly, collect promising t
    # limit dirs to ~16 to save time
    if len(dirs) > 16:
        # prefer smaller tol? keep evenly
        dirs = dirs[:16]
    scored_starts = []
    active_set0 = set(zip(iu[0][order[:50]].tolist(), iu[1][order[:50]].tolist()))
    for di, (D, tol, sval) in enumerate(dirs):
        if time.time() > deadline - 50.0:
            break
        best_t = None; best_mt = 1e9; best_Xt = None
        # track first topology change: first t where argmax not in active_set0
        topo_t = None; topo_X = None
        for t in steps_line:
            Xt = _norm(x0 + t * D)
            # fast max
            G = Xt @ Xt.T
            np.fill_diagonal(G, -2.0)
            mt = float(G.max())
            if mt < best_mt:
                best_mt = mt; best_t = t; best_Xt = Xt
            if topo_t is None and mt <= m_cur + 1e-9:
                flat = int(np.argmax(G))
                ii, jj = flat // n, flat % n
                if ii > jj:
                    ii, jj = jj, ii
                if (ii, jj) not in active_set0:
                    topo_t = t; topo_X = Xt.copy()
        # keep the line minimum and the topology-change point
        if best_Xt is not None:
            scored_starts.append((best_mt, best_Xt.copy(), f"floppy tol={tol} t={best_t:.2e}"))
        if topo_X is not None and topo_t != best_t:
            Gt = topo_X @ topo_X.T
            np.fill_diagonal(Gt, -2.0)
            mt2 = float(Gt.max())
            scored_starts.append((mt2, topo_X.copy(), f"floppy-topo tol={tol} t={topo_t:.2e}"))
    # sort by line max
    scored_starts.sort(key=lambda z: z[0])

    # polish top floppy starts with big-radius LP (+ light LSE first)
    n_floppy_polish = min(5, len(scored_starts))
    for k in range(n_floppy_polish):
        if time.time() > deadline - 35.0:
            break
        _, Xs, _ = scored_starts[k]
        try:
            # light anneal to relax
            Xa = _anneal_lse(Xs, steps=250, beta0=80.0, beta1=400.0, lr0=0.02, lr1=0.002)
        except Exception:
            Xa = Xs
        try:
            Xp, vp = _lp_polish_big(Xa, min(deadline - 20.0, time.time() + 7.0), r_init=1e-2, gap=1e-3, cap=3000)
        except Exception:
            continue
        if vp < best_val - 1e-15:
            best_val, best = vp, Xp.copy()
            # update incumbent for further drops? keep x0 as best for remaining? update
            x0, m_cur = best.copy(), best_val
            G0 = x0 @ x0.T; c0 = G0[iu]; m_cur = float(c0.max())

    # ---- systematic single-contact drops ----
    # recompute tight pairs from current best x0
    G0 = x0 @ x0.T
    c0 = G0[iu]
    m_cur = float(c0.max())
    order = np.argsort(-c0)
    # take up to 12 tight pairs spread out (every 2nd to diversify)
    drop_list = []
    for k in range(min(24, len(order))):
        if k % 2 == 0 or k < 6:
            drop_list.append((int(iu[0][order[k]]), int(iu[1][order[k]])))
        if len(drop_list) >= 12:
            break
    for di, (fi, fj) in enumerate(drop_list):
        if time.time() > deadline - 8.0:
            break
        target = m_cur - 1e-3
        try:
            # initial push apart
            Xd = x0.copy()
            # tangent separation direction
            c = float(Xd[fi] @ Xd[fj])
            gi = Xd[fj] - c * Xd[fi]
            gj = Xd[fi] - c * Xd[fj]
            ni = np.linalg.norm(gi) + 1e-30; nj = np.linalg.norm(gj) + 1e-30
            gi /= ni; gj /= nj
            push = 0.02
            Xd[fi] = _norm((Xd[fi] - push * gi)[None, :])[0]
            Xd[fj] = _norm((Xd[fj] - push * gj)[None, :])[0]
            Xa = _anneal_lse(Xd, steps=400, beta0=60.0, beta1=350.0, lr0=0.02, lr1=0.003,
                             forbid=(fi, fj), forbid_target=target, forbid_w=8.0)
            # check constraint satisfied roughly; if not, skip LP? still try
            Xp, vp = _lp_polish_big(Xa, min(deadline - 5.0, time.time() + 5.0), r_init=1e-2, gap=1e-3, cap=3000)
        except Exception:
            continue
        if vp < best_val - 1e-15:
            best_val, best = vp, Xp.copy()
            x0, m_cur = best.copy(), best_val

    # final safety: return best (incumbent unchanged if no improvement)
    return _norm(best)
