import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _norm_rows(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return x / n

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(np.max(g))

def _lp_polish(x, seconds, r0=1e-3):
    x = _norm_rows(x.copy())
    n, d = x.shape
    best = _max_cos(x)
    xb = x.copy()
    radius = float(r0)
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        c = (x @ x.T)[iu]
        m = c.max()
        sel = c > m - max(20 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        if p == 0:
            break
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(),
                               (np.repeat(np.arange(n), d), np.arange(n * d))),
                              shape=(n, nv)).tocsr()
            obj = np.zeros(nv); obj[-1] = 1.0
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
        y = _norm_rows(y)
        v = _max_cos(y)
        if v < best:
            x, best, xb = y, v, y.copy()
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return xb

def _tangent_bases(x):
    n, d = x.shape
    dm = d - 1
    Bs = np.zeros((n, d, dm))
    e1 = np.zeros(d); e1[0] = 1.0
    e2 = np.zeros(d); e2[1] = 1.0
    I = np.eye(d)
    for i in range(n):
        xi = x[i]
        e = e1 if abs(float(xi[0])) < 0.99 else e2
        u = e - xi
        nu = np.linalg.norm(u)
        if nu < 1e-12:
            # fallback: QR of random fixed matrix orthogonal to xi
            q, _ = np.linalg.qr(np.eye(d) - np.outer(xi, xi) + 1e-9 * np.eye(d))
            # first column may not be xi; build via SVD
            Bs[i] = q[:, 1:dm+1] if q.shape[1] > dm else q[:, :dm]
            continue
        u = u / nu
        H = I - 2.0 * np.outer(u, u)
        # first column of H is xi (H e = xi), rest spans tangent
        # H[:,0] should equal xi up to sign; take columns 1:
        Bs[i] = H[:, 1:]
        # re-orthogonalize against xi numerically
        # Bs should satisfy xi @ Bs ~ 0
    return Bs

def solve():
    t0 = time.time()
    DEADLINE = 172.0
    n, d, dm = 55, 21, 20
    # load incumbents deterministically
    best = None
    best_val = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.asarray(np.load(fn), dtype=float)
            if a.shape == (55, 21):
                a = _norm_rows(a)
                v = _max_cos(a)
                if v < best_val:
                    best_val = v
                    best = a.copy()
        except Exception:
            continue
    if best is None:
        rng = np.random.RandomState(0)
        best = _norm_rows(rng.randn(n, d))
        best_val = _max_cos(best)
    x0 = best.copy()
    x0_val = best_val

    try:
        # active set within 1e-7 of max
        G = x0 @ x0.T
        iu = np.triu_indices(n, 1)
        c_all = G[iu]
        m0 = float(c_all.max())
        thr = m0 - 1e-7
        sel_idx = np.where(c_all > thr)[0]
        # cap / ensure size for cheap SVD
        if len(sel_idx) > 1200:
            order = np.argsort(-c_all[sel_idx])[:1200]
            sel_idx = sel_idx[order]
        if len(sel_idx) < 60:
            # widen to 1e-6 to have enough contacts
            sel_idx = np.where(c_all > m0 - 1e-6)[0]
            if len(sel_idx) > 1200:
                order = np.argsort(-c_all[sel_idx])[:1200]
                sel_idx = sel_idx[order]
        I_act = iu[0][sel_idx]
        J_act = iu[1][sel_idx]
        p = len(sel_idx)
        c_sel = c_all[sel_idx]
        if p < 20:
            return _norm_rows(x0)

        Bs = _tangent_bases(x0)  # n x d x dm
        N = n * dm
        Jmat = np.zeros((p, N), dtype=float)
        # build contact Jacobian
        for k in range(p):
            i = int(I_act[k]); j = int(J_act[k])
            # x_j^T B_i and x_i^T B_j
            Jmat[k, i*dm:(i+1)*dm] = Bs[i].T @ x0[j]
            Jmat[k, j*dm:(j+1)*dm] = Bs[j].T @ x0[i]

        # regularised least-squares stresses: (J J^T + reg I) w = 1
        K = Jmat @ Jmat.T
        diag_mean = float(np.trace(K) / max(p, 1))
        reg = max(1e-9, 1e-7 * max(diag_mean, 1e-12))
        try:
            w = np.linalg.solve(K + reg * np.eye(p), np.ones(p))
        except Exception:
            try:
                w, *_ = np.linalg.lstsq(K + reg * np.eye(p), np.ones(p), rcond=None)
            except Exception:
                return _norm_rows(x0)
        if not np.all(np.isfinite(w)):
            return _norm_rows(x0)
        # point stress sums (use positive part for hub detection, keep raw for ranking)
        wpos = np.maximum(w, 0.0)
        if wpos.sum() <= 0:
            wpos = np.abs(w)
        pt_stress = np.zeros(n)
        np.add.at(pt_stress, I_act, wpos)
        np.add.at(pt_stress, J_act, wpos)
        hubs = np.argsort(-pt_stress)
        # keep top 6 hubs
        hubs = hubs[:6]
        # hub subsets: singletons + pairs + triples (deterministic, capped)
        subsets = []
        for h in hubs[:4]:
            subsets.append([int(h)])
        if len(hubs) >= 2:
            subsets.append([int(hubs[0]), int(hubs[1])])
            subsets.append([int(hubs[1]), int(hubs[2])])
        if len(hubs) >= 3:
            subsets.append([int(hubs[0]), int(hubs[1]), int(hubs[2])])
        if len(hubs) >= 4:
            subsets.append([int(hubs[0]), int(hubs[1]), int(hubs[2]), int(hubs[3])])
        # dedup
        seen = set(); uniq = []
        for s in subsets:
            t = tuple(sorted(s))
            if t not in seen:
                seen.add(t); uniq.append(list(t))
        subsets = uniq[:8]

        # order contacts by stress descending
        contact_order = np.argsort(-w)
        best_cand = None
        best_cand_val = x0_val

        # logarithmic step sizes
        steps = np.logspace(-4.0, -0.30, 20)  # 1e-4 .. ~0.5

        for sset in subsets:
            if time.time() - t0 > DEADLINE - 25:
                break
            sset_f = set(sset)
            # incident mask
            inc = np.array([(int(I_act[k]) in sset_f or int(J_act[k]) in sset_f) for k in range(p)])
            inc_idx = np.where(inc)[0]
            if len(inc_idx) == 0:
                continue
            # top 20-40 high-stress among incident
            # sort incident by w descending
            o = inc_idx[np.argsort(-w[inc_idx])]
            Kdel = 30
            if len(o) < 20:
                Kdel = len(o)
            else:
                Kdel = min(40, len(o))
                Kdel = max(20, Kdel)
                # use 30 if available
                if len(o) >= 30:
                    Kdel = 30
            del_idx = o[:Kdel]
            keep_mask = np.ones(p, dtype=bool)
            keep_mask[del_idx] = False
            J_rem = Jmat[keep_mask]
            J_del = Jmat[del_idx]
            m_del = J_del.shape[0]
            if m_del == 0 or J_rem.shape[0] == 0:
                continue
            # steepest descent for deleted set
            g = -np.mean(J_del, axis=0)  # N
            gn = np.linalg.norm(g)
            if gn < 1e-14 or not np.isfinite(gn):
                continue
            # economy SVD of remaining Jacobian (capped)
            try:
                U, S, Vt = np.linalg.svd(J_rem, full_matrices=False)
            except Exception:
                continue
            if not np.all(np.isfinite(S)):
                continue
            smax = float(S.max()) if S.size else 0.0
            tol = max(smax * 1e-10, 1e-12)
            r = int(np.sum(S > tol))
            if r > 0 and r < Vt.shape[0]:
                Vr = Vt[:r, :]
                proj = Vr @ g
                u = g - Vr.T @ proj
            elif r == 0:
                u = g.copy()
            else:
                # full rank: projection removes everything; use smallest singular vector
                u = Vt[-1, :].copy()
                if float(np.mean(J_del @ u)) > 0:
                    u = -u
            un = float(np.linalg.norm(u))
            if not np.isfinite(un) or un < 1e-12:
                # fallback to last singular vector oriented to decrease deleted
                try:
                    u = Vt[-1, :].copy()
                    if float(np.mean(J_del @ u)) > 0:
                        u = -u
                    un = float(np.linalg.norm(u))
                except Exception:
                    continue
                if not np.isfinite(un) or un < 1e-12:
                    continue
            u = u / un
            # ambient direction
            Umat = u.reshape(n, dm)  # per-point tangent coeffs
            DX = np.zeros((n, d))
            for i in range(n):
                DX[i] = Bs[i] @ Umat[i]
            # DX should be tangent; remove residual normal component
            DX = DX - (np.sum(DX * x0, axis=1, keepdims=True)) * x0
            dn = float(np.linalg.norm(DX))
            if not np.isfinite(dn) or dn < 1e-14:
                continue
            # line search after retraction
            for t in steps:
                y = x0 + float(t) * DX
                y = _norm_rows(y)
                v = _max_cos(y)
                if v < best_cand_val - 1e-12:
                    best_cand_val = v
                    best_cand = y.copy()
            # also try a couple of larger/smaller extremes deterministically
            # (already covered by logspace)
        if best_cand is not None and best_cand_val < x0_val - 1e-12:
            # final short LP polish only for accepted improvement
            remain = DEADLINE - (time.time() - t0)
            if remain > 3.0:
                secs = min(10.0, remain - 1.0)
                try:
                    pol = _lp_polish(best_cand, seconds=secs, r0=1e-3)
                    pv = _max_cos(pol)
                    if pv < best_cand_val:
                        best_cand = pol
                        best_cand_val = pv
                except Exception:
                    pass
            return _norm_rows(best_cand)
        else:
            return _norm_rows(x0)
    except Exception:
        try:
            return _norm_rows(x0)
        except Exception:
            rng = np.random.RandomState(0)
            return _norm_rows(rng.randn(55, 21))
