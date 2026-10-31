import time
import numpy as np

def _norm_rows(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return x / n

def _max_cos(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def _tangent_bases(x):
    n = x.shape[0]
    T = np.empty((n, 4, 3), dtype=np.float64)
    for i in range(n):
        xi = x[i]
        # pick 3 axes least aligned, orthogonalize
        # use QR of 4x4 [xi | I] trick: build basis via SVD of projector
        # simple Gram-Schmidt on standard basis
        order = np.argsort(np.abs(xi))  # smallest first
        vecs = []
        for a in order:
            e = np.zeros(4)
            e[a] = 1.0
            # remove xi component
            e = e - float(np.dot(e, xi)) * xi
            for v in vecs:
                e = e - float(np.dot(e, v)) * v
            nv = float(np.linalg.norm(e))
            if nv > 1e-8:
                e /= nv
                vecs.append(e)
            if len(vecs) == 3:
                break
        # fallback
        while len(vecs) < 3:
            v = np.random.randn(4)
            v = v - float(np.dot(v, xi)) * xi
            for vv in vecs:
                v = v - float(np.dot(v, vv)) * vv
            nv = float(np.linalg.norm(v))
            if nv < 1e-8:
                continue
            vecs.append(v / nv)
        T[i, :, 0] = vecs[0]
        T[i, :, 1] = vecs[1]
        T[i, :, 2] = vecs[2]
    return T

def _global_Q(x, T):
    n = x.shape[0]
    m = 3 * n
    Qraw = np.zeros((m, 6), dtype=np.float64)
    col = 0
    for a in range(4):
        for b in range(a + 1, 4):
            # y_i[a]=x_i[b], y_i[b]=-x_i[a]
            for i in range(n):
                xi = x[i]
                y = np.zeros(4)
                y[a] = xi[b]
                y[b] = -xi[a]
                # y already tangent; coordinates in T
                q = T[i].T @ y  # 3
                Qraw[i * 3:(i + 1) * 3, col] = q
            col += 1
    # orthonormalize
    try:
        Q, _ = np.linalg.qr(Qraw)
    except Exception:
        Q, _ = np.linalg.qr(Qraw + 1e-12 * np.random.randn(*Qraw.shape))
    return Q  # m x 6 orthonormal

def _contact_J(x, T, I, J):
    n = x.shape[0]
    p = I.shape[0]
    m = 3 * n
    Jac = np.zeros((p, m), dtype=np.float64)
    # vectorized per contact loop (p~450 fine)
    for k in range(p):
        i = int(I[k]); j = int(J[k])
        vi = T[i].T @ x[j]  # 3
        vj = T[j].T @ x[i]
        Jac[k, i * 3:(i + 1) * 3] = vi
        Jac[k, j * 3:(j + 1) * 3] = vj
    return Jac

def _lp_polish(x, deadline, r0=0.003, cap=2500, maxit=120):
    from scipy.optimize import linprog
    from scipy.sparse import coo_matrix
    x = _norm_rows(x.copy())
    n, d = x.shape
    try:
        best = _max_cos(x)
    except Exception:
        return x, 1e9
    iu0 = np.triu_indices(n, 1)
    radius = float(r0)
    it = 0
    while time.time() < deadline and radius > 1e-14 and it < maxit:
        it += 1
        G = x @ x.T
        c = G[iu0]
        m = float(np.max(c))
        thr = m - max(20.0 * radius, 1e-9)
        idx = np.where(c > thr)[0]
        if idx.size == 0:
            radius *= 0.5
            continue
        if idx.size > cap:
            part = np.argpartition(c[idx], -cap)[-cap:]
            idx = idx[part]
        I = iu0[0][idx]; J = iu0[1][idx]
        csel = c[idx]
        p = idx.size
        nv = n * d + 1
        try:
            rows = np.repeat(np.arange(p), 2 * d + 1)
            cI = I[:, None] * d + np.arange(d)
            cJ = J[:, None] * d + np.arange(d)
            cols = np.concatenate([cI, cJ, np.full((p, 1), n * d)], axis=1).ravel()
            vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], axis=1).ravel()
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
        try:
            dx = res.x[:-1].reshape(n, d)
        except Exception:
            radius *= 0.5
            continue
        y = _norm_rows(x + dx)
        try:
            v = _max_cos(y)
        except Exception:
            radius *= 0.5
            continue
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x, best

def _deg_hash(x, tol=5e-7):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    m = float(np.max(G))
    B = G > (m - tol)
    deg = np.sum(B, axis=1)
    return (tuple(sorted(deg.tolist())), round(float(m) * 1e7)), deg, m

def solve():
    t0 = time.time()
    HARD_END = t0 + 172.0
    FLIP_END = t0 + 120.0
    n, d = 153, 4
    rng = np.random.default_rng(7)
    best_x = None; best_v = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=np.float64)
            if a.shape == (n, d):
                a = _norm_rows(a)
                v = _max_cos(a)
                if v < best_v:
                    best_v = v; best_x = a.copy()
        except Exception:
            pass
    if best_x is None:
        best_x = _norm_rows(rng.standard_normal(size=(n, d)))
        best_v = _max_cos(best_x)
    global_x = best_x.copy(); global_v = best_v
    # quick initial polish to get true local optimum
    try:
        global_x, global_v = _lp_polish(global_x, min(HARD_END - 5.0, t0 + 12.0), r0=0.003)
    except Exception:
        pass
    base_x = global_x.copy(); base_v = global_v
    seen = set()
    try:
        hk0, _, _ = _deg_hash(base_x)
        seen.add(hk0)
    except Exception:
        pass

    try:
        # ---- build active contacts within 1e-9 ----
        G0 = base_x @ base_x.T
        iu_all = np.triu_indices(n, 1)
        c_all = G0[iu_all]
        m0 = float(np.max(c_all))
        ACT_TOL = 1e-9
        act_mask = c_all > (m0 - ACT_TOL)
        act_idx = np.where(act_mask)[0]
        Iact = iu_all[0][act_idx]; Jact = iu_all[1][act_idx]
        p_act = act_idx.size
        if p_act < 10 or p_act > 3000:
            raise ValueError("bad active count")
        # degrees from active set
        deg = np.zeros(n, dtype=int)
        np.add.at(deg, Iact, 1)
        np.add.at(deg, Jact, 1)
        # tangent bases + global rotations + Jacobian
        T = _tangent_bases(base_x)
        Q = _global_Q(base_x, T)  # 3n x 6
        Jfull = _contact_J(base_x, T, Iact, Jact)  # p x 3n
        # ---- dual weights via NNLS: min ||J^T w|| s.t. w>=0, sum=1 ----
        try:
            from scipy.optimize import nnls
            Aeq = Jfull.T  # 3n x p
            alpha = 1.0
            M = np.vstack([Aeq, alpha * np.ones((1, p))])
            b = np.zeros(M.shape[0]); b[-1] = alpha
            # scale for stability
            w, _ = nnls(M, b, maxiter=10 * p)
            if np.sum(w) <= 1e-12 or np.any(~np.isfinite(w)):
                raise ValueError("nnls degenerate")
            w = w / max(1e-300, float(np.sum(w)))
        except Exception:
            w = np.ones(p_act) / max(1, p_act)
        # rank contacts: prefer degree 3/4, small weight
        order = []
        for k in range(p_act):
            i = int(Iact[k]); j = int(Jact[k])
            md = int(min(deg[i], deg[j]))
            pri = 0 if md <= 4 else 1
            order.append((pri, float(w[k]), k))
        order.sort(key=lambda t: (t[0], t[1]))
        weak_ks = [t[2] for t in order[:20]]
        # candidate f pairs: slack < 2e-4 sharing vertex with e
        SLACK_F = 2e-4
        # precompute slack
        slack = m0 - c_all
        # map vertex -> list of pair positions? just brute per e
        combos = []  # (e_rank, slack_f, e_k, f_pos)
        for erank, ek in enumerate(weak_ks):
            ei = int(Iact[ek]); ej = int(Jact[ek])
            # find non-contact pairs sharing ei or ej
            # brute over all pairs
            for pos in range(c_all.size):
                s = float(slack[pos])
                if s <= ACT_TOL or s >= SLACK_F:
                    continue
                a = int(iu_all[0][pos]); b = int(iu_all[1][pos])
                if a == ei or b == ei or a == ej or b == ej:
                    combos.append((erank, s, ek, pos))
        # keep at most 5 closest f per e, then global cap
        from collections import defaultdict
        by_e = defaultdict(list)
        for tup in combos:
            by_e[tup[2]].append(tup)
        filt = []
        for ek, lst in by_e.items():
            lst.sort(key=lambda t: t[1])
            filt.extend(lst[:6])
        # sort globally by e_rank then slack
        filt.sort(key=lambda t: (t[0], t[1]))
        # cap to ~80 attempts
        filt = filt[:80]

        m3n = 3 * n
        # precompute row for f quickly via T? will compute per combo
        for (erank, sf, ek, fpos) in filt:
            if time.time() > FLIP_END:
                break
            try:
                ei = int(Iact[ek]); ej = int(Jact[ek])
                fa = int(iu_all[0][fpos]); fb = int(iu_all[1][fpos])
                # remaining Jacobian without e
                keep = np.ones(p_act, dtype=bool); keep[ek] = False
                Jrem = Jfull[keep]  # (p-1) x 3n
                # augmented with global rotations penalty
                beta = 50.0
                Jaug = np.vstack([Jrem, beta * Q.T])  # (p+5) x 3n
                # smallest right singular vector
                try:
                    _, _, Vh = np.linalg.svd(Jaug, full_matrices=False)
                except Exception:
                    continue
                v = Vh[-1].copy()  # 3n
                # ensure orthogonal-ish to Q (should be); project residual
                # remove Q component explicitly
                coeff = Q.T @ v
                v = v - Q @ coeff
                nv = float(np.linalg.norm(v))
                if nv < 1e-12:
                    continue
                v /= nv
                # row for f
                rowf = np.zeros(m3n)
                rowf[fa * 3:(fa + 1) * 3] = T[fa].T @ base_x[fb]
                rowf[fb * 3:(fb + 1) * 3] = T[fb].T @ base_x[fa]
                df = float(rowf @ v)
                if abs(df) < 1e-12:
                    continue
                # orient so linearised cosine of f INCREASES (toward active)
                if df < 0:
                    v = -v; df = -df
                # line search walk 0.001..0.03
                # incremental deltas
                delta = 0.002
                nsteps = 15  # up to 0.03
                cur_alpha = 0.0
                best_alpha = 0.0
                # cache per-point direction
                Vmat = v.reshape(n, 3)  # rows per point
                # ambient directions D_i = T_i @ Vmat_i
                D = np.einsum('nij,nj->ni', T, Vmat)  # n x 4
                feas_tol = 5e-05
                # evaluate stepwise
                cf0 = float(base_x[fa] @ base_x[fb])
                last_feas_alpha = 0.0
                last_cf = cf0
                alpha = 0.0
                feasible_found = False
                for s_idx in range(nsteps):
                    if time.time() > FLIP_END:
                        break
                    alpha += delta
                    if alpha > 0.031:
                        break
                    Y = base_x + alpha * D
                    Y = _norm_rows(Y)
                    # compute max and f value
                    # full Gram for max (cheap)
                    G = Y @ Y.T
                    np.fill_diagonal(G, -2.0)
                    my = float(np.max(G))
                    cf = float(Y[fa] @ Y[fb])
                    # "any other cosine above current maximum by more than step tolerance"
                    # check max over pairs != f
                    # approximate by my but if argmax is f, allow it to reach m0
                    # find max excluding f: cheap: if my corresponds to f, second max?
                    # simpler: compute max_excl
                    # to avoid O(n^2) argmax twice, just compute:
                    if cf >= my - 1e-12:
                        # f is the max; need second max
                        # mask f entry
                        G[fa, fb] = -2.0; G[fb, fa] = -2.0
                        m_excl = float(np.max(G))
                    else:
                        m_excl = my
                    if m_excl > m0 + feas_tol:
                        break  # no further feasible
                    # feasible
                    last_feas_alpha = alpha
                    last_cf = cf
                    feasible_found = True
                    if cf >= m0 - 1e-9:
                        break  # reached active level
                if not feasible_found or last_feas_alpha < 1e-12:
                    continue
                # require some movement of f toward active (slack reduced)
                if last_cf <= cf0 + 1e-7:
                    continue
                Yfin = _norm_rows(base_x + last_feas_alpha * D)
                # polish
                t_pol = min(time.time() + 2.5, FLIP_END + 2.0, HARD_END - 8.0)
                if t_pol - time.time() < 0.4:
                    continue
                Yp, vp = _lp_polish(Yfin, t_pol, r0=0.004)
                try:
                    hk, _, _ = _deg_hash(Yp)
                except Exception:
                    continue
                is_new_hash = (hk not in seen)
                if vp < global_v - 1e-12:
                    global_x = Yp.copy(); global_v = float(vp)
                    base_x = Yp.copy(); base_v = float(vp)
                    seen.add(hk)
                    # refresh structures from new base for remaining flips?
                    # recompute lightweight: m0,T,Q,Jfull,w,combos? to keep time, only refresh if big gain
                    # do a light refresh of m0/T/Q/Jfull for subsequent trials
                    try:
                        G0 = base_x @ base_x.T
                        c_all = G0[iu_all]
                        m0 = float(np.max(c_all))
                        act_mask = c_all > (m0 - ACT_TOL)
                        act_idx = np.where(act_mask)[0]
                        Iact = iu_all[0][act_idx]; Jact = iu_all[1][act_idx]
                        p_act = act_idx.size
                        if p_act >= 10 and p_act <= 3000:
                            T = _tangent_bases(base_x)
                            Q = _global_Q(base_x, T)
                            Jfull = _contact_J(base_x, T, Iact, Jact)
                            deg = np.zeros(n, dtype=int)
                            np.add.at(deg, Iact, 1)
                            np.add.at(deg, Jact, 1)
                            slack = m0 - c_all
                            # note: weak_ks/combos not refreshed to save time; continue with updated J for mechanism
                    except Exception:
                        pass
                elif is_new_hash and vp < global_v + 2e-04:
                    seen.add(hk)
                    # keep diverse but don't switch base; optionally try polishing diverse point later
                    pass
                # else discard
            except Exception:
                continue
    except Exception:
        pass
    # final polish of best with remaining time
    try:
        t_left = HARD_END - time.time()
        if t_left > 2.0:
            global_x, global_v = _lp_polish(global_x, HARD_END - 1.0, r0=0.003)
    except Exception:
        pass
    global_x = _norm_rows(global_x)
    return global_x
