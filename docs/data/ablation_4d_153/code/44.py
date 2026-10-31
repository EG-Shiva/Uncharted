import time
import numpy as np

def _norm(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return x / n

def _max_cos(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def _lp_polish(x, deadline, cap=2500, r0=1e-3):
    from scipy.optimize import linprog
    from scipy.sparse import coo_matrix
    x = _norm(x)
    n, d = x.shape
    best = _max_cos(x)
    best_x = x.copy()
    radius = r0
    iu0 = np.triu_indices(n, 1)
    it = 0
    while time.time() < deadline and radius > 1e-14 and it < 300:
        it += 1
        G = x @ x.T
        c = G[iu0]
        m = float(c.max())
        thresh = m - max(20.0 * radius, 1e-9)
        idx = np.where(c > thresh)[0]
        if idx.size == 0:
            break
        if idx.size > cap:
            order = np.argpartition(c[idx], -cap)[-cap:]
            idx = idx[order]
        I = iu0[0][idx]; J = iu0[1][idx]
        csel = c[idx]
        p = idx.size
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cI = I[:, None] * d + np.arange(d)
        cJ = J[:, None] * d + np.arange(d)
        cols = np.concatenate([cI, cJ, np.full((p, 1), n * d)], axis=1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], axis=1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y = _norm(y)
        v = _max_cos(y)
        if v < best:
            x = y; best = v; best_x = y.copy()
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return best_x, best

def _hash_key(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    m = float(np.max(G))
    B = G > (m - 5e-7)
    deg = np.sum(B, axis=1)
    return (tuple(sorted(deg.tolist())), round(float(m) * 1e7))

def solve():
    t0 = time.time()
    HARD_END = t0 + 174.0
    FLIP_END = t0 + 120.0
    n, d = 153, 4
    best_x = None
    best_v = 1e9
    for fname in ("best.npy", "record.npy"):
        try:
            a = np.load(fname)
            a = np.asarray(a, dtype=float)
            if a.shape == (n, d):
                a = _norm(a)
                if np.all(np.isfinite(a)):
                    v = _max_cos(a)
                    if v < best_v:
                        best_v = v
                        best_x = a.copy()
        except Exception:
            pass
    if best_x is None:
        rng0 = np.random.default_rng(0)
        best_x = _norm(rng0.standard_normal(size=(n, d)))
        best_v = _max_cos(best_x)
    global_best_x = best_x.copy()
    global_best_v = best_v
    seen = set()
    try:
        seen.add(_hash_key(global_best_x))
    except Exception:
        pass

    X0 = global_best_x.copy()
    m0 = global_best_v

    # --- build tangent bases ---
    try:
        Ntan = n * 3
        Ts = np.zeros((n, 4, 3))
        for i in range(n):
            xi = X0[i]
            # null space of xi (1x4) via svd
            _, _, Vh = np.linalg.svd(xi.reshape(1, 4))
            Ti = Vh[1:, :].T  # 4x3
            # ensure orthonormal (svd already)
            Ts[i] = Ti
        # contacts within 1e-9
        G0 = X0 @ X0.T
        iu_all = np.triu_indices(n, 1)
        c_all = G0[iu_all]
        m_ref = float(np.max(c_all[np.isfinite(c_all)])) if False else m0
        # recompute with diag excluded
        Gtmp = G0.copy()
        np.fill_diagonal(Gtmp, -2.0)
        m_ref = float(np.max(Gtmp))
        slack_all = m_ref - c_all
        contact_mask = slack_all < 1e-9
        cI = iu_all[0][contact_mask]
        cJ = iu_all[1][contact_mask]
        p = cI.size
        if p < 10 or p > 3000:
            raise ValueError("bad p")
        deg = np.zeros(n, dtype=int)
        for a_, b_ in zip(cI, cJ):
            deg[a_] += 1
            deg[b_] += 1
        # Jacobian p x Ntan
        J = np.zeros((p, Ntan))
        for k in range(p):
            i = int(cI[k]); j = int(cJ[k])
            Ti = Ts[i]; Tj = Ts[j]
            J[k, i*3:(i+1)*3] = Ti.T @ X0[j]
            J[k, j*3:(j+1)*3] = Tj.T @ X0[i]
        # global rotation basis Q : Ntan x 6
        pairs = [(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)]
        Q = np.zeros((Ntan, 6))
        for qk, (a_, b_) in enumerate(pairs):
            for i in range(n):
                xi = X0[i]
                y = np.zeros(4)
                y[a_] = xi[b_]
                y[b_] = -xi[a_]
                Q[i*3:(i+1)*3, qk] = Ts[i].T @ y
        Q, _ = np.linalg.qr(Q)
        # dual weights via NNLS
        try:
            from scipy.optimize import nnls
            pen = 1000.0
            sp = np.sqrt(pen)
            A = np.vstack([J.T, sp * np.ones((1, p))])
            b = np.zeros(Ntan + 1); b[-1] = sp
            w, _ = nnls(A, b)
            if np.sum(w) <= 0 or not np.all(np.isfinite(w)):
                raise ValueError("nnls bad")
            w = w / max(1e-12, np.sum(w))
        except Exception:
            w = np.ones(p) / max(1, p)
        # rank contacts: small weight first, boost deg 3/4
        rank_score = np.zeros(p)
        for k in range(p):
            i = int(cI[k]); j = int(cJ[k])
            md = deg[i] if deg[i] < deg[j] else deg[j]
            bonus = 0.0
            if md == 3:
                bonus = 0.005
            elif md == 4:
                bonus = 0.002
            rank_score[k] = float(w[k]) - bonus
        order_e = np.argsort(rank_score)
        E_best = order_e[:min(20, p)]
        # build flip tasks
        tasks = []  # (e_pos, f_i, f_j, slack_f)
        for epos in E_best:
            ek = int(epos)
            ei = int(cI[ek]); ej = int(cJ[ek])
            # non-contact pairs sharing vertex with e, slack <2e-4
            # scan all pairs
            cand_f = []
            for idx in range(c_all.size):
                s = float(slack_all[idx])
                if s < 1e-9 or s >= 2e-4:
                    continue
                a_ = int(iu_all[0][idx]); b_ = int(iu_all[1][idx])
                if a_ == ei or b_ == ei or a_ == ej or b_ == ej:
                    cand_f.append((s, a_, b_))
            cand_f.sort(key=lambda t: t[0])
            for (s, a_, b_) in cand_f[:3]:
                tasks.append((ek, a_, b_, s))
        # sort tasks by weight rank then slack
        # map ek to rank
        rank_of = {int(e): r for r, e in enumerate(E_best)}
        tasks.sort(key=lambda t: (rank_of.get(int(t[0]), 99), t[3]))

        # cache mechanisms per e
        mech_cache = {}
        def get_mech(ek):
            if ek in mech_cache:
                return mech_cache[ek]
            mask = np.ones(p, dtype=bool); mask[ek] = False
            Jr = J[mask]  # (p-1) x N
            Gg = Jr.T @ Jr  # N x N
            # projector
            # P G P + lam QQt
            # compute PG = G - Q(Q^T G) etc efficiently
            QtG = Q.T @ Gg  # 6 x N
            # PGP = G - Q QtG - (QtG)^T Q^T? Let's compute directly with matrices (459 ok)
            P = np.eye(Ntan) - Q @ Q.T
            PGP = P @ Gg @ P
            tr = float(np.trace(Gg)) / max(1, Ntan)
            lam = 10.0 * (tr + 1.0)
            K = PGP + lam * (Q @ Q.T)
            # eigh smallest
            vals, vecs = np.linalg.eigh(K)
            v = vecs[:, 0]
            # ensure orthogonal to Q (should be)
            v = v - Q @ (Q.T @ v)
            nv = float(np.linalg.norm(v))
            if nv < 1e-12:
                return None
            v = v / nv
            mech_cache[ek] = v
            return v

        # precompute Dx per e after orientation? orientation per f
        for (ek, fi, fj, s_f) in tasks:
            if time.time() > FLIP_END - 1.0:
                break
            v = get_mech(int(ek))
            if v is None:
                continue
            # gradient row for f
            Ti = Ts[fi]; Tj = Ts[fj]
            gf = np.zeros(Ntan)
            gf[fi*3:(fi+1)*3] = Ti.T @ X0[fj]
            gf[fj*3:(fj+1)*3] = Tj.T @ X0[fi]
            dot = float(gf @ v)
            if abs(dot) < 2e-3:
                continue
            sgn = -1.0 if dot > 0 else 1.0
            vv = sgn * v
            # tangent displacements in R4
            Dx = np.zeros((n, 4))
            for i in range(n):
                Dx[i] = Ts[i] @ vv[i*3:(i+1)*3]
            # projected line search: incremental 0.002 up to 0.03
            cf0 = float(X0[fi] @ X0[fj])
            t_cur = 0.0
            cur_X = X0
            cur_cf = cf0
            feasible = False
            step_inc = 0.002
            for _ in range(15):
                if time.time() > FLIP_END - 0.9:
                    break
                t_nxt = t_cur + step_inc
                if t_nxt > 0.0301:
                    break
                Xn = X0 + t_nxt * Dx
                Xn = _norm(Xn)
                # fast max
                Gn = Xn @ Xn.T
                np.fill_diagonal(Gn, -2.0)
                m_nxt = float(np.max(Gn))
                tol = 5.0 * t_nxt * t_nxt + 2e-9
                if m_nxt > m_ref + tol:
                    break
                cfn = float(Xn[fi] @ Xn[fj])
                if not (cfn < cur_cf - 1e-9):
                    break
                t_cur = t_nxt
                cur_X = Xn
                cur_cf = cfn
                feasible = True
                if (m_ref - cfn) < 1e-9:
                    break
            if not feasible or t_cur < 0.0009:
                continue
            # polish
            dl = min(time.time() + 0.9, FLIP_END)
            if dl <= time.time() + 0.05:
                break
            try:
                Yp, vp = _lp_polish(cur_X, dl, cap=2000, r0=8e-4)
            except Exception:
                continue
            try:
                hk = _hash_key(Yp)
            except Exception:
                continue
            if hk in seen:
                # only keep if improves
                if vp < global_best_v - 1e-12:
                    global_best_v = vp
                    global_best_x = Yp.copy()
                    seen.add(hk)
            else:
                seen.add(hk)
                if vp < global_best_v:
                    global_best_v = vp
                    global_best_x = Yp.copy()
                # keep diverse archive implicitly via seen; do not degrade global best
            # early exit if clearly improved a lot? continue to use time
    except Exception:
        pass

    # final full polish on global best with remaining time
    try:
        dl = HARD_END - 1.0
        if dl > time.time():
            global_best_x, global_best_v = _lp_polish(global_best_x, dl, cap=3000, r0=1e-3)
    except Exception:
        pass
    global_best_x = _norm(global_best_x)
    return global_best_x
