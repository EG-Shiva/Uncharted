import time
import numpy as np
from collections import defaultdict

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
        k = int(np.argmax(np.abs(xi)))
        axes = [a for a in range(4) if a != k]
        V = np.zeros((4, 3), dtype=np.float64)
        for c, a in enumerate(axes):
            e = np.zeros(4)
            e[a] = 1.0
            V[:, c] = e - float(xi[a]) * xi
        Q, _ = np.linalg.qr(V)
        # ensure tangent (qr sign fixes)
        T[i] = Q
    return T

def _global_Q(x, T):
    n = x.shape[0]
    m = 3 * n
    Qraw = np.zeros((m, 6), dtype=np.float64)
    col = 0
    for a in range(4):
        for b in range(a + 1, 4):
            for i in range(n):
                xi = x[i]
                y = np.zeros(4)
                y[a] = xi[b]
                y[b] = -xi[a]
                Qraw[i * 3:(i + 1) * 3, col] = T[i].T @ y
            col += 1
    Q, _ = np.linalg.qr(Qraw)
    return Q

def _contact_J(x, T, I, J):
    p = I.shape[0]
    m = 3 * x.shape[0]
    Jac = np.zeros((p, m), dtype=np.float64)
    for k in range(p):
        i = int(I[k]); j = int(J[k])
        Jac[k, i * 3:(i + 1) * 3] = T[i].T @ x[j]
        Jac[k, j * 3:(j + 1) * 3] = T[j].T @ x[i]
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
            raise ValueError("bad active")
        deg = np.zeros(n, dtype=int)
        np.add.at(deg, Iact, 1)
        np.add.at(deg, Jact, 1)
        T = _tangent_bases(base_x)
        Q = _global_Q(base_x, T)
        Jfull = _contact_J(base_x, T, Iact, Jact)
        # dual weights via NNLS
        try:
            from scipy.optimize import nnls
            Aeq = Jfull.T
            alpha = 1.0
            M = np.vstack([Aeq, alpha * np.ones((1, p_act))])
            b = np.zeros(M.shape[0]); b[-1] = alpha
            w, _ = nnls(M, b, maxiter=10 * p_act)
            if np.sum(w) <= 1e-12 or np.any(~np.isfinite(w)):
                raise ValueError("nnls bad")
            w = w / max(1e-300, float(np.sum(w)))
        except Exception:
            w = np.ones(p_act) / max(1, p_act)
        order = []
        for k in range(p_act):
            i = int(Iact[k]); j = int(Jact[k])
            md = int(min(deg[i], deg[j]))
            pri = 0 if md <= 4 else 1
            order.append((pri, float(w[k]), k))
        order.sort(key=lambda t: (t[0], t[1]))
        weak_ks = [t[2] for t in order[:20]]
        SLACK_F = 2e-4
        slack = m0 - c_all
        combos = []
        for erank, ek in enumerate(weak_ks):
            ei = int(Iact[ek]); ej = int(Jact[ek])
            for pos in range(c_all.size):
                s = float(slack[pos])
                if s <= ACT_TOL or s >= SLACK_F:
                    continue
                a = int(iu_all[0][pos]); bb = int(iu_all[1][pos])
                if a == ei or bb == ei or a == ej or bb == ej:
                    combos.append((erank, s, ek, pos))
        by_e = defaultdict(list)
        for tup in combos:
            by_e[tup[2]].append(tup)
        filt = []
        for ek, lst in by_e.items():
            lst.sort(key=lambda t: t[1])
            filt.extend(lst[:6])
        filt.sort(key=lambda t: (t[0], t[1]))
        filt = filt[:80]
        m3n = 3 * n
        beta = 50.0
        Qt = Q.T.copy()
        alphas = [0.001, 0.002, 0.004, 0.007, 0.01, 0.015, 0.02, 0.03]
        for (erank, sf, ek, fpos) in filt:
            if time.time() > FLIP_END:
                break
            try:
                fa = int(iu_all[0][fpos]); fb = int(iu_all[1][fpos])
                keep = np.ones(p_act, dtype=bool); keep[ek] = False
                Jrem = Jfull[keep]
                Jaug = np.vstack([Jrem, beta * Qt])
                Gram = Jaug.T @ Jaug
                try:
                    vals_e, vecs_e = np.linalg.eigh(Gram)
                except Exception:
                    continue
                v = vecs_e[:, 0].copy()
                coeff = Q.T @ v
                v = v - Q @ coeff
                nv = float(np.linalg.norm(v))
                if nv < 1e-12 or not np.isfinite(nv):
                    continue
                v /= nv
                rowf = np.zeros(m3n)
                rowf[fa * 3:(fa + 1) * 3] = T[fa].T @ base_x[fb]
                rowf[fb * 3:(fb + 1) * 3] = T[fb].T @ base_x[fa]
                df = float(rowf @ v)
                if abs(df) < 1e-12 or not np.isfinite(df):
                    continue
                if df < 0:
                    v = -v; df = -df
                Vmat = v.reshape(n, 3)
                D = np.einsum('nij,nj->ni', T, Vmat)
                feas_tol = 5e-05
                best_alpha = 0.0
                best_cf = float(base_x[fa] @ base_x[fb])
                found = False
                for al in alphas:
                    if time.time() > FLIP_END:
                        break
                    Y = _norm_rows(base_x + al * D)
                    G = Y @ Y.T
                    np.fill_diagonal(G, -2.0)
                    my = float(np.max(G))
                    cf = float(Y[fa] @ Y[fb])
                    if my <= m0 + feas_tol and cf > best_cf:
                        best_cf = cf
                        best_alpha = al
                        found = True
                if not found or best_alpha == 0.0:
                    continue
                Y0 = _norm_rows(base_x + best_alpha * D)
                # quick reject if f did not close meaningfully
                if best_cf < m0 - 2e-4 and (best_cf - float(base_x[fa] @ base_x[fb])) < 0.3 * sf:
                    continue
                dl = min(FLIP_END, time.time() + 3.0)
                if dl >= HARD_END - 15.0:
                    break
                Yp, vp = _lp_polish(Y0, dl, r0=0.003, maxit=60)
                try:
                    hk, _, _ = _deg_hash(Yp)
                except Exception:
                    continue
                is_new = (hk not in seen)
                if is_new:
                    seen.add(hk)
                # retain only if genuinely different hash or improves
                if (not is_new) and (vp >= global_v):
                    continue
                # for non-improving new hashes, require not too bad to count
                # but global best only updated on improvement
                if vp < global_v - 1e-12:
                    global_x = Yp.copy(); global_v = vp
                    # keep base fixed per spec; do not move base
                # also if new hash within tolerance, keep as candidate for fallback? no
            except Exception:
                continue
    except Exception:
        pass
    # final polish of best found
    try:
        rem = HARD_END - time.time() - 1.0
        if rem > 3.0:
            global_x, global_v = _lp_polish(global_x, time.time() + min(rem, 30.0), r0=0.002, maxit=150)
    except Exception:
        pass
    try:
        global_x = _norm_rows(global_x)
    except Exception:
        pass
    return global_x
