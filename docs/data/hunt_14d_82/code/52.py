import time
import numpy as np

def _norm(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return X / n

def _stats(X):
    G = X @ X.T
    iu = np.triu_indices(X.shape[0], 1)
    c = G[iu]
    s = np.sort(c)[::-1]
    m = float(s[0])
    mean100 = float(np.mean(s[:100]))
    top500 = s[:500]
    tau = 1e-3
    e = np.exp((top500 - m) / tau)
    lse = float(m + tau * np.log(np.sum(e) + 1e-300))
    return m, mean100, lse, c, G

def _lex_less(a, b, eps=1e-12):
    if a[0] < b[0] - eps:
        return True
    if a[0] > b[0] + eps:
        return False
    if a[1] < b[1] - eps:
        return True
    if a[1] > b[1] + eps:
        return False
    if a[2] < b[2] - 1e-11:
        return True
    return False

def _quench(X0, steps, tau_s, tau_e, lr_s, lr_e, mu=0.9, t_dead=None):
    X = X0.copy()
    n, d = X.shape
    V = np.zeros_like(X)
    for s in range(steps):
        if t_dead is not None and (s % 100 == 0) and time.time() > t_dead:
            break
        frac = s / max(1, steps - 1)
        tau = tau_s * (1 - frac) + tau_e * frac
        lr = lr_s * (1 - frac) + lr_e * frac
        G = X @ X.T
        np.fill_diagonal(G, -10.0)
        m = float(np.max(G))
        # softmax weights
        Q = (G - m) / tau
        np.clip(Q, -50.0, 0.0, out=Q)
        E = np.exp(Q)
        np.fill_diagonal(E, 0.0)
        S = float(np.sum(E))
        if not np.isfinite(S) or S <= 0:
            break
        W = E / S
        Gr = W @ X
        dot = np.sum(Gr * X, axis=1, keepdims=True)
        Gt = Gr - dot * X
        # clip per-point grad norm for stability
        gn = np.linalg.norm(Gt, axis=1, keepdims=True) + 1e-300
        # no hard clip, just avoid explosion
        V = mu * V + Gt
        X = X - lr * V
        X = _norm(X)
    return X

def solve():
    t0 = time.time()
    OPT_END = t0 + 122.0
    LP_END = t0 + 132.0
    HARD = t0 + 138.0
    rng = np.random.default_rng(0)
    n, d = 82, 14
    cands = []
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=float)
            if a.shape == (n, d) and np.all(np.isfinite(a)):
                cands.append(_norm(a))
        except Exception:
            pass
    if not cands:
        cands.append(_norm(rng.standard_normal((n, d))))
    # pick best by max
    def _mx(A):
        G = A @ A.T
        np.fill_diagonal(G, -2.0)
        return float(np.max(G))
    cands.sort(key=_mx)
    X_best = cands[0].copy()
    m, m100, lse, _, _ = _stats(X_best)
    best_lex = (m, m100, lse)
    best_max = m

    iu0 = np.triu_indices(n, 1)
    I0 = iu0[0]; J0 = iu0[1]

    # Stage 1: quick polish of incumbent with gentle quench
    try:
        Y = _quench(X_best, 600, 8e-3, 6e-4, 0.6, 0.03, mu=0.88, t_dead=OPT_END)
        m2, m100_2, lse2, _, _ = _stats(Y)
        if _lex_less((m2, m100_2, lse2), best_lex):
            if m2 <= best_max + 1e-12:
                X_best = Y
                best_lex = (m2, m100_2, lse2)
                if m2 < best_max:
                    best_max = m2
    except Exception:
        pass

    # Stage 2: kicked basin hopping + quench
    it = 0
    # precompute worst-point selector from current best
    X_cur_base = X_best.copy()
    while time.time() < OPT_END:
        it += 1
        if time.time() > HARD - 20.0:
            break
        cyc = it % 6
        if cyc == 0:
            ks, scale, steps = 82, 0.015, 350
            t_s, t_e = 6e-3, 5e-4
            l_s, l_e = 0.5, 0.02
            full_rand = False
        elif cyc == 1:
            ks, scale, steps = 12, 0.06, 450
            t_s, t_e = 8e-3, 5e-4
            l_s, l_e = 0.6, 0.02
            full_rand = False
        elif cyc == 2:
            ks, scale, steps = 6, 0.25, 550
            t_s, t_e = 1e-2, 6e-4
            l_s, l_e = 0.8, 0.03
            full_rand = True
        elif cyc == 3:
            ks, scale, steps = 3, 0.4, 550
            t_s, t_e = 1e-2, 4e-4
            l_s, l_e = 0.8, 0.02
            full_rand = True
        elif cyc == 4:
            ks, scale, steps = 20, 0.03, 400
            t_s, t_e = 6e-3, 5e-4
            l_s, l_e = 0.5, 0.02
            full_rand = False
        else:
            ks, scale, steps = 8, 0.12, 500
            t_s, t_e = 8e-3, 5e-4
            l_s, l_e = 0.7, 0.02
            full_rand = False

        # find worst points in X_best (top-500 involvement)
        try:
            G = X_best @ X_best.T
            c_all = G[I0, J0]
            k500 = 500
            part = np.argpartition(-c_all, k500 - 1)[:k500]
            cnt = np.zeros(n, dtype=int)
            np.add.at(cnt, I0[part], 1)
            np.add.at(cnt, J0[part], 1)
            order = np.argsort(-cnt, kind="stable")
            # add small random tie-break every few iters
            if it % 7 == 0:
                order = order[np.argsort(-cnt[order] + rng.standard_normal(n) * 0.5, kind="stable")]
            sel = order[:ks]
        except Exception:
            sel = rng.choice(n, size=ks, replace=False)

        Y0 = X_best.copy()
        if full_rand and (it % 2 == 0):
            # far-hole reinsertion: randomize selected points fully
            Y0[sel] = rng.standard_normal((len(sel), d))
            Y0 = _norm(Y0)
            # light tangent kick to the rest to break symmetry
            pert = rng.standard_normal((n, d)) * 0.005
            pert -= (np.sum(pert * Y0, axis=1, keepdims=True)) * Y0
            Y0 = _norm(Y0 + pert)
        else:
            pert = rng.standard_normal((n, d)) * 0.0
            p_sel = rng.standard_normal((len(sel), d)) * scale
            # tangent component
            p_sel -= (np.sum(p_sel * Y0[sel], axis=1, keepdims=True)) * Y0[sel]
            Y0[sel] = Y0[sel] + p_sel
            # tiny global jitter
            g = rng.standard_normal((n, d)) * (scale * 0.1)
            g -= (np.sum(g * Y0, axis=1, keepdims=True)) * Y0
            Y0 = _norm(Y0 + g)

        try:
            Yq = _quench(Y0, steps, t_s, t_e, l_s, l_e, mu=0.88, t_dead=OPT_END)
        except Exception:
            continue
        try:
            m2, m100_2, lse2, _, _ = _stats(Yq)
        except Exception:
            continue
        new_lex = (m2, m100_2, lse2)
        # accept strictly improving max; also accept tail-only improvement to diversify base every ~10 iters
        if m2 < best_max - 1e-12:
            X_best = Yq
            best_lex = new_lex
            best_max = m2
            X_cur_base = Yq.copy()
        elif abs(m2 - best_max) <= 1e-12 and _lex_less(new_lex, best_lex):
            X_best = Yq
            best_lex = new_lex
            X_cur_base = Yq.copy()
        else:
            # occasional random restart quench from scratch to probe new basin
            if it % 12 == 0 and time.time() < OPT_END - 10.0:
                try:
                    R = _norm(rng.standard_normal((n, d)))
                    Rq = _quench(R, 500, 1e-2, 6e-4, 0.8, 0.03, mu=0.88, t_dead=OPT_END)
                    mr, mr100, lr_, _, _ = _stats(Rq)
                    if mr < best_max - 5e-4:  # only take if competitive
                        X_best = Rq
                        best_lex = (mr, mr100, lr_)
                        best_max = mr
                except Exception:
                    pass

    # Stage 3: final trust-region minimax LP micro-squeeze (small, fast)
    try:
        from scipy.optimize import linprog
        from scipy.sparse import coo_matrix
        X_cur = X_best.copy()
        cur_lex = best_lex
        radius = 3e-4
        for _lp_it in range(12):
            if time.time() > LP_END:
                break
            G = X_cur @ X_cur.T
            c_all = G[I0, J0]
            m_cur = float(np.max(c_all))
            # active: top 700 pairs
            K = 700
            part = np.argpartition(-c_all, K - 1)[:K]
            # sort for determinism
            o = np.argsort(-c_all[part])
            act = part[o]
            Ia = I0[act]; Ja = J0[act]; ca = c_all[act]
            p = len(act)
            nv = n * d + 1
            nd = n * d
            M = np.empty((p, 2 * d + 1))
            M[:, :d] = X_cur[Ja]
            M[:, d:2 * d] = X_cur[Ia]
            M[:, 2 * d] = -1.0
            vals = M.ravel()
            C = np.empty((p, 2 * d + 1), dtype=int)
            cd = np.arange(d)
            C[:, :d] = Ia[:, None] * d + cd
            C[:, d:2 * d] = Ja[:, None] * d + cd
            C[:, 2 * d] = nd
            cols = C.ravel()
            rows = np.repeat(np.arange(p), 2 * d + 1)
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((X_cur.ravel(), (np.repeat(np.arange(n), d), np.arange(nd))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv); obj[-1] = 1.0
            try:
                res = linprog(obj, A_ub=A_ub, b_ub=-ca, A_eq=A_eq, b_eq=np.zeros(n),
                              bounds=[(-radius, radius)] * nd + [(None, None)],
                              method="highs", options={"time_limit": 0.8})
            except Exception:
                break
            if res is None or res.status != 0:
                radius *= 0.5
                if radius < 1e-7:
                    break
                continue
            delta = res.x[:-1].reshape(n, d)
            Y = _norm(X_cur + delta)
            m2, m100_2, lse2, _, _ = _stats(Y)
            if m2 < m_cur - 1e-12 or (abs(m2 - m_cur) <= 1e-12 and (m100_2 < cur_lex[1] - 1e-12)):
                X_cur = Y
                cur_lex = (m2, m100_2, lse2)
                if m2 < best_max - 1e-12:
                    X_best = Y.copy()
                    best_max = m2
                    best_lex = cur_lex
                elif abs(m2 - best_max) <= 1e-12 and _lex_less(cur_lex, best_lex):
                    X_best = Y.copy()
                    best_lex = cur_lex
                radius = min(radius * 1.3, 8e-4)
            else:
                radius *= 0.5
                if radius < 1e-7:
                    break
    except Exception:
        pass

    X_best = _norm(X_best)
    return X_best
