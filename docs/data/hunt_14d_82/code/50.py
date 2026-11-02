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
    return float(np.max(G))

def _lex_stats(X):
    G = X @ X.T
    iu = np.triu_indices(X.shape[0], 1)
    c = G[iu]
    s = np.sort(c)[::-1]
    m = float(s[0])
    m100 = float(np.mean(s[:100]))
    top500 = s[:500]
    tau = 1e-3
    e = np.exp((top500 - m) / tau)
    lse = float(m + tau * np.log(np.sum(e) + 1e-300))
    return m, m100, lse

def solve():
    t0 = time.time()
    OPT_END = t0 + 128.0
    HARD_END = t0 + 135.0
    n, d = 82, 14
    rng = np.random.default_rng(0)

    # ---- load incumbent ----
    cands = []
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=float)
            if a.shape == (n, d) and np.all(np.isfinite(a)):
                a = _norm(a)
                cands.append(a)
        except Exception:
            pass
    if len(cands) == 0:
        X_best = _norm(rng.standard_normal((n, d)))
    else:
        cands.sort(key=lambda c: _max_cos(c))
        X_best = cands[0].copy()
    X_best = _norm(X_best)
    best_max = _max_cos(X_best)
    try:
        bm, bm100, blse = _lex_stats(X_best)
        best_lex = (bm, bm100, blse)
    except Exception:
        best_lex = (best_max, best_max, best_max)

    def time_left():
        return HARD_END - time.time()

    def consider(X):
        nonlocal X_best, best_max, best_lex
        if X is None:
            return False
        X = np.asarray(X, dtype=float)
        if X.shape != (n, d) or not np.all(np.isfinite(X)):
            return False
        X = _norm(X)
        m = _max_cos(X)
        if m < best_max - 1e-12:
            try:
                a, b, c = _lex_stats(X)
                best_lex = (a, b, c)
            except Exception:
                pass
            best_max = m
            X_best = X.copy()
            return True
        return False

    # ============ Phase A: symmetric cyclic orbit, Adam on generators ============
    try:
        k = 6  # 5*14 + 12 = 82 differentiable
        # init generators
        Yg = rng.standard_normal((k, d)) * 1.0
        # Adam state
        m1 = np.zeros_like(Yg); m2 = np.zeros_like(Yg)
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        lr0 = 0.03
        # shift plan: first 5 gens full 14, last gen first 12 shifts
        shifts = []
        for gi in range(5):
            for s in range(14):
                shifts.append((gi, s))
        for s in range(12):
            shifts.append((5, s))
        assert len(shifts) == 82
        tau_start, tau_end = 0.02, 0.004
        stepsA = 500
        for step in range(stepsA):
            if time.time() > OPT_END - 60:
                break
            if time.time() > HARD_END - 8:
                break
            frac = step / max(1, stepsA - 1)
            tau = tau_start * (1 - frac) + tau_end * frac
            lr = lr0 * (1.0 - 0.5 * frac)
            Gn = _norm(Yg)
            # build orbit X (82,14)
            X = np.empty((n, d))
            for idx, (gi, s) in enumerate(shifts):
                X[idx] = np.roll(Gn[gi], s)
            # Gram + softmax weights (upper only)
            G = X @ X.T
            np.fill_diagonal(G, -np.inf)
            mx = float(np.max(G))
            # weights
            S = (G - mx) / tau
            S[G == -np.inf] = -1e9
            np.clip(S, -50, 0, out=S)
            E = np.exp(S)
            E[G == -np.inf] = 0.0
            iu = np.triu_indices(n, 1)
            Eu = E[iu]
            ssum = float(np.sum(Eu))
            if not np.isfinite(ssum) or ssum <= 0:
                ssum = 1.0
                Eu = np.ones_like(Eu) / len(Eu)
            else:
                Eu = Eu / ssum
            # Grad_X = Wsym @ X
            W = np.zeros((n, n))
            W[iu] = Eu
            W = W + W.T
            GradX = W @ X
            # pullback to generators
            GradG = np.zeros_like(Gn)
            for idx, (gi, s) in enumerate(shifts):
                g = GradX[idx]
                # un-roll
                GradG[gi] += np.roll(g, -s)
            # to Yg via tangent projection / radius
            r = np.linalg.norm(Yg, axis=1, keepdims=True)
            r[r == 0] = 1.0
            Xn = Yg / r
            dot = np.sum(GradG * Xn, axis=1, keepdims=True)
            GradY = (GradG - dot * Xn) / r
            # Adam update
            m1 = beta1 * m1 + (1 - beta1) * GradY
            m2 = beta2 * m2 + (1 - beta2) * (GradY ** 2)
            b1c = 1 - beta1 ** (step + 1)
            b2c = 1 - beta2 ** (step + 1)
            upd = (m1 / b1c) / (np.sqrt(m2 / b2c) + eps)
            Yg = Yg - lr * upd
            # occasional eval
            if step % 100 == 99 or step == stepsA - 1:
                Gn2 = _norm(Yg)
                X2 = np.empty((n, d))
                for idx2, (gi2, s2) in enumerate(shifts):
                    X2[idx2] = np.roll(Gn2[gi2], s2)
                consider(X2)
        # final symmetric point (also keep for release phase)
        Gn = _norm(Yg)
        Xsym = np.empty((n, d))
        for idx, (gi, s) in enumerate(shifts):
            Xsym[idx] = np.roll(Gn[gi], s)
        Xsym = _norm(Xsym)
        sym_best = Xsym.copy()
    except Exception:
        sym_best = None

    # ============ Phase B: combinatorial sparse seeds + greedy ============
    Xcomb_best = None
    try:
        a_grid = [0.4, 0.7]
        for a_val in a_grid:
            if time.time() > OPT_END - 55:
                break
            M = 2200
            P = np.zeros((M, d))
            for t in range(M):
                # support size 2-4
                sz = int(rng.integers(2, 5))
                pos = rng.choice(d, size=sz, replace=False)
                v = np.zeros(d)
                for p_ in pos:
                    r_ = rng.random()
                    if r_ < 0.3:
                        val = 1.0 if rng.random() < 0.5 else -1.0
                    elif r_ < 0.6:
                        val = a_val if rng.random() < 0.5 else -a_val
                    else:
                        # uniform in [-1,1] sparse jitter
                        val = float(rng.uniform(-1, 1))
                        if abs(val) < 0.2:
                            val = 0.5 * np.sign(val) if val != 0 else 0.5
                    v[p_] = val
                nv = np.linalg.norm(v)
                if nv == 0 or not np.isfinite(nv):
                    v[0] = 1.0
                    nv = 1.0
                P[t] = v / nv
            # greedy select 82
            sel = [int(rng.integers(0, M))]
            # precompute dots? iterative
            remaining = np.ones(M, dtype=bool)
            remaining[sel[0]] = False
            for it in range(1, n):
                if time.time() > HARD_END - 8:
                    break
                Smat = P[sel]  # (it, d)
                # scores for remaining: max dot
                # chunk to save memory
                cand_idx = np.where(remaining)[0]
                D = P[cand_idx] @ Smat.T
                mxv = np.max(D, axis=1)
                # pick minimal max
                bi = int(np.argmin(mxv))
                chosen = int(cand_idx[bi])
                sel.append(chosen)
                remaining[chosen] = False
            Xc = _norm(P[np.array(sel)])
            # quick local eval
            if Xcomb_best is None or _max_cos(Xc) < _max_cos(Xcomb_best):
                Xcomb_best = Xc.copy()
            consider(Xc)
    except Exception:
        pass

    # ============ Phase C: release symmetry - full Adam on seeds ============
    try:
        seeds = []
        if sym_best is not None:
            seeds.append(sym_best)
        if Xcomb_best is not None:
            seeds.append(Xcomb_best)
        # also include a perturbed incumbent for diversity (small kick)
        if len(seeds) < 2:
            pert = rng.standard_normal((n, d)) * 0.02
            pert = pert - (np.sum(pert * X_best, axis=1, keepdims=True)) * X_best
            seeds.append(_norm(X_best + pert))
        for si, Sd in enumerate(seeds):
            if time.time() > OPT_END - 40:
                break
            Y = Sd.copy()  # will treat Y as pre-norm params
            m1f = np.zeros_like(Y); m2f = np.zeros_like(Y)
            stepsC = 250
            lr0f = 0.02
            tau_f = 0.008
            for st in range(stepsC):
                if time.time() > OPT_END - 40:
                    break
                if time.time() > HARD_END - 8:
                    break
                if st == 150:
                    tau_f = 0.004
                    lr0f = 0.01
                r = np.linalg.norm(Y, axis=1, keepdims=True)
                r[r == 0] = 1.0
                X = Y / r
                G = X @ X.T
                np.fill_diagonal(G, -np.inf)
                mx = float(np.max(G))
                S = (G - mx) / tau_f
                S[G == -np.inf] = -1e9
                np.clip(S, -50, 0, out=S)
                E = np.exp(S)
                E[G == -np.inf] = 0.0
                iu = np.triu_indices(n, 1)
                Eu = E[iu]
                ss = float(np.sum(Eu))
                if not np.isfinite(ss) or ss <= 0:
                    break
                Eu = Eu / ss
                W = np.zeros((n, n))
                W[iu] = Eu
                W = W + W.T
                GradX = W @ X
                dot = np.sum(GradX * X, axis=1, keepdims=True)
                GradY = (GradX - dot * X) / r
                m1f = 0.9 * m1f + 0.1 * GradY
                m2f = 0.999 * m2f + 0.001 * (GradY ** 2)
                b1c = 1 - 0.9 ** (st + 1)
                b2c = 1 - 0.999 ** (st + 1)
                upd = (m1f / b1c) / (np.sqrt(m2f / b2c) + 1e-8)
                Y = Y - lr0f * upd
            consider(_norm(Y))
    except Exception:
        pass

    # ============ Phase D: brief wide-active-set LP polish on incumbent ============
    try:
        X_cur = X_best.copy()
        cur_max = best_max
        try:
            cm, cm100, clse = best_lex
            cur_lex = (cm, cm100, clse)
        except Exception:
            cur_lex = (cur_max, cur_max, cur_max)
        iu0 = np.triu_indices(n, 1)
        I0 = iu0[0]; J0 = iu0[1]
        radius = 5e-4
        rmin, rmax = 2e-7, 1e-3
        nd = n * d
        it = 0
        while time.time() < OPT_END and time.time() < HARD_END - 3:
            it += 1
            if it > 40:
                break
            if radius < rmin:
                radius = 5e-4
                # tiny tangent jitter
                pert = rng.standard_normal((n, d)) * 1e-5
                pert = pert - (np.sum(pert * X_cur, axis=1, keepdims=True)) * X_cur
                X_cur = _norm(X_cur + pert)
                cur_max = _max_cos(X_cur)
                continue
            G = X_cur @ X_cur.T
            c_all = G[I0, J0]
            m_cur = float(np.max(c_all))
            # wide active set: top 1500
            K = 1500
            if len(c_all) > K:
                part = np.argpartition(-c_all, K - 1)[:K]
                # sort for determinism
                o = np.argsort(-c_all[part])
                act = part[o]
            else:
                act = np.arange(len(c_all))
            Ia = I0[act]; Ja = J0[act]; ca = c_all[act]
            p_ = len(act)
            # build minimax LP: min t s.t. Xj*dxi + Xi*dxj - t <= -ca
            rows = np.repeat(np.arange(p_), 2 * d + 1)
            C = np.empty((p_, 2 * d + 1), dtype=int)
            cd = np.arange(d)
            C[:, :d] = Ia[:, None] * d + cd
            C[:, d:2 * d] = Ja[:, None] * d + cd
            C[:, 2 * d] = nd
            cols = C.ravel()
            Mmat = np.empty((p_, 2 * d + 1))
            Mmat[:, :d] = X_cur[Ja]
            Mmat[:, d:2 * d] = X_cur[Ia]
            Mmat[:, 2 * d] = -1.0
            vals = Mmat.ravel()
            try:
                A_ub = coo_matrix((vals, (rows, cols)), shape=(p_, nd + 1)).tocsr()
                A_eq = coo_matrix((X_cur.ravel(), (np.repeat(np.arange(n), d), np.arange(nd))), shape=(n, nd + 1)).tocsr()
                obj = np.zeros(nd + 1); obj[-1] = 1.0
                b_ub = -ca
                res = linprog(obj, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=np.zeros(n),
                              bounds=[(-radius, radius)] * nd + [(None, None)],
                              method="highs", options={"time_limit": 1.0})
            except Exception:
                radius *= 0.5
                continue
            if res is not None and res.status == 0:
                delta = res.x[:-1].reshape(n, d)
                Y2 = _norm(X_cur + delta)
                m_n = _max_cos(Y2)
                # accept on max improvement (allow tiny tol) else tail check
                improved = False
                if m_n < cur_max - 1e-12:
                    improved = True
                elif m_n <= cur_max + 1e-12:
                    # check tail mean-top100 not worse and max not worse
                    try:
                        mn, mn100, mlse = _lex_stats(Y2)
                        if mn <= cur_lex[0] + 1e-12 and (mn100 < cur_lex[1] - 1e-12 or mlse < cur_lex[2] - 1e-11):
                            # only accept if global best not worsened
                            if mn <= best_max + 1e-12:
                                improved = True
                    except Exception:
                        pass
                if improved:
                    X_cur = Y2
                    cur_max = m_n
                    try:
                        a_, b_, c_ = _lex_stats(X_cur)
                        cur_lex = (a_, b_, c_)
                    except Exception:
                        cur_lex = (cur_max, cur_max, cur_max)
                    consider(X_cur)
                    # sync cur to best if best improved elsewhere
                    cur_max = min(cur_max, best_max)
                    radius = min(radius * 1.25, rmax)
                else:
                    radius *= 0.5
            else:
                radius *= 0.5
    except Exception:
        pass

    X_best = _norm(X_best)
    return X_best
