import time
import numpy as np
from scipy.optimize import minimize, linprog
from scipy.sparse import coo_matrix

def _norm(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n = np.maximum(n, 1e-12)
    return X / n

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def _lbfgs_free(X0, beta, maxiter):
    X0 = _norm(X0)
    n, d = X0.shape
    iu0, iu1 = np.triu_indices(n, 1)
    y0 = X0.ravel().copy()
    def fun(y):
        Y = y.reshape(n, d)
        rn = np.linalg.norm(Y, axis=1, keepdims=True)
        rn = np.maximum(rn, 1e-12)
        Xn = Y / rn
        G = Xn @ Xn.T
        g = G[iu0, iu1]
        m = float(g.max())
        e = np.exp(beta * (g - m))
        S = float(e.sum()) + 1e-300
        loss = m + np.log(S) / beta
        w = e / S
        W = np.zeros((n, n))
        W[iu0, iu1] = w
        W = W + W.T
        GX = W @ Xn
        dot = (GX * Xn).sum(axis=1, keepdims=True)
        GY = (GX - dot * Xn) / rn
        return float(loss), GY.ravel()
    try:
        res = minimize(fun, y0, jac=True, method="L-BFGS-B",
                       options={"maxiter": maxiter, "ftol": 1e-12, "gtol": 1e-8, "maxfun": 20000})
        Y = res.x.reshape(n, d)
        return _norm(Y)
    except Exception:
        return X0

def _lp_polish(X, deadline, r_init=1.0e-3, max_steps=12):
    X = _norm(X.copy())
    n, d = X.shape
    iu0, iu1 = np.triu_indices(n, 1)
    best = _max_cos(X)
    bx = X.copy()
    radius = r_init
    steps = 0
    while time.time() < deadline and steps < max_steps:
        steps += 1
        if radius < 1e-14:
            break
        Gf = X @ X.T
        c = Gf[iu0, iu1]
        m = float(c.max())
        thr = m - max(18.0 * radius, 1e-9)
        sel = c > thr
        I = iu0[sel]; J = iu1[sel]; csel = c[sel]
        p = len(I)
        if p == 0:
            break
        if p > 3500:
            k2 = 3500
            part = np.argpartition(-csel, k2 - 1)[:k2]
            I, J, csel = I[part], J[part], csel[part]
            p = k2
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([X[J], X[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, n * d + 1)).tocsr()
            A_eq = coo_matrix((X.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, n * d + 1)).tocsr()
            obj = np.zeros(n * d + 1); obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = _norm(X + res.x[:-1].reshape(n, d))
        v = _max_cos(y)
        if v < best:
            X, best = y, v
            bx = y.copy()
            radius = min(radius * 1.5, 0.03)
        else:
            radius *= 0.5
        if p < 10 and radius < 1e-6:
            break
    return bx, best

def _build_pool():
    N = 5 ** 6
    tmp = np.arange(N, dtype=np.int64)
    digits = np.empty((N, 6), dtype=np.int64)
    for k in range(6):
        digits[:, k] = tmp % 5 - 2
        tmp //= 5
    mask = np.any(digits != 0, axis=1)
    digits = digits[mask]
    norms = np.sqrt(np.sum(digits * digits, axis=1).astype(float))
    Xn = digits.astype(float) / norms[:, None]
    rounded = np.round(Xn, 8)
    seen = set()
    keep = []
    for i in range(rounded.shape[0]):
        key = tuple(rounded[i].tolist())
        if key not in seen:
            seen.add(key)
            keep.append(i)
    P = Xn[np.array(keep, dtype=int)]
    return _norm(P)

def _greedy(P, need=219):
    M = P.shape[0]
    selected = [0]
    cur_max = P @ P[0]
    cur_max[0] = np.inf
    cur_cnt = np.ones(M, dtype=np.int64)
    cur_cnt[0] = 10 ** 9
    for k in range(1, need):
        min_m = float(np.min(cur_max))
        cand = np.where(cur_max <= min_m + 1e-12)[0]
        if len(cand) > 1:
            cn = cur_cnt[cand]
            mc = int(np.min(cn))
            cand2 = cand[cn == mc]
            if len(cand2) > 1:
                vals = cur_max[cand2]
                best = int(cand2[np.argmin(vals)])
            else:
                best = int(cand2[0])
        else:
            best = int(cand[0])
        selected.append(best)
        if k == need - 1:
            break
        ndots = P @ P[best]
        eps = 1e-12
        greater = ndots > cur_max + eps
        equal = np.abs(ndots - cur_max) <= eps
        # do not touch already selected (inf)
        equal[cur_max > 1e9] = False
        greater[cur_max > 1e9] = False
        cur_max[greater] = ndots[greater]
        cur_cnt[greater] = 1
        cur_cnt[equal] += 1
        cur_max[best] = np.inf
        cur_cnt[best] = 10 ** 9
    return np.array(selected, dtype=int)

def _swap_improve(P, sel, deadline, max_passes=15):
    n = sel.shape[0]
    M = P.shape[0]
    sel = sel.copy()
    is_sel = np.zeros(M, dtype=bool)
    is_sel[sel] = True
    ar = np.arange(M)
    for _ in range(max_passes):
        if time.time() > deadline:
            break
        X = P[sel]
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        m = float(np.max(G))
        C = G >= m - 1e-12
        deg = C.sum(axis=1)
        crit = np.where(deg > 0)[0]
        if len(crit) == 0:
            break
        D = P @ X.T
        idx1 = np.argmax(D, axis=1)
        r1 = D[ar, idx1].copy()
        D[ar, idx1] = -2.0
        r2 = D.max(axis=1)
        r1[is_sel] = np.inf
        r2[is_sel] = np.inf
        # m_rest per crit
        m_rests = {}
        for s in crit:
            Gm = G.copy()
            Gm[s, :] = -2.0
            Gm[:, s] = -2.0
            m_rests[int(s)] = float(np.max(Gm))
        best_ach = np.inf
        best_sw = None
        for s in crit:
            s = int(s)
            m_rest = m_rests[s]
            mn = np.where(idx1 != s, r1, r2)
            p_star = int(np.argmin(mn))
            v = float(mn[p_star])
            ach = m_rest if m_rest > v else v
            if ach < best_ach - 1e-15:
                best_ach = ach
                best_sw = (s, p_star)
        if best_sw is None:
            break
        if best_ach < m - 1e-12:
            s_star, p_star = best_sw
            old = int(sel[s_star])
            sel[s_star] = int(p_star)
            is_sel[old] = False
            is_sel[int(p_star)] = True
        else:
            break
    return sel

def solve():
    t0 = time.time()
    BUDGET = 168.0
    n, d = 219, 6
    rng = np.random.RandomState(0)
    # fallback incumbent (not used for init)
    fb = None
    fb_val = 1e9
    for fn in ["best.npy", "record.npy"]:
        try:
            v = np.asarray(np.load(fn), dtype=float)
            if v.shape == (n, d):
                v = _norm(v)
                c = _max_cos(v)
                if c < fb_val:
                    fb_val = c
                    fb = v.copy()
        except Exception:
            pass
    if fb is None:
        fb = _norm(rng.randn(n, d))
        fb_val = _max_cos(fb)

    # 1) lattice pool
    try:
        P = _build_pool()
    except Exception:
        return fb
    # 2) greedy + swaps
    try:
        sel = _greedy(P, need=n)
        sel = _swap_improve(P, sel, deadline=t0 + 22.0, max_passes=15)
        X0 = P[sel].copy()
    except Exception:
        X0 = None

    cands = []
    if X0 is not None:
        cands.append((X0, _max_cos(X0)))

    # 3) orbit-constrained stage: D6 perms order 12, 18 reps + 3 free
    try:
        if time.time() - t0 < 40.0:
            # build D6 perms
            rot = tuple([(j - 1) % 6 for j in range(6)])
            ref = tuple([0, 5, 4, 3, 2, 1])
            # closure
            def compose(p1, s1, p2, s2):
                p = tuple(p1[p2[j]] for j in range(6))
                s = tuple(s2[j] * s1[p2[j]] for j in range(6))
                return p, s
            gens = [(rot, (1, 1, 1, 1, 1, 1)), (ref, (1, 1, 1, 1, 1, 1))]
            seen = set()
            elems = []
            ident = (tuple(range(6)), (1, 1, 1, 1, 1, 1))
            stack = [ident]
            seen.add(ident)
            while stack:
                cur = stack.pop()
                elems.append(cur)
                for g in gens:
                    for a, b in [(cur, g), (g, cur)]:
                        np_, ns_ = compose(a[0], a[1], b[0], b[1])
                        key = (np_, ns_)
                        if key not in seen:
                            seen.add(key)
                            stack.append(key)
            Gsz = len(elems)
            perms = np.array([list(p) for p, s in elems], dtype=int)
            signs = np.array([list(s) for p, s in elems], dtype=float)
            # use only if size 12 else still use
            R, F = 18, 3
            # adjust to fit 219
            if R * Gsz + F != n:
                # adapt F
                F = n - R * Gsz
                if F < 0 or F > 12:
                    R = n // Gsz
                    F = n - R * Gsz
            Y0 = _norm(rng.randn(R + F, d))
            iu0, iu1 = np.triu_indices(n, 1)
            def expand(Yn):
                Xo = np.empty((R * Gsz + F, d))
                for i in range(R):
                    r = Yn[i]
                    for g in range(Gsz):
                        Xo[i * Gsz + g] = signs[g] * r[perms[g]]
                for k in range(F):
                    Xo[R * Gsz + k] = Yn[R + k]
                return Xo
            def orb_fun(y, beta):
                Y = y.reshape(R + F, d)
                rn = np.linalg.norm(Y, axis=1, keepdims=True)
                rn = np.maximum(rn, 1e-12)
                Yn = Y / rn
                X = expand(Yn)
                G = X @ X.T
                g = G[iu0, iu1]
                m = float(g.max())
                e = np.exp(beta * (g - m))
                S = float(e.sum()) + 1e-300
                loss = m + np.log(S) / beta
                w = e / S
                W = np.zeros((n, n))
                W[iu0, iu1] = w
                W = W + W.T
                GX = W @ X
                # backprop to Yn
                GY = np.zeros_like(Yn)
                for i in range(R):
                    acc = np.zeros(d)
                    for gix in range(Gsz):
                        gg = GX[i * Gsz + gix]
                        p = perms[gix]; s = signs[gix]
                        for j in range(d):
                            acc[p[j]] += s[j] * gg[j]
                    GY[i] = acc
                for k in range(F):
                    GY[R + k] = GX[R * Gsz + k]
                dot = (GY * Yn).sum(axis=1, keepdims=True)
                GD = (GY - dot * Yn) / rn
                return float(loss), GD.ravel()
            y = Y0.ravel().copy()
            for beta, mi in [(80.0, 80), (400.0, 80)]:
                if time.time() - t0 > BUDGET - 90:
                    break
                try:
                    res = minimize(lambda yy: orb_fun(yy, beta), y, jac=True, method="L-BFGS-B",
                                   options={"maxiter": mi, "ftol": 1e-12, "gtol": 1e-7, "maxfun": 10000})
                    y = res.x.copy()
                except Exception:
                    break
            Y = (y.reshape(R + F, d))
            Y = _norm(Y)
            Xo = _norm(expand(Y))
            if Xo.shape[0] == n:
                # release symmetry: free relax a bit
                Xo = _lbfgs_free(Xo, 120.0, 40)
                cands.append((Xo.copy(), _max_cos(Xo)))
    except Exception:
        pass

    # 4) free relax of discrete candidate with beta schedule 50->500->5000(capped)
    bestX = None
    bestV = 1e9
    try:
        for Xc, vc in cands:
            X = Xc.copy()
            # small jitter to break lattice exactness
            if time.time() - t0 < BUDGET - 60:
                X = _lbfgs_free(X, 50.0, 70)
            if time.time() - t0 < BUDGET - 50:
                X = _lbfgs_free(X, 500.0, 70)
            if time.time() - t0 < BUDGET - 40:
                # very sharp: use 2000 to avoid overflow stall
                X = _lbfgs_free(X, 2000.0, 60)
            v = _max_cos(X)
            if v < bestV:
                bestV = v
                bestX = X.copy()
    except Exception:
        pass

    if bestX is None:
        # fallback to first cand or fb
        if len(cands) > 0:
            bestX = cands[0][0].copy()
            bestV = cands[0][1]
        else:
            return fb

    # 5) LP polish remaining time
    try:
        dl = min(t0 + BUDGET - 4.0, time.time() + 18.0)
        if time.time() < dl - 1.0:
            bestX, bestV = _lp_polish(bestX, dl, r_init=1.0e-3, max_steps=14)
    except Exception:
        pass

    # guaranteed fallback comparison
    try:
        if fb_val < bestV:
            return fb
    except Exception:
        pass
    return _norm(bestX)
