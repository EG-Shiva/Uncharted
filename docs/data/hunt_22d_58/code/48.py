import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _norm(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return X / n

def _lp_polish(x, seconds, t_deadline):
    x = _norm(x)
    n, d = x.shape
    best = _max_cos(x)
    radius = 8e-4
    start = time.time()
    iu = np.triu_indices(n, 1)
    it = 0
    while time.time() - start < seconds and time.time() < t_deadline and radius > 1e-13:
        it += 1
        if it > 30:
            break
        c = (x @ x.T)[iu]
        m = float(c.max())
        window = max(20.0 * radius, 4e-5)
        # keep at most 650 most-violated
        order = np.argsort(-c, kind='stable')
        k = 0
        for idx in order:
            if c[idx] > m - window and k < 650:
                k += 1
            elif c[idx] <= m - window:
                break
            if k >= 650:
                break
        if k == 0:
            break
        sel_idx = order[:k]
        I = iu[0][sel_idx]
        J = iu[1][sel_idx]
        p = k
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        except Exception:
            break
        obj = np.zeros(nv)
        obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c[sel_idx], A_eq=A_eq, b_eq=np.zeros(n),
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
            x, best, radius = y, v, min(radius * 1.5, 0.02)
        else:
            radius *= 0.5
    return x

def _sparsify_descent(X, cap, steps, lr0=0.008, lr1=0.0012, alpha=90.0, temp=0.00035, wc=1.0, ws=0.7):
    n, d = X.shape
    thresh = cap - 0.002
    ll0, ll1 = np.log(lr0), np.log(lr1)
    for s in range(steps):
        f = s / max(1, steps - 1)
        lr = float(np.exp(ll0 + (ll1 - ll0) * f))
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        # smooth count gradient
        Z = (G - thresh) / temp
        np.clip(Z, -20, 20, out=Z)
        S = 1.0 / (1.0 + np.exp(-Z))
        np.fill_diagonal(S, 0.0)
        D = S * (1.0 - S) / temp
        np.fill_diagonal(D, 0.0)
        GradC = D @ X
        # high-temp softmax gradient
        E = np.exp(alpha * (G - m))
        np.fill_diagonal(E, 0.0)
        sw = float(E.sum())
        if not np.isfinite(sw) or sw <= 0:
            GradS = np.zeros_like(X)
        else:
            GradS = (E @ X) / sw
        def _tang(Gr):
            dot = np.sum(Gr * X, axis=1, keepdims=True)
            Gt = Gr - dot * X
            return Gt
        GtC = _tang(GradC)
        GtS = _tang(GradS)
        nc = float(np.linalg.norm(GtC, axis=1).mean()) + 1e-12
        ns = float(np.linalg.norm(GtS, axis=1).mean()) + 1e-12
        if m > cap + 0.0004:
            # must pull max down hard
            Gt = 0.15 * GtC / nc + 2.0 * GtS / ns
        elif m > cap:
            Gt = 0.4 * GtC / nc + 1.2 * GtS / ns
        else:
            Gt = wc * GtC / nc + ws * GtS / ns
        X = X - lr * Gt
        X = _norm(X)
    return X

def _near_stats(X, cap):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    m = float(G.max())
    thresh = cap - 0.002
    cnt_pairs = int(np.count_nonzero(np.triu(G > thresh, 1)))
    deg = np.count_nonzero(G > thresh, axis=1)
    nlow = int(np.count_nonzero(deg < 10))
    return m, cnt_pairs, deg, nlow, G

def _low_stress_kick(X, cap, G, rng, eps=0.016):
    n, d = X.shape
    # active set: top P pairs
    iu = np.triu_indices(n, 1)
    c = G[iu]
    P = min(260, len(c))
    order = np.argpartition(-c, P - 1)[:P]
    I = iu[0][order]
    J = iu[1][order]
    p = len(I)
    A = np.zeros((p, n * d))
    # fill rows
    for k in range(p):
        i = int(I[k]); j = int(J[k])
        A[k, i * d:(i + 1) * d] = X[j]
        A[k, j * d:(j + 1) * d] = X[i]
    try:
        _, _, Vt = np.linalg.svd(A, full_matrices=False)
    except Exception:
        return X, False
    # smallest 2 singular vectors
    cand_dirs = []
    for t in range(min(3, Vt.shape[0])):
        v = Vt[-(1 + t)]
        Vm = v.reshape(n, d)
        dot = np.sum(Vm * X, axis=1, keepdims=True)
        Vtang = Vm - dot * X
        nrm = float(np.linalg.norm(Vtang)) + 1e-12
        Vtang /= nrm
        cand_dirs.append(Vtang)
    # heavily loaded contacts: top pairs by G + degree
    thresh = cap - 0.002
    deg = np.count_nonzero(G > thresh, axis=1)
    # score pairs by G + deg sum
    scores = c[order] + 0.0002 * (deg[I] + deg[J])
    o2 = np.argsort(-scores)
    topk = min(6, p)
    endpoints = []
    for k in range(topk):
        idx = o2[k]
        endpoints.append(int(I[idx])); endpoints.append(int(J[idx]))
    # unique, sorted by degree descending
    endpoints = sorted(set(endpoints), key=lambda i: -deg[i])[:6]
    if len(endpoints) == 0:
        return X, False
    # current smooth objective
    def _smooth_obj(Y):
        Gg = Y @ Y.T
        np.fill_diagonal(Gg, -2.0)
        mm = float(Gg.max())
        thr = cap - 0.002
        Z = np.clip((Gg - thr) / 0.00035, -20, 20)
        S = 1.0 / (1.0 + np.exp(-Z))
        np.fill_diagonal(S, 0.0)
        cnt = float(S.sum() / 2.0)
        # penalty for exceeding cap
        viol = max(0.0, mm - cap) * 20000.0
        return cnt + viol, mm
    base_obj, base_m = _smooth_obj(X)
    bestX = X
    bestObj = base_obj
    improved = False
    # try combinations
    for Vd in cand_dirs:
        for sgn in (1.0, -1.0):
            for subset in (endpoints[:2], endpoints[:3], endpoints[2:5] if len(endpoints) >= 5 else endpoints[:2]):
                Y = X.copy()
                for idx in subset:
                    vv = Vd[idx]
                    vn = float(np.linalg.norm(vv))
                    if vn < 1e-9:
                        # random tangent
                        r = rng.standard_normal(d)
                        r = r - float(r @ X[idx]) * X[idx]
                        rn = float(np.linalg.norm(r)) + 1e-12
                        vv = r / rn
                        vn = 1.0
                    step = eps * sgn / (np.sqrt(len(subset)))
                    Y[idx] = Y[idx] + step * Vd[idx] * (np.sqrt(n) * 1.0)
                    # scale: Vd is global unit, per-point ~ 1/sqrt(n); boost
                Y = _norm(Y)
                o, mm = _smooth_obj(Y)
                if o < bestObj - 1e-9:
                    bestObj = o
                    bestX = Y
                    improved = True
    # also try pure random tangent kick on worst hub if no improvement
    if not improved:
        worst = int(np.argmax(deg))
        Y = X.copy()
        r = rng.standard_normal(d)
        r = r - float(r @ Y[worst]) * Y[worst]
        r /= float(np.linalg.norm(r)) + 1e-12
        Y[worst] = _norm((Y[worst] + eps * r)[None, :])[0]
        o, mm = _smooth_obj(Y)
        if o < bestObj:
            bestX = Y
            improved = True
    return bestX, improved

def solve():
    t0 = time.time()
    T_END = t0 + 172.0
    HOP_END = t0 + 138.0
    rng = np.random.default_rng(12345)
    best = None
    best_val = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            c = np.load(fn).astype(float)
            if c.shape != (58, 22):
                continue
            c = _norm(c)
            v = _max_cos(c)
            if v < best_val:
                best_val = v
                best = c.copy()
        except Exception:
            continue
    if best is None:
        z = rng.standard_normal((58, 22))
        best = _norm(z)
        best_val = _max_cos(best)
    n, d = 58, 22
    deltas = [1e-3, 3e-4, 1e-4]
    cycle = 0
    # quick LP polish budget helper
    while time.time() < HOP_END and cycle < 30:
        cycle += 1
        # start from incumbent with kick proportional to delta cycle
        cur_delta = deltas[(cycle - 1) % len(deltas)]
        # alternate kick sizes
        X = best.copy()
        sig = 0.006 + 0.004 * ((cycle // 3) % 3)
        if cycle % 4 == 0:
            # stronger escape: reinsert worst hub into hole
            G0 = X @ X.T
            np.fill_diagonal(G0, -2.0)
            m0 = float(G0.max())
            deg0 = np.count_nonzero(G0 > m0 - 0.004, axis=1)
            victim = int(np.argmax(deg0))
            mask = np.ones(n, dtype=bool); mask[victim] = False
            Y = X[mask]
            C = rng.standard_normal((2500, d)); C = _norm(C)
            worst = (C @ Y.T).max(axis=1)
            X[victim] = C[int(np.argmin(worst))]
            X = _norm(X + sig * 0.5 * rng.standard_normal(X.shape))
        else:
            X = _norm(X + sig * rng.standard_normal(X.shape))
        cap0 = best_val + cur_delta
        # stage sparsify at relaxed cap
        for stage_i, delta in enumerate([cur_delta, cur_delta * 0.3, 0.0]):
            if time.time() > HOP_END:
                break
            cap = best_val + delta
            # fewer steps when time short
            remain = HOP_END - time.time()
            steps = 550 if remain > 30 else (300 if remain > 12 else 150)
            # hotter alpha early, colder later
            alpha = 70.0 if stage_i == 0 else (110.0 if stage_i == 1 else 140.0)
            try:
                X = _sparsify_descent(X, cap, steps, lr0=0.008, lr1=0.0012, alpha=alpha)
            except Exception:
                break
            # low-stress kicks to reduce count (2 attempts)
            try:
                m, cntp, deg, nlow, G = _near_stats(X, cap)
                for _ in range(2):
                    if time.time() > HOP_END:
                        break
                    Xn, _ = _low_stress_kick(X, cap, G, rng, eps=0.016 if stage_i == 0 else 0.010)
                    # accept if not worse in max beyond cap tolerance
                    m2 = _max_cos(Xn)
                    if m2 <= cap + 0.0006:
                        X = Xn
                        m, cntp, deg, nlow, G = _near_stats(X, cap)
                    else:
                        break
            except Exception:
                pass
            # LP polish after each stage, short budget
            try:
                m_before = _max_cos(X)
                # only polish if feasible-ish
                if m_before < cap + 0.001:
                    budget = 4.0
                    if HOP_END - time.time() < budget + 1:
                        budget = max(1.0, HOP_END - time.time() - 0.5)
                    Xp = _lp_polish(X, budget, HOP_END)
                    vp = _max_cos(Xp)
                    # keep polished if it improves max (even if count rises)
                    if vp < m_before:
                        X = Xp
                    # track global best
                    if vp < best_val:
                        best = Xp.copy()
                        best_val = vp
            except Exception:
                pass
            # early exit if already beat incumbent and sparse
            if best_val < 0.06662 and time.time() > HOP_END - 5:
                break
        # final check of X itself
        try:
            v = _max_cos(X)
            if v < best_val and time.time() < T_END - 5:
                best = X.copy()
                best_val = v
        except Exception:
            pass
    # final LP polish with remaining budget
    try:
        budget = min(28.0, max(4.0, T_END - time.time() - 2.0))
        if budget > 1.5:
            best = _lp_polish(best, budget, T_END - 1.0)
    except Exception:
        pass
    best = _norm(best)
    # sanitize
    best = np.where(np.isfinite(best), best, 0.0)
    for i in range(best.shape[0]):
        nr = float(np.linalg.norm(best[i]))
        if not np.isfinite(nr) or nr < 1e-12:
            v = rng.standard_normal(d)
            v /= float(np.linalg.norm(v)) + 1e-12
            best[i] = v
    best = _norm(best)
    return best
