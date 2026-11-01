import time
import numpy as np

def _norm_rows(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[~np.isfinite(n)] = 1.0
    n[n == 0] = 1.0
    return x / n

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _quench_lse(X, steps, T0, T1, lr, mu=0.9, track=True):
    n, d = X.shape
    V = np.zeros_like(X)
    lt0 = np.log(T0); lt1 = np.log(T1)
    if track:
        G0 = X @ X.T
        np.fill_diagonal(G0, -2.0)
        bv = float(G0.max()); bX = X.copy()
    else:
        bv = 1e100; bX = None
    for s in range(steps):
        frac = s / max(1, steps - 1)
        T = float(np.exp(lt0 + (lt1 - lt0) * frac))
        G = X @ X.T
        np.fill_diagonal(G, -1e100)
        m = float(G.max())
        Q = (G - m) / T
        np.clip(Q, -50, 0, out=Q)
        E = np.exp(Q)
        np.fill_diagonal(E, 0.0)
        rs = E.sum(axis=1, keepdims=True)
        rs[rs == 0] = 1.0
        Grad = (E @ X) / rs
        dot = np.sum(Grad * X, axis=1, keepdims=True)
        Gt = Grad - dot * X
        gn = np.linalg.norm(Gt, axis=1, keepdims=True) + 1e-12
        mk = gn[:, 0] > 2.0
        if np.any(mk):
            Gt[mk] *= (2.0 / gn[mk])
        V = mu * V + Gt
        X = X - lr * V
        X = _norm_rows(X)
        if track and ((s % 200 == 0) or s == steps - 1):
            Gc = X @ X.T
            np.fill_diagonal(Gc, -2.0)
            vc = float(Gc.max())
            if vc < bv:
                bv = vc; bX = X.copy()
    if track:
        return bX, bv
    return X, _max_cos(X)

def _quench_riesz(X, steps, s_pow, lr, mu=0.9):
    n, d = X.shape
    V = np.zeros_like(X)
    for s in range(steps):
        G = X @ X.T
        np.fill_diagonal(G, -1.0)
        D2 = np.clip(2.0 - 2.0 * G, 1e-6, 4.0)
        W = np.power(D2, -s_pow * 0.5 - 1.0) * s_pow
        np.fill_diagonal(W, 0.0)
        Grad = W @ X
        dot = np.sum(Grad * X, axis=1, keepdims=True)
        Gt = Grad - dot * X
        gn = float(np.median(np.linalg.norm(Gt, axis=1)) + 1e-12)
        Gt = Gt / gn
        V = mu * V + Gt
        X = X - lr * V
        X = _norm_rows(X)
    return X

def _single_refine(s, R, steps=180, T=0.004, lr=0.12, mu=0.8):
    s = s / (np.linalg.norm(s) + 1e-12)
    v = np.zeros_like(s)
    for _ in range(steps):
        g = R @ s
        m = float(g.max())
        e = np.exp(np.clip((g - m) / T, -50, 0))
        den = float(e.sum()) + 1e-30
        Grad = (R.T @ e) / den
        Gt = Grad - float(Grad @ s) * s
        v = mu * v + Gt
        s = s - lr * v
        nn = float(np.linalg.norm(s))
        if not np.isfinite(nn) or nn == 0:
            break
        s = s / nn
    return s

def _find_hole(R, rng, n_sample=5000, n_cand=3):
    d = R.shape[1]
    Z = _norm_rows(rng.standard_normal(size=(n_sample, d)))
    S = Z @ R.T
    scores = S.max(axis=1)
    idx = np.argpartition(scores, n_cand)[:n_cand]
    best_s = None; best_v = 1e100
    for k in idx:
        s0 = Z[k]
        s1 = _single_refine(s0, R, steps=150, T=0.005, lr=0.12)
        s1 = _single_refine(s1, R, steps=150, T=0.0015, lr=0.07)
        v = float((R @ s1).max())
        if v < best_v:
            best_v = v; best_s = s1
    best_s = best_s / (np.linalg.norm(best_s) + 1e-12)
    return best_s

def _lp_polish(x, seconds, deadline=None):
    from scipy.optimize import linprog
    from scipy.sparse import coo_matrix
    x = _norm_rows(x.copy())
    n, d = x.shape
    try:
        best = _max_cos(x)
    except Exception:
        return x
    radius = 1.0e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds:
        if deadline is not None and time.time() > deadline:
            break
        if radius < 1e-14:
            break
        G = x @ x.T
        cuv = G[iu]
        m = float(cuv.max())
        sel = cuv > m - max(20 * radius, 1e-9)
        if np.count_nonzero(sel) == 0:
            break
        I = iu[0][sel]; J = iu[1][sel]
        p = len(I)
        if p > 1500:
            order = np.argsort(-cuv[sel])[:1500]
            I, J = I[order], J[order]
            csel = cuv[sel][order]
            p = 1500
        else:
            csel = cuv[sel]
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        except Exception:
            radius *= 0.5
            continue
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
        y = _norm_rows(y)
        v = _max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.02)
        else:
            radius *= 0.5
    return x

def _descriptors(X):
    G = X @ X.T
    try:
        eig = np.linalg.eigvalsh(G)
        eig = np.sort(eig)[::-1]
    except Exception:
        eig = np.zeros(X.shape[0])
    iu = np.triu_indices(X.shape[0], 1)
    c = G[iu]
    m = float(c.max())
    slack = np.sort(m - c)
    thr = 1e-9
    deg = (G > m - 1e-7).sum(axis=1) - 1
    deg = np.sort(deg)
    return eig, slack[:500], deg, m

def solve():
    t0 = time.time()
    HARD_END = t0 + 172.0
    rng = np.random.default_rng(20261101)
    pool = []  # list of dicts
    best_score = 1e100
    best_X = None

    def add_to_pool(X):
        nonlocal best_score, best_X
        X = _norm_rows(X.copy())
        sc = _max_cos(X)
        eig, slack, deg, _ = _descriptors(X)
        if sc < best_score:
            best_score = sc
            best_X = X.copy()
        # keep if within 0.002 of best (use current best_score after update;
        # allow slightly larger window to keep diversity)
        if sc <= best_score + 0.002:
            # diversity check: eig distance
            distinct = True
            for e in pool:
                d = float(np.linalg.norm(eig - e['eig']) / max(1.0, float(np.linalg.norm(eig))))
                sdiff = abs(sc - e['score'])
                if d < 2e-4 and sdiff < 5e-06:
                    distinct = False
                    # replace if better
                    if sc < e['score']:
                        e.update({'X': X.copy(), 'score': sc, 'eig': eig, 'slack': slack, 'deg': deg})
                        distinct = None
                    break
            if distinct:
                pool.append({'X': X.copy(), 'score': sc, 'eig': eig, 'slack': slack, 'deg': deg})
                # cap pool at 15, drop worst
                if len(pool) > 15:
                    pool.sort(key=lambda e: e['score'])
                    del pool[-1]
        return sc

    # continuation configs: (N, d, nstarts)
    configs = [(72, 12, 2), (75, 12, 2), (80, 12, 1), (85, 12, 1), (71, 10, 2), (71, 11, 2)]
    EXPLORE_END = t0 + 118.0
    for (Nc, dc, ns) in configs:
        for si in range(ns):
            if time.time() > EXPLORE_END or time.time() > HARD_END - 55:
                break
            X0 = _norm_rows(rng.standard_normal(size=(Nc, dc)))
            # Stage A: Riesz spread
            try:
                Xa = _quench_riesz(X0, steps=500, s_pow=4.0, lr=0.15)
            except Exception:
                Xa = X0
            # Stage B: LSE quench
            try:
                Xb, _ = _quench_lse(Xa, steps=1000, T0=0.01, T1=0.0008, lr=0.20, track=True)
            except Exception:
                Xb = Xa
            if time.time() > EXPLORE_END or time.time() > HARD_END - 55:
                # still try to use Xb if shape matches
                pass
            if dc < 12:
                # embed into 12D with small random extra coords
                Xe = np.zeros((Xb.shape[0], 12))
                Xe[:, :dc] = Xb
                Xe[:, dc:] = rng.standard_normal(size=(Xb.shape[0], 12 - dc)) * 0.15
                Xe = _norm_rows(Xe)
                try:
                    Xe, _ = _quench_lse(Xe, steps=700, T0=0.005, T1=0.0005, lr=0.15, track=True)
                except Exception:
                    pass
                # greedy hole re-insertion path: 2 kills
                for _k in range(2):
                    if time.time() > HARD_END - 50:
                        break
                    try:
                        G = Xe @ Xe.T
                        np.fill_diagonal(G, -2.0)
                        rowmax = G.max(axis=1)
                        victim = int(np.argmax(rowmax))
                        R = np.delete(Xe, victim, axis=0)
                        h = _find_hole(R, rng, n_sample=4000, n_cand=2)
                        Xe[victim] = h
                        Xe, _ = _quench_lse(Xe, steps=300, T0=0.003, T1=0.0005, lr=0.12, track=False)
                        Xe, _ = _quench_lse(Xe, steps=200, T0=0.001, T1=0.0004, lr=0.08, track=True)
                    except Exception:
                        break
                Xfinal = Xe
                # light LP
                try:
                    Xfinal = _lp_polish(Xfinal, seconds=0.8, deadline=HARD_END - 45)
                except Exception:
                    pass
                add_to_pool(Xfinal)
            else:
                Xcur = Xb
                # greedy deletion path to 71 with re-polish after every deletion
                while Xcur.shape[0] > 71:
                    if time.time() > EXPLORE_END + 10 or time.time() > HARD_END - 50:
                        # if out of time, delete remaining at once by rowmax order
                        G = Xcur @ Xcur.T
                        np.fill_diagonal(G, -2.0)
                        rowmax = G.max(axis=1)
                        order = np.argsort(rowmax)  # ascending keep best
                        keep = order[:71]
                        keep = np.sort(keep)
                        Xcur = Xcur[keep]
                        break
                    try:
                        G = Xcur @ Xcur.T
                        np.fill_diagonal(G, -2.0)
                        rowmax = G.max(axis=1)
                        m = float(G.max())
                        loads = (G > m - 0.002).sum(axis=1)
                        # primary rowmax, secondary loads
                        victim = int(np.lexsort((loads, rowmax))[-1])
                        Xcur = np.delete(Xcur, victim, axis=0)
                        # re-polish after every single deletion (short path-follow)
                        Xcur, _ = _quench_lse(Xcur, steps=300, T0=0.004, T1=0.0006, lr=0.14, track=False)
                    except Exception:
                        break
                try:
                    Xcur, _ = _quench_lse(Xcur, steps=600, T0=0.004, T1=0.0004, lr=0.12, track=True)
                except Exception:
                    pass
                try:
                    Xcur = _lp_polish(Xcur, seconds=0.8, deadline=HARD_END - 45)
                except Exception:
                    pass
                if Xcur.shape == (71, 12):
                    add_to_pool(Xcur)
        if time.time() > EXPLORE_END or time.time() > HARD_END - 55:
            break

    # ensure at least fallback random if pool empty
    if not pool:
        Xf = _norm_rows(rng.standard_normal(size=(71, 12)))
        Xf, _ = _quench_lse(Xf, steps=800, T0=0.01, T1=0.001, lr=0.2, track=True)
        add_to_pool(Xf)

    # intensive LP polishing across diverse elite pool
    pool.sort(key=lambda e: e['score'])
    # keep at least ten if we have them; if fewer, generate extra perturbed copies to reach ten
    if len(pool) < 10 and time.time() < HARD_END - 40:
        # expand pool by perturbing best
        base = pool[0]['X'].copy() if pool else _norm_rows(rng.standard_normal(size=(71, 12)))
        need = min(10 - len(pool), 6)
        for _ in range(need):
            if time.time() > HARD_END - 40:
                break
            Xn = base + 0.02 * rng.standard_normal(size=base.shape)
            Xn = _norm_rows(Xn)
            try:
                Xn, _ = _quench_lse(Xn, steps=400, T0=0.003, T1=0.0005, lr=0.12, track=True)
            except Exception:
                pass
            add_to_pool(Xn)
        pool.sort(key=lambda e: e['score'])

    # intensive polish: allocate remaining time across top entries
    pool.sort(key=lambda e: e['score'])
    topk = min(len(pool), 6)
    for k in range(topk):
        if time.time() > HARD_END - 6:
            break
        e = pool[k]
        secs = 4.0 if k < 2 else 2.0
        # leave margin
        secs = min(secs, max(0.5, HARD_END - 5 - time.time()))
        if secs <= 0.2:
            break
        try:
            Xp = _lp_polish(e['X'], seconds=secs, deadline=HARD_END - 4)
            sc = _max_cos(Xp)
            if sc < e['score'] - 1e-12:
                eig, slack, deg, _ = _descriptors(Xp)
                e['X'] = Xp; e['score'] = sc; e['eig'] = eig; e['slack'] = slack; e['deg'] = deg
                if sc < best_score:
                    best_score = sc; best_X = Xp.copy()
        except Exception:
            continue
        # short LSE tighten between LP rounds for top-2
        if k < 2 and time.time() < HARD_END - 8:
            try:
                Xt, _ = _quench_lse(e['X'], steps=300, T0=0.0012, T1=0.0003, lr=0.08, track=True)
                Xt = _lp_polish(Xt, seconds=1.5, deadline=HARD_END - 4)
                sc2 = _max_cos(Xt)
                if sc2 < e['score']:
                    eig, slack, deg, _ = _descriptors(Xt)
                    e['X'] = Xt; e['score'] = sc2; e['eig'] = eig; e['slack'] = slack; e['deg'] = deg
                    if sc2 < best_score:
                        best_score = sc2; best_X = Xt.copy()
            except Exception:
                pass

    pool.sort(key=lambda e: e['score'])
    final = pool[0]['X'] if pool else best_X
    if final is None:
        final = _norm_rows(rng.standard_normal(size=(71, 12)))
    final = _norm_rows(np.asarray(final, dtype=float))
    # tiny shrink guard not needed; ensure unit rows
    return final
