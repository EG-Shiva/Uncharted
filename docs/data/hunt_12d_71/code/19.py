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
    return float(g.max())

def _quench(X, steps, T0, T1, lr, mu=0.9, t_deadline=None):
    n, d = X.shape
    V = np.zeros_like(X)
    lt0 = np.log(T0); lt1 = np.log(T1)
    G0 = X @ X.T
    np.fill_diagonal(G0, -2.0)
    best_v = float(G0.max())
    best_X = X.copy()
    for s in range(steps):
        if t_deadline is not None and (s % 250 == 0) and time.time() > t_deadline:
            break
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
        # clip extreme tangential grads
        big = gn[:, 0] > 2.0
        if np.any(big):
            Gt[big] *= (2.0 / gn[big])
        V = mu * V + Gt
        X = X - lr * V
        X = _norm_rows(X)
        # track best occasionally (cheap max already available next iter,
        # so evaluate from current G proxy every 100 steps)
        if (s % 100 == 0) or (s == steps - 1):
            Gc = X @ X.T
            np.fill_diagonal(Gc, -2.0)
            vc = float(Gc.max())
            if vc < best_v:
                best_v = vc
                best_X = X.copy()
    return best_X, best_v

def _refine_single(s, R, steps=250, T=0.004, lr=0.12, mu=0.8):
    s = s / (np.linalg.norm(s) + 1e-12)
    v = np.zeros_like(s)
    for _ in range(steps):
        g = R @ s
        m = float(g.max())
        e = np.exp(np.clip((g - m) / T, -50, 0))
        den = e.sum() + 1e-30
        Grad = (R.T @ e) / den
        Gt = Grad - float(Grad @ s) * s
        v = mu * v + Gt
        s = s - lr * v
        s = s / (np.linalg.norm(s) + 1e-12)
    return s

def _find_hole(R, rng, n_sample=16000, n_cand=4):
    n, d = R.shape
    Z = _norm_rows(rng.standard_normal(size=(n_sample, d)))
    S = Z @ R.T
    scores = S.max(axis=1)
    idx = np.argpartition(scores, n_cand)[:n_cand]
    best_s = None
    best_v = 1e100
    for k in idx:
        s0 = Z[k]
        s1 = _refine_single(s0, R, steps=250, T=0.005, lr=0.12)
        s1 = _refine_single(s1, R, steps=250, T=0.0015, lr=0.07)
        v = float((R @ s1).max())
        if v < best_v:
            best_v = v
            best_s = s1
    return best_s / (np.linalg.norm(best_s) + 1e-12)

def _lp_polish(x, seconds, deadline=None):
    x = _norm_rows(x.copy())
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds:
        if deadline is not None and time.time() > deadline:
            break
        if radius < 1e-14:
            break
        cmat = (x @ x.T)[iu]
        m = float(cmat.max())
        sel = cmat > m - max(20 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        if p == 0:
            break
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cc = np.concatenate([I[:, None] * d + np.arange(d),
                             J[:, None] * d + np.arange(d),
                             np.full((p, 1), n * d)], 1).ravel()
        vv = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vv, (rows, cc)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        except Exception:
            break
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-cmat[sel], A_eq=A_eq, b_eq=np.zeros(n),
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
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def solve():
    t0 = time.time()
    TIME_LIMIT = 173.0
    rng = np.random.default_rng(1)
    cands = []
    for fn in ("best.npy", "record.npy"):
        try:
            a = _norm_rows(np.asarray(np.load(fn), dtype=np.float64))
            if a.shape == (71, 12):
                cands.append(a)
        except Exception:
            pass
    if not cands:
        cands.append(_norm_rows(rng.standard_normal(size=(71, 12))))
    bestX = min(cands, key=_max_cos).copy()
    bestVal = _max_cos(bestX)
    curX = bestX.copy()
    curVal = bestVal
    archive = [(bestVal, bestX.copy())]

    def push(v, X):
        archive.append((v, X.copy()))
        archive.sort(key=lambda t: t[0])
        del archive[5:]

    explore_deadline = t0 + 126.0
    bh_T = 0.0015
    it = 0
    while time.time() < explore_deadline:
        it += 1
        if time.time() > explore_deadline:
            break
        if it % 3 == 0:
            # hole re-insertion: kill hubs with many near-contacts
            base = bestX if rng.random() < 0.75 else curX
            bv = bestVal if rng.random() < 0.75 else curVal
            G = base @ base.T
            np.fill_diagonal(G, -2.0)
            thr = bv - 0.008
            cnt = (G > thr).sum(axis=1)
            order = np.argsort(-cnt)
            topk = order[:10]
            nrem = 1
            r = rng.random()
            if r < 0.25:
                nrem = 2
            elif r < 0.35:
                nrem = 3
            victims = []
            # mostly kill worst hubs, sometimes random hub
            for _ in range(nrem):
                if rng.random() < 0.75:
                    # pick worst not already picked
                    for c in topk:
                        if int(c) not in victims:
                            victims.append(int(c))
                            break
                else:
                    c = int(rng.choice(topk))
                    if c not in victims:
                        victims.append(c)
                    else:
                        victims.append(int(rng.integers(0, 71)))
            victims = sorted(set(victims))
            mask = np.ones(71, dtype=bool)
            mask[victims] = False
            Xt_base = base[mask]
            for _ in range(len(victims)):
                s_new = _find_hole(Xt_base, rng, n_sample=14000, n_cand=4)
                Xt_base = np.vstack([Xt_base, s_new[None, :]])
            Xt = Xt_base
            T0 = float(rng.choice([0.012, 0.008, 0.006, 0.015, 0.010]))
            T1 = float(rng.choice([0.0008, 0.0004, 0.0012, 0.0006, 0.0003]))
            lr = float(rng.choice([0.05, 0.04, 0.06, 0.03, 0.07]))
            steps = int(rng.choice([1800, 2500, 3200, 2000, 1500]))
            Xt, _ = _quench(Xt, steps=steps, T0=T0, T1=T1, lr=lr, mu=0.9, t_deadline=explore_deadline)
            v = _max_cos(Xt)
        else:
            base = bestX if rng.random() < 0.65 else curX
            Xt = base.copy()
            k = int(rng.choice([1, 2, 2, 3, 4, 1]))
            idx = rng.choice(71, size=k, replace=False)
            if rng.random() < 0.06:
                Xt[idx] = _norm_rows(rng.standard_normal(size=(k, 12)))
            else:
                sigma = float(np.exp(rng.uniform(np.log(0.015), np.log(0.5))))
                Xt[idx] = Xt[idx] + rng.standard_normal(size=(k, 12)) * sigma
                Xt = _norm_rows(Xt)
            T0 = float(rng.choice([0.010, 0.007, 0.015, 0.012]))
            T1 = float(rng.choice([0.0005, 0.001, 0.0003, 0.0008]))
            lr = float(rng.choice([0.05, 0.03, 0.07, 0.04]))
            steps = int(rng.choice([1500, 2200, 3000, 1800]))
            Xt, _ = _quench(Xt, steps=steps, T0=T0, T1=T1, lr=lr, mu=0.9, t_deadline=explore_deadline)
            v = _max_cos(Xt)
        # guarded low-T tightening for promising basins
        if v < bestVal + 0.003 and time.time() < explore_deadline - 0.5:
            Xt2, v2 = _quench(Xt.copy(), steps=600, T0=0.0008, T1=0.0002, lr=0.02, mu=0.9, t_deadline=explore_deadline)
            # _quench already returns best seen; recompute to be safe
            v2 = _max_cos(Xt2)
            if v2 < v:
                Xt, v = Xt2, v2
        if v < bestVal:
            bestVal = v
            bestX = Xt.copy()
            curX = Xt.copy()
            curVal = v
            push(v, Xt)
        else:
            if v < bestVal + 0.004:
                if len(archive) < 5 or v < archive[-1][0]:
                    push(v, Xt)
            if v < curVal or rng.random() < np.exp(-(v - curVal) / bh_T):
                curX = Xt.copy()
                curVal = v
        if it % 25 == 0:
            curX = bestX.copy()
            curVal = bestVal

    # polish archive, best first
    arch = sorted(archive, key=lambda t: t[0])[:4]
    remain = TIME_LIMIT - (time.time() - t0)
    if remain > 14 and len(arch) > 0:
        # short tighten + LP per archive entry
        for (av, ax) in list(arch):
            if time.time() - t0 > TIME_LIMIT - 13:
                break
            y, vy = _quench(ax.copy(), steps=1200, T0=0.001, T1=0.00015, lr=0.02, mu=0.9, t_deadline=t0 + TIME_LIMIT - 12)
            vy = _max_cos(y)
            y2 = _lp_polish(y if vy < av else ax.copy(), seconds=1.2, deadline=t0 + TIME_LIMIT - 11)
            v2 = _max_cos(y2)
            if v2 < bestVal:
                bestVal = v2
                bestX = y2.copy()
            elif vy < bestVal:
                bestVal = vy
                bestX = y.copy()
    if time.time() - t0 < TIME_LIMIT - 2:
        y, _ = _quench(bestX.copy(), steps=2000, T0=0.0012, T1=0.00012, lr=0.018, mu=0.9, t_deadline=t0 + TIME_LIMIT - 8)
        vy = _max_cos(y)
        if vy < bestVal:
            bestVal = vy
            bestX = y.copy()
        rem2 = TIME_LIMIT - (time.time() - t0)
        if rem2 > 2:
            y2 = _lp_polish(bestX, seconds=rem2 - 0.5, deadline=t0 + TIME_LIMIT - 0.3)
            v2 = _max_cos(y2)
            if v2 < bestVal:
                bestX = y2.copy()
                bestVal = v2
    bestX = _norm_rows(bestX)
    return bestX
