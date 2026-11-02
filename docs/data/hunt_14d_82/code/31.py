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

def _adam_quench(X, steps, lr0, lr1, b0, b1):
    X = _norm(X.copy())
    n, d = X.shape
    m1 = np.zeros_like(X)
    m2 = np.zeros_like(X)
    b_1, b_2, eps = 0.9, 0.999, 1e-8
    p1, p2 = 1.0, 1.0
    import math
    l0, l1 = math.log(lr0), math.log(lr1)
    e0, e1 = math.log(b0), math.log(b1)
    for s in range(steps):
        frac = s / max(steps - 1, 1)
        lr = float(np.exp(l0 + (l1 - l0) * frac))
        beta = float(np.exp(e0 + (e1 - e0) * frac))
        G = X @ X.T
        np.fill_diagonal(G, -10.0)
        gm = float(np.max(G))
        E = np.exp(np.clip(beta * (G - gm), -30.0, 0.0))
        np.fill_diagonal(E, 0.0)
        Gr = E @ X
        dot = np.sum(Gr * X, axis=1, keepdims=True)
        Gt = Gr - dot * X
        p1 *= b_1
        p2 *= b_2
        m1 = b_1 * m1 + (1.0 - b_1) * Gt
        m2 = b_2 * m2 + (1.0 - b_2) * (Gt * Gt)
        mh = m1 / (1.0 - p1)
        vh = m2 / (1.0 - p2)
        upd = mh / (np.sqrt(vh) + eps)
        X = X - lr * upd
        X = _norm(X)
    return X

def _lp_polish(X, deadline, radius_init=2e-3, cap=1500, max_it=25):
    X = _norm(X.copy())
    n, d = X.shape
    best = _max_cos(X)
    radius = float(radius_init)
    iu = np.triu_indices(n, 1)
    it = 0
    while it < max_it and time.time() < deadline and radius > 1e-14:
        it += 1
        G = X @ X.T
        c_all = G[iu]
        m = float(np.max(c_all))
        thr = m - max(20.0 * radius, 1e-9)
        sel = c_all > thr
        cnt = int(np.sum(sel))
        if cnt == 0:
            break
        if cnt > cap:
            idx = np.where(sel)[0]
            vals = c_all[idx]
            part = np.argpartition(-vals, cap - 1)[:cap]
            m2 = np.zeros_like(sel, dtype=bool)
            m2[idx[part]] = True
            sel = m2
        I = iu[0][sel]
        J = iu[1][sel]
        c = c_all[sel]
        p = len(I)
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vls = np.concatenate([X[J], X[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vls, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((X.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv)
            obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-c, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                          method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        Y = X + res.x[:-1].reshape(n, d)
        Y = _norm(Y)
        v = _max_cos(Y)
        if v < best:
            X, best = Y, v
            radius = min(radius * 1.5, 0.02)
        else:
            radius *= 0.5
    return X

def _far_hole(fixed, rng, n_try=8000):
    d = fixed.shape[1]
    Z = rng.standard_normal((n_try, d))
    Z = _norm(Z)
    S = Z @ fixed.T
    w = S.max(axis=1)
    return Z[int(np.argmin(w))].copy()

def solve():
    t0 = time.time()
    BUDGET = 165.0
    t_end = t0 + BUDGET
    rng = np.random.default_rng(0)
    n, d = 82, 14
    cands = []
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=float)
            if a.shape == (n, d):
                a = _norm(a)
                if np.all(np.isfinite(a)):
                    cands.append(a)
        except Exception:
            pass
    if len(cands) == 0:
        xb = _norm(rng.standard_normal((n, d)))
    else:
        cands.sort(key=lambda c: _max_cos(c))
        xb = cands[0].copy()
    best_val = _max_cos(xb)
    Xbest = xb.copy()
    archive = [(best_val, Xbest.copy())]

    def push(X, v):
        archive.append((v, X.copy()))
        archive.sort(key=lambda t: t[0])
        del archive[8:]

    # Phase 1: diverse jammed seeds from scratch + large kicks of incumbent
    # pure random quenches
    for k in range(4):
        if time.time() > t_end - 30:
            break
        R = _norm(rng.standard_normal((n, d)))
        Rq = _adam_quench(R, steps=700, lr0=0.02, lr1=0.002, b0=15.0, b1=250.0)
        v = _max_cos(Rq)
        push(Rq, v)
        if v < best_val:
            best_val, Xbest = v, Rq.copy()
    # large perturbations of incumbent to new basins
    for sigma in (0.08, 0.15, 0.25):
        if time.time() > t_end - 28:
            break
        T = Xbest + sigma * rng.standard_normal((n, d))
        T = _norm(T)
        Tq = _adam_quench(T, steps=600, lr0=0.015, lr1=0.002, b0=20.0, b1=300.0)
        v = _max_cos(Tq)
        push(Tq, v)
        if v < best_val:
            best_val, Xbest = v, Tq.copy()
    # light LP on current best to gauge
    if time.time() < t_end - 25:
        tmp = _lp_polish(Xbest, deadline=min(t_end - 20, time.time() + 2.0), radius_init=1e-3)
        v = _max_cos(tmp)
        push(tmp, v)
        if v < best_val:
            best_val, Xbest = v, tmp.copy()

    it = 0
    sigmas = [0.02, 0.05, 0.10, 0.18, 0.03, 0.07]
    while time.time() < t_end - 12:
        it += 1
        if time.time() >= t_end - 12:
            break
        # pick base: mostly best, sometimes diverse
        if rng.random() < 0.65 or len(archive) == 1:
            base = Xbest.copy()
            base_val = best_val
        else:
            j = int(rng.integers(1, len(archive)))
            base_val, base = archive[j][0], archive[j][1].copy()
            # occasionally use best anyway if diverse is much worse
            if base_val > best_val + 0.015 and rng.random() < 0.7:
                base = Xbest.copy()
                base_val = best_val
        G = base @ base.T
        np.fill_diagonal(G, -2.0)
        m0 = float(np.max(G))
        flat = int(np.argmax(G))
        wi, wj = flat // n, flat % n
        rowmax = G.max(axis=1)
        order = np.argsort(-rowmax)
        r = rng.random()
        trial = base.copy()
        if r < 0.35:
            sigma = sigmas[it % len(sigmas)]
            trial = base + sigma * rng.standard_normal((n, d))
            trial = _norm(trial)
        elif r < 0.65:
            sigma = 0.05 + 0.10 * rng.random()
            k = int(rng.integers(3, 9))
            victims = set([int(wi), int(wj)])
            for q in range(k):
                victims.add(int(order[int(rng.integers(0, min(n, 12)))]))
            for _ in range(2):
                victims.add(int(rng.integers(0, n)))
            victims = np.array(sorted(victims), dtype=int)
            trial[victims] = base[victims] + sigma * rng.standard_normal((len(victims), d))
            trial = _norm(trial)
        else:
            # hole reinsertion: delete 1-2 worst points
            nv = 1 if rng.random() < 0.6 else 2
            victims = [int(order[q]) for q in range(nv)]
            # also sometimes delete a random low-contact point
            if rng.random() < 0.3:
                victims[-1] = int(rng.integers(0, n))
            for vic in victims:
                mask = np.ones(n, dtype=bool)
                mask[vic] = False
                fixed = trial[mask]
                h = _far_hole(fixed, rng, n_try=8000)
                # quick local refine of hole: few tangent steps
                for _ in range(40):
                    g = fixed @ h
                    mm = float(np.max(g))
                    w = np.exp(np.clip((g - mm) / 0.03, -30, 0))
                    grad = w @ fixed
                    grad = grad - (grad @ h) * h
                    nn = float(np.linalg.norm(grad))
                    if nn < 1e-12 or not np.isfinite(nn):
                        break
                    h = h - (0.08 / (nn + 1e-12)) * grad
                    h = h / (np.linalg.norm(h) + 1e-18)
                trial[vic] = h
            # small shake of neighbours to accommodate
            trial = trial + 0.005 * rng.standard_normal((n, d))
            trial = _norm(trial)
        # quench
        if r < 0.5:
            trial_q = _adam_quench(trial, steps=500, lr0=0.015, lr1=0.0015, b0=25.0, b1=350.0)
        else:
            trial_q = _adam_quench(trial, steps=650, lr0=0.02, lr1=0.001, b0=15.0, b1=250.0)
        vq = _max_cos(trial_q)
        if vq < best_val - 1e-12:
            # promising: LP polish with short budget
            budget = 1.2 if vq < best_val - 2e-5 else 0.8
            if time.time() + budget + 11.0 > t_end:
                budget = max(0.3, t_end - 11.0 - time.time())
            tdl = min(t_end - 10.0, time.time() + budget)
            if tdl - time.time() > 0.25:
                polished = _lp_polish(trial_q, deadline=tdl, radius_init=1e-3)
                vp = _max_cos(polished)
                push(polished, vp)
                push(trial_q, vq)
                if vp < best_val:
                    Xbest, best_val = polished.copy(), vp
                elif vq < best_val:
                    Xbest, best_val = trial_q.copy(), vq
            else:
                push(trial_q, vq)
                Xbest, best_val = trial_q.copy(), vq
        elif vq < best_val + 0.004:
            push(trial_q, vq)
            # occasional LP on near-best to try to squeeze below
            if it % 7 == 0 and vq < best_val + 0.001 and time.time() < t_end - 16:
                tdl = min(t_end - 10.0, time.time() + 0.6)
                polished = _lp_polish(trial_q, deadline=tdl, radius_init=5e-4)
                vp = _max_cos(polished)
                push(polished, vp)
                if vp < best_val:
                    Xbest, best_val = polished.copy(), vp
        else:
            # keep diversity with small prob
            if rng.random() < 0.05:
                push(trial_q, vq)
        # keep archive bounded, refresh
        if it % 30 == 0 and len(archive) > 3:
            # drop worst diverse that is far above best
            archive.sort(key=lambda t: t[0])
            del archive[6:]

    # final squeeze of global best
    if time.time() < t_end - 1.0:
        dl1 = min(t_end - 0.5, time.time() + max(1.0, t_end - 2.0 - time.time()))
        Xbest = _lp_polish(Xbest, deadline=dl1, radius_init=2e-3)
        if time.time() < t_end - 0.3:
            Xbest = _lp_polish(Xbest, deadline=t_end - 0.1, radius_init=3e-4)
    Xbest = _norm(Xbest)
    return Xbest
