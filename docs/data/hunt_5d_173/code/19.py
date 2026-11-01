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

def _lp_polish(X, seconds):
    X = _norm(np.asarray(X, dtype=np.float64).copy())
    n, d = X.shape
    best = _max_cos(X)
    if seconds <= 0:
        return X
    radius = 1e-3
    start = time.time()
    iu0, iu1 = np.triu_indices(n, 1)
    ar = np.arange(d)
    while time.time() - start < seconds and radius > 1e-14:
        if time.time() - start >= seconds:
            break
        G = X @ X.T
        c = G[iu0, iu1]
        m = float(np.max(c))
        thr = m - max(20.0 * radius, 1e-9)
        sel = c > thr
        if not np.any(sel):
            break
        I = iu0[sel]; J = iu1[sel]
        p = len(I)
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + ar, J[:, None] * d + ar, np.full((p, 1), n * d)], axis=1).ravel()
        vals = np.concatenate([X[J], X[I], -np.ones((p, 1))], axis=1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((X.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c[sel], A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = X + res.x[:-1].reshape(n, d)
        y = _norm(y)
        v = _max_cos(y)
        if v < best:
            X, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return X

def _smooth_opt(X0, beta0, beta1, iters, lr):
    X = np.asarray(X0, dtype=np.float64).copy()
    n, d = X.shape
    iu0, iu1 = np.triu_indices(n, 1)
    M = np.zeros_like(X); V = np.zeros_like(X)
    b1, b2, eps = 0.9, 0.999, 1e-8
    for t in range(1, iters + 1):
        frac = (t - 1) / max(iters - 1, 1)
        beta = beta0 + (beta1 - beta0) * frac
        cur_lr = lr * (1.0 - 0.5 * frac)
        G = X @ X.T
        gmax = float(np.max(G[iu0, iu1]))
        B = np.exp(beta * (G - gmax))
        np.fill_diagonal(B, 0.0)
        Grad = B @ X
        dot = np.sum(Grad * X, axis=1, keepdims=True)
        Gt = Grad - dot * X
        M = b1 * M + (1.0 - b1) * Gt
        V = b2 * V + (1.0 - b2) * (Gt * Gt)
        mhat = M / (1.0 - b1 ** t)
        vhat = V / (1.0 - b2 ** t)
        X = X - cur_lr * mhat / (np.sqrt(vhat) + eps)
        X = _norm(X)
    return X

def _riesz_opt(X0, s, iters, lr, rng):
    X = _norm(np.asarray(X0, dtype=np.float64).copy())
    n, d = X.shape
    M = np.zeros_like(X); V = np.zeros_like(X)
    b1, b2, eps = 0.9, 0.999, 1e-8
    for t in range(1, iters + 1):
        G = X @ X.T
        np.fill_diagonal(G, 1.0)
        r2 = np.clip(2.0 - 2.0 * G, 1e-12, 4.0)
        # r^{-s-2}
        w = r2 ** (-(s + 2.0) / 2.0)
        np.fill_diagonal(w, 0.0)
        w = w * s
        row = np.sum(w, axis=1, keepdims=True)
        Grad = w @ X - row * X
        dot = np.sum(Grad * X, axis=1, keepdims=True)
        Gt = Grad - dot * X
        # clip huge grads
        gn = np.linalg.norm(Gt, axis=1, keepdims=True)
        Gt = Gt / np.maximum(1.0, gn / 10.0)
        M = b1 * M + (1.0 - b1) * Gt
        V = b2 * V + (1.0 - b2) * (Gt * Gt)
        mhat = M / (1.0 - b1 ** t)
        vhat = V / (1.0 - b2 ** t)
        X = X - lr * mhat / (np.sqrt(vhat) + eps)
        X = _norm(X)
    return X

def _contacts_deg(X, thr=1e-7):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    m = float(np.max(G))
    cnt = np.sum(G > m - thr, axis=1)
    return m, cnt, G

def _greedy_void_insert(X_rem, k, rng, P=15000):
    cur = X_rem.copy()
    d = cur.shape[1]
    for _ in range(k):
        C = _norm(rng.standard_normal((P, d)))
        S = C @ cur.T
        smax = np.max(S, axis=1)
        # pick among best 20 for diversity
        idx = np.argpartition(smax, 20)[:20]
        # bias to very best but random
        w = np.exp(-60.0 * (smax[idx] - np.min(smax[idx])))
        w = w / np.sum(w)
        pick = idx[rng.choice(len(idx), p=w)]
        v = C[pick] + 0.005 * rng.standard_normal(d)
        v = v / np.linalg.norm(v)
        cur = np.vstack([cur, v[None, :]])
    return cur

def solve():
    rng = np.random.default_rng(0)
    np.random.seed(0)
    t0 = time.time()
    HARD = 175.0
    n, d = 173, 5
    X_best = None
    best_val = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            A = np.load(fn)
            A = np.asarray(A, dtype=np.float64)
            if A.shape != (n, d):
                continue
            A = _norm(A)
            v = _max_cos(A)
            if v < best_val:
                best_val = v
                X_best = A.copy()
        except Exception:
            continue
    if X_best is None:
        X_best = _norm(rng.standard_normal((n, d)))
        best_val = _max_cos(X_best)
    # population: best + 2 diverse explorers
    pop = [(X_best.copy(), best_val)]
    # make explorers by moderate shake + riesz
    try:
        e1 = _norm(X_best + 0.08 * rng.standard_normal((n, d)))
        e1 = _riesz_opt(e1, 6.0, 400, 0.02, rng)
        e1 = _smooth_opt(e1, 40.0, 120.0, 500, 0.012)
        pop.append((e1, _max_cos(e1)))
    except Exception:
        pass
    try:
        e2 = _norm(rng.standard_normal((n, d)))
        e2 = _riesz_opt(e2, 5.0, 500, 0.02, rng)
        e2 = _smooth_opt(e2, 30.0, 100.0, 500, 0.014)
        pop.append((e2, _max_cos(e2)))
    except Exception:
        pass

    explore_end = t0 + HARD - 30.0
    it = 0
    while time.time() < explore_end:
        it += 1
        if time.time() >= explore_end:
            break
        # pick parent round robin, favour explorers sometimes
        pi = it % max(len(pop), 1)
        if pi >= len(pop):
            pi = 0
        Xp0, vp0 = pop[pi]
        mode = it % 7
        try:
            m0, cnt, G0 = _contacts_deg(Xp0, thr=1e-7)
        except Exception:
            cnt = np.zeros(n)
        Xc = None
        if mode == 0:
            # small jitter exploit best
            base, _ = pop[0]
            Xc = _norm(base + 0.02 * rng.standard_normal((n, d)))
            Xc = _smooth_opt(Xc, 50.0, 160.0, 900, 0.013)
        elif mode == 1:
            # Riesz continuation from random / explorer
            if rng.random() < 0.5:
                Xt = _norm(rng.standard_normal((n, d)))
            else:
                Xt = _norm(Xp0 + 0.15 * rng.standard_normal((n, d)))
            s = float(rng.choice([4.0, 6.0, 8.0, 12.0]))
            Xt = _riesz_opt(Xt, s, 600, 0.02, rng)
            Xc = _smooth_opt(Xt, 40.0, 150.0, 800, 0.014)
        elif mode == 2:
            # overload neighbourhood cap removal
            order = np.argsort(-cnt)
            centre = int(order[0]) if rng.random() < 0.7 else int(rng.integers(0, n))
            xc = Xp0[centre]
            dots = Xp0 @ xc
            k = int(rng.integers(8, 20))
            rem = np.argsort(-dots)[:k]
            mask = np.ones(n, dtype=bool); mask[rem] = False
            X_rem = Xp0[mask]
            Xc = _greedy_void_insert(X_rem, k, rng)
            Xc = _norm(Xc + 0.008 * rng.standard_normal((n, d)))
            Xc = _smooth_opt(Xc, 40.0, 140.0, 900, 0.014)
        elif mode == 3:
            # random pole cap removal
            u = _norm(rng.standard_normal((d,)))
            dots = Xp0 @ u
            k = int(rng.integers(10, 24))
            rem = np.argsort(-dots)[:k]
            mask = np.ones(n, dtype=bool); mask[rem] = False
            X_rem = Xp0[mask]
            Xc = _greedy_void_insert(X_rem, k, rng)
            Xc = _norm(Xc + 0.01 * rng.standard_normal((n, d)))
            # local riesz then sharpen
            Xc = _riesz_opt(Xc, 6.0, 300, 0.015, rng)
            Xc = _smooth_opt(Xc, 40.0, 140.0, 800, 0.014)
        elif mode == 4:
            # drop high-degree, reinsert voids (equilibration)
            k = int(rng.integers(3, 7))
            order = np.argsort(-cnt)
            rem = order[:k]
            # occasionally include rattler
            if rng.random() < 0.4:
                ratt = int(np.argmin(cnt))
                rem = np.unique(np.concatenate([rem[:k-1], [ratt]]))
                while len(rem) < k:
                    rem = np.append(rem, int(rng.integers(0, n)))
                    rem = np.unique(rem)
            mask = np.ones(n, dtype=bool); mask[rem] = False
            X_rem = Xp0[mask]
            Xc = _greedy_void_insert(X_rem, k, rng)
            Xc = _smooth_opt(Xc, 50.0, 160.0, 900, 0.013)
        elif mode == 5:
            # directed repulsion: push overloaded points away from contact centroid
            Xj = Xp0.copy()
            G = Xp0 @ Xp0.T
            np.fill_diagonal(G, -2.0)
            m = float(np.max(G))
            # contacts mask tight
            Cm = G > m - 1e-6
            deg = np.sum(Cm, axis=1)
            hi = np.where(deg >= np.median(deg) + 2)[0]
            for i in hi:
                nbrs = np.where(Cm[i])[0]
                if len(nbrs) == 0:
                    continue
                cent = np.sum(Xp0[nbrs], axis=0)
                # tangent component
                cent_t = cent - np.dot(cent, Xp0[i]) * Xp0[i]
                step = 0.03 + 0.03 * rng.random()
                Xj[i] = Xj[i] - step * cent_t
            # also pull rattler to largest void: sample
            lo = int(np.argmin(deg))
            if deg[lo] <= 1:
                C = _norm(rng.standard_normal((8000, d)))
                S = C @ Xj.T
                sm = np.max(S, axis=1)
                Xj[lo] = C[int(np.argmin(sm))]
            Xj = _norm(Xj)
            Xc = _smooth_opt(Xj, 50.0, 170.0, 900, 0.012)
        else:
            # large kick + riesz + smooth
            Xt = _norm(Xp0 + 0.25 * rng.standard_normal((n, d)))
            Xt = _riesz_opt(Xt, 8.0, 400, 0.02, rng)
            Xc = _smooth_opt(Xt, 30.0, 120.0, 900, 0.016)
        if Xc is None:
            continue
        try:
            vc = _max_cos(Xc)
        except Exception:
            continue
        # update global best (with short LP if near)
        if vc < best_val + 0.004 and vc > best_val - 0.02:
            remain = explore_end - time.time()
            if remain > 1.5:
                budget = 2.0 if vc > best_val + 0.001 else 3.5
                budget = min(budget, remain - 0.2)
                try:
                    Xlp = _lp_polish(Xc, budget)
                    vlp = _max_cos(Xlp)
                    if vlp < vc:
                        Xc, vc = Xlp, vlp
                except Exception:
                    pass
        if vc < best_val:
            X_best = Xc.copy(); best_val = vc
            pop[0] = (X_best.copy(), best_val)
            # keep pop size bounded
            if len(pop) > 4:
                # drop worst
                vals = [v for _, v in pop]
                pop.pop(int(np.argmax(vals)))
        else:
            # accept into pop if not too bad and diverse
            if vc < best_val + 0.012:
                # diversity check: mean dot? just angle distance via correlation
                dup = False
                for (Xq, _) in pop:
                    # align? cheap: compare sorted Gram? use max cos diff + mean abs?
                    # use simple: try optimal rotation via Procrustes? skip, use energy diff
                    if abs(_max_cos(Xq) - vc) < 1e-7:
                        # check configuration distance after alignment by linear assignment is heavy; use degree hist
                        try:
                            _, cq, _ = _contacts_deg(Xq, thr=1e-7)
                            if np.sum(np.abs(np.sort(cq) - np.sort(cnt))) < 5:
                                dup = True
                                break
                        except Exception:
                            pass
                if not dup:
                    pop.append((Xc.copy(), vc))
                    if len(pop) > 4:
                        # keep best + 3 most diverse (drop median worst not best)
                        # never drop index 0
                        worst = 1 + int(np.argmax([v for _, v in pop[1:]]))
                        pop.pop(worst)
        # time guard
        if time.time() >= explore_end:
            break
    # final intensive polish
    elapsed = time.time() - t0
    rem = HARD - (time.time() - t0)
    rem = float(np.clip(rem, 3.0, 29.0))
    if elapsed < 178.0:
        try:
            X_best = _lp_polish(X_best, min(rem, 178.0 - (time.time() - t0) - 0.5))
        except Exception:
            pass
    X_best = _norm(X_best)
    return X_best
