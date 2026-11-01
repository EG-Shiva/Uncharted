import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _norm_rows(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[~np.isfinite(n)] = 1.0
    n[n == 0] = 1.0
    return X / n

def _lp_polish(X, seconds, deadline=None):
    X = _norm_rows(np.asarray(X, dtype=float).copy())
    n, d = X.shape
    try:
        best = _max_cos(X)
    except Exception:
        return X
    radius = 2e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds:
        if deadline is not None and time.time() > deadline:
            break
        if radius < 1e-14:
            break
        G = X @ X.T
        c = G[iu]
        m = float(c.max())
        sel = c > m - max(20.0 * radius, 1e-9)
        I = iu[0][sel]; J = iu[1][sel]
        cs = c[sel]
        p = len(I)
        if p == 0:
            break
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([X[J], X[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((X.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv); obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-cs, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = X + res.x[:-1].reshape(n, d)
        y = _norm_rows(y)
        if not np.all(np.isfinite(y)):
            radius *= 0.5
            continue
        v = _max_cos(y)
        if v < best:
            X, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return X

def _quench(X_init, steps, T0, T1, lr0, lr1, rng=None, noise=0.0):
    X = np.asarray(X_init, dtype=float).copy()
    n, d = X.shape
    m1 = np.zeros_like(X)
    v1 = np.zeros_like(X)
    b1, b2, eps = 0.9, 0.999, 1e-8
    lT0, lT1 = np.log(T0), np.log(T1)
    ll0, ll1 = np.log(lr0), np.log(lr1)
    G0 = X @ X.T
    np.fill_diagonal(G0, -2.0)
    best_m = float(G0.max())
    best_X = X.copy()
    for s in range(1, steps + 1):
        frac = (s - 1) / max(steps - 1, 1)
        T = float(np.exp(lT0 * (1 - frac) + lT1 * frac))
        lr = float(np.exp(ll0 * (1 - frac) + ll1 * frac))
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        if m < best_m:
            best_m = m
            best_X = X.copy()
        D = (G - m) / T
        E = np.exp(D)
        np.fill_diagonal(E, 0.0)
        S = float(E.sum() * 0.5)
        if not np.isfinite(S) or S <= 0:
            S = 1.0
        Gr = (E @ X) / S
        dot = np.sum(Gr * X, axis=1, keepdims=True)
        Gt = Gr - dot * X
        if rng is not None and noise > 0:
            N = rng.standard_normal((n, d)) * noise
            N -= X * np.sum(N * X, axis=1, keepdims=True)
            Gt = Gt + N
        if not np.all(np.isfinite(Gt)):
            break
        m1 = b1 * m1 + (1 - b1) * Gt
        v1 = b2 * v1 + (1 - b2) * (Gt * Gt)
        c1 = 1.0 - b1 ** s
        c2 = 1.0 - b2 ** s
        step = (m1 / c1) / (np.sqrt(v1 / c2) + eps)
        sn = float(np.linalg.norm(step))
        mxn = 8.0 * np.sqrt(n * d)
        if sn > mxn:
            step *= (mxn / sn)
        X = X - lr * step
        X = _norm_rows(X)
    Gf = X @ X.T
    np.fill_diagonal(Gf, -2.0)
    mf = float(Gf.max())
    if mf < best_m:
        best_m = mf
        best_X = X.copy()
    return best_X

def _greedy_hole(Y, rng, n_try=3000):
    d = Y.shape[1]
    C = rng.standard_normal((n_try, d))
    C = _norm_rows(C)
    D = C @ Y.T
    w = D.max(axis=1)
    return C[np.argmin(w)]

def _refine_single(cand, Y):
    cc = cand.reshape(1, -1).copy()
    cc = _norm_rows(cc)
    for s in range(180):
        T = 0.004 * (0.995 ** s) + 0.0003
        lr = 0.02 * (0.99 ** s) + 0.002
        dots = (cc @ Y.T).ravel()
        mm = float(dots.max())
        E = np.exp((dots - mm) / T)
        E /= (E.sum() + 1e-30)
        g = (E @ Y).reshape(1, -1)
        g = g - cc * float((g * cc).sum())
        cc = cc - lr * g
        cc = _norm_rows(cc)
    return cc.ravel() / (np.linalg.norm(cc.ravel()) + 1e-30)

def _relocate_try(X, cur_max, rng):
    n, d = X.shape
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    pmx = G.max(axis=1)
    order = np.argsort(-pmx)
    for ii in range(min(n, 6)):
        i = int(order[ii])
        mask = np.ones(n, dtype=bool); mask[i] = False
        Y = X[mask]
        Go = Y @ Y.T
        np.fill_diagonal(Go, -2.0)
        others_max = float(Go.max()) if Go.size else -2.0
        if others_max >= cur_max - 1e-12:
            continue
        C = _norm_rows(rng.standard_normal((2500, d)))
        D = C @ Y.T
        w = D.max(axis=1)
        bi = int(np.argmin(w))
        if float(w[bi]) >= cur_max:
            continue
        cand = _refine_single(C[bi], Y)
        dots = (Y @ cand).ravel()
        bv = float(dots.max())
        if bv < cur_max - 1e-7:
            Xn = X.copy()
            Xn[i] = cand
            Xn = _norm_rows(Xn)
            nm = _max_cos(Xn)
            if nm < cur_max:
                return Xn, nm, True
    return X, cur_max, False

def solve():
    t0 = time.time()
    HARD_END = t0 + 174.0
    FINAL_POLISH = 22.0
    HOP_END = HARD_END - FINAL_POLISH - 1.0
    rng = np.random.default_rng(71771)
    best_X = None
    try:
        best_X = np.load("best.npy").astype(float)
    except Exception:
        pass
    if best_X is None or best_X.shape != (71, 12) or not np.all(np.isfinite(best_X)):
        try:
            best_X = np.load("record.npy").astype(float)
        except Exception:
            best_X = rng.standard_normal((71, 12))
    best_X = _norm_rows(np.asarray(best_X, dtype=float))
    best_score = _max_cos(best_X)
    n, d = 71, 12
    curX = best_X.copy()
    curVal = best_score
    archive = [(best_score, best_X.copy())]
    sigmas = [0.015, 0.04, 0.09, 0.20, 0.32, 0.06, 0.12]
    it = 0
    accept_T = 0.0012
    while time.time() < HOP_END:
        it += 1
        if time.time() > HOP_END - 2.0:
            break
        # pick base: mostly best, sometimes cur/archive
        rr = rng.random()
        if rr < 0.65:
            base = best_X
        elif rr < 0.85:
            base = curX
        else:
            base = archive[int(rng.integers(len(archive)))][1]
        Gc = base @ base.T
        np.fill_diagonal(Gc, -2.0)
        pmx = Gc.max(axis=1)
        cnt = (Gc > best_score - 2e-4).sum(axis=1)
        order_worst = np.argsort(-pmx - 0.0005 * cnt)
        r = rng.random()
        X0 = base.copy()
        if r < 0.32:
            sigma = sigmas[it % len(sigmas)]
            noise = rng.standard_normal((n, d))
            w = np.ones(n)
            w[order_worst[:12]] *= 2.2
            noise *= (sigma * w[:, None])
            noise -= X0 * np.sum(noise * X0, axis=1, keepdims=True)
            X0 = _norm_rows(X0 + noise)
            steps = 900 if sigma < 0.1 else 1200
            T0 = 0.012 if sigma < 0.1 else 0.018
            Xq = _quench(X0, steps, T0, 0.00035, 0.025, 0.0025, rng if sigma > 0.05 else None, noise=0.0005 if sigma > 0.1 else 0.0)
        elif r < 0.55:
            k = int(rng.integers(2, 6))
            kill = order_worst[:k].copy()
            if rng.random() < 0.25:
                kill = rng.choice(n, size=k, replace=False)
            for idx in kill:
                mask = np.ones(n, dtype=bool); mask[idx] = False
                Y = X0[mask]
                X0[idx] = _greedy_hole(Y, rng, n_try=3000)
            X0 = _norm_rows(X0)
            Xq = _quench(X0, 1100, 0.016, 0.0003, 0.03, 0.002, rng, noise=0.0008)
        elif r < 0.70:
            # worst-pair hole surgery
            iu = np.triu_indices(n, 1)
            Gf = X0 @ X0.T
            gv = Gf[iu]
            bi = int(np.argmax(gv))
            a = int(iu[0][bi]); b = int(iu[1][bi])
            # move the one with larger second-best
            ra = np.partition(Gf[a], -3)[-3]
            rb = np.partition(Gf[b], -3)[-3]
            mv = a if ra > rb else b
            if rng.random() < 0.3:
                mv = a if rng.random() < 0.5 else b
            mask = np.ones(n, dtype=bool); mask[mv] = False
            Y = X0[mask]
            hole = _greedy_hole(Y, rng, n_try=4000)
            hole = _refine_single(hole, Y)
            X0[mv] = hole
            # small jitter to neighbours of the pair
            nbr = np.where(Gf[mv] > best_score - 0.003)[0]
            if len(nbr) > 0:
                jn = nbr[:8]
                X0[jn] = _norm_rows(X0[jn] + rng.standard_normal((len(jn), d)) * 0.02)
            X0 = _norm_rows(X0)
            Xq = _quench(X0, 1100, 0.014, 0.0003, 0.028, 0.002, rng, noise=0.0006)
        elif r < 0.85:
            # coherent cluster shake around most overloaded point
            thr = best_score - 0.004
            adj = (Gc > thr)
            np.fill_diagonal(adj, False)
            seed = int(order_worst[0])
            # BFS component
            seen = {seed}
            stack = [seed]
            while stack:
                u = stack.pop()
                nb = np.where(adj[u])[0]
                for v in nb:
                    if int(v) not in seen and len(seen) < 24:
                        seen.add(int(v))
                        stack.append(int(v))
            comp = np.array(sorted(seen), dtype=int)
            direction = rng.standard_normal((1, d))
            direction -= X0[comp[:1]] * np.sum(direction * X0[comp[:1]], axis=1, keepdims=True)
            dn = np.linalg.norm(direction) + 1e-12
            direction /= dn
            shift = float(rng.uniform(0.10, 0.28))
            X0[comp] = _norm_rows(X0[comp] + shift * direction + rng.standard_normal((len(comp), d)) * 0.04)
            # light jitter elsewhere
            X0 = _norm_rows(X0 + rng.standard_normal((n, d)) * 0.004 - X0 * 0.0)
            X0 = _norm_rows(X0)
            Xq = _quench(X0, 1250, 0.019, 0.0004, 0.034, 0.003, rng, noise=0.001)
        else:
            # partial restart from archive/best keeping 60
            keep = rng.choice(n, size=60, replace=False)
            X0 = base.copy()
            complement = np.array([i for i in range(n) if i not in set(keep.tolist())])
            rng.shuffle(complement)
            for idx in complement:
                mask = np.ones(n, dtype=bool); mask[idx] = False
                Y = X0[mask]
                X0[idx] = _greedy_hole(Y, rng, n_try=2000)
            X0 = _norm_rows(X0)
            Xq = _quench(X0, 1350, 0.022, 0.0004, 0.04, 0.003, rng, noise=0.0012)
        try:
            qscore = _max_cos(Xq)
        except Exception:
            continue
        if qscore < best_score + 0.004 and time.time() < HOP_END - 3.0:
            try:
                Xq2, q2, _ = _relocate_try(Xq, qscore, rng)
                # second pass if improved
                if q2 < qscore:
                    Xq3, q3, _ = _relocate_try(Xq2, q2, rng)
                    Xq, qscore = Xq3, q3
                else:
                    Xq, qscore = Xq2, q2
            except Exception:
                pass
        if qscore < best_score + 0.0006 and (it % 6 == 0):
            try:
                Xq = _lp_polish(Xq, 0.9, deadline=HOP_END)
                qscore = _max_cos(Xq)
            except Exception:
                pass
        if qscore < best_score - 1e-12:
            best_X, best_score = Xq.copy(), qscore
            archive.append((best_score, best_X.copy()))
            if len(archive) > 6:
                archive = sorted(archive, key=lambda z: z[0])[:6]
            curX, curVal = Xq.copy(), qscore
        else:
            # Metropolis accept for exploration
            if qscore < curVal or rng.random() < np.exp(-(qscore - curVal) / accept_T):
                curX, curVal = Xq.copy(), qscore
            # occasional restart cur to best
            if rng.random() < 0.05:
                curX, curVal = best_X.copy(), best_score
        if rng.random() < 0.12 and time.time() < HOP_END - 5:
            try:
                Xf = _quench(best_X, 450, 0.002, 0.00015, 0.008, 0.001)
                sf = _max_cos(Xf)
                if sf < best_score:
                    best_X, best_score = Xf, sf
                    archive.append((sf, Xf.copy()))
            except Exception:
                pass
    try:
        if time.time() < HARD_END - 2:
            Xf = _quench(best_X, 700, 0.003, 0.00012, 0.010, 0.0012)
            sf = _max_cos(Xf)
            if sf < best_score:
                best_X, best_score = Xf, sf
    except Exception:
        pass
    # guarded archive tightening
    try:
        arch_sorted = sorted(archive, key=lambda z: z[0])[:3]
        for vv, XX in arch_sorted:
            if time.time() > HARD_END - 8:
                break
            try:
                Xc = _quench(XX, 500, 0.0015, 0.0001, 0.006, 0.001)
                sc = _max_cos(Xc)
                if sc < best_score:
                    best_X, best_score = Xc, sc
            except Exception:
                pass
    except Exception:
        pass
    try:
        remain = HARD_END - time.time() - 0.5
        if remain > 3:
            Xp = _lp_polish(best_X, remain, deadline=HARD_END - 0.3)
            sp = _max_cos(Xp)
            if sp < best_score:
                best_X = Xp
    except Exception:
        pass
    best_X = _norm_rows(best_X)
    return best_X
