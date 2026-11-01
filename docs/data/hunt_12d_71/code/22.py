import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _norm_rows(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[~np.isfinite(n)] = 1.0
    n[n == 0] = 1.0
    return X / n

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _quench(X_init, steps, T0, T1, lr0, lr1, rng=None, noise=0.0):
    X = np.array(X_init, dtype=float, copy=True)
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
        # softmax weights, guard overflow
        # E = exp((G-m)/T), diag 0
        D = (G - m) / T
        # clip very negative to avoid underflow cost (still fine)
        np.clip(D, -50, 0, out=D)
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

def _lp_polish(X, seconds, deadline=None):
    X = _norm_rows(np.array(X, dtype=float, copy=True))
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

def _refine_cand(cc, Y):
    cc = cc.reshape(1, -1).copy()
    cc = _norm_rows(cc)
    for s in range(160):
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
    return (cc.ravel() / (np.linalg.norm(cc.ravel()) + 1e-30))

def _relocate_try(X, cur_max, rng):
    n, d = X.shape
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    pmx = G.max(axis=1)
    order = np.argsort(-pmx)
    for ii in range(min(n, 5)):
        i = int(order[ii])
        mask = np.ones(n, dtype=bool); mask[i] = False
        Y = X[mask]
        Go = Y @ Y.T
        np.fill_diagonal(Go, -2.0)
        others_max = float(Go.max()) if Go.size else -2.0
        if others_max >= cur_max - 1e-12:
            continue
        C = rng.standard_normal((2500, d))
        C = _norm_rows(C)
        D = C @ Y.T
        w = D.max(axis=1)
        bi = int(np.argmin(w))
        bv = float(w[bi])
        if bv >= cur_max - 1e-12:
            continue
        cand = _refine_cand(C[bi], Y)
        bv2 = float((Y @ cand).max())
        if bv2 >= cur_max - 2e-7:
            continue
        Xn = X.copy()
        Xn[i] = cand
        nm = _max_cos(Xn)
        if nm < cur_max - 1e-12:
            X = Xn
            cur_max = nm
    return X, cur_max

def solve():
    t0 = time.time()
    HARD_END = t0 + 173.0
    FINAL_POLISH = 20.0
    HOP_END = HARD_END - FINAL_POLISH - 1.0
    rng = np.random.default_rng(7112026)
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
    cur_X = best_X.copy()
    cur_score = best_score
    it = 0
    sigmas = [0.015, 0.04, 0.09, 0.20, 0.32, 0.06, 0.12]
    while time.time() < HOP_END:
        it += 1
        if time.time() > HOP_END - 2.0:
            break
        # Metropolis temperature anneal
        frac_time = (time.time() - t0) / max(HOP_END - t0, 1e-9)
        Tmet = 0.0012 * (1 - frac_time) + 0.00008 * frac_time
        r = rng.random()
        base = cur_X if rng.random() < 0.65 else best_X
        # per-point stats on base
        Gb = base @ base.T
        np.fill_diagonal(Gb, -2.0)
        pmx = Gb.max(axis=1)
        cnt = (Gb > cur_score - 5e-4).sum(axis=1) if False else (Gb > float(Gb.max()) - 5e-4).sum(axis=1)
        order_worst = np.argsort(-pmx - 0.0003 * cnt)
        X0 = base.copy()
        if r < 0.40:
            # worst-pair targeted shake: jammed pair + overloaded
            sigma = sigmas[it % len(sigmas)]
            # find worst pair
            ij = int(np.argmax(Gb)) if False else None
            # use triu argmax
            iu_tmp = np.triu_indices(n, 1)
            # Gb already has -2 diag, max over all includes diag? diag -2 so fine
            flat = Gb[Angular if False else iu_tmp] if False else Gb[iu_tmp]
            k = int(np.argmax(flat))
            wi, wj = int(iu_tmp[0][k]), int(iu_tmp[1][k])
            noise = rng.standard_normal((n, d)) * (sigma * 0.4)
            # amplify worst points
            w = np.ones(n)
            k2 = 8
            w[order_worst[:k2]] *= 2.4
            w[wi] *= 2.0; w[wj] *= 2.0
            noise *= w[:, None] / 0.4 if False else w[:, None]
            # actually noise already scaled; rescale:
            # noise = base scale sigma*0.4 * w -> redo simply:
            noise = rng.standard_normal((n, d)) * (sigma * w[:, None])
            noise -= X0 * np.sum(noise * X0, axis=1, keepdims=True)
            X0 = X0 + noise
            X0 = _norm_rows(X0)
            steps = 900 if sigma < 0.1 else 1150
            T0 = 0.012 if sigma < 0.1 else 0.018
            Xq = _quench(X0, steps, T0, 0.00035, 0.025, 0.0025,
                         rng if sigma > 0.05 else None, noise=0.0005 if sigma > 0.1 else 0.0)
        elif r < 0.68:
            # kill-and-greedy-refill worst points
            k = int(rng.integers(2, 6))
            kill = order_worst[:k].copy()
            if rng.random() < 0.25:
                kill = rng.choice(n, size=k, replace=False)
            X0[kill] = _norm_rows(rng.standard_normal((k, d)))
            for idx in kill:
                mask = np.ones(n, dtype=bool); mask[idx] = False
                Y = X0[mask]
                C = _norm_rows(rng.standard_normal((3000, d)))
                D = C @ Y.T
                wv = D.max(axis=1)
                X0[idx] = C[np.argmin(wv)]
            X0 = _norm_rows(X0)
            Xq = _quench(X0, 1050, 0.016, 0.0003, 0.03, 0.002, rng, noise=0.0008)
        elif r < 0.86:
            # coherent cap move + cluster shake
            u = _norm_rows(rng.standard_normal((1, d))).ravel()
            proj = X0 @ u
            thr = np.quantile(proj, rng.uniform(0.6, 0.9))
            idx = np.where(proj >= thr)[0]
            if len(idx) < 3:
                idx = rng.choice(n, size=int(rng.integers(6, 16)), replace=False)
            shift = rng.standard_normal(d)
            shift -= u * float(shift @ u)
            shift /= (np.linalg.norm(shift) + 1e-12)
            mag = rng.uniform(0.15, 0.45)
            X0[idx] = X0[idx] + mag * shift[None, :] + rng.standard_normal((len(idx), d)) * 0.08
            # also jitter a few random
            jr = rng.choice(n, size=4, replace=False)
            X0[jr] += rng.standard_normal((4, d)) * 0.1
            X0 = _norm_rows(X0)
            Xq = _quench(X0, 1200, 0.019, 0.00035, 0.032, 0.0025, rng, noise=0.001)
        else:
            # partial fresh refill keeping 60 best
            keep = rng.choice(n, size=60, replace=False)
            X0 = best_X.copy()
            comp = np.array([i for i in range(n) if i not in set(keep.tolist())])
            for idx in comp:
                mask = np.ones(n, dtype=bool); mask[idx] = False
                Y = X0[mask]
                C = _norm_rows(rng.standard_normal((2000, d)))
                D = C @ Y.T
                wv = D.max(axis=1)
                X0[idx] = C[np.argmin(wv)]
            X0 = _norm_rows(X0)
            Xq = _quench(X0, 1300, 0.021, 0.0004, 0.038, 0.003, rng, noise=0.0012)
        try:
            qscore = _max_cos(Xq)
        except Exception:
            continue
        if qscore < best_score + 0.004 and time.time() < HOP_END - 3.0:
            try:
                Xq, qscore = _relocate_try(Xq, qscore, rng)
            except Exception:
                pass
        if qscore < best_score + 0.0005 and (it % 6 == 0):
            try:
                Xq = _lp_polish(Xq, 1.0, deadline=HOP_END)
                qscore = _max_cos(Xq)
            except Exception:
                pass
        if qscore < best_score - 1e-12:
            best_X, best_score = Xq, qscore
            cur_X, cur_score = Xq, qscore
        else:
            # Metropolis accept for exploration
            if qscore < cur_score or rng.random() < np.exp((cur_score - qscore) / max(Tmet, 1e-9)):
                cur_X, cur_score = Xq, qscore
        if rng.random() < 0.12 and time.time() < HOP_END - 5:
            try:
                Xf = _quench(best_X, 450, 0.002, 0.00015, 0.008, 0.001)
                sf = _max_cos(Xf)
                if sf < best_score:
                    best_X, best_score = Xf, sf
                    cur_X, cur_score = Xf, sf
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
