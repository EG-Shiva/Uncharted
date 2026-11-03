import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _norm(X):
    n = np.linalg.norm(X, axis=1, keepdims=True) + 1e-30
    return X / n

def _rand_sphere(rng, n, d):
    return _norm(rng.standard_normal((n, d)))

def _adam_anneal(X0, beta0, beta1, lr0, lr1, steps, lam0, lam1, s_pow, rng, deadline, noise=0.0):
    n, d = X0.shape
    Y = X0.copy()
    m1 = np.zeros_like(Y)
    v1 = np.zeros_like(Y)
    b1, b2, eps = 0.9, 0.999, 1e-8
    t = 0
    for s in range(steps):
        if time.time() > deadline:
            break
        f = s / max(steps - 1, 1)
        beta = beta0 + (beta1 - beta0) * f
        lr = lr0 + (lr1 - lr0) * f
        lam = lam0 + (lam1 - lam0) * f
        nn = np.sqrt(np.sum(Y * Y, axis=1, keepdims=True)) + 1e-30
        X = Y / nn
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        mx = float(G.max())
        E = np.exp(beta * (G - mx))
        np.fill_diagonal(E, 0.0)
        S = float(E.sum() / 2.0) + 1e-300
        Grad = (E / S) @ X
        if lam > 0:
            # Riesz spreading
            D = 2.0 - 2.0 * G
            np.fill_diagonal(D, 1.0)
            D = np.maximum(D, 1e-6)
            P = D ** (-(s_pow / 2.0 + 1.0))
            np.fill_diagonal(P, 0.0)
            GradR = (P @ X)
            # scale to comparable rms
            rL = float(np.sqrt(np.mean(Grad * Grad))) + 1e-30
            rR = float(np.sqrt(np.mean(GradR * GradR))) + 1e-30
            Grad = Grad + lam * (rL / rR) * GradR
        dot = np.sum(Grad * X, axis=1, keepdims=True)
        GradX = Grad - dot * X
        GradY = GradX / nn
        # clip
        gn = np.sqrt(np.mean(GradY * GradY)) + 1e-30
        if gn > 10.0:
            GradY *= (10.0 / gn)
        t += 1
        m1 = b1 * m1 + (1 - b1) * GradY
        v1 = b2 * v1 + (1 - b2) * GradY * GradY
        mhat = m1 / (1 - b1 ** t)
        vhat = v1 / (1 - b2 ** t)
        Y -= lr * mhat / (np.sqrt(vhat) + eps)
        if noise > 0:
            amp = noise * (1.0 - f)
            N = rng.standard_normal((n, d))
            # project to tangent of X
            Nd = np.sum(N * X, axis=1, keepdims=True)
            N -= Nd * X
            Y += amp * N * np.sqrt(np.sum(Y * Y, axis=1, keepdims=True)).mean() / 8.0
    return _norm(Y)

def _effective_counts(X, thr=5e-4):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    m = float(G.max())
    C = (G > m - thr).sum(axis=1)
    # also max per point
    Mp = G.max(axis=1)
    return C, Mp, m, G

def _greedy_reinsert(Xkeep, k_add, rng, pool=4000):
    n0, d = Xkeep.shape
    if k_add <= 0:
        return Xkeep.copy()
    P = _rand_sphere(rng, pool, d)
    # score pool by max cos to keep
    if n0 > 0:
        S = P @ Xkeep.T
        smax = S.max(axis=1)
    else:
        smax = np.zeros(pool)
    chosen = []
    sel_mask = np.ones(pool, dtype=bool)
    cur = Xkeep.copy()
    for _ in range(k_add):
        # pick best available (lowest max)
        # consider also repulsion among chosen via cur
        if cur.shape[0] > 0:
            S2 = P @ cur.T
            m2 = S2.max(axis=1)
        else:
            m2 = smax
        m2 = np.where(sel_mask, m2, np.inf)
        idx = int(np.argmin(m2))
        chosen.append(P[idx].copy())
        sel_mask[idx] = False
        cur = np.vstack([cur, P[idx:idx+1]])
    if len(chosen) == 0:
        return Xkeep.copy()
    return np.vstack([Xkeep] + [c[None, :] for c in chosen])

def _peel_reinsert(X, k_remove, rng, thr=5e-4):
    n, d = X.shape
    C, Mp, m, G = _effective_counts(X, thr=thr)
    # overload score: combine count and max
    # remove top k_remove by (C, Mp)
    order = np.lexsort((Mp, C))  # ascending
    victims = order[n - k_remove:]
    keep_idx = [i for i in range(n) if i not in set(victims.tolist())]
    Xkeep = X[keep_idx]
    Xnew = _greedy_reinsert(Xkeep, k_remove, rng)
    return _norm(Xnew)

def _tangent_kick(X, sigma, rng, idx=None):
    N = rng.standard_normal(X.shape)
    dot = np.sum(N * X, axis=1, keepdims=True)
    N -= dot * X
    if idx is not None:
        M = np.zeros((X.shape[0], 1))
        M[idx] = 1.0
        N = N * (0.25 + 0.75 * M) if False else N
        # scale per point
        S = np.full((X.shape[0], 1), 0.25 * sigma)
        S[idx] = sigma
        Y = X + S * N
    else:
        Y = X + sigma * N
    return _norm(Y)

def _lp_polish(x, deadline):
    x = _norm(x)
    n, d = x.shape
    try:
        best = _max_cos(x)
    except Exception:
        return x
    radius = 1e-3
    iu = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        Gfull = x @ x.T
        c = Gfull[iu]
        m = float(c.max())
        thr = m - max(20.0 * radius, 1e-9)
        sel = c > thr
        if int(np.count_nonzero(sel)) == 0:
            break
        if int(np.count_nonzero(sel)) > 2500:
            idx = np.where(sel)[0]
            order = np.argsort(-c[idx])[:2500]
            mask = np.zeros_like(sel, dtype=bool)
            mask[idx[order]] = True
            sel = mask
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        if p == 0:
            break
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv); obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-c[sel], A_eq=A_eq, b_eq=np.zeros(n),
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
            x, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def solve():
    t0 = time.time()
    deadline = t0 + 168.0
    n, d = 138, 8
    rng = np.random.default_rng(7)
    best = None
    best_val = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            vv = np.load(fn).astype(float)
            if vv.shape == (n, d):
                vv = _norm(vv)
                c = _max_cos(vv)
                if c < best_val:
                    best_val = c
                    best = vv.copy()
        except Exception:
            pass
    if best is None:
        best = _rand_sphere(rng, n, d)
        best_val = _max_cos(best)
    incumbent = best.copy()
    # candidate pool for final polish
    pool = [(best_val, best.copy())]

    # fast initial anneal of incumbent with spreading to balance a bit
    # basin hopping
    hop_deadline = t0 + 128.0
    it = 0
    # precompute strategies cycle
    while time.time() < hop_deadline:
        it += 1
        mode = it % 6
        try:
            if mode == 0:
                # small global kick + anneal
                X0 = _tangent_kick(incumbent, 0.06 + 0.04 * rng.random(), rng)
                X1 = _adam_anneal(X0, 50.0, 500.0, 0.02, 0.004, 650, 0.08, 0.0, 2.0, rng, hop_deadline, noise=0.0)
            elif mode == 1:
                # targeted overload kick
                C, Mp, m, G = _effective_counts(incumbent, thr=8e-4)
                # overloaded: top 20 by C
                order = np.argsort(C)
                over = order[-20:]
                # rattlers: C==0
                X0 = incumbent.copy()
                N = rng.standard_normal((n, d))
                dot = np.sum(N * X0, axis=1, keepdims=True)
                N -= dot * X0
                sig = np.full((n, 1), 0.04)
                sig[over] = 0.30
                # rattlers get medium kick to voids via random reinit for a few
                rat = np.where(C == 0)[0]
                if len(rat) > 0:
                    sig[rat] = 0.18
                X0 = _norm(X0 + sig * N)
                X1 = _adam_anneal(X0, 60.0, 600.0, 0.018, 0.004, 700, 0.05, 0.0, 3.0, rng, hop_deadline, noise=0.0)
            elif mode == 2:
                # peel overloaded + greedy reinsert
                k = int(rng.integers(8, 19))
                thr = float(rng.choice([3e-4, 8e-4, 2e-3]))
                X0 = _peel_reinsert(incumbent, k, rng, thr=thr)
                # small kick after reinsert
                X0 = _tangent_kick(X0, 0.03, rng)
                X1 = _adam_anneal(X0, 50.0, 550.0, 0.02, 0.004, 750, 0.06, 0.0, 2.0, rng, hop_deadline, noise=0.0)
            elif mode == 3:
                # peel random cluster: pick random point + its nearest neighbours
                G = incumbent @ incumbent.T
                np.fill_diagonal(G, -2.0)
                seed = int(rng.integers(0, n))
                nn_order = np.argsort(-G[seed])[:12]
                k = int(rng.integers(10, 18))
                victims = list(nn_order[:k])
                keep = [i for i in range(n) if i not in set(victims)]
                Xkeep = incumbent[keep]
                X0 = _greedy_reinsert(Xkeep, len(victims), rng, pool=5000)
                X0 = _tangent_kick(X0, 0.02, rng)
                X1 = _adam_anneal(X0, 60.0, 600.0, 0.018, 0.003, 750, 0.04, 0.0, 2.0, rng, hop_deadline, noise=0.0)
            elif mode == 4:
                # larger global kick to change basin
                X0 = _tangent_kick(best, 0.25 + 0.2 * rng.random(), rng)
                X1 = _adam_anneal(X0, 40.0, 500.0, 0.022, 0.004, 650, 0.10, 0.0, 1.5, rng, hop_deadline, noise=0.0)
            else:
                # random fresh basin with short anneal (E8-flavoured: sample from best + uniform mix)
                if rng.random() < 0.5:
                    X0 = _rand_sphere(rng, n, d)
                else:
                    # splice: half best half random
                    X0 = incumbent.copy()
                    sel = rng.choice(n, size=n // 3, replace=False)
                    X0[sel] = _rand_sphere(rng, len(sel), d)
                    X0 = _norm(X0)
                X1 = _adam_anneal(X0, 40.0, 450.0, 0.022, 0.005, 600, 0.10, 0.0, 2.0, rng, hop_deadline, noise=0.0)
        except Exception:
            continue
        try:
            v = _max_cos(X1)
        except Exception:
            continue
        # keep pool of top 4
        pool.append((v, X1.copy()))
        pool.sort(key=lambda t: t[0])
        pool = pool[:4]
        if v < best_val - 1e-12:
            best_val = v
            best = X1.copy()
            incumbent = X1.copy()
        elif v < _max_cos(incumbent) + 0.002 and rng.random() < 0.35:
            # accept move to different basin occasionally
            incumbent = X1.copy()
        elif rng.random() < 0.08:
            incumbent = best.copy()
            # diversify incumbent slightly
            incumbent = _tangent_kick(incumbent, 0.10, rng)
        if time.time() > hop_deadline:
            break

    # final polish top candidates
    pool.sort(key=lambda t: t[0])
    # deduplicate roughly: keep best 3
    final_deadline = deadline - 2.0
    # allocate LP time: most to best, some to second
    cands = [p[1] for p in pool[:3]]
    polished = []
    for ci, Xc in enumerate(cands):
        if time.time() > final_deadline - 1.0:
            break
        # short re-anneal at high beta before LP for best precision
        try:
            remain = final_deadline - time.time()
            if remain > 12 and ci == 0:
                Xc2 = _adam_anneal(Xc, 300.0, 900.0, 0.006, 0.0015, 400, 0.0, 0.0, 2.0, rng, final_deadline, noise=0.0)
            else:
                Xc2 = Xc
            # LP budget
            if ci == 0:
                lp_budget = final_deadline - time.time()
            else:
                lp_budget = min(12.0, (final_deadline - time.time()) / 2.0)
            if lp_budget > 2.0:
                Xp = _lp_polish(Xc2, time.time() + lp_budget)
            else:
                Xp = Xc2
            v = _max_cos(Xp)
            polished.append((v, Xp))
            if v < best_val:
                best_val = v
                best = Xp.copy()
        except Exception:
            continue
    # safety
    best = _norm(best)
    # tiny shrink not needed (rows already unit); ensure finite
    best = np.where(np.isfinite(best), best, 0.0)
    # renormalize after fix
    nr = np.linalg.norm(best, axis=1, keepdims=True)
    nr = np.where(nr < 1e-12, 1.0, nr)
    best = best / nr
    return best
