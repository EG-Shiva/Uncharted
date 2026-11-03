import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def rowwise_max(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return G.max(axis=1), G

def lp_snap(x, deadline, cap=2500):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    try:
        best = max_cos(x)
    except Exception:
        return x
    radius = 2e-3
    iu = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        Gfull = x @ x.T
        c = Gfull[iu]
        m = float(c.max())
        thr = m - max(20.0 * radius, 1e-9)
        sel = c > thr
        cnt = int(np.count_nonzero(sel))
        if cnt == 0:
            break
        if cnt > cap:
            idx = np.where(sel)[0]
            order = np.argpartition(-c[idx], cap)[:cap]
            mask = np.zeros_like(sel, dtype=bool)
            mask[idx[order]] = True
            sel = mask
        I, J = iu[0][sel], iu[1][sel]
        p = int(np.count_nonzero(sel))
        csel = c[sel]
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
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
        yn = np.linalg.norm(y, axis=1, keepdims=True)
        yn[yn == 0] = 1.0
        y = y / yn
        try:
            v = max_cos(y)
        except Exception:
            radius *= 0.5
            continue
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def adam_quench(X, k0, k1, steps, lr0, deadline):
    n, d = X.shape
    m1 = np.zeros_like(X)
    m2 = np.zeros_like(X)
    b1 = 0.9; b2 = 0.999; eps = 1e-8
    b1p = 1.0; b2p = 1.0
    for t in range(1, steps + 1):
        if (t & 63) == 0 and time.time() > deadline:
            break
        frac = t / steps
        k = k0 + (k1 - k0) * frac
        lr = lr0 * (0.15 + 0.85 * 0.5 * (1.0 + np.cos(np.pi * frac)))
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        M = float(G.max())
        E = np.exp(k * (G - M))
        np.fill_diagonal(E, 0.0)
        S = float(E.sum() * 0.5)
        if not np.isfinite(S) or S <= 0:
            break
        P = E / S
        grad = P @ X
        dot = np.sum(grad * X, axis=1, keepdims=True)
        g = grad - dot * X
        b1p *= b1; b2p *= b2
        m1 = b1 * m1 + (1.0 - b1) * g
        m2 = b2 * m2 + (1.0 - b2) * (g * g)
        m1h = m1 / (1.0 - b1p)
        m2h = m2 / (1.0 - b2p)
        step = lr * m1h / (np.sqrt(m2h) + eps)
        sn = np.linalg.norm(step, axis=1, keepdims=True)
        cap = 0.02
        over = sn > cap
        if np.any(over):
            step = np.where(over, step * (cap / np.maximum(sn, 1e-12)), step)
        X = X - step
        X = X / np.linalg.norm(X, axis=1, keepdims=True)
    return X

def solve():
    t0 = time.time()
    HARD_END = t0 + 177.0
    SEARCH_END = t0 + 154.0
    rng = np.random.default_rng(0)
    X_best = None
    best_val = 1e9
    for fn in ["best.npy", "record.npy"]:
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=float)
            if a.shape == (176, 9):
                an = a / np.linalg.norm(a, axis=1, keepdims=True)
                v = max_cos(an)
                if v < best_val:
                    best_val = v
                    X_best = an.copy()
        except Exception:
            continue
    if X_best is None:
        X_best = rng.standard_normal((176, 9))
        X_best /= np.linalg.norm(X_best, axis=1, keepdims=True)
        best_val = max_cos(X_best)
    n, d = 176, 9
    # contact weights
    try:
        Gb = X_best @ X_best.T
        np.fill_diagonal(Gb, -2.0)
        contacts = np.sum(Gb > best_val - 1e-7, axis=1).astype(float)
        rowmax = Gb.max(axis=1)
    except Exception:
        contacts = np.ones(n)
        rowmax = np.full(n, best_val)
    X_second = X_best.copy()
    second_val = best_val
    sigmas = [0.008, 0.015, 0.025, 0.04, 0.065, 0.10]
    k_sched = [(40.0, 160.0), (70.0, 240.0), (110.0, 320.0), (50.0, 180.0)]
    lr_sched = [0.003, 0.0045, 0.002, 0.0035]
    steps_sched = [550, 750, 450, 650]
    trial = 0
    lp_calls = 0
    while time.time() < SEARCH_END:
        trial += 1
        if HARD_END - time.time() < 4.0:
            break
        # pick base: mostly best, sometimes second
        if trial % 9 == 0 and second_val < best_val + 0.006:
            base = X_second
            base_val = second_val
        else:
            base = X_best
            base_val = best_val
        X_try = base.copy()
        do_kill = (trial % 4 == 0)
        if do_kill:
            # kill 1-3 overloaded hubs
            try:
                Gtmp = base @ base.T
                np.fill_diagonal(Gtmp, -2.0)
                rm = Gtmp.max(axis=1)
                cc = np.sum(Gtmp > base_val - 1e-6, axis=1)
                score = rm + 0.001 * cc
                k = 1 + (trial // 4) % 3  # 1,2,3 cycle
                if trial % 16 == 0:
                    k = 4
                worst = np.argpartition(-score, k)[:k]
            except Exception:
                worst = rng.choice(n, size=2, replace=False)
                k = 2
            mask = np.ones(n, dtype=bool)
            mask[worst] = False
            cur = X_try[mask].copy()
            # sequential void reinsert
            for j in worst:
                # sample candidates, pick deepest hole
                nt = 2500
                T = rng.standard_normal((nt, d))
                T /= np.linalg.norm(T, axis=1, keepdims=True)
                dots = T @ cur.T
                mm = dots.max(axis=1)
                bi = int(np.argmin(mm))
                cand = T[bi].copy()
                # quick local hill climb on sphere: 60 steps small random
                bmm = float(mm[bi])
                step_s = 0.05
                for _ in range(60):
                    pr = cand + rng.standard_normal(d) * step_s
                    pr /= np.linalg.norm(pr)
                    v_ = float(np.max(pr @ cur.T))
                    if v_ < bmm:
                        bmm = v_
                        cand = pr
                    step_s *= 0.985
                X_try[j] = cand
                cur = np.vstack([cur, cand[None, :]])
            # small tangent jitter to rest to diversify
            sig = 0.005
            noi = rng.standard_normal((n, d)) * sig
            dn = np.sum(noi * X_try, axis=1, keepdims=True)
            noi -= dn * X_try
            X_try += noi
            X_try /= np.linalg.norm(X_try, axis=1, keepdims=True)
        else:
            sigma = sigmas[(trial - 1) % len(sigmas)]
            mu = float(contacts.mean())
            sd = float(contacts.std() + 1e-9)
            w = 1.0 + 0.6 * np.clip((contacts - mu) / (sd + 1.0), -1.0, 2.0)
            if trial % 5 == 0:
                try:
                    j = int(np.argmax(rowmax))
                    w[j] *= 2.0
                except Exception:
                    pass
            noise = rng.standard_normal((n, d)) * (sigma * w[:, None])
            dn = np.sum(noise * base, axis=1, keepdims=True)
            noise -= dn * base
            X_try = base + noise
            X_try /= np.linalg.norm(X_try, axis=1, keepdims=True)
        ki = k_sched[(trial - 1) % len(k_sched)]
        lr0 = lr_sched[(trial - 1) % len(lr_sched)]
        steps = steps_sched[(trial - 1) % len(steps_sched)]
        q_dead = min(time.time() + 1.0, SEARCH_END)
        X_q = adam_quench(X_try, ki[0], ki[1], steps, lr0, q_dead)
        try:
            vq = max_cos(X_q)
        except Exception:
            continue
        # track second best for diversity
        if vq < second_val and vq >= best_val:
            second_val = vq
            X_second = X_q.copy()
        if vq < best_val - 1e-12:
            if time.time() < HARD_END - 2.0 and lp_calls < 30:
                lp_dead = min(time.time() + 1.4, HARD_END - 1.0)
                X_p = lp_snap(X_q, lp_dead)
                lp_calls += 1
                try:
                    vp = max_cos(X_p)
                except Exception:
                    continue
                if vp < best_val:
                    # demote old best to second
                    if best_val < second_val or trial % 3 == 0:
                        pass
                    X_second = X_best.copy()
                    second_val = best_val
                    X_best = X_p
                    best_val = vp
                    try:
                        Gb = X_best @ X_best.T
                        np.fill_diagonal(Gb, -2.0)
                        contacts = np.sum(Gb > best_val - 1e-7, axis=1).astype(float)
                        rowmax = Gb.max(axis=1)
                    except Exception:
                        pass
                    continue
            # if LP not done, still accept quench win
            if vq < best_val:
                X_second = X_best.copy()
                second_val = best_val
                X_best = X_q
                best_val = vq
                try:
                    Gb = X_best @ X_best.T
                    np.fill_diagonal(Gb, -2.0)
                    contacts = np.sum(Gb > best_val - 1e-7, axis=1).astype(float)
                    rowmax = Gb.max(axis=1)
                except Exception:
                    pass
            continue
        # near-miss LP occasionally
        if vq < best_val + 0.0015 and (trial % 6 == 0) and lp_calls < 30 and time.time() < SEARCH_END - 1.0:
            if HARD_END - time.time() < 3.0:
                continue
            lp_dead = min(time.time() + 0.9, HARD_END - 1.0)
            X_p = lp_snap(X_q, lp_dead)
            lp_calls += 1
            try:
                vp = max_cos(X_p)
            except Exception:
                continue
            if vp < best_val - 1e-12:
                X_second = X_best.copy()
                second_val = best_val
                X_best = X_p
                best_val = vp
                try:
                    Gb = X_best @ X_best.T
                    np.fill_diagonal(Gb, -2.0)
                    contacts = np.sum(Gb > best_val - 1e-7, axis=1).astype(float)
                    rowmax = Gb.max(axis=1)
                except Exception:
                    pass
            elif vp < second_val:
                second_val = vp
                X_second = X_p.copy()
    if time.time() < HARD_END - 0.6:
        try:
            X_best = lp_snap(X_best, HARD_END - 0.25)
        except Exception:
            pass
    X_best = X_best / np.linalg.norm(X_best, axis=1, keepdims=True)
    return X_best
