import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(np.max(g))

def adam_quench(x_init, steps, T0, T1, lr):
    x = x_init.copy()
    b1, b2, eps = 0.9, 0.999, 1e-8
    m = np.zeros_like(x)
    v = np.zeros_like(x)
    lT0 = np.log(T0); lT1 = np.log(T1)
    for s in range(1, steps + 1):
        lt = lT0 + (lT1 - lT0) * (s / steps)
        T = float(np.exp(lt))
        G = x @ x.T
        np.fill_diagonal(G, -10.0)
        M = float(G.max())
        W = np.exp((G - M) / T)
        np.fill_diagonal(W, 0.0)
        Grad = (W @ x) / T
        dot = np.sum(Grad * x, axis=1, keepdims=True)
        Gt = Grad - dot * x
        m = b1 * m + (1.0 - b1) * Gt
        v = b2 * v + (1.0 - b2) * (Gt * Gt)
        c1 = 1.0 - b1 ** s
        c2 = 1.0 - b2 ** s
        upd = (m / c1) / (np.sqrt(v / c2) + eps)
        x = x - lr * upd
        nr = np.linalg.norm(x, axis=1, keepdims=True)
        nr[nr == 0] = 1.0
        x /= nr
    return x

def hard_push(x_init, iters=120, lr0=0.003, band=0.002):
    x = x_init.copy()
    n, d = x.shape
    for it in range(iters):
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        mval = float(G.max())
        M = (G > mval - band)
        # gradient: for each i sum over contacting j of tangential part of x_j
        # vectorized: (M @ x) - diag-ish correction
        S = M @ x  # (n,d)
        coeff = (G * M).sum(axis=1, keepdims=True)  # sum_j c_ij
        Gt = S - coeff * x
        # scale lr down
        lr = lr0 * (1.0 - 0.7 * it / max(iters, 1))
        # normalize per-row magnitude to avoid blowup
        gn = np.linalg.norm(Gt, axis=1, keepdims=True) + 1e-12
        # only move points that have contacts
        has = M.sum(axis=1, keepdims=True) > 0
        step = np.where(has, Gt / np.maximum(gn, 3.0), 0.0)
        x = x - lr * step
        nr = np.linalg.norm(x, axis=1, keepdims=True)
        nr[nr == 0] = 1.0
        x /= nr
    return x

def lp_polish(x, seconds, t_end):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = max_cos(x)
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and time.time() < t_end:
        if radius < 1e-14:
            break
        c = (x @ x.T)[iu]
        mcur = float(c.max())
        sel = c > mcur - max(20 * radius, 1e-9)
        if int(sel.sum()) > 2600:
            idx = np.argpartition(-c, 2600)[:2600]
            mask = np.zeros_like(sel)
            mask[idx] = True
            sel = sel & mask
        I, J = iu[0][sel], iu[1][sel]
        p, nv = len(I), n * d + 1
        if p == 0:
            break
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
            res = linprog(obj, A_ub=A_ub, b_ub=-c[sel], A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        nr = np.linalg.norm(y, axis=1, keepdims=True)
        nr[nr == 0] = 1.0
        y /= nr
        vv = max_cos(y)
        if vv < best:
            x, best, radius = y, vv, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def refine_hole(q_init, X_rem, steps=250, T=0.004, lr=0.06):
    q = q_init.copy()
    q /= (np.linalg.norm(q) + 1e-12)
    for _ in range(steps):
        dots = X_rem @ q
        M = float(dots.max())
        w = np.exp((dots - M) / T)
        s = float(w.sum())
        if s == 0 or not np.isfinite(s):
            break
        g = (w[:, None] * X_rem).sum(axis=0) / (s * T)
        gt = g - float(g @ q) * q
        q = q - lr * gt
        nq = float(np.linalg.norm(q))
        if nq == 0 or not np.isfinite(nq):
            break
        q /= nq
    return q

def greedy_holes(X_rem, rng, k, n_samp=12000):
    d = X_rem.shape[1]
    S = rng.standard_normal((n_samp, d))
    S /= np.linalg.norm(S, axis=1, keepdims=True)
    chosen = []
    cur = X_rem
    for _ in range(k):
        if chosen:
            aug = np.vstack([cur] + [c[None, :] for c in chosen])
        else:
            aug = cur
        dots = S @ aug.T
        worst = dots.max(axis=1)
        bi = int(np.argmin(worst))
        q0 = S[bi].copy()
        q = refine_hole(q0, aug, steps=220, T=0.004, lr=0.06)
        chosen.append(q)
        # suppress neighbourhood to diversify
        dd = S @ q
        keep = dd < 0.97
        if int(keep.sum()) > 1500:
            S = S[keep]
    return chosen

def solve():
    t0 = time.time()
    T_LIMIT = 170.0
    t_end = t0 + T_LIMIT
    rng = np.random.default_rng(12345)
    cands = []
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.array(np.load(fn), dtype=float)
            a /= np.linalg.norm(a, axis=1, keepdims=True)
            if a.shape == (102, 12) and np.all(np.isfinite(a)):
                cands.append((max_cos(a), a))
        except Exception:
            pass
    n, d = 102, 12
    if len(cands) == 0:
        a = rng.standard_normal((n, d))
        a /= np.linalg.norm(a, axis=1, keepdims=True)
        x_best = a; best = max_cos(a)
    else:
        cands.sort(key=lambda t: t[0])
        best, x_best = cands[0][0], cands[0][1].copy()
    elites = [(best, x_best.copy())]
    if len(cands) > 1:
        elites.append((cands[1][0], cands[1][1].copy()))
    # current wandering state
    cur = x_best.copy(); cur_val = best
    hop_end = t_end - 16.0
    it = 0
    stuck = 0
    while time.time() < hop_end and it < 800:
        it += 1
        frac = max(0.0, min(1.0, (time.time() - t0) / max(hop_end - t0, 1e-9)))
        Temp = 0.004 * (1 - frac) + 0.00025 * frac
        # pick base: wandering cur, sometimes best, sometimes second elite
        r = rng.random()
        if r < 0.62:
            base = cur
        elif r < 0.85:
            base = x_best
        else:
            elites.sort(key=lambda t: t[0])
            base = elites[min(len(elites)-1, int(rng.integers(1, len(elites)+1)) if len(elites)>1 else 0)][1] if len(elites)>1 else x_best
            # fallback
            if not isinstance(base, np.ndarray):
                base = x_best
        G0 = base @ base.T
        np.fill_diagonal(G0, -2.0)
        rowmax = G0.max(axis=1)
        mval = float(rowmax.max())
        cnt = (G0 > mval - 0.0015).sum(axis=1)
        order = np.argsort(-rowmax)
        order_c = np.argsort(-cnt)
        mode = it % 7
        x_try = base.copy()
        if mode == 0:
            sig = float(rng.uniform(0.015, 0.04))
            x_try = base + sig * rng.standard_normal((n, d))
            steps = int(rng.integers(400, 700)); T0 = float(rng.uniform(0.02, 0.05)); T1 = float(rng.uniform(0.0025, 0.0045)); lr = float(rng.uniform(0.0018, 0.0028))
        elif mode == 1:
            sig = float(rng.uniform(0.06, 0.13))
            x_try = base + sig * rng.standard_normal((n, d))
            steps = int(rng.integers(450, 800)); T0 = float(rng.uniform(0.03, 0.06)); T1 = float(rng.uniform(0.002, 0.004)); lr = float(rng.uniform(0.0018, 0.003))
        elif mode == 2:
            k = int(rng.integers(2, 6))
            worst = order_c[:k].copy()
            rng.shuffle(worst)
            mask = np.ones(n, dtype=bool); mask[worst] = False
            holes = greedy_holes(base[mask], rng, k, n_samp=10000)
            for wi, h in zip(worst, holes):
                x_try[wi] = h + 0.004 * rng.standard_normal(d)
            x_try[mask] += 0.0035 * rng.standard_normal((n - k, d))
            steps = int(rng.integers(500, 850)); T0 = float(rng.uniform(0.018, 0.04)); T1 = float(rng.uniform(0.002, 0.0038)); lr = float(rng.uniform(0.0018, 0.0026))
        elif mode == 3:
            ci = int(order_c[int(rng.integers(0, 8))])
            Gr = (base @ base[ci]); Gr[ci] = -2.0
            nn = int(rng.integers(5, 12))
            nbr = np.argsort(-Gr)[:nn]
            cl = np.append(nbr, ci)
            x_try[cl] += float(rng.uniform(0.05, 0.12)) * rng.standard_normal((len(cl), d))
            x_try += 0.003 * rng.standard_normal((n, d))
            steps = int(rng.integers(450, 800)); T0 = float(rng.uniform(0.02, 0.05)); T1 = float(rng.uniform(0.0025, 0.004)); lr = float(rng.uniform(0.0018, 0.0028))
        elif mode == 4:
            # break worst contacting pair apart
            iu0 = np.argmax(G0)
            ii, jj = divmod(int(iu0), n)
            v = base[ii] - base[jj]
            v /= (np.linalg.norm(v) + 1e-12)
            amp = float(rng.uniform(0.05, 0.12))
            x_try[ii] = base[ii] + amp * v + 0.01 * rng.standard_normal(d)
            x_try[jj] = base[jj] - amp * v + 0.01 * rng.standard_normal(d)
            x_try += 0.004 * rng.standard_normal((n, d))
            steps = int(rng.integers(450, 800)); T0 = float(rng.uniform(0.015, 0.04)); T1 = float(rng.uniform(0.002, 0.004)); lr = float(rng.uniform(0.0018, 0.0028))
        elif mode == 5:
            # elite crossover with Procrustes alignment
            if len(elites) >= 2:
                elites.sort(key=lambda t: t[0])
                A = elites[0][1]; B = elites[int(rng.integers(1, len(elites)))][1]
                # align B to A
                try:
                    U, _, Vt = np.linalg.svd(B.T @ A, full_matrices=False)
                    R = U @ Vt
                    B = B @ R
                except Exception:
                    pass
                msk = rng.random(n) < 0.5
                x_try[msk] = B[msk]
                x_try += float(rng.uniform(0.02, 0.06)) * rng.standard_normal((n, d))
            else:
                x_try = base + 0.05 * rng.standard_normal((n, d))
            steps = int(rng.integers(450, 800)); T0 = float(rng.uniform(0.02, 0.05)); T1 = float(rng.uniform(0.0025, 0.004)); lr = float(rng.uniform(0.0018, 0.0028))
        else:
            wi = int(order[0])
            mask = np.ones(n, dtype=bool); mask[wi] = False
            holes = greedy_holes(base[mask], rng, 1, n_samp=8000)
            x_try[wi] = holes[0] + 0.004 * rng.standard_normal(d)
            x_try += float(rng.uniform(0.008, 0.028)) * rng.standard_normal((n, d))
            steps = int(rng.integers(400, 750)); T0 = float(rng.uniform(0.015, 0.04)); T1 = float(rng.uniform(0.0025, 0.0045)); lr = float(rng.uniform(0.0018, 0.0028))
        x_try /= np.linalg.norm(x_try, axis=1, keepdims=True)
        x_q = adam_quench(x_try, steps, T0, T1, lr)
        # cheap hard-max squeeze
        try:
            x_q = hard_push(x_q, iters=100, lr0=0.0025, band=0.002)
        except Exception:
            pass
        vq = max_cos(x_q)
        # wandering acceptance
        if vq < cur_val:
            cur, cur_val = x_q.copy(), vq
            stuck = 0
        else:
            stuck += 1
            if rng.random() < np.exp(-(vq - cur_val) / max(Temp, 1e-12)):
                cur, cur_val = x_q.copy(), vq
            if stuck > 35:
                cur, cur_val = x_best.copy(), best
                stuck = 0
        if vq < best - 1e-12:
            # snap only if promising and time allows
            budget = min(3.0, max(0.8, (hop_end - time.time()) / 6.0))
            if budget > 0.7 and time.time() + budget < hop_end:
                try:
                    x_q = lp_polish(x_q, budget, hop_end)
                    vq = max_cos(x_q)
                    # extra hard push after LP sometimes helps
                except Exception:
                    pass
            if vq < best:
                x_best, best = x_q.copy(), vq
                cur, cur_val = x_q.copy(), vq
                stuck = 0
                elites.append((best, x_best.copy()))
                elites.sort(key=lambda t: t[0])
                elites = elites[:4]
        elif vq < elites[-1][0] - 1e-12 and len(elites) < 4 and (it % 3 == 0):
            elites.append((vq, x_q.copy()))
            elites.sort(key=lambda t: t[0])
        if time.time() >= hop_end:
            break
    remain = t_end - time.time() - 1.5
    if remain > 2.0:
        try:
            x_best = lp_polish(x_best, min(remain, 14.0), t_end - 0.8)
        except Exception:
            pass
    x_best /= np.linalg.norm(x_best, axis=1, keepdims=True)
    return x_best
