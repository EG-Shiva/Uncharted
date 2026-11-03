import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _lp_polish(x, seconds, t_deadline=None):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while True:
        if time.time() - start >= seconds:
            break
        if t_deadline is not None and time.time() >= t_deadline:
            break
        if radius < 1e-14:
            break
        c = (x @ x.T)[iu]
        m = c.max()
        sel = c > m - max(20 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p, nv = len(I), n * d + 1
        if p == 0:
            break
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d),
                                           np.arange(n * d))), shape=(n, nv)).tocsr()
        except Exception:
            break
        obj = np.zeros(nv)
        obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c[sel], A_eq=A_eq,
                          b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                          method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        v = _max_cos(y)
        if v < best:
            x, best, radius = y, v, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def _smooth_softmax(x_init, steps, beta0, beta1, lr0, lr1, mu=0.85):
    X = x_init / np.linalg.norm(x_init, axis=1, keepdims=True)
    n, d = X.shape
    V = np.zeros_like(X)
    for t in range(steps):
        frac = t / max(steps - 1, 1)
        beta = beta0 * (1 - frac) + beta1 * frac
        lr = lr0 * (1 - frac) + lr1 * frac
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        m = G.max()
        E = np.exp(beta * (G - m))
        np.fill_diagonal(E, 0.0)
        S = E.sum() * 0.5
        if not np.isfinite(S) or S <= 0:
            break
        W = E / S
        grad = W @ X
        dot = np.sum(grad * X, axis=1, keepdims=True)
        gt = grad - dot * X
        nr = np.sqrt(np.sum(gt * gt, axis=1, keepdims=True)) + 1e-12
        scale = np.minimum(1.0, 5.0 / nr)
        gt = gt * scale
        V = mu * V + gt
        X = X - lr * V
        nrm = np.linalg.norm(X, axis=1, keepdims=True)
        nrm[nrm == 0] = 1.0
        X /= nrm
    return X

def _riesz_relax(x_init, steps, s0, s1, lr0, lr1, mu=0.85):
    X = x_init / np.linalg.norm(x_init, axis=1, keepdims=True)
    n, d = X.shape
    V = np.zeros_like(X)
    for t in range(steps):
        frac = t / max(steps - 1, 1)
        s = s0 * (1 - frac) + s1 * frac
        lr = lr0 * (1 - frac) + lr1 * frac
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        R2 = np.clip(2.0 - 2.0 * G, 1e-6, 4.0)
        # w = r^{-s-2} = (R2)^{-(s+2)/2}
        W = np.exp(-0.5 * (s + 2.0) * np.log(R2))
        np.fill_diagonal(W, 0.0)
        S = W.sum() * 0.5 + 1e-30
        if not np.isfinite(S):
            break
        W = W / S
        grad = W @ X
        dot = np.sum(grad * X, axis=1, keepdims=True)
        gt = grad - dot * X
        nr = np.sqrt(np.sum(gt * gt, axis=1, keepdims=True)) + 1e-12
        scale = np.minimum(1.0, 5.0 / nr)
        gt = gt * scale
        V = mu * V + gt
        X = X - lr * V
        nrm = np.linalg.norm(X, axis=1, keepdims=True)
        nrm[nrm == 0] = 1.0
        X /= nrm
    return X

def _find_void(X, rng, n_sample=8000):
    n, d = X.shape
    C = rng.standard_normal((n_sample, d))
    C /= np.linalg.norm(C, axis=1, keepdims=True)
    dots = C @ X.T
    m = dots.max(axis=1)
    return C[int(np.argmin(m))]

def solve():
    t0 = time.time()
    DEADLINE = t0 + 176.0
    rng = np.random.default_rng(12345)
    best_X = None
    best_v = np.inf
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=float)
            if a.shape == (94, 13):
                a /= np.linalg.norm(a, axis=1, keepdims=True)
                v = _max_cos(a)
                if v < best_v:
                    best_v = v
                    best_X = a.copy()
        except Exception:
            pass
    if best_X is None:
        best_X = rng.standard_normal((94, 13))
        best_X /= np.linalg.norm(best_X, axis=1, keepdims=True)
        best_v = _max_cos(best_X)
    n, d = 94, 13
    cur_X = best_X.copy()
    cur_v = best_v
    elites = [(best_v, best_X.copy())]

    def push_elite(v, X):
        for vv, _ in elites:
            if abs(vv - v) < 5e-7:
                return
        elites.append((v, X.copy()))
        elites.sort(key=lambda e: e[0])
        del elites[5:]

    T0, T1 = 0.004, 0.00015
    it = 0
    while True:
        if time.time() > DEADLINE - 16:
            break
        it += 1
        frac_time = (time.time() - t0) / max(1e-9, (DEADLINE - 16 - t0))
        frac_time = min(1.0, max(0.0, frac_time))
        Temp = T0 * (1 - frac_time) + T1 * frac_time
        # choose base
        u = rng.random()
        if u < 0.55:
            base = best_X
        elif u < 0.85:
            base = cur_X
        else:
            base = elites[rng.integers(0, len(elites))][1]
        cand = base.copy()
        mode = int(rng.integers(0, 8))
        if mode == 0:
            sigma = 0.006 + 0.004 * rng.random()
            cand = cand + sigma * rng.standard_normal(cand.shape)
            cand /= np.linalg.norm(cand, axis=1, keepdims=True)
            steps, kind = 1400 + int(rng.integers(0, 800)), 0
            b0, b1 = 50.0, 350.0 + 100 * rng.random()
        elif mode == 1:
            sigma = 0.025 + 0.02 * rng.random()
            cand = cand + sigma * rng.standard_normal(cand.shape)
            cand /= np.linalg.norm(cand, axis=1, keepdims=True)
            steps, kind = 1800 + int(rng.integers(0, 800)), 0
            b0, b1 = 40.0, 400.0
        elif mode == 2:
            k = int(rng.integers(2, 6))
            idx = rng.choice(n, size=k, replace=False)
            R = rng.standard_normal((k, d))
            R /= np.linalg.norm(R, axis=1, keepdims=True)
            cand[idx] = R
            steps, kind = 2000 + int(rng.integers(0, 600)), 1 if rng.random() < 0.5 else 0
            b0, b1 = 40.0, 420.0
        elif mode == 3:
            # void reinsertion of 1-3 worst
            G = cand @ cand.T
            np.fill_diagonal(G, -2.0)
            rm = G.max(axis=1)
            order = np.argsort(rm)
            k = int(rng.integers(1, 4))
            for j in range(k):
                worst = int(order[-(j + 1)])
                v = _find_void(cand, rng, 10000)
                cand[worst] = v + 1e-3 * rng.standard_normal(d)
            cand /= np.linalg.norm(cand, axis=1, keepdims=True)
            steps, kind = 2200 + int(rng.integers(0, 500)), 0
            b0, b1 = 35.0, 450.0
        elif mode == 4:
            G = cand @ cand.T
            np.fill_diagonal(G, -2.0)
            thresh = best_v - 0.002
            contacts = (G > thresh).sum(axis=1)
            worst = int(np.argmax(contacts))
            v = _find_void(cand, rng, 8000)
            cand[worst] = v + 1e-3 * rng.standard_normal(d)
            # second victim sometimes
            if rng.random() < 0.5:
                worst2 = int(np.argsort(contacts)[-2])
                v2 = _find_void(cand, rng, 6000)
                cand[worst2] = v2 + 1e-3 * rng.standard_normal(d)
            cand += 0.005 * rng.standard_normal(cand.shape)
            cand /= np.linalg.norm(cand, axis=1, keepdims=True)
            steps, kind = 2000 + int(rng.integers(0, 600)), 1 if rng.random() < 0.4 else 0
            b0, b1 = 40.0, 420.0
        elif mode == 5:
            # cap cluster shift
            uvec = rng.standard_normal(d)
            uvec /= np.linalg.norm(uvec)
            proj = cand @ uvec
            k = int(rng.integers(5, 12))
            idx = np.argsort(proj)[-k:]
            g = rng.standard_normal(d)
            g /= np.linalg.norm(g) + 1e-12
            amp = 0.04 + 0.04 * rng.random()
            cand[idx] += amp * g[None, :] + 0.008 * rng.standard_normal((k, d))
            cand /= np.linalg.norm(cand, axis=1, keepdims=True)
            steps, kind = 1800 + int(rng.integers(0, 600)), 0
            b0, b1 = 45.0, 380.0
        elif mode == 6:
            # subset random rotation-like shear
            k = int(rng.integers(12, 30))
            idx = rng.choice(n, size=k, replace=False)
            A = rng.standard_normal((d, d)) * 0.06
            A = A - A.T
            M = np.eye(d) + A
            cand[idx] = cand[idx] @ M
            cand[idx] += 0.005 * rng.standard_normal((k, d))
            cand /= np.linalg.norm(cand, axis=1, keepdims=True)
            steps, kind = 1800 + int(rng.integers(0, 600)), 1 if rng.random() < 0.5 else 0
            b0, b1 = 40.0, 400.0
        else:
            # worst-pair split
            G = cand @ cand.T
            np.fill_diagonal(G, -2.0)
            flat = int(np.argmax(G))
            ii, jj = divmod(flat, n)
            # push apart tangentially
            di = cand[ii] @ cand[jj]
            ti = cand[jj] - di * cand[ii]
            tj = cand[ii] - di * cand[jj]
            ti /= np.linalg.norm(ti) + 1e-12
            tj /= np.linalg.norm(tj) + 1e-12
            st = 0.02 + 0.02 * rng.random()
            cand[ii] = cand[ii] - st * ti + 0.004 * rng.standard_normal(d)
            cand[jj] = cand[jj] - st * tj + 0.004 * rng.standard_normal(d)
            cand += 0.003 * rng.standard_normal(cand.shape)
            cand /= np.linalg.norm(cand, axis=1, keepdims=True)
            steps, kind = 1600 + int(rng.integers(0, 600)), 0
            b0, b1 = 45.0, 400.0
        lr0 = 0.012 + 0.006 * rng.random()
        lr1 = 0.0015 + 0.001 * rng.random()
        mu = 0.82 + 0.08 * rng.random()
        if kind == 0:
            cand2 = _smooth_softmax(cand, steps=steps, beta0=b0, beta1=b1, lr0=lr0, lr1=lr1, mu=mu)
        else:
            s0 = 12.0 + 8.0 * rng.random()
            s1 = 60.0 + 60.0 * rng.random()
            cand2 = _riesz_relax(cand, steps=steps, s0=s0, s1=s1, lr0=lr0 * 0.6, lr1=lr1 * 0.6, mu=mu)
        v = _max_cos(cand2)
        # SA update of current walk
        if v < cur_v or rng.random() < np.exp(-(v - cur_v) / max(Temp, 1e-9)):
            cur_X, cur_v = cand2, v
        if v < best_v + 0.004:
            push_elite(v, cand2)
        # decide LP polish
        do_lp = False
        budget = 1.2
        if v < best_v - 1e-12:
            do_lp = True
            budget = 2.2
        elif v < best_v + 0.0012 and (rng.random() < 0.55):
            do_lp = True
            budget = 1.8
        elif v < best_v + 0.003 and (rng.random() < 0.18):
            do_lp = True
            budget = 1.2
        if do_lp:
            if time.time() > DEADLINE - 15:
                # skip, save for final
                if v < best_v:
                    best_X, best_v = cand2, v
                continue
            cand2p = _lp_polish(cand2, seconds=budget, t_deadline=DEADLINE - 13)
            vp = _max_cos(cand2p)
            if vp < cur_v and rng.random() < 0.7:
                cur_X, cur_v = cand2p, vp
            if vp < best_v - 1e-13:
                best_X, best_v = cand2p, vp
                push_elite(vp, cand2p)
            elif v < best_v - 1e-13:
                best_X, best_v = cand2, v
    remain = DEADLINE - time.time() - 0.8
    if remain > 1.5:
        best_X = _lp_polish(best_X, seconds=min(remain, 12.0), t_deadline=DEADLINE - 0.4)
    best_X /= np.linalg.norm(best_X, axis=1, keepdims=True)
    return best_X
