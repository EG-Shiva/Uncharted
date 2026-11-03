import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())


def adam_quench(x_init, steps, T0, T1, lr):
    x = x_init.copy()
    n, d = x.shape
    m = np.zeros_like(x)
    v = np.zeros_like(x)
    b1, b2, eps = 0.9, 0.999, 1e-8
    l0 = float(np.log(T0))
    l1 = float(np.log(T1))
    for s in range(1, steps + 1):
        lt = l0 + (l1 - l0) * (s / steps)
        T = float(np.exp(lt))
        G = x @ x.T
        np.fill_diagonal(G, -10.0)
        Mx = float(G.max())
        # softmax weights on near-max pairs
        W = np.exp((G - Mx) / T)
        np.fill_diagonal(W, 0.0)
        Gr = W @ x
        dot = np.sum(Gr * x, axis=1, keepdims=True)
        Gt = Gr - dot * x
        m = b1 * m + (1.0 - b1) * Gt
        v = b2 * v + (1.0 - b2) * (Gt * Gt)
        c1 = 1.0 - b1 ** s
        c2 = 1.0 - b2 ** s
        mh = m / c1
        vh = v / c2
        upd = mh / (np.sqrt(vh) + eps)
        x = x - lr * upd
        nr = np.linalg.norm(x, axis=1, keepdims=True)
        nr[nr == 0] = 1.0
        x /= nr
    return x


def refine_hole(q, X_rem, iters=120, T=0.006, lr=0.08):
    for _ in range(iters):
        dots = X_rem @ q
        Mx = float(dots.max())
        w = np.exp((dots - Mx) / T)
        s = float(w.sum())
        if s == 0.0 or not np.isfinite(s):
            break
        g = (w[:, None] * X_rem).sum(axis=0) / s
        gt = g - float(np.dot(g, q)) * q
        q = q - lr * gt
        n = float(np.linalg.norm(q))
        if n == 0.0 or not np.isfinite(n):
            break
        q = q / n
    return q


def lp_polish(x, seconds, t_end):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    try:
        best = max_cos(x)
    except Exception:
        return x
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and time.time() < t_end:
        if radius < 1e-14:
            break
        try:
            c = (x @ x.T)[iu]
        except Exception:
            break
        mcur = float(c.max())
        sel = c > mcur - max(20.0 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p, nv = len(I), n * d + 1
        if p == 0:
            break
        try:
            rows = np.repeat(np.arange(p), 2 * d + 1)
            cols = np.concatenate([I[:, None] * d + np.arange(d),
                                   J[:, None] * d + np.arange(d),
                                   np.full((p, 1), n * d)], 1).ravel()
            vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(),
                               (np.repeat(np.arange(n), d), np.arange(n * d))),
                              shape=(n, nv)).tocsr()
            obj = np.zeros(nv)
            obj[-1] = 1.0
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
        try:
            y = x + res.x[:-1].reshape(n, d)
        except Exception:
            radius *= 0.5
            continue
        nr = np.linalg.norm(y, axis=1, keepdims=True)
        nr[nr == 0] = 1.0
        y /= nr
        if not np.all(np.isfinite(y)):
            radius *= 0.5
            continue
        vv = max_cos(y)
        if vv < best:
            x, best, radius = y, vv, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x


def solve():
    t0 = time.time()
    T_LIMIT = 170.0
    t_end = t0 + T_LIMIT
    rng = np.random.default_rng(7)
    n, d = 102, 12

    cands = []
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.array(np.load(fn), dtype=float)
            a /= np.linalg.norm(a, axis=1, keepdims=True)
            if a.shape == (n, d) and np.all(np.isfinite(a)):
                cands.append((max_cos(a), a))
        except Exception:
            pass
    if len(cands) == 0:
        a = rng.standard_normal((n, d))
        a /= np.linalg.norm(a, axis=1, keepdims=True)
        x_best = a
        best = max_cos(a)
    else:
        cands.sort(key=lambda t: t[0])
        best, x_best = cands[0][0], cands[0][1].copy()
    x_cur = x_best.copy()
    try:
        cur_val = max_cos(x_cur)
    except Exception:
        cur_val = best

    trial = 0
    # reserve ~26s for final LP
    while time.time() < t_end - 28.0 and trial < 2000:
        trial += 1
        # choose base: mostly wander, sometimes restart from best
        if trial % 5 == 0 or rng.random() < 0.30:
            base = x_best
        else:
            base = x_cur
        u = float(rng.random())
        x_try = None
        steps = 700
        T0 = 0.018
        T1 = 0.0022
        lr = 0.003

        if u < 0.25:
            # global jitter
            sigma = float(rng.uniform(0.010, 0.050))
            x_try = base + sigma * rng.standard_normal((n, d))
            x_try /= np.linalg.norm(x_try, axis=1, keepdims=True)
            steps = int(rng.integers(500, 900))
            T0 = float(rng.uniform(0.012, 0.025))
            T1 = float(rng.uniform(0.0012, 0.0035))
            lr = float(rng.uniform(0.002, 0.004))
        elif u < 0.50:
            # kick worst points heavily
            G = base @ base.T
            np.fill_diagonal(G, -2.0)
            rowmax = G.max(axis=1)
            order = np.argsort(-rowmax)
            k = int(rng.integers(2, 6))
            idx = order[:k]
            x_try = base.copy()
            x_try[idx] += float(rng.uniform(0.06, 0.16)) * rng.standard_normal((k, d))
            x_try += float(rng.uniform(0.002, 0.008)) * rng.standard_normal((n, d))
            x_try /= np.linalg.norm(x_try, axis=1, keepdims=True)
            steps = int(rng.integers(600, 1000))
            T0 = float(rng.uniform(0.015, 0.030))
            T1 = float(rng.uniform(0.0010, 0.0030))
            lr = float(rng.uniform(0.0022, 0.0045))
        elif u < 0.75:
            # delete worst k and greedy hole re-insertion
            G = base @ base.T
            np.fill_diagonal(G, -2.0)
            rowmax = G.max(axis=1)
            order = np.argsort(-rowmax)
            k = int(rng.integers(1, 5))
            worst = order[:k]
            mask = np.ones(n, dtype=bool)
            mask[worst] = False
            X_rem = base[mask]
            S = rng.standard_normal((12000, d))
            S /= np.linalg.norm(S, axis=1, keepdims=True)
            X_aug = X_rem.copy()
            new_pts = []
            for _ in range(k):
                dots = S @ X_aug.T
                worstdot = dots.max(axis=1)
                bi = int(np.argmin(worstdot))
                q = S[bi].copy()
                q = refine_hole(q, X_aug, iters=120, T=0.006, lr=0.08)
                new_pts.append(q)
                X_aug = np.vstack([X_aug, q[None, :]])
            x_try = np.vstack([X_rem] + [q[None, :] for q in new_pts])
            x_try += float(rng.uniform(0.001, 0.004)) * rng.standard_normal((n, d))
            x_try /= np.linalg.norm(x_try, axis=1, keepdims=True)
            steps = int(rng.integers(600, 1000))
            T0 = float(rng.uniform(0.012, 0.025))
            T1 = float(rng.uniform(0.0010, 0.0028))
            lr = float(rng.uniform(0.002, 0.004))
        else:
            # cluster shake: point + nearest neighbours
            ci = int(rng.integers(n))
            Gr = (base @ base[ci]).copy()
            Gr[ci] = -2.0
            nn = int(rng.integers(4, 8))
            nbr = np.argsort(-Gr)[:nn]
            cl = np.append(nbr, ci)
            x_try = base.copy()
            x_try[cl] += float(rng.uniform(0.05, 0.12)) * rng.standard_normal((len(cl), d))
            x_try += float(rng.uniform(0.002, 0.006)) * rng.standard_normal((n, d))
            x_try /= np.linalg.norm(x_try, axis=1, keepdims=True)
            steps = int(rng.integers(500, 900))
            T0 = float(rng.uniform(0.012, 0.028))
            T1 = float(rng.uniform(0.0012, 0.0032))
            lr = float(rng.uniform(0.002, 0.0042))

        if x_try is None or x_try.shape != (n, d):
            continue
        if not np.all(np.isfinite(x_try)):
            continue
        try:
            x_q = adam_quench(x_try, steps, T0, T1, lr)
        except Exception:
            continue
        if not np.all(np.isfinite(x_q)):
            continue
        try:
            v = max_cos(x_q)
        except Exception:
            continue
        if not np.isfinite(v):
            continue
        if v < best - 1e-12:
            best = v
            x_best = x_q.copy()
            x_cur = x_q.copy()
            cur_val = v
        else:
            # wandering acceptance: improve current or occasional sideways
            if v < cur_val - 1e-12 or float(rng.random()) < 0.08:
                x_cur = x_q.copy()
                cur_val = v

        if time.time() >= t_end - 28.0:
            break

    # final snap with remaining budget
    remain = float(t_end - time.time() - 1.0)
    if remain > 3.0:
        try:
            x_best = lp_polish(x_best, min(25.0, remain), t_end - 0.5)
        except Exception:
            pass
    x_best /= np.linalg.norm(x_best, axis=1, keepdims=True)
    return x_best
