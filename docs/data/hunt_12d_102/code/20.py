import time
import numpy as np
from scipy.optimize import linprog, minimize
from scipy.sparse import coo_matrix


def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())


def adam_quench(x_init, steps, T0, T1, lr, seed_jit=0.0):
    x = x_init.copy()
    n, d = x.shape
    m = np.zeros_like(x)
    v = np.zeros_like(x)
    b1, b2, eps = 0.9, 0.999, 1e-8
    lT0 = np.log(T0)
    lT1 = np.log(T1)
    for s in range(1, steps + 1):
        lt = lT0 + (lT1 - lT0) * (s / steps)
        T = float(np.exp(lt))
        G = x @ x.T
        np.fill_diagonal(G, -10.0)
        M = G.max()
        W = np.exp((G - M) / T)
        np.fill_diagonal(W, 0.0)
        Grad = (W @ x) / T
        dot = np.sum(Grad * x, axis=1, keepdims=True)
        Gt = Grad - dot * x
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


def mom_quench(x_init, steps, T0, T1, lr, mu=0.85):
    # heavy-ball projected gradient on softmax, different basin than Adam
    x = x_init.copy()
    n, d = x.shape
    vel = np.zeros_like(x)
    lT0 = np.log(T0)
    lT1 = np.log(T1)
    for s in range(1, steps + 1):
        frac = s / steps
        T = float(np.exp(lT0 + (lT1 - lT0) * frac))
        lrs = lr * (1.0 - 0.5 * frac)
        G = x @ x.T
        np.fill_diagonal(G, -10.0)
        M = G.max()
        E = np.exp((G - M) / T)
        np.fill_diagonal(E, 0.0)
        S = E.sum()
        if S == 0 or not np.isfinite(S):
            continue
        P = E / S
        Grad = (P @ x)
        dot = np.sum(Grad * x, axis=1, keepdims=True)
        Gt = Grad - dot * x
        # normalize scale to keep step stable
        nrm = np.linalg.norm(Gt) + 1e-12
        Gt = Gt / (nrm / (n * 1.0) + 1e-12)
        # clip
        vel = mu * vel + Gt
        x = x - lrs * vel
        nr = np.linalg.norm(x, axis=1, keepdims=True)
        nr[nr == 0] = 1.0
        x /= nr
    return x


def lbfgs_quench(x_init, T, maxiter=120):
    n, d = x_init.shape
    y0 = x_init.ravel().copy()

    def fun_grad(y):
        Y = y.reshape(n, d)
        nr = np.linalg.norm(Y, axis=1, keepdims=True)
        nr[nr == 0] = 1.0
        X = Y / nr
        G = X @ X.T
        np.fill_diagonal(G, -10.0)
        M = G.max()
        E = np.exp((G - M) / T)
        np.fill_diagonal(E, 0.0)
        S = E.sum(axis=1)  # not used
        # sum over i<j
        S2 = E.sum() / 2.0 + 1e-300
        L = M + T * np.log(S2)
        P = E / (2.0 * S2)  # symmetric, rows sum to weights; Grad=P@X gives correct grad
        # P diag 0 already
        GradX = P @ X
        dot = np.sum(GradX * X, axis=1, keepdims=True)
        Gt = (GradX - dot * X) / nr
        return float(L), Gt.ravel()

    try:
        res = minimize(fun_grad, y0, jac=True, method="L-BFGS-B",
                       options={"maxiter": maxiter, "ftol": 1e-12, "gtol": 1e-8})
        Y = res.x.reshape(n, d)
        nr = np.linalg.norm(Y, axis=1, keepdims=True)
        nr[nr == 0] = 1.0
        return Y / nr
    except Exception:
        return x_init


def pair_sweep(x, sweeps=30, step=0.02):
    x = x.copy()
    n = x.shape[0]
    for _ in range(sweeps):
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        k = int(np.argmax(G))
        i = k // n
        j = k % n
        gij = G[i, j]
        # push apart tangentially
        ci = x[i] @ x[j]
        fi = x[j] - ci * x[i]
        fj = x[i] - ci * x[j]
        x[i] = x[i] - step * fi
        x[j] = x[j] - step * fj
        ni = np.linalg.norm(x[i])
        nj = np.linalg.norm(x[j])
        if ni == 0 or nj == 0 or not np.isfinite(ni + nj):
            break
        x[i] /= ni
        x[j] /= nj
        # early stop if good enough? continue to smooth
    return x


def refine_hole(q0, Xcur, steps=250, T=0.006, lr=0.08):
    q = q0.copy()
    nrm = np.linalg.norm(q)
    if nrm == 0:
        return q0
    q /= nrm
    for _ in range(steps):
        dots = Xcur @ q
        M = dots.max()
        w = np.exp((dots - M) / T)
        s = w.sum()
        if s == 0 or not np.isfinite(s):
            break
        w /= s
        g = Xcur.T @ w
        gt = g - (g @ q) * q
        q = q - lr * gt
        nrm = np.linalg.norm(q)
        if nrm == 0 or not np.isfinite(nrm):
            break
        q /= nrm
        lr *= 0.998
    return q


def best_holes(X_fixed, k, rng, n_samp=20000):
    d = X_fixed.shape[1]
    outs = []
    Xcur = X_fixed.copy()
    for _ in range(k):
        S = rng.standard_normal((n_samp, d))
        S /= np.linalg.norm(S, axis=1, keepdims=True)
        dots = S @ Xcur.T
        worst = dots.max(axis=1)
        bi = int(np.argmin(worst))
        q0 = S[bi]
        q = refine_hole(q0, Xcur)
        outs.append(q)
        Xcur = np.vstack([Xcur, q[None, :]])
    return np.array(outs)


def lp_polish(x, seconds, t_end):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = max_cos(x)
    radius = 1e-3
    iu = np.triu_indices(n, 1)
    while time.time() < t_end and (time.time() - (t_end - seconds)) < seconds:
        if radius < 1e-14:
            break
        c = (x @ x.T)[iu]
        mcur = c.max()
        sel = c > mcur - max(20 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p, nv = len(I), n * d + 1
        if p == 0:
            break
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))),
                          shape=(n, nv)).tocsr()
        obj = np.zeros(nv)
        obj[-1] = 1.0
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


def solve():
    t0 = time.time()
    T_LIMIT = 172.0
    t_end = t0 + T_LIMIT
    rng = np.random.default_rng(7)
    cands = []
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.array(np.load(fn), dtype=float)
            a /= np.linalg.norm(a, axis=1, keepdims=True)
            if a.shape == (102, 12) and np.all(np.isfinite(a)):
                cands.append((max_cos(a), a))
        except Exception:
            pass
    if len(cands) == 0:
        a = rng.standard_normal((102, 12))
        a /= np.linalg.norm(a, axis=1, keepdims=True)
        x_best = a
        best = max_cos(a)
    else:
        cands.sort(key=lambda t: t[0])
        best, x_best = cands[0][0], cands[0][1].copy()
    n, d = 102, 12

    it = 0
    # hopping phase, reserve ~22s for final
    while time.time() < t_end - 24.0 and it < 500:
        it += 1
        mode = it % 7
        # contact overload stats from incumbent
        G0 = x_best @ x_best.T
        np.fill_diagonal(G0, -2.0)
        m0 = G0.max()
        cnt = (G0 > m0 - 0.008).sum(axis=1)
        rowmax = G0.max(axis=1)
        order = np.argsort(-rowmax)
        cnt_order = np.argsort(-cnt)
        x_try = x_best.copy()

        if mode == 0:
            x_try = x_best + 0.014 * rng.standard_normal((n, d))
            x_try /= np.linalg.norm(x_try, axis=1, keepdims=True)
            x_q = adam_quench(x_try, 650, 0.022, 0.0035, 0.0022)
            x_q = pair_sweep(x_q, 25, 0.02)
        elif mode == 1:
            # rupture overloaded hub: hub + nearest neighbours
            hub = int(cnt_order[0])
            dots = (x_best @ x_best[hub])
            dots[hub] = -2.0
            nbr = np.argsort(-dots)[:6]
            kill = np.append(nbr, hub)  # 7 points
            mask = np.ones(n, dtype=bool)
            mask[kill] = False
            holes = best_holes(x_best[mask], len(kill), rng, n_samp=12000)
            x_try[mask] = x_best[mask] + 0.003 * rng.standard_normal((n - len(kill), d))
            x_try[kill] = holes + 0.004 * rng.standard_normal((len(kill), d))
            x_try /= np.linalg.norm(x_try, axis=1, keepdims=True)
            x_q = adam_quench(x_try, 900, 0.025, 0.003, 0.002)
            x_q = pair_sweep(x_q, 20, 0.015)
        elif mode == 2:
            # kick top-3 worst hard
            idx = order[:3]
            x_try[idx] += 0.10 * rng.standard_normal((3, d))
            x_try += 0.004 * rng.standard_normal((n, d))
            x_try /= np.linalg.norm(x_try, axis=1, keepdims=True)
            x_q = mom_quench(x_try, 700, 0.02, 0.0025, 0.02, mu=0.85)
        elif mode == 3:
            # single best-hole re-insertion of worst point
            wi = int(order[0])
            mask = np.ones(n, dtype=bool)
            mask[wi] = False
            holes = best_holes(x_best[mask], 1, rng, n_samp=20000)
            x_try[wi] = holes[0] + 0.003 * rng.standard_normal(d)
            x_try[mask] += 0.0035 * rng.standard_normal((n - 1, d))
            x_try /= np.linalg.norm(x_try, axis=1, keepdims=True)
            x_q = adam_quench(x_try, 800, 0.018, 0.0028, 0.002)
        elif mode == 4:
            # multi-hole 2-3 worst
            kk = int(rng.integers(2, 4))
            kill = order[:kk].copy()
            mask = np.ones(n, dtype=bool)
            mask[kill] = False
            holes = best_holes(x_best[mask], kk, rng, n_samp=15000)
            x_try[kill] = holes + 0.004 * rng.standard_normal((kk, d))
            x_try[mask] += 0.003 * rng.standard_normal((n - kk, d))
            x_try /= np.linalg.norm(x_try, axis=1, keepdims=True)
            # L-BFGS smooth-max at two temperatures
            T = float(rng.choice([0.004, 0.007, 0.011]))
            x_q = lbfgs_quench(x_try, T, maxiter=150)
            x_q = pair_sweep(x_q, 15, 0.015)
        elif mode == 5:
            # cluster shake: random centre + 5 nearest
            ci = int(rng.integers(n))
            Gr = (x_best @ x_best[ci])
            Gr[ci] = -2.0
            nbr = np.argsort(-Gr)[:5]
            cl = np.append(nbr, ci)
            x_try[cl] += 0.07 * rng.standard_normal((len(cl), d))
            x_try += 0.003 * rng.standard_normal((n, d))
            x_try /= np.linalg.norm(x_try, axis=1, keepdims=True)
            x_q = adam_quench(x_try, 850, 0.022, 0.0032, 0.0021)
        else:
            # global jitter + L-BFGS
            x_try = x_best + 0.020 * rng.standard_normal((n, d))
            x_try /= np.linalg.norm(x_try, axis=1, keepdims=True)
            T = float(rng.choice([0.005, 0.008]))
            x_q = lbfgs_quench(x_try, T, maxiter=120)
            x_q = adam_quench(x_q, 400, 0.010, 0.0025, 0.0018)

        vq = max_cos(x_q)
        if vq < best - 1e-12:
            budget = min(3.5, max(0.9, (t_end - 24.0 - time.time()) / 6.0))
            if time.time() + budget < t_end - 23.0 and budget > 0.8:
                x_q = lp_polish(x_q, budget, min(t_end - 22.0, time.time() + budget))
                vq = max_cos(x_q)
            if vq < best:
                x_best, best = x_q, vq
        elif vq < best + 0.003 and (it % 6 == 0):
            budget = min(2.0, max(0.7, (t_end - 24.0 - time.time()) / 10.0))
            if budget > 0.65 and time.time() + budget < t_end - 23.0:
                x_p = lp_polish(x_q, budget, time.time() + budget)
                vp = max_cos(x_p)
                if vp < best:
                    x_best, best = x_p, vp
        if time.time() >= t_end - 24.0:
            break

    # final squeeze: LP + tiny shake + LP
    remain = t_end - time.time() - 1.5
    if remain > 3.0:
        x_best = lp_polish(x_best, min(remain * 0.6, 13.0), t_end - 1.0)
        # tiny tangential shake to jump to adjacent LP basin
        if t_end - time.time() > 5.0:
            rng2 = np.random.default_rng(12345)
            bestv = max_cos(x_best)
            for _ in range(3):
                if t_end - time.time() < 2.5:
                    break
                pert = 1e-4 * rng2.standard_normal((n, d))
                # project tangential
                pert = pert - np.sum(pert * x_best, axis=1, keepdims=True) * x_best
                y = x_best + pert
                y /= np.linalg.norm(y, axis=1, keepdims=True)
                y = lp_polish(y, 1.5, time.time() + 1.5)
                vy = max_cos(y)
                if vy < bestv:
                    x_best, bestv = y, vy
    x_best /= np.linalg.norm(x_best, axis=1, keepdims=True)
    return x_best
