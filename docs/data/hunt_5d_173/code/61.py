import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())


def _row_max(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return g.max(axis=1)


def _lp_polish(x, deadline, radius0=2e-3, cap=2200):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = float(radius0)
    iu0, iu1 = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        G = x @ x.T
        c_all = G[iu0, iu1]
        m = float(c_all.max())
        thresh = m - max(20.0 * radius, 1e-9)
        sel = c_all > thresh
        cnt = int(np.count_nonzero(sel))
        if cnt == 0:
            radius *= 0.5
            continue
        if cnt > cap:
            idx = np.where(sel)[0]
            order = np.argsort(c_all[idx])[::-1][:cap]
            msel = np.zeros_like(sel, dtype=bool)
            msel[idx[order]] = True
            sel = msel
        I = iu0[sel]
        J = iu1[sel]
        csel = c_all[sel]
        p = len(I)
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d),
                                           np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv)
            obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq,
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
        yn = np.linalg.norm(y, axis=1, keepdims=True)
        yn[yn == 0] = 1.0
        y = y / yn
        v = _max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x, best


def _anneal(x, steps, T0, T1, s0, s1):
    x = x.copy()
    for s in range(steps):
        frac = s / max(1, steps - 1)
        T = T0 * ((T1 / T0) ** frac) if T0 != T1 else T0
        step = s0 + (s1 - s0) * frac
        G = x @ x.T
        np.fill_diagonal(G, -1e9)
        m = float(G.max())
        W = np.exp((G - m) / T)
        np.fill_diagonal(W, 0.0)
        rs = W.sum(axis=1, keepdims=True) + 1e-12
        grad = (W @ x) / rs
        gd = (grad * x).sum(axis=1, keepdims=True)
        gt = grad - gd * x
        x = x - step * gt
        xn = np.linalg.norm(x, axis=1, keepdims=True)
        xn[xn == 0] = 1.0
        x = x / xn
    return x


def _riesz(x, steps, s_pow, lr0, lr1):
    x = x.copy()
    n, d = x.shape
    for it in range(steps):
        frac = it / max(1, steps - 1)
        lr = lr0 + (lr1 - lr0) * frac
        G = x @ x.T
        G = np.clip(G, -1.0, 0.999999)
        r2 = np.maximum(2.0 - 2.0 * G, 1e-12)
        np.fill_diagonal(r2, np.inf)
        # weight = r^{-s-2}
        W = np.power(r2, -s_pow / 2.0 - 1.0)
        np.fill_diagonal(W, 0.0)
        rs = W.sum(axis=1, keepdims=True) + 1e-12
        A = (W @ x) / rs
        ad = (A * x).sum(axis=1, keepdims=True)
        Ft = A - ad * x
        # normalize force scale
        fn = np.linalg.norm(Ft, axis=1, keepdims=True).mean() + 1e-12
        Ft = Ft / fn
        x = x - lr * Ft
        xn = np.linalg.norm(x, axis=1, keepdims=True)
        xn[xn == 0] = 1.0
        x = x / xn
    return x


def _void_fill(rem, k, rng, n_cand=4000):
    d = rem.shape[1]
    cand = rng.standard_normal((n_cand, d))
    cand /= np.linalg.norm(cand, axis=1, keepdims=True)
    s_max = (cand @ rem.T).max(axis=1)
    chosen = []
    for _ in range(k):
        idx = int(np.argmin(s_max))
        pt = cand[idx].copy()
        chosen.append(pt)
        upd = cand @ pt
        np.maximum(s_max, upd, out=s_max)
        s_max[idx] = 2.0
    return np.stack(chosen, axis=0)


def _rand_Q(d, rng, eps):
    M = np.eye(d) + eps * rng.standard_normal((d, d))
    Q, R = np.linalg.qr(M)
    dd = np.diag(R)
    s = np.sign(dd)
    s[s == 0] = 1.0
    Q = Q * s[None, :]
    if np.linalg.det(Q) < 0:
        Q[:, 0] = -Q[:, 0]
    return Q


def _haar_Q(d, rng):
    M = rng.standard_normal((d, d))
    Q, R = np.linalg.qr(M)
    dd = np.diag(R)
    s = np.sign(dd)
    s[s == 0] = 1.0
    Q = Q * s[None, :]
    if np.linalg.det(Q) < 0:
        Q[:, 0] = -Q[:, 0]
    return Q


def solve():
    t0 = time.time()
    BUDGET = 168.0
    rng = np.random.default_rng(7)
    n, d = 173, 5
    cands = []
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=float)
            if a.shape == (n, d):
                nr = np.linalg.norm(a, axis=1, keepdims=True)
                nr[nr == 0] = 1.0
                a = a / nr
                if np.all(np.isfinite(a)):
                    cands.append(a)
        except Exception:
            pass
    if not cands:
        a = rng.standard_normal((n, d))
        a /= np.linalg.norm(a, axis=1, keepdims=True)
        cands.append(a)
    x_best = None
    best = 2.0
    for a in cands:
        v = _max_cos(a)
        if v < best:
            best = v
            x_best = a.copy()
    pool = [(best, x_best.copy())]

    def base_pick(it):
        # mostly best, sometimes diverse restart
        if len(pool) == 1 or (it % 5 != 4):
            return pool[0][1]
        # pick random pool member
        k = rng.integers(len(pool))
        return pool[k][1]

    it = 0
    while True:
        remain = BUDGET - (time.time() - t0)
        if remain < 7.0:
            break
        it += 1
        mode = it % 7
        xb = base_pick(it)
        x_try = xb.copy()
        try:
            if mode == 0:
                # overload -> rattler transplant: make rattler load-bearing
                rm = _row_max(x_try)
                ratt = int(np.argmin(rm))
                worst = np.argsort(rm)[-4:]
                # generate points near rattler
                sig = 0.25 + 0.1 * rng.random()
                pts = np.repeat(x_try[ratt:ratt + 1], len(worst), axis=0) + sig * rng.standard_normal((len(worst), d))
                pts /= np.linalg.norm(pts, axis=1, keepdims=True)
                # spread them: repel among themselves a bit
                x_try[worst] = pts
                # jitter rattler neighbours slightly
                x_try = _riesz(x_try, 120, 8.0, 0.05, 0.02)
                x_try = _anneal(x_try, 450, 0.02, 0.002, 0.35, 0.12)
            elif mode == 1:
                # greedy void refill of worst points
                rm = _row_max(x_try)
                k = int(rng.integers(6, 16))
                worst = np.argsort(rm)[-k:]
                mask = np.ones(n, dtype=bool)
                mask[worst] = False
                rem = x_try[mask]
                new_pts = _void_fill(rem, k, rng, n_cand=4000)
                # small jitter
                new_pts = new_pts + 0.01 * rng.standard_normal(new_pts.shape)
                new_pts /= np.linalg.norm(new_pts, axis=1, keepdims=True)
                x_try[worst] = new_pts
                x_try = _anneal(x_try, 550, 0.025, 0.002, 0.4, 0.12)
            elif mode == 2:
                # rigid cap rotation (coherent move)
                u = rng.standard_normal(d)
                u /= np.linalg.norm(u)
                proj = x_try @ u
                m = int(rng.integers(10, 30))
                cap_idx = np.argsort(proj)[-m:]
                eps = 0.15 + 0.35 * rng.random()
                Q = _rand_Q(d, rng, eps)
                x_try[cap_idx] = x_try[cap_idx] @ Q.T
                x_try /= np.linalg.norm(x_try, axis=1, keepdims=True)
                x_try = _anneal(x_try, 550, 0.02, 0.002, 0.35, 0.12)
            elif mode == 3:
                # subset orthogonal remix: large topology jump
                u = rng.standard_normal(d)
                u /= np.linalg.norm(u)
                proj = x_try @ u
                # take random cap ~ half
                thr = np.quantile(proj, 0.5 + 0.2 * rng.random())
                S = np.where(proj > thr)[0]
                if len(S) < 5 or len(S) > n - 5:
                    S = rng.choice(n, size=40, replace=False)
                Q = _haar_Q(d, rng)
                x_try[S] = x_try[S] @ Q.T
                x_try /= np.linalg.norm(x_try, axis=1, keepdims=True)
                x_try = _riesz(x_try, 100, 6.0, 0.06, 0.02)
                x_try = _anneal(x_try, 650, 0.03, 0.002, 0.45, 0.12)
            elif mode == 4:
                # random subset reinit + void fill
                k = int(rng.integers(12, 32))
                idx = rng.choice(n, size=k, replace=False)
                mask = np.ones(n, dtype=bool)
                mask[idx] = False
                rem = x_try[mask]
                new_pts = _void_fill(rem, k, rng, n_cand=5000)
                x_try[idx] = new_pts
                x_try = _anneal(x_try, 650, 0.03, 0.002, 0.45, 0.12)
            elif mode == 5:
                # global shake + two-stage
                sig = 0.05 + 0.25 * rng.random()
                x_try = x_try + sig * rng.standard_normal(x_try.shape)
                x_try /= np.linalg.norm(x_try, axis=1, keepdims=True)
                x_try = _riesz(x_try, 80, 10.0, 0.04, 0.015)
                x_try = _anneal(x_try, 600, 0.03, 0.003, 0.4, 0.13)
            else:
                # fresh diverse start via void incremental + anneal
                a = rng.standard_normal((n, d))
                a /= np.linalg.norm(a, axis=1, keepdims=True)
                # mix with best: keep 60%
                keep = rng.choice(n, size=100, replace=False)
                a[keep] = xb[keep]
                a /= np.linalg.norm(a, axis=1, keepdims=True)
                x_try = _riesz(a, 150, 5.0, 0.08, 0.02)
                x_try = _anneal(x_try, 700, 0.04, 0.003, 0.5, 0.13)
        except Exception:
            continue
        v_try = _max_cos(x_try)
        # filter hopeless before expensive LP
        if v_try > best + 0.015 and not (it % 7 == 6):
            # still maybe keep diversity without LP
            continue
        lp_budget = min(2.8, BUDGET - (time.time() - t0) - 3.0)
        if lp_budget < 0.8:
            break
        try:
            x_pol, v_pol = _lp_polish(x_try, time.time() + lp_budget, radius0=3e-3)
        except Exception:
            continue
        if v_pol < best - 1e-12:
            best = v_pol
            x_best = x_pol.copy()
            pool[0] = (best, x_best.copy())
            # prune pool worse than best+0.004
            pool = [p for p in pool if p[0] < best + 0.004][:5]
        elif v_pol < best + 0.004:
            # add diverse if far in config space (cheap proxy: mean abs Gram diff)
            try:
                diverse = True
                for _, px in pool:
                    # correlation after no alignment: quick check
                    if np.mean((x_pol - px) ** 2) < 1e-6:
                        diverse = False
                        break
                if diverse and len(pool) < 5:
                    pool.append((v_pol, x_pol.copy()))
                    # keep sorted
                    pool = sorted(pool, key=lambda z: z[0])[:5]
                elif diverse and v_pol < pool[-1][0] - 2e-5:
                    pool[-1] = (v_pol, x_pol.copy())
                    pool = sorted(pool, key=lambda z: z[0])[:5]
            except Exception:
                pass

    remain = BUDGET - (time.time() - t0)
    if remain > 2.0 and x_best is not None:
        try:
            x_best, best = _lp_polish(x_best, time.time() + remain - 0.5, radius0=5e-4)
        except Exception:
            pass
    x_best = x_best / np.linalg.norm(x_best, axis=1, keepdims=True)
    return x_best
