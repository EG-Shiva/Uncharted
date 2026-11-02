import time
import numpy as np
from scipy.optimize import linprog, minimize
from scipy.sparse import coo_matrix


def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())


def _lp_polish(x, deadline, r0=1e-3):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    try:
        best = _max_cos(x)
    except Exception:
        return x
    radius = r0
    iu0, iu1 = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        try:
            G = x @ x.T
            c = G[iu0, iu1]
            m = c.max()
            sel_thr = m - max(20 * radius, 1e-9)
            sel = c > sel_thr
            if np.count_nonzero(sel) == 0:
                sel[np.argmax(c)] = True
            I = iu0[sel]; J = iu1[sel]
            csel = c[sel]
            p = len(I)
            nv = n * d + 1
            rows = np.repeat(np.arange(p), 2 * d + 1)
            cI = np.arange(d)[None, :].repeat(p, axis=0)
            # columns: I*d+o, J*d+o, last
            cols_I = I[:, None] * d + cI
            cols_J = J[:, None] * d + cI
            cols_T = np.full((p, 1), n * d)
            cols = np.concatenate([cols_I, cols_J, cols_T], axis=1).ravel()
            vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], axis=1).ravel()
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv); obj[-1] = 1.0
            bounds = [(-radius, radius)] * (n * d) + [(None, None)]
            res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=bounds, method="highs")
            if res.status != 0:
                radius *= 0.5
                continue
            y = x + res.x[:-1].reshape(n, d)
            yn = np.linalg.norm(y, axis=1, keepdims=True)
            yn[yn < 1e-12] = 1.0
            y = y / yn
            v = _max_cos(y)
            if v < best:
                x, best = y, v
                radius = min(radius * 1.5, 0.05)
            else:
                radius *= 0.5
        except Exception:
            radius *= 0.5
            continue
    return x


def _smooth_polish(x_init, deadline, taus, iters):
    x = x_init / np.linalg.norm(x_init, axis=1, keepdims=True)
    n, d = x.shape
    iu0, iu1 = np.triu_indices(n, 1)
    Np = len(iu0)
    y = x.ravel().copy()
    for tau, maxiter in zip(taus, iters):
        if time.time() > deadline - 0.2:
            break
        def fg(yflat, _tau=tau):
            Y = yflat.reshape(n, d)
            rn = np.linalg.norm(Y, axis=1, keepdims=True)
            rn = np.maximum(rn, 1e-12)
            X = Y / rn
            G = X @ X.T
            g = G[iu0, iu1]
            m = g.max()
            e = np.exp((g - m) / _tau)
            s = e.sum()
            if s <= 0 or not np.isfinite(s):
                return float(m), np.zeros_like(yflat)
            L = m + _tau * np.log(s / Np)
            w = e / s
            W = np.zeros((n, n))
            W[iu0, iu1] = w
            W = W + W.T
            GX = W @ X
            dot = (GX * X).sum(axis=1, keepdims=True)
            GY = (GX - dot * X) / rn
            return float(L), GY.ravel()
        try:
            res = minimize(fg, y, method="L-BFGS-B", jac=True,
                           options={"maxiter": int(maxiter), "maxfun": int(maxiter * 5),
                                    "ftol": 1e-14, "gtol": 1e-9})
            y = res.x
        except Exception:
            break
        # renormalize y to avoid drift
        try:
            Yt = y.reshape(n, d)
            nt = np.linalg.norm(Yt, axis=1, keepdims=True)
            nt[nt < 1e-12] = 1.0
            Yt = Yt / nt
            y = Yt.ravel()
        except Exception:
            break
    Xf = y.reshape(n, d)
    Xf = Xf / np.linalg.norm(Xf, axis=1, keepdims=True)
    return Xf


def solve():
    t0 = time.time()
    DEADLINE = t0 + 174.0
    rng = np.random.default_rng(12345)
    np.random.seed(0)
    # load starting points
    best_x = None
    best_v = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=np.float64)
            if a.shape != (163, 4):
                continue
            a = a / np.linalg.norm(a, axis=1, keepdims=True)
            if not np.all(np.isfinite(a)):
                continue
            v = _max_cos(a)
            if v < best_v:
                best_v = v
                best_x = a.copy()
        except Exception:
            continue
    if best_x is None:
        z = rng.standard_normal((163, 4))
        z /= np.linalg.norm(z, axis=1, keepdims=True)
        best_x = z
        best_v = _max_cos(z)
    n, d = best_x.shape

    def contacts_of(X, thr=1e-7):
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        m = G.max()
        per_max = G.max(axis=1)
        cnt = (G > m - thr).sum(axis=1)
        return G, m, per_max, cnt

    # quick rattler hole-fill to start
    try:
        _, m0, per_max0, cnt0 = contacts_of(best_x, thr=1e-9)
        ratt = np.where(cnt0 == 0)[0]
        if len(ratt) > 0:
            # find deep holes with random candidates
            cand = rng.standard_normal((4000, 4))
            cand /= np.linalg.norm(cand, axis=1, keepdims=True)
            # score candidates by max dot to fixed points
            for r in ratt:
                others = np.delete(best_x, r, axis=0)
                dots = cand @ others.T
                md = dots.max(axis=1)
                bi = int(np.argmin(md))
                if md[bi] < per_max0[r]:
                    best_x[r] = cand[bi]
            best_x /= np.linalg.norm(best_x, axis=1, keepdims=True)
            nv = _max_cos(best_x)
            if nv < best_v:
                best_v = nv
    except Exception:
        pass

    trial = 0
    # reserve ~12s for final polish
    while time.time() < DEADLINE - 13.0:
        trial += 1
        if time.time() > DEADLINE - 13.0:
            break
        try:
            G, mcur, per_max, cnt = contacts_of(best_x, thr=1e-9)
        except Exception:
            break
        order_worst = np.argsort(-per_max)
        ratt_idx = np.where(cnt == 0)[0]
        mode = trial % 5
        xs = best_x.copy()
        if mode == 0:  # global small shake
            sigma = [0.02, 0.008, 0.03, 0.012][(trial // 5) % 4]
            xs = best_x + sigma * rng.standard_normal((n, d))
        elif mode == 1:  # subset shake of worst + rattlers
            k = [10, 6, 12, 4][(trial // 5) % 4]
            sigma = [0.06, 0.10, 0.04, 0.15][(trial // 5) % 4]
            idx = list(order_worst[:k]) + list(ratt_idx)
            idx = np.unique(np.array(idx, dtype=int))
            # add few random to diversify
            extra = rng.choice(n, size=min(3, n), replace=False)
            idx = np.unique(np.concatenate([idx, extra]))
            xs[idx] = best_x[idx] + sigma * rng.standard_normal((len(idx), d))
        elif mode == 2:  # worst-point reinsertion
            k = [3, 5, 2, 4][(trial // 5) % 4]
            rem = order_worst[:k]
            keep_mask = np.ones(n, dtype=bool)
            keep_mask[rem] = False
            kept = best_x[keep_mask]
            # greedy farthest insertion from big candidate pool
            pool = rng.standard_normal((800, 4))
            pool /= np.linalg.norm(pool, axis=1, keepdims=True)
            # score pool against kept
            scored = (pool @ kept.T).max(axis=1)
            # pick k diverse best: iteratively pick best and penalize closeness
            chosen = []
            pool_left = pool.copy()
            score_left = scored.copy()
            for _ in range(k):
                bi = int(np.argmin(score_left))
                chosen.append(pool_left[bi])
                # penalize neighbours of chosen
                dd = pool_left @ chosen[-1]
                score_left = np.maximum(score_left, dd - 0.02 * rng.random(len(score_left)))
                score_left[bi] = 1e9
            for r, ch in zip(rem, chosen):
                xs[r] = ch + 0.005 * rng.standard_normal(d)
        elif mode == 3:  # rattler + neighbourhood shake
            sigma = 0.08
            # pick most overloaded region
            order_best_contact = np.argsort(-cnt)
            idx = list(order_best_contact[:6]) + list(ratt_idx)
            idx = np.unique(np.array(idx, dtype=int))
            xs[idx] = best_x[idx] + sigma * rng.standard_normal((len(idx), d))
        else:  # medium global + reinsert one worst
            sigma = 0.01
            xs = best_x + sigma * rng.standard_normal((n, d))
            # relocate single worst to hole
            w = int(order_worst[0])
            cand = rng.standard_normal((2000, 4))
            cand /= np.linalg.norm(cand, axis=1, keepdims=True)
            others = best_x[np.arange(n) != w]
            md = (cand @ others.T).max(axis=1)
            xs[w] = cand[int(np.argmin(md))]
        xs /= np.linalg.norm(xs, axis=1, keepdims=True)
        # smooth phase with limited budget
        now = time.time()
        s_dead = min(now + 3.2, DEADLINE - 10.0)
        if s_dead <= now + 0.3:
            break
        # choose sharpness by trial
        if trial % 2 == 0:
            taus = [0.015, 0.006, 0.0025]
            iters = [60, 60, 80]
        else:
            taus = [0.02, 0.008, 0.003]
            iters = [50, 50, 70]
        try:
            ys = _smooth_polish(xs, s_dead, taus, iters)
        except Exception:
            continue
        now2 = time.time()
        l_dead = min(now2 + 2.2, DEADLINE - 10.0)
        if l_dead > now2 + 0.3:
            try:
                ys = _lp_polish(ys, l_dead, r0=8e-4)
            except Exception:
                pass
        try:
            ys /= np.linalg.norm(ys, axis=1, keepdims=True)
            v = _max_cos(ys)
        except Exception:
            continue
        if v < best_v - 1e-12:
            best_x = ys
            best_v = v
            # opportunistic extra polish on improvement if time allows
            if time.time() < DEADLINE - 14.0:
                try:
                    best_x = _lp_polish(best_x, min(time.time() + 2.0, DEADLINE - 10.0), r0=5e-4)
                    best_x /= np.linalg.norm(best_x, axis=1, keepdims=True)
                    best_v = _max_cos(best_x)
                except Exception:
                    pass
        # early exit if clearly record-beating and time low? continue to improve
    # final thorough polish of best
    try:
        remain = DEADLINE - time.time() - 1.0
        if remain > 2.0:
            # one more smooth refinement at small tau
            sd = time.time() + min(4.0, remain * 0.35)
            best_x = _smooth_polish(best_x, sd, [0.006, 0.0025, 0.0012], [80, 120, 150])
            best_x /= np.linalg.norm(best_x, axis=1, keepdims=True)
        remain = DEADLINE - time.time() - 0.5
        if remain > 1.0:
            best_x = _lp_polish(best_x, time.time() + remain, r0=1e-3)
            best_x /= np.linalg.norm(best_x, axis=1, keepdims=True)
    except Exception:
        pass
    best_x = best_x / np.linalg.norm(best_x, axis=1, keepdims=True)
    # safety: replace non-finite
    if not np.all(np.isfinite(best_x)):
        best_x = np.load("best.npy").astype(np.float64)
        best_x /= np.linalg.norm(best_x, axis=1, keepdims=True)
    return best_x
