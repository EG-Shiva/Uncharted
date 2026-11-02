import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def solve():
    t0 = time.time()
    DEADLINE = t0 + 170.0
    rng = np.random.default_rng(0)

    # ---- load start ----
    try:
        b = np.load("best.npy")
    except Exception:
        b = None
    try:
        r = np.load("record.npy")
    except Exception:
        r = None
    cands = []
    for a in (b, r):
        if a is not None:
            a = np.asarray(a, dtype=float)
            if a.shape == (74, 13):
                a = a / np.linalg.norm(a, axis=1, keepdims=True)
                cands.append(( _max_cos(a), a))
    if len(cands) == 0:
        a = rng.standard_normal((74, 13))
        a /= np.linalg.norm(a, axis=1, keepdims=True)
        cands.append((_max_cos(a), a))
    cands.sort(key=lambda t: t[0])
    Xbest = cands[0][1].copy()
    best_val = cands[0][0]
    n, d = Xbest.shape  # 74,13
    iu0 = np.triu_indices(n, 1)
    iu_r = iu0[0]
    iu_c = iu0[1]

    # ---------- helpers ----------
    def eval_max(X):
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        return float(G.max())

    def per_point_max_idx(X):
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        pm = G.max(axis=1)
        order = np.argsort(-pm)
        # worst pair
        k = int(np.argmax(G))  # flat over n*n, diag is -2 so not chosen
        a = k // n; bb = k % n
        return pm, order, (a, bb)

    def adam_stages(X, stages, track_best=True):
        # stages: list of (tau, steps, lr)
        nonlocal_best_v = eval_max(X)
        nonlocal_best = X.copy()
        cur = X.copy()
        m1 = np.zeros_like(cur)
        m2 = np.zeros_like(cur)
        b1, b2, eps = 0.9, 0.999, 1e-8
        step_counter = 0
        for (tau, steps, lr) in stages:
            # reset moments per stage for stability
            m1[:] = 0; m2[:] = 0
            for s in range(steps):
                step_counter += 1
                G = cur @ cur.T
                g = G[iu_r, iu_c]  # 2701
                mx = g.max()
                # softmax weights
                # (g-mx)/tau <=0
                w = np.exp((g - mx) / tau)
                sw = w.sum()
                if sw == 0 or not np.isfinite(sw):
                    break
                w /= sw
                S = np.zeros((n, n), dtype=float)
                S[iu_r, iu_c] = w
                S = S + S.T
                grad = S @ cur
                # tangent projection
                dot = np.sum(grad * cur, axis=1, keepdims=True)
                gt = grad - dot * cur
                # adam
                m1 = b1 * m1 + (1 - b1) * gt
                m2 = b2 * m2 + (1 - b2) * (gt * gt)
                c1 = 1 - b1 ** step_counter
                c2 = 1 - b2 ** step_counter
                m1h = m1 / c1
                m2h = m2 / c2
                upd = m1h / (np.sqrt(m2h) + eps)  # scaled ~ +-1
                # project update to tangent
                d2 = np.sum(upd * cur, axis=1, keepdims=True)
                upd_t = upd - d2 * cur
                # trust: clip per-row norm
                nr = np.linalg.norm(upd_t, axis=1, keepdims=True) + 1e-16
                # lr is absolute move; upd_t ~ O(1), so scale
                # clip to max 1.0 then * lr
                scale = np.minimum(1.0, 1.0 / nr)
                # actually upd_t already O(1); just apply lr
                cur = cur - lr * upd_t * scale
                # retraction
                nn = np.linalg.norm(cur, axis=1, keepdims=True)
                nn[nn == 0] = 1.0
                cur /= nn
                if track_best and (step_counter % 50 == 0 or s == steps - 1):
                    v = eval_max(cur)
                    if v < nonlocal_best_v:
                        nonlocal_best_v = v
                        nonlocal_best = cur.copy()
                if time.time() > DEADLINE - 26.0:
                    break
            if time.time() > DEADLINE - 26.0:
                break
            step_counter = 0  # reset bias correction per stage? keep continuous? reset ok
        if track_best:
            return nonlocal_best, nonlocal_best_v
        else:
            v = eval_max(cur)
            return cur, v

    def micro_separate(X, val, eps_list=None):
        # hill climb on worst pair along steepest separation + random tangent tries
        if eps_list is None:
            eps_list = [3e-4, 1e-4, 3e-5, 1e-5, 3e-6, 1e-6, 3e-3, 1e-3]
        cur = X.copy()
        cur_v = val
        for _ in range(12):
            G = cur @ cur.T
            np.fill_diagonal(G, -2.0)
            flat = int(np.argmax(G))
            a = flat // n; bb = flat % n
            improved = False
            # try moving a away from bb and bb away from a
            for idx, other in ((a, bb), (bb, a)):
                xi = cur[idx]; xj = cur[other]
                dot = float(xi @ xj)
                direc = xj - dot * xi
                nd = np.linalg.norm(direc)
                if nd < 1e-14:
                    continue
                direc /= nd
                for eps in eps_list:
                    trial = cur.copy()
                    nv = xi - eps * direc
                    nv /= np.linalg.norm(nv)
                    trial[idx] = nv
                    v = eval_max(trial)
                    if v < cur_v - 1e-12:
                        cur = trial; cur_v = v; improved = True
                        break
                if improved:
                    break
            if improved:
                continue
            # random tangent moves for a,bb
            for _r in range(30):
                for idx in (a, bb):
                    xi = cur[idx]
                    z = rng.standard_normal(d)
                    z = z - (z @ xi) * xi
                    nz = np.linalg.norm(z)
                    if nz < 1e-14:
                        continue
                    z /= nz
                    eps = float(rng.choice(np.array([2e-5, 5e-5, 1.5e-4, 4e-4])))
                    trial = cur.copy()
                    nv = xi + eps * z
                    nv /= np.linalg.norm(nv)
                    trial[idx] = nv
                    v = eval_max(trial)
                    if v < cur_v - 1e-12:
                        cur = trial; cur_v = v; improved = True
                        break
                if improved:
                    break
            if not improved:
                break
        return cur, cur_v

    def lp_polish(X, seconds):
        X = X / np.linalg.norm(X, axis=1, keepdims=True)
        best = eval_max(X)
        bestX = X.copy()
        radius = 1e-3
        start = time.time()
        while time.time() - start < seconds and radius > 1e-14:
            if time.time() > DEADLINE - 2.0:
                break
            G = X @ X.T
            c = G[iu_r, iu_c]
            m = c.max()
            thr = max(20 * radius, 1e-9)
            sel = c > m - thr
            # cap size for speed
            p_all = int(sel.sum())
            if p_all == 0:
                radius *= 0.5
                continue
            if p_all > 1500:
                # keep largest 1500
                idx = np.where(sel)[0]
                order = np.argsort(-c[idx])[:1500]
                sel2 = np.zeros_like(sel, dtype=bool)
                sel2[idx[order]] = True
                sel = sel2
            I = iu_r[sel]; J = iu_c[sel]
            p = len(I)
            nv = n * d + 1
            rows = np.repeat(np.arange(p), 2 * d + 1)
            cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
            vals = np.concatenate([X[J], X[I], -np.ones((p, 1))], 1).ravel()
            try:
                A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
                A_eq = coo_matrix((X.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
                obj = np.zeros(nv); obj[-1] = 1.0
                res = linprog(obj, A_ub=A_ub, b_ub=-c[sel], A_eq=A_eq, b_eq=np.zeros(n),
                              bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
            except Exception:
                radius *= 0.5
                continue
            if res.status != 0:
                radius *= 0.5
                continue
            y = X + res.x[:-1].reshape(n, d)
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            v = eval_max(y)
            if v < best:
                X, best = y, v
                bestX = y.copy()
                radius = min(radius * 1.5, 0.05)
            else:
                radius *= 0.5
        return bestX

    # ---- immediate micro fix + light smooth polish ----
    Xbest, best_val = micro_separate(Xbest, best_val)
    Xp, vp = adam_stages(Xbest, [(0.008, 200, 0.01), (0.002, 300, 0.006), (0.0007, 300, 0.004)])
    if vp < best_val:
        Xbest, best_val = Xp, vp
    Xbest, best_val = micro_separate(Xbest, best_val)

    # ---- basin hopping ----
    sigmas = [0.0004, 0.0012, 0.004, 0.012, 0.035, 0.09]
    n_trials = 300
    for trial in range(n_trials):
        if time.time() > DEADLINE - 26.0:
            break
        sigma = sigmas[trial % len(sigmas)]
        # worst points
        _, order, _ = per_point_max_idx(Xbest)
        if trial % 2 == 0:
            # global shake
            Z = rng.standard_normal((n, d))
            # tangent
            dprod = np.sum(Z * Xbest, axis=1, keepdims=True)
            Zt = Z - dprod * Xbest
            nrm = np.linalg.norm(Zt, axis=1, keepdims=True) + 1e-16
            Zt /= nrm
            Xt = Xbest + sigma * Zt
        else:
            k = [2, 4, 8, 16][(trial // 2) % 4]
            worst = order[:k]
            Z = rng.standard_normal((n, d))
            dprod = np.sum(Z * Xbest, axis=1, keepdims=True)
            Zt = Z - dprod * Xbest
            nrm = np.linalg.norm(Zt, axis=1, keepdims=True) + 1e-16
            Zt /= nrm
            step = np.full((n, 1), sigma * 0.12)
            step[worst] = sigma
            Xt = Xbest + step * Zt
        Xt /= np.linalg.norm(Xt, axis=1, keepdims=True)
        # polish trial: shorter for large sigma exploration, longer for small
        if sigma < 0.005:
            stages = [(0.005, 150, 0.008), (0.0015, 250, 0.005), (0.0005, 250, 0.003)]
        else:
            stages = [(0.01, 150, 0.01), (0.003, 250, 0.007), (0.001, 250, 0.004)]
        Xt_p, vt_p = adam_stages(Xt, stages)
        # micro separate on trial
        Xt_p, vt_p = micro_separate(Xt_p, vt_p)
        if vt_p < best_val - 1e-12:
            Xbest, best_val = Xt_p, vt_p

    # ---- final squeeze ----
    remain = DEADLINE - time.time()
    if remain > 8.0:
        # small final adam with tiny tau to equalize
        Xf, vf = adam_stages(Xbest, [(0.001, 400, 0.004), (0.00035, 400, 0.002)])
        if vf < best_val:
            Xbest, best_val = Xf, vf
        Xbest, best_val = micro_separate(Xbest, best_val)
    remain = DEADLINE - time.time()
    # use leftover (up to ~22s) for LP
    if remain > 4.0:
        lp_time = min(remain - 1.0, 22.0)
        Xlp = lp_polish(Xbest, lp_time)
        vlp = eval_max(Xlp)
        if vlp < best_val:
            Xbest, best_val = Xlp, vlp
        # one more tiny adam after LP
        if time.time() < DEADLINE - 1.0:
            Xbest, best_val = micro_separate(Xbest, best_val)

    Xbest /= np.linalg.norm(Xbest, axis=1, keepdims=True)
    # safety: remove NaN
    Xbest = np.nan_to_num(Xbest, nan=0.0, posinf=0.0, neginf=0.0)
    nn = np.linalg.norm(Xbest, axis=1, keepdims=True)
    nn[nn == 0] = 1.0
    Xbest /= nn
    return Xbest
