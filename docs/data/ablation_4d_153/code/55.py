import time
import numpy as np

def _norm_rows(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return x / n

def _max_cos(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def _row_max(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    rm = np.max(G, axis=1)
    return rm, float(np.max(G))

def _lp_polish(x, deadline, r0=0.004, cap=2500, maxit=80):
    from scipy.optimize import linprog
    from scipy.sparse import coo_matrix
    x = _norm_rows(x.copy())
    n, d = x.shape
    try:
        best = _max_cos(x)
    except Exception:
        return x, 1e9
    best_x = x.copy()
    iu0 = np.triu_indices(n, 1)
    radius = float(r0)
    it = 0
    while time.time() < deadline and radius > 1e-14 and it < maxit:
        it += 1
        G = x @ x.T
        c = G[iu0]
        m = float(np.max(c))
        thr = m - max(20.0 * radius, 1e-9)
        idx = np.where(c > thr)[0]
        if idx.size == 0:
            radius *= 0.5
            continue
        if idx.size > cap:
            part = np.argpartition(c[idx], -cap)[-cap:]
            idx = idx[part]
        I = iu0[0][idx]; J = iu0[1][idx]
        csel = c[idx]
        p = idx.size
        nv = n * d + 1
        try:
            rows = np.repeat(np.arange(p), 2 * d + 1)
            cI = I[:, None] * d + np.arange(d)
            cJ = J[:, None] * d + np.arange(d)
            cols = np.concatenate([cI, cJ, np.full((p, 1), n * d)], axis=1).ravel()
            vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], axis=1).ravel()
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv); obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        try:
            dx = res.x[:-1].reshape(n, d)
        except Exception:
            radius *= 0.5
            continue
        y = _norm_rows(x + dx)
        try:
            v = _max_cos(y)
        except Exception:
            radius *= 0.5
            continue
        if v < best:
            x, best = y, v
            best_x = y.copy()
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return best_x, best

def _smooth_patch(x, patch, steps, tau0, tau1, lr0, lr1, mom=0.85, rng=None):
    n, d = x.shape
    patch = np.asarray(patch, dtype=int)
    mask = np.zeros(n, dtype=bool)
    mask[patch] = True
    vel = np.zeros((len(patch), d))
    lt0, lt1 = np.log(tau0), np.log(tau1)
    ll0, ll1 = np.log(lr0), np.log(lr1)
    # map global idx -> local pos for vel; simpler keep full vel
    vfull = np.zeros_like(x)
    for t in range(steps):
        f = t / max(1, steps - 1)
        tau = float(np.exp(lt0 + (lt1 - lt0) * f))
        lr = float(np.exp(ll0 + (ll1 - ll0) * f))
        G = x @ x.T
        np.fill_diagonal(G, -np.inf)
        m = float(np.max(G))
        # weights
        W = np.exp((G - m) / tau)
        np.fill_diagonal(W, 0.0)
        row = np.sum(W, axis=1, keepdims=True)
        row[row < 1e-14] = 1e-14
        grad = (W @ x) / row
        dot = np.sum(grad * x, axis=1, keepdims=True)
        grad = grad - dot * x
        # only patch moves
        g = grad[mask]
        # clip
        gn = np.linalg.norm(g, axis=1, keepdims=True)
        scale = np.ones_like(gn)
        big = gn > 2.0
        scale[big] = 2.0 / np.maximum(gn[big], 1e-12)
        g = g * scale
        vfull[mask] = mom * vfull[mask] + (1.0 - mom) * g
        bc = 1.0 - mom ** (t + 1)
        step = vfull[mask] / bc
        x[mask] = x[mask] - lr * step
        # renormalize patch
        nr = np.linalg.norm(x[mask], axis=1, keepdims=True)
        nr[nr == 0] = 1.0
        x[mask] = x[mask] / nr
    return x

def _smooth_full(x, steps, tau0, tau1, lr0, lr1, mom=0.9):
    n, d = x.shape
    vel = np.zeros_like(x)
    lt0, lt1 = np.log(tau0), np.log(tau1)
    ll0, ll1 = np.log(lr0), np.log(lr1)
    for t in range(steps):
        f = t / max(1, steps - 1)
        tau = float(np.exp(lt0 + (lt1 - lt0) * f))
        lr = float(np.exp(ll0 + (ll1 - ll0) * f))
        G = x @ x.T
        np.fill_diagonal(G, -np.inf)
        m = float(np.max(G))
        W = np.exp((G - m) / tau)
        np.fill_diagonal(W, 0.0)
        row = np.sum(W, axis=1, keepdims=True)
        row[row < 1e-12] = 1e-12
        grad = (W @ x) / row
        dot = np.sum(grad * x, axis=1, keepdims=True)
        grad = grad - dot * x
        gn = np.linalg.norm(grad, axis=1, keepdims=True)
        big = gn > 2.0
        if np.any(big):
            grad = grad * np.where(big, 2.0 / np.maximum(gn, 1e-12), 1.0)
        vel = mom * vel + (1.0 - mom) * grad
        bc = 1.0 - mom ** (t + 1)
        step = vel / bc
        x = x - lr * step
        nr = np.linalg.norm(x, axis=1, keepdims=True)
        nr[nr == 0] = 1.0
        x = x / nr
    return x

def _deg_key(x, tol=5e-7):
    try:
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        m = float(np.max(G))
        B = G > (m - tol)
        deg = tuple(sorted(np.sum(B, axis=1).tolist()))
        return (deg, round(m * 1e7))
    except Exception:
        return None

def solve():
    t0 = time.time()
    HARD_END = t0 + 172.0
    EXPLORE_END = t0 + 148.0
    n, d = 153, 4
    rng = np.random.default_rng(20261031)
    best_x = None; best_v = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=np.float64)
            if a.shape == (n, d) and np.all(np.isfinite(a)):
                a = _norm_rows(a)
                v = _max_cos(a)
                if v < best_v:
                    best_v = v; best_x = a.copy()
        except Exception:
            pass
    if best_x is None:
        best_x = _norm_rows(rng.standard_normal((n, d)))
        best_v = _max_cos(best_x)
    # initial jam
    try:
        best_x, best_v = _lp_polish(best_x, min(HARD_END - 10.0, t0 + 12.0), r0=0.003)
    except Exception:
        pass
    cur_x = best_x.copy(); cur_v = best_v
    seen = set()
    k0 = _deg_key(best_x)
    if k0 is not None:
        seen.add(k0)
    # precompute triu for hole search etc.
    patch_sizes = [12, 18, 25, 35]
    it = 0
    # quick full smooth diversification pool evaluated without LP first
    while time.time() < EXPLORE_END:
        it += 1
        time_left = HARD_END - time.time()
        if time_left < 8.0:
            break
        mode = it % 10
        base = cur_x if (it % 3 != 0) else best_x
        y = base.copy()
        if mode <= 5:
            # patch melt
            k = patch_sizes[(it // 2) % len(patch_sizes)]
            # choose center: worst point half the time
            try:
                rm, _ = _row_max(base)
                if rng.random() < 0.55:
                    center = int(np.argmax(rm))
                else:
                    center = int(rng.integers(0, n))
                Gc = base @ base[center]
                order = np.argsort(-Gc)
                patch = order[:k]
            except Exception:
                patch = rng.choice(n, size=k, replace=False)
                center = int(patch[0])
            # randomize patch: full random or big kick
            r = rng.random()
            if r < 0.5:
                y[patch] = _norm_rows(rng.standard_normal((k, d)))
            elif r < 0.8:
                noise = rng.standard_normal((k, d)) * 0.35
                y[patch] = _norm_rows(y[patch] + noise)
            else:
                # rotate patch coherently a bit + random
                noise = rng.standard_normal((k, d)) * 0.12
                y[patch] = _norm_rows(y[patch] + noise)
            # anneal patch
            steps = 600 if k <= 18 else 800
            tau0 = 0.02 if (it % 2 == 0) else 0.035
            tau1 = 0.004 + 0.002 * rng.random()
            lr0 = 0.06; lr1 = 0.004
            try:
                y = _smooth_patch(y, patch, steps, tau0, tau1, lr0, lr1, mom=0.85)
            except Exception:
                continue
            # also a few full steps to settle interface
            try:
                y = _smooth_full(y, 120, max(tau1, 0.006), 0.003, 0.01, 0.002, mom=0.9)
            except Exception:
                pass
            try:
                v_pre = _max_cos(y)
            except Exception:
                continue
            # gate: only LP if promising or diverse enough
            # allow worse by up to 0.015 to change basin
            if v_pre > best_v + 0.015:
                # still maybe accept as cur wander? no
                continue
            dk = _deg_key(y)
            is_new = (dk not in seen) if dk is not None else True
            # LP polish short
            lp_budget = 3.0
            if v_pre < best_v + 1e-4 and is_new:
                lp_budget = 4.5
            try:
                yp, vp = _lp_polish(y, min(HARD_END - 4.0, time.time() + lp_budget), r0=0.008)
            except Exception:
                continue
            if dk is not None:
                seen.add(dk)
                try:
                    dk2 = _deg_key(yp)
                    if dk2 is not None:
                        seen.add(dk2)
                except Exception:
                    pass
            if vp < best_v - 1e-12:
                best_x = yp.copy(); best_v = vp
                cur_x = yp.copy(); cur_v = vp
            elif vp < cur_v + 4e-4 and is_new and vp < best_v + 2e-3:
                # sideways drift to new basin
                cur_x = yp.copy(); cur_v = vp
            elif v_pre < cur_v and rng.random() < 0.1:
                cur_x = y.copy(); cur_v = v_pre
        elif mode == 6 or mode == 7:
            # hole reinsertion: remove worst, insert at largest hole
            try:
                rm, _ = _row_max(base)
                worst = int(np.argmax(rm))
                # find hole via sampling
                S = 4000
                samp = _norm_rows(rng.standard_normal((S, d)))
                # max dot to existing (excluding worst)
                keep = np.ones(n, dtype=bool); keep[worst] = False
                Xk = base[keep]
                dots = samp @ Xk.T
                md = np.max(dots, axis=1)
                hole = samp[int(np.argmin(md))]
                # a couple of candidate holes: best + random orthogonal jitter
                y[worst] = hole
                # jitter neighbours of hole slightly?
                Gwh = base @ hole
                nbr = np.argsort(-Gwh)[:10]
                noise = rng.standard_normal((len(nbr), d)) * 0.02
                y[nbr] = _norm_rows(y[nbr] + noise)
                y = _smooth_full(y, 500, 0.02, 0.004, 0.04, 0.003, mom=0.88)
                v_pre = _max_cos(y)
                if v_pre > best_v + 0.01:
                    continue
                yp, vp = _lp_polish(y, min(HARD_END - 4.0, time.time() + 4.0), r0=0.008)
                dk2 = _deg_key(yp)
                if dk2 is not None:
                    seen.add(dk2)
                if vp < best_v - 1e-12:
                    best_x = yp.copy(); best_v = vp
                    cur_x = yp.copy(); cur_v = vp
                elif vp < cur_v + 3e-4:
                    cur_x = yp.copy(); cur_v = vp
            except Exception:
                continue
        else:
            # global kick + full anneal (basin hop)
            try:
                sig = 0.06 if mode == 8 else 0.14
                noise = rng.standard_normal((n, d)) * sig
                dot = np.sum(noise * y, axis=1, keepdims=True)
                noise = noise - dot * y
                y = _norm_rows(y + noise)
                y = _smooth_full(y, 700, 0.025, 0.004, 0.05, 0.003, mom=0.9)
                v_pre = _max_cos(y)
                if v_pre > best_v + 0.01:
                    # keep wandering a bit
                    if v_pre < cur_v:
                        cur_x = y.copy(); cur_v = v_pre
                    continue
                yp, vp = _lp_polish(y, min(HARD_END - 4.0, time.time() + 4.0), r0=0.006)
                dk2 = _deg_key(yp)
                if dk2 is not None:
                    if dk2 in seen and vp >= best_v - 1e-12:
                        # already visited
                        if vp < cur_v:
                            cur_x = yp.copy(); cur_v = vp
                        continue
                    seen.add(dk2)
                if vp < best_v - 1e-12:
                    best_x = yp.copy(); best_v = vp
                    cur_x = yp.copy(); cur_v = vp
                elif vp < cur_v + 3e-4:
                    cur_x = yp.copy(); cur_v = vp
            except Exception:
                continue
        # occasional reset cur to best to intensify
        if it % 25 == 0:
            cur_x = best_x.copy(); cur_v = best_v
    # final intensive polish of best
    try:
        time_left = HARD_END - time.time()
        if time_left > 3.0:
            best_x, best_v = _lp_polish(best_x, time.time() + max(1.0, time_left - 0.5), r0=0.002)
    except Exception:
        pass
    best_x = _norm_rows(best_x)
    return best_x
