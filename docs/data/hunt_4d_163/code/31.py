import time
import numpy as np
from scipy.optimize import linprog, minimize
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _lp_polish(x, deadline, r0=1e-3):
    x = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
    n, d = x.shape
    try:
        best = _max_cos(x)
    except Exception:
        return x, 1.0
    radius = r0
    iu0, iu1 = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        try:
            G = x @ x.T
            c = G[iu0, iu1]
            m = float(c.max())
            sel = c > m - max(20.0 * radius, 1e-9)
            if np.count_nonzero(sel) == 0:
                sel[int(np.argmax(c))] = True
            I = iu0[sel]; J = iu1[sel]
            csel = c[sel]
            p = len(I)
            nv = n * d + 1
            rows = np.repeat(np.arange(p), 2 * d + 1)
            tmpc = np.empty((p, 2 * d + 1), dtype=int)
            ar = np.arange(d)
            tmpc[:, :d] = I[:, None] * d + ar
            tmpc[:, d:2 * d] = J[:, None] * d + ar
            tmpc[:, 2 * d] = n * d
            cols = tmpc.ravel()
            cc = np.empty((p, 2 * d + 1), dtype=float)
            cc[:, :d] = x[J]
            cc[:, d:2 * d] = x[I]
            cc[:, 2 * d] = -1.0
            vals = cc.ravel()
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv); obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
            if res.status != 0:
                radius *= 0.5
                continue
            y = x + res.x[:-1].reshape(n, d)
            y /= np.maximum(np.linalg.norm(y, axis=1, keepdims=True), 1e-12)
            v = _max_cos(y)
            if v < best:
                x, best = y, v
                radius = min(radius * 1.5, 0.05)
            else:
                radius *= 0.5
        except Exception:
            radius *= 0.5
            continue
    return x, best

def _gd_soft(x0, steps, T0, T1, lr0, mu, rng, noise0=0.0, deadline=None):
    x = x0.copy()
    n, d = x.shape
    v = np.zeros_like(x)
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    best_val = float(G.max())
    best = x.copy()
    lratio = np.log(T1 / T0) if T0 > 0 and T1 > 0 else 0.0
    for k in range(steps):
        if deadline is not None and (k & 127) == 0 and time.time() > deadline:
            break
        frac = k / max(1, steps - 1)
        T = T0 * np.exp(lratio * frac)
        if T < 1e-12:
            T = 1e-12
        lr = lr0 * (1.0 - 0.55 * frac)
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        E = np.exp((G - m) / T)
        np.fill_diagonal(E, 0.0)
        S = float(E.sum() * 0.5)
        if S < 1e-300 or not np.isfinite(S):
            S = 1e-300
        grad = (E @ x) / S
        grad -= (grad * x).sum(axis=1, keepdims=True) * x
        v = mu * v + (1.0 - mu) * grad
        x = x - lr * v
        if noise0 > 0:
            amp = noise0 * (1.0 - frac)
            if amp > 0:
                nz = rng.randn(n, d) * amp
                nz -= (nz * x).sum(axis=1, keepdims=True) * x
                x = x + nz
        x /= np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
        if m < best_val:
            best_val = m
            best = x.copy()
    return best, best_val

def _gd_riesz(x0, steps, S0, S1, lr0, mu, rng, noise0=0.0, deadline=None):
    x = x0.copy()
    n, d = x.shape
    v = np.zeros_like(x)
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    best_val = float(G.max())
    best = x.copy()
    lratio = np.log(S1 / S0) if S0 > 0 and S1 > 0 else 0.0
    for k in range(steps):
        if deadline is not None and (k & 127) == 0 and time.time() > deadline:
            break
        frac = k / max(1, steps - 1)
        s = S0 * np.exp(lratio * frac)
        lr = lr0 * (1.0 - 0.5 * frac)
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        R2 = np.maximum(2.0 - 2.0 * G, 1e-9)
        np.fill_diagonal(R2, 1e30)
        # power weights, clipped
        with np.errstate(over='ignore', divide='ignore', invalid='ignore'):
            W = R2 ** (-s * 0.5 - 1.0)
        np.fill_diagonal(W, 0.0)
        W = np.minimum(W, 1e18)
        S = float(W.sum() * 0.5)
        if S < 1e-300 or not np.isfinite(S):
            S = 1e-300
        W = W / S
        grad = (W @ x)
        grad -= (grad * x).sum(axis=1, keepdims=True) * x
        gn = float(np.linalg.norm(grad))
        if gn > 1e3:
            grad *= (1e3 / gn)
        v = mu * v + (1.0 - mu) * grad
        x = x - lr * v
        if noise0 > 0:
            amp = noise0 * (1.0 - frac)
            if amp > 0:
                nz = rng.randn(n, d) * amp
                nz -= (nz * x).sum(axis=1, keepdims=True) * x
                x = x + nz
        x /= np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
        if m < best_val:
            best_val = m
            best = x.copy()
    return best, best_val

def _lbfgs_smooth(x_init, deadline, taus=(0.004, 0.0012, 0.0005), iters=(90, 90, 130)):
    x = x_init / np.maximum(np.linalg.norm(x_init, axis=1, keepdims=True), 1e-12)
    n, d = x.shape
    iu0, iu1 = np.triu_indices(n, 1)
    Np = len(iu0)
    y = x.ravel().copy()
    for tau, maxiter in zip(taus, iters):
        if time.time() > deadline - 0.4:
            break
        def fg(yflat, _tau=tau):
            Y = yflat.reshape(n, d)
            rn = np.maximum(np.linalg.norm(Y, axis=1, keepdims=True), 1e-12)
            X = Y / rn
            g = (X @ X.T)[iu0, iu1]
            m = float(g.max())
            e = np.exp((g - m) / _tau)
            s = float(e.sum())
            if s <= 0 or not np.isfinite(s):
                return float(m), np.zeros_like(yflat)
            L = m + _tau * np.log(s / Np)
            w = e / s
            W = np.zeros((n, n))
            W[iu0, iu1] = w
            W = W + W.T
            GX = W @ X
            GY = (GX - (GX * X).sum(axis=1, keepdims=True) * X) / rn
            return float(L), GY.ravel()
        try:
            res = minimize(fg, y, method="L-BFGS-B", jac=True,
                           options={"maxiter": int(maxiter), "maxfun": int(maxiter * 4),
                                    "ftol": 1e-14, "gtol": 1e-9})
            y = res.x
        except Exception:
            break
    Yf = y.reshape(n, d)
    Yf /= np.maximum(np.linalg.norm(Yf, axis=1, keepdims=True), 1e-12)
    return Yf, _max_cos(Yf)

def _find_hole(x_fixed, rng, n_samples=6000):
    Y = rng.randn(n_samples, 4)
    Y /= np.maximum(np.linalg.norm(Y, axis=1, keepdims=True), 1e-12)
    D = Y @ x_fixed.T
    vals = D.max(axis=1)
    idx = int(np.argmin(vals))
    y = Y[idx].copy()
    vy = float(vals[idx])
    for _ in range(150):
        dots = x_fixed @ y
        j = int(np.argmax(dots))
        g = x_fixed[j] - dots[j] * y
        ng = float(np.linalg.norm(g))
        if ng < 1e-12:
            break
        improved = False
        for step in (0.03, 0.01, 0.003):
            yn = y - step * g / ng
            yn /= max(np.linalg.norm(yn), 1e-12)
            vn = float((x_fixed @ yn).max())
            if vn < vy:
                y, vy = yn, vn
                improved = True
                break
        if not improved:
            break
    return y, vy

def _rand_rot(rng, scale):
    A = rng.randn(4, 4) * scale
    A = A - A.T
    Q, _ = np.linalg.qr(np.eye(4) + A)
    if float(np.linalg.det(Q)) < 0:
        Q[:, 0] = -Q[:, 0]
    return Q

def _crossover(a, b, rng):
    n = a.shape[0]
    na_half = n // 2
    idx_a = rng.choice(n, na_half, replace=False)
    cur_list = [a[i] for i in idx_a]
    cur = np.array(cur_list)
    # greedy farthest fill from b (+ jittered copies)
    pool = b.copy()
    pool += rng.randn(*pool.shape) * 0.01
    pool /= np.maximum(np.linalg.norm(pool, axis=1, keepdims=True), 1e-12)
    used = np.zeros(n, dtype=bool)
    while cur.shape[0] < n:
        D = cur @ pool.T  # (m, n)
        worst = D.max(axis=0)  # for each pool point, closest to cur
        worst[used] = 2.0
        j = int(np.argmin(worst))
        used[j] = True
        cur = np.vstack([cur, pool[j:j+1]])
        if used.all():
            # add random holes if pool exhausted (should not happen since n pool + half = >n? pool has n, need ~82)
            break
    if cur.shape[0] > n:
        cur = cur[:n]
    # if short, fill holes
    while cur.shape[0] < n:
        h, _ = _find_hole(cur, rng, n_samples=4000)
        cur = np.vstack([cur, h.reshape(1, 4)])
    return cur

def solve():
    t0 = time.time()
    DEADLINE = t0 + 172.0
    SEARCH_END = t0 + 126.0
    rng = np.random.RandomState(20261102)
    n = 163
    try:
        bx = np.load("best.npy")
    except Exception:
        bx = None
    try:
        rec = np.load("record.npy")
    except Exception:
        rec = None
    cands = []
    if bx is not None:
        bx = np.asarray(bx, dtype=float)
        if bx.shape == (n, 4) and np.all(np.isfinite(bx)):
            bx = bx / np.maximum(np.linalg.norm(bx, axis=1, keepdims=True), 1e-12)
            cands.append(bx)
    if rec is not None:
        rec = np.asarray(rec, dtype=float)
        if rec.shape == (n, 4) and np.all(np.isfinite(rec)):
            rec = rec / np.maximum(np.linalg.norm(rec, axis=1, keepdims=True), 1e-12)
            cands.append(rec)
    if len(cands) == 0:
        cands.append(rng.randn(n, 4) / np.linalg.norm(rng.randn(n, 4), axis=1, keepdims=True))
    vals = [_max_cos(c) for c in cands]
    best_x = cands[int(np.argmin(vals))].copy()
    best_val = float(min(vals))
    archive = [(best_val, best_x.copy())]

    def archive_add(v, x):
        # distinct check
        for vv, _ in archive:
            if abs(v - vv) < 1e-10:
                # duplicate basin, keep better only
                if v < vv - 1e-12:
                    pass
                else:
                    # if equal, skip
                    if v >= vv:
                        return
        archive.append((v, x.copy()))
        archive.sort(key=lambda t: t[0])
        del archive[8:]

    def pick_base(t, rng):
        if len(archive) == 1:
            return best_x
        # mostly best, sometimes 2nd/3rd
        r = rng.rand()
        if r < 0.65:
            return best_x
        k = rng.randint(1, len(archive))
        return archive[k][1]

    trial = 0
    while time.time() < SEARCH_END:
        if time.time() > DEADLINE - 45:
            break
        trial += 1
        base = pick_base(trial, rng)
        mode = trial % 9
        cur = None
        use_riesz = False
        # build starting point
        if mode == 0:
            sigma = [0.25, 0.12, 0.05][(trial // 9) % 3]
            cur = base + rng.randn(n, 4) * sigma
            cur /= np.linalg.norm(cur, axis=1, keepdims=True)
        elif mode == 1:
            sigma = 0.06 + 0.04 * rng.rand()
            cur = base + rng.randn(n, 4) * sigma
            cur /= np.linalg.norm(cur, axis=1, keepdims=True)
            use_riesz = True
        elif mode == 2:
            u = rng.randn(4); u /= np.linalg.norm(u)
            dots = base @ u
            q = 0.55 + 0.35 * rng.rand()
            thresh = np.quantile(dots, q)
            idx = np.where(dots > thresh)[0]
            if len(idx) < 4:
                idx = np.argsort(dots)[-15:]
            R = _rand_rot(rng, 0.35 + 0.65 * rng.rand())
            cur = base.copy()
            cur[idx] = (R @ cur[idx].T).T
            cur += rng.randn(n, 4) * 0.006
            cur /= np.linalg.norm(cur, axis=1, keepdims=True)
        elif mode == 3:
            u = rng.randn(4); u /= np.linalg.norm(u)
            dots = np.abs(base @ u)
            idx = np.where(dots < 0.30)[0]
            if len(idx) < 6:
                idx = np.argsort(dots)[:25]
            R = _rand_rot(rng, 0.4 + 0.6 * rng.rand())
            cur = base.copy()
            cur[idx] = (R @ cur[idx].T).T
            cur[idx] += rng.randn(len(idx), 4) * 0.04
            cur /= np.linalg.norm(cur, axis=1, keepdims=True)
            use_riesz = (rng.rand() < 0.5)
        elif mode == 4:
            G = base @ base.T
            np.fill_diagonal(G, -2.0)
            m = float(G.max())
            deg = (G > m - 0.003).sum(axis=1)
            hub = int(np.argmax(deg))
            nbrs = np.where(G[hub] > m - 0.010)[0]
            if len(nbrs) > 14:
                nbrs = nbrs[np.argsort(G[hub][nbrs])[-14:]]
            group = np.unique(np.concatenate([[hub], nbrs]))
            cur = base.copy()
            j = int(np.argmax(G[hub]))
            away = cur[hub] - cur[j]
            away -= (away @ cur[hub]) * cur[hub]
            na = np.linalg.norm(away)
            if na > 1e-12:
                cur[hub] += (away / na) * (0.15 + 0.15 * rng.rand())
            cur[group] += rng.randn(len(group), 4) * 0.12
            cur += rng.randn(n, 4) * 0.005
            cur /= np.linalg.norm(cur, axis=1, keepdims=True)
        elif mode == 5:
            G = base @ base.T
            np.fill_diagonal(G, -2.0)
            m = float(G.max())
            deg = (G > m - 0.003).sum(axis=1)
            mrep = 3 + (trial % 6)  # 3..8
            worst = np.argsort(deg)[-mrep:]
            keep = np.ones(n, dtype=bool); keep[worst] = False
            fixed = base[keep].copy()
            cur = base.copy()
            for w in worst:
                h, _ = _find_hole(fixed, rng, n_samples=6000)
                cur[w] = h
                fixed = np.vstack([fixed, h.reshape(1, 4)])
            cur += rng.randn(n, 4) * 0.005
            cur /= np.linalg.norm(cur, axis=1, keepdims=True)
            use_riesz = (rng.rand() < 0.4)
        elif mode == 6:
            if len(archive) >= 2:
                k2 = rng.randint(0, len(archive))
                a2 = archive[k2][1]
                # ensure a != base occasionally
                cur = _crossover(base, a2, rng)
                cur += rng.randn(n, 4) * 0.004
                cur /= np.linalg.norm(cur, axis=1, keepdims=True)
            else:
                cur = base + rng.randn(n, 4) * 0.15
                cur /= np.linalg.norm(cur, axis=1, keepdims=True)
        elif mode == 7:
            cur = rng.randn(n, 4)
            cur /= np.linalg.norm(cur, axis=1, keepdims=True)
            # seed partially with best holes? mix 20 points from best
            if rng.rand() < 0.5:
                idx = rng.choice(n, 30, replace=False)
                cur[idx] = base[idx] + rng.randn(30, 4) * 0.05
                cur /= np.linalg.norm(cur, axis=1, keepdims=True)
            use_riesz = True
        else:
            # pair-break + random cap
            G = base @ base.T
            np.fill_diagonal(G, -2.0)
            ij = int(np.argmax(G.ravel()))
            ii, jj = divmod(ij, n)
            cur = base.copy()
            for kk, sgn in ((ii, 1.0), (jj, -1.0)):
                d = cur[kk] - sgn * 0.08 * cur[jj if kk == ii else ii]
                d /= max(np.linalg.norm(d), 1e-12)
                cur[kk] = d
            u = rng.randn(4); u /= np.linalg.norm(u)
            dots = base @ u
            idx = np.where(dots > np.quantile(dots, 0.85))[0]
            R = _rand_rot(rng, 0.5)
            cur[idx] = (R @ cur[idx].T).T
            cur += rng.randn(n, 4) * 0.006
            cur /= np.linalg.norm(cur, axis=1, keepdims=True)

        # annealed descent
        if use_riesz:
            s0 = 6.0 + 4.0 * rng.rand()
            s1 = 45.0 + 25.0 * rng.rand()
            cur_b, cur_v = _gd_riesz(cur, steps=3200, S0=s0, S1=s1, lr0=0.05 + 0.02 * rng.rand(),
                                     mu=0.86, rng=rng, noise0=0.002, deadline=SEARCH_END)
            # follow with short soft-max to target max directly
            if cur_v < best_val + 0.006 and time.time() < SEARCH_END - 0.5:
                cur_b2, cur_v2 = _gd_soft(cur_b, steps=1800, T0=0.006, T1=0.0005,
                                          lr0=0.03, mu=0.88, rng=rng, noise0=0.0, deadline=SEARCH_END)
                if cur_v2 < cur_v:
                    cur_b, cur_v = cur_b2, cur_v2
        else:
            T0 = 0.010 + 0.008 * rng.rand()
            T1 = 0.0004 + 0.0004 * rng.rand()
            cur_b, cur_v = _gd_soft(cur, steps=3800, T0=T0, T1=T1, lr0=0.038 + 0.010 * rng.rand(),
                                    mu=0.85 + 0.05 * rng.rand(), rng=rng,
                                    noise0=0.0015 + 0.0015 * rng.rand(), deadline=SEARCH_END)

        if cur_v < best_val + 0.004 and time.time() < SEARCH_END - 0.8:
            try:
                lb_x, lb_v = _lbfgs_smooth(cur_b, deadline=min(SEARCH_END, time.time() + 3.5))
                if lb_v < cur_v:
                    cur_b, cur_v = lb_x, lb_v
            except Exception:
                pass
        if cur_v < best_val + 0.0015 and time.time() < SEARCH_END - 0.5:
            try:
                lp_x, lp_v = _lp_polish(cur_b, deadline=min(SEARCH_END, time.time() + 2.0), r0=8e-4)
                if lp_v < cur_v:
                    cur_b, cur_v = lp_x, lp_v
            except Exception:
                pass
        # archive update
        if cur_v < best_val - 1e-12:
            best_val, best_x = cur_v, cur_b.copy()
            archive_add(cur_v, cur_b)
        elif cur_v < best_val + 0.004:
            if len(archive) < 8:
                # avoid exact dup
                dup = False
                for vv, xx in archive:
                    if abs(vv - cur_v) < 5e-9:
                        dup = True
                        break
                if not dup:
                    archive_add(cur_v, cur_b)
            else:
                # replace worst if better than worst and distinct
                worst = archive[-1][0]
                if cur_v < worst - 2e-6:
                    dup = any(abs(vv - cur_v) < 5e-9 for vv, _ in archive)
                    if not dup:
                        archive[-1] = (cur_v, cur_b.copy())
                        archive.sort(key=lambda t: t[0])

    # final polish on best
    try:
        if time.time() < DEADLINE - 3.0:
            bx2, bv2 = _lbfgs_smooth(best_x, deadline=DEADLINE - 12.0,
                                     taus=(0.003, 0.001, 0.0004), iters=(150, 150, 220))
            if bv2 < best_val:
                best_x, best_val = bx2, bv2
    except Exception:
        pass
    try:
        if time.time() < DEADLINE - 2.0:
            px, pv = _lp_polish(best_x, deadline=DEADLINE - 0.5, r0=1e-3)
            if pv < best_val:
                best_x, best_val = px, pv
    except Exception:
        pass
    best_x = best_x / np.maximum(np.linalg.norm(best_x, axis=1, keepdims=True), 1e-12)
    return best_x
