import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())


def adam_quench(x_init, steps, T0, T1, lr, mask=None, seed_step=0):
    x = x_init.copy()
    n, d = x.shape
    m = np.zeros_like(x)
    v = np.zeros_like(x)
    b1, b2, eps = 0.9, 0.999, 1e-8
    lT0 = np.log(T0)
    lT1 = np.log(T1)
    if mask is None:
        idx = None
    else:
        idx = np.where(mask)[0]
        if len(idx) == 0:
            return x
    for s in range(1, steps + 1):
        lt = lT0 + (lT1 - lT0) * (s / steps)
        T = float(np.exp(lt))
        G = x @ x.T
        np.fill_diagonal(G, -10.0)
        Mx = float(G.max())
        W = np.exp((G - Mx) / T)
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
        if idx is None:
            x = x - lr * upd
            nr = np.linalg.norm(x, axis=1, keepdims=True)
            nr[nr == 0] = 1.0
            x /= nr
        else:
            x[idx] = x[idx] - lr * upd[idx]
            nr = np.linalg.norm(x[idx], axis=1, keepdims=True)
            nr[nr == 0] = 1.0
            x[idx] /= nr
    return x


def find_hole(X_rem, rng, n_samp=12000):
    d = X_rem.shape[1]
    S = rng.standard_normal((n_samp, d))
    S /= np.linalg.norm(S, axis=1, keepdims=True)
    dots = S @ X_rem.T
    worst = dots.max(axis=1)
    bi = int(np.argmin(worst))
    return S[bi], float(worst[bi])


def refine_hole(q0, X_rem, iters=120, T=0.004, lr=0.05):
    q = q0.copy()
    q /= np.linalg.norm(q)
    for _ in range(iters):
        dots = X_rem @ q
        mm = float(dots.max())
        w = np.exp((dots - mm) / T)
        s = float(w.sum())
        if s == 0 or not np.isfinite(s):
            break
        g = (w @ X_rem) / s
        # project to tangent at q and descend
        g = g - float(g @ q) * q
        q = q - lr * g
        nq = float(np.linalg.norm(q))
        if nq == 0 or not np.isfinite(nq):
            break
        q /= nq
    return q


def lp_polish(x, seconds, t_end):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = max_cos(x)
    radius = 1.5e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and time.time() < t_end:
        if radius < 1e-14:
            break
        G = x @ x.T
        allc = G[iu]
        mcur = float(allc.max())
        thresh = mcur - max(20 * radius, 1e-9)
        sel = allc > thresh
        # cap pairs for speed
        if int(sel.sum()) > 2500:
            # keep top 2000
            part = np.argpartition(-allc, 2000)[:2000]
            mask = np.zeros_like(sel)
            mask[part] = True
            sel = mask
        I, J = iu[0][sel], iu[1][sel]
        p, nv = len(I), n * d + 1
        if p == 0:
            break
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv)
        obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-allc[sel], A_eq=A_eq, b_eq=np.zeros(n),
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
    # elite pool: best + second
    x_second = x_best.copy()
    second = best + 1e-3

    explore_end = t0 + 142.0
    it = 0
    while time.time() < explore_end and time.time() < t_end - 30.0:
        it += 1
        # choose base: mostly best, sometimes second for diversity
        if it % 7 == 6 and second < best + 0.004:
            base = x_second
        else:
            base = x_best
        G0 = base @ base.T
        np.fill_diagonal(G0, -2.0)
        rowmax = G0.max(axis=1)
        order = np.argsort(-rowmax)
        mcur = float(rowmax.max())
        loads = np.sum(G0 > mcur - 1e-3, axis=1)
        load_order = np.argsort(-loads)
        mode = int(rng.integers(0, 5))
        x_try = base.copy()
        mask = None
        if mode == 0:
            # coordinated hub-cluster shift: open many contacts at once
            hub = int(load_order[int(rng.integers(0, 3))])
            gr = base @ base[hub]
            gr[hub] = -2.0
            k = 6 + int(rng.integers(0, 4))  # 6..9
            nbr = np.argsort(-gr)[:k]
            cl = np.append(nbr, hub)
            common = rng.standard_normal(d)
            common -= float(common @ base[hub]) * base[hub]
            cn = float(np.linalg.norm(common))
            if cn > 1e-9:
                common /= cn
            amp = 0.05 + 0.03 * float(rng.random())
            for ci in cl:
                # project common to tangent at ci
                cc = common - float(common @ base[ci]) * base[ci]
                x_try[ci] = base[ci] + amp * cc + 0.035 * rng.standard_normal(d)
            # tiny global jitter
            x_try += 0.0015 * rng.standard_normal((n, d))
            steps, T0, T1, lr = 650, 0.020, 0.0025, 0.0022
        elif mode == 1:
            # multi-point delete + sequential best-hole re-insert
            nk = 2 + int(rng.integers(0, 2))  # 2..3
            # pick worst + random high-load mix
            worst = []
            worst.append(int(order[0]))
            pool = load_order[:12].tolist()
            rng.shuffle(pool)
            for c in pool:
                if c not in worst and len(worst) < nk:
                    worst.append(int(c))
            keep = np.ones(n, dtype=bool)
            keep[np.array(worst)] = False
            X_rem = base[keep]
            cur_list = [X_rem]
            # sequentially insert
            new_pts = []
            for _ in worst:
                Xc = np.vstack(cur_list + ([np.vstack(new_pts)] if len(new_pts) > 0 else []))
                q, _ = find_hole(Xc, rng, n_samp=12000)
                q = refine_hole(q, Xc, iters=100, T=0.004, lr=0.05)
                new_pts.append(q[None, :])
            for wi, qp in zip(worst, new_pts):
                x_try[wi] = qp[0] + 0.004 * rng.standard_normal(d)
            x_try += 0.002 * rng.standard_normal((n, d))
            steps, T0, T1, lr = 800, 0.022, 0.0022, 0.0020
        elif mode == 2:
            # cavity expansion: find hole, push blockers away
            q, _ = find_hole(base, rng, n_samp=12000)
            q = refine_hole(q, base, iters=80, T=0.005, lr=0.05)
            dots_h = base @ q
            k = 6 + int(rng.integers(0, 3))
            blk = np.argsort(-dots_h)[:k]
            push = 0.25 + 0.2 * float(rng.random())
            for bi in blk:
                h_tan = q - float(q @ base[bi]) * base[bi]
                x_try[bi] = base[bi] - push * h_tan + 0.02 * rng.standard_normal(d)
            x_try += 0.0015 * rng.standard_normal((n, d))
            steps, T0, T1, lr = 700, 0.018, 0.0022, 0.0022
        elif mode == 3:
            # subset-freeze quench: kick random subset, optimize subset first
            k = 8 + int(rng.integers(0, 8))
            idx = rng.choice(n, size=k, replace=False)
            # bias toward high contacts
            if float(rng.random()) < 0.5:
                idx[:4] = load_order[:4]
            x_try[idx] += 0.06 * rng.standard_normal((k, d))
            x_try += 0.002 * rng.standard_normal((n, d))
            x_try /= np.linalg.norm(x_try, axis=1, keepdims=True)
            # stage 1: subset only
            smask = np.zeros(n, dtype=bool)
            smask[idx] = True
            x_try = adam_quench(x_try, 300, 0.015, 0.004, 0.0025, mask=smask)
            mask = None
            steps, T0, T1, lr = 500, 0.012, 0.0020, 0.0020
        else:
            # global small kick exploit
            x_try = base + 0.010 * rng.standard_normal((n, d))
            steps, T0, T1, lr = 550, 0.014, 0.0028, 0.0020
        x_try /= np.linalg.norm(x_try, axis=1, keepdims=True)
        try:
            x_q = adam_quench(x_try, steps, T0, T1, lr, mask=mask)
        except Exception:
            continue
        v = max_cos(x_q)
        if v < best - 1e-12:
            # demote old best
            x_second, second = x_best, best
            best = v
            x_best = x_q
            # opportunistic extra squeeze
            if time.time() < explore_end - 6:
                try:
                    x_e = adam_quench(x_best, 250, 0.007, 0.0015, 0.0015)
                    ve = max_cos(x_e)
                    if ve < best:
                        best, x_best = ve, x_e
                except Exception:
                    pass
        elif v < second and v < best + 0.003:
            second, x_second = v, x_q

    remain = t_end - time.time()
    if remain > 6:
        try:
            x_sq = adam_quench(x_best, 400, 0.009, 0.0016, 0.0015)
            vs = max_cos(x_sq)
            if vs < best:
                best, x_best = vs, x_sq
        except Exception:
            pass
        remain = t_end - time.time()
        lp_time = min(remain - 1.0, 26.0)
        if lp_time > 3:
            try:
                x_p = lp_polish(x_best, lp_time, t_end - 1.0)
                vp = max_cos(x_p)
                if vp < best:
                    x_best, best = x_p, vp
            except Exception:
                pass
    x_best /= np.linalg.norm(x_best, axis=1, keepdims=True)
    return x_best
