import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _normalize(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return x / n

def _adam_smooth(x, steps, tau0, tau1, lr0, lr1):
    n, d = x.shape
    m = np.zeros_like(x)
    v = np.zeros_like(x)
    b1, b2, eps = 0.9, 0.999, 1e-8
    ltau = np.log(tau1 / tau0) if tau0 > 0 and tau1 > 0 and tau0 != tau1 else 0.0
    llr = np.log(lr1 / lr0) if lr0 > 0 and lr1 > 0 and lr0 != lr1 else 0.0
    tt = 0
    for s in range(steps):
        frac = s / max(1, steps - 1)
        tau = tau0 * np.exp(ltau * frac)
        lr = lr0 * np.exp(llr * frac)
        if tau < 5e-05:
            tau = 5e-05
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        mg = float(G.max())
        W = np.exp((G - mg) / tau)
        np.fill_diagonal(W, 0.0)
        # row balance: divide by row sum to avoid overloaded points dominating
        rs = W.sum(axis=1, keepdims=True) + 1e-12
        # mix global and balanced weighting
        grad = (W @ x) / np.sqrt(rs)
        dot = np.sum(grad * x, axis=1, keepdims=True)
        gt = grad - dot * x
        tt += 1
        m = b1 * m + (1.0 - b1) * gt
        v = b2 * v + (1.0 - b2) * (gt * gt)
        mh = m / (1.0 - b1 ** tt)
        vh = v / (1.0 - b2 ** tt)
        step = lr * mh / (np.sqrt(vh) + eps)
        x = x - step
        x = _normalize(x)
    return x

def _hard_push(x, iters, margin=0.0012, step0=0.003):
    cur = _max_cos(x)
    step = step0
    for _ in range(iters):
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        cmax = float(G.max())
        mask = G > (cmax - margin)
        grad = mask.astype(np.float64) @ x
        dot = np.sum(grad * x, axis=1, keepdims=True)
        gt = grad - dot * x
        rn = np.linalg.norm(gt, axis=1, keepdims=True)
        nz = (rn[:, 0] > 1e-12)
        if not np.any(nz):
            break
        direc = np.zeros_like(gt)
        direc[nz] = gt[nz] / rn[nz]
        xtry = _normalize(x - step * direc)
        vtry = _max_cos(xtry)
        if vtry < cmax:
            x = xtry
            cur = vtry
            step = min(step * 1.25, 0.006)
        else:
            step *= 0.5
            if step < 1e-7:
                break
    return x

def _refine_void(others, v, iters=200):
    lr = 0.05
    tau = 0.006
    for _ in range(iters):
        dots = others @ v
        m0 = float(dots.max())
        w = np.exp((dots - m0) / tau)
        s = w.sum()
        if s <= 0 or not np.isfinite(s):
            break
        w /= s
        g = w @ others
        gt = g - float(g @ v) * v
        nrm = float(np.linalg.norm(gt))
        if nrm < 1e-12 or not np.isfinite(nrm):
            break
        gt /= nrm
        v2 = v - lr * gt
        n2 = float(np.linalg.norm(v2))
        if n2 < 1e-12:
            lr *= 0.5
            continue
        v2 /= n2
        m2 = float((others @ v2).max())
        if m2 < m0:
            v = v2
            lr = min(lr * 1.06, 0.08)
        else:
            lr *= 0.5
            if lr < 1e-6:
                break
    return v / np.linalg.norm(v)

def _void_insert(x_try, rng, krep, n_cand=2048):
    n, d = x_try.shape
    for _ in range(krep):
        G = x_try @ x_try.T
        np.fill_diagonal(G, -2.0)
        per = G.max(axis=1)
        iw = int(np.argmax(per))
        cand = rng.standard_normal((n_cand, d))
        cand /= np.linalg.norm(cand, axis=1, keepdims=True)
        others = np.delete(x_try, iw, axis=0)
        dots = cand @ others.T
        mdots = dots.max(axis=1)
        ib = int(np.argmin(mdots))
        v = cand[ib]
        v = _refine_void(others, v, iters=160)
        x_try[iw] = v
    return x_try

def _lp_polish(x, seconds, r0=0.002):
    x = _normalize(x.copy())
    n, d = x.shape
    best = _max_cos(x)
    radius = float(r0)
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        Gfull = x @ x.T
        c_all = Gfull[iu]
        m = float(c_all.max())
        thresh = m - max(20.0 * radius, 1e-7)
        sel = c_all > thresh
        cnt = int(sel.sum())
        if cnt == 0:
            radius *= 0.5
            continue
        if cnt > 4000:
            idx = np.argpartition(-c_all, 4000)[:4000]
            mask = np.zeros_like(sel)
            mask[idx] = True
            sel = mask
            cnt = 4000
        I = iu[0][sel]
        J = iu[1][sel]
        p = len(I)
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        except Exception:
            radius *= 0.5
            continue
        obj = np.zeros(nv)
        obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c_all[sel], A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y = _normalize(y)
        vv = _max_cos(y)
        if vv < best:
            x, best = y, vv
            radius = min(radius * 1.5, 0.02)
        else:
            radius *= 0.5
        if time.time() - start > seconds - 0.4:
            break
    return x

def _micro_greedy(x, seconds, rng):
    # tiny tangent random moves, accept if max improves; squeezes last 1e-7
    start = time.time()
    n, d = x.shape
    best = _max_cos(x)
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    # focus on points involved in top pairs
    step = 2e-4
    it = 0
    while time.time() - start < seconds:
        it += 1
        if it % 50 == 1:
            G = x @ x.T
            np.fill_diagonal(G, -2.0)
            per = G.max(axis=1)
            order = np.argsort(-per)
            # pick among top 20 worst points
            k = int(rng.integers(0, min(20, n)))
            i = int(order[k])
        else:
            i = int(rng.integers(0, n))
        # random tangent direction
        r = rng.standard_normal(d)
        r -= float(r @ x[i]) * x[i]
        rn = float(np.linalg.norm(r))
        if rn < 1e-12:
            continue
        r /= rn
        # try +/- with current step
        for sgn in (1.0, -1.0):
            y = x.copy()
            v = x[i] + sgn * step * r
            v /= np.linalg.norm(v)
            y[i] = v
            # quick bound: only need max involving i and current best
            dots = y @ y.T
            np.fill_diagonal(dots, -2.0)
            vv = float(dots.max())
            if vv < best:
                x = y
                best = vv
                step = min(step * 1.15, 5e-4)
                break
        else:
            # no improvement, occasionally shrink
            if it % 30 == 0:
                step *= 0.9
                if step < 2e-6:
                    step = 2e-6
        if it > 20000:
            break
    return x

def solve():
    t0 = time.time()
    BUDGET = 172.0
    rng = np.random.default_rng(12345)
    np.random.seed(1)
    best_x = None
    best_val = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn).astype(np.float64)
            if a.shape == (93, 13):
                a = _normalize(a)
                va = _max_cos(a)
                if va < best_val:
                    best_val = va
                    best_x = a.copy()
        except Exception:
            pass
    if best_x is None:
        best_x = _normalize(rng.standard_normal((93, 13)))
        best_val = _max_cos(best_x)
    n, d = 93, 13
    HOP_END = t0 + 138.0
    trial = 0
    # short initial squeeze of loaded best to establish baseline
    try:
        x0 = _lp_polish(best_x, 2.0, r0=0.001)
        v0 = _max_cos(x0)
        if v0 < best_val and np.all(np.isfinite(x0)):
            best_x, best_val = x0, v0
    except Exception:
        pass
    while time.time() < HOP_END and (time.time() - t0) < BUDGET - 32:
        mode = trial % 6
        if mode == 0:
            sigma = 0.008
            krep = 0
            steps, tau0, tau1, lr0, lr1 = 800, 0.012, 0.0006, 0.004, 0.0008
            hard_iters, margin = 200, 0.0015
        elif mode == 1:
            sigma = 0.022
            krep = 1
            steps, tau0, tau1, lr0, lr1 = 1000, 0.010, 0.0004, 0.005, 0.0006
            hard_iters, margin = 250, 0.0012
        elif mode == 2:
            sigma = 0.045
            krep = 1
            steps, tau0, tau1, lr0, lr1 = 1200, 0.012, 0.0003, 0.005, 0.0005
            hard_iters, margin = 280, 0.0010
        elif mode == 3:
            sigma = 0.080
            krep = 2
            steps, tau0, tau1, lr0, lr1 = 1200, 0.015, 0.0004, 0.006, 0.0006
            hard_iters, margin = 280, 0.0012
        elif mode == 4:
            sigma = 0.130
            krep = 2
            steps, tau0, tau1, lr0, lr1 = 1400, 0.018, 0.0005, 0.006, 0.0007
            hard_iters, margin = 300, 0.0015
        else:
            sigma = 0.030
            krep = 3
            steps, tau0, tau1, lr0, lr1 = 1100, 0.008, 0.00025, 0.004, 0.0004
            hard_iters, margin = 250, 0.0008
        x_try = best_x + sigma * rng.standard_normal((n, d))
        x_try = _normalize(x_try)
        # cluster shake for larger modes: pick a random center and shake its neighbours more
        if mode in (2, 3, 4):
            c = int(rng.integers(0, n))
            Gc = x_try @ x_try[c]
            nn = np.argsort(-Gc)[:8]
            x_try[nn] += 0.5 * sigma * rng.standard_normal((len(nn), d))
            x_try = _normalize(x_try)
        if krep > 0:
            x_try = _void_insert(x_try, rng, krep)
        x_opt = _adam_smooth(x_try, steps, tau0, tau1, lr0, lr1)
        x_opt = _hard_push(x_opt, hard_iters, margin=margin, step0=0.003)
        v = _max_cos(x_opt)
        if v < best_val - 1e-12 and np.all(np.isfinite(x_opt)):
            # squeeze promising with quick LP
            remain = BUDGET - (time.time() - t0)
            if v < best_val + 0.002 and remain > 40:
                try:
                    x_opt2 = _lp_polish(x_opt, 1.5, r0=0.0015)
                    v2 = _max_cos(x_opt2)
                    if v2 < v and np.all(np.isfinite(x_opt2)):
                        x_opt, v = x_opt2, v2
                except Exception:
                    pass
            if v < best_val:
                best_x, best_val = x_opt, v
        trial += 1
        if trial > 3000:
            break
    remain = BUDGET - (time.time() - t0) - 1.0
    if remain > 4:
        lp_time = min(remain * 0.75, 26.0)
        try:
            xp = _lp_polish(best_x, lp_time, r0=0.0015)
            vp = _max_cos(xp)
            if vp < best_val and np.all(np.isfinite(xp)):
                best_x, best_val = xp, vp
        except Exception:
            pass
        remain2 = BUDGET - (time.time() - t0) - 0.8
        if remain2 > 2:
            # ultra-fine radii to equalize degenerate contacts
            try:
                xp2 = _lp_polish(best_x, min(remain2 * 0.5, 8.0), r0=2e-4)
                vp2 = _max_cos(xp2)
                if vp2 < best_val and np.all(np.isfinite(xp2)):
                    best_x, best_val = xp2, vp2
            except Exception:
                pass
            remain3 = BUDGET - (time.time() - t0) - 0.5
            if remain3 > 1.5:
                try:
                    best_x = _micro_greedy(best_x, min(remain3, 5.0), rng)
                    best_val = _max_cos(best_x)
                except Exception:
                    pass
    best_x = _normalize(best_x)
    best_x = np.nan_to_num(best_x)
    best_x = _normalize(best_x)
    return best_x
