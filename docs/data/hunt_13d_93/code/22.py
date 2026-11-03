import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _per_row(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return g.max(axis=1)

def _norm(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return x / n

def _adam(x, steps, tau0, tau1, lr0, lr1, deadline=1e18):
    n, d = x.shape
    m = np.zeros_like(x)
    v = np.zeros_like(x)
    b1, b2, eps = 0.9, 0.999, 1e-8
    ltau = np.log(tau1 / tau0) if tau0 > 0 and tau1 > 0 and tau0 != tau1 else 0.0
    llr = np.log(lr1 / lr0) if lr0 > 0 and lr1 > 0 and lr0 != lr1 else 0.0
    t = 0
    for s in range(steps):
        if (s & 127) == 0 and time.time() > deadline:
            break
        f = s / max(1, steps - 1)
        tau = tau0 * np.exp(ltau * f)
        lr = lr0 * np.exp(llr * f)
        if tau < 1e-12:
            tau = 1e-12
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        mg = float(G.max())
        W = np.exp((G - mg) / tau)
        np.fill_diagonal(W, 0.0)
        grad = W @ x
        dot = np.sum(grad * x, axis=1, keepdims=True)
        gt = grad - dot * x
        t += 1
        m = b1 * m + (1.0 - b1) * gt
        v = b2 * v + (1.0 - b2) * (gt * gt)
        mh = m / (1.0 - b1 ** t)
        vh = v / (1.0 - b2 ** t)
        x = x - lr * mh / (np.sqrt(vh) + eps)
        x = _norm(x)
    return x

def _hard(x, iters, margin=0.003, step0=0.003, deadline=1e18):
    step = step0
    for it in range(iters):
        if (it % 40 == 0) and time.time() > deadline:
            break
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        cm = float(G.max())
        mask = G > (cm - margin)
        grad = mask.astype(np.float64) @ x
        dot = np.sum(grad * x, axis=1, keepdims=True)
        gt = grad - dot * x
        rn = np.linalg.norm(gt, axis=1, keepdims=True)
        nz = rn[:, 0] > 1e-12
        if not np.any(nz):
            break
        dirc = np.zeros_like(gt)
        dirc[nz] = gt[nz] / rn[nz]
        xt = _norm(x - step * dirc)
        vt = _max_cos(xt)
        if vt < cm:
            x = xt
            step = min(step * 1.2, 0.005)
        else:
            step *= 0.5
            if step < 1e-7:
                break
    return x

def _find_void(others, rng, M=4096):
    d = others.shape[1]
    cand = rng.standard_normal((M, d))
    cand = _norm(cand)
    dots = cand @ others.T
    md = dots.max(axis=1)
    v = cand[int(np.argmin(md))].copy()
    lr = 0.05
    tau = 0.008
    for _ in range(120):
        dv = others @ v
        m0 = float(dv.max())
        w = np.exp((dv - m0) / tau)
        s = float(w.sum())
        if s <= 0 or not np.isfinite(s):
            break
        w /= s
        g = w @ others
        gt = g - float(g @ v) * v
        nr = float(np.linalg.norm(gt))
        if nr < 1e-12 or not np.isfinite(nr):
            break
        gt /= nr
        v2 = v - lr * gt
        n2 = float(np.linalg.norm(v2))
        if n2 < 1e-12:
            lr *= 0.5
            continue
        v2 /= n2
        m2 = float((others @ v2).max())
        if m2 < m0:
            v = v2
            lr = min(lr * 1.05, 0.08)
        else:
            lr *= 0.5
            if lr < 1e-6:
                break
    return v / np.linalg.norm(v)

def _lp_polish(x, seconds, r0=0.002):
    x = _norm(x.copy())
    n, d = x.shape
    best = _max_cos(x)
    radius = float(r0)
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        Gf = x @ x.T
        call = Gf[iu]
        m = float(call.max())
        thresh = m - max(20.0 * radius, 1e-7)
        sel = call > thresh
        cnt = int(sel.sum())
        if cnt > 4000:
            idx = np.argpartition(-call, 4000)[:4000]
            mask = np.zeros_like(sel)
            mask[idx] = True
            sel = mask
            cnt = 4000
        if cnt == 0:
            radius *= 0.5
            continue
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
            res = linprog(obj, A_ub=A_ub, b_ub=-call[sel], A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y = _norm(y)
        vv = _max_cos(y)
        if vv < best:
            x, best = y, vv
            radius = min(radius * 1.5, 0.02)
        else:
            radius *= 0.5
        if time.time() - start > seconds - 0.4:
            break
    return x

def solve():
    t0 = time.time()
    BUDGET = 172.0
    HOP_END = t0 + 138.0
    rng = np.random.default_rng(7)
    np.random.seed(7)
    best_x = None
    best_val = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn).astype(np.float64)
            if a.shape == (93, 13):
                a = _norm(a)
                va = _max_cos(a)
                if va < best_val:
                    best_val = va
                    best_x = a.copy()
        except Exception:
            pass
    if best_x is None:
        best_x = _norm(rng.standard_normal((93, 13)))
        best_val = _max_cos(best_x)
    n, d = 93, 13
    try:
        tmp = _adam(best_x.copy(), steps=500, tau0=0.008, tau1=0.001, lr0=0.004, lr1=0.001, deadline=t0 + 10.0)
        tmp = _hard(tmp, iters=120, deadline=t0 + 12.0)
        vv = _max_cos(tmp)
        if vv < best_val and np.all(np.isfinite(tmp)):
            best_x, best_val = tmp, vv
    except Exception:
        pass
    sigmas = [0.005, 0.010, 0.020, 0.040, 0.080, 0.150]
    trial = 0
    while time.time() < HOP_END and (time.time() - t0) < BUDGET - 32.0:
        sigma = sigmas[trial % len(sigmas)]
        mode = trial % 5
        x_try = best_x + sigma * rng.standard_normal((n, d))
        x_try = _norm(x_try)
        if sigma >= 0.019:
            krep = 1 if sigma < 0.07 else 2
            for _ in range(krep):
                try:
                    per = _per_row(x_try)
                    iw = int(np.argmax(per))
                    M = 4096 if sigma < 0.07 else 8192
                    others = np.delete(x_try, iw, axis=0)
                    x_try[iw] = _find_void(others, rng, M=M)
                except Exception:
                    pass
            x_try = _norm(x_try)
        # targeted pair split for small sigma: push current max pair apart a bit to leave basin
        if sigma < 0.03:
            try:
                G = x_try @ x_try.T
                np.fill_diagonal(G, -2.0)
                k = int(np.argmax(G))
                ii = k // n
                jj = k % n
                if ii != jj:
                    eps = sigma * 0.5
                    xi = x_try[ii].copy(); xj = x_try[jj].copy()
                    di = xj - float(xj @ xi) * xi
                    dj = xi - float(xi @ xj) * xj
                    ni = float(np.linalg.norm(di)); nj = float(np.linalg.norm(dj))
                    if ni > 1e-12 and nj > 1e-12:
                        di /= ni; dj /= nj
                        x_try[ii] = xi - eps * di
                        x_try[jj] = xj - eps * dj
                        x_try = _norm(x_try)
            except Exception:
                pass
        dl = HOP_END + 5.0
        try:
            if mode <= 1:
                x_opt = _adam(x_try, steps=900, tau0=0.010, tau1=0.0010, lr0=0.004, lr1=0.0008, deadline=dl)
                x_opt = _hard(x_opt, iters=150, deadline=dl)
            elif mode <= 3:
                x_opt = _adam(x_try, steps=1200, tau0=0.013, tau1=0.0007, lr0=0.0045, lr1=0.0006, deadline=dl)
                x_opt = _hard(x_opt, iters=200, deadline=dl)
            else:
                x_opt = _adam(x_try, steps=1500, tau0=0.016, tau1=0.0006, lr0=0.005, lr1=0.0005, deadline=dl)
                x_opt = _hard(x_opt, iters=220, deadline=dl)
        except Exception:
            trial += 1
            continue
        try:
            v = _max_cos(x_opt)
        except Exception:
            trial += 1
            continue
        if v < best_val - 1e-12 and np.all(np.isfinite(x_opt)):
            best_x, best_val = x_opt, v
            try:
                tmp2 = _adam(best_x.copy(), steps=350, tau0=0.001, tau1=0.0003, lr0=0.001, lr1=0.0003, deadline=dl)
                v2 = _max_cos(tmp2)
                if v2 < best_val:
                    best_x, best_val = tmp2, v2
            except Exception:
                pass
        trial += 1
        if trial > 1000:
            break
    remain = BUDGET - (time.time() - t0) - 1.0
    if remain > 5:
        ft = min(remain * 0.4, 18.0)
        dl = time.time() + ft
        try:
            best_x = _adam(best_x, steps=2000, tau0=0.003, tau1=0.0003, lr0=0.002, lr1=0.0003, deadline=dl)
            best_x = _hard(best_x, iters=300, deadline=time.time() + 4.0)
            bv = _max_cos(best_x)
            if bv < best_val:
                best_val = bv
        except Exception:
            pass
        remain2 = BUDGET - (time.time() - t0) - 0.5
        if remain2 > 3:
            try:
                xp = _lp_polish(best_x, seconds=min(remain2, 26.0), r0=0.002)
                vp = _max_cos(xp)
                if vp < best_val and np.all(np.isfinite(xp)):
                    best_x, best_val = xp, vp
            except Exception:
                pass
    best_x = _norm(best_x)
    best_x = np.nan_to_num(best_x)
    best_x = _norm(best_x)
    return best_x
