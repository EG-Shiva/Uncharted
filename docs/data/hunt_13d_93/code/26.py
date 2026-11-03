import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _norm(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return X / n

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _p_continuation(X0, ps, steps_list, lrs, shift=0.15, track_every=10, deadline=1e18):
    X = _norm(X0.copy())
    n, d = X.shape
    best_val = _max_cos(X)
    best_X = X.copy()
    for p, steps, lr in zip(ps, steps_list, lrs):
        m = np.zeros_like(X)
        v = np.zeros_like(X)
        t = 0
        b1, b2, eps = 0.9, 0.999, 1e-8
        # effective active monitoring (diagnostic only)
        for s in range(steps):
            if time.time() > deadline:
                break
            G = X @ X.T
            np.fill_diagonal(G, -1.0)
            S = G - shift
            S[S < 0] = 0.0
            # diag already 0
            smax = float(S.max())
            if (not np.isfinite(smax)) or smax <= 1e-300:
                break
            R = S / smax
            mask = R > 0
            Rp1 = np.zeros_like(R)
            with np.errstate(over='ignore', under='ignore', invalid='ignore'):
                np.power(R, p - 1.0, out=Rp1, where=mask)
            Rp = Rp1 * R
            Q = float(Rp.sum() * 0.5)
            if (not np.isfinite(Q)) or Q <= 0:
                break
            scale = Q ** (1.0 / p - 1.0)
            A = Rp1 * scale
            grad = A @ X
            dot = np.einsum('ij,ij->i', grad, X)[:, None]
            gt = grad - dot * X
            t += 1
            m = b1 * m + (1.0 - b1) * gt
            v = b2 * v + (1.0 - b2) * (gt * gt)
            mh = m / (1.0 - b1 ** t)
            vh = v / (1.0 - b2 ** t)
            upd = lr * mh / (np.sqrt(vh) + eps)
            d2 = np.einsum('ij,ij->i', upd, X)[:, None]
            updt = upd - d2 * X
            X = X - updt
            X = _norm(X)
            dm = np.einsum('ij,ij->i', m, X)[:, None]
            m = m - dm * X
            if (s % track_every == 0) or (s == steps - 1):
                vv = _max_cos(X)
                if vv < best_val:
                    best_val = vv
                    best_X = X.copy()
        # rescale implicitly by resetting momentum next stage
        fv = _max_cos(X)
        if fv < best_val:
            best_val = fv
            best_X = X.copy()
        if time.time() > deadline:
            break
    return best_X, best_val, X

def _smooth_adam(X, steps, tau0, tau1, lr0, lr1, deadline=1e18):
    X = _norm(X.copy())
    n, d = X.shape
    m = np.zeros_like(X)
    v = np.zeros_like(X)
    b1, b2, eps = 0.9, 0.999, 1e-8
    t = 0
    ltau = np.log(tau1 / tau0) if tau0 != tau1 and tau0 > 0 and tau1 > 0 else 0.0
    llr = np.log(lr1 / lr0) if lr0 != lr1 and lr0 > 0 and lr1 > 0 else 0.0
    for s in range(steps):
        if (s % 200 == 0) and time.time() > deadline:
            break
        f = s / max(1, steps - 1)
        tau = tau0 * np.exp(ltau * f)
        lr = lr0 * np.exp(llr * f)
        if tau < 1e-12:
            tau = 1e-12
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        mg = float(G.max())
        W = np.exp((G - mg) / tau)
        np.fill_diagonal(W, 0.0)
        grad = W @ X
        dot = np.sum(grad * X, axis=1, keepdims=True)
        gt = grad - dot * X
        t += 1
        m = b1 * m + (1.0 - b1) * gt
        v = b2 * v + (1.0 - b2) * (gt * gt)
        mh = m / (1.0 - b1 ** t)
        vh = v / (1.0 - b2 ** t)
        step = lr * mh / (np.sqrt(vh) + eps)
        # tangent project
        d2 = np.sum(step * X, axis=1, keepdims=True)
        stept = step - d2 * X
        X = X - stept
        X = _norm(X)
    return X

def _hard_topK(X, iters, step0, K, deadline=1e18):
    X = _norm(X.copy())
    best = _max_cos(X)
    bestX = X.copy()
    step = step0
    n, d = X.shape
    iu0, iu1 = np.triu_indices(n, 1)
    K = min(K, len(iu0))
    grad = np.empty_like(X)
    for it in range(iters):
        if (it % 100 == 0) and time.time() > deadline:
            break
        G = X @ X.T
        c = G[iu0, iu1]
        if K < len(c):
            idx = np.argpartition(-c, K)[:K]
        else:
            idx = np.arange(len(c))
        I = iu0[idx]; J = iu1[idx]
        grad.fill(0.0)
        np.add.at(grad, I, X[J])
        np.add.at(grad, J, X[I])
        grad /= float(len(idx))
        dot = np.einsum('ij,ij->i', grad, X)[:, None]
        gt = grad - dot * X
        ng = float(np.linalg.norm(gt))
        if ng == 0 or not np.isfinite(ng):
            step *= 0.5
            if step < 1e-13:
                break
            continue
        Xt = X - step * gt
        Xt = _norm(Xt)
        vv = _max_cos(Xt)
        if vv < best:
            X = Xt
            best = vv
            bestX = X.copy()
            step = min(step * 1.3, 0.02)
        else:
            step *= 0.5
            if step < 1e-13:
                break
    return bestX, best

def _lp_polish(x, seconds, r0=0.0002):
    x = _norm(x.copy())
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
        if cnt > 4000:
            idx = np.argpartition(-c_all, 4000)[:4000]
            mask = np.zeros_like(sel)
            mask[idx] = True
            sel = mask
            cnt = 4000
        if cnt == 0:
            radius *= 0.5
            continue
        I = iu[0][sel]; J = iu[1][sel]
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
        obj = np.zeros(nv); obj[-1] = 1.0
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
        y = _norm(y)
        vv = _max_cos(y)
        if vv < best:
            x, best = y, vv
            radius = min(radius * 1.5, 0.02)
        else:
            radius *= 0.5
        if time.time() - start > seconds - 0.3:
            break
    return x

def solve():
    t0 = time.time()
    BUDGET = 170.0
    HOP_DEADLINE = t0 + 118.0
    n, d = 93, 13
    # load seeds
    best_x = None; best_val = 1e9
    rec = None
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=np.float64)
            if a.shape == (93, 13):
                a = _norm(a)
                va = _max_cos(a)
                if va < best_val:
                    best_val = va
                    best_x = a.copy()
                if fn == "record.npy":
                    rec = a.copy()
        except Exception:
            pass
    rng = np.random.default_rng(0)
    if best_x is None:
        best_x = _norm(rng.standard_normal((n, d)))
        best_val = _max_cos(best_x)
    if rec is None:
        rec = best_x.copy()
    # build starts: best, record, perturbed sigma in {0.15,0.3,0.6}, random
    starts = []
    starts.append(best_x.copy())
    starts.append(rec.copy())
    def pert(base, sig, seed):
        r = np.random.default_rng(seed)
        Y = base + sig * r.standard_normal(size=base.shape)
        return _norm(Y)
    starts.append(pert(best_x, 0.15, 1))
    starts.append(pert(best_x, 0.30, 2))
    starts.append(pert(best_x, 0.60, 3))
    starts.append(_norm(np.random.default_rng(4).standard_normal((n, d))))
    starts.append(pert(rec, 0.30, 5))
    starts.append(pert(best_x, 0.15, 6))
    ps = [8, 16, 32, 64, 128, 256]
    steps_list = [350, 350, 300, 300, 250, 250]
    lrs = [3e-3, 2e-3, 1.2e-3, 8e-4, 5e-4, 3e-4]
    cands = []  # (val, bestX)
    # also keep global best
    gbest_X = best_x.copy(); gbest = best_val
    for si, X0 in enumerate(starts):
        if time.time() > HOP_DEADLINE:
            break
        try:
            bX, bv, fX = _p_continuation(X0, ps, steps_list, lrs, shift=0.15, track_every=10, deadline=HOP_DEADLINE)
        except Exception:
            continue
        # also consider final
        try:
            fv = _max_cos(fX)
            if fv < bv:
                bv = fv; bX = fX.copy()
        except Exception:
            pass
        cands.append((bv, bX.copy()))
        if bv < gbest:
            gbest = bv; gbest_X = bX.copy()
    # promote best 2-3 to hard polish
    cands.sort(key=lambda t: t[0])
    top = cands[:3] if len(cands) >= 3 else cands
    # polish each
    polished = []
    for bv, bX in top:
        if time.time() > t0 + BUDGET - 12:
            break
        try:
            # low-T smooth to settle, then topK
            X1 = _smooth_adam(bX, steps=600, tau0=3e-4, tau1=3e-5, lr0=0.01, lr1=0.002, deadline=t0 + BUDGET - 10)
            X2, v2 = _hard_topK(X1, iters=500, step0=0.0015, K=500, deadline=t0 + BUDGET - 8)
            v1 = _max_cos(X2)
            # keep better of bX and X2
            if v1 < bv:
                bv = v1; bX = X2.copy()
        except Exception:
            pass
        # small LP squeeze if time
        remain = (t0 + BUDGET - 6) - time.time()
        if remain > 1.2:
            try:
                Xlp = _lp_polish(bX, seconds=min(3.0, remain - 0.5), r0=0.00015)
                vlp = _max_cos(Xlp)
                if vlp < bv:
                    bv = vlp; bX = Xlp.copy()
            except Exception:
                pass
        polished.append((bv, bX))
        if bv < gbest:
            gbest = bv; gbest_X = bX.copy()
    # final quick LP on global best if time
    remain = (t0 + BUDGET - 2) - time.time()
    if remain > 1.5:
        try:
            Xf = _lp_polish(gbest_X, seconds=min(4.0, remain - 0.5), r0=8e-05)
            vf = _max_cos(Xf)
            if vf < gbest:
                gbest = vf; gbest_X = Xf.copy()
        except Exception:
            pass
    gbest_X = _norm(gbest_X)
    return gbest_X
