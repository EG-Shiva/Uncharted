import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _norm(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n = np.maximum(n, 1e-12)
    return x / n

def _max_cos(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _per_max(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return G.max(axis=1)

def _smooth(x, steps, T0, T1, lr, iu0, iu1):
    n, d = x.shape
    m1 = np.zeros_like(x)
    v1 = np.zeros_like(x)
    b1 = 0.9; b2 = 0.999; eps = 1e-8
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    best_val = float(G.max())
    best_x = x.copy()
    lT0 = np.log(T0); lT1 = np.log(T1)
    for t in range(1, steps+1):
        frac = t / steps
        T = float(np.exp(lT0 + (lT1-lT0)*frac))
        G = x @ x.T
        g = G[iu0, iu1]
        mg = float(g.max())
        if mg < best_val:
            best_val = mg
            best_x = x.copy()
        e = np.exp((g-mg)/T)
        s = float(e.sum())
        if not np.isfinite(s) or s <= 0:
            continue
        W = np.zeros((n, n), dtype=x.dtype)
        W[iu0, iu1] = e/s
        W = W + W.T
        grad = W @ x
        dot = np.sum(grad*x, axis=1, keepdims=True)
        gt = grad - dot*x
        m1 = b1*m1 + (1.0-b1)*gt
        v1 = b2*v1 + (1.0-b2)*(gt*gt)
        bc1 = 1.0 - b1**t
        bc2 = 1.0 - b2**t
        delta = lr*(m1/bc1)/(np.sqrt(v1/bc2)+eps)
        x = x - delta
        x = _norm(x)
    return best_x, best_val

def _find_hole(Xk, rng, n_samples=12000, refine_steps=80):
    d = Xk.shape[1]
    Z = rng.standard_normal(size=(n_samples, d))
    Z = _norm(Z)
    D = Z @ Xk.T
    worst = D.max(axis=1)
    idx = int(np.argmin(worst))
    z = Z[idx].copy()
    # refine hole: minimize softmax max dot
    T = 0.008; lr = 0.03
    for _ in range(refine_steps):
        dots = Xk @ z
        m = float(dots.max())
        w = np.exp((dots-m)/T)
        w = w/(w.sum()+1e-12)
        g = w @ Xk
        g = g - float(np.dot(g, z))*z
        z = z - lr*g
        nz = float(np.linalg.norm(z))
        if nz < 1e-12:
            break
        z = z/nz
    return z

def _lp_polish(x, deadline):
    x = _norm(x)
    n, d = x.shape
    best = _max_cos(x)
    radius = 2e-3
    iu0, iu1 = np.triu_indices(n, 1)
    ar = np.arange(d)
    while True:
        rem = deadline - time.time()
        if rem < 0.35 or radius < 1e-14:
            break
        G = x @ x.T
        c_all = G[iu0, iu1]
        m = float(c_all.max())
        thr = m - max(20.0*radius, 1e-9)
        sel = c_all > thr
        cnt = int(sel.sum())
        if cnt == 0:
            break
        if cnt > 4000:
            idx = np.where(sel)[0]
            order = np.argsort(-c_all[idx])[:4000]
            mask = np.zeros_like(sel)
            mask[idx[order]] = True
            sel = mask
        I = iu0[sel]; J = iu1[sel]
        csel = c_all[sel]
        p = I.shape[0]
        rows = np.repeat(np.arange(p), 2*d+1)
        cols = np.concatenate([I[:,None]*d+ar, J[:,None]*d+ar, np.full((p,1), n*d)], axis=1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p,1))], axis=1).ravel()
        try:
            A_ub = coo_matrix((vals,(rows,cols)), shape=(p, n*d+1)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n),d), np.arange(n*d))), shape=(n, n*d+1)).tocsr()
            obj = np.zeros(n*d+1); obj[-1]=1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius,radius)]*(n*d)+[(None,None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n,d)
        y = _norm(y)
        v = _max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius*1.5, 0.03)
        else:
            radius *= 0.5
    return x, best

def solve():
    t00 = time.time()
    BUDGET = 170.0
    rng = np.random.default_rng(1)
    n, d = 131, 9
    iu0, iu1 = np.triu_indices(n, 1)
    best_x = None; best_s = 1e9
    for fn in ("best.npy","record.npy"):
        try:
            a = np.load(fn).astype(float)
            if a.shape != (n,d):
                a = np.reshape(a,(n,d))
            a = _norm(a)
            s = _max_cos(a)
            if s < best_s:
                best_s = s; best_x = a.copy()
        except Exception:
            pass
    if best_x is None:
        best_x = _norm(rng.standard_normal(size=(n,d)))
        best_s = _max_cos(best_x)
    # short initial polish
    try:
        best_x, best_s2 = _lp_polish(best_x.copy(), time.time()+3.0)
        if best_s2 < best_s:
            best_s = best_s2
        else:
            pass
    except Exception:
        pass
    x_best = best_x.copy(); score_best = float(best_s)
    trial = 0
    HOP_END = t00 + BUDGET - 13.0
    while time.time() < HOP_END:
        trial += 1
        mode = trial % 10
        Xt = x_best.copy()
        if mode == 0:
            sigma = 0.008
            Xt = _norm(Xt + sigma*rng.standard_normal(size=Xt.shape))
            Xt,_ = _smooth(Xt, 500, 0.02, 0.003, 0.010, iu0, iu1)
        elif mode == 1:
            sigma = 0.020
            Xt = _norm(Xt + sigma*rng.standard_normal(size=Xt.shape))
            Xt,_ = _smooth(Xt, 600, 0.025, 0.004, 0.008, iu0, iu1)
        elif mode == 2:
            # worst -> hole
            pm = _per_max(Xt)
            wi = int(np.argmax(pm))
            keep = np.delete(np.arange(n), wi)
            z = _find_hole(Xt[keep], rng)
            Xt[wi] = z + 0.002*rng.standard_normal(size=d)
            Xt = _norm(Xt)
            Xt,_ = _smooth(Xt, 600, 0.015, 0.0025, 0.008, iu0, iu1)
        elif mode == 3:
            sigma = 0.012
            Xt = _norm(Xt + sigma*rng.standard_normal(size=Xt.shape))
            Xt,_ = _smooth(Xt, 500, 0.012, 0.002, 0.012, iu0, iu1)
        elif mode == 4:
            pm = _per_max(Xt)
            ws = np.argsort(-pm)[:2]
            keep = np.delete(np.arange(n), ws)
            Xk = Xt[keep]
            for k, wi in enumerate(ws):
                z = _find_hole(Xk, rng, n_samples=8000)
                Xt[wi] = z
                Xk = np.vstack([Xk, z[None,:]])
            Xt = _norm(Xt + 0.002*rng.standard_normal(size=Xt.shape))
            Xt,_ = _smooth(Xt, 700, 0.02, 0.003, 0.008, iu0, iu1)
        elif mode == 5:
            # hub kick: high contact count
            G = Xt @ Xt.T
            np.fill_diagonal(G, -2.0)
            m = float(G.max())
            cnt = np.sum(G > m-5e-7, axis=1)
            order = np.argsort(-cnt)
            pool = order[:10]
            ki = rng.choice(pool, size=2, replace=False)
            Xt = Xt + 0.010*rng.standard_normal(size=Xt.shape)
            for wi in ki:
                # randomize one, hole for other
                pass
            # replace first with random, second with hole
            r = rng.standard_normal(size=d); r/=max(np.linalg.norm(r),1e-12)
            Xt[ki[0]] = r
            keep = np.delete(np.arange(n), ki[1])
            Xt[ki[1]] = _find_hole(Xt[keep], rng, n_samples=8000)
            Xt = _norm(Xt)
            Xt,_ = _smooth(Xt, 600, 0.02, 0.003, 0.009, iu0, iu1)
        elif mode == 6:
            sigma = 0.035
            Xt = _norm(Xt + sigma*rng.standard_normal(size=Xt.shape))
            Xt,_ = _smooth(Xt, 700, 0.03, 0.005, 0.007, iu0, iu1)
        elif mode == 7:
            wi = int(rng.integers(0, n))
            keep = np.delete(np.arange(n), wi)
            Xt[wi] = _find_hole(Xt[keep], rng, n_samples=8000)
            Xt = _norm(Xt)
            Xt,_ = _smooth(Xt, 500, 0.015, 0.0025, 0.009, iu0, iu1)
        elif mode == 8:
            # rattler shuffle: move low-contact points
            G = Xt @ Xt.T
            np.fill_diagonal(G, -2.0)
            m = float(G.max())
            cnt = np.sum(G > m-1e-6, axis=1)
            ratt = np.where(cnt==0)[0]
            if len(ratt)==0:
                ratt = np.argsort(cnt)[:3]
            else:
                # also add worst
                pass
            for wi in ratt[:3]:
                keep = np.delete(np.arange(n), wi)
                Xt[wi] = _find_hole(Xt[keep], rng, n_samples=6000)
            Xt = _norm(Xt + 0.003*rng.standard_normal(size=Xt.shape))
            Xt,_ = _smooth(Xt, 600, 0.018, 0.003, 0.008, iu0, iu1)
        else:
            sigma = 0.12 if (trial % 30 == 9) else 0.05
            Xt = _norm(Xt + sigma*rng.standard_normal(size=Xt.shape))
            Xt,_ = _smooth(Xt, 800, 0.03, 0.004, 0.007, iu0, iu1)
        v_quick = _max_cos(Xt)
        # filter: only LP if promising
        if v_quick > score_best + 0.004:
            # accept rarely if far? skip LP
            if v_quick < score_best:
                x_best = Xt.copy(); score_best = v_quick
            continue
        if time.time() >= HOP_END:
            if v_quick < score_best:
                x_best = Xt.copy(); score_best = v_quick
            break
        # short LP
        try:
            Xp, vp = _lp_polish(Xt, time.time()+1.1)
        except Exception:
            continue
        if vp < score_best - 1e-12:
            x_best = Xp.copy(); score_best = vp
    # final long polish
    rem = BUDGET - (time.time()-t00) - 0.8
    if rem > 1.0:
        rem = min(rem, 11.0)
        try:
            Xp, vp = _lp_polish(x_best.copy(), time.time()+rem)
            if vp < score_best:
                x_best = Xp.copy(); score_best = vp
        except Exception:
            pass
    x_best = _norm(x_best)
    return x_best
