import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _norm(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return X / n

def _maxcos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def _rowmax(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return G.max(axis=1)

def _adam_smooth(x, steps, tau0, tau1, lr0, lr1):
    n, d = x.shape
    m = np.zeros_like(x)
    v = np.zeros_like(x)
    b1, b2, eps = 0.9, 0.999, 1e-8
    tt = 0
    ltau = np.log(tau1 / tau0) if tau0 != tau1 else 0.0
    llr = np.log(lr1 / lr0) if lr0 != lr1 else 0.0
    for s in range(steps):
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
        rs = W.sum(axis=1, keepdims=True) + 1e-12
        grad = (W @ x) / rs
        dot = (grad * x).sum(axis=1, keepdims=True)
        gt = grad - dot * x
        tt += 1
        m = b1 * m + (1.0 - b1) * gt
        vv = b2 * v + (1.0 - b2) * (gt * gt)
        v = vv
        mh = m / (1.0 - b1 ** tt)
        vh = vv / (1.0 - b2 ** tt)
        step = lr * mh / (np.sqrt(vh) + eps)
        # project step to tangent
        dot2 = (step * x).sum(axis=1, keepdims=True)
        step_t = step - dot2 * x
        x = x - step_t
        x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x

def _hardK(X, iters, step0, K, iu0, iu1):
    best = _maxcos(X)
    step = step0
    mtot = len(iu0)
    K = int(min(K, mtot))
    grad = np.empty_like(X)
    for _ in range(iters):
        G = X @ X.T
        c = G[iu0, iu1]
        if K < mtot:
            idx = np.argpartition(-c, K)[:K]
        else:
            idx = np.arange(mtot)
        I = iu0[idx]
        J = iu1[idx]
        grad.fill(0.0)
        np.add.at(grad, I, X[J])
        np.add.at(grad, J, X[I])
        grad /= float(len(idx))
        dot = (grad * X).sum(axis=1, keepdims=True)
        gt = grad - dot * X
        ng = float(np.linalg.norm(gt))
        if ng == 0 or not np.isfinite(ng):
            step *= 0.5
            if step < 1e-13:
                break
            continue
        Xt = X - step * gt
        Xt /= np.linalg.norm(Xt, axis=1, keepdims=True)
        G2 = Xt @ Xt.T
        np.fill_diagonal(G2, -2.0)
        vv = float(G2.max())
        if vv < best:
            X = Xt
            best = vv
            step = min(step * 1.25, 0.02)
        else:
            step *= 0.5
            if step < 1e-13:
                break
    return X, best

def _find_void(others, rng, n_cand=5000):
    d = others.shape[1]
    C = rng.standard_normal((n_cand, d))
    C /= np.linalg.norm(C, axis=1, keepdims=True)
    D = C @ others.T
    worst = D.max(axis=1)
    vv = C[int(np.argmin(worst))].copy()
    lr = 0.05
    tau = 0.008
    for _ in range(150):
        dots = others @ vv
        m0 = float(dots.max())
        w = np.exp((dots - m0) / tau)
        s = float(w.sum())
        if s <= 0 or not np.isfinite(s):
            break
        w /= s
        g = w @ others
        gt = g - float(g @ vv) * vv
        nrm = float(np.linalg.norm(gt))
        if nrm < 1e-12 or not np.isfinite(nrm):
            break
        gt /= nrm
        v2 = vv - lr * gt
        n2 = float(np.linalg.norm(v2))
        if n2 < 1e-12:
            lr *= 0.5
            continue
        v2 /= n2
        m2 = float((others @ v2).max())
        if m2 < m0:
            vv = v2
            lr = min(lr * 1.05, 0.08)
        else:
            lr *= 0.5
            if lr < 1e-6:
                break
    vv /= np.linalg.norm(vv)
    return vv

def _lp_polish(x, seconds, r_init=2e-4):
    x = _norm(x.copy())
    n, d = x.shape
    best = _maxcos(x)
    radius = float(r_init)
    start = time.time()
    iu = np.triu_indices(n, 1)
    ar = np.arange(d)
    while time.time() - start < seconds and radius > 1e-13:
        Gfull = x @ x.T
        c_all = Gfull[iu]
        mm = float(c_all.max())
        thresh = max(25.0 * radius, 5e-8)
        sel = c_all > mm - thresh
        cnt = int(sel.sum())
        if cnt == 0:
            radius *= 0.5
            continue
        if cnt > 2500:
            idx = np.argpartition(-c_all, 2500)[:2500]
            mask = np.zeros_like(sel)
            mask[idx] = True
            sel = mask
        I = iu[0][sel]
        J = iu[1][sel]
        p = len(I)
        csub = c_all[sel]
        nv = n * d + 1
        try:
            rows = np.repeat(np.arange(p), 2 * d + 1)
            cols = np.concatenate([I[:, None] * d + ar, J[:, None] * d + ar, np.full((p, 1), n * d)], axis=1).ravel()
            vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], axis=1).ravel()
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            r_eq = np.repeat(np.arange(n), d)
            c_eq = np.arange(n * d)
            A_eq = coo_matrix((x.ravel(), (r_eq, c_eq)), shape=(n, nv)).tocsr()
            obj = np.zeros(nv)
            obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-csub, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y = _norm(y)
        vv = _maxcos(y)
        if vv < best:
            x, best = y, vv
            radius = min(radius * 1.5, 0.02)
        else:
            radius *= 0.5
    return x, best

def solve():
    t0 = time.time()
    DEADLINE = 172.0
    HOP_END = t0 + 132.0
    rng = np.random.default_rng(1)
    np.random.seed(1)
    n, d = 93, 13
    iu0, iu1 = np.triu_indices(n, 1)
    best = None
    best_val = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            if a.shape != (93, 13):
                continue
            a = _norm(np.asarray(a, dtype=np.float64))
            vv = _maxcos(a)
            if vv < best_val:
                best_val = vv
                best = a.copy()
        except Exception:
            pass
    if best is None:
        best = _norm(rng.standard_normal((93, 13)))
        best_val = _maxcos(best)

    def multi_void(X, k):
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        rm = G.max(axis=1)
        # pick k worst distinct (greedy far apart)
        order = np.argsort(-rm)
        chosen = []
        for cand in order:
            ok = True
            for c in chosen:
                if float(X[cand] @ X[c]) > 0.7:
                    ok = False
                    break
            if ok:
                chosen.append(int(cand))
            if len(chosen) >= k:
                break
        Xn = X.copy()
        keep_mask = np.ones(n, dtype=bool)
        keep_mask[np.array(chosen)] = False
        base = Xn[keep_mask]
        for idx_c, c in enumerate(chosen):
            vv = _find_void(base, rng, n_cand=5000)
            Xn[c] = vv
            base = np.vstack([base, vv[None, :]])
        Xn += 0.0015 * rng.standard_normal(Xn.shape)
        return _norm(Xn)

    trial = 0
    while time.time() < HOP_END and trial < 80:
        mode = trial % 6
        try:
            if mode == 0:
                sigma = [0.025, 0.012, 0.035, 0.008][(trial // 6) % 4]
                Xt = _norm(best + sigma * rng.standard_normal(best.shape))
            elif mode == 1:
                # rattler shake: move lowest rowmax points
                rm = _rowmax(best)
                low = np.argsort(rm)[:10]
                Xt = best.copy()
                Xt += 0.0015 * rng.standard_normal(Xt.shape)
                Xt[low] += 0.05 * rng.standard_normal((len(low), d))
                # also jitter bottleneck pair slightly
                hi = np.argsort(-rm)[:2]
                Xt[hi] += 0.008 * rng.standard_normal((len(hi), d))
                Xt = _norm(Xt)
            elif mode == 2:
                k = 2 + (trial % 3)
                Xt = multi_void(best, k)
            elif mode == 3:
                # cap shake
                u = rng.standard_normal(d)
                u /= np.linalg.norm(u)
                dots = best @ u
                cap = np.where(dots > 0.35)[0]
                if len(cap) < 5:
                    cap = np.argsort(-dots)[:8]
                Xt = best.copy() + 0.002 * rng.standard_normal(best.shape)
                # random rotation within cap to change topology
                Q, _ = np.linalg.qr(rng.standard_normal((d, d)))
                if np.linalg.det(Q) < 0:
                    Q[:, 0] = -Q[:, 0]
                # small rotation: mix
                ang = 0.15 + 0.1 * rng.random()
                R = np.eye(d) * np.cos(ang) + Q * 0.0
                # simpler: perturb cap strongly
                Xt[cap] += 0.04 * rng.standard_normal((len(cap), d))
                Xt = _norm(Xt)
            elif mode == 4:
                # subset resample: replace 6 random points by voids
                sub = rng.choice(n, size=6, replace=False)
                Xt = best.copy()
                mask = np.ones(n, dtype=bool)
                mask[sub] = False
                base = Xt[mask]
                for c in sub:
                    vv = _find_void(base, rng, n_cand=3000)
                    Xt[c] = vv
                    base = np.vstack([base, vv[None, :]])
                Xt += 0.002 * rng.standard_normal(Xt.shape)
                Xt = _norm(Xt)
            else:
                sigma = 0.015
                Xt = _norm(best + sigma * rng.standard_normal(best.shape))
        except Exception:
            trial += 1
            continue
        # annealed smooth with varied schedule
        try:
            if mode in (2, 4):
                Xt = _adam_smooth(Xt, steps=1000, tau0=0.015, tau1=0.0006, lr0=0.02, lr1=0.002)
            elif mode == 1:
                Xt = _adam_smooth(Xt, steps=900, tau0=0.008, tau1=0.0004, lr0=0.015, lr1=0.0015)
            else:
                Xt = _adam_smooth(Xt, steps=800, tau0=0.010, tau1=0.0007, lr0=0.015, lr1=0.001)
        except Exception:
            trial += 1
            continue
        if time.time() > DEADLINE - 4:
            vv = _maxcos(Xt)
            if vv < best_val:
                best, best_val = Xt, vv
            trial += 1
            continue
        try:
            # first bulk-average descent with large K to tie plateau
            Xb, vb = _hardK(Xt, 300, 0.0015, 800, iu0, iu1)
            # then focused descent on top
            Xc, vc = _hardK(Xb, 300, 0.001, 120, iu0, iu1)
            if vc < vb:
                Xb, vb = Xc, vc
            else:
                # try even more focused
                Xd, vd = _hardK(Xb, 200, 0.0006, 40, iu0, iu1)
                if vd < vb:
                    Xb, vb = Xd, vd
        except Exception:
            trial += 1
            continue
        if vb < best_val - 1e-12:
            best, best_val = Xb, vb
        trial += 1

    # final polish with remaining time
    try:
        # multiscale hard to equalize then focus
        for K, s0, it in [(800, 0.001, 500), (250, 0.0007, 500), (80, 0.0004, 500)]:
            if time.time() > t0 + DEADLINE - 8:
                break
            b, bv = _hardK(best, it, s0, K, iu0, iu1)
            if bv < best_val:
                best, best_val = b, bv
        remain = (t0 + DEADLINE) - time.time()
        if remain > 6:
            lb, lv = _lp_polish(best, seconds=max(1.0, remain - 3.0), r_init=1.5e-4)
            if lv < best_val:
                best, best_val = lb, lv
            if time.time() < t0 + DEADLINE - 2:
                b, bv = _hardK(best, 350, 0.0004, 150, iu0, iu1)
                if bv < best_val:
                    best, best_val = b, bv
    except Exception:
        pass
    best = _norm(best)
    return best
