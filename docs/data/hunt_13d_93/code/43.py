import time
import numpy as np

def _norm(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return X / n

def _maxcos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _rowmax(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return G.max(axis=1)

def _balanced_adam(X, steps, tau0, tau1, lr0, lr1, deadline=1e18):
    n, d = X.shape
    m = np.zeros_like(X)
    v = np.zeros_like(X)
    b1, b2, eps = 0.9, 0.999, 1e-8
    tt = 0
    ltau = np.log(tau1 / tau0) if tau0 != tau1 else 0.0
    llr = np.log(lr1 / lr0) if lr0 != lr1 else 0.0
    for s in range(steps):
        if (s % 50 == 0) and time.time() > deadline:
            break
        f = s / max(1, steps - 1)
        tau = tau0 * np.exp(ltau * f)
        lr = lr0 * np.exp(llr * f)
        if tau < 1e-12:
            tau = 1e-12
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        rmax = G.max(axis=1, keepdims=True)
        W = np.exp((G - rmax) / tau)
        np.fill_diagonal(W, 0.0)
        rs = W.sum(axis=1, keepdims=True) + 1e-12
        Wn = W / rs
        Ws = Wn + Wn.T
        grad = Ws @ X
        dot = np.sum(grad * X, axis=1, keepdims=True)
        gt = grad - dot * X
        if not np.all(np.isfinite(gt)):
            continue
        tt += 1
        m = b1 * m + (1.0 - b1) * gt
        v = b2 * v + (1.0 - b2) * (gt * gt)
        mh = m / (1.0 - b1 ** tt)
        vh = v / (1.0 - b2 ** tt)
        step = lr * mh / (np.sqrt(vh) + eps)
        d2 = np.sum(step * X, axis=1, keepdims=True)
        step_t = step - d2 * X
        X = X - step_t
        X = _norm(X)
    return X

def _global_adam(X, steps, tau0, tau1, lr0, lr1, deadline=1e18):
    n, d = X.shape
    m = np.zeros_like(X)
    v = np.zeros_like(X)
    b1, b2, eps = 0.9, 0.999, 1e-8
    tt = 0
    ltau = np.log(tau1 / tau0) if tau0 != tau1 else 0.0
    llr = np.log(lr1 / lr0) if lr0 != lr1 else 0.0
    for s in range(steps):
        if (s % 50 == 0) and time.time() > deadline:
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
        if not np.all(np.isfinite(gt)):
            continue
        tt += 1
        m = b1 * m + (1.0 - b1) * gt
        v = b2 * v + (1.0 - b2) * (gt * gt)
        mh = m / (1.0 - b1 ** tt)
        vh = v / (1.0 - b2 ** tt)
        step = lr * mh / (np.sqrt(vh) + eps)
        d2 = np.sum(step * X, axis=1, keepdims=True)
        step_t = step - d2 * X
        X = X - step_t
        X = _norm(X)
    return X

def _greedy_topk(X, iters, K0, K1, step0, deadline=1e18):
    n, d = X.shape
    iu0 = np.triu_indices(n, 1)[0]
    iu1 = np.triu_indices(n, 1)[1]
    P = len(iu0)
    Xc = _norm(X.copy())
    best = _maxcos(Xc)
    bestX = Xc.copy()
    step = float(step0)
    grad = np.empty_like(Xc)
    for it in range(iters):
        if (it % 40 == 0) and time.time() > deadline:
            break
        frac = it / max(1, iters - 1)
        K = int(round(K0 + (K1 - K0) * frac))
        K = max(1, min(K, P))
        G = Xc @ Xc.T
        c = G[iu0, iu1]
        if K < P:
            idx = np.argpartition(-c, K)[:K]
        else:
            idx = np.arange(P)
        I = iu0[idx]
        J = iu1[idx]
        grad.fill(0.0)
        np.add.at(grad, I, Xc[J])
        np.add.at(grad, J, Xc[I])
        grad /= float(len(idx))
        dot = np.sum(grad * Xc, axis=1, keepdims=True)
        gt = grad - dot * Xc
        ng = float(np.linalg.norm(gt))
        if ng == 0 or not np.isfinite(ng):
            step *= 0.5
            if step < 1e-13:
                step = float(step0) * 0.3
                continue
            continue
        Xt = _norm(Xc - step * gt / (ng / n + 1e-12) * 0.5 - step * 0.5 * gt)
        # simpler: try normalized + unnormalized blend via direct step
        # fallback direct:
        Xt2 = _norm(Xc - step * gt)
        v2 = _maxcos(Xt2)
        v1 = _maxcos(Xt)
        if v1 < v2:
            Xt = Xt
            vv = v1
        else:
            Xt = Xt2
            vv = v2
        if vv < best:
            best = vv
            bestX = Xt.copy()
            Xc = Xt
            step = min(step * 1.3, 0.01)
        else:
            step *= 0.55
            if step < 1e-12:
                step = float(step0) * 0.3
    return bestX, best

def _find_void(others, rng, n_cand=4000):
    d = others.shape[1]
    C = rng.standard_normal((n_cand, d))
    C /= np.linalg.norm(C, axis=1, keepdims=True)
    D = C @ others.T
    worst = D.max(axis=1)
    bi = int(np.argmin(worst))
    vv = C[bi].copy()
    lr = 0.05
    tau = 0.006
    for _ in range(120):
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

def _lp_polish(x, seconds):
    try:
        from scipy.optimize import linprog
        from scipy.sparse import coo_matrix
    except Exception:
        return x
    x = _norm(x.copy())
    n, d = x.shape
    best = _maxcos(x)
    radius = 8e-4
    start = time.time()
    iu = np.triu_indices(n, 1)
    ar = np.arange(d)
    while time.time() - start < seconds and radius > 1e-14:
        Gfull = x @ x.T
        c_all = Gfull[iu]
        mm = float(c_all.max())
        thresh = max(25.0 * radius, 5e-9)
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
        if getattr(res, "status", 1) != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y = _norm(y)
        v = _maxcos(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.01)
        else:
            radius *= 0.5
    return x

def solve():
    t0 = time.time()
    BUDGET = 168.0
    deadline_final = t0 + BUDGET
    rng = np.random.default_rng(0)
    np.random.seed(0)
    n, d = 93, 13

    best_val = 1e9
    x_best = None
    for fn in ("best.npy", "record.npy"):
        try:
            A = np.load(fn)
            A = np.asarray(A, dtype=float)
            if A.shape != (n, d):
                continue
            A = _norm(A)
            if not np.all(np.isfinite(A)):
                continue
            v = _maxcos(A)
            if v < best_val:
                best_val = v
                x_best = A.copy()
        except Exception:
            pass
    if x_best is None:
        A = rng.standard_normal(size=(n, d))
        A = _norm(A)
        x_best = A
        best_val = _maxcos(A)
    Xbest0 = x_best.copy()

    pool = [(best_val, x_best.copy())]

    # hopping phase: reserve last ~38s for final polish
    hop_end = t0 + 128.0
    trial = 0
    # schedules
    hub_sigmas = [0.10, 0.18, 0.28, 0.12, 0.22, 0.08, 0.15, 0.30]
    glob_sigmas = [0.008, 0.015, 0.005, 0.02, 0.010, 0.025, 0.008, 0.012]
    n_reins = [3, 2, 4, 3, 2, 4, 3, 2]
    while time.time() < hop_end and time.time() < deadline_final - 42:
        trial += 1
        si = (trial - 1) % len(hub_sigmas)
        hub_sigma = hub_sigmas[si]
        glob_sigma = glob_sigmas[si]
        nr = n_reins[si]
        # occasional big jump
        if trial % 9 == 0:
            hub_sigma = 0.45
            glob_sigma = 0.05
            nr = 5
        Xc = x_best.copy()
        # analyse current best topology
        try:
            G = Xc @ Xc.T
            np.fill_diagonal(G, -2.0)
            gmax = float(G.max())
            rowm = G.max(axis=1)
            # contacts at gmax-5e-7
            thr = 5e-7
            Ccnt = (G > gmax - thr).sum(axis=1)
            # hub = max contacts, tie by rowm
            hub = int(np.argmax(Ccnt * 10 + (rowm - rowm.min()) / max(1e-12, rowm.max() - rowm.min())))
            neigh = np.where(G[hub] > gmax - 2e-3)[0]
            ratt = np.where(rowm < gmax - 5e-7)[0]
            if len(ratt) == 0:
                ratt = np.argsort(rowm)[:6]
            # shake hub neighbourhood
            for idx in ([hub] + neigh.tolist()):
                Xc[idx] = Xc[idx] + hub_sigma * rng.standard_normal(d)
            # global small shake
            Xc = Xc + glob_sigma * rng.standard_normal(size=Xc.shape)
            Xc = _norm(Xc)
            # reinsert worst rattlers into voids
            if len(ratt) > 0:
                # pick nr with smallest rowm
                order = np.argsort(rowm[ratt])
                chosen = ratt[order[:min(nr, len(ratt))]]
                fixed_mask = np.ones(n, dtype=bool)
                fixed_mask[chosen] = False
                for ci in chosen:
                    others = Xc[fixed_mask]
                    vv = _find_void(others, rng, n_cand=3500)
                    Xc[ci] = vv
                    fixed_mask[ci] = True
            Xc = _norm(Xc)
        except Exception:
            Xc = _norm(x_best + 0.1 * rng.standard_normal(size=x_best.shape))

        # optimize: balanced then global then greedy-topk short
        try:
            dl = min(hop_end, deadline_final - 42)
            Xc = _balanced_adam(Xc, steps=320, tau0=0.015, tau1=0.0025, lr0=0.008, lr1=0.002, deadline=dl)
            if time.time() > dl:
                continue
            Xc = _global_adam(Xc, steps=260, tau0=0.004, tau1=0.0007, lr0=0.006, lr1=0.002, deadline=dl)
            if time.time() > dl:
                continue
            Xc, vv = _greedy_topk(Xc, iters=180, K0=350, K1=60, step0=0.004, deadline=dl)
        except Exception:
            continue
        try:
            vnew = _maxcos(Xc)
        except Exception:
            continue
        if not np.isfinite(vnew):
            continue
        pool.append((float(vnew), Xc.copy()))
        pool = sorted(pool, key=lambda t: t[0])[:8]
        if vnew < best_val - 1e-12:
            best_val = float(vnew)
            x_best = Xc.copy()

    # pick top candidates for final polish
    pool = sorted(pool, key=lambda t: t[0])[:4]
    # final extended polish on best 2
    final_best = x_best.copy()
    final_val = best_val
    for ci, (vv, XX) in enumerate(pool[:2]):
        if time.time() > deadline_final - 34:
            break
        try:
            dl = deadline_final - 12
            Y = _norm(XX.copy())
            # short balanced re-equalize then strong global
            Y = _balanced_adam(Y, steps=250, tau0=0.010, tau1=0.002, lr0=0.006, lr1=0.0015, deadline=dl)
            Y = _global_adam(Y, steps=500, tau0=0.003, tau1=0.0005, lr0=0.005, lr1=0.001, deadline=dl)
            Y, vy = _greedy_topk(Y, iters=600, K0=250, K1=30, step0=0.003, deadline=dl)
            vy = _maxcos(Y)
            if vy < final_val:
                final_val = vy
                final_best = Y.copy()
        except Exception:
            continue
    # LP finish on final best
    try:
        remain = deadline_final - time.time() - 2.0
        if remain > 4:
            Ylp = _lp_polish(final_best, seconds=min(remain, 14.0))
            vLp = _maxcos(Ylp)
            if np.isfinite(vLp) and vLp < final_val:
                final_val = vLp
                final_best = Ylp.copy()
    except Exception:
        pass
    final_best = _norm(final_best)
    # strict feasibility: shrink tiny margin? just renormalize, no zero rows
    return final_best
