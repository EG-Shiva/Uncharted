import time
import numpy as np

def _norm(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return X / n

def _maxcos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def _p_cont(X0, p_list, sps, lr0, lr1, shift):
    n, d = X0.shape
    X = _norm(X0.copy())
    iu0, iu1 = np.triu_indices(n, 1)
    m = np.zeros_like(X)
    v = np.zeros_like(X)
    b1, b2, eps = 0.9, 0.999, 1e-8
    total = len(p_list) * sps
    tglob = 0
    best = _maxcos(X)
    bestX = X.copy()
    for p in p_list:
        m[:] = 0.0
        v[:] = 0.0
        tt = 0
        for s in range(sps):
            frac = tglob / max(1, total - 1)
            lr = lr0 * (lr1 / lr0) ** frac
            tglob += 1
            G = X @ X.T
            c = G[iu0, iu1]
            sc = c - shift
            pos = sc > 0
            gt = np.zeros_like(X)
            if np.any(pos):
                logs = np.full_like(c, -np.inf)
                logs[pos] = np.log(np.maximum(sc[pos], 1e-16))
                lmax = float(np.max(logs[pos]))
                e = np.zeros_like(c)
                e[pos] = np.exp((p - 1.0) * (logs[pos] - lmax))
                esum = float(e.sum())
                if esum > 0 and np.isfinite(esum):
                    w = e / esum
                    W = np.zeros((n, n), dtype=np.float64)
                    W[iu0, iu1] = w
                    W[iu1, iu0] = w
                    grad = W @ X
                    dot = np.einsum('ij,ij->i', grad, X)[:, None]
                    gt = grad - dot * X
            ng = float(np.linalg.norm(gt))
            if ng < 1e-15 or not np.isfinite(ng):
                continue
            tt += 1
            m = b1 * m + (1.0 - b1) * gt
            v = b2 * v + (1.0 - b2) * (gt * gt)
            mh = m / (1.0 - b1 ** tt)
            vh = v / (1.0 - b2 ** tt)
            step = lr * mh / (np.sqrt(vh) + eps)
            # tangent projection of step
            d2 = np.einsum('ij,ij->i', step, X)[:, None]
            step = step - d2 * X
            X = _norm(X - step)
            # track true max cheaply (reuse G next iter? just periodic)
            if (s % 25 == 0) or (s == sps - 1):
                cur = _maxcos(X)
                if cur < best:
                    best = cur
                    bestX = X.copy()
    cur = _maxcos(X)
    if best < cur:
        return bestX, best
    return X, cur

def _smooth_adam(x, steps, tau0, tau1, lr0, lr1):
    n, d = x.shape
    m = np.zeros_like(x)
    v = np.zeros_like(x)
    b1, b2, eps = 0.9, 0.999, 1e-8
    ltau = np.log(tau1 / tau0) if tau0 != tau1 else 0.0
    llr = np.log(lr1 / lr0) if lr0 != lr1 else 0.0
    tt = 0
    for s in range(steps):
        f = s / max(1, steps - 1)
        tau = tau0 * np.exp(ltau * f)
        lr = lr0 * np.exp(llr * f)
        if tau < 5e-5:
            tau = 5e-5
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        mg = float(G.max())
        W = np.exp((G - mg) / tau)
        np.fill_diagonal(W, 0.0)
        rs = W.sum(axis=1, keepdims=True) + 1e-12
        grad = (W @ x) / np.sqrt(rs)
        dot = np.einsum('ij,ij->i', grad, x)[:, None]
        gt = grad - dot * x
        if not np.all(np.isfinite(gt)):
            break
        tt += 1
        m = b1 * m + (1.0 - b1) * gt
        v = b2 * v + (1.0 - b2) * (gt * gt)
        mh = m / (1.0 - b1 ** tt)
        vh = v / (1.0 - b2 ** tt)
        step = lr * mh / (np.sqrt(vh) + eps)
        d2 = np.einsum('ij,ij->i', step, x)[:, None]
        step = step - d2 * x
        x = _norm(x - step)
    return x

def _hardK(X, iters, step0, K0, K1, deadline=1e18):
    n, d = X.shape
    iu0, iu1 = np.triu_indices(n, 1)
    mtot = len(iu0)
    best = _maxcos(X)
    bestX = X.copy()
    step = step0
    grad = np.empty_like(X)
    for it in range(iters):
        if (it % 40 == 0) and time.time() > deadline:
            break
        frac = it / max(1, iters - 1)
        K = int(round(K0 + (K1 - K0) * frac))
        K = max(1, min(K, mtot))
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
        dot = np.einsum('ij,ij->i', grad, X)[:, None]
        gt = grad - dot * X
        rn = np.linalg.norm(gt, axis=1, keepdims=True)
        nz = rn[:, 0] > 1e-14
        direc = np.zeros_like(gt)
        direc[nz] = gt[nz] / np.maximum(rn[nz], 1e-300)
        Xt = _norm(X - step * direc)
        vv = _maxcos(Xt)
        if vv < best:
            best = vv
            bestX = Xt.copy()
            X = Xt
            step = min(step * 1.25, 0.006)
        else:
            step *= 0.55
            if step < 1e-12:
                X = bestX.copy()
                step = step0 * 0.25
                if it > iters * 0.65:
                    break
    return bestX, best

def _refine_void(others, vv, iters=120):
    lr = 0.05
    tau = 0.006
    for _ in range(iters):
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
            lr = min(lr * 1.06, 0.08)
        else:
            lr *= 0.5
            if lr < 1e-6:
                break
    nrm = float(np.linalg.norm(vv))
    if nrm == 0 or not np.isfinite(nrm):
        return vv
    return vv / nrm

def _void_replace(x, rng, nrep=1, ncand=2500):
    n, d = x.shape
    for _ in range(nrep):
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        per = G.max(axis=1)
        iw = int(np.argmax(per))
        others = np.delete(x, iw, axis=0)
        C = rng.standard_normal((ncand, d))
        C /= np.linalg.norm(C, axis=1, keepdims=True)
        md = (C @ others.T).max(axis=1)
        vv = C[int(np.argmin(md))].copy()
        vv = _refine_void(others, vv, iters=120)
        x[iw] = vv
    return x

def _lp_polish(x, seconds, r_init=0.0012):
    try:
        from scipy.optimize import linprog
        from scipy.sparse import coo_matrix
    except Exception:
        return x, _maxcos(x)
    x = _norm(x.copy())
    n, d = x.shape
    best = _maxcos(x)
    radius = float(r_init)
    start = time.time()
    iu = np.triu_indices(n, 1)
    ar = np.arange(d)
    while time.time() - start < seconds and radius > 1e-14:
        cfull = (x @ x.T)[iu]
        m = float(cfull.max())
        thresh = max(22.0 * radius, 5e-8)
        sel = cfull > m - thresh
        if int(sel.sum()) == 0:
            radius *= 0.5
            continue
        if int(sel.sum()) > 3000:
            idx = np.argpartition(-cfull, 3000)[:3000]
            mask = np.zeros_like(sel)
            mask[idx] = True
            sel = mask
        I = iu[0][sel]
        J = iu[1][sel]
        p = len(I)
        csub = cfull[sel]
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + ar, J[:, None] * d + ar, np.full((p, 1), n * d)], axis=1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], axis=1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        except Exception:
            radius *= 0.5
            continue
        obj = np.zeros(nv)
        obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-csub, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = _norm(x + res.x[:-1].reshape(n, d))
        vv = _maxcos(y)
        if vv < best:
            x, best = y, vv
            radius = min(radius * 1.5, 0.008)
        else:
            radius *= 0.5
        if time.time() - start > seconds - 0.3:
            break
    return x, best

def solve():
    t0 = time.time()
    BUDGET = 172.0
    rng = np.random.default_rng(0)
    n, d = 93, 13
    best_val = 1e9
    best_x = None
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.asarray(np.load(fn), dtype=np.float64)
            if a.shape == (n, d):
                a = _norm(a)
                va = _maxcos(a)
                if va < best_val:
                    best_val = va
                    best_x = a.copy()
        except Exception:
            continue
    if best_x is None:
        best_x = _norm(rng.standard_normal((n, d)))
        best_val = _maxcos(best_x)
    try:
        rec = _norm(np.asarray(np.load("record.npy"), dtype=np.float64))
    except Exception:
        rec = best_x.copy()

    starts = []
    starts.append(best_x.copy())
    starts.append(rec.copy())
    starts.append(_norm(best_x + 0.15 * rng.standard_normal((n, d))))
    starts.append(_norm(best_x + 0.30 * rng.standard_normal((n, d))))
    starts.append(_norm(best_x + 0.60 * rng.standard_normal((n, d))))
    starts.append(_norm(rec + 0.30 * rng.standard_normal((n, d))))
    starts.append(_norm(rng.standard_normal((n, d))))
    starts.append(_norm(rng.standard_normal((n, d))))

    p_list = [8, 16, 32, 64, 128, 256]
    results = [(best_val, best_x.copy())]
    for si, s0 in enumerate(starts):
        if time.time() - t0 > 80.0:
            break
        sh = 0.15 if si % 2 == 0 else 0.18
        sps = 250 if si < 5 else 180
        try:
            Xc, vc = _p_cont(s0, p_list, sps, 3e-3, 3e-4, sh)
        except Exception:
            continue
        results.append((vc, Xc))
        if vc < best_val:
            best_val = vc
            best_x = Xc.copy()

    results.sort(key=lambda t: t[0])
    # promote top 3 to soft+hard polish
    cands = []
    for va, xa in results[:3]:
        if time.time() - t0 > 115.0:
            break
        try:
            xs = _smooth_adam(xa.copy(), steps=350, tau0=0.006, tau1=0.0012, lr0=2.5e-3, lr1=4e-4)
            Xp, vp = _hardK(xs, iters=700, step0=0.0025, K0=450, K1=35, deadline=t0 + 130.0)
        except Exception:
            continue
        cands.append((vp, Xp))
        if vp < best_val:
            best_val = vp
            best_x = Xp.copy()
    if cands:
        cands.sort(key=lambda t: t[0])
        best_val, best_x = cands[0][0], cands[0][1].copy()

    # basin hops around current best to change topology
    hop_rng = np.random.default_rng(1)
    nhop = 0
    while time.time() - t0 < 152.0 and nhop < 40:
        nhop += 1
        sig = 0.012 if nhop % 5 != 0 else 0.05
        xt = best_x + sig * hop_rng.standard_normal((n, d))
        xt = _norm(xt)
        if nhop % 7 == 0:
            try:
                xt = _void_replace(xt.copy(), hop_rng, nrep=1)
            except Exception:
                pass
        try:
            if nhop % 3 == 0:
                xt = _smooth_adam(xt, steps=150, tau0=0.004, tau1=0.001, lr0=1.5e-3, lr1=3e-4)
            Xp, vp = _hardK(xt, iters=220, step0=0.0018, K0=300, K1=30, deadline=t0 + 155.0)
        except Exception:
            continue
        if vp < best_val:
            best_val = vp
            best_x = Xp.copy()

    # final LP squeeze + short hard polish
    rem = (t0 + BUDGET) - time.time() - 2.0
    if rem > 2.0:
        try:
            Xl, vl = _lp_polish(best_x.copy(), seconds=min(14.0, rem * 0.6))
            if vl < best_val:
                best_val = vl
                best_x = Xl.copy()
        except Exception:
            pass
    if time.time() - t0 < t0 + BUDGET - 3.0:
        try:
            Xp2, vp2 = _hardK(best_x.copy(), iters=300, step0=0.001, K0=180, K1=25, deadline=t0 + BUDGET - 1.0)
            if vp2 < best_val:
                best_val = vp2
                best_x = Xp2.copy()
        except Exception:
            pass

    best_x = _norm(best_x)
    return best_x
