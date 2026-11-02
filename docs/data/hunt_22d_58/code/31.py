import time
import numpy as np

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _count_above(X, t):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return int((G > t).sum() // 2)

def _degrees(X, thr):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return (G > thr).sum(axis=1)

def _adam_pump(X0, t, steps, lr=0.008, beta0=40.0, beta1=400.0,
               margin=0.01, t0=None, deadline=1e18):
    X = X0.copy()
    n, d = X.shape
    m1 = np.zeros_like(X)
    v1 = np.zeros_like(X)
    b1, b2, eps = 0.9, 0.999, 1e-8
    lb0, lb1 = np.log(beta0), np.log(beta1)
    for s in range(steps):
        if t0 is not None and (s % 500 == 0) and (time.time() - t0 > deadline):
            break
        f = s / max(1, steps - 1)
        beta = float(np.exp(lb0 + (lb1 - lb0) * f))
        lr_s = lr * (1.0 - 0.7 * f)
        G = X @ X.T
        np.fill_diagonal(G, -1.0)
        M = G > (t - margin)
        # exclude diagonal already false
        if not np.any(M):
            break
        D = G - t
        Z = beta * D
        Zc = np.clip(Z, -60.0, 60.0)
        sig = 1.0 / (1.0 + np.exp(-Zc))
        # softplus(beta*D)/beta stable
        # for D>0: D + log1p(exp(-Z))/beta else log1p(exp(Z))/beta
        e_neg = np.exp(-np.clip(Z, 0.0, 60.0))
        e_pos = np.exp(np.clip(Z, -60.0, 0.0))
        soft_pos = D + np.log1p(e_neg) / beta
        soft_neg = np.log1p(e_pos) / beta
        soft = np.where(D > 0, soft_pos, soft_neg)
        W = 2.0 * soft * sig
        W[~M] = 0.0
        np.fill_diagonal(W, 0.0)
        Grad = W @ X
        proj = (Grad * X).sum(axis=1, keepdims=True)
        Gt = Grad - X * proj
        m1 = b1 * m1 + (1.0 - b1) * Gt
        v1 = b2 * v1 + (1.0 - b2) * (Gt * Gt)
        bc1 = 1.0 - b1 ** (s + 1)
        bc2 = 1.0 - b2 ** (s + 1)
        mh = m1 / bc1
        vh = v1 / bc2
        upd = mh / (np.sqrt(vh) + eps)
        # re-project update to tangent
        uproj = (upd * X).sum(axis=1, keepdims=True)
        upd_t = upd - X * uproj
        X = X - lr_s * upd_t
        nr = np.linalg.norm(X, axis=1, keepdims=True)
        nr[nr == 0] = 1.0
        X = X / nr
        # early feasible exit
        if (s % 200 == 199):
            G2 = X @ X.T
            np.fill_diagonal(G2, -2.0)
            if float(G2.max()) <= t:
                break
    return X

def _lp_polish_weighted(X, seconds, eps_sec=2e-3):
    try:
        from scipy.optimize import linprog
        from scipy.sparse import coo_matrix
    except Exception:
        return X
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    n, d = X.shape
    best = _max_cos(X)
    start = time.time()
    radius = 1e-3
    iu = np.triu_indices(n, 1)
    tau = 0.002
    while time.time() - start < seconds and radius > 1e-14:
        G = X @ X.T
        c = G[iu]
        m = float(c.max())
        sel = c > m - max(20 * radius, 1e-9)
        I0, J0 = iu[0][sel], iu[1][sel]
        cw0 = c[sel]
        # stress weights from previous multipliers proxy
        w0 = np.exp(np.clip((cw0 - m) / tau, -20, 0))
        keep = w0 > 1e-4
        # always keep the absolute max pairs
        keep = keep | (cw0 >= m - 1e-12)
        I, J, cw, w = I0[keep], J0[keep], cw0[keep], w0[keep]
        if len(I) == 0:
            break
        # secondary gradient g = W_sym @ X
        # objective coeffs for dx: eps_sec * g
        g = np.zeros_like(X)
        # accumulate (loop over pairs, p~few hundred, fine)
        for k in range(len(I)):
            i, j, wk = int(I[k]), int(J[k]), float(w[k])
            g[i] += wk * X[j]
            g[j] += wk * X[i]
        p = len(I)
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([X[J], X[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((X.ravel(), (np.repeat(np.arange(n), d),
                                           np.arange(n * d))), shape=(n, nv)).tocsr()
        except Exception:
            break
        obj = np.zeros(nv)
        obj[:-1] = eps_sec * g.ravel()
        obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-cw, A_eq=A_eq,
                          b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                          method="highs")
        except Exception:
            break
        if res.status != 0:
            radius *= 0.5
            continue
        y = X + res.x[:-1].reshape(n, d)
        nr = np.linalg.norm(y, axis=1, keepdims=True)
        nr[nr == 0] = 1.0
        y = y / nr
        v = _max_cos(y)
        if v < best:
            X, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
        if time.time() - start > seconds:
            break
    return X

def _quench(X, steps, lr0, lr1, tau0, tau1):
    for s in range(steps):
        f = s / max(1, steps - 1)
        lr = lr0 * (lr1 / lr0) ** f if lr0 != lr1 else lr0
        tau = tau0 * (tau1 / tau0) ** f if tau0 != tau1 else tau0
        G = X @ X.T
        np.fill_diagonal(G, -1.0)
        m = G.max()
        W = np.exp((G - m) / tau)
        np.fill_diagonal(W, 0.0)
        WX = W @ X
        WG = (W * G).sum(axis=1, keepdims=True)
        F = WX - X * WG
        rs = W.sum(axis=1, keepdims=True) + 1e-12
        F /= rs
        X = X - lr * F
        X /= np.linalg.norm(X, axis=1, keepdims=True)
    return X

def _greedy_push(X, steps, eps0, eps1):
    n = X.shape[0]
    for s in range(steps):
        f = s / max(1, steps - 1)
        eps = eps0 * (eps1 / eps0) ** f if eps0 != eps1 else eps0
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        k = int(np.argmax(G))
        i = k // n
        j = k % n
        g = float(G[i, j])
        xi = X[i].copy(); xj = X[j].copy()
        ti = xj - g * xi; tj = xi - g * xj
        ni = np.linalg.norm(ti) + 1e-12
        nj = np.linalg.norm(tj) + 1e-12
        X[i] = xi - eps * ti / ni
        X[j] = xj - eps * tj / nj
        X[i] /= np.linalg.norm(X[i]); X[j] /= np.linalg.norm(X[j])
    return X

def solve():
    t_start = time.time()
    HARD = 160.0  # hard return deadline from start
    rng = np.random.default_rng(0)
    cands = []
    for fn in ("best.npy", "record.npy"):
        try:
            c = np.load(fn).astype(float)
            if c.shape == (58, 22):
                c = c / np.linalg.norm(c, axis=1, keepdims=True)
                c = np.where(np.isfinite(c), c, 0.0)
                nr = np.linalg.norm(c, axis=1, keepdims=True)
                nr[nr == 0] = 1.0
                c = c / nr
                cands.append(c)
        except Exception:
            pass
    if not cands:
        c = rng.standard_normal((58, 22))
        c /= np.linalg.norm(c, axis=1, keepdims=True)
        cands.append(c)
    vals = [_max_cos(c) for c in cands]
    best = cands[int(np.argmin(vals))].copy()
    best_val = float(min(vals))
    cur = best.copy()
    cur_viol = _count_above(cur, best_val - 0.0003)

    # target slightly below incumbent
    t_target = best_val - 0.0003
    # snap to suggested ladder if close
    if best_val > 0.0663 and t_target < 0.0660:
        t_target = 0.0663
    outer = 0
    lp_every = 3
    while time.time() - t_start < 118 and outer < 40:
        outer += 1
        remain = HARD - (time.time() - t_start)
        if remain < 30:
            break
        steps = 4000 if outer < 8 else 3000
        # anneal beta upward harder in later rounds
        b0 = 30.0 + 5.0 * outer
        b1 = 300.0 + 100.0 * outer
        if b1 > 1200:
            b1 = 1200.0
        lr = 0.009 if outer % 2 == 1 else 0.006
        Xn = _adam_pump(cur, t_target, steps, lr=lr, beta0=b0, beta1=b1,
                        margin=0.01, t0=t_start, deadline=115)
        mv = _max_cos(Xn)
        nv_t = _count_above(Xn, t_target)
        # success: t-feasible -> new record region
        if mv < best_val - 1e-12:
            best = Xn.copy(); best_val = mv
            # LP-polish feasible point quickly
            if HARD - (time.time() - t_start) > 25:
                best = _lp_polish_weighted(best, seconds=3.0)
                best_val = _max_cos(best)
            # lower target
            t_target = best_val - 0.0003
            cur = best.copy()
            cur_viol = _count_above(cur, t_target)
            continue
        # alternate LP secondary step on Xn occasionally
        if outer % lp_every == 0 and HARD - (time.time() - t_start) > 35:
            try:
                Xlp = _lp_polish_weighted(Xn, seconds=2.5)
                mlp = _max_cos(Xlp)
                if mlp < mv - 1e-12:
                    Xn, mv = Xlp, mlp
                    nv_t = _count_above(Xn, t_target)
                    if mv < best_val - 1e-12:
                        best = Xn.copy(); best_val = mv
                        t_target = best_val - 0.0003
                        cur = best.copy(); cur_viol = _count_above(cur, t_target)
                        continue
            except Exception:
                pass
        # topology check: markedly fewer contacts => new basin
        Gtmp = Xn @ Xn.T
        np.fill_diagonal(Gtmp, -2.0)
        mtmp = float(Gtmp.max())
        degs = (Gtmp > mtmp - 1e-7).sum(axis=1)
        mn, med = int(degs.min()), float(np.median(degs))
        if mn <= 8 or mn <= med * 0.4:
            # de-jammed: keep new topology even if slightly worse
            if mv < best_val + 0.002:
                cur = Xn.copy()
                cur_viol = nv_t
                # also try small perturbation of best toward this basin next round?
                continue
        # progress in violation count?
        if nv_t < cur_viol:
            cur = Xn.copy(); cur_viol = nv_t
        else:
            # restart from incumbent with perturbation; larger kick if stuck
            sigma = 0.004 if (outer % 4 != 0) else 0.012
            # perturb high-degree points more
            Gc = cur @ cur.T
            np.fill_diagonal(Gc, -2.0)
            mc = float(Gc.max())
            cnt = (Gc > mc - 0.004).sum(axis=1)
            order = np.argsort(-cnt)
            pert = np.zeros_like(cur)
            # perturb top-6 loaded points harder
            k = min(6, cur.shape[0])
            victims = order[:k]
            pert += 0.0
            nxt = best.copy()
            nxt[victims] += sigma * 2.0 * rng.standard_normal((k, 22))
            nxt += (sigma * 0.3) * rng.standard_normal(nxt.shape)
            nxt /= np.linalg.norm(nxt, axis=1, keepdims=True)
            cur = nxt
            cur_viol = _count_above(cur, t_target)
        # if violations at t_target are few, tighten: keep same t, else maybe relax t upward slightly?
        # if stuck with many violations, raise t a bit toward best to make progress
        if outer % 6 == 5 and cur_viol > 200:
            t_target = min(best_val - 1e-4, t_target + 0.0001)
    # final polish of best
    if HARD - (time.time() - t_start) > 8:
        try:
            best = _quench(best, 2500, 0.008, 0.0006, 0.004, 0.0007)
        except Exception:
            pass
    if HARD - (time.time() - t_start) > 6:
        try:
            best = _greedy_push(best, 1500, 0.001, 0.00005)
        except Exception:
            pass
    if HARD - (time.time() - t_start) > 12:
        try:
            secs = min(8.0, HARD - (time.time() - t_start) - 3.0)
            if secs > 2:
                best = _lp_polish_weighted(best, seconds=secs)
        except Exception:
            pass
    best /= np.linalg.norm(best, axis=1, keepdims=True)
    return best
