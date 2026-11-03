import time
import numpy as np
from scipy.optimize import linprog, minimize
from scipy.sparse import coo_matrix


def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())


def contact_counts(x, thresh=2e-3):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    m = float(g.max())
    cnt = np.sum(g > m - thresh, axis=1)
    return m, cnt


def lp_polish(x, seconds, r0=2e-3, rmax=0.05):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = max_cos(x)
    radius = r0
    start = time.time()
    iu0 = np.triu_indices(n, 1)
    I0, J0 = iu0[0], iu0[1]
    while time.time() - start < seconds and radius > 1e-14:
        if time.time() - start > seconds - 0.05:
            break
        G = x @ x.T
        cfull = G[I0, J0]
        m = float(cfull.max())
        if m < best:
            best = m
        sel = cfull > m - max(20.0 * radius, 1e-9)
        cnt = int(sel.sum())
        if cnt == 0:
            radius *= 0.5
            continue
        if cnt > 3000:
            idx = np.argpartition(-cfull, 3000)[:3000]
            msk = np.zeros_like(sel)
            msk[idx] = True
            sel = msk
            cnt = 3000
        I = I0[sel]
        J = J0[sel]
        c = cfull[sel]
        nv = n * d + 1
        cols = np.concatenate([(I[:, None] * d + np.arange(d)).ravel(),
                               (J[:, None] * d + np.arange(d)).ravel(),
                               np.full(cnt, n * d)]).astype(np.int64)
        rows = np.repeat(np.arange(cnt), 2 * d + 1)
        vals = np.concatenate([x[J], x[I], -np.ones((cnt, 1))], axis=1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(cnt, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv)
        obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        v = max_cos(y)
        if v < best - 1e-15:
            x, best = y, v
            radius = min(radius * 1.5, rmax)
        else:
            radius *= 0.5
    return x


def fast_gd(X0, steps=700, tau0=0.045, tau1=0.007, lr0=0.05, mu=0.65):
    X = X0.copy()
    n, d = X.shape
    V = np.zeros_like(X)
    for it in range(steps):
        frac = it / max(steps - 1, 1)
        tau = tau0 * (tau1 / tau0) ** frac
        G = X @ X.T
        np.fill_diagonal(G, -10.0)
        m = float(G.max())
        E = np.exp((G - m) / tau)
        np.fill_diagonal(E, 0.0)
        S = float(E.sum())
        if not np.isfinite(S) or S <= 0:
            break
        W = E / S
        Grad = 2.0 * (W @ X)
        dot = np.sum(Grad * X, axis=1, keepdims=True)
        Gt = Grad - dot * X
        mx = float(np.sqrt((Gt * Gt).sum(axis=1)).max())
        if mx < 1e-14 or not np.isfinite(mx):
            break
        cur_lr = lr0 * (1.0 - 0.5 * frac)
        step = Gt / mx * cur_lr
        V = mu * V - step
        X = X + V
        X /= np.linalg.norm(X, axis=1, keepdims=True)
    return X


def lbfgs_full(x0, taus=(0.015, 0.007), maxiters=(70, 90)):
    n, d = x0.shape
    flat = (x0 / np.linalg.norm(x0, axis=1, keepdims=True)).ravel().copy()
    for tau, mi in zip(taus, maxiters):
        def fun(f):
            X = f.reshape(n, d)
            norms = np.linalg.norm(X, axis=1)
            norms = np.maximum(norms, 1e-12)
            Xn = X / norms[:, None]
            G = Xn @ Xn.T
            np.fill_diagonal(G, -10.0)
            m = float(G.max())
            E = np.exp((G - m) / tau)
            np.fill_diagonal(E, 0.0)
            S = float(E.sum())
            if S <= 0:
                S = 1e-300
            val = m + tau * np.log(S / 2.0 + 1e-300)
            W = E / S
            Gw = W @ Xn
            gXn = 2.0 * Gw
            dot = (gXn * Xn).sum(axis=1, keepdims=True)
            gX = (gXn - dot * Xn) / norms[:, None]
            pen = 0.1 * float(((norms - 1.0) ** 2).sum())
            val = val + pen
            gX = gX + (0.2 * (norms - 1.0))[:, None] * Xn
            return val, gX.ravel()
        try:
            res = minimize(fun, flat, jac=True, method="L-BFGS-B",
                           options={"maxiter": mi, "ftol": 1e-14, "gtol": 1e-7, "maxfun": mi * 3})
            flat = res.x
        except Exception:
            break
    X = flat.reshape(n, d)
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    return X


def subset_lbfgs(X0, idx, tau=0.012, maxiter=90):
    n, d = X0.shape
    k = len(idx)
    if k == 0:
        return X0
    Xfix = X0 / np.linalg.norm(X0, axis=1, keepdims=True)
    xvar0 = Xfix[idx].copy()
    flat0 = xvar0.ravel().copy()
    idx = np.asarray(idx, dtype=np.int64)

    def fun(f):
        Xv = f.reshape(k, d)
        nv = np.linalg.norm(Xv, axis=1)
        nv = np.maximum(nv, 1e-12)
        Xvn = Xv / nv[:, None]
        Xfull = Xfix.copy()
        Xfull[idx] = Xvn
        G = Xfull @ Xfull.T
        np.fill_diagonal(G, -10.0)
        m = float(G.max())
        E = np.exp((G - m) / tau)
        np.fill_diagonal(E, 0.0)
        S = float(E.sum())
        if S <= 0:
            S = 1e-300
        val = m + tau * np.log(S / 2.0 + 1e-300)
        W = E / S
        # grad w.r.t normalized full
        Gfull_grad = 2.0 * (W @ Xfull)
        gvn = Gfull_grad[idx]
        dot = (gvn * Xvn).sum(axis=1, keepdims=True)
        gv = (gvn - dot * Xvn) / nv[:, None]
        pen = 0.1 * float(((nv - 1.0) ** 2).sum())
        val = val + pen
        gv = gv + (0.2 * (nv - 1.0))[:, None] * Xvn
        return val, gv.ravel()

    try:
        res = minimize(fun, flat0, jac=True, method="L-BFGS-B",
                       options={"maxiter": maxiter, "ftol": 1e-14, "gtol": 1e-7, "maxfun": maxiter * 3})
        Xv = res.x.reshape(k, d)
        Xv /= np.linalg.norm(Xv, axis=1, keepdims=True)
        Xout = Xfix.copy()
        Xout[idx] = Xv
        return Xout
    except Exception:
        return Xfix


def refine_hole(X_keep, start, steps=250, tau=0.012, lr=0.05):
    v = start / np.linalg.norm(start)
    for it in range(steps):
        dots = X_keep @ v
        m = float(dots.max())
        w = np.exp((dots - m) / tau)
        s = float(w.sum())
        if not np.isfinite(s) or s <= 0:
            break
        w = w / s
        g = w @ X_keep
        gt = g - float(g @ v) * v
        nrm = float(np.linalg.norm(gt))
        if nrm < 1e-14:
            break
        cur = lr * (1.0 - 0.5 * it / steps)
        v = v - gt / nrm * cur
        v /= np.linalg.norm(v)
    return v


def find_holes(X_keep, k, rng):
    n_keep, d = X_keep.shape
    cand = rng.randn(12000, d)
    cand /= np.linalg.norm(cand, axis=1, keepdims=True)
    scores = (cand @ X_keep.T).max(axis=1)
    order = np.argsort(scores)
    picked = []
    for ii in order:
        if len(picked) >= k:
            break
        v = cand[ii]
        ok = True
        for q in picked:
            if float(v @ q) > 0.5:
                ok = False
                break
        if ok:
            picked.append(v.copy())
    out = []
    for v in picked:
        out.append(refine_hole(X_keep, v))
    while len(out) < k:
        v = rng.randn(d)
        v /= np.linalg.norm(v)
        out.append(refine_hole(X_keep, v))
    return np.array(out)


def random_sphere(rng, k, d):
    a = rng.randn(k, d)
    a /= np.linalg.norm(a, axis=1, keepdims=True)
    return a


def solve():
    np.random.seed(0)
    rng = np.random.RandomState(20261103)
    t0 = time.time()
    DEADLINE = t0 + 172.0
    try:
        xb = np.load("best.npy")
    except Exception:
        try:
            xb = np.load("record.npy")
        except Exception:
            xb = random_sphere(rng, 162, 8)
    xb = np.asarray(xb, dtype=float).reshape(162, 8)
    xb = xb / np.linalg.norm(xb, axis=1, keepdims=True)
    best_val = max_cos(xb)
    x_best = xb.copy()

    # brief snap of starting point if time allows
    if DEADLINE - time.time() > 30:
        try:
            xt = lp_polish(x_best, min(2.5, DEADLINE - time.time() - 20))
            vt = max_cos(xt)
            if vt < best_val:
                x_best, best_val = xt, vt
        except Exception:
            pass

    trial = 0
    n, d = 162, 8
    while time.time() < DEADLINE - 12.0:
        trial += 1
        _, cnt = contact_counts(x_best, thresh=2e-3)
        _, cnt_tight = contact_counts(x_best, thresh=1e-7)
        order = np.argsort(cnt)
        mode = trial % 6
        x0 = x_best.copy()

        if mode == 0:
            sig = float(rng.uniform(0.008, 0.03))
            x0 = x0 + sig * rng.randn(n, d)
            x0 /= np.linalg.norm(x0, axis=1, keepdims=True)
            xs = fast_gd(x0, steps=650, tau0=0.04, tau1=0.007, lr0=0.045, mu=0.6)
            vs = max_cos(xs)
            if vs > best_val + 0.02:
                continue
            xs = lbfgs_full(xs, taus=(0.015, 0.007), maxiters=(60, 80))
            vs = max_cos(xs)
            if vs < best_val - 1e-12:
                pass
            elif vs > best_val + 0.004:
                continue
            if time.time() > DEADLINE - 12.0:
                if vs < best_val - 1e-12:
                    x_best, best_val = xs, vs
                continue
            try:
                xl = lp_polish(xs, max(0.4, min(1.6, DEADLINE - 12.0 - time.time())))
            except Exception:
                continue
            vl = max_cos(xl)
            if vl < best_val - 1e-12:
                x_best, best_val = xl, vl
            elif vs < best_val - 1e-12:
                x_best, best_val = xs, vs

        elif mode == 1:
            k = int(rng.randint(7, 15))
            idx = np.argsort(cnt)[-k:]
            sig = float(rng.uniform(0.07, 0.20))
            x0[idx] = x0[idx] + sig * rng.randn(k, d)
            x0 = x0 + 0.004 * rng.randn(n, d)
            x0 /= np.linalg.norm(x0, axis=1, keepdims=True)
            xs = fast_gd(x0, steps=750, tau0=0.05, tau1=0.006, lr0=0.05, mu=0.7)
            vs = max_cos(xs)
            if vs > best_val + 0.02:
                continue
            xs = lbfgs_full(xs, taus=(0.012, 0.006), maxiters=(60, 90))
            vs = max_cos(xs)
            if vs > best_val + 0.004:
                continue
            if time.time() > DEADLINE - 12.0:
                if vs < best_val - 1e-12:
                    x_best, best_val = xs, vs
                continue
            try:
                xl = lp_polish(xs, max(0.4, min(1.6, DEADLINE - 12.0 - time.time())))
            except Exception:
                continue
            vl = max_cos(xl)
            if vl < best_val - 1e-12:
                x_best, best_val = xl, vl
            elif vs < best_val - 1e-12:
                x_best, best_val = xs, vs

        elif mode == 2:
            # subset surgery on worst cluster
            k = int(rng.randint(6, 12))
            idx = np.argsort(cnt)[-k:]
            sig = float(rng.uniform(0.05, 0.14))
            x0[idx] = x0[idx] + sig * rng.randn(k, d)
            x0 /= np.linalg.norm(x0, axis=1, keepdims=True)
            try:
                xs = subset_lbfgs(x0, idx, tau=float(rng.uniform(0.010, 0.020)), maxiter=90)
                xs = subset_lbfgs(xs, idx, tau=0.006, maxiter=90)
            except Exception:
                continue
            xs = fast_gd(xs, steps=400, tau0=0.02, tau1=0.006, lr0=0.03, mu=0.6)
            vs = max_cos(xs)
            if vs > best_val + 0.012:
                continue
            xs2 = lbfgs_full(xs, taus=(0.008,), maxiters=(80,))
            vs2 = max_cos(xs2)
            xs = xs2
            vs = vs2
            if vs > best_val + 0.004:
                continue
            if time.time() > DEADLINE - 12.0:
                if vs < best_val - 1e-12:
                    x_best, best_val = xs, vs
                continue
            try:
                xl = lp_polish(xs, max(0.4, min(1.8, DEADLINE - 12.0 - time.time())))
            except Exception:
                continue
            vl = max_cos(xl)
            if vl < best_val - 1e-12:
                x_best, best_val = xl, vl
            elif vs < best_val - 1e-12:
                x_best, best_val = xs, vs

        elif mode == 3:
            # delete overloaded + reinsert in holes
            k = int(rng.randint(1, 4))
            # pick victims: mostly high contacts
            hi = np.argsort(cnt)[-10:]
            victims = list(rng.choice(hi, size=min(k, len(hi)), replace=False))
            if len(victims) < k:
                extra = rng.choice(n, size=k - len(victims), replace=False)
                for e in extra:
                    if int(e) not in victims:
                        victims.append(int(e))
            victims = np.array(victims, dtype=int)[:k]
            keep_mask = np.ones(n, dtype=bool)
            keep_mask[victims] = False
            X_keep = x_best[keep_mask]
            try:
                holes = find_holes(X_keep, k, rng)
            except Exception:
                continue
            x0[ victims] = holes + float(rng.uniform(0.005, 0.02)) * rng.randn(k, d)
            # jitter overloaded neighbours a bit
            nb = np.argsort(cnt)[-4:]
            nb = nb[~np.isin(nb, victims)]
            if len(nb) > 0:
                x0[nb] = x0[nb] + 0.03 * rng.randn(len(nb), d)
            x0 /= np.linalg.norm(x0, axis=1, keepdims=True)
            xs = fast_gd(x0, steps=800, tau0=0.05, tau1=0.006, lr0=0.05, mu=0.7)
            vs = max_cos(xs)
            if vs > best_val + 0.02:
                continue
            xs = lbfgs_full(xs, taus=(0.013, 0.006), maxiters=(60, 90))
            vs = max_cos(xs)
            if vs > best_val + 0.004:
                continue
            if time.time() > DEADLINE - 12.0:
                if vs < best_val - 1e-12:
                    x_best, best_val = xs, vs
                continue
            try:
                xl = lp_polish(xs, max(0.4, min(1.8, DEADLINE - 12.0 - time.time())))
            except Exception:
                continue
            vl = max_cos(xl)
            if vl < best_val - 1e-12:
                x_best, best_val = xl, vl
            elif vs < best_val - 1e-12:
                x_best, best_val = xs, vs

        elif mode == 4:
            sig = float(rng.uniform(0.03, 0.085))
            x0 = x0 + sig * rng.randn(n, d)
            # split current worst pair slightly
            try:
                G = x_best @ x_best.T
                np.fill_diagonal(G, -2.0)
                ij = np.argmax(G)
                ii, jj = divmod(int(ij), n)
                direction = x0[ii] - x0[jj]
                direction /= (np.linalg.norm(direction) + 1e-12)
                push = float(rng.uniform(0.02, 0.07))
                x0[ii] = x0[ii] + push * direction
                x0[jj] = x0[jj] - push * direction
            except Exception:
                pass
            x0 /= np.linalg.norm(x0, axis=1, keepdims=True)
            xs = fast_gd(x0, steps=700, tau0=0.045, tau1=0.006, lr0=0.05, mu=0.65)
            vs = max_cos(xs)
            if vs > best_val + 0.02:
                continue
            xs = lbfgs_full(xs, taus=(0.014, 0.0065), maxiters=(60, 85))
            vs = max_cos(xs)
            if vs > best_val + 0.004:
                continue
            if time.time() > DEADLINE - 12.0:
                if vs < best_val - 1e-12:
                    x_best, best_val = xs, vs
                continue
            try:
                xl = lp_polish(xs, max(0.4, min(1.6, DEADLINE - 12.0 - time.time())))
            except Exception:
                continue
            vl = max_cos(xl)
            if vl < best_val - 1e-12:
                x_best, best_val = xl, vl
            elif vs < best_val - 1e-12:
                x_best, best_val = xs, vs

        else:
            # rattler + low-contact shake
            ratt = np.where(cnt_tight <= 2)[0]
            if len(ratt) == 0:
                ratt = order[:3]
            k = int(rng.randint(4, 11))
            idx_h = np.argsort(cnt)[-max(k // 2, 1):]
            idx_l = order[:k - len(idx_h)]
            idx = np.concatenate([idx_h, idx_l, ratt[:1]])
            idx = np.unique(idx)
            sig = float(rng.uniform(0.05, 0.16))
            x0[idx] = x0[idx] + sig * rng.randn(len(idx), d)
            x0 = x0 + 0.004 * rng.randn(n, d)
            x0 /= np.linalg.norm(x0, axis=1, keepdims=True)
            xs = fast_gd(x0, steps=650, tau0=0.04, tau1=0.007, lr0=0.045, mu=0.65)
            vs = max_cos(xs)
            if vs > best_val + 0.02:
                continue
            xs = lbfgs_full(xs, taus=(0.013, 0.006), maxiters=(60, 85))
            vs = max_cos(xs)
            if vs > best_val + 0.004:
                continue
            if time.time() > DEADLINE - 12.0:
                if vs < best_val - 1e-12:
                    x_best, best_val = xs, vs
                continue
            try:
                xl = lp_polish(xs, max(0.4, min(1.6, DEADLINE - 12.0 - time.time())))
            except Exception:
                continue
            vl = max_cos(xl)
            if vl < best_val - 1e-12:
                x_best, best_val = xl, vl
            elif vs < best_val - 1e-12:
                x_best, best_val = xs, vs

    try:
        remain = DEADLINE - time.time() - 0.8
        if remain > 1.5:
            x_best = lp_polish(x_best, min(remain, 10.0), r0=1e-3)
    except Exception:
        pass
    x_best /= np.linalg.norm(x_best, axis=1, keepdims=True)
    return x_best
