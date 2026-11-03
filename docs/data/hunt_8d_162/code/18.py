import time
import numpy as np
from scipy.optimize import linprog, minimize
from scipy.sparse import coo_matrix


def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())


def contacts_cnt(x, thresh):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    m = float(g.max())
    cnt = np.sum(g > m - thresh, axis=1)
    return m, cnt


def lp_polish(x, seconds, r0=1e-3, rmax=0.05):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = max_cos(x)
    radius = r0
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        if time.time() - start > seconds - 0.05:
            break
        c = (x @ x.T)[iu]
        m = float(c.max())
        sel = c > m - max(20 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        if p == 0:
            break
        if p > 3000:
            idx = np.argpartition(-c[sel], 3000)[:3000]
            I, J = I[idx], J[idx]
            cc = c[sel][idx]
            p = 3000
        else:
            cc = c[sel]
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv)
        obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-cc, A_eq=A_eq, b_eq=np.zeros(n),
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
        if v < best:
            x, best, radius = y, v, min(radius * 1.5, rmax)
        else:
            radius *= 0.5
    return x


def smooth_refine(x0, taus=(0.04, 0.015), maxiters=(80, 120)):
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
            grad_Xn = 2.0 * Gw
            dot = (grad_Xn * Xn).sum(axis=1, keepdims=True)
            grad_X = (grad_Xn - dot * Xn) / norms[:, None]
            pen = 0.1 * float(((norms - 1.0) ** 2).sum())
            val = val + pen
            grad_X = grad_X + (0.2 * (norms - 1.0))[:, None] * Xn
            return val, grad_X.ravel()
        try:
            res = minimize(fun, flat, jac=True, method="L-BFGS-B",
                           options={"maxiter": mi, "ftol": 1e-14, "gtol": 1e-7, "maxfun": mi * 3})
            flat = res.x
        except Exception:
            break
    X = flat.reshape(n, d)
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    return X


def smooth_subset(x0, S, tau=0.02, maxiter=120):
    n, d = x0.shape
    x0 = x0 / np.linalg.norm(x0, axis=1, keepdims=True)
    S = np.asarray(S, dtype=int)
    k = len(S)
    if k == 0 or k >= n:
        return smooth_refine(x0, taus=(tau,), maxiters=(maxiter,))
    mask = np.ones(n, dtype=bool)
    mask[S] = False
    Xf = x0[mask]
    flat = x0[S].ravel().copy()
    def fun(f):
        Xs = f.reshape(k, d)
        ns = np.linalg.norm(Xs, axis=1)
        ns = np.maximum(ns, 1e-12)
        Xsn = Xs / ns[:, None]
        G1 = Xsn @ Xf.T
        G2 = Xsn @ Xsn.T
        np.fill_diagonal(G2, -10.0)
        m = float(max(float(G1.max()) if G1.size else -2.0, float(G2.max()) if G2.size else -2.0))
        E1 = np.exp((G1 - m) / tau) if G1.size else np.zeros((k, 0))
        E2 = np.exp((G2 - m) / tau)
        np.fill_diagonal(E2, 0.0)
        Ssum = float(E1.sum() + E2.sum() / 2.0)
        if Ssum <= 0:
            Ssum = 1e-300
        val = m + tau * np.log(Ssum + 1e-300)
        W1 = E1 / Ssum
        W2 = E2 / Ssum
        grad_Xn = W1 @ Xf + W2 @ Xsn
        dot = (grad_Xn * Xsn).sum(axis=1, keepdims=True)
        grad_X = (grad_Xn - dot * Xsn) / ns[:, None]
        pen = 0.1 * float(((ns - 1.0) ** 2).sum())
        val = val + pen
        grad_X = grad_X + (0.2 * (ns - 1.0))[:, None] * Xsn
        return val, grad_X.ravel()
    try:
        res = minimize(fun, flat, jac=True, method="L-BFGS-B",
                       options={"maxiter": maxiter, "ftol": 1e-14, "gtol": 1e-7, "maxfun": maxiter * 3})
        flat = res.x
    except Exception:
        pass
    X = x0.copy()
    Xs = flat.reshape(k, d)
    Xs /= np.linalg.norm(Xs, axis=1, keepdims=True)
    X[S] = Xs
    return X


def random_sphere(rng, k, d):
    a = rng.randn(k, d)
    a /= np.linalg.norm(a, axis=1, keepdims=True) + 1e-300
    return a


def best_hole(others, rng, nsamp=12000):
    d = others.shape[1]
    C = rng.randn(nsamp, d)
    C /= np.linalg.norm(C, axis=1, keepdims=True) + 1e-300
    dots = C @ others.T
    worst = dots.max(axis=1)
    bi = int(np.argmin(worst))
    return C[bi]


def near_identity_Q(rng, d, eps):
    M = np.eye(d) + eps * rng.randn(d, d)
    Q, R = np.linalg.qr(M)
    s = np.sign(np.diag(R))
    s[s == 0] = 1.0
    Q = Q * s[None, :]
    if np.linalg.det(Q) < 0:
        Q[:, 0] = -Q[:, 0]
    return Q


def subset_lp_step(x, S, radius, margin):
    n, d = x.shape
    S = np.asarray(S, dtype=int)
    k = len(S)
    if k == 0:
        return x
    pos = -np.ones(n, dtype=int)
    pos[S] = np.arange(k)
    G = x @ x.T
    iu = np.triu_indices(n, 1)
    cfull = G[iu]
    m = float(cfull.max())
    sel = cfull > m - margin
    I0, J0 = iu[0][sel], iu[1][sel]
    keep = np.array([(i in pos and pos[i] >= 0) or (j in pos and pos[j] >= 0) for i, j in zip(I0, J0)])
    # 'in pos' check via set for speed
    sset = set(S.tolist())
    keep = np.array([ (a in sset) or (b in sset) for a, b in zip(I0, J0)])
    I0, J0 = I0[keep], J0[keep]
    p = len(I0)
    if p == 0:
        return x
    if p > 2500:
        idx = np.argpartition(-cfull[sel][keep], 2500)[:2500]
        I0, J0 = I0[idx], J0[idx]
        p = 2500
    c = np.array([float(x[a] @ x[b]) for a, b in zip(I0, J0)])
    nv = k * d + 1
    rows_list = []
    cols_list = []
    vals_list = []
    for r in range(p):
        a, b = int(I0[r]), int(J0[r])
        if a in sset:
            pa = int(pos[a])
            for t in range(d):
                rows_list.append(r); cols_list.append(pa * d + t); vals_list.append(float(x[b, t]))
        if b in sset:
            pb = int(pos[b])
            for t in range(d):
                rows_list.append(r); cols_list.append(pb * d + t); vals_list.append(float(x[a, t]))
        rows_list.append(r); cols_list.append(k * d); vals_list.append(-1.0)
    A_ub = coo_matrix((np.array(vals_list), (np.array(rows_list), np.array(cols_list))), shape=(p, nv)).tocsr()
    # tangency: x_s . dx_s = 0
    er = np.repeat(np.arange(k), d)
    ec = np.arange(k * d)
    ev = x[S].ravel()
    A_eq = coo_matrix((ev, (er, ec)), shape=(k, nv)).tocsr()
    obj = np.zeros(nv); obj[-1] = 1.0
    try:
        res = linprog(obj, A_ub=A_ub, b_ub=-c, A_eq=A_eq, b_eq=np.zeros(k),
                      bounds=[(-radius, radius)] * (k * d) + [(None, None)], method="highs")
    except Exception:
        return x
    if res.status != 0:
        return x
    dx = res.x[:-1].reshape(k, d)
    y = x.copy()
    y[S] = y[S] + dx
    y /= np.linalg.norm(y, axis=1, keepdims=True)
    return y


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
    trial = 0
    while time.time() < DEADLINE - 13.0:
        trial += 1
        if time.time() > DEADLINE - 13.0:
            break
        m, cnt = contacts_cnt(x_best, thresh=2e-3)
        _, cnt_tight = contacts_cnt(x_best, thresh=1e-7)
        order = np.argsort(cnt)
        n, d = 162, 8
        mode = int(rng.randint(0, 7))
        x0 = x_best.copy()
        subset_S = None
        if mode == 0:
            sigma = float(rng.uniform(0.006, 0.025))
            x0 = x0 + sigma * rng.randn(n, d)
        elif mode == 1:
            k = int(rng.randint(6, 16))
            idx = np.argsort(cnt)[-k:]
            sigma = float(rng.uniform(0.06, 0.20))
            x0[idx] = x0[idx] + sigma * rng.randn(k, d)
            x0 = x0 + 0.004 * rng.randn(n, d)
        elif mode == 2:
            ratt = np.where(cnt_tight <= 2)[0]
            if len(ratt) == 0:
                ratt = order[:4]
            ri = int(rng.choice(ratt))
            others = np.delete(x_best, ri, axis=0)
            ns = int(rng.choice([8000, 15000]))
            x0[ri] = best_hole(others, rng, nsamp=ns)
            # shake overloaded neighbours
            k2 = int(rng.randint(3, 7))
            idx2 = np.argsort(cnt)[-k2:]
            x0[idx2] = x0[idx2] + 0.05 * rng.randn(k2, d)
            x0 = x0 + 0.003 * rng.randn(n, d)
            subset_S = np.concatenate([[ri], idx2])
        elif mode == 3:
            k = int(rng.randint(2, 6))
            idx_del = np.argsort(cnt)[-k:]
            keep = np.delete(x_best, idx_del, axis=0)
            # greedy sequential reinsert with spread
            new_pts = []
            cur = keep
            for _ in range(k):
                cand = random_sphere(rng, 8000, d)
                if len(new_pts):
                    cur2 = np.vstack([cur] + new_pts)
                else:
                    cur2 = cur
                dots = cand @ cur2.T
                worst = dots.max(axis=1)
                # enforce spread among candidates implicitly by picking diverse best
                bi = int(np.argmin(worst))
                new_pts.append(cand[bi:bi + 1])
            x0 = np.vstack([keep] + new_pts)
            # random permute to avoid index bias
            perm = rng.permutation(n)
            # need to map: we want x0 aligned; just permute rows
            x0 = x0[rng.permutation(n)] if False else x0
            subset_S = idx_del
        elif mode == 4:
            # rigid cap rotation
            u = random_sphere(rng, 1, d)[0]
            dots = x_best @ u
            mcap = int(rng.randint(8, 30))
            cap = np.argsort(dots)[-mcap:]
            eps = float(rng.uniform(0.05, 0.30))
            Q = near_identity_Q(rng, d, eps)
            x0[cap] = (Q @ x0[cap].T).T
            x0 = x0 + 0.004 * rng.randn(n, d)
        elif mode == 5:
            # directed subset LP move on overloaded set
            k = int(rng.randint(10, 24))
            idx_h = np.argsort(cnt)[-k:]
            ratt = np.where(cnt_tight <= 2)[0]
            S = idx_h.tolist() + ratt.tolist()
            S = np.unique(np.array(S, dtype=int))
            # limit size
            if len(S) > 30:
                S = np.concatenate([idx_h, rng.choice(S, size=30 - len(idx_h), replace=False)])
                S = np.unique(S)
            rad = float(rng.uniform(0.04, 0.12))
            marg = max(20 * rad, 0.004)
            y = subset_lp_step(x0, S, rad, marg)
            # small jitter after
            y[S] = y[S] + 0.008 * rng.randn(len(S), d)
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            x0 = y
            subset_S = S
        else:
            k = int(rng.randint(5, 13))
            kh = k // 2
            kl = k - kh
            idx_h = np.argsort(cnt)[-kh:] if kh > 0 else np.array([], dtype=int)
            idx_l = np.argsort(cnt)[:kl]
            idx = np.concatenate([idx_h, idx_l]).astype(int)
            sigma = float(rng.uniform(0.05, 0.16))
            x0[idx] = x0[idx] + sigma * rng.randn(len(idx), d)
            x0 = x0 + 0.003 * rng.randn(n, d)
        x0 /= np.linalg.norm(x0, axis=1, keepdims=True)
        # subset pre-polish if applicable
        if subset_S is not None and len(np.asarray(subset_S)) > 0 and len(np.asarray(subset_S)) < n:
            try:
                tau_s = float(rng.uniform(0.012, 0.03))
                x0 = smooth_subset(x0, np.asarray(subset_S).astype(int), tau=tau_s, maxiter=80)
            except Exception:
                pass
        tau1 = float(rng.uniform(0.025, 0.06))
        tau2 = float(rng.uniform(0.007, 0.02))
        mi1 = int(rng.choice([50, 70, 90]))
        mi2 = int(rng.choice([70, 100, 130]))
        try:
            xs = smooth_refine(x0, taus=(tau1, tau2), maxiters=(mi1, mi2))
        except Exception:
            continue
        vs = max_cos(xs)
        if vs > best_val + 0.015:
            if vs < best_val:
                x_best = xs; best_val = vs
            continue
        if time.time() > DEADLINE - 13.0:
            if vs < best_val - 1e-12:
                x_best = xs; best_val = vs
            continue
        remain = DEADLINE - 13.0 - time.time()
        budget = min(1.5, max(0.4, remain / 3.0))
        # only LP if within striking distance
        if vs > best_val + 0.006:
            if vs < best_val - 1e-12:
                x_best = xs; best_val = vs
            continue
        try:
            xl = lp_polish(xs, budget)
        except Exception:
            continue
        vl = max_cos(xl)
        if vl < best_val - 1e-12:
            x_best = xl; best_val = vl
        elif vs < best_val - 1e-12:
            x_best = xs; best_val = vs
    try:
        remain = DEADLINE - time.time() - 1.0
        if remain > 2.0:
            x_best = lp_polish(x_best, min(remain, 11.0), r0=2e-3)
    except Exception:
        pass
    x_best /= np.linalg.norm(x_best, axis=1, keepdims=True)
    return x_best
