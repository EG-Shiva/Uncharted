import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _mc(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def _nm(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return X / n

def _lp_deadline(X, deadline):
    X = _nm(X.copy())
    n, d = X.shape
    best = _mc(X)
    rad = 1e-3
    iu = np.triu_indices(n, 1)
    while time.time() < deadline and rad > 1e-14:
        C = (X @ X.T)[iu]
        m = float(np.max(C))
        sel = C > m - max(20.0 * rad, 1e-9)
        I = iu[0][sel]; J = iu[1][sel]
        cs = C[sel]
        p = len(I)
        if p == 0:
            break
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([X[J], X[I], -np.ones((p, 1))], 1).ravel()
        Aub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        Aeq = coo_matrix((X.ravel(), (np.repeat(np.arange(n), d),
                                      np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            r = linprog(obj, A_ub=Aub, b_ub=-cs, A_eq=Aeq, b_eq=np.zeros(n),
                        bounds=[(-rad, rad)] * (n * d) + [(None, None)],
                        method="highs")
        except Exception:
            break
        if r.status != 0:
            rad *= 0.5
            continue
        Y = _nm(X + r.x[:-1].reshape(n, d))
        v = _mc(Y)
        if v < best:
            X, best = Y, v
            rad = min(rad * 1.5, 0.05)
        else:
            rad *= 0.5
    return X, best

def _plane_rot(rng, theta):
    A = rng.randn(4, 2)
    Q, _ = np.linalg.qr(A)
    e1 = Q[:, 0]; e2 = Q[:, 1]
    c = np.cos(theta); s = np.sin(theta)
    R = np.eye(4) + (c - 1.0) * (np.outer(e1, e1) + np.outer(e2, e2)) + s * (np.outer(e2, e1) - np.outer(e1, e2))
    return R

def _rand_orth(rng):
    Z = rng.randn(4, 4)
    Q, Rq = np.linalg.qr(Z)
    Q = Q * np.sign(np.diag(Rq) + 1e-30)
    if np.linalg.det(Q) < 0:
        Q[:, 0] = -Q[:, 0]
    return Q

def solve():
    t0 = time.time()
    HARD_END = t0 + 170.0
    SEARCH_END = t0 + 136.0
    rng = np.random.RandomState(1684031)
    n, d = 168, 4
    try:
        xb = np.load("best.npy").astype(float)
    except Exception:
        xb = rng.randn(n, d)
    xb = _nm(xb)
    try:
        xr = _nm(np.load("record.npy").astype(float))
        if xr.shape == (n, d) and _mc(xr) < _mc(xb):
            xb = xr.copy()
    except Exception:
        pass
    if xb.shape != (n, d):
        xb = _nm(rng.randn(n, d))
    best_val = _mc(xb)
    x_best = xb.copy()
    x_cur = xb.copy()
    cur_val = best_val

    def refine(X0, steps, b0=30.0, b1=600.0, lr0=0.016, lr1=0.0007, mu=0.88):
        X = X0.copy()
        vel = np.zeros_like(X)
        lb0 = np.log(b0); lb1 = np.log(b1)
        for s in range(steps):
            f = s / max(1, steps - 1)
            beta = float(np.exp(lb0 + (lb1 - lb0) * f))
            lr = lr0 * (1.0 - f) + lr1 * f
            G = X @ X.T
            np.fill_diagonal(G, -2.0)
            cm = float(np.max(G))
            E = np.exp(np.clip(beta * (G - cm), -25.0, 0.0))
            np.fill_diagonal(E, 0.0)
            rs = np.sum(E, axis=1, keepdims=True) + 1e-12
            grad = (E @ X) / rs
            gt = grad - np.sum(grad * X, axis=1, keepdims=True) * X
            sc = np.mean(np.linalg.norm(gt, axis=1)) + 1e-12
            gt = gt / sc
            vel = mu * vel + (1.0 - mu) * gt
            X = X - lr * vel
            X = _nm(X)
            if time.time() > SEARCH_END:
                break
        return X

    def refine_single(s0, Xref, steps=100, b0=45.0, b1=280.0, lr0=0.09, lr1=0.006):
        s = s0 / (np.linalg.norm(s0) + 1e-30)
        lb0 = np.log(b0); lb1 = np.log(b1)
        for it in range(steps):
            f = it / max(1, steps - 1)
            beta = float(np.exp(lb0 + (lb1 - lb0) * f))
            lr = lr0 * (1.0 - f) + lr1 * f
            dots = Xref @ s
            cm = float(np.max(dots))
            w = np.exp(np.clip(beta * (dots - cm), -25.0, 0.0))
            w /= (np.sum(w) + 1e-30)
            g = w @ Xref
            gt = g - np.dot(g, s) * s
            gn = np.linalg.norm(gt) + 1e-12
            gt = gt / gn
            s = s - lr * gt
            ns = np.linalg.norm(s)
            if ns < 1e-12:
                s = rng.randn(4); ns = np.linalg.norm(s)
            s = s / ns
        return s

    def void_pt(Xref, nsamp=9000):
        C = rng.randn(nsamp, 4)
        C /= (np.linalg.norm(C, axis=1, keepdims=True) + 1e-30)
        D = C @ Xref.T
        m = D.max(axis=1)
        return C[int(np.argmin(m))]

    # quick squeeze of incumbent to set baseline
    try:
        x_best, best_val = _lp_deadline(x_best, min(t0 + 8.0, SEARCH_END))
        x_cur, cur_val = x_best.copy(), best_val
    except Exception:
        pass

    trial = 0
    while time.time() < SEARCH_END:
        trial += 1
        if time.time() > SEARCH_END - 1.0:
            break
        if trial % 5 == 0 and cur_val < best_val + 0.004:
            base = x_cur
        else:
            base = x_best
        typ = trial % 12
        X = base.copy()
        if typ == 0:
            X = _nm(X + 0.012 * rng.randn(n, d))
            X = refine(X, 450, 25.0, 500.0, 0.014, 0.0007)
        elif typ == 1:
            X = _nm(X + 0.05 * rng.randn(n, d))
            X = refine(X, 550, 30.0, 700.0, 0.016, 0.0007)
        elif typ == 2:
            # hemisphere fault: rotate one side
            u = rng.randn(4); u /= np.linalg.norm(u)
            pr = X @ u
            thr = np.median(pr) + rng.randn() * 0.1
            mask = pr > thr
            if mask.sum() == 0 or mask.sum() == n:
                mask[:84] = True; mask[84:] = False
            th = float(rng.uniform(0.06, 0.30))
            R = _plane_rot(rng, th)
            X[mask] = X[mask] @ R.T
            X = _nm(X + 0.004 * rng.randn(n, d))
            X = refine(X, 600, 28.0, 650.0, 0.015, 0.0007)
        elif typ == 3:
            # cap fault: rotate geographic cap
            c = rng.randint(n)
            dots = X[c] @ X.T
            k = int(rng.choice([10, 16, 24, 34]))
            idx = np.argsort(-dots)[:k]
            th = float(rng.uniform(0.08, 0.35))
            R = _plane_rot(rng, th)
            X[idx] = X[idx] @ R.T
            X = _nm(X + 0.004 * rng.randn(n, d))
            X = refine(X, 600, 28.0, 650.0, 0.015, 0.0007)
        elif typ == 4:
            # global projective stretch (coherent dozens-point move)
            Z = rng.randn(4, 4)
            S = (Z + Z.T) * 0.5
            S /= (np.linalg.norm(S) + 1e-12)
            sig = float(rng.uniform(0.05, 0.16))
            M = np.eye(4) + sig * S
            X = _nm(X @ M.T)
            X = refine(X, 550, 25.0, 600.0, 0.015, 0.0007)
        elif typ == 5:
            # contact-guided cap delete + void reinsert
            G = X @ X.T
            np.fill_diagonal(G, -2.0)
            m = float(np.max(G))
            deg = np.sum(G > m - 0.004, axis=1)
            k_move = int(rng.choice([8, 12, 18, 26]))
            c = int(np.argmax(deg + rng.randn(n) * 0.5))
            order = np.argsort(-(X[c] @ X.T))
            victims = order[:k_move]
            keep = np.ones(n, dtype=bool); keep[victims] = False
            Xk = X[keep]
            newp = []
            for si in range(k_move):
                Xr = Xk if si == 0 else np.vstack([Xk, np.array(newp)])
                c0 = void_pt(Xr, nsamp=9000)
                xn = refine_single(c0, Xr, steps=100, b0=45.0, b1=280.0)
                newp.append(xn)
            X[victims] = np.array(newp)
            X = _nm(X + 0.005 * rng.randn(n, d))
            X = refine(X, 600, 30.0, 750.0, 0.015, 0.0007)
        elif typ == 6:
            # splice two rotated copies along a plane (crossover fault)
            R = _rand_orth(rng)
            Y = base @ R.T
            u = rng.randn(4); u /= np.linalg.norm(u)
            thr = float(np.median(np.concatenate([X @ u, Y @ u])))
            mX = (X @ u) > thr
            mY = (Y @ u) > thr
            Ax = X[mX]; Ay = Y[~mY]
            pool = np.vstack([Ax, Ay])
            if len(pool) >= n:
                sel = rng.choice(len(pool), n, replace=False)
                X = pool[sel]
            else:
                need = n - len(pool)
                extra = rng.randn(need, 4)
                X = _nm(np.vstack([pool, extra]))
            X = _nm(X)
            X = refine(X, 650, 22.0, 550.0, 0.017, 0.0008)
        elif typ == 7:
            X = _nm(X + 0.02 * rng.randn(n, d))
            X = refine(X, 500, 80.0, 1500.0, 0.010, 0.0005, mu=0.85)
        elif typ == 8:
            # random void reinsert 1-3
            k_move = int(rng.choice([1, 2, 3]))
            victims = rng.choice(n, size=k_move, replace=False)
            keep = np.ones(n, dtype=bool); keep[victims] = False
            Xk = X[keep]
            newp = []
            for si in range(k_move):
                Xr = Xk if si == 0 else np.vstack([Xk, np.array(newp)])
                c0 = void_pt(Xr, nsamp=11000)
                xn = refine_single(c0, Xr, steps=120, b0=35.0, b1=300.0)
                newp.append(xn)
            X[victims] = np.array(newp)
            X = _nm(X + 0.004 * rng.randn(n, d))
            X = refine(X, 520, 30.0, 700.0, 0.014, 0.0006)
        elif typ == 9:
            # double fault: two successive plane splits
            for _ in range(2):
                u = rng.randn(4); u /= np.linalg.norm(u)
                pr = X @ u
                mask = pr > np.median(pr)
                R = _plane_rot(rng, float(rng.uniform(0.05, 0.22)))
                X[mask] = X[mask] @ R.T
            X = _nm(X + 0.004 * rng.randn(n, d))
            X = refine(X, 600, 26.0, 650.0, 0.015, 0.0007)
        elif typ == 10:
            X = _nm(X + 0.13 * rng.randn(n, d))
            X = refine(X, 650, 20.0, 500.0, 0.018, 0.0008)
        else:
            X = _nm(X + 0.26 * rng.randn(n, d))
            X = refine(X, 700, 15.0, 400.0, 0.020, 0.0009)
        v = _mc(X)
        if v < cur_val:
            x_cur, cur_val = X.copy(), v
        elif v < cur_val + 0.003 and rng.rand() < 0.12:
            x_cur, cur_val = X.copy(), v
        if trial % 25 == 0 and cur_val > best_val + 0.005:
            x_cur, cur_val = x_best.copy(), best_val
        if v < best_val + 0.0008 and time.time() < SEARCH_END - 3.0:
            do_lp = False
            if v < best_val:
                do_lp = True
            elif v < best_val + 0.00015 and (trial % 3 == 0):
                do_lp = True
            if do_lp:
                dl = min(time.time() + 1.3, SEARCH_END)
                try:
                    Xs, vs = _lp_deadline(X, dl)
                except Exception:
                    vs = v; Xs = X
                if vs < best_val:
                    x_best, best_val = Xs.copy(), vs
                    x_cur, cur_val = Xs.copy(), vs
                elif vs < cur_val:
                    x_cur, cur_val = Xs.copy(), vs
            else:
                if v < best_val:
                    x_best, best_val = X.copy(), v
                    x_cur, cur_val = X.copy(), v
        else:
            if v < best_val:
                x_best, best_val = X.copy(), v
                x_cur, cur_val = X.copy(), v
    try:
        x_best, best_val = _lp_deadline(x_best, HARD_END - 1.0)
    except Exception:
        pass
    return _nm(x_best)
