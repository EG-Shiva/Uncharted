import time
import numpy as np
from scipy.optimize import linprog, linear_sum_assignment
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _per_max(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return G.max(axis=1)

def _stress_vec(X, beta=200.0):
    G = X @ X.T
    np.fill_diagonal(G, -10.0)
    m = float(G.max())
    E = np.exp(beta * (G - m))
    np.fill_diagonal(E, 0.0)
    return E.sum(axis=1), m, G

def _e8_roots():
    d = 8
    pts = []
    for i in range(d):
        for j in range(i+1, d):
            for si in (1.0, -1.0):
                for sj in (1.0, -1.0):
                    v = np.zeros(d)
                    v[i] = si
                    v[j] = sj
                    pts.append(v)
    for mask in range(256):
        v = np.ones(d) * 0.5
        nminus = 0
        for k in range(d):
            if (mask >> k) & 1:
                v[k] = -0.5
                nminus += 1
        if nminus % 2 == 0:
            pts.append(v)
    A = np.array(pts, dtype=np.float64)
    A = A / np.sqrt(2.0)
    A /= np.linalg.norm(A, axis=1, keepdims=True)
    return A

def _balanced_quench(X, steps, b0, b1, lr0, lr1, lam=0.12, mu=0.85, t_lim=None):
    n, d = X.shape
    X = X.copy()
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    best_m = _max_cos(X)
    best = X.copy()
    V = np.zeros_like(X)
    one_mu = 1.0 - mu
    lb0, lb1 = np.log(b0), np.log(b1)
    ll0, ll1 = np.log(lr0), np.log(lr1)
    for s in range(steps):
        if t_lim is not None and (s & 127) == 0 and time.time() > t_lim:
            break
        frac = s / max(steps - 1, 1)
        beta = np.exp(lb0 + (lb1 - lb0) * frac)
        lr = np.exp(ll0 + (ll1 - ll0) * frac)
        G = X @ X.T
        np.fill_diagonal(G, -10.0)
        m = float(G.max())
        if m < best_m:
            best_m = m
            best = X.copy()
        E = np.exp(beta * (G - m))
        np.fill_diagonal(E, 0.0)
        Sg = E.sum() / max(n, 1) + 1e-30
        Grad_g = (E @ X) / Sg
        bl = beta * 0.25
        if bl < 8.0:
            bl = 8.0
        mx = G.max(axis=1, keepdims=True)
        W = np.exp(bl * (G - mx))
        np.fill_diagonal(W, 0.0)
        Rs = W.sum(axis=1, keepdims=True) + 1e-30
        Grad_l = (W @ X) / Rs
        Grad = Grad_g + lam * Grad_l
        dot = (Grad * X).sum(axis=1, keepdims=True)
        Gt = Grad - dot * X
        nn = np.sqrt((Gt * Gt).sum(axis=1, keepdims=True)) + 1e-12
        scale = np.minimum(1.0, 3.0 / (nn + 1e-12))
        Gt = Gt * scale
        V = mu * V + one_mu * Gt
        X = X - lr * V
        X /= np.linalg.norm(X, axis=1, keepdims=True)
    return best

def _hard_polish(X, steps=150, lr0=0.003, lr1=0.00012, tol0=3e-4, tol1=2e-6):
    X = X.copy()
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    for s in range(steps):
        frac = s / max(steps - 1, 1)
        tol = tol0 + (tol1 - tol0) * frac
        lr = lr0 * (lr1 / lr0) ** frac if lr0 > 0 and lr1 > 0 else lr0
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        M = (G > (m - tol)).astype(np.float64)
        np.fill_diagonal(M, 0.0)
        cnt = M.sum(axis=1, keepdims=True) + 1e-12
        active = (cnt[:, 0] > 0.5)
        if not np.any(active):
            continue
        Grad = (M @ X) / cnt
        dot = (Grad * X).sum(axis=1, keepdims=True)
        Gt = Grad - dot * X
        nn = np.sqrt((Gt * Gt).sum(axis=1, keepdims=True)) + 1e-12
        step = np.zeros_like(Gt)
        step[active] = Gt[active] / nn[active]
        X = X - lr * step
        X /= np.linalg.norm(X, axis=1, keepdims=True)
    return X

def _refine_hole(Y, p, iters=150, beta=90.0, lr0=0.12, lr1=0.004):
    p = p / (np.linalg.norm(p) + 1e-30)
    for it in range(iters):
        frac = it / max(iters - 1, 1)
        lr = lr0 * (lr1 / lr0) ** frac
        g = Y @ p
        mx = g.max()
        w = np.exp(beta * (g - mx))
        s = w.sum() + 1e-12
        A = (w @ Y) / s
        At = A - float(A @ p) * p
        p = p - lr * At
        nr = np.linalg.norm(p)
        if nr == 0:
            break
        p /= nr
    return p

def _find_holes(Y, k, rng, beta=90.0, n_probe=8000):
    d = Y.shape[1]
    P = rng.randn(n_probe, d)
    P /= np.linalg.norm(P, axis=1, keepdims=True)
    CH = 2000
    bs = np.empty(n_probe)
    for a in range(0, n_probe, CH):
        b = min(a + CH, n_probe)
        D = P[a:b] @ Y.T
        bs[a:b] = D.max(axis=1)
    order = np.argsort(bs)
    holes = []
    for idx in order:
        p = P[idx]
        ok = True
        for q in holes:
            if abs(float(p @ q)) > 0.88:
                ok = False
                break
        if ok:
            Yext = Y if len(holes) == 0 else np.vstack([Y, np.array(holes)])
            pr = _refine_hole(Yext, p.copy(), iters=160, beta=beta + rng.rand() * 40.0)
            holes.append(pr)
            if len(holes) >= k:
                break
    j = 0
    while len(holes) < k:
        p = P[order[j % len(order)]].copy()
        j += 1
        Yext = Y if len(holes) == 0 else np.vstack([Y, np.array(holes)])
        holes.append(_refine_hole(Yext, p, iters=140, beta=beta))
    return holes

def _lp_polish(x, seconds, r0=1.2e-3):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = r0
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        c = (x @ x.T)[iu]
        m = float(c.max())
        sel = c > m - max(20 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        if p == 0:
            radius *= 0.5
            continue
        if p > 3000:
            idx = np.argsort(c[sel])[-3000:]
            I, J = I[idx], J[idx]
            cc = c[sel][idx]
            p = 3000
        else:
            cc = c[sel]
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
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
        v = _max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def _procrustes_child(A, B, rng, sigma=0.015):
    n, d = A.shape
    try:
        C = -(A @ B.T)
        ri, ci = linear_sum_assignment(C)
        Bp = np.empty_like(B)
        Bp[ri] = B[ci]
        M = Bp.T @ A
        U, _, Vt = np.linalg.svd(M, full_matrices=False)
        Q = U @ Vt
        Bal = Bp @ Q
        mask = rng.rand(n) < 0.5
        Ch = A.copy()
        Ch[mask] = Bal[mask]
        # average with counterpart for smoothness + noise
        Ch = 0.5 * (A + Bal)
        # random pick perturbation alternative
        Ch = Ch + rng.randn(n, d) * sigma
        Ch /= np.linalg.norm(Ch, axis=1, keepdims=True)
        return Ch
    except Exception:
        return None

def solve():
    t0 = time.time()
    BUDGET = 174.0
    t_explore = t0 + 118.0
    n, d = 168, 8
    rng = np.random.RandomState(7)
    best = None
    best_val = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn).astype(np.float64)
            if a.shape != (n, d):
                continue
            a = a / np.linalg.norm(a, axis=1, keepdims=True)
            if not np.all(np.isfinite(a)):
                continue
            va = _max_cos(a)
            if va < best_val:
                best_val = va
                best = a.copy()
        except Exception:
            pass
    if best is None:
        best = rng.randn(n, d)
        best /= np.linalg.norm(best, axis=1, keepdims=True)
        best_val = _max_cos(best)
    E8 = _e8_roots()
    elite = [(best_val, best.copy())]
    def push(v, X):
        nonlocal best_val, best
        elite.append((v, X.copy()))
        elite.sort(key=lambda e: e[0])
        del elite[5:]
        if v < best_val - 1e-15:
            best_val = v
            best = X.copy()
    it = 0
    while time.time() < t_explore:
        if time.time() + 2.5 > t0 + BUDGET:
            break
        it += 1
        r = rng.rand()
        cur = elite[0][1]
        try:
            if r < 0.32:
                X = cur.copy()
                st, mcur, _ = _stress_vec(X, beta=200.0)
                imax = _per_max(X)
                k = 6 + int(rng.randint(9))
                order_s = np.argsort(st)
                order_w = np.argsort(imax)
                victims = set()
                for _ in range(k):
                    u = rng.rand()
                    cand_pool = None
                    if u < 0.55:
                        cand_pool = order_s[::-1]
                    elif u < 0.85:
                        cand_pool = order_w[::-1]
                    else:
                        vv = int(rng.randint(n))
                        while vv in victims:
                            vv = int(rng.randint(n))
                        victims.add(vv)
                        continue
                    for idx in cand_pool:
                        if int(idx) not in victims:
                            victims.add(int(idx))
                            break
                victims = sorted(victims)
                keep = np.ones(n, bool)
                keep[victims] = False
                Y = X[keep]
                holes = _find_holes(Y, len(victims), rng, beta=70.0 + rng.rand() * 60.0)
                for vi, h in zip(victims, holes):
                    X[vi] = h
                X = X + rng.randn(n, d) * 0.0035
                X /= np.linalg.norm(X, axis=1, keepdims=True)
                b0 = 25.0 + rng.rand() * 25.0
                b1 = 280.0 + rng.rand() * 340.0
                lr0 = 0.004 + rng.rand() * 0.004
                steps = 850 + int(rng.randint(500))
                lam = 0.03 + rng.rand() * 0.09
                X = _balanced_quench(X, steps, b0, b1, lr0, lr0 * 0.12, lam, t_lim=t_explore)
                X = _hard_polish(X, steps=120, lr0=0.002, lr1=0.0002, tol0=3e-4, tol1=2e-6)
                v = _max_cos(X)
                push(v, X)
            elif r < 0.50:
                X = cur.copy()
                mcur = _max_cos(X)
                imax = _per_max(X)
                tol_r = 1.5e-4 + rng.rand() * 2.5e-4
                ratt = np.where(imax < mcur - tol_r)[0]
                if len(ratt) < 4 or len(ratt) > 70:
                    # fallback to heavy kick
                    st, _, _ = _stress_vec(X, beta=200.0)
                    worst = np.argsort(st)[-24:]
                    X[worst] += rng.randn(len(worst), d) * (0.07 + rng.rand() * 0.08)
                    X /= np.linalg.norm(X, axis=1, keepdims=True)
                    X = _balanced_quench(X, 900 + int(rng.randint(400)), 30.0 + rng.rand() * 20.0, 260.0 + rng.rand() * 300.0, 0.005, 0.0005, 0.05 + rng.rand() * 0.06, t_lim=t_explore)
                    X = _hard_polish(X, steps=120)
                    v = _max_cos(X)
                    push(v, X)
                    continue
                keep = np.ones(n, bool)
                keep[ratt] = False
                Y = X[keep]
                holes = _find_holes(Y, len(ratt), rng, beta=80.0 + rng.rand() * 50.0)
                for vi, h in zip(ratt, holes):
                    X[vi] = h
                X = X + rng.randn(n, d) * 0.0025
                X /= np.linalg.norm(X, axis=1, keepdims=True)
                X = _balanced_quench(X, 900 + int(rng.randint(400)), 25.0 + rng.rand() * 25.0, 300.0 + rng.rand() * 300.0, 0.005, 0.0005, 0.06 + rng.rand() * 0.07, t_lim=t_explore)
                X = _hard_polish(X, steps=130)
                v = _max_cos(X)
                push(v, X)
            elif r < 0.62:
                idx = rng.choice(240, n, replace=False)
                X = E8[idx].copy()
                sigma = 0.008 + rng.rand() * 0.012
                X = X + rng.randn(n, d) * sigma
                X /= np.linalg.norm(X, axis=1, keepdims=True)
                b0 = 20.0 + rng.rand() * 20.0
                b1 = 300.0 + rng.rand() * 300.0
                X = _balanced_quench(X, 1000 + int(rng.randint(500)), b0, b1, 0.005, 0.0005, 0.04 + rng.rand() * 0.08, t_lim=t_explore)
                X = _hard_polish(X, steps=150, lr0=0.0025, lr1=0.0002, tol0=4e-4, tol1=2e-6)
                v = _max_cos(X)
                push(v, X)
            elif r < 0.72:
                X = rng.randn(n, d)
                X /= np.linalg.norm(X, axis=1, keepdims=True)
                X = _balanced_quench(X, 700, 15.0, 120.0, 0.006, 0.001, 0.08, t_lim=t_explore)
                if time.time() > t_explore:
                    continue
                X = _balanced_quench(X, 900, 80.0, 500.0 + rng.rand() * 200.0, 0.005, 0.0004, 0.04, t_lim=t_explore)
                X = _hard_polish(X, steps=120)
                v = _max_cos(X)
                push(v, X)
            elif r < 0.86:
                X = cur.copy()
                st, _, _ = _stress_vec(X, beta=200.0)
                nworst = 20 + int(rng.randint(12))
                worst = np.argsort(st)[-nworst:]
                sigma_big = 0.07 + rng.rand() * 0.09
                X[worst] += rng.randn(len(worst), d) * sigma_big
                mask = np.ones(n, bool)
                mask[worst] = False
                X[mask] += rng.randn(mask.sum(), d) * 0.004
                X /= np.linalg.norm(X, axis=1, keepdims=True)
                b0 = 30.0 + rng.rand() * 20.0
                b1 = 260.0 + rng.rand() * 320.0
                steps = 900 + int(rng.randint(500))
                lam = 0.02 + rng.rand() * 0.07
                X = _balanced_quench(X, steps, b0, b1, 0.005, 0.0005, lam, t_lim=t_explore)
                X = _hard_polish(X, steps=120)
                v = _max_cos(X)
                push(v, X)
            else:
                if len(elite) < 2:
                    X = cur.copy()
                    X = X + rng.randn(n, d) * 0.01
                    X /= np.linalg.norm(X, axis=1, keepdims=True)
                    X = _balanced_quench(X, 900, 30.0, 400.0, 0.005, 0.0005, 0.06, t_lim=t_explore)
                    v = _max_cos(X)
                    push(v, X)
                    continue
                a_idx = int(rng.randint(len(elite)))
                b_idx = int(rng.randint(len(elite)))
                if a_idx == b_idx:
                    b_idx = (b_idx + 1) % len(elite)
                A = elite[a_idx][1]
                B = elite[b_idx][1]
                Ch = _procrustes_child(A, B, rng, sigma=0.012 + rng.rand() * 0.012)
                if Ch is None:
                    continue
                Ch = _balanced_quench(Ch, 900 + int(rng.randint(400)), 30.0, 350.0 + rng.rand() * 250.0, 0.005, 0.0005, 0.05 + rng.rand() * 0.06, t_lim=t_explore)
                Ch = _hard_polish(Ch, steps=120)
                v = _max_cos(Ch)
                push(v, Ch)
        except Exception:
            continue
    # final intensification
    Xb = elite[0][1].copy()
    Xb = _balanced_quench(Xb, 1200, 80.0, 700.0, 0.004, 0.00035, 0.05, t_lim=t0 + BUDGET - 32.0)
    Xb = _hard_polish(Xb, steps=220, lr0=0.0022, lr1=0.0001, tol0=2.5e-4, tol1=1e-6)
    vb = _max_cos(Xb)
    push(vb, Xb)
    time_left = (t0 + BUDGET - 2.0) - time.time()
    if time_left > 5.0:
        Xb = elite[0][1].copy()
        Xb = _lp_polish(Xb, seconds=min(time_left * 0.7, 30.0))
        vb = _max_cos(Xb)
        push(vb, Xb)
        time_left2 = (t0 + BUDGET - 1.0) - time.time()
        if time_left2 > 4.0:
            # tiny kicks + LP to squeeze plateau
            cur = elite[0][1].copy()
            for _ in range(4):
                if time.time() > t0 + BUDGET - 1.5:
                    break
                Y = cur + rng.randn(n, d) * 2e-4
                Y /= np.linalg.norm(Y, axis=1, keepdims=True)
                Y = _lp_polish(Y, seconds=min(4.0, (t0 + BUDGET - 1.0 - time.time()) / 2.0))
                vy = _max_cos(Y)
                if vy < _max_cos(cur):
                    cur = Y
            push(_max_cos(cur), cur)
    Xf = elite[0][1].copy()
    Xf /= np.linalg.norm(Xf, axis=1, keepdims=True)
    return Xf
