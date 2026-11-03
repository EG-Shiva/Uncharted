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


def build_E8():
    pts = []
    # 112: two coords +-1, six zeros
    for i in range(8):
        for j in range(i + 1, 8):
            for si in (1.0, -1.0):
                for sj in (1.0, -1.0):
                    v = np.zeros(8)
                    v[i] = si
                    v[j] = sj
                    pts.append(v)
    # 128: all +-1/2 with even number of minus
    for mask in range(256):
        v = np.ones(8) * 0.5
        nm = 0
        for k in range(8):
            if (mask >> k) & 1:
                v[k] = -0.5
                nm += 1
        if nm % 2 == 0:
            pts.append(v.copy())
    E = np.array(pts, dtype=float)
    assert E.shape == (240, 8)
    E = E / np.linalg.norm(E, axis=1, keepdims=True)
    return E


def greedy_delete(E, rng, eps=1e-12):
    X = E.copy()
    # delete 78 -> 162
    cur = np.arange(X.shape[0])
    Xc = X
    for _ in range(240 - 162):
        n = Xc.shape[0]
        G = Xc @ Xc.T
        np.fill_diagonal(G, -2.0)
        rowmax = G.max(axis=1)
        gmax = float(rowmax.max())
        cand = np.where(rowmax >= gmax - 1e-9)[0]
        if len(cand) == 1:
            rm = cand[0]
        else:
            # contact count near own rowmax
            best_c = -1
            best_set = []
            for i in cand:
                c = int(np.sum(G[i] > rowmax[i] - 1e-7))
                if c > best_c:
                    best_c = c
                    best_set = [i]
                elif c == best_c:
                    best_set.append(i)
            if len(best_set) == 1:
                rm = best_set[0]
            else:
                # randomised tie break, slight preference to larger index spread
                # add tiny noise to break symmetry deterministically via rng
                rm = int(rng.choice(best_set))
        keep = [i for i in range(n) if i != rm]
        Xc = Xc[keep]
    return Xc


def anneal_stage(x0, taus, iters):
    n, d = x0.shape
    flat = (x0 / np.linalg.norm(x0, axis=1, keepdims=True)).ravel().copy()
    for tau, mi in zip(taus, iters):
        def fun(f, tau=tau):
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
                           options={"maxiter": mi, "ftol": 1e-14, "gtol": 1e-6, "maxfun": mi * 4})
            flat = res.x
        except Exception:
            break
    X = flat.reshape(n, d)
    X /= np.linalg.norm(X, axis=1, keepdims=True)
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
        else:
            cc = c[sel]
            p = len(I)
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


def solve():
    t0 = time.time()
    DEADLINE = t0 + 172.0
    rng = np.random.RandomState(12345)
    # load incumbent as safety
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

    E8 = build_E8()
    # random rotation of E8 to diversify (E8 symmetric, rotation gives same geometry but
    # different L-BFGS path after numerical noise; keep one unrotated too)
    starts = []
    # greedy runs with different tie-break seeds
    for gi in range(4):
        r2 = np.random.RandomState(1000 + gi)
        # small random rotation for diversity except first
        if gi == 0:
            Er = E8
        else:
            M = r2.randn(8, 8)
            Q, R = np.linalg.qr(M)
            s = np.sign(np.diag(R)); s[s == 0] = 1
            Q = Q * s[None, :]
            Er = (E8 @ Q.T)
            Er /= np.linalg.norm(Er, axis=1, keepdims=True)
        g = greedy_delete(Er, r2)
        starts.append(g)
        if time.time() > DEADLINE - 100:
            break
    # random 162-subsets of E8
    for ri in range(3):
        r2 = np.random.RandomState(5000 + ri)
        idx = r2.choice(240, size=162, replace=False)
        starts.append(E8[idx].copy())

    taus_full = [1.0 / 30.0, 1.0 / 100.0, 1.0 / 300.0, 1.0 / 1000.0]
    # shorter iters for many starts; longer for best later
    for si, s0 in enumerate(starts):
        if time.time() > DEADLINE - 60:
            break
        # jitter to break exact E8 symmetry
        s0 = s0 + 1e-4 * rng.randn(*s0.shape)
        s0 /= np.linalg.norm(s0, axis=1, keepdims=True)
        if si < 4:
            iters = [50, 50, 60, 60]
        else:
            iters = [40, 40, 50, 50]
        xopt = anneal_stage(s0, taus_full, iters)
        v = max_cos(xopt)
        if v < best_val:
            best_val = v
            x_best = xopt.copy()

    # hole repair / thin-swap if rattler or imbalance reappears
    n, d = 162, 8
    for rep in range(6):
        if time.time() > DEADLINE - 38:
            break
        m, cnt = contacts_cnt(x_best, thresh=2e-3)
        _, cnt_tight = contacts_cnt(x_best, thresh=1e-7)
        ratt = np.where(cnt == 0)[0]
        if len(ratt) == 0:
            ratt = np.where(cnt_tight <= 1)[0]
        if len(ratt) == 0:
            # shake overloaded cluster instead
            order = np.argsort(cnt)
            k = int(rng.randint(6, 12))
            idx_h = order[-k:]
            x0 = x_best.copy()
            x0[idx_h] += 0.05 * rng.randn(k, d)
            x0 += 0.003 * rng.randn(n, d)
            x0 /= np.linalg.norm(x0, axis=1, keepdims=True)
            xopt = anneal_stage(x0, [0.01, 0.0033, 0.001], [40, 40, 40])
            v = max_cos(xopt)
            if v < best_val:
                best_val = v
                x_best = xopt.copy()
            else:
                break
        else:
            ri = int(ratt[rng.randint(len(ratt))])
            others = np.delete(x_best, ri, axis=0)
            nh = best_hole(others, rng, nsamp=15000)
            x0 = x_best.copy()
            x0[ri] = nh
            # also nudge top overloaded neighbours
            order = np.argsort(cnt)
            k2 = min(5, n - 1)
            idx2 = order[-k2:]
            x0[idx2] += 0.03 * rng.randn(k2, d)
            x0 /= np.linalg.norm(x0, axis=1, keepdims=True)
            xopt = anneal_stage(x0, [0.01, 0.0033, 0.001], [50, 50, 60])
            v = max_cos(xopt)
            if v < best_val - 1e-12:
                best_val = v
                x_best = xopt.copy()
            else:
                # try greedy swap acceptance with tiny tolerance to change topology
                if v < best_val + 5e-5 and rep < 2:
                    # accept sideways to escape basin occasionally
                    x_best = xopt.copy()
                    best_val = v
                else:
                    # no progress
                    pass
                if rep >= 3:
                    break

    # final selection vs incumbent already done; LP polish remaining time
    remain = DEADLINE - time.time() - 4.0
    if remain > 5:
        x_best = lp_polish(x_best, seconds=min(remain, 30.0))
        x_best = x_best / np.linalg.norm(x_best, axis=1, keepdims=True)

    # safety: if somehow worse than loaded best, return loaded best polished? keep true best
    try:
        v_final = max_cos(x_best)
        if v_final > best_val + 1e-12:
            pass
        # ensure better than input; if E8 path failed, xb may be better
        vx = max_cos(xb)
        if vx < max_cos(x_best):
            x_best = xb.copy()
    except Exception:
        pass
    x_best = x_best / np.linalg.norm(x_best, axis=1, keepdims=True)
    return x_best
