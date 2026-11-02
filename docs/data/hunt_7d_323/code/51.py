import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _worst_per_point(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return G.max(axis=1)

def _anneal(x, steps, lr0, a0, a1, noise0, rng):
    n, d = x.shape
    # local copy assumed
    for s in range(steps):
        frac = s / max(1, steps - 1)
        alpha = a0 + (a1 - a0) * frac
        lr = lr0 * (1.0 - 0.5 * frac)
        # gram
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        # weights, dense exp is cheap for 323
        # clip to avoid overflow (alpha*(G-m) <=0)
        W = np.exp(alpha * (G - m))
        np.fill_diagonal(W, 0.0)
        # sparsify tiny weights to reduce noise
        # threshold 1e-7
        # W[W < 1e-7] = 0.0  # optional, keep dense for speed
        sumW = W.sum(axis=1, keepdims=True) + 1e-12
        Wx = W @ x
        wG = (W * G).sum(axis=1, keepdims=True)
        grad = (Wx - wG * x) / sumW
        # step
        x = x - lr * grad
        if noise0 > 0:
            ns = noise0 * (1.0 - frac * 0.7)
            if ns > 1e-12:
                nz = rng.standard_normal(size=x.shape) * ns
                # make tangential
                dotn = (nz * x).sum(axis=1, keepdims=True)
                nz -= dotn * x
                x = x + nz
        # project
        xn = np.linalg.norm(x, axis=1, keepdims=True)
        xn[xn == 0] = 1.0
        x /= xn
    return x

def _greedy_reinsert(y, to_replace, rng, n_cand=20000):
    n, d = y.shape
    fixed_mask = np.ones(n, dtype=bool)
    fixed_mask[np.array(to_replace, dtype=int)] = False
    X_fixed = y[fixed_mask]
    for k in to_replace:
        cand = rng.standard_normal(size=(n_cand, d))
        cand /= np.linalg.norm(cand, axis=1, keepdims=True)
        # dots to fixed + already replaced
        # build current set
        # for speed use fixed + current y for replaced except k
        # simple: use all except k
        others = np.delete(y, k, axis=0)
        dots = cand @ others.T
        worst = dots.max(axis=1)
        bi = int(np.argmin(worst))
        y[k] = cand[bi]
    return y

def _lp_polish(x, seconds, init_radius=5e-3):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = float(init_radius)
    start = time.time()
    iuI, iuJ = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        if time.time() - start >= seconds:
            break
        G = x @ x.T
        c = G[iuI, iuJ]
        m = float(c.max())
        thresh = m - max(20.0 * radius, 1e-9)
        sel = c > thresh
        cnt = int(sel.sum())
        if cnt == 0:
            break
        if cnt > 3000:
            idx = np.where(sel)[0]
            cidx = c[idx]
            part = np.argpartition(cidx, -3000)[-3000:]
            sel = np.zeros_like(sel, dtype=bool)
            sel[idx[part]] = True
            cnt = 3000
        I = iuI[sel]; J = iuJ[sel]
        csel = c[sel]
        p = cnt
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.empty(p * (2 * d + 1), dtype=np.int64)
        vals = np.empty(p * (2 * d + 1), dtype=np.float64)
        Cb = cols.reshape(p, 2 * d + 1)
        Vb = vals.reshape(p, 2 * d + 1)
        ar = np.arange(d)
        Cb[:, :d] = I[:, None] * d + ar
        Cb[:, d:2 * d] = J[:, None] * d + ar
        Cb[:, 2 * d] = n * d
        Vb[:, :d] = x[J]
        Vb[:, d:2 * d] = x[I]
        Vb[:, 2 * d] = -1.0
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        yn = np.linalg.norm(y, axis=1, keepdims=True)
        yn[yn == 0] = 1.0
        y = y / yn
        v = _max_cos(y)
        if v < best:
            x = y; best = v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def solve():
    t0 = time.time()
    rng = np.random.default_rng(12345)
    x0 = None; b0 = np.inf
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn).astype(np.float64)
            if a.shape == (323, 7):
                a = a / np.linalg.norm(a, axis=1, keepdims=True)
                v = _max_cos(a)
                if v < b0:
                    b0 = v; x0 = a.copy()
        except Exception:
            pass
    if x0 is None:
        x0 = rng.standard_normal(size=(323, 7))
        x0 /= np.linalg.norm(x0, axis=1, keepdims=True)
        b0 = _max_cos(x0)
    x_best = x0.copy()
    best = float(b0)
    n, d = x_best.shape

    HOP_DEADLINE = 150.0  # leave time for final polish; total must be <180
    # adapt to actual start time; solve must finish in 180s
    it = 0
    lp_calls = 0
    # precompute for speed? nothing
    while True:
        el = time.time() - t0
        if el > HOP_DEADLINE:
            break
        remain = HOP_DEADLINE - el
        if remain < 1.0:
            break
        x = x_best.copy()
        mode = it % 7
        if mode == 0:
            sig = 0.015
            x = x + rng.standard_normal(size=x.shape) * sig
            x /= np.linalg.norm(x, axis=1, keepdims=True)
            steps, lr0, a0, a1, nz = 900, 0.04, 60.0, 260.0, 0.002
        elif mode == 1:
            w = _worst_per_point(x)
            k = 16
            idx = np.argpartition(w, -k)[-k:]
            x[idx] += rng.standard_normal(size=(k, d)) * 0.09
            x[idx] /= np.linalg.norm(x[idx], axis=1, keepdims=True)
            steps, lr0, a0, a1, nz = 1000, 0.04, 60.0, 280.0, 0.0015
        elif mode == 2:
            # coordinated cap: random direction, top cluster common shift
            gdir = rng.standard_normal(size=d); gdir /= np.linalg.norm(gdir)
            proj = x @ gdir
            K = int(rng.integers(10, 22))
            idx = np.argpartition(proj, -K)[-K:]
            common = rng.standard_normal(size=d); common /= np.linalg.norm(common)
            # make common roughly orthogonal to mean?
            mag = 0.10 + 0.05 * rng.random()
            x[idx] += common[None, :] * mag + rng.standard_normal(size=(K, d)) * 0.025
            x[idx] /= np.linalg.norm(x[idx], axis=1, keepdims=True)
            steps, lr0, a0, a1, nz = 1000, 0.045, 50.0, 260.0, 0.002
        elif mode == 3:
            # hub/rattler greedy reinsertion
            w = _worst_per_point(x)
            k = int(rng.integers(4, 7))
            idx = np.argpartition(w, -12)[-12:]
            # pick k among worst 12 randomly + maybe rattler via contacts approx
            pick = rng.choice(idx, size=k, replace=False).tolist()
            x = _greedy_reinsert(x, [int(v) for v in pick], rng, n_cand=20000)
            x += rng.standard_normal(size=x.shape) * 0.004
            x /= np.linalg.norm(x, axis=1, keepdims=True)
            steps, lr0, a0, a1, nz = 1100, 0.04, 60.0, 280.0, 0.001
        elif mode == 4:
            # void-directed: find empty direction, drag worst points toward it
            cand = rng.standard_normal(size=(8000, d))
            cand /= np.linalg.norm(cand, axis=1, keepdims=True)
            dots = cand @ x.T
            worst = dots.max(axis=1)
            vi = int(np.argmin(worst))
            vdir = cand[vi]
            w = _worst_per_point(x)
            K = 8
            idx = np.argpartition(w, -K)[-K:]
            for kk in idx:
                t = 0.25 + 0.15 * rng.random()
                nv = (1 - t) * x[int(kk)] + t * vdir + rng.standard_normal(size=d) * 0.02
                nv /= np.linalg.norm(nv)
                x[int(kk)] = nv
            x += rng.standard_normal(size=x.shape) * 0.003
            x /= np.linalg.norm(x, axis=1, keepdims=True)
            steps, lr0, a0, a1, nz = 1000, 0.04, 60.0, 280.0, 0.0015
        elif mode == 5:
            sig = 0.032
            x = x + rng.standard_normal(size=x.shape) * sig
            x /= np.linalg.norm(x, axis=1, keepdims=True)
            steps, lr0, a0, a1, nz = 900, 0.05, 50.0, 250.0, 0.0025
        else:
            # contact-balanced: kick two random caps in opposite directions to open channel
            g1 = rng.standard_normal(size=d); g1 /= np.linalg.norm(g1)
            proj = x @ g1
            K = 10
            top = np.argpartition(proj, -K)[-K:]
            bot = np.argpartition(proj, K)[:K]
            u = rng.standard_normal(size=d); u /= np.linalg.norm(u)
            x[top] += u[None, :] * 0.08 + rng.standard_normal(size=(K, d)) * 0.02
            x[bot] -= u[None, :] * 0.08 + rng.standard_normal(size=(K, d)) * 0.02
            x /= np.linalg.norm(x, axis=1, keepdims=True)
            steps, lr0, a0, a1, nz = 1000, 0.045, 55.0, 270.0, 0.002
        try:
            y = _anneal(x, steps=steps, lr0=lr0, a0=a0, a1=a1, noise0=nz, rng=rng)
        except Exception:
            it += 1
            continue
        v = _max_cos(y)
        if v < best - 1e-12:
            # promising new basin: short LP snap to verify true value if time allows
            el2 = time.time() - t0
            if el2 < 130 and lp_calls < 8:
                try:
                    ys = _lp_polish(y, seconds=2.2, init_radius=0.006)
                    vs = _max_cos(ys)
                    lp_calls += 1
                    if vs < best - 1e-12:
                        best = vs; x_best = ys.copy()
                    elif v < best:
                        # keep annealed if LP failed to improve but raw is better
                        # (LP value is authoritative)
                        if vs < v:
                            best = vs; x_best = ys.copy()
                        else:
                            best = v; x_best = y.copy()
                    else:
                        # LP did not beat best, but y raw beat best? use LP value
                        pass
                except Exception:
                    best = v; x_best = y.copy()
            else:
                best = v; x_best = y.copy()
        elif v < best + 0.0025 and lp_calls < 6 and (it % 9 == 4):
            # occasional polish of near-best basins to not miss hidden winners
            el2 = time.time() - t0
            if el2 < 125 and (time.time() - t0) < 135:
                try:
                    ys = _lp_polish(y, seconds=2.0, init_radius=0.005)
                    vs = _max_cos(ys)
                    lp_calls += 1
                    if vs < best - 1e-12:
                        best = vs; x_best = ys.copy()
                except Exception:
                    pass
        it += 1

    remain = 178.0 - (time.time() - t0)
    if remain > 2.5:
        try:
            x_best = _lp_polish(x_best, seconds=min(remain - 0.5, 24.0), init_radius=2e-3)
        except Exception:
            pass
    x_best = x_best / np.linalg.norm(x_best, axis=1, keepdims=True)
    return x_best
