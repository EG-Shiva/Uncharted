import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def _norm(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return x / n


def _max_cos(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))


def _lp_polish(x, deadline, r0=2e-3, cap=2200, maxit=200):
    x = _norm(x)
    n, d = x.shape
    best = _max_cos(x)
    best_x = x.copy()
    radius = r0
    iu = np.triu_indices(n, 1)
    it = 0
    while time.time() < deadline and radius > 1e-14 and it < maxit:
        it += 1
        G = x @ x.T
        c = G[iu]
        m = float(c.max())
        thresh = m - max(20.0 * radius, 1e-9)
        idx = np.where(c > thresh)[0]
        if idx.size == 0:
            radius = min(radius * 1.5, 0.05)
            continue
        if idx.size > cap:
            o = np.argpartition(c[idx], -cap)[-cap:]
            idx = idx[o]
        I = iu[0][idx]
        J = iu[1][idx]
        csel = c[idx]
        p = idx.size
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], axis=1).ravel()
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
            res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y = _norm(y)
        v = _max_cos(y)
        if v < best:
            x = y
            best = v
            best_x = y.copy()
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return best_x, best


def _smooth(x, steps, tau0, tau1, lr0, lr1, mom=0.85, fix_mask=None):
    # fix_mask: boolean array True=free, False=fixed ; if None all free
    n, d = x.shape
    vel = np.zeros_like(x)
    lt0, lt1 = np.log(tau0), np.log(tau1)
    ll0, ll1 = np.log(lr0), np.log(lr1)
    for t in range(steps):
        f = t / max(1, steps - 1)
        tau = float(np.exp(lt0 + (lt1 - lt0) * f))
        lr = float(np.exp(ll0 + (ll1 - ll0) * f))
        G = x @ x.T
        np.fill_diagonal(G, -np.inf)
        m = float(np.max(G))
        W = np.exp((G - m) / tau)
        np.fill_diagonal(W, 0.0)
        row = np.sum(W, axis=1, keepdims=True)
        row[row < 1e-12] = 1e-12
        grad = (W @ x) / row
        dot = np.sum(grad * x, axis=1, keepdims=True)
        grad = grad - dot * x
        gn = np.linalg.norm(grad, axis=1, keepdims=True)
        scale = np.ones_like(gn)
        big = gn > 2.0
        scale[big] = 2.0 / np.maximum(gn[big], 1e-12)
        grad = grad * scale
        vel = mom * vel + (1.0 - mom) * grad
        bc = 1.0 - mom ** (t + 1)
        step = vel / bc
        if fix_mask is not None:
            step = step * fix_mask[:, None]
        x = x - lr * step
        x = _norm(x)
    return x


def _cap_rotation(x, rng, k, theta):
    n, d = x.shape
    u = rng.standard_normal(d)
    u /= np.linalg.norm(u)
    dots = x @ u
    idx = np.argpartition(-dots, k)[:k]
    # random 2-plane
    A = rng.standard_normal((d, d))
    Q, _ = np.linalg.qr(A)
    if np.linalg.det(Q) < 0:
        Q[:, 0] = -Q[:, 0]
    a = Q[:, 0]
    b = Q[:, 1]
    c = np.cos(theta)
    s = np.sin(theta)
    R = np.eye(d) + (c - 1.0) * (np.outer(a, a) + np.outer(b, b)) + s * (np.outer(a, b) - np.outer(b, a))
    y = x.copy()
    y[idx] = (R @ x[idx].T).T
    return _norm(y)


def _field_move(x, rng, sigma):
    n, d = x.shape
    M = rng.standard_normal((d, d)) * sigma
    # remove pure antisymmetric (rotation does nothing) to keep effective part
    # keep symmetric + trace part
    V = M @ x.T  # 4 x n
    V = V.T  # n x 4
    dot = np.sum(V * x, axis=1, keepdims=True)
    Vt = V - dot * x
    y = x + Vt
    return _norm(y)


def _hole_reinsert(x, rng, k):
    n, d = x.shape
    seed = int(rng.integers(0, n))
    dots = x @ x[seed]
    knn = np.argpartition(-dots, k)[:k]
    fixed = np.ones(n, dtype=bool)
    fixed[knn] = False
    # find largest hole w.r.t. fixed points
    S = 2500
    cand = rng.standard_normal((S, d))
    cand /= np.linalg.norm(cand, axis=1, keepdims=True)
    Gf = cand @ x[~fixed].T if np.any(~fixed == False) else cand @ x.T
    # actually fixed points are ~fixed==True
    Gf = cand @ x[fixed].T
    worst = np.max(Gf, axis=1)
    hole = cand[int(np.argmin(worst))]
    y = x.copy()
    # place k points around hole with spread
    spread = 0.15 + 0.1 * float(rng.random())
    for ii, j in enumerate(knn):
        v = rng.standard_normal(d) * spread
        v = v - np.dot(v, hole) * hole
        p = hole + v
        p /= np.linalg.norm(p)
        y[j] = p
    return _norm(y), knn


def _subset_kick(x, rng, k, sigma):
    n, d = x.shape
    idx = rng.choice(n, size=k, replace=False)
    y = x.copy()
    noise = rng.standard_normal((k, d)) * sigma
    dot = np.sum(noise * y[idx], axis=1, keepdims=True)
    noise = noise - dot * y[idx]
    y[idx] = y[idx] + noise
    return _norm(y)


def solve():
    t0 = time.time()
    HARD_END = t0 + 172.0
    EXPLORE_END = t0 + 142.0
    n, d = 153, 4
    rng = np.random.default_rng(7)

    best_x = None
    best_v = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=float)
            if a.shape == (n, d):
                a = _norm(a)
                if np.all(np.isfinite(a)):
                    v = _max_cos(a)
                    if v < best_v:
                        best_v = v
                        best_x = a.copy()
        except Exception:
            pass
    if best_x is None:
        best_x = _norm(rng.standard_normal((n, d)))
        best_v = _max_cos(best_x)

    # quick initial polish to lock local optimum
    try:
        best_x, best_v = _lp_polish(best_x, min(HARD_END - 5, time.time() + 8.0), r0=1e-3)
    except Exception:
        pass

    trial = 0
    # parameter cycles
    while time.time() < EXPLORE_END:
        trial += 1
        mode = trial % 4
        try:
            if mode == 1:
                k = int(rng.integers(5, 13))
                theta = float(rng.uniform(0.03, 0.16))
                y = _cap_rotation(best_x, rng, k, theta)
                y = _smooth(y, 260, 0.012, 0.0025, 0.06, 0.012)
                y, v = _lp_polish(y, min(HARD_END - 2, time.time() + 3.2), r0=2e-3)
            elif mode == 2:
                k = int(rng.integers(5, 11))
                y, knn = _hole_reinsert(best_x, rng, k)
                # local relax of inserted points first
                free = np.zeros(n, dtype=float)
                free[knn] = 1.0
                # need boolean for fix inside smooth: use fix_mask float
                # run smooth with mask: build mask array
                mask = np.zeros(n, dtype=float)
                mask[knn] = 1.0
                # emulate masked smooth by temporary function
                # do full smooth but with two stages: masked then full
                yy = y.copy()
                # masked stage
                n_, d_ = yy.shape
                vel = np.zeros_like(yy)
                lt0, lt1 = np.log(0.008), np.log(0.002)
                ll0, ll1 = np.log(0.08), np.log(0.015)
                steps = 200
                for t in range(steps):
                    f = t / max(1, steps - 1)
                    tau = float(np.exp(lt0 + (lt1 - lt0) * f))
                    lr = float(np.exp(ll0 + (ll1 - ll0) * f))
                    G = yy @ yy.T
                    np.fill_diagonal(G, -np.inf)
                    m = float(np.max(G))
                    W = np.exp((G - m) / tau)
                    np.fill_diagonal(W, 0.0)
                    row = np.sum(W, axis=1, keepdims=True)
                    row[row < 1e-12] = 1e-12
                    grad = (W @ yy) / row
                    dot = np.sum(grad * yy, axis=1, keepdims=True)
                    grad = grad - dot * yy
                    vel = 0.85 * vel + 0.15 * grad
                    bc = 1.0 - 0.85 ** (t + 1)
                    step = vel / bc
                    step = step * mask[:, None]
                    yy = yy - lr * step
                    yy = _norm(yy)
                y = _smooth(yy, 150, 0.008, 0.002, 0.04, 0.01)
                y, v = _lp_polish(y, min(HARD_END - 2, time.time() + 3.2), r0=2e-3)
            elif mode == 3:
                sigma = float(rng.uniform(0.04, 0.16))
                y = _field_move(best_x, rng, sigma)
                # add small noise to break symmetry
                y = _subset_kick(y, rng, 4, 0.01)
                y = _smooth(y, 260, 0.012, 0.0025, 0.06, 0.012)
                y, v = _lp_polish(y, min(HARD_END - 2, time.time() + 3.2), r0=2e-3)
            else:
                k = int(rng.integers(8, 21))
                sigma = float(rng.uniform(0.05, 0.22))
                y = _subset_kick(best_x, rng, k, sigma)
                y = _smooth(y, 300, 0.015, 0.002, 0.07, 0.01)
                y, v = _lp_polish(y, min(HARD_END - 2, time.time() + 3.0), r0=1.5e-3)
        except Exception:
            continue
        if v < best_v - 1e-12:
            best_v = v
            best_x = y.copy()
            # intensify around new best immediately
            try:
                best_x, best_v = _lp_polish(best_x, min(HARD_END - 2, time.time() + 5.0), r0=8e-4)
            except Exception:
                pass
        # safety
        if time.time() >= HARD_END - 28:
            break

    # final thorough polish: alternate smooth + LP
    try:
        remain = HARD_END - time.time() - 1.0
        if remain > 5:
            # light anneal to settle second shell without leaving basin
            best_x = _smooth(best_x, 200, 0.005, 0.001, 0.02, 0.005)
            best_x, best_v = _lp_polish(best_x, time.time() + remain, r0=1e-3)
    except Exception:
        pass

    best_x = _norm(best_x)
    # tiny shrink guard (no-op, ensures finite)
    best_x = np.where(np.isfinite(best_x), best_x, 0.0)
    nr = np.linalg.norm(best_x, axis=1, keepdims=True)
    nr[nr == 0] = 1.0
    best_x = best_x / nr
    return best_x
