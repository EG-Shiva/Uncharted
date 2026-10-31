import time
import numpy as np

def _norm(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return x / n

def _max_cos(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def _lp_polish(x, deadline, r0=0.003, cap=2200, maxit=120):
    from scipy.optimize import linprog
    from scipy.sparse import coo_matrix
    x = _norm(x.copy())
    n, d = x.shape
    try:
        best = _max_cos(x)
    except Exception:
        return x, 1e9
    best_x = x.copy()
    iu = np.triu_indices(n, 1)
    radius = float(r0)
    it = 0
    while time.time() < deadline and radius > 1e-14 and it < maxit:
        it += 1
        G = x @ x.T
        c = G[iu]
        m = float(np.max(c))
        thr = m - max(20.0 * radius, 1e-9)
        idx = np.where(c > thr)[0]
        if idx.size == 0:
            radius = min(radius * 1.5, 0.05)
            continue
        if idx.size > cap:
            part = np.argpartition(c[idx], -cap)[-cap:]
            idx = idx[part]
        I = iu[0][idx]
        J = iu[1][idx]
        csel = c[idx]
        p = idx.size
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cI = I[:, None] * d + np.arange(d)
        cJ = J[:, None] * d + np.arange(d)
        cols = np.concatenate([cI, cJ, np.full((p, 1), n * d)], axis=1).ravel()
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
        try:
            dx = res.x[:-1].reshape(n, d)
        except Exception:
            radius *= 0.5
            continue
        y = _norm(x + dx)
        try:
            v = _max_cos(y)
        except Exception:
            radius *= 0.5
            continue
        if v < best:
            x = y
            best = v
            best_x = y.copy()
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return best_x, best

def _smooth(x, steps, tau0, tau1, lr0, lr1, mom=0.85):
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
        big = gn > 2.0
        scale = np.ones_like(gn)
        scale[big] = 2.0 / np.maximum(gn[big], 1e-12)
        grad = grad * scale
        vel = mom * vel + (1.0 - mom) * grad
        bc = 1.0 - mom ** (t + 1)
        step = vel / bc
        x = x - lr * step
        x = _norm(x)
    return x

def _find_hole(Xfix, rng, S=4000):
    d = Xfix.shape[1]
    cand = rng.standard_normal((S, d))
    cand /= np.linalg.norm(cand, axis=1, keepdims=True)
    Gf = cand @ Xfix.T
    worst = np.max(Gf, axis=1)
    i = int(np.argmin(worst))
    return cand[i]

def _hole_reinsert_move(x, rng, k):
    n, d = x.shape
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    m = float(np.max(G))
    B = G > (m - 1e-9)
    deg = np.sum(B, axis=1)
    seed = int(np.argmax(deg + rng.random(n) * 0.5))
    dots = x @ x[seed]
    knn = np.argpartition(-dots, k)[:k]
    keep = np.ones(n, dtype=bool)
    keep[knn] = False
    Xfix = x[keep]
    newpts = []
    cur = Xfix
    for _ in range(k):
        h = _find_hole(cur, rng, S=4000)
        h = h + 0.001 * rng.standard_normal(d)
        h /= np.linalg.norm(h)
        newpts.append(h)
        cur = np.vstack([cur, h[None, :]])
    y = x.copy()
    y[knn] = np.array(newpts)
    return _norm(y)

def _subset_randomize(x, rng, k):
    n, d = x.shape
    y = x.copy()
    idx = rng.choice(n, size=k, replace=False)
    r = rng.standard_normal((k, d))
    r /= np.linalg.norm(r, axis=1, keepdims=True)
    y[idx] = r
    return _norm(y)

def _gauss_kick(x, rng, sigma):
    y = x + sigma * rng.standard_normal(x.shape)
    return _norm(y)

def _cap_rotation(x, rng, k, theta):
    n, d = x.shape
    u = rng.standard_normal(d)
    u /= np.linalg.norm(u)
    dots = x @ u
    idx = np.argpartition(-dots, k)[:k]
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
    V = (M @ x.T).T
    dot = np.sum(V * x, axis=1, keepdims=True)
    Vt = V - dot * x
    y = x + Vt
    return _norm(y)

def solve():
    t0 = time.time()
    HARD_END = t0 + 172.0
    SEARCH_END = t0 + 140.0
    n, d = 153, 4
    rng = np.random.default_rng(12345)
    best_x = None
    best_v = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=np.float64)
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
    global_x = best_x.copy()
    global_v = best_v
    try:
        dl = min(HARD_END - 5.0, t0 + 12.0)
        if dl > time.time():
            global_x, global_v = _lp_polish(global_x, dl, r0=0.003, maxit=150)
    except Exception:
        pass
    base_x = global_x.copy()
    base_v = global_v
    trial = 0
    while time.time() < SEARCH_END:
        trial += 1
        if time.time() > HARD_END - 20.0:
            break
        mode = trial % 5
        try:
            if mode == 0:
                k = int(rng.integers(3, 9))
                y = _hole_reinsert_move(base_x, rng, k)
                y = _smooth(y, steps=250, tau0=0.02, tau1=0.003, lr0=0.05, lr1=0.005)
            elif mode == 1:
                k = int(rng.integers(5, 13))
                y = _subset_randomize(base_x, rng, k)
                y = _smooth(y, steps=350, tau0=0.03, tau1=0.002, lr0=0.08, lr1=0.005)
            elif mode == 2:
                sigma = float(rng.uniform(0.015, 0.07))
                y = _gauss_kick(base_x, rng, sigma)
                y = _smooth(y, steps=300, tau0=0.025, tau1=0.0025, lr0=0.06, lr1=0.005)
            elif mode == 3:
                k = int(rng.integers(10, 26))
                th = float(rng.uniform(0.04, 0.22))
                y = _cap_rotation(base_x, rng, k, th)
                y = _smooth(y, steps=250, tau0=0.02, tau1=0.003, lr0=0.05, lr1=0.005)
            else:
                sigma = float(rng.uniform(0.05, 0.25))
                y = _field_move(base_x, rng, sigma)
                y = _smooth(y, steps=300, tau0=0.025, tau1=0.0025, lr0=0.06, lr1=0.005)
        except Exception:
            continue
        try:
            v0 = _max_cos(y)
        except Exception:
            continue
        if v0 > base_v + 0.03:
            continue
        try:
            dl2 = min(SEARCH_END, time.time() + 2.5)
            if dl2 >= HARD_END - 15.0:
                break
            yp, vp = _lp_polish(y, dl2, r0=0.003, maxit=60)
        except Exception:
            continue
        if vp < global_v - 1e-12:
            global_x = yp.copy()
            global_v = vp
            base_x = yp.copy()
            base_v = vp
        elif vp < base_v + 0.0005 and (trial % 17 == 0):
            # occasional sideways move to different basin for diversity
            # only if clearly different (avoid returning)
            try:
                G1 = yp @ yp.T
                np.fill_diagonal(G1, -2.0)
                m1 = float(np.max(G1))
                G0 = base_x @ base_x.T
                np.fill_diagonal(G0, -2.0)
                # accept as new base if not much worse to explore from there briefly
                # but keep global best separate
                if vp < base_v - 1e-12:
                    base_x = yp.copy()
                    base_v = vp
                elif rng.random() < 0.1:
                    base_x = yp.copy()
                    base_v = vp
            except Exception:
                pass
        # if base stagnates, occasionally restart from global best with big kick
        if trial % 25 == 0:
            base_x = global_x.copy()
            base_v = global_v
    try:
        rem = HARD_END - time.time() - 1.0
        if rem > 3.0:
            global_x, global_v = _lp_polish(global_x, time.time() + min(rem, 28.0), r0=0.002, maxit=180)
    except Exception:
        pass
    try:
        global_x = _norm(global_x)
    except Exception:
        pass
    return global_x
