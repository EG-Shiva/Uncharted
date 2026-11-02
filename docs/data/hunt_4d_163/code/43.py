import time
import numpy as np
from scipy.optimize import linprog, minimize
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _contacts(x, thr=1e-9):
    n = x.shape[0]
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    m = float(G.max())
    C = G >= m - thr
    deg = C.sum(axis=1)
    iu0, iu1 = np.triu_indices(n, 1)
    c = G[iu0, iu1]
    return m, deg, iu0, iu1, c

def _small_rot(rng, eps):
    S = rng.randn(4, 4) * eps
    S = (S - S.T) * 0.5
    M = np.eye(4) + S
    Q, R = np.linalg.qr(M)
    d = np.diag(R)
    Q = Q * np.sign(d + 1e-12)
    if np.linalg.det(Q) < 0:
        Q[:, 0] = -Q[:, 0]
    return Q

def _rand_dirs(rng, m):
    X = rng.randn(m, 4)
    X /= np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
    return X

def _find_hole(rng, keep, n_cand=4000):
    if keep.shape[0] == 0:
        v = rng.randn(4)
        return v / np.linalg.norm(v)
    cand = _rand_dirs(rng, n_cand)
    D = cand @ keep.T
    mx = D.max(axis=1)
    j = int(np.argmin(mx))
    return cand[j]

def _hub_to_hole(x, rng, n_move=2, thr=1e-9, n_cand=4000):
    n = x.shape[0]
    y = x.copy()
    G = y @ y.T
    np.fill_diagonal(G, -2.0)
    m = float(G.max())
    deg = (G >= m - thr).sum(axis=1)
    order = np.argsort(-deg)
    hubs = order[:max(1, n_move)].copy()
    # remove hubs one by one, place at holes w.r.t remainder
    for h in hubs:
        mask = np.ones(n, dtype=bool)
        mask[h] = False
        keep = y[mask]
        hole = _find_hole(rng, keep, n_cand=n_cand)
        # small jitter
        hole = hole + rng.randn(4) * 0.005
        hole /= max(np.linalg.norm(hole), 1e-12)
        y[h] = hole
    return y

def _subset_reinit(x, rng, k=8, n_cand=2000):
    n = x.shape[0]
    y = x.copy()
    idx = rng.choice(n, size=min(k, n), replace=False)
    # replace sequentially with holes
    for h in idx:
        mask = np.ones(n, dtype=bool)
        mask[h] = False
        keep = y[mask]
        hole = _find_hole(rng, keep, n_cand=n_cand)
        hole = hole + rng.randn(4) * 0.01
        hole /= max(np.linalg.norm(hole), 1e-12)
        y[h] = hole
    return y

def _cap_rotate(x, rng):
    n = x.shape[0]
    y = x.copy()
    c = _rand_dirs(rng, 1)[0]
    cos_thr = float(np.cos(rng.uniform(0.45, 1.0)))
    dots = y @ c
    mask = dots >= cos_thr
    cnt = int(mask.sum())
    if cnt < 2 or cnt > n - 2:
        # fallback: pick random subset around a point
        ci = int(rng.randint(n))
        d2 = y @ y[ci]
        thr = float(np.sort(d2)[max(0, n - 1 - int(rng.randint(8, 30)))])
        mask = d2 >= thr
        cnt = int(mask.sum())
        if cnt < 2:
            return y
    eps = float(rng.uniform(0.02, 0.22))
    Q = _small_rot(rng, eps)
    y[mask] = (Q @ y[mask].T).T
    y /= np.maximum(np.linalg.norm(y, axis=1, keepdims=True), 1e-12)
    return y

def _gd_softmax(x0, steps, T0, T1, lr0, mu, rng, noise0=0.0, deadline=None):
    x = x0.copy()
    n, d = x.shape
    v = np.zeros_like(x)
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    best_val = float(G.max())
    best = x.copy()
    if T0 <= 0:
        T0 = 1e-12
    if T1 <= 0:
        T1 = 1e-12
    lratio = np.log(T1 / T0)
    for k in range(steps):
        if deadline is not None and (k & 127) == 0 and time.time() > deadline:
            break
        frac = k / max(1, steps - 1)
        T = T0 * np.exp(lratio * frac)
        if T < 5e-5:
            T = 5e-5
        lr = lr0 * (1.0 - 0.55 * frac)
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        E = np.exp((G - m) / T)
        np.fill_diagonal(E, 0.0)
        S = float(E.sum() * 0.5)
        if S < 1e-300 or not np.isfinite(S):
            S = 1e-300
        grad = (E @ x) / S
        grad -= (grad * x).sum(axis=1, keepdims=True) * x
        v = mu * v + (1.0 - mu) * grad
        x = x - lr * v
        if noise0 > 0:
            amp = noise0 * (1.0 - frac)
            if amp > 0:
                nz = rng.randn(n, d) * amp
                nz -= (nz * x).sum(axis=1, keepdims=True) * x
                x = x + nz
        x /= np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
        if m < best_val:
            best_val = m
            best = x.copy()
    return best, best_val

def _lbfgs_smooth(x_init, deadline, taus=(0.002, 0.0006), iters=(70, 70)):
    x = x_init / np.maximum(np.linalg.norm(x_init, axis=1, keepdims=True), 1e-12)
    n, d = x.shape
    iu0, iu1 = np.triu_indices(n, 1)
    Np = len(iu0)
    y = x.ravel().copy()
    for tau, maxiter in zip(taus, iters):
        if time.time() > deadline - 0.4:
            break
        def fg(yflat, _tau=tau):
            Y = yflat.reshape(n, d)
            rn = np.maximum(np.linalg.norm(Y, axis=1, keepdims=True), 1e-12)
            X = Y / rn
            g = (X @ X.T)[iu0, iu1]
            m = float(g.max())
            e = np.exp((g - m) / _tau)
            s = float(e.sum())
            if s <= 0 or not np.isfinite(s):
                return float(m), np.zeros_like(yflat)
            L = m + _tau * np.log(s / Np)
            w = e / s
            W = np.zeros((n, n))
            W[iu0, iu1] = w
            W = W + W.T
            GX = W @ X
            GY = (GX - (GX * X).sum(axis=1, keepdims=True) * X) / rn
            return float(L), GY.ravel()
        try:
            res = minimize(fg, y, method="L-BFGS-B", jac=True,
                           options={"maxiter": int(maxiter), "maxfun": int(maxiter * 4),
                                    "ftol": 1e-14, "gtol": 1e-9})
            y = res.x
            Yt = y.reshape(n, d)
            nt = np.maximum(np.linalg.norm(Yt, axis=1, keepdims=True), 1e-12)
            y = (Yt / nt).ravel()
        except Exception:
            break
    Xf = y.reshape(n, d)
    Xf /= np.maximum(np.linalg.norm(Xf, axis=1, keepdims=True), 1e-12)
    return Xf

def _lp_polish(x, deadline, r0=0.002, cap=1300):
    x = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
    n, d = x.shape
    try:
        best = _max_cos(x)
    except Exception:
        return x, 1.0
    radius = r0
    iu0, iu1 = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        try:
            G = x @ x.T
            c = G[iu0, iu1]
            m = float(c.max())
            sel = c > m - max(20.0 * radius, 1e-9)
            if np.count_nonzero(sel) == 0:
                sel[int(np.argmax(c))] = True
            if int(sel.sum()) > cap:
                idx = np.argsort(-c)[:cap]
                s2 = np.zeros_like(sel)
                s2[idx] = True
                sel = s2
            I = iu0[sel]; J = iu1[sel]
            csel = c[sel]
            p = len(I)
            nv = n * d + 1
            rows = np.repeat(np.arange(p), 2 * d + 1)
            tmpc = np.empty((p, 2 * d + 1), dtype=int)
            ar = np.arange(d)
            tmpc[:, :d] = I[:, None] * d + ar
            tmpc[:, d:2 * d] = J[:, None] * d + ar
            tmpc[:, 2 * d] = n * d
            cols = tmpc.ravel()
            cc = np.empty((p, 2 * d + 1), dtype=float)
            cc[:, :d] = x[J]
            cc[:, d:2 * d] = x[I]
            cc[:, 2 * d] = -1.0
            vals = cc.ravel()
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv); obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
            if res.status != 0:
                radius *= 0.5
                continue
            y = x + res.x[:-1].reshape(n, d)
            y /= np.maximum(np.linalg.norm(y, axis=1, keepdims=True), 1e-12)
            vv = _max_cos(y)
            if vv < best:
                x, best = y, vv
                radius = min(radius * 1.5, 0.05)
            else:
                radius *= 0.5
        except Exception:
            radius *= 0.5
            continue
    return x, best

def solve():
    t0 = time.time()
    DEADLINE = t0 + 172.0
    FINAL_START = DEADLINE - 12.0
    rng = np.random.RandomState(9176342)
    n = 163
    best_x = None
    best_val = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=np.float64)
            if a.shape != (n, 4) or not np.all(np.isfinite(a)):
                continue
            a /= np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-12)
            v = _max_cos(a)
            if v < best_val:
                best_val = v
                best_x = a.copy()
        except Exception:
            continue
    if best_x is None:
        best_x = _rand_dirs(rng, n)
        best_val = _max_cos(best_x)
    # elite archive
    elite = [(best_val, best_x.copy())]
    # quick diverse seeds: a few jittered copies annealed briefly to populate elite
    it = 0
    # main basin-hopping loop
    while time.time() < FINAL_START - 1.0:
        it += 1
        # choose base
        r = rng.rand()
        if r < 0.65 or len(elite) == 1:
            base = best_x
            base_val = best_val
        elif r < 0.9:
            k = int(rng.randint(len(elite)))
            base_val, base = elite[k]
            base = base.copy()
        else:
            # fresh random
            base = _rand_dirs(rng, n)
            base_val = _max_cos(base)
        op = int(rng.randint(5))
        try:
            if op == 0:
                sig = [0.008, 0.03, 0.10, 0.20][int(rng.randint(4))]
                x0 = base + rng.randn(n, 4) * sig
                x0 /= np.maximum(np.linalg.norm(x0, axis=1, keepdims=True), 1e-12)
            elif op == 1:
                nm = int(rng.choice([1, 2, 3, 3, 4]))
                x0 = _hub_to_hole(base, rng, n_move=nm, n_cand=5000)
            elif op == 2:
                kk = int(rng.choice([4, 8, 14, 22]))
                x0 = _subset_reinit(base, rng, k=kk, n_cand=3000)
            elif op == 3:
                x0 = _cap_rotate(base, rng)
                # add small jitter after rotation
                x0 = x0 + rng.randn(n, 4) * 0.004
                x0 /= np.maximum(np.linalg.norm(x0, axis=1, keepdims=True), 1e-12)
            else:
                # combined hub + cap
                x0 = _hub_to_hole(base, rng, n_move=int(rng.randint(1, 4)), n_cand=4000)
                if rng.rand() < 0.6:
                    x0 = _cap_rotate(x0, rng)
        except Exception:
            continue
        # anneal params varied
        T0 = float(rng.choice([0.004, 0.007, 0.012]))
        T1 = float(rng.choice([0.00015, 0.0003, 0.0006]))
        lr0 = float(rng.uniform(0.12, 0.30))
        mu = float(rng.uniform(0.82, 0.92))
        noise0 = float(rng.choice([0.0, 0.0, 0.002, 0.006]))
        steps = int(rng.choice([900, 1400, 2000]))
        # limit steps near end
        remain = FINAL_START - 1.0 - time.time()
        if remain < 8:
            steps = min(steps, 600)
        try:
            xa, va = _gd_softmax(x0, steps, T0, T1, lr0, mu, rng, noise0=noise0, deadline=FINAL_START - 1.0)
        except Exception:
            continue
        # quick LBFGS if promising
        if va < best_val + 3e-4 and time.time() < FINAL_START - 3.0:
            try:
                xf = _lbfgs_smooth(xa, FINAL_START - 1.0, taus=(0.002, 0.0006), iters=(60, 60))
                vf = _max_cos(xf)
                if vf < va:
                    xa, va = xf, vf
            except Exception:
                pass
        # LP polish if very promising
        improved = False
        if va < best_val + 1.5e-4 and time.time() < FINAL_START - 2.0:
            try:
                # short LP budget: cap deadline
                xp, vp = _lp_polish(xa, min(time.time() + 2.2, FINAL_START - 1.0), r0=0.002, cap=1300)
                if vp < va:
                    xa, va = xp, vp
            except Exception:
                pass
        if va < best_val - 1e-12:
            best_val = va
            best_x = xa.copy()
            elite.append((va, xa.copy()))
            # keep elite small, sorted
            elite = sorted(elite, key=lambda e: e[0])[:6]
        elif va < best_val + 8e-4:
            # diversity: add with small prob
            if len(elite) < 6 and rng.rand() < 0.25:
                elite.append((va, xa.copy()))
                elite = sorted(elite, key=lambda e: e[0])[:6]
            elif rng.rand() < 0.05:
                # replace worst if better than worst
                if va < elite[-1][0]:
                    elite[-1] = (va, xa.copy())
                    elite = sorted(elite, key=lambda e: e[0])[:6]
        # occasional elite pruning by time
        if time.time() > FINAL_START - 1.0:
            break
    # final intensive polish of best
    try:
        if time.time() < DEADLINE - 1.0:
            # medium LBFGS then LP
            best_x = _lbfgs_smooth(best_x, DEADLINE - 8.0, taus=(0.003, 0.001, 0.00045), iters=(80, 80, 100))
            best_val = _max_cos(best_x)
    except Exception:
        pass
    try:
        if time.time() < DEADLINE - 0.5:
            best_x, best_val = _lp_polish(best_x, DEADLINE - 0.3, r0=0.003, cap=1400)
    except Exception:
        pass
    best_x /= np.maximum(np.linalg.norm(best_x, axis=1, keepdims=True), 1e-12)
    return best_x
