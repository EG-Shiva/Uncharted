import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _expand_reps(R, cos_list, sin_list, add_pole):
    K = R.shape[0]
    p = len(cos_list[0])
    norb = K * p
    X = np.empty((norb + (1 if add_pole else 0), 7), dtype=np.float64)
    # planes
    for k in range(3):
        c = cos_list[k]
        s = sin_list[k]
        Ra = R[:, 2*k]
        Rb = R[:, 2*k+1]
        # (K,p)
        X0 = Ra[:, None] * c[None, :] - Rb[:, None] * s[None, :]
        X1 = Ra[:, None] * s[None, :] + Rb[:, None] * c[None, :]
        X0f = X0.reshape(-1)
        X1f = X1.reshape(-1)
        # ordering: orbit-major: index o*p+j
        X[:norb, 2*k] = X0f
        X[:norb, 2*k+1] = X1f
    h = R[:, 6]
    X[:norb, 6] = np.repeat(h, p)
    if add_pole:
        X[norb, :] = 0.0
        X[norb, 6] = 1.0
    return X

def _sym_adam(Rinit, p, speeds, steps, tau0, tau1, lr0, lr1, deadline, seed_off=0):
    K = Rinit.shape[0]
    add_pole = (K * p < 323)
    # precompute cos/sin
    cos_list, sin_list = [], []
    for sk in speeds:
        th = 2.0 * np.pi * float(sk) / float(p)
        ang = th * np.arange(p, dtype=np.float64)
        cos_list.append(np.cos(ang))
        sin_list.append(np.sin(ang))
    R = Rinit.copy()
    R = R / np.linalg.norm(R, axis=1, keepdims=True)
    m1 = np.zeros_like(R)
    m2 = np.zeros_like(R)
    b1, b2, eps = 0.9, 0.999, 1e-8
    n = K * p + (1 if add_pole else 0)
    iuI, iuJ = np.triu_indices(n, 1)
    best_R = R.copy()
    Xb = _expand_reps(R, cos_list, sin_list, add_pole)
    best_v = _max_cos(Xb)
    t = 0
    for s in range(steps):
        if s % 40 == 0 and time.time() >= deadline:
            break
        frac = s / max(1, steps - 1)
        tau = tau0 + (tau1 - tau0) * frac
        lr = lr0 + (lr1 - lr0) * frac
        X = _expand_reps(R, cos_list, sin_list, add_pole)
        G = X @ X.T
        c = G[iuI, iuJ]
        mx = float(c.max())
        e = np.exp((c - mx) / tau)
        S = float(e.sum())
        if not np.isfinite(S) or S == 0:
            break
        w = e / S
        W = np.zeros((n, n), dtype=np.float64)
        W[iuI, iuJ] = w
        W = W + W.T
        Wx = W @ X
        dot = np.sum(Wx * X, axis=1, keepdims=True)
        Gt = Wx - dot * X
        # pullback to reps (ignore pole row)
        if add_pole:
            Gt_o = Gt[:K*p].reshape(K, p, 7)
        else:
            Gt_o = Gt.reshape(K, p, 7)
        Grep = np.zeros_like(R)
        for k in range(3):
            ck = cos_list[k]
            sk_ = sin_list[k]
            g0 = Gt_o[:, :, 2*k]
            g1 = Gt_o[:, :, 2*k+1]
            # dRa = sum_j c*g0 + s*g1 ; dRb = sum_j -s*g0 + c*g1
            Grep[:, 2*k] = np.sum(g0 * ck[None, :] + g1 * sk_[None, :], axis=1)
            Grep[:, 2*k+1] = np.sum(-g0 * sk_[None, :] + g1 * ck[None, :], axis=1)
        Grep[:, 6] = np.sum(Gt_o[:, :, 6], axis=1)
        # tangent project on rep spheres
        d = np.sum(Grep * R, axis=1, keepdims=True)
        Grep = Grep - d * R
        # Adam
        t += 1
        m1 = b1 * m1 + (1 - b1) * Grep
        m2 = b2 * m2 + (1 - b2) * (Grep * Grep)
        mh = m1 / (1 - b1 ** t)
        vh = m2 / (1 - b2 ** t)
        R = R - lr * mh / (np.sqrt(vh) + eps)
        # renormalize
        nr = np.linalg.norm(R, axis=1, keepdims=True)
        nr[nr == 0] = 1.0
        R = R / nr
        if (s % 50 == 49) or (s == steps - 1):
            Xc = _expand_reps(R, cos_list, sin_list, add_pole)
            v = _max_cos(Xc)
            if v < best_v:
                best_v = v
                best_R = R.copy()
            if time.time() >= deadline:
                break
    return best_R, best_v, (cos_list, sin_list, add_pole)

def _smooth_descent_full(x, steps, tau0, tau1, lr0, lr1, mu, deadline):
    n, d = x.shape
    iuI, iuJ = np.triu_indices(n, 1)
    vel = np.zeros_like(x)
    best = _max_cos(x)
    best_x = x.copy()
    for s in range(steps):
        if s % 25 == 0 and time.time() >= deadline:
            break
        frac = s / max(1, steps - 1)
        tau = tau0 + (tau1 - tau0) * frac
        lr = lr0 + (lr1 - lr0) * frac
        G = x @ x.T
        c = G[iuI, iuJ]
        m = float(c.max())
        e = np.exp((c - m) / tau)
        S = float(e.sum())
        if S == 0 or not np.isfinite(S):
            break
        w = e / S
        Wx = np.zeros_like(x)
        np.add.at(Wx, iuI, w[:, None] * x[iuJ])
        np.add.at(Wx, iuJ, w[:, None] * x[iuI])
        dot = np.sum(Wx * x, axis=1, keepdims=True)
        Gt = Wx - dot * x
        vel = mu * vel + Gt
        x = x - lr * vel
        xn = np.linalg.norm(x, axis=1, keepdims=True)
        xn[xn == 0] = 1.0
        x = x / xn
        if (s % 60 == 59) or (s == steps - 1):
            v = _max_cos(x)
            if v < best:
                best = v
                best_x = x.copy()
    return best_x

def _lp_polish(x, deadline, init_radius=5e-3):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = float(init_radius)
    iuI, iuJ = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        G = x @ x.T
        c = G[iuI, iuJ]
        m = float(c.max())
        thresh = m - max(20.0 * radius, 1e-9)
        sel = c > thresh
        cnt = int(sel.sum())
        if cnt == 0:
            break
        if cnt > 2200:
            idx = np.where(sel)[0]
            cidx = c[idx]
            part = np.argpartition(cidx, -2200)[-2200:]
            sel = np.zeros_like(sel, dtype=bool)
            sel[idx[part]] = True
            cnt = 2200
        I = iuI[sel]
        J = iuJ[sel]
        csel = c[sel]
        p_ = cnt
        nv = n * d + 1
        rows = np.repeat(np.arange(p_), 2 * d + 1)
        cols = np.empty(p_ * (2 * d + 1), dtype=np.int64)
        vals = np.empty(p_ * (2 * d + 1), dtype=np.float64)
        Cm = cols.reshape(p_, 2 * d + 1)
        Vm = vals.reshape(p_, 2 * d + 1)
        ar = np.arange(d)
        Cm[:, :d] = I[:, None] * d + ar
        Cm[:, d:2*d] = J[:, None] * d + ar
        Cm[:, 2*d] = n * d
        Vm[:, :d] = x[J]
        Vm[:, d:2*d] = x[I]
        Vm[:, 2*d] = -1.0
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p_, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv)
        obj[-1] = 1.0
        remain = deadline - time.time()
        if remain <= 0.05:
            break
        tl = min(remain - 0.02, 8.0)
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs",
                          options={"time_limit": float(max(0.1, tl))})
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
        if not np.all(np.isfinite(y)):
            radius *= 0.5
            continue
        v = _max_cos(y)
        if v < best:
            x = y
            best = v
            radius = min(radius * 1.5, 0.03)
        else:
            radius *= 0.5
        if time.time() >= deadline:
            break
    return x

def solve():
    t0 = time.time()
    HARD = t0 + 165.0
    SYM_END = t0 + 92.0
    FULL_END = t0 + 160.0
    rng = np.random.default_rng(7)
    # load incumbent
    x_inc = None
    b_inc = np.inf
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn).astype(np.float64)
            if a.shape == (323, 7):
                a = a / np.linalg.norm(a, axis=1, keepdims=True)
                if np.all(np.isfinite(a)):
                    v = _max_cos(a)
                    if v < b_inc:
                        b_inc = v
                        x_inc = a.copy()
        except Exception:
            pass
    if x_inc is None:
        a = rng.standard_normal(size=(323, 7))
        a /= np.linalg.norm(a, axis=1, keepdims=True)
        x_inc = a
        b_inc = _max_cos(a)
    x_best = x_inc.copy()
    best = float(b_inc)
    # ---- symmetric phase ----
    configs = []
    for sp in [(1, 5, 11), (1, 2, 3), (1, 3, 8), (1, 4, 9)]:
        configs.append((23, 14, sp))
    for sp in [(1, 4, 9), (1, 2, 5), (1, 3, 7)]:
        configs.append((19, 17, sp))
    sym_best_X = None
    sym_best_v = np.inf
    # quick trials
    for (p, K, sp) in configs:
        if time.time() >= SYM_END:
            break
        for restart in range(2):
            if time.time() >= SYM_END:
                break
            R0 = rng.standard_normal(size=(K, 7))
            R0 /= np.linalg.norm(R0, axis=1, keepdims=True)
            # avoid degenerate tiny radii: add small floor by mixing
            dl = min(SYM_END, time.time() + 6.0, HARD - 30)
            if dl - time.time() < 0.3:
                break
            Rb, vb, meta = _sym_adam(R0, p, sp, steps=700, tau0=0.02, tau1=0.003,
                                    lr0=0.08, lr1=0.01, deadline=dl)
            if vb < sym_best_v and np.isfinite(vb):
                sym_best_v = vb
                cos_list, sin_list, add_pole = meta
                sym_best_X = _expand_reps(Rb, cos_list, sin_list, add_pole)
                sym_best_R = Rb.copy()
                sym_best_meta = meta
                sym_best_sp = sp
                sym_best_p = p
    # refine top symmetric if found
    if sym_best_X is not None and time.time() < SYM_END:
        # longer refine on the winning stratum with fresh inits near winner + random
        try:
            p = sym_best_p
            sp = sym_best_sp
            K = sym_best_R.shape[0]
            for extra in range(3):
                if time.time() >= SYM_END:
                    break
                if extra == 0:
                    R0 = sym_best_R + rng.standard_normal(size=sym_best_R.shape) * 0.05
                else:
                    R0 = rng.standard_normal(size=(K, 7))
                R0 /= np.linalg.norm(R0, axis=1, keepdims=True)
                dl = min(SYM_END, time.time() + 7.0)
                Rb, vb, meta = _sym_adam(R0, p, sp, steps=1200, tau0=0.012, tau1=0.0015,
                                        lr0=0.06, lr1=0.008, deadline=dl)
                if vb < sym_best_v:
                    sym_best_v = vb
                    cos_list, sin_list, add_pole = meta
                    sym_best_X = _expand_reps(Rb, cos_list, sin_list, add_pole)
                    sym_best_R = Rb.copy()
                    sym_best_meta = meta
        except Exception:
            pass
    # update global best with raw symmetric (usually worse but keep if lucky)
    if sym_best_X is not None:
        try:
            v = _max_cos(sym_best_X)
            if v < best:
                best = v
                x_best = sym_best_X.copy()
        except Exception:
            pass
    # ---- full-space phase: break symmetry ----
    candidates = []
    if sym_best_X is not None:
        candidates.append(sym_best_X.copy())
        # perturbed copies
        for sig in (0.01, 0.03):
            if time.time() >= FULL_END:
                break
            xp = sym_best_X + rng.standard_normal(size=sym_best_X.shape) * sig
            xp /= np.linalg.norm(xp, axis=1, keepdims=True)
            candidates.append(xp)
    # also include incumbent perturbed slightly
    candidates.append(x_inc.copy())
    # iterate candidates with smooth + LP
    for xc in candidates:
        if time.time() >= FULL_END:
            break
        x = xc.copy()
        # smooth descent short
        dl1 = min(time.time() + 1.2, FULL_END, HARD - 6)
        if dl1 - time.time() > 0.2:
            try:
                x = _smooth_descent_full(x, steps=500, tau0=0.012, tau1=0.0025,
                                         lr0=3.0, lr1=0.5, mu=0.8, deadline=dl1)
            except Exception:
                pass
        dl2 = min(time.time() + 4.0, FULL_END, HARD - 5)
        if dl2 - time.time() > 0.3:
            try:
                x = _lp_polish(x, deadline=dl2, init_radius=0.006)
            except Exception:
                pass
        try:
            v = _max_cos(x)
        except Exception:
            continue
        if v < best and np.all(np.isfinite(x)):
            best = v
            x_best = x.copy()
    # hill-climb global best briefly
    it = 0
    while time.time() < FULL_END and time.time() < HARD - 6 and it < 12:
        it += 1
        x = x_best.copy()
        # random kick to worst points
        try:
            G = x @ x.T
            np.fill_diagonal(G, -2.0)
            wpp = G.max(axis=1)
            k = 16
            idx = np.argpartition(wpp, -k)[-k:]
            x[idx] += rng.standard_normal(size=(k, 7)) * 0.04
            x[idx] /= np.linalg.norm(x[idx], axis=1, keepdims=True)
        except Exception:
            x = x + rng.standard_normal(size=x.shape) * 0.01
            x /= np.linalg.norm(x, axis=1, keepdims=True)
        dl1 = min(time.time() + 0.8, FULL_END, HARD - 5)
        if dl1 - time.time() > 0.2:
            try:
                x = _smooth_descent_full(x, steps=350, tau0=0.010, tau1=0.002,
                                         lr0=2.5, lr1=0.4, mu=0.75, deadline=dl1)
            except Exception:
                pass
        dl2 = min(time.time() + 2.5, FULL_END, HARD - 4)
        if dl2 - time.time() > 0.3:
            try:
                x = _lp_polish(x, deadline=dl2, init_radius=0.005)
            except Exception:
                pass
        try:
            v = _max_cos(x)
        except Exception:
            continue
        if v < best - 1e-12:
            best = v
            x_best = x.copy()
    # final squeeze
    if time.time() < HARD - 1.5:
        try:
            dl = min(HARD - 0.8, time.time() + max(1.0, HARD - 0.8 - time.time()))
            x_best = _lp_polish(x_best, deadline=dl, init_radius=2e-3)
        except Exception:
            pass
    x_best = x_best / np.linalg.norm(x_best, axis=1, keepdims=True)
    # sanitize, never return worse than incumbent if something broke
    try:
        vb = _max_cos(x_best)
        if not np.isfinite(vb) or vb > best + 1e-12:
            pass
        if not np.isfinite(vb) or vb > b_inc and np.all(np.isfinite(x_best)):
            # keep x_best only if it beats incumbent else revert to incumbent
            if vb > b_inc:
                x_best = x_inc.copy()
    except Exception:
        x_best = x_inc.copy()
    x_best[~np.isfinite(x_best)] = 0.0
    nn = np.linalg.norm(x_best, axis=1, keepdims=True)
    for i in range(x_best.shape[0]):
        if not np.isfinite(nn[i, 0]) or nn[i, 0] < 1e-12:
            vv = rng.standard_normal(size=(7,))
            vv /= np.linalg.norm(vv)
            x_best[i] = vv
    x_best = x_best / np.linalg.norm(x_best, axis=1, keepdims=True)
    return x_best
