import time
import numpy as np
from scipy.optimize import linprog, minimize
from scipy.sparse import coo_matrix

def _norm(X):
    nrm = np.linalg.norm(X, axis=1, keepdims=True)
    nrm = np.maximum(nrm, 1e-12)
    return X / nrm

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _degs(X, tol=1e-7):
    n = X.shape[0]
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    m = float(G.max())
    mask = G > m - tol
    deg = mask.sum(axis=1).astype(int)
    return m, deg, G

def _anneal(x0, steps, T0, T1, lr, mu, rng, active=None, noise0=0.0, deadline=None):
    x = x0.copy()
    n, d = x.shape
    if active is None:
        act = np.ones(n, dtype=bool)
    else:
        act = np.asarray(active, dtype=bool)
    vel = np.zeros_like(x)
    G0 = x @ x.T
    np.fill_diagonal(G0, -2.0)
    bv = float(G0.max())
    bx = x.copy()
    if T0 <= 0 or T1 <= 0:
        T0 = max(T0, 1e-12); T1 = max(T1, 1e-12)
    lratio = np.log(T1 / T0)
    for k in range(steps):
        if deadline is not None and (k & 127) == 0 and time.time() > deadline:
            break
        frac = k / max(1, steps - 1)
        T = T0 * np.exp(lratio * frac)
        if T < 5e-5:
            T = 5e-5
        lrk = lr * (1.0 - 0.55 * frac)
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        E = np.exp((G - m) / T)
        np.fill_diagonal(E, 0.0)
        S = float(E.sum() * 0.5)
        if not np.isfinite(S) or S < 1e-300:
            S = 1e-300
        Gr = (E @ x) / S
        Gr -= (Gr * x).sum(axis=1, keepdims=True) * x
        if not np.all(act):
            Gr[~act] = 0.0
            vel[~act] = 0.0
        vel = mu * vel + (1.0 - mu) * Gr
        if not np.all(act):
            vel[~act] = 0.0
        x[act] = x[act] - lrk * vel[act]
        if noise0 > 0:
            amp = noise0 * (1.0 - frac)
            if amp > 0:
                na = int(act.sum())
                if na > 0:
                    nz = rng.randn(na, d) * amp
                    xa = x[act]
                    nz -= (nz * xa).sum(axis=1, keepdims=True) * xa
                    xa = xa + nz
                    xa /= np.maximum(np.linalg.norm(xa, axis=1, keepdims=True), 1e-12)
                    x[act] = xa
        xn = np.linalg.norm(x[act], axis=1, keepdims=True)
        x[act] /= np.maximum(xn, 1e-12)
        if m < bv:
            bv = m
            bx = x.copy()
    return bx, bv

def _lbfgs(x_init, deadline, taus=(0.002, 0.0006), iters=(60, 60)):
    x = _norm(x_init.copy())
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
        except Exception:
            break
    Yf = y.reshape(n, d)
    return _norm(Yf)

def _lp_polish(x, deadline, r0=0.003, cap=1100):
    x = _norm(x.copy())
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
            if int(sel.sum()) == 0:
                sel[int(np.argmax(c))] = True
            if int(sel.sum()) > cap:
                idx = np.argsort(-c)[:cap]
                s2 = np.zeros_like(sel, dtype=bool)
                s2[idx] = True
                sel = s2
            I = iu0[sel]; J = iu1[sel]
            csel = c[sel]
            p = len(I)
            nv = n * d + 1
            rows = np.repeat(np.arange(p), 2 * d + 1)
            cols = np.empty((p, 2 * d + 1), dtype=int)
            ar = np.arange(d)
            cols[:, :d] = I[:, None] * d + ar
            cols[:, d:2 * d] = J[:, None] * d + ar
            cols[:, 2 * d] = n * d
            cols = cols.ravel()
            vals = np.empty((p, 2 * d + 1), dtype=float)
            vals[:, :d] = x[J]
            vals[:, d:2 * d] = x[I]
            vals[:, 2 * d] = -1.0
            vals = vals.ravel()
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv); obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
            if res.status != 0:
                radius *= 0.5
                continue
            y = x + res.x[:-1].reshape(n, d)
            y = _norm(y)
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

def _refine_hole(c0, Y, T=0.002, iters=120, lr=0.06):
    c = c0 / max(float(np.linalg.norm(c0)), 1e-12)
    for _ in range(iters):
        dots = Y @ c
        m = float(dots.max())
        e = np.exp((dots - m) / T)
        s = float(e.sum())
        if s <= 0 or not np.isfinite(s):
            break
        w = e / s
        g = w @ Y
        g = g - float(g @ c) * c
        c = c - lr * g
        nrm = float(np.linalg.norm(c))
        if nrm < 1e-12:
            break
        c = c / nrm
    return c

def _find_hole(Y, rng, n_sample=15000, n_refine=15):
    C = rng.randn(n_sample, 4)
    C = _norm(C)
    D = Y @ C.T
    mx = D.max(axis=0)
    idx = np.argsort(mx)[:n_refine]
    best_c = None
    best_v = 1e9
    for j in idx:
        c = _refine_hole(C[j], Y)
        v = float((Y @ c).max())
        if v < best_v:
            best_v = v
            best_c = c
    return best_c, best_v

def solve():
    t0 = time.time()
    DEADLINE = t0 + 172.0
    rng = np.random.RandomState(916331)
    n = 163
    best_x = None
    best_val = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=np.float64)
            if a.shape != (n, 4) or not np.all(np.isfinite(a)):
                continue
            a = _norm(a)
            v = _max_cos(a)
            if v < best_val:
                best_val = v
                best_x = a.copy()
        except Exception:
            continue
    if best_x is None:
        best_x = _norm(rng.randn(n, 4))
        best_val = _max_cos(best_x)
    incumbent = best_x.copy()
    inc_val = best_val
    base = incumbent.copy()
    base_val = inc_val
    trial = 0
    lp_count = 0
    while time.time() < DEADLINE - 14.0 and trial < 60:
        trial += 1
        remain = DEADLINE - 14.0 - time.time()
        if remain <= 1.0:
            break
        mode = trial % 5  # 0,1 cap ; 2,3 hub ; 4 shake
        k = int([12, 9, 15, 8, 10][trial % 5] + rng.randint(-2, 3))
        k = max(6, min(20, k))
        try:
            cur_base = base.copy()
            if mode in (0, 1):
                u = rng.randn(4); u /= max(float(np.linalg.norm(u)), 1e-12)
                # bias center toward a random existing point sometimes
                if rng.rand() < 0.6:
                    u = cur_base[rng.randint(n)] + 0.35 * rng.randn(4)
                    u /= max(float(np.linalg.norm(u)), 1e-12)
                dots = cur_base @ u
                del_idx = np.argsort(-dots)[:k]
            elif mode in (2, 3):
                m, deg, G = _degs(cur_base, tol=1e-7)
                order = np.argsort(-deg + rng.randn(n) * 0.8)
                hub = int(order[0] if rng.rand() < 0.7 else order[rng.randint(min(5, n))])
                row = G[hub].copy()
                row[hub] = -2.0
                # contact neighbours
                cn = np.where(row > m - 1e-7)[0].tolist()
                # fill with nearest neighbours spatially
                near = np.argsort(-row).tolist()
                sel = [hub]
                for q in cn + near:
                    if q not in sel:
                        sel.append(int(q))
                    if len(sel) >= k:
                        break
                del_idx = np.array(sel[:k], dtype=int)
            else:
                # global tangent shake, no deletion
                sigma = 0.03 + 0.03 * rng.rand()
                nz = rng.randn(n, 4) * sigma
                nz -= (nz * cur_base).sum(axis=1, keepdims=True) * cur_base
                cur = _norm(cur_base + nz)
                cur, vc = _anneal(cur, steps=550, T0=0.004, T1=0.0004, lr=0.08, mu=0.85,
                                 rng=rng, noise0=0.002, deadline=min(time.time() + 4.0, DEADLINE - 14.0))
                cur, vc = _anneal(cur, steps=400, T0=0.0009, T1=0.00015, lr=0.05, mu=0.88,
                                  rng=rng, noise0=0.0, deadline=min(time.time() + 3.0, DEADLINE - 14.0))
                if time.time() < DEADLINE - 16.0 and vc < base_val + 0.001:
                    cur = _lbfgs(cur, min(time.time() + 2.5, DEADLINE - 14.0))
                    vc = _max_cos(cur)
                # metropolis update
                if vc < inc_val:
                    incumbent, inc_val = cur.copy(), vc
                    base, base_val = cur.copy(), vc
                elif vc < base_val or rng.rand() < np.exp(-(vc - base_val) / 0.0012):
                    base, base_val = cur.copy(), vc
                continue
            keep = np.ones(n, dtype=bool)
            keep[del_idx] = False
            Y = cur_base[keep].copy()
            # relax remainder
            Y, _ = _anneal(Y, steps=500, T0=0.005, T1=0.0004, lr=0.09, mu=0.85,
                           rng=rng, noise0=0.003, deadline=min(time.time() + 3.0, DEADLINE - 14.0))
            cur = Y.copy()
            for t in range(k):
                if time.time() > DEADLINE - 14.0:
                    break
                hc, hv = _find_hole(cur, rng, n_sample=12000, n_refine=12)
                if hc is None:
                    hc = rng.randn(4); hc /= float(np.linalg.norm(hc))
                cur = _norm(np.vstack([cur, hc.reshape(1, 4)]))
                # local relax of newcomer + close neighbours
                try:
                    dvec = cur[:-1] @ cur[-1]
                    nbr = np.where(dvec > 0.72)[0]
                except Exception:
                    nbr = np.array([], dtype=int)
                act = np.zeros(cur.shape[0], dtype=bool)
                act[-1] = True
                if len(nbr) > 0:
                    act[nbr] = True
                if int(act.sum()) <= 2 or (t % 4 == 3):
                    act[:] = True
                cur, _ = _anneal(cur, steps=260, T0=0.0025, T1=0.0003, lr=0.06, mu=0.85,
                                 rng=rng, active=act, noise0=0.0,
                                 deadline=min(time.time() + 1.0, DEADLINE - 13.0))
            if cur.shape[0] != n:
                continue
            cur = _norm(cur)
            cur, vc = _anneal(cur, steps=500, T0=0.003, T1=0.00035, lr=0.07, mu=0.86,
                              rng=rng, noise0=0.0, deadline=min(time.time() + 3.0, DEADLINE - 14.0))
            if time.time() < DEADLINE - 15.0:
                cur, vc2 = _anneal(cur, steps=350, T0=0.0008, T1=0.00012, lr=0.045, mu=0.88,
                                   rng=rng, noise0=0.0, deadline=min(time.time() + 2.5, DEADLINE - 14.0))
                vc = min(vc, _max_cos(cur))
            # quick lbfgs if promising
            if vc < base_val + 0.0012 and time.time() < DEADLINE - 16.0:
                cur = _lbfgs(cur, min(time.time() + 2.0, DEADLINE - 14.5))
                vc = _max_cos(cur)
            # LP polish only if near best or topology jump promising
            if vc < inc_val - 2e-7:
                if time.time() < DEADLINE - 13.0 and lp_count < 10:
                    cur, vc = _lp_polish(cur, min(time.time() + 6.0, DEADLINE - 8.0), r0=0.003)
                    lp_count += 1
                incumbent, inc_val = cur.copy(), vc
                base, base_val = cur.copy(), vc
            elif vc < base_val + 0.0008 and time.time() < DEADLINE - 16.0 and lp_count < 10 and (trial % 4 == 0 or vc < inc_val + 1e-6):
                cur2, vc2 = _lp_polish(cur, min(time.time() + 4.0, DEADLINE - 10.0), r0=0.0025)
                lp_count += 1
                if vc2 < inc_val:
                    incumbent, inc_val = cur2.copy(), vc2
                    base, base_val = cur2.copy(), vc2
                elif vc2 < base_val or rng.rand() < np.exp(-(vc2 - base_val) / 0.0012):
                    base, base_val = cur2.copy(), vc2
                elif vc < base_val or rng.rand() < np.exp(-(vc - base_val) / 0.0015):
                    base, base_val = cur.copy(), vc
            else:
                if vc < base_val or rng.rand() < np.exp(-(vc - base_val) / 0.0015):
                    base, base_val = cur.copy(), vc
        except Exception:
            continue
    # final squeeze of incumbent
    try:
        if time.time() < DEADLINE:
            incumbent, _ = _anneal(incumbent, steps=600, T0=0.0009, T1=0.0001, lr=0.04, mu=0.9,
                                   rng=rng, noise0=0.0, deadline=min(time.time() + 4.0, DEADLINE - 4.0))
        if time.time() < DEADLINE - 3.0:
            incumbent = _lbfgs(incumbent, DEADLINE - 2.5)
        if time.time() < DEADLINE - 1.0:
            incumbent, inc_val = _lp_polish(incumbent, DEADLINE - 0.2, r0=0.002)
    except Exception:
        pass
    incumbent = _norm(np.asarray(incumbent, dtype=np.float64))
    return incumbent
