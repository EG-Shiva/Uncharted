import time
import numpy as np
from scipy.optimize import minimize, linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _per_max(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return G.max(axis=1)

def _lp_polish(x, deadline, r0=8e-4, cap_p=2500):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = r0
    iu = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        if time.time() > deadline - 0.3:
            break
        c = (x @ x.T)[iu]
        m = float(c.max())
        sel = c > m - max(20.0 * radius, 1e-9)
        I = iu[0][sel]; J = iu[1][sel]
        p = len(I)
        if p == 0:
            radius *= 0.5
            continue
        csel = c[sel]
        if p > cap_p:
            od = np.argsort(-csel)[:cap_p]
            I, J = I[od], J[od]
            csel = csel[od]
            p = cap_p
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv); obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        step = res.x[:-1].reshape(n, d)
        y = x + step
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        v = _max_cos(y)
        if v < best - 1e-15:
            x, best = y, v
            radius = min(radius * 1.5, 0.02)
        else:
            improved = False
            for sh in (0.3, 0.1):
                yy = x + step * sh
                yy /= np.linalg.norm(yy, axis=1, keepdims=True)
                vv = _max_cos(yy)
                if vv < best:
                    x, best = yy, vv
                    improved = True
                    break
            if not improved:
                radius *= 0.5
    return x

def _full_fg(IU, beta):
    def fg(yf):
        Y = yf.reshape(68, 12)
        nr = np.linalg.norm(Y, axis=1, keepdims=True)
        nr = np.maximum(nr, 1e-12)
        X = Y / nr
        g = (X @ X.T)[IU]
        m = float(g.max())
        e = np.exp(beta * (g - m))
        S = float(e.sum()) + 1e-300
        F = m + np.log(S) / beta
        w = e / S
        P = np.zeros((68, 68))
        P[IU] = w
        P = P + P.T
        GX = P @ X
        dot = np.sum(GX * X, axis=1, keepdims=True)
        GY = (GX - dot * X) / nr
        return F, GY.ravel()
    return fg

def _run_full(x0, beta, maxiter):
    IU = np.triu_indices(68, 1)
    fg = _full_fg(IU, beta)
    y0 = (x0 / np.linalg.norm(x0, axis=1, keepdims=True)).ravel().copy()
    try:
        r = minimize(fg, y0, jac=True, method='L-BFGS-B',
                     options={'maxiter': maxiter, 'ftol': 1e-16, 'gtol': 1e-8, 'maxfun': maxiter * 3})
        Y = r.x.reshape(68, 12)
        Y /= np.linalg.norm(Y, axis=1, keepdims=True)
        return Y
    except Exception:
        return x0

def _run_subset(fixed, Z0, beta, maxiter):
    # fixed (nf,d), Z (m,d) variable; LSE over cross + internal pairs
    m = Z0.shape[0]
    d = fixed.shape[1]
    def fg(zf):
        Z = zf.reshape(m, d)
        nr = np.linalg.norm(Z, axis=1, keepdims=True)
        nr = np.maximum(nr, 1e-12)
        Zn = Z / nr
        C = Zn @ fixed.T  # m x nf
        Gz = Zn @ Zn.T
        iu = np.triu_indices(m, 1)
        g1 = C.ravel()
        g2 = Gz[iu] if m > 1 else np.empty(0)
        g = np.concatenate([g1, g2]) if g2.size else g1
        mx = float(g.max())
        e = np.exp(beta * (g - mx))
        S = float(e.sum()) + 1e-300
        F = mx + np.log(S) / beta
        w = e / S
        w1 = w[:g1.size].reshape(m, -1)
        GX = w1 @ fixed  # m x d
        if g2.size:
            P = np.zeros((m, m))
            P[iu] = w[g1.size:]
            P = P + P.T
            GX = GX + P @ Zn
        dot = np.sum(GX * Zn, axis=1, keepdims=True)
        GZ = (GX - dot * Zn) / nr
        return F, GZ.ravel()
    try:
        r = minimize(fg, Z0.ravel().copy(), jac=True, method='L-BFGS-B',
                     options={'maxiter': maxiter, 'ftol': 1e-15, 'gtol': 1e-8, 'maxfun': maxiter * 3})
        Z = r.x.reshape(m, d)
        Z /= np.linalg.norm(Z, axis=1, keepdims=True)
        return Z
    except Exception:
        Z = Z0 / np.linalg.norm(Z0, axis=1, keepdims=True)
        return Z

def _best_hole(Xr, rng, ncand=20000):
    d = Xr.shape[1]
    cand = rng.standard_normal((ncand, d))
    cand /= np.linalg.norm(cand, axis=1, keepdims=True)
    sc = (cand @ Xr.T).max(axis=1)
    return cand[int(np.argmin(sc))]

def _opt_single(Xr, z0, beta, maxiter=80):
    def fg(z):
        r = max(float(np.linalg.norm(z)), 1e-12)
        y = z / r
        s = Xr @ y
        mm = float(s.max())
        e = np.exp(beta * (s - mm))
        S = float(e.sum()) + 1e-300
        F = mm + np.log(S) / beta
        w = e / S
        gy = Xr.T @ w
        gz = (gy - float(gy @ y) * y) / r
        return F, gz
    try:
        r = minimize(fg, z0, jac=True, method='L-BFGS-B',
                     options={'maxiter': maxiter, 'ftol': 1e-15, 'gtol': 1e-9, 'maxfun': maxiter * 2})
        z = r.x
        z /= (np.linalg.norm(z) + 1e-300)
        return z
    except Exception:
        z = z0 / (np.linalg.norm(z0) + 1e-300)
        return z

def solve():
    t0 = time.time()
    TL = 172.0
    rng = np.random.default_rng(987654321)
    cands = []
    try:
        xb = np.load("best.npy").astype(float)
        xb /= np.linalg.norm(xb, axis=1, keepdims=True)
        cands.append(xb)
    except Exception:
        pass
    try:
        xr = np.load("record.npy").astype(float)
        xr /= np.linalg.norm(xr, axis=1, keepdims=True)
        cands.append(xr)
    except Exception:
        pass
    if not cands:
        x0 = rng.standard_normal((68, 12))
        x0 /= np.linalg.norm(x0, axis=1, keepdims=True)
        cands.append(x0)
    x_best = cands[0].copy()
    best_val = _max_cos(x_best)
    for c in cands[1:]:
        v = _max_cos(c)
        if v < best_val:
            best_val = v
            x_best = c.copy()
    # initial squeeze
    x_best = _lp_polish(x_best, min(t0 + 8.0, t0 + TL - 30.0))
    best_val = _max_cos(x_best)
    cur = x_best.copy()
    cur_val = best_val
    explore_end = t0 + TL - 24.0
    trial = 0
    stagn = 0
    while time.time() < explore_end:
        trial += 1
        elapsed = time.time() - t0
        frac = elapsed / max(explore_end - t0, 1e-9)
        T = max(2e-6, 2.5e-3 * (1.0 - frac) + 2e-6)
        u = rng.random()
        prop = None
        try:
            if u < 0.38:
                # global kick + quench
                ls = rng.uniform(-2.6, -1.05)
                sigma = 10.0 ** ls
                x = cur + sigma * rng.standard_normal((68, 12))
                x /= np.linalg.norm(x, axis=1, keepdims=True)
                b1 = float(rng.choice([100.0, 140.0, 180.0]))
                x = _run_full(x, b1, 70)
                if time.time() > explore_end:
                    prop = x
                else:
                    x = _run_full(x, 380.0, 60)
                    prop = x
            elif u < 0.68:
                # subset reinit + joint opt
                m = int(rng.integers(3, 9))
                pm = _per_max(cur)
                nworst = min(68, m + 4)
                order = np.argsort(-pm)
                # take 2 worst + random from top-nworst + random overall
                idx = set()
                idx.add(int(order[0]))
                if m > 1:
                    idx.add(int(order[1]))
                while len(idx) < m:
                    if rng.random() < 0.7:
                        idx.add(int(order[int(rng.integers(0, nworst))]))
                    else:
                        idx.add(int(rng.integers(0, 68)))
                idx = np.array(sorted(idx), dtype=int)
                mask = np.ones(68, dtype=bool); mask[idx] = False
                fixed = cur[mask]
                Z0 = np.empty((m, 12))
                placed = []
                for k in range(m):
                    if len(placed) == 0:
                        Xr = fixed
                    else:
                        Xr = np.vstack([fixed, np.vstack(placed)])
                    h = _best_hole(Xr, rng, ncand=16000)
                    # perturb a bit
                    h = h + 0.02 * rng.standard_normal(12)
                    h /= np.linalg.norm(h)
                    Z0[k] = h
                    placed.append(h.reshape(1, 12))
                Z = _run_subset(fixed, Z0, 260.0, 120)
                # refine cross with higher beta
                Z = _run_subset(fixed, Z, 550.0, 60)
                Xp = np.empty((68, 12))
                Xp[mask] = fixed
                Xp[idx] = Z
                # light global relax
                Xp = _run_full(Xp, 220.0, 50)
                prop = Xp
            elif u < 0.86:
                # hole chain on worst points (coordinate descent)
                Xp = cur.copy()
                pm = _per_max(Xp)
                k = int(rng.integers(2, 5))
                order = np.argsort(-pm)[:k]
                for j in order:
                    mk = np.ones(68, dtype=bool); mk[j] = False
                    Xr = Xp[mk]
                    curm = float((Xr @ Xp[j]).max())
                    hole = _best_hole(Xr, rng, ncand=20000)
                    best_y = Xp[j].copy(); best_m = curm
                    starts = [Xp[j].copy(), hole]
                    pv = hole + 0.03 * rng.standard_normal(12)
                    pv /= np.linalg.norm(pv); starts.append(pv)
                    for s in starts:
                        for beta in (150.0, 500.0):
                            y = _opt_single(Xr, s.copy(), beta, maxiter=70)
                            mm = float((Xr @ y).max())
                            if mm < best_m:
                                best_m, best_y = mm, y
                    Xp[j] = best_y
                # global light touch
                Xp = _run_full(Xp, 300.0, 40)
                prop = Xp
            else:
                # fresh diverse restart with moderate beta ladder
                Xp = rng.standard_normal((68, 12))
                Xp /= np.linalg.norm(Xp, axis=1, keepdims=True)
                Xp = _run_full(Xp, 80.0, 70)
                Xp = _run_full(Xp, 250.0, 70)
                prop = Xp
        except Exception:
            continue
        if prop is None:
            continue
        prop = prop / np.linalg.norm(prop, axis=1, keepdims=True)
        pv = _max_cos(prop)
        # elitist update
        if pv < best_val - 1e-15:
            x_best = prop.copy(); best_val = pv
            cur = prop.copy(); cur_val = pv
            stagn = 0
        elif pv < cur_val:
            cur = prop.copy(); cur_val = pv
            stagn = 0
        else:
            stagn += 1
            if rng.random() < np.exp(-(pv - cur_val) / max(T, 1e-12)):
                cur = prop.copy(); cur_val = pv
            # reheat / restart from best if stuck
            if stagn > 35:
                # jump back to best with medium kick
                sigma = 0.015
                cur = x_best + sigma * rng.standard_normal((68, 12))
                cur /= np.linalg.norm(cur, axis=1, keepdims=True)
                cur_val = _max_cos(cur)
                stagn = 0
        if time.time() > explore_end:
            break
    # final intensification: L-BFGS ladder + LP
    try:
        if time.time() < t0 + TL - 20.0:
            x_best = _run_full(x_best, 350.0, 120)
            bv = _max_cos(x_best)
            if bv < best_val:
                best_val = bv
    except Exception:
        pass
    try:
        x_best = _lp_polish(x_best, t0 + TL - 1.0, r0=6e-4, cap_p=3000)
        best_val = _max_cos(x_best)
    except Exception:
        pass
    x_best = x_best / np.linalg.norm(x_best, axis=1, keepdims=True)
    return x_best
