import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _adam_quench(X, steps, beta0, beta1, lr0, lr1, rownorm):
    n, d = X.shape
    m = np.zeros_like(X)
    v = np.zeros_like(X)
    b1 = 0.9; b2 = 0.999; eps = 1e-8
    lb0 = np.log(beta0); lb1 = np.log(beta1)
    ll0 = np.log(lr0); ll1 = np.log(lr1)
    for t in range(1, steps + 1):
        frac = t / steps
        beta = np.exp(lb0 + (lb1 - lb0) * frac)
        lr = np.exp(ll0 + (ll1 - ll0) * frac)
        G = X @ X.T
        np.fill_diagonal(G, -10.0)
        gmax = G.max()
        W = np.exp(beta * (G - gmax))
        np.fill_diagonal(W, 0.0)
        if rownorm:
            S = W.sum(axis=1, keepdims=True) + 1e-12
            Grad = (W @ X) / S
        else:
            Z = W.sum()
            if Z == 0:
                continue
            Grad = (W @ X) / Z
        dot = (Grad * X).sum(axis=1, keepdims=True)
        Gt = Grad - dot * X
        m *= b1; m += (1.0 - b1) * Gt
        v *= b2; v += (1.0 - b2) * (Gt * Gt)
        bc1 = 1.0 - b1 ** t; bc2 = 1.0 - b2 ** t
        mhat = m / bc1; vhat = v / bc2
        step = mhat / (np.sqrt(vhat) + eps)
        X = X - lr * step
        nrm = np.sqrt((X * X).sum(axis=1, keepdims=True))
        nrm[nrm == 0] = 1.0
        X /= nrm
    return X

def _hard_descent(X, steps=250, lr0=0.003, lr1=0.00015, tol0=3e-4, tol1=2e-6):
    n, d = X.shape
    for s in range(steps):
        frac = s / max(steps - 1, 1)
        tol = tol0 + (tol1 - tol0) * frac
        lr = lr0 * (lr1 / lr0) ** frac if lr0 > 0 and lr1 > 0 else lr0
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        mx = G.max()
        M = (G > (mx - tol)).astype(np.float64)
        cnt = M.sum(axis=1, keepdims=True) + 1e-12
        Grad = (M @ X) / cnt
        dot = (Grad * X).sum(axis=1, keepdims=True)
        Gt = Grad - dot * X
        nn = np.sqrt((Gt * Gt).sum(axis=1, keepdims=True)) + 1e-12
        active = (cnt[:, 0] > 0.5)
        step = np.zeros_like(Gt)
        step[active] = Gt[active] / nn[active]
        X = X - lr * step
        nrm = np.sqrt((X * X).sum(axis=1, keepdims=True))
        nrm[nrm == 0] = 1.0
        X /= nrm
    return X

def _refine_hole(Y, p, iters=180, beta=90.0, lr0=0.12, lr1=0.004):
    p = p / np.linalg.norm(p)
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

def _find_voids(Y, rng, n_probe=12000, n_top=6):
    d = Y.shape[1]
    P = rng.randn(n_probe, d)
    P /= np.linalg.norm(P, axis=1, keepdims=True)
    CH = 4000
    bs = np.empty(n_probe)
    for a in range(0, n_probe, CH):
        b = min(a + CH, n_probe)
        D = P[a:b] @ Y.T
        bs[a:b] = D.max(axis=1)
    order = np.argsort(bs)
    cands = []
    for idx in order[:max(n_top * 8, 40)]:
        p = P[idx].copy()
        ok = True
        for q in cands:
            if abs(float(p @ q)) > 0.90:
                ok = False
                break
        if ok:
            cands.append(p)
        if len(cands) >= n_top:
            break
    while len(cands) < n_top:
        cands.append(P[order[len(cands)]].copy())
    out = []
    for p in cands[:n_top]:
        q = _refine_hole(Y, p, iters=180, beta=80.0 + rng.rand() * 40.0,
                         lr0=0.10 + rng.rand() * 0.06, lr1=0.003)
        out.append(q)
    return out

def _joint_reopt(Y, P, steps=320, beta0=25.0, beta1=320.0, lr0=0.06, lr1=0.004):
    k, d = P.shape
    m = Y.shape[0]
    Mt = np.zeros_like(P); Vt = np.zeros_like(P)
    b1 = 0.9; b2 = 0.999; eps = 1e-8
    lb0 = np.log(beta0); lb1 = np.log(beta1)
    ll0 = np.log(lr0); ll1 = np.log(lr1)
    for t in range(1, steps + 1):
        frac = t / steps
        beta = np.exp(lb0 + (lb1 - lb0) * frac)
        lr = np.exp(ll0 + (ll1 - ll0) * frac)
        A = P @ Y.T
        B = P @ P.T
        np.fill_diagonal(B, -10.0)
        gmax = max(float(A.max()), float(B.max()))
        WA = np.exp(beta * (A - gmax))
        WB = np.exp(beta * (B - gmax))
        S = WA.sum(axis=1, keepdims=True) + WB.sum(axis=1, keepdims=True) + 1e-12
        Grad = (WA @ Y + WB @ P) / S
        dot = (Grad * P).sum(axis=1, keepdims=True)
        Gt = Grad - dot * P
        Mt *= b1; Mt += (1.0 - b1) * Gt
        Vt *= b2; Vt += (1.0 - b2) * (Gt * Gt)
        mhat = Mt / (1.0 - b1 ** t); vhat = Vt / (1.0 - b2 ** t)
        P = P - lr * mhat / (np.sqrt(vhat) + eps)
        P /= np.linalg.norm(P, axis=1, keepdims=True)
    return P

def _lp_polish(x, seconds, r0=1e-3):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = r0
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        c = (x @ x.T)[iu]
        mval = c.max()
        sel = c > mval - max(20 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        if p == 0:
            break
        if p > 2200:
            idx = np.argsort(c[sel])[-2200:]
            I, J = I[idx], J[idx]
            cc = c[sel][idx]
            p = 2200
        else:
            cc = c[sel]
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        # fix vals construction
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d),
                                       np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-cc, A_eq=A_eq,
                          b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                          method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        vv = _max_cos(y)
        if vv < best:
            x, best, radius = y, vv, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def _build_Rpowers(m, freqs):
    Rp = np.zeros((m, 8, 8))
    for k in range(m):
        R = np.zeros((8, 8))
        for pl in range(4):
            th = 2.0 * np.pi * freqs[pl] * k / m
            c = np.cos(th); s = np.sin(th)
            R[2*pl, 2*pl] = c; R[2*pl, 2*pl+1] = -s
            R[2*pl+1, 2*pl] = s; R[2*pl+1, 2*pl+1] = c
        Rp[k] = R
    return Rp

def _orbit_seed_quench(rng, m_choice=7, steps=550):
    m = m_choice
    ns = 168 // m
    freqs = rng.randint(1, m, size=4).astype(float)
    # avoid zero freq (randint 1..m-1 already); ensure non-degenerate
    Rp = _build_Rpowers(m, freqs)
    Rt = Rp.transpose(0, 2, 1)
    Z = rng.randn(ns, 8)
    Z /= np.linalg.norm(Z, axis=1, keepdims=True)
    Mt = np.zeros_like(Z); Vt = np.zeros_like(Z)
    b1 = 0.9; b2 = 0.999; eps = 1e-8
    lb0 = np.log(25.0); lb1 = np.log(380.0)
    ll0 = np.log(0.02); ll1 = np.log(0.002)
    X = np.zeros((168, 8))
    for t in range(1, steps + 1):
        frac = t / steps
        beta = np.exp(lb0 + (lb1 - lb0) * frac)
        lr = np.exp(ll0 + (ll1 - ll0) * frac)
        for s in range(ns):
            zs = Z[s]
            for k in range(m):
                X[s*m+k] = Rp[k] @ zs
        G = X @ X.T
        np.fill_diagonal(G, -10.0)
        gmax = G.max()
        W = np.exp(beta * (G - gmax))
        np.fill_diagonal(W, 0.0)
        S = W.sum(axis=1, keepdims=True) + 1e-12
        Grad = (W @ X) / S
        Gz = np.zeros_like(Z)
        for s in range(ns):
            acc = np.zeros(8)
            for k in range(m):
                acc += Rt[k] @ Grad[s*m+k]
            Gz[s] = acc / m
        dot = (Gz * Z).sum(axis=1, keepdims=True)
        Gt = Gz - dot * Z
        Mt *= b1; Mt += (1.0 - b1) * Gt
        Vt *= b2; Vt += (1.0 - b2) * (Gt * Gt)
        mhat = Mt / (1.0 - b1 ** t); vhat = Vt / (1.0 - b2 ** t)
        Z = Z - lr * mhat / (np.sqrt(vhat) + eps)
        Z /= np.linalg.norm(Z, axis=1, keepdims=True)
    for s in range(ns):
        for k in range(m):
            X[s*m+k] = Rp[k] @ Z[s]
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    return X

def _antipodal_quench(rng, steps=650):
    ns = 84
    Z = rng.randn(ns, 8)
    Z /= np.linalg.norm(Z, axis=1, keepdims=True)
    Mt = np.zeros_like(Z); Vt = np.zeros_like(Z)
    b1 = 0.9; b2 = 0.999; eps = 1e-8
    lb0 = np.log(30.0); lb1 = np.log(450.0)
    ll0 = np.log(0.015); ll1 = np.log(0.0015)
    for t in range(1, steps + 1):
        frac = t / steps
        beta = np.exp(lb0 + (lb1 - lb0) * frac)
        lr = np.exp(ll0 + (ll1 - ll0) * frac)
        X = np.vstack([Z, -Z])
        G = X @ X.T
        np.fill_diagonal(G, -10.0)
        gmax = G.max()
        W = np.exp(beta * (G - gmax))
        np.fill_diagonal(W, 0.0)
        S = W.sum(axis=1, keepdims=True) + 1e-12
        Grad = (W @ X) / S
        Gtop = Grad[:ns]; Gbot = Grad[ns:]
        Gz = Gtop - Gbot
        dot = (Gz * Z).sum(axis=1, keepdims=True)
        Gt = Gz - dot * Z
        Mt *= b1; Mt += (1.0 - b1) * Gt
        Vt *= b2; Vt += (1.0 - b2) * (Gt * Gt)
        mhat = Mt / (1.0 - b1 ** t); vhat = Vt / (1.0 - b2 ** t)
        Z = Z - lr * mhat / (np.sqrt(vhat) + eps)
        Z /= np.linalg.norm(Z, axis=1, keepdims=True)
    X = np.vstack([Z, -Z])
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    return X

def solve():
    t0 = time.time()
    BUDGET = 175.0
    PHASE1 = 138.0
    best = None
    best_val = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn).astype(np.float64)
            if a.shape != (168, 8):
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
        r0 = np.random.RandomState(11)
        best = r0.randn(168, 8)
        best /= np.linalg.norm(best, axis=1, keepdims=True)
        best_val = _max_cos(best)
    n, d = best.shape
    rng = np.random.RandomState(987654321)
    elite_vals = [best_val]
    elite_arrs = [best.copy()]
    def elite_add(v, X):
        for ev in elite_vals:
            if abs(v - ev) < 1e-12:
                return
        if len(elite_vals) < 5:
            elite_vals.append(v); elite_arrs.append(X.copy())
        else:
            k = int(np.argmax(elite_vals))
            if v < elite_vals[k]:
                elite_vals[k] = v; elite_arrs[k] = X.copy()
    it = 0
    stall = 0
    deadline1 = t0 + PHASE1
    m_choices = [7, 12, 14, 21, 24]
    while time.time() < deadline1:
        if time.time() + 3.0 > t0 + BUDGET:
            break
        it += 1
        u = rng.rand()
        if rng.rand() < 0.80:
            X0 = best.copy()
        else:
            X0 = elite_arrs[rng.randint(len(elite_arrs))].copy()
        X = X0.copy()
        r = rng.rand()
        if r < 0.30:
            # global kick + balanced quench
            sigmas = (0.006, 0.012, 0.025, 0.05)
            sigma = sigmas[rng.randint(len(sigmas))]
            if stall > 20:
                sigma *= 1.6
            X = X + rng.randn(n, d) * sigma
            X /= np.linalg.norm(X, axis=1, keepdims=True)
            beta0 = 30.0 + rng.rand() * 30.0
            beta1 = 280.0 + rng.rand() * 520.0
            steps = 850 + int(rng.randint(850))
            lr0 = 0.003 + rng.rand() * 0.005
            rownorm = bool(rng.rand() < 0.72)
            Xq = _adam_quench(X, steps, beta0, beta1, lr0, lr0 * 0.10, rownorm)
            Xq = _hard_descent(Xq, steps=220, lr0=0.003, lr1=0.00015)
        elif r < 0.58:
            # joint multi-void reinsertion of worst + rattlers
            G = X @ X.T
            np.fill_diagonal(G, -2.0)
            rowmax = G.max(axis=1)
            k = 2 + int(rng.randint(3))  # 2..4
            nw = max(1, k // 2)
            idx_worst = np.argsort(rowmax)[-nw:]
            idx_rattle = np.argsort(rowmax)[:max(1, k - nw)]
            # occasionally add random
            if rng.rand() < 0.25:
                idx_rattle = np.append(idx_rattle, rng.randint(n))
                idx_rattle = np.unique(idx_rattle)[:k]
            idx = np.unique(np.concatenate([idx_worst, idx_rattle]))
            while len(idx) < k:
                idx = np.unique(np.append(idx, rng.randint(n)))
            idx = idx[:k]
            keep = np.ones(n, dtype=bool); keep[idx] = False
            Y = X[keep]
            voids = _find_voids(Y, rng, n_probe=9000 + int(rng.randint(6000)), n_top=max(6, k+2))
            P0 = np.array(voids[:k])
            # diversify: if k>1 ensure spread (already diverse)
            Pj = _joint_reopt(Y, P0, steps=280 + int(rng.randint(150)),
                              beta0=25.0, beta1=300.0 + rng.rand()*200.0,
                              lr0=0.06, lr1=0.004)
            X[idx] = Pj
            X /= np.linalg.norm(X, axis=1, keepdims=True)
            beta0 = 35.0
            beta1 = 350.0 + rng.rand() * 450.0
            steps = 950 + int(rng.randint(850))
            lr0 = 0.004 + rng.rand() * 0.004
            rownorm = bool(rng.rand() < 0.7)
            Xq = _adam_quench(X, steps, beta0, beta1, lr0, lr0 * 0.12, rownorm)
            Xq = _hard_descent(Xq, steps=260, lr0=0.0032, lr1=0.00014)
        elif r < 0.74:
            # hub shatter
            G = X @ X.T
            np.fill_diagonal(G, -2.0)
            mm = G.max()
            deg = (G > mm - 0.002).sum(axis=1)
            hub = int(np.argmax(deg))
            if deg[hub] < 3:
                hub = int(np.argmax(G.max(axis=1)))
            order = np.argsort(-G[hub])
            nb = int(min(n - 1, max(5, deg[hub] + 3)))
            idx = order[:nb]
            idx = np.unique(np.concatenate([[hub], idx]))
            sigma = 0.035 + rng.rand() * 0.05
            noise = np.zeros((n, d))
            noise[idx] = rng.randn(len(idx), d) * sigma
            if rng.rand() < 0.4:
                rm = G.max(axis=1)
                jr = int(np.argmin(rm))
                others = np.delete(X, jr, axis=0)
                try:
                    y = _find_voids(others, rng, n_probe=6000, n_top=3)[0]
                    X[jr] = y
                except Exception:
                    pass
            X = X + noise
            X /= np.linalg.norm(X, axis=1, keepdims=True)
            beta0 = 30.0 + rng.rand() * 25.0
            beta1 = 300.0 + rng.rand() * 500.0
            steps = 1000 + int(rng.randint(800))
            lr0 = 0.004 + rng.rand() * 0.004
            Xq = _adam_quench(X, steps, beta0, beta1, lr0, lr0 * 0.10, bool(rng.rand() < 0.6))
            Xq = _hard_descent(Xq, steps=240, lr0=0.003, lr1=0.00014)
        elif r < 0.90:
            # orbit-symmetric trial then relax
            mc = m_choices[rng.randint(len(m_choices))]
            try:
                Xo = _orbit_seed_quench(rng, m_choice=mc, steps=480 + int(rng.randint(250)))
            except Exception:
                Xo = best.copy() + rng.randn(n, d) * 0.02
                Xo /= np.linalg.norm(Xo, axis=1, keepdims=True)
            # break symmetry slightly
            Xo = Xo + rng.randn(n, d) * (0.004 + rng.rand() * 0.008)
            Xo /= np.linalg.norm(Xo, axis=1, keepdims=True)
            beta0 = 35.0; beta1 = 380.0 + rng.rand() * 400.0
            steps = 900 + int(rng.randint(700))
            lr0 = 0.004 + rng.rand() * 0.003
            Xq = _adam_quench(Xo, steps, beta0, beta1, lr0, lr0 * 0.10, True)
            Xq = _hard_descent(Xq, steps=260, lr0=0.003, lr1=0.00013)
        else:
            # antipodal restart then relax
            try:
                Xa = _antipodal_quench(rng, steps=550 + int(rng.randint(250)))
            except Exception:
                Xa = rng.randn(n, d)
                Xa /= np.linalg.norm(Xa, axis=1, keepdims=True)
            Xa = Xa + rng.randn(n, d) * (0.003 + rng.rand() * 0.007)
            Xa /= np.linalg.norm(Xa, axis=1, keepdims=True)
            beta0 = 35.0; beta1 = 380.0 + rng.rand() * 400.0
            steps = 900 + int(rng.randint(700))
            lr0 = 0.004 + rng.rand() * 0.003
            Xq = _adam_quench(Xa, steps, beta0, beta1, lr0, lr0 * 0.10, True)
            Xq = _hard_descent(Xq, steps=260, lr0=0.003, lr1=0.00013)
        if not np.all(np.isfinite(Xq)):
            stall += 1
            continue
        vq = _max_cos(Xq)
        if vq < best_val:
            best_val = float(vq)
            best = Xq.copy()
            elite_add(best_val, best)
            stall = 0
            # quick LP squeeze on new best if time allows
            if time.time() + 6.0 < t0 + BUDGET and time.time() < deadline1 - 4:
                try:
                    Xs = _lp_polish(best, 1.2, r0=6e-4)
                    vs = _max_cos(Xs)
                    if vs < best_val and np.all(np.isfinite(Xs)):
                        best_val = float(vs); best = Xs.copy()
                        elite_add(best_val, best)
                except Exception:
                    pass
        elif vq < best_val + 3e-4:
            elite_add(float(vq), Xq)
            if (it % 10 == 0) and (time.time() + 5.0 < t0 + BUDGET) and (time.time() < deadline1 - 5):
                try:
                    Xs = _lp_polish(Xq, 0.9, r0=5e-4)
                    vs = _max_cos(Xs)
                    if vs < best_val and np.all(np.isfinite(Xs)):
                        best_val = float(vs); best = Xs.copy()
                        elite_add(best_val, best)
                        stall = 0
                    elif vs < best_val + 1e-4:
                        elite_add(float(vs), Xs)
                except Exception:
                    pass
            stall += 1
        else:
            stall += 1
    # final squeeze
    remain = (t0 + BUDGET) - time.time() - 1.0
    if remain > 4.0:
        try:
            best = _hard_descent(best, steps=300, lr0=0.002, lr1=0.0001, tol0=2e-4, tol1=1e-6)
            bv = _max_cos(best)
            if bv < best_val:
                best_val = float(bv)
        except Exception:
            pass
        try:
            lp_time = min(remain - 1.0, 22.0)
            if lp_time > 2.0:
                Xp = _lp_polish(best, lp_time, r0=8e-4)
                vp = _max_cos(Xp)
                if vp < best_val and np.all(np.isfinite(Xp)):
                    best = Xp; best_val = float(vp)
        except Exception:
            pass
        # tiny hill climb around best
        try:
            for _ in range(6):
                if time.time() + 2.0 > t0 + BUDGET:
                    break
                cand = best + rng.randn(n, d) * 0.002
                cand /= np.linalg.norm(cand, axis=1, keepdims=True)
                cand = _adam_quench(cand, 350, 60.0, 500.0, 0.003, 0.0003, True)
                cand = _hard_descent(cand, steps=150, lr0=0.002, lr1=0.0001)
                vc = _max_cos(cand)
                if vc < best_val:
                    # squeeze
                    try:
                        cs = _lp_polish(cand, 1.0, r0=4e-4)
                        vs2 = _max_cos(cs)
                        if vs2 < vc and np.all(np.isfinite(cs)):
                            cand, vc = cs, vs2
                    except Exception:
                        pass
                    if vc < best_val:
                        best = cand; best_val = float(vc)
                else:
                    break
        except Exception:
            pass
    best = best / np.linalg.norm(best, axis=1, keepdims=True)
    return best
