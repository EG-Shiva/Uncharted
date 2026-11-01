import time
import numpy as np
from scipy.optimize import linprog, minimize
from scipy.sparse import coo_matrix

def _norm_rows(X):
    X = np.asarray(X, dtype=np.float64)
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[~np.isfinite(n)] = 1.0
    n[n == 0] = 1.0
    return X / n

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _row_max(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return G.max(axis=1)

def _quench_hb(X, steps, T0, T1, lr, mu, deadline=None, track=True):
    n, d = X.shape
    V = np.zeros_like(X)
    lt0 = np.log(T0); lt1 = np.log(T1)
    bv = 1e100; bX = None
    if track:
        G0 = X @ X.T; np.fill_diagonal(G0, -2.0)
        bv = float(G0.max()); bX = X.copy()
    for s in range(steps):
        if deadline is not None and (s % 300 == 0) and time.time() > deadline:
            break
        frac = s / max(1, steps - 1)
        T = float(np.exp(lt0 + (lt1 - lt0) * frac))
        G = X @ X.T
        np.fill_diagonal(G, -1e100)
        m = float(G.max())
        Q = (G - m) / T
        np.clip(Q, -50, 0, out=Q)
        E = np.exp(Q)
        np.fill_diagonal(E, 0.0)
        S = float(E.sum())
        if S == 0 or not np.isfinite(S):
            S = 1.0
        Grad = (E @ X) / (S * 0.5 + 1e-30) * 0.5
        # mix row-balanced gradient for rattlers
        rs = E.sum(axis=1, keepdims=True)
        rs[rs == 0] = 1.0
        GradRow = (E @ X) / rs
        Grad = 0.6 * Grad + 0.4 * GradRow
        dot = np.sum(Grad * X, axis=1, keepdims=True)
        Gt = Grad - dot * X
        gn = np.linalg.norm(Gt, axis=1, keepdims=True) + 1e-12
        bad = gn[:, 0] > 2.0
        if np.any(bad):
            Gt[bad] *= (2.0 / gn[bad])
        V = mu * V + Gt
        X = X - lr * V
        X = _norm_rows(X)
        if track and ((s % 120 == 0) or s == steps - 1):
            Gc = X @ X.T; np.fill_diagonal(Gc, -2.0)
            vc = float(Gc.max())
            if vc < bv:
                bv = vc; bX = X.copy()
    if track:
        return bX, bv
    return X, _max_cos(X)

def _quench_adam(X, steps, T0, T1, lr, deadline=None, b1=0.9, b2=0.999, eps=1e-8):
    n, d = X.shape
    Mt = np.zeros_like(X); Vt = np.zeros_like(X)
    lt0 = np.log(T0); lt1 = np.log(T1)
    G0 = X @ X.T; np.fill_diagonal(G0, -2.0)
    bv = float(G0.max()); bX = X.copy()
    for s in range(1, steps + 1):
        if deadline is not None and (s % 300 == 0) and time.time() > deadline:
            break
        frac = (s - 1) / max(1, steps - 1)
        T = float(np.exp(lt0 + (lt1 - lt0) * frac))
        G = X @ X.T
        np.fill_diagonal(G, -1e100)
        m = float(G.max())
        Q = (G - m) / T
        np.clip(Q, -50, 0, out=Q)
        E = np.exp(Q)
        np.fill_diagonal(E, 0.0)
        S = float(E.sum()) * 0.5 + 1e-30
        Grad = (E @ X) / (S * 2.0)
        dot = np.sum(Grad * X, axis=1, keepdims=True)
        Gt = Grad - dot * X
        Mt = b1 * Mt + (1 - b1) * Gt
        Vt = b2 * Vt + (1 - b2) * (Gt * Gt)
        ch1 = 1 - b1 ** s; ch2 = 1 - b2 ** s
        Mh = Mt / ch1; Vh = Vt / ch2
        Step = Mh / (np.sqrt(Vh) + eps)
        # limit step norm per row
        sn = np.linalg.norm(Step, axis=1, keepdims=True) + 1e-12
        Step = Step / np.maximum(1.0, sn / 3.0)
        X = X - lr * Step
        X = _norm_rows(X)
        if (s % 120 == 0) or s == steps:
            Gc = X @ X.T; np.fill_diagonal(Gc, -2.0)
            vc = float(Gc.max())
            if vc < bv:
                bv = vc; bX = X.copy()
    return bX, bv

def _lbfgs_lse(X0, T, maxiter=150):
    n, d = X0.shape
    Y0 = X0.ravel().copy()
    def fun_grad(Y):
        Ym = Y.reshape(n, d)
        nm = np.linalg.norm(Ym, axis=1, keepdims=True) + 1e-30
        X = Ym / nm
        G = X @ X.T
        iu = np.triu_indices(n, 1)
        g = G[iu]
        m = float(g.max())
        Q = (g - m) / T
        np.clip(Q, -50, 0, out=Q)
        E = np.exp(Q)
        S = float(E.sum()) + 1e-30
        L = m + T * np.log(S / len(g))
        # gradient w.r.t X
        # build full E matrix
        Ef = np.zeros((n, n))
        Ef[iu] = E
        Ef = Ef + Ef.T
        GradX = (Ef @ X) / S
        # chain to Y
        dot = np.sum(GradX * X, axis=1, keepdims=True)
        GradY = (GradX - dot * X) / nm
        return L, GradY.ravel()
    try:
        res = minimize(fun_grad, Y0, jac=True, method='L-BFGS-B',
                       options={'maxiter': maxiter, 'ftol': 1e-14, 'gtol': 1e-9, 'maxfun': maxiter * 2})
        Yf = res.x.reshape(n, d)
        Xf = _norm_rows(Yf)
        return Xf, _max_cos(Xf)
    except Exception:
        return X0, _max_cos(X0)

def _single_refine(s, R, steps=150, T=0.004, lr=0.12, mu=0.8):
    s = s / (np.linalg.norm(s) + 1e-12)
    v = np.zeros_like(s)
    for _ in range(steps):
        g = R @ s
        m = float(g.max())
        e = np.exp(np.clip((g - m) / T, -50, 0))
        den = float(e.sum()) + 1e-30
        Grad = (R.T @ e) / den
        Gt = Grad - float(Grad @ s) * s
        v = mu * v + Gt
        s = s - lr * v
        nn = float(np.linalg.norm(s))
        if not np.isfinite(nn) or nn == 0:
            break
        s = s / nn
    return s

def _find_hole(R, rng, n_sample=4000):
    d = R.shape[1]
    Z = rng.standard_normal(size=(n_sample, d))
    Z = _norm_rows(Z)
    S = Z @ R.T
    scores = S.max(axis=1)
    idx = np.argpartition(scores, 2)[:2]
    best_s = None; best_v = 1e100
    for k in idx:
        s0 = Z[k]
        s1 = _single_refine(s0, R, steps=150, T=0.005, lr=0.12)
        s1 = _single_refine(s1, R, steps=150, T=0.0015, lr=0.07)
        v = float((R @ s1).max())
        if v < best_v:
            best_v = v; best_s = s1
    return best_s / (np.linalg.norm(best_s) + 1e-12)

def _lp_polish(x, seconds, deadline=None, r0=2.5e-3):
    x = _norm_rows(x.copy())
    n, d = x.shape
    try:
        best = _max_cos(x)
    except Exception:
        return x
    radius = r0
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds:
        if deadline is not None and time.time() > deadline:
            break
        if radius < 1e-14:
            break
        G = x @ x.T
        cuv = G[iu]
        m = float(cuv.max())
        band = max(30.0 * radius, 2e-4)
        sel = cuv > m - band
        I = iu[0][sel]; J = iu[1][sel]
        p = len(I)
        if p == 0:
            break
        if p > 2200:
            order = np.argsort(-cuv[sel])[:2200]
            I, J = I[order], J[order]
            csel = cuv[sel][order]
        else:
            csel = cuv[sel]
            p = len(I)
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        except Exception:
            radius *= 0.5
            continue
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
        y = _norm_rows(y)
        if not np.all(np.isfinite(y)):
            radius *= 0.5
            continue
        v = _max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def solve():
    t0 = time.time()
    HARD_END = t0 + 174.0
    HOP_END = HARD_END - 34.0
    rng = np.random.default_rng(7)
    n, d = 71, 12
    best_X = None; best_s = 1e100
    def consider(X, s):
        global_best = False
        nonlocal best_X, best_s
        if not np.isfinite(s):
            return False
        X = _norm_rows(X)
        if s < best_s:
            best_s = float(s); best_X = X.copy()
            return True
        return False
    # load files
    loaded = []
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.asarray(np.load(fn), dtype=np.float64)
            if a.shape == (n, d) and np.all(np.isfinite(a)):
                a = _norm_rows(a)
                loaded.append(a)
                s = _max_cos(a)
                consider(a, s)
        except Exception:
            pass
    elites = []
    escores = []
    if best_X is not None:
        elites.append(best_X.copy()); escores.append(best_s)
    # diverse seeds (deliberately NOT starting near best)
    def rand_mat():
        return _norm_rows(rng.standard_normal((n, d)))
    def antipodal_mat():
        m = 35
        Y = _norm_rows(rng.standard_normal((m, d)))
        # push to hemisphere via random pole
        pole = _norm_rows(rng.standard_normal((1, d)))[0]
        for i in range(m):
            if float(Y[i] @ pole) < 0.02:
                Y[i] = _norm_rows((Y[i] - pole * float(Y[i] @ pole) + pole * 0.15)[None, :])[0]
                Y[i] = Y[i] / (np.linalg.norm(Y[i]) + 1e-12)
        Z1 = _norm_rows(rng.standard_normal((1, d)))[0]
        A = np.concatenate([Y, -Y, Z1[None, :]], axis=0)
        return _norm_rows(A + rng.standard_normal((n, d)) * 0.02)
    def cube_mat():
        # vertices of hypercube projected
        V = rng.choice([-1.0, 1.0], size=(n, d)).astype(float)
        V += rng.standard_normal((n, d)) * 0.35
        return _norm_rows(V)
    def perturbed_best(sigma):
        if best_X is None:
            return rand_mat()
        return _norm_rows(best_X + rng.standard_normal((n, d)) * sigma)
    seeds = []
    for _ in range(3):
        seeds.append(rand_mat())
    for _ in range(2):
        seeds.append(antipodal_mat())
    for _ in range(2):
        seeds.append(cube_mat())
    if best_X is not None:
        seeds.append(perturbed_best(0.55))
        seeds.append(perturbed_best(0.35))
    # quick quench each seed
    for X0 in seeds:
        if time.time() > HOP_END - 50:
            break
        T0 = float(10 ** rng.uniform(-2.4, -1.7))
        T1 = float(10 ** rng.uniform(-4.0, -3.2))
        lr = float(10 ** rng.uniform(-1.2, -0.7))
        mu = float(rng.uniform(0.82, 0.93))
        try:
            Xq, sq = _quench_hb(X0.copy(), steps=650, T0=T0, T1=T1, lr=lr, mu=mu, deadline=HOP_END)
        except Exception:
            continue
        consider(Xq, sq)
        elites.append(Xq.copy()); escores.append(float(sq))
    # keep diverse top 6
    if len(elites) > 0:
        od = np.argsort(np.array(escores))
        elites = [elites[i] for i in od[:6]]
        escores = [float(escores[i]) for i in od[:6]]
    else:
        elites = [rand_mat()]; escores = [0.5]
        best_X = elites[0].copy(); best_s = escores[0]
    tabu = set()
    def fingerprint(X):
        G = X @ X.T
        iu = np.triu_indices(n, 1)
        g = np.sort(G[iu])[-12:]
        return tuple(np.round(g, 5))
    it = 0
    while time.time() < HOP_END:
        it += 1
        tleft = HOP_END - time.time()
        if tleft < 1.0:
            break
        # pick base
        r = rng.random()
        if r < 0.45 or len(elites) == 1:
            base = best_X
        else:
            base = elites[int(rng.integers(0, len(elites)))]
        X0 = base.copy()
        u = rng.random()
        if u < 0.25:
            sigma = float(10 ** rng.uniform(-2.4, -0.4))
            X0 = base + rng.standard_normal((n, d)) * sigma
            X0 = _norm_rows(X0)
        elif u < 0.55:
            k = int(rng.integers(4, 10))
            direction = rng.standard_normal(d)
            direction /= (np.linalg.norm(direction) + 1e-12)
            if rng.random() < 0.5:
                direction = base[int(rng.integers(0, n))].copy() + rng.standard_normal(d) * 0.35
                direction /= (np.linalg.norm(direction) + 1e-12)
            proj = base @ direction
            idx = np.argpartition(-proj, k)[:k]
            keep = np.ones(n, dtype=bool); keep[idx] = False
            R = base[keep]
            for j in idx:
                if rng.random() < 0.35:
                    h = _norm_rows(rng.standard_normal((1, d)))[0]
                else:
                    try:
                        h = _find_hole(R, rng, n_sample=3500)
                    except Exception:
                        h = _norm_rows(rng.standard_normal((1, d)))[0]
                X0[j] = h
                R = np.concatenate([R, h[None, :]], axis=0)
            sig = float(10 ** rng.uniform(-2.6, -1.1))
            X0[keep] += rng.standard_normal((np.count_nonzero(keep), d)) * sig
            X0 = _norm_rows(X0)
        elif u < 0.80:
            k = 1 if rng.random() < 0.45 else (2 if rng.random() < 0.7 else 3)
            try:
                rm = _row_max(base)
                order = np.argsort(-rm + rng.random(n) * 0.002)
                idx = order[:k]
            except Exception:
                idx = rng.choice(n, size=k, replace=False)
            keep = np.ones(n, dtype=bool); keep[idx] = False
            R = base[keep]
            for j in idx:
                try:
                    h = _find_hole(R, rng, n_sample=4500)
                except Exception:
                    h = _norm_rows(rng.standard_normal((1, d)))[0]
                X0[j] = h
                R = np.concatenate([R, h[None, :]], axis=0)
            sig = float(10 ** rng.uniform(-2.6, -1.0))
            X0[keep] += rng.standard_normal((n - k, d)) * sig
            X0 = _norm_rows(X0)
        else:
            # subset rotation / swap topology change
            k = int(rng.integers(6, 16))
            idx = rng.choice(n, size=k, replace=False)
            if rng.random() < 0.5:
                # random orthogonal twist on subset
                A = rng.standard_normal((d, d))
                Q, _ = np.linalg.qr(A)
                if float(np.linalg.det(Q)) < 0:
                    Q[:, 0] = -Q[:, 0]
                ang = float(rng.uniform(0.1, 0.6))
                Rq = np.eye(d) * np.cos(ang) + Q * np.sin(ang) * 0.3
                # orthonormalize-ish: just mix
                X0[idx] = _norm_rows(X0[idx] + (X0[idx] @ Q.T) * ang * 0.5 + rng.standard_normal((k, d)) * 0.05)
            else:
                # swap + noise
                perm = idx[rng.permutation(k)]
                X0[idx] = base[perm] + rng.standard_normal((k, d)) * 0.08
                X0 = _norm_rows(X0)
        if tleft < 6:
            steps = 900
        elif tleft < 25:
            steps = 1500
        else:
            steps = 1800
            if rng.random() < 0.2:
                steps = 2600
        try:
            if rng.random() < 0.45:
                T0 = float(10 ** rng.uniform(-2.5, -1.8))
                T1 = float(10 ** rng.uniform(-4.2, -3.3))
                lr = float(10 ** rng.uniform(-3.2, -2.4)) if True else 0.002
                # adam lr small
                lr_a = float(10 ** rng.uniform(-3.0, -2.2))
                X1, s1 = _quench_adam(X0, steps=min(steps, 1600), T0=T0, T1=T1, lr=lr_a, deadline=HOP_END)
            else:
                T0 = float(10 ** rng.uniform(-2.5, -1.6))
                T1 = float(10 ** rng.uniform(-4.2, -3.1))
                lr = float(10 ** rng.uniform(-1.4, -0.7))
                mu = float(rng.uniform(0.82, 0.94))
                X1, s1 = _quench_hb(X0, steps=steps, T0=T0, T1=T1, lr=lr, mu=mu, deadline=HOP_END)
        except Exception:
            continue
        if not np.isfinite(s1):
            continue
        # tabu
        try:
            fp = fingerprint(X1)
            if fp in tabu:
                continue
            tabu.add(fp)
            if len(tabu) > 4000:
                tabu.clear()
        except Exception:
            pass
        improved = consider(X1, s1)
        # elite archive with diversity (chordal distance)
        if s1 < max(escores) + 2e-4 and s1 < 0.26:
            dup = False
            for e in elites:
                try:
                    c = float(np.abs((e * X1).sum(axis=1)).mean())
                    if c > 0.9995 and abs(_max_cos(e) - s1) < 2e-6:
                        dup = True
                        break
                except Exception:
                    pass
            if not dup:
                elites.append(X1.copy()); escores.append(float(s1))
                od = np.argsort(np.array(escores))
                elites = [elites[i] for i in od[:6]]
                escores = [escores[i] for i in od[:6]]
        # opportunistic LBFGS if near best
        if s1 < best_s + 3e-4 and tleft > 20 and (it % 4 == 0):
            try:
                Xl, sl = _lbfgs_lse(X1.copy(), T=6e-4, maxiter=80)
                consider(Xl, sl)
                if sl < max(escores):
                    elites.append(Xl.copy()); escores.append(float(sl))
                    od = np.argsort(np.array(escores))
                    elites = [elites[i] for i in od[:6]]
                    escores = [escores[i] for i in od[:6]]
            except Exception:
                pass
    # final polish on top elites
    try:
        od = np.argsort(np.array(escores))
        elites = [elites[i] for i in od]
        escores = [escores[i] for i in od]
    except Exception:
        elites = [best_X]; escores = [best_s]
    for ei in range(min(3, len(elites))):
        if time.time() > HARD_END - 6:
            break
        try:
            Xc = elites[ei].copy()
            # low-T adam squeeze
            Xc, sc = _quench_adam(Xc, steps=1400, T0=1.2e-3, T1=2.0e-4, lr=0.0018, deadline=HARD_END - 12)
            consider(Xc, sc)
            # LBFGS cascade
            for Tt, mi in ((8e-4, 120), (3e-4, 120), (1.2e-4, 80)):
                if time.time() > HARD_END - 10:
                    break
                Xc, sc = _lbfgs_lse(Xc, T=Tt, maxiter=mi)
                consider(Xc, sc)
            # LP
            tlp = min(6.0, max(1.0, (HARD_END - time.time()) / (3 - ei + 1) * 0.5))
            if time.time() + tlp < HARD_END - 1:
                Xc = _lp_polish(Xc, seconds=tlp, deadline=HARD_END - 1)
                consider(Xc, _max_cos(Xc))
            # tiny hill climb on worst points
            for _ in range(120):
                if time.time() > HARD_END - 1.5:
                    break
                try:
                    rm = _row_max(Xc)
                    j = int(np.argmax(rm))
                    prop = Xc.copy()
                    tang = rng.standard_normal(d)
                    tang -= float(tang @ prop[j]) * prop[j]
                    tn = float(np.linalg.norm(tang))
                    if tn == 0:
                        continue
                    tang /= tn
                    step = float(rng.uniform(1e-4, 1.5e-3))
                    prop[j] = prop[j] + tang * step
                    prop[j] /= (np.linalg.norm(prop[j]) + 1e-12)
                    v = _max_cos(prop)
                    if v < sc:
                        Xc = prop; sc = v
                        consider(Xc, sc)
                except Exception:
                    break
        except Exception:
            continue
    best_X = _norm_rows(best_X)
    # tiny shrink to guarantee strict feasibility / no numerical edge
    return np.ascontiguousarray(best_X, dtype=np.float64)
