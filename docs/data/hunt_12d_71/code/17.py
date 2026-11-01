import time
import numpy as np
from scipy.optimize import linprog, minimize
from scipy.sparse import coo_matrix

def _norm_rows(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return x / n

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _relax(X, steps, T0, T1, lr, mu=0.9, t_deadline=None, track_best=False):
    n, d = X.shape
    V = np.zeros_like(X)
    lt0 = np.log(T0); lt1 = np.log(T1)
    bestV = _max_cos(X) if track_best else None
    bestX = X.copy() if track_best else None
    for s in range(steps):
        if t_deadline is not None and (s % 300 == 0) and time.time() > t_deadline:
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
        rs = E.sum(axis=1, keepdims=True)
        rs[rs == 0] = 1.0
        Grad = (E @ X) / rs
        dot = np.sum(Grad * X, axis=1, keepdims=True)
        Gt = Grad - dot * X
        gn = np.linalg.norm(Gt, axis=1, keepdims=True) + 1e-12
        sc = np.ones_like(gn)
        mk = (gn[:, 0] > 2.0)
        sc[mk] = 2.0 / gn[mk]
        Gt = Gt * sc
        V = mu * V + Gt
        X = X - lr * V
        X = _norm_rows(X)
        if track_best and (s % 250 == 0 or s == steps - 1):
            v = _max_cos(X)
            if v < bestV:
                bestV = v
                bestX = X.copy()
    if track_best:
        return bestX, bestV
    return X

def _single_refine(s, R, steps=250, T=0.004, lr=0.12, mu=0.8):
    s = s / (np.linalg.norm(s) + 1e-12)
    v = np.zeros_like(s)
    for _ in range(steps):
        g = R @ s
        m = float(g.max())
        e = np.exp(np.clip((g - m) / T, -50, 0))
        den = e.sum() + 1e-30
        Grad = (R.T @ e) / den
        Gt = Grad - float(Grad @ s) * s
        v = mu * v + Gt
        s = s - lr * v
        s = s / (np.linalg.norm(s) + 1e-12)
    return s

def _find_hole(R, rng, n_sample=10000, n_cand=3):
    n, d = R.shape
    Z = _norm_rows(rng.standard_normal(size=(n_sample, d)))
    S = Z @ R.T
    scores = S.max(axis=1)
    idx = np.argpartition(scores, n_cand)[:n_cand]
    best_s = None
    best_v = 1e100
    for k in idx:
        s0 = Z[k]
        s1 = _single_refine(s0, R, steps=220, T=0.005, lr=0.12)
        s1 = _single_refine(s1, R, steps=220, T=0.0015, lr=0.07)
        v = float((R @ s1).max())
        if v < best_v:
            best_v = v
            best_s = s1
    return best_s / (np.linalg.norm(best_s) + 1e-12)

def _lp_polish(x, seconds, t_hard=None):
    x = _norm_rows(x.copy())
    n, d = x.shape
    best = _max_cos(x)
    radius = 2e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while True:
        if time.time() - start >= seconds:
            break
        if t_hard is not None and time.time() > t_hard:
            break
        if radius < 1e-14:
            break
        cmat = (x @ x.T)[iu]
        m = float(cmat.max())
        sel = cmat > m - max(20 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        if p == 0:
            break
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cc = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vv = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vv, (rows, cc)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        except Exception:
            break
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-cmat[sel], A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y = _norm_rows(y)
        v = _max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def _lbfgs_smooth(X0, betas, iters):
    n, d = X0.shape
    iu = np.triu_indices(n, 1)
    Y = _norm_rows(X0.copy())
    for beta, it in zip(betas, iters):
        def fun(Yflat, beta=beta):
            Ym = Yflat.reshape(n, d)
            rn = np.linalg.norm(Ym, axis=1, keepdims=True)
            rn = np.maximum(rn, 1e-12)
            X = Ym / rn
            G = X @ X.T
            gv = G[iu]
            gm = gv.max()
            ee = np.exp(beta * (gv - gm))
            S = ee.sum()
            F = gm + np.log(S) / beta
            w = ee / S
            W = np.zeros((n, n))
            W[iu] = w
            W = W + W.T
            gradX = W @ X
            dot = (gradX * X).sum(axis=1, keepdims=True)
            gradY = (gradX - dot * X) / rn
            r1 = rn.squeeze() - 1.0
            pen = float((r1 ** 2).mean())
            Ftot = float(F) + pen
            gradY = gradY + (2.0 / n) * r1[:, None] * X
            return Ftot, gradY.ravel()
        try:
            res = minimize(fun, Y.ravel(), method='L-BFGS-B', jac=True,
                           options={'maxiter': int(it), 'ftol': 1e-12, 'gtol': 1e-7, 'maxcor': 12})
            Y = res.x.reshape(n, d)
        except Exception:
            pass
        Y = _norm_rows(Y)
    return _norm_rows(Y)

def solve():
    t0 = time.time()
    TIME_LIMIT = 173.0
    HARD_END = t0 + TIME_LIMIT
    rng = np.random.default_rng(0)
    cands = []
    for fn in ("best.npy", "record.npy"):
        try:
            a = _norm_rows(np.asarray(np.load(fn), dtype=np.float64))
            if a.shape == (71, 12) and np.all(np.isfinite(a)):
                cands.append(a)
        except Exception:
            pass
    if not cands:
        cands.append(_norm_rows(rng.standard_normal(size=(71, 12))))
    bestX = min(cands, key=_max_cos).copy()
    bestVal = _max_cos(bestX)
    curX = bestX.copy(); curVal = bestVal
    archive = [(bestVal, bestX.copy())]
    def push(v, X):
        archive.append((v, X.copy()))
        archive.sort(key=lambda t: t[0])
        del archive[5:]
    explore_dead = t0 + 118.0
    bh_T = 0.0012
    it = 0
    while time.time() < explore_dead:
        it += 1
        if time.time() > explore_dead:
            break
        mode = it % 6
        if mode == 0 or mode == 3:
            # hole reinsertion of overloaded points
            base = bestX if rng.random() < 0.75 else curX
            bv = bestVal if rng.random() < 0.75 else curVal
            G = base @ base.T
            np.fill_diagonal(G, -2.0)
            m = float(G.max())
            # soft overload score
            Tc = 0.003
            W = np.exp(np.clip((G - m) / Tc, -50, 0))
            score = W.sum(axis=1)
            order = np.argsort(-score)
            topk = order[:10]
            nrem = 2 if (it % 12 == 0) else 1
            victims = []
            # pick worst + random among top
            victims.append(int(topk[0]))
            for _ in range(nrem - 1):
                victims.append(int(topk[int(rng.integers(1, 6))]))
            victims = sorted(set(victims))
            mask = np.ones(71, dtype=bool); mask[victims] = False
            Rb = base[mask]
            for _ in range(len(victims)):
                s_new = _find_hole(Rb, rng, n_sample=10000, n_cand=3)
                Rb = np.vstack([Rb, s_new[None, :]])
            Xt = Rb
            T0 = float(rng.choice([0.008, 0.012, 0.006, 0.015]))
            T1 = float(rng.choice([0.0008, 0.0004, 0.0012, 0.0006]))
            lr = float(rng.choice([0.04, 0.06, 0.03, 0.05]))
            steps = int(rng.choice([1600, 2200, 3000, 1900]))
            Xt = _relax(Xt, steps=steps, T0=T0, T1=T1, lr=lr, mu=0.9, t_deadline=explore_dead)
        elif mode == 5:
            # coherent cap move
            base = bestX if rng.random() < 0.7 else curX
            Xt = base.copy()
            u = _norm_rows(rng.standard_normal(size=(1, 12)))[0]
            dots = Xt @ u
            k = int(rng.choice([3, 4, 5, 6]))
            cap = np.argsort(-dots)[:k]
            sig_c = float(np.exp(rng.uniform(np.log(0.02), np.log(0.15))))
            sig_i = float(np.exp(rng.uniform(np.log(0.005), np.log(0.05))))
            common = rng.standard_normal(size=(12,)) * sig_c
            Xt[cap] = Xt[cap] + common[None, :] + rng.standard_normal(size=(k, 12)) * sig_i
            Xt = _norm_rows(Xt)
            T0 = float(rng.choice([0.01, 0.007, 0.014]))
            T1 = float(rng.choice([0.0005, 0.001, 0.0003]))
            lr = float(rng.choice([0.05, 0.03, 0.07]))
            steps = int(rng.choice([1500, 2200, 2800]))
            Xt = _relax(Xt, steps=steps, T0=T0, T1=T1, lr=lr, mu=0.9, t_deadline=explore_dead)
        else:
            base = bestX if rng.random() < 0.7 else curX
            Xt = base.copy()
            G = base @ base.T
            np.fill_diagonal(G, -2.0)
            m = float(G.max())
            cnt = (G > m - 0.01).sum(axis=1).astype(float) + 0.5
            prob = cnt / cnt.sum()
            k = int(rng.choice([1, 2, 2, 3, 4]))
            idx = rng.choice(71, size=k, replace=False, p=prob)
            sigma = float(np.exp(rng.uniform(np.log(0.015), np.log(0.45))))
            if rng.random() < 0.06:
                Xt[idx] = _norm_rows(rng.standard_normal(size=(k, 12)))
            else:
                Xt[idx] = Xt[idx] + rng.standard_normal(size=(k, 12)) * sigma
                Xt = _norm_rows(Xt)
            T0 = float(rng.choice([0.01, 0.007, 0.015]))
            T1 = float(rng.choice([0.0005, 0.001, 0.0003]))
            lr = float(rng.choice([0.05, 0.03, 0.07]))
            steps = int(rng.choice([1400, 2000, 2600]))
            Xt = _relax(Xt, steps=steps, T0=T0, T1=T1, lr=lr, mu=0.9, t_deadline=explore_dead)
        v = _max_cos(Xt)
        if v < bestVal + 0.003 and time.time() < explore_dead - 0.5:
            Xt2 = _relax(Xt.copy(), steps=500, T0=0.0008, T1=0.0002, lr=0.02, mu=0.9, t_deadline=explore_dead)
            v2 = _max_cos(Xt2)
            if v2 < v:
                Xt, v = Xt2, v2
        if v < bestVal:
            bestVal = v; bestX = Xt.copy()
            curX = Xt.copy(); curVal = v
            push(v, Xt)
        else:
            if v < bestVal + 0.004:
                if len(archive) < 5 or v < archive[-1][0]:
                    push(v, Xt)
            if v < curVal or rng.random() < np.exp(-(v - curVal) / bh_T):
                curX = Xt.copy(); curVal = v
        if it % 30 == 0:
            curX = bestX.copy(); curVal = bestVal
    # archive tighten: L-BFGS + LP, best first
    arch = sorted(archive, key=lambda t: t[0])[:4]
    remain = HARD_END - time.time()
    if remain > 16 and len(arch) > 0:
        per = min(6.0, (remain - 11.0) / max(1, len(arch)))
        per = max(per, 1.0)
        for av, ax in list(arch):
            if time.time() > HARD_END - 12:
                break
            y = _relax(ax.copy(), steps=1200, T0=0.001, T1=0.00015, lr=0.02, mu=0.9, t_deadline=HARD_END - 11)
            try:
                y = _lbfgs_smooth(y, betas=[800, 2500, 6000], iters=[120, 120, 120])
            except Exception:
                pass
            vy = _max_cos(y)
            y2 = _lp_polish(y if vy < av else ax.copy(), seconds=max(0.8, per * 0.5), t_hard=HARD_END - 11)
            v2 = _max_cos(y2)
            cand_v, cand_x = (v2, y2) if v2 < vy else (vy, y)
            if cand_v < bestVal:
                bestVal = cand_v; bestX = cand_x.copy()
    if time.time() < HARD_END - 2:
        y = _relax(bestX.copy(), steps=2000, T0=0.0012, T1=0.00012, lr=0.018, mu=0.9, t_deadline=HARD_END - 8)
        try:
            y = _lbfgs_smooth(y, betas=[1000, 3000, 8000], iters=[150, 150, 150])
        except Exception:
            pass
        vy = _max_cos(y)
        if vy < bestVal:
            bestVal = vy; bestX = y.copy()
        rem2 = HARD_END - time.time()
        if rem2 > 2:
            y2 = _lp_polish(bestX, seconds=rem2 - 0.6, t_hard=HARD_END - 0.4)
            if _max_cos(y2) < bestVal:
                bestX = y2.copy()
    return _norm_rows(bestX)
