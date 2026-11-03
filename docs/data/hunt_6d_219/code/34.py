import time
import numpy as np
from scipy.optimize import minimize, linprog
from scipy.sparse import coo_matrix

def _norm(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n = np.maximum(n, 1e-12)
    return X / n

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _per_max(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return np.max(G, axis=1), G

def _fast_relax(X, iters, lr0=0.02, margin0=0.004, mask=None):
    X = _norm(X)
    n, d = X.shape
    best = _max_cos(X)
    bestX = X.copy()
    cur = X.copy()
    lr = lr0
    margin = margin0
    if mask is None:
        active = None
    else:
        active = np.asarray(mask, dtype=int)
        if len(active) == 0:
            return bestX, best
    for it in range(iters):
        G = cur @ cur.T
        np.fill_diagonal(G, -2.0)
        m = float(np.max(G))
        tau = margin * 0.2 + 1e-12
        W = np.exp((G - m) / tau)
        W[G < m - margin * 5.0] = 0.0
        np.fill_diagonal(W, 0.0)
        Grad = W @ cur
        proj = np.sum(Grad * cur, axis=1, keepdims=True)
        Gt = Grad - proj * cur
        if active is not None:
            Gt2 = np.zeros_like(Gt)
            Gt2[active] = Gt[active]
            Gt = Gt2
        gn = float(np.max(np.linalg.norm(Gt, axis=1)))
        if not np.isfinite(gn) or gn < 1e-14:
            margin *= 0.5
            if margin < 1e-7:
                break
            continue
        Xn = cur.copy()
        if active is None:
            Xn = cur - (lr / gn) * Gt
            Xn = _norm(Xn)
        else:
            Xn[active] = cur[active] - (lr / gn) * Gt[active]
            Xn[active] = _norm(Xn[active])
        G2 = Xn @ Xn.T
        np.fill_diagonal(G2, -2.0)
        mn = float(np.max(G2))
        if mn < m:
            cur = Xn
            if mn < best:
                best = mn
                bestX = Xn.copy()
            if it % 200 == 0:
                margin *= 0.85
        else:
            lr *= 0.6
            if lr < 1e-6:
                margin *= 0.7
                lr = lr0 * 0.3
                if margin < 5e-7:
                    break
    return bestX, best

def _refine_single(v, Others, iters=160, lr0=0.08, rng=None):
    v = v / max(1e-12, np.linalg.norm(v))
    lr = lr0
    best = float(np.max(Others @ v))
    bv = v.copy()
    for it in range(iters):
        dots = Others @ v
        m = float(np.max(dots))
        tau = 0.003
        w = np.exp((dots - m) / tau)
        w[dots < m - 0.015] = 0.0
        g = w @ Others
        gt = g - float(g @ v) * v
        nm = float(np.linalg.norm(gt))
        if nm < 1e-14 or not np.isfinite(nm):
            break
        vn = v - (lr / nm) * gt * 0.3
        vn = vn / max(1e-12, np.linalg.norm(vn))
        mv = float(np.max(Others @ vn))
        if mv < m:
            v = vn
            if mv < best:
                best = mv
                bv = vn.copy()
        else:
            lr *= 0.7
            if lr < 1e-4:
                break
    return bv

def solve():
    t0 = time.time()
    BUDGET = 170.0
    HOP_END = t0 + 128.0
    rng = np.random.RandomState(20261103)
    n, d = 219, 6
    iu0, iu1 = np.triu_indices(n, 1)

    x_best = None
    best_val = 1e9
    for fn in ["best.npy", "record.npy"]:
        try:
            v = np.load(fn)
            v = np.asarray(v, dtype=float)
            if v.shape == (n, d):
                v = _norm(v)
                c = _max_cos(v)
                if c < best_val:
                    best_val = c
                    x_best = v.copy()
        except Exception:
            pass
    if x_best is None:
        x_best = _norm(rng.randn(n, d))
        best_val = _max_cos(x_best)

    def run_lbfgs(x0, beta, maxiter, mask=None):
        # mask: indices to optimize; others frozen via large penalty (fix by masking grad and resetting)
        y0full = x0.copy()
        if mask is None:
            y0 = x0.ravel().copy()
            nn, dd = n, d
            def fun(y):
                Y = y.reshape(nn, dd)
                rn = np.linalg.norm(Y, axis=1, keepdims=True)
                rn = np.maximum(rn, 1e-12)
                X = Y / rn
                G = X @ X.T
                g = G[iu0, iu1]
                m = float(g.max())
                e = np.exp(beta * (g - m))
                S = float(e.sum()) + 1e-300
                loss = m + np.log(S) / beta
                w = e / S
                W = np.zeros((nn, nn), dtype=float)
                W[iu0, iu1] = w
                W = W + W.T
                GX = W @ X
                dot = (GX * X).sum(axis=1, keepdims=True)
                GY = (GX - dot * X) / rn
                return float(loss), GY.ravel()
            try:
                res = minimize(fun, y0, jac=True, method="L-BFGS-B",
                               options={"maxiter": maxiter, "ftol": 1e-12, "gtol": 1e-7, "maxfun": 20000})
                Y = res.x.reshape(nn, dd)
                return _norm(Y)
            except Exception:
                return _norm(x0)
        else:
            sub = np.asarray(mask, dtype=int)
            k = len(sub)
            if k == 0:
                return _norm(x0)
            fixed = x0.copy()
            y0 = x0[sub].ravel().copy()
            # precompute iu for mapping? use full X each eval
            def fun2(y):
                Ysub = y.reshape(k, d)
                rn_s = np.linalg.norm(Ysub, axis=1, keepdims=True)
                rn_s = np.maximum(rn_s, 1e-12)
                X = fixed.copy()
                X[sub] = Ysub / rn_s
                # need grad w.r.t Ysub only
                G = X @ X.T
                g = G[iu0, iu1]
                m = float(g.max())
                e = np.exp(beta * (g - m))
                S = float(e.sum()) + 1e-300
                loss = m + np.log(S) / beta
                w = e / S
                W = np.zeros((n, n), dtype=float)
                W[iu0, iu1] = w
                W = W + W.T
                GX = W @ X
                dot = (GX * X).sum(axis=1, keepdims=True)
                GXt = (GX - dot * X)
                GYs = GXt[sub] / rn_s
                return float(loss), GYs.ravel()
            try:
                res = minimize(fun2, y0, jac=True, method="L-BFGS-B",
                               options={"maxiter": maxiter, "ftol": 1e-12, "gtol": 1e-7, "maxfun": 20000})
                Ysub = res.x.reshape(k, d)
                X = fixed.copy()
                X[sub] = _norm(Ysub)
                return _norm(X)
            except Exception:
                return _norm(x0)

    def lp_polish(x, deadline, r_init=1.1e-3):
        x = _norm(x.copy())
        best = _max_cos(x)
        bx = x.copy()
        radius = r_init
        while time.time() < deadline and time.time() - t0 < BUDGET - 1.5:
            if radius < 1e-14:
                break
            Gf = x @ x.T
            c = Gf[iu0, iu1]
            m = float(c.max())
            thr = m - max(20.0 * radius, 1e-9)
            sel = c > thr
            I = iu0[sel]; J = iu1[sel]; csel = c[sel]
            p = len(I)
            if p == 0:
                break
            if p > 3600:
                kk = 3600
                part = np.argpartition(-csel, kk - 1)[:kk]
                I, J, csel = I[part], J[part], csel[part]
                p = kk
            rows = np.repeat(np.arange(p), 2 * d + 1)
            cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
            vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, n * d + 1)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, n * d + 1)).tocsr()
            obj = np.zeros(n * d + 1); obj[-1] = 1.0
            try:
                res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq, b_eq=np.zeros(n),
                              bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
            except Exception:
                radius *= 0.5
                continue
            if res.status != 0:
                radius *= 0.5
                continue
            y = _norm(x + res.x[:-1].reshape(n, d))
            v = _max_cos(y)
            if v < best:
                x, best = y, v
                bx = y.copy()
                radius = min(radius * 1.5, 0.03)
            else:
                radius *= 0.5
            if p < 10 and radius < 1e-6:
                break
        return bx, best

    pool = [(best_val, x_best.copy())]
    cur_X = x_best.copy()
    cur_val = best_val
    trial = 0
    temp0 = 0.0015

    while time.time() < HOP_END and time.time() - t0 < BUDGET - 42:
        trial += 1
        elapsed = time.time() - t0
        frac = elapsed / max(1e-9, (HOP_END - t0))
        temp = temp0 * max(0.05, 1.0 - frac)
        # base selection
        rsel = rng.rand()
        if rsel < 0.55:
            base_X = x_best; base_val = best_val
        elif len(pool) > 1 and rsel < 0.75:
            pi = int(rng.randint(0, len(pool)))
            base_X = pool[pi][1]; base_val = pool[pi][0]
        else:
            base_X = cur_X; base_val = cur_val
        Gcur = base_X @ base_X.T
        np.fill_diagonal(Gcur, -2.0)
        ppm = Gcur.max(axis=1)
        order = np.argsort(-ppm)
        mcur = float(Gcur.max())
        x_init = base_X.copy()
        mode = int(rng.randint(0, 6))
        sub_hint = None
        try:
            if mode == 0:
                # kill-and-fill: remove k worst, greedy reinsert into voids
                k = int(rng.randint(8, 24))
                victims = list(map(int, order[:k]))
                keep = np.array([i for i in range(n) if i not in set(victims)], dtype=int)
                fixed = x_init[keep]
                # sequential maximin insert
                new_pts = []
                for vi in range(len(victims)):
                    cand = _norm(rng.randn(2500, d))
                    dots = cand @ fixed.T
                    if len(new_pts) > 0:
                        np_pts = np.stack(new_pts)
                        d2 = cand @ np_pts.T
                        dots = np.concatenate([dots, d2], axis=1)
                    worst = dots.max(axis=1)
                    bi = int(np.argmin(worst))
                    v0 = cand[bi]
                    others = fixed if len(new_pts) == 0 else np.concatenate([fixed, np.stack(new_pts)], axis=0)
                    v0 = _refine_single(v0, others, iters=120)
                    new_pts.append(v0)
                for idx, vv in zip(victims, new_pts):
                    x_init[idx] = vv
                x_init = _norm(x_init)
                # neighbourhood jitter
                sub = set(victims)
                for w in victims[:6]:
                    nn = np.argsort(-Gcur[w])[:6]
                    for u in nn:
                        sub.add(int(u))
                sub_hint = np.array(sorted(sub), dtype=int)
                # frozen-subset pre-relax on victims+neighbours
                x_init, _ = _fast_relax(x_init, iters=250, lr0=0.03, margin0=0.006, mask=sub_hint)
            elif mode == 1:
                # cap shear: large collective move of a spherical cap
                u = _norm(rng.randn(1, d))[0]
                proj = x_init @ u
                q = np.quantile(proj, 1.0 - rng.uniform(0.10, 0.28))
                cap = np.where(proj >= q)[0]
                if len(cap) < 8:
                    cap = np.argsort(-proj)[:20]
                # random shear direction perpendicular-ish
                w = rng.randn(d); w = w - float(w @ u) * u
                nw = float(np.linalg.norm(w))
                if nw < 1e-12:
                    w = rng.randn(d)
                    nw = float(np.linalg.norm(w))
                w /= nw
                alpha = 0.18 + 0.25 * rng.rand()
                for gidx in cap:
                    xi = x_init[gidx]
                    tw = w - float(w @ xi) * xi
                    nm = float(np.linalg.norm(tw))
                    if nm > 1e-12:
                        tw /= nm
                    else:
                        tw = np.zeros(d)
                    x_init[gidx] = xi + alpha * tw + 0.02 * (rng.randn(d) - float(rng.randn(d) @ xi) * 0 if False else 0)
                # add tangent noise
                pert = rng.randn(len(cap), d)
                for ii, gidx in enumerate(cap):
                    xi = x_init[gidx]
                    # project
                    p = pert[ii] - float(pert[ii] @ xi) * xi
                    x_init[gidx] = xi + 0.03 * p
                x_init[cap] = _norm(x_init[cap])
                sub_hint = cap.copy()
                x_init, _ = _fast_relax(x_init, iters=250, lr0=0.03, margin0=0.006, mask=cap)
            elif mode == 2:
                # transplant worst point into rattler void
                ratt = int(np.argmin(ppm))
                # void centre ~ rattler position; place victim near it but at distance ~ current best
                vw = int(order[0])
                # sample around rattler: points with dot ~ mcur*0.9?
                # generate candidates near rattler: ratt + noise
                best_c = None; best_m = 1e9
                others_idx = [i for i in range(n) if i != vw]
                others = x_init[others_idx]
                for _ in range(3000):
                    cand = x_init[ratt] + 0.35 * rng.randn(d)
                    cand = cand / max(1e-12, np.linalg.norm(cand))
                    # also try pure random half the time
                    if rng.rand() < 0.3:
                        cand = _norm(rng.randn(1, d))[0]
                    mv = float(np.max(others @ cand))
                    if mv < best_m:
                        best_m = mv; best_c = cand
                x_init[vw] = best_c
                # jitter its new neighbours
                dots2 = x_init @ best_c
                dots2[vw] = -2
                nn = np.argsort(-dots2)[:10]
                sub = set([vw, ratt] + list(map(int, nn)))
                sub_hint = np.array(sorted(sub), dtype=int)
                pert = rng.randn(len(sub_hint), d)
                for ii, gidx in enumerate(sub_hint):
                    xi = x_init[gidx]
                    p = pert[ii] - float(pert[ii] @ xi) * xi
                    x_init[gidx] = x_init[gidx] + 0.02 * p
                x_init[sub_hint] = _norm(x_init[sub_hint])
                x_init, _ = _fast_relax(x_init, iters=250, lr0=0.03, margin0=0.006, mask=sub_hint)
            elif mode == 3:
                # worst block strong kick + frozen relax
                k = int(rng.randint(16, 34))
                seeds = order[:max(5, k // 3)]
                candl = list(map(int, seeds))
                for s in list(seeds[:6]):
                    nn = np.argsort(-Gcur[s])[:5]
                    for u in nn:
                        if int(u) not in candl:
                            candl.append(int(u))
                        if len(candl) >= k + 4:
                            break
                while len(candl) < k:
                    candl.append(int(rng.randint(0, n)))
                sub = np.array(sorted(set(candl))[:k], dtype=int)
                pert = rng.randn(len(sub), d)
                for ii, gidx in enumerate(sub):
                    xi = x_init[gidx]
                    p = pert[ii] - float(pert[ii] @ xi) * xi
                    nm = float(np.linalg.norm(p))
                    if nm > 1e-12:
                        p /= nm
                    x_init[gidx] = xi + (0.10 + 0.12 * rng.rand()) * p
                x_init[sub] = _norm(x_init[sub])
                sub_hint = sub
                x_init, _ = _fast_relax(x_init, iters=220, lr0=0.03, margin0=0.006, mask=sub)
            elif mode == 4:
                # pair-break with antipodal push + crowd noise
                i = int(order[0]); j = int(np.argmax(Gcur[i]))
                for (a, b) in [(i, j), (j, i)]:
                    xa = x_init[a]; xb = x_init[b]
                    t = xb - float(xb @ xa) * xa
                    nm = float(np.linalg.norm(t))
                    if nm > 1e-12:
                        t /= nm
                        x_init[a] = xa - (0.10 + 0.08 * rng.rand()) * t
                x_init[[i, j]] = _norm(x_init[[i, j]])
                crowd = list(map(int, np.argsort(-ppm)[:14]))
                sub = np.array(sorted(set(crowd + [i, j])), dtype=int)
                pert = rng.randn(len(sub), d)
                for ii, gidx in enumerate(sub):
                    xi = x_init[gidx]
                    p = pert[ii] - float(pert[ii] @ xi) * xi
                    x_init[gidx] = xi + (0.04 + 0.04 * rng.rand()) * p
                x_init[sub] = _norm(x_init[sub])
                sub_hint = sub
            else:
                # broad medium kick on random block
                k = int(rng.randint(30, 60))
                idx = rng.choice(n, size=k, replace=False)
                pert = rng.randn(n, d)
                for ii in range(n):
                    xi = x_init[ii]
                    p = pert[ii] - float(pert[ii] @ xi) * xi
                    pert[ii] = p
                sig = np.full((n, 1), 0.004)
                sig[idx] = 0.06 + 0.05 * rng.rand()
                x_init = _norm(x_init + sig * pert)
                sub_hint = idx
        except Exception:
            x_init = base_X.copy()
            sub_hint = None

        # two-stage refine: fast global then subset LBFGS then global LBFGS
        try:
            Xi, vi = _fast_relax(x_init, iters=350, lr0=0.02, margin0=0.003)
        except Exception:
            Xi, vi = x_init, _max_cos(x_init)
        # decide LBFGS budget by remaining time
        remain = BUDGET - (time.time() - t0)
        if remain < 30:
            maxit = 25
        elif remain < 60:
            maxit = 40
        else:
            maxit = 55
        try:
            beta = 220.0 + 180.0 * rng.rand()
            # occasionally do subset LBFGS first
            if sub_hint is not None and len(sub_hint) < n and rng.rand() < 0.5:
                Xs = run_lbfgs(Xi, beta * 0.6, maxiter=30, mask=sub_hint)
                Xf = run_lbfgs(Xs, beta, maxiter=maxit)
            else:
                Xf = run_lbfgs(Xi, beta, maxiter=maxit)
            vf = _max_cos(Xf)
        except Exception:
            Xf, vf = Xi, vi
        # acceptance
        improved_global = vf < best_val - 1e-12
        if improved_global:
            best_val = vf
            x_best = Xf.copy()
            pool = [(best_val, x_best.copy())] + pool[:4]
            cur_X, cur_val = Xf.copy(), vf
        else:
            # anneal: accept worse as wanderer
            if vf < cur_val or rng.rand() < np.exp(-max(0.0, vf - cur_val) / max(1e-9, temp)):
                cur_X, cur_val = Xf.copy(), vf
                # keep pool diverse (max 5)
                pool.append((vf, Xf.copy()))
                pool = sorted(pool, key=lambda z: z[0])[:5]
            elif vi < cur_val:
                cur_X, cur_val = Xi.copy(), vi
        if time.time() - t0 > HOP_END:
            break

    # final sharpen best
    try:
        Xb = x_best.copy()
        # annealed LBFGS chain
        for beta, mi in [(250.0, 60), (450.0, 60), (700.0, 40)]:
            if time.time() - t0 > BUDGET - 18:
                break
            Xb = run_lbfgs(Xb, beta, maxiter=mi)
        # LP squeeze with remaining time
        deadline = t0 + BUDGET - 0.8
        Xb, vb = lp_polish(Xb, deadline)
        if vb < best_val:
            best_val = vb
            x_best = Xb.copy()
    except Exception:
        pass
    x_best = _norm(x_best)
    return x_best
