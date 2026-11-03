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

def _refine_single(v, Others, iters=120, lr0=0.08):
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

def _fast_relax(X, iters, lr0=0.025, margin0=0.004, mask=None):
    X = _norm(X)
    n = X.shape[0]
    best = _max_cos(X)
    bestX = X.copy()
    cur = X.copy()
    lr = lr0
    margin = margin0
    active = None if mask is None else np.asarray(mask, dtype=int)
    if active is not None and len(active) == 0:
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

def solve():
    t0 = time.time()
    BUDGET = 172.0
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
        if mask is None:
            y0 = x0.ravel().copy()
            nn = n
            def fun(y):
                Y = y.reshape(nn, d)
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
                W = np.zeros((nn, nn))
                W[iu0, iu1] = w
                W = W + W.T
                GX = W @ X
                dot = (GX * X).sum(axis=1, keepdims=True)
                GY = (GX - dot * X) / rn
                return float(loss), GY.ravel()
            try:
                res = minimize(fun, y0, jac=True, method="L-BFGS-B",
                               options={"maxiter": maxiter, "ftol": 1e-12, "gtol": 1e-7, "maxfun": 20000})
                Y = res.x.reshape(nn, d)
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
            def fun2(y):
                Ysub = y.reshape(k, d)
                rn_s = np.linalg.norm(Ysub, axis=1, keepdims=True)
                rn_s = np.maximum(rn_s, 1e-12)
                X = fixed.copy()
                X[sub] = Ysub / rn_s
                G = X @ X.T
                g = G[iu0, iu1]
                m = float(g.max())
                e = np.exp(beta * (g - m))
                S = float(e.sum()) + 1e-300
                loss = m + np.log(S) / beta
                w = e / S
                W = np.zeros((n, n))
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

    def lp_polish(x, deadline, r_init=1.0e-3):
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
            try:
                A_ub = coo_matrix((vals, (rows, cols)), shape=(p, n * d + 1)).tocsr()
                A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, n * d + 1)).tocsr()
                obj = np.zeros(n * d + 1); obj[-1] = 1.0
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

    def cone_transplant(X, r_idx, hubs, dot_target):
        X = X.copy()
        r = X[r_idx] / max(1e-12, np.linalg.norm(X[r_idx]))
        # build orthonormal basis perpendicular to r
        for h in hubs:
            # random perp directions -> cone candidates
            best_c = None
            best_s = 1e9
            # need fixed set excluding h
            mask_keep = np.ones(n, dtype=bool); mask_keep[h] = False
            fixed = X[mask_keep]
            s_target = np.sqrt(max(0.0, 1.0 - dot_target * dot_target))
            # batch sampling
            for _b in range(2):
                Z = rng.randn(1500, d)
                Z -= (Z @ r)[:, None] * r[None, :]
                zn = np.linalg.norm(Z, axis=1, keepdims=True) + 1e-12
                Z /= zn
                cand = dot_target * r[None, :] + s_target * Z
                cand = _norm(cand)
                dots = cand @ fixed.T
                worst = dots.max(axis=1)
                bi = int(np.argmin(worst))
                if worst[bi] < best_s:
                    best_s = float(worst[bi])
                    best_c = cand[bi]
            # refine best cone point against fixed
            best_c = _refine_single(best_c, fixed, iters=80)
            X[h] = best_c
        return _norm(X)

    pool = [(best_val, x_best.copy())]
    cur_X = x_best.copy()
    cur_val = best_val
    trial = 0
    while time.time() < HOP_END and time.time() - t0 < BUDGET - 42:
        trial += 1
        # select base
        u = rng.rand()
        if u < 0.5:
            base_X = x_best; base_val = best_val
        elif len(pool) > 1 and u < 0.75:
            pi = int(rng.randint(0, len(pool)))
            base_X = pool[pi][1]; base_val = pool[pi][0]
        else:
            base_X = cur_X; base_val = cur_val
        ppm, Gcur = _per_max(base_X)
        mcur = float(Gcur.max())
        # contact counts with tol 1e-7 for rattler detection
        thr_c = mcur - 1e-7
        cnt = np.sum(Gcur > thr_c, axis=1) - 1  # diag is -2 so not counted; actually need correct
        # Gcur diag -2, so sum over >thr gives contacts
        cnt = np.sum(Gcur > thr_c, axis=1)
        rattlers = np.where(cnt == 0)[0]
        order_hi = np.argsort(-ppm)
        order_lo = np.argsort(ppm)
        x_init = base_X.copy()
        mode = int(rng.randint(0, 5))
        try:
            if mode == 0:
                # rattler hub transplant: force 2-4 hubs onto cone around rattler
                r = int(rattlers[0]) if len(rattlers) > 0 else int(order_lo[0])
                nh = int(rng.randint(2, 5))
                hubs = [int(x) for x in order_hi[:12] if x != r][:nh+2]
                # ensure hubs not including r
                hubs = [h for h in hubs if h != r][:nh]
                if len(hubs) == 0:
                    hubs = [int(order_hi[0])] if int(order_hi[0]) != r else [int(order_hi[1])]
                dt = mcur - 0.002 - 0.002 * rng.rand()
                dt = float(np.clip(dt, 0.60, 0.692))
                x_init = cone_transplant(x_init, r, hubs, dt)
                sub = set(hubs); sub.add(r)
                # add rattler nearest neighbours
                nn = np.argsort(-Gcur[r])[:8]
                for qq in nn:
                    sub.add(int(qq))
                sub = np.array(sorted(sub), dtype=int)
                x_init, _ = _fast_relax(x_init, iters=250, lr0=0.03, margin0=0.006, mask=sub)
                x_init = run_lbfgs(x_init, beta=600, maxiter=60, mask=sub)
                x_init = run_lbfgs(x_init, beta=900, maxiter=60, mask=None)
            elif mode == 1:
                # kill worst + void fill
                k = int(rng.randint(3, 8))
                victims = [int(x) for x in order_hi[:k]]
                keep_mask = np.ones(n, dtype=bool)
                keep_mask[victims] = False
                fixed = x_init[keep_mask]
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
                    v0 = _refine_single(v0, others, iters=100)
                    new_pts.append(v0)
                for idx, vv in zip(victims, new_pts):
                    x_init[idx] = vv
                x_init = _norm(x_init)
                sub = set(victims)
                for w in victims[:4]:
                    nn = np.argsort(-Gcur[w])[:5]
                    for qq in nn:
                        sub.add(int(qq))
                sub = np.array(sorted(sub), dtype=int)
                x_init, _ = _fast_relax(x_init, iters=250, lr0=0.03, margin0=0.006, mask=sub)
                x_init = run_lbfgs(x_init, beta=700, maxiter=70, mask=None)
            elif mode == 2:
                # cap shear + rotation
                uu = _norm(rng.randn(1, d))[0]
                proj = x_init @ uu
                q = np.quantile(proj, 1.0 - rng.uniform(0.10, 0.30))
                cap = np.where(proj >= q)[0]
                if len(cap) < 8:
                    cap = np.argsort(-proj)[:20]
                w = rng.randn(d); w = w - float(w @ uu) * uu
                nw = float(np.linalg.norm(w))
                if nw < 1e-12:
                    w = rng.randn(d); nw = float(np.linalg.norm(w))
                w /= nw
                alpha = 0.15 + 0.25 * rng.rand()
                for gi in cap:
                    xi = x_init[gi]
                    coeff = float(xi @ uu)
                    xi2 = xi + alpha * (coeff - float(q)) * w
                    x_init[gi] = xi2
                x_init = _norm(x_init)
                # tangential noise on cap
                sig = 0.05 + 0.1 * rng.rand()
                for gi in cap:
                    nz = rng.randn(d)
                    nz -= float(nz @ x_init[gi]) * x_init[gi]
                    x_init[gi] += sig * nz / (np.linalg.norm(nz) + 1e-12)
                x_init = _norm(x_init)
                x_init, _ = _fast_relax(x_init, iters=300, lr0=0.025, margin0=0.005, mask=cap)
                x_init = run_lbfgs(x_init, beta=800, maxiter=70, mask=None)
            elif mode == 3:
                # hub neighbourhood strong shake
                nh2 = int(rng.randint(10, 26))
                hubs2 = [int(x) for x in order_hi[:nh2]]
                sig = 0.12 + 0.15 * rng.rand()
                for gi in hubs2:
                    nz = rng.randn(d)
                    nz -= float(nz @ x_init[gi]) * x_init[gi]
                    nn2 = np.linalg.norm(nz) + 1e-12
                    x_init[gi] = x_init[gi] + sig * nz / nn2
                # also shake rattler strongly
                if len(rattlers) > 0:
                    r = int(rattlers[0])
                    nz = rng.randn(d)
                    nz -= float(nz @ x_init[r]) * x_init[r]
                    x_init[r] = x_init[r] + 0.3 * nz / (np.linalg.norm(nz) + 1e-12)
                x_init = _norm(x_init)
                x_init, _ = _fast_relax(x_init, iters=300, lr0=0.03, margin0=0.006, mask=None)
                x_init = run_lbfgs(x_init, beta=500, maxiter=60, mask=None)
            else:
                # global small shake + anneal
                sig = 0.02 + 0.03 * rng.rand()
                nz = rng.randn(n, d)
                nz -= (nz * x_init).sum(1, keepdims=True) * x_init
                nn2 = np.linalg.norm(nz, axis=1, keepdims=True) + 1e-12
                x_init = x_init + sig * nz / nn2
                x_init = _norm(x_init)
                x_init, _ = _fast_relax(x_init, iters=200, lr0=0.02, margin0=0.004, mask=None)
                x_init = run_lbfgs(x_init, beta=1000, maxiter=60, mask=None)
        except Exception:
            continue
        try:
            v = _max_cos(x_init)
        except Exception:
            continue
        # acceptance
        if v < cur_val:
            cur_X = x_init.copy(); cur_val = v
        if v < best_val - 1e-12:
            x_best = x_init.copy(); best_val = v
            pool.append((v, x_init.copy()))
            if len(pool) > 8:
                pool = sorted(pool, key=lambda z: z[0])[:8]
            # quick LP squeeze on new best if time allows
            if time.time() - t0 < BUDGET - 45:
                try:
                    bx, bv = lp_polish(x_best, deadline=time.time() + 4.0, r_init=8e-4)
                    if bv < best_val:
                        x_best = bx; best_val = bv
                        pool[0] = (bv, bx.copy())
                    cur_X = x_best.copy(); cur_val = best_val
                except Exception:
                    pass
        elif v < best_val + 0.001 and rng.rand() < 0.25:
            # diversity
            pool.append((v, x_init.copy()))
            if len(pool) > 8:
                pool = sorted(pool, key=lambda z: z[0])[:8]
        # opportunistic LP if close
        if len(pool) > 1 and trial % 12 == 0 and time.time() - t0 < BUDGET - 45:
            try:
                # polish second best occasionally to explore different basin
                sb = sorted(pool, key=lambda z: z[0])[1][1]
                bx2, bv2 = lp_polish(sb, deadline=time.time() + 2.0, r_init=6e-4)
                if bv2 < best_val:
                    x_best = bx2; best_val = bv2
                    pool.append((bv2, bx2.copy()))
                    pool = sorted(pool, key=lambda z: z[0])[:8]
            except Exception:
                pass

    # final sharpening: increasing beta + LP
    try:
        x_best = run_lbfgs(x_best, beta=1200, maxiter=120, mask=None)
        v = _max_cos(x_best)
        if v < best_val:
            best_val = v
    except Exception:
        pass
    try:
        x_best = run_lbfgs(x_best, beta=2000, maxiter=120, mask=None)
        v = _max_cos(x_best)
        if v < best_val:
            best_val = v
    except Exception:
        pass
    try:
        remain = BUDGET - (time.time() - t0) - 0.5
        if remain > 3:
            bx, bv = lp_polish(x_best, deadline=time.time() + min(remain, 25.0), r_init=9e-4)
            if bv < best_val:
                x_best, best_val = bx, bv
    except Exception:
        pass
    x_best = _norm(x_best)
    return x_best
