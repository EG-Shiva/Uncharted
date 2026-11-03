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
            Z = np.zeros_like(Gt)
            Z[active] = Gt[active]
            Gt = Z
        gn = float(np.max(np.linalg.norm(Gt, axis=1)))
        if not np.isfinite(gn) or gn < 1e-14:
            margin *= 0.5
            if margin < 1e-7:
                break
            continue
        if active is None:
            Xn = _norm(cur - (lr / gn) * Gt)
        else:
            Xn = cur.copy()
            Xn[active] = _norm(cur[active] - (lr / gn) * Gt[active])
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

def _refine_single(v, Others, iters=120, lr0=0.08):
    v = v / max(1e-12, np.linalg.norm(v))
    lr = lr0
    best = float(np.max(Others @ v))
    bv = v.copy()
    for _ in range(iters):
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
    HOP_END = t0 + 122.0
    rng = np.random.RandomState(777)
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
            def fun(y):
                Y = y.reshape(n, d)
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
                W = np.zeros((n, n))
                W[iu0, iu1] = w
                W = W + W.T
                GX = W @ X
                dot = (GX * X).sum(axis=1, keepdims=True)
                GY = (GX - dot * X) / rn
                return float(loss), GY.ravel()
            try:
                res = minimize(fun, y0, jac=True, method="L-BFGS-B",
                               options={"maxiter": maxiter, "ftol": 1e-12, "gtol": 1e-7, "maxfun": 20000})
                return _norm(res.x.reshape(n, d))
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
                return float(loss), (GXt[sub] / rn_s).ravel()
            try:
                res = minimize(fun2, y0, jac=True, method="L-BFGS-B",
                               options={"maxiter": maxiter, "ftol": 1e-12, "gtol": 1e-7, "maxfun": 20000})
                X = fixed.copy()
                X[sub] = _norm(res.x.reshape(k, d))
                return _norm(X)
            except Exception:
                return _norm(x0)

    def lp_polish(x, deadline, r_init=1.0e-3, max_steps=14):
        x = _norm(x.copy())
        best = _max_cos(x)
        bx = x.copy()
        radius = r_init
        steps = 0
        while time.time() < deadline and steps < max_steps and time.time() - t0 < BUDGET - 1.5:
            steps += 1
            if radius < 1e-14:
                break
            Gf = x @ x.T
            c = Gf[iu0, iu1]
            m = float(c.max())
            thr = m - max(18.0 * radius, 1e-9)
            sel = c > thr
            I = iu0[sel]; J = iu1[sel]; csel = c[sel]
            p = len(I)
            if p == 0:
                break
            if p > 3500:
                kk = 3500
                part = np.argpartition(-csel, kk - 1)[:kk]
                I, J, csel = I[part], J[part], csel[part]
            rows = np.repeat(np.arange(len(I)), 2 * d + 1)
            cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((len(I), 1), n * d)], 1).ravel()
            vals = np.concatenate([x[J], x[I], -np.ones((len(I), 1))], 1).ravel()
            A_ub = coo_matrix((vals, (rows, cols)), shape=(len(I), n * d + 1)).tocsr()
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
        return bx, best

    cur_X = x_best.copy()
    cur_val = best_val
    pool = [(best_val, x_best.copy())]
    trial = 0
    while time.time() < HOP_END and time.time() - t0 < BUDGET - 45:
        trial += 1
        # select base
        r = rng.rand()
        if r < 0.5:
            base_X = x_best
        elif len(pool) > 1 and r < 0.75:
            base_X = pool[int(rng.randint(0, len(pool)))][1]
        else:
            base_X = cur_X
        base_X = _norm(base_X.copy())
        Gc = base_X @ base_X.T
        np.fill_diagonal(Gc, -2.0)
        prow = Gc.max(axis=1)
        mcur = float(Gc.max())
        # contact counts within 1e-9
        cnt = (Gc > mcur - 1e-9).sum(axis=1)
        rattlers = np.where(cnt == 0)[0]
        hubs = np.argsort(-cnt)[:12]
        worst = np.argsort(-prow)[:12]
        x_init = base_X.copy()
        mode = trial % 5
        sub_hint = None
        if mode in (0, 1):
            # rattler-integrating kill-and-fill: always include rattler(s) + hubs + worst
            k = int(rng.randint(10, 20))
            victims = set()
            for rr in rattlers[:2]:
                victims.add(int(rr))
            # add hubs/worst
            cand_pool = list(map(int, np.concatenate([hubs[:6], worst[:6]])))
            rng.shuffle(cand_pool)
            for c_ in cand_pool:
                if len(victims) >= k:
                    break
                victims.add(c_)
            while len(victims) < k:
                victims.add(int(rng.randint(0, n)))
            victims = list(victims)
            keep = np.array([i for i in range(n) if i not in set(victims)], dtype=int)
            fixed = x_init[keep]
            new_pts = []
            for _ in victims:
                cand = _norm(rng.randn(2200, d))
                dots = cand @ fixed.T
                if new_pts:
                    dots2 = cand @ np.stack(new_pts).T
                    dots = np.concatenate([dots, dots2], axis=1)
                bi = int(np.argmin(dots.max(axis=1)))
                v0 = cand[bi]
                others = fixed if not new_pts else np.concatenate([fixed, np.stack(new_pts)], axis=0)
                v0 = _refine_single(v0, others, iters=100)
                new_pts.append(v0)
            for idx, vv in zip(victims, new_pts):
                x_init[idx] = vv
            x_init = _norm(x_init)
            # rattler neighbourhood re-jam: freeze rest, relax victims+nearest
            sub = set(victims)
            for w in victims[:5]:
                nn = np.argsort(-(x_init @ x_init[w]))[1:7]
                for u in nn:
                    sub.add(int(u))
            # also add rattler neighbours explicitly to force >=5 contacts
            if len(rattlers) > 0:
                rr = int(rattlers[0]) if int(rattlers[0]) in victims else victims[0]
                nn2 = np.argsort(-(x_init @ x_init[rr]))[1:16]
                for u in nn2:
                    sub.add(int(u))
            sub_hint = np.array(sorted(sub), dtype=int)
            x_init, _ = _fast_relax(x_init, iters=220, lr0=0.03, margin0=0.006, mask=sub_hint)
        elif mode == 2:
            # cap shear + rotation to change topology
            u = _norm(rng.randn(1, d))[0]
            proj = x_init @ u
            q = np.quantile(proj, 1.0 - rng.uniform(0.10, 0.25))
            cap = np.where(proj >= q)[0]
            if len(cap) < 8:
                cap = np.argsort(-proj)[:20]
            w = rng.randn(d); w = w - float(w @ u) * u
            nw = float(np.linalg.norm(w))
            if nw < 1e-12:
                w = rng.randn(d); nw = float(np.linalg.norm(w))
            w /= nw
            alpha = 0.15 + 0.30 * rng.rand()
            for gi in cap:
                xi = x_init[gi]
                xi2 = xi + alpha * (float(xi @ u)) * w
                x_init[gi] = xi2
            x_init = _norm(x_init)
            # tangential noise on cap
            nz = rng.randn(len(cap), d)
            nz -= (nz * x_init[cap]).sum(1, keepdims=True) * x_init[cap]
            x_init[cap] = _norm(x_init[cap] + 0.03 * nz)
            sub_hint = cap
            x_init, _ = _fast_relax(x_init, iters=200, lr0=0.025, margin0=0.005, mask=None)
        elif mode == 3:
            # hub dispersal: strong tangential kick to hubs + neighbours
            targets = list(map(int, np.concatenate([hubs[:8], worst[:4]])))
            sub = set(targets)
            for w_ in targets[:4]:
                nn = np.argsort(-Gc[w_])[:5]
                for u in nn:
                    sub.add(int(u))
            sub = np.array(sorted(sub), dtype=int)
            nz = rng.randn(len(sub), d)
            nz -= (nz * x_init[sub]).sum(1, keepdims=True) * x_init[sub]
            amp = 0.10 + 0.15 * rng.rand()
            x_init[sub] = _norm(x_init[sub] + amp * nz)
            sub_hint = sub
            x_init, _ = _fast_relax(x_init, iters=220, lr0=0.03, margin0=0.006, mask=sub_hint)
        else:
            # global tangential annealing
            nz = rng.randn(n, d)
            nz -= (nz * x_init).sum(1, keepdims=True) * x_init
            nn = np.linalg.norm(nz, axis=1, keepdims=True) + 1e-12
            amp = 0.04 + 0.05 * rng.rand()
            x_init = _norm(x_init + amp * nz / nn * np.sqrt(d))
            x_init, _ = _fast_relax(x_init, iters=250, lr0=0.02, margin0=0.004, mask=None)
            sub_hint = None
        # L-BFGS sharpening: subset first then global
        beta = 350 if (trial % 3) else 800
        if sub_hint is not None and len(sub_hint) < n and rng.rand() < 0.7:
            x_init = run_lbfgs(x_init, beta, 120, mask=sub_hint)
        x_init = run_lbfgs(x_init, beta, 220, mask=None)
        v = _max_cos(x_init)
        # acceptance
        if v < cur_val - 1e-12 or rng.rand() < 0.12:
            cur_X, cur_val = x_init, v
        if v < best_val - 2e-12:
            x_best, best_val = x_init.copy(), v
            pool.append((v, x_init.copy()))
            if len(pool) > 8:
                pool.sort(key=lambda t: t[0])
                pool = pool[:8]
            # quick LP on new best if time allows
            if time.time() - t0 < BUDGET - 30:
                bx, bv = lp_polish(x_best, t0 + BUDGET - 12.0, r_init=8e-4, max_steps=6)
                if bv < best_val:
                    x_best, best_val = bx, bv
                    pool.append((bv, bx.copy()))
        elif v < best_val + 0.0015 and len(pool) < 8 and rng.rand() < 0.1:
            pool.append((v, x_init.copy()))

    # final polish: escalating beta + full LP
    for beta, mi in [(600, 300), (1500, 300), (3000, 200)]:
        if time.time() - t0 > BUDGET - 22:
            break
        x_best = run_lbfgs(x_best, beta, mi, mask=None)
        bv = _max_cos(x_best)
        if bv < best_val:
            best_val = bv
    if time.time() - t0 < BUDGET - 3:
        bx, bv = lp_polish(x_best, t0 + BUDGET - 1.0, r_init=1.0e-3, max_steps=14)
        if bv < best_val:
            x_best, best_val = bx, bv
    return _norm(x_best)
