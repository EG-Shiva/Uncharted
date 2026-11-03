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
    return G.max(axis=1), float(G.max())

def _solve_inner():
    t0 = time.time()
    BUDGET = 168.0
    rng = np.random.RandomState(42)
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

    def run_lbfgs(x0, beta, maxiter):
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
                           options={"maxiter": maxiter, "ftol": 1e-12, "gtol": 1e-8, "maxfun": 20000})
            Y = res.x.reshape(n, d)
            return _norm(Y)
        except Exception:
            return _norm(x0)

    def fast_relax(X, iters, lr0=0.025, margin0=0.004, noise0=0.0):
        X = _norm(X)
        cur = X.copy()
        best = _max_cos(cur)
        bestX = cur.copy()
        lr = lr0
        margin = margin0
        for it in range(iters):
            if time.time() - t0 > BUDGET - 4:
                break
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
            gn = float(np.max(np.linalg.norm(Gt, axis=1)))
            if not np.isfinite(gn) or gn < 1e-14:
                margin *= 0.6
                if margin < 5e-7:
                    break
                continue
            step = (lr / gn) * Gt
            if noise0 > 0:
                R = rng.randn(*cur.shape)
                R -= (R * cur).sum(1, keepdims=True) * cur
                rn2 = np.linalg.norm(R, axis=1, keepdims=True) + 1e-12
                dec = 1.0 - it / float(max(1, iters))
                step = step + (noise0 * dec) * (R / rn2) * 0.15
            Xn = _norm(cur - step)
            G2 = Xn @ Xn.T
            np.fill_diagonal(G2, -2.0)
            mn = float(np.max(G2))
            allow = 1e-12 if noise0 == 0 else 3e-4 * (1.0 - it / float(max(1, iters)))
            if mn < m + allow:
                cur = Xn
                if mn < best:
                    best = mn
                    bestX = Xn.copy()
                if it % 200 == 199:
                    margin *= 0.85
            else:
                lr *= 0.6
                if lr < 1e-6:
                    margin *= 0.7
                    lr = lr0 * 0.3
                    if margin < 5e-7:
                        break
        return bestX, best

    def lp_polish(X, deadline, r_init=1.0e-3, max_steps=12):
        X = _norm(X.copy())
        best = _max_cos(X)
        bx = X.copy()
        radius = r_init
        steps = 0
        while time.time() < deadline and time.time() - t0 < BUDGET - 1.5 and steps < max_steps:
            steps += 1
            if radius < 1e-14:
                break
            Gf = X @ X.T
            c = Gf[iu0, iu1]
            m = float(c.max())
            thr = m - max(18.0 * radius, 1e-9)
            sel = c > thr
            I = iu0[sel]; J = iu1[sel]; csel = c[sel]
            p = len(I)
            if p == 0:
                break
            if p > 3500:
                k = 3500
                part = np.argpartition(-csel, k - 1)[:k]
                I, J, csel = I[part], J[part], csel[part]
                p = k
            rows = np.repeat(np.arange(p), 2 * d + 1)
            cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
            vals = np.concatenate([X[J], X[I], -np.ones((p, 1))], 1).ravel()
            try:
                A_ub = coo_matrix((vals, (rows, cols)), shape=(p, n * d + 1)).tocsr()
                A_eq = coo_matrix((X.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, n * d + 1)).tocsr()
                obj = np.zeros(n * d + 1); obj[-1] = 1.0
                res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq, b_eq=np.zeros(n),
                              bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
            except Exception:
                radius *= 0.5
                continue
            if res.status != 0:
                radius *= 0.5
                continue
            y = _norm(X + res.x[:-1].reshape(n, d))
            v = _max_cos(y)
            if v < best:
                X, best = y, v
                bx = y.copy()
                radius = min(radius * 1.5, 0.03)
            else:
                radius *= 0.5
            if p < 10 and radius < 1e-6:
                break
        return bx, best

    def balanced_void(others, n_cand=8000, alpha=0.3):
        C = _norm(rng.randn(n_cand, d))
        D = C @ others.T
        # d1 and d5
        # partial sort: use argpartition for speed
        # full sort small n so just sort
        Ds = np.sort(D, axis=1)[:, ::-1]
        d1 = Ds[:, 0]
        d5 = Ds[:, 4] if Ds.shape[1] >= 5 else Ds[:, -1]
        score = d1 + alpha * (d1 - d5)
        bi = int(np.argmin(score))
        return C[bi]

    def refine_void(v, others, iters=80, lr0=0.05, alpha=0.3):
        v = v / max(1e-12, np.linalg.norm(v))
        bestv = v.copy()
        dots = others @ v
        s = np.sort(dots)[::-1]
        bestscore = float(s[0] + alpha * (s[0] - s[4] if len(s) >= 5 else 0))
        lr = lr0
        for _ in range(iters):
            dots = others @ v
            m = float(dots.max())
            # weights: top1 weight 1, next 4 weight alpha-ish via softmax
            tau = 0.003
            w = np.exp((dots - m) / tau)
            w[dots < m - 0.02] = 0.0
            # boost 2-5th to encourage clustering: add extra weight
            order = np.argsort(-dots)
            w2 = w.copy()
            # encourage d5 closeness: weight top5 equally a bit
            top5 = order[:5]
            w2[top5] += alpha * 0.5
            g = w2 @ others
            gt = g - float(g @ v) * v
            nn = float(np.linalg.norm(gt))
            if nn < 1e-14 or not np.isfinite(nn):
                break
            vn = v - (lr / nn) * gt
            vn /= max(1e-12, np.linalg.norm(vn))
            d2 = others @ vn
            s2 = np.sort(d2)[::-1]
            sc2 = float(s2[0] + alpha * (s2[0] - s2[4] if len(s2) >= 5 else 0))
            if sc2 < bestscore:
                v = vn
                bestscore = sc2
                bestv = vn.copy()
            else:
                lr *= 0.7
                if lr < 1e-4:
                    break
        return bestv

    def integrate_rattlers(X):
        # move each rattler (0 contacts at 1e-9) to feasible balanced spot just below max
        X = _norm(X.copy())
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        deg = np.sum(G > m - 1e-9, axis=1)
        rats = np.where(deg == 0)[0]
        if len(rats) == 0:
            return X
        for r in rats:
            if time.time() - t0 > BUDGET - 8:
                break
            others = X[np.arange(n) != r]
            # search feasible candidates with d1 < m - 5e-4, maximize d5
            C = _norm(rng.randn(20000, d))
            D = C @ others.T
            d1 = D.max(axis=1)
            feas = d1 < m - 5e-4
            if np.sum(feas) < 5:
                feas = d1 < m - 1e-4
            if np.sum(feas) == 0:
                continue
            Ds = np.sort(D[feas], axis=1)[:, ::-1]
            d5 = Ds[:, 4] if Ds.shape[1] >= 5 else Ds[:, -1]
            bi = int(np.argmax(d5))
            cand = _norm(C[feas][bi:bi+1])[0]
            # refine keeping d1 < m
            v = cand.copy()
            lr = 0.03
            bestv = v.copy()
            bd = float((others @ v).max())
            # want to push d5 up while keeping d1 < m-2e-4
            b5 = float(np.sort(others @ v)[::-1][4])
            for _ in range(120):
                dots = others @ v
                # gradient to increase 5th-nearest: weight top5
                order = np.argsort(-dots)
                top5 = order[:5]
                w = np.zeros_like(dots)
                w[top5] = 1.0
                # repel the top1 if too close to m
                g = w @ others
                gt = g - float(g @ v) * v
                nn = float(np.linalg.norm(gt))
                if nn < 1e-14:
                    break
                vn = v + (lr / nn) * gt  # ascent on dots (increase d5)
                vn /= max(1e-12, np.linalg.norm(vn))
                dn = others @ vn
                d1n = float(dn.max())
                if d1n > m - 2e-4:
                    lr *= 0.6
                    if lr < 1e-4:
                        break
                    continue
                s5 = float(np.sort(dn)[::-1][4])
                if s5 > b5:
                    v = vn
                    b5 = s5
                    bestv = vn.copy()
                else:
                    lr *= 0.7
                    if lr < 1e-4:
                        break
            X[r] = bestv
        return _norm(X)

    # initial rattler integration attempt
    try:
        Xi = integrate_rattlers(x_best)
        # light joint re-jam from integrated start
        Xr, _ = fast_relax(Xi, iters=250, lr0=0.02, margin0=0.003)
        Xr = run_lbfgs(Xr, 300.0, 60)
        vr = _max_cos(Xr)
        if vr < best_val:
            x_best = Xr.copy()
            best_val = vr
            x_best, best_val = lp_polish(x_best, time.time() + 5.0, r_init=1.0e-3, max_steps=14)
            best_val = _max_cos(x_best)
        else:
            # keep integrated position even if not better, as new basin seed?
            # only keep if close
            if vr < best_val + 0.002:
                # polish integrated without full move? evaluate
                pass
    except Exception:
        pass

    # main shatter-and-repair loop
    trial = 0
    # varying kill sizes
    while time.time() - t0 < BUDGET - 12:
        trial += 1
        try:
            ppm, mcur = _per_max(x_best)
            Gc = x_best @ x_best.T
            np.fill_diagonal(Gc, -2.0)
            deg = np.sum(Gc > mcur - 1e-9, axis=1)
            rats = set(np.where(deg == 0)[0].tolist())
            # choose K cycling 10..24
            Klist = [12, 16, 20, 24, 10, 18, 14, 22]
            K = Klist[(trial - 1) % len(Klist)]
            # rank hubs by degree then per_max
            order_deg = np.argsort(-deg)
            order_worst = np.argsort(-ppm)
            kill = []
            # take top hubs (exclude rattlers)
            for o in order_deg:
                if int(o) in rats:
                    continue
                kill.append(int(o))
                if len(kill) >= K // 2:
                    break
            for o in order_worst:
                if int(o) in kill or int(o) in rats:
                    continue
                kill.append(int(o))
                if len(kill) >= K - 2:
                    break
            # fill remainder randomly (exclude rats + already killed)
            while len(kill) < K:
                c = int(rng.randint(0, n))
                if c in kill or c in rats:
                    continue
                # prefer neighbours of hubs: with prob check contact
                kill.append(c)
            kill = sorted(kill)
            keep_mask = np.ones(n, dtype=bool)
            keep_mask[kill] = False
            cur_keep = x_best[keep_mask].copy()
            # sequential balanced fill
            new_pts = []
            alpha = 0.25 + 0.15 * ((trial % 3) / 2.0)
            for k in range(K):
                if time.time() - t0 > BUDGET - 10:
                    break
                base = cur_keep if len(new_pts) == 0 else np.vstack([cur_keep] + new_pts)
                v = balanced_void(base, n_cand=7000, alpha=alpha)
                v = refine_void(v, base, iters=60, alpha=alpha)
                new_pts.append(v.reshape(1, d))
            cur = np.vstack([cur_keep] + new_pts)
            # random permutation to avoid index bias in lbfgs? keep order but shuffle kill positions
            # reinsert in original positions: simpler just use cur as is (size n)
            cur = _norm(cur)
            # tangential shake to break ties (stronger early, weaker later)
            shake = 0.02 + 0.02 * ((trial % 4) / 3.0)
            R = rng.randn(*cur.shape)
            R -= (R * cur).sum(1, keepdims=True) * cur
            Rn = np.linalg.norm(R, axis=1, keepdims=True) + 1e-12
            cur = _norm(cur + shake * R / Rn)
            # re-optimise
            cur_r, _ = fast_relax(cur, iters=280, lr0=0.028, margin0=0.0045, noise0=0.012 if (trial % 2 == 1) else 0.0)
            if time.time() - t0 > BUDGET - 10:
                break
            cur_r = run_lbfgs(cur_r, 140.0, 50)
            if time.time() - t0 > BUDGET - 9:
                break
            cur_r = run_lbfgs(cur_r, 450.0, 50)
            vcur = _max_cos(cur_r)
            if vcur < best_val:
                # polish immediately
                dl = min(time.time() + 6.0, t0 + BUDGET - 2)
                cur_p, vp = lp_polish(cur_r, dl, r_init=1.0e-3, max_steps=14)
                if vp < vcur:
                    cur_r, vcur = cur_p, vp
                if vcur < best_val - 1e-12:
                    x_best = cur_r.copy()
                    best_val = vcur
                    # try integrating new rattlers if any
                    try:
                        Xi2 = integrate_rattlers(x_best)
                        if time.time() - t0 < BUDGET - 8:
                            Xr2, _ = fast_relax(Xi2, iters=150, lr0=0.015, margin0=0.0025)
                            Xr2 = run_lbfgs(Xr2, 350.0, 40)
                            vr2 = _max_cos(Xr2)
                            if vr2 < best_val:
                                dl2 = min(time.time() + 4.0, t0 + BUDGET - 2)
                                Xr2p, vr2p = lp_polish(Xr2, dl2, r_init=8e-4, max_steps=10)
                                if vr2p < vr2:
                                    Xr2, vr2 = Xr2p, vr2p
                                if vr2 < best_val:
                                    x_best, best_val = Xr2.copy(), vr2
                    except Exception:
                        pass
            elif vcur < best_val + 0.0015 and trial % 7 == 0:
                # occasional polish of near-miss to explore LP vertex
                dl = min(time.time() + 3.0, t0 + BUDGET - 2)
                cur_p, vp = lp_polish(cur_r, dl, r_init=8e-4, max_steps=8)
                if vp < best_val:
                    x_best, best_val = cur_p.copy(), vp
        except Exception:
            continue

    # final squeeze
    try:
        if time.time() - t0 < BUDGET - 1:
            x_best, best_val = lp_polish(x_best, t0 + BUDGET - 0.5, r_init=6e-4, max_steps=10)
    except Exception:
        pass
    return _norm(x_best)

def solve():
    return _solve_inner()
