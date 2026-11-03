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

def _contact_deg(X, m, tol=2e-9):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return np.sum(G > m - tol, axis=1)

def _relax(X, iters, lr0=0.025, margin0=0.004, rng=None, noise0=0.0):
    X = _norm(X)
    n = X.shape[0]
    cur = X.copy()
    best = _max_cos(cur)
    bestX = cur.copy()
    lr = lr0
    margin = margin0
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
        gn = float(np.max(np.linalg.norm(Gt, axis=1)))
        if not np.isfinite(gn) or gn < 1e-14:
            margin *= 0.6
            if margin < 5e-7:
                break
            continue
        step = (lr / gn) * Gt
        if noise0 > 0 and rng is not None:
            R = rng.randn(*cur.shape)
            R -= (R * cur).sum(1, keepdims=True) * cur
            rn = np.linalg.norm(R, axis=1, keepdims=True) + 1e-12
            dec = 1.0 - it / float(max(1, iters))
            step = step + (noise0 * dec * lr) * R / rn
        Xn = _norm(cur - step)
        G2 = Xn @ Xn.T
        np.fill_diagonal(G2, -2.0)
        mn = float(np.max(G2))
        allow = 1e-12 if noise0 == 0 else 2e-4 * (1.0 - it / float(max(1, iters)))
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

def _lbfgs(X, beta, maxiter):
    X = _norm(X)
    n, d = X.shape
    iu0, iu1 = np.triu_indices(n, 1)
    y0 = X.ravel().copy()
    def fun(y):
        Y = y.reshape(n, d)
        rn = np.linalg.norm(Y, axis=1, keepdims=True)
        rn = np.maximum(rn, 1e-12)
        Xn = Y / rn
        G = Xn @ Xn.T
        g = G[iu0, iu1]
        m = float(g.max())
        e = np.exp(beta * (g - m))
        S = float(e.sum()) + 1e-300
        loss = m + np.log(S) / beta
        w = e / S
        W = np.zeros((n, n))
        W[iu0, iu1] = w
        W = W + W.T
        GX = W @ Xn
        dot = (GX * Xn).sum(axis=1, keepdims=True)
        GY = (GX - dot * Xn) / rn
        return float(loss), GY.ravel()
    try:
        res = minimize(fun, y0, jac=True, method="L-BFGS-B",
                       options={"maxiter": maxiter, "ftol": 1e-12, "gtol": 1e-8, "maxfun": 20000})
        Y = res.x.reshape(n, d)
        return _norm(Y)
    except Exception:
        return X

def _lp_polish(X, t0, BUDGET, deadline, r_init=1.2e-3, max_steps=25):
    X = _norm(X.copy())
    n, d = X.shape
    iu0, iu1 = np.triu_indices(n, 1)
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
        thr = m - max(20.0 * radius, 1e-9)
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
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, n * d + 1)).tocsr()
        A_eq = coo_matrix((X.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, n * d + 1)).tocsr()
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

def _void_pick(X, rng, n_cand=6000):
    n, d = X.shape
    C = _norm(rng.randn(n_cand, d))
    D = C @ X.T
    worst = D.max(axis=1)
    bi = int(np.argmin(worst))
    return C[bi]

def _refine_void(v, others, iters=150, lr0=0.08):
    v = v / max(1e-12, np.linalg.norm(v))
    lr = lr0
    best = float(np.max(others @ v))
    bestv = v.copy()
    for _ in range(iters):
        dots = others @ v
        m = float(np.max(dots))
        tau = 0.002
        w = np.exp((dots - m) / tau)
        w[dots < m - 0.012] = 0.0
        g = w @ others
        gt = g - float(g @ v) * v
        nrm = float(np.linalg.norm(gt))
        if nrm < 1e-14 or not np.isfinite(nrm):
            break
        vn = v - (lr / nrm) * gt
        vn /= max(1e-12, np.linalg.norm(vn))
        mn = float(np.max(others @ vn))
        if mn < m:
            v = vn
            if mn < best:
                best = mn
                bestv = vn.copy()
        else:
            lr *= 0.7
            if lr < 1e-5:
                break
    return bestv, best

def solve():
    t0 = time.time()
    BUDGET = 168.0
    rng = np.random.RandomState(7)
    n, d = 219, 6
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
    x_init_best = x_best.copy()

    def anneal_reopt(X, fast_iters=300, betas=(120.0, 350.0), iters_list=(60, 60)):
        Xc, _ = _relax(X, iters=fast_iters, lr0=0.025, margin0=0.004, rng=rng, noise0=0.0)
        for b, mi in zip(betas, iters_list):
            if time.time() - t0 > BUDGET - 8:
                break
            Xc = _lbfgs(Xc, b, mi)
        return Xc, _max_cos(Xc)

    # ---- N-down: 219 -> 200 (delete 19 worst incl rattler) ----
    try:
        ppm, mcur = _per_max(x_best)
        deg = _contact_deg(x_best, mcur, tol=1e-9)
        ratt = list(np.where(deg == 0)[0])
        order = np.argsort(-ppm)
        kill = []
        for r in ratt:
            kill.append(int(r))
            if len(kill) >= 2:
                break
        for o in order:
            if int(o) not in kill:
                kill.append(int(o))
            if len(kill) >= 19:
                break
        kill = np.array(sorted(kill[:19]), dtype=int)
        keep = np.array([i for i in range(n) if i not in set(kill.tolist())], dtype=int)
        X200 = x_best[keep].copy()
        # optimise 200 freely until jammed
        X200, _ = _relax(X200, iters=600, lr0=0.03, margin0=0.005)
        if time.time() - t0 < BUDGET - 60:
            X200 = _lbfgs(X200, 200.0, 90)
        if time.time() - t0 < BUDGET - 55:
            X200 = _lbfgs(X200, 600.0, 90)
        if time.time() - t0 < BUDGET - 50:
            X200, _ = _lp_polish(X200, t0, BUDGET, t0 + 14.0, r_init=1.2e-3, max_steps=12)
        # ---- regrow 200 -> 219 one at a time ----
        cur = X200.copy()
        for target in range(201, 220):
            if time.time() - t0 > BUDGET - 45:
                break
            v = _void_pick(cur, rng, n_cand=7000)
            v, _ = _refine_void(v, cur, iters=120)
            cur = np.vstack([cur, v.reshape(1, d)])
            cur = _norm(cur)
            # annealed soft-max global reopt + light relax
            a, _ = _relax(cur, iters=220, lr0=0.022, margin0=0.0035)
            # keep better of relaxed vs direct by true max? use relaxed start for lbfgs
            cur2 = _lbfgs(a, 110.0, 45)
            if time.time() - t0 < BUDGET - 40:
                cur2 = _lbfgs(cur2, 380.0, 45)
            cur = cur2
            # LP polish after every addition (short) except to save time every 2nd does full
            if target % 2 == 0 or target == 219:
                dl = min(time.time() + 2.2, t0 + BUDGET - 30)
                if time.time() < dl - 0.3:
                    cur, _ = _lp_polish(cur, t0, BUDGET, dl, r_init=9e-4, max_steps=8)
            # small void jitter of newest point occasionally
            if target < 219 and (target % 5 == 0):
                # shake newest slightly via void re-pick if it is worst
                ppmc, _ = _per_max(cur)
                wi = int(np.argmax(ppmc))
                if wi == len(cur) - 1:
                    vv = _void_pick(cur[np.arange(len(cur)) != wi], rng, n_cand=3000)
                    vv, _ = _refine_void(vv, cur[np.arange(len(cur)) != wi], iters=60)
                    cur[wi] = vv
                    cur = _norm(cur)
        if cur.shape[0] == n:
            # final tighten of grown code
            grown = _lbfgs(cur, 500.0, 70)
            gv = _max_cos(grown)
            # LP squeeze
            if time.time() - t0 < BUDGET - 25:
                grown2, gv2 = _lp_polish(grown, t0, BUDGET, time.time() + 6.0, r_init=1.0e-3, max_steps=14)
                if gv2 < gv:
                    grown, gv = grown2, gv2
            if gv < best_val:
                best_val = gv
                x_best = grown.copy()
    except Exception:
        pass

    # ---- reverse: overfilled 225-230 prune back (alternating trials) ----
    try:
        if time.time() - t0 < BUDGET - 30:
            base = x_best.copy()
            # choose overfill size depending on remaining time
            Nextra = 8 if (BUDGET - (time.time() - t0)) > 45 else 6
            curO = base.copy()
            for _ in range(Nextra):
                v = _void_pick(curO, rng, n_cand=6000)
                v, _ = _refine_void(v, curO, iters=100)
                curO = np.vstack([curO, v.reshape(1, d)])
            curO = _norm(curO)
            # optimise overfilled
            curO, _ = _relax(curO, iters=350, lr0=0.025, margin0=0.004)
            if time.time() - t0 < BUDGET - 25:
                curO = _lbfgs(curO, 160.0, 60)
            if time.time() - t0 < BUDGET - 22:
                curO = _lbfgs(curO, 450.0, 60)
            # prune worst one-by-one back to 219 with reopt
            while curO.shape[0] > n and time.time() - t0 < BUDGET - 14:
                ppmO, _ = _per_max(curO)
                wi = int(np.argmax(ppmO))
                mask = np.arange(curO.shape[0]) != wi
                curO = curO[mask].copy()
                curO, _ = _relax(curO, iters=180, lr0=0.02, margin0=0.003)
                curO = _lbfgs(curO, 300.0, 40)
            if curO.shape[0] == n:
                ov = _max_cos(curO)
                if time.time() - t0 < BUDGET - 10:
                    curO2, ov2 = _lp_polish(curO, t0, BUDGET, time.time() + 5.0, r_init=1.0e-3, max_steps=12)
                    if ov2 < ov:
                        curO, ov = curO2, ov2
                if ov < best_val:
                    best_val = ov
                    x_best = curO.copy()
    except Exception:
        pass

    # ---- final squeeze of global best ----
    try:
        if time.time() - t0 < BUDGET - 4:
            Xf = _lbfgs(x_best, 600.0, 80)
            vf = _max_cos(Xf)
            if vf < best_val:
                best_val = vf
                x_best = Xf.copy()
        if time.time() - t0 < BUDGET - 2:
            dl = t0 + BUDGET - 0.8
            Xp, vp = _lp_polish(x_best, t0, BUDGET, dl, r_init=1.0e-3, max_steps=25)
            if vp < best_val:
                best_val = vp
                x_best = Xp.copy()
    except Exception:
        pass
    # safety: if somehow worse than initial, return initial
    try:
        iv = _max_cos(x_init_best)
        if _max_cos(x_best) > iv and iv < 0.694:
            x_best = x_init_best.copy()
    except Exception:
        pass
    return _norm(np.asarray(x_best, dtype=float))
