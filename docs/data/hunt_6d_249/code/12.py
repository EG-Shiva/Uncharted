import time
import numpy as np
from scipy.optimize import linprog, minimize
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def solve():
    t0 = time.time()
    DEADLINE = t0 + 172.0
    np.random.seed(0)
    rng = np.random.default_rng(0)
    n, d = 249, 6

    # load start
    best_x = None
    best_v = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn).astype(float)
            if a.shape == (n, d):
                a = a / np.linalg.norm(a, axis=1, keepdims=True)
                v = _max_cos(a)
                if v < best_v:
                    best_v = v
                    best_x = a.copy()
        except Exception:
            pass
    if best_x is None:
        best_x = rng.normal(size=(n, d))
        best_x /= np.linalg.norm(best_x, axis=1, keepdims=True)
        best_v = _max_cos(best_x)

    cur = best_x.copy()
    cur_v = best_v

    iu0, iu1 = np.triu_indices(n, 1)

    def make_fun(alpha):
        def fun(yflat):
            Y = yflat.reshape(n, d)
            nrm = np.sqrt(np.sum(Y * Y, axis=1))
            nrm = np.maximum(nrm, 1e-12)
            X = Y / nrm[:, None]
            G = X @ X.T
            g = G[iu0, iu1]
            m = g.max()
            e = np.exp(alpha * (g - m))
            S = e.sum()
            f = m + np.log(S) / alpha
            w = e / S
            W = np.zeros((n, n), dtype=float)
            W[iu0, iu1] = w
            W = W + W.T
            Gx = W @ X
            s = np.sum(Gx * X, axis=1)
            Gy = (Gx - X * s[:, None]) / nrm[:, None]
            diff = nrm - 1.0
            f = f + 1e-4 * np.mean(diff * diff)
            Gy = Gy + (X * (2e-4 * diff / n)[:, None])
            return float(f), Gy.ravel().astype(float)
        return fun

    def lse_polish(xinit, alpha, maxiter):
        fun = make_fun(alpha)
        try:
            res = minimize(fun, xinit.ravel(), jac=True, method="L-BFGS-B",
                           options={"maxiter": int(maxiter), "maxfun": int(maxiter*3),
                                    "ftol": 1e-14, "gtol": 1e-7, "maxcor": 12})
            y = res.x.reshape(n, d)
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            v = _max_cos(y)
            return y, v
        except Exception:
            return xinit, _max_cos(xinit)

    # Stage 1: anneal from incumbent
    alphas = [80.0, 250.0, 700.0, 1500.0, 2800.0]
    iters = [120, 140, 160, 180, 220]
    for alpha, mi in zip(alphas, iters):
        if time.time() > DEADLINE - 95:
            break
        y, v = lse_polish(cur, alpha, mi)
        if v < best_v:
            best_v = v
            best_x = y.copy()
        if v < cur_v:
            cur, cur_v = y, v
        else:
            # keep best as continuation if cur stuck
            if best_v < cur_v - 2e-7:
                cur, cur_v = best_x.copy(), best_v

    # Stage 2: basin hops - worst kicks + hole reinsertion
    hop = 0
    sigmas = [0.04, 0.025, 0.015, 0.008, 0.004]
    while time.time() < DEADLINE - 32:
        hop += 1
        if time.time() > DEADLINE - 33:
            break
        G = cur @ cur.T
        np.fill_diagonal(G, -1.0)
        row_max = G.max(axis=1)
        order = np.argsort(row_max)  # ascending
        # alternate move types
        do_hole = (hop % 3 == 0)
        trial = cur.copy()
        if do_hole:
            # replace single worst point by largest hole
            wi = int(order[-1])
            # sample candidates
            M = 8000
            C = rng.normal(size=(M, d))
            C /= np.linalg.norm(C, axis=1, keepdims=True)
            # max dot to others (exclude wi)
            # compute dots M x n
            D = C @ cur.T
            # ignore column wi by setting -2
            D[:, wi] = -2.0
            md = D.max(axis=1)
            bi = int(np.argmin(md))
            # local refine best hole: few projected gradient steps on max-dot
            cand = C[bi].copy()
            step = 0.05
            # mask others
            for _ in range(60):
                dots = cur @ cand
                dots[wi] = -1e9
                # softmax weights to focus on worst
                tau = 0.01
                mm = dots.max()
                e = np.exp((dots - mm) / tau)
                w = e / e.sum()
                grad = (w @ cur)  # weighted mean direction
                # project tangent
                grad_t = grad - np.dot(grad, cand) * cand
                nc = np.linalg.norm(grad_t) + 1e-12
                # move away from crowders
                ncand = cand - step * grad_t / nc * 0.1
                ncand /= np.linalg.norm(ncand) + 1e-12
                # evaluate
                ndots = cur @ ncand
                ndots[wi] = -2.0
                odots = cur @ cand
                odots[wi] = -2.0
                if ndots.max() < odots.max():
                    cand = ncand
                    step = min(step * 1.1, 0.1)
                else:
                    step *= 0.6
                    if step < 1e-4:
                        break
            trial[wi] = cand
            # small shake of neighbours
            # shake top-3 worst slightly to help relax
            K2 = 3
            worst2 = order[-K2:]
            trial[worst2] += rng.normal(scale=0.005, size=(K2, d))
            trial /= np.linalg.norm(trial, axis=1, keepdims=True)
            y, v = lse_polish(trial, 900.0, 110)
        else:
            K = 6 if (hop % 2 == 1) else 10
            sigma = sigmas[(hop // 2) % len(sigmas)]
            worst = order[-K:]
            trial[worst] += rng.normal(scale=sigma, size=(K, d))
            # tiny global jitter
            trial += rng.normal(scale=2e-4, size=(n, d))
            trial /= np.linalg.norm(trial, axis=1, keepdims=True)
            y, v = lse_polish(trial, 900.0 if hop < 30 else 1500.0, 110)
        if v < cur_v - 1e-9:
            cur, cur_v = y, v
            if v < best_v:
                best_v = v
                best_x = y.copy()
        elif v < best_v:
            best_v = v
            best_x = y.copy()
            # jump to best occasionally
            if hop % 5 == 0:
                cur, cur_v = y, v
        # occasional restart from best with shake if stuck many hops
        if hop % 12 == 0 and time.time() < DEADLINE - 45:
            if cur_v > best_v + 5e-7:
                cur, cur_v = best_x.copy(), best_v

    # Stage 3: short re-anneal of best
    if time.time() < DEADLINE - 28:
        for alpha, mi in [(700.0, 120), (1800.0, 150)]:
            if time.time() > DEADLINE - 26:
                break
            y, v = lse_polish(best_x, alpha, mi)
            if v < best_v:
                best_v = v
                best_x = y.copy()
        cur, cur_v = best_x.copy(), best_v

    # Stage 4: LP trust-region polish to jam ties
    def lp_polish(x, t_end):
        x = x / np.linalg.norm(x, axis=1, keepdims=True)
        nn, dd = x.shape
        best = _max_cos(x)
        radius = 6e-4
        iut0, iut1 = np.triu_indices(nn, 1)
        while time.time() < t_end - 0.8 and radius > 1e-14:
            c = (x @ x.T)[iut0, iut1]
            m = c.max()
            sel_thr = m - max(20 * radius, 5e-7)
            sel = c > sel_thr
            cnt = int(np.count_nonzero(sel))
            if cnt > 6000:
                idx = np.argsort(c)[-6000:]
                msel = np.zeros_like(sel)
                msel[idx] = True
                sel = msel
                cnt = 6000
            if cnt == 0:
                break
            I = iut0[sel]; J = iut1[sel]
            p, nv = cnt, nn * dd + 1
            rows = np.repeat(np.arange(p), 2 * dd + 1)
            cols = np.concatenate([I[:, None] * dd + np.arange(dd), J[:, None] * dd + np.arange(dd), np.full((p, 1), nn * dd)], 1).ravel()
            vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
            try:
                A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
                A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(nn), dd), np.arange(nn * dd))), shape=(nn, nv)).tocsr()
            except Exception:
                break
            obj = np.zeros(nv); obj[-1] = 1.0
            try:
                tl = max(0.5, t_end - time.time() - 0.4)
                res = linprog(obj, A_ub=A_ub, b_ub=-c[sel], A_eq=A_eq, b_eq=np.zeros(nn),
                              bounds=[(-radius, radius)] * (nn * dd) + [(None, None)], method="highs",
                              options={"time_limit": tl})
            except Exception:
                radius *= 0.5
                continue
            if res.status != 0:
                radius *= 0.5
                continue
            y = x + res.x[:-1].reshape(nn, dd)
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            vv = _max_cos(y)
            if vv < best:
                x, best = y, vv
                radius = min(radius * 1.5, 0.02)
            else:
                radius *= 0.5
        return x, best

    if time.time() < DEADLINE - 3:
        try:
            yp, vp = lp_polish(best_x, DEADLINE - 0.5)
            if vp < best_v:
                best_v = vp
                best_x = yp
        except Exception:
            pass

    best_x = best_x / np.linalg.norm(best_x, axis=1, keepdims=True)
    return best_x
