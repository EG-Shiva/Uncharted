import time
import numpy as np

def _norm(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return X / n

def _maxcos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def solve():
    t0 = time.time()
    DEADLINE = 172.0
    rng = np.random.default_rng(12345)
    np.random.seed(0)
    n, d = 93, 13

    best = None
    best_val = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=np.float64)
            if a.shape != (93, 13):
                continue
            a = _norm(a)
            v = _maxcos(a)
            if v < best_val:
                best_val = v
                best = a.copy()
        except Exception:
            pass
    if best is None:
        best = _norm(rng.standard_normal((n, d)))
        best_val = _maxcos(best)

    iu0, iu1 = np.triu_indices(n, 1)

    def adam_anneal(X, steps, tau0, tau1, lr0, b1=0.9, b2=0.999, eps=1e-8):
        m = np.zeros_like(X)
        v = np.zeros_like(X)
        t = 0
        ltau = np.log(tau1 / tau0) if tau0 != tau1 else 0.0
        for s in range(steps):
            if time.time() - t0 > DEADLINE:
                break
            frac = s / max(1, steps - 1)
            tau = tau0 * np.exp(ltau * frac)
            lr = lr0 * (0.35 + 0.65 * (1.0 - frac))
            G = X @ X.T
            np.fill_diagonal(G, -2.0)
            mg = float(np.max(G))
            W = np.exp((G - mg) / tau)
            np.fill_diagonal(W, 0.0)
            grad = W @ X
            dot = np.sum(grad * X, axis=1, keepdims=True)
            gt = grad - dot * X
            t += 1
            m = b1 * m + (1.0 - b1) * gt
            v = b2 * v + (1.0 - b2) * (gt * gt)
            mh = m / (1.0 - b1 ** t)
            vh = v / (1.0 - b2 ** t)
            step = lr * mh / (np.sqrt(vh) + eps)
            # project step to tangent to reduce norm drift
            sd = np.sum(step * X, axis=1, keepdims=True)
            step_t = step - sd * X
            X = _norm(X - step_t)
        return X

    def lp_polish(X, secs, r_init=0.002, p_cap=1800):
        from scipy.optimize import linprog
        from scipy.sparse import coo_matrix
        t_start = time.time()
        X = _norm(X.copy())
        cur = _maxcos(X)
        radius = r_init
        while (time.time() - t_start < secs and time.time() - t0 < DEADLINE - 1.0 and radius > 1e-13):
            G = X @ X.T
            c_all = G[iu0, iu1]
            m = float(np.max(c_all))
            delta = max(12.0 * radius, 2e-4)
            # select near-active
            if delta >= 1.0:
                sel_idx = np.arange(len(c_all))
            else:
                sel_idx = np.where(c_all > m - delta)[0]
            if len(sel_idx) == 0:
                sel_idx = np.array([int(np.argmax(c_all))])
            if len(sel_idx) > p_cap:
                # keep largest p_cap
                part = np.argpartition(-c_all[sel_idx], p_cap)[:p_cap]
                sel_idx = sel_idx[part]
            I = iu0[sel_idx]
            J = iu1[sel_idx]
            p = len(sel_idx)
            c_sel = c_all[sel_idx]
            nv = n * d + 1
            rows = np.repeat(np.arange(p), 2 * d + 1)
            col_I = (I[:, None] * d + np.arange(d)).ravel()
            col_J = (J[:, None] * d + np.arange(d)).ravel()
            cols = np.empty(p * (2 * d + 1), dtype=np.int64)
            cols[0::(2*d+1)] = 0  # placeholder, will overwrite vectorized below
            # build cols vectorized
            cols = np.concatenate([ (I[:, None]*d + np.arange(d)).ravel() if False else None ], axis=0) if False else None
            # simpler: build with tile
            cb = np.empty((p, 2*d+1), dtype=np.int64)
            cb[:, :d] = I[:, None]*d + np.arange(d)
            cb[:, d:2*d] = J[:, None]*d + np.arange(d)
            cb[:, 2*d] = n*d
            cols = cb.ravel()
            vals = np.concatenate([X[J].ravel(), X[I].ravel(), -np.ones(p)], axis=0)
            # vals order must match cols order row-major: [xJ row, xI row, -1] per row
            vals2 = np.empty((p, 2*d+1), dtype=np.float64)
            vals2[:, :d] = X[J]
            vals2[:, d:2*d] = X[I]
            vals2[:, 2*d] = -1.0
            vals = vals2.ravel()
            try:
                A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
                # A_eq: x_i . dx_i = 0
                er = np.repeat(np.arange(n), d)
                ec = np.arange(n*d)
                ev = X.ravel()
                A_eq = coo_matrix((ev, (er, ec)), shape=(n, nv)).tocsr()
                obj = np.zeros(nv); obj[-1] = 1.0
                bnds = [(-radius, radius)]*(n*d) + [(None, None)]
                res = linprog(obj, A_ub=A_ub, b_ub=-c_sel, A_eq=A_eq, b_eq=np.zeros(n), bounds=bnds, method="highs")
            except Exception:
                radius *= 0.5
                continue
            if res.status != 0:
                radius *= 0.5
                continue
            dx = res.x[:-1].reshape(n, d)
            # limit predicted improvement check
            Y = _norm(X + dx)
            vv = _maxcos(Y)
            if vv < cur - 1e-12:
                X, cur = Y, vv
                radius = min(radius * 1.5, 0.02)
            else:
                radius *= 0.5
                # if very close, try accepting tiny move that keeps same max but reduces near contacts? no
            # early stop if radius tiny and no progress
        return X, cur

    def void_reinsert(X, rng):
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        rowmax = G.max(axis=1)
        k = int(np.argmax(rowmax))
        M = 5000
        C = _norm(rng.standard_normal((M, d)))
        D = C @ X.T
        D[:, k] = -2.0
        worst = D.max(axis=1)
        idx = int(np.argmin(worst))
        Xn = X.copy()
        Xn[k] = C[idx]
        return Xn

    def pair_split(X, rng, alpha=0.02):
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        flat = int(np.argmax(G))
        a = flat // n; b = flat % n
        Xn = X.copy()
        # push apart along tangent
        Xn[a] = Xn[a] - alpha * X[b]
        Xn[b] = Xn[b] - alpha * X[a]
        Xn += 0.002 * rng.standard_normal(Xn.shape)
        return _norm(Xn)

    # initial LP squeeze of loaded best (short)
    try:
        Xp, vp = lp_polish(best, secs=4.0, r_init=0.0015)
        if vp < best_val:
            best, best_val = Xp, vp
    except Exception:
        pass

    # elite pool for diversity
    elites = [(best_val, best.copy())]

    sigmas = [0.006, 0.012, 0.025, 0.05, 0.09, 0.16]
    HOP_END = t0 + 148.0
    trial = 0
    while time.time() < HOP_END and time.time() - t0 < DEADLINE - 28:
        # pick base: mostly best, sometimes 2nd elite
        if len(elites) > 1 and (trial % 7 == 6):
            # second best
            base = elites[1][1]
        else:
            base = best
        sigma = sigmas[trial % len(sigmas)]
        mode = trial % 5
        try:
            if mode == 0:
                Xt = _norm(base + sigma * rng.standard_normal(base.shape))
            elif mode == 1:
                G = base @ base.T
                np.fill_diagonal(G, -2.0)
                rm = G.max(axis=1)
                k = 8 if sigma < 0.05 else 5
                worst = np.argsort(rm)[-k:]
                Xt = base.copy()
                Xt[worst] += sigma * rng.standard_normal((k, d))
                Xt += 0.002 * rng.standard_normal(Xt.shape)
                Xt = _norm(Xt)
            elif mode == 2:
                Xt = void_reinsert(base, rng)
                Xt += (sigma*0.3) * rng.standard_normal(Xt.shape)
                Xt = _norm(Xt)
            elif mode == 3:
                Xt = pair_split(base, rng, alpha=min(sigma, 0.03))
            else:
                # block rotation of random subset
                Xt = base.copy()
                sub = rng.choice(n, size=12, replace=False)
                Xt[sub] += sigma * rng.standard_normal((12, d))
                Xt = _norm(Xt)
        except Exception:
            Xt = _norm(base + sigma * rng.standard_normal(base.shape))

        # anneal
        steps = 900 if sigma < 0.06 else 700
        tau0 = 0.012 if trial % 2 == 0 else 0.018
        Xt = adam_anneal(Xt, steps=steps, tau0=tau0, tau1=0.0015, lr0=0.006)
        if time.time() - t0 > DEADLINE - 2:
            v = _maxcos(Xt)
            if v < best_val:
                best, best_val = Xt, v
            break
        # quick LP polish
        try:
            Xt2, v2 = lp_polish(Xt, secs=1.2, r_init=0.0012)
        except Exception:
            v2 = _maxcos(Xt); Xt2 = Xt
        if v2 < best_val - 1e-12:
            best, best_val = Xt2, v2
            elites.append((best_val, best.copy()))
            elites.sort(key=lambda t: t[0])
            elites = elites[:3]
        elif v2 < elites[-1][0] and len(elites) < 3:
            elites.append((v2, Xt2.copy()))
            elites.sort(key=lambda t: t[0])
        trial += 1
        # adapt: if time low break
        if trial > 120:
            break

    # final intensive polish on best
    remain = DEADLINE - (time.time() - t0) - 1.0
    if remain > 2:
        try:
            # a bit of extra adam at low temp to smooth then LP
            best = adam_anneal(best, steps=min(600, int(remain*250)), tau0=0.003, tau1=0.0006, lr0=0.002)
            remain2 = DEADLINE - (time.time() - t0) - 0.8
            if remain2 > 2:
                Xf, vf = lp_polish(best, secs=remain2, r_init=0.0015)
                if vf < best_val:
                    best, best_val = Xf, vf
        except Exception:
            pass

    best = _norm(best)
    return best
