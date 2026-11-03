import time
import numpy as np
from scipy.optimize import minimize

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def _norm_rows(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n = np.maximum(n, 1e-12)
    return X / n

def _smooth_opt(X0, beta, maxiter, maxcor=8):
    n, d = X0.shape
    Y0 = X0.copy()
    # work on unnormalised Y, init norms=1
    def fg(Yflat):
        Y = Yflat.reshape(n, d)
        nrm = np.linalg.norm(Y, axis=1, keepdims=True)
        nrm = np.maximum(nrm, 1e-12)
        X = Y / nrm
        G = X @ X.T
        # max over strict upper
        # use full max for shift (diag is 1, so mask diag)
        np.fill_diagonal(G, -2.0)
        M = float(np.max(G))
        D = beta * (G - M)
        np.clip(D, -30.0, 0.0, out=D)
        E = np.exp(D, out=D)
        np.fill_diagonal(E, 0.0)
        Spair = float(np.sum(E) * 0.5)
        if Spair <= 0 or not np.isfinite(Spair):
            Spair = 1e-300
        loss = M + np.log(Spair) / beta
        # gradient: GX = E @ X / Spair
        GX = (E @ X) / Spair
        dot = np.sum(GX * X, axis=1, keepdims=True)
        GY = (GX - dot * X) / nrm
        if not np.all(np.isfinite(GY)):
            GY = np.nan_to_num(GY, nan=0.0, posinf=0.0, neginf=0.0)
        return float(loss), GY.ravel()
    try:
        res = minimize(fg, Y0.ravel(), jac=True, method="L-BFGS-B",
                       options={"maxiter": maxiter, "ftol": 1e-20,
                                "gtol": 1e-12, "maxfun": maxiter * 6,
                                "maxcor": maxcor})
        Y = res.x.reshape(n, d)
    except Exception:
        Y = Y0
    Y = _norm_rows(Y)
    return Y

def _deep_hole_candidates(Xfix, rng, n_cand=2000):
    # sample uniform on S^5, return candidate with minimal max-dot
    d = Xfix.shape[1]
    C = rng.standard_normal((n_cand, d))
    C /= np.maximum(np.linalg.norm(C, axis=1, keepdims=True), 1e-12)
    dots = C @ Xfix.T
    m = np.max(dots, axis=1)
    i = int(np.argmin(m))
    return C[i], float(m[i])

def _refine_hole(y0, Xfix, iters=40, step0=0.05, beta=4000.0):
    y = y0 / max(np.linalg.norm(y0), 1e-12)
    best = y.copy()
    dots = Xfix @ y
    best_m = float(np.max(dots))
    cur = y.copy()
    for it in range(iters):
        dots = Xfix @ cur
        M = float(np.max(dots))
        D = beta * (dots - M)
        np.clip(D, -30.0, 0.0, out=D)
        w = np.exp(D)
        s = float(np.sum(w))
        if s <= 0 or not np.isfinite(s):
            break
        w /= s
        g = w @ Xfix
        gt = g - float(g @ cur) * cur
        nn = float(np.linalg.norm(gt))
        if nn < 1e-14:
            break
        gt /= nn
        step = step0 / (1.0 + it * 0.08)
        nxt = cur - step * gt
        nn2 = float(np.linalg.norm(nxt))
        if nn2 < 1e-12:
            continue
        nxt /= nn2
        mn = float(np.max(Xfix @ nxt))
        if mn < best_m:
            best_m = mn
            best = nxt.copy()
        cur = nxt
    return best

def solve():
    t0 = time.time()
    DEADLINE = 168.0
    def rem():
        return DEADLINE - (time.time() - t0)
    n, d = 250, 6
    rng = np.random.default_rng(12345)
    # load
    best = None
    best_val = np.inf
    for fn in ("best.npy", "record.npy"):
        try:
            z = np.asarray(np.load(fn), dtype=np.float64)
            if z.shape == (n, d):
                z = _norm_rows(z)
                if np.all(np.isfinite(z)):
                    v = _max_cos(z)
                    if v < best_val:
                        best_val = v
                        best = z.copy()
        except Exception:
            continue
    if best is None:
        z = rng.standard_normal((n, d))
        z = _norm_rows(z)
        best = z
        best_val = _max_cos(z)
    cur_best = best.copy()
    cur_best_val = float(best_val)

    iu = np.triu_indices(n, 1)

    # stage 0: direct annealed polish of loaded best (no perturbation)
    if rem() > 30:
        try:
            X = cur_best.copy()
            for beta, mi in ((5000.0, 200), (15000.0, 250), (35000.0, 250)):
                if rem() < 15:
                    break
                X = _smooth_opt(X, beta, maxiter=mi)
                v = _max_cos(X)
                if v < cur_best_val:
                    cur_best_val = v
                    cur_best = X.copy()
                # early stop if no progress? continue ladder anyway
        except Exception:
            pass

    # basin hopping with excision / jitter
    hop = 0
    while rem() > 22.0:
        hop += 1
        try:
            G = cur_best @ cur_best.T
            gv = G[iu]
            M = float(np.max(gv))
            # plateau threshold adaptive: include ~1500 tightest
            # use M-2e-6 as plateau window
            thr = M - 2e-6
            sel = gv >= thr
            I = iu[0][sel]; J = iu[1][sel]
            deg = np.zeros(n, dtype=np.int64)
            np.add.at(deg, I, 1)
            np.add.at(deg, J, 1)
            # max pair
            k = int(np.argmax(gv))
            a = int(iu[0][k]); b = int(iu[1][k])
            # build neighbour set of a,b via plateau edges
            adj = [set() for _ in range(n)]
            for u, v_ in zip(I.tolist(), J.tolist()):
                adj[u].add(v_); adj[v_].add(u)
            # BFS component around a
            comp = set([a, b])
            frontier = [a, b]
            for _ in range(2):
                nxt = []
                for u in frontier:
                    for w in adj[u]:
                        if w not in comp and len(comp) < 60:
                            comp.add(w); nxt.append(w)
                frontier = nxt
                if not frontier:
                    break
            comp = list(comp)
            # decide mode
            mode = hop % 3
            Xw = cur_best.copy()
            if mode == 0:
                # global small jitter
                sig = 0.0015 if (hop % 6 == 0) else 0.0008
                Xw = Xw + sig * rng.standard_normal((n, d))
                Xw = _norm_rows(Xw)
                excised = []
            else:
                # excise m points
                if mode == 1:
                    m = int(rng.integers(4, 9))
                else:
                    m = int(rng.integers(8, 15))
                # prefer high-degree + component + random
                order = np.argsort(-deg, kind="stable")
                cand_pool = list(order[:40]) + comp
                # dedupe preserve order
                seen = set(); pool = []
                for u in cand_pool:
                    if u not in seen:
                        seen.add(u); pool.append(u)
                # pick m: half from pool top random, half random rattlers/any
                ratt = np.where(deg == 0)[0].tolist()
                chosen = set()
                rng.shuffle(pool)
                for u in pool:
                    if len(chosen) >= m // 2 + (m % 2):
                        break
                    chosen.add(int(u))
                # fill rest randomly
                while len(chosen) < m:
                    if ratt and rng.random() < 0.5:
                        chosen.add(int(ratt[int(rng.integers(0, len(ratt)))]))
                    else:
                        chosen.add(int(rng.integers(0, n)))
                chosen = list(chosen)
                # sequentially regrow with deep holes
                # fixed mask
                for idx in chosen:
                    mask = np.ones(n, dtype=bool); mask[idx] = True
                    # exclude other chosen not yet placed? use current Xw fixed others
                    Xfix = Xw[mask]
                    y0, _ = _deep_hole_candidates(Xfix, rng, n_cand=2500)
                    y0 = _refine_hole(y0, Xfix, iters=30, step0=0.06, beta=4000.0)
                    Xw[idx] = y0
                Xw = _norm_rows(Xw)
            # quick smooth evaluation ladder (short)
            Xh = Xw
            for beta, mi in ((6000.0, 120), (18000.0, 150)):
                if rem() < 14:
                    break
                Xh = _smooth_opt(Xh, beta, maxiter=mi)
            vh = _max_cos(Xh)
            if vh < cur_best_val - 1e-12:
                # refine further with high beta
                if rem() > 14:
                    Xh2 = _smooth_opt(Xh, 35000.0, maxiter=200)
                    vh2 = _max_cos(Xh2)
                    if vh2 < vh:
                        Xh = Xh2; vh = vh2
                cur_best_val = float(vh)
                cur_best = Xh.copy()
            elif vh < cur_best_val + 5e-6 and rem() > 40 and rng.random() < 0.15:
                # occasional neutral move to change basin (keep search moving)
                # do longer refinement from this basin once
                Xh2 = _smooth_opt(Xh, 30000.0, maxiter=150)
                vh2 = _max_cos(Xh2)
                if vh2 < cur_best_val:
                    cur_best_val = float(vh2); cur_best = Xh2.copy()
        except Exception:
            continue
        # safety: if hop count huge, break
        if hop > 60:
            break

    # final intensive polish
    try:
        X = cur_best.copy()
        schedule = ((8000.0, 200), (20000.0, 250), (45000.0, 250))
        for beta, mi in schedule:
            if rem() < 8:
                break
            Xn = _smooth_opt(X, beta, maxiter=mi)
            v = _max_cos(Xn)
            if v <= _max_cos(X):
                X = Xn
                if v < cur_best_val:
                    cur_best_val = v
                    cur_best = X.copy()
            else:
                # keep best anyway if improved over global
                if v < cur_best_val:
                    cur_best_val = v
                    cur_best = Xn.copy()
                X = Xn if rng.random() < 0.3 else X
    except Exception:
        pass

    out = _norm_rows(cur_best.copy())
    return out
