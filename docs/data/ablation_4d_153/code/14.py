import time
import numpy as np


def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(np.max(g))


def _random_sphere(n, d, rng):
    x = rng.standard_normal(size=(n, d))
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x


def _smooth_opt(x, steps, tau0, tau1, lr0, lr1, mom=0.88):
    n, d = x.shape
    vel = np.zeros_like(x)
    lt0, lt1 = np.log(tau0), np.log(tau1)
    ll0, ll1 = np.log(lr0), np.log(lr1)
    for t in range(steps):
        f = t / max(1, steps - 1)
        tau = float(np.exp(lt0 + (lt1 - lt0) * f))
        lr = float(np.exp(ll0 + (ll1 - ll0) * f))
        G = x @ x.T
        np.fill_diagonal(G, -np.inf)
        m = float(np.max(G))
        W = np.exp((G - m) / tau)
        np.fill_diagonal(W, 0.0)
        row = np.sum(W, axis=1, keepdims=True)
        GX = W @ x
        grad = np.zeros_like(x)
        mask = row[:, 0] > 1e-12
        grad[mask] = GX[mask] / row[mask]
        dot = np.sum(grad * x, axis=1, keepdims=True)
        grad -= dot * x
        gn = np.linalg.norm(grad, axis=1, keepdims=True)
        over = gn[:, 0] > 3.0
        if np.any(over):
            grad[over] *= (3.0 / gn[over])
        vel = mom * vel + (1.0 - mom) * grad
        bc = 1.0 - mom ** (t + 1)
        step = vel / bc
        x = x - lr * step
        nr = np.linalg.norm(x, axis=1, keepdims=True)
        nr[nr == 0] = 1.0
        x = x / nr
    return x


def _lp_polish(x, deadline):
    from scipy.optimize import linprog
    from scipy.sparse import coo_matrix
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    iu = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        G = x @ x.T
        c = G[iu]
        m = float(c.max())
        thresh = m - max(20.0 * radius, 1e-9)
        idx = np.where(c > thresh)[0]
        p0 = idx.size
        if p0 == 0:
            break
        if p0 > 3000:
            part = np.argpartition(c[idx], -3000)[-3000:]
            idx = idx[part]
            p0 = 3000
        csel = c[idx]
        I = iu[0][idx]
        J = iu[1][idx]
        p = p0
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cI = I[:, None] * d + np.arange(d)
        cJ = J[:, None] * d + np.arange(d)
        cols = np.concatenate([cI, cJ, np.full((p, 1), n * d)], axis=1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], axis=1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix(
            (x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))),
            shape=(n, nv)).tocsr()
        obj = np.zeros(nv)
        obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq,
                          b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                          method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y = y / np.linalg.norm(y, axis=1, keepdims=True)
        v = _max_cos(y)
        if v < best:
            x = y
            best = v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x, best


def solve():
    t0 = time.time()
    HARD_END = t0 + 175.0
    FINAL_RESERVE = 28.0
    n, d = 153, 4
    rng = np.random.default_rng(0)

    best_x = None
    best_v = 1e9
    for fname in ("best.npy", "record.npy"):
        try:
            a = np.load(fname)
            if a.shape == (n, d):
                a = a / np.linalg.norm(a, axis=1, keepdims=True)
                v = _max_cos(a)
                if v < best_v:
                    best_v = v
                    best_x = a.copy()
        except Exception:
            pass
    if best_x is None:
        best_x = _random_sphere(n, d, rng)
        best_v = _max_cos(best_x)

    global_x = best_x.copy()
    global_v = best_v

    sigmas = [0.002, 0.005, 0.01, 0.02, 0.04, 0.08, 0.15, 0.30, 0.60]

    trial = -1
    while time.time() < HARD_END - FINAL_RESERVE:
        trial += 1
        # time guard for at least a short trial
        time_left = HARD_END - FINAL_RESERVE - time.time()
        if time_left < 1.0:
            break
        if trial % 10 == 9:
            # fresh random start
            x0 = _random_sphere(n, d, rng)
            tau0 = float(rng.uniform(0.02, 0.04))
            tau1 = float(rng.uniform(0.0008, 0.003))
            lr0 = float(rng.uniform(0.06, 0.12))
            lr1 = float(rng.uniform(0.004, 0.012))
            mom = float(rng.uniform(0.82, 0.92))
            steps = int(rng.integers(2000, 3200))
            # cap steps if little time
            if time_left < 8:
                steps = min(steps, 1200)
        else:
            sigma = sigmas[trial % len(sigmas)]
            noise = rng.standard_normal(size=(n, d)) * sigma
            noise -= np.sum(noise * global_x, axis=1, keepdims=True) * global_x
            x0 = global_x + noise
            x0 = x0 / np.linalg.norm(x0, axis=1, keepdims=True)
            # rattler to void relocation every 4th trial
            if trial % 4 == 3:
                try:
                    Gtmp = x0 @ x0.T
                    np.fill_diagonal(Gtmp, -np.inf)
                    rm = np.max(Gtmp, axis=1)
                    gm = float(np.max(rm))
                    ratt = np.where(rm < gm - 3e-4)[0]
                    if ratt.size > 0:
                        S = 15000
                        cand = rng.standard_normal(size=(S, d))
                        cand /= np.linalg.norm(cand, axis=1, keepdims=True)
                        dots = cand @ x0.T
                        md = np.max(dots, axis=1)
                        order = np.argsort(md)
                        chosen = []
                        for idx in order:
                            if len(chosen) >= ratt.size:
                                break
                            c = cand[idx]
                            ok = True
                            for j in chosen:
                                if float(c @ cand[j]) > 0.80:
                                    ok = False
                                    break
                            if ok:
                                chosen.append(int(idx))
                        for r_, ci in zip(ratt, chosen):
                            x0[r_] = cand[ci]
                except Exception:
                    pass
            tau0 = float(rng.uniform(0.012, 0.035))
            tau1 = float(rng.uniform(0.0007, 0.003))
            lr0 = float(rng.uniform(0.04, 0.12))
            lr1 = float(rng.uniform(0.003, 0.012))
            mom = float(rng.uniform(0.80, 0.92))
            steps = int(rng.integers(1200, 2600))
            if time_left < 8:
                steps = min(steps, 1000)

        x1 = _smooth_opt(x0, steps, tau0, tau1, lr0, lr1, mom)
        v1 = _max_cos(x1)

        # refine promising candidates with short LP
        if v1 < global_v + 0.002 and time.time() < HARD_END - FINAL_RESERVE - 2.0:
            try:
                x1p, v1p = _lp_polish(x1, min(time.time() + 1.5, HARD_END - FINAL_RESERVE))
                x1, v1 = x1p, v1p
            except Exception:
                pass

        if v1 < global_v:
            global_x = x1.copy()
            global_v = v1

    # final long polish of the champion
    try:
        gx, gv = _lp_polish(global_x, HARD_END - 1.0)
        global_x = gx
    except Exception:
        pass
    global_x = global_x / np.linalg.norm(global_x, axis=1, keepdims=True)
    return global_x
