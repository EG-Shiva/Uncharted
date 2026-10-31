import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(np.max(g))


def _lp_polish(x, deadline, r0=3e-3):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = r0
    iu0 = np.triu_indices(n, 1)
    ar = np.arange(d)
    while time.time() < deadline and radius > 1e-14:
        cmat = x @ x.T
        c = cmat[iu0]
        m = float(np.max(c))
        thr = m - max(20.0 * radius, 1e-9)
        sel = c > thr
        cnt = int(np.sum(sel))
        if cnt == 0:
            break
        # cap constraints for speed: keep tightest 2500
        if cnt > 2500:
            # select tightest 2500
            idx_all = np.where(sel)[0]
            order = np.argsort(-c[idx_all])[:2500]
            keep = np.zeros_like(sel, dtype=bool)
            keep[idx_all[order]] = True
            sel = keep
            cnt = 2500
        I = iu0[0][sel]
        J = iu0[1][sel]
        p = cnt
        nv = n * d + 1
        w = 2 * d + 1
        rows = np.repeat(np.arange(p), w)
        cols2d = np.empty((p, w), dtype=np.int64)
        vals2d = np.empty((p, w), dtype=np.float64)
        cols2d[:, :d] = I[:, None] * d + ar[None, :]
        cols2d[:, d:2 * d] = J[:, None] * d + ar[None, :]
        cols2d[:, 2 * d] = n * d
        vals2d[:, :d] = x[J]
        vals2d[:, d:2 * d] = x[I]
        vals2d[:, 2 * d] = -1.0
        try:
            A_ub = coo_matrix((vals2d.ravel(), (rows, cols2d.ravel())), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv)
            obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-c[sel], A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        yn = np.linalg.norm(y, axis=1, keepdims=True)
        if not np.all(np.isfinite(yn)) or np.any(yn < 1e-12):
            radius *= 0.5
            continue
        y /= yn
        v = _max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.02)
        else:
            radius *= 0.5
    return x, best


def _smooth_descent(x, steps=220, T0=0.025, T1=0.004, lr0=0.6, lr1=0.06, mu=0.85):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    vel = np.zeros_like(x)
    for s in range(steps):
        frac = s / max(steps - 1, 1)
        T = T0 * (T1 / T0) ** frac
        lr = lr0 * (1 - frac) + lr1 * frac
        G = x @ x.T
        np.fill_diagonal(G, -np.inf)
        M = float(np.max(G))
        E = np.exp((G - M) / T)
        S = float(np.sum(E))
        if not np.finite(S) or S <= 0:
            break
        Grad = (E @ x) / S
        dot = np.sum(Grad * x, axis=1, keepdims=True)
        Gt = Grad - dot * x
        vel = mu * vel + Gt
        step = lr * vel
        gn = float(np.max(np.linalg.norm(step, axis=1)))
        maxmove = 0.05
        if gn > maxmove and gn > 0:
            step = step * (maxmove / gn)
        x = x - step
        x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x


def solve():
    t0 = time.time()
    HARD_END = t0 + 172.0
    EXPL_END = t0 + 138.0
    rng = np.random.default_rng(7)

    try:
        xb = np.load("best.npy").astype(np.float64)
    except Exception:
        xb = None
    try:
        xr = np.load("record.npy").astype(np.float64)
    except Exception:
        xr = None
    cands = []
    if xb is not None:
        xb = xb / np.linalg.norm(xb, axis=1, keepdims=True)
        cands.append(xb)
    if xr is not None:
        xr = xr / np.linalg.norm(xr, axis=1, keepdims=True)
        cands.append(xr)
    if len(cands) == 0:
        cands.append(rng.normal(size=(116, 6)))
    best_x = None
    best_v = 1e9
    for c in cands:
        c = c / np.linalg.norm(c, axis=1, keepdims=True)
        v = _max_cos(c)
        if v < best_v:
            best_v = v
            best_x = c.copy()
    n, d = best_x.shape
    cur_x = best_x.copy()
    cur_v = best_v

    Ttemp0 = 0.002
    Ttemp1 = 2e-05

    trial = 0
    while time.time() < EXPL_END:
        trial += 1
        elapsed = time.time() - t0
        frac_e = elapsed / max(EXPL_END - t0, 1e-9)
        Temp = Ttemp0 * (Ttemp1 / Ttemp0) ** min(max(frac_e, 0.0), 1.0)

        # contact info of current
        G0 = cur_x @ cur_x.T
        np.fill_diagonal(G0, -2.0)
        m0 = float(np.max(G0))
        per0 = np.sum(G0 > (m0 - 1e-7), axis=1)
        order = np.argsort(per0)

        Xtry = cur_x.copy()
        r = float(rng.random())
        if r < 0.28:
            sigma = float(rng.choice([0.012, 0.025, 0.04, 0.015]))
            Xtry = Xtry + rng.normal(scale=sigma, size=(n, d))
            Xtry /= np.linalg.norm(Xtry, axis=1, keepdims=True)
            steps = 170
        elif r < 0.43:
            sigma = 0.08 + 0.08 * float(rng.random())
            noise = rng.normal(scale=sigma, size=(n, d))
            k = int(rng.integers(3, 9))
            idx = rng.choice(n, size=k, replace=False)
            noise[idx] *= 2.5
            Xtry = Xtry + noise
            Xtry /= np.linalg.norm(Xtry, axis=1, keepdims=True)
            steps = 300
        elif r < 0.58:
            # hub-targeted kick
            noise = rng.normal(scale=0.012, size=(n, d))
            top = np.argsort(-per0)[:8]
            noise[top] += rng.normal(scale=0.22, size=(8, d))
            ratt = np.where(per0 <= 0.5)[0]
            if len(ratt) > 0:
                noise[ratt] += rng.normal(scale=0.10, size=(len(ratt), d))
            Xtry = Xtry + noise
            Xtry /= np.linalg.norm(Xtry, axis=1, keepdims=True)
            steps = 280
        elif r < 0.75:
            # void-filling reinsertion of 1-3 points
            if int(rng.integers(0, 2)) == 0:
                # pick worst hub else rattler
                if float(rng.random()) < 0.5:
                    idxs = [int(np.argmax(per0))]
                else:
                    ratt = np.where(per0 <= 0.5)[0]
                    if len(ratt) > 0:
                        idxs = [int(rng.choice(ratt))]
                    else:
                        idxs = [int(order[0])]
                    if float(rng.random()) < 0.4:
                        # second point: another rattler / low contact
                        ratt2 = np.where(per0 <= 1.5)[0]
                        cand2 = [i for i in ratt2 if i not in idxs]
                        if len(cand2) > 0:
                            idxs.append(int(rng.choice(np.array(cand2))))
            else:
                k = int(rng.integers(2, 5))
                idxs = list(rng.choice(n, size=k, replace=False))
            keep_mask = np.ones(n, dtype=bool)
            keep_mask[idxs] = False
            others = cur_x[keep_mask]
            for idx in idxs:
                Nc = 3000
                Rc = rng.normal(size=(Nc, d))
                Rc /= np.linalg.norm(Rc, axis=1, keepdims=True)
                dots = Rc @ others.T
                worst = np.max(dots, axis=1)
                bi = int(np.argmin(worst))
                Xtry[idx] = Rc[bi]
                # update others to include this for sequential fill
                others = np.vstack([others, Rc[bi:bi + 1]])
            Xtry = Xtry + rng.normal(scale=0.012, size=(n, d))
            Xtry /= np.linalg.norm(Xtry, axis=1, keepdims=True)
            steps = 300
        elif r < 0.90:
            # worst-pair split
            iu = np.triu_indices(n, 1)
            c = G0[iu]
            K = min(6, len(c))
            topk = np.argsort(-c)[:K]
            for tk in topk:
                i = int(iu[0][tk]); j = int(iu[1][tk])
                # tangent directions away
                tj_i = Xtry[j] - np.dot(Xtry[j], Xtry[i]) * Xtry[i]
                tj_j = Xtry[i] - np.dot(Xtry[i], Xtry[j]) * Xtry[j]
                amp = 0.05 + 0.05 * float(rng.random())
                Xtry[i] -= amp * tj_i / (np.linalg.norm(tj_i) + 1e-12)
                Xtry[j] -= amp * tj_j / (np.linalg.norm(tj_j) + 1e-12)
            Xtry = Xtry + rng.normal(scale=0.015, size=(n, d))
            Xtry /= np.linalg.norm(Xtry, axis=1, keepdims=True)
            steps = 250
        else:
            k = int(rng.integers(2, 6))
            idx = rng.choice(n, size=k, replace=False)
            Rr = rng.normal(size=(k, d))
            Rr /= np.linalg.norm(Rr, axis=1, keepdims=True)
            Xtry[idx] = Rr
            Xtry = Xtry + rng.normal(scale=0.01, size=(n, d))
            Xtry /= np.linalg.norm(Xtry, axis=1, keepdims=True)
            steps = 320

        Xtry = _smooth_descent(Xtry, steps=steps, T0=0.025, T1=0.0035, lr0=0.6, lr1=0.06, mu=0.85)
        v_smooth = _max_cos(Xtry)
        if v_smooth > cur_v + 0.025 and v_smooth > best_v + 0.02:
            continue
        budget = 1.5 if steps <= 200 else 2.0
        deadline = min(time.time() + budget, EXPL_END + 1.0, HARD_END - 12.0)
        if deadline <= time.time() + 0.08:
            break
        Xpol, vpol = _lp_polish(Xtry, deadline)
        # Metropolis acceptance w.r.t. cur
        accept = False
        if vpol < cur_v - 1e-13:
            accept = True
        else:
            if Temp > 0:
                prob = np.exp((cur_v - vpol) / Temp)
                if float(rng.random()) < prob:
                    accept = True
        if accept:
            cur_x = Xpol.copy()
            cur_v = vpol
        if vpol < best_v - 5e-13:
            best_v = vpol
            best_x = Xpol.copy()
        elif vpol < best_v - 1e-14:
            best_x = Xpol.copy()
            best_v = vpol
        # if stuck far above best, occasionally restart from best
        if cur_v > best_v + 0.004 and float(rng.random()) < 0.3:
            cur_x = best_x.copy()
            cur_v = best_v

    remain = HARD_END - time.time()
    if remain > 4.0:
        fp_budget = min(remain - 2.0, 30.0)
        Xf, vf = _lp_polish(best_x, time.time() + fp_budget, r0=2e-3)
        if vf <= best_v:
            best_x, best_v = Xf, vf

    best_x = best_x / np.linalg.norm(best_x, axis=1, keepdims=True)
    return best_x
