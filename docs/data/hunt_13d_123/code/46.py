import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _lp_polish(x, seconds, t_deadline, iu0, iu1, p_cap=1200):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    best_x = x.copy()
    radius = 1e-3
    start = time.time()
    end = min(start + seconds, t_deadline)
    it = 0
    while time.time() < end and radius > 1e-14 and it < 40:
        it += 1
        if time.time() >= t_deadline - 0.5:
            break
        G = x @ x.T
        c_all = G[iu0, iu1]
        m = float(c_all.max())
        window = max(20.0 * radius, 1e-9)
        sel = c_all > m - window
        cnt = int(np.count_nonzero(sel))
        if cnt == 0:
            break
        if cnt > p_cap:
            idx = np.where(sel)[0]
            part = np.argpartition(-c_all[idx], p_cap - 1)[:p_cap]
            keep = np.zeros_like(sel)
            keep[idx[part]] = True
            sel = keep
            cnt = p_cap
        I = iu0[sel]; J = iu1[sel]
        csel = c_all[sel]
        p = cnt
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([(I[:, None] * d + np.arange(d)).reshape(p, d),
                               (J[:, None] * d + np.arange(d)).reshape(p, d),
                               np.full((p, 1), n * d)], axis=1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], axis=1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            erows = np.repeat(np.arange(n), d)
            ecols = np.arange(n * d)
            A_eq = coo_matrix((x.ravel(), (erows, ecols)), shape=(n, nv)).tocsr()
            obj = np.zeros(nv); obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs",
                          options={"time_limit": 1.5})
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        yn = np.linalg.norm(y, axis=1, keepdims=True)
        yn[yn == 0] = 1.0
        y = y / yn
        v = _max_cos(y)
        if v < best:
            best = v
            best_x = y.copy()
            x = y
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
            if radius < 1e-7:
                x = best_x.copy()
        if time.time() >= end:
            break
    return best_x, best

def solve():
    t0 = time.time()
    HARD = t0 + 165.0
    n, d = 123, 13
    rng = np.random.default_rng(0)
    np.random.seed(0)
    iu0, iu1 = np.triu_indices(n, 1)
    best = None
    best_val = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=float)
            if a.shape == (n, d):
                a = a / np.linalg.norm(a, axis=1, keepdims=True)
                if np.all(np.isfinite(a)):
                    v = _max_cos(a)
                    if v < best_val:
                        best_val = v
                        best = a.copy()
        except Exception:
            pass
    if best is None:
        best = rng.normal(size=(n, d))
        best /= np.linalg.norm(best, axis=1, keepdims=True)
        best_val = _max_cos(best)
    # quick 1s squeeze of incumbent (capped)
    try:
        if HARD - time.time() > 20:
            c, v = _lp_polish(best, 1.5, HARD - 1.0, iu0, iu1, p_cap=1200)
            if v < best_val and np.all(np.isfinite(c)):
                best, best_val = c, v
    except Exception:
        pass
    thetas = [0.03, 0.05, 0.08, 0.12, 0.15, 0.04, 0.07, 0.10]
    it = 0
    # precompute contact threshold for surgery
    while time.time() < HARD - 9.0 and it < 200:
        it += 1
        if time.time() >= HARD - 9.0:
            break
        # contact graph within 1e-7 of max on incumbent
        try:
            G = best @ best.T
            np.fill_diagonal(G, -2.0)
            m = float(G.max())
            # degrees with 1e-7
            B = G > m - 1e-7
            deg = B.sum(axis=1)
            # rattler reinsert every 12 iters if rattler with 1e-9
            if it % 12 == 0:
                Br = G > m - 1e-9
                degr = Br.sum(axis=1)
                ratt = np.where(degr == 0)[0]
                if len(ratt) > 0 and time.time() < HARD - 20:
                    r = int(ratt[0])
                    keep_mask = np.ones(n, dtype=bool); keep_mask[r] = False
                    Yr = best[keep_mask]
                    cand = rng.normal(size=(1500, d))
                    cand /= np.linalg.norm(cand, axis=1, keepdims=True)
                    sc = (cand @ Yr.T).max(axis=1)
                    bi = int(np.argmin(sc))
                    y = best.copy(); y[r] = cand[bi]
                    c2, v2 = _lp_polish(y, 2.0, HARD - 1.0, iu0, iu1, p_cap=1200)
                    if v2 < best_val - 1e-12:
                        best, best_val = c2, v2
                        continue
            order = np.argsort(-deg)
            top = order[:12]
            hub = int(top[(it - 1) % len(top)])
            nbrs = np.where(B[hub])[0]
            nbrs = nbrs[nbrs != hub]
            if len(nbrs) < 6:
                continue
            kdrop = int(rng.integers(2, 6))
            kdrop = min(kdrop, len(nbrs) - 3)
            perm = rng.permutation(len(nbrs))
            drop = nbrs[perm[:kdrop]]
            keep_nbrs = nbrs[perm[kdrop:]]
            # tangent basis at hub
            xh = best[hub]
            R = rng.normal(size=(d, d - 1))
            R = R - np.outer(xh, xh @ R)
            Q, _ = np.linalg.qr(R)
            # rigidity stress: keep matrix projected
            Mk = best[keep_nbrs] @ Q  # kk x (d-1)
            try:
                U, S, Vt = np.linalg.svd(Mk, full_matrices=False)
            except Exception:
                continue
            mode_choice = ((it - 1) // 12) % 2
            vidx = -1 - mode_choice
            if abs(vidx) > Vt.shape[0]:
                vidx = -1
            v = Vt[vidx]
            dvec = Q @ v
            dn = np.linalg.norm(dvec)
            if dn < 1e-12 or not np.isfinite(dn):
                continue
            dvec = dvec / dn
            # orient to open drop set: want x_j . dvec < 0 on average
            cd = float((best[drop] @ dvec).mean())
            if not np.isfinite(cd):
                continue
            if cd > 0:
                dvec = -dvec
            elif cd == 0:
                if rng.random() < 0.5:
                    dvec = -dvec
            theta = float(thetas[(it - 1) % len(thetas)])
            # small jitter of mode index sign already cycled; move hub
            y = best.copy()
            y[hub] = np.cos(theta) * xh + np.sin(theta) * dvec
            y[hub] /= np.linalg.norm(y[hub])
            # tiny global tangential jitter to avoid exact saddle
            jmag = 2e-4
            if jmag > 0:
                Z = rng.normal(size=(n, d)) * jmag
                # project tangential
                Z = Z - (Z * y).sum(axis=1, keepdims=True) * y
                y = y + Z
                y /= np.linalg.norm(y, axis=1, keepdims=True)
            # capped polish
            c2, v2 = _lp_polish(y, 2.2, HARD - 1.0, iu0, iu1, p_cap=1200)
            if v2 < best_val - 5e-13 and np.all(np.isfinite(c2)):
                best, best_val = c2, v2
        except Exception:
            # ensure time check, continue
            if time.time() >= HARD - 9.0:
                break
            continue
    # final snap with remaining time
    try:
        remain = HARD - time.time()
        if remain > 1.5:
            c, v = _lp_polish(best, min(remain - 0.5, 6.0), HARD, iu0, iu1, p_cap=1500)
            if v < best_val and np.all(np.isfinite(c)):
                best, best_val = c, v
    except Exception:
        pass
    best = best / np.linalg.norm(best, axis=1, keepdims=True)
    return best
