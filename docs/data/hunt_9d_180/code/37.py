import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())


def _per_point_max(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return g.max(axis=1)


def lp_polish(x, seconds, radius_init=1e-3, max_con=2500):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    try:
        best = _max_cos(x)
    except Exception:
        return x
    radius = radius_init
    start = time.time()
    iu0 = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        try:
            G = x @ x.T
            c_full = G[iu0]
            m = float(c_full.max())
        except Exception:
            break
        tol = max(20.0 * radius, 1e-9)
        sel = c_full > (m - tol)
        cnt = int(np.count_nonzero(sel))
        if cnt == 0:
            break
        if cnt > max_con:
            idx = np.where(sel)[0]
            order = np.argsort(-c_full[idx])[:max_con]
            mask = np.zeros_like(sel, dtype=bool)
            mask[idx[order]] = True
            sel = mask
            cnt = max_con
        I = iu0[0][sel]
        J = iu0[1][sel]
        c = c_full[sel]
        p = cnt
        nv = n * d + 1
        try:
            rows = np.repeat(np.arange(p), 2 * d + 1)
            c1 = np.arange(d)
            cols = np.concatenate([I[:, None] * d + c1, J[:, None] * d + c1, np.full((p, 1), n * d)], 1).ravel()
            vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        except Exception:
            radius *= 0.5
            continue
        obj = np.zeros(nv)
        obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs",
                          options={"time_limit": 1.0})
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        try:
            y = x + res.x[:-1].reshape(n, d)
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            v = _max_cos(y)
        except Exception:
            radius *= 0.5
            continue
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
        if time.time() - start >= seconds:
            break
    return x


def largest_void(x, rng, n_cand=16000, refine=120):
    n, d = x.shape
    try:
        cand = rng.randn(n_cand, d)
        cand /= np.linalg.norm(cand, axis=1, keepdims=True)
        dots = cand @ x.T
        vals = dots.max(axis=1)
        i0 = int(np.argmin(vals))
        r = cand[i0].copy()
        best_v = float(vals[i0])
    except Exception:
        r = rng.randn(d)
        r /= np.linalg.norm(r)
        best_v = float(np.max(x @ r))
        return r, best_v
    step = 0.05
    for _ in range(refine):
        try:
            dr = x @ r
            j = int(np.argmax(dr))
            g = x[j] - float(x[j] @ r) * r
            ng = float(np.linalg.norm(g))
            if not np.isfinite(ng) or ng < 1e-12:
                break
            g /= ng
            r2 = r - step * g
            n2 = float(np.linalg.norm(r2))
            if not np.isfinite(n2) or n2 < 1e-12:
                step *= 0.5
                continue
            r2 /= n2
            v2 = float(np.max(x @ r2))
            if v2 < best_v:
                r, best_v = r2, v2
            else:
                step *= 0.7
                if step < 1e-5:
                    break
        except Exception:
            break
    return r, best_v


def solve():
    np.random.seed(1)
    t0 = time.time()
    deadline = t0 + 176.0
    delete_deadline = t0 + 100.0
    rng = np.random.RandomState(1)
    best = None
    best_val = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=float)
            if a.shape == (180, 9):
                a = a / np.linalg.norm(a, axis=1, keepdims=True)
                if np.all(np.isfinite(a)):
                    v = _max_cos(a)
                    if v < best_val:
                        best_val = v
                        best = a.copy()
        except Exception:
            pass
    if best is None:
        r0 = np.random.RandomState(0)
        best = r0.randn(180, 9)
        best /= np.linalg.norm(best, axis=1, keepdims=True)
        best_val = _max_cos(best)
    # quick snap not needed; best already polished
    ncyc = 0
    # alternate modes 0,1,2
    while time.time() < delete_deadline and ncyc < 14:
        if time.time() > deadline - 45.0:
            break
        mode = ncyc % 3
        ncyc += 1
        try:
            G = best @ best.T
            np.fill_diagonal(G, -2.0)
            m0 = float(G.max())
            # contacts with tight tol
            thr_c = m0 - 1e-7
            contacts = np.count_nonzero(G > thr_c, axis=1)
            ppm = G.max(axis=1)
            # sum of near-max for tie break
            thr_s = m0 - 1e-4
            mask_s = G > thr_s
            sums = (np.where(mask_s, G, 0.0)).sum(axis=1)
        except Exception:
            continue
        try:
            if mode == 0:
                # N-1: delete most overloaded
                order = np.lexsort((-sums, -contacts))
                # lexsort sorts by last key primary? we want contacts primary
                # so recompute: primary contacts desc, secondary sums desc
                idx_sort = np.argsort(contacts * 1e6 + sums * 1e3)
                idx_del = int(idx_sort[-1])
                x179 = np.delete(best, idx_del, axis=0)
                # polish hole open
                rem = delete_deadline - time.time()
                if rem < 4.0:
                    break
                x179 = lp_polish(x179, min(2.5, max(0.8, rem - 3.5)), radius_init=0.01)
                r, _ = largest_void(x179, rng, n_cand=16000, refine=100)
                x180 = np.vstack([x179, r.reshape(1, -1)])
                x180 /= np.linalg.norm(x180, axis=1, keepdims=True)
                rem2 = delete_deadline - time.time()
                if rem2 < 2.0:
                    v = _max_cos(x180)
                    if v < best_val:
                        best, best_val = x180, v
                    break
                x180p = lp_polish(x180, min(3.0, max(0.8, rem2 - 1.0)), radius_init=0.005)
                v = _max_cos(x180p)
                if v < best_val:
                    best, best_val = x180p, v
            elif mode == 1:
                # N-2: delete rattler + high-contact
                ratt = np.where(contacts == 0)[0]
                if len(ratt) > 0:
                    # pick rattler with smallest ppm (most isolated)
                    idx_r = int(ratt[np.argmin(ppm[ratt])])
                else:
                    idx_r = int(np.argmin(contacts * 1e6 + sums))
                idx_h = int(np.argmax(contacts * 1e6 + sums))
                if idx_h == idx_r:
                    idx_h = int(np.argsort(contacts * 1e6 + sums)[-2])
                dels = sorted([idx_r, idx_h])
                x178 = np.delete(best, dels, axis=0)
                rem = delete_deadline - time.time()
                if rem < 5.0:
                    break
                x178 = lp_polish(x178, min(2.0, max(0.8, rem - 4.0)), radius_init=0.01)
                r1, _ = largest_void(x178, rng, n_cand=14000, refine=80)
                xt = np.vstack([x178, r1.reshape(1, -1)])
                xt /= np.linalg.norm(xt, axis=1, keepdims=True)
                r2, _ = largest_void(xt, rng, n_cand=14000, refine=80)
                x180 = np.vstack([xt, r2.reshape(1, -1)])
                x180 /= np.linalg.norm(x180, axis=1, keepdims=True)
                rem2 = delete_deadline - time.time()
                if rem2 < 2.0:
                    v = _max_cos(x180)
                    if v < best_val:
                        best, best_val = x180, v
                    break
                x180p = lp_polish(x180, min(3.0, max(0.8, rem2 - 1.0)), radius_init=0.005)
                v = _max_cos(x180p)
                if v < best_val:
                    best, best_val = x180p, v
            else:
                # 182 -> delete 2 worst
                rem = delete_deadline - time.time()
                if rem < 6.0:
                    break
                r1, _ = largest_void(best, rng, n_cand=12000, refine=60)
                xt = np.vstack([best, r1.reshape(1, -1)])
                xt /= np.linalg.norm(xt, axis=1, keepdims=True)
                r2, _ = largest_void(xt, rng, n_cand=12000, refine=60)
                x182 = np.vstack([xt, r2.reshape(1, -1)])
                x182 /= np.linalg.norm(x182, axis=1, keepdims=True)
                # short polish of 182
                x182 = lp_polish(x182, min(2.0, max(0.7, (rem - 4.0) * 0.4)), radius_init=0.004)
                # delete two worst by per-point max
                try:
                    G2 = x182 @ x182.T
                    np.fill_diagonal(G2, -2.0)
                    ppm2 = G2.max(axis=1)
                    worst = np.argsort(-ppm2)[:2]
                    worst = sorted([int(w) for w in worst])
                    x180 = np.delete(x182, worst, axis=0)
                except Exception:
                    continue
                rem2 = delete_deadline - time.time()
                if rem2 < 2.0:
                    v = _max_cos(x180)
                    if v < best_val:
                        best, best_val = x180, v
                    break
                x180p = lp_polish(x180, min(3.0, max(0.8, rem2 - 1.0)), radius_init=0.005)
                v = _max_cos(x180p)
                if v < best_val:
                    best, best_val = x180p, v
        except Exception:
            continue
    # final long polish, at most 40s
    remain = deadline - time.time() - 1.0
    if remain > 2.0:
        try:
            secs = min(remain, 38.0)
            best = lp_polish(best, secs, radius_init=1e-3)
        except Exception:
            pass
    best = best / np.linalg.norm(best, axis=1, keepdims=True)
    return best
