import time
import numpy as np
from scipy.sparse import coo_matrix
from scipy.optimize import linprog

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def _norm_rows(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n = np.maximum(n, 1e-12)
    return X / n

def _adam_lse(X0, steps=260, lr=0.006, b0=800.0, b1=8000.0, seed=0):
    n, d = X0.shape
    Y = X0.copy()
    m = np.zeros_like(Y)
    v = np.zeros_like(Y)
    b1m = 0.9
    b2m = 0.999
    eps = 1e-8
    p1 = 1.0
    p2 = 1.0
    l0 = np.log(b0)
    l1 = np.log(b1)
    for t in range(1, steps + 1):
        f = (t - 1) / max(1, steps - 1)
        beta = float(np.exp(l0 * (1.0 - f) + l1 * f))
        nrm = np.sqrt(np.sum(Y * Y, axis=1, keepdims=True))
        nrm = np.maximum(nrm, 1e-12)
        X = Y / nrm
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        M = float(np.max(G))
        if not np.isfinite(M):
            break
        D = G - M
        D *= beta
        np.clip(D, -30.0, 0.0, out=D)
        np.exp(D, out=D)
        np.fill_diagonal(D, 0.0)
        S = float(np.sum(D))
        if not np.isfinite(S) or S <= 0:
            break
        D /= S
        GX = D @ X
        GX *= 2.0
        dot = np.sum(GX * X, axis=1, keepdims=True)
        GY = GX - dot * X
        GY /= nrm
        if not np.all(np.isfinite(GY)):
            break
        m *= b1m
        m += (1.0 - b1m) * GY
        v *= b2m
        v += (1.0 - b2m) * (GY * GY)
        p1 *= b1m
        p2 *= b2m
        Mh = m / (1.0 - p1)
        Vh = v / (1.0 - p2)
        Y -= lr * Mh / (np.sqrt(Vh) + eps)
        if not np.all(np.isfinite(Y)):
            break
    return _norm_rows(Y)

def _lp_polish(x, best_v, deadline, k_top=3200, r0=1.5e-3, maxit=70):
    n, d = x.shape
    x = _norm_rows(x)
    try:
        bv = _max_cos(x)
        if bv < best_v:
            best_v = bv
    except Exception:
        return x, best_v
    iu0, iu1 = np.triu_indices(n, 1)
    ar = np.arange(d)
    radius = r0
    it = 0
    while time.time() < deadline - 1.0 and radius > 1e-13 and it < maxit:
        it += 1
        Gall = x @ x.T
        c_all = Gall[iu0, iu1]
        m = float(np.max(c_all))
        window = max(18.0 * radius, 4e-7)
        sel = c_all > m - window
        cnt = int(np.count_nonzero(sel))
        if cnt == 0:
            break
        if cnt > k_top:
            idx = np.argpartition(-c_all, k_top)[:k_top]
            I = iu0[idx]
            J = iu1[idx]
            c_sel = c_all[idx]
        else:
            I = iu0[sel]
            J = iu1[sel]
            c_sel = c_all[sel]
        p = I.shape[0]
        nv = n * d + 1
        try:
            cols = np.concatenate([I[:, None] * d + ar, J[:, None] * d + ar, np.full((p, 1), n * d)], axis=1).ravel()
            rows = np.repeat(np.arange(p), 2 * d + 1)
            vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], axis=1).ravel()
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv)
            obj[-1] = 1.0
            bounds = [(-radius, radius)] * (n * d) + [(None, float(m))]
            res = linprog(obj, A_ub=A_ub, b_ub=-c_sel, A_eq=A_eq, b_eq=np.zeros(n), bounds=bounds, method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        dx = np.asarray(res.x[:-1], dtype=np.float64).reshape(n, d)
        y = _norm_rows(x + dx)
        try:
            vv = _max_cos(y)
        except Exception:
            radius *= 0.5
            continue
        if vv < best_v - 1e-12 and np.all(np.isfinite(y)):
            x = y
            best_v = vv
            radius = min(radius * 1.4, 0.010)
        else:
            radius *= 0.5
    return x, best_v

def solve():
    t_start = time.time()
    BUDGET = 168.0
    def remaining():
        return BUDGET - (time.time() - t_start)
    n, d = 250, 6
    best = None
    best_val = np.inf
    for fn in ("best.npy", "record.npy"):
        try:
            z = np.asarray(np.load(fn), dtype=np.float64)
            if z.shape == (n, d):
                z = _norm_rows(z)
                if np.all(np.isfinite(z)):
                    vv = _max_cos(z)
                    if vv < best_val:
                        best_val = vv
                        best = z.copy()
        except Exception:
            continue
    if best is None:
        rng = np.random.default_rng(0)
        best = rng.standard_normal((n, d))
        best = _norm_rows(best)
        best_val = _max_cos(best)
    cur = best.copy()
    cur_val = float(best_val)

    # Stage 1: Adam equalise then LP push
    if remaining() > 20.0:
        try:
            Xa = _adam_lse(cur, steps=220, lr=0.006, b0=800.0, b1=6000.0)
            va = _max_cos(Xa)
            # accept if not much worse (equalisation may lift max slightly)
            if va < cur_val + 3e-5 and np.all(np.isfinite(Xa)):
                cur = Xa
                cur_val = va
                if va < best_val:
                    best = Xa.copy()
                    best_val = va
        except Exception:
            pass
    if remaining() > 15.0:
        deadline = t_start + BUDGET - 55.0
        if deadline > time.time() + 5.0:
            try:
                cur, cur_val = _lp_polish(cur, min(cur_val, best_val), deadline, k_top=3200, r0=1.5e-3, maxit=60)
            except Exception:
                pass
            if cur_val < best_val and np.all(np.isfinite(cur)):
                best = cur.copy()
                best_val = cur_val
            else:
                # keep cur at best for next stage
                cur = best.copy()
                cur_val = float(best_val)

    # Stage 2: stronger beta Adam + LP
    if remaining() > 20.0:
        try:
            Xa = _adam_lse(best, steps=260, lr=0.004, b0=1500.0, b1=12000.0)
            va = _max_cos(Xa)
            if va < best_val + 2e-5 and np.all(np.isfinite(Xa)):
                cur = Xa
                cur_val = va
            else:
                cur = best.copy()
                cur_val = float(best_val)
        except Exception:
            cur = best.copy()
            cur_val = float(best_val)
        deadline = t_start + BUDGET - 30.0
        if deadline > time.time() + 5.0:
            try:
                cur, cur_val = _lp_polish(cur, best_val, deadline, k_top=3800, r0=8e-4, maxit=60)
            except Exception:
                pass
            if cur_val < best_val and np.all(np.isfinite(cur)):
                best = cur.copy()
                best_val = cur_val

    # Stage 3: small random kicks to change topology, short Adam+LP
    rng = np.random.default_rng(12345)
    nkick = 3
    for ki in range(nkick):
        if remaining() < 22.0:
            break
        sigma = [1.2e-4, 2.5e-4, 5e-4][ki % 3]
        try:
            noise = rng.standard_normal((n, d))
            # tangent projection w.r.t best
            dot = np.sum(noise * best, axis=1, keepdims=True)
            noise -= dot * best
            cand = _norm_rows(best + sigma * noise)
            # quick Adam smooth (few steps) to re-equalise
            cand2 = _adam_lse(cand, steps=80, lr=0.003, b0=2000.0, b1=8000.0)
            try:
                v2 = _max_cos(cand2)
            except Exception:
                continue
            if v2 > best_val + 5e-5:
                continue
            deadline = time.time() + min(14.0, remaining() - 6.0)
            if deadline <= time.time() + 2.0:
                break
            cand2, v3 = _lp_polish(cand2, best_val, deadline, k_top=2800, r0=6e-4, maxit=25)
            if v3 < best_val and np.all(np.isfinite(cand2)):
                best = cand2.copy()
                best_val = v3
        except Exception:
            continue

    # Stage 4: final fine LP on best
    if remaining() > 8.0:
        deadline = t_start + BUDGET - 1.5
        try:
            curf, vf = _lp_polish(best.copy(), best_val, deadline, k_top=4000, r0=3e-4, maxit=50)
            if vf < best_val and np.all(np.isfinite(curf)):
                best = curf.copy()
                best_val = vf
        except Exception:
            pass

    out = _norm_rows(best.copy())
    if not np.all(np.isfinite(out)):
        out = np.where(np.isfinite(out), out, 0.0)
        nrm = np.linalg.norm(out, axis=1, keepdims=True)
        bad = (nrm[:, 0] < 1e-12) | (~np.isfinite(nrm[:, 0]))
        out[bad, 0] = 1.0
        out = _norm_rows(out)
    return out
