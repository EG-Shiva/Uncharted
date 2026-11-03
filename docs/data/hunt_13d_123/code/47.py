import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _lp_polish(X, seconds, p_cap=1500):
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    n, d = X.shape
    best = _max_cos(X)
    best_x = X.copy()
    radius = 1e-3
    start = time.time()
    iu0, iu1 = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        G = X @ X.T
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
        I = iu0[sel]; J = iu1[sel]
        csel = c_all[sel]
        p = int(sel.sum())
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([(I[:, None] * d + np.arange(d)).reshape(p, d),
                               (J[:, None] * d + np.arange(d)).reshape(p, d),
                               np.full((p, 1), n * d)], axis=1).ravel()
        vals = np.concatenate([X[J], X[I], -np.ones((p, 1))], axis=1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            erows = np.repeat(np.arange(n), d)
            ecols = np.arange(n * d)
            A_eq = coo_matrix((X.ravel(), (erows, ecols)), shape=(n, nv)).tocsr()
            obj = np.zeros(nv); obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = X + res.x[:-1].reshape(n, d)
        yn = np.linalg.norm(y, axis=1, keepdims=True)
        yn[yn == 0] = 1.0
        y = y / yn
        v = _max_cos(y)
        if v < best:
            best = v
            best_x = y.copy()
            X = y
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
            if radius < 1e-7:
                X = best_x.copy()
        if time.time() - start >= seconds:
            break
    return best_x

def solve():
    t0 = time.time()
    HARD = 168.0
    n, d = 123, 13
    rng = np.random.default_rng(7)

    best = None
    best_val = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=float)
            if a.shape == (n, d):
                a = a / np.linalg.norm(a, axis=1, keepdims=True)
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

    cur = best.copy()
    cur_val = float(best_val)

    # fixed random projector for tangent bases (deterministic)
    G0 = rng.normal(size=(d, d - 1))
    alphas = [0.03, 0.06, 0.10, 0.15, 0.045, 0.09, 0.12, 0.05]

    iu0, iu1 = np.triu_indices(n, 1)
    it = 0
    while time.time() - t0 < HARD - 12:
        it += 1
        # --- contact graph of cur within 1e-7 of max ---
        G = cur @ cur.T
        np.fill_diagonal(G, -2.0)
        mcur = float(G.max())
        thr = mcur - 1e-7
        # degrees for hub selection
        deg = (G > thr).sum(axis=1)
        hubs = np.argsort(-deg)[:6]
        # contact pairs
        sel_all = (G[iu0, iu1] > thr)
        I_all = iu0[sel_all]; J_all = iu1[sel_all]
        c_all = G[iu0[sel_all], iu1[sel_all]]
        p_full = len(I_all)
        if p_full < 10 or p_full > 4000:
            # cap to tightest 2500 if too many
            if p_full > 4000:
                order = np.argsort(-c_all)[:2500]
                I_all = I_all[order]; J_all = J_all[order]
                p_full = 2500
            elif p_full < 10:
                # random shake fallback
                alpha = alphas[(it-1) % len(alphas)]
                pert = rng.normal(size=(n, d)) * 0.05
                pert -= (pert * cur).sum(axis=1, keepdims=True) * cur
                y = cur + pert
                y /= np.linalg.norm(y, axis=1, keepdims=True)
                rem = HARD - (time.time() - t0) - 4
                if rem < 2:
                    break
                yp = _lp_polish(y, min(4.0, rem))
                vp = _max_cos(yp)
                if vp < best_val:
                    best, best_val = yp.copy(), vp
                    cur, cur_val = yp.copy(), vp
                elif vp < cur_val + 0.002:
                    cur, cur_val = yp.copy(), vp
                continue
        # cap rows for SVD speed
        if p_full > 2200:
            order = np.argsort(-c_all)[:2200]
            Ic = I_all[order]; Jc = J_all[order]
        else:
            Ic = I_all; Jc = J_all
        p = len(Ic)

        # --- tangent bases ---
        # B[i] is d x (d-1)
        Bs = np.empty((n, d, d - 1))
        for i in range(n):
            xi = cur[i]
            Gp = G0 - xi[:, None] * (xi @ G0)
            Q, _ = np.linalg.qr(Gp)
            # fix sign deterministically
            Bs[i] = Q

        # --- tangent rigidity matrix p x N ---
        N = n * (d - 1)
        Rt = np.empty((p, N), dtype=float)
        # precompute projections B_i^T x_j
        # loop over pairs (p up to 2200, cheap)
        for k in range(p):
            i = int(Ic[k]); j = int(Jc[k])
            bi = Bs[i].T @ cur[j]   # (d-1,)
            bj = Bs[j].T @ cur[i]
            Rt[k, i*(d-1):(i+1)*(d-1)] = bi
            Rt[k, j*(d-1):(j+1)*(d-1)] = bj
            # need zero elsewhere
        # zero-fill: Rt uninitialized elsewhere -> must zero first
        # We filled only two blocks per row; rest is garbage. Fix:
        # Rebuild correctly with zeros (redo efficiently)
        # To save time, zero out using mask approach: we already allocated garbage,
        # so rebuild in vectorized way:
        # Actually redo properly:
        Rt2 = np.zeros((p, N), dtype=float)
        for k in range(p):
            i = int(Ic[k]); j = int(Jc[k])
            Rt2[k, i*(d-1):(i+1)*(d-1)] = Rt[k, i*(d-1):(i+1)*(d-1)]
            Rt2[k, j*(d-1):(j+1)*(d-1)] = Rt[k, j*(d-1):(j+1)*(d-1)]
        Rt = Rt2
        del Rt2

        # --- sparse SVD approx via dense SVD (smallest modes) ---
        try:
            U, S, Vt = np.linalg.svd(Rt, full_matrices=False)
        except Exception:
            # fallback shake
            alpha = alphas[(it-1) % len(alphas)]
            pert = rng.normal(size=(n, d)) * alpha * 0.3
            pert -= (pert * cur).sum(axis=1, keepdims=True) * cur
            y = cur + pert
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            rem = HARD - (time.time() - t0) - 4
            if rem < 2:
                break
            yp = _lp_polish(y, min(3.0, rem))
            vp = _max_cos(yp)
            if vp < best_val:
                best, best_val = yp.copy(), vp
                cur, cur_val = yp.copy(), vp
            elif vp < cur_val + 0.002:
                cur, cur_val = yp.copy(), vp
            continue

        K = min(16, Vt.shape[0])
        modes = Vt[-K:, :]  # smallest singular values, each (N,)

        # --- choose dropped-contact set involving top hub ---
        hub = int(hubs[(it-1) % len(hubs)])
        # rows incident to hub
        inc = np.where((Ic == hub) | (Jc == hub))[0]
        if len(inc) == 0:
            # rattler hub? pick random rows
            Didx = rng.choice(p, size=min(3, p), replace=False)
        else:
            # sort incident by tightness (use original cosine)
            # need cosine values for Ic/Jc
            cos_inc = np.array([cur[int(Ic[k])] @ cur[int(Jc[k])] for k in inc])
            order = np.argsort(-cos_inc)
            inc_sorted = inc[order]
            nd = 2 + (it % 2)  # 2 or 3 contacts to break
            nd = min(nd, len(inc_sorted))
            # cycle offset so different sets
            off = (it // len(hubs)) % max(1, len(inc_sorted))
            pick = [(off + q) % len(inc_sorted) for q in range(nd)]
            Didx = inc_sorted[pick]
        Dmask = np.zeros(p, dtype=bool)
        Dmask[Didx] = True

        alpha = alphas[(it-1) % len(alphas)]

        # --- score modes: want large negative stretch on D, small elsewhere ---
        best_score = -1e100
        best_v = None
        for mi in range(K):
            v = modes[mi]
            stretch = Rt @ v  # d(cosine)/ds
            drop = float((-stretch[Didx]).mean())
            preserve = float(np.abs(stretch[~Dmask]).mean()) if p - len(Didx) > 0 else 0.0
            # rigidity stress weighting: penalize preserve
            score = drop - 0.35 * preserve
            # also reward moving hub + rattler: norm of hub block
            hub_norm = float(np.linalg.norm(v[hub*(d-1):(hub+1)*(d-1)]))
            score += 0.05 * hub_norm
            # consider both signs
            if score > best_score:
                best_score = score
                best_v = v.copy()
            if -drop - 0.35 * preserve + 0.05 * hub_norm > best_score:
                # flipped sign equivalent to -v: drop' = mean(stretch)
                best_score = -drop - 0.35 * preserve + 0.05 * hub_norm
                best_v = -v.copy()
        if best_v is None:
            continue

        # --- move along mode by alpha radians (average) ---
        Dx = np.empty((n, d))
        for i in range(n):
            Dx[i] = Bs[i] @ best_v[i*(d-1):(i+1)*(d-1)]
        rms = float(np.sqrt((Dx * Dx).sum() / n)) + 1e-20
        s = alpha / rms
        # cap per-point step to avoid blowup
        norms = np.linalg.norm(Dx, axis=1)
        mx = float(norms.max()) * s
        if mx > 0.6:
            s *= 0.6 / mx
        Y = cur + s * Dx
        Y /= np.linalg.norm(Y, axis=1, keepdims=True)

        # quick reject if catastrophically bad
        vY = _max_cos(Y)
        if vY > 0.45:
            # reduce step and retry once
            Y2 = cur + 0.4 * s * Dx
            Y2 /= np.linalg.norm(Y2, axis=1, keepdims=True)
            if _max_cos(Y2) < vY:
                Y = Y2
                vY = _max_cos(Y)

        rem = HARD - (time.time() - t0)
        if rem < 3:
            break
        budget = min(5.0, rem - 2.0)
        if budget < 1.0:
            break
        Yp = _lp_polish(Y, budget)
        vp = _max_cos(Yp)
        if vp < best_val - 1e-12:
            best, best_val = Yp.copy(), vp
            cur, cur_val = Yp.copy(), vp
        elif vp < cur_val + 0.002 and vp < 0.32:
            # walk to new basin even if slightly worse
            cur, cur_val = Yp.copy(), vp
        else:
            # stay at cur but add tiny deterministic jitter to vary next SVD drop set
            pass
        # time guard
        if time.time() - t0 > HARD - 6:
            break

    # final safety: return best normalized
    best = best / np.linalg.norm(best, axis=1, keepdims=True)
    return best
