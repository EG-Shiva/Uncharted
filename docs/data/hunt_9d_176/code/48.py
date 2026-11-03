import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def smooth_polish(X, steps=500, a0=40.0, a1=350.0, lr=0.003):
    x = X.copy()
    n, d = x.shape
    m = np.zeros_like(x)
    vv = np.zeros_like(x)
    b1, b2, eps = 0.9, 0.999, 1e-8
    for t in range(steps):
        frac = t / max(steps - 1, 1)
        alpha = a0 * ((a1 / a0) ** frac)
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        M = float(np.max(G))
        E = np.exp(alpha * (G - M))
        np.fill_diagonal(E, 0.0)
        S = float(E.sum() * 0.5)
        if S <= 0 or not np.isfinite(S):
            break
        Grad = (E @ x) / S
        dot = np.sum(Grad * x, axis=1, keepdims=True)
        Gt = Grad - dot * x
        m = b1 * m + (1 - b1) * Gt
        vv = b2 * vv + (1 - b2) * (Gt * Gt)
        mhat = m / (1 - b1 ** (t + 1))
        vhat = vv / (1 - b2 ** (t + 1))
        step = mhat / (np.sqrt(vhat) + eps)
        np.clip(step, -3.0, 3.0, out=step)
        lrt = lr * (1.0 - 0.5 * frac)
        x = x - lrt * step
        nr = np.linalg.norm(x, axis=1, keepdims=True)
        nr[nr == 0] = 1.0
        x /= nr
        if not np.all(np.isfinite(x)):
            break
    nr = np.linalg.norm(x, axis=1, keepdims=True)
    nr[nr == 0] = 1.0
    return x / nr

def lp_polish(X, seconds, radius_init=3e-3, cap=2800):
    x = X / np.linalg.norm(X, axis=1, keepdims=True)
    n, d = x.shape
    best = max_cos(x)
    radius = radius_init
    start = time.time()
    iu0 = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        Gfull = x @ x.T
        c_all = Gfull[iu0]
        mval = float(c_all.max())
        thresh = mval - max(20 * radius, 1e-9)
        sel = c_all > thresh
        cnt = int(np.count_nonzero(sel))
        if cnt == 0:
            break
        if cnt > cap:
            idx_all = np.where(sel)[0]
            order = np.argpartition(-c_all[idx_all], cap)[:cap]
            keep = np.zeros_like(sel)
            keep[idx_all[order]] = True
            sel = keep
            cnt = cap
        I = iu0[0][sel]
        J = iu0[1][sel]
        c = c_all[sel]
        p = cnt
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))),
                              shape=(n, nv)).tocsr()
        except Exception:
            radius *= 0.5
            continue
        obj = np.zeros(nv)
        obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                          method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        nr = np.linalg.norm(y, axis=1, keepdims=True)
        nr[nr == 0] = 1.0
        y /= nr
        v = max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
        if time.time() - start >= seconds:
            break
    return x

def solve():
    t0 = time.time()
    BUDGET = 168.0
    rng = np.random.default_rng(0)
    try:
        xb = np.load("best.npy")
    except Exception:
        try:
            xb = np.load("record.npy")
        except Exception:
            xb = rng.standard_normal((176, 9))
    xb = np.asarray(xb, dtype=float)
    if xb.shape != (176, 9):
        xb = rng.standard_normal((176, 9))
    xb = xb / np.linalg.norm(xb, axis=1, keepdims=True)
    best_val = max_cos(xb)
    best_x = xb.copy()
    try:
        xr = np.load("record.npy")
        xr = np.asarray(xr, dtype=float)
        if xr.shape == (176, 9):
            xr = xr / np.linalg.norm(xr, axis=1, keepdims=True)
            vr = max_cos(xr)
            if vr < best_val:
                best_val, best_x = vr, xr.copy()
    except Exception:
        pass

    n, d = 176, 9
    iu_all = np.triu_indices(n, 1)
    amps = [0.01, 0.03, 0.06, 0.1]
    outer = 0

    while time.time() - t0 < BUDGET - 14.0:
        outer += 1
        x0 = best_x.copy()
        # --- active contact graph slack<1e-3 ---
        G0 = x0 @ x0.T
        np.fill_diagonal(G0, -2.0)
        m0 = float(np.max(G0))
        c_all = G0[iu_all]
        slack = m0 - c_all
        sel_active = slack < 1e-3
        if int(np.count_nonzero(sel_active)) > 3000:
            idx = np.where(sel_active)[0]
            order = np.argsort(c_all[idx])[::-1][:3000]
            keep = np.zeros_like(sel_active)
            keep[idx[order]] = True
            sel_active = keep
        I = iu_all[0][sel_active]; J = iu_all[1][sel_active]
        cc = c_all[sel_active]
        p = len(I)
        # positive equilibrium stress approx via softmax
        if p > 0:
            w = np.exp(300.0 * (cc - m0))
            w = w / (w.sum() + 1e-30)
            # least-squares refinement: try to balance (one step)
            # point stress
            pt_stress = np.zeros(n)
            np.add.at(pt_stress, I, w)
            np.add.at(pt_stress, J, w)
        else:
            pt_stress = np.ones(n) / n
        deg = np.zeros(n, dtype=int)
        if p > 0:
            np.add.at(deg, I, 1)
            np.add.at(deg, J, 1)
        hub_order = np.argsort(-(pt_stress + 0.1 * deg / max(1, deg.max())))
        hubs = hub_order[:5]

        # --- rigidity Jacobian A (p x n*d) ---
        modes = None
        try:
            from scipy.sparse.linalg import svds
            if p >= 30:
                rows = np.repeat(np.arange(p), 2 * d)
                colI = (I[:, None] * d + np.arange(d)).ravel()
                colJ = (J[:, None] * d + np.arange(d)).ravel()
                cols = np.concatenate([colI, colJ])
                vI = (x0[J] - cc[:, None] * x0[I]).ravel()
                vJ = (x0[I] - cc[:, None] * x0[J]).ravel()
                vals = np.concatenate([vI, vJ])
                A = coo_matrix((vals, (rows, cols)), shape=(p, n * d)).tocsr()
                k = 24
                k = min(k, min(p, n * d) - 2)
                if k >= 4:
                    u, s, vt = svds(A, k=k, which='SM', tol=1e-3, maxiter=250)
                    # vt: k x nd, smallest first? sort by s
                    order_s = np.argsort(s)
                    vt = vt[order_s, :]
                    modes = vt.T  # nd x k
        except Exception:
            modes = None
        if modes is None:
            # fallback random tangent modes
            k = 20
            R = rng.standard_normal((n * d, k))
            modes = R

        nd, k = modes.shape
        # reshape modes to (k, n, d) and project to tangent
        M3 = modes.T.reshape(k, n, d)
        # project each mode to tangent
        dotm = np.sum(M3 * x0[None, :, :], axis=2, keepdims=True)
        M3 = M3 - dotm * x0[None, :, :]
        # normalize each mode
        for mi in range(k):
            nn = np.linalg.norm(M3[mi])
            if nn > 1e-12:
                M3[mi] /= nn

        # --- spectral partitioning of contact graph ---
        try:
            from scipy.sparse.linalg import eigsh
            # adjacency
            Adj = coo_matrix((np.ones(p), (np.concatenate([I, J]), np.concatenate([J, I]))), shape=(n, n)).toarray()
            Deg = Adj.sum(axis=1)
            Lap = np.diag(Deg) - Adj
            # two smallest eigenvectors
            wv, vv2 = eigsh(Lap, k=2, which='SM', tol=1e-3, maxiter=500)
            fied = vv2[:, np.argsort(wv)[1]]
            sideA = fied > 0
            sideB = ~sideA
            if sideA.sum() == 0 or sideA.sum() == n:
                sideA = np.arange(n) % 2 == 0
                sideB = ~sideA
        except Exception:
            sideA = np.arange(n) % 2 == 0
            sideB = ~sideA
        hub = int(hubs[0])
        hub_side = sideA if sideA[hub] else sideB
        # hub neighbourhood (1-hop)
        neigh = np.zeros(n, dtype=bool)
        neigh[hub] = True
        if p > 0:
            neigh[I[J == hub]] = True if False else neigh[I[J == hub]]
            # vectorized:
            for hh in hubs[:3]:
                neigh[hub] = True
                mask_h = (I == hh) | (J == hh)
                # neighbors
                nb = np.unique(np.concatenate([J[I == hh], I[J == hh]])) if np.any(mask_h) else np.array([], dtype=int)
                neigh[nb] = True
        masks = []
        masks.append(hub_side.copy())
        masks.append(~hub_side)
        masks.append(neigh.copy())
        masks.append(np.ones(n, dtype=bool))

        # --- propose collective displacements ---
        n_cand = 44
        cands = []
        cands_scores = []
        for ci in range(n_cand):
            nmix = int(rng.integers(1, 4))  # 1..3
            midx = rng.choice(k, size=nmix, replace=False)
            coeff = rng.standard_normal(nmix)
            D = np.tensordot(coeff, M3[midx], axes=(0, 0))  # n x d
            # re-project (mixture stays tangent)
            mi_choice = int(rng.integers(0, len(masks)))
            mask = masks[mi_choice]
            # occasionally random subset mask for diversity
            if rng.random() < 0.15:
                mask = rng.random(n) < 0.5
                if mask.sum() == 0:
                    mask[0] = True
            amp = amps[int(rng.integers(0, len(amps)))]
            if rng.random() < 0.5:
                amp = -amp
            disp = np.zeros_like(D)
            disp[mask] = D[mask]
            # scale to rms amplitude over moved points
            cntm = int(mask.sum())
            if cntm == 0:
                continue
            rms = np.sqrt(np.sum(disp[mask] ** 2) / cntm)
            if rms < 1e-12:
                continue
            disp *= (abs(amp) / rms)
            if amp < 0:
                disp *= -1.0
            y = x0 + disp
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            v = max_cos(y)
            cands.append(y)
            cands_scores.append(v)
            if time.time() - t0 > BUDGET - 14.0:
                break
        if len(cands) == 0:
            break
        cands_scores = np.array(cands_scores)
        order_c = np.argsort(cands_scores)
        # screen top survivors with 500 Adam steps
        n_screen = min(6, len(cands))
        screened = []
        screened_vals = []
        for si in range(n_screen):
            y = cands[order_c[si]]
            # quick exact pre-check: only screen if promising
            # allow slightly worse to escape basin
            if cands_scores[order_c[si]] > best_val + 0.01 and si >= 3:
                continue
            ys = smooth_polish(y, steps=500, a0=40.0, a1=350.0, lr=0.003)
            vs = max_cos(ys)
            screened.append(ys)
            screened_vals.append(vs)
            if time.time() - t0 > BUDGET - 12.0:
                break
        if len(screened_vals) == 0:
            continue
        screened_vals = np.array(screened_vals)
        bi = int(np.argmin(screened_vals))
        best_screened = screened[bi]
        best_screened_val = float(screened_vals[bi])
        # LP-polish only single best screened candidate under hard deadline
        rem = BUDGET - (time.time() - t0)
        # reserve ~7s for final polish
        lp_sec = min(5.0, max(1.0, rem - 8.0))
        if rem > 9.0:
            best_screened = lp_polish(best_screened, seconds=lp_sec, radius_init=3e-3)
            best_screened_val = max_cos(best_screened)
        if best_screened_val < best_val - 1e-12:
            best_val = best_screened_val
            best_x = best_screened.copy()
        else:
            # no improvement: try small elitist jitter to avoid stall (cheap, no LP)
            if outer % 3 == 0 and time.time() - t0 < BUDGET - 12.0:
                tmp = best_x + 0.005 * rng.standard_normal(best_x.shape)
                tmp /= np.linalg.norm(tmp, axis=1, keepdims=True)
                tmp = smooth_polish(tmp, steps=400, a0=80.0, a1=500.0, lr=0.0015)
                vt = max_cos(tmp)
                if vt < best_val - 1e-12:
                    best_val, best_x = vt, tmp
        # loop continues with updated best (new modes next outer)

    # final polish of incumbent with remaining time
    rem = BUDGET - (time.time() - t0)
    rem = min(max(rem - 0.5, 0.0), 8.0)
    if rem > 1.0:
        best_x = lp_polish(best_x, seconds=rem, radius_init=1e-3)
    best_x = best_x / np.linalg.norm(best_x, axis=1, keepdims=True)
    return best_x
