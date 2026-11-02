import time
import numpy as np
from scipy.optimize import linprog, minimize
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _lp_polish(x, deadline, r0=1e-3):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    try:
        best = _max_cos(x)
    except Exception:
        return x
    radius = r0
    iu0, iu1 = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        try:
            G = x @ x.T
            c = G[iu0, iu1]
            m = c.max()
            sel = c > m - max(20.0 * radius, 1e-9)
            if np.count_nonzero(sel) == 0:
                sel[np.argmax(c)] = True
            I = iu0[sel]; J = iu1[sel]
            csel = c[sel]
            p = len(I)
            nv = n * d + 1
            rows = np.repeat(np.arange(p), 2 * d + 1)
            cI = np.arange(d)[None, :].repeat(p, axis=0)
            cols_I = I[:, None] * d + cI
            cols_J = J[:, None] * d + cI
            cols_T = np.full((p, 1), n * d)
            cols = np.concatenate([cols_I, cols_J, cols_T], axis=1).ravel()
            vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], axis=1).ravel()
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv); obj[-1] = 1.0
            bounds = [(-radius, radius)] * (n * d) + [(None, None)]
            res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=bounds, method="highs")
            if res.status != 0:
                radius *= 0.5
                continue
            y = x + res.x[:-1].reshape(n, d)
            yn = np.linalg.norm(y, axis=1, keepdims=True)
            yn[yn < 1e-12] = 1.0
            y = y / yn
            v = _max_cos(y)
            if v < best:
                x, best = y, v
                radius = min(radius * 1.5, 0.05)
            else:
                radius *= 0.5
        except Exception:
            radius *= 0.5
            continue
    return x

def _smooth_lbfgs(x_init, deadline, taus=(0.008, 0.003, 0.001), maxiters=(120, 120, 150)):
    x = x_init / np.linalg.norm(x_init, axis=1, keepdims=True)
    n, d = x.shape
    iu0, iu1 = np.triu_indices(n, 1)
    Np = len(iu0)
    y = x.ravel().copy()
    for tau, maxiter in zip(taus, maxiters):
        if time.time() > deadline - 0.15:
            break
        def fg(yflat, _tau=tau):
            Y = yflat.reshape(n, d)
            rn = np.linalg.norm(Y, axis=1, keepdims=True)
            rn = np.maximum(rn, 1e-12)
            X = Y / rn
            G = X @ X.T
            g = G[iu0, iu1]
            m = g.max()
            e = np.exp((g - m) / _tau)
            s = e.sum()
            if s <= 0 or not np.isfinite(s):
                return float(m), np.zeros_like(yflat)
            L = m + _tau * np.log(s / Np)
            w = e / s
            W = np.zeros((n, n))
            W[iu0, iu1] = w
            W = W + W.T
            GX = W @ X
            dot = (GX * X).sum(axis=1, keepdims=True)
            GY = (GX - dot * X) / rn
            return float(L), GY.ravel()
        try:
            res = minimize(fg, y, method="L-BFGS-B", jac=True,
                           options={"maxiter": int(maxiter), "maxfun": int(maxiter*5),
                                    "ftol": 1e-14, "gtol": 1e-8})
            y = res.x
            Yt = y.reshape(n, d)
            nt = np.linalg.norm(Yt, axis=1, keepdims=True)
            nt[nt < 1e-12] = 1.0
            y = (Yt / nt).ravel()
        except Exception:
            break
    Xf = y.reshape(n, d)
    Xf = Xf / np.linalg.norm(Xf, axis=1, keepdims=True)
    return Xf

def _per_point_max(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return G.max(axis=1), G.max(), G

def _find_hole(X_fixed, rng, n_samples=25000):
    Y = rng.standard_normal((n_samples, 4))
    Y /= np.linalg.norm(Y, axis=1, keepdims=True)
    D = Y @ X_fixed.T
    vals = D.max(axis=1)
    idx = int(np.argmin(vals))
    return Y[idx], float(vals[idx])

def solve():
    t0 = time.time()
    DEADLINE = t0 + 174.0
    SEARCH_END = t0 + 148.0
    rng = np.random.default_rng(7)
    best_x = None
    best_v = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=np.float64)
            if a.shape != (163, 4):
                continue
            a = a / np.linalg.norm(a, axis=1, keepdims=True)
            if not np.all(np.isfinite(a)):
                continue
            v = _max_cos(a)
            if v < best_v:
                best_v = v
                best_x = a.copy()
        except Exception:
            continue
    if best_x is None:
        z = rng.standard_normal((163, 4))
        z /= np.linalg.norm(z, axis=1, keepdims=True)
        best_x = z
        best_v = _max_cos(z)
    n, d = 163, 4
    trial = 0
    # quick initial LP to ensure local optimum baseline (short)
    # skip long polish now; do at end
    while time.time() < SEARCH_END:
        trial += 1
        if time.time() > SEARCH_END - 2.0:
            break
        cur = best_x.copy()
        mode = trial % 7
        try:
            ppm, mval, G = _per_point_max(cur)
        except Exception:
            ppm = np.ones(n); G = cur @ cur.T; mval = best_v
        if mode == 0:
            sigma = 0.015 + 0.01 * ((trial // 7) % 3)
            cur = cur + rng.standard_normal(cur.shape) * sigma
            cur /= np.linalg.norm(cur, axis=1, keepdims=True)
        elif mode == 1:
            # shake random subset strongly
            k = 15 + (trial % 11)
            idx = rng.choice(n, size=k, replace=False)
            sigma = 0.12
            cur[idx] = cur[idx] + rng.standard_normal((k, d)) * sigma
            cur /= np.linalg.norm(cur, axis=1, keepdims=True)
        elif mode == 2:
            # worst-point ejection to holes
            k = 2 + (trial % 5)  # 2..6
            worst = np.argsort(ppm)[-k:]
            keep_mask = np.ones(n, dtype=bool)
            keep_mask[worst] = False
            Xkeep = cur[keep_mask]
            # sequentially find holes
            new_pts = []
            Xaug = Xkeep.copy()
            for _ in range(k):
                h, _ = _find_hole(Xaug, rng, n_samples=25000)
                new_pts.append(h)
                Xaug = np.vstack([Xaug, h[None, :]])
            cur[worst] = np.array(new_pts)
            # tiny jitter to kept
            cur = cur + rng.standard_normal(cur.shape) * 0.005
            cur /= np.linalg.norm(cur, axis=1, keepdims=True)
        elif mode == 3:
            # rattler / low-contact relocation + worst
            thr = mval - 1e-7
            # contacts count
            np.fill_diagonal(G, -2.0)
            cnt = (G > thr).sum(axis=1)
            ratt = np.where(cnt == 0)[0]
            # pick up to 3 rattlers/low + 2 worst
            cand = list(ratt[:4])
            worst2 = list(np.argsort(ppm)[-3:])
            for w in worst2:
                if w not in cand:
                    cand.append(w)
            if len(cand) == 0:
                cand = list(np.argsort(ppm)[-3:])
            cand = np.array(cand[:5])
            keep_mask = np.ones(n, dtype=bool)
            keep_mask[cand] = False
            Xkeep = cur[keep_mask]
            Xaug = Xkeep.copy()
            new_pts = []
            for _ in range(len(cand)):
                h, _ = _find_hole(Xaug, rng, n_samples=25000)
                new_pts.append(h)
                Xaug = np.vstack([Xaug, h[None, :]])
            cur[cand] = np.array(new_pts)
            cur /= np.linalg.norm(cur, axis=1, keepdims=True)
        elif mode == 4:
            # jammed high-degree shake
            thr2 = mval - 0.002
            G2 = cur @ cur.T
            np.fill_diagonal(G2, -2.0)
            deg = (G2 > thr2).sum(axis=1)
            idx = np.argsort(deg)[-12:]
            noise = rng.standard_normal(cur.shape) * 0.01
            noise[idx] += rng.standard_normal((len(idx), d)) * 0.10
            cur = cur + noise
            cur /= np.linalg.norm(cur, axis=1, keepdims=True)
        elif mode == 5:
            # single worst to hole + medium global shake
            worst = int(np.argmax(ppm))
            keep_mask = np.ones(n, dtype=bool)
            keep_mask[worst] = False
            h, _ = _find_hole(cur[keep_mask], rng, n_samples=30000)
            cur[worst] = h
            cur = cur + rng.standard_normal(cur.shape) * 0.02
            cur /= np.linalg.norm(cur, axis=1, keepdims=True)
        else:
            # pair-split: take top pairs and push apart + jitter
            G3 = cur @ cur.T
            np.fill_diagonal(G3, -2.0)
            # find top 6 pairs
            iu0, iu1 = np.triu_indices(n, 1)
            c = G3[iu0, iu1]
            top = np.argsort(c)[-6:]
            for t in top:
                i = int(iu0[t]); j = int(iu1[t])
                # push i and j apart along difference in tangent space
                diff = cur[i] - cur[j]
                diff = diff - (diff @ cur[i]) * cur[i]
                dn = np.linalg.norm(diff)
                if dn > 1e-12:
                    diff /= dn
                    cur[i] = cur[i] + diff * 0.03
                    cur[j] = cur[j] - diff * 0.03
            cur = cur + rng.standard_normal(cur.shape) * 0.008
            cur /= np.linalg.norm(cur, axis=1, keepdims=True)

        # smooth optimize with limited time per trial
        per_trial_deadline = min(SEARCH_END, time.time() + 2.6)
        try:
            sm = _smooth_lbfgs(cur, per_trial_deadline)
        except Exception:
            continue
        try:
            v = _max_cos(sm)
        except Exception:
            continue
        # accept if promising -> LP confirm
        if v < best_v + 0.003 and v < 0.90:
            lp_dead = min(SEARCH_END, time.time() + 1.5)
            # only full LP if close to best or better after smooth
            if v < best_v + 0.001:
                try:
                    pol = _lp_polish(sm, lp_dead)
                    vp = _max_cos(pol)
                except Exception:
                    pol, vp = sm, v
                if vp < best_v:
                    best_x, best_v = pol, vp
            else:
                if v < best_v:
                    best_x, best_v = sm, v
        # also direct update if smooth alone beats best (rare without LP)
        if v < best_v:
            # quick LP already attempted above if threshold; ensure update
            if v < best_v:
                best_x, best_v = sm if 'pol' not in locals() else (best_x if best_v <= v else sm), min(best_v, v)
    # final polish: annealed smooth + long LP
    try:
        # final smooth with finer taus
        rem = DEADLINE - time.time()
        if rem > 5:
            sm_dead = time.time() + min(12.0, rem - 8.0)
            best_x = _smooth_lbfgs(best_x, sm_dead, taus=(0.005, 0.002, 0.0008, 0.0004), maxiters=(150, 150, 150, 120))
    except Exception:
        pass
    try:
        best_x = _lp_polish(best_x, DEADLINE - 0.5, r0=2e-3)
    except Exception:
        pass
    best_x = best_x / np.linalg.norm(best_x, axis=1, keepdims=True)
    return best_x
