import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _norm(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return X / n

def _maxcos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def _tangent_bases(X):
    n, d = X.shape
    T = np.empty((n, d, d - 1))
    for i in range(n):
        x = X[i]
        m = int(np.argmax(np.abs(x)))
        C = np.zeros((d, d - 1))
        col = 0
        xv = float(x[m])
        if abs(xv) < 1e-12:
            xv = 1e-12
        for k in range(d):
            if k == m:
                continue
            v = np.zeros(d)
            v[k] = 1.0
            v[m] = -float(x[k]) / xv
            C[:, col] = v
            col += 1
        Q, _ = np.linalg.qr(C)
        T[i] = Q
    return T

def _lp_polish(X, deadline):
    X = _norm(X.copy())
    n, d = X.shape
    best = _maxcos(X)
    bx = X.copy()
    radius = 2.0e-3
    iu = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        G = X @ X.T
        c = G[iu]
        m = float(np.max(c))
        thr = m - max(20.0 * radius, 1e-9)
        idx = np.where(c > thr)[0]
        if idx.size == 0:
            break
        if idx.size > 1400:
            o = np.argsort(c[idx])[-1400:]
            idx = idx[o]
        sel = np.zeros_like(c, dtype=bool)
        sel[idx] = True
        I = iu[0][sel]; J = iu[1][sel]
        cs = c[sel]
        p = len(I)
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], axis=1).ravel()
        vals = np.concatenate([X[J], X[I], -np.ones((p, 1))], axis=1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((X.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        except Exception:
            break
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-cs, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        try:
            y = X + res.x[:-1].reshape(n, d)
        except Exception:
            radius *= 0.5
            continue
        y = _norm(y)
        v = _maxcos(y)
        if v < best:
            best = v; bx = y.copy(); X = y
            radius = min(radius * 1.5, 0.05)
        else:
            if v <= best + 1e-15:
                X = y
            radius *= 0.5
        if time.time() >= deadline:
            break
    return bx, best

def solve():
    t0 = time.time()
    HARD_END = t0 + 171.0
    rng = np.random.default_rng(1)
    n, d = 116, 6
    def clean(a):
        try:
            a = np.asarray(a, dtype=float)
        except Exception:
            return None
        if a.shape != (116, 6) or not np.all(np.isfinite(a)):
            return None
        nn = np.linalg.norm(a, axis=1, keepdims=True)
        if np.any(nn < 1e-12):
            return None
        return a / nn
    cands = []
    for f in ("best.npy", "record.npy"):
        try:
            c = clean(np.load(f))
            if c is not None:
                cands.append(c)
        except Exception:
            pass
    if not cands:
        cands.append(_norm(rng.standard_normal((116, 6))))
    vals = [_maxcos(c) for c in cands]
    bi = int(np.argmin(vals))
    best_x = cands[bi].copy()
    best_v = float(vals[bi])
    cur_x = best_x.copy()
    cur_v = best_v
    STEPS = [1e-2, 3e-3, 1e-3, 3e-4]
    # quick initial squeeze if time allows
    if HARD_END - time.time() > 20:
        try:
            bx, bv = _lp_polish(best_x, min(time.time() + 2.0, HARD_END - 15.0))
            if bv < best_v:
                best_x, best_v = bx, bv
                cur_x, cur_v = bx.copy(), bv
        except Exception:
            pass
    it = 0
    stall = 0
    while time.time() < HARD_END - 9.0:
        it += 1
        if time.time() > HARD_END - 12.0:
            break
        # rebuild tangent bases and rigidity matrix
        try:
            T = _tangent_bases(cur_x)
        except Exception:
            break
        try:
            G = cur_x @ cur_x.T
            np.fill_diagonal(G, -2.0)
            m0 = float(np.max(G))
            iu0 = np.triu_indices(n, 1)
            c0 = G[iu0]
            slack = m0 - c0
            act = np.where(slack < 3e-7)[0]
            if act.size < 10:
                act = np.argsort(slack)[:400]
            if act.size > 900:
                # keep tightest 900 to keep SVD cheap
                o = np.argsort(slack[act])[:900]
                act = act[o]
            I = iu0[0][act]; J = iu0[1][act]
            p = len(I)
            R = np.zeros((p, n * 5))
            for t in range(p):
                ii = int(I[t]); jj = int(J[t])
                try:
                    a = T[ii].T @ cur_x[jj]
                    b = T[jj].T @ cur_x[ii]
                except Exception:
                    a = np.zeros(5); b = np.zeros(5)
                R[t, ii * 5:(ii + 1) * 5] = a
                R[t, jj * 5:(jj + 1) * 5] = b
        except Exception:
            break
        # SVD nullspace
        try:
            Uc, sv, Vt = np.linalg.svd(R, full_matrices=True)
        except Exception:
            # fallback small noise
            cur_x = _norm(cur_x + rng.normal(scale=1e-4, size=(n, d)))
            cur_v = _maxcos(cur_x)
            continue
        nv_all = n * 5
        nsv = sv.shape[0]
        # pool of null / near-null directions (rows of Vt)
        if p < nv_all:
            Dexact = nv_all - p
            Dex = min(Dexact, 50)
            pool_rows = []
            # exact null: last Dex rows
            for k in range(Dex):
                pool_rows.append(Vt[nv_all - 1 - k, :])
            # add a few smallest non-zero
            Knear = min(12, nsv)
            for k in range(Knear):
                pool_rows.append(Vt[nsv - 1 - k, :])
            pool = np.stack(pool_rows, axis=0) if len(pool_rows) > 0 else Vt[-40:, :]
        else:
            K = min(60, nv_all)
            pool = Vt[nv_all - K:, :]
        Kpool = pool.shape[0]
        # build candidate direction list in 580-dim
        cand_dirs = []
        for k in range(Kpool):
            cand_dirs.append(pool[k, :].copy())
        # random combos in pool subspace
        Nrand = 70
        try:
            coeff = rng.standard_normal((Nrand, Kpool))
            for r in range(Nrand):
                v = coeff[r, :] @ pool
                nn = float(np.linalg.norm(v))
                if nn < 1e-15:
                    continue
                cand_dirs.append(v / nn)
        except Exception:
            pass
        # convert each dir to physical U and evaluate steps
        best_trial_v = 1e9
        best_trial_x = None
        # precompute? loop
        for v in cand_dirs:
            if time.time() > HARD_END - 9.0:
                break
            # build U
            try:
                U = np.empty((n, d))
                mx = 0.0
                for i in range(n):
                    a = v[i * 5:(i + 1) * 5]
                    u = T[i] @ a
                    U[i, :] = u
                    q = float(np.dot(u, u))
                    if q > mx:
                        mx = q
                mx = float(np.sqrt(mx))
                if not np.isfinite(mx) or mx < 1e-15:
                    continue
                U = U / mx
            except Exception:
                continue
            for st in STEPS:
                try:
                    Xt = cur_x + st * U
                    Xt = _norm(Xt)
                    vv = _maxcos(Xt)
                except Exception:
                    continue
                if vv < best_trial_v:
                    best_trial_v = vv
                    best_trial_x = Xt.copy()
                    if vv < best_v - 1e-12:
                        pass
            # early exit if big win
        if best_trial_x is None:
            stall += 1
            cur_x = _norm(cur_x + rng.normal(scale=2e-4, size=(n, d)))
            cur_v = _maxcos(cur_x)
            continue
        if best_trial_v < cur_v - 5e-13:
            cur_x = best_trial_x.copy()
            cur_v = float(best_trial_v)
            stall = 0
            if cur_v < best_v - 5e-13:
                best_v = cur_v
                best_x = cur_x.copy()
                # short LP polish on improvement
                dl = min(time.time() + 1.6, HARD_END - 8.0)
                if dl > time.time() + 0.25:
                    try:
                        Xp, vp = _lp_polish(cur_x, dl)
                        if vp < best_v:
                            best_v = vp; best_x = Xp.copy()
                        # stay at polished point even if equal
                        cur_x, cur_v = Xp.copy(), float(vp) if vp < cur_v + 1e-9 else (cur_x.copy(), cur_v)
                        if vp <= cur_v:
                            cur_x, cur_v = Xp.copy(), float(vp)
                    except Exception:
                        pass
            else:
                # polish occasionally even without global best
                if it % 4 == 0:
                    dl = min(time.time() + 1.0, HARD_END - 8.0)
                    if dl > time.time() + 0.25:
                        try:
                            Xp, vp = _lp_polish(cur_x, dl)
                            if vp < cur_v:
                                cur_x, cur_v = Xp.copy(), float(vp)
                                if vp < best_v:
                                    best_v = vp; best_x = Xp.copy()
                        except Exception:
                            pass
        else:
            # no lowering direction: drift sideways if close, else tiny random in nullspace
            stall += 1
            if best_trial_v < cur_v + 5e-7 and best_trial_x is not None:
                cur_x = best_trial_x.copy()
                cur_v = float(best_trial_v)
                # occasional LP to re-jam sideways point
                if stall % 3 == 0:
                    dl = min(time.time() + 1.0, HARD_END - 8.0)
                    if dl > time.time() + 0.25:
                        try:
                            Xp, vp = _lp_polish(cur_x, dl)
                            if vp < best_v:
                                best_v = vp; best_x = Xp.copy()
                                cur_x, cur_v = Xp.copy(), float(vp)
                                stall = 0
                            elif vp < cur_v:
                                cur_x, cur_v = Xp.copy(), float(vp)
                        except Exception:
                            pass
            else:
                # kick in nullspace + tiny full noise
                try:
                    k0 = int(rng.integers(0, Kpool))
                    v0 = pool[k0, :]
                    U0 = np.empty((n, d))
                    mx = 0.0
                    for i in range(n):
                        a = v0[i * 5:(i + 1) * 5]
                        u = T[i] @ a
                        U0[i, :] = u
                        q = float(np.dot(u, u))
                        if q > mx:
                            mx = q
                    mx = float(np.sqrt(mx)) + 1e-15
                    U0 = U0 / mx
                    cur_x = _norm(cur_x + 3e-3 * U0 + rng.normal(scale=3e-4, size=(n, d)))
                    cur_v = _maxcos(cur_x)
                except Exception:
                    cur_x = _norm(cur_x + rng.normal(scale=5e-4, size=(n, d)))
                    cur_v = _maxcos(cur_x)
            if stall >= 6:
                # reset to best to avoid wandering too far
                cur_x = best_x.copy(); cur_v = best_v
                stall = 0
                # small null perturbation of best to get new basin view
                try:
                    T2 = _tangent_bases(cur_x)
                    # quick random tangent step
                    cur_x = _norm(cur_x + rng.normal(scale=8e-4, size=(n, d)))
                    cur_v = _maxcos(cur_x)
                except Exception:
                    pass
    rem = HARD_END - time.time()
    if rem > 4.0:
        try:
            fb = min(rem - 1.0, 20.0)
            Xf, vf = _lp_polish(best_x, time.time() + fb)
            if vf <= best_v:
                best_x, best_v = Xf, vf
        except Exception:
            pass
    best_x = _norm(best_x)
    return best_x
