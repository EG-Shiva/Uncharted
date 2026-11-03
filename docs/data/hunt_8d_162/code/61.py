import time
import numpy as np
from scipy.optimize import minimize, linprog
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _lp_polish(X, seconds, r0=1e-3, rmax=0.05, cap=3000):
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    n, d = X.shape
    best = _max_cos(X)
    radius = r0
    t0 = time.time()
    iuI, iuJ = np.triu_indices(n, 1)
    while time.time() - t0 < seconds and radius > 1e-14:
        G = X @ X.T
        cf = G[iuI, iuJ]
        m = float(cf.max())
        if m < best:
            best = m
        sel = cf > m - max(20.0 * radius, 1e-9)
        cnt = int(sel.sum())
        if cnt == 0:
            radius *= 0.5
            continue
        if cnt > cap:
            idx = np.argpartition(-cf, cap)[:cap]
            mask = np.zeros_like(sel)
            mask[idx] = True
            sel = mask
            cnt = cap
        I = iuI[sel]; J = iuJ[sel]; c = cf[sel]
        p = cnt
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([(I[:, None] * d + np.arange(d)).reshape(p, d),
                               (J[:, None] * d + np.arange(d)).reshape(p, d),
                               np.full((p, 1), n * d)], axis=1).ravel()
        vals = np.concatenate([X[J], X[I], -np.ones((p, 1))], axis=1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((X.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv); obj[-1] = 1.0
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
        Y = X + res.x[:-1].reshape(n, d)
        Y /= np.linalg.norm(Y, axis=1, keepdims=True)
        v = _max_cos(Y)
        if v < best - 1e-15:
            X = Y; best = v
            radius = min(radius * 1.5, rmax)
        else:
            radius *= 0.5
        if time.time() - t0 >= seconds:
            break
    return X

def _free_obj_factory(n, d, alpha, shift=0.5):
    def fun(Yf):
        Y = Yf.reshape(n, d)
        nr = np.maximum(np.linalg.norm(Y, axis=1, keepdims=True), 1e-12)
        X = Y / nr
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        M = alpha * (G - shift)
        np.clip(M, -700, 200, out=M)
        Emat = np.exp(M)
        E = float(Emat.sum() * 0.5)
        GradX = alpha * (Emat @ X)
        dot = np.sum(GradX * X, axis=1, keepdims=True)
        GradY = (GradX - dot * X) / nr
        return E, GradY.ravel()
    return fun

def _smooth_free(Xinit, alphas, iters, t_start, budget_left, shift=0.5):
    n, d = Xinit.shape
    Yf = Xinit.ravel().copy()
    for alpha, mi in zip(alphas, iters):
        if time.time() - t_start > budget_left:
            break
        fun = _free_obj_factory(n, d, float(alpha), shift)
        try:
            res = minimize(fun, Yf, jac=True, method="L-BFGS-B",
                           options={"maxiter": int(mi), "ftol": 1e-12, "gtol": 1e-5, "maxfun": int(mi) * 20})
            Yf = res.x
        except Exception:
            break
    Y = Yf.reshape(n, d)
    Y /= np.linalg.norm(Y, axis=1, keepdims=True)
    return Y

def _build_e8():
    roots = []
    sq2 = np.sqrt(2.0)
    for i in range(8):
        for j in range(i + 1, 8):
            for si in (-1.0, 1.0):
                for sj in (-1.0, 1.0):
                    v = np.zeros(8)
                    v[i] = si; v[j] = sj
                    roots.append(v / sq2)
    for mask in range(256):
        nminus = 0
        v = np.empty(8)
        for k in range(8):
            bit = (mask >> k) & 1
            s = 1.0 if bit else -1.0
            if s < 0:
                nminus += 1
            v[k] = s * 0.5
        if nminus % 2 == 0:
            roots.append(v / sq2)
    return np.array(roots)

def _delete_to_162(E8, rng_or_none):
    idx = list(range(E8.shape[0]))
    Gfull = E8 @ E8.T
    while len(idx) > 162:
        sub = E8[idx]
        G = sub @ sub.T
        np.fill_diagonal(G, -2.0)
        cnt = (G > 0.499).sum(axis=1)
        mx = cnt.max()
        cands = np.where(cnt == mx)[0]
        if len(cands) > 1 and rng_or_none is not None:
            pick = rng_or_none.integers(0, len(cands))
            victim = cands[pick]
        else:
            # deterministic: also break ties by sum
            if len(cands) > 1:
                s = G[cands].sum(axis=1)
                victim = cands[int(np.argmax(s))]
            else:
                victim = cands[0]
        del idx[int(victim)]
    return E8[idx].copy()

def _cyclic_expand(Y20, Ye):
    parts = []
    for k in range(Y20.shape[0]):
        r = Y20[k]
        for s in range(8):
            parts.append(np.roll(r, s))
    Z = np.vstack(parts + [Ye])
    return Z

def _smooth_cyclic(Y20init, Yeinit, alphas, iters, t_start, budget_left, shift=0.5):
    nrep, d = Y20init.shape
    ne = Yeinit.shape[0]
    nfull = nrep * 8 + ne
    y0 = np.concatenate([Y20init.ravel(), Yeinit.ravel()]).copy()
    for alpha, mi in zip(alphas, iters):
        if time.time() - t_start > budget_left:
            break
        al = float(alpha)
        def fun(yf, al=al):
            Y20 = yf[:nrep * d].reshape(nrep, d)
            Ye = yf[nrep * d:].reshape(ne, d)
            Z = _cyclic_expand(Y20, Ye)
            nr = np.maximum(np.linalg.norm(Z, axis=1, keepdims=True), 1e-12)
            X = Z / nr
            G = X @ X.T
            np.fill_diagonal(G, -2.0)
            M = al * (G - shift)
            np.clip(M, -700, 200, out=M)
            Emat = np.exp(M)
            E = float(Emat.sum() * 0.5)
            Gx = al * (Emat @ X)
            dot = np.sum(Gx * X, axis=1, keepdims=True)
            GradZ = (Gx - dot * X) / nr
            GradY20 = np.zeros_like(Y20)
            pos = 0
            for k in range(nrep):
                acc = np.zeros(d)
                for s in range(8):
                    gz = GradZ[pos]; pos += 1
                    acc += np.roll(gz, -s)
                GradY20[k] = acc
            GradYe = GradZ[pos:pos + ne]
            grad = np.concatenate([GradY20.ravel(), GradYe.ravel()])
            return E, grad
        try:
            res = minimize(fun, y0, jac=True, method="L-BFGS-B",
                           options={"maxiter": int(mi), "ftol": 1e-12, "gtol": 1e-5, "maxfun": int(mi) * 20})
            y0 = res.x
        except Exception:
            break
    Y20 = y0[:nrep * d].reshape(nrep, d)
    Ye = y0[nrep * d:].reshape(ne, d)
    Z = _cyclic_expand(Y20, Ye)
    Z /= np.linalg.norm(Z, axis=1, keepdims=True)
    return Z

def _smooth_antipodal(Yinit, alphas, iters, t_start, budget_left, shift=0.35):
    nrep, d = Yinit.shape
    nfull = nrep * 2
    y0 = Yinit.ravel().copy()
    for alpha, mi in zip(alphas, iters):
        if time.time() - t_start > budget_left:
            break
        al = float(alpha)
        def fun(yf, al=al):
            Y = yf.reshape(nrep, d)
            Z = np.vstack([Y, -Y])
            nr = np.maximum(np.linalg.norm(Z, axis=1, keepdims=True), 1e-12)
            X = Z / nr
            G = X @ X.T
            np.fill_diagonal(G, -2.0)
            M = al * (G - shift)
            np.clip(M, -700, 200, out=M)
            Emat = np.exp(M)
            E = float(Emat.sum() * 0.5)
            Gx = al * (Emat @ X)
            dot = np.sum(Gx * X, axis=1, keepdims=True)
            GradZ = (Gx - dot * X) / nr
            GradY = GradZ[:nrep] - GradZ[nrep:]
            return E, GradY.ravel()
        try:
            res = minimize(fun, y0, jac=True, method="L-BFGS-B",
                           options={"maxiter": int(mi), "ftol": 1e-12, "gtol": 1e-5, "maxfun": int(mi) * 20})
            y0 = res.x
        except Exception:
            break
    Y = y0.reshape(nrep, d)
    Z = np.vstack([Y, -Y])
    # antipodal gives 2*nrep points; need exactly 162 -> nrep must be 81
    Z /= np.linalg.norm(Z, axis=1, keepdims=True)
    # dedupe antipodal duplicates of norm issues
    if Z.shape[0] != 162:
        # adjust by truncation/padding (should not happen)
        if Z.shape[0] > 162:
            Z = Z[:162]
    return Z

def solve():
    t_start = time.time()
    BUDGET = 175.0
    n, d = 162, 8
    rng = np.random.default_rng(20261103)
    # load incumbent
    X_inc = None
    try:
        xb = np.load("best.npy").astype(float)
        if xb.shape == (162, 8):
            xb = xb / np.linalg.norm(xb, axis=1, keepdims=True)
            if np.all(np.isfinite(xb)):
                X_inc = xb
    except Exception:
        pass
    if X_inc is None:
        try:
            xr = np.load("record.npy").astype(float)
            if xr.shape == (162, 8):
                X_inc = xr / np.linalg.norm(xr, axis=1, keepdims=True)
        except Exception:
            pass
    if X_inc is None:
        a = rng.standard_normal(size=(162, 8))
        a /= np.linalg.norm(a, axis=1, keepdims=True)
        X_inc = a
    best_val = _max_cos(X_inc)
    X_best = X_inc.copy()
    X_struct_best = None
    struct_val = 1e9
    # ---------- Phase 1: E8 greedy delete + relax ----------
    try:
        E8 = _build_e8()
        # deterministic deletion
        S0 = _delete_to_162(E8, None)
        rng1 = np.random.default_rng(11)
        S1 = _delete_to_162(E8, rng1)
        for k, S in enumerate([S0, S1]):
            if time.time() - t_start > 60:
                break
            try:
                sig = 0.006 if k == 0 else 0.012
                Sp = S + sig * rng.standard_normal(size=S.shape)
                Sp /= np.linalg.norm(Sp, axis=1, keepdims=True)
                So = _smooth_free(Sp, [50, 120], [45, 45], t_start, 70.0)
                v = _max_cos(So)
                if v < struct_val:
                    struct_val = v
                    X_struct_best = So.copy()
                # symmetry-preserving-ish quick LP on subset? do short LP
                if time.time() - t_start < 90:
                    Slp = _lp_polish(So, 2.0, r0=0.002)
                    vlp = _max_cos(Slp)
                    if vlp < struct_val:
                        struct_val = vlp
                        X_struct_best = Slp.copy()
            except Exception:
                continue
    except Exception:
        pass
    # ---------- Phase 2a: cyclic orbits 20x8+2 ----------
    try:
        if time.time() - t_start < 110:
            Y20 = rng.standard_normal(size=(20, 8))
            Y20 /= np.linalg.norm(Y20, axis=1, keepdims=True)
            # seed a few reps from incumbent for diversity
            try:
                pick = rng.choice(162, 6, replace=False)
                Y20[:6] = X_inc[pick] + 0.02 * rng.standard_normal(size=(6, 8))
                Y20 /= np.linalg.norm(Y20, axis=1, keepdims=True)
            except Exception:
                pass
            Ye = rng.standard_normal(size=(2, 8))
            Ye /= np.linalg.norm(Ye, axis=1, keepdims=True)
            Zc = _smooth_cyclic(Y20, Ye, [40, 110], [50, 50], t_start, 125.0)
            vc = _max_cos(Zc)
            if vc < struct_val:
                struct_val = vc
                X_struct_best = Zc.copy()
    except Exception:
        pass
    # ---------- Phase 2b: antipodal 81x2 ----------
    try:
        if time.time() - t_start < 135:
            Ya = rng.standard_normal(size=(81, 8))
            Ya /= np.linalg.norm(Ya, axis=1, keepdims=True)
            Za = _smooth_antipodal(Ya, [40, 100], [45, 45], t_start, 145.0)
            if Za.shape == (162, 8):
                va = _max_cos(Za)
                if va < struct_val:
                    struct_val = va
                    X_struct_best = Za.copy()
    except Exception:
        pass
    # ---------- Phase 3: gradual symmetry release ----------
    try:
        if X_struct_best is not None and time.time() - t_start < 150:
            Xr = X_struct_best.copy()
            for sig in [0.004, 0.010]:
                if time.time() - t_start > 152:
                    break
                try:
                    Xp = Xr + sig * rng.standard_normal(size=Xr.shape)
                    Xp /= np.linalg.norm(Xp, axis=1, keepdims=True)
                    Xo = _smooth_free(Xp, [90, 150], [40, 40], t_start, 158.0)
                    vo = _max_cos(Xo)
                    if vo < struct_val:
                        struct_val = vo
                        Xr = Xo.copy()
                    else:
                        # keep released version anyway for final polish if close
                        Xr = Xo.copy()
                        struct_val = vo
                        # if much worse, revert to best release? keep going
                except Exception:
                    continue
            X_struct_best = Xr.copy()
            # final unrestricted LP polish on structured result
            remain = 172.0 - (time.time() - t_start)
            if remain > 4:
                try:
                    X_struct_best = _lp_polish(X_struct_best, min(remain, 25.0), r0=0.003)
                    struct_val = _max_cos(X_struct_best)
                except Exception:
                    pass
    except Exception:
        pass
    # ---------- Final selection with fallback ----------
    try:
        if X_struct_best is not None:
            if struct_val < best_val - 1e-12:
                X_best = X_struct_best.copy()
                best_val = struct_val
            else:
                # even if structured not better, try polishing incumbent with leftover time
                remain = 173.0 - (time.time() - t_start)
                if remain > 4:
                    try:
                        Xp = _lp_polish(X_best, min(remain, 20.0), r0=0.002)
                        vp = _max_cos(Xp)
                        if vp < best_val:
                            X_best = Xp.copy()
                            best_val = vp
                    except Exception:
                        pass
        else:
            remain = 173.0 - (time.time() - t_start)
            if remain > 4:
                try:
                    Xp = _lp_polish(X_best, min(remain, 20.0), r0=0.002)
                    if _max_cos(Xp) < best_val:
                        X_best = Xp.copy()
                except Exception:
                    pass
    except Exception:
        pass
    X_best = np.nan_to_num(X_best, nan=0.0, posinf=0.0, neginf=0.0)
    nr = np.linalg.norm(X_best, axis=1, keepdims=True)
    nr[nr == 0] = 1.0
    X_best = X_best / nr
    # strict feasibility: ensure shape
    if X_best.shape != (162, 8):
        return X_inc
    # crash-proof fallback to incumbent if worse due to numerics
    try:
        if _max_cos(X_best) > best_val + 1e-12:
            # X_best should already be best; if struct made it worse, return incumbent polished? keep X_best as min
            pass
        # ensure we return at least incumbent if our result is worse than loaded best
        if _max_cos(X_best) > _max_cos(X_inc) + 1e-12:
            return X_inc
    except Exception:
        pass
    return X_best
