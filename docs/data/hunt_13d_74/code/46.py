import time
import numpy as np
from scipy.optimize import minimize, linprog
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _normalize(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return X / n

def solve():
    t0 = time.time()
    LIM = 172.0
    TRIAL_BUDGET = 120.0
    t_trial_end = t0 + TRIAL_BUDGET
    rng = np.random.default_rng(20261102)
    n, d = 74, 13

    cands = []
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=np.float64)
            if a.shape == (74, 13):
                a = _normalize(a)
                cands.append(a)
        except Exception:
            pass
    if not cands:
        cands.append(_normalize(rng.standard_normal((74, 13))))
    vals = [_max_cos(a) for a in cands]
    bi = int(np.argmin(vals))
    best = cands[bi].copy()
    best_val = float(vals[bi])

    IU = np.triu_indices(n, 1)
    I0 = IU[0].astype(np.int64)
    J0 = IU[1].astype(np.int64)

    def contact_info(X, thr=1e-7):
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        gmax = float(G.max())
        cnt = (G > gmax - thr).sum(axis=1)
        # top neighbour for each i
        topnb = np.argmax(G, axis=1)
        return cnt, gmax, G, topnb

    def lse_polish(X0, tau, maxiter, freeze=None):
        # freeze: boolean (n,) True=fixed
        if freeze is None:
            free_idx = np.arange(n)
        else:
            free_idx = np.where(~freeze)[0]
        if len(free_idx) == 0:
            return X0.copy()
        Xfix = X0.copy()
        y0 = X0[free_idx].ravel().copy()
        # map free position to row
        def fun(yf):
            Y = np.empty((n, d))
            Y[:] = Xfix  # for fixed rows Y=X (unit); for free overwritten
            Y[free_idx] = yf.reshape(len(free_idx), d)
            r2 = (Y * Y).sum(axis=1, keepdims=True)
            r = np.sqrt(np.maximum(r2, 1e-24))
            r = np.maximum(r, 1e-12)
            X = Y / r
            G = X @ X.T
            g = G[I0, J0]
            mmax = float(g.max())
            e = np.exp((g - mmax) / tau)
            S = float(e.sum())
            if not np.isfinite(S) or S <= 0:
                return float(mmax), np.zeros_like(yf)
            F = mmax + tau * np.log(S + 1e-300)
            w = e / (S + 1e-300)
            Wx = np.zeros_like(X)
            np.add.at(Wx, I0, w[:, None] * X[J0])
            np.add.at(Wx, J0, w[:, None] * X[I0])
            dot = (Wx * X).sum(axis=1, keepdims=True)
            gradY = (Wx - dot * X) / r
            gf = gradY[free_idx].ravel()
            if not np.all(np.isfinite(gf)):
                gf = np.zeros_like(yf)
            return float(F), gf
        try:
            res = minimize(fun, y0, jac=True, method="L-BFGS-B",
                           options={"maxiter": int(maxiter), "ftol": 1e-20,
                                    "gtol": 1e-7, "maxfun": int(maxiter * 8)})
            Y = np.empty((n, d))
            Y[:] = Xfix
            Y[free_idx] = res.x.reshape(len(free_idx), d)
        except Exception:
            return X0.copy()
        rn = np.linalg.norm(Y, axis=1, keepdims=True)
        rn[rn == 0] = 1.0
        X = Y / rn
        if not np.all(np.isfinite(X)):
            return X0.copy()
        return _normalize(X)

    def lp_polish(X, seconds, radius_init=1e-3):
        X = _normalize(X.copy())
        cur = _max_cos(X)
        start = time.time()
        radius = radius_init
        iu0 = IU
        while time.time() - start < seconds and radius > 1e-14:
            if time.time() - t0 > LIM - 3:
                break
            Gf = (X @ X.T)[iu0]
            m = float(Gf.max())
            sel = Gf > m - max(20.0 * radius, 1e-9)
            if int(np.count_nonzero(sel)) > 2500:
                thr = np.sort(Gf)[-2500]
                sel = Gf >= thr
            I = iu0[0][sel]; J = iu0[1][sel]
            c = Gf[sel]
            p = len(I)
            if p == 0:
                break
            nv = n * d + 1
            rows = np.repeat(np.arange(p), 2 * d + 1)
            cI = np.tile(np.arange(d), p)
            # cols: I*d+k, J*d+k, last
            cols_I = (np.repeat(I, d) * d + cI)
            cols_J = (np.repeat(J, d) * d + cI)
            cols_t = np.full(p, n * d)
            cols = np.concatenate([cols_I, cols_J, cols_t])
            # need per-row ordering: row r has [I-part d, J-part d, t]
            # build vals in same row order
            vals_I = X[J].repeat(1, axis=0)  # X[J] shape (p,d)
            vals_J = X[I]
            # rows repeat already interleaves? repeat gives [row0*d... ] need block
            # Simpler: build with loop-free reshape:
            # rows array is [0..0,1..1...] with 2d+1 each; cols above must match
            # Our cols_I is grouped by p then d, which matches rows grouping first d entries.
            # So reorder cols to [I_d, J_d, t] per row:
            cols_per_row = np.empty((p, 2 * d + 1), dtype=np.int64)
            cols_per_row[:, :d] = (I[:, None] * d + np.arange(d))
            cols_per_row[:, d:2 * d] = (J[:, None] * d + np.arange(d))
            cols_per_row[:, 2 * d] = n * d
            cols = cols_per_row.ravel()
            vals_per_row = np.empty((p, 2 * d + 1))
            vals_per_row[:, :d] = X[J]
            vals_per_row[:, d:2 * d] = X[I]
            vals_per_row[:, 2 * d] = -1.0
            vals = vals_per_row.ravel()
            try:
                A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
                A_eq = coo_matrix((X.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            except Exception:
                radius *= 0.5
                continue
            obj = np.zeros(nv); obj[-1] = 1.0
            try:
                res = linprog(obj, A_ub=A_ub, b_ub=-c, A_eq=A_eq, b_eq=np.zeros(n),
                              bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
            except Exception:
                radius *= 0.5
                continue
            if res.status != 0:
                radius *= 0.5
                continue
            Y = X + res.x[:-1].reshape(n, d)
            Y = _normalize(Y)
            v = _max_cos(Y)
            if v < cur:
                X, cur = Y, v
                radius = min(radius * 1.5, 0.02)
            else:
                radius *= 0.5
        return X, cur

    def best_hole(Xfixed, n_cand=20000):
        C = _normalize(rng.standard_normal((n_cand, d)))
        S = C @ Xfixed.T
        sc = S.max(axis=1)
        i = int(np.argmin(sc))
        return C[i], float(sc[i])

    def perturb_tangent(c, lo=0.05, hi=0.20):
        theta = float(rng.uniform(lo, hi))
        r = rng.standard_normal(d)
        r = r - float(r @ c) * c
        nr = float(np.linalg.norm(r))
        if nr < 1e-12:
            r = rng.standard_normal(d)
            r = r - float(r @ c) * c
            nr = float(np.linalg.norm(r)) + 1e-12
        r /= nr
        cn = np.cos(theta) * c + np.sin(theta) * r
        return cn / (np.linalg.norm(cn) + 1e-12)

    # archive keyed by sorted degree sequence
    cnt0, gmax0, _, _ = contact_info(best, thr=1e-7)
    archive = set()
    archive.add(tuple(sorted(cnt0.tolist())))

    # build trial queue: deterministic shuffled enumeration
    cnt, gmax, Gfull, topnb = contact_info(best, thr=1e-7)
    rattlers = np.where(cnt == 0)[0].tolist()
    hubs = np.argsort(-cnt)[:10].tolist()
    # hubs defined as degree>=7, fallback to top-10
    hubs7 = [int(i) for i in range(n) if cnt[i] >= 7]
    if len(hubs7) == 0:
        hubs7 = hubs
    if len(rattlers) == 0:
        # lowest-degree points act as rattlers
        rattlers = np.argsort(cnt)[:4].tolist()

    trials = []
    # single hub moves paired with rattler hole context
    for h in hubs7:
        for r in rattlers:
            trials.append(("single", int(h), int(r)))
    # double moves: hub + its highest-contact neighbour, paired with rattler index for ordering
    for h in hubs7:
        nb = int(topnb[h])
        if nb == h:
            continue
        trials.append(("double", int(h), int(nb)))
    # hub+rattler double-hole moves
    for h in hubs7:
        for r in rattlers:
            trials.append(("hub_rattler", int(h), int(r)))
    # shuffle deterministically
    perm = rng.permutation(len(trials))
    trials = [trials[i] for i in perm]

    # short frozen anneal settings
    n_cand_global = 20000

    for ttype, a, b in trials:
        if time.time() > t_trial_end or time.time() - t0 > LIM - 25:
            break
        try:
            Xcur = best.copy()
            if ttype == "single":
                h = a
                fixed_idx = [i for i in range(n) if i != h]
                Xf = Xcur[fixed_idx]
                cstar, _ = best_hole(Xf, n_cand=n_cand_global)
                cstar = perturb_tangent(cstar, 0.05, 0.20)
                Xcur[h] = cstar
                moving = [h]
            elif ttype == "double":
                h = a; nb = b
                if nb == h:
                    continue
                fixed_idx = [i for i in range(n) if i != h and i != nb]
                Xf = Xcur[fixed_idx]
                c1, _ = best_hole(Xf, n_cand=n_cand_global)
                c1p = perturb_tangent(c1, 0.05, 0.20)
                # second hole conditioned on first
                Xf2 = np.vstack([Xf, c1p[None, :]])
                c2, _ = best_hole(Xf2, n_cand=8000)
                c2p = perturb_tangent(c2, 0.05, 0.20)
                Xcur[h] = c1p
                Xcur[nb] = c2p
                moving = [h, nb]
            else:  # hub_rattler
                h = a; r = b
                if r == h:
                    continue
                fixed_idx = [i for i in range(n) if i != h and i != r]
                Xf = Xcur[fixed_idx]
                c1, _ = best_hole(Xf, n_cand=n_cand_global)
                c1p = perturb_tangent(c1, 0.05, 0.20)
                Xf2 = np.vstack([Xf, c1p[None, :]])
                c2, _ = best_hole(Xf2, n_cand=8000)
                c2p = perturb_tangent(c2, 0.05, 0.20)
                Xcur[h] = c1p
                Xcur[r] = c2p
                moving = [h, r]

            freeze = np.zeros(n, dtype=bool)
            # freeze transplanted points first: freeze=True means fixed
            # we want them frozen => all others free? No: freeze transplanted, optimize rest?
            # Spec: "Freeze the transplanted point for first ~200 steps" => transplanted fixed, rest moves
            for m in moving:
                freeze[m] = True
            # stage 1: frozen anneal at tau=1e-3 (~200 steps -> use maxiter 120 for speed, two calls)
            Xcur = lse_polish(Xcur, 1e-3, 110, freeze=freeze)
            if time.time() - t0 > LIM - 20:
                vcur = _max_cos(Xcur)
                if vcur < best_val:
                    best, best_val = _normalize(Xcur), vcur
                continue
            # unfreeze and polish at tau=3e-4, 8e-5
            Xcur = lse_polish(Xcur, 3e-4, 90, freeze=None)
            Xcur = lse_polish(Xcur, 8e-5, 90, freeze=None)
            vcur = _max_cos(Xcur)
            cntc, _, _, _ = contact_info(Xcur, thr=1e-7)
            key = tuple(sorted(cntc.tolist()))
            is_new = (key not in archive)
            # only spend final LP polish if degree seq differs or within 1e-6 of best
            if is_new or (vcur < best_val + 1e-6):
                Xcur2, vcur2 = lp_polish(Xcur, seconds=1.0, radius_init=1e-3)
                # extra tiny LSE after LP
                Xcur2 = lse_polish(Xcur2, 8e-5, 40, freeze=None)
                vcur2 = _max_cos(Xcur2)
                archive.add(key)
                # also add post-LP key
                try:
                    cnt2, _, _, _ = contact_info(Xcur2, thr=1e-7)
                    archive.add(tuple(sorted(cnt2.tolist())))
                except Exception:
                    pass
                if vcur2 < best_val - 1e-12:
                    best, best_val = _normalize(Xcur2.copy()), float(vcur2)
                    # refresh hub/rattler lists if improved substantially? keep original queue
                elif vcur < best_val - 1e-12:
                    best, best_val = _normalize(Xcur.copy()), float(vcur)
                    archive.add(key)
            else:
                archive.add(key)
                if vcur < best_val - 1e-12:
                    best, best_val = _normalize(Xcur.copy()), float(vcur)
        except Exception:
            continue

    # final polish on best with remaining time
    try:
        if time.time() - t0 < LIM - 4:
            best = lse_polish(best, 3e-4, 80, freeze=None)
            best = lse_polish(best, 8e-5, 100, freeze=None)
            bv = _max_cos(best)
            if bv < best_val:
                best_val = bv
            if time.time() - t0 < LIM - 6:
                best2, bv2 = lp_polish(best, seconds=min(4.0, max(0.5, LIM - (time.time() - t0) - 2.0)), radius_init=5e-4)
                if bv2 < best_val:
                    best, best_val = best2, bv2
    except Exception:
        pass

    best = _normalize(best)
    best[~np.isfinite(best)] = 0.0
    nr = np.linalg.norm(best, axis=1)
    for i in range(n):
        if nr[i] < 1e-12 or not np.isfinite(nr[i]):
            best[i] = _normalize(rng.standard_normal((1, d)))[0]
    best = _normalize(best)
    return best
