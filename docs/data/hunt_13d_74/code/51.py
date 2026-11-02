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
        cands.append(_normalize(rng.standard_normal((n, 13))))
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
        cnt = (G > gmax - thr).sum(axis=1).astype(int)
        return cnt, gmax, G

    def lse_polish_mask(X0, tau, maxiter, frozen=None):
        # frozen: boolean array n, True means fixed
        if frozen is None:
            frozen = np.zeros(n, dtype=bool)
        else:
            frozen = np.asarray(frozen, dtype=bool)
        free_idx = np.where(~frozen)[0]
        nf = len(free_idx)
        if nf == 0:
            return X0.copy()
        Xfix = X0.copy()
        y0 = X0[free_idx].ravel().copy()
        # map from full row to free position
        pos = np.full(n, -1, dtype=int)
        pos[free_idx] = np.arange(nf)
        I0f = pos[I0]
        J0f = pos[J0]
        # for pairs where one end frozen, need Xfix values
        def fun(yf):
            Yf = yf.reshape(nf, d)
            r2 = (Yf * Yf).sum(axis=1, keepdims=True)
            r = np.sqrt(np.maximum(r2, 1e-24))
            r = np.maximum(r, 1e-12)
            Xf = Yf / r
            # build full X
            X = np.empty((n, d), dtype=np.float64)
            X[free_idx] = Xf
            X[frozen] = Xfix[frozen]
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
            # gradient w.r.t free only
            Wf = Wx[free_idx]
            dot = (Wf * Xf).sum(axis=1, keepdims=True)
            gradY = (Wf - dot * Xf) / r
            return float(F), gradY.ravel()
        try:
            res = minimize(fun, y0, jac=True, method="L-BFGS-B",
                           options={"maxiter": int(maxiter), "ftol": 1e-20,
                                    "gtol": 1e-7, "maxfun": int(maxiter * 8)})
            Yf = res.x.reshape(nf, d)
        except Exception:
            return X0.copy()
        rn = np.linalg.norm(Yf, axis=1, keepdims=True)
        rn[rn == 0] = 1.0
        Xout = X0.copy()
        Xout[free_idx] = Yf / rn
        if not np.all(np.isfinite(Xout)):
            return X0.copy()
        return _normalize(Xout)

    def lp_polish(X, seconds, radius_init=1e-3):
        X = _normalize(X.copy())
        cur = _max_cos(X)
        radius = float(radius_init)
        start = time.time()
        iu = np.triu_indices(n, 1)
        while time.time() - start < seconds and radius > 1e-14:
            if time.time() - t0 > LIM - 2:
                break
            Cfull = (X @ X.T)[iu]
            m = float(Cfull.max())
            sel = Cfull > m - max(20.0 * radius, 1e-9)
            if np.count_nonzero(sel) > 2500:
                thr = np.sort(Cfull)[-2500]
                sel = Cfull >= thr
            I = iu[0][sel]; J = iu[1][sel]
            c = Cfull[sel]
            p = len(I)
            if p == 0:
                break
            nv = n * d + 1
            rows = np.repeat(np.arange(p), 2 * d + 1)
            cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
            vals = np.concatenate([X[J], X[I], -np.ones((p, 1))], 1).ravel()
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

    def hole_candidate(Xkeep, n_cand=20000):
        cand = _normalize(rng.standard_normal((n_cand, d)))
        S = cand @ Xkeep.T
        sc = S.max(axis=1)
        k = int(np.argmin(sc))
        return cand[k], float(sc[k])

    def perturb_radial(y, lo=0.05, hi=0.20):
        theta = float(rng.uniform(lo, hi))
        v = rng.standard_normal(d)
        v = v - float(v @ y) * y
        nv = float(np.linalg.norm(v))
        if nv < 1e-12 or not np.isfinite(nv):
            return y
        v /= nv
        return np.cos(theta) * y + np.sin(theta) * v

    # initial contact analysis on incumbent
    cnt0, gmax0, G0 = contact_info(best, thr=1e-7)
    rattlers = list(np.where(cnt0 == 0)[0])
    hubs = list(np.argsort(-cnt0)[:10])
    hubs = [h for h in hubs if cnt0[h] >= 7]
    if len(hubs) == 0:
        hubs = list(np.argsort(-cnt0)[:6])
    if len(rattlers) == 0:
        # fallback: lowest degree points treated as rattlers
        rattlers = list(np.argsort(cnt0)[:6])

    # build trial list: singles (hub -> hole ignoring hub+rattler) and doubles (hub+neighbour)
    trials = []
    for h in hubs:
        for r in rattlers:
            trials.append(("single", int(h), int(r), -1))
    # hub neighbours: highest-degree contact, tie by max cosine
    for h in hubs:
        row = G0[h].copy()
        row[h] = -2.0
        # contacts
        cmask = row > gmax0 - 1e-7
        if np.any(cmask):
            cidx = np.where(cmask)[0]
            # pick max degree among contacts
            nb = int(cidx[np.argmax(cnt0[cidx])])
        else:
            nb = int(np.argmax(row))
        if nb == h:
            continue
        trials.append(("double", int(h), int(nb), -1))
    # shuffle deterministic
    order = np.arange(len(trials))
    rng.shuffle(order)
    trials = [trials[i] for i in order]

    # archive keyed by sorted degree sequence
    archive = set()
    tup0 = tuple(sorted(cnt0.tolist()))
    archive.add(tup0)

    # helper degree seq
    def deg_seq(X):
        c, _, _ = contact_info(X, thr=1e-7)
        return tuple(sorted(c.tolist())), c

    for ti, tr in enumerate(trials):
        if time.time() > t_trial_end or time.time() - t0 > LIM - 25:
            break
        kind = tr[0]
        Xcur = best.copy()
        frozen = np.zeros(n, dtype=bool)
        try:
            if kind == "single":
                h, r = tr[1], tr[2]
                if h == r:
                    continue
                keep_idx = [i for i in range(n) if i != h and i != r]
                Xkeep = Xcur[keep_idx]
                ystar, _ = hole_candidate(Xkeep, n_cand=20000)
                ystar = perturb_radial(ystar, 0.05, 0.20)
                ystar = ystar / (np.linalg.norm(ystar) + 1e-15)
                Xcur[h] = ystar
                frozen[h] = True
                victims = [h]
            else:  # double: hub + neighbour
                h, nb = tr[1], tr[2]
                if h == nb:
                    continue
                keep_idx = [i for i in range(n) if i != h and i != nb]
                Xkeep = Xcur[keep_idx]
                y1, _ = hole_candidate(Xkeep, n_cand=20000)
                y1 = perturb_radial(y1, 0.05, 0.20)
                y1 = y1 / (np.linalg.norm(y1) + 1e-15)
                Xcur[h] = y1
                # second hole conditioned on first
                Xkeep2 = np.vstack([Xkeep, y1[None, :]])
                y2, _ = hole_candidate(Xkeep2, n_cand=20000)
                y2 = perturb_radial(y2, 0.05, 0.20)
                y2 = y2 / (np.linalg.norm(y2) + 1e-15)
                Xcur[nb] = y2
                frozen[h] = True
                frozen[nb] = True
                victims = [h, nb]
        except Exception:
            continue

        try:
            # stage 1: frozen annealed LSE at tau=1e-3 (~200 steps)
            Xcur = lse_polish_mask(Xcur, 1e-3, 200, frozen=frozen)
            if time.time() > t_trial_end or time.time() - t0 > LIM - 20:
                vcur = _max_cos(Xcur)
                if vcur < best_val - 1e-12:
                    best, best_val = _normalize(Xcur), vcur
                continue
            # unfreeze, polish at tau=3e-4
            Xcur = lse_polish_mask(Xcur, 3e-4, 150, frozen=None)
            vmid = _max_cos(Xcur)
            seq, _ = deg_seq(Xcur)
            is_new = (seq not in archive)
            # only spend final polish if novel topology or competitive
            if (not is_new) and (vmid > best_val + 1e-6):
                # not worth LP; still record
                continue
            # final polish: tau=8e-5 + 1-2s LP
            Xcur = lse_polish_mask(Xcur, 8e-5, 150, frozen=None)
            Xcur, vcur = lp_polish(Xcur, 1.2, radius_init=8e-4)
            seq2, _ = deg_seq(Xcur)
            archive.add(seq)
            archive.add(seq2)
            if vcur < best_val - 1e-12:
                best = _normalize(Xcur).copy()
                best_val = float(vcur)
                # refresh contact info for remaining trials? keep original hubs/rattlers order
                # update trial end? keep strict 120s
        except Exception:
            continue

    # final polish on best with remaining time
    try:
        if time.time() - t0 < LIM - 8:
            Xf = best.copy()
            Xf = lse_polish_mask(Xf, 3e-4, 120, frozen=None)
            if time.time() - t0 < LIM - 6:
                Xf = lse_polish_mask(Xf, 8e-5, 150, frozen=None)
            if time.time() - t0 < LIM - 4:
                Xf, vf = lp_polish(Xf, min(2.0, max(0.5, LIM - 3 - (time.time() - t0))), radius_init=5e-4)
                if vf < best_val:
                    best, best_val = Xf, vf
    except Exception:
        pass

    best = _normalize(best)
    # sanitize
    best[~np.isfinite(best)] = 0.0
    nr = np.linalg.norm(best, axis=1)
    for i in range(n):
        if nr[i] < 1e-12 or not np.isfinite(nr[i]):
            best[i] = _normalize(rng.standard_normal((1, d)))[0]
    best = _normalize(best)
    return best
