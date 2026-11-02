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
    TRIAL_BUDGET_END = t0 + 120.0
    rng = np.random.default_rng(0)
    n, d = 74, 13
    cands = []
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=np.float64)
            if a.shape == (n, d):
                cands.append(_normalize(a))
        except Exception:
            pass
    if not cands:
        cands.append(_normalize(rng.standard_normal((n, d))))
    vals = [_max_cos(a) for a in cands]
    best = cands[int(np.argmin(vals))].copy()
    best_val = float(min(vals))

    IU = np.triu_indices(n, 1)
    I0 = IU[0].astype(np.int64); J0 = IU[1].astype(np.int64)
    M = len(I0)

    def lse_value_grad_full(X, tau):
        G = X @ X.T
        g = G[I0, J0]
        mmax = float(g.max())
        e = np.exp((g - mmax) / tau)
        S = float(e.sum())
        if not np.isfinite(S) or S <= 0:
            return float(mmax), np.zeros_like(X), mmax
        F = mmax + tau * np.log(S + 1e-300)
        w = e / (S + 1e-300)
        Wx = np.zeros_like(X)
        np.add.at(Wx, I0, w[:, None] * X[J0])
        np.add.at(Wx, J0, w[:, None] * X[I0])
        return F, Wx, mmax

    def lse_polish_freeze(X0, tau, maxiter, frozen=None):
        X0 = _normalize(X0)
        if frozen is None or len(frozen) == 0:
            y0 = X0.ravel().copy()
            def fun(yf):
                Y = yf.reshape(n, d)
                r2 = (Y * Y).sum(axis=1, keepdims=True)
                r = np.sqrt(np.maximum(r2, 1e-24))
                r = np.maximum(r, 1e-12)
                X = Y / r
                F, Wx, _ = lse_value_grad_full(X, tau)
                dot = (Wx * X).sum(axis=1, keepdims=True)
                gradY = (Wx - dot * X) / r
                return float(F), gradY.ravel()
            try:
                res = minimize(fun, y0, jac=True, method="L-BFGS-B",
                               options={"maxiter": int(maxiter), "ftol": 1e-20,
                                        "gtol": 1e-7, "maxfun": int(maxiter * 8)})
                Y = res.x.reshape(n, d)
            except Exception:
                return X0
            rn = np.linalg.norm(Y, axis=1, keepdims=True); rn[rn == 0] = 1.0
            X = Y / rn
            if not np.all(np.isfinite(X)):
                return X0
            return _normalize(X)
        else:
            frozen = np.asarray(list(frozen), dtype=int)
            fset = set(frozen.tolist())
            free = np.array([i for i in range(n) if i not in fset], dtype=int)
            mf = len(free)
            if mf == 0:
                return X0
            Xfroz = X0[frozen].copy()
            y0 = X0[free].ravel().copy()
            def fun2(yf):
                Yf = yf.reshape(mf, d)
                r2 = (Yf * Yf).sum(axis=1, keepdims=True)
                r = np.sqrt(np.maximum(r2, 1e-24))
                r = np.maximum(r, 1e-12)
                Xfree = Yf / r
                X = np.empty((n, d), dtype=np.float64)
                X[free] = Xfree
                X[frozen] = Xfroz
                F, Wx, _ = lse_value_grad_full(X, tau)
                dot = (Wx[free] * Xfree).sum(axis=1, keepdims=True)
                gradYf = (Wx[free] - dot * Xfree) / r
                return float(F), gradYf.ravel()
            try:
                res = minimize(fun2, y0, jac=True, method="L-BFGS-B",
                               options={"maxiter": int(maxiter), "ftol": 1e-20,
                                        "gtol": 1e-7, "maxfun": int(maxiter * 8)})
                Yf = res.x.reshape(mf, d)
            except Exception:
                return X0
            rn = np.linalg.norm(Yf, axis=1, keepdims=True); rn[rn == 0] = 1.0
            Xfree = Yf / rn
            X = np.empty((n, d), dtype=np.float64)
            X[free] = Xfree
            X[frozen] = Xfroz
            if not np.all(np.isfinite(X)):
                return X0
            return _normalize(X)

    def lp_polish(X, seconds, radius_init=1e-3):
        X = _normalize(X)
        bestv = _max_cos(X)
        start = time.time()
        radius = float(radius_init)
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
            I = iu[0][sel].astype(np.int64); J = iu[1][sel].astype(np.int64)
            c = Cfull[sel]
            p = len(I)
            if p == 0:
                break
            nv = n * d + 1
            try:
                rows = np.repeat(np.arange(p), 2 * d + 1)
                cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
                vals = np.concatenate([X[J], X[I], -np.ones((p, 1))], 1).ravel()
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
            if v < bestv:
                X, bestv = Y, v
                radius = min(radius * 1.5, 0.02)
            else:
                radius *= 0.5
        return X

    def contact_info(X, slack=1e-7):
        G = X @ X.T
        np.fill_diagonal(G, -np.inf)
        gmax = float(G.max())
        mask = G > (gmax - slack)
        deg = mask.sum(axis=1).astype(int)
        prow = G.max(axis=1)
        return G, gmax, prow, deg

    def hole_candidates(Xkeep, n_cand=20000):
        cand = _normalize(rng.standard_normal((n_cand, d)))
        S = cand @ Xkeep.T
        sc = S.max(axis=1)
        return cand, sc

    def perturb_rad(y, lo=0.05, hi=0.20):
        theta = float(rng.uniform(lo, hi))
        v = rng.standard_normal(d)
        v = v - float(v @ y) * y
        nv = float(np.linalg.norm(v))
        if nv < 1e-12 or not np.finite(nv):
            return _normalize(y.reshape(1, -1))[0]
        v = v / nv
        yn = np.cos(theta) * y + np.sin(theta) * v
        nn = float(np.linalg.norm(yn))
        if nn == 0 or not np.finite(nn):
            return y
        return yn / nn

    # light baseline to stabilize degrees (cheap)
    if time.time() - t0 < LIM - 130:
        try:
            Xb = lse_polish_freeze(best, 8e-5, 60, frozen=None)
            vb = _max_cos(Xb)
            if vb < best_val:
                best, best_val = Xb.copy(), float(vb)
        except Exception:
            pass

    _, _, _, deg0 = contact_info(best, slack=1e-7)
    seq0 = tuple(sorted(deg0.tolist()))
    archive = set([seq0])

    # build deterministic shuffled enumeration of rattler/hub choices
    def build_trials():
        G, gmax, prow, deg = contact_info(best, slack=1e-7)
        rattlers = np.where(deg == 0)[0].tolist()
        hubs = np.where(deg >= 7)[0].tolist()
        if len(rattlers) == 0:
            rattlers = np.argsort(deg)[:8].tolist()
        if len(hubs) == 0:
            hubs = np.argsort(-deg)[:8].tolist()
        # neighbour with highest contact degree (tie: max cosine)
        trials = []
        for h in hubs:
            row = G[h].copy()
            row[h] = -np.inf
            nbr = int(np.argmax(row))
            # single-hub trials paired with each rattler (for shuffle diversity)
            for r in rattlers:
                trials.append(("single", int(h), int(r)))
            # double trials hub+nbr paired with rattler
            for r in rattlers:
                if nbr != h:
                    trials.append(("double", int(h), int(nbr), int(r)))
        # shuffle deterministically
        perm = rng.permutation(len(trials))
        trials = [trials[i] for i in perm]
        return trials, rattlers, hubs

    trials, _, _ = build_trials()
    # limit enumeration length
    if len(trials) > 80:
        trials = trials[:80]
    t_idx = 0
    # refresh trials periodically if best changes a lot? keep static for determinism but rebuild once if stalled
    while time.time() < TRIAL_BUDGET_END and time.time() - t0 < LIM - 30 and t_idx < len(trials):
        tr = trials[t_idx]; t_idx += 1
        if time.time() - t0 > LIM - 30:
            break
        try:
            Gc, gmaxc, prowc, degc = contact_info(best, slack=1e-7)
        except Exception:
            continue
        # parse trial
        try:
            if tr[0] == "single":
                _, hub, rat = tr
                victims = [int(hub)]
            else:
                _, hub, nbr, rat = tr
                # verify nbr still highest-contact neighbour of hub; recompute
                row = Gc[int(hub)].copy(); row[int(hub)] = -np.inf
                nnbr = int(np.argmax(row))
                victims = [int(hub), int(nnbr)]
                victims = list(dict.fromkeys(victims))[:2]
            victims = [int(v) for v in victims if 0 <= int(v) < n]
            if len(victims) == 0:
                continue
            keep_mask = np.ones(n, dtype=bool); keep_mask[victims] = False
            Xkeep = best[keep_mask]
            # sample 20000 candidates scored to non-moving points
            cand, sc = hole_candidates(Xkeep, n_cand=20000)
            # best global candidate
            bi = int(np.argmin(sc))
            # for diversity: if rattler-influenced, sometimes pick among top-20 random
            # deterministic choice based on t_idx
            if (t_idx % 4) == 0:
                # pick random among top 20 holes
                topk = np.argpartition(sc, 20)[:20]
                bi = int(topk[int(rng.integers(len(topk)))])
            y1 = cand[bi].copy()
            # perturb 0.05-0.20 rad tangent
            y1p = perturb_rad(y1, 0.05, 0.20)
            Xt = best.copy()
            Xt[victims[0]] = y1p
            if len(victims) == 2:
                # second hole conditioned on first
                Xkeep2 = np.vstack([Xkeep, y1p.reshape(1, -1)])
                cand2, sc2 = hole_candidates(Xkeep2, n_cand=12000)
                bi2 = int(np.argmin(sc2))
                y2 = cand2[bi2].copy()
                y2p = perturb_rad(y2, 0.05, 0.20)
                Xt[victims[1]] = y2p
            Xt = _normalize(Xt)
            frozen = victims  # freeze transplanted
            # phase 1: frozen annealed LSE at tau=1e-3 (~200 steps -> use 120 for speed, plus 80)
            Xp = lse_polish_freeze(Xt, 1e-3, 120, frozen=frozen)
            # extra frozen steps to reach ~200
            if time.time() - t0 > LIM - 28:
                continue
            Xp = lse_polish_freeze(Xp, 1e-3, 80, frozen=frozen)
            # unfreeze polish tau=3e-4, 8e-5
            Xp = lse_polish_freeze(Xp, 3e-4, 60, frozen=None)
            if time.time() - t0 > LIM - 26:
                continue
            Xp = lse_polish_freeze(Xp, 8e-5, 80, frozen=None)
            vpost = _max_cos(Xp)
            # quick reject if far
            if vpost > best_val + 5e-3:
                # still record archive to avoid repeat
                try:
                    _, _, _, dtmp = contact_info(Xp, slack=1e-7)
                    archive.add(tuple(sorted(dtmp.tolist())))
                except Exception:
                    pass
                continue
            _, _, _, degp = contact_info(Xp, slack=1e-7)
            seqp = tuple(sorted(degp.tolist()))
            is_new = (seqp not in archive)
            archive.add(seqp)
            need_lp = is_new or (vpost < best_val + 1e-6)
            if need_lp:
                # 1-2 s trust-region LP polish
                budget = 1.2
                # give a bit more if very promising
                if vpost < best_val + 1e-6:
                    budget = 2.0
                # cap by remaining trial budget
                remain_trial = TRIAL_BUDGET_END - time.time()
                if remain_trial < 0.5:
                    budget = min(budget, 0.6)
                Xp = lp_polish(Xp, seconds=budget, radius_init=1e-3)
                vpost = _max_cos(Xp)
                # small extra LSE after LP
                Xp2 = lse_polish_freeze(Xp, 8e-5, 40, frozen=None)
                v2 = _max_cos(Xp2)
                if v2 < vpost:
                    Xp, vpost = Xp2, float(v2)
            if vpost < best_val - 1e-13:
                best, best_val = Xp.copy(), float(vpost)
                # rebuild trials once to reflect new topology if big gain? keep archive
                if len(archive) > 200:
                    pass
        except Exception:
            continue

    # final polish on best only
    if time.time() - t0 < LIM - 1.0:
        try:
            Xf = lse_polish_freeze(best, 3e-4, 60, frozen=None)
            vf = _max_cos(Xf)
            if vf < best_val:
                best_val, best = float(vf), Xf.copy()
            Xf = lse_polish_freeze(best, 8e-5, 100, frozen=None)
            vf = _max_cos(Xf)
            if vf < best_val:
                best_val, best = float(vf), Xf.copy()
            # short LP if time
            rem = (t0 + LIM - 0.6) - time.time()
            if rem > 1.0:
                Xl = lp_polish(best, seconds=min(rem - 0.4, 3.0), radius_init=8e-4)
                vl = _max_cos(Xl)
                if vl < best_val:
                    best_val, best = float(vl), Xl.copy()
                    Xf2 = lse_polish_freeze(best, 8e-5, 60, frozen=None)
                    vf2 = _max_cos(Xf2)
                    if vf2 < best_val:
                        best_val, best = float(vf2), Xf2.copy()
        except Exception:
            pass
    best = _normalize(best)
    return best
