import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _norm(x):
    n = np.linalg.norm(x, axis=1, keepdims=True) + 1e-300
    return x / n

def _mc(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def _lp_polish(x, t0, deadline, r0=2e-3):
    x = _norm(x.copy())
    n, d = x.shape
    best = _mc(x)
    xb = x.copy()
    radius = r0
    iu0 = np.triu_indices(n, 1)
    ar = np.arange(d)
    while time.time() < deadline - 0.5 and radius > 1e-13:
        G = x @ x.T
        cf = G[iu0]
        m = float(cf.max())
        thr = m - max(20.0 * radius, 1e-9)
        sel = cf > thr
        I = iu0[0][sel]; J = iu0[1][sel]; cs = cf[sel]
        p = len(I)
        if p == 0:
            radius *= 0.5
            continue
        if p > 3000:
            idx = np.argpartition(-cs, 3000)[:3000]
            I, J, cs = I[idx], J[idx], cs[idx]
            p = 3000
        nv = n * d + 1
        cols = np.concatenate([np.repeat(I * d, d) + np.tile(ar, p),
                               np.repeat(J * d, d) + np.tile(ar, p),
                               np.full(p, n * d)])
        rows = np.concatenate([np.repeat(np.arange(p), d),
                               np.repeat(np.arange(p), d),
                               np.arange(p)])
        vals = np.concatenate([x[J].ravel(), x[I].ravel(), -np.ones(p)])
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv); obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-cs, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = _norm(x + res.x[:-1].reshape(n, d))
        v = _mc(y)
        if v < best:
            best = v; xb = y.copy(); x = y
            radius = min(radius * 1.5, 0.02)
        else:
            radius *= 0.5
            if radius < 5e-7 and p > 1500:
                break
    return xb

def solve():
    t0 = time.time()
    DEAD = t0 + 175.0
    rng = np.random.default_rng(0)
    n, d = 184, 8
    try:
        xb0 = np.load("best.npy")
    except Exception:
        xb0 = None
    try:
        xr0 = np.load("record.npy")
    except Exception:
        xr0 = None
    cands = []
    for c in (xb0, xr0):
        try:
            a = _norm(np.asarray(c, dtype=np.float64).reshape(n, d))
            cands.append((_mc(a), a))
        except Exception:
            pass
    if cands:
        cands.sort(key=lambda t: t[0])
        gval, gbest = cands[0][0], cands[0][1].copy()
    else:
        gbest = _norm(rng.standard_normal((n, d)))
        gval = _mc(gbest)
    LP_THRESH = 0.49998295  # lab best; LP only after beating it
    # target continuation: decreasing feasibility thresholds
    taus = [0.49997, 0.49995, 0.49992, 0.49990, 0.49985, 0.49980]
    X = gbest.copy()
    cur_max = _mc(X)
    # Adam state reused per tau
    b1, b2, eps = 0.9, 0.999, 1e-8
    for ti, tau in enumerate(taus):
        if time.time() > DEAD - 35:
            break
        # skip taus already satisfied with margin? still try lower
        # run up to R retries at this tau without reverting to incumbent
        R = 3
        STEPS = 500
        lr_base = 0.003 if ti < 2 else 0.002
        active_thr = tau - 0.02
        c_hinge = tau  # sparse hinge sum max(0,G-tau)^2; (+0.002 margin absorbed into feasibility check)
        for retry in range(R):
            if time.time() > DEAD - 35:
                break
            m1 = np.zeros_like(X)
            m2 = np.zeros_like(X)
            # init active mask
            G0 = X @ X.T
            np.fill_diagonal(G0, -2.0)
            act = (G0 > active_thr)
            np.fill_diagonal(act, False)
            best_retry = cur_max
            stagn = 0
            # adapt lr slightly per retry
            lr = lr_base * (0.7 ** retry) if retry > 0 else lr_base
            for t in range(1, STEPS + 1):
                if time.time() > DEAD - 35:
                    break
                if t % 20 == 1:
                    G = X @ X.T
                    np.fill_diagonal(G, -2.0)
                    cur_max_tmp = float(np.max(G))
                    act = (G > active_thr)
                    np.fill_diagonal(act, False)
                else:
                    G = X @ X.T
                    np.fill_diagonal(G, -2.0)
                    cur_max_tmp = float(np.max(G))
                # track global best (feasibility topology change, no revert)
                if cur_max_tmp < gval:
                    gval = cur_max_tmp
                    gbest = X.copy()
                if cur_max_tmp < best_retry - 1e-12:
                    best_retry = cur_max_tmp
                    stagn = 0
                else:
                    stagn += 1
                # hinge energy on active set
                # H = relu(G - c) * act
                H = G - c_hinge
                H[H < 0] = 0.0
                H[~act] = 0.0
                np.fill_diagonal(H, 0.0)
                # gradient 2*H@X
                grad = 2.0 * (H @ X)
                dot = np.sum(grad * X, axis=1, keepdims=True)
                gt = grad - dot * X
                m1 = b1 * m1 + (1 - b1) * gt
                m2 = b2 * m2 + (1 - b2) * (gt * gt)
                bc1 = 1 - b1 ** t
                bc2 = 1 - b2 ** t
                mh = m1 / bc1
                vh = m2 / bc2
                upd = lr * mh / (np.sqrt(vh) + eps)
                # keep update tangential-ish
                X = X - upd
                X = _norm(X)
                # early exit if feasible at tau
                if cur_max_tmp < tau:
                    break
                # mild stagnation: shrink lr once
                if stagn == 300:
                    lr *= 0.5
            Gf = X @ X.T
            np.fill_diagonal(Gf, -2.0)
            cur_max = float(np.max(Gf))
            if cur_max < gval:
                gval = cur_max
                gbest = X.copy()
            if cur_max < tau:
                # feasible: go to next lower tau, carry X forward
                break
            # stagnated: kick restricted to violated endpoints + high dual-stress points
            # stress proxy: per-point hinge sum (LP dual analogue)
            Hv = np.maximum(0.0, Gf - c_hinge)
            Hv[Hv < 0] = 0.0
            stress = Hv.sum(axis=1)
            # endpoints of worst pairs
            iu = np.triu_indices(n, 1)
            cf = Gf[iu]
            kord = np.argpartition(-cf, min(40, len(cf)-1))[:40]
            # sort top few
            kord = kord[np.argsort(-cf[kord])]
            cand = set()
            for kk in kord[:12]:
                cand.add(int(iu[0][kk])); cand.add(int(iu[1][kk]))
            sord = np.argsort(-stress, kind="stable")
            for s in sord[:8]:
                cand.add(int(s))
            cand = list(cand)
            # choose nk victims
            nk = 3 + int(rng.integers(0, 3))  # 3-5
            # bias to highest stress
            probs = stress[np.array(cand)] + 1e-12
            probs = probs / probs.sum()
            # if probs degenerate, uniform
            try:
                victims = rng.choice(cand, size=min(nk, len(cand)), replace=False, p=probs)
            except Exception:
                victims = rng.choice(cand, size=min(nk, len(cand)), replace=False)
            sigma = 0.015 * (1.3 ** retry)  # slightly larger on repeated stagnation
            for vi in victims:
                vi = int(vi)
                v = rng.standard_normal(d)
                v = v - np.dot(v, X[vi]) * X[vi]
                vn = np.linalg.norm(v) + 1e-300
                v = v / vn
                X[vi] = X[vi] + sigma * v
                X[vi] /= (np.linalg.norm(X[vi]) + 1e-300)
            # do NOT revert to incumbent; retry from kicked X
            # if last retry and still infeasible at this tau, stop going lower
            if retry == R - 1:
                pass
        # after retries, check feasibility; if not feasible, don't attempt harder taus
        if cur_max >= tau:
            break
        # else continue to next tau
    # trust-region LP only after satisfying target below lab best
    if gval < LP_THRESH and time.time() < DEAD - 3:
        secs_left = DEAD - 1.0 - time.time()
        if secs_left > 3:
            gbest = _lp_polish(gbest, time.time(), DEAD - 1.0)
            gbest = _norm(gbest)
            # also try polishing current wandering X if better
            nv = _mc(gbest)
            if nv < gval:
                gval = nv
    else:
        # no improvement: return incumbent (ensure normalized)
        pass
    gbest = _norm(gbest)
    return gbest
