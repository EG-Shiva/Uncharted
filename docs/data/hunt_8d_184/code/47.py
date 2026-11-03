import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _norm(x):
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-300)

def _max_cos(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def _lp_polish(x, seconds, deadline):
    x = _norm(x.copy())
    n, d = x.shape
    best = _max_cos(x)
    xb = x.copy()
    radius = 2e-3
    t0 = time.time()
    iu = np.triu_indices(n, 1)
    ar = np.arange(d)
    while time.time() - t0 < seconds:
        if time.time() > deadline:
            break
        if radius < 1e-14:
            break
        G = x @ x.T
        c = G[iu]
        m = float(np.max(c))
        sel = c > m - max(20.0 * radius, 2e-9)
        I = iu[0][sel]; J = iu[1][sel]; cs = c[sel]
        p = len(I)
        if p == 0:
            break
        if p > 3000:
            idx = np.argpartition(-cs, 3000)[:3000]
            I, J, cs = I[idx], J[idx], cs[idx]
            p = 3000
        nv = n * d + 1
        rows = np.concatenate([np.repeat(np.arange(p), d), np.repeat(np.arange(p), d), np.arange(p)])
        cols = np.concatenate([np.repeat(I * d, d) + np.tile(ar, p),
                               np.repeat(J * d, d) + np.tile(ar, p),
                               np.full(p, n * d)])
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
        v = _max_cos(y)
        if v < best:
            best = float(v); xb = y.copy(); x = y
            radius = min(radius * 1.5, 0.03)
        else:
            radius *= 0.5
    return xb

def solve():
    t0 = time.time()
    DEADLINE = t0 + 174.0
    rng = np.random.default_rng(12345)
    n, d = 184, 8
    try:
        x0 = np.asarray(np.load("best.npy"), dtype=np.float64).reshape(n, d)
    except Exception:
        x0 = np.asarray(np.load("record.npy"), dtype=np.float64).reshape(n, d)
    # also consider record
    try:
        xr = np.asarray(np.load("record.npy"), dtype=np.float64).reshape(n, d)
        xr = _norm(xr)
        if _max_cos(xr) < _max_cos(_norm(x0)):
            x0 = xr
    except Exception:
        pass
    x0 = _norm(x0)
    gbest = x0.copy()
    gval = _max_cos(gbest)
    incumbent_thr = 0.49998295  # must beat to allow LP
    # if incumbent already below (lab best slightly below), set threshold to gval
    # we require strictly below gval to trigger LP; use min
    thresh_trigger = min(incumbent_thr, gval - 1e-12) if gval < incumbent_thr + 1e-9 else incumbent_thr

    targets = [0.49997, 0.49995, 0.49992, 0.49990, 0.49985, 0.49980]
    # start from current best
    cur_x = gbest.copy()
    cur_val = gval
    b1, b2, eps = 0.9, 0.999, 1e-8
    lr_base = 0.012

    feasible_below = False
    # time budget for continuation: leave 45s for LP
    CONT_END = DEADLINE - 45.0

    for tau in targets:
        if time.time() > CONT_END:
            break
        # if tau is above current (already satisfied), skip to lower
        if cur_val <= tau:
            continue
        # Adam state on unconstrained Y
        Y = cur_x.copy()
        M = np.zeros_like(Y)
        V = np.zeros_like(Y)
        tstep = 0
        active = None
        best_tau_x = cur_x.copy()
        best_tau_val = cur_val
        stagn = 0
        # budget per target
        budget = 1400
        patience = 260
        for it in range(budget):
            if time.time() > CONT_END:
                break
            if it % 20 == 0:
                Xn_tmp = Y / (np.linalg.norm(Y, axis=1, keepdims=True) + 1e-300)
                Gtmp = Xn_tmp @ Xn_tmp.T
                np.fill_diagonal(Gtmp, -2.0)
                active = Gtmp > (tau - 0.02)
            # forward
            nrm = np.linalg.norm(Y, axis=1, keepdims=True) + 1e-300
            Xn = Y / nrm
            G = Xn @ Xn.T
            np.fill_diagonal(G, -2.0)
            cur_max = float(np.max(G))
            # track
            if cur_max < best_tau_val - 1e-12:
                best_tau_val = cur_max
                best_tau_x = Xn.copy()
                stagn = 0
                # global improvement?
                if cur_max < gval:
                    gval = cur_max
                    gbest = Xn.copy()
                    if gval < thresh_trigger:
                        feasible_below = True
            else:
                stagn += 1
            if cur_max <= tau:
                # feasibility at prescribed threshold: advance
                cur_x = Xn.copy()
                cur_val = cur_max
                best_tau_x = Xn.copy()
                best_tau_val = cur_max
                stagn = 0
                break
            # hinge weights with margin 0.002, sparse via active set
            # W = 2*relu(G - tau + 0.002) masked
            W = G - tau + 0.002
            np.maximum(W, 0.0, out=W)
            if active is not None:
                W[~active] = 0.0
            np.fill_diagonal(W, 0.0)
            # energy zero check (strict): all zero
            # gradient
            W *= 2.0
            grad_X = W @ Xn
            dot = np.sum(grad_X * Xn, axis=1, keepdims=True)
            grad_Y = (grad_X - dot * Xn) / nrm
            tstep += 1
            M = b1 * M + (1.0 - b1) * grad_Y
            V = b2 * V + (1.0 - b2) * (grad_Y * grad_Y)
            mhat = M / (1.0 - b1 ** tstep)
            vhat = V / (1.0 - b2 ** tstep)
            # slight decay of lr within target
            lr = lr_base * (1.0 - 0.3 * (it / max(budget - 1, 1)))
            Y -= lr * mhat / (np.sqrt(vhat) + eps)
            # keep norms sane
            # (no explicit renorm of Y; Xn handles it)

            if stagn >= patience:
                # kicks restricted to endpoints of violated pairs + largest dual-stress approx
                # violation magnitudes
                Gv = Xn @ Xn.T
                np.fill_diagonal(Gv, -2.0)
                viol = Gv - tau + 0.002
                np.maximum(viol, 0.0, out=viol)
                np.fill_diagonal(viol, 0.0)
                stress = viol.sum(axis=1)  # LP dual stress proxy
                # top violated pairs
                iu0 = np.triu_indices(n, 1)
                flat = Gv[iu0]
                k = min(6, flat.size)
                # argpartition for top-k
                idx = np.argpartition(-flat, k)[:k]
                cand = set()
                for q in idx:
                    if flat[q] > tau - 0.02:
                        cand.add(int(iu0[0][q])); cand.add(int(iu0[1][q]))
                order = np.argsort(-stress)
                for o in order[:8]:
                    cand.add(int(o))
                victims = list(cand)[:12]
                if len(victims) == 0:
                    victims = order[:6].tolist()
                # kick magnitude varies
                sig = 0.045 if (it < budget // 2) else 0.025
                for vv in victims:
                    xv = Xn[vv]
                    r = rng.standard_normal(d)
                    r -= float(np.dot(r, xv)) * xv
                    nn = float(np.linalg.norm(r))
                    if nn < 1e-12 or not np.isfinite(nn):
                        continue
                    r /= nn
                    # apply kick to Y (keep same norm scale ~1)
                    Y[vv] = xv + sig * r
                    M[vv, :] = 0.0
                    V[vv, :] *= 0.25
                stagn = 0
                # do NOT return to incumbent: continue from kicked Y (retry)
        # end iters for this tau
        # update cur to best found for this tau (even if not feasible, keep progress without reverting to old incumbent? use best_tau)
        if best_tau_val < cur_val:
            cur_x = best_tau_x.copy()
            cur_val = best_tau_val
            if cur_val <= tau:
                continue  # go to next lower tau
            else:
                # stagnated without reaching tau: retry same tau once more is handled by next loop?
                # if improved but not feasible, we already spent budget; try next tau only if improved a lot?
                # To follow continuation, break to avoid wasting on impossible lower targets unless we made progress.
                # If progress > 2e-06, attempt same tau again with fresh budget (one extra retry)
                # implement one extra retry by staying on same tau via simple extension
                pass
        # if no feasibility (best_tau_val > tau), stop advancing to lower (harder) targets
        if best_tau_val > tau:
            # if we improved global, we could try again with new kicks rather than going lower
            # do one more round on same tau if time allows and improvement happened
            if best_tau_val < gval + 1e-12 and best_tau_val < cur_val + 1e-12:
                pass
            break
        # else feasible -> loop continues to next tau
        if feasible_below and cur_val < thresh_trigger:
            # will LP later; continue to push lower if time
            pass

    # trust-region LP only after first satisfying below trigger
    if feasible_below or gval < thresh_trigger:
        remain = DEADLINE - time.time() - 1.0
        if remain > 6.0:
            lp_t = min(remain, 42.0)
            try:
                polished = _lp_polish(gbest, lp_t, DEADLINE - 0.5)
                pv = _max_cos(polished)
                if pv < gval:
                    gbest = polished
                    gval = pv
                else:
                    # keep best
                    gbest = gbest
            except Exception:
                pass
    # final safety: if no improvement, return (polished?) incumbent best (gbest is at least x0)
    gbest = _norm(gbest)
    if not np.all(np.isfinite(gbest)):
        gbest = _norm(x0)
    return gbest
