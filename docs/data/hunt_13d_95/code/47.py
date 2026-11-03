import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

RECORD = 0.244097856927

def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def lp_polish(x, seconds, radius_init=1e-3, t_deadline=None):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = max_cos(x)
    radius = float(radius_init)
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        if t_deadline is not None and time.time() > t_deadline:
            break
        if time.time() - start > seconds - 0.05:
            break
        c = (x @ x.T)[iu]
        m = float(c.max())
        thresh = m - max(20 * radius, 1e-9)
        sel = c > thresh
        if int(sel.sum()) > 1500:
            idx = np.argpartition(c, -1500)[-1500:]
            mask = np.zeros_like(sel, dtype=bool)
            mask[idx] = True
            sel = mask
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        if p == 0:
            break
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv)
        obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c[sel], A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        v = max_cos(y)
        if v < best:
            x, best, radius = y, v, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def get_contacts_duals(x, radius=8e-4, slack_thr=1e-9):
    n, d = x.shape
    iu = np.triu_indices(n, 1)
    cfull = (x @ x.T)[iu]
    m = float(cfull.max())
    sel_all = cfull > m - max(slack_thr, 1e-12)
    I0, J0 = iu[0][sel_all], iu[1][sel_all]
    c0 = cfull[sel_all]
    p = len(I0)
    if p == 0:
        return I0, J0, c0, np.zeros(0), m
    nv = n * d + 1
    rows = np.repeat(np.arange(p), 2 * d + 1)
    cols = np.concatenate([I0[:, None] * d + np.arange(d), J0[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
    vals = np.concatenate([x[J0], x[I0], -np.ones((p, 1))], 1).ravel()
    A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
    A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
    obj = np.zeros(nv); obj[-1] = 1.0
    try:
        res = linprog(obj, A_ub=A_ub, b_ub=-c0, A_eq=A_eq, b_eq=np.zeros(n),
                      bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
    except Exception:
        return I0, J0, c0, np.zeros(p), m
    if res.status != 0:
        return I0, J0, c0, np.zeros(p), m
    duals = None
    try:
        if hasattr(res, "ineqlin") and res.ineqlin is not None:
            mg = getattr(res.ineqlin, "marginals", None)
            if mg is not None:
                duals = np.asarray(mg, dtype=float)
            elif isinstance(res.ineqlin, dict) and "marginals" in res.ineqlin:
                duals = np.asarray(res.ineqlin["marginals"], dtype=float)
        if duals is None and isinstance(res, dict) and "ineqlin" in res:
            v = res["ineqlin"]
            if isinstance(v, dict) and "marginals" in v:
                duals = np.asarray(v["marginals"], dtype=float)
            elif hasattr(v, "marginals"):
                duals = np.asarray(v.marginals, dtype=float)
        if duals is None:
            mg2 = getattr(res, "marginals", None)
            if mg2 is not None:
                duals = np.asarray(mg2, dtype=float).ravel()[:p]
    except Exception:
        duals = None
    if duals is None or duals.shape[0] != p or not np.all(np.isfinite(duals)):
        # fallback: larger slack => lower stress
        slack = m - c0
        duals = -slack
    return I0, J0, c0, duals, m

def adam_block(x, S, targets, steps_per=350, lr=0.004, scale=1e-4, margin=0.025, seed=0):
    n, d = x.shape
    S = np.asarray(sorted(S), dtype=int)
    k = len(S)
    is_var = np.zeros(n, dtype=bool)
    is_var[S] = True
    Yfull = x.copy()
    Yn = x[S].copy()
    Yn /= np.linalg.norm(Yn, axis=1, keepdims=True)
    Yfull[S] = Yn
    b1, b2, eps = 0.9, 0.999, 1e-8
    for ti, t in enumerate(targets):
        lrt = lr * (0.7 ** ti)
        m1 = np.zeros_like(Yn)
        v1 = np.zeros_like(Yn)
        # active mask: pairs with at least one end in S and current cos > t-margin
        # recompute once per target using current Yfull
        G0 = Yfull @ Yfull.T
        iu_r, iu_c = np.triu_indices(n, 1)
        c0 = G0[iu_r, iu_c]
        keep = (c0 > (t - margin)) & (is_var[iu_r] | is_var[iu_c])
        # always keep pairs within S that are close even if slightly lower
        Mk = np.zeros((n, n), dtype=bool)
        Mk[iu_r[keep], iu_c[keep]] = True
        Mk[iu_c[keep], iu_r[keep]] = True
        for s in range(steps_per):
            Yv = Yfull[S]
            G = Yfull @ Yfull.T
            Z = (G - t) / scale
            Zc = np.clip(Z, -30.0, 30.0)
            sig = 1.0 / (1.0 + np.exp(-Zc))
            # for large Z use 1
            sig = np.where(Z > 30.0, 1.0, np.where(Z < -30.0, 0.0, sig))
            coeff = sig / scale
            coeff[~Mk] = 0.0
            np.fill_diagonal(coeff, 0.0)
            Grad_full = coeff @ Yfull
            Gv = Grad_full[S]
            dot = np.sum(Gv * Yv, axis=1, keepdims=True)
            Gt = Gv - dot * Yv
            m1 = b1 * m1 + (1 - b1) * Gt
            v1 = b2 * v1 + (1 - b2) * (Gt * Gt)
            bc1 = 1 - b1 ** (s + 1 + ti * steps_per)
            bc2 = 1 - b2 ** (s + 1 + ti * steps_per)
            mh = m1 / bc1
            vh = v1 / bc2
            step = lrt * mh / (np.sqrt(vh) + eps)
            # limit step norm
            Ynew = Yv - step
            Ynew /= np.linalg.norm(Ynew, axis=1, keepdims=True)
            Yfull[S] = Ynew
    return Yfull / np.linalg.norm(Yfull, axis=1, keepdims=True)

def solve():
    t0 = time.time()
    DEADLINE = t0 + 150.0
    BUDGET_HARD = t0 + 172.0
    rng = np.random.default_rng(0)
    try:
        xr = np.load("record.npy").astype(float)
    except Exception:
        xr = None
    try:
        xb = np.load("best.npy").astype(float)
    except Exception:
        xb = None
    cands = []
    if xb is not None and xb.shape == (95, 13):
        cands.append(xb / np.linalg.norm(xb, axis=1, keepdims=True))
    if xr is not None and xr.shape == (95, 13):
        cands.append(xr / np.linalg.norm(xr, axis=1, keepdims=True))
    if not cands:
        rr = rng.normal(size=(95, 13))
        rr /= np.linalg.norm(rr, axis=1, keepdims=True)
        cands.append(rr)
    vals = [max_cos(c) for c in cands]
    bi = int(np.argmin(vals))
    x_best = cands[bi].copy()
    best_val = float(vals[bi])
    # one LP polish first (as directed)
    try:
        remain = DEADLINE - (time.time() - t0) - 60.0
        ps = float(np.clip(remain, 2.0, 8.0)) if remain > 0 else 2.0
        x_best = lp_polish(x_best, seconds=min(ps, 8.0), radius_init=1e-3, t_deadline=DEADLINE - 55)
        x_best /= np.linalg.norm(x_best, axis=1, keepdims=True)
        best_val = max_cos(x_best)
    except Exception:
        pass
    targets_list = [0.2438, 0.2433, 0.2425]
    it = 0
    # surgery loop
    while time.time() < DEADLINE - 8.0 and time.time() < BUDGET_HARD - 12.0:
        it += 1
        if time.time() > DEADLINE - 6.0:
            break
        try:
            I0, J0, c0, duals, mcur = get_contacts_duals(x_best)
        except Exception:
            break
        p = len(I0)
        if p < 10:
            break
        order = np.argsort(np.abs(duals))  # low stress first
        # choose number of contacts 10-30 deterministically varied
        ncon = int([12, 18, 24, 15, 20, 28, 10, 30][(it - 1) % 8])
        ncon = min(ncon, p)
        # accumulate points 8-20
        Sset = set()
        chosen = []
        for idx in order:
            if len(chosen) >= ncon:
                break
            a, b = int(I0[idx]), int(J0[idx])
            # add if not exploding point count
            newpts = (0 if a in Sset else 1) + (0 if b in Sset else 1)
            if len(Sset) + newpts > 20 and len(chosen) >= 10:
                continue
            chosen.append(idx)
            Sset.add(a); Sset.add(b)
        # ensure 8-20 points: expand/trim
        if len(Sset) < 8:
            for idx in order:
                if len(Sset) >= 10:
                    break
                a, b = int(I0[idx]), int(J0[idx])
                if len(Sset) >= 20:
                    break
                Sset.add(a); Sset.add(b)
                if idx not in chosen:
                    chosen.append(idx)
        if len(Sset) > 20:
            # keep first points by appearance, drop extras by rebuilding
            Slist = sorted(Sset)[:20]
            Sset = set(Slist)
        if len(Sset) < 5 or len(chosen) < 8:
            # fallback random block around rattler/low contact region
            G = x_best @ x_best.T
            np.fill_diagonal(G, -2.0)
            rowmax = G.max(axis=1)
            # pick lowest rowmax points
            idxs = np.argsort(rowmax)[:10]
            Sset = set(int(v) for v in idxs[:12])
        Sarr = np.array(sorted(Sset), dtype=int)
        # small random jitter choice of targets
        jit = (rng.random() - 0.5) * 1e-4
        targets = [t + jit for t in targets_list]
        # occasionally try only first two targets
        if it % 4 == 0:
            targets = targets[:2]
        steps = 300 if time.time() > DEADLINE - 40 else 380
        lr = 0.004 if it % 2 == 1 else 0.006
        try:
            Y = adam_block(x_best, Sarr, targets, steps_per=steps, lr=lr, scale=1e-4, margin=0.025, seed=it)
        except Exception:
            continue
        Y /= np.linalg.norm(Y, axis=1, keepdims=True)
        # quick check: if catastrophically bad skip polish? still polish may recover, but save time
        vm = max_cos(Y)
        if vm > best_val + 0.03:
            continue
        # unfreeze all, LP-polish short
        left = DEADLINE - time.time()
        if left < 4.0:
            break
        psec = min(2.5, max(1.0, left * 0.12))
        try:
            Yp = lp_polish(Y, seconds=psec, radius_init=0.008, t_deadline=DEADLINE - 4.0)
        except Exception:
            continue
        Yp /= np.linalg.norm(Yp, axis=1, keepdims=True)
        v = max_cos(Yp)
        if v < best_val - 1e-12:
            x_best = Yp.copy()
            best_val = v
            # early exit if clearly beating record by margin
            if best_val < RECORD - 5e-4 and time.time() > t0 + 60:
                pass
        # if stuck, occasionally perturb S block randomly to diversify next dual extraction
        if it % 6 == 0 and time.time() < DEADLINE - 20:
            # tiny random nudge of worst point to change topology
            G2 = x_best @ x_best.T
            np.fill_diagonal(G2, -2.0)
            rm = G2.max(axis=1)
            w = int(np.argmax(rm))
            xj = x_best[w] + 0.002 * rng.normal(size=(13,))
            xj /= np.linalg.norm(xj)
            # accept if not too bad after micro polish? just keep for diversity if it doesn't hurt much
            tmp = x_best.copy(); tmp[w] = xj
            if max_cos(tmp) < best_val + 0.002:
                x_best = tmp
                try:
                    x_best = lp_polish(x_best, seconds=1.0, radius_init=0.004, t_deadline=DEADLINE - 4.0)
                    nv2 = max_cos(x_best)
                    if nv2 < best_val:
                        best_val = nv2
                except Exception:
                    pass
    # final polish of best within remaining time
    try:
        left = min(BUDGET_HARD - time.time() - 1.5, 25.0)
        if left > 2.0:
            x_best = lp_polish(x_best, seconds=left * 0.6, radius_init=8e-4)
            l2 = BUDGET_HARD - time.time() - 1.0
            if l2 > 2.0:
                x_best = lp_polish(x_best, seconds=min(l2, 10.0), radius_init=4e-4)
    except Exception:
        pass
    x_best /= np.linalg.norm(x_best, axis=1, keepdims=True)
    # safety fallback to record if NaN or worse and record exists
    try:
        if not np.all(np.isfinite(x_best)):
            raise ValueError("nonfinite")
        if xr is not None:
            xrf = xr / np.linalg.norm(xr, axis=1, keepdims=True)
            if max_cos(x_best) > max_cos(xrf) + 1e-9:
                # still return best found (could still be record-beating vs published?); ensure valid
                # but guarantee we never return worse than loaded best if best file was better
                pass
    except Exception:
        try:
            x_best = (xb if xb is not None else xr)
            x_best = x_best / np.linalg.norm(x_best, axis=1, keepdims=True)
        except Exception:
            pass
    x_best /= np.linalg.norm(x_best, axis=1, keepdims=True)
    return x_best
