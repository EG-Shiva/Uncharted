import time
import numpy as np

def _norm(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return X / n

def _maxcos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _core_adam(X, steps, b0=50.0, b1=600.0, lr0=0.02, lr1=0.003):
    n, d = X.shape
    Y = X.copy()
    m = np.zeros_like(Y)
    vv = np.zeros_like(Y)
    b1m, b2m, eps = 0.9, 0.999, 1e-8
    p1 = 1.0
    p2 = 1.0
    lb = np.log(b1 / b0)
    ll = np.log(lr1 / lr0)
    for s in range(steps):
        f = s / max(1, steps - 1)
        beta = b0 * np.exp(lb * f)
        lr = lr0 * np.exp(ll * f)
        G = Y @ Y.T
        np.fill_diagonal(G, -10.0)
        gm = float(G.max())
        E = np.exp(beta * (G - gm))
        np.fill_diagonal(E, 0.0)
        S = float(E.sum())
        if not np.isfinite(S) or S <= 0:
            break
        Gr = 2.0 * (E @ Y) / S
        Gt = Gr - np.sum(Gr * Y, axis=1, keepdims=True) * Y
        p1 *= b1m
        p2 *= b2m
        m *= b1m
        m += (1 - b1m) * Gt
        vv *= b2m
        vv += (1 - b2m) * (Gt * Gt)
        mh = m / (1 - p1)
        vh = vv / (1 - p2)
        Y = Y - lr * mh / (np.sqrt(vh) + eps)
        Y = _norm(Y)
    return Y

def _refine_hole(v, Y, iters=100, T=0.005, lr=0.25):
    v = v / (np.linalg.norm(v) + 1e-12)
    for k in range(iters):
        f = k / max(1, iters - 1)
        Tk = T * (0.4 ** f)
        lrk = lr * (0.5 ** f)
        s = Y @ v
        mm = float(s.max())
        w = np.exp((s - mm) / Tk)
        sw = float(w.sum())
        if not np.isfinite(sw) or sw <= 0:
            break
        w /= sw
        g = w @ Y
        g = g - float(np.dot(g, v)) * v
        v = v - lrk * g
        nv = float(np.linalg.norm(v))
        if not np.isfinite(nv) or nv == 0:
            break
        v = v / nv
    return v

def _local_adam(Cur, sub_idx, steps=350, beta=350.0, lr=0.012):
    # Cur (m,d) normalized; optimize rows sub_idx, freeze rest
    m_, d_ = Cur.shape
    Y = Cur.copy()
    sub = np.asarray(sub_idx, dtype=int)
    if len(sub) == 0:
        return Y
    mm = np.zeros((len(sub), d_))
    vv = np.zeros((len(sub), d_))
    b1m, b2m, eps = 0.9, 0.999, 1e-8
    p1 = 1.0
    p2 = 1.0
    for s in range(steps):
        G = Y @ Y.T
        np.fill_diagonal(G, -10.0)
        gm = float(G.max())
        E = np.exp(beta * (G - gm))
        np.fill_diagonal(E, 0.0)
        S = float(E.sum())
        if not np.isfinite(S) or S <= 0:
            break
        Gr = 2.0 * (E @ Y) / S
        Gt = Gr - np.sum(Gr * Y, axis=1, keepdims=True) * Y
        Gs = Gt[sub]
        p1 *= b1m
        p2 *= b2m
        mm *= b1m
        mm += (1 - b1m) * Gs
        vv *= b2m
        vv += (1 - b2m) * (Gs * Gs)
        mh = mm / (1 - p1)
        vh = vv / (1 - p2)
        Y[sub] = Y[sub] - lr * mh / (np.sqrt(vh) + eps)
        Y[sub] = _norm(Y[sub])
    return _norm(Y)

def _lp_polish(x, t_end, max_constraints=1500):
    try:
        from scipy.optimize import linprog
        from scipy.sparse import coo_matrix
    except Exception:
        return x
    x = _norm(x)
    n, d = x.shape
    try:
        iu = np.triu_indices(n, 1)
    except Exception:
        return x
    try:
        best = _maxcos(x)
    except Exception:
        return x
    radius = 8e-4
    while time.time() < t_end - 0.5 and radius > 1e-14:
        try:
            c = (x @ x.T)[iu]
        except Exception:
            break
        mcur = float(c.max())
        thr = max(20 * radius, 1e-9)
        sel = c > mcur - thr
        cnt = int(sel.sum())
        if cnt == 0:
            break
        if cnt > max_constraints:
            idx = np.argpartition(-c, max_constraints)[:max_constraints]
            mask = np.zeros_like(sel)
            mask[idx] = True
            sel = mask
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        nv = n * d + 1
        try:
            rows = np.repeat(np.arange(p), 2 * d + 1)
            cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
            vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv)
            obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-c[sel], A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            break
        if getattr(res, "status", 1) != 0:
            radius *= 0.5
            continue
        try:
            y = x + np.asarray(res.x[:-1]).reshape(n, d)
        except Exception:
            radius *= 0.5
            continue
        y = _norm(y)
        if not np.all(np.isfinite(y)):
            radius *= 0.5
            continue
        v = _maxcos(y)
        if v < best:
            x, best, radius = y, v, min(radius * 1.5, 0.02)
        else:
            radius *= 0.5
    return x

def solve():
    t0 = time.time()
    deadline = t0 + 150.0
    hard_end = t0 + 168.0
    rng = np.random.default_rng(7)

    best_global = None
    bv_global = 1e9
    record_val = 0.168263492524
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn).astype(float)
            if a.shape == (76, 15):
                a = _norm(a)
                if np.all(np.isfinite(a)):
                    v = _maxcos(a)
                    if v < bv_global:
                        bv_global = v
                        best_global = a.copy()
                    if fn == "record.npy":
                        record_val = float(v)
        except Exception:
            pass
    if best_global is None:
        best_global = _norm(rng.normal(size=(76, 15)))
        bv_global = _maxcos(best_global)

    # identify 11 rattlers: points with smallest per-point max
    try:
        G0 = best_global @ best_global.T
        np.fill_diagonal(G0, -2.0)
        rowmax = G0.max(axis=1)
        order = np.argsort(rowmax)
        ratt = set(order[:11].tolist())
        keep = [i for i in range(76) if i not in ratt]
        core = best_global[keep].copy()
    except Exception:
        core = best_global[:65].copy()
    core = _norm(core)
    # core anneal beta 50->600, 3000 steps (split to respect deadline)
    steps_done = 0
    total_steps = 3000
    # do in chunks to allow time check
    while steps_done < total_steps and time.time() < deadline - 40:
        chunk = min(750, total_steps - steps_done)
        # approximate continuation of schedule by running sub-anneal
        # map chunk position to global frac for beta/lr: run _core_adam on current with narrowed range
        f0 = steps_done / total_steps
        f1 = (steps_done + chunk) / total_steps
        b0 = 50.0 * (600.0 / 50.0) ** f0
        b1 = 50.0 * (600.0 / 50.0) ** f1
        lr0 = 0.02 * (0.003 / 0.02) ** f0
        lr1 = 0.02 * (0.003 / 0.02) ** f1
        core = _core_adam(core, chunk, b0=b0, b1=b1, lr0=lr0, lr1=lr1)
        steps_done += chunk

    try:
        core_opt = _maxcos(core)
    except Exception:
        core_opt = bv_global
    # threshold schedule: start at core optimum, slowly raised to record-eps
    target = min(bv_global - 1e-9, 0.168263492524 - 1e-9)
    # if core_opt already above target, just aim slightly below core_opt path
    if not np.isfinite(target):
        target = 0.168263492524 - 1e-9
    if core_opt >= target:
        target = core_opt + 0.004

    cur = core.copy()  # (65,15)
    n_need = 76 - cur.shape[0]
    for k in range(n_need):
        if time.time() > deadline - 8:
            break
        frac_thr = (k + 1) / n_need
        thr = core_opt + (target - core_opt) * frac_thr
        best_try = None
        best_try_v = 1e9
        # up to 2 attempts, keep best; first uses 20000 candidates
        for att in range(2):
            if time.time() > deadline - 6:
                break
            NS = 20000 if att == 0 else 8000
            try:
                C = _norm(rng.normal(size=(NS, 15)))
                scores = (C @ cur.T).max(axis=1)
                bi = int(np.argmin(scores))
                v0 = C[bi]
                v0 = _refine_hole(v0, cur, n_iter=100, T=0.005, lr=0.25)
                cand = np.vstack([cur, v0[None, :]])
                cand = _norm(cand)
                # local re-opt: inserted + neighbours >0.05
                try:
                    s = (cand[:-1] @ cand[-1])
                    nb = np.where(s > 0.05)[0].tolist()
                except Exception:
                    nb = []
                sub_idx = nb + [cand.shape[0] - 1]
                # dedupe
                sub_idx = sorted(set(sub_idx))
                # cap subset size for speed
                if len(sub_idx) > 20:
                    # keep inserted + top 19 neighbours
                    try:
                        ss = (cand[:-1] @ cand[-1])
                        o = np.argsort(-ss)[:19].tolist()
                        sub_idx = sorted(set(o + [cand.shape[0] - 1]))
                    except Exception:
                        sub_idx = sub_idx[:20]
                if time.time() < deadline - 12:
                    cand2 = _local_adam(cand, sub_idx, steps=350, beta=350.0, lr=0.012)
                else:
                    cand2 = cand
                # capped LP polish at most 1500 constraints, 10s
                t_end = min(time.time() + 10.0, deadline)
                remain = t_end - time.time()
                if remain > 1.2 and cand2.shape[0] >= 66:
                    try:
                        cand2 = _lp_polish(cand2, t_end, max_constraints=1500)
                    except Exception:
                        pass
                v = _maxcos(cand2)
                if v < best_try_v:
                    best_try_v = v
                    best_try = cand2.copy()
                # accept immediately if below threshold
                if v <= thr:
                    break
            except Exception:
                continue
        if best_try is None:
            # quick random fill
            try:
                v0 = _norm(rng.normal(size=(1, 15)))[0]
                best_try = _norm(np.vstack([cur, v0[None, :]]))
                best_try_v = _maxcos(best_try)
            except Exception:
                break
        # accept if below slowly-raised threshold, else accept best anyway to progress
        # (threshold enforces discipline but we must complete the set)
        cur = best_try
        # update core_opt reference? keep original; optionally track cur max
        # small time guard
        if time.time() > hard_end - 8:
            break

    # pad if incomplete (deadline hit)
    if cur.shape[0] < 76:
        try:
            miss = 76 - cur.shape[0]
            C = _norm(rng.normal(size=(miss * 500, 15)))
            for _ in range(miss):
                sc = (C @ cur.T).max(axis=1)
                bi = int(np.argmin(sc))
                cur = np.vstack([cur, C[bi][None, :]])
                # remove used
                # keep C as is
            cur = _norm(cur)
        except Exception:
            pass
    # ensure exactly 76
    try:
        if cur.shape[0] > 76:
            cur = cur[:76]
        cur = _norm(cur)
        cur_v = _maxcos(cur)
    except Exception:
        return _norm(best_global)
    # return best complete 76-point set found, falling back to best.npy
    if cur.shape == (76, 15) and np.all(np.isfinite(cur)) and cur_v < bv_global:
        out = _norm(cur)
    else:
        # if cur is complete but worse, still return best_global (fallback)
        # to guarantee no regression; spec wants best complete set, fallback to best.npy
        if cur.shape == (76, 15) and np.all(np.isfinite(cur)):
            # if we are within tiny margin of best, return cur? prefer best_global for scoring
            out = _norm(best_global)
            # if cur beats record but not best_global (rare), keep cur
            # already handled; so return best_global
        else:
            out = _norm(best_global)
    if not np.all(np.isfinite(out)) or out.shape != (76, 15):
        out = _norm(best_global)
    return _norm(out)
