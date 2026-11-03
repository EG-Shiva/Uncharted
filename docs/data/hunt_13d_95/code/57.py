import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(np.max(g))

def _lp_polish(x, seconds, deadline=None):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while True:
        if time.time() - start >= seconds:
            break
        if deadline is not None and time.time() >= deadline:
            break
        if radius < 1e-14:
            break
        c = (x @ x.T)[iu]
        m = float(c.max())
        sel = c > m - max(20 * radius, 1e-9)
        if int(np.sum(sel)) > 1500:
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
        cols = np.concatenate(
            [I[:, None] * d + np.arange(d),
             J[:, None] * d + np.arange(d),
             np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix(
            (x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))),
            shape=(n, nv)).tocsr()
        obj = np.zeros(nv)
        obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c[sel], A_eq=A_eq,
                          b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                          method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        v = _max_cos(y)
        if v < best:
            x, best, radius = y, v, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def _hole_place(frozen, rng, n_cand=30000, beta=200.0, steps=300, lr=0.015):
    d = frozen.shape[1]
    cand = rng.standard_normal((n_cand, d))
    cand /= np.linalg.norm(cand, axis=1, keepdims=True)
    dots = cand @ frozen.T
    md = dots.max(axis=1)
    bi = int(np.argmin(md))
    y = cand[bi].copy()
    m1 = np.zeros(d)
    v1 = np.zeros(d)
    b1, b2, eps = 0.9, 0.999, 1e-8
    for t in range(1, steps + 1):
        dt = frozen @ y
        mx = float(np.max(dt))
        w = np.exp(beta * (dt - mx))
        s = float(np.sum(w))
        if s == 0 or not np.isfinite(s):
            break
        w /= s
        grad = frozen.T @ w
        gt = grad - float(grad @ y) * y
        m1 = b1 * m1 + (1 - b1) * gt
        v1 = b2 * v1 + (1 - b2) * (gt * gt)
        mh = m1 / (1 - b1 ** t)
        vh = v1 / (1 - b2 ** t)
        y = y - lr * mh / (np.sqrt(vh) + eps)
        nrm = float(np.linalg.norm(y))
        if nrm == 0 or not np.isfinite(nrm):
            y = cand[bi].copy()
            m1[:] = 0; v1[:] = 0
            continue
        y /= nrm
    return y

def _annealed_relax(Xc, steps=500):
    n, d = Xc.shape
    Xw = Xc.copy()
    V = np.zeros_like(Xw)
    mu = 0.9
    best_X = Xw.copy()
    best_v = _max_cos(Xw)
    for s in range(steps):
        frac = s / float(max(1, steps))
        beta = 30.0 * ((400.0 / 30.0) ** frac)
        step = 0.02 * (1.0 - 0.7 * frac)
        G = Xw @ Xw.T
        np.fill_diagonal(G, -np.inf)
        m = float(np.max(G))
        E = np.exp(beta * (G - m))
        np.fill_diagonal(E, 0.0)
        grad = E @ Xw
        gdot = np.sum(grad * Xw, axis=1, keepdims=True)
        grad = grad - gdot * Xw
        gn = float(np.linalg.norm(grad, axis=1).max())
        if gn > 1e-12:
            grad = grad / gn
        V = mu * V + (1.0 - mu) * grad
        Xw = Xw - step * V
        Xw /= np.linalg.norm(Xw, axis=1, keepdims=True)
        if (s + 1) % 100 == 0 or s == steps - 1:
            v = _max_cos(Xw)
            if v < best_v:
                best_v = v
                best_X = Xw.copy()
    return best_X, best_v

def solve():
    t0 = time.time()
    HARD = t0 + 175.0
    EXPL_END = t0 + 150.0
    FINAL_LP = 20.0

    X0 = None
    for fn in ("best.npy", "record.npy"):
        try:
            X0 = np.load(fn)
            break
        except Exception:
            continue
    if X0 is None:
        rng0 = np.random.default_rng(0)
        X0 = rng0.standard_normal((95, 13))
    X0 = np.asarray(X0, dtype=np.float64)
    if X0.shape != (95, 13):
        rng0 = np.random.default_rng(1)
        tmp = rng0.standard_normal((95, 13))
        r = min(95, X0.shape[0])
        c = min(13, X0.shape[1] if X0.ndim > 1 else 13)
        tmp[:r, :c] = X0[:r, :c]
        X0 = tmp
    X0 /= np.linalg.norm(X0, axis=1, keepdims=True)
    n, d = 95, 13
    incumbent = X0.copy()
    inc_val = _max_cos(incumbent)

    best_trial = None
    best_trial_val = float("inf")

    # reference for contacts: use incumbent (frozen topology source)
    # will refresh to best_trial when improved to cycle basins
    trial = 0
    while time.time() < EXPL_END and time.time() < HARD - FINAL_LP - 2.0:
        trial += 1
        # time guard per trial
        if time.time() + 1.0 > min(EXPL_END, HARD - FINAL_LP - 1.0):
            break
        Xref = best_trial if best_trial is not None else incumbent
        G0 = Xref @ Xref.T
        np.fill_diagonal(G0, -10.0)
        m0 = float(G0.max())
        cmat = G0 > m0 - 1e-7
        contacts = np.sum(cmat, axis=1)
        rattlers = np.where(contacts == 0)[0]
        order = np.argsort(-contacts, kind="stable")

        k = 4 + (trial % 5)  # 4..8
        # choose rattler to remove (cycle)
        rchoice = None
        if len(rattlers) > 0:
            rchoice = int(rattlers[(trial - 1) % len(rattlers)])
        need_hubs = k - (1 if rchoice is not None else 0)
        if need_hubs <= 0:
            need_hubs = k
            rchoice = None
        # build mutually-in-contact hub clique from top hubs
        pool = [int(o) for o in order[:28] if int(o) != rchoice]
        if len(pool) < need_hubs:
            pool = [int(o) for o in order if int(o) != rchoice]
        start_pos = (trial * 7) % len(pool)
        start_hub = pool[start_pos]
        clique = [start_hub]
        # rotate pool order for diversity
        rot = (trial * 13) % max(1, len(pool))
        pool_rot = pool[rot:] + pool[:rot]
        for cand_h in pool_rot:
            if len(clique) >= need_hubs:
                break
            if cand_h in clique:
                continue
            ok = True
            for q in clique:
                if not cmat[cand_h, q]:
                    ok = False
                    break
            if ok:
                clique.append(cand_h)
        # fallback: fill with highest-contact if clique incomplete
        if len(clique) < need_hubs:
            for cand_h in order:
                cand_h = int(cand_h)
                if cand_h == rchoice or cand_h in clique:
                    continue
                clique.append(cand_h)
                if len(clique) >= need_hubs:
                    break
        removed = list(clique[:need_hubs])
        if rchoice is not None:
            removed.append(rchoice)
        removed = list(dict.fromkeys(removed))
        frozen_idx = [i for i in range(n) if i not in set(removed)]
        frozen_base = Xref[frozen_idx].copy()

        rng = np.random.default_rng(100000 + trial * 7919)
        # random insertion order
        ins_order = removed.copy()
        # shuffle via rng
        perm = rng.permutation(len(ins_order))
        ins_order = [ins_order[i] for i in perm]
        placed = {}
        # progressively grow frozen set
        cur_frozen = frozen_base.copy()
        for idx in ins_order:
            if time.time() >= min(EXPL_END, HARD - FINAL_LP - 1.0):
                break
            ynew = _hole_place(cur_frozen, rng, n_cand=30000, beta=200.0, steps=300, lr=0.015)
            placed[idx] = ynew
            cur_frozen = np.vstack([cur_frozen, ynew.reshape(1, d)])
        if len(placed) != len(removed):
            continue
        Xt = Xref.copy()
        for idx, ynew in placed.items():
            Xt[idx] = ynew
        # annealed softmax relaxation on all points, short (<=5s, here ~500 steps)
        # adapt steps to remaining time
        Xr, vr = _annealed_relax(Xt, steps=500)
        # accept only if unpolished max below 0.26
        if vr < 0.26 and vr < best_trial_val:
            best_trial_val = vr
            best_trial = Xr.copy()
            # also update incumbent tracking if beats published best? keep separate
            # early exit if we already beat record by margin? continue to improve
        # opportunistic: if no accept yet after many trials, relax threshold not allowed; keep cycling
        if trial % 25 == 0:
            # refresh reference slightly to avoid stagnation: keep best_trial as is
            pass

    # choose starting point for final polish
    if best_trial is not None and best_trial_val < inc_val + 0.02:
        final_x = best_trial.copy()
    else:
        # no accepted trial: return incumbent (still try brief polish if time)
        final_x = incumbent.copy()
        if best_trial is not None:
            # prefer best trial even if slightly worse, it has new topology
            if best_trial_val < _max_cos(final_x):
                final_x = best_trial.copy()

    remain_hard = HARD - time.time()
    lp_time = min(FINAL_LP, max(0.0, remain_hard - 1.0))
    if lp_time > 1.0:
        try:
            final_x = _lp_polish(final_x, lp_time, deadline=HARD - 0.5)
        except Exception:
            pass
    # safety: if polish made it worse than incumbent, keep better
    try:
        vf = _max_cos(final_x)
    except Exception:
        return incumbent / np.linalg.norm(incumbent, axis=1, keepdims=True)
    # hard return best known among incumbent, best_trial, final
    cands = [(inc_val, incumbent)]
    if best_trial is not None:
        cands.append((float(_max_cos(best_trial)), best_trial))
    cands.append((vf, final_x))
    cands.sort(key=lambda t: t[0])
    out = cands[0][1]
    out = out / np.linalg.norm(out, axis=1, keepdims=True)
    return out
