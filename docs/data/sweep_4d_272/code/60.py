import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def _norm(x):
    x = np.asarray(x, dtype=float)
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[~np.isfinite(n)] = 1.0
    n[n < 1e-12] = 1.0
    x = x / n
    x[~np.isfinite(x)] = 0.0
    n2 = np.linalg.norm(x, axis=1, keepdims=True)
    bad = (n2[:, 0] < 1e-12)
    if np.any(bad):
        x[bad] = np.array([1.0, 0.0, 0.0, 0.0])
        x = x / np.linalg.norm(x, axis=1, keepdims=True)
    return x


def _score_tuple(x):
    n = x.shape[0]
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    m = float(G.max())
    iu = np.triu_indices(n, 1)
    c = G[iu]
    cnt = int((c >= m - 1e-6).sum())
    if c.size >= 100:
        part = np.partition(c, c.size - 100)
        s100 = float(part[c.size - 100:].sum())
    else:
        s100 = float(c.sum())
    return (m, cnt, s100)


def _lp_solve(x, radius, b_ub_extra=None, sel_I=None, sel_J=None, sel_c=None, mcur=None):
    n, d = x.shape
    if sel_I is None:
        iu = np.triu_indices(n, 1)
        cfull = (x @ x.T)[iu]
        mcur = float(cfull.max())
        sel = cfull > mcur - max(20.0 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        c = cfull[sel]
    else:
        I, J, c = sel_I, sel_J, sel_c
    p = len(I)
    if p == 0:
        return None, I, J, c, mcur
    nv = n * d + 1
    rows = np.repeat(np.arange(p), 2 * d + 1)
    cols = np.concatenate([I[:, None] * d + np.arange(d),
                           J[:, None] * d + np.arange(d),
                           np.full((p, 1), n * d)], axis=1).ravel()
    vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], axis=1).ravel()
    A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
    A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
    obj = np.zeros(nv)
    obj[-1] = 1.0
    b_ub = -c if b_ub_extra is None else (-c + b_ub_extra)
    try:
        res = linprog(obj, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=np.zeros(n),
                      bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
    except Exception:
        return None, I, J, c, mcur
    return res, I, J, c, mcur


def _extract_duals(res, p):
    if res is None:
        return None
    try:
        if hasattr(res, "ineqlin") and res.ineqlin is not None:
            iq = res.ineqlin
            if isinstance(iq, dict):
                mg = iq.get("marginals", None)
            else:
                mg = getattr(iq, "marginals", None)
            if mg is not None:
                mg = np.asarray(mg, dtype=float).ravel()
                if mg.size == p:
                    return np.abs(mg)
    except Exception:
        pass
    try:
        mg = getattr(res, "marginals", None)
        if mg is not None:
            mg = np.asarray(mg, dtype=float).ravel()
            if mg.size >= p:
                return np.abs(mg[:p])
    except Exception:
        pass
    return None


def _lp_polish(x, seconds, deadline=1e18, exclude_set=None, exclude_iters=0):
    x = _norm(x)
    n, d = x.shape
    best = _score_tuple(x)
    best_x = x.copy()
    radius = 1e-3
    start = time.time()
    it = 0
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and time.time() < deadline and radius > 1e-14:
        cfull = (x @ x.T)[iu]
        mcur = float(cfull.max())
        sel = cfull > mcur - max(20.0 * radius, 1e-9)
        if exclude_set is not None and it < exclude_iters:
            # build mask to drop excluded pairs
            I0, J0 = iu[0][sel], iu[1][sel]
            keep = np.ones(len(I0), dtype=bool)
            for k in range(len(I0)):
                a, b = int(I0[k]), int(J0[k])
                if a > b:
                    a, b = b, a
                if (a, b) in exclude_set:
                    keep[k] = False
            if not np.all(keep):
                # if everything excluded, skip exclusion this iter
                if keep.sum() > 50:
                    sel_idx = np.where(sel)[0][keep]
                    sel2 = np.zeros_like(sel)
                    sel2[sel_idx] = True
                    sel = sel2
        I, J = iu[0][sel], iu[1][sel]
        c = cfull[sel]
        p = len(I)
        if p == 0:
            break
        res, _, _, _, _ = _lp_solve(x, radius, sel_I=I, sel_J=J, sel_c=c)
        if res is None or getattr(res, "status", 1) != 0:
            radius *= 0.5
            it += 1
            continue
        y = x + res.x[:-1].reshape(n, d)
        y = _norm(y)
        sc = _score_tuple(y)
        if sc[0] < best[0] - 1e-13 or (abs(sc[0] - best[0]) <= 1e-13 and sc[1:] <= best[1:]):
            # accept strict improvement or equal-max with no worse counts
            # but require actual max not worse for plain polish
            if sc[0] <= best[0] + 1e-12:
                # for plain polish only accept max improvement; for exclude allow equal
                if exclude_set is None:
                    if sc[0] < best[0] - 1e-13:
                        x, best, best_x = y, sc, y.copy()
                        radius = min(radius * 1.5, 0.05)
                    else:
                        radius *= 0.5
                else:
                    x, best, best_x = y, sc, y.copy()
                    radius = min(radius * 1.5, 0.05)
            else:
                radius *= 0.5
        else:
            # check pure max improvement even if counts worse
            if sc[0] < best[0] - 1e-13:
                x, best, best_x = y, sc, y.copy()
                radius = min(radius * 1.5, 0.05)
            else:
                radius *= 0.5
        it += 1
    return best_x


def _diverse_select(I, J, stress, K, max_per):
    order = np.argsort(-stress)
    n = int(max(I.max(), J.max())) + 1
    cnt = np.zeros(n, dtype=int)
    sel_idx = []
    for o in order:
        if len(sel_idx) >= K:
            break
        i, j = int(I[o]), int(J[o])
        if cnt[i] >= max_per or cnt[j] >= max_per:
            continue
        sel_idx.append(int(o))
        cnt[i] += 1
        cnt[j] += 1
    # relax if not enough
    if len(sel_idx) < K:
        for o in order:
            if len(sel_idx) >= K:
                break
            if int(o) in sel_idx:
                continue
            sel_idx.append(int(o))
    return sel_idx


def solve():
    t0 = time.time()
    deadline = t0 + 172.0
    try:
        x0 = np.load("best.npy")
    except Exception:
        x0 = np.load("record.npy")
    x0 = np.asarray(x0, dtype=float)
    if x0.shape != (272, 4):
        flat = x0.ravel()
        need = 272 * 4
        rng0 = np.random.default_rng(0)
        if flat.size < need:
            flat = np.concatenate([flat, rng0.standard_normal((need - flat.size,))])
        x0 = flat[:need].reshape(272, 4)
    x0 = _norm(x0)

    # initial short polish to reach local optimum
    x_best = _lp_polish(x0, seconds=6, deadline=deadline - 60)
    best_tup = _score_tuple(x_best)

    deltas = [1e-4, 3e-4, 1e-3, 3e-3]
    Klist = [20, 30, 40]
    alphas = [1, 2, 4, 8, 16]
    max_per_list = [3, 4]

    rnd = 0
    # several topology-change rounds
    while time.time() < deadline - 25 and rnd < 10:
        # ranking LP at current best
        radius0 = 2e-3 if (rnd % 2 == 0) else 3e-3
        res0, I0, J0, c0, m0 = _lp_solve(x_best, radius0)
        if res0 is None or getattr(res0, "status", 1) != 0:
            # fallback polish and retry
            x_best = _lp_polish(x_best, seconds=2, deadline=deadline - 20)
            best_tup = _score_tuple(x_best)
            rnd += 1
            continue
        stress = _extract_duals(res0, len(I0))
        if stress is None:
            # fallback: tighter = higher stress
            stress = (c0 - c0.min() + 1e-12)
            # invert: larger c -> larger stress
            stress = c0.copy()
        # collect candidates: for each (K,delta) solve modified LP
        cand_infos = []  # (unpolished_tuple, K, delta, alpha, y, exclude_set, sel_idx)
        max_per = max_per_list[rnd % len(max_per_list)]
        for K in Klist:
            if time.time() > deadline - 25:
                break
            sel_idx = _diverse_select(I0, J0, stress, K, max_per)
            if len(sel_idx) < 10:
                continue
            selmask = np.zeros(len(I0), dtype=bool)
            selmask[sel_idx] = True
            excl = set()
            for o in sel_idx:
                a, b = int(I0[o]), int(J0[o])
                if a > b:
                    a, b = b, a
                excl.add((a, b))
            for delta in deltas:
                if time.time() > deadline - 25:
                    break
                extra = np.zeros(len(I0))
                extra[selmask] = -float(delta)
                resm, _, _, _, _ = _lp_solve(x_best, radius0, b_ub_extra=extra,
                                             sel_I=I0, sel_J=J0, sel_c=c0)
                if resm is None or getattr(resm, "status", 1) != 0:
                    continue
                dx = resm.x[:-1].reshape(x_best.shape)
                if not np.all(np.isfinite(dx)):
                    continue
                dn = np.linalg.norm(dx)
                if dn < 1e-12 or dn > 10:
                    continue
                for a in alphas:
                    y = _norm(x_best + float(a) * dx)
                    tup = _score_tuple(y)
                    cand_infos.append((tup, K, float(delta), int(a), y, excl, sel_idx))
        if not cand_infos:
            rnd += 1
            continue
        # rank by lexicographic tuple, keep top few diverse
        cand_infos.sort(key=lambda e: (e[0][0], e[0][1], e[0][2]))
        # take top 3 with distinct (K,delta) if possible
        picked = []
        seen = set()
        for e in cand_infos:
            key = (e[1], e[2])
            if key in seen and len(picked) >= 2:
                continue
            picked.append(e)
            seen.add(key)
            if len(picked) >= 3:
                break
        improved = False
        for (tup, K, delta, a, y, excl, sidx) in picked:
            if time.time() > deadline - 22:
                break
            # re-polish with selected contacts initially excluded
            yp = _lp_polish(y, seconds=3.0, deadline=deadline - 18,
                            exclude_set=excl, exclude_iters=4)
            # then short normal polish
            yp = _lp_polish(yp, seconds=1.5, deadline=deadline - 16)
            syp = _score_tuple(yp)
            # lexicographic non-worsening acceptance
            if (syp[0] < best_tup[0] - 1e-12) or \
               (abs(syp[0] - best_tup[0]) <= 1e-12 and (syp[1], syp[2]) <= (best_tup[1], best_tup[2])):
                x_best = yp
                best_tup = syp
                improved = True
                break
            # also accept tiny max improvement even if counts worse
            if syp[0] < best_tup[0] - 5e-13:
                x_best = yp
                best_tup = syp
                improved = True
                break
        if not improved:
            # keep searching: if best unpolished candidate is already much worse, do nothing;
            # otherwise try next round with different max_per/radius (loop continues)
            pass
        rnd += 1
        # early exit if clearly record-beating and time low
        if best_tup[0] < 0.898160315150 - 1e-7 and time.time() > deadline - 40:
            break

    # final polish with remaining time
    rem = deadline - time.time() - 2.0
    if rem > 3:
        x_best = _lp_polish(x_best, seconds=min(rem, 20), deadline=deadline - 1)
    return _norm(x_best)
