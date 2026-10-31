import time
import numpy as np


def _norm_rows(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[~np.isfinite(n)] = 1.0
    n[n == 0] = 1.0
    return x / n


def _max_cos(x):
    try:
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        return float(np.max(G))
    except Exception:
        return 1.0


def _loads_and_max(x, tau=1e-4):
    # returns m, per-point load, total load
    try:
        G = x @ x.T
        np.fill_diagonal(G, -np.inf)
        m = float(np.max(G))
        # exp((G-m)/tau), diagonal -> 0
        with np.errstate(over='ignore', under='ignore'):
            W = np.exp((G - m) / tau)
        # fix diagonal / non-finite
        np.fill_diagonal(W, 0.0)
        W[~np.isfinite(W)] = 0.0
        # clip: W in [0,1] for valid entries
        per = np.sum(W, axis=1)
        tot = float(np.sum(W) * 0.5)
        return m, per, tot, G
    except Exception:
        n = x.shape[0]
        return 1.0, np.ones(n), 1e18, None


def _degree_hash(x, m=None, tol=5e-7):
    try:
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        if m is None:
            m = float(np.max(G))
        deg = np.sum(G > (m - tol), axis=1)
        return tuple(sorted(deg.tolist())), round(float(m) * 1e7)
    except Exception:
        return tuple(), 0


def _smooth_opt(x, steps, tau0=0.02, tau1=0.001, lr0=0.08, lr1=0.008, mom=0.85):
    try:
        n, d = x.shape
        vel = np.zeros_like(x)
        lt0, lt1 = np.log(tau0), np.log(tau1)
        ll0, ll1 = np.log(lr0), np.log(lr1)
        for t in range(int(steps)):
            f = t / max(1, int(steps) - 1)
            tau = float(np.exp(lt0 + (lt1 - lt0) * f))
            lr = float(np.exp(ll0 + (ll1 - ll0) * f))
            G = x @ x.T
            np.fill_diagonal(G, -np.inf)
            m = float(np.max(G))
            if not np.isfinite(m):
                break
            with np.errstate(over='ignore', under='ignore'):
                W = np.exp((G - m) / tau)
            np.fill_diagonal(W, 0.0)
            W[~np.isfinite(W)] = 0.0
            row = np.sum(W, axis=1, keepdims=True)
            GX = W @ x
            grad = np.zeros_like(x)
            mask = (row[:, 0] > 1e-12)
            grad[mask] = GX[mask] / row[mask]
            dot = np.sum(grad * x, axis=1, keepdims=True)
            grad -= dot * x
            # clip
            gn = np.linalg.norm(grad, axis=1, keepdims=True)
            over = (gn[:, 0] > 3.0)
            if np.any(over):
                grad[over] *= (3.0 / np.maximum(gn[over], 1e-12))
            vel = mom * vel + (1.0 - mom) * grad
            bc = 1.0 - mom ** (t + 1)
            step = vel / max(bc, 1e-12)
            x = x - lr * step
            x = _norm_rows(x)
            if not np.all(np.isfinite(x)):
                break
        return _norm_rows(x)
    except Exception:
        try:
            return _norm_rows(x)
        except Exception:
            return x


def _lp_polish(x, deadline, cap=2500, maxit=200, r0=1e-3):
    try:
        from scipy.optimize import linprog
        from scipy.sparse import coo_matrix
    except Exception:
        return x, _max_cos(x)
    try:
        x = _norm_rows(np.asarray(x, dtype=float))
        n, d = x.shape
        best = _max_cos(x)
        best_x = x.copy()
        radius = float(r0)
        iu0 = np.triu_indices(n, 1)
        it = 0
        while time.time() < deadline and radius > 1e-14 and it < maxit:
            it += 1
            try:
                G = x @ x.T
                c = G[iu0]
                m = float(c.max())
                if not np.isfinite(m):
                    break
                thresh = m - max(20.0 * radius, 1e-9)
                idx = np.where(c > thresh)[0]
                if idx.size == 0:
                    radius = min(radius * 1.5, 0.05)
                    continue
                if idx.size > cap:
                    try:
                        o = np.argpartition(c[idx], -cap)[-cap:]
                        idx = idx[o]
                    except Exception:
                        idx = idx[:cap]
                I = iu0[0][idx]
                J = iu0[1][idx]
                csel = c[idx]
                p = idx.size
                nv = n * d + 1
                rows = np.repeat(np.arange(p), 2 * d + 1)
                cI = I[:, None] * d + np.arange(d)
                cJ = J[:, None] * d + np.arange(d)
                cols = np.concatenate([cI, cJ, np.full((p, 1), n * d)], axis=1).ravel()
                vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], axis=1).ravel()
                A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
                A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
                obj = np.zeros(nv)
                obj[-1] = 1.0
                res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq, b_eq=np.zeros(n),
                              bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
            except Exception:
                try:
                    radius *= 0.5
                except Exception:
                    break
                continue
            try:
                if res.status != 0:
                    radius *= 0.5
                    continue
                y = x + res.x[:-1].reshape(n, d)
                y = _norm_rows(y)
                if not np.all(np.isfinite(y)):
                    radius *= 0.5
                    continue
                v = _max_cos(y)
                if v < best:
                    x = y
                    best = v
                    best_x = y.copy()
                    radius = min(radius * 1.5, 0.05)
                else:
                    radius *= 0.5
            except Exception:
                try:
                    radius *= 0.5
                except Exception:
                    break
                continue
        return best_x, best
    except Exception:
        try:
            return x, _max_cos(x)
        except Exception:
            return x, 1.0


def _hole_points(x_inc, k, rng, n_cand=20000, sep=0.80):
    try:
        n, d = x_inc.shape
        cand = rng.standard_normal(size=(n_cand, d))
        cand = _norm_rows(cand)
        dots = cand @ x_inc.T
        md = np.max(dots, axis=1)
        order = np.argsort(md)
        chosen = []
        chosen_idx = []
        for idx in order:
            if len(chosen) >= k:
                break
            c = cand[idx]
            ok = True
            for cc in chosen:
                if float(c @ cc) > sep:
                    ok = False
                    break
            if ok:
                chosen.append(c)
                chosen_idx.append(int(idx))
            if len(chosen) >= k:
                break
        # fallback: just take lowest md if separation too strict
        if len(chosen) < k:
            for idx in order:
                if int(idx) in chosen_idx:
                    continue
                chosen.append(cand[idx])
                if len(chosen) >= k:
                    break
        return np.array(chosen[:k], dtype=float)
    except Exception:
        try:
            h = rng.standard_normal(size=(k, x_inc.shape[1]))
            return _norm_rows(h)
        except Exception:
            return np.zeros((k, x_inc.shape[1]))


def solve():
    t0 = time.time()
    HARD_END = t0 + 174.0
    BRANCH_DEADLINE = t0 + 160.0
    n, d = 153, 4
    rng = np.random.default_rng(0)

    # load incumbent
    best_x = None
    best_v = 1e9
    for fname in ("best.npy", "record.npy"):
        try:
            a = np.load(fname)
            a = np.asarray(a, dtype=float)
            if a.shape == (n, d) and np.all(np.isfinite(a)):
                a = _norm_rows(a)
                v = _max_cos(a)
                if v < best_v:
                    best_v = v
                    best_x = a.copy()
        except Exception:
            continue
    if best_x is None:
        try:
            best_x = _norm_rows(rng.standard_normal(size=(n, d)))
        except Exception:
            best_x = np.ones((n, d)) / 2.0
        best_v = _max_cos(best_x)
    global_x = best_x.copy()
    global_v = float(best_v)

    leaves = []  # list of (153,4) arrays

    extras = [6, 9, 12]
    beam_width = 3
    tau_load = 1e-4

    for extra in extras:
        if time.time() > BRANCH_DEADLINE:
            break
        if time.time() > HARD_END - 20.0:
            break
        try:
            # hole insertion
            holes = _hole_points(global_x if global_x.shape[0] == n else best_x, int(extra), rng)
            xN = np.vstack([best_x, holes])
            xN = _norm_rows(xN)
            # brief smooth minimax
            # smaller steps for larger N to save time
            steps = 600 if xN.shape[0] <= 162 else 450
            # reduce steps if time is short
            tleft = HARD_END - time.time()
            if tleft < 60:
                steps = min(steps, 300)
            if tleft < 30:
                steps = min(steps, 150)
            try:
                xN = _smooth_opt(xN, steps)
            except Exception:
                pass
            # one short LP polish
            try:
                dl = min(time.time() + 3.0, HARD_END - 15.0, BRANCH_DEADLINE)
                if dl - time.time() > 0.5:
                    xN, _ = _lp_polish(xN, dl, cap=2000, maxit=60, r0=8e-4)
            except Exception:
                pass

            # beam prune down to 153
            beam = [xN]
            while True:
                try:
                    curN = beam[0].shape[0]
                except Exception:
                    break
                if curN <= 153:
                    break
                if time.time() > BRANCH_DEADLINE:
                    break
                # generate children
                pool = []  # (score_max, score_tot, child)
                for xb in beam:
                    try:
                        cn = xb.shape[0]
                        m, per, tot, G = _loads_and_max(xb, tau=tau_load)
                        # candidates with largest per-point load
                        D = min(4, cn)
                        try:
                            order = np.argsort(-per)
                        except Exception:
                            order = np.arange(cn)
                        cands = order[:D]
                        for del_idx in cands:
                            try:
                                child = np.delete(xb, int(del_idx), axis=0)
                                cm, cper, ctot, _ = _loads_and_max(child, tau=tau_load)
                                pool.append((float(cm), float(ctot), child))
                            except Exception:
                                continue
                    except Exception:
                        continue
                if not pool:
                    break
                # sort by (max, tot)
                try:
                    pool.sort(key=lambda t: (t[0], t[1]))
                except Exception:
                    pass
                # keep beam_width with diverse degree hash
                new_beam = []
                seen = set()
                for cm, ctot, child in pool:
                    try:
                        dh, _ = _degree_hash(child)
                    except Exception:
                        dh = tuple()
                    if dh in seen:
                        continue
                    seen.add(dh)
                    new_beam.append(child)
                    if len(new_beam) >= beam_width:
                        break
                # if too few due to dedup, fill from pool
                if len(new_beam) < beam_width:
                    for cm, ctot, child in pool:
                        if len(new_beam) >= beam_width:
                            break
                        # avoid exact duplicate object
                        dup = False
                        for b in new_beam:
                            if b is child:
                                dup = True
                                break
                        if not dup:
                            # allow duplicate hash now
                            new_beam.append(child)
                            if len(new_beam) >= beam_width:
                                break
                        if len(new_beam) >= beam_width:
                            break
                if not new_beam:
                    break
                beam = new_beam
            # collect leaves that reached 153
            for b in beam:
                try:
                    if b.shape == (153, 4) and np.all(np.isfinite(b)):
                        leaves.append(_norm_rows(b))
                except Exception:
                    continue
        except Exception:
            continue

    # deduplicate leaves by degree hash
    uniq = []
    seen_hash = set()
    for lf in leaves:
        try:
            m = _max_cos(lf)
            dh, rh = _degree_hash(lf, m=m)
            key = (dh, rh)
            if key in seen_hash:
                continue
            seen_hash.add(key)
            uniq.append(lf)
        except Exception:
            continue
    # cap leaves to at most 9 to fit time
    if len(uniq) > 9:
        try:
            scored = [( _max_cos(lf), lf) for lf in uniq]
            scored.sort(key=lambda t: t[0])
            uniq = [lf for _, lf in scored[:9]]
        except Exception:
            uniq = uniq[:9]
    leaves = uniq

    # LP-polish each beam leaf
    polished = []
    for li, lf in enumerate(leaves):
        if time.time() > HARD_END - 3.0:
            break
        if time.time() > BRANCH_DEADLINE + 12.0 and li > 0:
            # allow a bit beyond branch deadline for polishing but stop if near hard end
            if time.time() > HARD_END - 15.0:
                break
        try:
            remain = max(1.0, HARD_END - 2.0 - time.time())
            left_n = max(1, len(leaves) - li)
            per_leaf = min(10.0, remain / left_n)
            if per_leaf < 0.8:
                # just score without polish
                polished.append((_max_cos(lf), lf))
                continue
            dl = time.time() + per_leaf
            # also wrap in try/except
            try:
                pf, pv = _lp_polish(lf, dl, cap=2500, maxit=200, r0=1e-3)
            except Exception:
                pf, pv = lf, _max_cos(lf)
            try:
                pv2 = _max_cos(pf)
            except Exception:
                pf, pv2 = lf, _max_cos(lf)
            polished.append((float(pv2), _norm_rows(pf)))
        except Exception:
            continue

    # pick best among polished, leaves, incumbent
    candidates = []
    try:
        candidates.append((float(global_v), global_x))
    except Exception:
        pass
    for lf in leaves:
        try:
            candidates.append((float(_max_cos(lf)), lf))
        except Exception:
            continue
    for pv, pf in polished:
        try:
            candidates.append((float(pv), pf))
        except Exception:
            continue
    # also consider incumbent best_x
    try:
        candidates.append((float(_max_cos(best_x)), best_x))
    except Exception:
        pass

    final_x = global_x
    final_v = float(global_v)
    try:
        # filter finite
        filt = []
        for v, x in candidates:
            try:
                if np.isfinite(v) and isinstance(x, np.ndarray) and x.shape == (n, d) and np.all(np.isfinite(x)):
                    # check rows nonzero
                    nr = np.linalg.norm(x, axis=1)
                    if np.all(nr > 1e-8):
                        filt.append((v, x))
            except Exception:
                continue
        if filt:
            filt.sort(key=lambda t: t[0])
            final_v, final_x = filt[0][0], filt[0][1]
    except Exception:
        pass

    try:
        final_x = _norm_rows(np.asarray(final_x, dtype=float))
        # final safety: if not finite or wrong shape, return incumbent
        if final_x.shape != (n, d) or not np.all(np.isfinite(final_x)):
            final_x = _norm_rows(best_x)
    except Exception:
        try:
            final_x = _norm_rows(best_x)
        except Exception:
            final_x = np.ones((n, d)) / 2.0

    return final_x
