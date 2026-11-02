import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def _max_cos_triu(x, iu0, iu1):
    G = x @ x.T
    c = G[iu0, iu1]
    return float(np.max(c)), G


def _norm(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return x / n


def micro_peel(x, best, deadline, rng, s_list=None, rand_sigmas=None, max_rand=300):
    n, d = x.shape
    if s_list is None:
        s_list = [5e-6, 2e-6, 1e-6, 5e-7, 2e-7, 1e-7]
    if rand_sigmas is None:
        rand_sigmas = [2e-6, 1e-6, 5e-7, 2e-7]
    ar = np.arange(n)
    fails = 0
    it = 0
    while time.time() < deadline:
        it += 1
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        flat = int(np.argmax(G))
        ii = flat // n
        jj = flat % n
        m = float(G[ii, jj])
        if m < best:
            best = m
        # masks for m_excl
        # compute m_excl for ii and jj
        # max over pairs not involving k: use G with row/col k masked
        # fast: overall second max fallback
        improved = False
        # try directed moves for both endpoints
        for k, partner in ((ii, jj), (jj, ii)):
            xk = x[k]
            xp = x[partner]
            c = float(np.dot(xk, xp))
            proj = xp - c * xk
            pn = float(np.linalg.norm(proj))
            if pn < 1e-14 or not np.finite(pn):
                continue
            unit = proj / pn
            # m_excl_k
            # compute by max of G excluding k
            # copy row/col mask via slicing: max of 3 blocks
            # simple: set row/col to -2 on a copy view
            # to avoid copy, compute max of G with mask
            # do small copy (453x453) - cheap
            # we recompute once per k
            Gtmp = G.copy()
            Gtmp[k, :] = -2.0
            Gtmp[:, k] = -2.0
            m_excl = float(np.max(Gtmp))
            for s in s_list:
                if time.time() >= deadline:
                    break
                yk = xk - s * unit
                nn = float(np.linalg.norm(yk))
                if nn == 0 or not np.finite(nn):
                    continue
                yk = yk / nn
                dv = x @ yk
                dv[k] = -2.0
                max1 = float(np.max(dv))
                newm = max1 if max1 > m_excl else m_excl
                if newm < best - 1e-12:
                    x[k] = yk
                    best = newm
                    # refresh G diagonal not needed; continue outer
                    improved = True
                    break
            if improved:
                break
        if improved:
            fails = 0
            continue
        # directed failed: try random tangent moves on ii,jj
        found = False
        for k in (ii, jj):
            Gtmp = G.copy()
            Gtmp[k, :] = -2.0
            Gtmp[:, k] = -2.0
            m_excl = float(np.max(Gtmp))
            xk = x[k].copy()
            for sigma in rand_sigmas:
                if found or time.time() >= deadline:
                    break
                for _ in range(max_rand // len(rand_sigmas)):
                    z = rng.normal(scale=sigma, size=d)
                    z = z - float(np.dot(z, xk)) * xk
                    yk = xk + z
                    nn = float(np.linalg.norm(yk))
                    if nn == 0 or not np.finite(nn):
                        continue
                    yk = yk / nn
                    dv = x @ yk
                    dv[k] = -2.0
                    max1 = float(np.max(dv))
                    newm = max1 if max1 > m_excl else m_excl
                    if newm < best - 1e-12:
                        x[k] = yk
                        best = newm
                        found = True
                        break
                if found:
                    break
            if found:
                break
        if found:
            fails = 0
            continue
        fails += 1
        if fails >= 2:
            break
        # tiny stall: sleep-free continue to let caller switch to LP
        break
    return x, best


def tiny_lp(x, best, deadline, r_init=5e-6):
    n, d = x.shape
    iu0 = np.triu_indices(n, 1)
    iu_r = iu0[0]
    iu_c = iu0[1]
    ar = np.arange(d)
    radius = float(r_init)
    # quick current best recompute
    G0 = x @ x.T
    best = float(np.max(G0[iu_r, iu_c]))
    while time.time() < deadline and radius > 1e-13:
        if deadline - time.time() < 2.2:
            break
        G = x @ x.T
        c = G[iu_r, iu_c]
        m = float(np.max(c))
        delta = max(12.0 * radius, 4e-7)
        sel = c > m - delta
        cnt = int(np.count_nonzero(sel))
        if cnt == 0:
            radius *= 0.5
            continue
        if cnt > 4000:
            idx = np.argpartition(-c, 4000)[:4000]
            # build mask for top 4000 among all? use idx directly
            I = iu_r[idx]
            J = iu_c[idx]
            csel = c[idx]
            # keep only those within delta? already top, keep all 4000
            # but ensure they are near max; if smallest far, trim
            keep = csel > m - delta
            if np.count_nonzero(keep) < 10:
                # keep top 500 anyway
                I = I[:500]
                J = J[:500]
                csel = csel[:500]
            else:
                I = I[keep]
                J = J[keep]
                csel = csel[keep]
        else:
            I = iu_r[sel]
            J = iu_c[sel]
            csel = c[sel]
        p = int(I.size)
        if p == 0:
            radius *= 0.5
            continue
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate(
            [I[:, None] * d + ar, J[:, None] * d + ar,
             np.full((p, 1), n * d)], axis=1
        ).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], axis=1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix(
                (x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))),
                shape=(n, nv)).tocsr()
            obj = np.zeros(nv)
            obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq,
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
        y = _norm(y)
        if not np.all(np.isfinite(y)):
            radius *= 0.5
            continue
        Gy = y @ y.T
        v = float(np.max(Gy[iu_r, iu_c]))
        if v < best - 1e-12:
            x = y
            best = v
            radius = min(radius * 1.4, 2e-5)
        else:
            radius *= 0.5
    return x, best


def solve():
    t0 = time.time()
    HARD_END = t0 + 172.0
    rng = np.random.default_rng(0)
    cands = []
    try:
        b = np.load("best.npy")
        cands.append(np.asarray(b, dtype=np.float64))
    except Exception:
        pass
    try:
        r = np.load("record.npy")
        cands.append(np.asarray(r, dtype=np.float64))
    except Exception:
        pass
    bestX = None
    bestV = float("inf")
    iu_tmp_r, iu_tmp_c = np.triu_indices(453, 1)
    for cc in cands:
        try:
            if cc.shape != (453, 5):
                flat = cc.ravel()
                need = 453 * 5
                if flat.size < need:
                    continue
                cc = flat[:need].reshape(453, 5)
            cc = _norm(cc)
            if not np.all(np.isfinite(cc)):
                continue
            G = cc @ cc.T
            vv = float(np.max(G[iu_tmp_r, iu_tmp_c]))
            if vv < bestV:
                bestV = vv
                bestX = cc.copy()
        except Exception:
            continue
    if bestX is None:
        bestX = rng.normal(size=(453, 5))
        bestX = _norm(bestX)
        G = bestX @ bestX.T
        bestV = float(np.max(G[iu_tmp_r, iu_tmp_c]))

    x = bestX.copy()
    best = bestV

    # Phase 1: peel isolated maxima
    try:
        x, best = micro_peel(x, best, deadline=t0 + 50.0, rng=rng)
    except Exception:
        pass

    # Phase 2: tiny LP
    if time.time() < t0 + 100.0:
        try:
            x, best = tiny_lp(x, best, deadline=min(t0 + 100.0, HARD_END - 40.0))
        except Exception:
            pass

    # re-peel
    if time.time() < t0 + 120.0:
        try:
            x, best = micro_peel(x, best, deadline=min(t0 + 120.0, HARD_END - 30.0), rng=rng)
        except Exception:
            pass

    # Phase 3: small basin hopping around worst points + repeel
    trial = 0
    n, d = x.shape
    glob_best_X = x.copy()
    glob_best = best
    while time.time() < t0 + 150.0 and time.time() < HARD_END - 22.0 and trial < 20:
        trial += 1
        try:
            G = glob_best_X @ glob_best_X.T
            np.fill_diagonal(G, -2.0)
            pm = np.max(G, axis=1)
            order = np.argsort(-pm)
            sig = [3e-4, 1.5e-4, 5e-4, 8e-5][(trial - 1) % 4]
            y = glob_best_X.copy()
            # perturb worst 40 plus light global noise
            worst = order[:40]
            y = y + rng.normal(scale=sig * 0.2, size=y.shape)
            y[worst] = y[worst] + rng.normal(scale=sig, size=(worst.size, d))
            y = _norm(y)
            Gy = y @ y.T
            v0 = float(np.max(Gy[iu_tmp_r, iu_tmp_c]))
            # short peel (2.5s)
            y2, v2 = micro_peel(y, v0, deadline=min(time.time() + 2.5, t0 + 150.0, HARD_END - 22.0), rng=rng,
                                max_rand=160)
            if v2 < glob_best - 1e-12 and np.all(np.isfinite(y2)):
                glob_best_X = y2.copy()
                glob_best = v2
                x = glob_best_X.copy()
                best = glob_best
        except Exception:
            continue

    x = glob_best_X.copy()
    best = glob_best

    # Final polish
    if time.time() < HARD_END - 3.0:
        try:
            x, best = tiny_lp(x, best, deadline=HARD_END - 1.5, r_init=3e-6)
        except Exception:
            pass
        try:
            x, best = micro_peel(x, best, deadline=HARD_END - 0.5, rng=rng, max_rand=120)
        except Exception:
            pass

    x = _norm(x)
    return x
