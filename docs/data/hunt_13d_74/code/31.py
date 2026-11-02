import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _mc(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _norm(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return X / n

def solve():
    t0 = time.time()
    DEADLINE = t0 + 165.0
    rng = np.random.default_rng(0)
    n0, d = 74, 13

    # ---- load ----
    cands = []
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=float)
            if a.shape == (74, 13):
                a = _norm(a)
                cands.append((_mc(a), a))
        except Exception:
            pass
    if not cands:
        a = _norm(rng.standard_normal((74, 13)))
        cands.append((_mc(a), a))
    cands.sort(key=lambda t: t[0])
    Xbest = cands[0][1].copy()
    best_val = float(cands[0][0])

    # ---------- generic helpers ----------
    def eval_max(X):
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        return float(G.max())

    def per_point_order(X):
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        pm = G.max(axis=1)
        order = np.argsort(-pm)
        k = int(np.argmax(G))
        nn = X.shape[0]
        return pm, order, (k // nn, k % nn)

    def adam_run(X, stages, deadline, track=True):
        cur = X.copy()
        nn_, dd_ = cur.shape
        iu_r, iu_c = np.triu_indices(nn_, 1)
        bv = eval_max(cur)
        bX = cur.copy()
        b1, b2, eps = 0.9, 0.999, 1e-8
        step_c = 0
        for (tau, steps, lr) in stages:
            m1 = np.zeros_like(cur)
            m2 = np.zeros_like(cur)
            for s in range(steps):
                if time.time() > deadline:
                    break
                step_c += 1
                G = cur @ cur.T
                g = G[iu_r, iu_c]
                mx = float(g.max())
                w = np.exp((g - mx) / tau)
                sw = float(w.sum())
                if not np.isfinite(sw) or sw <= 0:
                    break
                w /= sw
                S = np.zeros((nn_, nn_))
                S[iu_r, iu_c] = w
                S = S + S.T
                grad = S @ cur
                dot = np.sum(grad * cur, axis=1, keepdims=True)
                gt = grad - dot * cur
                m1 = b1 * m1 + (1 - b1) * gt
                m2 = b2 * m2 + (1 - b2) * (gt * gt)
                c1 = 1 - b1 ** step_c
                c2 = 1 - b2 ** step_c
                m1h = m1 / max(c1, 1e-16)
                m2h = m2 / max(c2, 1e-16)
                upd = m1h / (np.sqrt(m2h) + eps)
                dp = np.sum(upd * cur, axis=1, keepdims=True)
                upd_t = upd - dp * cur
                nr = np.linalg.norm(upd_t, axis=1, keepdims=True) + 1e-16
                scale = np.minimum(1.0, 1.0 / nr)
                cur = cur - lr * upd_t * scale
                cur = _norm(cur)
                if track and (step_c % 50 == 0 or s == steps - 1):
                    v = eval_max(cur)
                    if v < bv:
                        bv = v
                        bX = cur.copy()
            if time.time() > deadline:
                break
        if track:
            return bX, bv
        else:
            return cur, eval_max(cur)

    def micro_sep(X, val, deadline=None):
        cur = X.copy()
        cur_v = val
        nn_ = cur.shape[0]
        dd_ = cur.shape[1]
        eps_list = [3e-4, 1e-4, 3e-5, 1e-5, 3e-6, 3e-3, 1e-3]
        for _ in range(12):
            if deadline is not None and time.time() > deadline:
                break
            G = cur @ cur.T
            np.fill_diagonal(G, -2.0)
            flat = int(np.argmax(G))
            a = flat // nn_; bb = flat % nn_
            improved = False
            for idx, other in ((a, bb), (bb, a)):
                xi = cur[idx]; xj = cur[other]
                dot = float(xi @ xj)
                direc = xj - dot * xi
                nd = float(np.linalg.norm(direc))
                if nd < 1e-14:
                    continue
                direc /= nd
                for eps in eps_list:
                    trial = cur.copy()
                    nv = xi - eps * direc
                    nv /= (np.linalg.norm(nv) + 1e-16)
                    trial[idx] = nv
                    v = eval_max(trial)
                    if v < cur_v - 1e-12:
                        cur = trial; cur_v = v; improved = True
                        break
                if improved:
                    break
            if improved:
                continue
            for _r in range(20):
                if deadline is not None and time.time() > deadline:
                    break
                done = False
                for idx in (a, bb):
                    xi = cur[idx]
                    z = rng.standard_normal(dd_)
                    z = z - (z @ xi) * xi
                    nz = float(np.linalg.norm(z))
                    if nz < 1e-14:
                        continue
                    z /= nz
                    eps = float(rng.choice(np.array([2e-5, 5e-5, 1.5e-4, 4e-4]))))
                    trial = cur.copy()
                    nv = xi + eps * z
                    nv /= (np.linalg.norm(nv) + 1e-16)
                    trial[idx] = nv
                    v = eval_max(trial)
                    if v < cur_v - 1e-12:
                        cur = trial; cur_v = v; improved = True; done = True
                        break
                if done:
                    break
            if not improved:
                break
        return cur, cur_v

    def hole_point(X, n_cand=8000, refine=120):
        dd_ = X.shape[1]
        C = _norm(rng.standard_normal((n_cand, dd_)))
        s = C @ X.T
        m = s.max(axis=1)
        bi = int(np.argmin(m))
        v = C[bi].copy()
        bv = float(m[bi])
        step = 0.15
        for _ in range(refine):
            dots = X @ v
            j = int(np.argmax(dots))
            c = float(dots[j])
            g = X[j] - c * v
            ng = float(np.linalg.norm(g))
            if ng < 1e-14:
                break
            g /= ng
            # try step
            nv = v - step * g
            nv /= (np.linalg.norm(nv) + 1e-16)
            nm = float((X @ nv).max())
            if nm < bv:
                v = nv; bv = nm
                step = min(step * 1.05, 0.2)
            else:
                step *= 0.92
                if step < 1e-5:
                    break
        return v

    def insert_holes(X, k):
        cur = X.copy()
        for _ in range(k):
            v = hole_point(cur)
            cur = np.vstack([cur, v.reshape(1, -1)])
        return _norm(cur)

    def point_stats(X):
        nn_ = X.shape[0]
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        M = float(G.max())
        pm = G.max(axis=1)
        deg = (G >= M - 1e-7).sum(axis=1).astype(float)
        # sum top-5
        s5 = np.zeros(nn_)
        for i in range(nn_):
            row = G[i]
            # top5 largest
            if nn_ >= 6:
                part = np.argpartition(-row, 5)[:5]
                s5[i] = float(row[part].sum())
            else:
                s5[i] = float(row.sum())
        return M, pm, deg, s5

    def worst_indices(X, kdel):
        # iterative deletion of worst one by one
        idx = list(range(X.shape[0]))
        curX = X.copy()
        out = []
        for _ in range(kdel):
            M, pm, deg, s5 = point_stats(curX)
            # lex order: pm desc, deg desc, s5 desc
            order = sorted(range(curX.shape[0]), key=lambda i: (-pm[i], -deg[i], -s5[i]))
            w = order[0]
            out.append(idx[w])
            # remove
            keep = [i for i in range(curX.shape[0]) if i != w]
            idx = [idx[i] for i in keep]
            curX = curX[keep]
        return out

    def prune_to(X, target):
        cur = X.copy()
        while cur.shape[0] > target:
            M, pm, deg, s5 = point_stats(cur)
            order = sorted(range(cur.shape[0]), key=lambda i: (-pm[i], -deg[i], -s5[i]))
            w = order[0]
            keep = [i for i in range(cur.shape[0]) if i != w]
            cur = cur[keep]
        return _norm(cur)

    def lp_polish(X, seconds):
        X = _norm(X)
        best = eval_max(X)
        bestX = X.copy()
        nn_ = X.shape[0]; dd_ = X.shape[1]
        iu_r, iu_c = np.triu_indices(nn_, 1)
        radius = 1e-3
        start = time.time()
        while time.time() - start < seconds and radius > 1e-14:
            if time.time() > DEADLINE - 1.0:
                break
            G = X @ X.T
            c = G[iu_r, iu_c]
            m = float(c.max())
            thr = max(20 * radius, 1e-9)
            sel = c > m - thr
            if int(sel.sum()) == 0:
                radius *= 0.5
                continue
            if int(sel.sum()) > 1500:
                idxx = np.where(sel)[0]
                od = np.argsort(-c[idxx])[:1500]
                sel2 = np.zeros_like(sel, dtype=bool)
                sel2[idxx[od]] = True
                sel = sel2
            I = iu_r[sel]; J = iu_c[sel]
            p = len(I)
            cc = c[sel]
            nv = nn_ * dd_ + 1
            rows = np.repeat(np.arange(p), 2 * dd_ + 1)
            cols = np.concatenate([I[:, None] * dd_ + np.arange(dd_), J[:, None] * dd_ + np.arange(dd_), np.full((p, 1), nn_ * dd_)], 1).ravel()
            vals = np.concatenate([X[J], X[I], -np.ones((p, 1))], 1).ravel()
            try:
                A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
                A_eq = coo_matrix((X.ravel(), (np.repeat(np.arange(nn_), dd_), np.arange(nn_ * dd_))), shape=(nn_, nv)).tocsr()
                obj = np.zeros(nv); obj[-1] = 1.0
                res = linprog(obj, A_ub=A_ub, b_ub=-cc, A_eq=A_eq, b_eq=np.zeros(nn_),
                              bounds=[(-radius, radius)] * (nn_ * dd_) + [(None, None)], method="highs")
            except Exception:
                radius *= 0.5
                continue
            if res.status != 0:
                radius *= 0.5
                continue
            y = X + res.x[:-1].reshape(nn_, dd_)
            y = _norm(y)
            v = eval_max(y)
            if v < best:
                X, best = y, v
                bestX = y.copy()
                radius = min(radius * 1.5, 0.05)
            else:
                radius *= 0.5
        return bestX

    def short_polish(X, deadline):
        # quick adam + micro
        if time.time() > deadline:
            return X, eval_max(X)
        Xa, va = adam_run(X, [(0.005, 150, 0.008), (0.0015, 200, 0.005), (0.0005, 200, 0.003)], deadline)
        Xa, va = micro_sep(Xa, va, deadline)
        return Xa, va

    # ---- initial light polish ----
    Xbest, best_val = micro_sep(Xbest, best_val)
    dl = min(DEADLINE - 100.0, time.time() + 6.0)
    if time.time() < dl:
        Xt, vt = adam_run(Xbest, [(0.008, 150, 0.01), (0.002, 200, 0.006)], dl)
        Xt, vt = micro_sep(Xt, vt)
        if vt < best_val:
            Xbest, best_val = Xt, vt

    # ---- Phase A: overshoot 74 -> 78 -> 74 ----
    if time.time() < DEADLINE - 60.0:
        try:
            X78 = insert_holes(Xbest, 4)
            dlA = time.time() + 24.0
            # anneal harder problem
            X78a, _ = adam_run(X78, [(0.010, 300, 0.012), (0.003, 400, 0.007), (0.001, 400, 0.004), (0.0004, 300, 0.002)], dlA)
            X78a, _ = micro_sep(X78a, eval_max(X78a), dlA)
            Xpr = prune_to(X78a, 74)
            Xpr, vpr = short_polish(Xpr, min(DEADLINE - 50.0, time.time() + 6.0))
            if vpr < best_val - 1e-12:
                Xbest, best_val = Xpr, vpr
            elif vpr < best_val:
                Xbest, best_val = Xpr, vpr
        except Exception:
            pass

    # ---- Phase B: undershoot 74 -> 70 -> 74 staged grow ----
    if time.time() < DEADLINE - 40.0:
        try:
            # drop 4 worst hubs from best
            wi = worst_indices(Xbest, 4)
            keep = [i for i in range(74) if i not in set(wi)]
            X70 = _norm(Xbest[keep])
            dlB = time.time() + 14.0
            X70a, _ = adam_run(X70, [(0.008, 300, 0.01), (0.002, 400, 0.006), (0.0006, 300, 0.003)], dlB)
            X70a, _ = micro_sep(X70a, eval_max(X70a), dlB)
            cur = X70a.copy()
            for k in range(4):
                if time.time() > DEADLINE - 35.0:
                    break
                v = hole_point(cur)
                cur = _norm(np.vstack([cur, v.reshape(1, -1)]))
                # polish between insertions
                cur, _ = adam_run(cur, [(0.004, 120, 0.007), (0.001, 150, 0.004)], min(DEADLINE - 30.0, time.time() + 2.5))
                cur, _ = micro_sep(cur, eval_max(cur))
            if cur.shape[0] == 74:
                cur, vc = short_polish(cur, min(DEADLINE - 25.0, time.time() + 5.0))
                if vc < best_val:
                    Xbest, best_val = cur, vc
        except Exception:
            pass

    # ---- Phase C: second overshoot 74 -> 76 -> 74 ----
    if time.time() < DEADLINE - 25.0:
        try:
            X76 = insert_holes(Xbest, 2)
            dlC = time.time() + 12.0
            X76a, _ = adam_run(X76, [(0.008, 250, 0.01), (0.002, 350, 0.006), (0.0005, 350, 0.003)], dlC)
            X76a, _ = micro_sep(X76a, eval_max(X76a), dlC)
            Xpr2 = prune_to(X76a, 74)
            Xpr2, vpr2 = short_polish(Xpr2, min(DEADLINE - 15.0, time.time() + 5.0))
            if vpr2 < best_val:
                Xbest, best_val = Xpr2, vpr2
        except Exception:
            pass

    # ---- small basin kicks with remaining time ----
    sigmas = [0.0006, 0.002, 0.008, 0.03]
    ti = 0
    while time.time() < DEADLINE - 12.0 and ti < 40:
        try:
            sigma = sigmas[ti % len(sigmas)]
            ti += 1
            _, order, _ = per_point_order(Xbest)
            Z = rng.standard_normal((n0, d))
            dp = np.sum(Z * Xbest, axis=1, keepdims=True)
            Zt = Z - dp * Xbest
            Zt = Zt / (np.linalg.norm(Zt, axis=1, keepdims=True) + 1e-16)
            if ti % 2 == 0:
                Xt = _norm(Xbest + sigma * Zt)
            else:
                k = [2, 4, 8][(ti // 2) % 3]
                worst = order[:k]
                step = np.full((n0, 1), sigma * 0.15)
                step[worst] = sigma
                Xt = _norm(Xbest + step * Zt)
            Xt_p, vt_p = adam_run(Xt, [(0.005, 120, 0.008), (0.0012, 180, 0.005)], min(DEADLINE - 11.0, time.time() + 2.0))
            Xt_p, vt_p = micro_sep(Xt_p, vt_p)
            if vt_p < best_val - 1e-12:
                Xbest, best_val = Xt_p, vt_p
        except Exception:
            break

    # ---- final LP squeeze ----
    remain = DEADLINE - time.time()
    if remain > 5.0:
        try:
            Xlp = lp_polish(Xbest, min(remain - 1.0, 9.0))
            vlp = eval_max(Xlp)
            if vlp < best_val:
                Xbest, best_val = Xlp, vlp
            Xbest, best_val = micro_sep(Xbest, best_val)
        except Exception:
            pass

    Xbest = _norm(Xbest)
    Xbest = np.nan_to_num(Xbest, nan=0.0, posinf=0.0, neginf=0.0)
    Xbest = _norm(Xbest)
    return Xbest
