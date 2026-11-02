import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def solve():
    t_start = time.time()
    HARD = 165.0  # internal deadline
    rng = np.random.default_rng(12345)

    def time_left():
        return HARD - (time.time() - t_start)

    # ---- load incumbent ----
    best_X = None
    best_val = np.inf
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn).astype(np.float64)
            if a.shape == (323, 7):
                a = a / np.linalg.norm(a, axis=1, keepdims=True)
                if not np.all(np.isfinite(a)):
                    continue
                v = _max_cos(a)
                if v < best_val:
                    best_val = v
                    best_X = a.copy()
        except Exception:
            continue
    if best_X is None:
        best_X = rng.standard_normal(size=(323, 7))
        best_X /= np.linalg.norm(best_X, axis=1, keepdims=True)
        best_val = _max_cos(best_X)
    incumbent_X = best_X.copy()
    incumbent_val = best_val

    # ---- symmetric builders ----
    def make_tables(m, speeds):
        th = 2.0 * np.pi / m
        k = np.arange(m)
        ca = np.cos(k * speeds[0] * th)
        sa = np.sin(k * speeds[0] * th)
        cb = np.cos(k * speeds[1] * th)
        sb = np.sin(k * speeds[1] * th)
        cc = np.cos(k * speeds[2] * th)
        sc = np.sin(k * speeds[2] * th)
        return ca, sa, cb, sb, cc, sc

    def build_full(Praw, tables, use_pole):
        ca, sa, cb, sb, cc, sc = tables
        m = ca.shape[0]
        n_orb = Praw.shape[0]
        z = Praw[:, :4]
        nrm = np.sqrt(np.sum(z * z, axis=1, keepdims=True) + 1e-16)
        u = z / nrm
        rho1 = np.abs(u[:, 0])
        rho2 = np.abs(u[:, 1])
        rho3 = np.abs(u[:, 2])
        h = u[:, 3]
        phi2 = Praw[:, 4]
        phi3 = Praw[:, 5]
        c2 = np.cos(phi2)[:, None]
        s2 = np.sin(phi2)[:, None]
        c3 = np.cos(phi3)[:, None]
        s3 = np.sin(phi3)[:, None]
        # (n_orb, m)
        x1 = rho1[:, None] * ca[None, :]
        x2 = rho1[:, None] * sa[None, :]
        x3 = rho2[:, None] * (c2 * cb[None, :] - s2 * sb[None, :])
        x4 = rho2[:, None] * (s2 * cb[None, :] + c2 * sb[None, :])
        x5 = rho3[:, None] * (c3 * cc[None, :] - s3 * sc[None, :])
        x6 = rho3[:, None] * (s3 * cc[None, :] + c3 * sc[None, :])
        x7 = h[:, None] * np.ones((1, m))
        blk = np.stack([x1, x2, x3, x4, x5, x6, x7], axis=2)  # (n_orb,m,7)
        X = blk.reshape(n_orb * m, 7)
        if use_pole:
            pole = np.zeros((1, 7))
            pole[0, 6] = 1.0
            X = np.vstack([X, pole])
        # normalize (already unit up to fp)
        X = X / np.linalg.norm(X, axis=1, keepdims=True)
        return X

    def eval_raw(Praw, tables, use_pole):
        X = build_full(Praw, tables, use_pole)
        return _max_cos(X), X

    # ---- symmetric evolutionary search ----
    configs = [
        (14, 23, (1, 5, 11), True),
        (14, 23, (1, 2, 7), True),
        (14, 23, (1, 3, 8), True),
        (17, 19, (1, 4, 9), False),
        (17, 19, (1, 7, 11), False),
    ]
    sym_best_X = None
    sym_best_val = np.inf

    # budget for symmetric phase: at most ~80s
    sym_deadline = min(time_left(), 80.0)
    sym_t0 = time.time()
    for (n_orb, m, speeds, use_pole) in configs:
        if time.time() - t_start >= HARD - 45.0:
            break
        if time.time() - sym_t0 >= sym_deadline:
            break
        tables = make_tables(m, speeds)
        # init
        Pbest = rng.standard_normal(size=(n_orb, 6)) * 0.7
        # scale phases a bit wider
        Pbest[:, 4:] *= 2.0
        try:
            bv, bX = eval_raw(Pbest, tables, use_pole)
        except Exception:
            continue
        sigma = 0.35
        noimp = 0
        it = 0
        # per-config time slice
        cfg_budget = min(16.0, max(4.0, (sym_deadline - (time.time() - sym_t0)) / (len(configs) - configs.index((n_orb, m, speeds, use_pole)))))
        cfg_t0 = time.time()
        batch = 24
        while True:
            it += 1
            if time.time() - cfg_t0 >= cfg_budget:
                break
            if time.time() - t_start >= HARD - 45.0:
                break
            if time.time() - sym_t0 >= sym_deadline:
                break
            # adapt sigma vector
            svec = np.array([1.0, 1.0, 1.0, 1.0, 1.6, 1.6]) * sigma
            try:
                noise = rng.standard_normal(size=(batch, n_orb, 6)) * svec[None, None, :]
            except Exception:
                break
            improved = False
            for b in range(batch):
                if time.time() - t_start >= HARD - 45.0:
                    break
                Pc = Pbest + noise[b]
                try:
                    v, _ = eval_raw(Pc, tables, use_pole)
                except Exception:
                    continue
                if v < bv:
                    bv = v
                    Pbest = Pc.copy()
                    improved = True
            if improved:
                noimp = 0
                sigma = min(0.5, sigma * 1.03)
            else:
                noimp += 1
                sigma *= 0.985
                if sigma < 0.015:
                    sigma = 0.015
                if noimp >= 60:
                    # large kick to change basin but keep best
                    sigma = 0.4
                    noimp = 0
                    # random restart around best with big step once
                    try:
                        Pc = Pbest + rng.standard_normal(size=(n_orb, 6)) * 0.6
                        v, _ = eval_raw(Pc, tables, use_pole)
                        if v < bv:
                            bv = v
                            Pbest = Pc.copy()
                    except Exception:
                        pass
            if it % 5 == 0:
                if time.time() - cfg_t0 >= cfg_budget:
                    break
        # finalize this config
        try:
            bv2, bX2 = eval_raw(Pbest, tables, use_pole)
        except Exception:
            continue
        if bv2 < sym_best_val:
            sym_best_val = bv2
            sym_best_X = bX2.copy()

    # ---- trust-region LP polish (full space, symmetry may break) ----
    def lp_polish(x, seconds):
        x = x / np.linalg.norm(x, axis=1, keepdims=True)
        n, d = x.shape
        best = _max_cos(x)
        best_x = x.copy()
        radius = 1e-3
        t0 = time.time()
        iuI_all, iuJ_all = np.triu_indices(n, 1)
        while (time.time() - t0 < seconds) and (time.time() - t_start < HARD - 5.0) and radius > 1e-14:
            G = x @ x.T
            c = G[iuI_all, iuJ_all]
            m = float(np.max(c))
            thresh = m - max(20.0 * radius, 1e-9)
            sel = c > thresh
            cnt = int(np.sum(sel))
            if cnt == 0:
                break
            if cnt > 2500:
                idx = np.where(sel)[0]
                cv = c[idx]
                part = np.argpartition(cv, -2500)[-2500:]
                sel2 = np.zeros_like(sel, dtype=bool)
                sel2[idx[part]] = True
                sel = sel2
                cnt = 2500
            I = iuI_all[sel]
            J = iuJ_all[sel]
            csel = c[sel]
            p = cnt
            nv = n * d + 1
            rows = np.repeat(np.arange(p), 2 * d + 1)
            cols = np.empty(p * (2 * d + 1), dtype=np.int64)
            vals = np.empty(p * (2 * d + 1), dtype=np.float64)
            crow = cols.reshape(p, 2 * d + 1)
            vrow = vals.reshape(p, 2 * d + 1)
            ar = np.arange(d)
            crow[:, :d] = I[:, None] * d + ar
            crow[:, d:2 * d] = J[:, None] * d + ar
            crow[:, 2 * d] = n * d
            vrow[:, :d] = x[J]
            vrow[:, d:2 * d] = x[I]
            vrow[:, 2 * d] = -1.0
            try:
                A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
                A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            except Exception:
                radius *= 0.5
                continue
            obj = np.zeros(nv)
            obj[-1] = 1.0
            remain = min(seconds - (time.time() - t0), HARD - 5.0 - (time.time() - t_start))
            if remain <= 0.2:
                break
            tl = float(min(5.0, max(0.5, remain - 0.2)))
            try:
                res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq, b_eq=np.zeros(n),
                              bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                              method="highs", options={"time_limit": tl})
            except Exception:
                radius *= 0.5
                continue
            if res.status != 0:
                radius *= 0.5
                continue
            try:
                y = x + res.x[:-1].reshape(n, d)
                yn = np.linalg.norm(y, axis=1, keepdims=True)
                yn[yn == 0] = 1.0
                y = y / yn
                v = _max_cos(y)
            except Exception:
                radius *= 0.5
                continue
            if v < best:
                x = y
                best = v
                best_x = y.copy()
                radius = min(radius * 1.5, 0.05)
            else:
                radius *= 0.5
        return best_x, best

    # polish symmetric best if exists
    polished_sym_X = None
    polished_sym_val = np.inf
    if sym_best_X is not None and time_left() > 12.0:
        secs = min(28.0, max(6.0, time_left() - 25.0))
        if secs > 1.0:
            try:
                polished_sym_X, polished_sym_val = lp_polish(sym_best_X, seconds=secs)
            except Exception:
                polished_sym_X = None

    # polish incumbent with remaining time
    polished_inc_X = None
    polished_inc_val = np.inf
    if time_left() > 8.0:
        secs = min(25.0, max(4.0, time_left() - 3.0))
        if secs > 1.0:
            try:
                polished_inc_X, polished_inc_val = lp_polish(incumbent_X, seconds=secs)
            except Exception:
                polished_inc_X = None

    # ---- select overall best ----
    cand = [(incumbent_X, incumbent_val)]
    if sym_best_X is not None:
        try:
            v = _max_cos(sym_best_X)
            cand.append((sym_best_X, v))
        except Exception:
            pass
    if polished_sym_X is not None:
        try:
            v = _max_cos(polished_sym_X)
            cand.append((polished_sym_X, v))
        except Exception:
            pass
    if polished_inc_X is not None:
        try:
            v = _max_cos(polished_inc_X)
            cand.append((polished_inc_X, v))
        except Exception:
            pass
    # also consider stored best_X (same as incumbent)
    best_out = incumbent_X
    best_v = incumbent_val
    for Xc, vc in cand:
        if np.all(np.isfinite(Xc)) and vc < best_v:
            best_v = vc
            best_out = Xc
    best_out = best_out / np.linalg.norm(best_out, axis=1, keepdims=True)
    # strict finite check
    if not np.all(np.isfinite(best_out)):
        best_out = incumbent_X
    return best_out
