import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _lp_polish(x, seconds, p_cap=1200):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    best_x = x.copy()
    radius = 1.2e-3
    start = time.time()
    iu0, iu1 = np.triu_indices(n, 1)
    itc = 0
    while time.time() - start < seconds and radius > 1e-14 and itc < 30:
        itc += 1
        G = x @ x.T
        c_all = G[iu0, iu1]
        m = c_all.max()
        window = max(25 * radius, 2e-9)
        sel = c_all > m - window
        cnt = int(np.count_nonzero(sel))
        if cnt == 0:
            break
        if cnt > p_cap:
            idx = np.where(sel)[0]
            part = np.argpartition(-c_all[idx], p_cap - 1)[:p_cap]
            keep = np.zeros_like(sel)
            keep[idx[part]] = True
            sel = keep
        I = iu0[sel]; J = iu1[sel]
        csel = c_all[sel]
        p = len(I)
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([(I[:, None] * d + np.arange(d)), (J[:, None] * d + np.arange(d)), np.full((p, 1), n * d)], axis=1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], axis=1).ravel()
        tl = max(0.25, seconds - (time.time() - start))
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv); obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs",
                          options={"time_limit": tl})
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        yn = np.linalg.norm(y, axis=1, keepdims=True)
        yn[yn == 0] = 1.0
        y = y / yn
        v = _max_cos(y)
        if v < best:
            best = v; best_x = y.copy(); x = y
            radius = min(radius * 1.6, 0.03)
        else:
            radius *= 0.5
        if time.time() - start >= seconds:
            break
    return best_x

def _tangent_data(X):
    n, d = X.shape
    s = -np.sign(X[:, 0])
    s[s == 0] = -1.0
    V = X.copy()
    V[:, 0] = X[:, 0] - s
    denom = np.einsum('ij,ij->i', V, V) + 1e-30
    return V, denom

def _build_R(X, V, denom, pairs_I, pairs_J, n, dm1):
    p = len(pairs_I)
    N = n * dm1
    R = np.empty((p, N), dtype=float)
    for k in range(p):
        i = int(pairs_I[k]); j = int(pairs_J[k])
        xi = X[i]; xj = X[j]
        vi = V[i]; vj = V[j]
        di = denom[i]; dj = denom[j]
        dot_i = float(np.dot(xj, vi))
        dot_j = float(np.dot(xi, vj))
        ci = xj[1:] - (2.0 * dot_i / di) * vi[1:]
        cj = xi[1:] - (2.0 * dot_j / dj) * vj[1:]
        row = np.zeros(N, dtype=float)
        row[i * dm1:(i + 1) * dm1] = ci
        row[j * dm1:(j + 1) * dm1] = cj
        R[k] = row
    return R

def _apply_mode(X, V, denom, y, step):
    n, d = X.shape
    dm1 = d - 1
    Y = y.reshape(n, dm1)
    bn = np.linalg.norm(Y, axis=1)
    mx = bn.max()
    if mx < 1e-18:
        return X.copy()
    scale = step / mx
    B = Y * scale
    sdot = np.einsum('ij,ij->i', V[:, 1:], B)
    s = 2.0 * sdot / denom
    DX = np.empty_like(X)
    DX[:, 0] = -s * V[:, 0]
    DX[:, 1:] = B - s[:, None] * V[:, 1:]
    Xn = X + DX
    Xn /= np.linalg.norm(Xn, axis=1, keepdims=True)
    return Xn

def solve():
    t0 = time.time()
    LIM = 168.0
    rng = np.random.default_rng(0)
    n, d = 123, 13
    dm1 = d - 1
    N = n * dm1
    X0 = None; bv = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, float)
            if a.shape == (n, d):
                a /= np.linalg.norm(a, axis=1, keepdims=True)
                v = _max_cos(a)
                if v < bv:
                    bv = v; X0 = a.copy()
        except Exception:
            pass
    if X0 is None:
        X0 = rng.standard_normal((n, d))
        X0 /= np.linalg.norm(X0, axis=1, keepdims=True)
        bv = _max_cos(X0)
    cur = X0.copy(); cur_val = float(bv)
    best = X0.copy(); best_val = float(bv)
    try:
        if time.time() - t0 < LIM - 50:
            cand = _lp_polish(best, 2.0)
            v = _max_cos(cand)
            if v < best_val:
                best = cand.copy(); best_val = v; cur = cand.copy(); cur_val = v
    except Exception:
        pass
    iu0, iu1 = np.triu_indices(n, 1)
    steps_cycle = np.array([0.12, 0.06, 0.09, 0.04, 0.15, 0.03, 0.10, 0.05, 0.07, 0.13])
    it = 0
    while time.time() - t0 < LIM - 14:
        it += 1
        remain = LIM - 14 - (time.time() - t0)
        if remain < 3.0:
            break
        try:
            Gc = cur @ cur.T
            np.fill_diagonal(Gc, -2.0)
            wmax = Gc.max(axis=1)
            c_all = Gc[iu0, iu1]
            m = float(c_all.max())
            thr = m - 1e-7
            msk = c_all > thr
            I0 = iu0[msk]; J0 = iu1[msk]; C0 = c_all[msk]
            if len(I0) == 0:
                I0 = iu0; J0 = iu1; C0 = c_all
            if len(I0) > 1800:
                part = np.argpartition(-C0, 1799)[:1800]
                I0 = I0[part]; J0 = J0[part]; C0 = C0[part]
            deg = np.bincount(np.concatenate([I0, J0]), minlength=n)
            hubs = np.argsort(-deg)
            hub = int(hubs[(it - 1) % min(5, n)])
            inc = np.where((I0 == hub) | (J0 == hub))[0]
            if len(inc) == 0:
                hub = int(hubs[0])
                inc = np.where((I0 == hub) | (J0 == hub))[0]
            kdrop = 3 + (it % 4)
            kdrop = min(kdrop, len(inc))
            if kdrop < 2 and len(I0) >= 2:
                kdrop = min(2, len(I0))
            sel_drop = rng.choice(inc, size=kdrop, replace=False) if len(inc) >= kdrop and kdrop > 0 else np.array([], dtype=int)
            # occasionally add one random non-hub contact to drop set
            if it % 3 == 0 and len(I0) > kdrop + 1 and kdrop > 0:
                extra = rng.integers(0, len(I0))
                if extra not in sel_drop:
                    sel_drop = np.append(sel_drop, extra)
            keep_mask = np.ones(len(I0), dtype=bool)
            if len(sel_drop) > 0:
                keep_mask[sel_drop] = False
            KI, KJ = I0[keep_mask], J0[keep_mask]
            DI, DJ = I0[~keep_mask], J0[~keep_mask]
            if len(KI) == 0 or len(DI) == 0:
                # fallback random kick
                trial = cur + 0.01 * rng.standard_normal(cur.shape)
                trial /= np.linalg.norm(trial, axis=1, keepdims=True)
                trial = _lp_polish(trial, 0.6)
                v = _max_cos(trial)
                if v < best_val - 1e-14:
                    best_val = v; best = trial.copy(); cur = trial.copy(); cur_val = v
                elif v < cur_val + 0.001:
                    cur = trial.copy(); cur_val = v
                continue
            V, denom = _tangent_data(cur)
            Rk = _build_R(cur, V, denom, KI, KJ, n, dm1)
            Rd = _build_R(cur, V, denom, DI, DJ, n, dm1)
            try:
                _, S, Vt = np.linalg.svd(Rk, full_matrices=False)
            except Exception:
                continue
            p_kept = Rk.shape[0]
            nc = 40
            if p_kept < N - 5:
                Rnd = rng.standard_normal((N, nc))
                Co = Vt @ Rnd
                Rnd = Rnd - Vt.T @ Co
                nr = np.linalg.norm(Rnd, axis=0) + 1e-20
                Rnd = Rnd / nr
                # score null candidates
                Drop = Rd @ Rnd  # kd x nc
                mean_abs = np.abs(Drop).mean(axis=0)
                hub_blocks = Rnd[hub * dm1:(hub + 1) * dm1, :THING] if False else None
                hb = np.linalg.norm(Rnd[hub * dm1:(hub + 1) * dm1, :], axis=0)
                kept_rms = np.zeros(nc)
                scores = (mean_abs + 0.5 * hb) / (kept_rms + 0.002)
                bi = int(np.argmax(scores))
                y_best = Rnd[:, bi].copy()
            else:
                M = min(25, Vt.shape[0])
                Vb = Vt[-M:, :].T  # N x M
                Sm = S[-M:]
                C = rng.standard_normal((M, nc))
                Yc = Vb @ C
                nr = np.linalg.norm(Yc, axis=0) + 1e-20
                Yc = Yc / nr
                # kept rms via singular values: for each cand, coeff in Vb basis
                # coeff = Vb^T y ; rms = sqrt(sum Sm^2 coeff^2 / p)
                Cb = Vb.T @ Yc  # M x nc
                kept_rms = np.sqrt(np.einsum('m,mj->j', Sm * Sm, Cb * Cb) / max(p_kept, 1))
                Drop = Rd @ Yc
                mean_abs = np.abs(Drop).mean(axis=0)
                hb = np.linalg.norm(Yc[hub * dm1:(hub + 1) * dm1, :], axis=0)
                scores = (mean_abs + 0.5 * hb) / (kept_rms + 0.002)
                bi = int(np.argmax(scores))
                y_best = Yc[:, bi].copy()
            sv = Rd @ y_best
            if float(sv.max()) > float((-sv).max()):
                y_best = -y_best
            step = float(steps_cycle[(it - 1) % len(steps_cycle)])
            trial = _apply_mode(cur, V, denom, y_best, step)
            # rattler / low-contact hole reinsertion
            try:
                Gt = trial @ trial.T
                np.fill_diagonal(Gt, -2.0)
                cmax = Gt.max(axis=1)
                rattlers = np.where(cmax < m - 0.005)[0]
                if len(rattlers) > 0:
                    for r in rattlers[:2]:
                        mask = np.ones(n, dtype=bool); mask[r] = False
                        Xr = trial[mask]
                        Ys = rng.standard_normal((4000, d))
                        Ys /= np.linalg.norm(Ys, axis=1, keepdims=True)
                        md = (Ys @ Xr.T).max(axis=1)
                        pi = int(np.argmin(md))
                        vnew = Ys[pi] + 0.002 * rng.standard_normal(d)
                        vnew /= np.linalg.norm(vnew)
                        trial[r] = vnew
            except Exception:
                pass
            lp_sec = 0.8 if remain > 30 else 0.55
            try:
                trial = _lp_polish(trial, lp_sec)
            except Exception:
                pass
            v = _max_cos(trial)
            if v < best_val - 1e-14:
                best_val = v; best = trial.copy(); cur = trial.copy(); cur_val = v
            else:
                tol = 0.002 * np.exp(-it / 30.0) + 0.0002
                if v < cur_val + tol:
                    cur = trial.copy(); cur_val = v
                else:
                    # revert toward best with small noise to diversify
                    if it % 20 == 0:
                        cur = best + 0.006 * rng.standard_normal(best.shape)
                        cur /= np.linalg.norm(cur, axis=1, keepdims=True)
                        cur_val = _max_cos(cur)
        except Exception:
            continue
        if time.time() - t0 >= LIM - 14:
            break
    fsec = LIM - (time.time() - t0) - 0.6
    fsec = float(np.clip(fsec, 1.0, 12.0))
    try:
        pol = _lp_polish(best, fsec)
        pv = _max_cos(pol)
        if pv < best_val:
            best = pol; best_val = pv
    except Exception:
        pass
    best /= np.linalg.norm(best, axis=1, keepdims=True)
    return best
