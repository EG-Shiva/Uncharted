import time
import numpy as np
from scipy.optimize import minimize, linprog
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def _per_max(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return np.max(G, axis=1)

def _fingerprint(X, K=300):
    G = X @ X.T
    iu = np.triu_indices(X.shape[0], 1)
    c = G[iu]
    c = np.sort(c)[::-1]
    return c[:K].copy()

def _hist_dist(a, b):
    return float(np.linalg.norm(a - b))

def _lse_opt(X0, betas, iters):
    n, d = X0.shape
    IU = np.triu_indices(n, 1)
    y = (X0 / np.linalg.norm(X0, axis=1, keepdims=True)).ravel().copy()
    for beta, mi in zip(betas, iters):
        def fg(yf, _beta=beta):
            Y = yf.reshape(n, d)
            nr = np.maximum(np.linalg.norm(Y, axis=1, keepdims=True), 1e-12)
            X = Y / nr
            g = (X @ X.T)[IU]
            m = float(np.max(g))
            e = np.exp(_beta * (g - m))
            S = float(np.sum(e)) + 1e-300
            F = m + np.log(S) / _beta
            w = e / S
            P = np.zeros((n, n))
            P[IU] = w
            P = P + P.T
            GX = P @ X
            dot = np.sum(GX * X, axis=1, keepdims=True)
            GY = (GX - dot * X) / nr
            return float(F), GY.ravel()
        try:
            r = minimize(fg, y, jac=True, method='L-BFGS-B',
                         options={'maxiter': mi, 'ftol': 1e-14, 'gtol': 1e-7,
                                  'maxfun': mi * 2, 'maxcor': 10})
            y = r.x.copy()
        except Exception:
            break
    Y = y.reshape(n, d)
    Y /= (np.linalg.norm(Y, axis=1, keepdims=True) + 1e-300)
    return Y

def _single_opt(Xr, z0, beta=800.0, mi=80):
    def fg(z):
        r = max(float(np.linalg.norm(z)), 1e-12)
        y = z / r
        s = Xr @ y
        m = float(np.max(s))
        e = np.exp(beta * (s - m))
        S = float(np.sum(e)) + 1e-300
        F = m + np.log(S) / beta
        w = e / S
        gy = Xr.T @ w
        gz = (gy - float(gy @ y) * y) / r
        return float(F), gz
    try:
        r = minimize(fg, z0.copy(), jac=True, method='L-BFGS-B',
                     options={'maxiter': mi, 'ftol': 1e-15, 'gtol': 1e-9, 'maxfun': mi * 2})
        z = r.x
        z /= (np.linalg.norm(z) + 1e-300)
        return z
    except Exception:
        z = z0 / (np.linalg.norm(z0) + 1e-300)
        return z

def _best_hole(Xr, rng, ncand=20000, refine=True):
    d = Xr.shape[1]
    cand = rng.standard_normal((ncand, d))
    cand /= (np.linalg.norm(cand, axis=1, keepdims=True) + 1e-300)
    sc = cand @ Xr.T
    ms = np.max(sc, axis=1)
    bi = int(np.argmin(ms))
    h = cand[bi]
    if refine:
        h = _single_opt(Xr, h, beta=600.0, mi=60)
    return h

def _lp_polish(x, deadline, r0=8e-4, cap_p=2500):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = r0
    iu = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        if time.time() > deadline - 0.4:
            break
        G = x @ x.T
        c = G[iu]
        m = float(np.max(c))
        sel = c > m - max(20.0 * radius, 1e-9)
        I = iu[0][sel]; J = iu[1][sel]
        p = len(I)
        if p == 0:
            radius *= 0.5
            continue
        csel = c[sel]
        if p > cap_p:
            od = np.argsort(-csel)[:cap_p]
            I, J = I[od], J[od]
            csel = csel[od]
            p = cap_p
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv); obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        step = res.x[:-1].reshape(n, d)
        y = x + step
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        v = _max_cos(y)
        if v < best - 1e-15:
            x, best = y, v
            radius = min(radius * 1.5, 0.02)
        else:
            ok = False
            for sh in (0.3, 0.1):
                yy = x + step * sh
                yy /= np.linalg.norm(yy, axis=1, keepdims=True)
                vv = _max_cos(yy)
                if vv < best:
                    x, best = yy, vv
                    ok = True
                    break
            if not ok:
                radius *= 0.5
    return x

def solve():
    t0 = time.time()
    T_END = t0 + 172.0
    HOP_END = t0 + 148.0
    n, d = 68, 12
    rng = np.random.default_rng(20261031)

    best = None
    best_s = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.asarray(np.load(fn), dtype=np.float64)
            if a.shape == (n, d):
                a = a / np.linalg.norm(a, axis=1, keepdims=True)
                if np.all(np.isfinite(a)):
                    s = _max_cos(a)
                    if s < best_s:
                        best_s = s
                        best = a.copy()
        except Exception:
            pass
    if best is None:
        best = rng.standard_normal((n, d))
        best /= np.linalg.norm(best, axis=1, keepdims=True)
        best_s = _max_cos(best)

    cur = best.copy()
    cur_s = best_s
    cur_fp = _fingerprint(cur)
    archive = [(best_s, cur_fp.copy())]

    def is_new_basin(fp, tol=8e-4):
        md = 1e9
        for _, af in archive:
            dd = _hist_dist(fp, af)
            if dd < md:
                md = dd
        return md, md > tol

    step = 0
    # initial short global polish of random fresh points to seed diversity (cheap)
    # then main kick-hole hopping
    while time.time() < HOP_END:
        step += 1
        remain = HOP_END - time.time()
        if remain < 3.0:
            break
        # temperature: allow uphill early, greedy late
        prog = (time.time() - t0) / max(1e-9, (HOP_END - t0))
        temp = 0.004 * (1.0 - prog) + 0.0002
        # choose kick size
        r = rng.random()
        if r < 0.35:
            k = 2
        elif r < 0.65:
            k = 3
        elif r < 0.88:
            k = 4
        else:
            k = 5
        # choose victims: worst per-max mostly, sometimes random/uniform
        pm = _per_max(cur)
        order = np.argsort(-pm)
        victims = []
        if rng.random() < 0.75:
            # worst-biased
            nw = min(n, k + 6)
            cand_pool = order[:nw].tolist()
            # sample k from pool weighted to worst
            perm = rng.permutation(len(cand_pool))
            victims = [cand_pool[i] for i in perm[:k]]
            if rng.random() < 0.25 and k >= 2:
                victims[-1] = int(rng.integers(0, n))
        else:
            victims = rng.choice(n, size=k, replace=False).tolist()
        victims = sorted(victims)
        keep_mask = np.ones(n, dtype=bool)
        keep_mask[victims] = False

        Xt = cur.copy()
        mode = step % 3  # 0: hole, 1: hole+noise, 2: random
        if mode == 2 and rng.random() < 0.5:
            # pure random reinsert for strong diversification
            for v in victims:
                z = rng.standard_normal(d)
                z /= (np.linalg.norm(z) + 1e-300)
                Xt[v] = z
        else:
            # sequential hole reinsertion
            placed = []
            for idx_vi, v in enumerate(victims):
                Xr_list = [cur[keep_mask]]
                if placed:
                    Xr_list.append(np.stack(placed, axis=0))
                Xr = np.concatenate(Xr_list, axis=0) if len(Xr_list) > 1 else Xr_list[0]
                nc = 12000 if remain > 60 else 8000
                h = _best_hole(Xr, rng, ncand=nc, refine=True)
                if mode == 1:
                    h = h + 0.08 * rng.standard_normal(d)
                    h /= (np.linalg.norm(h) + 1e-300)
                Xt[v] = h
                placed.append(Xt[v].copy())
            # small global shake to break exact contacts
            sig = 0.015 if rng.random() < 0.7 else 0.05
            Xt = Xt + sig * rng.standard_normal((n, d))
            Xt /= (np.linalg.norm(Xt, axis=1, keepdims=True) + 1e-300)

        # staged re-optimization: short if far from best, longer if promising start
        # quick estimate: maxcos of Xt before opt
        try:
            s0 = _max_cos(Xt)
        except Exception:
            continue
        if s0 > cur_s + 0.08:
            betas = [400.0, 1500.0]
            iters = [60, 80]
        else:
            betas = [400.0, 1500.0, 5000.0]
            iters = [60, 90, 90]
        # reduce iters when time short
        if remain < 25:
            iters = [int(max(30, x * 0.6)) for x in iters]
        try:
            Xn = _lse_opt(Xt, betas, iters)
        except Exception:
            continue
        sn = _max_cos(Xn)
        fp = _fingerprint(Xn)
        md, newb = is_new_basin(fp, tol=6e-4)
        archive.append((sn, fp.copy()))
        if len(archive) > 400:
            archive = archive[-400:]

        improved_best = sn < best_s - 1e-12
        if improved_best:
            best, best_s = Xn.copy(), sn
            cur, cur_s, cur_fp = Xn.copy(), sn, fp.copy()
            continue
        # acceptance: greedy if same basin, tempered if new basin
        if newb:
            if sn < cur_s + temp:
                cur, cur_s, cur_fp = Xn.copy(), sn, fp.copy()
            elif rng.random() < 0.02 and sn < cur_s + 0.01:
                cur, cur_s, cur_fp = Xn.copy(), sn, fp.copy()
        else:
            if sn < cur_s - 1e-12:
                cur, cur_s, cur_fp = Xn.copy(), sn, fp.copy()
        # occasional restart from best with fresh kick if stuck
        if step % 12 == 0 and time.time() < HOP_END - 10:
            if cur_s > best_s + 0.002:
                cur = best.copy(); cur_s = best_s; cur_fp = _fingerprint(cur)
        # occasional fresh random start injection
        if step % 15 == 0 and time.time() < HOP_END - 20:
            Xr0 = rng.standard_normal((n, d))
            Xr0 /= (np.linalg.norm(Xr0, axis=1, keepdims=True) + 1e-300)
            try:
                Xr1 = _lse_opt(Xr0, [400.0, 1500.0], [50, 60])
                sr1 = _max_cos(Xr1)
                fp1 = _fingerprint(Xr1)
                archive.append((sr1, fp1.copy()))
                if sr1 < cur_s + temp and _hist_dist(fp1, cur_fp) > 6e-4:
                    cur, cur_s, cur_fp = Xr1, sr1, fp1
                if sr1 < best_s:
                    best, best_s = Xr1.copy(), sr1
            except Exception:
                pass

    # final intensification: re-optimize best + cur with high beta then LP
    cands = [(best_s, best)]
    if cur_s < best_s + 0.005:
        cands.append((cur_s, cur.copy()))
    # also try averaging best with cur under alignment? skip to save time; just polish
    polished = []
    for s0, X0 in cands:
        if time.time() > T_END - 12.0:
            polished.append((s0, X0))
            continue
        try:
            X1 = _lse_opt(X0, [1000.0, 4000.0, 12000.0], [90, 90, 70])
        except Exception:
            X1 = X0
        s1 = _max_cos(X1)
        polished.append((s1, X1))
    polished.sort(key=lambda z: z[0])
    Xf, sf = polished[0][1], polished[0][0]
    # LP squeeze
    try:
        Xf = _lp_polish(Xf, min(T_END - 1.0, time.time() + 12.0))
        sf = _max_cos(Xf)
    except Exception:
        pass
    # safety: return best seen
    if sf > best_s and best is not None:
        # check polished best vs original best polished? keep min
        if best_s < sf:
            Xf = best
            # quick LP on best if time
            if time.time() < T_END - 4.0:
                try:
                    Xf = _lp_polish(best, T_END - 0.5)
                except Exception:
                    Xf = best
    Xf = Xf / np.linalg.norm(Xf, axis=1, keepdims=True)
    return Xf
