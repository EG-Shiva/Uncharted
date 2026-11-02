import time
import numpy as np
from scipy.optimize import linprog, minimize
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def solve():
    t0 = time.time()
    DEADLINE = t0 + 171.0
    FINAL_RESERVE = 25.0
    n, d8, d9 = 121, 8, 9
    iu = np.triu_indices(n, 1)
    Iall, Jall = iu[0], iu[1]

    cands = []
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn).astype(float)
            if a.shape == (n, d8):
                a = a / np.linalg.norm(a, axis=1, keepdims=True)
                if np.all(np.isfinite(a)):
                    cands.append(a)
        except Exception:
            pass
    if not cands:
        rng0 = np.random.default_rng(0)
        X0 = rng0.standard_normal((n, d8))
        X0 /= np.linalg.norm(X0, axis=1, keepdims=True)
        cands = [X0]
    vals = [_max_cos(a) for a in cands]
    Xbest = cands[int(np.argmin(vals))].copy()
    Vbest = float(min(vals))
    Xrec = cands[0].copy()

    # ---------- helpers ----------
    def make_9d_fun(beta, lam):
        b = float(beta); l = float(lam)
        def fun(flat):
            X = flat.reshape(n, d9)
            nr = np.linalg.norm(X, axis=1, keepdims=True)
            nr = np.maximum(nr, 1e-12)
            Y = X / nr
            G = Y @ Y.T
            c = G[iu]
            m = float(c.max())
            e = np.exp(b * (c - m))
            s = float(e.sum())
            if not np.isfinite(s) or s <= 0:
                return float(m), np.zeros_like(flat)
            f = m + np.log(s) / b
            w = e / s
            W = np.zeros((n, n))
            W[Iall, Jall] = w
            W[Jall, Iall] = w
            GY = W @ Y
            dot = (GY * Y).sum(axis=1, keepdims=True)
            GX = (GY - dot * Y) / nr
            if l != 0.0:
                z = Y[:, 8:9]  # (n,1)
                f += l * float((z ** 2).mean())
                GP = np.zeros_like(Y)
                GP[:, 8:9] = (2.0 * l / float(n)) * z
                dotp = (GP * Y).sum(axis=1, keepdims=True)
                GX += (GP - dotp * Y) / nr
            return float(f), GX.ravel()
        return fun

    def make_8d_lse(beta):
        b = float(beta)
        def fun(flat):
            X = flat.reshape(n, d8)
            nr = np.linalg.norm(X, axis=1, keepdims=True)
            nr = np.maximum(nr, 1e-12)
            Y = X / nr
            G = Y @ Y.T
            c = G[iu]
            m = float(c.max())
            e = np.exp(b * (c - m))
            s = float(e.sum())
            if not np.isfinite(s) or s <= 0:
                return float(m), np.zeros_like(flat)
            f = m + np.log(s) / b
            w = e / s
            W = np.zeros((n, n))
            W[Iall, Jall] = w
            W[Jall, Iall] = w
            GY = W @ Y
            dot = (GY * Y).sum(axis=1, keepdims=True)
            GX = (GY - dot * Y) / nr
            return float(f), GX.ravel()
        return fun

    def make_8d_hinge(t):
        tt = float(t)
        def fun(flat):
            X = flat.reshape(n, d8)
            nr = np.linalg.norm(X, axis=1, keepdims=True)
            nr = np.maximum(nr, 1e-12)
            Y = X / nr
            G = Y @ Y.T
            c = G[iu]
            diff = c - tt
            pos = diff > 0
            if np.any(pos):
                f = float((diff[pos] ** 2).sum())
                w = np.zeros_like(c)
                w[pos] = 2.0 * diff[pos]
            else:
                f = 0.0
                w = np.zeros_like(c)
            W = np.zeros((n, n))
            W[Iall, Jall] = w
            W[Jall, Iall] = w
            GY = W @ Y
            dot = (GY * Y).sum(axis=1, keepdims=True)
            GX = (GY - dot * Y) / nr
            return float(f), GX.ravel()
        return fun

    def run_lbfgs(fun, Xf, maxiter):
        try:
            res = minimize(fun, Xf, jac=True, method="L-BFGS-B",
                           options={"maxiter": int(maxiter), "ftol": 1e-20,
                                    "gtol": 1e-14, "maxfun": int(maxiter * 8)})
            return np.asarray(res.x, dtype=float)
        except Exception:
            return Xf

    def lp_polish(x, seconds, radius_init=3e-4):
        start = time.time()
        x = x / np.linalg.norm(x, axis=1, keepdims=True)
        best = _max_cos(x)
        radius = radius_init
        it = 0
        while time.time() - start < seconds and time.time() < DEADLINE - 1.0:
            it += 1
            if it > 200 or radius < 1e-14:
                break
            G = x @ x.T
            cf = G[iu]
            m = float(cf.max())
            thr = max(20.0 * radius, 2e-9)
            sel = cf > m - thr
            cnt = int(sel.sum())
            if cnt == 0:
                radius *= 0.5
                continue
            if cnt > 2500:
                kth = np.sort(cf)[-2500]
                sel = cf >= kth
                cnt = int(sel.sum())
                if cnt > 3000:
                    idxp = np.argpartition(-cf, 3000)[:3000]
                    mk = np.zeros_like(sel)
                    mk[idxp] = True
                    sel = mk
                    cnt = 3000
            c = cf[sel]
            I = iu[0][sel]; J = iu[1][sel]
            p = cnt
            nv = n * d8 + 1
            rows = np.repeat(np.arange(p), 2 * d8 + 1)
            cols = np.concatenate([I[:, None] * d8 + np.arange(d8),
                                   J[:, None] * d8 + np.arange(d8),
                                   np.full((p, 1), n * d8)], 1).ravel()
            valsA = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
            try:
                A_ub = coo_matrix((valsA, (rows, cols)), shape=(p, nv)).tocsr()
                A_eq = coo_matrix((x.ravel(),
                                   (np.repeat(np.arange(n), d8), np.arange(n * d8))),
                                  shape=(n, nv)).tocsr()
            except Exception:
                radius *= 0.5
                continue
            obj = np.zeros(nv); obj[-1] = 1.0
            try:
                res = linprog(obj, A_ub=A_ub, b_ub=-c, A_eq=A_eq,
                              b_eq=np.zeros(n),
                              bounds=[(-radius, radius)] * (n * d8) + [(None, None)],
                              method="highs")
            except Exception:
                radius *= 0.5
                continue
            if res.status != 0:
                radius *= 0.5
                continue
            y = x + res.x[:-1].reshape(n, d8)
            yn = np.linalg.norm(y, axis=1, keepdims=True)
            yn[yn == 0] = 1.0
            y = y / yn
            v = _max_cos(y)
            if v < best:
                x, best = y, v
                radius = min(radius * 1.6, 0.02)
            else:
                radius *= 0.5
        return x, best

    # ---------- build 9D lifts ----------
    rng = np.random.default_rng(7)
    lifts = []
    # lift 0: incumbent + tiny z
    z0 = 0.05 * rng.standard_normal((n, 1))
    X9_0 = np.hstack([Xrec, z0])
    X9_0 /= np.linalg.norm(X9_0, axis=1, keepdims=True)
    lifts.append(("inc_small", X9_0))
    # lift 1: incumbent + larger z + 8D noise
    rng1 = np.random.default_rng(11)
    z1 = 0.18 * rng1.standard_normal((n, 1))
    N1 = 0.03 * rng1.standard_normal((n, d8))
    X9_1 = np.hstack([Xrec + N1, z1])
    X9_1 /= np.linalg.norm(X9_1, axis=1, keepdims=True)
    lifts.append(("inc_big", X9_1))
    # lift 2: best (maybe same) with random-sign z
    rng2 = np.random.default_rng(23)
    z2 = 0.10 * rng2.standard_normal((n, 1))
    X9_2 = np.hstack([Xbest + 0.01 * rng2.standard_normal((n, d8)), z2])
    X9_2 /= np.linalg.norm(X9_2, axis=1, keepdims=True)
    lifts.append(("best_mix", X9_2))
    # lift 3,4: fresh random 9D
    for si, seed in enumerate([101, 202]):
        r = np.random.default_rng(seed)
        Xr = r.standard_normal((n, d9))
        Xr /= np.linalg.norm(Xr, axis=1, keepdims=True)
        lifts.append((f"rand{si}", Xr))

    lam_schedule = [10.0, 30.0, 100.0, 300.0, 1000.0]

    for bidx, (bname, X9init) in enumerate(lifts):
        if time.time() > DEADLINE - FINAL_RESERVE - 2.0:
            break
        try:
            Xf = X9init.ravel().copy()
            # free 9D LogSumExp reorganization
            for beta, mi in [(2000.0, 60), (12000.0, 80)]:
                if time.time() > DEADLINE - FINAL_RESERVE - 1.0:
                    break
                fun = make_9d_fun(beta, 0.0)
                Xf = run_lbfgs(fun, Xf, mi)
            # penalized flattening
            for li, lam in enumerate(lam_schedule):
                if time.time() > DEADLINE - FINAL_RESERVE - 1.0:
                    break
                beta = 12000.0 if li < 2 else 20000.0
                fun = make_9d_fun(beta, lam)
                Xf = run_lbfgs(fun, Xf, 70)
                # early check of z energy
                try:
                    Xt = Xf.reshape(n, d9)
                    Xt = Xt / np.linalg.norm(Xt, axis=1, keepdims=True)
                    zrms = float(np.sqrt((Xt[:, 8] ** 2).mean()))
                except Exception:
                    zrms = 1.0
                # if z already tiny and lam large, keep going (need topology lock)
                if zrms > 0.08 and lam >= 300.0:
                    # badly unflattenable, will likely discard; still try one more
                    pass
            X9 = Xf.reshape(n, d9)
            X9 /= np.linalg.norm(X9, axis=1, keepdims=True)
            zrms = float(np.sqrt((X9[:, 8] ** 2).mean()))
            zmax = float(np.abs(X9[:, 8]).max())
            # project to R8
            X8p = X9[:, :8].copy()
            nr8 = np.linalg.norm(X8p, axis=1, keepdims=True)
            nr8 = np.maximum(nr8, 1e-12)
            X8p = X8p / nr8
            Vp = _max_cos(X8p)
            # discard if unflattenable
            if not np.isfinite(Vp) or Vp > 0.462:
                continue
            # hinge descent from projected value toward 0.4595
            # build descending targets
            t_low = 0.4595
            # 3-4 targets interpolating
            t_start = Vp - 0.0002
            if t_start < t_low:
                targets = [t_low]
            else:
                targets = [t_start, max((t_start + t_low) / 2.0, t_low), t_low]
                # ensure strictly descending and below Vp
                targets = sorted(set([float(t) for t in targets]), reverse=True)
            Xc = X8p.copy()
            for ti, tt in enumerate(targets):
                if time.time() > DEADLINE - FINAL_RESERVE - 1.0:
                    break
                if tt >= _max_cos(Xc) - 1e-12:
                    # already below target, push lower
                    pass
                funh = make_8d_hinge(tt)
                Xfh = run_lbfgs(funh, Xc.ravel().copy(), 90)
                Xn = Xfh.reshape(n, d8)
                Xn /= np.linalg.norm(Xn, axis=1, keepdims=True)
                # accept if not exploding
                vn = _max_cos(Xn)
                if vn < _max_cos(Xc) + 0.0015:
                    Xc = Xn
                # if feasible, continue to next lower target
            # smooth 8D LogSumExp polish
            for beta, mi in [(5000.0, 60), (20000.0, 70)]:
                if time.time() > DEADLINE - FINAL_RESERVE - 1.0:
                    break
                fun8 = make_8d_lse(beta)
                Xf8 = run_lbfgs(fun8, Xc.ravel().copy(), mi)
                Xn = Xf8.reshape(n, d8)
                Xn /= np.linalg.norm(Xn, axis=1, keepdims=True)
                vn = _max_cos(Xn)
                if vn < _max_cos(Xc) + 0.0005:
                    Xc = Xn
            Vc = _max_cos(Xc)
            if Vc < Vbest - 1e-12:
                # quick light LP to confirm
                Xc2, Vc2 = lp_polish(Xc, seconds=1.0, radius_init=2e-4)
                if Vc2 < Vc:
                    Xc, Vc = Xc2, Vc2
                Xbest, Vbest = Xc, Vc
            elif Vc < Vbest + 0.0008 and time.time() < DEADLINE - FINAL_RESERVE - 8.0:
                Xc2, Vc2 = lp_polish(Xc, seconds=0.8, radius_init=2e-4)
                if Vc2 < Vbest - 1e-12:
                    Xbest, Vbest = Xc2, Vc2
        except Exception:
            continue

    # final trust-region LP polishing of best flattened branch
    rem = DEADLINE - time.time() - 1.0
    if rem > 2.0:
        try:
            Xbest, Vbest = lp_polish(Xbest, seconds=min(rem, 30.0), radius_init=3e-4)
        except Exception:
            pass
    Xbest = Xbest / np.linalg.norm(Xbest, axis=1, keepdims=True)
    return Xbest
