import time
import numpy as np
from scipy.optimize import linprog, minimize
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _contacts(x, tol=1e-9):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    m = float(G.max())
    return G, m, G > m - tol

def _contact_set(x, tol=1e-9):
    n = x.shape[0]
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    m = float(G.max())
    iu0, iu1 = np.triu_indices(n, 1)
    c = G[iu0, iu1]
    sel = c > m - tol
    s = set((int(i) * n + int(j)) for i, j in zip(iu0[sel], iu1[sel]))
    return s, m

def _jaccard(a, b):
    if not a and not b:
        return 0.0
    u = len(a | b)
    if u == 0:
        return 0.0
    return 1.0 - len(a & b) / u

def _lp_polish(x, deadline, r0=0.003, cap=1300):
    x = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
    n, d = x.shape
    try:
        best = _max_cos(x)
    except Exception:
        return x, 1.0
    radius = r0
    iu0, iu1 = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        try:
            G = x @ x.T
            c = G[iu0, iu1]
            m = float(c.max())
            thresh = max(20.0 * radius, 1e-9)
            sel = c > m - thresh
            cnt = int(np.count_nonzero(sel))
            if cnt == 0:
                sel[np.argmax(c)] = True
                cnt = 1
            if cnt > cap:
                idx = np.argsort(-c)[:cap]
                sel2 = np.zeros_like(sel, dtype=bool)
                sel2[idx] = True
                sel = sel2
            I = iu0[sel]; J = iu1[sel]
            csel = c[sel]
            p = len(I)
            nv = n * d + 1
            rows = np.repeat(np.arange(p), 2 * d + 1)
            cols = np.empty((p, 2 * d + 1), dtype=int)
            ar = np.arange(d)
            cols[:, :d] = I[:, None] * d + ar
            cols[:, d:2 * d] = J[:, None] * d + ar
            cols[:, 2 * d] = n * d
            cols = cols.ravel()
            vals = np.empty((p, 2 * d + 1), dtype=float)
            vals[:, :d] = x[J]
            vals[:, d:2 * d] = x[I]
            vals[:, 2 * d] = -1.0
            vals = vals.ravel()
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv); obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
            if res.status != 0:
                radius *= 0.5
                continue
            y = x + res.x[:-1].reshape(n, d)
            y /= np.maximum(np.linalg.norm(y, axis=1, keepdims=True), 1e-12)
            v = _max_cos(y)
            if v < best:
                x, best = y, v
                radius = min(radius * 1.5, 0.05)
            else:
                radius *= 0.5
        except Exception:
            radius *= 0.5
            continue
    return x, best

def _anneal(x0, steps, T0, T1, lr0, mu, rng, noise0=0.0, deadline=None, active=None):
    x = x0.copy()
    n, d = x.shape
    if active is None:
        act = np.ones(n, dtype=bool)
    else:
        act = np.asarray(active, dtype=bool)
    nact = int(act.sum())
    if nact == 0:
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        return x.copy(), float(G.max())
    vel = np.zeros_like(x)
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    bv = float(G.max())
    bx = x.copy()
    if T0 <= 0 or T1 <= 0:
        T0 = max(T0, 1e-12); T1 = max(T1, 1e-12)
        lratio = 0.0
    else:
        lratio = np.log(T1 / T0)
    for k in range(steps):
        if deadline is not None and (k & 127) == 0 and time.time() > deadline:
            break
        frac = k / max(1, steps - 1)
        T = T0 * np.exp(lratio * frac)
        if T < 5e-5:
            T = 5e-5
        lr = lr0 * (1.0 - 0.55 * frac)
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        E = np.exp((G - m) / T)
        np.fill_diagonal(E, 0.0)
        S = float(E.sum() * 0.5)
        if not np.isfinite(S) or S < 1e-300:
            S = 1e-300
        Grad = (E @ x) / S
        Grad -= (Grad * x).sum(axis=1, keepdims=True) * x
        if nact != n:
            Grad[~act] = 0.0
        vel = mu * vel + (1.0 - mu) * Grad
        if nact != n:
            vel[~act] = 0.0
        x[act] = x[act] - lr * vel[act]
        if noise0 > 0:
            amp = noise0 * (1.0 - frac)
            if amp > 0:
                nz = rng.randn(nact, d) * amp
                xa = x[act]
                nz -= (nz * xa).sum(axis=1, keepdims=True) * xa
                x[act] = xa + nz
        xn = np.linalg.norm(x[act], axis=1, keepdims=True)
        xn = np.maximum(xn, 1e-12)
        x[act] /= xn
        if m < bv:
            bv = m
            bx = x.copy()
    return bx, bv

def _lbfgs_smooth(x_init, deadline, taus=(0.004, 0.0012, 0.0005), iters=(80, 80, 100)):
    x = x_init / np.maximum(np.linalg.norm(x_init, axis=1, keepdims=True), 1e-12)
    n, d = x.shape
    iu0, iu1 = np.triu_indices(n, 1)
    Np = len(iu0)
    y = x.ravel().copy()
    for tau, maxiter in zip(taus, iters):
        if time.time() > deadline - 0.4:
            break
        def fg(yflat, _tau=tau):
            Y = yflat.reshape(n, d)
            rn = np.maximum(np.linalg.norm(Y, axis=1, keepdims=True), 1e-12)
            X = Y / rn
            g = (X @ X.T)[iu0, iu1]
            m = float(g.max())
            e = np.exp((g - m) / _tau)
            s = float(e.sum())
            if s <= 0 or not np.isfinite(s):
                return float(m), np.zeros_like(yflat)
            L = m + _tau * np.log(s / Np)
            w = e / s
            W = np.zeros((n, n))
            W[iu0, iu1] = w
            W = W + W.T
            GX = W @ X
            GY = (GX - (GX * X).sum(axis=1, keepdims=True) * X) / rn
            return float(L), GY.ravel()
        try:
            res = minimize(fg, y, method="L-BFGS-B", jac=True,
                           options={"maxiter": int(maxiter), "maxfun": int(maxiter * 4),
                                    "ftol": 1e-14, "gtol": 1e-9})
            y = res.x
        except Exception:
            break
    Yf = y.reshape(n, d)
    Yf /= np.maximum(np.linalg.norm(Yf, axis=1, keepdims=True), 1e-12)
    return Yf

def _refine_hole(x_fixed, y, iters=160):
    y = y / max(float(np.linalg.norm(y)), 1e-12)
    vy = float((x_fixed @ y).max())
    for _ in range(iters):
        dots = x_fixed @ y
        j = int(np.argmax(dots))
        g = x_fixed[j] - dots[j] * y
        ng = float(np.linalg.norm(g))
        if ng < 1e-12:
            break
        g /= ng
        improved = False
        for step in (0.03, 0.01, 0.003, 0.001):
            yn = y - step * g
            nyn = float(np.linalg.norm(yn))
            if nyn < 1e-12:
                continue
            yn /= nyn
            vn = float((x_fixed @ yn).max())
            if vn < vy:
                y, vy = yn, vn
                improved = True
                break
        if not improved:
            break
    return y, vy

def solve():
    t0 = time.time()
    DEADLINE = t0 + 172.0
    SEARCH_END = t0 + 118.0
    rng = np.random.RandomState(20261102)
    n = 163
    best_x = None
    best_val = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=np.float64)
            if a.shape != (n, 4) or not np.all(np.isfinite(a)):
                continue
            a /= np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-12)
            v = _max_cos(a)
            if v < best_val:
                best_val = v
                best_x = a.copy()
        except Exception:
            continue
    if best_x is None:
        best_x = rng.randn(n, 4)
        best_x /= np.linalg.norm(best_x, axis=1, keepdims=True)
        best_val = _max_cos(best_x)
    best_set, _ = _contact_set(best_x)
    elite = [(best_val, best_x.copy(), best_set)]
    kvals = [10, 6, 14, 18]
    trial = 0
    while time.time() < SEARCH_END and time.time() < DEADLINE - 50:
        trial += 1
        k = kvals[trial % len(kvals)]
        # occasionally vary k by +-2 for diversity
        if rng.rand() < 0.3:
            k = int(np.clip(k + rng.choice([-2, 2]), 4, 20))
        try:
            G = best_x @ best_x.T
            np.fill_diagonal(G, -2.0)
            m0 = float(G.max())
            deg = (G > m0 - 1e-9).sum(axis=1)
            # stress score: degree + tie-break by worst neighbour cosine sum (proxy for dual)
            worst = G.max(axis=1)
            order = np.lexsort((worst, deg))[::-1]
            pool = order[:max(2 * k, k + 6)]
            if len(pool) < k:
                del_idx = order[:k]
            else:
                sel = rng.choice(len(pool), size=k, replace=False)
                del_idx = pool[sel]
            keep_mask = np.ones(n, dtype=bool)
            keep_mask[del_idx] = False
            fixed = best_x[keep_mask].copy()
            # polish thinned code well below target
            fixed_b, _ = _anneal(fixed, steps=1100, T0=0.007, T1=0.00045, lr0=0.035, mu=0.86, rng=rng, noise0=0.0015, deadline=SEARCH_END)
            # short LP on thinned set
            fixed_b, vf = _lp_polish(fixed_b, min(SEARCH_END, time.time() + 1.2), r0=0.004)
            cur = fixed_b.copy()
            # sequential greedy reinsertion
            ok = True
            for ti in range(k):
                if time.time() > SEARCH_END:
                    ok = False
                    break
                # 20k uniform candidates
                Y = rng.randn(20000, 4)
                Y /= np.maximum(np.linalg.norm(Y, axis=1, keepdims=True), 1e-12)
                Dmax = (Y @ cur.T).max(axis=1)
                idx20 = np.argpartition(Dmax, 20)[:20]
                by = None
                bv = 1e9
                for ii in idx20:
                    yr, vr = _refine_hole(cur, Y[ii])
                    if vr < bv:
                        bv = vr
                        by = yr
                cur = np.vstack([cur, by.reshape(1, 4)])
                # local relax: inserted + neighbours > 0.75
                last = cur[-1]
                dots = cur[:-1] @ last
                neigh = np.where(dots > 0.75)[0]
                act = np.zeros(len(cur), dtype=bool)
                act[-1] = True
                act[neigh] = True
                cur, _ = _anneal(cur, steps=550, T0=0.004, T1=0.0004, lr0=0.030, mu=0.85, rng=rng, noise0=0.0008, deadline=SEARCH_END, active=act)
            if not ok:
                continue
            # light global settle
            cur2, v2 = _anneal(cur, steps=700, T0=0.0032, T1=0.0004, lr0=0.030, mu=0.86, rng=rng, noise0=0.0, deadline=SEARCH_END)
            vcur = _max_cos(cur2)
            if vcur > best_val + 0.0025:
                # stalled far above target: restart with different order next trial
                continue
            cset, _ = _contact_set(cur2)
            jd = _jaccard(best_set, cset)
            do_full = (jd > 0.1)
            if do_full:
                # LBFGS + LP only for topologically novel candidates
                lb_dead = min(SEARCH_END + 6, DEADLINE - 12)
                cur3 = _lbfgs_smooth(cur2, lb_dead)
                v3 = _max_cos(cur3)
                # keep better of cur2/cur3 for LP
                lp_in = cur3 if v3 < vcur else cur2
                lp_dead = min(time.time() + 4.0, DEADLINE - 10)
                if time.time() < SEARCH_END + 5:
                    lp_out, vlp = _lp_polish(lp_in, lp_dead, r0=0.0025)
                else:
                    lp_out, vlp = lp_in, _max_cos(lp_in)
                # update
                if vlp < best_val - 1e-12:
                    best_val = vlp
                    best_x = (lp_out / np.linalg.norm(lp_out, axis=1, keepdims=True)).copy()
                    best_set = _contact_set(best_x)[0]
                    elite.append((best_val, best_x.copy(), best_set))
                    if len(elite) > 6:
                        elite = sorted(elite, key=lambda z: z[0])[:6]
                elif jd > 0.25 and len(elite) < 6:
                    # keep distinct archive member even if not better
                    elite.append((vlp, lp_out.copy(), cset))
            else:
                if vcur < best_val:
                    best_val = vcur
                    best_x = (cur2 / np.linalg.norm(cur2, axis=1, keepdims=True)).copy()
                    best_set = cset
                    elite.append((best_val, best_x.copy(), best_set))
        except Exception:
            continue
    # final intensive polish of incumbent
    try:
        if time.time() < DEADLINE - 3:
            bx = _lbfgs_smooth(best_x, DEADLINE - 6)
            if _max_cos(bx) < best_val:
                best_x = bx
                best_val = _max_cos(bx)
        if time.time() < DEADLINE - 2:
            best_x, best_val2 = _lp_polish(best_x, DEADLINE - 1.0, r0=0.002)
            best_val = min(best_val, best_val2)
    except Exception:
        pass
    best_x = best_x / np.maximum(np.linalg.norm(best_x, axis=1, keepdims=True), 1e-12)
    return best_x
