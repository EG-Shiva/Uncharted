import time
import numpy as np
from scipy.optimize import linprog, minimize
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _anneal_full(x0, steps, T0, T1, lr0, mu, rng, noise0=0.0, deadline=None):
    x = x0.copy()
    n, d = x.shape
    vel = np.zeros_like(x)
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    bv = float(G.max())
    bx = x.copy()
    lratio = np.log(T1 / T0) if T0 > 0 and T1 > 0 else 0.0
    for k in range(steps):
        if deadline is not None and (k & 127) == 0 and time.time() > deadline:
            break
        frac = k / max(1, steps - 1)
        T = T0 * np.exp(lratio * frac)
        if T < 8e-5:
            T = 8e-5
        lr = lr0 * (1.0 - 0.55 * frac)
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        E = np.exp((G - m) / T)
        np.fill_diagonal(E, 0.0)
        S = float(E.sum() * 0.5)
        if not np.isfinite(S) or S < 1e-300:
            S = 1e-300
        grad = (E @ x) / S
        grad -= (grad * x).sum(axis=1, keepdims=True) * x
        vel = mu * vel + (1.0 - mu) * grad
        x = x - lr * vel
        if noise0 > 0:
            amp = noise0 * (1.0 - frac)
            if amp > 0:
                nz = rng.randn(n, d) * amp
                nz -= (nz * x).sum(axis=1, keepdims=True) * x
                x = x + nz
        x /= np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
        if m < bv:
            bv = m
            bx = x.copy()
    return bx, bv

def _anneal_subset(x0, mobile, steps, T0, T1, lr0, mu, rng, noise0=0.0, deadline=None):
    x = x0.copy()
    n, d = x.shape
    vel = np.zeros_like(x)
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    bv = float(G.max())
    bx = x.copy()
    lratio = np.log(T1 / T0) if T0 > 0 and T1 > 0 else 0.0
    mob = mobile.astype(bool)
    nmob = int(mob.sum())
    if nmob == 0:
        return bx, bv
    for k in range(steps):
        if deadline is not None and (k & 127) == 0 and time.time() > deadline:
            break
        frac = k / max(1, steps - 1)
        T = T0 * np.exp(lratio * frac)
        if T < 8e-5:
            T = 8e-5
        lr = lr0 * (1.0 - 0.55 * frac)
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        E = np.exp((G - m) / T)
        np.fill_diagonal(E, 0.0)
        S = float(E.sum() * 0.5)
        if not np.isfinite(S) or S < 1e-300:
            S = 1e-300
        grad = (E @ x) / S
        grad -= (grad * x).sum(axis=1, keepdims=True) * x
        grad[~mob] = 0.0
        vel[~mob] = 0.0
        vel[mob] = mu * vel[mob] + (1.0 - mu) * grad[mob]
        x[mob] = x[mob] - lr * vel[mob]
        if noise0 > 0:
            amp = noise0 * (1.0 - frac)
            if amp > 0:
                xm = x[mob]
                nz = rng.randn(nmob, d) * amp
                nz -= (nz * xm).sum(axis=1, keepdims=True) * xm
                x[mob] = xm + nz
        xn = np.linalg.norm(x[mob], axis=1, keepdims=True)
        x[mob] /= np.maximum(xn, 1e-12)
        if m < bv:
            bv = m
            bx = x.copy()
    return bx, bv

def _lbfgs_smooth(x_init, deadline, taus=(0.004, 0.0012, 0.00045), iters=(90, 90, 120)):
    x = x_init / np.maximum(np.linalg.norm(x_init, axis=1, keepdims=True), 1e-12)
    n, d = x.shape
    iu0, iu1 = np.triu_indices(n, 1)
    Np = len(iu0)
    y = x.ravel().copy()
    for tau, maxiter in zip(taus, iters):
        if time.time() > deadline - 0.35:
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
    return Yf, _max_cos(Yf)

def _lp_polish(x, deadline, r0=1e-3, cap=1400):
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
            sel = c > m - max(20.0 * radius, 1e-9)
            cnt = int(np.count_nonzero(sel))
            if cnt == 0:
                sel[int(np.argmax(c))] = True
                cnt = 1
            if cnt > cap:
                idx = np.argsort(-c)[:cap]
                s2 = np.zeros_like(sel, dtype=bool)
                s2[idx] = True
                sel = s2
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
            vals = np.empty((p, 2 * d + 1), dtype=float)
            vals[:, :d] = x[J]
            vals[:, d:2 * d] = x[I]
            vals[:, 2 * d] = -1.0
            A_ub = coo_matrix((vals.ravel(), (rows, cols.ravel())), shape=(p, nv)).tocsr()
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

def _find_hole(x_fixed, rng, n_samples=7000):
    Y = rng.randn(n_samples, 4)
    Y /= np.maximum(np.linalg.norm(Y, axis=1, keepdims=True), 1e-12)
    D = Y @ x_fixed.T
    vals = D.max(axis=1)
    idx = int(np.argmin(vals))
    y = Y[idx].copy()
    vy = float(vals[idx])
    for _ in range(160):
        dots = x_fixed @ y
        j = int(np.argmax(dots))
        g = x_fixed[j] - dots[j] * y
        ng = float(np.linalg.norm(g))
        if ng < 1e-12:
            break
        improved = False
        for step in (0.03, 0.01, 0.003):
            yn = y - step * g / ng
            nyn = np.linalg.norm(yn)
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

def _collective_move(base, rng, amp):
    n = base.shape[0]
    M = rng.randn(4, 4)
    M = (M + M.T) * 0.5
    M -= np.trace(M) / 4.0 * np.eye(4)
    A = rng.randn(4, 4)
    A = (A - A.T) * 0.5
    a1 = rng.rand()
    Mx = base @ M.T
    proj = (Mx * base).sum(axis=1, keepdims=True)
    VM = Mx - proj * base
    VA = base @ A.T
    V = a1 * VM + (1.0 - a1) * VA
    nv = np.linalg.norm(V, axis=1, keepdims=True)
    med = np.median(nv) + 1e-12
    V = V / med * amp
    cur = base + V
    cur /= np.maximum(np.linalg.norm(cur, axis=1, keepdims=True), 1e-12)
    return cur

def _rand_rot(rng, scale):
    A = rng.randn(4, 4) * scale
    A = A - A.T
    Q, _ = np.linalg.qr(np.eye(4) + A)
    return Q

def solve():
    t0 = time.time()
    DEADLINE = t0 + 172.0
    SEARCH_END = t0 + 126.0
    rng = np.random.RandomState(20261102)
    n = 163
    try:
        bx = np.load("best.npy")
    except Exception:
        try:
            bx = np.load("record.npy")
        except Exception:
            bx = rng.randn(n, 4)
    bx = np.asarray(bx, dtype=float)
    if bx.shape != (n, 4) or not np.all(np.isfinite(bx)):
        bx = rng.randn(n, 4)
    bx = bx / np.maximum(np.linalg.norm(bx, axis=1, keepdims=True), 1e-12)
    best_x = bx.copy()
    best_val = _max_cos(best_x)
    archive = [(best_val, best_x.copy())]

    def push(x, v):
        nonlocal best_x, best_val
        archive.append((v, x.copy()))
        archive.sort(key=lambda t: t[0])
        del archive[6:]
        if v < best_val - 1e-12:
            best_val = v
            best_x = x.copy()

    trial = 0
    while time.time() < SEARCH_END:
        if time.time() > DEADLINE - 44:
            break
        trial += 1
        # pick base: mostly best, sometimes 2nd/3rd for diversity
        if len(archive) == 1 or (trial % 6) != 5:
            base = best_x
        else:
            base = archive[(trial // 6) % len(archive)][1]
        mode = trial % 9
        try:
            if mode == 0:
                cur = rng.randn(n, 4)
                cur /= np.linalg.norm(cur, axis=1, keepdims=True)
                cur_b, cur_v = _anneal_full(cur, steps=4200, T0=0.020, T1=0.0006, lr0=0.045, mu=0.85, rng=rng, noise0=0.004, deadline=SEARCH_END)
            elif mode == 1:
                sig = [0.02, 0.06, 0.15, 0.30][(trial // 9) % 4]
                cur = base + rng.randn(n, 4) * sig
                cur /= np.maximum(np.linalg.norm(cur, axis=1, keepdims=True), 1e-12)
                cur_b, cur_v = _anneal_full(cur, steps=3800, T0=0.010, T1=0.0005, lr0=0.036, mu=0.88, rng=rng, noise0=0.002, deadline=SEARCH_END)
            elif mode == 2:
                # collective low-rank field: symmetry-enforced coherent motion
                amp = 0.06 + 0.20 * rng.rand()
                cur = _collective_move(base, rng, amp)
                cur += rng.randn(n, 4) * 0.004
                cur /= np.maximum(np.linalg.norm(cur, axis=1, keepdims=True), 1e-12)
                cur_b, cur_v = _anneal_full(cur, steps=4000, T0=0.012, T1=0.0005, lr0=0.040, mu=0.86, rng=rng, noise0=0.0015, deadline=SEARCH_END)
            elif mode == 3:
                # floppy subcluster: freeze 85%, re-anneal mobile subset alone
                G = base @ base.T
                np.fill_diagonal(G, -2.0)
                m = float(G.max())
                deg = (G > m - 0.003).sum(axis=1)
                k = 10 + (trial % 9)
                worst = np.argsort(deg)[-k:]
                # add random neighbors to make contiguous floppy patch
                u = rng.randn(4); u /= np.linalg.norm(u)
                dots = base @ u
                cap = np.argsort(dots)[-k:]
                mobile = np.zeros(n, dtype=bool)
                mobile[worst] = True
                mobile[cap] = True
                cur = base.copy()
                cur[mobile] = rng.randn(int(mobile.sum()), 4)
                cur[mobile] /= np.linalg.norm(cur[mobile], axis=1, keepdims=True)
                # first fit mobile into holes w.r.t. frozen
                fixed = base[~mobile]
                for ii in np.where(mobile)[0]:
                    h, _ = _find_hole(fixed, rng, n_samples=4000)
                    cur[ii] = h
                    fixed = np.vstack([fixed, h.reshape(1, 4)])
                cur_b, cur_v = _anneal_subset(cur, mobile, steps=3000, T0=0.012, T1=0.0004, lr0=0.045, mu=0.86, rng=rng, noise0=0.002, deadline=SEARCH_END)
                # then release all
                cur_b2, cur_v2 = _anneal_full(cur_b, steps=2200, T0=0.004, T1=0.0003, lr0=0.030, mu=0.88, rng=rng, noise0=0.0005, deadline=SEARCH_END)
                cur_b, cur_v = cur_b2, cur_v2
            elif mode == 4:
                # kill-and-fill worst points into deep holes
                G = base @ base.T
                np.fill_diagonal(G, -2.0)
                m = float(G.max())
                deg = (G > m - 0.003).sum(axis=1)
                mrep = 3 + (trial % 6)
                worst = np.argsort(deg)[-mrep:]
                keep = np.ones(n, dtype=bool); keep[worst] = False
                fixed = base[keep].copy()
                cur = base.copy()
                for w in worst:
                    h, _ = _find_hole(fixed, rng, n_samples=7000)
                    cur[w] = h
                    fixed = np.vstack([fixed, h.reshape(1, 4)])
                cur += rng.randn(n, 4) * 0.005
                cur /= np.maximum(np.linalg.norm(cur, axis=1, keepdims=True), 1e-12)
                cur_b, cur_v = _anneal_full(cur, steps=4200, T0=0.011, T1=0.0005, lr0=0.040, mu=0.86, rng=rng, noise0=0.0012, deadline=SEARCH_END)
            elif mode == 5:
                # explicit worst-pair break with large tangential push
                G = base @ base.T
                np.fill_diagonal(G, -2.0)
                ij = int(np.argmax(G.ravel()))
                ii, jj = divmod(ij, n)
                cur = base.copy()
                for a, b in ((ii, jj), (jj, ii)):
                    away = cur[a] - cur[b]
                    away -= (away @ cur[a]) * cur[a]
                    na = np.linalg.norm(away)
                    if na > 1e-12:
                        cur[a] += (away / na) * (0.08 + 0.10 * rng.rand())
                # shake their contact neighborhoods too
                m = float(G.max())
                nbr = np.where((G[ii] > m - 0.010) | (G[jj] > m - 0.010))[0]
                if len(nbr) > 0:
                    cur[nbr] += rng.randn(len(nbr), 4) * 0.06
                cur /= np.maximum(np.linalg.norm(cur, axis=1, keepdims=True), 1e-12)
                cur_b, cur_v = _anneal_full(cur, steps=4000, T0=0.013, T1=0.0005, lr0=0.040, mu=0.85, rng=rng, noise0=0.002, deadline=SEARCH_END)
            elif mode == 6:
                # rigid cap rotation with large angle
                u = rng.randn(4); u /= np.linalg.norm(u)
                dots = base @ u
                q = 0.80 + 0.12 * rng.rand()
                thresh = np.quantile(dots, q)
                idx = np.where(dots > thresh)[0]
                if len(idx) < 5:
                    idx = np.argsort(dots)[-14:]
                R = _rand_rot(rng, 0.12 + 0.30 * rng.rand())
                cur = base.copy()
                cur[idx] = (R @ cur[idx].T).T
                cur += rng.randn(n, 4) * 0.004
                cur /= np.maximum(np.linalg.norm(cur, axis=1, keepdims=True), 1e-12)
                cur_b, cur_v = _anneal_full(cur, steps=4000, T0=0.012, T1=0.0005, lr0=0.040, mu=0.86, rng=rng, noise0=0.0015, deadline=SEARCH_END)
            elif mode == 7:
                # antipodal band twist
                u = rng.randn(4); u /= np.linalg.norm(u)
                dots = np.abs(base @ u)
                idx = np.where(dots < 0.28)[0]
                if len(idx) < 6:
                    idx = np.argsort(dots)[:22]
                R = _rand_rot(rng, 0.15 + 0.25 * rng.rand())
                cur = base.copy()
                cur[idx] = (R @ cur[idx].T).T
                cur[idx] += rng.randn(len(idx), 4) * 0.05
                cur /= np.maximum(np.linalg.norm(cur, axis=1, keepdims=True), 1e-12)
                cur_b, cur_v = _anneal_full(cur, steps=3800, T0=0.014, T1=0.0006, lr0=0.040, mu=0.84, rng=rng, noise0=0.002, deadline=SEARCH_END)
            else:
                # chained collective + hole reinsertion
                amp = 0.10 + 0.15 * rng.rand()
                cur = _collective_move(base, rng, amp)
                G = cur @ cur.T
                np.fill_diagonal(G, -2.0)
                m = float(G.max())
                deg = (G > m - 0.003).sum(axis=1)
                w = int(np.argmax(deg))
                fixed = np.delete(cur, w, axis=0)
                h, _ = _find_hole(fixed, rng, n_samples=6000)
                cur[w] = h
                cur /= np.maximum(np.linalg.norm(cur, axis=1, keepdims=True), 1e-12)
                cur_b, cur_v = _anneal_full(cur, steps=4200, T0=0.012, T1=0.0005, lr0=0.040, mu=0.86, rng=rng, noise0=0.0015, deadline=SEARCH_END)
        except Exception:
            continue
        # L-BFGS smoothing for promising candidates
        if cur_v < best_val + 0.004 and time.time() < SEARCH_END - 0.8:
            try:
                lb_x, lb_v = _lbfgs_smooth(cur_b, deadline=min(SEARCH_END, time.time() + 3.5))
                if lb_v < cur_v:
                    cur_b, cur_v = lb_x, lb_v
            except Exception:
                pass
        # Metropolis-style archive insert to keep diversity
        if cur_v < best_val:
            push(cur_b, cur_v)
        elif cur_v < best_val + 0.002 and rng.rand() < 0.15:
            push(cur_b, cur_v)
        # quick LP confirm if at/near best
        if cur_v < best_val + 1e-4 and cur_v < 0.90 and time.time() < SEARCH_END - 1.0:
            try:
                lp_x, lp_v = _lp_polish(cur_b, deadline=min(SEARCH_END, time.time() + 2.5), r0=2e-3)
                if lp_v < cur_v - 1e-9:
                    push(lp_x, lp_v)
            except Exception:
                pass

    # final intensive polish of the best
    try:
        fx, fv = _lbfgs_smooth(best_x, deadline=DEADLINE - 14.0,
                               taus=(0.002, 0.0008, 0.00035), iters=(150, 150, 200))
        if fv < best_val:
            best_x, best_val = fx, fv
    except Exception:
        pass
    try:
        best_x, best_val = _lp_polish(best_x, deadline=DEADLINE - 1.0, r0=1.5e-3)
    except Exception:
        pass
    best_x = best_x / np.maximum(np.linalg.norm(best_x, axis=1, keepdims=True), 1e-12)
    return best_x
