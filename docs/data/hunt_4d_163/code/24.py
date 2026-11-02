import time
import numpy as np
from scipy.optimize import linprog, minimize
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _lp_polish(x, deadline, r0=1e-3):
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
            if np.count_nonzero(sel) == 0:
                sel[np.argmax(c)] = True
            I = iu0[sel]; J = iu1[sel]
            csel = c[sel]
            p = len(I)
            nv = n * d + 1
            rows = np.repeat(np.arange(p), 2 * d + 1)
            tmpc = np.empty((p, 2 * d + 1), dtype=int)
            ar = np.arange(d)
            tmpc[:, :d] = I[:, None] * d + ar
            tmpc[:, d:2 * d] = J[:, None] * d + ar
            tmpc[:, 2 * d] = n * d
            cols = tmpc.ravel()
            cc = np.empty((p, 2 * d + 1), dtype=float)
            cc[:, :d] = x[J]
            cc[:, d:2 * d] = x[I]
            cc[:, 2 * d] = -1.0
            vals = cc.ravel()
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

def _anneal(x0, steps, T0, T1, lr0, mu, rng, noise0=0.0, deadline=None):
    x = x0.copy()
    n, d = x.shape
    v = np.zeros_like(x)
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    best_val = float(G.max())
    best = x.copy()
    lratio = np.log(T1 / T0) if T0 > 0 and T1 > 0 else 0.0
    for k in range(steps):
        if deadline is not None and (k & 255) == 0 and time.time() > deadline:
            break
        frac = k / max(1, steps - 1)
        T = T0 * np.exp(lratio * frac)
        if T < 1e-12:
            T = 1e-12
        lr = lr0 * (1.0 - 0.55 * frac)
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        E = np.exp((G - m) / T)
        np.fill_diagonal(E, 0.0)
        S = float(E.sum() * 0.5)
        if S < 1e-300 or not np.isfinite(S):
            S = 1e-300
        grad = (E @ x) / S
        grad -= (grad * x).sum(axis=1, keepdims=True) * x
        v = mu * v + (1.0 - mu) * grad
        x = x - lr * v
        if noise0 > 0:
            amp = noise0 * (1.0 - frac)
            if amp > 0:
                nz = rng.randn(n, d) * amp
                nz -= (nz * x).sum(axis=1, keepdims=True) * x
                x = x + nz
        x /= np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
        if m < best_val:
            best_val = m
            best = x.copy()
    return best, best_val

def _lbfgs_smooth(x_init, deadline, taus=(0.0035, 0.0012, 0.0005), iters=(100, 100, 150)):
    x = x_init / np.maximum(np.linalg.norm(x_init, axis=1, keepdims=True), 1e-12)
    n, d = x.shape
    iu0, iu1 = np.triu_indices(n, 1)
    Np = len(iu0)
    logNp = np.log(Np)
    y = x.ravel().copy()
    for tau, maxiter in zip(taus, iters):
        if time.time() > deadline - 0.3:
            break
        def fg(yflat, _tau=tau):
            Y = yflat.reshape(n, d)
            rn = np.linalg.norm(Y, axis=1, keepdims=True)
            rn = np.maximum(rn, 1e-12)
            X = Y / rn
            g = (X @ X.T)[iu0, iu1]
            m = float(g.max())
            e = np.exp((g - m) / _tau)
            s = float(e.sum())
            if s <= 0 or not np.isfinite(s):
                return float(m), np.zeros_like(yflat)
            L = m + _tau * (np.log(s) - logNp)
            w = e / s
            W = np.zeros((n, n))
            W[iu0, iu1] = w
            W = W + W.T
            GX = W @ X
            dot = (GX * X).sum(axis=1, keepdims=True)
            GY = (GX - dot * X) / rn
            return float(L), GY.ravel()
        try:
            res = minimize(fg, y, method="L-BFGS-B", jac=True,
                           options={"maxiter": int(maxiter), "maxfun": int(maxiter * 6),
                                    "ftol": 1e-14, "gtol": 1e-8})
            y = res.x
        except Exception:
            break
    Xf = y.reshape(n, d)
    Xf /= np.maximum(np.linalg.norm(Xf, axis=1, keepdims=True), 1e-12)
    return Xf, _max_cos(Xf)

def _find_hole(fixed, rng, n_samples=15000):
    Y = rng.randn(n_samples, 4)
    Y /= np.maximum(np.linalg.norm(Y, axis=1, keepdims=True), 1e-12)
    D = Y @ fixed.T
    vals = D.max(axis=1)
    order = np.argsort(vals)[:8]
    best_y = None
    best_v = 1e9
    for idx in order:
        y = Y[int(idx)].copy()
        vy = float(vals[int(idx)])
        # tangential refinement away from closest point
        step = 0.03
        for it in range(200):
            dots = fixed @ y
            j = int(np.argmax(dots))
            g = fixed[j] - dots[j] * y
            ng = float(np.linalg.norm(g))
            if ng < 1e-12:
                break
            dirc = g / ng
            improved = False
            for st in (step, step * 0.3):
                yn = y - st * dirc
                yn /= max(float(np.linalg.norm(yn)), 1e-12)
                vn = float((fixed @ yn).max())
                if vn < vy:
                    y, vy = yn, vn
                    improved = True
                    break
            if not improved:
                step *= 0.6
                if step < 1e-5:
                    break
        if vy < best_v:
            best_v = vy
            best_y = y
    return best_y, best_v

def _splice(a, b, rng):
    # hemisphere splice crossover
    n = a.shape[0]
    u = rng.randn(4); u /= np.linalg.norm(u)
    da = a @ u
    db = b @ u
    # take positive side from a, non-positive from b
    pts = []
    ptsA = a[da > 0]
    ptsB = b[db <= 0]
    cur = np.vstack([ptsA, ptsB]) if len(ptsA) + len(ptsB) > 0 else a.copy()
    # adjust count to n
    cur = cur / np.maximum(np.linalg.norm(cur, axis=1, keepdims=True), 1e-12)
    # if too many, greedily delete one of closest pair
    while cur.shape[0] > n:
        G = cur @ cur.T
        np.fill_diagonal(G, -2.0)
        i, j = np.unravel_index(int(np.argmax(G)), G.shape)
        # delete the one with larger sum of top-3 dots (more crowded)
        si = np.sort(G[i])[::-1][:3].sum()
        sj = np.sort(G[j])[::-1][:3].sum()
        drop = i if si > sj else j
        cur = np.delete(cur, drop, axis=0)
    # if too few, fill with holes
    while cur.shape[0] < n:
        h, _ = _find_hole(cur, rng, n_samples=8000)
        cur = np.vstack([cur, h.reshape(1, 4)])
    # shuffle order to avoid bias
    perm = rng.permutation(n)
    cur = cur[perm]
    return cur

def solve():
    t0 = time.time()
    DEADLINE = t0 + 173.0
    SEARCH_END = t0 + 138.0
    rng = np.random.RandomState(1234567)
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
    elites = [(best_val, best_x.copy())]

    def push_elite(x, v):
        for _, ex in elites:
            if float(np.abs(ex - x).max()) < 1e-9:
                return
        elites.append((v, x.copy()))
        elites.sort(key=lambda t: t[0])
        del elites[6:]

    trial = 0
    while time.time() < SEARCH_END:
        if time.time() > DEADLINE - 35:
            break
        trial += 1
        mode = trial % 7
        cur = None
        if mode == 0:
            cur = rng.randn(n, 4)
            cur /= np.linalg.norm(cur, axis=1, keepdims=True)
            T0, T1, lr0, mu, steps, noise = 0.018, 0.0007, 0.045, 0.85, 4500, 0.004
        elif mode == 1:
            sigmas = [0.015, 0.05, 0.12, 0.25]
            sigma = sigmas[(trial // 7) % 4]
            base = elites[rng.randint(len(elites))][1] if len(elites) > 1 and rng.rand() < 0.4 else best_x
            cur = base + rng.randn(n, 4) * sigma
            cur /= np.linalg.norm(cur, axis=1, keepdims=True)
            T0, T1, lr0, mu, steps, noise = 0.010, 0.0006, 0.035, 0.88, 3500, 0.002
        elif mode == 2:
            # worst-point surgery
            base = best_x
            G = base @ base.T
            np.fill_diagonal(G, -2.0)
            m = float(G.max())
            deg = (G > m - 0.0025).sum(axis=1)
            mrep = 2 + (trial % 5)
            worst = np.argsort(deg)[-mrep:]
            keep = np.ones(n, dtype=bool); keep[worst] = False
            fixed = base[keep].copy()
            cur = base.copy()
            for w in worst:
                h, _ = _find_hole(fixed, rng, n_samples=8000)
                cur[w] = h
                fixed = np.vstack([fixed, h.reshape(1, 4)])
            cur = cur + rng.randn(n, 4) * 0.004
            cur /= np.linalg.norm(cur, axis=1, keepdims=True)
            T0, T1, lr0, mu, steps, noise = 0.012, 0.0006, 0.040, 0.86, 4000, 0.001
        elif mode == 3:
            # cap shake / reshuffle
            base = best_x
            u = rng.randn(4); u /= np.linalg.norm(u)
            dots = base @ u
            thresh = np.quantile(dots, 0.86)
            idx = np.where(dots > thresh)[0]
            cur = base.copy()
            if len(idx) > 1 and rng.rand() < 0.5:
                # full reshuffle inside cap: random points with dot>thresh
                for k in idx:
                    v = rng.randn(4)
                    v -= (v @ u) * u
                    v /= max(float(np.linalg.norm(v)), 1e-12)
                    r = rng.rand() * 0.55 + 0.15
                    # random point at angular radius r from u
                    y = np.sqrt(max(0.0, 1 - r * r)) * u + r * v
                    cur[k] = y
            else:
                cur[idx] += rng.randn(len(idx), 4) * 0.28
            cur /= np.linalg.norm(cur, axis=1, keepdims=True)
            T0, T1, lr0, mu, steps, noise = 0.014, 0.0008, 0.040, 0.84, 3500, 0.003
        elif mode == 4:
            # pair-break: take top edges, kick endpoints apart
            base = best_x
            G = base @ base.T
            np.fill_diagonal(G, -2.0)
            iu0, iu1 = np.triu_indices(n, 1)
            c = G[iu0, iu1]
            top = np.argsort(c)[-10:]
            cur = base.copy()
            for t in top:
                i = int(iu0[t]); j = int(iu1[t])
                # push i away from j tangentially
                d = float(c[t])
                gi = base[i] - d * base[j]
                gj = base[j] - d * base[i]
                ni = float(np.linalg.norm(gi)); nj = float(np.linalg.norm(gj))
                amp = 0.10 + 0.08 * rng.rand()
                if ni > 1e-12:
                    cur[i] = base[i] + (gi / ni) * amp + rng.randn(4) * 0.02
                if nj > 1e-12:
                    cur[j] = base[j] + (gj / nj) * amp + rng.randn(4) * 0.02
            cur /= np.linalg.norm(cur, axis=1, keepdims=True)
            T0, T1, lr0, mu, steps, noise = 0.011, 0.0005, 0.038, 0.87, 3800, 0.0015
        elif mode == 5:
            # splice crossover of two elites (or best jitter if only one)
            if len(elites) >= 2:
                a = elites[rng.randint(len(elites))][1]
                b = elites[rng.randint(len(elites))][1]
                cur = _splice(a, b, rng)
                cur = cur + rng.randn(n, 4) * 0.006
                cur /= np.linalg.norm(cur, axis=1, keepdims=True)
            else:
                cur = best_x + rng.randn(n, 4) * 0.09
                cur /= np.linalg.norm(cur, axis=1, keepdims=True)
            T0, T1, lr0, mu, steps, noise = 0.013, 0.0006, 0.040, 0.85, 4000, 0.002
        else:
            # global rotation + shake
            base = best_x
            cur = base + rng.randn(n, 4) * 0.10
            cur /= np.linalg.norm(cur, axis=1, keepdims=True)
            A = rng.randn(4, 4) * 0.05
            A = A - A.T
            Q, _ = np.linalg.qr(np.eye(4) + A)
            cur = cur @ Q.T
            cur /= np.linalg.norm(cur, axis=1, keepdims=True)
            T0, T1, lr0, mu, steps, noise = 0.016, 0.0007, 0.042, 0.85, 3500, 0.003

        cb, cv = _anneal(cur, steps=steps, T0=T0, T1=T1, lr0=lr0, mu=mu, rng=rng, noise0=noise, deadline=SEARCH_END)
        # L-BFGS refine (cheap, big gain near basin bottom)
        try:
            cb2, cv2 = _lbfgs_smooth(cb, deadline=SEARCH_END)
        except Exception:
            cb2, cv2 = cb, cv
        if cv2 < cv:
            cb, cv = cb2, cv2
        else:
            # still keep lbfgs best-ever tracking: cv already best along anneal; check final point
            vf = _max_cos(cb2)
            if vf < cv:
                cb, cv = cb2, vf

        if cv < best_val - 1e-12:
            if cv < best_val - 2e-5 and time.time() < DEADLINE - 40:
                try:
                    pol, pv = _lp_polish(cb, deadline=min(time.time() + 2.5, DEADLINE - 30))
                except Exception:
                    pol, pv = cb, cv
                if pv < best_val:
                    best_x, best_val = pol, pv
                    push_elite(pol, pv)
                elif cv < best_val:
                    best_x, best_val = cb, cv
                    push_elite(cb, cv)
            else:
                best_x, best_val = cb, cv
                push_elite(cb, cv)
        elif cv < best_val + 0.002:
            push_elite(cb, cv)
        if best_val < 0.84:
            break

    remain = DEADLINE - time.time() - 1.5
    if remain > 3:
        # polish top elites, keep absolute best
        elites_sorted = sorted(elites, key=lambda t: t[0])[:3]
        # ensure best_x first
        cand_list = [(best_val, best_x)] + [e for e in elites_sorted if e[0] > best_val + 1e-12]
        for vi, xi in cand_list:
            if time.time() > DEADLINE - 1.0:
                break
            try:
                px, pv = _lp_polish(xi, deadline=min(time.time() + remain / max(1, len(cand_list)), DEADLINE - 0.8))
            except Exception:
                continue
            if pv < best_val:
                best_x, best_val = px, pv
            if pv <= best_val + 1e-12 and pv < vi:
                # update elite copy
                pass
    best_x = best_x / np.maximum(np.linalg.norm(best_x, axis=1, keepdims=True), 1e-12)
    return best_x
