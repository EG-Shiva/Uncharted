import time
import numpy as np

def _norm(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return X / n

def _mc(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _rowmax(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return G.max(axis=1), float(G.max()), G

def solve():
    t0 = time.time()
    LIM = 172.0
    rng = np.random.default_rng(0)
    n, d = 74, 13
    cands = []
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=np.float64)
            if a.shape == (74, 13):
                cands.append(_norm(a))
        except Exception:
            pass
    if not cands:
        cands.append(_norm(rng.standard_normal((n, d))))
    vals = [_mc(a) for a in cands]
    best = cands[int(np.argmin(vals))].copy()
    best_val = float(min(vals))

    def adam_penalised(X, steps, tau0, tau1, lam0, lam1, lr0, seed_rng):
        m = np.zeros_like(X)
        v = np.zeros_like(X)
        cnt = 0
        for s in range(steps):
            if time.time() - t0 > LIM - 8:
                break
            frac = s / max(1, steps - 1)
            # geometric tau, linear lambda
            tau = tau0 * ((tau1 / max(tau0, 1e-18)) ** frac) if tau0 > 0 and tau1 > 0 else tau0
            lam = lam0 + (lam1 - lam0) * frac
            G = X @ X.T
            np.fill_diagonal(G, -10.0)
            gmax = float(G.max())
            # LSE weights
            E = np.exp((G - gmax) / tau)
            np.fill_diagonal(E, 0.0)
            S = float(E.sum())
            if not np.isfinite(S) or S <= 0:
                continue
            Gr = (E @ X) / S
            dot = (Gr * X).sum(axis=1, keepdims=True)
            Gt_lse = Gr - dot * X
            # equalisation subgradient (mean squared gap)
            # rowmax and argmax
            G2 = X @ X.T
            np.fill_diagonal(G2, -2.0)
            row = G2.max(axis=1)
            js = G2.argmax(axis=1)
            diff = row - gmax  # <=0
            # tangent toward row-neighbour
            # Eq_grad = (2/n) * diff_i * t_i
            Xn = X[js]  # (n,d)
            t = Xn - row[:, None] * X
            # t already tangent
            Eq = (2.0 / n) * diff[:, None] * t
            Gt = Gt_lse + lam * Eq
            cnt += 1
            m[:] = 0.9 * m + 0.1 * Gt
            v[:] = 0.999 * v + 0.001 * (Gt * Gt)
            c1 = 1.0 - 0.9 ** cnt
            c2 = 1.0 - 0.999 ** cnt
            mh = m / c1
            vh = v / c2
            lr = lr0 * (1.0 - 0.5 * frac)
            X -= lr * mh / (np.sqrt(vh) + 1e-8)
            nr = np.linalg.norm(X, axis=1, keepdims=True)
            nr[nr == 0] = 1.0
            X /= nr
        return X

    def micro_split(X, cur, steps=250):
        step = 4e-6
        for _ in range(steps):
            if time.time() - t0 > LIM - 5:
                break
            G = X @ X.T
            np.fill_diagonal(G, -2.0)
            k = int(np.argmax(G))
            i = k // n
            j = k % n
            c = float(G[i, j])
            if c < cur - 1e-12:
                cur = c
            xi = X[i].copy()
            xj = X[j].copy()
            gi = xj - c * xi
            gj = xi - c * xj
            ni = float(np.linalg.norm(gi))
            nj = float(np.linalg.norm(gj))
            if ni < 1e-12 or nj < 1e-12:
                break
            gi /= ni
            gj /= nj
            improved = False
            for mode in (0, 1, 2):
                Y = X.copy()
                if mode == 0:
                    Y[i] = xi - step * gi
                    Y[j] = xj - step * gj
                elif mode == 1:
                    Y[i] = xi - step * 2.0 * gi
                else:
                    Y[j] = xj - step * 2.0 * gj
                Y[i] /= np.linalg.norm(Y[i])
                Y[j] /= np.linalg.norm(Y[j])
                vv = _mc(Y)
                if vv < cur:
                    X[:] = Y
                    cur = vv
                    improved = True
                    break
            if improved:
                step = min(step * 1.2, 5e-5)
            else:
                step *= 0.5
                if step < 5e-13:
                    break
        return X, cur

    def split_relocate(Xc, rattler, hub, dot_target, rng):
        xh = Xc[hub].copy()
        G = Xc @ Xc.T
        np.fill_diagonal(G, -2.0)
        gmax = float(G.max())
        # tight neighbours of hub
        thr = gmax - 0.015
        nbrs = np.where(G[hub] > thr)[0]
        if len(nbrs) == 0:
            nbrs = np.argsort(-G[hub])[:4]
        # tangent components
        T = Xc[nbrs] - G[hub, nbrs][:, None] * xh
        s = T.sum(axis=0)
        # project to tangent at xh
        s = s - float(s @ xh) * xh
        nn = float(np.linalg.norm(s))
        if nn < 1e-9:
            # random tangent
            r = rng.standard_normal(d)
            r = r - float(r @ xh) * xh
            nn = float(np.linalg.norm(r))
            if nn < 1e-12:
                return None
            u = r / nn
        else:
            u = -s / nn  # away from neighbours
        # eps so dot(xh,new)==dot_target
        dt = float(np.clip(dot_target, -0.9, 0.99))
        if dt <= 0:
            eps = 10.0
        else:
            eps = float(np.sqrt(max(1.0 / (dt * dt) - 1.0, 1e-6)))
        y = xh + eps * u
        ny = float(np.linalg.norm(y))
        if ny < 1e-12 or not np.isfinite(ny):
            return None
        y /= ny
        return y

    # initial quick penalised squeeze of best
    Xb = best.copy()
    Xb = adam_penalised(Xb, steps=400, tau0=0.006, tau1=0.0012, lam0=0.3, lam1=2.0, lr0=0.008, seed_rng=rng)
    Xb, vb = micro_split(Xb, _mc(Xb), steps=200)
    if vb < best_val:
        best, best_val = Xb.copy(), float(vb)

    # SA basin hopping with pair-splitting
    # cooling schedule
    T0, T1 = 6e-5, 2e-6
    n_trials = 200
    cur = best.copy()
    cur_val = float(best_val)
    trial = 0
    while time.time() - t0 < LIM - 22 and trial < n_trials:
        remain = LIM - 22 - (time.time() - t0)
        # fraction for cooling
        frac_all = trial / max(1, n_trials - 1)
        Temp = T0 * ((T1 / T0) ** frac_all)
        # identify rattlers and hubs from cur
        row, gmax, G = _rowmax(cur)
        # rattlers: smallest rowmax
        order_low = np.argsort(row)
        # hubs: degree with threshold gmax-2e-3, fallback to row value
        cnt = (G > (gmax - 2e-3)).sum(axis=1)
        order_hub = np.argsort(-cnt - 0.1 * (row - row.min()))
        # cycle
        rattler = int(order_low[trial % 10])  # among 10 lowest
        hub = int(order_hub[trial % 6])  # among 6 highest
        if rattler == hub:
            hub = int(order_hub[(trial + 3) % n])
        # dot target just above current best (deliberately worse)
        delta = 0.004 + 0.008 * rng.random()
        dot_target = cur_val + delta
        dot_target = min(dot_target, 0.35)
        y = split_relocate(cur, rattler, hub, dot_target, rng)
        if y is None:
            trial += 1
            continue
        Xt = cur.copy()
        Xt[rattler] = y
        # small jitter others to break symmetry
        Xt = _norm(Xt + 0.0015 * rng.standard_normal(Xt.shape))
        # anneal lambda upward within polish
        lam1 = 0.5 + 3.0 * frac_all  # increases over trials
        Xt = adam_penalised(Xt, steps=350, tau0=0.005, tau1=0.0009, lam0=0.2, lam1=lam1, lr0=0.007, seed_rng=rng)
        # light micro split
        Xt, vt = micro_split(Xt, _mc(Xt), steps=120)
        vt = float(vt)
        d = vt - cur_val
        accept = False
        if vt < cur_val - 1e-12:
            accept = True
        elif d <= 1e-4:
            # SA acceptance
            p = float(np.exp(-d / max(Temp, 1e-12)))
            if rng.random() < p:
                accept = True
        if accept:
            cur = Xt.copy()
            cur_val = vt
            if vt < best_val - 1e-12:
                best = Xt.copy()
                best_val = vt
        else:
            # even if rejected as cur, keep global best check with larger tolerance?
            if vt < best_val - 1e-12:
                best = Xt.copy()
                best_val = vt
                cur = Xt.copy()
                cur_val = vt
        trial += 1
        # occasionally restart cur from best to avoid drift too high
        if cur_val > best_val + 8e-5:
            cur = best.copy()
            cur_val = float(best_val)

    # final exact minimax squeeze on best
    Xf = best.copy()
    # staged tiny-tau polish with growing lambda (equalisation)
    Xf = adam_penalised(Xf, steps=600, tau0=0.002, tau1=0.0005, lam0=0.5, lam1=3.0, lr0=0.004, seed_rng=rng)
    Xf, vf = micro_split(Xf, _mc(Xf), steps=400)
    if vf < best_val:
        best, best_val = Xf.copy(), float(vf)
    # extra fine polish
    if time.time() - t0 < LIM - 6:
        Xf = adam_penalised(best.copy(), steps=300, tau0=0.0008, tau1=0.0004, lam0=1.0, lam1=4.0, lr0=0.002, seed_rng=rng)
        Xf, vf = micro_split(Xf, _mc(Xf), steps=300)
        if vf < best_val:
            best, best_val = Xf.copy(), float(vf)

    best = _norm(best)
    return best
