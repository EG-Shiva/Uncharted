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

def solve():
    t0 = time.time()
    LIM = 172.0
    rng = np.random.default_rng(0)
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
        cands.append(_norm(rng.standard_normal((74, 13))))
    vals = [_mc(a) for a in cands]
    best = cands[int(np.argmin(vals))].copy()
    best_val = float(min(vals))
    n, d = 74, 13

    def adam(X, sched, m, v, cnt):
        for tau, steps, lr0 in sched:
            for s in range(steps):
                if time.time() - t0 > LIM - 6:
                    return X
                G = X @ X.T
                np.fill_diagonal(G, -10.0)
                gmax = float(G.max())
                E = np.exp((G - gmax) / tau)
                np.fill_diagonal(E, 0.0)
                S = float(E.sum())
                if not np.isfinite(S) or S <= 0:
                    continue
                Gr = (E @ X) / S
                dot = (Gr * X).sum(axis=1, keepdims=True)
                Gt = Gr - dot * X
                m[:] = 0.9 * m + 0.1 * Gt
                v[:] = 0.999 * v + 0.001 * (Gt * Gt)
                cnt[0] += 1
                c1 = 1.0 - 0.9 ** cnt[0]
                c2 = 1.0 - 0.999 ** cnt[0]
                mh = m / c1
                vh = v / c2
                lr = lr0 * (1.0 - 0.5 * s / max(1, steps))
                X -= lr * mh / (np.sqrt(vh) + 1e-8)
                nr = np.linalg.norm(X, axis=1, keepdims=True)
                nr[nr == 0] = 1.0
                X /= nr
        return X

    def worst_pair(X):
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        k = int(np.argmax(G))
        return k // n, k % n, float(G.max())

    def worst_order(X):
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        return np.argsort(-G.max(axis=1))

    def hole_best_for(X, idx, n_cand=4000, steps=160):
        # find y minimizing max_j<>i dot(y,Xj)
        mask = np.ones(n, dtype=bool); mask[idx] = False
        O = X[mask]
        cand = _norm(rng.standard_normal((n_cand, d)))
        sc = cand @ O.T
        scmax = sc.max(axis=1)
        order = np.argsort(scmax)
        cur = float((X[idx] @ O.T).max())
        best_y = X[idx].copy()
        best_s = cur
        # try top 3 distinct holes
        for t in range(3):
            y = cand[order[t]].copy()
            lr = 0.03
            for _ in range(steps):
                dv = O @ y
                dm = float(dv.max())
                act = dv > dm - 0.01
                if not np.any(act):
                    break
                g = O[act].mean(axis=0)
                gt = g - float(g @ y) * y
                nn = float(np.linalg.norm(gt))
                if nn < 1e-12:
                    break
                gt /= nn
                yn = y - lr * gt
                nn2 = float(np.linalg.norm(yn))
                if nn2 == 0:
                    lr *= 0.5
                    continue
                yn /= nn2
                dn = float((O @ yn).max())
                if dn < dm:
                    y = yn
                    lr = min(lr * 1.08, 0.05)
                else:
                    lr *= 0.5
                    if lr < 1e-6:
                        break
            s = float((O @ y).max())
            if s < best_s:
                best_s = s
                best_y = y
        return best_y, best_s, cur

    def sweep_single(X):
        # try to improve worst few points in place
        cur_global = _mc(X)
        order = worst_order(X)
        improved = False
        for idx in order[:4]:
            if time.time() - t0 > LIM - 5:
                break
            y, s, _ = hole_best_for(X, int(idx), n_cand=3000, steps=120)
            if s < cur_global - 1e-12:
                X[int(idx)] = y
                cur_global = _mc(X)
                improved = True
                # after success, update order? continue to next
        return X, cur_global, improved

    def pair_shift_hill(X, trials=25):
        cur = _mc(X)
        i, j, _ = worst_pair(X)
        for _ in range(trials):
            if time.time() - t0 > LIM - 4:
                break
            # random common translation orthogonalised
            r = rng.standard_normal(d)
            # make orthogonal to xi,xj partly
            r = r - float(r @ X[i]) * X[i]
            r = r - float(r @ X[j]) * X[j]
            nn = float(np.linalg.norm(r))
            if nn < 1e-9:
                continue
            r /= nn
            eps = float(rng.uniform(-0.01, 0.01))
            Y = X.copy()
            Y[i] = _norm((X[i] + eps * r).reshape(1, -1))[0]
            Y[j] = _norm((X[j] + eps * r).reshape(1, -1))[0]
            v = _mc(Y)
            if v < cur:
                X[:] = Y
                cur = v
                i, j, _ = worst_pair(X)
        # also try opposite moves for worst pair
        for _ in range(trials):
            if time.time() - t0 > LIM - 4:
                break
            i, j, _ = worst_pair(X)
            r = rng.standard_normal(d)
            r1 = r - float(r @ X[i]) * X[i]
            r2 = -r - float((-r) @ X[j]) * X[j]
            n1 = float(np.linalg.norm(r1)); n2 = float(np.linalg.norm(r2))
            if n1 < 1e-9 or n2 < 1e-9:
                continue
            e = float(rng.uniform(0.002, 0.008))
            Y = X.copy()
            Y[i] = _norm((X[i] - e * r1 / n1).reshape(1, -1))[0]
            Y[j] = _norm((X[j] - e * r2 / n2).reshape(1, -1))[0]
            v = _mc(Y)
            if v < cur:
                X[:] = Y
                cur = v
        return X, cur

    def farthest_cand(X, exclude, n_cand=6000):
        cand = _norm(rng.standard_normal((n_cand, d)))
        S = cand @ X.T
        if exclude is not None:
            # allow overwrite of exclude by ignoring it
            S[:, exclude] = -10.0
        sc = S.max(axis=1)
        return cand[int(np.argmin(sc))]

    full = [(0.02, 250, 0.014), (0.008, 450, 0.009), (0.003, 650, 0.0055), (0.001, 550, 0.003)]
    fast = [(0.008, 300, 0.009), (0.003, 450, 0.005), (0.001, 300, 0.003)]
    tiny = [(0.003, 250, 0.004), (0.0008, 250, 0.002)]

    # initial squeeze
    X0 = best.copy()
    m = np.zeros_like(X0); v = np.zeros_like(X0); c = [0]
    X0 = adam(X0, full, m, v, c)
    X0, vg, _ = sweep_single(X0)
    X0, vg = pair_shift_hill(X0, trials=15)
    m = np.zeros_like(X0); v = np.zeros_like(X0); c = [0]
    X0 = adam(X0, tiny, m, v, c)
    vg = _mc(X0)
    if vg < best_val:
        best, best_val = X0.copy(), float(vg)

    basin = 0
    while time.time() - t0 < LIM - 9:
        remain = LIM - (time.time() - t0)
        sched = fast if (remain < 40 or basin % 3 == 0) else full
        if remain < 15:
            sched = tiny
        Xc = best.copy()
        mode = basin % 7
        if mode == 0:
            Xc = _norm(Xc + 0.010 * rng.standard_normal(Xc.shape))
        elif mode == 1:
            G = Xc @ Xc.T
            np.fill_diagonal(G, -2.0)
            gm = float(G.max())
            cntact = (G > gm - 0.004).sum(axis=1)
            order = np.argsort(-cntact)
            k = 6 + (basin % 5)
            idx = order[:k]
            Ns = 0.004 * rng.standard_normal(Xc.shape)
            Ns[idx] += 0.028 * rng.standard_normal((len(idx), d))
            Xc = _norm(Xc + Ns)
        elif mode == 2:
            wo = worst_order(Xc)[0]
            Xc[int(wo)] = farthest_cand(Xc, int(wo), n_cand=6000)
            Xc = _norm(Xc + 0.003 * rng.standard_normal(Xc.shape))
        elif mode == 3:
            for _ in range(2):
                wo = int(worst_order(Xc)[0])
                Xc[wo] = farthest_cand(Xc, wo, n_cand=5000)
            Xc = _norm(Xc + 0.003 * rng.standard_normal(Xc.shape))
        elif mode == 4:
            i, j, _ = worst_pair(Xc)
            Xc[i] -= 0.012 * Xc[j]
            Xc[j] -= 0.012 * Xc[i]
            Xc = _norm(Xc + 0.006 * rng.standard_normal(Xc.shape))
        elif mode == 5:
            Xc = _norm(Xc + 0.006 * rng.standard_normal(Xc.shape))
        else:
            i, j, _ = worst_pair(Xc)
            r = _norm(rng.standard_normal((1, d)))[0]
            r = r - float(r @ Xc[i]) * Xc[i] - float(r @ Xc[j]) * Xc[j]
            nn = float(np.linalg.norm(r))
            if nn > 1e-9:
                r /= nn
                e = 0.008
                Xc[i] = Xc[i] + e * r
                Xc[j] = Xc[j] + e * r
                Xc = _norm(Xc)
            Xc = _norm(Xc + 0.004 * rng.standard_normal(Xc.shape))
        m = np.zeros_like(Xc); v = np.zeros_like(Xc); cnt = [0]
        Xc = adam(Xc, sched, m, v, cnt)
        Xc, vg2, _ = sweep_single(Xc)
        Xc, vg2 = pair_shift_hill(Xc, trials=12)
        # final tiny balance
        m = np.zeros_like(Xc); v = np.zeros_like(Xc); cnt = [0]
        Xc = adam(Xc, tiny, m, v, cnt)
        vg2 = _mc(Xc)
        if vg2 < best_val - 1e-12:
            best = Xc.copy()
            best_val = float(vg2)
        basin += 1
        if basin > 300:
            break

    best = _norm(best)
    return best
