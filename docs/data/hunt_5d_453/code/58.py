import time
import numpy as np

def solve():
    t0 = time.time()
    DEAD = t0 + 172.0
    rng = np.random.default_rng(0)
    np.random.seed(0)
    n, d = 453, 5

    def max_cos(X):
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        return float(np.max(G))

    # load incumbent
    cands = []
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=np.float64)
            if a.shape == (n, d):
                nn = np.linalg.norm(a, axis=1, keepdims=True)
                nn[nn == 0] = 1.0
                a = a / nn
                if np.all(np.isfinite(a)):
                    cands.append(a)
        except Exception:
            pass
    if not cands:
        a = rng.standard_normal((n, d))
        a /= np.linalg.norm(a, axis=1, keepdims=True)
        cands.append(a)
    vals = sorted([max_cos(c) for c in cands])
    best_val = float(vals[0])
    # pick the best candidate
    best_X = None
    for c in cands:
        if abs(max_cos(c) - best_val) < 1e-15:
            best_X = c.copy()
            break
    best = [best_val]

    def anneal(x, steps, tau0, tau1, lr0, lr1, mu):
        v = np.zeros_like(x)
        ltau = np.log(tau1 / tau0)
        llr = np.log(lr1 / lr0)
        for it in range(steps):
            if (it & 31) == 0 and time.time() > DEAD - 6.0:
                break
            frac = it / max(steps - 1, 1)
            tau = tau0 * np.exp(ltau * frac)
            lr = lr0 * np.exp(llr * frac)
            G = x @ x.T
            np.fill_diagonal(G, -2.0)
            m = float(np.max(G))
            if m < best[0]:
                best[0] = m
                # copy out via best_X holder
                best_X_holder[0] = x.copy()
            P = np.exp((G - m) / tau)
            np.fill_diagonal(P, 0.0)
            Grad = P @ x
            Gt = Grad - np.sum(Grad * x, axis=1, keepdims=True) * x
            v = mu * v + (1.0 - mu) * Gt
            x = x - lr * v
            x /= np.linalg.norm(x, axis=1, keepdims=True)
            if m > best[0] + 0.02:
                x = best_X_holder[0].copy()
                v *= 0.5
        return x

    best_X_holder = [best_X.copy()]

    def sync():
        nonlocal best_X, best_val
        if best_X_holder[0] is not None and best[0] < best_val:
            best_val = float(best[0])
            best_X = best_X_holder[0].copy()

    # quick initial polish of incumbent
    x = best_X.copy()
    x = anneal(x, 700, 0.008, 0.0007, 6e-4, 6e-5, 0.82)
    sync()

    def largest_holes(Xref, exclude, k, n_cand=3500):
        C = rng.standard_normal((n_cand, d))
        C /= np.linalg.norm(C, axis=1, keepdims=True)
        D = C @ Xref.T
        if len(exclude) > 0:
            D[:, list(exclude)] = -2.0
        cmax = D.max(axis=1)
        order = np.argsort(cmax)
        picked = []
        for idx in order:
            c = C[idx]
            ok = True
            for q in picked:
                if float(c @ q) > 0.90:
                    ok = False
                    break
            if ok:
                picked.append(c.copy())
                if len(picked) >= k:
                    break
        return picked

    def row_max_and_deg(X):
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        m = float(np.max(G))
        rmax = G.max(axis=1)
        deg = np.sum(G > m - 1e-9, axis=1)
        return G, m, rmax, deg

    cyc = 0
    no_imp = 0
    # main hopping; reserve ~20s for final polish
    while time.time() < DEAD - 22.0:
        cyc += 1
        if time.time() > DEAD - 22.0:
            break
        base = best_X.copy()
        G0, m0, rmax0, deg0 = row_max_and_deg(base)
        flat = int(np.argmax(G0))
        wi = flat // n
        wj = flat % n
        mode = cyc % 6
        # adaptive sigma
        if no_imp > 8:
            sig = 0.035
        elif no_imp > 4:
            sig = 0.018
        else:
            sig = [0.006, 0.012, 0.022, 0.008][cyc % 4]
        y = base.copy()
        if mode == 0:
            # random subset shake
            k = 30 + (cyc % 3) * 15
            idx = rng.choice(n, size=k, replace=False)
            nz = rng.standard_normal((k, d)) * sig
            for t, ii in enumerate(idx):
                v_ = nz[t] - float(nz[t] @ y[ii]) * y[ii]
                y[ii] = y[ii] + v_
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            steps, tau0, tau1, lr0, lr1, mu = 450, 0.007, 0.0007, 5e-4, 5e-5, 0.82
        elif mode == 1:
            # worst neighbourhood shake
            s = {int(wi), int(wj)}
            for kk in (int(wi), int(wj)):
                row = G0[kk].copy()
                row[kk] = -2.0
                tn = np.argpartition(row, -8)[-8:]
                for tt in tn:
                    s.add(int(tt))
            for _ in range(12):
                s.add(int(rng.integers(0, n)))
            arr = np.array(list(s), dtype=int)
            nz = rng.standard_normal((len(arr), d)) * (sig * 1.2)
            for t, ii in enumerate(arr):
                v_ = nz[t] - float(nz[t] @ y[ii]) * y[ii]
                y[ii] = y[ii] + v_
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            steps, tau0, tau1, lr0, lr1, mu = 500, 0.006, 0.0006, 5e-4, 5e-5, 0.85
        elif mode == 2:
            # single ejection: worst-degree or worst-rowmax to hole
            if cyc % 2 == 0:
                worst = int(np.argmax(rmax0))
            else:
                worst = int(np.argmax(deg0))
            holes = largest_holes(y, [worst], 1)
            if holes:
                y[worst] = holes[0]
            # shake neighbours of hole/old position lightly
            idx = rng.choice(n, size=15, replace=False)
            nz = rng.standard_normal((15, d)) * (sig * 0.7)
            for t, ii in enumerate(idx):
                v_ = nz[t] - float(nz[t] @ y[ii]) * y[ii]
                y[ii] = y[ii] + v_
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            steps, tau0, tau1, lr0, lr1, mu = 550, 0.008, 0.0005, 6e-4, 4e-5, 0.82
        elif mode == 3:
            # double ejection
            o1 = np.argsort(rmax0)[-6:]
            o2 = np.argsort(deg0)[-6:]
            pool = list(dict.fromkeys(list(o1) + list(o2) + [int(wi), int(wj)]))
            # pick 2-3 distinct worst
            rng.shuffle(pool)
            excl = pool[:3] if len(pool) >= 3 else pool[:2]
            holes = largest_holes(y, excl, len(excl))
            for e, h in zip(excl, holes):
                y[int(e)] = h
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            steps, tau0, tau1, lr0, lr1, mu = 600, 0.009, 0.0005, 6e-4, 4e-5, 0.80
        elif mode == 4:
            # cap rotation: pick random axis, rotate cap
            u = rng.standard_normal(d)
            u /= np.linalg.norm(u)
            proj = y @ u
            thr = np.quantile(proj, 0.80)
            mask = proj > thr
            idx = np.where(mask)[0]
            if idx.size >= 5:
                # random small rotation in 5D: Q = I + skew
                A = rng.standard_normal((d, d)) * 0.35
                A = A - A.T
                # exp approx via Cayley or just I+A normalized per-point tangentially
                rot = A - np.mean(A, axis=0, keepdims=True) * 0
                for ii in idx:
                    v_ = rot @ y[ii]
                    v_ = v_ - float(v_ @ y[ii]) * y[ii]
                    y[ii] = y[ii] + v_ * 0.35
                y /= np.linalg.norm(y, axis=1, keepdims=True)
            # plus small global noise
            nz = rng.standard_normal((n, d)) * (sig * 0.25)
            nz -= np.sum(nz * y, axis=1, keepdims=True) * y
            y = y + nz
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            steps, tau0, tau1, lr0, lr1, mu = 550, 0.008, 0.0006, 5e-4, 5e-5, 0.83
        else:
            # overload shake: shake top-degree points harder
            top = np.argsort(deg0)[-25:]
            nz = rng.standard_normal((25, d)) * (sig * 1.5)
            for t, ii in enumerate(top):
                v_ = nz[t] - float(nz[t] @ y[int(ii)]) * y[int(ii)]
                y[int(ii)] = y[int(ii)] + v_
            # plus random 10
            idx = rng.choice(n, size=10, replace=False)
            nz2 = rng.standard_normal((10, d)) * sig
            for t, ii in enumerate(idx):
                v_ = nz2[t] - float(nz2[t] @ y[ii]) * y[ii]
                y[ii] = y[ii] + v_
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            steps, tau0, tau1, lr0, lr1, mu = 450, 0.005, 0.0006, 4e-4, 5e-5, 0.85

        prev = float(best[0])
        y = anneal(y, steps, tau0, tau1, lr0, lr1, mu)
        sync()
        if best[0] < prev - 1e-12:
            no_imp = 0
        else:
            no_imp += 1

    # final polish on best
    sync()
    x = best_X.copy()
    # low-temp anneal
    if time.time() < DEAD - 14.0:
        x = anneal(x, 1200, 0.003, 0.0002, 2.5e-4, 2e-5, 0.88)
        sync()
        x = best_X.copy()
    # greedy worst-pair micro shaving
    eps = 4e-4
    it_n = 900
    for it in range(it_n):
        if (it % 60) == 0 and time.time() > DEAD - 2.5:
            break
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        m = float(np.max(G))
        if m < best[0]:
            best[0] = m
            best_X_holder[0] = x.copy()
        flat = int(np.argmax(G))
        i = flat // n
        j = flat % n
        cij = float(x[i] @ x[j])
        ti = x[j] - cij * x[i]
        ni = float(np.linalg.norm(ti))
        if ni < 1e-12:
            continue
        ti /= ni
        tj = x[i] - cij * x[j]
        nj = float(np.linalg.norm(tj))
        if nj < 1e-12:
            continue
        tj /= nj
        done = False
        for sc in (eps, eps * 0.5):
            vi = x[i] - sc * ti
            vi /= np.linalg.norm(vi)
            vj = x[j] - sc * tj
            vj /= np.linalg.norm(vj)
            oi = x[i].copy()
            oj = x[j].copy()
            x[i] = vi
            x[j] = vj
            G2 = x @ x.T
            np.fill_diagonal(G2, -2.0)
            m2 = float(np.max(G2))
            if m2 < m - 1e-13:
                if m2 < best[0]:
                    best[0] = m2
                    best_X_holder[0] = x.copy()
                done = True
                break
            else:
                x[i] = oi
                x[j] = oj
        if done:
            eps = min(eps * 1.02, 6e-4)
        else:
            eps *= 0.97
            if eps < 2e-06:
                eps = 2e-06
    sync()
    X = best_X.copy()
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    return X
