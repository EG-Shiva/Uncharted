import time
import numpy as np

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def solve():
    import time as _time
    import numpy as _np
    from scipy.optimize import minimize as _minimize
    t0 = _time.time()
    T_END = t0 + 172.0
    RESERVE = 22.0
    rng = _np.random.default_rng(12345)
    _np.random.seed(12345)
    n, d = 249, 6
    I0, J0 = _np.triu_indices(n, 1)

    # load incumbents
    cands = []
    for fn in ("best.npy", "record.npy"):
        try:
            a = _np.load(fn)
            a = _np.asarray(a, dtype=float)
            if a.shape == (n, d):
                an = a / _np.maximum(_np.linalg.norm(a, axis=1, keepdims=True), 1e-12)
                if _np.all(_np.isfinite(an)):
                    cands.append(an)
        except Exception:
            pass
    if not cands:
        a = rng.standard_normal(size=(n, d))
        a /= _np.linalg.norm(a, axis=1, keepdims=True)
        cands.append(a)
    vals = [_max_cos(a) for a in cands]
    bestX = cands[int(_np.argmin(vals))].copy()
    bestVal = float(_np.min(vals))

    def full_soft(Xinit, taus, iters, creg=0.08):
        Xf = (Xinit / _np.maximum(_np.linalg.norm(Xinit, axis=1, keepdims=True), 1e-12)).ravel().copy()
        for tau, mi in zip(taus, iters):
            if _time.time() > T_END - RESERVE:
                break
            tau = float(tau)
            def fg(xf, tau=tau):
                X = xf.reshape(n, d)
                r = _np.sqrt(_np.sum(X*X, axis=1))
                r = _np.maximum(r, 1e-12)
                Xn = X / r[:, None]
                G = Xn @ Xn.T
                g = G[I0, J0]
                gm = _np.max(g)
                e = _np.exp((g-gm)/tau)
                S = _np.sum(e)
                Lsoft = gm + tau*_np.log(S)
                w = e / S
                W = _np.zeros((n, n))
                W[I0, J0] = w
                W = W + W.T
                A = W @ Xn
                proj = _np.sum(A*Xn, axis=1)
                G0 = (A - proj[:, None]*Xn) / r[:, None]
                diff = r - 1.0
                L = Lsoft + creg*_np.sum(diff*diff)
                Gr = G0 + (2.0*creg*diff)[:, None]*Xn
                return float(L), Gr.ravel()
            try:
                res = _minimize(fg, Xf, jac=True, method='L-BFGS-B',
                                options={'maxiter': int(mi), 'maxfun': int(mi*3),
                                         'ftol': 1e-14, 'gtol': 1e-7, 'maxcor': 8})
                Xf = res.x
            except Exception:
                break
            Xm = Xf.reshape(n, d)
            rn = _np.maximum(_np.linalg.norm(Xm, axis=1, keepdims=True), 1e-12)
            Xf = (Xm/rn).ravel()
        Xo = Xf.reshape(n, d)
        Xo /= _np.maximum(_np.linalg.norm(Xo, axis=1, keepdims=True), 1e-12)
        return Xo

    def greedy_descent(Xinit, deadline, step0=2e-3):
        cur = Xinit.copy()
        cur /= _np.maximum(_np.linalg.norm(cur, axis=1, keepdims=True), 1e-12)
        curVal = _max_cos(cur)
        step = float(step0)
        while _time.time() < deadline and step > 5e-13:
            if _time.time() > T_END - 2:
                break
            G = cur @ cur.T
            _np.fill_diagonal(G, -2.0)
            m = float(G.max())
            window = max(10.0*step, 3e-4)
            mask = G > (m - window)
            F = mask.astype(float) @ cur
            dot = _np.sum(F*cur, axis=1)
            Ft = F - dot[:, None]*cur
            trial = cur - step*Ft
            trial /= _np.maximum(_np.linalg.norm(trial, axis=1, keepdims=True), 1e-12)
            v = _max_cos(trial)
            if v < curVal:
                cur = trial
                curVal = v
                step = min(step*1.3, 0.01)
            else:
                step *= 0.5
        return cur, curVal

    def layered_init(split, L):
        # split e.g. 3 means 3+3
        d1 = int(split); d2 = d - d1
        th = _np.linspace(0.12, _np.pi/2-0.12, L)
        X = _np.zeros((n, d))
        perm = rng.permutation(n)
        for idx, k in enumerate(perm):
            li = idx % L
            t = th[li] + 0.03*rng.standard_normal()
            c, s = _np.cos(t), _np.sin(t)
            u = rng.standard_normal(size=(d1,)); u /= _np.linalg.norm(u)
            v = rng.standard_normal(size=(d2,)); v /= _np.linalg.norm(v)
            X[k, :d1] = c*u
            X[k, d1:] = s*v
        # normalize (already)
        X /= _np.linalg.norm(X, axis=1, keepdims=True)
        return X

    def greedy_refill(Xbase, del_idx, pool_size=20000):
        del_idx = _np.asarray(del_idx, dtype=int)
        keep_mask = _np.ones(n, dtype=bool)
        keep_mask[del_idx] = False
        Xkeep = Xbase[keep_mask]
        Pool = rng.standard_normal(size=(pool_size, d))
        Pool /= _np.linalg.norm(Pool, axis=1, keepdims=True)
        D = Pool @ Xkeep.T
        curmax = D.max(axis=1)
        used = _np.zeros(pool_size, dtype=bool)
        new_pts = []
        for _ in range(len(del_idx)):
            curmax[used] = _np.inf
            bi = int(_np.argmin(curmax))
            used[bi] = True
            pt = Pool[bi].copy()
            new_pts.append(pt)
            dd = Pool @ pt
            _np.maximum(curmax, dd, out=curmax)
            curmax[used] = _np.inf
        new_pts = _np.stack(new_pts, axis=0)
        Out = _np.empty_like(Xbase)
        Out[keep_mask] = Xkeep
        Out[del_idx] = new_pts
        return Out

    def crossover(A, B):
        z = rng.standard_normal(size=(d,)); z /= _np.linalg.norm(z)
        da = A @ z; db = B @ z
        # take upper half from A, lower half from B
        ia = _np.where(da > _np.median(da))[0]
        ib = _np.where(db <= _np.median(db))[0]
        Xa = A[ia]; Xb = B[ib]
        C = _np.vstack([Xa, Xb])
        # adjust to exactly n
        if C.shape[0] > n:
            # drop points with largest max-dot greedily
            while C.shape[0] > n:
                G = C @ C.T
                _np.fill_diagonal(G, -2.0)
                rm = G.max(axis=1)
                drop = int(_np.argmax(rm))
                C = _np.delete(C, drop, axis=0)
        elif C.shape[0] < n:
            need = n - C.shape[0]
            extra = rng.standard_normal(size=(need+20, d))
            extra /= _np.linalg.norm(extra, axis=1, keepdims=True)
            # greedy pick need
            D = extra @ C.T
            cm = D.max(axis=1) if C.shape[0]>0 else _np.zeros(extra.shape[0])
            picks = []
            used = _np.zeros(extra.shape[0], dtype=bool)
            for _ in range(need):
                cm[used] = _np.inf
                bi = int(_np.argmin(cm))
                used[bi]=True
                picks.append(extra[bi])
                dd = extra @ extra[bi]
                _np.maximum(cm, dd, out=cm)
            C = _np.vstack([C, _np.stack(picks,axis=0)])
        # shuffle
        C = C[rng.permutation(C.shape[0])]
        return C[:n]

    # light jam of incumbent to set baseline
    try:
        if _time.time() < T_END - 60:
            t = full_soft(bestX, [0.0015, 0.0006], [150, 180], creg=0.08)
            v = _max_cos(t)
            if v < bestVal:
                bestVal = v; bestX = t
    except Exception:
        pass

    archive = [(bestVal, bestX.copy())]

    # quick diverse screening
    trial_count = 0
    # pre-make a couple of random annealed codes for crossover pool
    while _time.time() < T_END - RESERVE - 8:
        trial_count += 1
        mode = trial_count % 5
        try:
            if mode == 1:
                # pure random restart, short anneal
                X0 = rng.standard_normal(size=(n, d))
                X0 /= _np.linalg.norm(X0, axis=1, keepdims=True)
                T = full_soft(X0, [0.012, 0.005, 0.002, 0.0009], [60, 70, 90, 110], creg=0.05)
            elif mode == 2:
                # layered organized restart
                split = int(rng.choice([1,2,3,3,4,5]))
                L = int(rng.choice([4,6,8,12]))
                X0 = layered_init(split, L)
                # small shake to break exact quantization
                X0 = X0 + 0.003*rng.standard_normal(size=(n,d))
                X0 /= _np.linalg.norm(X0, axis=1, keepdims=True)
                T = full_soft(X0, [0.008, 0.003, 0.001, 0.0005], [70, 90, 120, 140], creg=0.06)
            elif mode == 3:
                # cap avalanche delete+refill from best
                center = rng.standard_normal(size=(d,)); center/= _np.linalg.norm(center)
                dots = bestX @ center
                k = int(rng.choice([25,35,50]))
                thr = _np.sort(dots)[-k] if k < n else -2
                # also hub BFS variant half the time
                if trial_count % 2 == 0:
                    G = bestX @ bestX.T
                    _np.fill_diagonal(G, -2.0)
                    rm = G.max(axis=1)
                    i0 = int(_np.argmax(rm))
                    order = _np.argsort(G[i0])[::-1]
                    del_idx = order[:k]
                else:
                    del_idx = _np.where(dots >= thr)[0][:k]
                T0 = greedy_refill(bestX, del_idx, pool_size=18000)
                T = full_soft(T0, [0.004, 0.0015, 0.0006], [90, 120, 150], creg=0.08)
            elif mode == 4:
                # crossover of archive members (or best + random)
                if len(archive) >= 2:
                    a = archive[rng.integers(len(archive))][1]
                    b = archive[rng.integers(len(archive))][1]
                else:
                    a = bestX
                    b = rng.standard_normal(size=(n,d)); b/= _np.linalg.norm(b,axis=1,keepdims=True)
                    b = full_soft(b, [0.01,0.004],[50,60], creg=0.05)
                C0 = crossover(a, b)
                C0 = C0 + 0.002*rng.standard_normal(size=(n,d))
                C0 /= _np.linalg.norm(C0, axis=1, keepdims=True)
                T = full_soft(C0, [0.005, 0.0018, 0.0007], [80, 110, 140], creg=0.07)
            else:
                # global shake of best with larger sigma to change basin
                sig = float(rng.choice([0.01, 0.02, 0.035]))
                X0 = bestX + sig*rng.standard_normal(size=(n,d))
                X0 /= _np.linalg.norm(X0, axis=1, keepdims=True)
                T = full_soft(X0, [0.006, 0.0022, 0.0008], [90, 120, 150], creg=0.08)
            v = _max_cos(T)
            # maintain diverse archive (keep best 6)
            archive.append((v, T.copy()))
            archive.sort(key=lambda t: t[0])
            archive = archive[:6]
            if v < bestVal:
                bestVal = v; bestX = T.copy()
                # extra greedy shave immediately
                if _time.time() < T_END - RESERVE - 15:
                    Tg, vg = greedy_descent(bestX, _time.time()+3.0, step0=1.5e-3)
                    if vg < bestVal:
                        bestVal = vg; bestX = Tg.copy()
                        archive[0]=(bestVal,bestX.copy())
        except Exception:
            continue
        if trial_count > 40:
            break

    # final intensive jam from best
    try:
        if _time.time() < T_END - 18:
            t = full_soft(bestX, [0.003, 0.0012, 0.0005, 0.00022], [200, 250, 280, 320], creg=0.06)
            v = _max_cos(t)
            if v < bestVal:
                bestVal = v; bestX = t
    except Exception:
        pass
    try:
        Tg, vg = greedy_descent(bestX, T_END-1.5, step0=1.2e-3)
        if vg < bestVal:
            bestVal = vg; bestX = Tg.copy()
    except Exception:
        pass

    bestX /= _np.maximum(_np.linalg.norm(bestX, axis=1, keepdims=True), 1e-12)
    return bestX
