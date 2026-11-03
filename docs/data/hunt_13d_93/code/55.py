import time
import numpy as np

def _norm(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return X / n

def _maxcos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def _rowmax(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return G.max(axis=1)

def solve():
    t0 = time.time()
    BUDGET = 170.0
    n, d = 93, 13
    rng = np.random.default_rng(7)

    iu0, iu1 = np.triu_indices(n, 1)

    base_best = None
    base_val = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=np.float64)
            if a.shape != (n, d):
                continue
            a = _norm(a)
            v = _maxcos(a)
            if v < base_val:
                base_val = v
                base_best = a.copy()
        except Exception:
            pass
    if base_best is None:
        base_best = _norm(rng.standard_normal((n, d)))
        base_val = _maxcos(base_best)

    # ---------- adam polish ----------
    def adam_polish(X, steps, tau0, tau1, lr0, lr1, mode, deadline=1e18):
        X = X.copy()
        m = np.zeros_like(X)
        v = np.zeros_like(X)
        b1, b2, eps = 0.9, 0.999, 1e-8
        tt = 0
        ltau = np.log(tau1 / tau0) if tau0 != tau1 and tau0 > 0 and tau1 > 0 else 0.0
        llr = np.log(lr1 / lr0) if lr0 != lr1 and lr0 > 0 and lr1 > 0 else 0.0
        for s in range(steps):
            if (s % 60 == 0) and time.time() > deadline:
                break
            f = s / max(1, steps - 1)
            tau = tau0 * np.exp(ltau * f)
            lr = lr0 * np.exp(llr * f)
            if tau < 1e-12:
                tau = 1e-12
            G = X @ X.T
            np.fill_diagonal(G, -2.0)
            if mode == 'bal':
                rmax = G.max(axis=1, keepdims=True)
                W = np.exp((G - rmax) / tau)
                np.fill_diagonal(W, 0.0)
                rs = W.sum(axis=1, keepdims=True) + 1e-12
                Wn = W / rs
                Ws = Wn + Wn.T
                grad = Ws @ X
            else:
                mg = float(G.max())
                W = np.exp((G - mg) / tau)
                np.fill_diagonal(W, 0.0)
                # normalize to avoid overflow scale (adam invariant but stabilizes)
                ss = float(W.sum()) + 1e-12
                grad = (W @ X) / (ss / n)
            dot = np.sum(grad * X, axis=1, keepdims=True)
            gt = grad - dot * X
            if not np.all(np.isfinite(gt)):
                continue
            tt += 1
            m = b1 * m + (1.0 - b1) * gt
            v = b2 * v + (1.0 - b2) * (gt * gt)
            mh = m / (1.0 - b1 ** tt)
            vh = v / (1.0 - b2 ** tt)
            step = lr * mh / (np.sqrt(vh) + eps)
            d2 = np.sum(step * X, axis=1, keepdims=True)
            step_t = step - d2 * X
            X = X - step_t
            X = _norm(X)
        return X

    def find_void(others, rng_, n_cand=3000):
        dc = others.shape[1]
        C = rng_.standard_normal((n_cand, dc))
        C /= np.linalg.norm(C, axis=1, keepdims=True) + 1e-300
        D = C @ others.T
        worst = D.max(axis=1)
        bi = int(np.argmin(worst))
        vv = C[bi].copy()
        tau = 0.008
        lr = 0.08
        for _ in range(150):
            dots = others @ vv
            m0 = float(dots.max())
            w = np.exp((dots - m0) / tau)
            s = float(w.sum()) + 1e-300
            w /= s
            grad = w @ others
            gt = grad - float(np.dot(grad, vv)) * vv
            vv = vv - lr * gt
            nn = float(np.linalg.norm(vv))
            if nn == 0 or not np.isfinite(nn):
                break
            vv /= nn
            lr *= 0.995
            tau *= 0.997
            if tau < 0.002:
                tau = 0.002
        return vv / (np.linalg.norm(vv) + 1e-300)

    bestX = base_best.copy()
    best_val = float(base_val)
    cur = base_best.copy()
    cur_val = float(base_val)

    # initial equalization to balanced plateau
    try:
        cur = adam_polish(cur, 900, 0.010, 0.003, 0.008, 0.002, 'bal', deadline=t0 + BUDGET - 25)
        cur = adam_polish(cur, 700, 0.004, 0.0006, 0.004, 0.001, 'glob', deadline=t0 + BUDGET - 25)
        vv = _maxcos(cur)
        if vv < best_val:
            best_val = vv
            bestX = cur.copy()
        cur_val = vv
        # if initial got worse than base, reset cur to best
        if cur_val > best_val + 0.001:
            cur = bestX.copy()
            cur_val = best_val
    except Exception:
        pass

    # ---------- basin hopping ----------
    it = 0
    DEAD_POLISH = t0 + 148.0
    while time.time() < DEAD_POLISH:
        it += 1
        r = float(rng.random())
        prop = None
        try:
            if r < 0.38:
                # perturb worst points
                rm = _rowmax(cur)
                k = 6 if rng.random() < 0.7 else 10
                worst = np.argpartition(-rm, k)[:k]
                prop = cur.copy()
                sig = 0.035 if rng.random() < 0.5 else 0.06
                nz = rng.standard_normal((k, d)) * sig
                for ii, idx in enumerate(worst):
                    nz[ii] -= float(np.dot(nz[ii], prop[idx])) * prop[idx]
                prop[worst] = prop[worst] + nz
                prop = _norm(prop)
            elif r < 0.65:
                prop = cur.copy()
                sig = 0.012 if rng.random() < 0.6 else 0.025
                nz = rng.standard_normal(prop.shape) * sig
                nz -= (np.sum(nz * prop, axis=1, keepdims=True)) * prop
                prop = _norm(prop + nz)
            elif r < 0.85:
                prop = cur.copy()
                G = prop @ prop.T
                np.fill_diagonal(G, -2.0)
                rm = G.max(axis=1)
                order = np.argsort(rm)  # smallest first = rattlers
                nrep = 1 if rng.random() < 0.6 else 2
                # pick among 8 smallest randomly to diversify
                cand_pool = order[:8]
                chosen = rng.choice(cand_pool, size=nrep, replace=False)
                for idx in chosen:
                    others = np.delete(prop, int(idx), axis=0)
                    vv2 = find_void(others, rng)
                    # small random tilt to avoid exact symmetry
                    tilt = rng.standard_normal(d) * 0.002
                    tilt -= float(np.dot(tilt, vv2)) * vv2
                    vv2 = vv2 + tilt
                    vv2 /= np.linalg.norm(vv2) + 1e-300
                    prop[int(idx)] = vv2
            else:
                prop = cur.copy()
                G = prop @ prop.T
                np.fill_diagonal(G, -2.0)
                cmax = float(G.max())
                # hub with loose threshold to capture plateau load
                cnts = (G > cmax - 0.0015).sum(axis=1)
                hub = int(np.argmax(cnts))
                neigh = np.where(G[hub] > cmax - 0.003)[0]
                neigh = neigh[:10]
                idxs = np.unique(np.concatenate([[hub], neigh]))
                sig = 0.03
                nz = rng.standard_normal((len(idxs), d)) * sig
                for ii, idx in enumerate(idxs):
                    nz[ii] -= float(np.dot(nz[ii], prop[int(idx)])) * prop[int(idx)]
                prop[idxs] = prop[idxs] + nz
                prop = _norm(prop)
        except Exception:
            continue
        # short polish
        try:
            prop = adam_polish(prop, 220, 0.006, 0.004, 0.006, 0.003, 'bal', deadline=t0 + BUDGET - 12)
            prop = adam_polish(prop, 220, 0.0025, 0.0008, 0.004, 0.0012, 'glob', deadline=t0 + BUDGET - 12)
        except Exception:
            continue
        try:
            v = _maxcos(prop)
        except Exception:
            continue
        if v < cur_val - 1e-12:
            cur = prop
            cur_val = v
            if v < best_val:
                best_val = v
                bestX = prop.copy()
        elif v < cur_val + 0.002:
            # simulated-annealing uphill acceptance to change topology
            T = 0.00025
            if float(rng.random()) < np.exp((cur_val - v) / T):
                cur = prop
                cur_val = v
        # occasional restart from best with fresh kick if stuck
        if it % 25 == 0 and time.time() < DEAD_POLISH:
            # intensify best a bit
            if cur_val > best_val + 0.0005:
                cur = bestX.copy()
                nz = rng.standard_normal(cur.shape) * 0.02
                nz -= (np.sum(nz * cur, axis=1, keepdims=True)) * cur
                cur = _norm(cur + nz)
                cur_val = _maxcos(cur)

    # ---------- final polish of best ----------
    try:
        Xf = bestX.copy()
        # balanced equalization
        Xf = adam_polish(Xf, 700, 0.005, 0.002, 0.004, 0.0015, 'bal', deadline=t0 + BUDGET - 3)
        # annealed hard-max squeeze
        Xf = adam_polish(Xf, 900, 0.0015, 0.00035, 0.003, 0.0008, 'glob', deadline=t0 + BUDGET - 3)
        Xf = adam_polish(Xf, 900, 0.0004, 0.00012, 0.0012, 0.0004, 'glob', deadline=t0 + BUDGET - 2)
        # greedy top-K hard descent
        nG = 500
        step = 0.002
        best_loc = _maxcos(Xf)
        best_loc_X = Xf.copy()
        P = len(iu0)
        for gi in range(nG):
            if gi % 50 == 0 and time.time() > t0 + BUDGET - 2:
                break
            frac = gi / max(1, nG - 1)
            K = int(round(120 - 100 * frac))
            K = max(8, min(K, P))
            G = Xf @ Xf.T
            c = G[iu0, iu1]
            idx = np.argpartition(-c, K)[:K]
            I = iu0[idx]
            J = iu1[idx]
            grad = np.zeros_like(Xf)
            np.add.at(grad, I, Xf[J])
            np.add.at(grad, J, Xf[I])
            grad /= float(K)
            dot = np.sum(grad * Xf, axis=1, keepdims=True)
            gt = grad - dot * Xf
            Xt = _norm(Xf - step * gt)
            vt = _maxcos(Xt)
            if vt < best_loc:
                best_loc = vt
                best_loc_X = Xt.copy()
                Xf = Xt
                step = min(step * 1.25, 0.005)
            else:
                step *= 0.6
                if step < 1e-6:
                    step = 0.001
        vf = _maxcos(best_loc_X)
        if vf < best_val:
            best_val = vf
            bestX = best_loc_X.copy()
    except Exception:
        pass

    return _norm(bestX)
