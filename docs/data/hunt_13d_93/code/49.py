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

def solve():
    import time
    import numpy as np
    t0 = time.time()
    LIM = 170.0
    def expired():
        return (time.time() - t0) > LIM
    n, d = 93, 13
    rng = np.random.default_rng(0)
    iu0, iu1 = np.triu_indices(n, 1)
    P = len(iu0)

    bestX = None
    best_val = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=np.float64)
            if a.shape != (93, 13):
                continue
            if not np.all(np.isfinite(a)):
                continue
            a = _norm(a)
            v = _maxcos(a)
            if v < best_val:
                best_val = v
                bestX = a.copy()
        except Exception:
            pass
    if bestX is None:
        bestX = _norm(rng.standard_normal((n, d)))
        best_val = _maxcos(bestX)
    X0 = bestX.copy()

    # ---------- smoothed Adam ----------
    def adam_run(Xin, steps, tau0, tau1, lr0, lr1, mode):
        Xc = Xin.copy()
        m = np.zeros_like(Xc)
        v = np.zeros_like(Xc)
        tt = 0
        loc_best = _maxcos(Xc)
        loc_X = Xc.copy()
        ltau = np.log(tau1 / tau0) if tau0 != tau1 and tau0 > 0 and tau1 > 0 else 0.0
        llr = np.log(lr1 / lr0) if lr0 != lr1 and lr0 > 0 and lr1 > 0 else 0.0
        for s in range(steps):
            if (s % 50 == 0) and expired():
                break
            f = s / max(1, steps - 1)
            tau = tau0 * np.exp(ltau * f)
            lr = lr0 * np.exp(llr * f)
            if tau < 1e-12:
                tau = 1e-12
            G = Xc @ Xc.T
            np.fill_diagonal(G, -2.0)
            if mode == 'balanced':
                rmax = G.max(axis=1, keepdims=True)
                W = np.exp((G - rmax) / tau)
                np.fill_diagonal(W, 0.0)
                rs = W.sum(axis=1, keepdims=True) + 1e-12
                Wn = W / rs
                Ws = Wn + Wn.T
                grad = Ws @ Xc
            else:
                mg = float(G.max())
                W = np.exp((G - mg) / tau)
                np.fill_diagonal(W, 0.0)
                grad = W @ Xc
            dot = np.sum(grad * Xc, axis=1, keepdims=True)
            gt = grad - dot * Xc
            if not np.all(np.isfinite(gt)):
                continue
            tt += 1
            m = 0.9 * m + 0.1 * gt
            vv = 0.999 * v + 0.001 * (gt * gt)
            v = vv
            mh = m / (1.0 - 0.9 ** tt)
            vh = v / (1.0 - 0.999 ** tt)
            step = lr * mh / (np.sqrt(vh) + 1e-8)
            step = step - np.sum(step * Xc, axis=1, keepdims=True) * Xc
            Xc = Xc - step
            Xc = _norm(Xc)
            if (s % 100 == 99) or (s == steps - 1):
                cur = _maxcos(Xc)
                if cur < loc_best:
                    loc_best = cur
                    loc_X = Xc.copy()
        return loc_X, loc_best

    def find_void(others, rng_, n_cand=2500):
        dd = others.shape[1]
        C = rng_.standard_normal((n_cand, dd))
        C /= np.linalg.norm(C, axis=1, keepdims=True) + 1e-16
        D = C @ others.T
        worst = D.max(axis=1)
        bi = int(np.argmin(worst))
        vv = C[bi].copy()
        lr = 0.05
        tau = 0.008
        for _ in range(120):
            dots = others @ vv
            m0 = float(dots.max())
            w = np.exp((dots - m0) / max(tau, 1e-12))
            w /= (w.sum() + 1e-12)
            grad = w @ others
            gt = grad - float(np.dot(grad, vv)) * vv
            vv = vv - lr * gt
            nn = float(np.linalg.norm(vv))
            if nn == 0 or not np.isfinite(nn):
                break
            vv /= nn
            lr *= 0.99
            tau = max(tau * 0.995, 0.0012)
            if expired():
                break
        return vv

    def greedy_run(Xin, iters, K, step0):
        Xc = _norm(Xin.copy())
        cur = _maxcos(Xc)
        bX = Xc.copy()
        bv = cur
        step = float(step0)
        grad = np.empty_like(Xc)
        for it in range(iters):
            if (it % 40 == 0) and expired():
                break
            G = Xc @ Xc.T
            c = G[iu0, iu1]
            if K < P:
                idx = np.argpartition(-c, K)[:K]
            else:
                idx = np.arange(P)
            I = iu0[idx]; J = iu1[idx]
            grad.fill(0.0)
            np.add.at(grad, I, Xc[J])
            np.add.at(grad, J, Xc[I])
            grad /= float(len(idx))
            dot = np.sum(grad * Xc, axis=1, keepdims=True)
            gt = grad - dot * Xc
            if not np.all(np.isfinite(gt)):
                step *= 0.5
                continue
            improved = False
            # try scaled steps
            for alpha in (1.0, 0.5, 0.25):
                Xt = _norm(Xc - step * alpha * gt)
                vt = _maxcos(Xt)
                if vt < bv - 1e-15:
                    bv = vt
                    bX = Xt.copy()
                    Xc = Xt
                    improved = True
                    step = min(step * 1.2, 0.01)
                    break
                elif vt < cur - 1e-15:
                    cur = vt
                    Xc = Xt
                    improved = True
                    break
            if not improved:
                step *= 0.5
                if step < 1e-13:
                    step = float(step0) * 0.4
        return bX, bv

    def rowmax_vec(X):
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        return G.max(axis=1), float(G.max())

    # initial balancing from record
    curX = bestX.copy()
    cur_val = best_val
    try:
        Xb, vb = adam_run(curX, steps=900, tau0=0.015, tau1=0.0025, lr0=0.02, lr1=0.005, mode='balanced')
        if vb < best_val:
            best_val = vb; bestX = Xb.copy()
        curX, cur_val = Xb, vb
        Xg, vg = adam_run(curX, steps=900, tau0=0.008, tau1=0.0009, lr0=0.015, lr1=0.004, mode='global')
        if vg < best_val:
            best_val = vg; bestX = Xg.copy()
        curX, cur_val = Xg, vg
        # keep explorer at best
        if vg > best_val:
            curX = bestX.copy(); cur_val = best_val
        else:
            curX = bestX.copy(); cur_val = best_val
    except Exception:
        pass

    # basin hopping with void surgery
    it_outer = 0
    while not expired():
        it_outer += 1
        if it_outer > 200:
            break
        try:
            rm, gm = rowmax_vec(curX)
        except Exception:
            curX = bestX.copy(); cur_val = best_val
            continue
        # worst pair
        try:
            G = curX @ curX.T
            np.fill_diagonal(G, -2.0)
            c = G[iu0, iu1]
            am = int(np.argmax(c))
            a = int(iu0[am]); b = int(iu1[am])
        except Exception:
            continue
        improved_outer = False
        # try void replacement for a and b
        for target in (a, b):
            if expired():
                break
            try:
                mask = np.ones(n, dtype=bool); mask[target] = False
                others = curX[mask]
                vv = find_void(others, rng)
                Xt = curX.copy(); Xt[target] = vv
                # short polish
                Xp, _ = adam_run(Xt, steps=250, tau0=0.005, tau1=0.001, lr0=0.01, lr1=0.003, mode='global')
                Xp2, _ = adam_run(Xp, steps=200, tau0=0.004, tau1=0.001, lr0=0.008, lr1=0.003, mode='balanced')
                vp = _maxcos(Xp2)
                if vp < best_val - 1e-12:
                    best_val = vp; bestX = Xp2.copy()
                    curX = Xp2.copy(); cur_val = vp
                    improved_outer = True
                    break
                elif vp < cur_val - 1e-12:
                    curX = Xp2.copy(); cur_val = vp
                    improved_outer = True
                    break
            except Exception:
                continue
        if improved_outer:
            continue
        # hub neighbourhood jitter + polish
        try:
            order = np.argsort(-rm)[:6]
            Xt = curX.copy()
            noise = rng.standard_normal((len(order), d)) * 0.025
            for k, idx_ in enumerate(order):
                p = noise[k] - float(np.dot(noise[k], Xt[idx_])) * Xt[idx_]
                Xt[idx_] = Xt[idx_] + p
            Xt = _norm(Xt)
            Xp, _ = adam_run(Xt, steps=350, tau0=0.006, tau1=0.0012, lr0=0.012, lr1=0.003, mode='balanced')
            Xp, _ = adam_run(Xp, steps=350, tau0=0.005, tau1=0.0008, lr0=0.010, lr1=0.003, mode='global')
            vp = _maxcos(Xp)
            if vp < best_val - 1e-12:
                best_val = vp; bestX = Xp.copy()
                curX = Xp.copy(); cur_val = vp
                continue
            # sideways: accept if more balanced and not much worse
            rm2, gm2 = rowmax_vec(Xp)
            # balance: std and min
            std0 = float(np.std(rm)); std2 = float(np.std(rm2))
            min0 = float(rm.min()); min2 = float(rm2.min())
            if (vp < cur_val + 5e-4 and (std2 < std0 - 1e-6 or min2 > min0 + 2e-4)):
                curX = Xp.copy(); cur_val = vp
                continue
            if vp < cur_val:
                curX = Xp.copy(); cur_val = vp
                continue
            # occasional restart from best with kick
            if it_outer % 6 == 0:
                kick = rng.standard_normal((n, d)) * 0.015
                kick -= np.sum(kick * bestX, axis=1, keepdims=True) * bestX
                Xk = _norm(bestX + kick)
                Xk2, _ = adam_run(Xk, steps=300, tau0=0.006, tau1=0.001, lr0=0.01, lr1=0.003, mode='global')
                vk2 = _maxcos(Xk2)
                if vk2 < best_val:
                    best_val = vk2; bestX = Xk2.copy()
                curX = Xk2.copy(); cur_val = vk2
                # if explorer too bad, reset to best
                if cur_val > best_val + 8e-4:
                    curX = bestX.copy(); cur_val = best_val
            else:
                if cur_val > best_val + 8e-4:
                    curX = bestX.copy(); cur_val = best_val
        except Exception:
            curX = bestX.copy(); cur_val = best_val
            continue
        if expired():
            break

    # final polish on best
    try:
        if not expired():
            Xf, vf = adam_run(bestX, steps=800, tau0=0.003, tau1=0.0006, lr0=0.008, lr1=0.002, mode='global')
            if vf < best_val:
                best_val = vf; bestX = Xf.copy()
        if not expired():
            Xf2, vf2 = greedy_run(bestX, iters=600, K=400, step0=0.004)
            if vf2 < best_val:
                best_val = vf2; bestX = Xf2.copy()
        if not expired():
            Xf3, vf3 = adam_run(bestX, steps=400, tau0=0.0015, tau1=0.0004, lr0=0.004, lr1=0.001, mode='balanced')
            if vf3 < best_val:
                best_val = vf3; bestX = Xf3.copy()
    except Exception:
        pass
    bestX = _norm(bestX)
    # guard finite
    bestX = np.where(np.isfinite(bestX), bestX, 0.0)
    # renormalize after guard
    nr = np.linalg.norm(bestX, axis=1, keepdims=True)
    nr[nr == 0] = 1.0
    bestX = bestX / nr
    # if somehow NaN, fallback
    if not np.all(np.isfinite(bestX)):
        bestX = _norm(X0)
    return bestX
