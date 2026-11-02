import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _lp_polish(x, seconds, t_deadline=None):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds:
        if t_deadline is not None and time.time() > t_deadline:
            break
        if radius < 1e-14:
            break
        G = x @ x.T
        iu_c = G[iu]
        m = float(iu_c.max())
        sel = iu_c > m - max(20 * radius, 1e-9)
        I = iu[0][sel]; J = iu[1][sel]
        c = iu_c[sel]
        p = len(I)
        if p == 0:
            break
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cc = np.empty((p, 2 * d + 1), dtype=float)
        cc[:, :d] = x[J]
        cc[:, d:2*d] = x[I]
        cc[:, 2*d] = -1.0
        cols_I = (I[:, None] * d + np.arange(d)).ravel()
        cols_J = (J[:, None] * d + np.arange(d)).ravel()
        cols = np.empty(p * (2*d+1), dtype=int)
        # interleave: for each row k, cols = [I*d .., J*d .., n*d]
        # build via loop-free concat per block
        # rows are repeated, so order is row-major
        tmp = np.empty((p, 2*d+1), dtype=int)
        tmp[:, :d] = I[:, None]*d + np.arange(d)
        tmp[:, d:2*d] = J[:, None]*d + np.arange(d)
        tmp[:, 2*d] = n*d
        cols = tmp.ravel()
        vals = cc.ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n*d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv); obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-c, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)]*(n*d)+[(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        v = _max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius*1.5, 0.05)
        else:
            radius *= 0.5
    return x, best

def _smooth_descent(x0, steps, T0, T1, lr0, mu, rng, noise0=0.0, t_deadline=None):
    x = x0.copy()
    n, d = x.shape
    v = np.zeros_like(x)
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    best_val = float(G.max())
    best = x.copy()
    log_ratio = np.log(T1/T0) if T0>0 and T1>0 else 0.0
    for k in range(steps):
        if t_deadline is not None and k % 500 == 0 and time.time() > t_deadline:
            break
        frac = k / max(1, steps-1)
        T = T0 * np.exp(log_ratio*frac)
        if T < 1e-12:
            T = 1e-12
        lr = lr0 * (1.0 - 0.55*frac)
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        # weights
        # use in-place-ish
        E = np.exp((G - m)/T)
        np.fill_diagonal(E, 0.0)
        S = E.sum()*0.5
        if S < 1e-300:
            S = 1e-300
        grad = (E @ x)/S
        dot = (grad*x).sum(axis=1, keepdims=True)
        grad -= dot*x
        # heavy ball (EMA)
        v = mu*v + (1.0-mu)*grad
        x = x - lr*v
        if noise0 > 0:
            amp = noise0*(1.0-frac)
            if amp > 0:
                nz = rng.randn(n, d)*amp
                nz -= (nz*x).sum(axis=1, keepdims=True)*x
                x = x + nz
        x /= np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
        if m < best_val:
            best_val = m
            best = x.copy()
    return best, best_val

def _find_hole(x_fixed, rng, n_samples=12000):
    Y = rng.randn(n_samples, 4)
    Y /= np.linalg.norm(Y, axis=1, keepdims=True)
    D = Y @ x_fixed.T
    vals = D.max(axis=1)
    idx = int(np.argmin(vals))
    y = Y[idx].copy()
    vy = float(vals[idx])
    # refine: minimise max dot by tangential steps away from closest point
    for _ in range(250):
        dots = x_fixed @ y
        j = int(np.argmax(dots))
        g = x_fixed[j] - dots[j]*y
        ng = np.linalg.norm(g)
        if ng < 1e-12:
            break
        step = 0.02
        yn = y - step*g/ng
        yn /= np.linalg.norm(yn)
        vn = float((x_fixed @ yn).max())
        if vn < vy:
            y, vy = yn, vn
        else:
            # try smaller
            step = 0.005
            yn2 = y - step*g/ng
            yn2 /= np.linalg.norm(yn2)
            vn2 = float((x_fixed @ yn2).max())
            if vn2 < vy:
                y, vy = yn2, vn2
            else:
                break
    return y, vy

def solve():
    t0 = time.time()
    DEADLINE = t0 + 172.0
    SEARCH_END = t0 + 132.0
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
    # quick LP squeeze of starting point is already done in parent; skip

    trial = 0
    # precompute sigmas
    while time.time() < SEARCH_END:
        if time.time() > DEADLINE - 35:
            break
        trial += 1
        mode = trial % 6
        if mode == 0:
            # full random restart, strong anneal
            cur = rng.randn(n, 4)
            cur /= np.linalg.norm(cur, axis=1, keepdims=True)
            cur_b, cur_v = _smooth_descent(cur, steps=5500, T0=0.018, T1=0.0007, lr0=0.045, mu=0.85, rng=rng, noise0=0.004, t_deadline=SEARCH_END)
        elif mode == 1:
            sigma = [0.02, 0.06, 0.15, 0.30][(trial//6) % 4]
            cur = best_x + rng.randn(n, 4)*sigma
            cur /= np.linalg.norm(cur, axis=1, keepdims=True)
            cur_b, cur_v = _smooth_descent(cur, steps=4500, T0=0.010, T1=0.0006, lr0=0.035, mu=0.88, rng=rng, noise0=0.002, t_deadline=SEARCH_END)
        elif mode == 2:
            # surgery: replace worst m points by holes
            mrep = 2 + (trial % 5)  # 2..6
            G = best_x @ best_x.T
            np.fill_diagonal(G, -2.0)
            m = float(G.max())
            deg = (G > m - 0.0025).sum(axis=1)
            worst = np.argsort(deg)[-mrep:]
            keep_mask = np.ones(n, dtype=bool); keep_mask[worst] = False
            fixed = best_x[keep_mask].copy()
            cur = best_x.copy()
            for w in worst:
                # hole w.r.t. current fixed set
                h, _ = _find_hole(fixed, rng, n_samples=8000)
                cur[w] = h
                fixed = np.vstack([fixed, h.reshape(1,4)])
            # small jitter others
            cur = cur + rng.randn(n,4)*0.005
            cur /= np.linalg.norm(cur, axis=1, keepdims=True)
            cur_b, cur_v = _smooth_descent(cur, steps=5000, T0=0.012, T1=0.0006, lr0=0.040, mu=0.86, rng=rng, noise0=0.001, t_deadline=SEARCH_END)
        elif mode == 3:
            # cap re-arrange: pick random pole, shake cap strongly
            u = rng.randn(4); u/=np.linalg.norm(u)
            dots = best_x @ u
            thresh = np.quantile(dots, 0.88)  # top ~12%
            idx = np.where(dots > thresh)[0]
            cur = best_x.copy()
            sig = 0.25
            cur[idx] += rng.randn(len(idx),4)*sig
            cur /= np.linalg.norm(cur, axis=1, keepdims=True)
            cur_b, cur_v = _smooth_descent(cur, steps=4000, T0=0.014, T1=0.0008, lr0=0.040, mu=0.84, rng=rng, noise0=0.003, t_deadline=SEARCH_END)
        elif mode == 4:
            # rattler/hub targeted: overload shake + hole for rattlers
            G = best_x @ best_x.T
            np.fill_diagonal(G, -2.0)
            m = float(G.max())
            deg = (G > m - 1e-9).sum(axis=1)
            # rattlers = deg==0, hubs = largest deg
            rattle = np.where(deg==0)[0]
            hub = np.argsort(deg)[-6:]
            cur = best_x.copy()
            # move rattlers to holes
            if len(rattle)>0:
                fixed = np.delete(cur, rattle, axis=0)
                for w in rattle:
                    h,_ = _find_hole(fixed, rng, n_samples=8000)
                    cur[w]=h
                    fixed=np.vstack([fixed,h.reshape(1,4)])
            cur[hub] += rng.randn(len(hub),4)*0.12
            cur /= np.linalg.norm(cur, axis=1, keepdims=True)
            cur_b, cur_v = _smooth_descent(cur, steps=4500, T0=0.009, T1=0.0005, lr0=0.032, mu=0.88, rng=rng, noise0=0.0015, t_deadline=SEARCH_END)
        else:
            # mirror / rotation jump: random orthogonal perturbation to escape basin
            cur = best_x + rng.randn(n,4)*0.10
            cur /= np.linalg.norm(cur, axis=1, keepdims=True)
            # random small global rotation
            A = rng.randn(4,4)*0.05
            A = A - A.T
            R = np.eye(4)+A
            # QR to orthogonalize
            Q,_ = np.linalg.qr(R)
            cur = cur @ Q.T
            cur /= np.linalg.norm(cur, axis=1, keepdims=True)
            cur_b, cur_v = _smooth_descent(cur, steps=4000, T0=0.016, T1=0.0007, lr0=0.042, mu=0.85, rng=rng, noise0=0.003, t_deadline=SEARCH_END)

        if cur_v < best_val - 1e-12:
            # confirm with short LP if promising and time allows
            if cur_v < best_val - 3e-5 and time.time() < DEADLINE-40:
                try:
                    pol, pv = _lp_polish(cur_b, seconds=2.5, t_deadline=DEADLINE-30)
                except Exception:
                    pol, pv = cur_b, cur_v
                if pv < best_val:
                    best_x, best_val = pol, pv
                elif cur_v < best_val:
                    best_x, best_val = cur_b, cur_v
            else:
                best_x, best_val = cur_b, cur_v
        # occasional opportunistic keep of near-best diverse? skip
        if best_val < 0.84:
            break

    # final polish with remaining budget
    remain = DEADLINE - time.time() - 2.0
    if remain > 3:
        try:
            px, pv = _lp_polish(best_x, seconds=min(remain, 28.0), t_deadline=DEADLINE-1.0)
            if pv <= best_val + 1e-12:
                # accept if not worse (allow tiny numerical)
                if pv < best_val:
                    best_x, best_val = px, pv
                else:
                    best_x = px
        except Exception:
            pass
    best_x = best_x / np.linalg.norm(best_x, axis=1, keepdims=True)
    return best_x
