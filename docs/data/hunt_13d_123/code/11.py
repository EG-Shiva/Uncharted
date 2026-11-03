import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _smooth_softmax(y, steps, T0, T1, lr0, lr1):
    n, d = y.shape
    for s in range(steps):
        f = s / max(steps - 1, 1)
        T = T0 + (T1 - T0) * f
        lr = lr0 + (lr1 - lr0) * f
        G = y @ y.T
        np.fill_diagonal(G, -10.0)
        m = G.max()
        E = np.exp((G - m) / T)
        np.fill_diagonal(E, 0.0)
        rs = E.sum(axis=1, keepdims=True) + 1e-20
        Avg = (E @ y) / rs
        dot = (Avg * y).sum(axis=1, keepdims=True)
        Gt = Avg - dot * y
        y = y - lr * Gt
        y /= np.linalg.norm(y, axis=1, keepdims=True)
    return y

def _topk_descent(y, steps, K, lr0, lr1):
    n, d = y.shape
    iu0, iu1 = np.triu_indices(n, 1)
    for s in range(steps):
        f = s / max(steps - 1, 1)
        lr = lr0 + (lr1 - lr0) * f
        G = y @ y.T
        c = G[iu0, iu1]
        if K < c.size:
            thr = np.partition(c, -K)[-K]
            sel = c >= thr
        else:
            sel = np.ones_like(c, dtype=bool)
        I = iu0[sel]; J = iu1[sel]
        # unweighted gradient from selected pairs
        grad = np.zeros_like(y)
        # accumulate: grad[I] += y[J], grad[J] += y[I]
        np.add.at(grad, I, y[J])
        np.add.at(grad, J, y[I])
        cnt = np.zeros(n)
        np.add.at(cnt, I, 1); np.add.at(cnt, J, 1)
        cnt = np.maximum(cnt, 1)[:, None]
        grad /= cnt
        dot = (grad * y).sum(axis=1, keepdims=True)
        Gt = grad - dot * y
        # normalize step by max norm to stabilize
        nm = np.linalg.norm(Gt)
        if nm > 1e-12:
            y = y - lr * Gt
            y /= np.linalg.norm(y, axis=1, keepdims=True)
    return y

def _refine_hole(v0, Xf, steps=120, T=0.006, lr0=0.3, lr1=0.02):
    y = v0.copy()
    y /= np.linalg.norm(y)
    for s in range(steps):
        f = s / max(steps - 1, 1)
        lr = lr0 + (lr1 - lr0) * f
        dots = Xf @ y
        m = dots.max()
        w = np.exp((dots - m) / T)
        w /= (w.sum() + 1e-20)
        avg = w @ Xf
        g = avg - (avg @ y) * y
        y = y - lr * g
        nrm = np.linalg.norm(y)
        if nrm == 0:
            break
        y /= nrm
    return y

def _best_hole(Xf, rng, n_cand=25000):
    d = Xf.shape[1]
    Y = rng.normal(size=(n_cand, d))
    Y /= np.linalg.norm(Y, axis=1, keepdims=True)
    dots = Y @ Xf.T
    md = dots.max(axis=1)
    order = np.argpartition(md, 4)[:4]
    # pick randomly among 4 best for diversity
    pick = order[int(rng.integers(0, 4))]
    v0 = Y[pick]
    v0 = v0 + 0.003 * rng.normal(size=(d,))
    v0 /= np.linalg.norm(v0)
    v = _refine_hole(v0, Xf)
    return v

def _lp_polish(x, seconds, p_cap=1500, max_it=12):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    best_x = x.copy()
    radius = 8e-4
    start = time.time()
    iu0, iu1 = np.triu_indices(n, 1)
    it = 0
    while time.time() - start < seconds and radius > 1e-13 and it < max_it:
        it += 1
        if time.time() - start >= seconds:
            break
        G = x @ x.T
        c_all = G[iu0, iu1]
        m = c_all.max()
        window = max(20.0 * radius, 2e-9)
        sel = c_all > m - window
        cnt = int(np.count_nonzero(sel))
        if cnt == 0:
            break
        if cnt > p_cap:
            idx = np.where(sel)[0]
            part = np.argpartition(-c_all[idx], p_cap - 1)[:p_cap]
            keep = np.zeros_like(sel)
            keep[idx[part]] = True
            sel = keep
            cnt = p_cap
        I = iu0[sel]; J = iu1[sel]
        csel = c_all[sel]
        p = cnt
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], axis=1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], axis=1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv); obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        yn = np.linalg.norm(y, axis=1, keepdims=True)
        yn[yn == 0] = 1.0
        y = y / yn
        v = _max_cos(y)
        if v < best:
            best = v
            best_x = y.copy()
            x = y
            radius = min(radius * 1.4, 0.02)
        else:
            radius *= 0.5
    return best_x

def solve():
    import time as _time
    import numpy as _np
    t0 = _time.time()
    LIM = 176.0
    rng = _np.random.default_rng(0)
    _np.random.seed(0)
    n, d = 123, 13
    X0 = None
    best0 = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            a = _np.load(fn)
            a = _np.asarray(a, dtype=float)
            if a.shape == (n, d):
                a = a / _np.linalg.norm(a, axis=1, keepdims=True)
                v = _max_cos(a)
                if v < best0:
                    best0 = v
                    X0 = a.copy()
        except Exception:
            pass
    if X0 is None:
        X0 = rng.normal(size=(n, d))
        X0 /= _np.linalg.norm(X0, axis=1, keepdims=True)
        best0 = _max_cos(X0)
    best = X0.copy()
    best_val = float(best0)
    cur = X0.copy()
    cur_val = float(best0)
    # quick LP squeeze of start (short)
    if _time.time() - t0 < LIM - 40:
        try:
            cand = _lp_polish(best, 1.5)
            v = _max_cos(cand)
            if v < best_val:
                best_val = v; best = cand.copy()
                cur_val = v; cur = cand.copy()
        except Exception:
            pass
    it = 0
    iu0_all, iu1_all = _np.triu_indices(n, 1)
    while _time.time() - t0 < LIM - 13:
        it += 1
        remain = LIM - 13 - (_time.time() - t0)
        if remain < 2.5:
            break
        trial = cur.copy()
        Gc = cur @ cur.T
        _np.fill_diagonal(Gc, -2.0)
        wmax = Gc.max(axis=1)
        thr_c = cur_val - 0.008
        counts = (Gc > thr_c).sum(axis=1)
        worst_idx = _np.argsort(-wmax)
        ratt_idx = _np.argsort(counts)
        mode = it % 5
        if mode == 0:
            sigma = 0.012 if (it % 10 != 0) else 0.03
            trial = trial + sigma * rng.normal(size=trial.shape)
            trial /= _np.linalg.norm(trial, axis=1, keepdims=True)
        elif mode == 1:
            # shake overloaded hubs + worst
            targets = set()
            for k in range(min(3, n)):
                targets.add(int(worst_idx[k]))
            # overloaded = highest counts
            oh = _np.argsort(-counts)
            for k in range(min(3, n)):
                targets.add(int(oh[k]))
            for k in range(2):
                targets.add(int(rng.integers(0, n)))
            targets = list(targets)
            trial += 0.003 * rng.normal(size=trial.shape)
            for idx in targets:
                trial[idx] += 0.22 * rng.normal(size=(d,))
            trial /= _np.linalg.norm(trial, axis=1, keepdims=True)
        elif mode == 2:
            # rattler hole reinsertion (1-2 points)
            nrep = 2 if (it % 2 == 0) else 1
            repl = [int(ratt_idx[k]) for k in range(nrep)]
            # also sometimes replace a worst point instead
            if it % 6 == 0:
                repl[0] = int(worst_idx[0])
            for r in repl:
                mask = _np.ones(n, dtype=bool); mask[r] = False
                Xr = trial[mask]
                v = _best_hole(Xr, rng, n_cand=22000)
                trial[r] = v
            trial += 0.0015 * rng.normal(size=trial.shape)
            trial /= _np.linalg.norm(trial, axis=1, keepdims=True)
        elif mode == 3:
            # split worst pair
            G2 = trial @ trial.T
            _np.fill_diagonal(G2, -2.0)
            ij = _np.argmax(G2)
            ii, jj = int(ij // n), int(ij % n)
            # move jj away from ii along tangent + noise
            vij = trial[jj] - (trial[jj] @ trial[ii]) * trial[ii]
            nm = _np.linalg.norm(vij)
            if nm < 1e-9:
                vij = rng.normal(size=(d,))
                nm = _np.linalg.norm(vij)
            vij /= nm
            trial[jj] = trial[jj] + 0.25 * vij + 0.05 * rng.normal(size=(d,))
            trial[jj] /= _np.linalg.norm(trial[jj])
            # jitter worst few
            for k in range(1, 4):
                idx = int(worst_idx[k])
                trial[idx] += 0.06 * rng.normal(size=(d,))
            trial /= _np.linalg.norm(trial, axis=1, keepdims=True)
        else:
            # random subset medium shake
            k = 8
            sel = rng.choice(n, size=k, replace=False)
            trial[sel] += 0.12 * rng.normal(size=(k, d))
            trial += 0.002 * rng.normal(size=trial.shape)
            trial /= _np.linalg.norm(trial, axis=1, keepdims=True)
        # smooth descent: alternate schedules
        try:
            if it % 3 == 0:
                trial = _smooth_softmax(trial, steps=220, T0=0.022, T1=0.004, lr0=0.4, lr1=0.05)
                trial = _topk_descent(trial, steps=120, K=250, lr0=0.08, lr1=0.01)
            elif it % 3 == 1:
                trial = _smooth_softmax(trial, steps=260, T0=0.014, T1=0.003, lr0=0.32, lr1=0.03)
                trial = _topk_descent(trial, steps=100, K=150, lr0=0.06, lr1=0.008)
            else:
                trial = _smooth_softmax(trial, steps=200, T0=0.03, T1=0.006, lr0=0.5, lr1=0.06)
                trial = _topk_descent(trial, steps=140, K=350, lr0=0.10, lr1=0.012)
        except Exception:
            pass
        # occasional short LP snap: only if promising or periodic
        v_smooth = _max_cos(trial)
        do_lp = False
        if v_smooth < best_val + 0.0015:
            do_lp = True
        elif it % 7 == 0 and remain > 25:
            do_lp = True
        if do_lp:
            lp_sec = 1.1 if remain > 25 else 0.7
            # cap to avoid overrun
            lp_sec = min(lp_sec, max(0.4, remain - 1.5))
            try:
                trial = _lp_polish(trial, lp_sec)
            except Exception:
                pass
        v = _max_cos(trial)
        if v < best_val - 1e-14:
            best_val = v; best = trial.copy()
            cur = trial.copy(); cur_val = v
        else:
            tol = 0.0035 * _np.exp(-it / 28.0) + 0.00015
            if v < cur_val + tol:
                cur = trial.copy(); cur_val = v
            if it % 25 == 0:
                cur = best + 0.008 * rng.normal(size=best.shape)
                cur /= _np.linalg.norm(cur, axis=1, keepdims=True)
                cur_val = _max_cos(cur)
        if _time.time() - t0 >= LIM - 13:
            break
    # final polish
    fsec = LIM - (_time.time() - t0) - 1.0
    fsec = min(max(fsec, 1.0), 12.0)
    try:
        pol = _lp_polish(best, fsec)
        pv = _max_cos(pol)
        if pv < best_val:
            best = pol; best_val = pv
    except Exception:
        pass
    best = best / _np.linalg.norm(best, axis=1, keepdims=True)
    return best
