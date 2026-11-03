import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _contacts(x, thr=1e-9):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    m = G.max()
    C = G > m - thr
    cnt = C.sum(axis=1)
    return m, cnt, G

def _smooth_anneal(y, steps, T0, T1, lr0, lr1):
    n, d = y.shape
    for s in range(steps):
        frac = s / max(steps - 1, 1)
        T = T0 + (T1 - T0) * frac
        lr = lr0 + (lr1 - lr0) * frac
        G = y @ y.T
        np.fill_diagonal(G, -10.0)
        m = G.max()
        # softmax weights, overflow safe
        E = np.exp((G - m) / T)
        np.fill_diagonal(E, 0.0)
        row = E.sum(axis=1, keepdims=True) + 1e-20
        Avg = (E @ y) / row
        dot = (Avg * y).sum(axis=1, keepdims=True)
        Gt = Avg - dot * y
        # clip step for stability
        y = y - lr * Gt
        y /= np.linalg.norm(y, axis=1, keepdims=True)
    return y

def _find_hole(x, rng, ntry=3000):
    n, d = x.shape
    Q = rng.normal(size=(ntry, d))
    Q /= np.linalg.norm(Q, axis=1, keepdims=True)
    S = Q @ x.T
    m = S.max(axis=1)
    i = int(np.argmin(m))
    q = Q[i].copy()
    # local refine: few projected steps away from neighbours
    for _ in range(60):
        s = x @ q
        # softmax push
        T = 0.01
        w = np.exp((s - s.max()) / T)
        g = (w[:, None] * x).sum(axis=0)
        g = g - (g @ q) * q
        nrm = np.linalg.norm(g)
        if nrm < 1e-12:
            break
        q = q - 0.05 * g / (nrm + 1e-12)
        q /= np.linalg.norm(q)
    return q

def _lp_polish(x, deadline, p_cap=2500):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    best_x = x.copy()
    radius = 1.5e-3
    iu0, iu1 = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        G = x @ x.T
        c_all = G[iu0, iu1]
        m = float(c_all.max())
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
        I = iu0[sel]; J = iu1[sel]; csel = c_all[sel]
        p = cnt
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([(I[:, None] * d + np.arange(d)).reshape(p, d),
                               (J[:, None] * d + np.arange(d)).reshape(p, d),
                               np.full((p, 1), n * d)], axis=1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], axis=1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            erows = np.repeat(np.arange(n), d)
            ecols = np.arange(n * d)
            A_eq = coo_matrix((x.ravel(), (erows, ecols)), shape=(n, nv)).tocsr()
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
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        v = _max_cos(y)
        if v < best:
            best = v
            best_x = y.copy()
            x = y
            radius = min(radius * 1.5, 0.02)
        else:
            radius *= 0.5
    return best_x

def solve():
    t0 = time.time()
    LIMIT = 178.0
    rng = np.random.default_rng(0)
    n, d = 123, 13
    X0 = None
    best_val0 = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=float)
            if a.shape == (n, d):
                a = a / np.linalg.norm(a, axis=1, keepdims=True)
                if np.all(np.isfinite(a)):
                    v = _max_cos(a)
                    if v < best_val0:
                        best_val0 = v
                        X0 = a.copy()
        except Exception:
            pass
    if X0 is None:
        X0 = rng.normal(size=(n, d))
        X0 /= np.linalg.norm(X0, axis=1, keepdims=True)
        best_val0 = _max_cos(X0)
    best = X0.copy()
    best_val = float(best_val0)
    cur = X0.copy()
    cur_val = float(best_val0)

    # main basin-hopping with cheap smooth repair
    it = 0
    # reserve 22s for final LP
    while time.time() - t0 < LIMIT - 24:
        it += 1
        remain = LIMIT - 24 - (time.time() - t0)
        if remain <= 1:
            break
        # choose source: mostly best, sometimes cur
        if rng.random() < 0.75:
            base = best
            base_val = best_val
        else:
            base = cur
            base_val = cur_val
        cand = base.copy()
        mode = it % 8
        if mode == 7:
            # hole reinsertion: move worst/rattler to hole
            try:
                m, cnt, G = _contacts(cand)
                ratt = int(np.argmin(cnt))
                # also pick one hub to slightly shake
                hub = int(np.argmax(cnt))
                q = _find_hole(cand, rng, ntry=2500)
                cand[ratt] = q
                # small shake of hub neighbourhood: shake hub a bit
                cand[hub] += 0.05 * rng.normal(size=d)
                cand[hub] /= np.linalg.norm(cand[hub])
            except Exception:
                k = 5
                idx = rng.choice(n, size=k, replace=False)
                cand[idx] += 0.08 * rng.normal(size=(k, d))
                cand[idx] /= np.linalg.norm(cand[idx], axis=1, keepdims=True)
        elif mode == 5 or mode == 6:
            # targeted hub shake
            try:
                m, cnt, G = _contacts(cand)
                order = np.argsort(-cnt)
                k = int(rng.choice([4, 6, 10]))
                idx = order[:k]
                sig = float(rng.choice([0.06, 0.12, 0.22]))
                cand[idx] += sig * rng.normal(size=(k, d))
                cand[idx] /= np.linalg.norm(cand[idx], axis=1, keepdims=True)
            except Exception:
                k = 8
                idx = rng.choice(n, size=k, replace=False)
                cand[idx] += 0.1 * rng.normal(size=(k, d))
                cand[idx] /= np.linalg.norm(cand[idx], axis=1, keepdims=True)
        else:
            k = int(rng.choice([3, 5, 8, 12, 20]))
            sig = float(rng.choice([0.02, 0.05, 0.10, 0.18, 0.30]))
            idx = rng.choice(n, size=k, replace=False)
            cand[idx] += sig * rng.normal(size=(k, d))
            cand[idx] /= np.linalg.norm(cand[idx], axis=1, keepdims=True)

        # cheap annealed smooth repair; adapt steps to remaining time
        if remain > 60:
            steps = 900
        elif remain > 30:
            steps = 650
        else:
            steps = 400
        # vary temperature schedule deterministically
        T0 = float(rng.choice([0.015, 0.025, 0.04]))
        T1 = 0.002
        lr0 = 0.06
        lr1 = 0.01
        try:
            cand = _smooth_anneal(cand, steps, T0, T1, lr0, lr1)
        except Exception:
            continue
        v = _max_cos(cand)
        if v < best_val:
            best_val = v
            best = cand.copy()
            cur = cand.copy()
            cur_val = v
        elif v < cur_val or rng.random() < 0.06:
            cur = cand.copy()
            cur_val = v
        # early exit if time low
        if time.time() - t0 >= LIMIT - 24:
            break

    # final tightening with short LP
    deadline = t0 + LIMIT - 1.5
    # give LP at most 22s
    lp_time = min(22.0, max(4.0, deadline - time.time()))
    try:
        tight = _lp_polish(best, time.time() + lp_time)
        v = _max_cos(tight)
        if v < best_val and np.all(np.isfinite(tight)):
            best = tight
            best_val = v
    except Exception:
        pass
    best = best / np.linalg.norm(best, axis=1, keepdims=True)
    return best
