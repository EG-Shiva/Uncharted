import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _contacts(X, thresh=1e-9):
    n = X.shape[0]
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    m = G.max()
    cnt = np.zeros(n, dtype=int)
    # count per point
    mask = G > m - thresh
    cnt = mask.sum(axis=1)
    return m, cnt, G

def _norm(X):
    nrm = np.linalg.norm(X, axis=1, keepdims=True)
    nrm[nrm == 0] = 1.0
    return X / nrm

def _lp_polish(X, deadline, radius_init=3e-3, p_cap=2500, max_iters=100):
    X = _norm(X)
    n, d = X.shape
    best = _max_cos(X)
    best_x = X.copy()
    x = X.copy()
    radius = radius_init
    iu0, iu1 = np.triu_indices(n, 1)
    it = 0
    while time.time() < deadline and radius > 1e-14 and it < max_iters:
        it += 1
        G = x @ x.T
        c_all = G[iu0, iu1]
        m = c_all.max()
        window = max(20 * radius, 1e-9)
        sel = c_all > m - window
        cnt = int(np.count_nonzero(sel))
        if cnt == 0:
            break
        if cnt > p_cap:
            idx = np.where(sel)[0]
            # keep tightest
            part = np.argpartition(-c_all[idx], p_cap - 1)[:p_cap]
            keep = np.zeros_like(sel)
            keep[idx[part]] = True
            sel = keep
            cnt = p_cap
        I = iu0[sel]; J = iu1[sel]; csel = c_all[sel]
        p = cnt
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([(I[:, None] * d + np.arange(d)).ravel(),
                               (J[:, None] * d + np.arange(d)).ravel(),
                               np.full(p, n * d)]).reshape(p, 2 * d + 1).ravel() if False else None
        # build cols directly
        Ic = (I[:, None] * d + np.arange(d))
        Jc = (J[:, None] * d + np.arange(d))
        cols = np.concatenate([Ic, Jc, np.full((p, 1), n * d)], axis=1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], axis=1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            erows = np.repeat(np.arange(n), d)
            ecols = np.arange(n * d)
            A_eq = coo_matrix((x.ravel(), (erows, ecols)), shape=(n, nv)).tocsr()
            obj = np.zeros(nv); obj[-1] = 1.0
            remain = deadline - time.time()
            if remain < 0.2:
                break
            tlim = min(2.0, max(0.3, remain * 0.5))
            res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                          method="highs", options={"time_limit": tlim, "presolve": True})
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y = _norm(y)
        v = _max_cos(y)
        if v < best:
            best = v
            best_x = y.copy()
            x = y
            radius = min(radius * 1.5, 0.02)
        else:
            radius *= 0.5
    return best_x, best

def _smooth(X, steps, T0, T1, lr0, lr1, deadline):
    y = X.copy()
    n, d = y.shape
    for s in range(steps):
        if time.time() >= deadline:
            break
        frac = s / max(steps - 1, 1)
        T = T0 + (T1 - T0) * frac
        lr = lr0 + (lr1 - lr0) * frac
        G = y @ y.T
        np.fill_diagonal(G, -10.0)
        m = G.max()
        # softmax weights, avoid overflow
        E = np.exp((G - m) / T)
        np.fill_diagonal(E, 0.0)
        row = E.sum(axis=1, keepdims=True) + 1e-20
        Avg = (E @ y) / row
        dot = (Avg * y).sum(axis=1, keepdims=True)
        Gt = Avg - dot * y
        # normalize grad scale
        gn = np.linalg.norm(Gt)
        if gn > 1e-12:
            # keep lr as is
            pass
        y = y - lr * Gt
        y = _norm(y)
    return y

def _best_hole(X_keep, rng, n_cand=12000, batch=3000):
    # find unit vector minimizing max dot with X_keep
    n, d = X_keep.shape
    best_v = None
    best_s = 1e9
    for _ in range(0, n_cand, batch):
        b = min(batch, n_cand - _)
        C = rng.normal(size=(b, d))
        C = _norm(C)
        S = (C @ X_keep.T).max(axis=1)
        i = int(np.argmin(S))
        if float(S[i]) < best_s:
            best_s = float(S[i])
            best_v = C[i].copy()
    return best_v

def solve():
    t0 = time.time()
    HARD = t0 + 172.0
    rng = np.random.default_rng(0)
    n, d = 123, 13
    X0 = None
    best_val = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=float)
            if a.shape == (n, d):
                a = _norm(a)
                v = _max_cos(a)
                if v < best_val:
                    best_val = v
                    X0 = a.copy()
        except Exception:
            pass
    if X0 is None:
        X0 = _norm(rng.normal(size=(n, d)))
        best_val = _max_cos(X0)
    best = X0.copy()
    # initial quick tighten
    try:
        if time.time() < HARD - 5:
            cand, v = _lp_polish(best, min(HARD - 2, time.time() + 5.0), radius_init=2e-3, max_iters=40)
            if v < best_val:
                best_val = v
                best = cand.copy()
    except Exception:
        pass
    cur = best.copy()
    cur_val = best_val
    it = 0
    # reserve final polish
    while time.time() < HARD - 22:
        it += 1
        if time.time() >= HARD - 22:
            break
        mode = it % 4
        trial = best.copy()  # kick from best (greedy)
        try:
            m, cnt, G = _contacts(trial, thresh=1e-9)
        except Exception:
            m, cnt = cur_val, np.zeros(n, dtype=int)
        if mode == 0:
            # rattler / random hole reinsertion
            # pick point with smallest contacts
            order = np.argsort(cnt)
            ridx = int(order[0]) if rng.random() < 0.8 else int(rng.integers(0, n))
            keep = np.delete(trial, ridx, axis=0)
            hv = _best_hole(keep, rng, n_cand=8000)
            # small jitter around hole
            hv = hv + 0.01 * rng.normal(size=d)
            hv = hv / np.linalg.norm(hv)
            trial[ridx] = hv
        elif mode == 1:
            # hub removal: pick among top-8 hubs
            order = np.argsort(-cnt)
            k = min(8, n)
            ridx = int(order[rng.integers(0, k)])
            keep = np.delete(trial, ridx, axis=0)
            hv = _best_hole(keep, rng, n_cand=12000)
            trial[ridx] = hv
        elif mode == 2:
            # cluster shake around worst point + nearest neighbours
            worst = int(np.argmax(cnt))
            # neighbours by dot
            sims = trial @ trial[worst]
            nn = np.argsort(-sims)[:12]
            sig = 0.02 + 0.02 * rng.random()
            for j in nn:
                noise = rng.normal(size=d)
                noise = noise - (noise @ trial[j]) * trial[j]
                trial[j] = trial[j] + sig * noise
            trial = _norm(trial)
        else:
            # global small shake
            sig = 0.004 + 0.006 * rng.random()
            noise = rng.normal(size=(n, d))
            proj = (noise * trial).sum(axis=1, keepdims=True)
            noise = noise - proj * trial
            trial = _norm(trial + sig * noise)
        # smooth anneal (fast)
        inner_dead = min(HARD - 20, time.time() + 1.2)
        T0 = 0.01 if mode != 3 else 0.005
        T1 = 0.0008
        try:
            trial = _smooth(trial, steps=250, T0=T0, T1=T1, lr0=0.15, lr1=0.02, deadline=inner_dead)
        except Exception:
            pass
        # short LP
        lp_dead = min(HARD - 20, time.time() + 2.5)
        if time.time() < lp_dead - 0.3:
            try:
                trial2, v2 = _lp_polish(trial, lp_dead, radius_init=2e-3, max_iters=25)
            except Exception:
                v2 = _max_cos(trial)
                trial2 = trial
        else:
            v2 = _max_cos(trial)
            trial2 = trial
        if v2 < best_val - 1e-12:
            best_val = float(v2)
            best = trial2.copy()
            cur_val = float(v2)
            cur = trial2.copy()
        elif v2 < cur_val and rng.random() < 0.05:
            cur_val = float(v2)
            cur = trial2.copy()
        # opportunistic: if stuck long, perturb cur from best with larger kick (already done)
    # final polish on best
    try:
        remain = HARD - time.time()
        if remain > 3:
            cand, v = _lp_polish(best, time.time() + min(remain - 1.0, 18.0), radius_init=1.5e-3, max_iters=80)
            if v < best_val:
                best_val = v
                best = cand.copy()
    except Exception:
        pass
    best = _norm(best)
    # tiny shrink guard: ensure no NaN/zero
    return best
