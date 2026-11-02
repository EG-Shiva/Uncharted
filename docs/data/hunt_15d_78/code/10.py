import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _personal_max(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return g.max(axis=1)

def _contacts_cnt(x, tol=1e-7):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    m = float(g.max())
    cnt = (g > m - tol).sum(axis=1).astype(int)
    return cnt, m, g

def _lp_polish(x, deadline, r_init=0.01, r_max=0.05, cap=2500):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = r_init
    iu0 = np.triu_indices(n, 1)
    nv = n * d + 1
    obj = np.zeros(nv); obj[-1] = 1.0
    while time.time() < deadline and radius > 1e-13:
        if time.time() + 0.35 > deadline:
            break
        G = x @ x.T
        c_all = G[iu0]
        m = c_all.max()
        thresh = max(20.0 * radius, 1e-9)
        sel = c_all > m - thresh
        if np.count_nonzero(sel) > cap:
            idx = np.where(sel)[0]
            order = np.argsort(-c_all[idx])[:cap]
            mask = np.zeros_like(sel, dtype=bool)
            mask[idx[order]] = True
            sel = mask
        I = iu0[0][sel]; J = iu0[1][sel]
        p = len(I)
        if p == 0:
            radius *= 0.5
            continue
        c_sel = c_all[sel]
        cols = np.concatenate([I[:, None]*d + np.arange(d), J[:, None]*d + np.arange(d), np.full((p,1), n*d)], axis=1).ravel()
        rows = np.repeat(np.arange(p), 2*d+1)
        vals = np.concatenate([x[J], x[I], -np.ones((p,1))], axis=1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n*d))), shape=(n, nv)).tocsr()
            bounds = [(-radius, radius)]*(n*d) + [(None, None)]
            res = linprog(obj, A_ub=A_ub, b_ub=-c_sel, A_eq=A_eq, b_eq=np.zeros(n), bounds=bounds, method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        yn = np.linalg.norm(y, axis=1, keepdims=True)
        if not np.all(np.isfinite(yn)) or np.any(yn < 1e-12):
            radius *= 0.5
            continue
        y = y / yn
        v = _max_cos(y)
        if v < best:
            x = y; best = v
            radius = min(radius*1.5, r_max)
        else:
            radius *= 0.5
    return x, best

def _smooth_anneal(x, steps, a0, a1, lr0, lr1, mu=0.85):
    n, d = x.shape
    xc = x.copy()
    v = np.zeros_like(xc)
    la0 = np.log(a0); la1 = np.log(a1)
    for s in range(steps):
        frac = s/(steps-1) if steps > 1 else 0.0
        alpha = np.exp(la0 + (la1-la0)*frac)
        lr = lr0 + (lr1-lr0)*frac
        G = xc @ xc.T
        np.fill_diagonal(G, -1e9)
        m = G.max()
        # all <=0 after shift, safe exp
        W = np.exp(alpha*(G - m))
        np.fill_diagonal(W, 0.0)
        grad = W @ xc
        rs = W.sum(axis=1, keepdims=True) + 1e-12
        grad = grad / rs
        dot = np.sum(grad*xc, axis=1, keepdims=True)
        gt = grad - dot*xc
        v = mu*v + gt
        vd = np.sum(v*xc, axis=1, keepdims=True)
        vt = v - vd*xc
        # clip extreme velocity
        mx = np.linalg.norm(vt, axis=1).max()
        if mx > 2.0:
            vt *= (2.0/mx)
            v = vt
        xc = xc - lr*vt
        xc /= np.linalg.norm(xc, axis=1, keepdims=True)
    return xc

def _greedy_replace(X, idxs, rng, n_cand=2000):
    X = X.copy()
    n, d = X.shape
    for idx in idxs:
        others = np.delete(X, idx, axis=0)
        cand = rng.standard_normal((n_cand, d))
        cand /= np.linalg.norm(cand, axis=1, keepdims=True)
        sc = (cand @ others.T).max(axis=1)
        X[idx] = cand[np.argmin(sc)]
    return X

def solve():
    t0 = time.time()
    HARD_END = t0 + 172.0
    rng = np.random.default_rng(0)
    n, d = 78, 15
    cand = []
    try:
        b = np.load("best.npy")
        if b.shape == (n, d):
            cand.append(b.astype(float))
    except Exception:
        pass
    try:
        r = np.load("record.npy")
        if r.shape == (n, d):
            cand.append(r.astype(float))
    except Exception:
        pass
    if not cand:
        x0 = rng.standard_normal((n, d)); x0 /= np.linalg.norm(x0, axis=1, keepdims=True)
        cand.append(x0)
    best_v = 1e9; best_x = None
    for c in cand:
        c = c/np.linalg.norm(c, axis=1, keepdims=True)
        v = _max_cos(c)
        if v < best_v:
            best_v = v; best_x = c.copy()
    x_best = best_x.copy(); v_best = best_v
    # initial squeeze
    if time.time() < HARD_END - 30:
        x_best, v_best = _lp_polish(x_best, min(time.time()+6.0, HARD_END-26), r_init=0.002, r_max=0.02)
    trial = 0
    while time.time() < HARD_END - 22:
        trial += 1
        rem = HARD_END - 22 - time.time()
        if rem < 3:
            break
        budget = min(7.0, max(3.0, rem/3.0))
        deadline = time.time() + budget
        mode = trial % 6
        x_try = x_best.copy()
        if mode == 1:
            x_try = x_try + 0.010*rng.standard_normal((n,d))
            x_try /= np.linalg.norm(x_try, axis=1, keepdims=True)
            x_try = _smooth_anneal(x_try, 700, 40, 200, 0.020, 0.003, 0.85)
            x_try, v_try = _lp_polish(x_try, deadline, r_init=0.012, r_max=0.05)
        elif mode == 2:
            x_try = x_try + 0.030*rng.standard_normal((n,d))
            x_try /= np.linalg.norm(x_try, axis=1, keepdims=True)
            x_try = _smooth_anneal(x_try, 1200, 20, 250, 0.030, 0.004, 0.88)
            x_try, v_try = _lp_polish(x_try, deadline, r_init=0.015, r_max=0.05)
        elif mode == 3:
            pm = _personal_max(x_try)
            order = np.argsort(-pm)
            k = 5 if (trial % 2 == 1) else 7
            worst = order[:k]
            noise = 0.004*rng.standard_normal((n,d))
            noise[worst] += 0.09*rng.standard_normal((k,d))
            x_try = x_try + noise
            x_try /= np.linalg.norm(x_try, axis=1, keepdims=True)
            x_try = _smooth_anneal(x_try, 1000, 25, 220, 0.025, 0.003, 0.86)
            x_try, v_try = _lp_polish(x_try, deadline, r_init=0.015, r_max=0.05)
        elif mode == 4:
            cnt, m, G = _contacts_cnt(x_try, tol=1e-7)
            order = np.argsort(cnt)
            k = 2 if (trial % 3 != 0) else 3
            repl = order[:k].tolist()
            # greedy best-of-many reinsertion
            x_try = _greedy_replace(x_try, repl, rng, n_cand=2000)
            mask = np.ones(n, dtype=bool); mask[repl] = False
            x_try[mask] += 0.004*rng.standard_normal((np.count_nonzero(mask), d))
            x_try /= np.linalg.norm(x_try, axis=1, keepdims=True)
            x_try, v_try = _lp_polish(x_try, deadline, r_init=0.010, r_max=0.04)
        elif mode == 5:
            # contact-break push on tightest pairs then anneal
            G = x_try @ x_try.T
            np.fill_diagonal(G, -2.0)
            iu = np.triu_indices(n, 1)
            c_all = G[iu]
            top = np.argsort(-c_all)[:24]
            I = iu[0][top]; J = iu[1][top]
            push = 0.025
            for a, b_ in zip(I, J):
                # push a away from b along tangent
                dvec = x_try[a] - (x_try[a] @ x_try[b_])*x_try[b_]
                # tangent at a w.r.t b direction
                t = x_try[a]*0  # placeholder
                # simple: move a opposite to b's projection
                pa = x_try[b_] - (x_try[b_] @ x_try[a])*x_try[a]
                pb = x_try[a] - (x_try[a] @ x_try[b_])*x_try[b_]
                na = np.linalg.norm(pa); nb = np.linalg.norm(pb)
                if na > 1e-12:
                    x_try[a] -= push*pa/na
                if nb > 1e-12:
                    x_try[b_] -= push*pb/nb
            x_try /= np.linalg.norm(x_try, axis=1, keepdims=True)
            x_try = _smooth_anneal(x_try, 900, 30, 220, 0.022, 0.003, 0.85)
            x_try, v_try = _lp_polish(x_try, deadline, r_init=0.012, r_max=0.05)
        else:  # mode 0: tail-focused shake + anneal
            cnt, _, _ = _contacts_cnt(x_try, tol=1e-7)
            order = np.argsort(cnt)
            tail = order[:16]
            noise = 0.004*rng.standard_normal((n,d))
            noise[tail] += 0.055*rng.standard_normal((len(tail), d))
            x_try = x_try + noise
            x_try /= np.linalg.norm(x_try, axis=1, keepdims=True)
            x_try = _smooth_anneal(x_try, 1100, 22, 240, 0.028, 0.003, 0.87)
            x_try, v_try = _lp_polish(x_try, deadline, r_init=0.014, r_max=0.05)
        if v_try < v_best - 1e-13:
            x_best = x_try; v_best = v_try
        # also opportunistic: if trial found equal but different, keep random walk
        elif v_try < v_best + 1e-12 and mode in (2, 0) and rng.random() < 0.05:
            # sidestep to different basin with same level to escape
            x_best = x_try; v_best = v_try
    if time.time() < HARD_END - 1:
        x_best, v_best = _lp_polish(x_best, HARD_END-0.6, r_init=0.003, r_max=0.02)
        if time.time() < HARD_END - 0.4:
            x_best, v_best = _lp_polish(x_best, HARD_END-0.2, r_init=0.0006, r_max=0.006)
    x_best = x_best/np.linalg.norm(x_best, axis=1, keepdims=True)
    if not np.all(np.isfinite(x_best)):
        x_best = best_x/np.linalg.norm(best_x, axis=1, keepdims=True)
    return x_best
