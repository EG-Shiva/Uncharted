import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def _degrees(X, m, tol=1e-9):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return (G >= m - tol).sum(axis=1), G

def _pump(X, t, steps, b0, b1, lr0, lr1, rng=None, noise=0.0):
    n, d = X.shape
    m1 = np.zeros_like(X)
    v1 = np.zeros_like(X)
    b1m, b2m = 0.9, 0.999
    eps = 1e-8
    lb0, lb1 = np.log(b0), np.log(b1)
    ll0, ll1 = np.log(lr0), np.log(lr1)
    for s in range(steps):
        f = s / max(1, steps - 1)
        beta = float(np.exp(lb0 + (lb1 - lb0) * f))
        lr = float(np.exp(ll0 + (ll1 - ll0) * f))
        G = X @ X.T
        # upper active mask via threshold t-0.01
        U = G - t
        act = U > -0.01
        np.fill_diagonal(act, False)
        if not np.any(act):
            # feasible: tiny random tangent drift to keep moving
            if noise > 0 and rng is not None:
                N = rng.standard_normal(X.shape)
                N -= (N * X).sum(axis=1, keepdims=True) * X
                X = X + noise * N
                X /= np.linalg.norm(X, axis=1, keepdims=True)
            continue
        z = beta * U
        zc = np.clip(z, -50, 50)
        Sig = np.zeros_like(G)
        # sigmoid only where active
        Sig[act] = 1.0 / (1.0 + np.exp(-zc[act]))
        # softplus = logaddexp(0,z)/beta
        P = np.zeros_like(G)
        P[act] = (np.logaddexp(0, z[act])) / beta
        W = np.zeros_like(G)
        W[act] = 2.0 * P[act] * Sig[act]
        # symmetrize (G symmetric so already sym)
        Gr = W @ X
        dot = (Gr * X).sum(axis=1, keepdims=True)
        R = Gr - dot * X
        m1 = b1m * m1 + (1 - b1m) * R
        v1 = b2m * v1 + (1 - b2m) * (R * R)
        bc1 = 1 - b1m ** (s + 2)
        bc2 = 1 - b2m ** (s + 2)
        mh = m1 / bc1
        vh = v1 / bc2
        Upd = lr * mh / (np.sqrt(vh) + eps)
        # project to tangent
        Ud = (Upd * X).sum(axis=1, keepdims=True)
        Upd = Upd - Ud * X
        # clip large steps
        sn = np.linalg.norm(Upd, axis=1, keepdims=True) + 1e-12
        clip = 0.02
        over = sn > clip
        Upd = np.where(over, Upd * (clip / sn), Upd)
        X = X - Upd
        if noise > 0 and rng is not None:
            N = rng.standard_normal(X.shape)
            N -= (N * X).sum(axis=1, keepdims=True) * X
            X = X + noise * (1.0 - f) * N
        nr = np.linalg.norm(X, axis=1, keepdims=True)
        nr[nr == 0] = 1.0
        X = X / nr
    return X

def _lp_sec_step(X, radius, stress, eps_sec=2e-3, cap=500):
    n, d = X.shape
    G = X @ X.T
    iu0 = np.triu_indices(n, 1)
    c = G[iu0]
    m = float(c.max())
    window = max(20.0 * radius, 5e-5)
    order = np.argsort(-c)
    sel = []
    for idx in order:
        if c[idx] > m - window and len(sel) < cap:
            sel.append(int(idx))
        elif c[idx] <= m - window:
            break
        if len(sel) >= cap:
            break
    if len(sel) == 0:
        return X, stress, False
    sel = np.array(sel)
    I = iu0[0][sel]; J = iu0[1][sel]
    csel = c[sel]
    p = len(sel)
    nv = n * d + 1
    rows = np.repeat(np.arange(p), 2 * d + 1)
    cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
    vals = np.concatenate([X[J], X[I], -np.ones((p, 1))], 1).ravel()
    try:
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((X.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
    except Exception:
        return X, stress, False
    # secondary gradient from stress weights
    if stress is None:
        sec = np.zeros((n, d))
    else:
        # stress is (n,n) symmetric nonneg
        sec = eps_sec * 2.0 * (stress @ X)
        # project? keep as is; linprog handles via bounds
    obj = np.zeros(nv)
    obj[:n * d] = sec.ravel()
    obj[-1] = 1.0
    try:
        res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq, b_eq=np.zeros(n),
                      bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                      method="highs", options={"time_limit": 1.2})
    except Exception:
        return X, stress, False
    if res.status != 0:
        return X, stress, False
    y = X + res.x[:-1].reshape(n, d)
    y /= np.linalg.norm(y, axis=1, keepdims=True) + 1e-16
    # update stress from duals if available else from tightness
    new_stress = stress
    try:
        marg = None
        if hasattr(res, 'ineqlin') and res.ineqlin is not None:
            marg = getattr(res.ineqlin, 'marginals', None)
            if marg is None:
                marg = getattr(res.ineqlin, 'dual', None)
        if marg is not None:
            marg = np.asarray(marg).ravel()
            if marg.size == p:
                w = np.maximum(0, -marg)
                if w.sum() > 0:
                    S = np.zeros((n, n))
                    np.add.at(S, (I, J), w)
                    S = S + S.T
                    # drop unfavourable (zero-dual) contacts: keep only positive + small uniform
                    S = 0.9 * (S / S.sum() * (n * 10)) + 0.1 * 0.01
                    new_stress = S
                else:
                    new_stress = stress
            else:
                raise ValueError("size mismatch")
        else:
            raise ValueError("no duals")
    except Exception:
        # fallback: softmax tightness as stress
        tau = max(2 * radius, 1e-4)
        w = np.exp((csel - m) / tau) + 1e-3
        S = np.zeros((n, n))
        np.add.at(S, (I, J), w)
        S = S + S.T
        # penalise large-degree hubs slightly: divide by degree
        deg = (np.abs(S).sum(axis=1, keepdims=True) + 1.0)
        S = S / deg
        new_stress = S
    return y, new_stress, True

def _lp_polish_minmax(X, seconds, deadline, cap=500):
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-16)
    best = _max_cos(X)
    radius = 8e-4
    start = time.time()
    n, d = X.shape
    iu0 = np.triu_indices(n, 1)
    itn = 0
    while time.time() - start < seconds and time.time() < deadline:
        if radius < 1e-13:
            break
        itn += 1
        if itn > 30:
            break
        G = X @ X.T
        c = G[iu0]
        m = float(c.max())
        window = max(20.0 * radius, 3e-5)
        order = np.argsort(-c)
        sel = []
        for idx in order:
            if c[idx] > m - window and len(sel) < cap:
                sel.append(int(idx))
            elif c[idx] <= m - window:
                break
            if len(sel) >= cap:
                break
        if not sel:
            break
        sel = np.array(sel)
        I = iu0[0][sel]; J = iu0[1][sel]
        p = len(sel)
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([X[J], X[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((X.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        except Exception:
            break
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c[sel], A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                          method="highs", options={"time_limit": 1.2})
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = X + res.x[:-1].reshape(n, d)
        y /= np.linalg.norm(y, axis=1, keepdims=True) + 1e-16
        v = _max_cos(y)
        if v < best - 1e-12:
            X, best = y, v
            radius = min(radius * 1.5, 0.02)
        else:
            radius *= 0.5
    return X, best

def solve():
    t0 = time.time()
    DEAD = t0 + 165.0
    rng = np.random.default_rng(7)
    cands = []
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn).astype(float)
            if a.shape == (58, 22) and np.all(np.isfinite(a)):
                a = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-16)
                cands.append(a)
        except Exception:
            pass
    if not cands:
        a = rng.standard_normal((58, 22))
        a /= np.linalg.norm(a, axis=1, keepdims=True)
        cands.append(a)
    vals = [_max_cos(a) for a in cands]
    bi = int(np.argmin(vals))
    incumbent = cands[bi].copy()
    best = incumbent.copy()
    best_val = float(vals[bi])
    # degrees of incumbent for reference
    try:
        deg0, _ = _degrees(incumbent, best_val)
        ref_min = float(deg0.min()); ref_med = float(np.median(deg0))
    except Exception:
        ref_min, ref_med = 0.0, 20.0

    # target schedule slightly below incumbent
    t_cur = best_val - 0.00032
    # clamp to spec examples but adaptive
    # ensure t_cur around 0.0663 if best is record
    targets = [t_cur, t_cur - 0.0003, t_cur - 0.0008, t_cur - 0.0013]
    cur = incumbent.copy()
    stress = None

    for ti, t in enumerate(targets):
        if time.time() > DEAD - 25:
            break
        # rounds per target
        for rnd in range(4):
            if time.time() > DEAD - 22:
                break
            # anneal beta upward across rounds
            b0 = 250.0 + 150 * rnd
            b1 = 900.0 + 500 * rnd + 400 * ti
            # pump from cur (first round from incumbent with small kick)
            if rnd == 0 and ti == 0:
                start = incumbent.copy()
                N = rng.standard_normal(start.shape)
                N -= (N * start).sum(axis=1, keepdims=True) * start
                start = start + 0.004 * N
                start /= np.linalg.norm(start, axis=1, keepdims=True)
            else:
                start = cur.copy()
            steps = 900 if rnd < 2 else 1200
            # keep steps small to stay in budget
            Xp = _pump(start, t, steps, b0, b1, 0.004, 0.0008, rng=rng, noise=0.0002 if rnd>0 else 0.0)
            if time.time() > DEAD - 20:
                cur = Xp
                break
            vp = _max_cos(Xp)
            viol = int((((Xp @ Xp.T) - t) > 0).sum() // 2 - 29  # approx pairs above t (diag excluded approx)
            # contact topology check
            try:
                degp, _ = _degrees(Xp, vp)
                mindeg = float(degp.min())
            except Exception:
                mindeg = 99
            # LP secondary step(s)
            rad = 0.002 if rnd < 2 else 0.001
            Xl, stress, ok = _lp_sec_step(Xp, rad, stress)
            vl = _max_cos(Xl) if ok else vp
            # choose better of pump vs LP
            if vl < vp - 1e-12:
                Xp, vp = Xl, vl
                try:
                    degp, _ = _degrees(Xp, vp)
                    mindeg = float(degp.min())
                except Exception:
                    pass
            else:
                # still update stress fallback once
                if stress is None:
                    _, stress, _ = _lp_sec_step(Xp, rad, None)
                    # revert Xp unchanged (function returned y but we ignore)
                    pass
            # feasibility at t ?
            if vp <= t:
                # polish and lower
                Xpol, vpol = _lp_polish_minmax(Xp, seconds=4.0, deadline=DEAD - 8)
                if vpol < best_val - 1e-12:
                    best, best_val = Xpol.copy(), float(vpol)
                    incumbent = Xpol.copy()
                    try:
                        deg0, _ = _degrees(incumbent, best_val)
                        ref_min = float(deg0.min())
                    except Exception:
                        pass
                cur = Xpol.copy()
                # lower t to new best - margin and break to next outer t
                t_new = best_val - 0.00025
                # update remaining targets
                for k in range(ti + 1, len(targets)):
                    targets[k] = min(targets[k], t_new - 0.0003 * (k - ti - 1))
                break
            else:
                # not feasible: topology-aware restart logic
                # if we created a markedly looser point (de-jammed), keep it as new start
                # else if blew up, restart from incumbent
                if vp > best_val + 0.004:
                    cur = incumbent.copy()
                    # small kick
                    N = rng.standard_normal(cur.shape)
                    N -= (N * cur).sum(axis=1, keepdims=True) * cur
                    cur = cur + 0.003 * N
                    cur /= np.linalg.norm(cur, axis=1, keepdims=True)
                    stress = None
                elif mindeg <= max(1.0, ref_min - 5.0) or mindeg < 6:
                    # de-jammed topology found: keep exploring from here
                    cur = Xp.copy()
                else:
                    # alternate: keep best of cur/Xp near incumbent basin
                    if vp < _max_cos(cur) - 1e-12:
                        cur = Xp.copy()
                    else:
                        # small perturbation of incumbent to try different basin
                        N = rng.standard_normal(incumbent.shape)
                        N -= (N * incumbent).sum(axis=1, keepdims=True) * incumbent
                        cur = incumbent + (0.002 + 0.001 * rnd) * N
                        cur /= np.linalg.norm(cur, axis=1, keepdims=True)
                # if violations small, tighten: continue same t
                # loop to next rnd
                continue
    # final: polish best with remaining time
    if time.time() < DEAD - 6:
        secs = max(1.0, min(10.0, DEAD - time.time() - 2.0))
        try:
            Xpol, vpol = _lp_polish_minmax(best, seconds=secs, deadline=DEAD - 1.0)
            if vpol < best_val:
                best, best_val = Xpol, vpol
        except Exception:
            pass
    best /= np.linalg.norm(best, axis=1, keepdims=True) + 1e-16
    return best
