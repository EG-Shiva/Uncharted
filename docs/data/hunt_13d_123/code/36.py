import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _lp_polish_capped(x, seconds, p_cap=2500):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    best_x = x.copy()
    radius = 1e-3
    start = time.time()
    deadline = start + seconds
    iu0, iu1 = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        G = x @ x.T
        c_all = G[iu0, iu1]
        m = float(c_all.max())
        window = max(20 * radius, 1e-9)
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
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
            if radius < 1e-7:
                x = best_x.copy()
        if time.time() >= deadline:
            break
    return best_x

def _stress_weights(X, tol=1e-6, iters=400, lr=0.01, alpha=0.5):
    n, d = X.shape
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    m = float(G.max())
    iu0, iu1 = np.triu_indices(n, 1)
    c_all = G[iu0, iu1]
    sel = c_all > m - tol
    I = iu0[sel]; J = iu1[sel]
    c = c_all[sel]
    p = len(I)
    if p == 0:
        S = np.zeros(n); nr = np.zeros(n)
        return I, J, c, np.ones(0), S, nr
    VI = X[J] - c[:, None] * X[I]
    VJ = X[I] - c[:, None] * X[J]
    w = np.ones(p, dtype=float)
    R = np.zeros((n, d))
    for _ in range(iters):
        R.fill(0.0)
        np.add.at(R, I, w[:, None] * VI)
        np.add.at(R, J, w[:, None] * VJ)
        RI = R[I]; RJ = R[J]
        grad = 2.0 * ((RI * VI).sum(axis=1) + (RJ * VJ).sum(axis=1)) + 2.0 * alpha * (w - 1.0)
        w = w - lr * grad
        np.clip(w, 0.0, None, out=w)
    R.fill(0.0)
    np.add.at(R, I, w[:, None] * VI)
    np.add.at(R, J, w[:, None] * VJ)
    S = np.zeros(n)
    np.add.at(S, I, w)
    np.add.at(S, J, w)
    nr = np.linalg.norm(R, axis=1)
    return I, J, c, w, S, nr

def _make_patch(X, S, nr, contacts, variant):
    n = X.shape[0]
    # rattler = lowest stress (tie -> lowest contacts)
    order_s = np.argsort(S, kind="stable")
    r0 = int(order_s[0])
    dots = X[r0] @ X.T
    dots[r0] = -2.0
    nearest = np.argsort(-dots, kind="stable")
    if variant == 0:
        m, k_near, k_low = 12, 6, 5
    elif variant == 1:
        m, k_near, k_low = 15, 7, 7
    else:
        m, k_near, k_low = 18, 8, 9
    patch = [r0]
    for k in nearest[:k_near]:
        if int(k) not in patch:
            patch.append(int(k))
    # lowest-stress hubs
    cnt_low = 0
    for k in order_s:
        if int(k) not in patch:
            patch.append(int(k))
            cnt_low += 1
            if cnt_low >= k_low:
                break
    # variant 2: add highest-residual / overload point to diversify
    if variant == 2:
        o = int(np.argmax(S))
        if o not in patch:
            patch.append(o)
        o2 = int(np.argmax(nr))
        if o2 not in patch:
            patch.append(o2)
    patch = patch[:m]
    # fill if short (should not happen)
    if len(patch) < m:
        for k in order_s:
            if int(k) not in patch:
                patch.append(int(k))
                if len(patch) >= m:
                    break
    return np.array(patch[:m], dtype=np.int64), int(r0)

def _repack_patch(X, patch, target, rng, deadline):
    n, d = X.shape
    m = len(patch)
    mask = np.ones(n, dtype=bool)
    mask[patch] = False
    Xf = X[mask]
    # greedy hole init
    Y = np.zeros((m, d))
    for k in range(m):
        K = 2000
        C = rng.normal(size=(K, d))
        C /= np.linalg.norm(C, axis=1, keepdims=True)
        md = C @ Xf.T
        mx = md.max(axis=1)
        if k > 0:
            md2 = C @ Y[:k].T
            mx2 = md2.max(axis=1)
            mx = np.maximum(mx, mx2)
        # pick among best 5 randomly for diversity
        order = np.argpartition(mx, 5)[:5]
        pick = order[int(rng.integers(0, 5))]
        Y[k] = C[pick]
    Y /= np.linalg.norm(Y, axis=1, keepdims=True)
    # annealed penalized softmax descent
    steps = 4000
    lam = 12.0
    for s in range(steps):
        if (s % 40) == 0 and time.time() >= deadline:
            break
        frac = s / max(steps - 1, 1)
        T = 0.02 + (0.003 - 0.02) * frac
        lr = 0.6 + (0.08 - 0.6) * frac
        Gyy = Y @ Y.T
        np.fill_diagonal(Gyy, -10.0)
        Gyf = Y @ Xf.T
        mval = float(max(Gyy.max(), Gyf.max()))
        Eyy = np.exp((Gyy - mval) / T)
        np.fill_diagonal(Eyy, 0.0)
        Eyf = np.exp((Gyf - mval) / T)
        sumE = Eyy.sum() + Eyf.sum() + 1e-30
        Avg = (Eyy @ Y + Eyf @ Xf) / sumE
        dot = (Avg * Y).sum(axis=1, keepdims=True)
        Gt = Avg - dot * Y
        # quadratic penalty for exceeding target
        pen_yy = np.maximum(Gyy - target, 0.0)
        pen_yf = np.maximum(Gyf - target, 0.0)
        if pen_yy.max() > 0 or pen_yf.max() > 0:
            Pyy = pen_yy.copy()
            np.fill_diagonal(Pyy, 0.0)
            PAvg = (Pyy @ Y + pen_yf @ Xf) / m
            pdot = (PAvg * Y).sum(axis=1, keepdims=True)
            Gt = Gt + lam * (PAvg - pdot * Y)
        Y = Y - lr * Gt
        Y /= np.linalg.norm(Y, axis=1, keepdims=True)
    Y /= np.linalg.norm(Y, axis=1, keepdims=True)
    Xnew = X.copy()
    Xnew[patch] = Y
    Xnew /= np.linalg.norm(Xnew, axis=1, keepdims=True)
    return Xnew

def solve():
    t0 = time.time()
    TIME_LIMIT = 170.0
    n, d = 123, 13
    rng0 = np.random.default_rng(0)
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
        X0 = rng0.normal(size=(n, d))
        X0 /= np.linalg.norm(X0, axis=1, keepdims=True)
        best_val0 = _max_cos(X0)
    best = X0.copy()
    best_val = float(best_val0)
    # at most three stress-selected patches
    for variant in range(3):
        # global time check
        now = time.time()
        elapsed = now - t0
        remaining = TIME_LIMIT - elapsed
        if remaining < 22.0:
            break
        # stress analysis on current best
        try:
            I, J, c, w, S, nr = _stress_weights(best, tol=1e-6, iters=350, lr=0.01, alpha=0.5)
        except Exception:
            continue
        G = best @ best.T
        np.fill_diagonal(G, -2.0)
        contacts = (G > best_val - 1e-9).sum(axis=1)
        try:
            patch, r0 = _make_patch(best, S, nr, contacts, variant)
        except Exception:
            continue
        if time.time() - t0 >= TIME_LIMIT - 21.0:
            break
        # repack only patch, strictly capped 15 s (and respect global budget)
        repack_sec = min(15.0, TIME_LIMIT - (time.time() - t0) - 6.0)
        if repack_sec <= 1.0:
            break
        repack_deadline = time.time() + repack_sec
        target = best_val - 2e-6
        rng = np.random.default_rng(1234 + variant * 777)
        try:
            cand = _repack_patch(best, patch, target, rng, repack_deadline)
        except Exception:
            continue
        if time.time() - t0 >= TIME_LIMIT - 6.0:
            # no time for polish, check raw candidate
            try:
                vraw = _max_cos(cand)
                if vraw < best_val:
                    best_val = vraw
                    best = cand.copy()
            except Exception:
                pass
            break
        # LP-polish whole configuration for at most 5 s
        lp_sec = min(5.0, TIME_LIMIT - (time.time() - t0) - 1.0)
        if lp_sec <= 0.5:
            try:
                vraw = _max_cos(cand)
                if vraw < best_val:
                    best_val = vraw
                    best = cand.copy()
            except Exception:
                pass
            break
        try:
            polished = _lp_polish_capped(cand, lp_sec, p_cap=2500)
        except Exception:
            continue
        try:
            v = _max_cos(polished)
        except Exception:
            continue
        if v < best_val - 1e-14:
            best_val = v
            best = polished.copy()
        # else discard (accept only improvement), continue to next patch
        if time.time() - t0 >= TIME_LIMIT - 2.0:
            break
    best = best / np.linalg.norm(best, axis=1, keepdims=True)
    return best
