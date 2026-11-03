import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())


def lp_polish_deadline(x, deadline, r_init=0.005, max_pairs=1800):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    try:
        best = max_cos(x)
    except Exception:
        return x
    iu0 = np.triu_indices(n, 1)
    radius = r_init
    it = 0
    while time.time() < deadline and radius > 1e-13 and it < 60:
        it += 1
        try:
            G = x @ x.T
            c_all = G[iu0]
            m = float(c_all.max())
        except Exception:
            break
        thresh = m - max(20.0 * radius, 1e-9)
        sel = c_all > thresh
        cnt = int(sel.sum())
        if cnt == 0:
            radius *= 0.5
            continue
        if cnt > max_pairs:
            part = np.argpartition(c_all, -max_pairs)[-max_pairs:]
            sel2 = np.zeros_like(sel)
            sel2[part] = True
            sel = sel2
        I = iu0[0][sel]
        J = iu0[1][sel]
        c_sel = c_all[sel]
        p = len(I)
        nv = n * d + 1
        cols = np.empty((p, 2 * d + 1), dtype=int)
        ar = np.arange(d)
        cols[:, :d] = I[:, None] * d + ar
        cols[:, d:2 * d] = J[:, None] * d + ar
        cols[:, 2 * d] = n * d
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = cols.ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], axis=1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        except Exception:
            radius *= 0.5
            continue
        obj = np.zeros(nv)
        obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c_sel, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        try:
            y = x + res.x[:-1].reshape(n, d)
            yn = np.linalg.norm(y, axis=1, keepdims=True)
            if not np.all(np.isfinite(yn)) or np.any(yn < 1e-12):
                radius *= 0.5
                continue
            y = y / yn
            v = max_cos(y)
        except Exception:
            radius *= 0.5
            continue
        if v < best:
            x, best = y, v
            radius = min(radius * 1.4, 0.02)
        else:
            radius *= 0.5
    return x


def fast_anneal(X, steps=300, beta0=30.0, beta1=180.0, lr0=0.03, lr1=0.003, mu=0.8):
    n, d = X.shape
    V = np.zeros_like(X)
    for s in range(steps):
        frac = s / max(1, steps - 1)
        beta = beta0 * (beta1 / beta0) ** frac
        lr = lr0 * (lr1 / lr0) ** frac
        G = X @ X.T
        np.fill_diagonal(G, -10.0)
        m = float(G.max())
        # sparsify far pairs
        cut = m - 9.0 / beta
        # exp weights
        W = np.exp((G - m) * beta)
        W[G < cut] = 0.0
        np.fill_diagonal(W, 0.0)
        Grad = W @ X
        proj = np.sum(Grad * X, axis=1, keepdims=True)
        Gt = Grad - proj * X
        gnorm = float(np.linalg.norm(Gt) / np.sqrt(n))
        if not np.isfinite(gnorm) or gnorm < 1e-16:
            continue
        Gt = Gt / gnorm
        V = mu * V + (1.0 - mu) * Gt
        vnorm = float(np.linalg.norm(V) / np.sqrt(n))
        if vnorm > 1e-16 and np.isfinite(vnorm):
            step = V / vnorm
        else:
            step = Gt
        X = X - lr * step
        X = X / np.linalg.norm(X, axis=1, keepdims=True)
    return X


def greedy_holes(X, idxs, rng, M=1200):
    n, d = X.shape
    for idx in idxs:
        R = rng.standard_normal((M, d))
        R /= np.linalg.norm(R, axis=1, keepdims=True)
        D = R @ X.T
        D[:, idx] = -2.0
        worst = D.max(axis=1)
        pick = int(np.argmin(worst))
        X[idx] = R[pick]
    return X


def solve():
    t0 = time.time()
    HARD_END = t0 + 174.0
    LOOP_END = t0 + 152.0
    rng = np.random.default_rng(20261103)
    n, d = 176, 9
    cands = []
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            if a.shape == (n, d) and np.all(np.isfinite(a)):
                nr = np.linalg.norm(a, axis=1, keepdims=True)
                if np.all(nr > 1e-12):
                    cands.append(a / nr)
        except Exception:
            pass
    if not cands:
        X0 = rng.standard_normal((n, d))
        X0 /= np.linalg.norm(X0, axis=1, keepdims=True)
        cands.append(X0)
    vals = [max_cos(c) for c in cands]
    bi = int(np.argmin(vals))
    best_X = cands[bi].copy()
    best_v = float(vals[bi])
    curr_X = best_X.copy()
    curr_v = best_v
    # quick initial squeeze
    try:
        if time.time() < LOOP_END:
            best_X = lp_polish_deadline(best_X, min(time.time() + 3.0, LOOP_END), r_init=0.0012)
            best_v = max_cos(best_X)
            curr_X = best_X.copy()
            curr_v = best_v
    except Exception:
        pass

    sigmas = [0.004, 0.009, 0.016, 0.03, 0.06, 0.11]
    T = 3.5e-4
    trial = 0
    while time.time() < LOOP_END:
        trial += 1
        if trial % 30 == 0:
            curr_X = best_X.copy()
            curr_v = best_v
        if time.time() > HARD_END - 14.0:
            break
        sigma = sigmas[(trial - 1) % len(sigmas)]
        mode = trial % 7
        # row maxima for targeting
        try:
            Gc = curr_X @ curr_X.T
            np.fill_diagonal(Gc, -2.0)
            rowmax = Gc.max(axis=1)
            order = np.argsort(rowmax)
            # contacts count for overload (within 2e-3 of global max)
            mglob = float(Gc.max())
            contacts = (Gc > mglob - 2e-3).sum(axis=1)
        except Exception:
            order = np.arange(n)
            rowmax = np.zeros(n)
            contacts = np.zeros(n, dtype=int)
            mglob = curr_v
        Xp = curr_X.copy()
        if mode == 0:
            Xp = Xp + sigma * rng.standard_normal((n, d))
            Xp /= np.linalg.norm(Xp, axis=1, keepdims=True)
        elif mode == 1:
            k = 14
            idx_w = order[-k:]
            noise = 0.0025 * rng.standard_normal((n, d))
            noise[idx_w] += (sigma * 1.6) * rng.standard_normal((k, d))
            Xp = Xp + noise
            Xp /= np.linalg.norm(Xp, axis=1, keepdims=True)
        elif mode == 2:
            idx = int(order[-1])
            try:
                Xp = greedy_holes(Xp, [idx], rng, M=1600)
            except Exception:
                pass
            Xp = Xp + 0.002 * rng.standard_normal((n, d))
            Xp /= np.linalg.norm(Xp, axis=1, keepdims=True)
        elif mode == 3:
            k = 3
            idxs = list(order[-k:])
            # also add most overloaded if different
            try:
                io = int(np.argmax(contacts))
                if io not in idxs:
                    idxs[0] = io
            except Exception:
                pass
            Xp = Xp + 0.003 * rng.standard_normal((n, d))
            Xp /= np.linalg.norm(Xp, axis=1, keepdims=True)
            try:
                Xp = greedy_holes(Xp, idxs, rng, M=900)
            except Exception:
                pass
        elif mode == 4:
            # neighbourhood patch around worst point
            try:
                w = int(order[-1])
                sim = (curr_X @ curr_X[w])
                sim[w] = -2.0
                nbr = np.argpartition(sim, -24)[-24:]
                patch = np.concatenate([[w], nbr])
                Xp[patch] += (sigma * 1.2) * rng.standard_normal((len(patch), d))
                Xp += 0.0015 * rng.standard_normal((n, d))
                Xp /= np.linalg.norm(Xp, axis=1, keepdims=True)
            except Exception:
                Xp = Xp + sigma * rng.standard_normal((n, d))
                Xp /= np.linalg.norm(Xp, axis=1, keepdims=True)
        elif mode == 5:
            k = 22
            idxs = rng.choice(n, size=k, replace=False)
            Xp[idxs] += sigma * rng.standard_normal((k, d))
            Xp /= np.linalg.norm(Xp, axis=1, keepdims=True)
        else:
            # overload shake: top-contact points
            try:
                k = 10
                idx_o = np.argpartition(contacts, -k)[-k:]
                Xp[idx_o] += (sigma * 1.3) * rng.standard_normal((k, d))
                Xp += 0.002 * rng.standard_normal((n, d))
                Xp /= np.linalg.norm(Xp, axis=1, keepdims=True)
            except Exception:
                Xp = Xp + sigma * 0.7 * rng.standard_normal((n, d))
                Xp /= np.linalg.norm(Xp, axis=1, keepdims=True)

        steps = 320 if sigma < 0.04 else 220
        try:
            # vary beta slightly per trial for diversity
            b0 = 25.0 + (trial % 3) * 10.0
            b1 = 150.0 + (trial % 4) * 30.0
            Xp = fast_anneal(Xp, steps=steps, beta0=b0, beta1=b1,
                             lr0=0.028 if sigma < 0.05 else 0.035, lr1=0.002)
        except Exception:
            pass
        try:
            v_fast = max_cos(Xp)
        except Exception:
            continue
        if not np.isfinite(v_fast):
            continue
        # selective LP polish
        do_lp = False
        if v_fast < best_v + 0.002:
            do_lp = True
        elif (trial % 8 == 0) and (v_fast < curr_v + 0.004):
            do_lp = True
        v = v_fast
        if do_lp:
            budget = 1.8 if sigma < 0.03 else 2.6
            deadline = min(time.time() + budget, LOOP_END, HARD_END - 12.0)
            if deadline > time.time() + 0.25:
                try:
                    r0 = 0.007 if sigma > 0.02 else 0.004
                    Xp = lp_polish_deadline(Xp, deadline, r_init=r0)
                    v = max_cos(Xp)
                except Exception:
                    pass
        if not np.isfinite(v):
            continue
        if v < best_v:
            best_v = v
            best_X = Xp.copy()
            curr_X = Xp.copy()
            curr_v = v
        else:
            if v < curr_v or rng.random() < np.exp((curr_v - v) / T):
                curr_X = Xp.copy()
                curr_v = v
    try:
        fd = HARD_END - 1.0
        if time.time() < fd:
            best_X = lp_polish_deadline(best_X, fd, r_init=0.0012)
    except Exception:
        pass
    best_X = best_X / np.linalg.norm(best_X, axis=1, keepdims=True)
    return best_X
