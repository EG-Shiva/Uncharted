import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _norm(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n = np.maximum(n, 1e-12)
    return x / n

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(np.max(g))

def _graph_hash(x, thresh):
    g = x @ x.T
    iu = np.triu_indices(x.shape[0], 1)
    b = (g[iu] > thresh)
    return hash(b.tobytes())

def _feas_adam(X0, tau, steps, a0, a1, lr0, lr1, w_lse=0.03, temp=0.001):
    n, d = X0.shape
    X = X0.copy()
    m = np.zeros_like(X)
    v = np.zeros_like(X)
    b1, b2, eps = 0.9, 0.999, 1e-8
    npairs = n * (n - 1) / 2.0
    best_X = X.copy()
    # initial true max
    G0 = X @ X.T
    np.fill_diagonal(G0, -2.0)
    best_m = float(np.max(G0))
    pen0 = 0.0
    for t in range(1, steps + 1):
        frac = t / steps
        alpha = a0 * (a1 / a0) ** frac
        lr = lr0 * (lr1 / lr0) ** frac
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        Mx = float(np.max(G))
        if Mx < best_m:
            best_m = Mx
            best_X = X.copy()
        # softplus penalty gradient weight
        Z = (G - tau) / temp
        # clip for stability
        Zc = np.clip(Z, -50.0, 50.0)
        # softplus = logaddexp(0,Z)
        SP = np.logaddexp(0.0, Zc)
        np.fill_diagonal(SP, 0.0)
        # sigmoid
        # use stable formula
        Sig = np.empty_like(Zc)
        pos = Zc >= 0
        Sig[pos] = 1.0 / (1.0 + np.exp(-Zc[pos]))
        e = np.exp(Zc[~pos])
        Sig[~pos] = e / (1.0 + e)
        np.fill_diagonal(Sig, 0.0)
        # penalty value (for feasibility check) - mean sp^2 over pairs
        # cheap estimate using symmetric sum /2
        # W_pen entries
        W = (2.0 * SP * Sig / temp) / npairs
        np.fill_diagonal(W, 0.0)
        grad_pen = W @ X
        # LogSumExp part
        E = np.exp(alpha * (G - Mx))
        np.fill_diagonal(E, 0.0)
        S = float(np.sum(E))
        if not np.isfinite(S) or S <= 0:
            grad = grad_pen
        else:
            grad_lse = (E @ X) / S
            grad = grad_pen + w_lse * grad_lse
        # tangent projection
        proj = np.sum(grad * X, axis=1, keepdims=True)
        gt = grad - proj * X
        if not np.all(np.isfinite(gt)):
            continue
        m = b1 * m + (1.0 - b1) * gt
        v = b2 * v + (1.0 - b2) * (gt * gt)
        mhat = m / (1.0 - b1 ** t)
        vhat = v / (1.0 - b2 ** t)
        step = lr * mhat / (np.sqrt(vhat) + eps)
        corr = np.sum(step * X, axis=1, keepdims=True)
        step = step - corr * X
        if not np.all(np.isfinite(step)):
            continue
        X = X - step
        X = _norm(X)
    # final eval
    Gf = X @ X.T
    np.fill_diagonal(Gf, -2.0)
    Mf = float(np.max(Gf))
    if Mf < best_m:
        best_m = Mf
        best_X = X.copy()
    # penalty near zero check
    Zf = np.clip((Gf - tau) / temp, -50, 50)
    SPf = np.logaddexp(0.0, Zf)
    np.fill_diagonal(SPf, 0.0)
    iu = np.triu_indices(n, 1)
    pen = float(np.mean(SPf[iu] ** 2))
    feasible = (Mf <= tau + 1e-6) or (pen < 1e-10)
    return X, best_X, best_m, Mf, pen, feasible

def _lp_polish(x, deadline):
    x = _norm(x)
    n, d = x.shape
    best = _max_cos(x)
    xb = x.copy()
    radius = 1e-3
    iu = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        cfull = (x @ x.T)[iu]
        m = float(cfull.max())
        sel = cfull > m - max(20 * radius, 1e-9)
        cnt = int(np.count_nonzero(sel))
        if cnt == 0:
            break
        if cnt > 1100:
            idx = np.where(sel)[0]
            order = np.argsort(-cfull[idx])[:1100]
            nsel = np.zeros_like(sel)
            nsel[idx[order]] = True
            sel = nsel
        csel = cfull[sel]
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d),
                                           np.arange(n * d))), shape=(n, nv)).tocsr()
        except Exception:
            break
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq,
                          b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                          method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y = _norm(y)
        vv = _max_cos(y)
        if vv < best:
            best = vv
            xb = y.copy()
            x = y
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return xb

def solve():
    t0 = time.time()
    LIMIT = 172.0
    deadline = t0 + LIMIT
    feas_end = t0 + 140.0
    rng = np.random.default_rng(12345)
    n, d = 55, 21
    incumbent = None
    inc_score = 1e9
    for fn in ["best.npy", "record.npy"]:
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=np.float64)
            if a.shape == (55, 21):
                a = _norm(a)
                s = _max_cos(a)
                if s < inc_score:
                    inc_score = s
                    incumbent = a.copy()
        except Exception:
            pass
    if incumbent is None:
        incumbent = _norm(rng.standard_normal(size=(n, d)))
        inc_score = _max_cos(incumbent)
    taus = [0.0662, 0.0658, 0.0654, 0.0650]
    # per-tau steps: total per start <=800
    steps_per_tau = 200
    best_overall = incumbent.copy()
    best_score = inc_score
    pool = [(incumbent.copy(), inc_score)]
    seen = set()
    seen.add(_graph_hash(incumbent, 0.06))
    n_random_needed = 20
    n_random_done = 0
    start_idx = 0
    # annealing params per tau level (beta and lr annealing inside)
    while time.time() < feas_end and start_idx < 300:
        start_idx += 1
        # choose start point
        if start_idx == 1:
            X0 = incumbent + 0.01 * rng.standard_normal(size=(n, d))
            X0 = _norm(X0)
        elif start_idx % 2 == 0 and n_random_done < 60:
            X0 = _norm(rng.standard_normal(size=(n, d)))
            n_random_done += 1
        else:
            # perturbed copy of incumbent or pool best
            if rng.random() < 0.6:
                base = best_overall
                sigma = 0.02 + 0.08 * rng.random()
            else:
                # random pool member
                pi = rng.integers(0, len(pool))
                base = pool[pi][0]
                sigma = 0.03 + 0.12 * rng.random()
            X0 = base + sigma * rng.standard_normal(size=(n, d))
            X0 = _norm(X0)
        # diversity: if hash seen, add extra kick
        h0 = _graph_hash(X0, 0.06)
        if h0 in seen:
            X0 = _norm(X0 + 0.05 * rng.standard_normal(size=(n, d)))
        Xcur = X0
        # sequential taus
        for ti, tau in enumerate(taus):
            if time.time() >= feas_end:
                break
            # vary annealing slightly per start to diversify
            a0 = 60.0 + 20.0 * rng.random()
            a1 = 500.0 + 200.0 * rng.random()
            lr0 = 0.02 + 0.015 * rng.random()
            lr1 = 0.002 + 0.002 * rng.random()
            Xnew, Xbest_traj, best_m_traj, Mf, pen, feas = _feas_adam(
                Xcur, tau, steps_per_tau, a0, a1, lr0, lr1)
            # track global best by true max
            if best_m_traj < best_score:
                best_score = best_m_traj
                best_overall = Xbest_traj.copy()
            # pool update with contact-graph indexing
            h = _graph_hash(Xnew, tau)
            if h not in seen:
                seen.add(h)
                pool.append((Xnew.copy(), Mf))
                if len(pool) > 60:
                    # drop worst
                    pool.sort(key=lambda z: z[1])
                    pool = pool[:60]
            else:
                # duplicate graph: keep only if better true max
                pool.append((Xnew.copy(), Mf))
                pool.sort(key=lambda z: z[1])
                pool = pool[:60]
            if feas:
                # record and lower tau
                Xcur = Xnew
                continue
            else:
                # not feasible for this tau -> stop lowering, keep Xbest
                Xcur = Xnew
                break
        # ensure at least 20 random configs
        if n_random_done >= n_random_needed and start_idx > 25:
            # allow early exit only on time
            pass
        if n_random_done < n_random_needed and start_idx > 250:
            continue
    # final LP polish of feasible candidate with smallest true max
    # pick best_overall
    polish_deadline = min(deadline - 2.0, time.time() + 30.0)
    if time.time() < polish_deadline:
        try:
            xp = _lp_polish(best_overall, polish_deadline)
            sp = _max_cos(xp)
            if sp < best_score:
                best_overall = xp
                best_score = sp
        except Exception:
            pass
    best_overall = _norm(best_overall)
    return best_overall
