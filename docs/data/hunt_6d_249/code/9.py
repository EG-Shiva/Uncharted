import time
import numpy as np
from scipy.optimize import linprog, minimize
from scipy.sparse import coo_matrix


def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(np.max(g))


def _lbfgs_soft(Xinit, tau, maxiter, creg=0.2):
    n, d = Xinit.shape
    iu0 = np.triu_indices(n, 1)
    I0, J0 = iu0

    def fg(xf):
        X = xf.reshape(n, d)
        r = np.sqrt(np.sum(X * X, axis=1))
        r = np.maximum(r, 1e-12)
        Xn = X / r[:, None]
        G = Xn @ Xn.T
        g = G[I0, J0]
        gmax = np.max(g)
        e = np.exp((g - gmax) / tau)
        S = np.sum(e)
        Lsoft = gmax + tau * np.log(S)
        w = e / S
        W = np.zeros((n, n), dtype=float)
        W[I0, J0] = w
        W = W + W.T
        A = W @ Xn
        proj = np.sum(A * Xn, axis=1)
        Grad0 = (A - proj[:, None] * Xn) / r[:, None]
        diff = r - 1.0
        L = Lsoft + creg * np.sum(diff * diff)
        Grad = Grad0 + (2.0 * creg * diff)[:, None] * (X / r[:, None])
        return float(L), Grad.ravel()
    Xf = Xinit.ravel().copy()
    try:
        res = minimize(fg, Xf, jac=True, method='L-BFGS-B',
                       options={'maxiter': int(maxiter), 'maxfun': int(maxiter * 5),
                                'ftol': 1e-14, 'gtol': 1e-7})
        Xf = res.x
    except Exception:
        pass
    Xout = Xf.reshape(n, d)
    Xout /= np.linalg.norm(Xout, axis=1, keepdims=True)
    return Xout


def _exact_polish(Xinit, deadline, t0, budget_end, step0=2e-3):
    cur = Xinit.copy()
    cur /= np.linalg.norm(cur, axis=1, keepdims=True)
    curVal = _max_cos(cur)
    step = step0
    n = cur.shape[0]
    while time.time() < deadline and time.time() < budget_end:
        if step < 1e-13:
            break
        G = cur @ cur.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        window = max(10.0 * step, 5e-4)
        if step < 5e-5:
            window = max(2e-4, 10.0 * step)
        mask = G > (m - window)
        F = mask.astype(float) @ cur
        dot = np.sum(F * cur, axis=1)
        Ft = F - dot[:, None] * cur
        fn = np.linalg.norm(Ft) + 1e-12
        trial = cur - step * Ft
        rn = np.linalg.norm(trial, axis=1, keepdims=True)
        rn = np.maximum(rn, 1e-12)
        trial /= rn
        v = _max_cos(trial)
        if v < curVal:
            cur = trial
            curVal = v
            step = min(step * 1.25, 0.01)
        else:
            step *= 0.5
            if step < 1e-7:
                # small random tangent kick to escape flat
                pass
    return cur, curVal


def _hole_replace(x, rng, n_cand=6000, n_worst=8):
    n, d = x.shape
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    row_max = np.max(G, axis=1)
    glob = float(np.max(G))
    order = np.argsort(-row_max)
    improved = False
    for k in range(min(n_worst, n)):
        idx = int(order[k])
        mask = np.ones(n, dtype=bool)
        mask[idx] = False
        Xo = x[mask]
        G2 = Xo @ Xo.T
        np.fill_diagonal(G2, -2.0)
        glob_excl = float(np.max(G2)) if G2.size else -2.0
        if glob_excl >= glob - 1e-9:
            # removing this point alone does not help enough, still try
            pass
        C = rng.standard_normal(size=(n_cand, d))
        C /= np.linalg.norm(C, axis=1, keepdims=True)
        D = C @ Xo.T
        cmax = np.max(D, axis=1)
        bi = int(np.argmin(cmax))
        bv = float(cmax[bi])
        new_glob = glob_excl if glob_excl > bv else bv
        if new_glob < glob - 1e-7:
            x = x.copy()
            x[idx] = C[bi]
            glob = new_glob
            G = x @ x.T
            np.fill_diagonal(G, -2.0)
            row_max = np.max(G, axis=1)
            order = np.argsort(-row_max)
            improved = True
    return x, improved


def _lp_polish(x, seconds, t_end=None):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds:
        if t_end is not None and time.time() > t_end:
            break
        if radius < 1e-14:
            break
        Gfull = x @ x.T
        c = Gfull[iu]
        m = float(np.max(c))
        sel = c > m - max(20 * radius, 1e-9)
        I = iu[0][sel]
        J = iu[1][sel]
        p = len(I)
        if p == 0:
            radius *= 0.5
            continue
        if p > 6000:
            # keep tightest 6000 to bound LP size
            idx = np.argsort(-c[sel])[:6000]
            I = I[idx]; J = J[idx]
            p = len(I)
            csv = c[sel][idx]
        else:
            csv = c[sel]
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))),
                          shape=(n, nv)).tocsr()
        obj = np.zeros(nv)
        obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-csv, A_eq=A_eq,
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
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        v = _max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x


def solve():
    t0 = time.time()
    BUDGET_END = t0 + 172.0
    rng = np.random.default_rng(0)
    np.random.seed(0)
    n, d = 249, 6
    # load starts
    cands = []
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=float)
            if a.shape == (n, d):
                a = a / np.linalg.norm(a, axis=1, keepdims=True)
                cands.append(a)
        except Exception:
            pass
    if not cands:
        X0 = rng.standard_normal(size=(n, d))
        X0 /= np.linalg.norm(X0, axis=1, keepdims=True)
        cands.append(X0)
    # pick best
    vals = [_max_cos(a) for a in cands]
    bi = int(np.argmin(vals))
    Xcur = cands[bi].copy()
    bestVal = float(vals[bi])
    bestX = Xcur.copy()

    def note(X):
        nonlocal bestX, bestVal
        v = _max_cos(X)
        if v < bestVal:
            bestVal = v
            bestX = X.copy()
        return v

    # Stage 1: soft L-BFGS annealing
    sched = [(0.012, 80), (0.006, 100), (0.003, 120), (0.0015, 140), (0.0007, 160)]
    for tau, it in sched:
        if time.time() > BUDGET_END - 60:
            break
        Xcur = _lbfgs_soft(Xcur, tau, it)
        note(Xcur)
        # interleave exact squeeze
        Xcur, _ = _exact_polish(Xcur, time.time() + 2.0, t0, BUDGET_END - 55)
        note(Xcur)

    # hole improvement on current
    if time.time() < BUDGET_END - 55:
        Xh, _ = _hole_replace(Xcur, rng, n_cand=8000, n_worst=10)
        v = note(Xh)
        if v < _max_cos(Xcur):
            Xcur = Xh.copy()
        # quick re-polish after hole
        Xcur = _lbfgs_soft(Xcur, 0.001, 80)
        note(Xcur)
        Xcur, _ = _exact_polish(Xcur, time.time() + 2.0, t0, BUDGET_END - 50)
        note(Xcur)

    # Stage 2: basin-hopping shakes targeting worst points
    sigmas = [0.04, 0.025, 0.015, 0.008, 0.05, 0.03]
    si = 0
    while time.time() < BUDGET_END - 48:
        G = Xcur @ Xcur.T
        np.fill_diagonal(G, -2.0)
        row_max = np.max(G, axis=1)
        order = np.argsort(-row_max)
        sigma = sigmas[si % len(sigmas)]
        si += 1
        # number to perturb: 2-6 worst + global jitter
        k = int(rng.integers(2, 7))
        worst = order[:k]
        Xtry = Xcur.copy()
        # tangential perturbation for worst
        pert = rng.standard_normal(size=(k, d)) * sigma
        # project to tangent
        for ii, idx in enumerate(worst):
            v = pert[ii]
            v = v - np.dot(v, Xtry[idx]) * Xtry[idx]
            Xtry[idx] = Xtry[idx] + v
        # tiny global jitter
        j = rng.standard_normal(size=(n, d)) * (sigma * 0.15)
        j -= np.sum(j * Xtry, axis=1, keepdims=True) * Xtry
        Xtry = Xtry + j
        Xtry /= np.linalg.norm(Xtry, axis=1, keepdims=True)
        # quick polish: one soft + short exact
        # adapt effort: smaller sigma -> finer polish
        Xtry = _lbfgs_soft(Xtry, 0.0012, 60)
        Xtry, _ = _exact_polish(Xtry, time.time() + 1.5, t0, BUDGET_END - 48, step0=1.5e-3)
        vtry = _max_cos(Xtry)
        if vtry < bestVal - 2e-9:
            Xcur = Xtry.copy()
            note(Xcur)
        elif vtry < _max_cos(Xcur) - 1e-9 and rng.random() < 0.3:
            Xcur = Xtry.copy()
        # occasionally try hole from best
        if si % 4 == 0:
            Xh2, _ = _hole_replace(Xcur if rng.random() < 0.5 else bestX.copy(), rng,
                                   n_cand=5000, n_worst=6)
            vh = _max_cos(Xh2)
            if vh < bestVal:
                Xcur = _lbfgs_soft(Xh2, 0.0008, 60)
                note(Xcur)
                Xcur, _ = _exact_polish(Xcur, time.time() + 1.2, t0, BUDGET_END - 48, step0=1e-3)
                note(Xcur)
            elif vh < _max_cos(Xcur):
                Xcur = Xh2
        # if stuck at best, restart from best with bigger kick
        if si % 9 == 0:
            Xcur = bestX.copy()

    # Stage 3: final polish of best
    Xfin = bestX.copy()
    # short soft finish
    if time.time() < BUDGET_END - 42:
        Xfin = _lbfgs_soft(Xfin, 0.0005, 120)
        v = _max_cos(Xfin)
        if v < bestVal:
            bestVal = v
            bestX = Xfin.copy()
        else:
            Xfin = bestX.copy()
    # LP polish with remaining time
    rem = BUDGET_END - time.time() - 2.0
    if rem > 8:
        try:
            Xlp = _lp_polish(Xfin, rem, t_end=BUDGET_END - 1.0)
            vlp = _max_cos(Xlp)
            if vlp < bestVal:
                bestX = Xlp
                bestVal = vlp
        except Exception:
            pass
    bestX = bestX / np.linalg.norm(bestX, axis=1, keepdims=True)
    return bestX
