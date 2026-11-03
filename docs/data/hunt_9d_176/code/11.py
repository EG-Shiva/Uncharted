import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))


def anneal(X, steps, beta0, beta1, lr0, lr1):
    n, d = X.shape
    for s in range(steps):
        frac = s / max(1, steps - 1)
        beta = beta0 * (beta1 / beta0) ** frac
        lr = lr0 * (lr1 / lr0) ** frac
        G = X @ X.T
        np.fill_diagonal(G, -10.0)
        m = float(np.max(G))
        W = np.exp((G - m) * beta)
        np.fill_diagonal(W, 0.0)
        Grad = W @ X
        proj = np.sum(Grad * X, axis=1, keepdims=True)
        Gt = Grad - proj * X
        gnorm = float(np.linalg.norm(Gt) / np.sqrt(n))
        if not np.isfinite(gnorm) or gnorm < 1e-16:
            continue
        X = X - (lr / gnorm) * Gt
        X = X / np.linalg.norm(X, axis=1, keepdims=True)
    return X


def hub_scores(X, beta=60.0):
    G = X @ X.T
    np.fill_diagonal(G, -10.0)
    m = float(np.max(G))
    S = np.exp((G - m) * beta).sum(axis=1)
    return S, m, G


def refine_hole(Y, v, steps=120, beta=80.0, lr0=0.05, lr1=0.005):
    v = v / np.linalg.norm(v)
    for s in range(steps):
        frac = s / max(1, steps - 1)
        lr = lr0 * (lr1 / lr0) ** frac
        d = Y @ v
        m = float(np.max(d))
        w = np.exp((d - m) * beta)
        w = w / (np.sum(w) + 1e-30)
        g = w @ Y
        gt = g - float(np.dot(g, v)) * v
        gn = float(np.linalg.norm(gt))
        if not np.isfinite(gn) or gn < 1e-16:
            break
        v = v - (lr / (gn + 1e-30)) * gt * min(1.0, gn * 5.0 + 0.2)
        # simpler normalized step:
        # v = v - lr * gt / (gn+1e-30) * 0.5
        nrm = float(np.linalg.norm(v))
        if not np.isfinite(nrm) or nrm < 1e-12:
            break
        v = v / nrm
    return v / np.linalg.norm(v)


def best_hole(Y, rng, n_samples=3000):
    d = Y.shape[1]
    C = rng.standard_normal((n_samples, d))
    C /= np.linalg.norm(C, axis=1, keepdims=True)
    D = Y @ C.T
    vals = np.max(D, axis=0)
    bi = int(np.argmin(vals))
    return C[bi]


def lp_polish_deadline(x, deadline, r_init=0.004, max_pairs=2200):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    try:
        best = max_cos(x)
    except Exception:
        return x
    radius = r_init
    iu0 = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        G = x @ x.T
        c_all = G[iu0]
        m = float(np.max(c_all))
        thresh = m - max(20.0 * radius, 1e-9)
        sel = c_all > thresh
        if int(np.sum(sel)) > max_pairs:
            part = np.argpartition(c_all, -max_pairs)[-max_pairs:]
            sel2 = np.zeros(len(c_all), dtype=bool)
            sel2[part] = True
            sel = sel2
        if int(np.sum(sel)) == 0:
            radius *= 0.5
            continue
        I = iu0[0][sel]
        J = iu0[1][sel]
        c_sel = c_all[sel]
        p = len(I)
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.empty((p, 2 * d + 1), dtype=int)
        cols[:, :d] = I[:, None] * d + np.arange(d)
        cols[:, d:2 * d] = J[:, None] * d + np.arange(d)
        cols[:, 2 * d] = n * d
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
        y = x + res.x[:-1].reshape(n, d)
        yn = np.linalg.norm(y, axis=1, keepdims=True)
        if not np.all(np.isfinite(yn)) or np.any(yn < 1e-12):
            radius *= 0.5
            continue
        y = y / yn
        v = max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.03)
        else:
            radius *= 0.5
        if time.time() >= deadline:
            break
    return x


def solve():
    t0 = time.time()
    HARD_END = t0 + 172.0
    FINAL_LP = 26.0
    LOOP_END = HARD_END - FINAL_LP
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
    if len(cands) == 0:
        X0 = rng.standard_normal((n, d))
        X0 /= np.linalg.norm(X0, axis=1, keepdims=True)
        cands.append(X0)
    vals = [max_cos(c) for c in cands]
    bi = int(np.argmin(vals))
    best_X = cands[bi].copy()
    best_v = float(vals[bi])
    curr_X = best_X.copy()
    curr_v = best_v
    it = 0
    stall = 0
    while time.time() < LOOP_END:
        it += 1
        r = it % 10
        if r < 3:
            k, sigma, mode = 1, 0.30, "hole"
        elif r < 5:
            k, sigma, mode = 2, 0.22, "hole"
        elif r == 5:
            k, sigma, mode = 1, 0.20, "hub"
        elif r == 6:
            k, sigma, mode = 5, 0.12, "random"
        elif r == 7:
            k, sigma, mode = 12, 0.07, "random"
        elif r == 8:
            k, sigma, mode = 3, 0.18, "hub"
        else:
            k, sigma, mode = 1, 0.09, "random"
        # occasional restart from best if stalled
        if stall > 25:
            curr_X = best_X.copy()
            curr_v = best_v
            stall = 0
        base = curr_X
        # find hubs on base
        try:
            scores, m0, _ = hub_scores(base, beta=60.0)
            order = np.argsort(-scores)
        except Exception:
            order = rng.permutation(n)
        trial = base.copy()
        if mode == "random":
            idx = rng.choice(n, size=k, replace=False)
            noise = rng.standard_normal((k, d))
            trial[idx] = trial[idx] + sigma * noise
            trial[idx] /= np.linalg.norm(trial[idx], axis=1, keepdims=True)
        elif mode == "hub":
            idx = order[:k]
            noise = rng.standard_normal((k, d))
            trial[idx] = trial[idx] + sigma * noise
            trial[idx] /= np.linalg.norm(trial[idx], axis=1, keepdims=True)
            # also small global jitter
            trial = trial + 0.01 * rng.standard_normal((n, d))
            trial /= np.linalg.norm(trial, axis=1, keepdims=True)
        else:  # hole: remove k worst, reinsert at holes
            worst = order[:k]
            keep_mask = np.ones(n, dtype=bool)
            keep_mask[worst] = False
            Y = trial[keep_mask]
            # sequentially insert
            new_pts = []
            Ycur = Y.copy()
            for _ in range(k):
                v0 = best_hole(Ycur, rng, n_samples=3000)
                v0 = refine_hole(Ycur, v0, steps=100, beta=80.0)
                new_pts.append(v0)
                Ycur = np.vstack([Ycur, v0[None, :]])
            trial[worst] = np.array(new_pts)
            # tiny jitter to kept points to break symmetry
            trial[keep_mask] = trial[keep_mask] + 0.005 * rng.standard_normal((n - k, d))
            trial /= np.linalg.norm(trial, axis=1, keepdims=True)
        # anneal: steps depend on k
        if k <= 2:
            steps = 380
        elif k <= 5:
            steps = 500
        else:
            steps = 600
        # time guard: reduce steps if little time left
        if LOOP_END - time.time() < 8:
            steps = 200
        trial = anneal(trial, steps=steps, beta0=22.0, beta1=260.0, lr0=0.022, lr1=0.002)
        # short hard-tail: extra high-beta focus
        trial = anneal(trial, steps=120, beta0=260.0, beta1=400.0, lr0=0.004, lr1=0.001)
        try:
            v = max_cos(trial)
        except Exception:
            stall += 1
            continue
        # selective LP refinement for promising trials
        if v < best_v + 5e-4 and time.time() < LOOP_END - 2:
            trial = lp_polish_deadline(trial, min(time.time() + 1.2, LOOP_END), r_init=0.003)
            try:
                v = max_cos(trial)
            except Exception:
                stall += 1
                continue
        if v < best_v - 1e-12:
            best_X, best_v = trial.copy(), float(v)
            curr_X, curr_v = trial.copy(), float(v)
            stall = 0
        elif v < curr_v - 1e-12:
            curr_X, curr_v = trial.copy(), float(v)
            stall = 0
        else:
            # sideways random-walk acceptance to keep exploring
            if v < best_v + 0.002 and rng.random() < 0.15:
                curr_X, curr_v = trial.copy(), float(v)
            stall += 1
        # safety time check
        if time.time() >= LOOP_END:
            break
    # final polish of best
    try:
        best_X = lp_polish_deadline(best_X, time.time() + FINAL_LP - 1.0, r_init=0.005)
    except Exception:
        pass
    best_X = best_X / np.linalg.norm(best_X, axis=1, keepdims=True)
    return best_X
