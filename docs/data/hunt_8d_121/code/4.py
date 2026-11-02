import time
import numpy as np
from scipy.optimize import linprog, minimize
from scipy.sparse import coo_matrix


def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())


def normed(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n = np.maximum(n, 1e-12)
    return x / n


def lp_polish_deadline(x, deadline):
    x = normed(x)
    n, d = x.shape
    best = max_cos(x)
    radius = 2e-3
    iu0 = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        G = x @ x.T
        c = G[iu0]
        m = c.max()
        sel = c > m - max(20.0 * radius, 1e-9)
        I = iu0[0][sel]
        J = iu0[1][sel]
        p = len(I)
        if p == 0:
            break
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv)
        obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c[sel], A_eq=A_eq,
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
        y = normed(y)
        v = max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x


def refine_lse(x0, beta, maxiter):
    n, d = x0.shape
    iu = np.triu_indices(n, 1)
    X0f = x0.ravel().copy()

    def fun_grad(Xf):
        X = Xf.reshape(n, d)
        R = np.linalg.norm(X, axis=1, keepdims=True)
        R = np.maximum(R, 1e-12)
        Y = X / R
        G = Y @ Y.T
        D = G[iu]
        M = D.max()
        E = np.exp(beta * (D - M))
        S = E.sum()
        loss = M + np.log(S) / beta
        Wv = E / S
        W = np.zeros((n, n))
        W[iu] = Wv
        W = W + W.T
        GradY = W @ Y
        dot = (GradY * Y).sum(axis=1, keepdims=True)
        GradYt = GradY - dot * Y
        GradX = GradYt / R
        return float(loss), GradX.ravel()

    try:
        res = minimize(fun_grad, X0f, jac=True, method="L-BFGS-B",
                       options={"maxiter": maxiter, "ftol": 1e-12, "gtol": 1e-7})
        Xopt = res.x.reshape(n, d)
    except Exception:
        return normed(x0)
    return normed(Xopt)


def solve():
    t0 = time.time()
    DEADLINE = t0 + 175.0
    rng = np.random.default_rng(0)

    try:
        b1 = np.load("best.npy")
    except Exception:
        b1 = None
    try:
        b2 = np.load("record.npy")
    except Exception:
        b2 = None

    cands = []
    if b1 is not None:
        cands.append(normed(b1))
    if b2 is not None:
        cands.append(normed(b2))
    if len(cands) == 0:
        cands.append(normed(rng.standard_normal((121, 8))))

    best = cands[0]
    best_val = max_cos(best)
    for c in cands[1:]:
        v = max_cos(c)
        if v < best_val:
            best, best_val = c, v

    n, d = best.shape  # 121, 8
    sigmas = [0.005, 0.01, 0.025, 0.06, 0.12, 0.25]
    betas = [40.0, 80.0, 150.0, 300.0]

    it = 0
    # exploration phase until ~130s
    while time.time() < t0 + 132.0:
        it += 1
        if time.time() > DEADLINE - 30:
            break
        sigma = sigmas[rng.integers(len(sigmas))]
        beta = betas[it % len(betas)]
        mode = it % 6

        if mode in (0, 1):
            prop = best + sigma * rng.standard_normal(best.shape)
            prop = normed(prop)
        elif mode == 2:
            # worst-pair separation kick
            G = best @ best.T
            np.fill_diagonal(G, -2.0)
            flat = int(np.argmax(G))
            a = flat // n
            b = flat % n
            step = float(rng.uniform(0.05, 0.25))
            prop = best.copy()
            dv = best[a] - best[b]
            prop[a] = prop[a] + step * dv
            prop[b] = prop[b] - step * dv
            prop = prop + 0.008 * rng.standard_normal(prop.shape)
            prop = normed(prop)
        elif mode == 3:
            # worst-point void re-insertion
            G = best @ best.T
            np.fill_diagonal(G, -2.0)
            rowmax = G.max(axis=1)
            worst = int(np.argmax(rowmax))
            mask = np.ones(n, dtype=bool)
            mask[worst] = False
            others = best[mask]
            K = 300
            cand_pts = normed(rng.standard_normal((K, d)))
            dots = cand_pts @ others.T
            md = dots.max(axis=1)
            pick = int(np.argmin(md))
            prop = best.copy()
            prop[worst] = cand_pts[pick]
            prop = prop + 0.005 * rng.standard_normal(prop.shape)
            prop = normed(prop)
        elif mode == 4:
            # subset shake
            prop = best.copy()
            k = int(rng.integers(5, 15))
            idx = rng.choice(n, size=k, replace=False)
            prop[idx] = prop[idx] + (sigma * 2.0) * rng.standard_normal((k, d))
            prop = prop + 0.003 * rng.standard_normal(prop.shape)
            prop = normed(prop)
        else:
            # small global + targeted noise on top pairs
            G = best @ best.T
            np.fill_diagonal(G, -2.0)
            # top 5 pairs
            flat_idx = np.argpartition(G.ravel(), -5)[-5:]
            involved = np.unique(np.concatenate([(flat_idx // n), (flat_idx % n)]))
            prop = best + 0.004 * rng.standard_normal(best.shape)
            prop[involved] = prop[involved] + sigma * rng.standard_normal((len(involved), d))
            prop = normed(prop)

        maxiter = 90 if sigma > 0.05 else 120
        # keep L-BFGS cheap when little time left
        if time.time() > t0 + 110:
            maxiter = 70
        refined = refine_lse(prop, beta, maxiter=maxiter)
        v = max_cos(refined)
        if v < best_val - 1e-12:
            best, best_val = refined, v
            continue
        # if close, try short LP squeeze (1.2s) to get true minimax value
        if v < best_val + 0.002 and time.time() < t0 + 128:
            sq = lp_polish_deadline(refined, min(time.time() + 1.2, DEADLINE - 25))
            vsq = max_cos(sq)
            if vsq < best_val - 1e-12:
                best, best_val = sq, vsq

    # final polish of best with remaining time
    if time.time() < DEADLINE - 2:
        best = lp_polish_deadline(best, DEADLINE - 1.5)
        best_val = max_cos(best)

    return normed(best)
