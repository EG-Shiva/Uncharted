import time
import numpy as np
from scipy.optimize import minimize, linprog
from scipy.sparse import coo_matrix


def max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())


def smooth_opt(X0, betas, iters, t_deadline=None):
    n, d = X0.shape
    Y = X0.copy()
    # ensure normalized
    Y = Y / np.linalg.norm(Y, axis=1, keepdims=True)
    iu0 = np.triu_indices(n, 1)
    I0, J0 = iu0
    for beta, maxiter in zip(betas, iters):
        if t_deadline is not None and time.time() > t_deadline:
            break
        y0 = Y.ravel()

        def fun(yflat, _beta=beta):
            Ym = yflat.reshape(n, d)
            r = np.linalg.norm(Ym, axis=1)
            r = np.where(r == 0, 1.0, r)
            X = Ym / r[:, None]
            G = X @ X.T
            D = G[I0, J0]
            m = D.max()
            E = np.exp(_beta * (D - m))
            S = E.sum()
            f = m + np.log(S) / _beta
            # small norm penalty to stop radial drift
            pen = 1e-4 * float(((r - 1.0) ** 2).sum())
            f = f + pen
            Wv = E / S
            # grad_x = W @ X
            # build via accumulation without full matrix
            Gx = np.zeros_like(X)
            # W symmetric: contribution to row I from col J and vice versa
            # Gx[I] += Wv * X[J]; Gx[J] += Wv * X[I]
            np.add.at(Gx, I0, Wv[:, None] * X[J0])
            np.add.at(Gx, J0, Wv[:, None] * X[I0])
            dot = (Gx * X).sum(axis=1)
            Gy = (Gx - dot[:, None] * X) / r[:, None]
            # penalty grad
            Gy = Gy + (2e-4 * (r - 1.0) / r)[:, None] * Ym * 0.0 + (2e-4 * (r - 1.0))[:, None] * (Ym / (r[:, None] ** 2) * 0 + 1.0 / 1.0) * 0.0
            # correct penalty grad: d/dr 1e-4*(r-1)^2 * dr/dYm ; dr/dYm = X
            Gy = Gy + (2e-4 * (r - 1.0))[:, None] * X
            return float(f), Gy.ravel()
        try:
            res = minimize(fun, y0, jac=True, method="L-BFGS-B",
                           options={"maxiter": maxiter, "maxfun": maxiter * 5,
                                    "ftol": 1e-22, "gtol": 1e-14})
            Y = res.x.reshape(n, d)
            nr = np.linalg.norm(Y, axis=1, keepdims=True)
            nr[nr == 0] = 1.0
            Y = Y / nr
        except Exception:
            break
        if not np.all(np.isfinite(Y)):
            Y = X0.copy()
            break
    nr = np.linalg.norm(Y, axis=1, keepdims=True)
    nr[nr == 0] = 1.0
    return Y / nr


def lp_polish(x, seconds):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = max_cos(x)
    radius = 5e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        c = (x @ x.T)[iu]
        m = c.max()
        sel = c > m - max(15.0 * radius, 2e-9)
        I, J = iu[0][sel], iu[1][sel]
        p, nv = len(I), n * d + 1
        if p == 0:
            break
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        except Exception:
            break
        obj = np.zeros(nv)
        obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c[sel], A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        v = max_cos(y)
        if v < best:
            x, best, radius = y, v, min(radius * 1.5, 0.03)
        else:
            radius *= 0.5
    return x


def far_reinsert(X, victims, rng, ncand=2000):
    n, d = X.shape
    for v in victims:
        cand = rng.standard_normal(size=(ncand, d))
        cand /= np.linalg.norm(cand, axis=1, keepdims=True)
        dots = cand @ X.T
        dots[:, v] = -10.0
        mv = dots.max(axis=1)
        bi = int(np.argmin(mv))
        X[v] = cand[bi]
    return X


def solve():
    t0 = time.time()
    deadline_explore = t0 + 146.0
    deadline_total = t0 + 174.0
    rng = np.random.default_rng(12345)

    best = None
    best_val = 1e9
    for fn in ["best.npy", "record.npy"]:
        try:
            a = np.load(fn).astype(float)
            a = a / np.linalg.norm(a, axis=1, keepdims=True)
            if a.shape == (94, 13) and np.all(np.isfinite(a)):
                v = max_cos(a)
                if v < best_val:
                    best_val = v
                    best = a.copy()
        except Exception:
            pass
    if best is None:
        best = rng.standard_normal(size=(94, 13))
        best /= np.linalg.norm(best, axis=1, keepdims=True)
        best_val = max_cos(best)
    n, d = best.shape

    # quick true-up of starting point with smoothing + short LP if time allows
    try:
        if time.time() < deadline_explore - 5:
            B = smooth_opt(best, [120.0, 600.0, 2500.0], [120, 100, 60],
                           t_deadline=deadline_explore)
            vB = max_cos(B)
            if vB < best_val:
                best, best_val = B, vB
    except Exception:
        pass

    cur = best.copy()
    cur_val = best_val
    sigmas = [0.004, 0.009, 0.02, 0.045, 0.09, 0.18]
    trial = 0
    T0 = 0.0022
    while time.time() < deadline_explore:
        trial += 1
        elapsed = time.time() - t0
        T = T0 * max(0.04, 1.0 - elapsed / 150.0)
        rchoice = rng.random()
        Xtry = cur.copy()
        if rchoice < 0.38:
            sigma = sigmas[int(rng.integers(0, len(sigmas)))]
            if rng.random() < 0.4:
                G = Xtry @ Xtry.T
                np.fill_diagonal(G, -2.0)
                mx = G.max()
                cnt = (G > mx - 2e-4).sum(axis=1)
                order_loose = np.argsort(cnt)
                order_jammed = np.argsort(-cnt)
                victims = set()
                k = int(rng.integers(2, 6))
                for _ in range(k):
                    if rng.random() < 0.5:
                        victims.add(int(order_loose[int(rng.integers(0, 10))]))
                    else:
                        victims.add(int(order_jammed[int(rng.integers(0, 10))]))
                victims = list(victims)
                noise = rng.standard_normal(size=Xtry.shape) * (sigma * 0.5)
                noise[victims, :] *= 4.0
                Xtry = Xtry + noise
            else:
                Xtry = Xtry + sigma * rng.standard_normal(size=Xtry.shape)
            Xtry /= np.linalg.norm(Xtry, axis=1, keepdims=True)
        elif rchoice < 0.62:
            # shake worst cluster strongly
            G = Xtry @ Xtry.T
            np.fill_diagonal(G, -2.0)
            worst = G.max(axis=1)
            order = np.argsort(-worst)
            k = int(rng.integers(1, 5))
            victims = [int(order[0])]
            if k > 1:
                pool = order[:14].tolist()
                pool = [p for p in pool if p not in victims]
                rng.shuffle(pool)
                for p in pool[:k - 1]:
                    victims.append(int(p))
            sigma = float(rng.choice(np.array([0.02, 0.05, 0.1])))
            noise = rng.standard_normal(size=Xtry.shape) * (sigma * 0.25)
            noise[victims, :] *= 3.0
            # extra push worst pair apart tangentially
            try:
                iu = np.triu_indices(n, 1)
                gg = (Xtry @ Xtry.T)[iu]
                bi = int(np.argmax(gg))
                a1, a2 = int(iu[0][bi]), int(iu[1][bi])
                push = 0.5 * sigma
                d1 = Xtry[a1] - (Xtry[a1] @ Xtry[a2]) * Xtry[a2]
                d2 = Xtry[a2] - (Xtry[a2] @ Xtry[a1]) * Xtry[a1]
                n1 = np.linalg.norm(d1); n2 = np.linalg.norm(d2)
                if n1 > 1e-12:
                    Xtry[a1] = Xtry[a1] + (push / n1) * d1 * 2.0
                if n2 > 1e-12:
                    Xtry[a2] = Xtry[a2] + (push / n2) * d2 * 2.0
            except Exception:
                pass
            Xtry = Xtry + noise
            Xtry /= np.linalg.norm(Xtry, axis=1, keepdims=True)
        elif rchoice < 0.85:
            # kill worst points and reinsert into largest hole
            G = Xtry @ Xtry.T
            np.fill_diagonal(G, -2.0)
            worst = G.max(axis=1)
            order = np.argsort(-worst)
            k = int(rng.integers(1, 4))
            victims = [int(order[0])]
            if k > 1:
                pool = order[:16].tolist()
                pool = [p for p in pool if p not in victims]
                rng.shuffle(pool)
                for p in pool[:k - 1]:
                    victims.append(int(p))
            Xtry = far_reinsert(Xtry, victims, rng, ncand=2000)
            Xtry = Xtry + 0.0025 * rng.standard_normal(size=Xtry.shape)
            Xtry /= np.linalg.norm(Xtry, axis=1, keepdims=True)
        else:
            # kill rattlers / low-contact points
            G = Xtry @ Xtry.T
            np.fill_diagonal(G, -2.0)
            mx = G.max()
            cnt = (G > mx - 2e-4).sum(axis=1)
            order = np.argsort(cnt)
            k = int(rng.integers(1, 4))
            # take from loosest 12
            pool = order[:12].tolist()
            rng.shuffle(pool)
            victims = [int(p) for p in pool[:k]]
            Xtry = far_reinsert(Xtry, victims, rng, ncand=2500)
            Xtry = Xtry + 0.002 * rng.standard_normal(size=Xtry.shape)
            Xtry /= np.linalg.norm(Xtry, axis=1, keepdims=True)

        # relax with escalating LSE
        try:
            Xtry = smooth_opt(Xtry, [60.0, 350.0, 1500.0], [70, 60, 45],
                              t_deadline=deadline_explore)
        except Exception:
            continue
        if not np.all(np.isfinite(Xtry)):
            continue
        v = max_cos(Xtry)
        if v < best_val - 1e-12:
            # extra squeeze immediately
            try:
                Xf = smooth_opt(Xtry, [3000.0, 8000.0], [40, 30],
                                t_deadline=deadline_explore)
                vf = max_cos(Xf)
                if vf < v:
                    Xtry, v = Xf, vf
            except Exception:
                pass
            best_val = v
            best = Xtry.copy()
            cur, cur_val = Xtry.copy(), v
        else:
            if v < cur_val or rng.random() < np.exp((cur_val - v) / max(T, 1e-7)):
                cur, cur_val = Xtry.copy(), v
            if rng.random() < 0.09:
                cur, cur_val = best.copy(), best_val

    # final intensive polish of global best
    try:
        tleft = deadline_total - time.time() - 2.0
        if tleft > 8:
            B = smooth_opt(best, [400.0, 1500.0, 5000.0], [150, 120, 80],
                           t_deadline=deadline_total - 22.0)
            if max_cos(B) < best_val:
                best = B
                best_val = max_cos(B)
    except Exception:
        pass
    try:
        trem = deadline_total - time.time() - 1.0
        if trem > 4:
            B2 = lp_polish(best, min(trem, 20.0))
            if max_cos(B2) < max_cos(best):
                best = B2
    except Exception:
        pass
    best = best / np.linalg.norm(best, axis=1, keepdims=True)
    return best
