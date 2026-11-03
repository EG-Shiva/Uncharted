import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(np.max(g))


def lp_polish(x, seconds):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = max_cos(x)
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        c = (x @ x.T)[iu]
        m = float(c.max())
        sel = c > m - max(20 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        if p == 0:
            break
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate(
            [I[:, None] * d + np.arange(d),
             J[:, None] * d + np.arange(d),
             np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix(
            (x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))),
            shape=(n, nv)).tocsr()
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
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        v = max_cos(y)
        if v < best:
            x, best, radius = y, v, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x


def soft_refine(Xc, steps=500, beta0=20.0, beta1=600.0, s0=0.03, s1=0.005, mu=0.9):
    n, d = Xc.shape
    Xw = Xc.copy()
    V = np.zeros_like(Xw)
    G0 = Xw @ Xw.T
    np.fill_diagonal(G0, -2.0)
    best_v = float(G0.max())
    best_X = Xw.copy()
    for s in range(steps):
        frac = s / float(max(1, steps - 1))
        beta = beta0 * ((beta1 / beta0) ** frac)
        step = s0 + (s1 - s0) * frac
        G = Xw @ Xw.T
        np.fill_diagonal(G, -np.inf)
        m = float(np.max(G))
        if m < best_v:
            best_v = m
            best_X = Xw.copy()
        E = np.exp(beta * (G - m))
        np.fill_diagonal(E, 0.0)
        grad = E @ Xw
        gdot = np.sum(grad * Xw, axis=1, keepdims=True)
        grad -= gdot * Xw
        gn = float(np.linalg.norm(grad, axis=1).max())
        if gn > 1e-12:
            grad /= gn
        V = mu * V + (1.0 - mu) * grad
        Xw = Xw - step * V
        Xw /= np.linalg.norm(Xw, axis=1, keepdims=True)
    v = max_cos(Xw)
    if v < best_v:
        best_v = v
        best_X = Xw.copy()
    return best_X, best_v


def greedy_fill(X, victims, rng, n_cand=25000):
    n, d = X.shape
    m = len(victims)
    fixed_mask = np.ones(n, dtype=bool)
    fixed_mask[np.asarray(victims)] = False
    Xf = X[fixed_mask]
    cand = rng.standard_normal((n_cand, d))
    cand /= np.linalg.norm(cand, axis=1, keepdims=True)
    D = cand @ Xf.T
    scores = D.max(axis=1)
    used = np.zeros(n_cand, dtype=bool)
    new_pts = []
    for _ in range(m):
        masked = np.where(used, np.inf, scores)
        bi = int(np.argmin(masked))
        used[bi] = True
        new_pts.append(cand[bi])
        d2 = cand @ cand[bi]
        # max with already chosen
        np.maximum(scores, d2, out=scores)
        scores[bi] = np.inf
    new_pts = np.array(new_pts)
    Xn = X.copy()
    Xn[np.asarray(victims)] = new_pts
    return Xn


def cap_victims(X, m, rng):
    n, d = X.shape
    if rng.random() < 0.5:
        v = rng.standard_normal(d)
        v /= np.linalg.norm(v)
    else:
        idx = int(rng.integers(n))
        v = X[idx] + 0.35 * rng.standard_normal(d)
        v /= np.linalg.norm(v)
    dots = X @ v
    order = np.argsort(-dots)
    return list(order[:m])


def contact_counts(X, tol=1e-7):
    G = X @ X.T
    np.fill_diagonal(G, -10.0)
    mx = float(G.max())
    C = np.sum(G > mx - tol, axis=1)
    return C, mx


def solve():
    t0 = time.time()
    X0 = None
    for fn in ("best.npy", "record.npy"):
        try:
            X0 = np.load(fn)
            break
        except Exception:
            continue
    if X0 is None:
        r0 = np.random.default_rng(0)
        X0 = r0.standard_normal((95, 13))
    X0 = np.asarray(X0, dtype=np.float64)
    if X0.shape != (95, 13):
        r0 = np.random.default_rng(1)
        tmp = r0.standard_normal((95, 13))
        r = min(95, X0.shape[0])
        c = min(13, X0.shape[1] if X0.ndim > 1 else 13)
        tmp[:r, :c] = X0[:r, :c]
        X0 = tmp
    X0 /= np.linalg.norm(X0, axis=1, keepdims=True)

    rng = np.random.default_rng(12345)
    n, d = (95, 13)

    best = X0.copy()
    best_val = max_cos(best)
    cur = best.copy()
    cur_val = best_val

    search_end = t0 + 152.0
    hard_end = t0 + 172.0
    attempt = 0

    while time.time() < search_end:
        attempt += 1
        elapsed = time.time() - t0
        # temperature for uphill acceptance (decaying)
        Temp = 0.0012 * (1.0 - elapsed / (search_end - t0 + 1e-9)) + 0.00015

        u = rng.random()
        Xc = None
        steps = 450
        if u < 0.48:
            # spatial cap excision
            m = int(rng.integers(4, 13))  # 4..12
            vict = cap_victims(cur, m, rng)
            Xc = greedy_fill(cur, vict, rng, n_cand=25000)
            Xc[vict] += 0.015 * rng.standard_normal((len(vict), d))
            Xc /= np.linalg.norm(Xc, axis=1, keepdims=True)
            steps = 520
        elif u < 0.66:
            # overloaded-to-void: replace worst k by greedy voids
            C, _ = contact_counts(cur, tol=1e-7)
            k = int(rng.integers(1, 4))  # 1..3
            worst = list(np.argsort(-C)[:k])
            # also include a rattler if exists to recycle wasted point
            ratt = np.where(C == 0)[0]
            if len(ratt) > 0 and rng.random() < 0.6:
                # replace worst[0] + rattler instead
                cand_v = worst[:max(1, k - 1)] + [int(ratt[0])] if k > 1 else [int(worst[0])]
                # dedup
                cand_v = list(dict.fromkeys(cand_v))
                worst = cand_v
            Xc = greedy_fill(cur, worst, rng, n_cand=22000)
            Xc[worst] += 0.012 * rng.standard_normal((len(worst), d))
            Xc /= np.linalg.norm(Xc, axis=1, keepdims=True)
            steps = 520
        elif u < 0.80:
            sig = 0.02 + 0.03 * rng.random()
            Xc = cur + sig * rng.standard_normal((n, d))
            Xc /= np.linalg.norm(Xc, axis=1, keepdims=True)
            steps = 420
        elif u < 0.92:
            sig = 0.08 + 0.06 * rng.random()
            Xc = cur + sig * rng.standard_normal((n, d))
            Xc /= np.linalg.norm(Xc, axis=1, keepdims=True)
            steps = 480
        else:
            sig = 0.20 + 0.12 * rng.random()
            Xc = cur + sig * rng.standard_normal((n, d))
            Xc /= np.linalg.norm(Xc, axis=1, keepdims=True)
            steps = 550

        Xr, vr = soft_refine(Xc, steps=steps)

        if vr < cur_val - 1e-12:
            cur, cur_val = Xr, vr
            if vr < best_val - 1e-12:
                # squeeze improvement with short LP
                rem = hard_end - time.time()
                if rem > 6:
                    try:
                        Xp = lp_polish(Xr, min(2.5, rem - 3.0))
                        vp = max_cos(Xp)
                    except Exception:
                        Xp, vp = Xr, vr
                    if vp < best_val:
                        best, best_val = Xp, vp
                        cur, cur_val = Xp, vp
                    else:
                        best, best_val = Xr, vr
                else:
                    best, best_val = Xr, vr
        else:
            # Metropolis uphill / lateral to change basin
            if vr < cur_val + 0.01 and rng.random() < np.exp((cur_val - vr) / max(Temp, 1e-9)):
                cur, cur_val = Xr, vr
            # if wandering too far, pull back toward best
            if cur_val > best_val + 0.006:
                # restart cur near best with medium shake
                cur = best + 0.06 * rng.standard_normal((n, d))
                cur /= np.linalg.norm(cur, axis=1, keepdims=True)
                cur_val = max_cos(cur)

        # occasional cheap polish of near-miss to reveal hidden gain
        if attempt % 14 == 0 and vr < best_val + 0.0006 and vr >= best_val:
            rem = hard_end - time.time()
            if rem > 8:
                try:
                    Xp = lp_polish(Xr, 1.2)
                    vp = max_cos(Xp)
                    if vp < best_val:
                        best, best_val = Xp, vp
                        cur, cur_val = Xp, vp
                except Exception:
                    pass
        if time.time() > hard_end - 20:
            break

    remaining = 178.5 - (time.time() - t0)
    if remaining > 3.0:
        try:
            best = lp_polish(best, min(max(3.0, remaining - 1.0), 16.0))
        except Exception:
            pass
    best /= np.linalg.norm(best, axis=1, keepdims=True)
    return best
