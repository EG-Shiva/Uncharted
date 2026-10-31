import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())


def _normalize(x):
    nrm = np.linalg.norm(x, axis=1, keepdims=True)
    nrm[~np.isfinite(nrm)] = 1.0
    nrm[nrm < 1e-12] = 1.0
    x = x / nrm
    x[~np.isfinite(x)] = 0.0
    nrm2 = np.linalg.norm(x, axis=1, keepdims=True)
    bad = (nrm2[:, 0] < 1e-12)
    if np.any(bad):
        x[bad] = np.array([1.0, 0.0, 0.0, 0.0])
        x = x / np.linalg.norm(x, axis=1, keepdims=True)
    return x


def _anneal(x, steps, tau0, tau1, lr0, lr1):
    n, d = x.shape
    m = np.zeros_like(x)
    v = np.zeros_like(x)
    b1, b2, eps = 0.9, 0.999, 1e-8
    for k in range(steps):
        frac = k / (steps - 1) if steps > 1 else 0.0
        tau = tau0 * (tau1 / tau0) ** frac
        lr = lr0 + (lr1 - lr0) * frac
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        gm = float(G.max())
        W = np.exp((G - gm) / tau)
        np.fill_diagonal(W, 0.0)
        S = float(W.sum())
        if not np.isfinite(S) or S <= 0:
            break
        W /= S
        grad = W @ x
        dot = (grad * x).sum(axis=1, keepdims=True)
        g = grad - dot * x
        m *= b1
        m += (1.0 - b1) * g
        v *= b2
        v += (1.0 - b2) * (g * g)
        bc1 = 1.0 - b1 ** (k + 1)
        bc2 = 1.0 - b2 ** (k + 1)
        mh = m / bc1
        vh = v / bc2
        step = mh / (np.sqrt(vh) + eps)
        np.clip(step, -2.0, 2.0, out=step)
        x = x - lr * step
        x = _normalize(x)
    return x


def _best_void(others, rng, n_cand=6000):
    cand = rng.standard_normal((n_cand, 4))
    cand = _normalize(cand)
    sims = cand @ others.T
    scores = sims.max(axis=1)
    bi = int(np.argmin(scores))
    b = cand[bi].copy()
    bs = float(scores[bi])
    step = 0.05
    for _ in range(250):
        prop = b + step * rng.standard_normal((32, 4))
        prop = _normalize(prop)
        sc = (prop @ others.T).max(axis=1)
        j = int(np.argmin(sc))
        if float(sc[j]) < bs:
            b = prop[j].copy()
            bs = float(sc[j])
            step *= 1.05
        else:
            step *= 0.96
        if step < 1e-5:
            break
    return b


def _personal_worst(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return G.max(axis=1)


def _lp_polish(x, seconds, deadline):
    x = _normalize(x)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and time.time() < deadline and radius > 1e-14:
        c = (x @ x.T)[iu]
        mcur = float(c.max())
        sel = c > mcur - max(20.0 * radius, 1e-9)
        I = iu[0][sel]
        J = iu[1][sel]
        p = len(I)
        nv = n * d + 1
        if p == 0:
            break
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], axis=1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], axis=1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
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
        y = _normalize(y)
        vv = _max_cos(y)
        if vv < best:
            x = y
            best = vv
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x


def _small_rotation(rng, sigma):
    M = np.eye(4) + sigma * rng.standard_normal((4, 4))
    Q, R = np.linalg.qr(M)
    dd = np.diag(R)
    s = np.sign(dd)
    s[s == 0] = 1.0
    Q = Q * s[None, :]
    if np.linalg.det(Q) < 0:
        Q[:, 0] = -Q[:, 0]
    return Q


def solve():
    t0 = time.time()
    deadline = t0 + 172.0
    rng = np.random.default_rng(12345)
    try:
        x0 = np.load("best.npy")
    except Exception:
        try:
            x0 = np.load("record.npy")
        except Exception:
            x0 = rng.standard_normal((272, 4))
    x0 = np.asarray(x0, dtype=float)
    if x0.shape != (272, 4):
        flat = x0.ravel()
        need = 272 * 4
        if flat.size < need:
            extra = rng.standard_normal((need - flat.size,))
            flat = np.concatenate([flat, extra])
        x0 = flat[:need].reshape(272, 4)
    x0 = _normalize(x0)
    best = x0.copy()
    best_val = _max_cos(best)

    it = 0
    # hopping + fresh starts until ~135s, reserve rest for LP
    while time.time() < t0 + 132.0 and time.time() < deadline - 38.0:
        it += 1
        if it % 3 == 0:
            # fresh random basin: longer anneal, two stages
            y = _normalize(rng.standard_normal((272, 4)))
            # stage 1 broad
            y = _anneal(y, 1600, 0.020, 0.002, 0.020, 0.003)
            if time.time() > t0 + 132.0 or time.time() > deadline - 38.0:
                vv = _max_cos(y)
                if vv < best_val:
                    best, best_val = y, vv
                break
            y = _anneal(y, 1200, 0.005, 0.0006, 0.008, 0.001)
            vv = _max_cos(y)
            if vv < best_val:
                best = y.copy()
                best_val = vv
        elif it % 3 == 1:
            # multi-point void reinsertion on best (topology change)
            y = best.copy()
            pw = _personal_worst(y)
            order = np.argsort(-pw)
            k = 4 + (it % 4) * 2  # 4,6,8,10
            pool = order[:24]
            # pick k victims: mostly worst, some random for diversity
            perm = rng.permutation(len(pool))[:k]
            victims = pool[perm]
            # greedy sequential refill
            fixed_mask = np.ones(272, dtype=bool)
            fixed_mask[victims] = False
            placed = []
            for vi in victims:
                if len(placed) == 0:
                    others = y[fixed_mask]
                else:
                    others = np.vstack([y[fixed_mask], np.stack(placed)])
                b = _best_void(others, rng, n_cand=5000)
                placed.append(b)
            for vi, b in zip(victims, placed):
                y[vi] = b
            # small kick to neighbours to break local jam
            y = y + 0.004 * rng.standard_normal((272, 4))
            y = _normalize(y)
            y = _anneal(y, 1100, 0.008, 0.0007, 0.009, 0.001)
            vv = _max_cos(y)
            if vv < best_val:
                best = y.copy()
                best_val = vv
        else:
            # rigid cluster rotation + kick
            y = best.copy()
            u = rng.standard_normal((4,))
            u /= np.linalg.norm(u)
            dpro = y @ u
            k = int(12 + (it % 5) * 4)  # 12..28
            idx = np.argsort(-dpro)[:k]
            Q = _small_rotation(rng, 0.35)
            mu = y[idx].mean(axis=0, keepdims=True)
            mu /= np.linalg.norm(mu)
            # rotate cluster rigidly around origin
            y[idx] = y[idx] @ Q.T
            y = y + 0.008 * rng.standard_normal((272, 4))
            y = _normalize(y)
            y = _anneal(y, 1100, 0.007, 0.0007, 0.009, 0.001)
            vv = _max_cos(y)
            if vv < best_val:
                best = y.copy()
                best_val = vv
        # opportunistic extra sharpening if we just improved
        if time.time() > t0 + 132.0:
            break

    # final LP tightening on the best found
    remain = deadline - time.time()
    if remain > 8.0:
        secs = min(30.0, remain - 3.0)
        try:
            best = _lp_polish(best, secs, deadline - 2.0)
        except Exception:
            pass
    best = _normalize(best)
    return best
