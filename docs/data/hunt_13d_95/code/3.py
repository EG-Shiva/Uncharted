import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())


def point_maxs(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return g.max(axis=1), g


def smooth_refine(x, steps, beta0, beta1, lr, max_step=0.03):
    n, d = x.shape
    beta = float(beta0)
    grow = (float(beta1) / float(beta0)) ** (1.0 / max(1, steps))
    for _ in range(steps):
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        gmax = float(G.max())
        # weights exp(beta*(G-gmax))
        E = np.exp(beta * (G - gmax))
        np.fill_diagonal(E, 0.0)
        S = float(E.sum())
        if not np.isfinite(S) or S <= 0:
            break
        grad = (E @ x) / S
        proj = np.sum(grad * x, axis=1, keepdims=True)
        gt = grad - proj * x
        step = lr * gt
        # per-row trust region clip
        nm = np.linalg.norm(step, axis=1, keepdims=True)
        # avoid div by zero
        scale = np.ones_like(nm)
        over = nm > max_step
        scale[over] = max_step / (nm[over] + 1e-16)
        step = step * scale
        xn = x - step
        nr = np.linalg.norm(xn, axis=1, keepdims=True)
        nr[nr == 0] = 1.0
        x = xn / nr
        beta *= grow
    return x


def lp_polish_deadline(x, deadline):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    try:
        best = max_cos(x)
    except Exception:
        return x
    radius = 1e-3
    iu = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        cfull = (x @ x.T)[iu]
        m = float(cfull.max())
        thr = m - max(20.0 * radius, 1e-9)
        sel = cfull > thr
        # cap constraints for speed: keep top 1800
        if int(sel.sum()) > 1800:
            # keep largest 1800
            idx = np.where(sel)[0]
            order = np.argsort(cfull[idx])[-1800:]
            mask = np.zeros_like(sel, dtype=bool)
            mask[idx[order]] = True
            sel = mask
        I = iu[0][sel]
        J = iu[1][sel]
        p = len(I)
        if p == 0:
            break
        c = cfull[sel]
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))),
                              shape=(n, nv)).tocsr()
        except Exception:
            break
        obj = np.zeros(nv)
        obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                          method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        nr = np.linalg.norm(y, axis=1, keepdims=True)
        nr[nr == 0] = 1.0
        y = y / nr
        if not np.all(np.isfinite(y)):
            radius *= 0.5
            continue
        v = max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.02)
        else:
            radius *= 0.5
    return x


def rand_sphere(rng, d):
    v = rng.randn(d)
    n = np.linalg.norm(v)
    if n == 0:
        v[0] = 1.0
        n = 1.0
    return v / n


def solve():
    rng = np.random.RandomState(0)
    t0 = time.time()
    hard_end = t0 + 172.0
    reserve_final = 18.0

    # load starts
    cand = []
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=float)
            if a.shape == (95, 13):
                nr = np.linalg.norm(a, axis=1, keepdims=True)
                nr[nr == 0] = 1.0
                a = a / nr
                if np.all(np.isfinite(a)):
                    cand.append(a)
        except Exception:
            pass
    if len(cand) == 0:
        # random start fallback
        a = rng.randn(95, 13)
        a /= np.linalg.norm(a, axis=1, keepdims=True)
        cand.append(a)
    best_x = None
    best_v = 1e9
    for a in cand:
        v = max_cos(a)
        if v < best_v:
            best_v = v
            best_x = a.copy()
    n, d = 95, 13

    # quick contact info for targeting
    def get_degrees(x, thr=1e-4):
        pm, G = point_maxs(x)
        gmax = float(G.max())
        cnt = np.sum(G > gmax - thr, axis=1)
        return cnt, pm, G, gmax

    trial = 0
    # main basin-hopping loop
    while time.time() < hard_end - reserve_final:
        trial += 1
        if time.time() > hard_end - reserve_final:
            break
        cnt, pm, Gcur, gmax = get_degrees(best_x, thr=2e-3)
        order_worst = np.argsort(-pm)  # largest point-max first
        order_loose = np.argsort(cnt)  # smallest contacts first

        mode = trial % 5
        x_try = best_x.copy()
        if mode == 0:
            # global small shake
            sigma = float(rng.choice([0.008, 0.015, 0.03]))
            x_try = x_try + sigma * rng.randn(n, d)
        elif mode == 1:
            # kick worst points
            k = int(rng.randint(4, 14))
            sel = order_worst[:k]
            sigma = float(rng.choice([0.05, 0.12, 0.25]))
            x_try[sel] += sigma * rng.randn(len(sel), d)
        elif mode == 2:
            # kick loose + hub
            k1 = int(rng.randint(1, 4))
            k2 = int(rng.randint(2, 8))
            hub = np.argsort(-cnt)[:k2]
            loose = order_loose[:k1]
            sel = np.unique(np.concatenate([hub, loose]))
            sigma = float(rng.choice([0.08, 0.15, 0.3]))
            x_try[sel] += sigma * rng.randn(len(sel), d)
        elif mode == 3:
            # random subset medium kick
            k = int(rng.randint(6, 20))
            sel = rng.choice(n, size=k, replace=False)
            sigma = float(rng.choice([0.05, 0.12, 0.22]))
            x_try[sel] += sigma * rng.randn(len(sel), d)
        else:
            # reinsert 1-2 points randomly + shake neighbors
            k = int(rng.randint(1, 3))
            # reinsert points among loose/worst mix
            pool = np.concatenate([order_loose[:10], order_worst[:10]])
            sel = rng.choice(pool, size=k, replace=False)
            for s in sel:
                x_try[s] = rand_sphere(rng, d)
            # small shake to all to break symmetry
            x_try += 0.01 * rng.randn(n, d)

        # renormalize
        nr = np.linalg.norm(x_try, axis=1, keepdims=True)
        nr[nr == 0] = 1.0
        x_try /= nr

        # smooth params random
        beta0 = float(rng.choice([40.0, 80.0, 120.0]))
        beta1 = float(rng.choice([250.0, 400.0, 600.0]))
        if beta1 < beta0:
            beta1 = beta0 * 3.0
        lr = float(rng.choice([0.6, 1.2, 2.0, 3.0]))
        steps = int(rng.choice([400, 700, 1100]))
        # adapt steps to remaining time (rough 0.15ms/step)
        # keep it bounded
        x_try = smooth_refine(x_try, steps, beta0, beta1, lr)

        v_try = max_cos(x_try)
        # promising if within window or better
        # window shrinks as we go
        window = 0.015 if trial < 20 else 0.008
        if v_try < best_v + window:
            # short LP refine if time allows
            dl = min(3.0, (hard_end - reserve_final) - time.time())
            if dl < 0.4:
                # no time for LP, just check direct
                if v_try < best_v - 1e-12:
                    best_x, best_v = x_try, v_try
                continue
            x_ref = lp_polish_deadline(x_try, time.time() + dl)
            v_ref = max_cos(x_ref)
            if v_ref < best_v - 1e-12:
                best_x, best_v = x_ref, v_ref
                # extend search: keep going
            elif v_try < best_v - 1e-12:
                # LP failed but smooth alone improved
                best_x, best_v = x_try, v_try
        else:
            # occasional acceptance of slightly worse to diversify? no
            pass

        # safety: if we already beat record by decent margin, spend more LP
        if best_v < 0.2435 and time.time() < hard_end - reserve_final - 5:
            pass

    # final squeeze on best
    t_rem = hard_end - time.time()
    if t_rem > 0.5:
        # a final smooth with large beta then LP
        try:
            best_x = smooth_refine(best_x, 400, 150.0, 800.0, 1.0, max_step=0.01)
        except Exception:
            pass
        t_rem = hard_end - time.time()
        if t_rem > 0.5:
            best_x = lp_polish_deadline(best_x, time.time() + t_rem - 0.5)

    best_x = best_x / np.linalg.norm(best_x, axis=1, keepdims=True)
    return best_x
