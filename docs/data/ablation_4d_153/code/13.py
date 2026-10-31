import time
import numpy as np


def _max_cos(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))


def _random_sphere(n, d, rng):
    x = rng.standard_normal(size=(n, d))
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x


def _smooth_opt(x, steps, tau0, tau1, lr0, lr1, mom=0.9, rng=None, jitter=0.0):
    n, d = x.shape
    vel = np.zeros_like(x)
    lt0, lt1 = np.log(tau0), np.log(tau1)
    ll0, ll1 = np.log(lr0), np.log(lr1)
    for t in range(steps):
        f = t / max(1, steps - 1)
        tau = float(np.exp(lt0 + (lt1 - lt0) * f))
        lr = float(np.exp(ll0 + (ll1 - ll0) * f))
        G = x @ x.T
        np.fill_diagonal(G, -np.inf)
        m = float(np.max(G))
        # softmax weights
        W = np.exp((G - m) / tau)
        np.fill_diagonal(W, 0.0)
        row = np.sum(W, axis=1, keepdims=True)
        row = np.maximum(row, 1e-12)
        grad = (W @ x) / row
        dot = np.sum(grad * x, axis=1, keepdims=True)
        grad = grad - dot * x
        gn = np.linalg.norm(grad, axis=1, keepdims=True)
        # clip
        bad = gn > 2.0
        if np.any(bad):
            grad = grad * np.where(bad, 2.0 / np.maximum(gn, 1e-12), 1.0)
        vel = mom * vel + (1.0 - mom) * grad
        bc = 1.0 - mom ** (t + 1)
        step = vel / bc
        x = x - lr * step
        if jitter > 0 and rng is not None:
            x = x + jitter * rng.standard_normal(size=x.shape) * 1e-4
        nr = np.linalg.norm(x, axis=1, keepdims=True)
        nr[nr == 0] = 1.0
        x = x / nr
    return x


def _hard_opt(x, steps, delta0, delta1, lr0, lr1, mom=0.9):
    # minimize sum of ReLU(G - (m-delta))^2 style via active set
    n, d = x.shape
    vel = np.zeros_like(x)
    ll0, ll1 = np.log(lr0), np.log(lr1)
    for t in range(steps):
        f = t / max(1, steps - 1)
        delta = delta0 + (delta1 - delta0) * f
        lr = float(np.exp(ll0 + (ll1 - ll0) * f))
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        m = float(np.max(G))
        thr = m - delta
        W = np.maximum(G - thr, 0.0)
        # square to emphasize worst
        W = W * W
        np.fill_diagonal(W, 0.0)
        row = np.sum(W, axis=1, keepdims=True)
        # if all zero, skip
        if np.max(row) < 1e-18:
            continue
        row = np.maximum(row, 1e-12)
        grad = (W @ x) / row
        # scale by mean activity to keep step stable
        dot = np.sum(grad * x, axis=1, keepdims=True)
        grad = grad - dot * x
        gn = np.linalg.norm(grad, axis=1, keepdims=True)
        bad = gn > 2.0
        if np.any(bad):
            grad = grad * np.where(bad, 2.0 / np.maximum(gn, 1e-12), 1.0)
        vel = mom * vel + (1.0 - mom) * grad
        bc = 1.0 - mom ** (t + 1)
        step = vel / bc
        x = x - lr * step
        nr = np.linalg.norm(x, axis=1, keepdims=True)
        nr[nr == 0] = 1.0
        x = x / nr
    return x


def _find_hole(x, exclude, rng, n_samples=20000, refine=400, tau0=0.02, tau1=0.002, lr0=0.05, lr1=0.005):
    n, d = x.shape
    S = rng.standard_normal(size=(n_samples, d))
    S /= np.linalg.norm(S, axis=1, keepdims=True)
    D = S @ x.T
    # exclude column(s)
    if np.isscalar(exclude):
        D[:, exclude] = -2.0
    else:
        D[:, np.asarray(exclude)] = -2.0
    md = np.max(D, axis=1)
    bi = int(np.argmin(md))
    y = S[bi].copy()
    lt0, lt1 = np.log(tau0), np.log(tau1)
    ll0, ll1 = np.log(lr0), np.log(lr1)
    mask = np.ones(n, dtype=bool)
    if np.isscalar(exclude):
        mask[exclude] = False
    else:
        mask[np.asarray(exclude)] = False
    Xm = x[mask]
    for t in range(refine):
        f = t / max(1, refine - 1)
        tau = float(np.exp(lt0 + (lt1 - lt0) * f))
        lr = float(np.exp(ll0 + (ll1 - ll0) * f))
        dv = Xm @ y
        m = float(np.max(dv))
        w = np.exp((dv - m) / tau)
        w = w / max(np.sum(w), 1e-12)
        g = w @ Xm
        g = g - np.dot(g, y) * y
        gn = np.linalg.norm(g)
        if gn > 2.0:
            g = g * (2.0 / gn)
        y = y - lr * g
        ny = np.linalg.norm(y)
        if ny == 0:
            y = S[bi]
        else:
            y = y / ny
    return y


def _lp_polish(x, deadline):
    from scipy.optimize import linprog
    from scipy.sparse import coo_matrix
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    iu = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        G = x @ x.T
        c = G[iu]
        m = float(c.max())
        thresh = m - max(20.0 * radius, 1e-9)
        idx = np.where(c > thresh)[0]
        if idx.size == 0:
            break
        if idx.size > 2500:
            order = np.argpartition(c[idx], -2500)[-2500:]
            idx = idx[order]
        I = iu[0][idx]
        J = iu[1][idx]
        csel = c[idx]
        p = idx.size
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cI = I[:, None] * d + np.arange(d)
        cJ = J[:, None] * d + np.arange(d)
        cols = np.concatenate([cI, cJ, np.full((p, 1), n * d)], axis=1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], axis=1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv)
        obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y = y / np.linalg.norm(y, axis=1, keepdims=True)
        v = _max_cos(y)
        if v < best:
            x = y
            best = v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
        if time.time() >= deadline:
            break
    return x, best


def solve():
    t0 = time.time()
    HARD_END = t0 + 175.0
    POLISH_START = HARD_END - 45.0
    rng = np.random.default_rng(0)
    n, d = 153, 4
    best_x = None
    best_v = 1e9
    for fname in ("best.npy", "record.npy"):
        try:
            a = np.load(fname)
            if a.shape == (n, d):
                a = a / np.linalg.norm(a, axis=1, keepdims=True)
                v = _max_cos(a)
                if v < best_v:
                    best_v = v
                    best_x = a.copy()
        except Exception:
            pass
    if best_x is None:
        best_x = _random_sphere(n, d, rng)
        best_v = _max_cos(best_x)
    cur_x = best_x.copy()
    cur_v = best_v

    sigmas = [0.004, 0.010, 0.025, 0.06, 0.12, 0.22, 0.40, 0.70]
    # preset schedules for smooth opt
    schedules = [
        (2500, 0.06, 0.004, 0.03, 0.002, 0.9),
        (3000, 0.03, 0.002, 0.02, 0.001, 0.9),
        (2000, 0.015, 0.0015, 0.025, 0.002, 0.85),
        (3500, 0.08, 0.003, 0.04, 0.003, 0.92),
    ]

    it = 0
    # contact degrees to pick victims/holes
    while time.time() < POLISH_START - 2.0 and it < 120:
        sigma = sigmas[it % len(sigmas)]
        sched = schedules[it % len(schedules)]
        steps, tau0, tau1, lr0, lr1, mom = sched
        # adapt steps to remaining time
        if time.time() > POLISH_START - 20.0:
            steps = min(steps, 1500)

        mode = it % 10
        # choose base
        if mode == 9:
            # fresh random restart
            trial = _random_sphere(n, d, rng)
            # quick coarse opt with large tau
            trial = _smooth_opt(trial, steps, tau0, tau1, lr0, lr1, mom)
            trial = _hard_opt(trial, 800, 0.05, 0.005, 0.01, 0.002, 0.9)
        elif mode == 4 or mode == 7:
            # hole relocation: move 1-3 worst points to holes
            base = best_x if (it % 3 != 0) else cur_x
            G = base @ base.T
            np.fill_diagonal(G, -2.0)
            m = float(np.max(G))
            # degree in near-contact graph
            thr = m - 1e-3
            deg = np.sum(G > thr, axis=1)
            # victims: highest degree + involved in max pair
            # find max pair
            ij = np.unravel_index(int(np.argmax(G)), G.shape)
            cand = list(ij)
            # add high-degree nodes
            order = np.argsort(-deg)
            for o in order:
                if o not in cand and len(cand) < 3:
                    # randomize: only add with prob
                    if rng.random() < 0.6:
                        cand.append(int(o))
                if len(cand) >= 3:
                    break
            # number to move: 1-3 depending on sigma
            nmove = 1 if sigma < 0.05 else (2 if sigma < 0.2 else 3)
            cand = cand[:nmove]
            trial = base.copy()
            for v in cand:
                h = _find_hole(base, v, rng)
                # add small noise
                h = h + sigma * 0.3 * rng.standard_normal(size=d)
                h /= np.linalg.norm(h)
                trial[v] = h
            # small global jitter
            trial = trial + sigma * 0.2 * rng.standard_normal(size=trial.shape)
            trial /= np.linalg.norm(trial, axis=1, keepdims=True)
            trial = _smooth_opt(trial, steps, tau0, tau1, lr0, lr1, mom)
            # extra hard polish sometimes
            if it % 2 == 0:
                trial = _hard_opt(trial, 600, 0.03, 0.003, 0.008, 0.0015, 0.9)
        else:
            base = best_x if (rng.random() < 0.7) else cur_x
            # tangent-ish Gaussian perturb
            noise = rng.standard_normal(size=base.shape)
            # make tangential to preserve norm approx
            dot = np.sum(noise * base, axis=1, keepdims=True)
            noise = noise - dot * base
            trial = base + sigma * noise
            trial /= np.linalg.norm(trial, axis=1, keepdims=True)
            trial = _smooth_opt(trial, steps, tau0, tau1, lr0, lr1, mom)
            if mode in (2, 5):
                trial = _hard_opt(trial, 500, 0.04, 0.004, 0.01, 0.002, 0.9)

        v = _max_cos(trial)
        if v < best_v:
            best_v = v
            best_x = trial.copy()
            cur_x = trial.copy()
            cur_v = v
        elif v < cur_v or rng.random() < np.exp(-max(0.0, v - cur_v) / max(sigma * 0.02, 1e-4)):
            # simulated-annealing style acceptance (temperature ~ sigma)
            cur_x = trial.copy()
            cur_v = v
        # occasional reheat from best
        if it % 15 == 14 and time.time() < POLISH_START - 5:
            cur_x = best_x.copy()
            cur_v = best_v
        it += 1
        if time.time() >= POLISH_START - 1.0:
            break

    # final polish of best with remaining time
    try:
        px, pv = _lp_polish(best_x, HARD_END - 1.0)
        if pv < best_v:
            best_x = px
            best_v = pv
    except Exception:
        pass
    best_x = best_x / np.linalg.norm(best_x, axis=1, keepdims=True)
    return best_x
