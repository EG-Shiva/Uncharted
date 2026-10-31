import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _norm(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[~np.isfinite(n)] = 1.0
    n[n < 1e-12] = 1.0
    return x / n

def _anneal(x, steps, tau0, tau1, lr0, lr1):
    n, d = x.shape
    m = np.zeros_like(x)
    v = np.zeros_like(x)
    b1, b2, eps = 0.9, 0.999, 1e-8
    ltau0 = np.log(tau0); ltau1 = np.log(tau1)
    for k in range(steps):
        frac = k / (steps - 1) if steps > 1 else 0.0
        tau = float(np.exp(ltau0 + (ltau1 - ltau0) * frac))
        lr = lr0 + (lr1 - lr0) * frac
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        gm = G.max()
        G -= gm
        G /= tau
        # clip for exp overflow
        np.clip(G, -30.0, 0.0, out=G)
        np.exp(G, out=G)
        np.fill_diagonal(G, 0.0)
        S = G.sum()
        if not np.isfinite(S) or S == 0:
            break
        G /= S
        grad = G @ x
        dot = (grad * x).sum(axis=1, keepdims=True)
        g = grad - dot * x
        m *= b1; m += (1 - b1) * g
        v *= b2; v += (1 - b2) * (g * g)
        bc1 = 1.0 - b1 ** (k + 1)
        bc2 = 1.0 - b2 ** (k + 1)
        step = (m / bc1) / (np.sqrt(v / bc2) + eps)
        np.clip(step, -2.0, 2.0, out=step)
        x = x - lr * step
        x = _norm(x)
    return x

def _find_void(x_fixed, rng, n_samples=15000):
    cand = rng.standard_normal((n_samples, 4))
    cand = _norm(cand)
    sims = cand @ x_fixed.T
    best = sims.max(axis=1)
    idx = int(best.argmin())
    b = cand[idx].copy()
    bs = float(best[idx])
    step = 0.04
    for _ in range(200):
        prop = b + step * rng.standard_normal((48, 4))
        prop = _norm(prop)
        sc = (prop @ x_fixed.T).max(axis=1)
        j = int(np.argmin(sc))
        if sc[j] < bs:
            b, bs = prop[j].copy(), float(sc[j])
            step *= 1.06
        else:
            step *= 0.96
        if step < 2e-5:
            break
    return b

def _lp_run(x, seconds, deadline, r0=8e-4):
    x = _norm(x)
    n, d = x.shape
    best = _max_cos(x)
    radius = r0
    iu = np.triu_indices(n, 1)
    tstart = time.time()
    while time.time() - tstart < seconds and time.time() < deadline and radius > 1e-13:
        cfull = (x @ x.T)[iu]
        mcur = float(cfull.max())
        thr = max(12.0 * radius, 2e-10)
        sel = cfull > mcur - thr
        if int(sel.sum()) == 0:
            break
        I, J = iu[0][sel], iu[1][sel]
        p, nv = len(I), n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-cfull[sel], A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = _norm(x + res.x[:-1].reshape(n, d))
        vv = _max_cos(y)
        if vv < best - 1e-15:
            x, best = y, vv
            radius = min(radius * 1.25, 0.02)
        elif vv < best + 1e-14:
            # neutral move, keep but shrink
            x = y
            radius *= 0.6
        else:
            radius *= 0.5
    return x, best

def solve():
    rng = np.random.default_rng(11)
    t0 = time.time()
    deadline = t0 + 172.0
    try:
        x0 = np.load("best.npy")
    except Exception:
        x0 = np.load("record.npy")
    x0 = np.asarray(x0, dtype=float)
    if x0.shape != (272, 4):
        flat = x0.ravel()
        need = 272 * 4
        if flat.size < need:
            flat = np.concatenate([flat, rng.standard_normal((need - flat.size,))])
        x0 = flat[:need].reshape(272, 4)
    x0 = np.where(np.isfinite(x0), x0, 0.0)
    x0 = _norm(x0)
    x_best = x0.copy()
    best_val = _max_cos(x_best)

    sigmas = [0.015, 0.05, 0.008, 0.03, 0.07, 0.004, 0.02, 0.10]
    taus = [(0.02, 0.0015), (0.008, 0.0006), (0.035, 0.0025), (0.012, 0.0008), (0.05, 0.003)]
    lrs = [(0.010, 0.0008), (0.005, 0.0004), (0.014, 0.0012), (0.007, 0.0005)]

    trial = 0
    # Stage 1: basin hopping, reserve ~50s for final polish
    while time.time() < t0 + 118 and time.time() < deadline - 52:
        trial += 1
        sigma = sigmas[trial % len(sigmas)]
        tau0, tau1 = taus[(trial * 3) % len(taus)]
        lr0, lr1 = lrs[(trial * 5) % len(lrs)]
        x = x_best + sigma * rng.standard_normal((272, 4))
        x = _norm(x)
        # topology change on most trials
        if trial % 2 == 0:
            nkill = 2 + (trial % 4)  # 2..5
            # contact degrees to pick victims
            G = x @ x.T
            np.fill_diagonal(G, -2.0)
            gm = float(G.max())
            # victims: endpoints of near-max pairs, biased to high degree
            degs = (G > gm - 1e-7).sum(axis=1)
            for _ in range(nkill):
                if rng.random() < 0.7:
                    # pick among highest degree
                    order = np.argsort(-degs)
                    pool = order[:10]
                    # pick victim as endpoint of current global max within pool if possible
                    G2 = x @ x.T
                    np.fill_diagonal(G2, -2.0)
                    am = int(G2.argmax())
                    ii, jj = divmod(am, 272)
                    if ii in pool or jj in pool:
                        victim = ii if (ii in pool and (rng.random() < 0.5 or jj not in pool)) else jj
                        # randomize between ii/jj if both in pool
                        if ii in pool and jj in pool:
                            victim = ii if rng.random() < 0.5 else jj
                    else:
                        victim = int(pool[rng.integers(0, len(pool))])
                else:
                    G2 = x @ x.T
                    np.fill_diagonal(G2, -2.0)
                    am = int(G2.argmax())
                    ii, jj = divmod(am, 272)
                    victim = ii if rng.random() < 0.5 else jj
                mask = np.ones(272, dtype=bool); mask[victim] = False
                vv = _find_void(x[mask], rng, n_samples=12000)
                x[victim] = vv
                # update degs roughly
                degs[victim] = 0
        elif trial % 7 == 0:
            # move a rattler (lowest personal max) to void to diversify voids
            G = x @ x.T
            np.fill_diagonal(G, -2.0)
            mx = G.max(axis=1)
            victim = int(mx.argmin())
            mask = np.ones(272, dtype=bool); mask[victim] = False
            vv = _find_void(x[mask], rng, n_samples=12000)
            x[victim] = vv
        steps = 550
        x = _anneal(x, steps, tau0, tau1, lr0, lr1)
        v = _max_cos(x)
        if v < best_val:
            best_val = v
            x_best = x.copy()
            # quick snap to lock in gain
            if time.time() < deadline - 60:
                xs, vs = _lp_run(x_best, 2.5, deadline - 40, r0=5e-4)
                if vs <= best_val:
                    x_best, best_val = xs, vs
        # opportunistic small improvement: keep near-misses for further annealing
        if v < best_val + 2e-6 and rng.random() < 0.25 and time.time() < deadline - 60:
            x2 = _anneal(x.copy(), 250, 0.004, 0.0004, 0.003, 0.0003)
            v2 = _max_cos(x2)
            if v2 < best_val:
                best_val = v2
                x_best = x2.copy()

    # Stage 2: intensify around best with low-temp anneals
    for _ in range(4):
        if time.time() > deadline - 48:
            break
        x = _anneal(x_best.copy(), 400, 0.005, 0.0005, 0.0035, 0.0003)
        v = _max_cos(x)
        if v < best_val:
            best_val = v
            x_best = x.copy()

    # Stage 3: alternating LP + micro-kick
    while time.time() < deadline - 3:
        remain = deadline - time.time() - 1.5
        if remain <= 1:
            break
        chunk = min(remain if best_val < 0.8982 else min(remain, 12.0), 14.0)
        xs, vs = _lp_run(x_best, chunk, deadline - 1.0, r0=8e-4 if xs_first if False else 8e-4)
        if vs < best_val - 1e-16:
            x_best, best_val = xs, vs
            continue
        elif vs <= best_val:
            x_best = xs
        # micro-kick to try neighbouring basin then re-snap
        if time.time() > deadline - 8:
            break
        # tiny perturbation + low-temp relax
        sig = 3e-4 if rng.random() < 0.5 else 1e-3
        xt = _norm(x_best + sig * rng.standard_normal((272, 4)))
        xt = _anneal(xt, 200, 0.003, 0.0004, 0.002, 0.0002)
        vt = _max_cos(xt)
        if vt < best_val:
            best_val = vt
            x_best = xt.copy()
        else:
            # if no progress, do one more LP chunk from kicked point if promising
            if vt < best_val + 5e-7:
                xs2, vs2 = _lp_run(xt, 2.0, deadline - 1.0, r0=3e-4)
                if vs2 < best_val:
                    best_val = vs2
                    x_best = xs2.copy()
            else:
                # shrink radius LP to converge tightly
                xs3, vs3 = _lp_run(x_best, 2.0, deadline - 1.0, r0=1e-4)
                if vs3 <= best_val:
                    x_best = xs3.copy()
                    best_val = vs3
                break

    if not np.all(np.isfinite(x_best)):
        x_best = x0.copy()
    x_best = _norm(x_best)
    return x_best
