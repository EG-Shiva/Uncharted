import time
import numpy as np

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(np.max(g))

def _per_point_max(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return np.max(g, axis=1)

def _norm(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return x / n

def _lp_polish_deadline(x, deadline):
    from scipy.optimize import linprog
    from scipy.sparse import coo_matrix
    x = _norm(x)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    iu = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        G = x @ x.T
        c = G[iu]
        m = float(c.max())
        thresh = m - max(20.0 * radius, 1e-9)
        sel = c > thresh
        p0 = int(np.count_nonzero(sel))
        if p0 == 0:
            break
        idx = np.where(sel)[0]
        if p0 > 3000:
            order = np.argpartition(c[idx], -3000)[-3000:]
            idx = idx[order]
            p0 = 3000
        I = iu[0][idx]; J = iu[1][idx]
        csel = c[idx]
        p = p0
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cI = (I[:, None] * d + np.arange(d))
        cJ = (J[:, None] * d + np.arange(d))
        cols = np.concatenate([cI, cJ, np.full((p, 1), n * d)], axis=1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], axis=1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv); obj[-1] = 1.0
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
        y = _norm(y)
        v = _max_cos(y)
        if v < best:
            x = y; best = v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x, best

def _adam_smooth(x, steps, tau0, tau1, lr0, lr1):
    n, d = x.shape
    m1 = np.zeros_like(x); v2 = np.zeros_like(x)
    lt0, lt1 = np.log(tau0), np.log(tau1)
    ll0, ll1 = np.log(lr0), np.log(lr1)
    b1, b2, eps = 0.9, 0.999, 1e-8
    for t in range(1, steps + 1):
        f = (t - 1) / max(1, steps - 1)
        tau = float(np.exp(lt0 + (lt1 - lt0) * f))
        lr = float(np.exp(ll0 + (ll1 - ll0) * f))
        G = x @ x.T
        np.fill_diagonal(G, -np.inf)
        mmax = float(np.max(G))
        W = np.exp((G - mmax) / tau)
        np.fill_diagonal(W, 0.0)
        row = np.sum(W, axis=1, keepdims=True)
        np.maximum(row, 1e-12, out=row)
        grad = (W @ x) / row
        grad = grad - np.sum(grad * x, axis=1, keepdims=True) * x
        m1 = b1 * m1 + (1 - b1) * grad
        v2 = b2 * v2 + (1 - b2) * grad * grad
        m1h = m1 / (1 - b1 ** t)
        v2h = v2 / (1 - b2 ** t)
        step = m1h / (np.sqrt(v2h) + eps)
        # clip per-row step norm to 2 for stability
        sn = np.linalg.norm(step, axis=1, keepdims=True)
        step = step * np.where(sn > 2.0, 2.0 / np.maximum(sn, 1e-12), 1.0)
        x = x - lr * step
        x = _norm(x)
    return x

def _best_single(Y, rng, S=5000, refine=200):
    # Y: (M,4) fixed others, find z minimizing max(Y@z)
    Z = rng.standard_normal(size=(S, 4))
    Z = _norm(Z)
    D = Z @ Y.T
    f = np.max(D, axis=1)
    k = min(3, S)
    idx = np.argpartition(f, k - 1)[:k]
    best_z = Z[idx[0]].copy(); best_f = float(f[idx[0]])
    for kk in idx:
        z = Z[kk].copy()
        cur_f = float(f[kk]); cur_z = z.copy()
        lr = 0.1
        for _ in range(refine):
            dots = Y @ z
            m = float(np.max(dots))
            if m < cur_f:
                cur_f = m; cur_z = z.copy()
            act = Y[dots > m - 1e-3]
            if act.shape[0] == 0:
                break
            g = np.mean(act, axis=0)
            g = g - np.dot(g, z) * z
            gn = np.linalg.norm(g)
            if gn > 1e-12:
                g = g / gn
            else:
                break
            z = z - lr * g
            z = z / np.linalg.norm(z)
            lr *= 0.995
        if cur_f < best_f:
            best_f = cur_f; best_z = cur_z.copy()
    return best_z, best_f

def solve():
    t0 = time.time()
    HARD = t0 + 172.0
    rng = np.random.default_rng(7)
    n, d = 153, 4
    best_x = None; best_v = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            if a.shape == (n, d):
                a = _norm(a)
                v = _max_cos(a)
                if v < best_v:
                    best_v = v; best_x = a.copy()
        except Exception:
            pass
    if best_x is None:
        best_x = _norm(rng.standard_normal(size=(n, d)))
        best_v = _max_cos(best_x)

    # Phase A: greedy single-point hole jumping (up to ~35s)
    deadlineA = min(t0 + 35.0, HARD - 100.0)
    # quick squeeze first if time allows
    # greedy loop
    while time.time() < deadlineA:
        m = best_v
        per = _per_point_max(best_x)
        order = np.argsort(-per)
        improved = False
        for ii in range(min(25, n)):
            if time.time() > deadlineA - 1.0:
                break
            i = int(order[ii])
            Y = np.delete(best_x, i, axis=0)
            z, f = _best_single(Y, rng, S=5000, refine=200)
            trial = best_x.copy(); trial[i] = z
            v = _max_cos(trial)
            if v < best_v - 1e-12:
                best_x = trial; best_v = v
                improved = True
                break
        if improved:
            continue
        # no single improving move: shuffle rattlers to new holes (diversify, no worsening)
        ratt = np.where(per < m - 1e-4)[0]
        if len(ratt) > 0 and time.time() < deadlineA - 2.0:
            for i in ratt[:6]:
                if time.time() > deadlineA - 0.5:
                    break
                Y = np.delete(best_x, i, axis=0)
                z, f = _best_single(Y, rng, S=6000, refine=200)
                trial = best_x.copy(); trial[i] = z
                v = _max_cos(trial)
                if v <= best_v + 1e-12:
                    # accept diversification (keep same score, new topology)
                    best_x = trial
                    # re-evaluate (same)
                    best_v = min(best_v, v)
        break

    # Phase B: basin hopping with Adam + occasional coordinate jumps
    sigmas = [0.008, 0.02, 0.05, 0.12, 0.28, 0.55, 0.03, 0.09]
    tau_cfgs = [(0.06, 0.004), (0.03, 0.002), (0.10, 0.006), (0.04, 0.0015)]
    lr_cfgs = [(0.02, 0.001), (0.03, 0.0005), (0.012, 0.0008), (0.025, 0.002)]
    it = 0
    cur_x = best_x.copy(); cur_v = best_v
    while time.time() < HARD - 32.0:
        sigma = sigmas[it % len(sigmas)]
        tc = tau_cfgs[(it // 2) % len(tau_cfgs)]
        lc = lr_cfgs[(it // 3) % len(lr_cfgs)]
        # propose start
        if it % 13 == 12:
            # full random restart
            start = _norm(rng.standard_normal(size=(n, d)))
        elif it % 5 == 4:
            # rattler-heavy shake: small noise on locked, big on rattlers
            per = _per_point_max(best_x)
            ratt = per < best_v - 1e-4
            noise = rng.standard_normal(size=(n, d)) * sigma
            # boost rattlers
            boost = np.where(ratt[:, None], 5.0, 1.0)
            start = _norm(best_x + noise * boost)
        else:
            # cluster move every 4th: rotate a random cap together
            if it % 4 == 3:
                u = rng.standard_normal(size=(4,)); u /= np.linalg.norm(u)
                proj = best_x @ u
                thr = np.quantile(proj, rng.uniform(0.5, 0.85))
                mask = proj > thr
                # random small rotation
                A = rng.standard_normal(size=(4, 4)) * 0.3 * sigma * 5.0
                A = A - A.T
                # exp approx: I + A
                R = np.eye(4) + A
                # orthogonalize via QR
                Q, _ = np.linalg.qr(R)
                start = best_x.copy()
                start[mask] = (best_x[mask] @ Q.T)
                start = _norm(start + rng.standard_normal(size=(n, d)) * sigma * 0.3)
            else:
                start = _norm(best_x + rng.standard_normal(size=(n, d)) * sigma)
        steps = 1600 if time.time() > HARD - 80 else 2200
        # shorten steps if little time
        try:
            opt = _adam_smooth(start, steps, tc[0], tc[1], lc[0], lc[1])
        except Exception:
            it += 1
            continue
        v = _max_cos(opt)
        # single-point polish if close
        if v < best_v + 0.008 and time.time() < HARD - 40:
            # try one greedy sweep on opt (worst 8)
            per2 = _per_point_max(opt)
            order2 = np.argsort(-per2)
            for ii in range(min(8, n)):
                if time.time() > HARD - 35:
                    break
                i = int(order2[ii])
                Y = np.delete(opt, i, axis=0)
                z, f = _best_single(Y, rng, S=3000, refine=120)
                trial = opt.copy(); trial[i] = z
                vt = _max_cos(trial)
                if vt < v - 1e-12:
                    opt = trial; v = vt
                    break
        if v < best_v - 1e-12:
            # promising: LP polish short to get true basin value
            dl = min(time.time() + 4.0, HARD - 30.0)
            opt2, v2 = _lp_polish_deadline(opt, dl)
            if v2 < best_v - 1e-12:
                best_x = opt2; best_v = v2
            else:
                # keep unpolished if actually better (should not happen) else keep polished as cur
                if v < best_v - 1e-12:
                    # re-polish longer next time; store opt as best anyway
                    best_x = opt2; best_v = v2
            cur_x = best_x.copy(); cur_v = best_v
        elif v < cur_v + 0.015 and it % 6 == 0 and time.time() < HARD - 50:
            # occasional LP on non-improving to explore nearby basin value
            dl = min(time.time() + 2.0, HARD - 30.0)
            opt2, v2 = _lp_polish_deadline(opt, dl)
            if v2 < best_v - 1e-12:
                best_x = opt2; best_v = v2
        it += 1
        if it > 200:
            break

    # Phase C: final trust-region LP polish
    try:
        best_x, best_v = _lp_polish_deadline(best_x, HARD - 1.0)
    except Exception:
        pass
    best_x = _norm(best_x)
    return best_x
