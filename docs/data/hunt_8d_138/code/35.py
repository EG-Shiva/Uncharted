import time
import numpy as np
from scipy.optimize import linprog, linear_sum_assignment
from scipy.sparse import coo_matrix
from scipy.linalg import orthogonal_procrustes


def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())


def _contact_sig(x, tol=1e-6):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    mv = float(g.max())
    mask = g > mv - tol
    per = mask.sum(axis=1)
    ncon = int(mask.sum() // 2)
    s = tuple(sorted(per.tolist()))
    return mv, ncon, s


def _cluster_key(mv, ncon):
    return (round(mv, 3), int(ncon // 25))


def _adam_smooth(x, steps, beta0, beta1, lr):
    n, d = x.shape
    m = np.zeros_like(x)
    v = np.zeros_like(x)
    b1, b2, eps = 0.9, 0.999, 1e-8
    for t in range(1, steps + 1):
        f = (t - 1) / max(steps - 1, 1)
        beta = beta0 + (beta1 - beta0) * f
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        mv = G.max()
        W = np.exp(beta * (G - mv))
        np.fill_diagonal(W, 0.0)
        grad = W @ x
        nd = np.sum(grad * x, axis=1, keepdims=True)
        grad = grad - nd * x
        m *= b1
        m += (1.0 - b1) * grad
        v *= b2
        v += (1.0 - b2) * (grad * grad)
        bc1 = 1.0 - b1 ** t
        bc2 = 1.0 - b2 ** t
        upd = (m / bc1) / (np.sqrt(v / bc2) + eps)
        nd2 = np.sum(upd * x, axis=1, keepdims=True)
        upd = upd - nd2 * x
        x = x - lr * upd
        x /= np.linalg.norm(x, axis=1, keepdims=True) + 1e-30
    return x


def _lp_polish(x, deadline, cap=2500):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    iu = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        Gfull = x @ x.T
        c = Gfull[iu]
        mv = c.max()
        thr = mv - max(20.0 * radius, 1e-9)
        sel = c > thr
        cnt = int(np.count_nonzero(sel))
        if cnt == 0:
            break
        if cnt > cap:
            idx = np.where(sel)[0]
            order = np.argsort(-c[idx])[:cap]
            mask = np.zeros_like(sel, dtype=bool)
            mask[idx[order]] = True
            sel = mask
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
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
        vv = _max_cos(y)
        if vv < best:
            x, best = y, vv
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x


def _crossover(A, B, rng, t, sigma):
    n, d = A.shape
    try:
        R, _ = orthogonal_procrustes(B, A)
    except Exception:
        R = np.eye(d)
    Br = B @ R
    C = A @ Br.T
    try:
        _, col = linear_sum_assignment(-C)
    except Exception:
        col = np.arange(n)
    Bp = Br[col]
    dots = np.sum(A * Bp, axis=1)
    dots = np.clip(dots, -1.0, 1.0)
    w = np.arccos(dots)
    sw = np.sin(w)
    child = np.empty_like(A)
    for i in range(n):
        if sw[i] < 1e-12 or w[i] < 1e-12:
            child[i] = (1.0 - t) * A[i] + t * Bp[i]
        else:
            a = np.sin((1.0 - t) * w[i]) / sw[i]
            b = np.sin(t * w[i]) / sw[i]
            child[i] = a * A[i] + b * Bp[i]
    child /= np.linalg.norm(child, axis=1, keepdims=True) + 1e-30
    if sigma > 0:
        nz = rng.normal(size=(n, d))
        nz -= np.sum(nz * child, axis=1, keepdims=True) * child
        child = child + sigma * nz
        child /= np.linalg.norm(child, axis=1, keepdims=True) + 1e-30
    return child


def solve():
    rng = np.random.default_rng(0)
    t0 = time.time()
    deadline = t0 + 168.0
    n, d = 138, 8
    P = 9
    base = None
    base_val = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            vv = np.load(fn).astype(float)
            if vv.shape == (n, d):
                vv /= np.linalg.norm(vv, axis=1, keepdims=True)
                c = _max_cos(vv)
                if c < base_val:
                    base_val = c
                    base = vv.copy()
        except Exception:
            pass
    if base is None:
        base = rng.normal(size=(n, d))
        base /= np.linalg.norm(base, axis=1, keepdims=True)
        base_val = _max_cos(base)
    # smooth betas (sharpness) and Metropolis temps per replica
    betas = np.linspace(30.0, 260.0, P)
    temps = np.array([0.004, 0.003, 0.0022, 0.0015, 0.0010, 0.0007, 0.0004, 0.00025, 0.00015])
    sigmas_init = [0.0, 0.03, 0.06, 0.10, 0.15, 0.22, 0.35, 0.5, 0.7]
    pop = []
    vals = []
    # replica 0 is the incumbent untouched
    pop.append(base.copy())
    vals.append(base_val)
    for k in range(1, P):
        y = base.copy()
        s = sigmas_init[k]
        if k >= 6 and (k % 2 == 0):
            # fresh random basin for high-temp replicas
            y = rng.normal(size=(n, d))
            y /= np.linalg.norm(y, axis=1, keepdims=True)
        else:
            y = y + s * rng.normal(size=(n, d))
            y /= np.linalg.norm(y, axis=1, keepdims=True)
        # short anneal to make distinct polished configs
        try:
            y = _adam_smooth(y, steps=150, beta0=10.0, beta1=float(betas[k]), lr=0.006)
        except Exception:
            pass
        pop.append(y)
        vals.append(_max_cos(y))
    pop = list(pop)
    vals = list(vals)
    best_idx = int(np.argmin(vals))
    gbest = pop[best_idx].copy()
    gbest_val = float(vals[best_idx])
    keys = []
    for i in range(P):
        try:
            _, ncon, _ = _contact_sig(pop[i])
            keys.append(_cluster_key(vals[i], ncon))
        except Exception:
            keys.append((round(vals[i], 3), 0))
    it = 0
    t_choices = [0.35, 0.5, 0.65]
    while time.time() < deadline - 9.0 and it < 1000:
        it += 1
        # 1) evolve one replica (round robin) under its smoothed objective
        idx = it % P
        T = float(temps[idx])
        btar = float(betas[idx])
        x = pop[idx].copy()
        # kick size scales with temperature
        kick = 0.02 + min(0.12, T * 22.0)
        if it % 17 == 0:
            kick = 0.25
        y = x + kick * rng.normal(size=(n, d))
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        steps = 220 if (it % 3 != 0) else 350
        try:
            y = _adam_smooth(y, steps=steps, beta0=max(8.0, btar * 0.15), beta1=btar, lr=0.006)
        except Exception:
            pass
        vnew = _max_cos(y)
        vold = float(vals[idx])
        accept = False
        if vnew < vold:
            accept = True
        else:
            try:
                if rng.random() < np.exp(-(vnew - vold) / max(T, 1e-9)):
                    accept = True
            except Exception:
                accept = False
        if accept:
            pop[idx] = y
            vals[idx] = vnew
            try:
                _, ncon, _ = _contact_sig(y)
                keys[idx] = _cluster_key(vnew, ncon)
            except Exception:
                keys[idx] = (round(vnew, 3), 0)
            if vnew < gbest_val - 1e-12:
                gbest_val = vnew
                gbest = y.copy()
        # 2) occasional Metropolis swap of two replicas
        if it % 3 == 0 and time.time() < deadline - 9.5:
            a = int(rng.integers(0, P))
            b = int(rng.integers(0, P))
            if a != b:
                # standard PT swap probability
                try:
                    dE = vals[a] - vals[b]
                    dB = 1.0 / max(temps[a], 1e-9) - 1.0 / max(temps[b], 1e-9)
                    # energies scaled: use max-cos directly
                    prob = np.exp(min(0.0, -dE * dB * 0.0005))
                except Exception:
                    prob = 0.15
                if rng.random() < min(1.0, max(0.05, prob)):
                    pop[a], pop[b] = pop[b], pop[a]
                    vals[a], vals[b] = vals[b], vals[a]
                    keys[a], keys[b] = keys[b], keys[a]
        # 3) recombination every few iters between different clusters
        if it % 4 == 0 and time.time() < deadline - 9.5:
            # pick two parents with different keys if possible
            order = rng.permutation(P)
            pi = int(order[0])
            pj = int(order[1])
            for _ in range(4):
                if keys[pi] != keys[pj]:
                    break
                pj = int(rng.integers(0, P))
            A = pop[pi]
            B = pop[pj]
            # alternate direction randomly
            if rng.random() < 0.5:
                A, B = B, A
            t = t_choices[it % 3]
            try:
                child = _crossover(A, B, rng, t, sigma=0.012)
            except Exception:
                child = (A + B) / 2.0
                child /= np.linalg.norm(child, axis=1, keepdims=True) + 1e-30
            # anneal child at medium beta
            bmix = float((betas[pi] + betas[pj]) * 0.5)
            try:
                child = _adam_smooth(child, steps=250, beta0=12.0, beta1=bmix, lr=0.006)
            except Exception:
                pass
            vc = _max_cos(child)
            # LP polish if promising
            if vc < gbest_val + 0.006 and time.time() < deadline - 10.0:
                lp_dead = min(deadline - 9.0, time.time() + 0.9)
                if lp_dead > time.time() + 0.15:
                    try:
                        child = _lp_polish(child, lp_dead, cap=2200)
                        vc = _max_cos(child)
                    except Exception:
                        pass
            try:
                _, ncon_c, _ = _contact_sig(child)
                kc = _cluster_key(vc, ncon_c)
            except Exception:
                kc = (round(vc, 3), 0)
            # insertion: best member of each signature cluster survives
            worst = int(np.argmax(vals))
            if kc in keys:
                # replace worst member of same cluster if better
                same = [i for i in range(P) if keys[i] == kc]
                worst_same = max(same, key=lambda i: vals[i])
                if vc < vals[worst_same]:
                    pop[worst_same] = child
                    vals[worst_same] = vc
                    keys[worst_same] = kc
                elif vc < vals[worst] and rng.random() < 0.3:
                    pop[worst] = child
                    vals[worst] = vc
                    keys[worst] = kc
            else:
                # novel topology: keep even if slightly worse (diversity)
                if vc < vals[worst] + 0.002:
                    if vc < vals[worst] or rng.random() < 0.5:
                        pop[worst] = child
                        vals[worst] = vc
                        keys[worst] = kc
            if vc < gbest_val - 1e-12:
                gbest_val = vc
                gbest = child.copy()
        # 4) light LP on global best occasionally
        if it % 20 == 0 and time.time() < deadline - 10.0:
            if gbest_val < min(vals) + 1e-12:
                lp_dead = min(deadline - 9.0, time.time() + 0.7)
                if lp_dead > time.time() + 0.15:
                    try:
                        nb = _lp_polish(gbest, lp_dead, cap=2200)
                        nv = _max_cos(nb)
                        if nv < gbest_val:
                            gbest_val = nv
                            gbest = nb
                            # inject into coldest replica
                            ci = int(np.argmin(temps))
                            pop[ci] = nb.copy()
                            vals[ci] = nv
                            _, ncon, _ = _contact_sig(nb)
                            keys[ci] = _cluster_key(nv, ncon)
                    except Exception:
                        pass
    # final tight polish of incumbent
    try:
        gbest = _adam_smooth(gbest, steps=250, beta0=60.0, beta1=380.0, lr=0.003)
        fin_dead = min(deadline, time.time() + 7.0)
        if fin_dead > time.time() + 0.3:
            gbest = _lp_polish(gbest, fin_dead, cap=3000)
    except Exception:
        pass
    gbest /= np.linalg.norm(gbest, axis=1, keepdims=True) + 1e-30
    return gbest
