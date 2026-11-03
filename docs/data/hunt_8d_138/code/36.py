import time
import numpy as np
from scipy.optimize import linprog, linear_sum_assignment
from scipy.sparse import coo_matrix
try:
    from scipy.linalg import orthogonal_procrustes
    _HAS_PROC = True
except Exception:
    _HAS_PROC = False

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

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
        grad_t = grad - nd * x
        m *= b1
        m += (1.0 - b1) * grad_t
        v *= b2
        v += (1.0 - b2) * (grad_t * grad_t)
        mhat = m / (1.0 - b1 ** t)
        vhat = v / (1.0 - b2 ** t)
        upd = mhat / (np.sqrt(vhat) + eps)
        nd2 = np.sum(upd * x, axis=1, keepdims=True)
        upd = upd - nd2 * x
        x = x - lr * upd
        x /= np.linalg.norm(x, axis=1, keepdims=True) + 1e-30
    return x

def _lp_polish(x, deadline, cap=2000):
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

def _contact_key(x, val):
    n = x.shape[0]
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    thr = val - 1e-6
    per = np.sum(G > thr, axis=1)
    ncon = int(np.sum(per) // 2)
    nratt = int(np.sum(per == 0))
    # coarse clustering: contact count bin + rattler bin + mean contacts bin
    return (ncon // 40, nratt // 5)

def _crossover(a, b, t, rng, sigma):
    n, d = a.shape
    bR = b
    try:
        if _HAS_PROC:
            R, _ = orthogonal_procrustes(b, a)
            bR = b @ R
    except Exception:
        bR = b
    try:
        cost = -(a @ bR.T)
        # ensure finite
        cost = np.clip(cost, -2.0, 2.0)
        ri, ci = linear_sum_assignment(cost)
        bM = np.empty_like(bR)
        bM[ri] = bR[ci]
    except Exception:
        bM = bR
    dots = np.sum(a * bM, axis=1)
    dots = np.clip(dots, -1.0, 1.0)
    om = np.arccos(dots)
    child = np.empty_like(a)
    for i in range(n):
        o = om[i]
        if o < 1e-9:
            child[i] = (1.0 - t) * a[i] + t * bM[i]
        else:
            s = np.sin(o)
            c1 = np.sin((1.0 - t) * o) / s
            c2 = np.sin(t * o) / s
            child[i] = c1 * a[i] + c2 * bM[i]
    if sigma > 0:
        # tangent noise
        nz = rng.normal(size=(n, d))
        # project to tangent
        nd = np.sum(nz * child, axis=1, keepdims=True)
        nz = nz - nd * child
        child = child + sigma * nz
    child /= np.linalg.norm(child, axis=1, keepdims=True) + 1e-30
    return child

def solve():
    rng = np.random.default_rng(0)
    t0 = time.time()
    deadline = t0 + 168.0
    n, d = 138, 8
    P = 10
    # load seeds
    seeds = []
    for fn in ("best.npy", "record.npy"):
        try:
            vv = np.load(fn).astype(float)
            if vv.shape == (n, d):
                vv /= np.linalg.norm(vv, axis=1, keepdims=True)
                seeds.append(vv)
        except Exception:
            pass
    if len(seeds) == 0:
        v = rng.normal(size=(n, d)); v /= np.linalg.norm(v, axis=1, keepdims=True)
        seeds.append(v)
    base = seeds[0]
    base_val = _max_cos(base)
    # beta ladder (smoothed objective sharpness) and temperature ladder
    betas = [25.0, 45.0, 70.0, 100.0, 135.0, 170.0, 215.0, 260.0, 310.0, 360.0]
    Temps = [0.00025, 0.0004, 0.0006, 0.0009, 0.0013, 0.0018, 0.0024, 0.0032, 0.0042, 0.0055]
    pop = []
    pop_val = []
    pop_key = []
    # build diverse initial population
    sigmas = [0.0, 0.02, 0.05, 0.08, 0.12, 0.18, 0.25, 0.35, 0.0, 0.5]
    # second seed if available
    for k in range(P):
        if k == 0:
            x = base.copy()
        elif k == 1 and len(seeds) > 1:
            x = seeds[1].copy()
        elif k == P - 1:
            # fresh random basin
            x = rng.normal(size=(n, d)); x /= np.linalg.norm(x, axis=1, keepdims=True)
            x = _adam_smooth(x, steps=250, beta0=8.0, beta1=180.0, lr=0.007)
            v = _max_cos(x)
            pop.append(x); pop_val.append(v); pop_key.append(_contact_key(x, v))
            continue
        else:
            s = sigmas[k % len(sigmas)]
            x = base + s * rng.normal(size=(n, d))
            x /= np.linalg.norm(x, axis=1, keepdims=True)
        # light anneal with its beta
        b = betas[k % len(betas)]
        try:
            x = _adam_smooth(x, steps=220, beta0=max(8.0, b * 0.15), beta1=b, lr=0.006)
        except Exception:
            pass
        v = _max_cos(x)
        pop.append(x); pop_val.append(v); pop_key.append(_contact_key(x, v))
        if time.time() > deadline - 120:
            break
    # light LP on top 3 to get polished members quickly
    order = np.argsort(pop_val)
    for r in range(min(3, len(pop))):
        idx = int(order[r])
        if time.time() > deadline - 110:
            break
        try:
            pop[idx] = _lp_polish(pop[idx], min(deadline - 90, time.time() + 1.2), cap=1800)
            pop_val[idx] = _max_cos(pop[idx])
            pop_key[idx] = _contact_key(pop[idx], pop_val[idx])
        except Exception:
            pass
    best_idx = int(np.argmin(pop_val))
    gbest = pop[best_idx].copy()
    gbest_val = float(pop_val[best_idx])

    it = 0
    t_choices = [0.35, 0.5, 0.65]
    while time.time() < deadline - 7.0:
        it += 1
        # --- parallel-tempering single-replica mutation (round robin) ---
        if it % 2 == 1:
            k = int(rng.integers(0, len(pop)))
            T = Temps[k % len(Temps)]
            b = betas[k % len(betas)]
            x = pop[k].copy()
            # kick size depends on temperature rank
            sigma = 0.02 + 0.20 * (k / max(len(pop) - 1, 1))
            if rng.random() < 0.25:
                sigma = 0.35
            x = x + sigma * rng.normal(size=(n, d))
            x /= np.linalg.norm(x, axis=1, keepdims=True)
            b0 = max(8.0, b * 0.2)
            try:
                x = _adam_smooth(x, steps=320, beta0=b0, beta1=b * 1.15, lr=0.006)
            except Exception:
                pass
            v = _max_cos(x)
            if v < gbest_val + 0.006 and time.time() < deadline - 8.0:
                try:
                    x = _lp_polish(x, min(deadline - 7.0, time.time() + 0.9), cap=1600)
                    v = _max_cos(x)
                except Exception:
                    pass
            # Metropolis acceptance into same slot (allows worse at high T)
            old = pop_val[k]
            if v < old or rng.random() < np.exp(-(v - old) / max(T, 1e-9)):
                pop[k] = x; pop_val[k] = v; pop_key[k] = _contact_key(x, v)
            if v < gbest_val - 1e-12:
                gbest_val = v; gbest = x.copy()
        else:
            # --- crossover offspring via Procrustes + Hungarian + slerp ---
            m = len(pop)
            i1 = int(rng.integers(0, m)); i2 = int(rng.integers(0, m))
            if i2 == i1:
                i2 = (i2 + 1) % m
            a = pop[i1]; bb = pop[i2]
            t = t_choices[it % len(t_choices)]
            sigma = float(rng.choice([0.015, 0.03, 0.05]))
            try:
                child = _crossover(a, bb, t, rng, sigma)
            except Exception:
                child = (a + bb); child /= np.linalg.norm(child, axis=1, keepdims=True)
            # anneal child with random beta from ladder
            bc = float(rng.choice(betas))
            try:
                child = _adam_smooth(child, steps=320, beta0=max(8.0, bc * 0.2), beta1=bc * 1.2, lr=0.006)
            except Exception:
                pass
            v = _max_cos(child)
            if v < gbest_val + 0.008 and time.time() < deadline - 8.0:
                try:
                    child = _lp_polish(child, min(deadline - 7.0, time.time() + 1.0), cap=1800)
                    v = _max_cos(child)
                except Exception:
                    pass
            key = _contact_key(child, v)
            # cluster-preserving insertion: keep best per signature cluster
            same = [q for q in range(len(pop)) if pop_key[q] == key]
            if len(same) > 0:
                # replace best-in-cluster only if improved
                bv = min(pop_val[q] for q in same)
                if v < bv - 1e-12:
                    # replace worst in cluster to keep size
                    w = max(same, key=lambda q: pop_val[q])
                    pop[w] = child; pop_val[w] = v; pop_key[w] = key
                # else discard (preserve existing basin)
            else:
                if len(pop) < 12:
                    pop.append(child); pop_val.append(v); pop_key.append(key)
                else:
                    worst = int(np.argmax(pop_val))
                    # allow slightly worse if topologically new
                    if v < pop_val[worst] + 0.004:
                        # replace worst, prefer replacing from largest cluster
                        # find largest cluster
                        from collections import Counter
                        cc = Counter(pop_key)
                        big, _ = cc.most_common(1)[0]
                        cand = [q for q in range(len(pop)) if pop_key[q] == big]
                        w2 = max(cand, key=lambda q: pop_val[q])
                        # don't kill global best slot
                        gb_slot = int(np.argmin(pop_val))
                        if w2 == gb_slot:
                            w2 = worst
                        pop[w2] = child; pop_val[w2] = v; pop_key[w2] = key
            if v < gbest_val - 1e-12:
                gbest_val = v; gbest = child.copy()
        # occasional Metropolis swap between replicas
        if it % 5 == 0 and len(pop) >= 2:
            i1 = int(rng.integers(0, len(pop))); i2 = int(rng.integers(0, len(pop)))
            if i1 != i2:
                # swap configs (beta stays with slot) with tempered prob
                T1 = Temps[i1 % len(Temps)]; T2 = Temps[i2 % len(Temps)]
                v1 = pop_val[i1]; v2 = pop_val[i2]
                # standard PT swap prob
                try:
                    logp = (v1 - v2) * (1.0 / max(T1, 1e-9) - 1.0 / max(T2, 1e-9))
                    # values are costs; swap if beneficial
                    if logp >= 0 or rng.random() < np.exp(logp):
                        pop[i1], pop[i2] = pop[i2].copy(), pop[i1].copy()
                        pop_val[i1], pop_val[i2] = pop_val[i2], pop_val[i1]
                        pop_key[i1], pop_key[i2] = pop_key[i2], pop_key[i1]
                    elif rng.random() < 0.05:
                        # random drift to avoid freeze
                        pop[i1], pop[i2] = pop[i2].copy(), pop[i1].copy()
                        pop_val[i1], pop_val[i2] = pop_val[i2], pop_val[i1]
                        pop_key[i1], pop_key[i2] = pop_key[i2], pop_key[i1]
                except Exception:
                    pass
        if it % 10 == 0:
            # refresh global best slot protection
            bi = int(np.argmin(pop_val))
            if pop_val[bi] < gbest_val - 1e-12:
                gbest_val = float(pop_val[bi]); gbest = pop[bi].copy()
            if gbest_val < pop_val[bi] - 1e-12:
                # inject global best into worst slot (elitism) without killing diversity too much
                w = int(np.argmax(pop_val))
                if w != bi:
                    pop[w] = gbest.copy(); pop_val[w] = gbest_val; pop_key[w] = _contact_key(gbest, gbest_val)
    # final tight polish of incumbent
    try:
        gbest = _adam_smooth(gbest, steps=200, beta0=80.0, beta1=400.0, lr=0.003)
        gbest = _lp_polish(gbest, min(deadline, time.time() + 6.0), cap=2600)
    except Exception:
        pass
    gbest /= np.linalg.norm(gbest, axis=1, keepdims=True) + 1e-30
    return gbest
