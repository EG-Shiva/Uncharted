import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def proj_unit(x):
    nrm = np.linalg.norm(x, axis=1, keepdims=True)
    nrm[nrm == 0] = 1.0
    x = x / nrm
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    nrm2 = np.linalg.norm(x, axis=1, keepdims=True)
    nrm2[nrm2 == 0] = 1.0
    return x / nrm2

def lp_polish_deadline(x, deadline, radius_init=1e-3):
    x = proj_unit(x)
    n, d = x.shape
    best = max_cos(x)
    radius = radius_init
    iu = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        G = x @ x.T
        c = G[iu]
        m = float(c.max())
        sel = c > m - max(20.0 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        if p == 0:
            radius *= 0.5
            continue
        csv = c[sel]
        if p > 2500:
            idx = np.argpartition(-csv, 2500)[:2500]
            I, J = I[idx], J[idx]
            csv = csv[idx]
            p = 2500
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        colsA = I[:, None] * d + np.arange(d)
        colsB = J[:, None] * d + np.arange(d)
        cols = np.concatenate([colsA, colsB, np.full((p, 1), n * d)], axis=1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], axis=1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv)
            obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-csv, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                          method="highs", options={"time_limit": 4.0, "presolve": True})
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y = proj_unit(y)
        v = max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
        if time.time() >= deadline:
            break
    return x, best

def smooth_repel(x, steps, lr, T0=0.02, T1=0.004):
    y = x.copy()
    n, d = y.shape
    for s in range(steps):
        G = y @ y.T
        np.fill_diagonal(G, -10.0)
        m = float(G.max())
        T = T0 + (T1 - T0) * (s / max(1, steps - 1))
        W = np.exp((G - m) / max(T, 1e-6))
        np.fill_diagonal(W, 0.0)
        grad = W @ y
        grad -= (grad * y).sum(axis=1, keepdims=True) * y
        gn = np.linalg.norm(grad, axis=1, keepdims=True) + 1e-12
        grad /= gn
        step = lr * (1.0 - 0.5 * s / steps)
        y = y - step * grad
        y = proj_unit(y)
    return y

def contact_info(x, tol=1e-9):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    m = float(G.max())
    C = G > m - tol
    cnts = C.sum(axis=1)
    worst = G.max(axis=1)
    return cnts, worst, m, G

def tangent_rand(rng, p):
    n, d = p.shape
    v = rng.normal(size=(n, d))
    v -= (v * p).sum(axis=1, keepdims=True) * p
    nv = np.linalg.norm(v, axis=1, keepdims=True) + 1e-12
    return v / nv

def perturb_cluster(x, rng, strength):
    n, d = x.shape
    cnts, worst, m, G = contact_info(x)
    # pick hub among most contacted / worst
    score = cnts.astype(float) + 0.5 * (worst - worst.min()) / (worst.max() - worst.min() + 1e-12)
    top = np.argsort(-score)[:6]
    h = int(top[rng.integers(0, len(top))])
    # neighbours by Gram
    order = np.argsort(-G[h])
    k = int(rng.integers(4, 9))
    cluster = order[1:k + 1].tolist()
    cluster.append(h)
    z = rng.normal(size=d)
    z /= (np.linalg.norm(z) + 1e-12)
    y = x.copy()
    # coherent shift along z (tangential part per point)
    for i in cluster:
        t = z - (z @ y[i]) * y[i]
        nt = np.linalg.norm(t)
        if nt < 1e-12:
            continue
        t /= nt
        ang = strength * (0.7 + 0.6 * rng.random())
        y[i] = np.cos(ang) * y[i] + np.sin(ang) * t
    # individual tangent jitter on cluster + tiny global
    jr = tangent_rand(rng, y[cluster]) * (strength * 0.5 * (0.5 + rng.random(size=(len(cluster), 1))))
    y[cluster] += jr
    glob = tangent_rand(rng, y) * (strength * 0.12)
    y = y + glob
    return proj_unit(y)

def find_void(x, rng, n_cand=60000, refine=150):
    n, d = x.shape
    # block sampling for memory
    best_v = None
    best_m = 1e9
    bs = 10000
    for s in range(0, n_cand, bs):
        b = min(bs, n_cand - s)
        C = rng.normal(size=(b, d))
        C = proj_unit(C)
        G = C @ x.T
        mv = G.max(axis=1)
        i = int(np.argmin(mv))
        if float(mv[i]) < best_m:
            best_m = float(mv[i])
            best_v = C[i].copy()
    v = best_v.copy()
    # refine: minimise soft-max to hole
    for s in range(refine):
        g = x @ v  # (n,)
        m = float(g.max())
        T = max(0.003, 0.02 * (1 - s / refine))
        w = np.exp((g - m) / T)
        w /= (w.sum() + 1e-12)
        grad = w @ x  # (d,)
        grad -= (grad @ v) * v
        gn = np.linalg.norm(grad) + 1e-12
        step = 0.05 * (1 - 0.5 * s / refine)
        v = v - step * grad / gn
        v /= (np.linalg.norm(v) + 1e-12)
    return v

def perturb_void_replace(x, rng):
    n, d = x.shape
    cnts, worst, m, G = contact_info(x)
    # worst offender: max worst, tie by contacts
    order = np.argsort(-(worst + 1e-6 * cnts))
    # usually replace 1, sometimes 2
    nv_rep = 1 if rng.random() < 0.7 else 2
    targets = order[:nv_rep].tolist()
    # occasionally also replace random high-contact
    if rng.random() < 0.25:
        targets[0] = int(np.argsort(-cnts)[rng.integers(0, 8)])
    y = x.copy()
    for t in targets:
        v = find_void(y, rng, n_cand=30000 if nv_rep == 1 else 20000, refine=120)
        # small jitter so not exactly at void (avoid degeneracy)
        jr = rng.normal(size=d)
        jr -= (jr @ v) * v
        jr /= (np.linalg.norm(jr) + 1e-12)
        v = v + jr * 0.005
        v /= np.linalg.norm(v)
        y[t] = v
    # light global jitter
    y = y + tangent_rand(rng, y) * 0.005
    return proj_unit(y)

def perturb_shake(x, rng, strength, nk):
    n, d = x.shape
    cnts, worst, m, G = contact_info(x)
    order_low = np.argsort(cnts)
    order_high = np.argsort(-cnts)
    targets = set()
    targets.update(order_low[:2].tolist())
    targets.update(order_high[:2].tolist())
    while len(targets) < nk:
        targets.add(int(rng.integers(0, n)))
    targets = list(targets)
    y = x.copy()
    for i in targets:
        v = rng.normal(size=d)
        v -= (v @ y[i]) * y[i]
        nv = np.linalg.norm(v)
        if nv < 1e-12:
            continue
        v /= nv
        ang = strength * (0.5 + rng.random())
        y[i] = np.cos(ang) * y[i] + np.sin(ang) * v
    y = y + tangent_rand(rng, y) * (strength * 0.2)
    return proj_unit(y)

def solve():
    T0 = time.time()
    LIMIT = 165.0
    deadline_final = T0 + LIMIT
    rng = np.random.default_rng(7)
    try:
        x0 = np.load("best.npy")
    except Exception:
        try:
            x0 = np.load("record.npy")
        except Exception:
            x0 = rng.normal(size=(54, 21))
    x0 = proj_unit(np.asarray(x0, dtype=float).reshape(54, 21))
    best = x0.copy()
    best_val = max_cos(best)
    # quick baseline polish
    if time.time() < deadline_final - 60:
        dl = min(time.time() + 8.0, deadline_final - 50)
        best, best_val = lp_polish_deadline(best, dl, radius_init=1e-3)
    # one smooth from random to get diversity seed (cheap)
    it = 0
    strengths = [0.03, 0.08, 0.16, 0.30, 0.05, 0.12, 0.25, 0.04]
    while time.time() < deadline_final - 22:
        it += 1
        mode = it % 5
        if mode in (0, 1):
            cand = perturb_cluster(best, rng, strength=float(strengths[(it - 1) % len(strengths)]))
            cand = smooth_repel(cand, steps=70, lr=0.02)
        elif mode == 2:
            cand = perturb_void_replace(best, rng)
            cand = smooth_repel(cand, steps=90, lr=0.025)
        elif mode == 3:
            s = float(strengths[(it - 1) % len(strengths)])
            nk = int(rng.integers(3, 12))
            cand = perturb_shake(best, rng, strength=s, nk=nk)
            cand = smooth_repel(cand, steps=80, lr=0.02)
        else:
            # occasional fresh random + anneal (escape far)
            R = proj_unit(rng.normal(size=(54, 21)))
            R = smooth_repel(R, steps=180, lr=0.05, T0=0.05, T1=0.003)
            # mix with best: replace worst 8 by random annealed points' voids? just use R
            cand = R
        budget = min(5.0, (deadline_final - 22 - time.time()) / 2.0)
        if budget < 0.8:
            break
        dl = time.time() + budget
        cand, v = lp_polish_deadline(cand, dl)
        if v < best_val - 1e-12:
            best, best_val = cand, v
        # every few iters try extra intensify around new best with tiny shake
        if it % 7 == 0 and time.time() < deadline_final - 25:
            cand2 = perturb_shake(best, rng, strength=0.02, nk=4)
            cand2 = smooth_repel(cand2, steps=40, lr=0.01)
            dl2 = min(time.time() + 3.0, deadline_final - 22)
            if dl2 - time.time() > 0.8:
                cand2, v2 = lp_polish_deadline(cand2, dl2)
                if v2 < best_val - 1e-12:
                    best, best_val = cand2, v2
    if time.time() < deadline_final - 0.5:
        best, best_val = lp_polish_deadline(best, deadline_final - 0.3, radius_init=5e-4)
    best = proj_unit(best)
    # tiny shrink guard (keep strictly feasible, no-op for cosine but avoids edge)
    return best
