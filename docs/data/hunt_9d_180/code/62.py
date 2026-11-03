import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(np.max(g))

def per_max(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return np.max(g, axis=1), float(np.max(g))

def lp_polish(x, seconds, deadline=None):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = max_cos(x)
    bx = x.copy()
    radius = 1e-3
    if deadline is None:
        deadline = time.time() + seconds
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and time.time() < deadline and radius > 1e-14:
        Gfull = x @ x.T
        iu_c = Gfull[iu]
        m = float(np.max(iu_c))
        sel = iu_c > m - max(20 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        csub = iu_c[sel]
        p = len(I)
        if p == 0:
            break
        if p > 3000:
            # keep tightest 3000 to keep LP fast
            idx = np.argpartition(-csub, 3000)[:3000]
            I, J, csub = I[idx], J[idx], csub[idx]
            p = 3000
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        except Exception:
            break
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-csub, A_eq=A_eq, b_eq=np.zeros(n),
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
            x, best, bx = y, v, y.copy()
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
        if p > 1500 and radius < 2e-4:
            break
    return bx

def find_void(others, rng, n_try=6000, n_refine=250):
    d = others.shape[1]
    cand = rng.randn(n_try, d)
    cand /= np.linalg.norm(cand, axis=1, keepdims=True)
    dots = cand @ others.T
    m = np.max(dots, axis=1)
    bi = int(np.argmin(m))
    y = cand[bi].copy()
    cur = float(m[bi])
    for k in range(n_refine):
        sig = 0.08 * (0.001 / 0.08) ** (k / max(1, n_refine - 1))
        prop = y + sig * rng.randn(d)
        prop /= np.linalg.norm(prop)
        pm = float(np.max(prop @ others.T))
        if pm < cur:
            y = prop
            cur = pm
    return y, cur

def subset_adam(x, active, frozen_max, steps=400, lr=0.02, T0=0.01, T1=0.001, rng=None):
    # tangential softmax Adam on active indices only
    n, d = x.shape
    k = len(active)
    amap = np.zeros(n, dtype=int) - 1
    amap[active] = np.arange(k)
    m1 = np.zeros((k, d))
    m2 = np.zeros((k, d))
    b1, b2, eps = 0.9, 0.999, 1e-8
    best_x = x.copy()
    # current global best estimate
    Gb = x @ x.T
    np.fill_diagonal(Gb, -2.0)
    best_val = float(np.max(Gb))
    cur = x.copy()
    t = 0
    for s in range(steps):
        t += 1
        frac = s / max(1, steps - 1)
        T = T0 * (T1 / T0) ** frac
        Xa = cur[active]  # k x d
        G = Xa @ cur.T  # k x n
        # zero self
        G[np.arange(k), active] = -2.0
        gmax_a = float(np.max(G)) if G.size else -2.0
        gmax = gmax_a if gmax_a > frozen_max else frozen_max
        E = np.exp((G - gmax) / T)
        # self already -2 -> tiny, explicitly zero
        E[np.arange(k), active] = 0.0
        row = np.sum(E, axis=1, keepdims=True) + 1e-12
        grad = E @ cur  # k x d
        dot = np.sum(grad * Xa, axis=1, keepdims=True)
        gtan = grad - dot * Xa
        gtan = gtan / row
        # clip extreme
        nm = np.linalg.norm(gtan, axis=1, keepdims=True) + 1e-12
        gtan = gtan * np.minimum(1.0, 1.0 / nm)
        m1 = b1 * m1 + (1 - b1) * gtan
        m2 = b2 * m2 + (1 - b2) * (gtan * gtan)
        m1h = m1 / (1 - b1 ** t)
        m2h = m2 / (1 - b2 ** t)
        upd = m1h / (np.sqrt(m2h) + eps)
        # project update to tangent
        ud = np.sum(upd * Xa, axis=1, keepdims=True)
        upd_t = upd - ud * Xa
        # clip update norm
        un = np.linalg.norm(upd_t, axis=1, keepdims=True) + 1e-12
        upd_t = upd_t * np.minimum(1.0, 0.5 / un)
        Xa2 = Xa - lr * upd_t
        Xa2 /= np.linalg.norm(Xa2, axis=1, keepdims=True)
        cur[active] = Xa2
        if (s & 31) == 31 or s == steps - 1:
            Gb2 = cur @ cur.T
            np.fill_diagonal(Gb2, -2.0)
            v = float(np.max(Gb2))
            if v < best_val:
                best_val = v
                best_x = cur.copy()
    return best_x, best_val

def solve():
    t0 = time.time()
    deadline = t0 + 178.0
    cycle_end = t0 + 145.0
    rng = np.random.RandomState(0)
    try:
        xb = np.load("best.npy")
    except Exception:
        xb = None
    try:
        xr = np.load("record.npy")
    except Exception:
        xr = None
    cands = []
    for c in (xb, xr):
        try:
            if c is None:
                continue
            c = np.asarray(c, dtype=float)
            if c.shape == (180, 9):
                c = c / np.linalg.norm(c, axis=1, keepdims=True)
                cands.append((max_cos(c), c))
        except Exception:
            pass
    if len(cands) == 0:
        x0 = rng.randn(180, 9)
        x0 /= np.linalg.norm(x0, axis=1, keepdims=True)
        x_best = x0
        b_best = max_cos(x0)
    else:
        cands.sort(key=lambda t: t[0])
        b_best, x_best = cands[0][0], cands[0][1].copy()
    x_best = x_best / np.linalg.norm(x_best, axis=1, keepdims=True)
    b_best = max_cos(x_best)
    n, d = x_best.shape
    # ---- build graph > 0.35 and clusters 4-18 ----
    try:
        G0 = x_best @ x_best.T
        np.fill_diagonal(G0, -10.0)
        adj = G0 > 0.35
        neigh = [np.where(adj[i])[0] for i in range(n)]
        # connected components
        seen = np.zeros(n, dtype=bool)
        pool = []
        for i in range(n):
            if seen[i]:
                continue
            stack = [i]; seen[i] = True; comp = []
            while stack:
                u = stack.pop(); comp.append(u)
                for w in neigh[u]:
                    if not seen[w]:
                        seen[w] = True; stack.append(w)
            if 4 <= len(comp) <= 18:
                pool.append(np.array(comp, dtype=int))
        # greedy grown clusters
        sizes = [6, 8, 10, 12, 14, 16, 18, 5, 9, 13]
        for rep in range(60):
            seed = int(rng.randint(n))
            tgt = sizes[rep % len(sizes)]
            cl = set([seed])
            front = set(neigh[seed].tolist())
            front.discard(seed)
            while len(cl) < tgt and front:
                # pick frontier node with most edges into cl (greedy), tie random
                fl = list(front)
                scores = np.array([np.sum(adj[f, list(cl)]) for f in fl])
                mx = np.max(scores)
                candf = [f for f, s in zip(fl, scores) if s == mx]
                pick = candf[int(rng.randint(len(candf)))]
                cl.add(pick)
                front.discard(pick)
                for w in neigh[pick]:
                    if w not in cl:
                        front.add(w)
            if 4 <= len(cl) <= 18:
                pool.append(np.array(sorted(cl), dtype=int))
        # dedup roughly
        uniq = []
        seen_sets = set()
        for c in pool:
            key = tuple(sorted(c.tolist()))
            if key not in seen_sets:
                seen_sets.add(key); uniq.append(c)
        pool = uniq
        if len(pool) == 0:
            pool = [rng.choice(n, size=10, replace=False) for _ in range(10)]
        # rank by external contacts
        per, gmax0 = per_max(x_best)
        ranked = []
        for c in pool:
            mask = np.zeros(n, dtype=bool); mask[c] = True
            # external edges >0.35
            ext35 = 0
            for i in c:
                ext35 += np.sum(adj[i][~mask])
            ext35 = ext35 // 1
            # external near-max contacts
            sub = G0[np.ix_(c, ~mask)] if np.sum(~mask) > 0 else np.zeros((len(c), 0))
            extMax = np.sum(sub > gmax0 - 1e-7) if sub.size else 0
            score = ext35 + 10 * extMax + len(c) * 0.1
            ranked.append((score, c))
        ranked.sort(key=lambda t: -t[0])
        clusters = [c for _, c in ranked[:20]]
    except Exception:
        clusters = [rng.choice(n, size=10, replace=False) for _ in range(10)]
    thetas = [0.01, 0.025, 0.05, 0.09]
    attempt = 0
    # cycle
    while time.time() < cycle_end and time.time() < deadline - 12:
        if time.time() > cycle_end:
            break
        cl = clusters[attempt % len(clusters)]
        theta = thetas[attempt % len(thetas)]
        # orthonormal a,b in R9
        try:
            A = rng.randn(d)
            A /= np.linalg.norm(A)
            B = rng.randn(d)
            B = B - np.dot(B, A) * A
            nb = np.linalg.norm(B)
            if nb < 1e-8:
                attempt += 1
                continue
            B /= nb
            a, b = A, B
            ct, st = np.cos(theta), np.sin(theta)
            R = np.eye(d) + st * (np.outer(a, b) - np.outer(b, a)) + (ct - 1.0) * (np.outer(a, a) + np.outer(b, b))
            x_try = x_best.copy()
            xc = x_try[cl]  # m x d
            # rotate: row @ R.T
            x_try[cl] = xc @ R.T
            # relocate 2 rattlers into largest voids
            per_try, gm_try = per_max(x_try)
            order = np.argsort(per_try)  # ascending: worst (smallest max) first = rattlers
            ratt = order[:2].tolist()
            # sequential void search
            for ri in ratt:
                if time.time() > cycle_end or time.time() > deadline - 12:
                    break
                mask = np.ones(n, dtype=bool); mask[ri] = False
                others = x_try[mask]
                y, _ = find_void(others, rng, n_try=6000, n_refine=250)
                x_try[ri] = y
            # active set: cluster + rattlers + external contact neighbours
            core = np.unique(np.concatenate([cl, np.array(ratt, dtype=int)]))
            # neighbours: points with high dot to core
            Dots = x_try[core] @ x_try.T
            mx_to_core = np.max(Dots, axis=0)
            # exclude core itself
            iscore = np.zeros(n, dtype=bool); iscore[core] = True
            cand_nb = np.where((~iscore) & (mx_to_core > 0.40))[0]
            if len(cand_nb) == 0:
                # fallback: top 20 by mx_to_core
                idx = np.argsort(-mx_to_core)
                cand_nb = np.array([j for j in idx if j not in set(core.tolist())][:20], dtype=int)
            if len(cand_nb) > 40:
                # keep 40 highest
                o = np.argsort(-mx_to_core[cand_nb])[:40]
                cand_nb = cand_nb[o]
            active = np.unique(np.concatenate([core, cand_nb])).astype(int)
            # frozen max among pairs with neither endpoint active
            # compute quickly: max over complement
            frozen = np.ones(n, dtype=bool); frozen[active] = False
            if np.sum(frozen) >= 2:
                Gf = x_try[frozen] @ x_try[frozen].T
                np.fill_diagonal(Gf, -2.0)
                frozen_max = float(np.max(Gf)) if Gf.size else -2.0
            else:
                frozen_max = -2.0
            # 400 steps tangential softmax Adam on active
            x_ad, v_ad = subset_adam(x_try, active, frozen_max, steps=400, lr=0.02, T0=0.01, T1=0.001, rng=rng)
            # at most 6 s full LP polish
            rem_cycle = cycle_end - time.time()
            rem_dead = deadline - 0.5 - time.time()
            sec = min(6.0, rem_cycle, rem_dead)
            if sec < 0.5:
                # still check adam result
                if v_ad < b_best - 1e-12:
                    x_best, b_best = x_ad, v_ad
                    # rebuild clusters? keep same but update ranking quickly: continue
                attempt += 1
                continue
            try:
                x_pol = lp_polish(x_ad, seconds=sec, deadline=min(deadline - 0.5, time.time() + sec))
                v_pol = max_cos(x_pol)
            except Exception:
                v_pol = v_ad; x_pol = x_ad
            if v_pol < b_best - 1e-12:
                x_best = x_pol.copy()
                b_best = v_pol
                # keep clusters (topology changed, but continue cycling)
            elif v_ad < b_best - 1e-12:
                x_best = x_ad.copy()
                b_best = v_ad
        except Exception:
            pass
        attempt += 1
        if attempt > 200:
            break
    x_best = x_best / np.linalg.norm(x_best, axis=1, keepdims=True)
    return x_best
