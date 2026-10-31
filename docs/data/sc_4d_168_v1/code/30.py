import time
import numpy as np
from scipy.optimize import linprog, linear_sum_assignment
from scipy.sparse import coo_matrix

def _max_cos(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def _lp_polish(x, deadline):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    iu = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        C = (x @ x.T)[iu]
        m = float(np.max(C))
        sel = C > m - max(20.0 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        if p == 0:
            break
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d),
                                           np.arange(n * d))), shape=(n, nv)).tocsr()
        except Exception:
            break
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-C[sel], A_eq=A_eq,
                          b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                          method="highs")
        except Exception:
            break
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        v = _max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def _contact_key(x, tol=1e-4):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    m = float(np.max(G))
    adj = G > (m - tol)
    # adj includes symmetric; diagonal false
    degs = np.sum(adj, axis=1).astype(int)
    sdegs = tuple(sorted(degs.tolist()))
    ncon = int(np.sum(adj) // 2)
    # histogram feature: mean of top-600 cosines quantized
    iu = np.triu_indices(x.shape[0], 1)
    c = (x @ x.T)[iu]
    top = np.sort(c)[-600:]
    # quantize to 1e-4
    q = tuple(np.round(top[::60], 4).tolist())
    return (ncon, sdegs, q)

def solve():
    t0 = time.time()
    HARD_END = t0 + 172.0
    SEARCH_END = t0 + 140.0
    FINAL_END = t0 + 170.0
    rng = np.random.RandomState(7)
    n, d = 168, 4

    try:
        xb = np.load("best.npy").astype(float)
    except Exception:
        xb = rng.randn(n, d)
    try:
        xr = np.load("record.npy").astype(float)
    except Exception:
        xr = xb.copy()
    def _norm(v):
        v = np.asarray(v, dtype=float)
        if v.shape != (n, d):
            return None
        v = v / np.linalg.norm(v, axis=1, keepdims=True)
        if not np.all(np.isfinite(v)):
            return None
        return v
    xb = _norm(xb)
    xr = _norm(xr)
    if xb is None:
        xb = rng.randn(n, d); xb /= np.linalg.norm(xb, axis=1, keepdims=True)
    if xr is None:
        xr = xb.copy()
    vb = _max_cos(xb); vr = _max_cos(xr)
    if vr < vb:
        x_best = xr.copy(); best_val = vr
    else:
        x_best = xb.copy(); best_val = vb
    # ensure finite
    init_cands = [x_best, xb, xr]

    def refine(x_init, steps, beta0=25.0, beta1=500.0, lr0=0.014, lr1=0.0007, mu=0.9):
        x = x_init.copy()
        vel = np.zeros_like(x)
        for s in range(steps):
            frac = s / max(1, steps - 1)
            beta = beta0 * (beta1 / beta0) ** frac
            lr = lr0 * (1.0 - frac) + lr1 * frac
            G = x @ x.T
            np.fill_diagonal(G, -2.0)
            cmax = float(np.max(G))
            E = np.exp(np.clip(beta * (G - cmax), -20.0, 0.0))
            np.fill_diagonal(E, 0.0)
            rs = np.sum(E, axis=1, keepdims=True) + 1e-12
            grad = (E @ x) / rs
            dot = np.sum(grad * x, axis=1, keepdims=True)
            gt = grad - dot * x
            vel = mu * vel + (1.0 - mu) * gt
            x = x - lr * vel
            x /= np.linalg.norm(x, axis=1, keepdims=True)
        return x

    def deepest_hole(X_keep, n_sample=8000, opt_steps=25):
        S = rng.randn(n_sample, 4)
        S /= np.linalg.norm(S, axis=1, keepdims=True)
        D = S @ X_keep.T
        m = D.max(axis=1)
        s = S[np.argmin(m)].copy()
        # quick local optimisation of s (minimise smooth max)
        for it in range(opt_steps):
            dots = X_keep @ s
            cm = float(np.max(dots))
            beta = 80.0
            w = np.exp(np.clip(beta * (dots - cm), -20.0, 0.0))
            w /= (np.sum(w) + 1e-30)
            g = w @ X_keep
            gt = g - np.dot(g, s) * s
            s = s - 0.06 * gt
            ns = np.linalg.norm(s)
            if ns < 1e-12:
                s = rng.randn(4); ns = np.linalg.norm(s)
            s /= ns
        return s

    def reinsert_k(X_keep, k, n_sample=8000):
        cur = X_keep.copy()
        outs = []
        for i in range(k):
            h = deepest_hole(cur, n_sample=n_sample)
            outs.append(h)
            cur = np.vstack([cur, h[None, :]])
        return np.vstack(outs) if outs else np.zeros((0, 4))

    def kick_offspring(parent):
        x = parent.copy()
        typ = rng.randint(0, 4)
        if typ == 0:
            sig = rng.uniform(0.02, 0.07)
            x = x + sig * rng.randn(n, d)
            x /= np.linalg.norm(x, axis=1, keepdims=True)
        elif typ == 1:
            sig = rng.uniform(0.08, 0.16)
            x = x + sig * rng.randn(n, d)
            x /= np.linalg.norm(x, axis=1, keepdims=True)
            # light refine later
        elif typ == 2:
            # cluster ejection: remove geographic cluster, reinsert via holes
            idx = rng.randint(0, n)
            dots = x @ x[idx]
            k = int(rng.randint(8, 19))
            order = np.argsort(-dots)
            cluster = order[:k]
            mask = np.ones(n, dtype=bool); mask[cluster] = False
            keep = x[mask]
            newp = reinsert_k(keep, k, n_sample=6000)
            x = np.vstack([keep, newp])
            # shuffle order to avoid bias
            perm = rng.permutation(n)
            x = x[perm]
            # small shake
            x = x + 0.01 * rng.randn(n, d)
            x /= np.linalg.norm(x, axis=1, keepdims=True)
        else:
            # cap shake: pick random direction, shake points in cap strongly
            u = rng.randn(4); u /= np.linalg.norm(u)
            proj = x @ u
            k = int(rng.randint(20, 50))
            idx = np.argsort(-proj)[:k]
            x[idx] = x[idx] + rng.uniform(0.05, 0.12) * rng.randn(k, d)
            x /= np.linalg.norm(x, axis=1, keepdims=True)
        return x

    def crossover_offspring(A, B):
        # align B to A via assignment + Kabsch
        try:
            D = A @ B.T
            ri, ci = linear_sum_assignment(-D)
            perm = np.empty(n, dtype=int)
            perm[ri] = ci
            Bm = B[perm]
            M = Bm.T @ A
            U, _, Vt = np.linalg.svd(M)
            R = U @ Vt
            if np.linalg.det(R) < 0:
                U[:, -1] *= -1
                R = U @ Vt
            Ba = B @ R
            Ba /= np.linalg.norm(Ba, axis=1, keepdims=True)
        except Exception:
            Ba = B.copy()
        u = rng.randn(4); u /= np.linalg.norm(u)
        k = int(rng.randint(56, 85))  # 1/3 to 1/2
        pb = Ba @ u
        pa = A @ u
        idxB = np.argsort(-pb)[:k]
        idxA = np.argsort(-pa)[:k]
        cap = Ba[idxB]
        mask = np.ones(n, dtype=bool); mask[idxA] = False
        keep = A[mask]
        x = np.vstack([keep, cap])
        # repair near-duplicates: dot > 0.99
        for _ in range(6):
            G = x @ x.T
            np.fill_diagonal(G, -2.0)
            m = float(np.max(G))
            if m < 0.985:
                break
            ij = np.unravel_index(int(np.argmax(G)), G.shape)
            # replace j with hole w.r.t. rest
            mask2 = np.ones(n, dtype=bool); mask2[ij[1]] = False
            h = deepest_hole(x[mask2], n_sample=6000, opt_steps=15)
            x[ij[1]] = h
        # small shake to break exact ties
        x = x + 0.008 * rng.randn(n, d)
        x /= np.linalg.norm(x, axis=1, keepdims=True)
        return x

    # archive init
    archive = []  # list of dicts
    seen = set()
    def try_add(x, v):
        nonlocal best_val, x_best
        if not np.all(np.isfinite(x)):
            return False
        key = _contact_key(x)
        if v < best_val:
            best_val = float(v); x_best = x.copy()
        if key in seen:
            # replace if better than existing same key
            for e in archive:
                if e['key'] == key and v < e['val']:
                    e['x'] = x.copy(); e['val'] = float(v)
                    return True
            return False
        # novel
        seen.add(key)
        archive.append({'x': x.copy(), 'val': float(v), 'key': key})
        # keep size <=10, evict worst (but keep best)
        if len(archive) > 10:
            # find worst index not the global best
            vals = [e['val'] for e in archive]
            wi = int(np.argmax(vals))
            # don't evict if newcomer is worst? already added; evict worst
            archive.pop(wi)
            # rebuild seen (keys of remaining)
            seen.clear()
            for e in archive:
                seen.add(e['key'])
        return True

    for xc in init_cands:
        try_add(xc, _max_cos(xc))
    # build diverse initial variants: strongly kicked + hole reinserted, refined lightly, no LP yet (save time) then short LP for a few
    for vi in range(4):
        if time.time() > SEARCH_END - 10:
            break
        p = x_best.copy()
        # strong kick
        sig = 0.12 if vi % 2 == 0 else 0.06
        y = p + sig * rng.randn(n, d)
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        if vi >= 2:
            # hole reinsert one point: remove worst point (max contacts) and reinsert
            G = y @ y.T; np.fill_diagonal(G, -2.0)
            m = float(np.max(G))
            # count contacts
            adj = G > m - 1e-4
            degs = adj.sum(axis=1)
            idx = int(np.argmax(degs))
            mask = np.ones(n, dtype=bool); mask[idx] = False
            h = deepest_hole(y[mask], n_sample=8000, opt_steps=25)
            y[idx] = h
        y = refine(y, steps=400)
        vref = _max_cos(y)
        # short LP polish 2 sec
        y = _lp_polish(y, min(time.time() + 2.0, SEARCH_END))
        try_add(y, _max_cos(y))

    # ensure archive has at least 2 entries; if not, add perturbed
    while len(archive) < 3 and time.time() < SEARCH_END:
        y = x_best + 0.05 * rng.randn(n, d)
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        y = refine(y, steps=300)
        try_add(y, _max_cos(y))

    it = 0
    while time.time() < SEARCH_END:
        it += 1
        if time.time() > SEARCH_END - 3.0:
            break
        # select parent (not always best): 35% best, else random
        vals = np.array([e['val'] for e in archive])
        ibest = int(np.argmin(vals))
        if rng.rand() < 0.35:
            ip = ibest
        else:
            ip = int(rng.randint(0, len(archive)))
        parent = archive[ip]['x']
        pval = archive[ip]['val']
        # choose operator: 55% kick, 45% crossover
        if rng.rand() < 0.55 or len(archive) < 2:
            raw = kick_offspring(parent)
        else:
            iq = int(rng.randint(0, len(archive)))
            if iq == ip:
                iq = (iq + 1) % len(archive)
            raw = crossover_offspring(parent, archive[iq]['x'])
        # refine
        steps = int(rng.choice([400, 550, 700]))
        y = refine(raw, steps=steps)
        vref = _max_cos(y)
        # worse-accepting pre-filter: if much worse than parent (+8e-4) skip LP to save time
        if vref > pval + 8e-4 and rng.rand() < 0.8:
            continue
        # LP polish with ~2.2s budget
        budget = min(2.2, SEARCH_END - time.time() - 0.5)
        if budget < 0.6:
            break
        y = _lp_polish(y, time.time() + budget)
        v = _max_cos(y)
        # acceptance threshold up to +2e-4 over parent
        if v > pval + 2e-4 + 1e-12:
            # allow occasional random acceptance of novel topology? skip
            # still track global best
            if v < best_val:
                try_add(y, v)
            continue
        key = _contact_key(y)
        is_novel = key not in seen
        if is_novel or v < min(vals) - 1e-12:
            # insert: if full and worse than worst, only insert if novel and within +2e-4 of worst
            worst = float(np.max(vals)) if len(vals) else v
            if len(archive) >= 10 and v > worst + 2e-4:
                if v < best_val:
                    try_add(y, v)
                continue
            try_add(y, v)
        else:
            # same topology but maybe better squeeze
            try_add(y, v)
        # update vals loop automatically
        # opportunistic: if improved global best, continue exploiting
    # final polish of best with remaining time
    if time.time() < FINAL_END:
        # try polishing top 2 archive members briefly and keep best
        archive_sorted = sorted(archive, key=lambda e: e['val'])
        for e in archive_sorted[:2]:
            if time.time() >= FINAL_END - 1.0:
                break
            yp = _lp_polish(e['x'], min(FINAL_END, time.time() + 6.0))
            vp = _max_cos(yp)
            if vp < best_val:
                best_val = vp; x_best = yp.copy()
        x_best = _lp_polish(x_best, FINAL_END)
        best_val = _max_cos(x_best)
    x_best = x_best / np.linalg.norm(x_best, axis=1, keepdims=True)
    return x_best
