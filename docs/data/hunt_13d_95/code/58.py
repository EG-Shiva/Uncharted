import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _contacts(x, tol=1e-9):
    n = x.shape[0]
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    m = float(G.max())
    cnt = (G > m - tol).sum(axis=1)
    return cnt, m, G

def _lp_polish(x, seconds, radius_init=1e-3, max_c=1500, t_end=None):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = float(radius_init)
    start = time.time()
    iu = np.triu_indices(n, 1)
    while True:
        now = time.time()
        if now - start >= seconds:
            break
        if t_end is not None and now >= t_end:
            break
        if radius < 1e-14:
            break
        if seconds - (now - start) < 0.05:
            break
        G = x @ x.T
        c = G[iu]
        m = float(c.max())
        thresh = m - max(20.0 * radius, 1e-9)
        sel = c > thresh
        cnt = int(sel.sum())
        if cnt == 0:
            break
        if cnt > max_c:
            idx = np.argpartition(c, -max_c)[-max_c:]
            mask = np.zeros_like(sel, dtype=bool)
            mask[idx] = True
            sel = mask
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(),
                               (np.repeat(np.arange(n), d), np.arange(n * d))),
                              shape=(n, nv)).tocsr()
        except Exception:
            break
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
        nr = np.linalg.norm(y, axis=1, keepdims=True)
        nr[nr == 0] = 1.0
        y = y / nr
        v = _max_cos(y)
        if v < best:
            x, best, radius = y, v, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def _largest_hole(X94, rng, n_samples=50000, n_best=20, refine_steps=250):
    n94, d = X94.shape
    # sample in chunks, keep best n_best with smallest max-dot
    best_vals = np.full(n_best, np.inf)
    best_vecs = np.zeros((n_best, d))
    chunk = 5000
    done = 0
    while done < n_samples:
        m = min(chunk, n_samples - done)
        cand = rng.standard_normal(size=(m, d))
        cand /= np.linalg.norm(cand, axis=1, keepdims=True)
        dots = cand @ X94.T
        md = dots.max(axis=1)
        # merge
        allv = np.concatenate([best_vals, md])
        alli = np.argpartition(allv, n_best)[:n_best]
        # need corresponding vectors
        allvecs = np.vstack([best_vecs, cand])
        nv = allv[alli]
        order = np.argsort(nv)
        best_vals = nv[order]
        best_vecs = allvecs[alli[order]]
        done += m
    V = best_vecs.copy()  # (n_best, d)
    # refine best 20 by subgradient on max-dot with decaying step
    lr0, lr1 = 0.08, 0.002
    for s in range(refine_steps):
        f = s / max(1, refine_steps - 1)
        lr = lr0 * (lr1 / lr0) ** f
        D = V @ X94.T  # (B,94)
        idx = D.argmax(axis=1)
        Gmax = X94[idx]  # (B,d)
        dots = np.sum(Gmax * V, axis=1, keepdims=True)
        Gt = Gmax - dots * V
        V = V - lr * Gt
        V /= np.linalg.norm(V, axis=1, keepdims=True)
    D = V @ X94.T
    md = D.max(axis=1)
    j = int(np.argmin(md))
    return V[j] / np.linalg.norm(V[j]), float(md[j])

def solve():
    t0 = time.time()
    EXPL_END = t0 + 140.0
    HARD_END = t0 + 170.0
    rng = np.random.default_rng(12345)
    try:
        xb = np.load("best.npy").astype(float)
    except Exception:
        xb = None
    try:
        xr = np.load("record.npy").astype(float)
    except Exception:
        xr = None
    cands = []
    if xb is not None and np.shape(xb) == (95, 13):
        xn = xb / np.linalg.norm(xb, axis=1, keepdims=True)
        if np.all(np.isfinite(xn)):
            cands.append(xn)
    if xr is not None and np.shape(xr) == (95, 13):
        xn = xr / np.linalg.norm(xr, axis=1, keepdims=True)
        if np.all(np.isfinite(xn)):
            cands.append(xn)
    if not cands:
        z = rng.standard_normal((95, 13))
        z /= np.linalg.norm(z, axis=1, keepdims=True)
        cands.append(z)
    vals = [_max_cos(c) for c in cands]
    bi = int(np.argmin(vals))
    x_init = cands[bi].copy()
    best_val0 = float(vals[bi])

    # initial quick state
    cur = x_init.copy()
    cur_val = best_val0
    global_best = cur.copy()
    global_val = cur_val
    results = [(global_val, global_best.copy())]

    # contact info for first deletion
    try:
        cnt0, _, _ = _contacts(cur)
        rattler0 = int(np.argmin(cnt0))
    except Exception:
        rattler0 = 94

    tabu = []  # recent delete positions
    last_insert = -100
    it = 0
    first = True
    while time.time() < EXPL_END:
        if HARD_END - time.time() < 32:
            break
        it += 1
        ncur = cur.shape[0]
        if ncur != 95:
            # repair
            cur = global_best.copy()
        try:
            cnt, mcur, _ = _contacts(cur)
        except Exception:
            break
        # choose delete index
        if first:
            del_idx = rattler0 if 0 <= rattler0 < 95 else int(np.argmin(cnt))
            first = False
        else:
            order = np.argsort(-cnt)  # hubs first
            del_idx = -1
            for o in order:
                o = int(o)
                if o == last_insert and it < 6:
                    continue
                if o in tabu:
                    continue
                del_idx = o
                break
            if del_idx < 0:
                # clear tabu partially
                tabu = tabu[len(tabu)//2:]
                for o in order:
                    o = int(o)
                    if o == last_insert and len(order) > 5:
                        continue
                    del_idx = o
                    break
            if del_idx < 0:
                del_idx = int(order[0])
        tabu.append(int(del_idx))
        if len(tabu) > 12:
            tabu = tabu[-12:]
        # delete -> 94
        try:
            X94 = np.delete(cur, del_idx, axis=0)
        except Exception:
            continue
        # LP/softmax polish remaining 94 (gain slack) - short budget
        rem = EXPL_END - time.time()
        if rem < 3:
            break
        sec94 = 0.7 if rem > 20 else 0.4
        try:
            X94 = _lp_polish(X94, seconds=sec94, radius_init=1e-3, t_end=EXPL_END)
        except Exception:
            pass
        # locate largest empty cap: sample 50000, refine best 20
        try:
            hole, _ = _largest_hole(X94, rng, n_samples=50000, n_best=20, refine_steps=200)
        except Exception:
            continue
        X95 = np.vstack([X94, hole.reshape(1, -1)])
        X95 /= np.linalg.norm(X95, axis=1, keepdims=True)
        # relax all 95
        rem2 = EXPL_END - time.time()
        if rem2 < 2:
            # store without relax
            v = _max_cos(X95)
            results.append((v, X95.copy()))
            if v < global_val:
                global_val, global_best = v, X95.copy()
            break
        sec95 = 0.9 if rem2 > 20 else 0.5
        try:
            X95r = _lp_polish(X95, seconds=sec95, radius_init=3e-3, t_end=EXPL_END)
        except Exception:
            X95r = X95
        v = _max_cos(X95r)
        results.append((float(v), X95r.copy()))
        if v < global_val - 1e-15:
            global_val = float(v)
            global_best = X95r.copy()
            cur = X95r.copy()
            cur_val = float(v)
            last_insert = 94  # new point at end (approx)
        else:
            # move basin if not too bad to enable topology change
            if v < global_val + 0.015:
                cur = X95r.copy()
                cur_val = float(v)
                last_insert = 94
            else:
                # revert to global best but keep tabu to force different deletion
                cur = global_best.copy()
                cur_val = global_val
                # last_insert unchanged
            # occasional restart from global best already done
        # safety: cap results length
        if len(results) > 60:
            # keep best 20 + last 5
            results.sort(key=lambda t: t[0])
            results = results[:20] + results[-5:]
    # rank distinct results by max cosine
    # dedup roughly by value
    results.append((global_val, global_best.copy()))
    # also include initial
    try:
        results.append((best_val0, x_init.copy()))
    except Exception:
        pass
    results.sort(key=lambda t: t[0])
    # pick best
    top = results[0][1].copy()
    top_val = float(results[0][0])
    # spend last ~30s LP-polishing best one
    tnow = time.time()
    sec_left = HARD_END - tnow - 1.0
    if sec_left > 2:
        sec_left = min(sec_left, 30.0)
        try:
            top = _lp_polish(top, seconds=sec_left, radius_init=5e-4, max_c=2000, t_end=HARD_END - 0.5)
            top_val = _max_cos(top)
        except Exception:
            pass
    top /= np.linalg.norm(top, axis=1, keepdims=True)
    # fallback safety: if top is NaN or worse than record by much, return best valid
    if not np.all(np.isfinite(top)):
        top = x_init.copy()
    # ensure we return the best among top and x_init
    try:
        tv = _max_cos(top)
        if tv > best_val0 and best_val0 < 0.3:
            # still return top (it is the explored one) unless top is much worse;
            # prefer the better one to guarantee valid record-chasing output
            if tv > best_val0 + 1e-12:
                # check other results
                for vv, xx in results:
                    if vv < tv:
                        top = xx.copy()
                        tv = vv
                        break
    except Exception:
        pass
    top /= np.linalg.norm(top, axis=1, keepdims=True)
    return top
