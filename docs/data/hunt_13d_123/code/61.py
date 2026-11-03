import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _normalize(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return x / n

def _lp_polish(x, seconds, p_cap=1400):
    x = _normalize(x)
    n, d = x.shape
    best = _max_cos(x)
    best_x = x.copy()
    radius = 1e-3
    start = time.time()
    iu0, iu1 = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        G = x @ x.T
        c_all = G[iu0, iu1]
        m = float(c_all.max())
        window = max(20.0 * radius, 1e-9)
        sel = c_all > m - window
        cnt = int(np.count_nonzero(sel))
        if cnt == 0:
            break
        if cnt > p_cap:
            idx = np.where(sel)[0]
            part = np.argpartition(-c_all[idx], p_cap - 1)[:p_cap]
            keep = np.zeros_like(sel)
            keep[idx[part]] = True
            sel = keep
            cnt = p_cap
        I = iu0[sel]; J = iu1[sel]; csel = c_all[sel]
        p = cnt
        nv = n * d + 1
        try:
            rows = np.repeat(np.arange(p), 2 * d + 1)
            cols = np.concatenate([(I[:, None] * d + np.arange(d)).reshape(p, d),
                                   (J[:, None] * d + np.arange(d)).reshape(p, d),
                                   np.full((p, 1), n * d)], axis=1).ravel()
            vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], axis=1).ravel()
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv); obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y = _normalize(y)
        v = _max_cos(y)
        if v < best:
            best = v; best_x = y.copy(); x = y
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
            if radius < 1e-7:
                x = best_x.copy()
        if time.time() - start >= seconds:
            break
    return best_x

def _greedy_holes(F, P, rng, Nc=10000, top0=20):
    d = F.shape[1]
    C = rng.randn(Nc, d)
    C = _normalize(C)
    Dmax = (C @ F.T).max(axis=1)
    best_scores = Dmax.copy()
    chosen = []
    for k in range(P):
        if k == 0:
            pool = np.argpartition(best_scores, top0 - 1)[:top0]
            # random among top to diversify basins
            sel = int(pool[rng.randint(len(pool))])
        else:
            sel = int(np.argmin(best_scores))
            if rng.rand() < 0.15:
                cand = np.argpartition(best_scores, 5)[:5]
                o = np.argsort(best_scores[cand])
                if len(o) > 1:
                    sel = int(cand[o[1]])
        chosen.append(C[sel])
        dots = C @ C[sel]
        np.maximum(best_scores, dots, out=best_scores)
        best_scores[sel] = 2.0
    return np.stack(chosen)

def _patch_anneal(full, patch_idx, steps=500, beta0=25.0, beta1=400.0, lr0=0.015, lr1=0.0015, mu=0.8):
    n, d = full.shape
    P = len(patch_idx)
    cur = full.copy()
    V = cur[patch_idx].copy()
    vel = np.zeros_like(V)
    pmap = np.zeros(n, dtype=int)
    # map global col -> position in V for self masking
    g2p = {int(g): i for i, g in enumerate(patch_idx)}
    for s in range(steps):
        f = s / max(steps - 1, 1)
        beta = beta0 * (beta1 / beta0) ** f
        lr = lr0 * (lr1 / lr0) ** f
        cur[patch_idx] = V
        G = V @ cur.T  # P x N
        for i, g in enumerate(patch_idx):
            G[i, int(g)] = -10.0
        gmax = float(G.max())
        W = np.exp(beta * (G - gmax))
        for i, g in enumerate(patch_idx):
            W[i, int(g)] = 0.0
        row = W.sum(axis=1, keepdims=True) + 1e-12
        grad = W @ cur
        dot = np.sum(grad * V, axis=1, keepdims=True)
        gt = (grad - dot * V) / row
        vel = mu * vel + gt
        V = V - lr * vel
        V = _normalize(V)
    cur[patch_idx] = V
    return _normalize(cur)

def _patch_for_hub(adj, hub, target=20, cap=25):
    one = np.where(adj[hub])[0]
    # include hub
    s = set([int(hub)] + [int(v) for v in one])
    if len(s) < target:
        # add two-hop sorted by degree
        two = set()
        for v in list(s):
            for w in np.where(adj[int(v)])[0]:
                two.add(int(w))
        two -= s
        deg2 = [(int(w), int(adj[int(w)].sum())) for w in two]
        deg2.sort(key=lambda t: -t[1])
        need = min(target - len(s), len(deg2))
        for k in range(need):
            s.add(deg2[k][0])
    lst = list(s)
    if len(lst) > cap:
        # keep closest to hub (adj true first, then arbitrary)
        # compute dots later outside; here trim by keeping hub + one-hop first
        keep = [int(hub)] + [int(v) for v in one if int(v) in s]
        rest = [v for v in lst if v not in keep]
        keep = keep[:cap]
        if len(keep) < cap:
            keep += rest[:cap - len(keep)]
        lst = keep
    return np.array(sorted(lst), dtype=int)

def solve():
    t0 = time.time()
    HOP_END = 132.0
    HARD = 168.0
    n, d = 123, 13
    rng = np.random.RandomState(7)
    best = None; best_val = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=float)
            if a.shape == (n, d):
                a = _normalize(a)
                v = _max_cos(a)
                if v < best_val:
                    best_val = v; best = a.copy()
        except Exception:
            pass
    if best is None:
        best = _normalize(rng.randn(n, d))
        best_val = _max_cos(best)
    x_best = best.copy()
    x_cur = best.copy()
    cur_val = best_val

    outer = 0
    # cycle hubs
    while time.time() - t0 < HOP_END and outer < 30:
        outer += 1
        # near-contact graph within 1e-4 of max on current walk state
        Gc = x_cur @ x_cur.T
        np.fill_diagonal(Gc, -2.0)
        m = float(Gc.max())
        adj = Gc > (m - 1e-4)
        deg = adj.sum(axis=1)
        hubs = np.argsort(-deg)
        # take ~8 distinct hubs spread out (avoid adjacent hubs same patch)
        picked = []
        used = set()
        for h in hubs:
            h = int(h)
            if h in used:
                continue
            picked.append(h)
            # suppress 1-hop neighbours from being separate hubs this round
            for nb in np.where(adj[h])[0]:
                used.add(int(nb))
            used.add(h)
            if len(picked) >= 8:
                break
        if len(picked) == 0:
            picked = [int(hubs[0])]
        for hi, h in enumerate(picked):
            if time.time() - t0 > HOP_END:
                break
            # vary target size 15-25 across hubs
            target = 16 + ((outer + hi) % 5) * 2  # 16,18,20,22,24
            patch = _patch_for_hub(adj, h, target=target, cap=25)
            if len(patch) < 10:
                continue
            frozen_mask = np.ones(n, dtype=bool); frozen_mask[patch] = False
            F = x_cur[frozen_mask]
            # greedy deepest-hole placement
            try:
                V0 = _greedy_holes(F, len(patch), rng, Nc=9000)
            except Exception:
                continue
            x_try = x_cur.copy()
            x_try[patch] = V0
            x_try = _normalize(x_try)
            # patch-only annealed softmax descent
            steps = 450 if len(patch) <= 20 else 350
            try:
                x_try = _patch_anneal(x_try, patch, steps=steps)
            except Exception:
                continue
            sv = _max_cos(x_try)
            # skip LP if catastrophically bad
            if sv > cur_val + 0.008 and sv > best_val + 0.008:
                # still allow walk occasionally? reject to save time
                continue
            remain = HOP_END - (time.time() - t0)
            if remain < 2.0:
                break
            lp_sec = 2.0 if sv < cur_val + 0.002 else 1.2
            lp_sec = min(lp_sec, remain)
            try:
                x_pol = _lp_polish(x_try, lp_sec)
            except Exception:
                continue
            fv = _max_cos(x_pol)
            if fv < best_val:
                best_val = fv; x_best = x_pol.copy()
                x_cur = x_pol.copy(); cur_val = fv
                # refresh graph baseline immediately
                break
            elif fv < cur_val + 0.0015:
                # lateral / slightly worse walk move: new basin, not restart from global best
                x_cur = x_pol.copy(); cur_val = fv
                # update adj incrementally for next hub in same round
                Gc = x_cur @ x_cur.T
                np.fill_diagonal(Gc, -2.0)
                m = float(Gc.max())
                adj = Gc > (m - 1e-4)
            # else reject, keep x_cur
        # if stuck, small global jitter of rattlers to diversify cur without losing best
        if time.time() - t0 > HOP_END:
            break
    remain = HARD - (time.time() - t0)
    if remain > 5:
        try:
            x_best = _lp_polish(x_best, min(remain, 25.0))
        except Exception:
            pass
    return _normalize(x_best)
