import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _lp_polish(x, seconds, p_cap=1400):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    best_x = x.copy()
    radius = 1.0e-3
    start = time.time()
    iu0, iu1 = np.triu_indices(n, 1)
    itc = 0
    while time.time() - start < seconds and radius > 1e-14 and itc < 30:
        itc += 1
        if time.time() - start >= seconds:
            break
        G = x @ x.T
        c_all = G[iu0, iu1]
        m = c_all.max()
        window = max(25 * radius, 2e-9)
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
        I = iu0[sel]; J = iu1[sel]
        csel = c_all[sel]
        p = len(I)
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([(I[:, None]*d+np.arange(d)), (J[:, None]*d+np.arange(d)), np.full((p,1), n*d)], axis=1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p,1))], axis=1).ravel()
        try:
            remain = seconds - (time.time()-start)
            if remain <= 0.05:
                break
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n*d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv); obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)]*(n*d)+[(None,None)], method="highs",
                          options={"time_limit": max(0.3, remain)})
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        yn = np.linalg.norm(y, axis=1, keepdims=True)
        yn[yn==0]=1.0
        y = y/yn
        v = _max_cos(y)
        if v < best:
            best = v; best_x = y.copy(); x = y
            radius = min(radius*1.6, 0.02)
        else:
            radius *= 0.5
    return best_x

def _project_simplex(v):
    n = v.shape[0]
    u = np.sort(v)[::-1]
    css = np.cumsum(u) - 1.0
    ind = np.arange(1, n+1)
    cond = u - css/ind > 0
    if not np.any(cond):
        return np.full(n, 1.0/n)
    rho = ind[cond][-1]
    theta = css[cond][-1]/float(rho)
    return np.maximum(v-theta, 0.0)

def _stress_info(x, m, window=1e-6, tight=1e-9):
    n, d = x.shape
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    deg_w = (G > m - window).sum(axis=1)
    deg_t = (G > m - tight).sum(axis=1)
    wmax = G.max(axis=1)
    resid = np.zeros(n)
    for i in range(n):
        nbrs = np.where(G[i] > m - window)[0]
        k = len(nbrs)
        if k == 0:
            resid[i] = 1e9
            continue
        if k == 1:
            j = nbrs[0]
            gij = G[i, j]
            resid[i] = float(np.sqrt(max(0.0, 1.0-gij*gij)))
            continue
        T = x[nbrs] - G[i, nbrs][:, None]*x[i][None, :]
        Gram = T @ T.T
        w = np.full(k, 1.0/k)
        # projected gradient on simplex to minimize ||T^T w||^2
        for _ in range(80):
            grad = 2.0*(Gram @ w)
            w = w - 0.15*grad
            w = _project_simplex(w)
        v = T.T @ w if False else (w[None, :]*T.T).sum(axis=1) if False else (T.T @ w)
        # T is k x d, w k -> residual vector d
        rvec = T.T @ w if T.shape[0]==k else T.T @ w
        # Actually T.T @ w gives d vector
        resid[i] = float(np.linalg.norm(T.T @ w))
    return G, deg_w, deg_t, wmax, resid

def _repack_patch(cur, free_idx, time_budget, rng, T0=0.02, T1=0.002):
    n, d = cur.shape
    free_idx = np.asarray(free_idx, dtype=int)
    k = len(free_idx)
    frozen_idx = np.asarray([i for i in range(n) if i not in set(free_idx.tolist())], dtype=int)
    Xfroz = cur[frozen_idx].copy()
    # frozen-frozen max (constant)
    if len(frozen_idx) > 1:
        Gff0 = Xfroz @ Xfroz.T
        np.fill_diagonal(Gff0, -2.0)
        fmax = float(Gff0.max())
    else:
        fmax = -2.0
    best_y = cur.copy()
    G0 = cur @ cur.T
    np.fill_diagonal(G0, -2.0)
    # free-involved max of cur
    fim0 = float(max(G0[free_idx][:, :].max() if k>0 else -2.0, -2.0))
    best_fim = fim0
    start = time.time()
    # restart 0 = jittered current, rest = random
    n_restarts = 0
    first = True
    while time.time() - start < time_budget:
        n_restarts += 1
        if n_restarts > 12:
            break
        y = cur.copy()
        if first:
            first = False
            y[free_idx] += 0.05*rng.normal(size=(k, d))
            y[free_idx] /= np.linalg.norm(y[free_idx], axis=1, keepdims=True)
            steps = 250
        else:
            y[free_idx] = rng.normal(size=(k, d))
            y[free_idx] /= np.linalg.norm(y[free_idx], axis=1, keepdims=True)
            steps = 300
        Yf = y[free_idx].copy()
        # annealed softmax descent on free points only
        for s in range(steps):
            if s % 50 == 0 and time.time() - start >= time_budget:
                break
            frac = s/max(steps-1, 1)
            T = T0+(T1-T0)*frac
            lr = 0.5*(1.0-frac)+0.05*frac
            # dots
            Gfb = Yf @ Xfroz.T
            Gff = Yf @ Yf.T
            # max for stability among free-involved
            m = Gfb.max() if Gfb.size>0 else -2.0
            if k > 1:
                tmp = Gff.copy()
                np.fill_diagonal(tmp, -10.0)
                m = max(m, float(tmp.max()))
            Efb = np.exp((Gfb-m)/T) if Gfb.size>0 else Gfb
            Eff = np.exp((Gff-m)/T)
            np.fill_diagonal(Eff, 0.0)
            row = Efb.sum(axis=1, keepdims=True)+Eff.sum(axis=1, keepdims=True)+1e-20
            Avg = (Efb @ Xfroz + Eff @ Yf)/row
            dot = (Avg*Yf).sum(axis=1, keepdims=True)
            Gt = Avg - dot*Yf
            Yf = Yf - lr*Gt
            Yf /= np.linalg.norm(Yf, axis=1, keepdims=True)
        y[free_idx] = Yf
        # evaluate free-involved max
        Gall = y @ y.T
        # free vs all
        fim = float(Gall[free_idx].max())
        # exclude diagonal self (1.0) -> need mask
        # Gall[free_idx] includes self 1.0; recompute properly
        # max over free-frozen and free-free(i<j)
        Gfb2 = y[free_idx] @ y[frozen_idx].T
        fim2 = float(Gfb2.max()) if Gfb2.size>0 else -2.0
        if k > 1:
            Gf2 = y[free_idx] @ y[free_idx].T
            np.fill_diagonal(Gf2, -2.0)
            fim2 = max(fim2, float(Gf2.max()))
        if fim2 < best_fim:
            best_fim = fim2
            best_y = y.copy()
        if time.time() - start >= time_budget:
            break
    return best_y

def solve():
    t0 = time.time()
    LIM = 172.0
    rng = np.random.default_rng(12345)
    n, d = 123, 13
    X0 = None; bv = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn); a = np.asarray(a, float)
            if a.shape == (n, d):
                a /= np.linalg.norm(a, axis=1, keepdims=True)
                if np.all(np.isfinite(a)):
                    v = _max_cos(a)
                    if v < bv:
                        bv = v; X0 = a.copy()
        except Exception:
            pass
    if X0 is None:
        X0 = rng.normal(size=(n, d)); X0 /= np.linalg.norm(X0, axis=1, keepdims=True); bv = _max_cos(X0)
    best = X0.copy(); best_val = float(bv)
    # brief initial tighten if time allows
    try:
        if time.time()-t0 < LIM-70:
            cand = _lp_polish(best, 2.0)
            v = _max_cos(cand)
            if v < best_val:
                best = cand; best_val = v
    except Exception:
        pass
    if time.time()-t0 >= LIM-25:
        best /= np.linalg.norm(best, axis=1, keepdims=True)
        return best
    # stress analysis on best
    try:
        G, deg_w, deg_t, wmax, resid = _stress_info(best, best_val, window=1e-6, tight=1e-9)
    except Exception:
        best /= np.linalg.norm(best, axis=1, keepdims=True)
        return best
    # rattler: smallest tight degree, tie-break by smallest wmax
    order_tight = np.lexsort((wmax, deg_t))
    rattler = int(order_tight[0])
    # nearest neighbours of rattler by angle
    dots_r = (best @ best[rattler]).copy()
    dots_r[rattler] = -2.0
    nn_r = np.argsort(-dots_r)
    # hubs by residual (large = bad equilibrium), exclude rattler
    hub_order = np.argsort(-resid)
    hub_order = [h for h in hub_order if h != rattler]
    # also overload hubs by degree
    # build up to 3 patches
    patches = []
    for pi in range(min(3, len(hub_order))):
        hub = int(hub_order[pi])
        # hub neighbourhood: hub + top contacts/nearest
        dots_h = (best @ best[hub]).copy()
        dots_h[hub] = -2.0
        nn_h = np.argsort(-dots_h)
        s = set()
        s.add(rattler)
        for q in nn_r[:4]:
            s.add(int(q))
        s.add(hub)
        for q in nn_h[:8]:
            if len(s) >= 14:
                break
            s.add(int(q))
        # adjust size to 10-18
        if len(s) < 10:
            # add lowest-stress (high resid) points
            for q in hub_order:
                if int(q) not in s:
                    s.add(int(q))
                if len(s) >= 12:
                    break
        if len(s) > 18:
            # prune farthest from hub/rattler, keep hub+rattler
            lst = list(s)
            # keep rattler, its 4nn, hub first
            keep = set([rattler, hub] + [int(q) for q in nn_r[:4]])
            rest = [q for q in lst if q not in keep]
            # sort rest by dots to hub descending, keep top to reach 18
            rest_sorted = sorted(rest, key=lambda q: -(best[q] @ best[hub]))
            need = 18 - len(keep)
            keep.update(rest_sorted[:need])
            s = keep
        patches.append(sorted(s))
        if time.time()-t0 >= LIM-25:
            break
    for pi, patch in enumerate(patches):
        if time.time()-t0 >= LIM-25:
            break
        # repack patch strictly capped 15s (use 12s to leave margin, last patch smaller)
        budget = 12.0
        # ensure at least 5s LP after + final margin
        remain = LIM - (time.time()-t0)
        if remain < budget + 6.0:
            budget = max(2.0, remain - 6.0)
        try:
            trial = _repack_patch(best, patch, budget, rng)
        except Exception:
            continue
        if time.time()-t0 >= LIM-20:
            break
        # LP-polish whole config at most 5 seconds
        try:
            lpb = min(5.0, max(1.0, LIM-(time.time()-t0)-2.0))
            trial2 = _lp_polish(trial, lpb)
        except Exception:
            trial2 = trial
        if time.time()-t0 >= LIM-2:
            break
        v = _max_cos(trial2)
        if v < best_val - 1e-15:
            best = trial2.copy(); best_val = v
            # refresh stress info for remaining patches? keep original hubs but update G quickly
            try:
                G, deg_w, deg_t, wmax, resid = _stress_info(best, best_val, window=1e-6, tight=1e-9)
            except Exception:
                pass
        # time checked after every stage (already)
    # final safety: no extra long polish (would risk timeout); tiny polish if ample time
    try:
        if time.time()-t0 < LIM-8:
            sec = min(4.0, LIM-(time.time()-t0)-1.0)
            if sec >= 1.0:
                pol = _lp_polish(best, sec)
                pv = _max_cos(pol)
                if pv < best_val:
                    best = pol; best_val = pv
    except Exception:
        pass
    best /= np.linalg.norm(best, axis=1, keepdims=True)
    return best
