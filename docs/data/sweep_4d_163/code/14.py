import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def row_maxs(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return g.max(axis=1), g

def contact_degrees(G, m, tol=1e-7):
    # number of contacts per point within tol of max
    B = G > (m - tol)
    np.fill_diagonal(B, False)
    return B.sum(axis=1)

def lp_polish(x, seconds):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = max_cos(x)
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        G = x @ x.T
        c_all = G[iu]
        m = float(c_all.max())
        sel = c_all > m - max(20 * radius, 1e-9)
        cnt = int(np.count_nonzero(sel))
        if cnt == 0:
            break
        if cnt > 3000:
            # keep top 3000
            vals_sel = c_all[sel]
            thr = np.partition(vals_sel, cnt - 3000)[cnt - 3000]
            # need to map back; simpler: threshold on c_all
            sel = c_all >= thr
            # if still too many due to ties, subsample
            idx = np.where(sel)[0]
            if len(idx) > 3500:
                keep = np.argsort(c_all[idx])[-3000:]
                nsel = np.zeros_like(sel)
                nsel[idx[keep]] = True
                sel = nsel
        I, J = iu[0][sel], iu[1][sel]
        c = c_all[sel]
        p = len(I)
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        # cols: I*d:(I*d+d), J*d:(J*d+d), last
        cols_I = (I[:, None] * d + np.arange(d)).ravel()
        cols_J = (J[:, None] * d + np.arange(d)).ravel()
        cols_T = np.full(p, n * d)
        cols = np.empty(p * (2 * d + 1), dtype=int)
        cols[0::(2*d+1)] = cols_I[0::d]  # placeholder, build correctly below
        # build directly
        C = np.concatenate([I[:, None]*d + np.arange(d), J[:, None]*d + np.arange(d), np.full((p,1), n*d)], axis=1).ravel()
        V = np.concatenate([x[J], x[I], -np.ones((p,1))], axis=1).ravel()
        try:
            A_ub = coo_matrix((V, (rows, C)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n*d))), shape=(n, nv)).tocsr()
        except Exception:
            radius *= 0.5
            continue
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)]*(n*d)+[(None, None)], method="highs")
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
            x, best = y, v
            radius = min(radius*1.5, 0.05)
        else:
            radius *= 0.5
    return x

def smooth_polish(x, iters=450, T0=0.015, T1=0.0015, lr0=0.01, lr1=0.0015):
    n, d = x.shape
    for k in range(iters):
        frac = k / max(iters-1, 1)
        T = T0 * (T1/T0)**frac
        lr = lr0 * (lr1/lr0)**frac
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        W = np.exp((G - m)/T)
        np.fill_diagonal(W, 0.0)
        grad = W @ x
        dot = np.sum(grad*x, axis=1, keepdims=True)
        gt = grad - dot*x
        rs = W.sum(axis=1, keepdims=True)
        step = gt / np.maximum(rs, 1e-12)
        sn = np.linalg.norm(step, axis=1, keepdims=True)
        step = step / np.maximum(sn, 1.0)
        x = x - lr*step
        x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x

def hard_push(x, steps=60, lr=0.004, tol=3e-4):
    # direct repulsion on near-max pairs
    n, d = x.shape
    for _ in range(steps):
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        W = (G > m - tol).astype(float)
        np.fill_diagonal(W, 0.0)
        rs = W.sum(axis=1, keepdims=True)
        if float(rs.max()) == 0:
            break
        grad = W @ x
        dot = np.sum(grad*x, axis=1, keepdims=True)
        gt = grad - dot*x
        # normalize per point
        nrm = np.linalg.norm(gt, axis=1, keepdims=True)
        gt = gt/np.maximum(nrm, 1e-12)
        # only move points that have contacts
        mask = (rs[:,0] > 0)[:,None]
        x = x - lr*gt*mask
        x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x

def find_voids(x, exclude, n_void=3, n_cand=20000, rng=None):
    n, d = x.shape
    keep_mask = np.ones(n, dtype=bool)
    keep_mask[exclude] = False
    Xk = x[keep_mask]
    cand = rng.normal(size=(n_cand, d))
    cand /= np.linalg.norm(cand, axis=1, keepdims=True)
    # score = max dot to kept set (lower better)
    # chunk to save memory
    best_idx = None
    best_scores = np.full(n_cand, 1e9)
    # compute in chunks
    chunk = 4000
    scores = np.full(n_cand, -2.0)
    for s in range(0, n_cand, chunk):
        e = min(s+chunk, n_cand)
        Gc = cand[s:e] @ Xk.T
        scores[s:e] = Gc.max(axis=1)
    # greedy diverse selection: pick best, then penalize closeness to picked voids?
    chosen = []
    used = np.zeros(n_cand, dtype=bool)
    for _ in range(n_void):
        # prefer low score and far from already chosen
        pen = np.zeros(n_cand)
        for ch in chosen:
            pen += np.maximum(0, cand @ cand[ch] - 0.7)
        tot = scores + 2.0*pen
        tot[used] = 1e9
        # suppress neighbours of picked to enforce diversity via used? just pick argmin
        j = int(np.argmin(tot))
        chosen.append(j)
        # mark close candidates as used to diversify
        dots = cand @ cand[j]
        used[dots > 0.90] = True
        used[j] = True
    return cand[chosen]

def local_tune_point(x, r, iters=120, rng=None):
    # optimize single point r with others frozen to minimize its max dot
    n, d = x.shape
    others = x[np.arange(n)!=r]
    best_v = x[r].copy()
    best_s = float(np.max(others @ best_v))
    cur = best_v.copy()
    sigma = 0.05
    for k in range(iters):
        prop = cur + rng.normal(scale=sigma, size=d)
        prop /= np.linalg.norm(prop)
        s = float(np.max(others @ prop))
        if s < best_s:
            best_s = s
            best_v = prop
            cur = prop
        else:
            # random walk with small prob
            if rng.random() < 0.05:
                cur = prop
        sigma *= 0.985
        # gradient step occasionally
        if k % 10 == 9:
            # gradient of max: move away from worst offender
            j = int(np.argmax(others @ cur))
            w = others[j]
            g = w - (w @ cur)*cur
            cur = cur - 0.02*g
            cur /= np.linalg.norm(cur)
            s2 = float(np.max(others @ cur))
            if s2 < best_s:
                best_s = s2
                best_v = cur.copy()
    x2 = x.copy()
    x2[r] = best_v
    return x2

def solve():
    rng = np.random.default_rng(12345)
    t_start = time.time()
    try:
        xb = np.load("best.npy")
    except Exception:
        xb = None
    try:
        xr = np.load("record.npy")
    except Exception:
        xr = None
    cands = []
    if xb is not None:
        cands.append(np.asarray(xb, dtype=float))
    if xr is not None:
        cands.append(np.asarray(xr, dtype=float))
    if len(cands)==0:
        z = rng.normal(size=(163,4)); z/=np.linalg.norm(z,axis=1,keepdims=True)
        cands.append(z)
    best = None; best_val = 1e9
    for c in cands:
        if c.shape != (163,4):
            continue
        c = c/np.linalg.norm(c,axis=1,keepdims=True)
        v = max_cos(c)
        if v < best_val:
            best_val = v; best = c.copy()
    best = best/np.linalg.norm(best,axis=1,keepdims=True)
    n, d = 163, 4
    # quick deep polish of starting point to get true local opt
    if time.time()-t_start < 165:
        best = lp_polish(best, seconds=4.0)
        best_val = max_cos(best)
    cur = best.copy(); cur_val = best_val
    trial = 0
    sigmas = [0.012, 0.03, 0.06, 0.02, 0.045, 0.008, 0.025, 0.05]
    # temperature for Metropolis on cur
    Tmet0, Tmet1 = 8e-5, 2e-6
    max_trials = 200
    while time.time()-t_start < 138 and trial < max_trials:
        frac = trial/max(40,1)
        Tmet = Tmet0*(Tmet1/Tmet0)**min(frac,1.0)
        sigma = sigmas[trial % len(sigmas)]
        move = trial % 5
        y = cur.copy()
        rm, Gcur = row_maxs(cur)
        mcur = float(rm.max())
        if move == 0:
            y = y + rng.normal(scale=sigma, size=y.shape)
        elif move == 1:
            k = 15
            idx = np.argsort(rm)[-k:]
            noise = rng.normal(scale=sigma*0.25, size=y.shape)
            noise[idx] = rng.normal(scale=sigma*2.5, size=(k,d))
            # also push worst pairs apart a bit
            y = y + noise
        elif move == 2:
            # rattler / low-degree void reinsertion
            degs = contact_degrees(Gcur, mcur, tol=5e-4)
            ratt = np.where(rm < mcur - 2e-4)[0]
            low = np.where(degs <= 1)[0]
            targets = np.unique(np.concatenate([ratt, low])) if len(ratt)+len(low)>0 else np.array([int(np.argmin(degs))])
            if len(targets) > 4:
                # pick 4 worst among them
                order = np.argsort(rm[targets])[:4]
                targets = targets[order]
            if len(targets)==0:
                targets = np.array([int(np.argmax(rm))])
            targets = targets[:3]
            # small shake all
            y = y + rng.normal(scale=sigma*0.4, size=y.shape)
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            voids = find_voids(y, exclude=targets, n_void=len(targets), n_cand=20000, rng=rng)
            for ti, r in enumerate(targets):
                y[r] = voids[ti]
                # local tune
                # build temp array for tuning
                tmp = y.copy()
                # quick local tune with others frozen (use y as context)
                tmp = local_tune_point(tmp, int(r), iters=80, rng=rng)
                y[r] = tmp[r]
        elif move == 3:
            # hub expansion: pick highest-degree hub
            degs = contact_degrees(Gcur, mcur, tol=1e-6)
            h = int(np.argmax(degs + rm*1e-6))
            dots = y @ y[h]
            idx = np.argsort(dots)[-16:]  # hub + 15 neighbours
            # push away from cluster mean
            cmean = y[idx].mean(axis=0); cmean/=np.linalg.norm(cmean)
            for ii in idx:
                # tangential push away from cmean
                push = y[ii] - (y[ii]@cmean)*cmean
                nn = np.linalg.norm(push)
                if nn > 1e-12:
                    push/=nn
                    y[ii] = y[ii] + (-0.6*(y[ii]@cmean - 1.0))*push*sigma*3.0 + rng.normal(scale=sigma, size=d)
                else:
                    y[ii] = y[ii] + rng.normal(scale=sigma*2, size=d)
            # add small noise to rest
            mask = np.ones(n, dtype=bool); mask[idx]=False
            y[mask] += rng.normal(scale=sigma*0.2, size=(int(mask.sum()), d))
        else:
            # contact-pair split informed step + noise
            # top pairs
            iu = np.triu_indices(n,1)
            Gall = y @ y.T
            c_all = Gall[iu]
            top = np.argsort(c_all)[-120:]
            I = iu[0][top]; J = iu[1][top]
            step = np.zeros_like(y)
            for a,b in zip(I,J):
                # push a away from b and vice versa (tangent)
                dab = y[a]@y[b]
                pa = y[a]-dab*y[b]; pb = y[b]-dab*y[a]
                step[a] += pa; step[b] += pb
            # project to tangent
            dot = np.sum(step*y, axis=1, keepdims=True)
            step = step - dot*y
            ns = np.linalg.norm(step, axis=1, keepdims=True)
            step = step/np.maximum(ns, 1e-12)
            has = (ns[:,0]>1e-12)
            y[has] += step[has]*sigma*1.5
            y += rng.normal(scale=sigma*0.5, size=y.shape)
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        # descent: smooth + hard + short LP
        it = 480 if sigma>0.03 else 600
        y = smooth_polish(y, iters=it)
        # occasional hard push for tight max
        if trial % 3 == 0:
            y = hard_push(y, steps=40, lr=0.003, tol=5e-4)
        remain = 138-(time.time()-t_start)
        sec = min(2.2, max(0.7, remain/12))
        if sec > 0.35:
            y = lp_polish(y, seconds=sec)
        v = max_cos(y)
        if v < best_val - 1e-12:
            best = y.copy(); best_val = v
            cur = y.copy(); cur_val = v
        else:
            # Metropolis on cur
            if v < cur_val or rng.random() < np.exp((cur_val - v)/max(Tmet,1e-12)):
                cur = y.copy(); cur_val = v
            # if cur stuck much worse than best, reset to best occasionally
            if cur_val > best_val + 3e-4 and trial % 7 == 6:
                cur = best.copy(); cur_val = best_val
        trial += 1
        if time.time()-t_start > 137:
            break
    # final intensive polish of best
    elapsed = time.time()-t_start
    budget = max(5.0, min(28.0, 170.0-elapsed))
    # annealed smooth at low T then LP
    if budget > 8:
        best = smooth_polish(best, iters=700, T0=0.008, T1=0.0006, lr0=0.006, lr1=0.0008)
        best = lp_polish(best, seconds=budget*0.6)
        best = smooth_polish(best, iters=300, T0=0.002, T1=0.0004, lr0=0.002, lr1=0.0005)
        best = lp_polish(best, seconds=budget*0.4)
    else:
        best = lp_polish(best, seconds=budget)
    best = best/np.linalg.norm(best,axis=1,keepdims=True)
    best = np.nan_to_num(best)
    best /= np.linalg.norm(best,axis=1,keepdims=True)
    return best
