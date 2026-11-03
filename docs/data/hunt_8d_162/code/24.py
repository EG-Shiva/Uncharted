import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _norm(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n = np.maximum(n, 1e-12)
    return X / n

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _contacts(X, thr=1e-9):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    gm = float(G.max())
    C = (G > gm - thr).sum(axis=1)
    return gm, C, G

def _adam_smooth(X, iters, a0, a1, lr0, lr1, noise0, noise1, rng, freeze=None):
    n, d = X.shape
    m = np.zeros_like(X)
    v = np.zeros_like(X)
    b1, b2, eps = 0.9, 0.999, 1e-8
    if freeze is not None:
        fr = np.zeros(n, dtype=bool)
        fr[freeze] = True
    else:
        fr = None
    for t in range(iters):
        f = t / max(1, iters - 1)
        if a0 == a1:
            a = float(a0)
        else:
            a = float(a0 * ((a1 / a0) ** f))
        if lr0 == lr1:
            lr = float(lr0)
        else:
            lr = float(lr0 * ((lr1 / lr0) ** f))
        if noise0 <= 0:
            nm = 0.0
        else:
            if noise1 is None or noise1 <= 0:
                nm = float(noise0 * (1.0 - f))
            else:
                nm = float(noise0 * ((noise1 / noise0) ** f))
        G = X @ X.T
        np.fill_diagonal(G, -10.0)
        Gm = float(G.max())
        W = np.exp(a * (G - Gm))
        np.fill_diagonal(W, 0.0)
        S = float(W.sum())
        if not np.isfinite(S) or S == 0:
            break
        H = W @ X
        WG = (W * G).sum(axis=1)
        Grad = H - WG[:, None] * X
        if fr is not None:
            Grad[fr] = 0.0
        m = b1 * m + (1.0 - b1) * Grad
        v = b2 * v + (1.0 - b2) * (Grad * Grad)
        bc1 = 1.0 - b1 ** (t + 1)
        bc2 = 1.0 - b2 ** (t + 1)
        mh = m / bc1
        vh = v / bc2
        step = lr * mh / (np.sqrt(vh) + eps)
        rad = (step * X).sum(axis=1, keepdims=True)
        step = step - rad * X
        if fr is not None:
            step[fr] = 0.0
        X = X - step
        if nm > 0:
            N = rng.randn(n, d)
            nd = (N * X).sum(axis=1, keepdims=True)
            N = N - nd * X
            if fr is not None:
                N[fr] = 0.0
            X = X + nm * N
        X = _norm(X)
    return X

def _lp_polish(X, seconds, t_deadline=None):
    X = _norm(np.array(X, dtype=float))
    n, d = X.shape
    best = _max_cos(X)
    radius = 1.5e-3
    start = time.time()
    iu0 = np.triu_indices(n, 1)
    iuI = iu0[0]; iuJ = iu0[1]
    ar = np.arange(d)
    rep = np.repeat(np.arange(n), d)
    while time.time() - start < seconds:
        if t_deadline is not None and time.time() > t_deadline:
            break
        if radius < 1e-14:
            break
        Gfull = X @ X.T
        cfull = Gfull[iuI, iuJ]
        m = float(cfull.max())
        sel = cfull > m - max(20.0 * radius, 1e-9)
        cnt = int(sel.sum())
        if cnt == 0:
            radius *= 0.5
            continue
        if cnt > 2800:
            idx = np.argpartition(-cfull, 2800)[:2800]
            mask = np.zeros_like(sel)
            mask[idx] = True
            sel = sel & mask
        I = iuI[sel]; J = iuJ[sel]
        p = len(I)
        if p == 0:
            break
        cij = (X[I] * X[J]).sum(axis=1)
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + ar, J[:, None] * d + ar, np.full((p, 1), n * d)], axis=1).ravel()
        vals = np.concatenate([X[J], X[I], -np.ones((p, 1))], axis=1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((X.ravel(), (rep, np.arange(n * d))), shape=(n, nv)).tocsr()
        except Exception:
            radius *= 0.5
            continue
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-cij, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                          method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = X + res.x[:-1].reshape(n, d)
        y = _norm(y)
        vv = _max_cos(y)
        if vv < best:
            X, best = y, vv
            radius = min(radius * 1.6, 0.05)
        else:
            radius *= 0.5
    return X

def _refine_hole(v0, X_rem, beta=90.0, iters=220, lr0=0.12):
    vv = v0 / max(1e-12, float(np.linalg.norm(v0)))
    for it in range(iters):
        dts = X_rem @ vv
        mx = float(dts.max())
        w = np.exp(beta * (dts - mx))
        s = float(w.sum())
        if not np.isfinite(s) or s <= 0:
            break
        g = (w / s) @ X_rem
        gt = g - float(g @ vv) * vv
        nn = float(np.linalg.norm(gt))
        if nn < 1e-12:
            break
        gt /= nn
        step = lr0 * (0.12 ** (it / float(max(1, iters))))
        vv = vv - step * gt
        vv = vv / max(1e-12, float(np.linalg.norm(vv)))
    return vv

def _best_hole(X_rem, rng, n_cand=22000, beta=90.0):
    d = X_rem.shape[1]
    C = _norm(rng.randn(n_cand, d))
    dots = C @ X_rem.T
    mm = dots.max(axis=1)
    order = np.argpartition(mm, 8)[:8]
    best_v = C[order[0]]
    best_s = 1e9
    for k in order:
        vv = _refine_hole(C[k], X_rem, beta=beta, iters=220, lr0=0.10)
        s = float((X_rem @ vv).max())
        if s < best_s:
            best_s = s
            best_v = vv
    return best_v

def _sequential_reinsert(X, victims, rng, n_cand=22000):
    n = X.shape[0]
    mask = np.ones(n, dtype=bool)
    mask[np.array(victims, dtype=int)] = False
    keep = X[mask]
    # reinsert one by one greedily
    for _ in victims:
        h = _best_hole(keep, rng, n_cand=n_cand)
        keep = np.vstack([keep, h.reshape(1, -1)])
    return _norm(keep)

def solve():
    t0 = time.time()
    deadline = t0 + 172.0
    rng = np.random.RandomState(17)
    Xb = None
    sb = 1e9
    for fn in ["best.npy", "record.npy"]:
        try:
            vv = np.load(fn).astype(float)
            if vv.shape == (162, 8):
                vv = _norm(vv)
                s = _max_cos(vv)
                if s < sb:
                    sb = s
                    Xb = vv.copy()
        except Exception:
            pass
    if Xb is None:
        Xb = _norm(rng.randn(162, 8))
        sb = _max_cos(Xb)
    n, d = 162, 8
    # short initial lock
    if deadline - time.time() > 10:
        try:
            Xp = _lp_polish(Xb, 3.0, t_deadline=deadline - 8)
            sp = _max_cos(Xp)
            if sp < sb:
                Xb, sb = Xp, sp
        except Exception:
            pass
    X_best = Xb.copy(); s_best = float(sb)
    X_cur = Xb.copy(); s_cur = float(sb)
    elites = [(s_best, X_best.copy())]
    def push_elite(s, X):
        elites.append((float(s), X.copy()))
        elites.sort(key=lambda z: z[0])
        # keep top 10 diverse (unique by score diff)
        del elites[10:]
    trial = 0
    # main basin hopping
    while time.time() < t0 + 112 and trial < 800:
        if time.time() > deadline - 55:
            break
        trial += 1
        # choose base: 55% best, 25% cur, 20% random elite
        r = rng.rand()
        if r < 0.55:
            base = X_best
        elif r < 0.80:
            base = X_cur
        else:
            if len(elites) > 1:
                base = elites[rng.randint(len(elites))][1]
            else:
                base = X_best
        Xt = base.copy()
        op = rng.randint(7)
        try:
            if op == 0:
                # sequential hole reinsert of overloaded + rattler
                gm, C, G = _contacts(Xt)
                order_hi = np.argsort(-C)
                order_lo = np.argsort(C)
                k_hi = 1 + (trial % 2)
                victims = order_hi[:k_hi].tolist()
                # add one loose point
                victims.append(int(order_lo[trial % n]))
                victims = list(dict.fromkeys(victims))[:3]
                Xt = _sequential_reinsert(Xt, victims, rng, n_cand=22000)
                Xt = _norm(Xt + rng.randn(n, d) * 0.0025)
                Xt = _adam_smooth(Xt, 550, 25, 480, 0.004, 0.001, 0.0018, 0.0, rng)
                Xt = _adam_smooth(Xt, 280, 150, 1100, 0.0015, 0.0005, 0.0, 0.0, rng)
            elif op == 1:
                # worst-pair surgery: move one endpoint to hole
                G = Xt @ Xt.T
                np.fill_diagonal(G, -2.0)
                iu = np.triu_indices(n, 1)
                cf = G[iu]
                k = 4
                top = np.argpartition(-cf, k)[:k]
                q = top[rng.randint(k)]
                i = int(iu[0][q]); j = int(iu[1][q])
                # delete i, find hole
                mask = np.ones(n, dtype=bool); mask[i] = False
                hole = _best_hole(Xt[mask], rng, n_cand=22000)
                Xt[i] = hole
                Xt = _norm(Xt + rng.randn(n, d) * 0.002)
                Xt = _adam_smooth(Xt, 600, 22, 500, 0.004, 0.001, 0.0015, 0.0, rng)
                Xt = _adam_smooth(Xt, 280, 150, 1100, 0.0015, 0.0005, 0.0, 0.0, rng)
            elif op == 2:
                # cap-local rearrange: freeze outside, anneal inside
                u = rng.randn(d); u /= np.linalg.norm(u)
                proj = Xt @ u
                # random cap size 15-35
                frac = 0.10 + 0.12 * rng.rand()
                kcap = int(n * frac)
                idx = np.argsort(-proj)[:kcap]
                # kick cap
                R = rng.randn(d, d) * 0.05
                R = R - R.T
                for ii in idx:
                    Xt[ii] = Xt[ii] + (R @ Xt[ii]) + rng.randn(d) * 0.012
                Xt = _norm(Xt)
                freeze = np.ones(n, dtype=bool)
                freeze[idx] = False
                fr_idx = np.where(freeze)[0]
                Xt = _adam_smooth(Xt, 550, 24, 520, 0.005, 0.0012, 0.002, 0.0, rng, freeze=fr_idx)
                Xt = _adam_smooth(Xt, 350, 120, 1000, 0.002, 0.0006, 0.0, 0.0, rng)
            elif op == 3:
                # global soft kick + anneal with random sharpness
                amt = 0.008 + 0.020 * rng.rand()
                Xt = _norm(Xt + rng.randn(n, d) * amt)
                a0 = 18 + 10 * rng.rand()
                a1 = 400 + 300 * rng.rand()
                Xt = _adam_smooth(Xt, 650, a0, a1, 0.0045, 0.001, 0.0025, 0.0, rng)
                Xt = _adam_smooth(Xt, 280, 150, 1100, 0.0015, 0.0005, 0.0, 0.0, rng)
            elif op == 4:
                # multi-point random shake
                k = 8 + rng.randint(8)
                idx = rng.choice(n, k, replace=False)
                Xt[idx] = _norm(Xt[idx] + rng.randn(k, d) * (0.10 + 0.10 * rng.rand()))
                Xt = _adam_smooth(Xt, 650, 20, 450, 0.0045, 0.001, 0.0025, 0.0, rng)
                Xt = _adam_smooth(Xt, 280, 150, 1100, 0.0015, 0.0005, 0.0, 0.0, rng)
            elif op == 5:
                # worst-pair repulsion push then anneal
                G = Xt @ Xt.T
                np.fill_diagonal(G, -2.0)
                iu = np.triu_indices(n, 1)
                cf = G[iu]
                k = 6
                top = np.argpartition(-cf, k)[:k]
                for qq in top:
                    i = int(iu[0][qq]); j = int(iu[1][qq])
                    di = Xt[i] - Xt[j]
                    amt = 0.02 + 0.02 * rng.rand()
                    Xt[i] = Xt[i] + di * amt
                    Xt[j] = Xt[j] - di * amt
                Xt = _norm(Xt)
                Xt = _adam_smooth(Xt, 600, 28, 550, 0.004, 0.001, 0.002, 0.0, rng)
                Xt = _adam_smooth(Xt, 280, 150, 1100, 0.0015, 0.0005, 0.0, 0.0, rng)
            else:
                # double-hole sequential with larger sample
                gm, C, G = _contacts(Xt)
                order_hi = np.argsort(-C)
                victims = order_hi[:2].tolist()
                Xt = _sequential_reinsert(Xt, victims, rng, n_cand=26000)
                Xt = _norm(Xt + rng.randn(n, d) * 0.003)
                Xt = _adam_smooth(Xt, 550, 25, 500, 0.004, 0.001, 0.0015, 0.0, rng)
                Xt = _adam_smooth(Xt, 300, 160, 1200, 0.0015, 0.0005, 0.0, 0.0, rng)
        except Exception:
            continue
        try:
            s_try = _max_cos(Xt)
        except Exception:
            continue
        if not np.isfinite(s_try):
            continue
        # sideways walk update of cur
        # accept if better than cur or with prob / temp
        temp = 0.0012 * max(0.0, 1.0 - (time.time() - t0) / 120.0) + 0.0002
        if s_try < s_cur or (s_try < s_cur + temp and rng.rand() < 0.35):
            X_cur, s_cur = Xt.copy(), float(s_try)
        if s_try < s_best - 1e-12:
            # lock new basin
            budget = min(5.0, max(2.0, (deadline - time.time()) / 10.0))
            if time.time() + budget < deadline - 8:
                try:
                    Xt2 = _lp_polish(Xt, budget, t_deadline=deadline - 8)
                    s2 = _max_cos(Xt2)
                except Exception:
                    Xt2, s2 = Xt, s_try
                if s2 < s_best:
                    X_best, s_best = Xt2.copy(), float(s2)
                    X_cur, s_cur = Xt2.copy(), float(s2)
                    push_elite(s2, Xt2)
                elif s_try < s_best:
                    X_best, s_best = Xt.copy(), float(s_try)
                    push_elite(s_try, Xt)
                else:
                    push_elite(s2, Xt2)
            else:
                X_best, s_best = Xt.copy(), float(s_try)
                X_cur, s_cur = Xt.copy(), float(s_try)
                push_elite(s_try, Xt)
        elif s_try < s_best + 0.004:
            push_elite(s_try, Xt)
    # polish elites
    elites.sort(key=lambda z: z[0])
    uniq = elites[:7]
    for _, Xc in list(uniq):
        if time.time() > deadline - 14:
            break
        remain = deadline - time.time() - 4
        if remain < 3:
            break
        budget = min(8.0, max(2.5, remain / 4.0))
        try:
            Xp = _lp_polish(np.array(Xc, copy=True), budget, t_deadline=deadline - 2)
            sp = _max_cos(Xp)
            if sp < s_best:
                X_best, s_best = Xp.copy(), float(sp)
        except Exception:
            pass
    if time.time() < deadline - 3:
        budget = min(11.0, deadline - time.time() - 1.5)
        if budget > 2:
            try:
                Xp = _lp_polish(X_best, budget, t_deadline=deadline - 1)
                sp = _max_cos(Xp)
                if sp < s_best:
                    X_best, s_best = Xp, sp
            except Exception:
                pass
    X_best = _norm(X_best)
    return X_best
