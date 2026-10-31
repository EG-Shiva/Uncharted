import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _mc(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def _nm(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return X / n

def _degrees(X, tol=1e-7):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    m = float(np.max(G))
    B = G > (m - tol)
    deg = np.sum(B, axis=1)
    return deg, m

def _polish(X, steps, b0=30.0, b1=420.0, lr0=0.015, lr1=0.001, mu=0.88, deadline=None):
    x = X.copy()
    vel = np.zeros_like(x)
    lb0 = np.log(b0); lb1 = np.log(b1)
    for s in range(steps):
        if deadline is not None and (s % 40 == 0) and time.time() > deadline:
            break
        f = s / max(1, steps - 1)
        beta = float(np.exp(lb0 + (lb1 - lb0) * f))
        lr = lr0 * (1.0 - f) + lr1 * f
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        cmax = float(np.max(G))
        E = np.exp(np.clip(beta * (G - cmax), -25.0, 0.0))
        np.fill_diagonal(E, 0.0)
        grad = E @ x
        dot = np.sum(grad * x, axis=1, keepdims=True)
        gt = grad - dot * x
        sc = float(np.mean(np.linalg.norm(gt, axis=1))) + 1e-12
        gt = gt / sc
        vel = mu * vel + (1.0 - mu) * gt
        x = x - lr * vel
        x = _nm(x)
    return x

def _lp(x, deadline):
    x = _nm(x.copy())
    n, d = x.shape
    best = _mc(x)
    rad = 1e-3
    iu = np.triu_indices(n, 1)
    while time.time() < deadline and rad > 1e-14:
        C = (x @ x.T)[iu]
        m = float(np.max(C))
        sel = C > m - max(20.0 * rad, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        if p == 0:
            break
        if p > 4000:
            # keep tightest 4000 to bound LP size
            idx = np.argsort(C[sel])[-4000:]
            I, J = I[idx], J[idx]
            cs = C[sel][idx]
            p = len(I)
        else:
            cs = C[sel]
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        Aub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        Aeq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            r = linprog(obj, A_ub=Aub, b_ub=-cs, A_eq=Aeq, b_eq=np.zeros(n),
                        bounds=[(-rad, rad)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            break
        if r.status != 0:
            rad *= 0.5
            continue
        y = _nm(x + r.x[:-1].reshape(n, d))
        v = _mc(y)
        if v < best:
            x, best, rad = y, v, min(rad * 1.5, 0.05)
        else:
            rad *= 0.5
    return x

def _holes(Xk, rng, nsamp=20000, nref=40, nref_iter=60):
    # random sphere samples
    S = rng.normal(size=(nsamp, 4))
    S = S / (np.linalg.norm(S, axis=1, keepdims=True) + 1e-30)
    D = S @ Xk.T
    worst = D.max(axis=1)
    idx = np.argpartition(worst, nref)[:nref]
    cand = S[idx]
    # local hole refinement: minimize softmax max-dot (vectorized)
    beta = 70.0
    lr = 0.06
    for _ in range(nref_iter):
        D2 = cand @ Xk.T
        m = D2.max(axis=1, keepdims=True)
        W = np.exp(np.clip(beta * (D2 - m), -20, 0))
        W /= (W.sum(axis=1, keepdims=True) + 1e-30)
        grad = W @ Xk
        dot = np.sum(grad * cand, axis=1, keepdims=True)
        gt = grad - dot * cand
        cand = cand - lr * gt
        cand = cand / (np.linalg.norm(cand, axis=1, keepdims=True) + 1e-30)
    Df = cand @ Xk.T
    wf = Df.max(axis=1)
    order = np.argsort(wf)
    return cand[order], wf[order]

def _diverse_pick(cand_sorted, k, thresh=0.80):
    picked = []
    for c in cand_sorted:
        ok = True
        for p in picked:
            if float(np.dot(c, p)) > thresh:
                ok = False
                break
        if ok:
            picked.append(c)
        if len(picked) >= k:
            break
    # fill if needed
    i = 0
    while len(picked) < k and i < len(cand_sorted):
        picked.append(cand_sorted[i]); i += 1
    return np.stack(picked, axis=0) if len(picked) > 0 else cand_sorted[:k]

def solve():
    t0 = time.time()
    HARD = t0 + 172.0
    SEARCH = t0 + 140.0
    rng = np.random.default_rng(0)
    cands = []
    for f in ("best.npy", "record.npy"):
        try:
            c = np.load(f).astype(float)
            if c.shape == (168, 4):
                c = _nm(c)
                if np.all(np.isfinite(c)):
                    cands.append(c)
        except Exception:
            pass
    if not cands:
        xb = _nm(rng.normal(size=(168, 4)))
    else:
        vs = [_mc(c) for c in cands]
        xb = cands[int(np.argmin(vs))].copy()
    bv = _mc(xb)
    x_best = xb.copy()
    N0 = 160
    B = 6
    KPC = 4  # children per parent
    # degrees of incumbent for pruning
    try:
        deg, _ = _degrees(xb, tol=2e-7)
    except Exception:
        deg = np.zeros(168)
    order = np.argsort(-deg)
    top_pool = order[:16]
    # build B pruned starts at N0
    beam = []
    # include one deterministic top-8 removal
    removal_sets = []
    removal_sets.append(order[:168 - N0])
    for b in range(1, B):
        # keep top-4 fixed, random 4 among next 12
        fixed = list(order[:4])
        pool = list(order[4:16])
        ch = rng.choice(len(pool), size=(168 - N0 - 4), replace=False)
        rem = np.array(fixed + [pool[i] for i in ch], dtype=int)
        removal_sets.append(rem)
    for rem in removal_sets:
        if time.time() > SEARCH - 30:
            break
        mask = np.ones(168, dtype=bool)
        mask[rem] = False
        Xp = xb[mask].copy()
        # fully polish smaller code (unjammed -> lower max cosine)
        try:
            Xp = _polish(Xp, steps=550, b0=28.0, b1=500.0, lr0=0.016, lr1=0.0009, deadline=SEARCH - 20)
        except Exception:
            pass
        v = _mc(Xp)
        beam.append((v, Xp))
    if not beam:
        beam = [(bv, xb[:N0].copy())]
    beam = sorted(beam, key=lambda t: t[0])[:B]
    # successive reinsertion N0 -> 168
    cur_m = beam[0][1].shape[0]
    for target in range(cur_m, 168):
        if time.time() > SEARCH - 12:
            break
        children = []
        # adapt sample size to remaining time
        remain = SEARCH - 12 - time.time()
        nsamp = 30000 if (remain > 60 and target < 164) else 18000
        for (pv, Xp) in beam:
            if time.time() > SEARCH - 12:
                break
            try:
                cand_sorted, _ = _holes(Xp, rng, nsamp=nsamp, nref=40, nref_iter=50)
            except Exception:
                continue
            try:
                picks = _diverse_pick(cand_sorted, KPC, thresh=0.80)
            except Exception:
                picks = cand_sorted[:KPC]
            for h in picks:
                Xc = np.vstack([Xp, h.reshape(1, 4)])
                Xc = _nm(Xc)
                try:
                    # short 200-step annealed polish at each size
                    Xc2 = _polish(Xc, steps=200, b0=30.0, b1=420.0, lr0=0.014, lr1=0.001, deadline=SEARCH - 10)
                except Exception:
                    Xc2 = Xc
                v = _mc(Xc2)
                children.append((v, Xc2))
                if time.time() > SEARCH - 10:
                    break
        if not children:
            break
        children = sorted(children, key=lambda t: t[0])[:B]
        beam = children
        # track global best if reached 168
        if beam and beam[0][1].shape[0] == 168 and beam[0][0] < bv:
            bv = beam[0][0]
            x_best = beam[0][1].copy()
    # ensure completions to 168 if interrupted early
    full = []
    for (v, Xp) in beam:
        m = Xp.shape[0]
        Xf = Xp.copy()
        while m < 168 and time.time() < SEARCH - 4:
            try:
                cand_sorted, _ = _holes(Xf, rng, nsamp=15000, nref=20, nref_iter=40)
                h = cand_sorted[0]
            except Exception:
                h = rng.normal(size=(4,))
                h = h / (np.linalg.norm(h) + 1e-30)
            Xf = _nm(np.vstack([Xf, h.reshape(1, 4)]))
            m += 1
        if Xf.shape[0] == 168:
            try:
                if time.time() < SEARCH - 2:
                    Xf = _polish(Xf, steps=250, b0=30.0, b1=500.0, lr0=0.012, lr1=0.0008, deadline=SEARCH)
            except Exception:
                pass
            full.append((_mc(Xf), Xf))
    if full:
        full = sorted(full, key=lambda t: t[0])
        for (v, Xf) in full[:3]:
            if v < bv:
                bv = v
                x_best = Xf.copy()
        # final LP polish best few completions
        for i in range(min(3, len(full))):
            if time.time() > HARD - 8:
                break
            v0, Xf0 = full[i]
            if v0 > bv + 0.002:
                continue
            try:
                Xp = _lp(Xf0, min(HARD - 2.0, time.time() + 4.0))
                vp = _mc(Xp)
                if vp < bv:
                    bv = vp
                    x_best = Xp.copy()
                # update full entry
                full[i] = (vp, Xp)
            except Exception:
                pass
        full = sorted(full, key=lambda t: t[0])
        if full[0][0] < bv:
            bv = full[0][0]
            x_best = full[0][1].copy()
    # safety: if nothing improved, LP-polish incumbent briefly
    if x_best.shape != (168, 4):
        x_best = _nm(xb.copy())
    if time.time() < HARD - 3:
        # one final squeeze on the returned config if it is the incumbent
        try:
            if bv >= _mc(xb) - 1e-12:
                Xp = _lp(x_best, min(HARD - 1.0, time.time() + 5.0))
                if _mc(Xp) < bv:
                    x_best = Xp
        except Exception:
            pass
    return _nm(x_best)
