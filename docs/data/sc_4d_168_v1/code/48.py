import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _nm(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return x / n

def _mc(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(np.max(g))

def _lp_polish(x, deadline, rad0=1.5e-3):
    x = _nm(x.copy())
    n, d = x.shape
    try:
        best = _mc(x)
    except Exception:
        return x
    rad = rad0
    iu = np.triu_indices(n, 1)
    while time.time() < deadline and rad > 1e-14:
        try:
            C = (x @ x.T)[iu]
        except Exception:
            break
        m = float(np.max(C))
        sel = C > m - max(20.0 * rad, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        cs = C[sel]
        p = len(I)
        if p == 0:
            break
        if p > 4000:
            # keep tightest only to keep LP fast
            idx = np.argsort(-cs)[:4000]
            I, J, cs = I[idx], J[idx], cs[idx]
            p = len(I)
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            Aub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            Aeq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv); obj[-1] = 1.0
            r = linprog(obj, A_ub=Aub, b_ub=-cs, A_eq=Aeq, b_eq=np.zeros(n),
                        bounds=[(-rad, rad)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            break
        if r.status != 0:
            rad *= 0.5
            continue
        try:
            y = _nm(x + r.x[:-1].reshape(n, d))
            v = _mc(y)
        except Exception:
            rad *= 0.5
            continue
        if v < best - 1e-15:
            x, best, rad = y, v, min(rad * 1.5, 0.05)
        else:
            rad *= 0.5
    return x

def _adam_penalty(X0, t, steps, lr0, lr1, deadline=None, b1=0.9, b2=0.999, eps=1e-8):
    X = _nm(X0.copy())
    n, d = X.shape
    m = np.zeros_like(X)
    v = np.zeros_like(X)
    ll0, ll1 = np.log(lr0), np.log(lr1)
    best_pen = np.inf
    Xbest = X.copy()
    best_max = np.inf
    last_imp = 0
    for s in range(1, steps + 1):
        if deadline is not None and s % 50 == 0 and time.time() > deadline:
            break
        f = s / max(1, steps)
        lr = float(np.exp(ll0 + (ll1 - ll0) * f))
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        R1 = G - t
        np.maximum(R1, 0.0, out=R1)
        np.fill_diagonal(R1, 0.0)
        R2 = G - t + 0.002
        np.maximum(R2, 0.0, out=R2)
        np.fill_diagonal(R2, 0.0)
        # penalty sum_{i<j}
        pen = (np.sum(R1 * R1) + 0.1 * np.sum(R2 * R2)) * 0.5
        # track
        if pen < best_pen - 1e-18:
            best_pen = float(pen)
            Xbest = X.copy()
            last_imp = s
            try:
                best_max = float(np.max(G))
            except Exception:
                pass
        # plateau early stop: if no improvement for long, break
        if s - last_imp > 600 and s > 800:
            break
        W = 2.0 * R1 + 0.2 * R2
        grad = W @ X
        dot = np.sum(grad * X, axis=1, keepdims=True)
        gt = grad - dot * X
        m = b1 * m + (1.0 - b1) * gt
        v = b2 * v + (1.0 - b2) * (gt * gt)
        bc1 = 1.0 - b1 ** s
        bc2 = 1.0 - b2 ** s
        mh = m / (bc1 + 1e-30)
        vh = v / (bc2 + 1e-30)
        step = lr * mh / (np.sqrt(vh) + eps)
        # project step to tangent
        sd = np.sum(step * X, axis=1, keepdims=True)
        st = step - sd * X
        X = _nm(X - st)
    # final eval of X (may be better than Xbest in max though worse in pen)
    try:
        Gf = X @ X.T
        np.fill_diagonal(Gf, -2.0)
        R1f = np.maximum(Gf - t, 0.0)
        R2f = np.maximum(Gf - t + 0.002, 0.0)
        penf = (np.sum(R1f * R1f) + 0.1 * np.sum(R2f * R2f)) * 0.5
        if penf < best_pen:
            best_pen = float(penf)
            Xbest = X.copy()
            best_max = float(np.max(Gf))
    except Exception:
        pass
    return Xbest, float(best_pen), float(best_max)

def _hub_scores(X, t):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    R1 = np.maximum(G - t, 0.0)
    R2 = np.maximum(G - t + 0.002, 0.0)
    np.fill_diagonal(R1, 0.0)
    np.fill_diagonal(R2, 0.0)
    score = np.sum(R1 * R1, axis=1) + 0.1 * np.sum(R2 * R2, axis=1)
    cnt = np.sum(G > t, axis=1).astype(float)
    cnt2 = np.sum(G > t - 0.002, axis=1).astype(float)
    return G, score, cnt, cnt2

def _perturb_hubs(X, t, rng, k=8):
    try:
        G, score, cnt, cnt2 = _hub_scores(X, t)
    except Exception:
        return _nm(X + 0.02 * rng.standard_normal(X.shape))
    # rank hubs by score then count
    order = np.argsort(-(score + 1e-9 * cnt))
    # number of hubs: at least 4, up to k, but only those with violation
    hubs = []
    for idx in order:
        if cnt[idx] > 0 or score[idx] > 0:
            hubs.append(int(idx))
        if len(hubs) >= k:
            break
    if len(hubs) == 0:
        hubs = order[:4].tolist()
    Y = X.copy()
    for i in hubs:
        theta = float(rng.uniform(0.05, 0.15))
        r = rng.standard_normal(4)
        r = r - float(np.dot(r, Y[i])) * Y[i]
        n = float(np.linalg.norm(r))
        if n < 1e-12:
            continue
        r /= n
        Y[i] = np.cos(theta) * Y[i] + np.sin(theta) * r
    return _nm(Y)

def solve():
    t0 = time.time()
    HARD = t0 + 172.0
    SEARCH = t0 + 148.0
    rng = np.random.RandomState(0)
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
        xb = _nm(rng.standard_normal((168, 4)))
        bv = _mc(xb)
    else:
        vs = [_mc(c) for c in cands]
        bi = int(np.argmin(vs))
        xb = cands[bi].copy()
        bv = float(vs[bi])
    x_best = xb.copy()
    # build descending target list just below current best
    fixed = [0.8592, 0.8590, 0.8587, 0.8584, 0.8580, 0.8575]
    dyn = [bv - 1e-05, bv - 3e-05, bv - 6e-05]
    ts = []
    for tt in dyn + fixed:
        if tt < bv - 1e-12:
            ts.append(float(tt))
    # strictly decreasing, dedup
    ts2 = []
    for tt in ts:
        if not ts2 or tt < ts2[-1] - 1e-12:
            ts2.append(tt)
    # keep at most 6
    ts = ts2[:6]
    if len(ts) == 0:
        ts = [bv - 1e-05]
    x_stage = x_best.copy()
    # homotopy stages
    for ti, t in enumerate(ts):
        if time.time() > SEARCH - 2.0:
            break
        # attempts per stage depends on remaining time
        n_att = 3 if ti < 2 else 2
        best_pen_stage = np.inf
        best_X_stage = x_stage.copy()
        X_init = x_stage.copy()
        success = False
        for att in range(n_att):
            if time.time() > SEARCH - 1.5:
                break
            # steps: larger for early stages
            steps = 1400 if ti < 2 else 1100
            # shrink steps if low time
            rem = SEARCH - time.time()
            if rem < 20:
                steps = min(steps, 600)
            bud = min(time.time() + 12.0, SEARCH)
            try:
                Xr, pen, mx = _adam_penalty(X_init, t, steps=steps, lr0=0.008, lr1=0.0008, deadline=bud)
            except Exception:
                continue
            # update stage best
            if pen < best_pen_stage:
                best_pen_stage = pen
                best_X_stage = Xr.copy()
            # diagnostics: violating count
            try:
                G = Xr @ Xr.T
                np.fill_diagonal(G, -2.0)
                viol = int(np.sum(G > t + 1e-9) // 2)
                mxv = float(np.max(G))
            except Exception:
                viol = 10**9
                mxv = np.inf
            # near-zero violation ? -> LP polish and record
            if viol == 0 or mxv <= t + 3e-06:
                try:
                    Xp = _lp_polish(Xr, min(time.time() + 2.0, HARD - 3.0))
                    vp = _mc(Xp)
                except Exception:
                    Xp, vp = Xr, mxv
                if vp < bv:
                    bv = vp
                    xb = Xp.copy()
                    x_best = xb.copy()
                # advance stage init to polished point
                x_stage = Xp.copy() if 'Xp' in locals() else Xr.copy()
                success = True
                break
            else:
                # opportunistic: if max improved over global best, LP-polish
                if mxv < bv - 2e-07 and time.time() < HARD - 5.0:
                    try:
                        Xp = _lp_polish(Xr, min(time.time() + 1.2, HARD - 3.0))
                        vp = _mc(Xp)
                        if vp < bv:
                            bv = vp
                            xb = Xp.copy()
                            x_best = xb.copy()
                    except Exception:
                        pass
                # restart from best-penalty point with hub rotation
                try:
                    base = Xr if pen <= best_pen_stage + 1e-18 else best_X_stage
                except Exception:
                    base = Xr
                k = 8 if att == 0 else 12
                try:
                    X_init = _perturb_hubs(base, t, rng, k=k)
                except Exception:
                    X_init = _nm(base + 0.03 * rng.standard_normal(base.shape))
        if success:
            continue
        else:
            # failed to reach t: keep best-penalty point as init for next?
            # Do NOT lower t further if far from feasible; break to refinement
            # Check if stage best still improves global max meaningfully
            try:
                mxb = _mc(best_X_stage)
            except Exception:
                mxb = np.inf
            x_stage = best_X_stage.copy()
            # if max worse than best by margin, revert to global best
            # if this stage's t is much lower than achievable, stop homotopy
            # Heuristic: if viol still large (>200) then t too low -> break
            try:
                G2 = best_X_stage @ best_X_stage.T
                np.fill_diagonal(G2, -2.0)
                viol2 = int(np.sum(G2 > t + 1e-9) // 2)
            except Exception:
                viol2 = 10**9
            if viol2 > 300 and ti >= 1:
                break
            # otherwise continue to next t only if we made progress (max reduced)
            # if no progress, break
            if mxb > bv - 1e-06 and viol2 > 50:
                # try one more aggressive hub restart at same t? already done
                # move on only if time allows but keep t (don't go lower)
                break
            # else continue loop (next lower t) from stage best
            continue
    # final intensive squeeze of global best + stage best
    if time.time() < HARD - 2.0:
        try:
            # LP polish current best
            xb = _lp_polish(x_best, HARD - 1.0)
            bv = _mc(xb)
            x_best = xb.copy()
        except Exception:
            pass
    # if stage point is competitive, polish it too
    if time.time() < HARD - 4.0:
        try:
            if x_stage.shape == (168, 4):
                vs = _mc(_nm(x_stage))
                if vs < bv + 0.002:
                    Xp = _lp_polish(_nm(x_stage), HARD - 1.0)
                    vp = _mc(Xp)
                    if vp < bv:
                        bv = vp
                        xb = Xp.copy()
        except Exception:
            pass
    xb = _nm(xb)
    return xb
