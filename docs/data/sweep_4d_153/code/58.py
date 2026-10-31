import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())


def contact_counts(x, m, thresh=1e-9):
    n = x.shape[0]
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    mask = G > m - thresh
    # count per point (each pair counted once per endpoint)
    cnt = (mask).sum(axis=1)
    # mask is symmetric, sum gives contacts per point
    return cnt, G


def smooth_weighted(x, iters, k0, k1, lr0, lr1, mu, rattler, wmult=5.0):
    n, d = x.shape
    v = np.zeros_like(x)
    for t in range(iters):
        frac = t / max(1, iters - 1)
        k = k0 * (k1 / k0) ** frac
        lr = lr0 * (lr1 / lr0) ** frac
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        moff = G.max()
        W = np.exp(k * (G - moff))
        np.fill_diagonal(W, 0.0)
        if rattler is not None and 0 <= rattler < n:
            W[rattler, :] *= wmult
            W[:, rattler] *= wmult
            # keep zero diagonal
            W[rattler, rattler] = 0.0
        rs = W.sum(axis=1, keepdims=True) + 1e-12
        grad = (W @ x) / rs
        dot = np.sum(grad * x, axis=1, keepdims=True)
        gt = grad - dot * x
        v = mu * v + gt
        vc = v / (1.0 - mu ** (t + 1)) if mu > 0 else v
        x = x - lr * vc
        x = x / np.linalg.norm(x, axis=1, keepdims=True)
    return x


def lp_polish(x, seconds):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = max_cos(x)
    radius, start = 1e-3, time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        c = (x @ x.T)[iu]
        m = c.max()
        sel = c > m - max(20 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p, nv = len(I), n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv)
        obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c[sel], A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        vv = max_cos(y)
        if vv < best:
            x, best, radius = y, vv, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x


def find_holes(x, rng, n_rand=50000, n_top=200, n_holes=5):
    n, d = x.shape
    # at least 50000 random sphere candidates
    chunk = 10000
    all_c = []
    all_m = []
    # generate in chunks to bound memory
    cands = rng.standard_normal((n_rand, d))
    cands /= np.linalg.norm(cands, axis=1, keepdims=True)
    # compute max dot in chunks
    md = np.empty(n_rand)
    cs = 5000
    for s in range(0, n_rand, cs):
        e = min(n_rand, s + cs)
        dots = cands[s:e] @ x.T
        md[s:e] = dots.max(axis=1)
    order = np.argsort(md)[:n_top]
    Y = cands[order].copy()  # (n_top, d)
    # local tangent-gradient minimisation of max_j <y,x_j>
    # vectorized argmax gradient descent on sphere
    steps = 70
    for t in range(steps):
        frac = t / max(1, steps - 1)
        step = 0.05 * (0.1) ** frac  # 0.05 -> 0.005
        D = Y @ x.T  # (n_top, n)
        jstar = D.argmax(axis=1)  # (n_top,)
        Xs = x[jstar]  # (n_top, d)
        ds = np.sum(Y * Xs, axis=1, keepdims=True)
        Gt = Xs - ds * Y
        Y = Y - step * Gt
        Y /= np.linalg.norm(Y, axis=1, keepdims=True)
    D2 = Y @ x.T
    md2 = D2.max(axis=1)
    o2 = np.argsort(md2)
    Y = Y[o2]
    md2 = md2[o2]
    # pick five largest well-separated holes (greedy separation)
    picked = []
    for sep in [0.90, 0.95, 0.97, 0.99]:
        picked = []
        for i in range(len(Y)):
            y = Y[i]
            ok = True
            for q in picked:
                if float(y @ q) > sep:
                    ok = False
                    break
            if ok:
                picked.append(y)
                if len(picked) >= n_holes:
                    break
        if len(picked) >= n_holes:
            break
    # if still short, fill with next best
    if len(picked) < n_holes:
        for i in range(len(Y)):
            y = Y[i]
            found = False
            for q in picked:
                if float(y @ q) > 0.999:
                    found = True
                    break
            if not found:
                # check duplicate
                dup = False
                for q in picked:
                    if np.allclose(y, q):
                        dup = True
                        break
                if not dup:
                    picked.append(y)
            if len(picked) >= n_holes:
                break
    return np.array(picked[:n_holes])


def solve():
    t0 = time.time()
    HARD_END = t0 + 168.0
    rng = np.random.default_rng(12345)
    n, d = 153, 4
    try:
        inc = np.load("best.npy")
    except Exception:
        try:
            inc = np.load("record.npy")
        except Exception:
            inc = rng.standard_normal((n, d))
    inc = np.asarray(inc, dtype=float)
    if inc.shape != (n, d):
        tmp = rng.standard_normal((n, d))
        inc = tmp
    inc = inc / np.linalg.norm(inc, axis=1, keepdims=True)
    inc_val = max_cos(inc)

    # identify zero-contact point (rattler) and <=4-contact points
    m0 = inc_val
    cnt0, G0 = contact_counts(inc, m0, thresh=1e-9)
    # rattler: minimal per-point max (0 contacts => far below max)
    per_max = G0.max(axis=1)
    # gap to global max
    gap = m0 - per_max
    rattler = int(np.argmax(gap))
    # verify it is really a rattler (gap large); if not, take min count point
    if gap[rattler] < 1e-7:
        # fallback: point with fewest contacts
        rattler = int(np.argmin(cnt0))
    low_pts = np.where(cnt0 <= 4)[0]

    # find five largest well-separated holes (with local refinement)
    try:
        holes = find_holes(inc, rng, n_rand=50000, n_top=200, n_holes=5)
    except Exception:
        holes = None
    if holes is None or len(holes) == 0:
        # fallback: return polished incumbent briefly
        try:
            xp = lp_polish(inc, min(8.0, HARD_END - time.time() - 2))
            if max_cos(xp) < inc_val:
                return xp / np.linalg.norm(xp, axis=1, keepdims=True)
        except Exception:
            pass
        return inc

    # trial distances between 0.01 and 0.08
    trial_dists = [0.015, 0.03, 0.05, 0.08]

    best_kept = None
    best_kept_val = 1e9

    # strict internal deadlines: reserve 20s at end for final polish
    # iterate holes x dists
    trial_id = 0
    for hi in range(len(holes)):
        h = holes[hi] / np.linalg.norm(holes[hi])
        # six nearest points to this hole (using incumbent geometry)
        dots_inc = inc @ h
        # exclude rattler from neighbours
        idx_all = np.argsort(-dots_inc)
        neigh = []
        for ii in idx_all:
            if ii == rattler:
                continue
            neigh.append(int(ii))
            if len(neigh) >= 6:
                break
        for s in trial_dists:
            trial_id += 1
            # deadline check
            if time.time() > HARD_END - 22:
                break
            x = inc.copy()
            # place rattler into hole
            x[rattler] = h
            # simultaneously displace six nearest along tangent away from hole
            for nb in neigh:
                p = x[nb]
                dh = float(p @ h)
                vec = h - dh * p  # tangent at p pointing toward hole
                nv = float(np.linalg.norm(vec))
                if nv < 1e-12:
                    continue
                unit = vec / nv
                p2 = p - s * unit
                p2 /= np.linalg.norm(p2)
                x[nb] = p2
            # annealed exponential repulsion with rattler weight x5
            # vary schedule slightly per trial for diversity (deterministic)
            k0 = 12.0
            k1 = 280.0 + (trial_id % 3) * 40.0
            lr0 = 0.04
            lr1 = 0.002
            mu = 0.85
            iters = 900
            # reduce iters if very late
            if time.time() > HARD_END - 45:
                iters = 500
            try:
                xo = smooth_weighted(x, iters, k0, k1, lr0, lr1, mu, rattler, wmult=5.0)
            except Exception:
                continue
            if time.time() > HARD_END - 20:
                # no time for LP, evaluate directly
                v = max_cos(xo)
                # check rattler contacts within 1e-5
                dr = float((xo[rattler] @ xo.T).max()) if False else None
                # compute rattler max excluding self
                dots_r = xo[rattler] @ xo.T
                dots_r[rattler] = -2.0
                rmax = float(dots_r.max())
                nclose = int(np.sum(dots_r > v - 1e-5))
                if (v - rmax) < 1e-5 and nclose >= 1 and v < best_kept_val:
                    best_kept_val = v
                    best_kept = xo.copy()
                continue
            try:
                # short trust-region LP polishing per trial
                xp = lp_polish(xo, 2.0)
            except Exception:
                xp = xo
            v = max_cos(xp)
            dots_r = xp[rattler] @ xp.T
            dots_r[rattler] = -2.0
            rmax = float(dots_r.max())
            nclose = int(np.sum(dots_r > v - 1e-5))
            # retain only if rattler has contacts within 1e-5 of final maximum
            if (v - rmax) > 1e-5 or nclose < 1:
                continue
            if v < best_kept_val:
                best_kept_val = v
                best_kept = xp.copy()
            if time.time() > HARD_END - 22:
                break
        if time.time() > HARD_END - 22:
            break

    # always fall back to incumbent if no trial finishes / no kept result
    if best_kept is None:
        # quick polish of incumbent with remaining time (bounded)
        remain = HARD_END - time.time()
        if remain > 3:
            try:
                xp = lp_polish(inc, min(remain - 1, 10.0))
                if max_cos(xp) < inc_val:
                    xp /= np.linalg.norm(xp, axis=1, keepdims=True)
                    return xp
            except Exception:
                pass
        return inc / np.linalg.norm(inc, axis=1, keepdims=True)

    # final longer polish of the best kept with remaining time
    remain = HARD_END - time.time()
    if remain > 3:
        try:
            xf = lp_polish(best_kept, min(remain - 1, 15.0))
            vf = max_cos(xf)
            # re-check rattler condition after final polish (rattler index may have moved role,
            # but we track same index; require condition still holds, else keep previous)
            dr = xf[rattler] @ xf.T
            dr[rattler] = -2.0
            rm = float(dr.max())
            if (vf - rm) < 1e-5:
                if vf < best_kept_val:
                    best_kept_val = vf
                    best_kept = xf
        except Exception:
            pass
    # return better of kept and incumbent
    if inc_val < best_kept_val and inc_val <= max_cos(best_kept):
        return inc / np.linalg.norm(inc, axis=1, keepdims=True)
    best_kept /= np.linalg.norm(best_kept, axis=1, keepdims=True)
    # strict feasibility: ensure lower than incumbent or at least valid
    return best_kept
