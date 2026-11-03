import time
import numpy as np

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _norm(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return X / n

def _p_continuation(X0, steps_per_stage, p_list, lr0, lr1, shift, seed_off):
    # Riemannian Adam on shifted p-norm with log-stable weights
    n, d = X0.shape
    X = _norm(X0.copy())
    iu0, iu1 = np.triu_indices(n, 1)
    m = np.zeros_like(X)
    v = np.zeros_like(X)
    b1, b2, eps = 0.9, 0.999, 1e-8
    total = len(p_list) * steps_per_stage
    tglob = 0
    # for monitoring
    best_seen = _max_cos(X)
    best_X = X.copy()
    for pi, p in enumerate(p_list):
        # rescale step by resetting moments at each p increase
        m[:] = 0.0
        v[:] = 0.0
        tt = 0
        for s in range(steps_per_stage):
            frac = tglob / max(1, total - 1)
            lr = lr0 * (lr1 / lr0) ** frac
            G = X @ X.T
            c = G[iu0, iu1]
            sc = c - shift
            # only positive part participates
            pos = sc > 0
            gt = np.zeros_like(X)
            Neff = 1.0
            if np.any(pos):
                logs = np.full_like(c, -np.inf)
                logs[pos] = np.log(np.maximum(sc[pos], 1e-16))
                lmax = np.max(logs[pos])
                # weights ~ s^{p-1} normalized
                e = np.zeros_like(c)
                # (p-1)*(log-logmax) stable
                e[pos] = np.exp((p - 1.0) * (logs[pos] - lmax))
                esum = e.sum()
                if esum > 0 and np.isfinite(esum):
                    w = e / esum
                    # effective number from q ~ s^p
                    eq = np.zeros_like(c)
                    eq[pos] = np.exp(float(p) * (logs[pos] - lmax))
                    qsum = eq.sum()
                    if qsum > 0:
                        q = eq[pos] / qsum
                        # entropy only over pos
                        ent = -np.sum(q * np.log(np.maximum(q, 1e-300)))
                        Neff = float(np.exp(ent))
                    # build symmetric grad operator
                    # grad = W @ X with W symmetric from w
                    # do via accumulation to avoid n x n dense alloc each time? use dense
                    W = np.zeros((n, n), dtype=np.float64)
                    W[iu0, iu1] = w
                    W[iu1, iu0] = w
                    grad = W @ X
                    dot = np.einsum('ij,ij->i', grad, X)[:, None]
                    gt = grad - dot * X
                else:
                    gt = np.zeros_like(X)
            # Adam step on tangent
            tt += 1
            tglob += 1
            # if gt is zero, small random tangent kick to avoid stall
            if np.linalg.norm(gt) < 1e-15:
                continue
            m = b1 * m + (1.0 - b1) * gt
            v = b2 * v + (1.0 - b2) * (gt * gt)
            mh = m / (1.0 - b1 ** tt)
            vh = v / (1.0 - b2 ** tt)
            step = lr * mh / (np.sqrt(vh) + eps)
            # project step to tangent (remove radial part, Adam may have introduced it)
            # keep as is then retract by normalization (standard Riemannian Adam approx)
            X = X - step
            X = _norm(X)
        # end stage: track true max
        cur = _max_cos(X)
        if cur < best_seen:
            best_seen = cur
            best_X = X.copy()
    # return last X and best seen in this run
    cur_last = _max_cos(X)
    if best_seen < cur_last:
        return best_X, best_seen
    else:
        return X, cur_last

def _hard_polish_topK(X, iters, step0, K0, K1, deadline=1e18):
    n, d = X.shape
    iu0, iu1 = np.triu_indices(n, 1)
    best = _max_cos(X)
    bestX = X.copy()
    step = step0
    for it in range(iters):
        if (it % 50 == 0) and time.time() > deadline:
            break
        frac = it / max(1, iters - 1)
        K = int(round(K0 + (K1 - K0) * frac))
        K = max(1, min(K, len(iu0)))
        G = X @ X.T
        c = G[iu0, iu1]
        # top-K indices
        if K < len(c):
            idx = np.argpartition(-c, K)[:K]
        else:
            idx = np.arange(len(c))
        I = iu0[idx]
        J = iu1[idx]
        grad = np.zeros_like(X)
        np.add.at(grad, I, X[J])
        np.add.at(grad, J, X[I])
        grad /= float(len(idx))
        dot = np.einsum('ij,ij->i', grad, X)[:, None]
        gt = grad - dot * X
        # row-normalized direction for stability (equalize per-point move)
        rn = np.linalg.norm(gt, axis=1, keepdims=True)
        nz = rn[:, 0] > 1e-14
        direc = np.zeros_like(gt)
        direc[nz] = gt[nz] / rn[nz]
        # try step
        Xt = X - step * direc
        Xt = _norm(Xt)
        vv = _max_cos(Xt)
        if vv < best:
            best = vv
            bestX = Xt.copy()
            X = Xt
            step = min(step * 1.25, 0.006)
        else:
            step *= 0.55
            if step < 1e-12:
                # restart step small from best
                X = bestX.copy()
                step = step0 * 0.25
                if it > iters * 0.7:
                    break
    return bestX, best

def _lp_tiny_polish(x, seconds):
    # best-effort trust-region LP on near-max pairs, guarded
    try:
        from scipy.optimize import linprog
        from scipy.sparse import coo_matrix
    except Exception:
        return x, _max_cos(x)
    x = _norm(x.copy())
    n, d = x.shape
    best = _max_cos(x)
    start = time.time()
    iu = np.triu_indices(n, 1)
    radius = 0.0015
    while time.time() - start < seconds and radius > 1e-13:
        cfull = (x @ x.T)[iu]
        m = float(cfull.max())
        thresh = max(20.0 * radius, 5e-8)
        sel = cfull > m - thresh
        cnt = int(sel.sum())
        if cnt == 0:
            radius *= 0.5
            continue
        if cnt > 3000:
            idx = np.argpartition(-cfull, 3000)[:3000]
            mask = np.zeros_like(sel)
            mask[idx] = True
            sel = mask
        I = iu[0][sel]
        J = iu[1][sel]
        p = len(I)
        csub = cfull[sel]
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        except Exception:
            radius *= 0.5
            continue
        obj = np.zeros(nv)
        obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-csub, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = _norm(x + res.x[:-1].reshape(n, d))
        vv = _max_cos(y)
        if vv < best:
            x, best = y, vv
            radius = min(radius * 1.5, 0.01)
        else:
            radius *= 0.5
        if time.time() - start > seconds - 0.3:
            break
    return x, best

def solve():
    t0 = time.time()
    BUDGET = 172.0
    rng = np.random.default_rng(0)
    n, d = 93, 13
    best_val = 1e9
    best_x = None
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=np.float64)
            if a.shape == (n, d):
                a = _norm(a)
                va = _max_cos(a)
                if va < best_val:
                    best_val = va
                    best_x = a.copy()
        except Exception:
            continue
    if best_x is None:
        best_x = _norm(rng.standard_normal((n, d)))
        best_val = _max_cos(best_x)
    try:
        rec = _norm(np.asarray(np.load("record.npy"), dtype=np.float64))
    except Exception:
        rec = best_x.copy()

    # build starting configs
    starts = []
    starts.append(("best", best_x.copy()))
    starts.append(("record", rec.copy()))
    # perturbed
    for sig, tag in [(0.15, "b015"), (0.30, "b030"), (0.60, "b060")]:
        starts.append((tag, _norm(best_x + sig * rng.standard_normal((n, d)))))
    starts.append(("r030", _norm(rec + 0.30 * rng.standard_normal((n, d)))))
    starts.append(("rand1", _norm(rng.standard_normal((n, d)))))
    starts.append(("rand2", _norm(rng.standard_normal((n, d)))))

    p_list = [8, 16, 32, 64, 128, 256]
    # steps per stage ~300; reduce if many starts and little time
    steps_per_stage = 300
    lr0, lr1 = 3e-3, 3e-4
    shift = 0.15

    results = []  # (val, X)
    # keep initial best as candidate
    results.append((best_val, best_x.copy()))

    # run continuation for each start with time check
    for si, (tag, s0) in enumerate(starts):
        if time.time() - t0 > 105.0:
            break
        # slightly vary shift/lr per start for diversity (deterministic)
        sh = shift if si % 2 == 0 else 0.18
        # fewer steps for random starts to save time
        sps = steps_per_stage if si < 5 else 220
        Xc, vc = _p_continuation(s0, steps_per_stage=sps, p_list=p_list,
                                 lr0=lr0, lr1=lr1, shift=sh, seed_off=si)
        results.append((vc, Xc))
        if vc < best_val:
            best_val = vc
            best_x = Xc.copy()

    # promote best 3 continuations to hard-max polish
    results.sort(key=lambda t: t[0])
    top = results[:3]
    polished = []
    for ti, (va, xa) in enumerate(top):
        if time.time() - t0 > 150.0:
            break
        Xp, vp = _hard_polish_topK(xa.copy(), iters=900, step0=0.0025, K0=300, K1=40,
                                   deadline=t0 + 165.0)
        polished.append((vp, Xp))
        if vp < best_val:
            best_val = vp
            best_x = Xp.copy()

    # final tiny LP squeeze on the single best if time remains
    if time.time() - t0 < 168.0:
        try:
            Xl, vl = _lp_tiny_polish(best_x.copy(), seconds=max(1.0, (t0 + BUDGET) - time.time() - 1.0))
            if vl < best_val:
                best_val = vl
                best_x = Xl.copy()
        except Exception:
            pass
        # one more short hard polish after LP
        if time.time() - t0 < 170.0:
            try:
                Xp2, vp2 = _hard_polish_topK(best_x.copy(), iters=300, step0=0.001, K0=150, K1=30,
                                             deadline=t0 + 173.0)
                if vp2 < best_val:
                    best_val = vp2
                    best_x = Xp2.copy()
            except Exception:
                pass

    best_x = _norm(best_x)
    return best_x
