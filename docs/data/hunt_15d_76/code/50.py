import time
import numpy as np


def _norm(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return X / n


def _maxcos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())


def _adam_lse_opt(X, t_end):
    n, d = X.shape
    m = np.zeros_like(X)
    vv = np.zeros_like(X)
    b1, b2, eps = 0.9, 0.999, 1e-8
    b1p, b2p = 1.0, 1.0
    beta0, beta1 = 25.0, 320.0
    lr0, lr1 = 0.008, 0.0009
    lb = np.log(beta1 / beta0)
    ll = np.log(lr1 / lr0)
    t_start = time.time()
    budget = t_end - t_start
    if budget <= 0.05:
        return X
    while True:
        now = time.time()
        if now >= t_end - 0.02:
            break
        frac = (now - t_start) / budget
        if frac < 0:
            frac = 0.0
        if frac > 1:
            frac = 1.0
        beta = beta0 * np.exp(lb * frac)
        lr = lr0 * np.exp(ll * frac)
        for _ in range(60):
            G = X @ X.T
            np.fill_diagonal(G, -10.0)
            gm = float(G.max())
            E = np.exp(beta * (G - gm))
            np.fill_diagonal(E, 0.0)
            S = float(E.sum())
            if S == 0 or not np.isfinite(S):
                break
            E /= S
            Gr = 2.0 * (E @ X)
            Gt = Gr - np.sum(Gr * X, axis=1, keepdims=True) * X
            b1p *= b1
            b2p *= b2
            m *= b1
            m += (1 - b1) * Gt
            vv *= b2
            vv += (1 - b2) * (Gt * Gt)
            denom1 = 1.0 - b1p
            denom2 = 1.0 - b2p
            if denom1 < 1e-12 or denom2 < 1e-12:
                break
            mhat = m / denom1
            vhat = vv / denom2
            X -= lr * mhat / (np.sqrt(vhat) + eps)
            nr = np.linalg.norm(X, axis=1, keepdims=True)
            nr[nr == 0] = 1.0
            X /= nr
            if time.time() >= t_end - 0.02:
                break
        if time.time() >= t_end - 0.02:
            break
    X = _norm(X)
    return X


def _characterize(X):
    n = X.shape[0]
    G = X @ X.T
    # eigenvalues of Gram
    try:
        ev = np.linalg.eigvalsh(G)
    except Exception:
        ev = np.zeros(n)
    ev = np.sort(ev)[::-1]
    # keep top 15 (d=15) non-trivial
    feat = ev[:15].copy()
    mc = _maxcos(X)
    # contact degrees with loose threshold (exploratory, not LP-polished)
    G2 = X @ X.T
    np.fill_diagonal(G2, -2.0)
    thr = 0.002
    deg = np.sum(G2 > mc - thr, axis=1)
    ratt = int(np.sum(deg == 0))
    # histogram (binned degrees)
    hist = deg.astype(float)
    return feat, mc, deg, ratt, hist


def _lp_polish(X, t_end):
    from scipy.optimize import linprog
    from scipy.sparse import coo_matrix
    X = _norm(X.copy())
    n, d = X.shape
    best = _maxcos(X)
    radius = 1e-3
    iu = np.triu_indices(n, 1)
    while time.time() < t_end - 0.3 and radius > 1e-14:
        if time.time() > t_end - 0.6:
            break
        c = (X @ X.T)[iu]
        mcur = float(c.max())
        thr = max(20 * radius, 1e-9)
        sel = c > mcur - thr
        if int(sel.sum()) > 1500:
            idx = np.argpartition(-c, 1500)[:1500]
            mask = np.zeros_like(sel)
            mask[idx] = True
            sel = mask
        I, J = iu[0][sel], iu[1][sel]
        p, nv = len(I), n * d + 1
        if p == 0:
            break
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([X[J], X[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((X.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv)
            obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-c[sel], A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            break
        if res.status != 0:
            radius *= 0.5
            continue
        Y = X + res.x[:-1].reshape(n, d)
        Y = _norm(Y)
        v = _maxcos(Y)
        if v < best:
            X, best = Y, v
            radius = min(radius * 1.5, 0.02)
        else:
            radius *= 0.5
    return X


def _make_starts():
    n, d = 76, 15
    starts = []
    # 5 random Gaussian
    for s in range(5):
        rng = np.random.default_rng(1000 + s)
        X = rng.normal(size=(n, d))
        X = _norm(X)
        starts.append(X)
    # 4 perturbed unions of two lower-dim codes in orthogonal blocks
    cfgs = [(7, 38), (5, 30), (8, 40), (7, 36)]
    for k, (d1, n1) in enumerate(cfgs):
        rng = np.random.default_rng(2000 + k)
        d2 = d - d1
        n2 = n - n1
        A = rng.normal(size=(n1, d1))
        A = _norm(A)
        B = rng.normal(size=(n2, d2))
        B = _norm(B)
        X = np.zeros((n, d))
        X[:n1, :d1] = A
        X[n1:, d1:] = B
        X += 0.08 * rng.normal(size=(n, d))
        X = _norm(X)
        starts.append(X)
    # 5 cyclic-orbit configurations
    # c0: single ring in dims 0,1 + noise elsewhere
    rng = np.random.default_rng(3000)
    ang = 2 * np.pi * np.arange(n) / n
    X = 0.45 * rng.normal(size=(n, d))
    X[:, 0] = np.cos(ang)
    X[:, 1] = np.sin(ang)
    starts.append(_norm(X))
    # c1: two rings in orthogonal planes
    rng = np.random.default_rng(3001)
    X = 0.45 * rng.normal(size=(n, d))
    a1 = 2 * np.pi * np.arange(38) / 38
    X[:38, 0] = np.cos(a1)
    X[:38, 1] = np.sin(a1)
    a2 = 2 * np.pi * np.arange(38) / 38
    X[38:, 2] = np.cos(a2)
    X[38:, 3] = np.sin(a2)
    starts.append(_norm(X))
    # c2: cyclic shifts of template
    rng = np.random.default_rng(3002)
    tpl = rng.normal(size=(d,))
    tpl /= np.linalg.norm(tpl)
    X = np.zeros((n, d))
    for i in range(n):
        X[i] = np.roll(tpl, i % d) + 0.25 * rng.normal(size=(d,))
    starts.append(_norm(X))
    # c3: three interleaved rings
    rng = np.random.default_rng(3003)
    X = 0.5 * rng.normal(size=(n, d))
    for i in range(n):
        r = i % 3
        a = 2 * np.pi * (i // 3) / 26.0 + r * 0.7
        X[i, 2 * r] = np.cos(a)
        X[i, 2 * r + 1] = np.sin(a)
    starts.append(_norm(X))
    # c4: rotation-block orbit (7 planes + 1 fixed)
    rng = np.random.default_rng(3004)
    base = rng.normal(size=(d,))
    base /= np.linalg.norm(base)
    X = np.zeros((n, d))
    for i in range(n):
        v = base.copy()
        for b in range(7):
            th = 2 * np.pi * i * (b + 1) / n
            c, s_ = np.cos(th), np.sin(th)
            x0, x1 = v[2 * b], v[2 * b + 1]
            v[2 * b] = c * x0 - s_ * x1
            v[2 * b + 1] = s_ * x0 + c * x1
        X[i] = v + 0.15 * rng.normal(size=(d,))
    starts.append(_norm(X))
    return starts  # 14 starts


def solve():
    import time
    t0 = time.time()
    deadline = t0 + 165.0
    n, d = 76, 15
    # fallback best for safety (not used as start)
    try:
        fb = np.load("best.npy").astype(float)
        fb = _norm(fb)
        fb_score = _maxcos(fb)
    except Exception:
        fb = None
        fb_score = 1e9
    try:
        rec = np.load("record.npy").astype(float)
        rec = _norm(rec)
        rec_score = _maxcos(rec)
        if fb is None or rec_score < fb_score:
            fb, fb_score = rec, rec_score
    except Exception:
        pass
    if fb is None:
        rng0 = np.random.default_rng(0)
        fb = _norm(rng0.normal(size=(n, d)))
        fb_score = _maxcos(fb)

    starts = _make_starts()
    # keep 12-16 starts (we have 14)
    results = []
    # reserve ~45s for final polish of 3 + margin
    reserve_polish = 45.0
    per_start = 6.5
    for si, S in enumerate(starts):
        if time.time() > deadline - reserve_polish - 1.0:
            break
        remaining = deadline - reserve_polish - time.time()
        nleft = len(starts) - si
        # shrink per-start if running late
        budget = min(per_start, max(2.0, remaining / max(1, nleft)))
        t_end = time.time() + budget
        if t_end > deadline - reserve_polish:
            t_end = deadline - reserve_polish
        X = S.copy()
        X = _adam_lse_opt(X, t_end)
        if time.time() > deadline - 1.0:
            # still record
            pass
        feat, mc, deg, ratt, hist = _characterize(X)
        results.append({"X": X.copy(), "score": float(mc), "feat": feat,
                        "deg": deg, "ratt": int(ratt)})
        if time.time() > deadline - reserve_polish - 0.5:
            # continue to clustering if out of exploratory budget
            pass

    if len(results) == 0:
        return fb

    # cluster near-duplicates: greedy by score
    order = sorted(range(len(results)), key=lambda i: results[i]["score"])
    clusters = []  # list of representative indices into results
    for idx in order:
        r = results[idx]
        dup = False
        for cidx in clusters:
            c = results[cidx]
            fd = float(np.linalg.norm(r["feat"] - c["feat"]))
            sd = abs(r["score"] - c["score"])
            # degree-hist distance
            hd = float(np.mean(np.sort(r["deg"]) != np.sort(c["deg"]))) if False else 0.0
            if fd < 0.6 and sd < 0.008:
                dup = True
                break
        if not dup:
            clusters.append(idx)
        if time.time() > deadline - reserve_polish + 2.0:
            # don't spend more time clustering
            pass

    # best representative from each distinct cluster retained; take 3 lowest-score clusters
    clusters = sorted(clusters, key=lambda i: results[i]["score"])
    top3 = clusters[:3]

    polished = []
    for k, idx in enumerate(top3):
        if time.time() > deadline - 2.0:
            break
        # 15-second LP polish each, but respect deadline
        secs = 15.0
        if k == len(top3) - 1:
            secs = min(secs, deadline - time.time() - 2.0)
        else:
            # leave time for remaining polishes
            left_after = (len(top3) - k - 1) * 15.0 + 2.0
            avail = deadline - time.time() - left_after
            secs = min(secs, max(3.0, avail))
        if secs < 2.0:
            polished.append((results[idx]["X"], results[idx]["score"]))
            continue
        t_end = time.time() + secs
        try:
            Y = _lp_polish(results[idx]["X"], t_end)
            s = _maxcos(Y)
        except Exception:
            Y, s = results[idx]["X"], results[idx]["score"]
        polished.append((Y, float(s)))
        if time.time() > deadline - 2.0:
            break

    # overall best among polished, unpolished exploratory, and fallback
    bestX, bestS = fb.copy(), float(fb_score)
    for r in results:
        if r["score"] < bestS:
            bestS, bestX = r["score"], r["X"].copy()
    for Y, s in polished:
        if s < bestS:
            bestS, bestX = s, Y.copy()

    bestX = _norm(bestX)
    if not np.all(np.isfinite(bestX)):
        return fb
    return bestX
