import time
import numpy as np

def _norm_rows(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[~np.isfinite(n)] = 1.0
    n[n == 0] = 1.0
    return x / n

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _softmax_quench(X, steps, T0, T1, lr, mu=0.9, deadline=None, track=True):
    n, d = X.shape
    V = np.zeros_like(X)
    lt0 = np.log(T0); lt1 = np.log(T1)
    if track:
        G0 = X @ X.T
        np.fill_diagonal(G0, -2.0)
        bv = float(G0.max())
        bX = X.copy()
    else:
        bv = 1e100; bX = None
    for s in range(steps):
        if deadline is not None and (s % 200 == 0) and time.time() > deadline:
            break
        frac = s / max(1, steps - 1)
        T = float(np.exp(lt0 + (lt1 - lt0) * frac))
        G = X @ X.T
        np.fill_diagonal(G, -1e100)
        m = float(G.max())
        Q = (G - m) / T
        np.clip(Q, -50, 0, out=Q)
        E = np.exp(Q)
        np.fill_diagonal(E, 0.0)
        rs = E.sum(axis=1, keepdims=True)
        rs[rs == 0] = 1.0
        Grad = (E @ X) / rs
        dot = np.sum(Grad * X, axis=1, keepdims=True)
        Gt = Grad - dot * X
        gn = np.linalg.norm(Gt, axis=1, keepdims=True) + 1e-12
        bad = gn[:, 0] > 2.0
        if np.any(bad):
            Gt[bad] *= (2.0 / gn[bad])
        V = mu * V + Gt
        X = X - lr * V
        X = _norm_rows(X)
        if track and ((s % 200 == 0) or s == steps - 1):
            Gc = X @ X.T
            np.fill_diagonal(Gc, -2.0)
            vc = float(Gc.max())
            if vc < bv:
                bv = vc
                bX = X.copy()
    if track:
        return bX, bv
    return X, _max_cos(X)

def _embed_to_12(Xlow, rng, eps=0.08):
    n, dl = Xlow.shape
    d = 12
    if dl >= 12:
        return _norm_rows(Xlow[:, :12].copy())
    Z = np.zeros((n, d))
    Z[:, :dl] = Xlow
    Z[:, dl:] = eps * rng.standard_normal(size=(n, d - dl))
    return _norm_rows(Z)

def _greedy_delete_path(X, target_n, rng, deadline=None):
    # X is (N,d) normalized; iteratively delete worst row-max point, re-polish after each deletion
    X = X.copy()
    while X.shape[0] > target_n:
        if deadline is not None and time.time() > deadline:
            # delete remaining in one shot by row-max order
            G = X @ X.T
            np.fill_diagonal(G, -2.0)
            mx = G.max(axis=1)
            order = np.argsort(mx)  # keep smallest
            keep = np.sort(order[:target_n])
            X = X[keep]
            break
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        mx = G.max(axis=1)
        # contact load: count pairs within 2e-3 of max
        m = float(G.max())
        loads = (G > m - 2e-3).sum(axis=1)
        score_row = mx + 1e-4 * loads
        victim = int(np.argmax(score_row))
        mask = np.ones(X.shape[0], dtype=bool)
        mask[victim] = False
        X = X[mask]
        # short re-polish after every single deletion (path following)
        X, _ = _softmax_quench(X, 300, 0.006, 0.0012, 0.20, mu=0.88, deadline=deadline, track=True)
    return _norm_rows(X)

def _gram_spectrum(X):
    try:
        G = X @ X.T
        w = np.linalg.eigvalsh(G)
        return np.sort(w)
    except Exception:
        return np.zeros(X.shape[0])

def _degree_seq(X, thr=1e-9):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    m = float(G.max())
    C = (G >= m - thr)
    deg = C.sum(axis=1)
    return np.sort(deg), m

def _lp_polish(x, seconds, deadline=None):
    from scipy.optimize import linprog
    from scipy.sparse import coo_matrix
    x = _norm_rows(x.copy())
    n, d = x.shape
    try:
        best = _max_cos(x)
    except Exception:
        return x
    radius = 1.2e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds:
        if deadline is not None and time.time() > deadline:
            break
        if radius < 1e-14:
            break
        cmat = (x @ x.T)[iu]
        m = float(cmat.max())
        sel = cmat > m - max(20 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        if p == 0:
            break
        if p > 1500:
            order = np.argsort(-cmat[sel])[:1500]
            I, J = I[order], J[order]
            csel = cmat[sel][order]
            p = 1500
        else:
            csel = cmat[sel]
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cc = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vv = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vv, (rows, cc)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        except Exception:
            radius *= 0.5
            continue
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y = _norm_rows(y)
        v = _max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def solve():
    t0 = time.time()
    HARD_END = t0 + 172.0
    GEN_END = t0 + 105.0
    # NEVER load best.npy as parent per assigned direction: fresh random starts only
    elite = []  # list of dicts
    # neighbouring tasks: (N, d, seed_offset)
    tasks = [
        (85, 12, 101), (80, 12, 102), (80, 12, 103),
        (75, 12, 104), (75, 12, 105),
        (72, 12, 106), (72, 12, 107),
        (71, 11, 108), (71, 10, 109),
        (71, 12, 110), (71, 12, 111),
    ]
    best_score = 1e100
    best_X = None
    for (Nn, dd, soff) in tasks:
        if time.time() > GEN_END or time.time() > HARD_END - 60:
            break
        rng = np.random.default_rng(777000 + soff)
        X0 = _norm_rows(rng.standard_normal(size=(Nn, dd)))
        # two-stage fresh anneal from random start
        X1, _ = _softmax_quench(X0, 600, 0.05, 0.01, 0.50, mu=0.90, deadline=GEN_END, track=True)
        if time.time() > GEN_END:
            Xq = X1
        else:
            Xq, _ = _softmax_quench(X1, 1100, 0.01, 0.0012, 0.30, mu=0.90, deadline=GEN_END, track=True)
        Xq = _norm_rows(Xq)
        # continuation to (71,12)
        if dd < 12:
            Xq = _embed_to_12(Xq, rng, eps=0.10)
            # follow path in 12D: polish
            Xq, _ = _softmax_quench(Xq, 700, 0.008, 0.0010, 0.25, mu=0.90, deadline=GEN_END, track=True)
        if Nn > 71:
            Xq = _greedy_delete_path(Xq, 71, rng, deadline=GEN_END)
        # ensure shape (71,12)
        if Xq.shape != (71, 12):
            if Xq.shape[0] != 71:
                continue
            Xq = _embed_to_12(Xq, rng)
        Xq = _norm_rows(Xq)
        # short tighten
        Xq, _ = _softmax_quench(Xq, 350, 0.004, 0.0008, 0.18, mu=0.88, deadline=GEN_END, track=True)
        sc = _max_cos(Xq)
        spec = _gram_spectrum(Xq)
        deg, _ = _degree_seq(Xq)
        elite.append({"X": Xq.copy(), "score": sc, "spec": spec, "deg": deg})
        if sc < best_score:
            best_score = sc
            best_X = Xq.copy()
    if best_X is None:
        rng = np.random.default_rng(0)
        best_X = _norm_rows(rng.standard_normal(size=(71, 12)))
        best_score = _max_cos(best_X)
        elite.append({"X": best_X.copy(), "score": best_score, "spec": _gram_spectrum(best_X), "deg": _degree_seq(best_X)[0]})
    # keep diverse pool within 0.002 of best
    elite = sorted(elite, key=lambda e: e["score"])
    bs = elite[0]["score"]
    pool = [e for e in elite if e["score"] <= bs + 0.002]
    # ensure at least ten if available, else all; diversify by spectrum distance
    if len(pool) > 10:
        # greedy diverse selection: start from best, add farthest spectrum
        chosen = [pool[0]]
        rest = pool[1:]
        while len(chosen) < 10 and rest:
            # distance = spectrum L2 + degree L1
            best_idx = -1; best_d = -1
            for i, e in enumerate(rest):
                dmin = min(float(np.linalg.norm(e["spec"] - c["spec"])) for c in chosen)
                if dmin > best_d:
                    best_d = dmin; best_idx = i
            chosen.append(rest.pop(best_idx))
        # also keep next best scores to fill intensive budget
        pool = chosen + [e for e in pool if e not in chosen][:0]
        # for polishing, use chosen + top scorers
        polish_list = chosen
        # add a couple more top scorers if time allows (up to 10 already)
    else:
        polish_list = pool[:10]
    # intensive LP polishing across pool (low-T tighten + LP)
    # sort polish_list by score
    polish_list = sorted(polish_list, key=lambda e: e["score"])
    time_left = HARD_END - time.time()
    per = min(8.0, max(2.0, (time_left - 4.0) / max(1, len(polish_list))))
    for e in polish_list:
        if time.time() > HARD_END - 5:
            break
        Xc = e["X"].copy()
        # low-T tightening before LP
        Xc, _ = _softmax_quench(Xc, 400, 0.003, 0.0006, 0.15, mu=0.88, deadline=HARD_END - 5, track=True)
        Xc = _lp_polish(Xc, seconds=per, deadline=HARD_END - 4)
        # final micro tighten
        Xc, _ = _softmax_quench(Xc, 200, 0.0015, 0.0005, 0.10, mu=0.88, deadline=HARD_END - 3, track=True)
        # one more quick LP with small radius
        Xc = _lp_polish(Xc, seconds=max(1.0, per * 0.5), deadline=HARD_END - 2)
        sc = _max_cos(Xc)
        e["X"] = Xc; e["score"] = sc
        if sc < best_score:
            best_score = sc
            best_X = Xc.copy()
    # final best across pool
    for e in elite:
        if e["score"] < best_score:
            best_score = e["score"]
            best_X = e["X"].copy()
    for e in polish_list:
        if e["score"] < best_score:
            best_score = e["score"]
            best_X = e["X"].copy()
    best_X = _norm_rows(best_X)
    # tiny shrink guard not needed; ensure finite
    best_X[~np.isfinite(best_X)] = 0.0
    nn = np.linalg.norm(best_X, axis=1, keepdims=True)
    nn[nn == 0] = 1.0
    best_X = best_X / nn
    return best_X
