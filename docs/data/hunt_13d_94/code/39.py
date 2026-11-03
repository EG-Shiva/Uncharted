import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _adam_lse(X, steps, b0, b1, lr0, lr1, deadline=None):
    n, d = X.shape
    m1 = np.zeros_like(X)
    m2 = np.zeros_like(X)
    b1m, b2m, eps = 0.9, 0.999, 1e-8
    for s in range(steps):
        if deadline is not None and (s & 63) == 0 and time.time() > deadline:
            break
        frac = s / max(1, steps - 1)
        beta = b0 * (b1 / b0) ** frac
        lr = lr0 * (lr1 / lr0) ** frac
        G = X @ X.T
        np.fill_diagonal(G, -10.0)
        mv = float(G.max())
        W = np.exp(beta * (G - mv))
        np.fill_diagonal(W, 0.0)
        S = float(W.sum())
        if not np.isfinite(S) or S <= 0:
            break
        W /= S
        grad = W @ X
        dot = (grad * X).sum(axis=1, keepdims=True)
        gt = grad - dot * X
        m1 = b1m * m1 + (1 - b1m) * gt
        m2 = b2m * m2 + (1 - b2m) * (gt * gt)
        bc1 = 1.0 - b1m ** (s + 1)
        bc2 = 1.0 - b2m ** (s + 1)
        step = lr * (m1 / bc1) / (np.sqrt(m2 / bc2) + eps)
        X = X - step
        nr = np.linalg.norm(X, axis=1, keepdims=True)
        nr[nr == 0] = 1.0
        X = X / nr
        if not np.all(np.isfinite(X)):
            break
    return X

def _best_insertion(F, rng, n_cand=20000):
    d = F.shape[1]
    C = rng.standard_normal(size=(n_cand, d))
    C /= np.linalg.norm(C, axis=1, keepdims=True)
    D = C @ F.T
    scores = D.max(axis=1)
    idx = int(np.argmin(scores))
    return C[idx].copy(), float(scores[idx])

def _lp_polish(x, deadline):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    iu = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        cmat = (x @ x.T)[iu]
        m = float(cmat.max())
        sel = cmat > m - max(20.0 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        if p == 0:
            break
        # cap constraints for speed
        if p > 1500:
            # keep tightest 1500
            idx = np.argpartition(-cmat[sel], 1500)[:1500]
            I, J = I[idx], J[idx]
            p = 1500
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        except Exception:
            break
        obj = np.zeros(nv)
        obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-cmat[sel][:p] if p != int(sel.sum()) else -cmat[sel],
                          A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        v = _max_cos(y)
        if v < best:
            x, best, radius = y, v, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x, best

def _gram_spec(X):
    n = X.shape[0]
    G = X @ X.T
    iu = np.triu_indices(n, 1)
    v = G[iu]
    v = np.sort(v)
    return v

def _grow_chain(X88, seed, deadline):
    rng = np.random.default_rng(seed)
    X = X88.copy()
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    # easier problem N=88 annealed
    X = _adam_lse(X, steps=600, b0=30.0, b1=300.0, lr0=0.01, lr1=0.001, deadline=deadline)
    # grow one at a time to 94
    for target in range(X.shape[0] + 1, 95):
        if time.time() > deadline - 1.0:
            break
        bv, _ = _best_insertion(X, rng, n_cand=20000)
        X = np.vstack([X, bv[None, :]])
        X /= np.linalg.norm(X, axis=1, keepdims=True)
        # 300 Adam steps after insertion
        X = _adam_lse(X, steps=300, b0=40.0, b1=300.0, lr0=0.008, lr1=0.001, deadline=deadline)
        # short hinge polish: sharp high-beta burst
        X = _adam_lse(X, steps=80, b0=200.0, b1=350.0, lr0=0.002, lr1=0.0005, deadline=deadline)
    return X

def _shrink_chain(X96, seed, deadline):
    rng = np.random.default_rng(seed)  # reserved for tie-break noise
    X = X96.copy()
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    X = _adam_lse(X, steps=600, b0=30.0, b1=300.0, lr0=0.01, lr1=0.001, deadline=deadline)
    while X.shape[0] > 94:
        if time.time() > deadline - 1.0:
            break
        n = X.shape[0]
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        thr = m - 0.005
        # summed positive contact excess
        E = np.maximum(0.0, G - thr).sum(axis=1)
        # if all zero (unlikely), use max per row
        if float(E.max()) <= 0:
            E = G.max(axis=1)
        idx = int(np.argmax(E + 1e-12 * rng.standard_normal(n)))
        X = np.delete(X, idx, axis=0)
        X = _adam_lse(X, steps=300, b0=40.0, b1=300.0, lr0=0.008, lr1=0.001, deadline=deadline)
        X = _adam_lse(X, steps=80, b0=200.0, b1=350.0, lr0=0.002, lr1=0.0005, deadline=deadline)
    return X

def solve():
    T0 = time.time()
    BUDGET = 176.0
    H_END = T0 + 120.0
    FINAL_END = T0 + BUDGET - 1.5
    rng0 = np.random.default_rng(20261102)
    incumbent = None
    inc_val = 1e9
    rec = None
    for fn in ["best.npy", "record.npy"]:
        try:
            a = np.load(fn).astype(float)
            a = a / np.linalg.norm(a, axis=1, keepdims=True)
            if a.shape == (94, 13) and np.all(np.isfinite(a)):
                v = _max_cos(a)
                if v < inc_val:
                    inc_val = v
                    incumbent = a.copy()
                if fn == "record.npy":
                    rec = a.copy()
        except Exception:
            pass
    if incumbent is None:
        incumbent = rng0.standard_normal(size=(94, 13))
        incumbent /= np.linalg.norm(incumbent, axis=1, keepdims=True)
        inc_val = _max_cos(incumbent)
    if rec is None:
        rec = incumbent.copy()
    outcomes = []
    # helper to make 88-subset by dropping 6
    def subset88(X, seed, drop_worst=False):
        rng = np.random.default_rng(seed)
        n = X.shape[0]
        if drop_worst and n == 94:
            G = X @ X.T
            np.fill_diagonal(G, -2.0)
            # drop points involved in largest pairs
            iu = np.triu_indices(n, 1)
            c = (X @ X.T)[iu]
            order = np.argsort(-c)
            drop = set()
            for k in order:
                i, j = int(iu[0][k]), int(iu[1][k])
                # drop the one with larger row-max (more jammed)
                ri = float(np.delete(G[i], i).max())
                rj = float(np.delete(G[j], j).max())
                cand = i if ri > rj else j
                if cand not in drop:
                    drop.add(cand)
                if len(drop) >= 6:
                    break
            keep = [i for i in range(n) if i not in drop]
            return X[np.array(keep)].copy()
        else:
            idx = rng.choice(n, size=88, replace=False)
            idx = np.sort(idx)
            return X[idx].copy()

    # Chain 1: record -> 88 (worst-drop) -> grow
    if time.time() < H_END - 15:
        try:
            X88 = subset88(rec, 11, drop_worst=True)
            X94 = _grow_chain(X88, seed=101, deadline=H_END)
            if X94.shape == (94, 13):
                outcomes.append(X94.copy())
        except Exception:
            pass
    # Chain 2: random 88 -> grow
    if time.time() < H_END - 15:
        try:
            R88 = rng0.standard_normal(size=(88, 13))
            R88 /= np.linalg.norm(R88, axis=1, keepdims=True)
            X94 = _grow_chain(R88, seed=202, deadline=H_END)
            if X94.shape == (94, 13):
                outcomes.append(X94.copy())
        except Exception:
            pass
    # Chain 3: incumbent/best -> 88 random subset -> grow
    if time.time() < H_END - 15:
        try:
            X88 = subset88(incumbent, 33, drop_worst=False)
            X94 = _grow_chain(X88, seed=303, deadline=H_END)
            if X94.shape == (94, 13):
                outcomes.append(X94.copy())
        except Exception:
            pass
    # Chain 4: squeezed N=96 random -> delete to 94
    if time.time() < H_END - 12:
        try:
            R96 = np.random.default_rng(404).standard_normal(size=(96, 13))
            R96 /= np.linalg.norm(R96, axis=1, keepdims=True)
            X94 = _shrink_chain(R96, seed=404, deadline=H_END)
            if X94.shape == (94, 13):
                outcomes.append(X94.copy())
        except Exception:
            pass
    # Chain 5: record + 2 insertions -> 96 -> shrink (different basin)
    if time.time() < H_END - 12:
        try:
            rng5 = np.random.default_rng(505)
            F = rec.copy()
            for _ in range(2):
                bv, _ = _best_insertion(F, rng5, n_cand=20000)
                F = np.vstack([F, bv[None, :]])
            # F is 96, squeeze-delete back to 94
            X94 = _shrink_chain(F, seed=505, deadline=H_END)
            if X94.shape == (94, 13):
                outcomes.append(X94.copy())
        except Exception:
            pass

    # score outcomes + incumbent
    scored = [(inc_val, incumbent.copy())]
    for X in outcomes:
        try:
            X = X / np.linalg.norm(X, axis=1, keepdims=True)
            if X.shape != (94, 13) or not np.all(np.isfinite(X)):
                continue
            v = _max_cos(X)
            scored.append((v, X.copy()))
        except Exception:
            continue
    scored.sort(key=lambda t: t[0])
    # distinct selection by sorted Gram spectra
    elites = []
    specs = []
    for v, X in scored:
        try:
            s = _gram_spec(X)
        except Exception:
            continue
        dup = False
        for s0 in specs:
            # normalized L2 distance
            if float(np.linalg.norm(s - s0)) < 1e-6:
                dup = True
                break
        if not dup:
            elites.append((v, X))
            specs.append(s)
        if len(elites) >= 4:
            break
    # LP-polish best two distinct
    to_polish = elites[:2] if len(elites) >= 2 else elites[:1]
    remaining = max(0.0, FINAL_END - time.time())
    per = remaining / max(1, len(to_polish)) if to_polish else 0
    per = min(max(per, 3.0), 20.0)
    final_best = incumbent.copy()
    final_val = inc_val
    # include unpolished elites in comparison
    for v, X in elites:
        if v < final_val:
            final_val, final_best = v, X.copy()
    for v, X in to_polish:
        if time.time() > FINAL_END - 1.0:
            break
        dl = min(time.time() + per, FINAL_END)
        try:
            P, pv = _lp_polish(X.copy(), dl)
            if pv < final_val and np.all(np.isfinite(P)):
                final_val, final_best = pv, P.copy()
        except Exception:
            continue
        # quick global Adam burst after LP if time allows
        if time.time() < dl - 2.0:
            try:
                Q = _adam_lse(P.copy(), steps=150, b0=150.0, b1=400.0, lr0=0.002, lr1=0.0005, deadline=dl)
                qv = _max_cos(Q)
                if qv < final_val:
                    final_val, final_best = qv, Q.copy()
                    # re-LP briefly
                    if time.time() < dl - 1.0:
                        P2, pv2 = _lp_polish(Q.copy(), dl)
                        if pv2 < final_val:
                            final_val, final_best = pv2, P2.copy()
            except Exception:
                pass
    # verify and return incumbent if failed
    try:
        final_best = final_best / np.linalg.norm(final_best, axis=1, keepdims=True)
        fv = _max_cos(final_best)
        if not np.all(np.isfinite(final_best)) or final_best.shape != (94, 13):
            return incumbent
        # strict: shrink tiny margin? not needed; just return
        if fv > inc_val and inc_val < 1e8:
            # still return the better one
            return incumbent
        return final_best
    except Exception:
        return incumbent
