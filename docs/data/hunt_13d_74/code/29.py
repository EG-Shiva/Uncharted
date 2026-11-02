import time
import numpy as np

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _normalize(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return X / n

def solve():
    t0 = time.time()
    LIM = 172.0
    END = t0 + LIM - 2.0
    rng = np.random.default_rng(12345)
    n0, d = 74, 13

    cands = []
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=np.float64)
            if a.shape == (74, 13):
                a = _normalize(a)
                if np.all(np.isfinite(a)):
                    cands.append(a)
        except Exception:
            pass
    if not cands:
        cands.append(_normalize(rng.standard_normal((74, 13))))
    vals = [_max_cos(a) for a in cands]
    bi = int(np.argmin(vals))
    best = cands[bi].copy()
    best_val = float(vals[bi])

    # ---------- helpers ----------
    def anneal_box(X, phases, deadline):
        m = np.zeros_like(X)
        v = np.zeros_like(X)
        cnt = 0
        for tau, steps, lr0 in phases:
            for s in range(steps):
                if (s & 31) == 0 and time.time() >= deadline:
                    return X
                G = X @ X.T
                np.fill_diagonal(G, -10.0)
                gmax = float(G.max())
                # LSE weights
                E = np.exp((G - gmax) / tau)
                np.fill_diagonal(E, 0.0)
                S = float(E.sum())
                if not np.isfinite(S) or S <= 0:
                    continue
                Gr = (E @ X) / S
                dot = (Gr * X).sum(axis=1, keepdims=True)
                Gt = Gr - dot * X
                m *= 0.9
                m += 0.1 * Gt
                v *= 0.999
                v += 0.001 * (Gt * Gt)
                cnt += 1
                c1 = 1.0 - 0.9 ** cnt
                c2 = 1.0 - 0.999 ** cnt
                if c1 < 1e-12 or c2 < 1e-12:
                    continue
                mhat = m / c1
                vhat = v / c2
                lr = lr0 * (1.0 - 0.5 * s / max(1, steps))
                X -= lr * mhat / (np.sqrt(vhat) + 1e-8)
                nr = np.linalg.norm(X, axis=1, keepdims=True)
                nr[nr == 0] = 1.0
                X /= nr
        return X

    def hole_point(X, n_cand=4000):
        C = rng.standard_normal((n_cand, d))
        C = _normalize(C)
        S = C @ X.T
        sc = S.max(axis=1)
        j = int(np.argmin(sc))
        return C[j].copy()

    def combined_order(X):
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        gmax = float(G.max())
        mx = G.max(axis=1)
        deg = (G > (gmax - 1e-7)).sum(axis=1).astype(np.float64)
        # sum of top-5 (off diag dominates)
        part = np.sort(G, axis=1)[:, -5:].sum(axis=1)
        score = mx + 1e-4 * deg + 1e-6 * part
        return score, mx, deg, float(gmax)

    def prune_to(Xbig, target):
        Xb = Xbig
        while Xb.shape[0] > target:
            score, _, _, _ = combined_order(Xb)
            idx = int(np.argmax(score))
            Xb = np.delete(Xb, idx, axis=0)
        return Xb

    def drop_worst(X, k):
        Xb = X
        for _ in range(k):
            score, _, _, _ = combined_order(Xb)
            idx = int(np.argmax(score))
            Xb = np.delete(Xb, idx, axis=0)
        return Xb

    def try_update(X):
        nonlocal best, best_val
        X = _normalize(X)
        v = _max_cos(X)
        if v < best_val:
            best_val = float(v)
            best = X.copy()
            return True
        return False

    # ---------- overshoot cycle ----------
    def overshoot_cycle(add_k, budget):
        # budget seconds total for this cycle
        if time.time() + 8 >= END:
            return
        t_start = time.time()
        t_ann_end = min(t_start + budget * 0.75, END - 6)
        t_end = min(t_start + budget, END - 1)
        Xb = best.copy()
        for _ in range(add_k):
            h = hole_point(Xb, n_cand=5000)
            Xb = np.vstack([Xb, h.reshape(1, d)])
            Xb = _normalize(Xb)
        # anneal enlarged (harder) problem under pressure
        phases = [
            (0.006, 2000, 0.02),
            (0.002, 2000, 0.010),
            (5e-4, 2000, 0.005),
            (1.5e-4, 2000, 0.002),
        ]
        Xb = anneal_box(Xb, phases, t_ann_end)
        # prune back to 74 by worst combined score
        Xb = prune_to(Xb, 74)
        # repolish
        phases2 = [(3e-4, 1200, 0.003), (8e-5, 1200, 0.0012), (4e-5, 800, 0.0006)]
        Xb = anneal_box(Xb, phases2, t_end)
        try_update(Xb)

    # ---------- undershoot cycle ----------
    def undershoot_cycle(drop_k, budget):
        if time.time() + 10 >= END:
            return
        t_start = time.time()
        t_small_end = min(t_start + budget * 0.4, END - 12)
        t_end = min(t_start + budget, END - 1)
        Xb = best.copy()
        Xb = drop_worst(Xb, drop_k)
        # anneal small (easier) problem to reorganise
        phases = [(0.004, 1500, 0.015), (0.001, 1500, 0.007), (3e-4, 1500, 0.003)]
        Xb = anneal_box(Xb, phases, t_small_end)
        # staged reinsertion one at a time with polishing between
        remain = t_end - time.time()
        per = remain / drop_k if drop_k > 0 else remain
        for _ in range(drop_k):
            if time.time() >= END - 2:
                break
            h = hole_point(Xb, n_cand=5000)
            Xb = np.vstack([Xb, h.reshape(1, d)])
            Xb = _normalize(Xb)
            dl = min(time.time() + per, END - 1)
            Xb = anneal_box(Xb, [(3e-4, 900, 0.003), (8e-5, 900, 0.001)], dl)
        # ensure exactly 74
        if Xb.shape[0] != 74:
            if Xb.shape[0] > 74:
                Xb = prune_to(Xb, 74)
            else:
                while Xb.shape[0] < 74 and time.time() < END - 2:
                    h = hole_point(Xb, n_cand=3000)
                    Xb = np.vstack([Xb, h.reshape(1, d)])
        if Xb.shape[0] == 74:
            # short final polish of grown code
            dl2 = min(time.time() + 2.0, END - 1)
            Xb = anneal_box(Xb, [(8e-5, 600, 0.001)], dl2)
            try_update(Xb)

    # alternate cycles within budget
    # cycle 1: overshoot +3
    if time.time() < END - 20:
        overshoot_cycle(3, 38.0)
    # cycle 2: undershoot to 70 (drop 4, grow 4)
    if time.time() < END - 20:
        undershoot_cycle(4, 42.0)
    # cycle 3: overshoot +4 (stronger pressure)
    if time.time() < END - 20:
        overshoot_cycle(4, 32.0)
    # cycle 4: overshoot +2 (gentle)
    if time.time() < END - 15:
        overshoot_cycle(2, 22.0)

    # final polish of best with remaining time
    if time.time() < END:
        Xf = best.copy()
        phases_f = [(2e-4, 2000, 0.002), (6e-5, 2000, 0.0008), (3e-5, 1500, 0.0004)]
        Xf = anneal_box(Xf, phases_f, END)
        try_update(Xf)

    best = _normalize(best)
    return best
