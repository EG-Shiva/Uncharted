import time
import numpy as np

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _stress_counts(X, beta=350.0, tol=5e-4):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    m = float(G.max())
    # stress (softmax row sum)
    try:
        W = np.exp(beta * (G - m))
        np.fill_diagonal(W, 0.0)
        stress = W.sum(axis=1)
    except Exception:
        stress = G.max(axis=1)
    cnt = (G > m - tol).sum(axis=1)
    pmax = G.max(axis=1)
    return m, stress, cnt, pmax, G

def _anneal(X, steps, b0, b1, lr0, mu, deadline=None):
    n, d = X.shape
    v = np.zeros_like(X)
    if b1 > b0:
        lb = np.log(b1 / b0)
    else:
        lb = 0.0
    for t in range(steps):
        if deadline is not None and (t & 127) == 0 and time.time() > deadline:
            break
        frac = t / max(1, steps - 1)
        beta = b0 * np.exp(lb * frac)
        lr = lr0 * (1.0 - 0.5 * frac)
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        gmax = G.max()
        W = np.exp(beta * (G - gmax))
        np.fill_diagonal(W, 0.0)
        S = W.sum(axis=1, keepdims=True) + 1e-300
        A = (W @ X) / S
        dot = (A * X).sum(axis=1, keepdims=True)
        Gt = A - dot * X
        v = mu * v + Gt
        vd = (v * X).sum(axis=1, keepdims=True)
        v -= vd * X
        X = X - lr * v
        nr = np.linalg.norm(X, axis=1, keepdims=True)
        np.maximum(nr, 1e-12, out=nr)
        X /= nr
    return X

def _add_holes(Xrem, k, rng, n_cand=9000):
    Xa = Xrem.copy()
    for _ in range(k):
        cand = rng.standard_normal((n_cand, 4))
        cand /= np.maximum(np.linalg.norm(cand, axis=1, keepdims=True), 1e-12)
        best_s = 2.0
        best_p = None
        bs = 3000
        for a in range(0, n_cand, bs):
            cc = cand[a:a+bs]
            dd = cc @ Xa.T
            mm = dd.max(axis=1)
            j = int(np.argmin(mm))
            if float(mm[j]) < best_s:
                best_s = float(mm[j])
                best_p = cc[j]
        jp = best_p + 0.003 * rng.standard_normal(4)
        nrm = np.linalg.norm(jp)
        if not np.isfinite(nrm) or nrm < 1e-12:
            jp = best_p
            nrm = np.linalg.norm(jp)
        jp /= max(1e-12, nrm)
        Xa = np.vstack([Xa, jp.reshape(1, 4)])
    return Xa

def _sig_vec(X, L=600):
    n = X.shape[0]
    iu = np.triu_indices(n, 1)
    c = (X @ X.T)[iu]
    m = c.max()
    s = m - c
    s = np.sort(s)
    if len(s) > L:
        s = s[:L]
    # log scale for comparison, clip
    return np.log10(np.maximum(s, 1e-16))

def _sig_dist(a, b):
    return float(np.mean((a - b) ** 2))

def _lp_polish(x, deadline):
    try:
        from scipy.optimize import linprog
        from scipy.sparse import coo_matrix
    except Exception:
        return x
    x = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
    n, d = x.shape
    try:
        best = _max_cos(x)
    except Exception:
        return x
    radius = 1e-3
    iu = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        c = (x @ x.T)[iu]
        mval = c.max()
        sel = c > mval - max(20 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        if p == 0:
            break
        nv = n * d + 1
        try:
            rows = np.repeat(np.arange(p), 2 * d + 1)
            cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
            vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv); obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-c[sel], A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            break
        if getattr(res, "status", 1) != 0:
            radius *= 0.5
            continue
        try:
            y = x + res.x[:-1].reshape(n, d)
        except Exception:
            radius *= 0.5
            continue
        y /= np.maximum(np.linalg.norm(y, axis=1, keepdims=True), 1e-12)
        vv = _max_cos(y)
        if vv < best:
            x, best, radius = y, vv, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def solve():
    t0 = time.time()
    TL = 172.0
    explore_end = t0 + 134.0
    rng = np.random.default_rng(292280304)
    n, d = 292, 4
    cands = []
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=np.float64)
            if a.shape == (n, d):
                nr = np.linalg.norm(a, axis=1, keepdims=True)
                if np.all(np.isfinite(a)) and float(nr.min()) > 1e-9:
                    cands.append(a / nr)
        except Exception:
            pass
    if len(cands) == 0:
        Xb = rng.standard_normal((n, d))
        Xb /= np.maximum(np.linalg.norm(Xb, axis=1, keepdims=True), 1e-12)
    else:
        Xb = cands[int(np.argmin([_max_cos(c) for c in cands]))].copy()
    best = _max_cos(Xb)
    Xbest = Xb.copy()
    try:
        s0 = _sig_vec(Xbest)
    except Exception:
        s0 = None
    bank = [s0] if s0 is not None else []
    it = 0
    down_ks = [1, 2, 3, 4, 6, 9, 12]
    up_ks = [1, 2, 3, 5, 8]
    while time.time() < explore_end:
        it += 1
        rem = explore_end - time.time()
        if rem < 2.0 and it > 4:
            break
        X = Xbest.copy()
        r = rng.random()
        # pick branch: down-ladder vs up-overshoot
        if r < 0.52:
            # DOWN: delete k highest-stress, reopt small, rebuild via holes
            k = down_ks[int(rng.integers(0, len(down_ks)))]
            if 292 - k < 280:
                k = 292 - 280
            try:
                m, stress, cnt, pmax, G = _stress_counts(X, beta=350.0)
            except Exception:
                stress = rng.random(n); cnt = np.zeros(n)
            if rng.random() < 0.5:
                key = stress + rng.random(n) * 1e-6 * (stress.max() + 1.0)
            else:
                key = cnt.astype(float) + rng.random(n) * 0.4
                # tie-break by pmax
                key += 0.1 * (pmax - pmax.min()) / max(1e-12, pmax.max() - pmax.min())
            idx = np.argsort(-key)[:k]
            keep = np.ones(n, dtype=bool); keep[idx] = False
            Xrem = X[keep]
            # reoptimise smaller code
            steps1 = int(rng.choice([600, 900, 1200]))
            b0 = float(rng.choice([15.0, 25.0, 40.0]))
            b1 = float(rng.choice([300.0, 500.0, 700.0]))
            lr0 = float(rng.choice([0.06, 0.09, 0.12]))
            mu = float(rng.choice([0.82, 0.86, 0.90]))
            est = steps1 * 0.00022 + 0.15
            if est > rem - 1.0:
                steps1 = max(300, int((rem - 1.2) / 0.00022))
                if steps1 < 250:
                    break
            Xrem = _anneal(Xrem, steps1, b0, b1, lr0, mu, deadline=explore_end - 0.8)
            # rebuild to 292 via largest holes
            nc = int(rng.choice([7000, 9000, 12000]))
            try:
                Xreb = _add_holes(Xrem, k, rng, n_cand=nc)
            except Exception:
                continue
            # small shake to break symmetry
            sig = float(rng.choice([0.002, 0.005, 0.012]))
            Xreb = Xreb + sig * rng.standard_normal(Xreb.shape)
            Xreb /= np.maximum(np.linalg.norm(Xreb, axis=1, keepdims=True), 1e-12)
            steps2 = int(rng.choice([800, 1200, 1600]))
            b0b = float(rng.choice([20.0, 35.0]))
            b1b = float(rng.choice([400.0, 650.0]))
            lr0b = float(rng.choice([0.06, 0.09]))
            mub = 0.86
            est2 = steps2 * 0.00022 + 0.15
            if est2 + 0.4 > rem - 0.5:
                steps2 = max(300, int((rem - 1.0) / 0.00022))
                if steps2 < 250:
                    break
            Xreb = _anneal(Xreb, steps2, b0b, b1b, lr0b, mub, deadline=explore_end - 0.4)
            # squeeze
            if time.time() < explore_end - 0.6:
                try:
                    Xreb = _anneal(Xreb, 200, 300.0, 800.0, 0.02, 0.85, deadline=explore_end - 0.2)
                except Exception:
                    pass
            try:
                val = _max_cos(Xreb)
            except Exception:
                continue
            dup = False
            try:
                sv = _sig_vec(Xreb)
                mind = min([_sig_dist(sv, s) for s in bank]) if len(bank) > 0 else 1.0
                if mind < 1e-8:
                    dup = True
            except Exception:
                sv = None; mind = 1.0
            # gate LP: only if promising and not obvious duplicate basin
            thresh = best + (1e-4 if dup else 3.5e-4)
            if np.isfinite(val) and val < thresh:
                lp_dl = min(time.time() + 3.5, explore_end - 0.15)
                if lp_dl - time.time() > 0.4:
                    try:
                        Xreb = _lp_polish(Xreb, lp_dl)
                    except Exception:
                        pass
                    try:
                        val = _max_cos(Xreb)
                        sv = _sig_vec(Xreb)
                    except Exception:
                        pass
            if np.isfinite(val) and val < best:
                best = val; Xbest = Xreb.copy()
                try:
                    bank.append(_sig_vec(Xbest))
                    if len(bank) > 40:
                        bank.pop(0)
                except Exception:
                    pass
            else:
                # keep diversity: if novel signature but slightly worse, occasional accept as restart base?
                if not dup and rng.random() < 0.03 and val < best + 8e-4:
                    # use as temporary base for next iter without overwriting best
                    # implement by small chance of setting Xbest to Xreb? no, keep best; instead ignore
                    pass
        else:
            # UP: overshoot to N+k, reopt, delete different point (nontrivial cycle)
            k = up_ks[int(rng.integers(0, len(up_ks)))]
            if 292 + k > 304:
                k = 304 - 292
            nc = int(rng.choice([7000, 9000, 11000]))
            try:
                Xup = _add_holes(X, k, rng, n_cand=nc)
            except Exception:
                continue
            steps1 = int(rng.choice([700, 1000, 1400]))
            b0 = float(rng.choice([15.0, 25.0, 35.0]))
            b1 = float(rng.choice([300.0, 500.0]))
            lr0 = float(rng.choice([0.06, 0.10]))
            mu = float(rng.choice([0.84, 0.88]))
            est = (steps1 * 0.00024) + 0.15
            if est > rem - 1.0:
                steps1 = max(300, int((rem - 1.2) / 0.00024))
                if steps1 < 250:
                    break
            Xup = _anneal(Xup, steps1, b0, b1, lr0, mu, deadline=explore_end - 0.8)
            # choose k to delete, ensuring nontrivial cycle (not exactly the inserted ones)
            try:
                m2, stress2, cnt2, pmax2, G2 = _stress_counts(Xup, beta=350.0)
            except Exception:
                continue
            if rng.random() < 0.6:
                key2 = stress2 + rng.random(Xup.shape[0]) * 1e-6 * (stress2.max() + 1.0)
            else:
                key2 = cnt2.astype(float) + rng.random(Xup.shape[0]) * 0.4
            order = np.argsort(-key2)
            del_idx = order[:k]
            tail = np.arange(292, 292 + k)
            if set(del_idx.tolist()) == set(tail.tolist()):
                # force nontrivial: replace one with best old stressed point
                for cand_i in order:
                    if int(cand_i) not in tail:
                        del_idx = np.array([int(cand_i)] + del_idx[1:].tolist())
                        break
            # also ensure at least one old deleted and at least one new kept (nontrivial)
            # if all deleted are new tail beyond above, already handled; if none deleted are new, force keep diversity half time
            is_new = np.array([int(v) >= 292 for v in del_idx])
            if is_new.sum() == 0 and rng.random() < 0.5:
                # force delete one new (the most stressed new) instead of least stressed old in del set
                new_stress = stress2[tail]
                worst_new = int(tail[int(np.argmax(new_stress))])
                del_idx[-1] = worst_new
            elif is_new.sum() == k and rng.random() < 0.5:
                # all new deleted = trivial revert; force delete an old instead
                for cand_i in order:
                    if int(cand_i) < 292:
                        del_idx[-1] = int(cand_i)
                        break
            keep2 = np.ones(Xup.shape[0], dtype=bool); keep2[del_idx] = False
            X2 = Xup[keep2]
            # reoptimise back at 292
            steps2 = int(rng.choice([800, 1200]))
            X2 = _anneal(X2, steps2, 25.0, 550.0, 0.08, 0.86, deadline=explore_end - 0.4)
            if time.time() < explore_end - 0.6:
                try:
                    X2 = _anneal(X2, 200, 300.0, 800.0, 0.02, 0.85, deadline=explore_end - 0.2)
                except Exception:
                    pass
            try:
                val = _max_cos(X2)
            except Exception:
                continue
            try:
                sv = _sig_vec(X2)
                mind = min([_sig_dist(sv, s) for s in bank]) if len(bank) > 0 else 1.0
                dup = mind < 1e-8
            except Exception:
                dup = False; sv = None
            thresh = best + (1e-4 if dup else 3.5e-4)
            if np.isfinite(val) and val < thresh:
                lp_dl = min(time.time() + 3.5, explore_end - 0.15)
                if lp_dl - time.time() > 0.4:
                    try:
                        X2 = _lp_polish(X2, lp_dl)
                    except Exception:
                        pass
                    try:
                        val = _max_cos(X2)
                    except Exception:
                        pass
            if np.isfinite(val) and val < best:
                best = val; Xbest = X2.copy()
                try:
                    bank.append(_sig_vec(Xbest))
                    if len(bank) > 40:
                        bank.pop(0)
                except Exception:
                    pass
    # final squeeze + polish
    try:
        Xbest = _anneal(Xbest, 500, 200.0, 900.0, 0.02, 0.85)
    except Exception:
        pass
    try:
        Xbest = _lp_polish(Xbest, t0 + TL - 0.8)
    except Exception:
        pass
    Xbest = Xbest / np.maximum(np.linalg.norm(Xbest, axis=1, keepdims=True), 1e-12)
    if not np.all(np.isfinite(Xbest)):
        Xbest = Xb
    return Xbest
