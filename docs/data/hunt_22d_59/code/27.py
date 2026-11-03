import time
import numpy as np

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def solve():
    t0 = time.time()
    BUDGET = 162.0
    np.random.seed(0)
    from scipy.optimize import linprog

    n, d = 59, 22
    # load
    cands = []
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn).astype(np.float64)
            if a.shape == (n, d):
                nr = np.linalg.norm(a, axis=1, keepdims=True); nr[nr == 0] = 1.0
                a = a / nr
                if np.all(np.isfinite(a)):
                    cands.append(a)
        except Exception:
            pass
    if not cands:
        a = np.random.randn(n, d); a /= np.linalg.norm(a, axis=1, keepdims=True)
        cands.append(a)
    vals = [_max_cos(c) for c in cands]
    best = cands[int(np.argmin(vals))].copy()
    best_val = float(min(vals))
    cur = best.copy()
    cur_val = best_val

    def adam_polish(X, steps, a0, a1, lr0, lr1):
        X = X.copy()
        m = np.zeros_like(X); v = np.zeros_like(X)
        b1, b2, eps = 0.9, 0.999, 1e-8
        la0, la1 = np.log(a0), np.log(a1)
        ll0, ll1 = np.log(lr0), np.log(lr1)
        cb = X.copy(); cv = _max_cos(X)
        for t in range(1, steps + 1):
            if (t & 127) == 0 and (time.time() - t0) > BUDGET:
                break
            frac = t / steps
            alpha = float(np.exp(la0 + (la1 - la0) * frac))
            lr = float(np.exp(ll0 + (ll1 - ll0) * frac))
            G = X @ X.T
            np.fill_diagonal(G, -2.0)
            M = float(np.max(G))
            G -= M; G *= alpha
            np.exp(G, out=G)
            np.fill_diagonal(G, 0.0)
            S = float(np.sum(G))
            if not np.isfinite(S) or S <= 0:
                continue
            Gr = (G @ X) / S
            Gr -= X * np.sum(Gr * X, axis=1, keepdims=True)
            m *= b1; m += (1.0 - b1) * Gr
            v *= b2; v += (1.0 - b2) * (Gr * Gr)
            c1 = 1.0 - b1 ** t; c2 = 1.0 - b2 ** t
            upd = (m / c1) / (np.sqrt(v / c2) + eps)
            X -= lr * upd
            X /= (np.linalg.norm(X, axis=1, keepdims=True) + 1e-30)
            if (t % 400 == 0) or (t == steps):
                nv = _max_cos(X)
                if nv < cv:
                    cv = nv; cb = X.copy()
        fv = _max_cos(X)
        if fv < cv:
            cv = fv; cb = X.copy()
        return cb, cv

    def tangent_lp_step(Xcur, Sset, M0, delta, cap_active=800, force_drop=2e-4, allow_rise=1e-4):
        # active pairs by current cosine
        G = Xcur @ Xcur.T
        iu = np.triu_indices(n, 1)
        gvals = G[iu]
        # keep pairs with g > M0-0.002 to bound blow-up, cap to largest cap_active
        mask = gvals > (M0 - 0.002)
        idx = np.where(mask)[0]
        if idx.size > cap_active:
            part = np.argpartition(gvals[idx], -cap_active)[-cap_active:]
            idx = idx[part]
        pairs = [(int(iu[0][k]), int(iu[1][k])) for k in idx]
        Slookup = set()
        if Sset is not None:
            for (a, b) in Sset:
                if a > b:
                    a, b = b, a
                Slookup.add((a, b))
        nons = [p for p in pairs if p not in Slookup]
        # also ensure S pairs included even if not in active window
        if Sset is not None:
            for p in Slookup:
                if p not in pairs:
                    pairs.append(p)
        Nv = n * d + 1
        m_ub = len(nons) + len(Slookup) + 1
        A_ub = np.zeros((m_ub, Nv), dtype=np.float64)
        b_ub = np.zeros((m_ub,), dtype=np.float64)
        r = 0
        for (i, j) in nons:
            cij = float(Xcur[i] @ Xcur[j])
            A_ub[r, i * d:(i + 1) * d] = Xcur[j]
            A_ub[r, j * d:(j + 1) * d] = Xcur[i]
            A_ub[r, -1] = -1.0
            b_ub[r] = -cij
            r += 1
        for (i, j) in Slookup:
            cij = float(Xcur[i] @ Xcur[j])
            A_ub[r, i * d:(i + 1) * d] = Xcur[j]
            A_ub[r, j * d:(j + 1) * d] = Xcur[i]
            A_ub[r, -1] = 0.0
            b_ub[r] = (M0 - force_drop) - cij
            r += 1
        # t <= M0+allow
        A_ub[r, -1] = 1.0
        b_ub[r] = M0 + allow_rise
        # equalities: Xi . dxi = 0
        A_eq = np.zeros((n, Nv), dtype=np.float64)
        for i in range(n):
            A_eq[i, i * d:(i + 1) * d] = Xcur[i]
        b_eq = np.zeros((n,), dtype=np.float64)
        c_obj = np.zeros((Nv,), dtype=np.float64); c_obj[-1] = 1.0
        bounds = [(-delta, delta)] * (n * d) + [(-1.0, 2.0)]
        try:
            res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                          bounds=bounds, method="highs",
                          options={"presolve": True, "time_limit": 4.0})
        except Exception:
            return None, None
        if res.status != 0 or res.x is None:
            return None, None
        dx = res.x[:n * d].reshape(n, d)
        # try to extract duals for nons rows
        duals = None
        try:
            ine = getattr(res, "ineqlin", None)
            if ine is not None:
                if hasattr(ine, "marginals"):
                    duals = np.asarray(ine.marginals, dtype=float).ravel()
                elif isinstance(ine, dict) and "marginals" in ine:
                    duals = np.asarray(ine["marginals"], dtype=float).ravel()
        except Exception:
            duals = None
        return dx, duals

    def standard_lp_duals(Xcur, M0, delta=0.006, cap_active=900):
        G = Xcur @ Xcur.T
        iu = np.triu_indices(n, 1)
        gvals = G[iu]
        thr = float(np.max(gvals)) - 1e-5
        idx = np.where(gvals >= thr)[0]
        if idx.size > cap_active:
            part = np.argpartition(gvals[idx], -cap_active)[-cap_active:]
            idx = idx[part]
        pairs = [(int(iu[0][k]), int(iu[1][k]), float(gvals[k])) for k in idx]
        Nv = n * d + 1
        m_ub = len(pairs)
        if m_ub == 0:
            return [], None
        A_ub = np.zeros((m_ub, Nv), dtype=np.float64)
        b_ub = np.zeros((m_ub,), dtype=np.float64)
        for r, (i, j, cij) in enumerate(pairs):
            A_ub[r, i * d:(i + 1) * d] = Xcur[j]
            A_ub[r, j * d:(j + 1) * d] = Xcur[i]
            A_ub[r, -1] = -1.0
            b_ub[r] = -cij
        A_eq = np.zeros((n, Nv), dtype=np.float64)
        for i in range(n):
            A_eq[i, i * d:(i + 1) * d] = Xcur[i]
        b_eq = np.zeros((n,), dtype=np.float64)
        c_obj = np.zeros((Nv,)); c_obj[-1] = 1.0
        bounds = [(-delta, delta)] * (n * d) + [(-1.0, 2.0)]
        try:
            res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                          bounds=bounds, method="highs",
                          options={"presolve": True, "time_limit": 4.0})
        except Exception:
            return pairs, None
        duals = None
        try:
            ine = getattr(res, "ineqlin", None)
            if ine is not None and hasattr(ine, "marginals"):
                duals = np.asarray(ine.marginals, dtype=float).ravel()
        except Exception:
            duals = None
        return pairs, duals

    # short initial polish to sit at basin
    if (time.time() - t0) < BUDGET - 10:
        best, best_val = adam_polish(best, steps=2500, a0=80, a1=900, lr0=0.004, lr1=0.0004)
        cur, cur_val = best.copy(), best_val

    # get tight contact list for subset sampling
    rng = np.random.default_rng(0)
    trial = 0
    # try to get LP marginals once for dominant-contact ranking
    dom_pairs = []
    try:
        prs, du = standard_lp_duals(cur, cur_val)
        if prs:
            if du is not None and len(du) == len(prs):
                order = np.argsort(np.abs(du))[::-1]
                dom_pairs = [(prs[k][0], prs[k][1]) for k in order[:120]]
            else:
                prs_sorted = sorted(prs, key=lambda x: -x[2])
                dom_pairs = [(a, b) for a, b, _ in prs_sorted[:120]]
    except Exception:
        dom_pairs = []
    if not dom_pairs:
        G = cur @ cur.T; np.fill_diagonal(G, -2.0)
        iu = np.triu_indices(n, 1)
        gv = G[iu]
        k = min(120, gv.size)
        top = np.argpartition(gv, -k)[-k:]
        dom_pairs = [(int(iu[0][t]), int(iu[1][t])) for t in top]

    while (time.time() - t0) < BUDGET - 18:
        trial += 1
        M0 = cur_val
        # select 20-50 dominant contacts: mix of top-marginal and random
        sz = int(rng.integers(20, 51))
        # refresh dominant list every few trials
        if trial % 4 == 0:
            try:
                prs, du = standard_lp_duals(cur, cur_val)
                if prs:
                    if du is not None and len(du) == len(prs):
                        order = np.argsort(np.abs(du))[::-1]
                        dom_pairs = [(prs[k][0], prs[k][1]) for k in order[:120]]
                    else:
                        prs_sorted = sorted(prs, key=lambda x: -x[2])
                        dom_pairs = [(a, b) for a, b, _ in prs_sorted[:120]]
            except Exception:
                pass
        if len(dom_pairs) >= sz:
            # 70% from top-60, 30% random from rest to diversify
            top60 = dom_pairs[:60]
            rest = dom_pairs[60:] if len(dom_pairs) > 60 else dom_pairs
            n_top = min(len(top60), int(sz * 0.7))
            n_rest = sz - n_top
            sel = []
            if n_top > 0:
                ii = rng.choice(len(top60), size=n_top, replace=False)
                sel += [top60[k] for k in ii]
            if n_rest > 0 and rest:
                ii = rng.choice(len(rest), size=min(n_rest, len(rest)), replace=False)
                sel += [rest[k] for k in ii]
            # if still short, random pairs from tight set
            while len(sel) < sz:
                a = int(rng.integers(0, n)); b = int(rng.integers(0, n))
                if a == b:
                    continue
                if a > b:
                    a, b = b, a
                if (a, b) not in sel:
                    sel.append((a, b))
            Sset = sel[:sz]
        else:
            Sset = dom_pairs
        # forced sequence of 5-10 relinearized LPs
        nseq = int(rng.integers(5, 9))
        delta = float(rng.choice([0.008, 0.012, 0.018]))
        Xf = cur.copy()
        ok = True
        for _ in range(nseq):
            if (time.time() - t0) > BUDGET - 18:
                ok = False
                break
            dx, _ = tangent_lp_step(Xf, Sset, M0, delta)
            if dx is None:
                ok = False
                break
            # apply with trust scaling (clip extreme)
            Xf = Xf + dx.reshape(n, d)
            Xf /= (np.linalg.norm(Xf, axis=1, keepdims=True) + 1e-30)
            # early abort if exploded
            if _max_cos(Xf) > M0 + 0.002:
                # still continue? break to polish
                pass
        if not ok:
            # fallback: small random shake of involved points
            inv = sorted(set([a for a, b in Sset] + [b for a, b in Sset]))
            Xf = cur.copy()
            Xf[inv] += 0.01 * np.random.randn(len(inv), d)
            Xf /= (np.linalg.norm(Xf, axis=1, keepdims=True) + 1e-30)
        # short Adam polish after forced move
        remain = BUDGET - (time.time() - t0)
        if remain < 8:
            break
        steps = 1800 if remain > 50 else 1200
        Yp, Yv = adam_polish(Xf, steps=steps, a0=60, a1=900, lr0=0.004, lr1=0.0004)
        if Yv < best_val - 1e-12:
            best, best_val = Yp.copy(), float(Yv)
            cur, cur_val = Yp.copy(), float(Yv)
        elif Yv < cur_val + 0.0004:
            # accept slightly worse to change basin
            cur, cur_val = Yp.copy(), float(Yv)
        else:
            # occasionally accept random subset move anyway every ~7 trials
            if trial % 7 == 0:
                cur, cur_val = Yp.copy(), float(Yv)

    # final squeeze
    if (time.time() - t0) < BUDGET - 2:
        best, best_val = adam_polish(best, steps=2500, a0=800, a1=2000, lr0=0.0012, lr1=0.00015)
    best /= (np.linalg.norm(best, axis=1, keepdims=True) + 1e-30)
    return best
