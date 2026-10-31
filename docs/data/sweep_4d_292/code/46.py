import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _norm(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return x / n

def _max_cos(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _anneal_full(x, steps, beta0, beta1, lr0, lr1, mu=0.85):
    x = _norm(x.copy())
    vel = np.zeros_like(x)
    lb = np.log(beta1 / beta0) if beta1 > beta0 else 0.0
    ll = np.log(lr1 / lr0) if abs(lr1 - lr0) > 1e-18 else 0.0
    for s in range(steps):
        frac = s / max(1, steps - 1)
        beta = beta0 * np.exp(lb * frac)
        lr = lr0 * np.exp(ll * frac)
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        gmax = G.max()
        W = np.exp(beta * (G - gmax))
        np.fill_diagonal(W, 0.0)
        r = W.sum(axis=1, keepdims=True)
        np.maximum(r, 1e-30, out=r)
        A = (W @ x) / r
        At = A - np.sum(A * x, axis=1, keepdims=True) * x
        vel = mu * vel + (1.0 - mu) * At
        bias = 1.0 - mu ** (s + 1)
        x = x - lr * (vel / max(bias, 1e-12))
        x = _norm(x)
    return x

def _anneal_masked(x, mask, steps, beta0, beta1, lr0, lr1, mu=0.85):
    x = _norm(x.copy())
    vel = np.zeros_like(x)
    lb = np.log(beta1 / beta0) if beta1 > beta0 else 0.0
    ll = np.log(lr1 / lr0) if abs(lr1 - lr0) > 1e-18 else 0.0
    m = np.asarray(mask, dtype=bool)
    if not np.any(m):
        return x
    for s in range(steps):
        frac = s / max(1, steps - 1)
        beta = beta0 * np.exp(lb * frac)
        lr = lr0 * np.exp(ll * frac)
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        gmax = G.max()
        W = np.exp(beta * (G - gmax))
        np.fill_diagonal(W, 0.0)
        r = W.sum(axis=1, keepdims=True)
        np.maximum(r, 1e-30, out=r)
        A = (W @ x) / r
        At = A - np.sum(A * x, axis=1, keepdims=True) * x
        vel = mu * vel + (1.0 - mu) * At
        bias = 1.0 - mu ** (s + 1)
        step = lr * (vel / max(bias, 1e-12))
        step[~m] = 0.0
        x = x - step
        x = _norm(x)
    return x

def _lp_polish(x, seconds, r0=8e-4):
    x = _norm(np.asarray(x, dtype=float).copy())
    n, d = x.shape
    best = _max_cos(x)
    xb = x.copy()
    if seconds <= 0:
        return xb
    start = time.time()
    radius = r0
    iu0 = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        G = x @ x.T
        c = G[iu0]
        m = c.max()
        thresh = m - max(20.0 * radius, 1e-9)
        sel = c > thresh
        if not np.any(sel):
            break
        I = iu0[0][sel]; J = iu0[1][sel]
        p = len(I)
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        ci = I[:, None] * d + np.arange(d)
        cj = J[:, None] * d + np.arange(d)
        cols = np.concatenate([ci, cj, np.full((p, 1), n * d)], axis=1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], axis=1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv); obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-c[sel], A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y = _norm(y)
        v = _max_cos(y)
        if v < best:
            best = v; xb = y.copy(); x = y
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return xb

def _contacts(x, tol=1e-7):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    m = float(G.max())
    iu = np.triu_indices(x.shape[0], 1)
    c = G[iu]
    sel = c > m - tol
    s = set(zip(iu[0][sel].tolist(), iu[1][sel].tolist()))
    deg = np.zeros(x.shape[0], dtype=int)
    for (i, j) in s:
        deg[i] += 1; deg[j] += 1
    return s, deg, m

def _refine_holes(holes, Xkeep, steps=220, lr0=0.03, lr1=0.005, beta0=60.0, beta1=450.0):
    h = _norm(holes.copy())
    vel = np.zeros_like(h)
    mu = 0.8
    lb = np.log(beta1 / beta0)
    ll = np.log(lr1 / lr0)
    for s in range(steps):
        frac = s / max(1, steps - 1)
        beta = beta0 * np.exp(lb * frac)
        lr = lr0 * np.exp(ll * frac)
        D = h @ Xkeep.T
        dm = D.max(axis=1, keepdims=True)
        W = np.exp(beta * (D - dm))
        r = W.sum(axis=1, keepdims=True)
        np.maximum(r, 1e-30, out=r)
        A = (W @ Xkeep) / r
        At = A - np.sum(A * h, axis=1, keepdims=True) * h
        vel = mu * vel + (1.0 - mu) * At
        bias = 1.0 - mu ** (s + 1)
        h = h - lr * (vel / max(bias, 1e-12))
        h = _norm(h)
    return h

def solve():
    t0 = time.time()
    DEAD = 172.0
    rng = np.random.default_rng(1234567)
    def _load(p):
        try:
            a = np.load(p)
            a = np.asarray(a, dtype=float)
            if a.shape == (292, 4) and np.all(np.isfinite(a)):
                n = np.linalg.norm(a, axis=1, keepdims=True); n[n == 0] = 1.0
                return a / n
        except Exception:
            pass
        return None
    best = _load("best.npy")
    rec = _load("record.npy")
    if best is None:
        best = _norm(rng.standard_normal((292, 4)))
    if rec is not None and _max_cos(rec) < _max_cos(best):
        best = rec.copy()
    best_val = _max_cos(best)
    incumbent = best.copy()
    inc_contacts, _, _ = _contacts(incumbent)
    cur = best.copy()
    cur_val = best_val
    archive = [incumbent.copy()]
    arch_vals = [best_val]
    ks = [4, 8, 12, 16, 20]
    it = 0
    TARGET = 0.90347
    while time.time() - t0 < DEAD - 12.0:
        it += 1
        k = ks[(it - 1) % len(ks)]
        # vary source: mostly cur, sometimes random archive member
        if len(archive) > 1 and rng.random() < 0.3:
            src_idx = int(rng.integers(0, len(archive)))
            src = archive[src_idx]
        else:
            src = cur
        cset, deg, msrc = _contacts(src, tol=1e-7)
        ratt = np.where(deg == 0)[0].tolist()
        low = np.where(deg <= 4)[0].tolist()
        high = np.where(deg >= 8)[0].tolist()
        # build deletion set mixture
        dell = []
        def _add(pool, nwant):
            cand = [i for i in pool if i not in dell]
            rng.shuffle(cand)
            for i in cand[:nwant]:
                dell.append(int(i))
        if ratt:
            _add(ratt, min(len(ratt), 1 + k // 10))
        nlow = max(1, int(round(0.35 * k))) if low else 0
        nhigh = max(1, int(round(0.35 * k))) if high else 0
        _add(low, nlow)
        _add(high, nhigh)
        # fill remainder: prefer low/high mix then random
        rest_pool = list(range(292))
        rng.shuffle(rest_pool)
        # prioritize remaining low/high first
        extra = []
        for i in low + high:
            if i not in dell:
                extra.append(i)
        rng.shuffle(extra)
        for i in extra:
            if len(dell) >= k:
                break
            dell.append(int(i))
        for i in rest_pool:
            if len(dell) >= k:
                break
            if int(i) not in dell:
                dell.append(int(i))
        dell = dell[:k]
        keep_idx = [i for i in range(292) if i not in set(dell)]
        Xkeep = src[keep_idx]
        # massive hole sampling >=65536
        N_CAND = 65536
        cand = rng.standard_normal((N_CAND, 4))
        cand = cand / np.linalg.norm(cand, axis=1, keepdims=True)
        bscore = np.empty(N_CAND)
        cs = 8192
        for a in range(0, N_CAND, cs):
            b = min(a + cs, N_CAND)
            D = cand[a:b] @ Xkeep.T
            bscore[a:b] = D.max(axis=1)
        order = np.argsort(bscore)
        H = 300
        h0 = cand[order[:H]]
        # refine best few hundred holes
        hR = _refine_holes(h0, Xkeep, steps=180)
        D2 = hR @ Xkeep.T
        s2 = D2.max(axis=1)
        o2 = np.argsort(s2)
        hR = hR[o2]; s2 = s2[o2]
        # greedy insertion under temporary separation target
        sel = []
        sel_scores = []
        for idx in range(H):
            if len(sel) >= k:
                break
            c = hR[idx]
            mk = float(s2[idx])
            if mk > TARGET:
                # still allow but check mutual below; first pass requires TARGET
                continue
            ok = True
            for s in sel:
                if float(s @ c) > TARGET:
                    ok = False; break
            if ok:
                sel.append(c)
        # relaxed second pass if needed
        relax = TARGET + 0.002
        if len(sel) < k:
            for idx in range(H):
                if len(sel) >= k:
                    break
                c = hR[idx]
                if any(np.allclose(c, s) for s in sel):
                    continue
                # check against already selected with relaxed bound
                ok = True
                for s in sel:
                    if float(s @ c) > relax:
                        ok = False; break
                if ok and float(c @ Xkeep.T.max() if False else 0) == 0:
                    pass
                if ok:
                    # avoid duplicate pick
                    already = False
                    for s in sel:
                        if np.linalg.norm(c - s) < 1e-12:
                            already = True; break
                    if not already:
                        # ensure not picked before (compare index)
                        dup = False
                        for s in sel:
                            if np.linalg.norm(c - s) < 1e-9:
                                dup = True; break
                        if not dup:
                            # check it wasn't already selected (by value)
                            sel.append(c)
                            if len(sel) >= k:
                                break
            # if still short due to logic, fill greedily minimizing max
            while len(sel) < k:
                best_c = None; best_m = 1e9
                for idx in range(H):
                    c = hR[idx]
                    skip = False
                    for s in sel:
                        if np.linalg.norm(c - s) < 1e-9:
                            skip = True; break
                    if skip:
                        continue
                    m1 = float(s2[idx])
                    m2 = -2.0
                    for s in sel:
                        m2 = max(m2, float(s @ c))
                    mm = max(m1, m2)
                    if mm < best_m:
                        best_m = mm; best_c = c
                if best_c is None:
                    break
                sel.append(best_c)
        sel = sel[:k]
        while len(sel) < k:
            r = rng.standard_normal((1, 4)); r = r / np.linalg.norm(r)
            sel.append(r[0])
        Xnew = np.vstack([Xkeep, np.array(sel)])
        # neighbour mask for initial local optimisation: inserted + neighbours cos>0.75
        nkeep = Xkeep.shape[0]
        ins_idx = np.arange(nkeep, 292)
        Dins = Xnew[:nkeep] @ Xnew[nkeep:].T
        neigh = np.where(Dins.max(axis=1) > 0.75)[0]
        mask = np.zeros(292, dtype=bool)
        mask[nkeep:] = True
        mask[neigh] = True
        # stage 1: local only
        try:
            Xnew = _anneal_masked(Xnew, mask, steps=250, beta0=40.0, beta1=350.0, lr0=0.03, lr1=0.004)
        except Exception:
            pass
        # stage 2: full anneal
        try:
            Xnew = _anneal_full(Xnew, steps=350, beta0=30.0, beta1=500.0, lr0=0.025, lr1=0.003)
        except Exception:
            pass
        # stage 3: LP polish (short)
        time_left = DEAD - (time.time() - t0)
        lp_sec = min(3.0, max(0.8, time_left - 8.0))
        try:
            Xnew = _lp_polish(Xnew, seconds=lp_sec, r0=8e-4)
        except Exception:
            pass
        v = _max_cos(Xnew)
        ns, _, _ = _contacts(Xnew, tol=1e-7)
        diff_inc = len(ns.symmetric_difference(inc_contacts))
        diff_cur_set, _, _ = _contacts(cur, tol=1e-7)
        diff_cur = len(ns.symmetric_difference(diff_cur_set))
        improved = v < best_val - 1e-12
        novel = diff_inc >= 20
        if improved:
            best = Xnew.copy(); best_val = v
            cur = Xnew.copy(); cur_val = v
            archive.append(Xnew.copy()); arch_vals.append(v)
            inc_contacts = ns  # update? keep original incumbent? keep original for diversity; but update best reference
            # keep original incumbent contacts for novelty measure? restore: recompute from best? we want diff vs original record basin;
            # keep inc as original: so revert
            inc_contacts, _, _ = _contacts(incumbent, tol=1e-7) if False else (ns, None, None)
            # actually keep incumbent fixed as initial; recompute
            inc_contacts, _, _ = _contacts(_load("best.npy") if _load("best.npy") is not None else incumbent, tol=1e-7)
            # simpler: keep inc_contacts as before unless we want new reference; retain old
            # To avoid confusion, reload original diff reference from initial incumbent copy
            # we stored incumbent never changed, so:
            inc_contacts, _, _ = _contacts(incumbent, tol=1e-7)
            # if improved, also update incumbent reference for future diff? keep original + best both
        elif novel:
            archive.append(Xnew.copy()); arch_vals.append(v)
            # move to new basin if not much worse
            if v < cur_val + 0.002:
                cur = Xnew.copy(); cur_val = v
            # cap archive
            if len(archive) > 12:
                # drop worst non-best
                worst = int(np.argmax(arch_vals))
                # never drop best
                bi = int(np.argmin(arch_vals))
                if worst != bi:
                    del archive[worst]; del arch_vals[worst]
        else:
            # Metropolis move to change basin occasionally
            frac = (time.time() - t0) / DEAD
            Temp = 0.0022 * (2.0e-05 / 0.0022) ** min(1.0, max(0.0, frac))
            if v < cur_val or rng.random() < np.exp(-(v - cur_val) / max(Temp, 1e-9)):
                # only accept if somewhat close or diff non-trivial and diff_cur>=10
                if diff_cur >= 10 or v < cur_val + 0.001:
                    cur = Xnew.copy(); cur_val = v
        if time.time() - t0 > DEAD - 6.0:
            break
    # final polish best with remaining time
    rem = DEAD - (time.time() - t0)
    if rem > 1.5:
        try:
            best = _lp_polish(best, seconds=min(rem - 0.5, 6.0), r0=6e-4)
            best_val = _max_cos(best)
        except Exception:
            pass
    best = _norm(best)
    return best
