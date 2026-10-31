import time
import numpy as np

def _norm(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return x / n

def _max_cos(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def _degrees(x, tol=5e-7):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    m = float(np.max(G))
    B = G > (m - tol)
    deg = np.sum(B, axis=1)
    rowmax = np.max(G, axis=1)
    return deg, rowmax, m, G

def _hash_key(x):
    deg, _, m, _ = _degrees(x, tol=5e-7)
    return (tuple(sorted(deg.tolist())), round(float(m) * 1e7))

def _smooth_opt(x, steps, tau0, tau1, lr0, lr1, mom=0.9):
    n, d = x.shape
    vel = np.zeros_like(x)
    lt0, lt1 = np.log(tau0), np.log(tau1)
    ll0, ll1 = np.log(lr0), np.log(lr1)
    for t in range(steps):
        f = t / max(1, steps - 1)
        tau = float(np.exp(lt0 + (lt1 - lt0) * f))
        lr = float(np.exp(ll0 + (ll1 - ll0) * f))
        G = x @ x.T
        np.fill_diagonal(G, -np.inf)
        m = float(np.max(G))
        W = np.exp((G - m) / tau)
        np.fill_diagonal(W, 0.0)
        row = np.sum(W, axis=1, keepdims=True)
        row[row < 1e-12] = 1e-12
        grad = (W @ x) / row
        dot = np.sum(grad * x, axis=1, keepdims=True)
        grad = grad - dot * x
        gn = np.linalg.norm(grad, axis=1, keepdims=True)
        scale = np.ones_like(gn)
        big = gn > 2.0
        scale[big] = 2.0 / np.maximum(gn[big], 1e-12)
        grad = grad * scale
        vel = mom * vel + (1.0 - mom) * grad
        bc = 1.0 - mom ** (t + 1)
        step = vel / bc
        x = x - lr * step
        x = _norm(x)
    return x

def _lp_polish(x, deadline, cap=2500, r0=1e-3):
    from scipy.optimize import linprog
    from scipy.sparse import coo_matrix
    x = _norm(x)
    n, d = x.shape
    best = _max_cos(x)
    radius = r0
    iu0 = np.triu_indices(n, 1)
    it = 0
    while time.time() < deadline and radius > 1e-14 and it < 300:
        it += 1
        G = x @ x.T
        c = G[iu0]
        m = float(c.max())
        thresh = m - max(20.0 * radius, 1e-9)
        idx = np.where(c > thresh)[0]
        if idx.size == 0:
            break
        if idx.size > cap:
            order = np.argpartition(c[idx], -cap)[-cap:]
            idx = idx[order]
        I = iu0[0][idx]; J = iu0[1][idx]
        csel = c[idx]
        p = idx.size
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cI = I[:, None] * d + np.arange(d)
        cJ = J[:, None] * d + np.arange(d)
        cols = np.concatenate([cI, cJ, np.full((p, 1), n * d)], axis=1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], axis=1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
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
        y = _norm(y)
        v = _max_cos(y)
        if v < best:
            x = y; best = v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x, best

def _hole_insert(X, rng, n_cand=50000, batch=10000):
    n, d = X.shape
    best_c = None
    best_s = np.inf
    nb = max(1, n_cand // batch)
    for _ in range(nb):
        C = rng.standard_normal(size=(batch, d))
        C = _norm(C)
        S = C @ X.T
        sm = S.max(axis=1)
        i = int(np.argmin(sm))
        if float(sm[i]) < best_s:
            best_s = float(sm[i])
            best_c = C[i].copy()
    return np.vstack([X, best_c.reshape(1, -1)])

def solve():
    t0 = time.time()
    HARD_END = t0 + 174.0
    FINAL_RESERVE = 30.0
    FINAL_START = HARD_END - FINAL_RESERVE
    rng = np.random.default_rng(12345)
    n, d = 153, 4
    best_x = None
    best_v = 1e9
    for fname in ("best.npy", "record.npy"):
        try:
            a = np.load(fname)
            if a.shape == (n, d):
                a = _norm(a)
                v = _max_cos(a)
                if v < best_v:
                    best_v = v
                    best_x = a.copy()
        except Exception:
            pass
    if best_x is None:
        best_x = _norm(rng.standard_normal(size=(n, d)))
        best_v = _max_cos(best_x)
    record_hash = _hash_key(best_x)
    seen = set([record_hash])
    global_best_x = best_x.copy()
    global_best_v = best_v

    def consider(Y, v):
        nonlocal global_best_x, global_best_v
        h = _hash_key(Y)
        # diversity: always record hash, keep best value
        # prefer new basin slightly: if new hash and within 2e-6, still note
        if h not in seen:
            seen.add(h)
        if v < global_best_v:
            global_best_v = v
            global_best_x = Y.copy()
            return True
        return False

    def time_left():
        return HARD_END - time.time()

    # Phase 1: overshoot chain 154,155,156
    chain = {}
    cur = global_best_x.copy()
    for N in (154, 155, 156):
        if time.time() > FINAL_START - 55:
            break
        # insert at largest hole from 50000 candidates
        cur = _hole_insert(cur, rng, n_cand=50000, batch=10000)
        # jointly soft-max optimise
        # adapt steps to remaining time
        steps = 1200 if time_left() > 80 else 700
        cur = _smooth_opt(cur, steps=steps, tau0=0.025, tau1=0.002, lr0=0.02, lr1=0.002, mom=0.9)
        dl = min(time.time() + 6.0, FINAL_START - 50)
        if dl > time.time() + 0.5:
            cur, v = _lp_polish(cur, deadline=dl, cap=2500, r0=1e-3)
        else:
            v = _max_cos(cur)
        chain[N] = (cur.copy(), float(v))

    # Phase 2: return to 153 by trial deletions
    # helper for short 1-sec polish scoring
    def polish_short(Y, budget=1.0):
        dl = min(time.time() + budget, FINAL_START - 5)
        if dl <= time.time() + 0.15:
            return Y, _max_cos(Y)
        try:
            Yp, vp = _lp_polish(Y, deadline=dl, cap=2000, r0=7e-4)
            return Yp, vp
        except Exception:
            return Y, _max_cos(Y)

    for N in sorted(chain.keys()):
        if time.time() > FINAL_START - 35:
            break
        XN, _ = chain[N]
        k = N - 153
        deg, rowmax, m, G = _degrees(XN, tol=1e-6)
        order = np.argsort(-deg)
        # candidate pool: inserted points (last k) + high-degree points
        inserted = list(range(N - k, N))
        cand_pool = list(inserted)
        for o in order:
            o = int(o)
            if o not in cand_pool and len(cand_pool) < 12:
                cand_pool.append(o)
        if k == 1:
            # trial-delete each candidate with 1-sec LP polish
            for c in cand_pool:
                if time.time() > FINAL_START - 20:
                    break
                mask = np.ones(N, dtype=bool); mask[c] = False
                Y = XN[mask].copy()
                Yp, vp = polish_short(Y, budget=1.0)
                consider(Yp, vp)
        else:
            # baseline: delete inserted points
            base_del = set(inserted)
            # evaluate baseline
            if time.time() < FINAL_START - 20:
                mask = np.ones(N, dtype=bool); mask[list(base_del)] = False
                Y0 = XN[mask].copy()
                Y0p, v0p = polish_short(Y0, budget=1.0)
                consider(Y0p, v0p)
            # variants: keep one inserted, delete one high-degree instead
            # need k deletions: replace one of base_del with a high-deg candidate
            high = [c for c in order[:10] if c not in base_del]
            for h in high[:8]:
                if time.time() > FINAL_START - 20:
                    break
                # swap: keep last inserted, delete h instead
                # choose to keep inserted[-1], delete h + other inserted[:-1]
                delset = set(inserted[:-1]) | {int(h)}
                if len(delset) != k:
                    continue
                mask = np.ones(N, dtype=bool); mask[list(delset)] = False
                Y = XN[mask].copy()
                Yp, vp = polish_short(Y, budget=1.0)
                consider(Yp, vp)

    # Phase 3: reverse path shrink to 142 then regrow
    if time.time() < FINAL_START - 40:
        try:
            deg0, rowmax0, m0, G0 = _degrees(global_best_x, tol=5e-7)
            # rattler: zero-contact with smallest rowmax
            zero = np.where(deg0 == 0)[0]
            if len(zero) > 0:
                rattler = int(zero[np.argmin(rowmax0[zero])])
            else:
                rattler = int(np.argmin(rowmax0))
            # 10 random others (deterministic)
            others = [i for i in range(n) if i != rattler]
            sel = rng.choice(len(others), size=10, replace=False)
            dels = set([rattler] + [others[i] for i in sel])
            mask = np.ones(n, dtype=bool); mask[list(dels)] = False
            Xs = global_best_x[mask].copy()  # 142
            # fully re-optimise there
            Xs = _smooth_opt(Xs, steps=2000 if time_left() > 70 else 1000,
                             tau0=0.03, tau1=0.002, lr0=0.025, lr1=0.002, mom=0.9)
            dl = min(time.time() + 6.0, FINAL_START - 30)
            if dl > time.time() + 0.5:
                Xs, _ = _lp_polish(Xs, deadline=dl, cap=2500, r0=1e-3)
            # greedily regrow to 153 with brief joint polish after each addition
            for M in range(143, 154):
                if time.time() > FINAL_START - 18:
                    break
                Xs = _hole_insert(Xs, rng, n_cand=50000, batch=10000)
                Xs = _smooth_opt(Xs, steps=600, tau0=0.02, tau1=0.003, lr0=0.015, lr1=0.002, mom=0.9)
            # polish endpoint
            if Xs.shape[0] == 153:
                dl2 = min(time.time() + 5.0, FINAL_START - 10)
                if dl2 > time.time() + 0.5:
                    Xs, vs = _lp_polish(Xs, deadline=dl2, cap=2500, r0=1e-3)
                else:
                    vs = _max_cos(Xs)
                consider(Xs, vs)
        except Exception:
            pass

    # Final full LP polish 30s on best 153 endpoint
    try:
        fb = _norm(global_best_x)
        dl = HARD_END - 1.0
        if time.time() < dl - 0.5:
            fb, fv = _lp_polish(fb, deadline=dl, cap=3000, r0=1e-3)
            if fv < global_best_v:
                global_best_x = fb
                global_best_v = fv
    except Exception:
        pass
    global_best_x = _norm(global_best_x)
    return global_best_x
