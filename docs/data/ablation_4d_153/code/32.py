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

def _degrees(x, m=None, tol=1e-6):
    G = x @ x.T
    if m is None:
        np.fill_diagonal(G, -2.0)
        m = float(np.max(G))
    else:
        np.fill_diagonal(G, -2.0)
    B = G > (m - tol)
    deg = np.sum(B, axis=1)
    rowmax = np.max(G, axis=1)
    return deg, rowmax, m, G

def _hash_key(x):
    deg, _, m, _ = _degrees(x, tol=5e-7)
    return (tuple(sorted(deg.tolist())), round(float(m)*1e7))

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
        # clip
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
    nb = n_cand // batch
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
    # reserve 30s final polish
    FINAL_RESERVE = 30.0
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

    def time_left():
        return HARD_END - time.time()

    # ---------- Phase 1: overshoot chain 154,155,156 ----------
    chain = {}
    cur = global_best_x.copy()
    for N in (154, 155, 156):
        if time_left() < (FINAL_RESERVE + 45):
            break
        cur = _hole_insert(cur, rng, n_cand=50000, batch=10000)
        # joint soft-max
        cur = _smooth_opt(cur, steps=500, tau0=0.02, tau1=0.002, lr0=0.12, lr1=0.008)
        dl = time.time() + min(3.0, max(0.5, time_left() - FINAL_RESERVE - 30))
        cur, vcur = _lp_polish(cur, dl)
        chain[N] = (cur.copy(), vcur)
        # keep track of hash diversity
        seen.add(_hash_key(cur))

    # ---------- Phase 2: prune each overshoot back to 153 ----------
    for N in sorted(list(chain.keys())):
        if time_left() < (FINAL_RESERVE + 12):
            break
        Y, _ = chain[N]
        Y = Y.copy()
        kdel = N - 153
        # inserted indices are last k overshoot points if chain sequential;
        # for N=154 inserted={153}, for 155 inserted from its parent etc.
        # conservatively treat last kdel as inserted
        inserted = set(range(Y.shape[0] - kdel, Y.shape[0]))
        # greedy delete one by one
        ok = True
        for step in range(kdel):
            if time.time() > HARD_END - FINAL_RESERVE - 5:
                ok = False
                break
            nn = Y.shape[0]
            deg, rowmax, mcur, _ = _degrees(Y, tol=1e-6)
            order = np.argsort(-deg)
            top = set(int(o) for o in order[:12])
            cands = sorted(inserted.union(top))
            cands = [c for c in cands if 0 <= c < nn]
            if not cands:
                cands = list(range(nn))
            # quick screen by deletion max_cos
            scored = []
            for c in cands:
                mask = np.ones(nn, dtype=bool); mask[c] = False
                Yd = Y[mask]
                v = _max_cos(Yd)
                scored.append((v, c))
            scored.sort()
            # take top 3 for short LP scoring
            shortlist = [c for _, c in scored[:3]]
            best_c = shortlist[0]
            best_scored_v = 1e9
            best_Y = None
            for c in shortlist:
                if time.time() > HARD_END - FINAL_RESERVE - 3:
                    break
                mask = np.ones(nn, dtype=bool); mask[c] = False
                Yd = Y[mask]
                # short LP polish ~0.7s
                Yp, vp = _lp_polish(Yd, time.time() + 0.7, cap=2000, r0=8e-4)
                if vp < best_scored_v:
                    best_scored_v = vp
                    best_c = c
                    best_Y = Yp
            # apply deletion
            mask = np.ones(nn, dtype=bool); mask[best_c] = False
            if best_Y is not None and best_Y.shape[0] == nn - 1:
                Y = best_Y
            else:
                Y = Y[mask]
            # update inserted set: remove best_c, shift indices above
            new_ins = set()
            for ii in inserted:
                if ii == best_c:
                    continue
                elif ii > best_c:
                    new_ins.add(ii - 1)
                else:
                    new_ins.add(ii)
            inserted = new_ins
        if not ok:
            continue
        # Y should be 153 now
        if Y.shape[0] != 153:
            continue
        # diversity: if hash repeats record, kick
        hk = _hash_key(Y)
        if hk in seen:
            # coordinated kick to leave basin
            noise = rng.standard_normal(size=Y.shape) * 0.02
            noise -= np.sum(noise * Y, axis=1, keepdims=True) * Y
            Y = _norm(Y + noise)
        # brief joint polish
        if time_left() > FINAL_RESERVE + 4:
            Y = _smooth_opt(Y, steps=300, tau0=0.012, tau1=0.002, lr0=0.08, lr1=0.006)
            Y, vy = _lp_polish(Y, time.time() + 1.5)
        else:
            vy = _max_cos(Y)
        seen.add(_hash_key(Y))
        if vy < global_best_v:
            global_best_v = vy
            global_best_x = Y.copy()

    # ---------- Phase 3: shrink to 142 and regrow ----------
    if time_left() > FINAL_RESERVE + 25:
        try:
            Xb = global_best_x.copy()
            deg, rowmax, mcur, _ = _degrees(Xb, tol=1e-6)
            # rattler = lowest degree, tie by largest slack (smallest rowmax)
            mindeg = deg.min()
            rattlers = np.where(deg == mindeg)[0]
            # pick rattler with smallest rowmax
            r0 = int(rattlers[int(np.argmin(rowmax[rattlers]))])
            rest = [i for i in range(n) if i != r0]
            pick10 = rng.choice(rest, size=10, replace=False)
            dell = set([r0] + [int(p) for p in pick10])
            keep = np.array([i for i in range(n) if i not in dell])
            Y = Xb[keep]  # 142
            # fully re-optimise at 142
            Y = _smooth_opt(Y, steps=900, tau0=0.03, tau1=0.0025, lr0=0.15, lr1=0.008)
            Y, _ = _lp_polish(Y, time.time() + 3.0)
            # greedily regrow to 153
            for gi in range(11):
                if time.time() > HARD_END - FINAL_RESERVE - 6:
                    break
                Y = _hole_insert(Y, rng, n_cand=20000, batch=10000)
                Y = _smooth_opt(Y, steps=220, tau0=0.012, tau1=0.002, lr0=0.08, lr1=0.006)
                Y, _ = _lp_polish(Y, time.time() + 0.5, cap=2000, r0=8e-4)
            if Y.shape[0] == 153:
                hk = _hash_key(Y)
                if hk in seen:
                    noise = rng.standard_normal(size=Y.shape) * 0.015
                    noise -= np.sum(noise * Y, axis=1, keepdims=True) * Y
                    Y = _norm(Y + noise)
                Y = _smooth_opt(Y, steps=300, tau0=0.01, tau1=0.002, lr0=0.07, lr1=0.005)
                Y, vy = _lp_polish(Y, time.time() + 2.0)
                seen.add(_hash_key(Y))
                if vy < global_best_v:
                    global_best_v = vy
                    global_best_x = Y.copy()
        except Exception:
            pass

    # ---------- Final full LP polish (reserve) ----------
    try:
        dl = HARD_END - 1.0
        # ensure at least a bit of time
        if dl > time.time():
            global_best_x, global_best_v = _lp_polish(global_best_x, dl, cap=3000, r0=1e-3)
    except Exception:
        pass
    global_best_x = _norm(global_best_x)
    return global_best_x
