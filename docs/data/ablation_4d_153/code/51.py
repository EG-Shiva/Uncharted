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

def _row_max(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    rm = np.max(G, axis=1)
    return rm, float(np.max(rm))

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
        big = gn > 2.0
        if np.any(big):
            grad = grad * np.where(big, 2.0 / np.maximum(gn, 1e-12), 1.0)
        vel = mom * vel + (1.0 - mom) * grad
        bc = 1.0 - mom ** (t + 1)
        step = vel / bc
        x = x - lr * step
        x = _norm(x)
    return x

def _lp_polish(x, deadline, cap=2500, r0=1e-3, maxit=200):
    from scipy.optimize import linprog
    from scipy.sparse import coo_matrix
    x = _norm(x)
    n, d = x.shape
    best = _max_cos(x)
    radius = r0
    iu0 = np.triu_indices(n, 1)
    it = 0
    while time.time() < deadline and radius > 1e-14 and it < maxit:
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

def _best_hole_insert(Y, rng, n_cand=20000, batch=10000):
    # Y is (m,4) with m=152, return (153,4) with new point at largest hole
    d = Y.shape[1]
    best_c = None
    best_s = np.inf
    nb = max(1, n_cand // batch)
    for _ in range(nb):
        C = rng.standard_normal(size=(batch, d))
        C = _norm(C)
        S = C @ Y.T
        sm = S.max(axis=1)
        i = int(np.argmin(sm))
        if float(sm[i]) < best_s:
            best_s = float(sm[i])
            best_c = C[i].copy()
    return np.vstack([Y, best_c.reshape(1, -1)])

def _tangent_kick(y, idx, sigma, rng):
    z = y.copy()
    k = len(idx)
    noise = rng.standard_normal(size=(k, y.shape[1])) * sigma
    base = z[idx]
    dot = np.sum(noise * base, axis=1, keepdims=True)
    noise = noise - dot * base
    z[idx] = _norm(base + noise)
    return z

def solve():
    t0 = time.time()
    HARD_END = t0 + 173.0
    FINAL_RESERVE = 22.0
    rng = np.random.default_rng(0)
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
    # initial tight polish of baseline
    try:
        best_x, best_v = _lp_polish(best_x, time.time() + 4.0, cap=2500, r0=1e-3)
    except Exception:
        pass
    global_best_x = best_x.copy()
    global_best_v = float(best_v)
    cur_x = global_best_x.copy()
    cur_v = global_best_v

    EXPLORE_END = HARD_END - FINAL_RESERVE
    it = 0
    # precompute sigmas schedule
    while time.time() < EXPLORE_END:
        it += 1
        if time.time() > EXPLORE_END - 1.0:
            break
        mode = it % 6
        base = cur_x if (it % 3 != 0) else global_best_x
        y = base.copy()
        try:
            if mode == 0:
                # global small kick
                sigma = 0.015 + 0.01 * ((it // 6) % 3)
                noise = rng.standard_normal(size=(n, d)) * sigma
                dot = np.sum(noise * y, axis=1, keepdims=True)
                noise = noise - dot * y
                y = _norm(y + noise)
                y = _smooth_opt(y, steps=350, tau0=0.015, tau1=0.0015, lr0=0.10, lr1=0.006)
            elif mode == 1:
                # subset kick biased to high rowmax vertices
                rm, _ = _row_max(y)
                # pick k worst + random
                k = 8 + (it % 7)
                worst = np.argsort(-rm)[:k]
                # replace half by random for diversity
                nr = k // 2
                if nr > 0:
                    cand = rng.choice(n, size=nr, replace=False)
                    worst[:nr] = cand
                sigma = 0.06 + 0.04 * ((it // 6) % 3)
                y = _tangent_kick(y, worst, sigma, rng)
                y = _smooth_opt(y, steps=350, tau0=0.012, tau1=0.0012, lr0=0.09, lr1=0.005)
            elif mode == 2:
                # hole reinsertion: delete worst vertex, reinsert at hole
                rm, _ = _row_max(y)
                rdel = int(np.argmax(rm))
                # sometimes delete random among top-5
                if (it % 2) == 0:
                    top5 = np.argsort(-rm)[:5]
                    rdel = int(top5[rng.integers(0, 5)])
                mask = np.ones(n, dtype=bool); mask[rdel] = False
                Ym = y[mask]
                y = _best_hole_insert(Ym, rng, n_cand=20000, batch=10000)
                y = _smooth_opt(y, steps=400, tau0=0.015, tau1=0.0015, lr0=0.10, lr1=0.006)
            elif mode == 3:
                # contact-guided patch kick: vertices in contacts + second shell
                G = y @ y.T
                np.fill_diagonal(G, -2.0)
                m = float(np.max(G))
                # contacts
                B = G > (m - 5e-4)
                deg = np.sum(B, axis=1)
                # second shell pairs
                iu = np.triu_indices(n, 1)
                c = G[iu]
                sel = np.where((c > m - 0.02) & (c < m - 5e-4))[0]
                verts = set()
                if sel.size > 0:
                    pick = sel[rng.integers(0, sel.size)] if sel.size > 1 else sel[0]
                    a = int(iu[0][pick]); b = int(iu[1][pick])
                    verts.add(a); verts.add(b)
                # add high-degree vertices
                hd = np.argsort(-deg)[:6]
                for v in hd[:3]:
                    verts.add(int(v))
                # add neighbours of second-shell verts
                idx = np.array(sorted(verts), dtype=int)
                if idx.size < 4:
                    extra = rng.choice(n, size=4 - idx.size, replace=False)
                    idx = np.concatenate([idx, extra])
                y = _tangent_kick(y, idx, 0.05, rng)
                y = _smooth_opt(y, steps=350, tau0=0.012, tau1=0.0012, lr0=0.09, lr1=0.005)
            elif mode == 4:
                # medium global kick for basin escape
                sigma = 0.10
                noise = rng.standard_normal(size=(n, d)) * sigma
                dot = np.sum(noise * y, axis=1, keepdims=True)
                noise = noise - dot * y
                y = _norm(y + noise)
                y = _smooth_opt(y, steps=450, tau0=0.025, tau1=0.002, lr0=0.14, lr1=0.007)
            else:
                # single rattler-style large move of one vertex to hole
                rm, _ = _row_max(y)
                rdel = int(np.argmax(rm))
                # move that vertex alone to hole position w.r.t others
                mask = np.ones(n, dtype=bool); mask[rdel] = False
                Ym = y[mask]
                # find hole
                C = _norm(rng.standard_normal(size=(8000, d)))
                S = C @ Ym.T
                sm = S.max(axis=1)
                bi = int(np.argmin(sm))
                y[rdel] = C[bi]
                y = _smooth_opt(y, steps=300, tau0=0.010, tau1=0.001, lr0=0.08, lr1=0.005)
        except Exception:
            continue
        try:
            v_smooth = _max_cos(y)
        except Exception:
            continue
        # screen: only LP-polish if promising
        thresh = min(global_best_v + 0.003, cur_v + 0.002)
        # occasionally polish even if slightly worse to allow wandering
        force = (it % 17 == 0)
        if (v_smooth < thresh) or force:
            if time.time() > EXPLORE_END - 1.0:
                break
            try:
                budget = 0.9 if v_smooth < global_best_v + 0.001 else 0.5
                dl = min(time.time() + budget, EXPLORE_END)
                if dl <= time.time() + 0.05:
                    continue
                yp, vp = _lp_polish(y, dl, cap=1800, r0=7e-4, maxit=120)
            except Exception:
                continue
            if vp < global_best_v - 1e-12:
                global_best_v = float(vp)
                global_best_x = yp.copy()
                cur_x = yp.copy(); cur_v = float(vp)
            elif vp < cur_v:
                cur_x = yp.copy(); cur_v = float(vp)
            elif force and (vp < cur_v + 0.0015) and rng.random() < 0.3:
                cur_x = yp.copy(); cur_v = float(vp)
            # if stuck, restart cur from global best occasionally
            if it % 25 == 0:
                cur_x = global_best_x.copy(); cur_v = global_best_v
        else:
            # accept smooth move as cur if it improves cur slightly (no LP)
            if v_smooth < cur_v:
                cur_x = y.copy(); cur_v = float(v_smooth)
    # final long polish
    try:
        dl = HARD_END - 0.8
        if dl > time.time():
            global_best_x, global_best_v = _lp_polish(global_best_x, dl, cap=3000, r0=1e-3, maxit=300)
    except Exception:
        pass
    global_best_x = _norm(global_best_x)
    return global_best_x
