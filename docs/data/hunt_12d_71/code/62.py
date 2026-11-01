import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _norm_rows(X):
    X = np.asarray(X, dtype=float)
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[~np.isfinite(n)] = 1.0
    n[n == 0] = 1.0
    return X / n

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _topk(X, k=15):
    n = X.shape[0]
    iu = np.triu_indices(n, 1)
    G = X @ X.T
    c = G[iu]
    s = np.sort(c)[::-1]
    return s[:k].copy(), float(s[0])

def _lex_better(a, b, tol=1e-12):
    for x, y in zip(a, b):
        if x < y - tol:
            return True
        if x > y + tol:
            return False
    return False

def _quench(X_init, steps, T0, T1, lr, mu=0.88, deadline=None):
    X = _norm_rows(np.array(X_init, dtype=float, copy=True))
    n, d = X.shape
    V = np.zeros_like(X)
    lt0 = np.log(max(T0, 1e-12)); lt1 = np.log(max(T1, 1e-12))
    G0 = X @ X.T
    np.fill_diagonal(G0, -2.0)
    best_m = float(G0.max())
    best_X = X.copy()
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
        X = _norm_rows(X - lr * V)
        if (s % 150 == 0) or (s == steps - 1):
            Gc = X @ X.T
            np.fill_diagonal(Gc, -2.0)
            vc = float(Gc.max())
            if vc < best_m:
                best_m = vc
                best_X = X.copy()
    return best_X, best_m

def _single_refine(s, R, steps=200, T=0.004, lr=0.12, mu=0.8):
    s = s / (np.linalg.norm(s) + 1e-12)
    v = np.zeros_like(s)
    for _ in range(steps):
        g = R @ s
        m = float(g.max())
        e = np.exp(np.clip((g - m) / T, -50, 0))
        den = e.sum() + 1e-30
        Grad = (R.T @ e) / den
        Gt = Grad - float(Grad @ s) * s
        v = mu * v + Gt
        s = s - lr * v
        nrm = np.linalg.norm(s) + 1e-12
        s = s / nrm
    return s

def _find_hole(R, rng, n_sample=9000, n_cand=3):
    n, d = R.shape
    Z = _norm_rows(rng.standard_normal(size=(n_sample, d)))
    S = Z @ R.T
    scores = S.max(axis=1)
    idx = np.argpartition(scores, n_cand)[:n_cand]
    best_s = None
    best_v = 1e100
    for k in idx:
        s0 = Z[k]
        s1 = _single_refine(s0, R, steps=200, T=0.005, lr=0.12)
        s1 = _single_refine(s1, R, steps=200, T=0.0012, lr=0.07)
        v = float((R @ s1).max())
        if v < best_v:
            best_v = v
            best_s = s1
    return best_s / (np.linalg.norm(best_s) + 1e-12)

def _lp_polish(X0, seconds, deadline=None, r0=1.6e-3, cap=1500):
    X = _norm_rows(np.array(X0, dtype=float, copy=True))
    n, d = X.shape
    try:
        best = _max_cos(X)
    except Exception:
        return X
    bestX = X.copy()
    radius = float(r0)
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds:
        if deadline is not None and time.time() > deadline:
            break
        if radius < 1e-14:
            break
        cmat = (X @ X.T)[iu]
        m = float(cmat.max())
        band = max(25.0 * radius, 8e-5)
        sel = cmat > m - band
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        if p == 0:
            break
        if p > cap:
            order = np.argsort(-cmat[sel])[:cap]
            I, J = I[order], J[order]
            csel = cmat[sel][order]
            p = cap
        else:
            csel = cmat[sel]
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cc = np.concatenate([I[:, None] * d + np.arange(d),
                             J[:, None] * d + np.arange(d),
                             np.full((p, 1), n * d)], 1).ravel()
        vv = np.concatenate([X[J], X[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vv, (rows, cc)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((X.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
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
        y = _norm_rows(X + res.x[:-1].reshape(n, d))
        v = _max_cos(y)
        if v < best - 1e-15:
            X, best = y, v
            bestX = y.copy()
            radius = min(radius * 1.5, 0.02)
        else:
            radius *= 0.5
    return bestX

def _greedy_d12(rng, n=71, d=12, target=2500):
    pool = []
    tries = 0
    while len(pool) < target and tries < target * 80:
        tries += 1
        v = rng.integers(-2, 3, size=d)
        if int(v.sum()) % 2 != 0:
            continue
        n2 = int((v * v).sum())
        if n2 not in (4, 6, 8):
            continue
        pool.append(v.astype(float) / np.sqrt(n2))
    if len(pool) < n:
        return _norm_rows(rng.standard_normal((n, d)))
    P = np.stack(pool, axis=0)
    m = P.shape[0]
    # start from two most distant
    Gp = P @ P.T
    np.fill_diagonal(Gp, 2.0)
    ai, aj = np.unravel_index(np.argmin(Gp), Gp.shape)
    sel_idx = [int(ai), int(aj)]
    sel_mask = np.zeros(m, dtype=bool)
    sel_mask[sel_idx] = True
    # greedy
    cur_max = np.full(m, 1e100)
    for k in sel_idx:
        cur_max = np.minimum(cur_max, (P @ P[k]))
    cur_max[sel_mask] = 1e100
    while len(sel_idx) < n:
        nxt = int(np.argmin(cur_max))
        sel_idx.append(nxt)
        sel_mask[nxt] = True
        cur_max[nxt] = 1e100
        cur_max = np.minimum(cur_max, (P @ P[nxt]))
        cur_max[sel_mask] = 1e100
    X = P[np.array(sel_idx)]
    # small jitter to break exact ties
    X = _norm_rows(X + rng.standard_normal(X.shape) * 1e-3)
    return X

def solve():
    t0 = time.time()
    HARD = t0 + 174.0
    HOP = HARD - 18.0
    rng = np.random.default_rng(0)
    n, d = 71, 12
    iu = np.triu_indices(n, 1)
    try:
        bx = np.load("best.npy").astype(float)
    except Exception:
        bx = None
    if bx is None or bx.shape != (71, 12) or not np.all(np.isfinite(bx)):
        try:
            bx = np.load("record.npy").astype(float)
        except Exception:
            bx = rng.standard_normal((n, d))
    bx = _norm_rows(np.asarray(bx, dtype=float))
    try:
        rc = _norm_rows(np.load("record.npy").astype(float))
        if _max_cos(rc) < _max_cos(bx):
            bx = rc.copy()
    except Exception:
        pass
    best_top, best_score = _topk(bx, 15)
    bestX = bx.copy()
    elites = [(best_score, best_top.copy(), bestX.copy())]

    def consider(X):
        nonlocal best_score, best_top, bestX
        try:
            nt, ns = _topk(X, 15)
        except Exception:
            return False
        # keep elite list size 4
        elites.append((ns, nt.copy(), X.copy()))
        elites.sort(key=lambda e: tuple(e[1].tolist()))
        del elites[4:]
        if _lex_better(nt, best_top) or ns < best_score - 1e-14:
            best_top, best_score, bestX = nt.copy(), ns, X.copy()
            return True
        return False

    # Phase A: diverse seeds
    # 1) D12 greedy seeds
    for rep in range(2):
        if time.time() > HOP - 5:
            break
        r2 = np.random.default_rng(100 + rep)
        X0 = _greedy_d12(r2, n, d, target=2200)
        Xq, _ = _quench(X0, 2200, 0.012, 0.0006, 0.16, 0.88, HOP)
        consider(Xq)
        Xq2, _ = _quench(Xq, 1200, 0.003, 0.00025, 0.10, 0.85, HOP)
        consider(Xq2)
    # 2) pure random seeds
    for rep in range(10):
        if time.time() > HOP - 5:
            break
        r2 = np.random.default_rng(1000 + rep * 17)
        X0 = _norm_rows(r2.standard_normal((n, d)))
        # vary schedule a bit
        T0 = 0.014 if rep % 2 == 0 else 0.008
        Xq, _ = _quench(X0, 2000, T0, 0.0005, 0.16, 0.88, HOP)
        is_new = consider(Xq)
        if is_new and time.time() < HOP - 10:
            Xp = _lp_polish(Xq, 1.2, HOP)
            consider(Xp)
    # 3) perturbed best seeds ( topology change )
    for rep in range(4):
        if time.time() > HOP - 5:
            break
        r2 = np.random.default_rng(2000 + rep)
        sig = [0.08, 0.18, 0.30, 0.12][rep % 4]
        Nt = r2.standard_normal((n, d)) * sig
        Nt -= np.sum(Nt * bestX, axis=1, keepdims=True) * bestX
        X0 = _norm_rows(bestX + Nt)
        Xq, _ = _quench(X0, 1800, 0.008, 0.0004, 0.15, 0.87, HOP)
        consider(Xq)
    # 4) antipodal-structured seed: 35 pairs + pole
    if time.time() < HOP - 5:
        try:
            r2 = np.random.default_rng(777)
            A = _norm_rows(r2.standard_normal((35, d)))
            pole = _norm_rows(r2.standard_normal((1, d)))
            X0 = np.concatenate([A, -A, pole], axis=0)
            X0 = _norm_rows(X0 + r2.standard_normal(X0.shape) * 2e-3)
            Xq, _ = _quench(X0, 2000, 0.010, 0.0004, 0.15, 0.88, HOP)
            consider(Xq)
        except Exception:
            pass

    # pick current from best
    curX = bestX.copy()
    cur_top, cur_score = best_top.copy(), best_score

    it = 0
    while time.time() < HOP:
        it += 1
        if HOP - time.time() < 2.5:
            break
        mode = it % 5
        Y = None
        try:
            G = curX @ curX.T
            cuv = G[iu]
            m = float(cuv.max())
        except Exception:
            curX = bestX.copy(); cur_top, cur_score = best_top.copy(), best_score
            continue
        r2 = np.random.default_rng(it * 7919 + 5)
        if mode == 0 or mode == 2:
            # hole reinsertion of worst point
            Gn = curX @ curX.T
            np.fill_diagonal(Gn, -2.0)
            rowmax = Gn.max(axis=1)
            w = int(np.argmax(rowmax))
            # sometimes second worst
            if mode == 2 and n > 1:
                order = np.argsort(-rowmax)
                w = int(order[0] if r2.random() < 0.7 else order[1])
            R = np.delete(curX, w, axis=0)
            try:
                hole = _find_hole(R, r2, n_sample=7000, n_cand=3)
            except Exception:
                hole = _norm_rows(r2.standard_normal((1, d)))[0]
            Y = np.concatenate([R, hole[None, :]], axis=0)
            # tiny jitter rest
            jit = r2.standard_normal(Y.shape) * 1.5e-3
            jit -= np.sum(jit * Y, axis=1, keepdims=True) * Y
            Y = _norm_rows(Y + jit)
        elif mode == 1:
            # hub shake: points with many near-contacts
            thr = m - 3e-4
            sel = cuv > thr
            cntp = np.zeros(n, dtype=int)
            np.add.at(cntp, iu[0][sel], 1)
            np.add.at(cntp, iu[1][sel], 1)
            hubs = np.where(cntp >= max(2, int(cntp.max())))[0]
            if len(hubs) == 0:
                hubs = np.array([int(iu[0][int(np.argmax(cuv))]), int(iu[1][int(np.argmax(cuv))])])
            Y = curX.copy()
            nh = min(len(hubs), 8)
            pick = hubs[r2.choice(len(hubs), size=nh, replace=False)] if len(hubs) >= nh else hubs
            jit = r2.standard_normal((len(pick), d)) * 0.03
            jit -= np.sum(jit * Y[pick], axis=1, keepdims=True) * Y[pick]
            Y[pick] = _norm_rows(Y[pick] + jit)
            # small global jitter
            j2 = r2.standard_normal(Y.shape) * 1e-3
            j2 -= np.sum(j2 * Y, axis=1, keepdims=True) * Y
            Y = _norm_rows(Y + j2)
        elif mode == 3:
            # max-pair split + rattler jitter
            flat = int(np.argmax(cuv))
            a = int(iu[0][flat]); b = int(iu[1][flat])
            xa = curX[a]; xb = curX[b]
            mid = xa + xb
            nm = float(np.linalg.norm(mid))
            if nm < 1e-12:
                mid = r2.standard_normal(d); nm = float(np.linalg.norm(mid))
            mid = mid / nm
            da = xa - float(xa @ mid) * mid
            nd = float(np.linalg.norm(da))
            Y = curX.copy()
            if nd > 1e-12:
                e = da / nd
                step = float(r2.uniform(3e-3, 1.2e-2))
                Y[a] = _norm_rows((xa + step * e)[None, :])[0]
                Y[b] = _norm_rows((xb - step * e)[None, :])[0]
            sel5 = cuv > (m - 5e-8)
            cntp = np.zeros(n, dtype=int)
            np.add.at(cntp, iu[0][sel5], 1)
            np.add.at(cntp, iu[1][sel5], 1)
            rats = np.where(cntp == 0)[0]
            if len(rats) > 0:
                jr = rats[r2.integers(0, len(rats), size=min(8, len(rats)))]
                jit = r2.standard_normal((len(jr), d)) * 6e-3
                jit -= np.sum(jit * Y[jr], axis=1, keepdims=True) * Y[jr]
                Y[jr] = _norm_rows(Y[jr] + jit)
        else:
            # global restart jitter from elite
            src = elites[int(r2.integers(0, len(elites)))][2]
            sig = float(r2.uniform(0.02, 0.09))
            Nt = r2.standard_normal(src.shape) * sig
            Nt -= np.sum(Nt * src, axis=1, keepdims=True) * src
            Y = _norm_rows(src + Nt)
        # quench short with low-T finish to change topology
        try:
            T0s = 0.006 if mode != 1 else 0.004
            Yq, _ = _quench(Y, 1100, T0s, 0.00035, 0.13, 0.86, HOP)
        except Exception:
            continue
        try:
            nt, ns = _topk(Yq, 15)
        except Exception:
            continue
        if _lex_better(nt, cur_top) or ns < cur_score - 1e-14:
            curX, cur_top, cur_score = Yq, nt, ns
            was_global = consider(Yq)
            if was_global and (HARD - time.time()) > 12:
                # squeeze new global best a bit
                try:
                    Yp = _lp_polish(Yq, 1.0, HOP)
                    ntp, nsp = _topk(Yp, 15)
                    if _lex_better(ntp, cur_top) or nsp < cur_score - 1e-14:
                        curX, cur_top, cur_score = Yp, ntp, nsp
                    consider(Yp)
                except Exception:
                    pass
        else:
            # sideways random restart if stuck long
            if it % 25 == 0:
                # jump to best with medium kick
                sig = 0.05
                Nt = rng.standard_normal(bestX.shape) * sig
                Nt -= np.sum(Nt * bestX, axis=1, keepdims=True) * bestX
                curX = _norm_rows(bestX + Nt)
                try:
                    cur_top, cur_score = _topk(curX, 15)
                except Exception:
                    curX = bestX.copy(); cur_top, cur_score = best_top.copy(), best_score

    # Final polish of top elites with remaining time
    elites_sorted = sorted(elites, key=lambda e: float(e[0]))[:3]
    # also include curX
    cands = [e[2] for e in elites_sorted]
    # ensure bestX first
    cands = [bestX] + [c for c in cands if _max_cos(c) > best_score + 1e-15][:2]
    remain = HARD - time.time() - 1.5
    if remain > 3:
        per = remain / max(1, len(cands))
        for c in cands:
            if time.time() > HARD - 2.0:
                break
            try:
                cq, _ = _quench(c, 1500, 0.0025, 0.0002, 0.09, 0.86, HARD - 2.0)
                consider(cq)
                c2 = _lp_polish(cq if _max_cos(cq) < _max_cos(c) else c, min(per * 0.6, 6.0), HARD - 1.0)
                consider(c2)
            except Exception:
                continue
    return _norm_rows(bestX.copy())
