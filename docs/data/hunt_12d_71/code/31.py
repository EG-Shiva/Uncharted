import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _norm_rows(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[~np.isfinite(n)] = 1.0
    n[n == 0] = 1.0
    return x / n

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _quench_adam(X, steps, T0, T1, lr, b1=0.9, b2=0.999, eps=1e-8, deadline=None, track=True):
    n, d = X.shape
    m1 = np.zeros_like(X); v1 = np.zeros_like(X)
    lt0 = np.log(T0); lt1 = np.log(T1)
    if track:
        G0 = X @ X.T; np.fill_diagonal(G0, -2.0)
        bv = float(G0.max()); bX = X.copy()
    else:
        bv = 1e100; bX = None
    t = 0
    for s in range(steps):
        if deadline is not None and (s % 250 == 0) and time.time() > deadline:
            break
        frac = s / max(1, steps - 1)
        T = float(np.exp(lt0 + (lt1 - lt0) * frac))
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        Q = (G - m) / T
        np.clip(Q, -50, 0, out=Q)
        E = np.exp(Q)
        np.fill_diagonal(E, 0.0)
        S = float(E.sum() * 0.5)
        if not np.isfinite(S) or S <= 0:
            S = 1.0
        Grad = (E @ X) / S
        dot = np.sum(Grad * X, axis=1, keepdims=True)
        Gt = Grad - dot * X
        if not np.all(np.isfinite(Gt)):
            break
        t += 1
        m1 = b1 * m1 + (1 - b1) * Gt
        v1 = b2 * v1 + (1 - b2) * (Gt * Gt)
        c1 = 1.0 - b1 ** t; c2 = 1.0 - b2 ** t
        step = (m1 / c1) / (np.sqrt(v1 / c2) + eps)
        # limit huge steps
        sn = float(np.linalg.norm(step))
        cap = 8.0 * np.sqrt(n * d)
        if sn > cap:
            step *= (cap / sn)
        X = X - lr * step
        X = _norm_rows(X)
        if track and ((s % 120 == 0) or s == steps - 1):
            Gc = X @ X.T; np.fill_diagonal(Gc, -2.0)
            vc = float(Gc.max())
            if vc < bv:
                bv = vc; bX = X.copy()
    if track:
        return bX, bv
    return X, _max_cos(X)

def _quench_mom(X, steps, T0, T1, lr, mu=0.9, deadline=None, track=True):
    n, d = X.shape
    V = np.zeros_like(X)
    lt0 = np.log(T0); lt1 = np.log(T1)
    if track:
        G0 = X @ X.T; np.fill_diagonal(G0, -2.0)
        bv = float(G0.max()); bX = X.copy()
    else:
        bv = 1e100; bX = None
    for s in range(steps):
        if deadline is not None and (s % 250 == 0) and time.time() > deadline:
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
        mk = gn[:, 0] > 2.0
        if np.any(mk):
            Gt[mk] *= (2.0 / gn[mk])
        V = mu * V + Gt
        X = X - lr * V
        X = _norm_rows(X)
        if track and ((s % 150 == 0) or s == steps - 1):
            Gc = X @ X.T; np.fill_diagonal(Gc, -2.0)
            vc = float(Gc.max())
            if vc < bv:
                bv = vc; bX = X.copy()
    if track:
        return bX, bv
    return X, _max_cos(X)

def _quench_riesz(X, steps, s_pow, lr, mu=0.9, deadline=None):
    n, d = X.shape
    V = np.zeros_like(X)
    for s in range(steps):
        if deadline is not None and (s % 250 == 0) and time.time() > deadline:
            break
        G = X @ X.T
        np.fill_diagonal(G, -1.0)
        D2 = np.clip(2.0 - 2.0 * G, 1e-6, 4.0)
        W = np.power(D2, -s_pow * 0.5 - 1.0) * s_pow
        np.fill_diagonal(W, 0.0)
        Grad = W @ X
        dot = np.sum(Grad * X, axis=1, keepdims=True)
        Gt = Grad - dot * X
        gn = float(np.median(np.linalg.norm(Gt, axis=1)) + 1e-12)
        Gt = Gt / gn
        V = mu * V + Gt
        X = X - lr * V
        X = _norm_rows(X)
    return X

def _refine_single(s, R, steps=180, T=0.004, lr=0.10, mu=0.8):
    s = s / (np.linalg.norm(s) + 1e-12)
    v = np.zeros_like(s)
    for _ in range(steps):
        g = R @ s
        m = float(g.max())
        e = np.exp(np.clip((g - m) / T, -50, 0))
        den = float(e.sum()) + 1e-30
        Grad = (R.T @ e) / den
        Gt = Grad - float(Grad @ s) * s
        v = mu * v + Gt
        s = s - lr * v
        nn = float(np.linalg.norm(s))
        if not np.isfinite(nn) or nn == 0:
            break
        s = s / nn
    return s

def _find_hole(R, rng, n_sample=6000, n_cand=3):
    d = R.shape[1]
    Z = rng.standard_normal(size=(n_sample, d))
    Z = _norm_rows(Z)
    S = Z @ R.T
    scores = S.max(axis=1)
    idx = np.argpartition(scores, n_cand)[:n_cand]
    best_s = None; best_v = 1e100
    for k in idx:
        s0 = Z[k]
        s1 = _refine_single(s0, R, steps=160, T=0.005, lr=0.10)
        s1 = _refine_single(s1, R, steps=160, T=0.0015, lr=0.06)
        v = float((R @ s1).max())
        if v < best_v:
            best_v = v; best_s = s1
    return best_s / (np.linalg.norm(best_s) + 1e-12)

def _row_scores(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return G.max(axis=1), G

def _lp_polish(x, seconds, deadline=None):
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
        G = x @ x.T
        cv = G[iu]
        m = float(cv.max())
        sel = cv > m - max(20 * radius, 1e-9)
        if np.count_nonzero(sel) == 0:
            break
        I = iu[0][sel]; J = iu[1][sel]
        cs = cv[sel]
        p = len(I)
        if p > 1600:
            od = np.argsort(-cs)[:1600]
            I, J, cs = I[od], J[od], cs[od]
            p = 1600
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        except Exception:
            radius *= 0.5
            continue
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-cs, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y = _norm_rows(y)
        if not np.all(np.isfinite(y)):
            radius *= 0.5
            continue
        v = _max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def _align(B, A):
    # orthogonal Q minimizing ||A - B Q||
    M = B.T @ A
    try:
        U, _, Vt = np.linalg.svd(M, full_matrices=False)
        Q = U @ Vt
    except Exception:
        Q = np.eye(A.shape[1])
    return B @ Q

def solve():
    t0 = time.time()
    HARD_END = t0 + 174.0
    HOP_END = HARD_END - 30.0
    rng = np.random.default_rng(7)
    n, d = 71, 12
    cands = []
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.asarray(np.load(fn), dtype=np.float64)
            if a.shape == (71, 12) and np.all(np.isfinite(a)):
                cands.append(_norm_rows(a))
        except Exception:
            pass
    if not cands:
        cands.append(_norm_rows(rng.standard_normal((n, d))))
    # score cands
    scored = [( _max_cos(a), a) for a in cands]
    scored.sort(key=lambda t: t[0])
    best_score, best_X = scored[0][0], scored[0][1].copy()
    elites = [a.copy() for _, a in scored[:2]]
    escores = [s for s, _ in scored[:2]]
    # quick fresh blood to leave basin
    for f in range(5):
        if time.time() > HOP_END - 70:
            break
        try:
            X0 = _norm_rows(rng.standard_normal((n, d)))
            T0 = float(10 ** rng.uniform(-2.4, -1.4))
            T1 = float(10 ** rng.uniform(-4.0, -3.0))
            lr = float(10 ** rng.uniform(-2.8, -2.0))
            Xq, sq = _quench_adam(X0, steps=800, T0=T0, T1=T1, lr=lr, deadline=HOP_END)
            elites.append(_norm_rows(Xq)); escores.append(float(sq))
            if float(sq) < best_score:
                best_score = float(sq); best_X = _norm_rows(Xq).copy()
        except Exception:
            continue
    od = np.argsort(np.array(escores))
    elites = [_norm_rows(elites[i]) for i in od[:5]]
    escores = [float(escores[i]) for i in od[:5]]
    if best_score <= min(escores):
        pass
    else:
        best_X = elites[0].copy(); best_score = escores[0]
    it = 0
    while time.time() < HOP_END:
        it += 1
        tleft = HOP_END - time.time()
        if tleft < 1.2:
            break
        # pick base
        r = rng.random()
        if r < 0.5 or len(elites) == 1:
            base = best_X
        elif r < 0.8:
            base = elites[int(rng.integers(0, len(elites)))]
        else:
            # tournament of 2
            a = int(rng.integers(0, len(elites))); b = int(rng.integers(0, len(elites)))
            base = elites[a] if escores[a] < escores[b] else elites[b]
        u = rng.random()
        X0 = base.copy()
        try:
            if u < 0.22:
                sigma = float(10 ** rng.uniform(-2.6, -0.4))
                X0 = base + rng.standard_normal((n, d)) * sigma
                X0 = _norm_rows(X0)
            elif u < 0.48:
                # cap kill large
                k = int(rng.integers(4, 13))
                direction = rng.standard_normal(d)
                direction /= (np.linalg.norm(direction) + 1e-12)
                if rng.random() < 0.5:
                    direction = base[int(rng.integers(0, n))].copy() + rng.standard_normal(d) * 0.3
                    direction /= (np.linalg.norm(direction) + 1e-12)
                proj = base @ direction
                idx = np.argpartition(-proj, k)[:k]
                R = np.delete(base, idx, axis=0)
                for j in idx:
                    if rng.random() < 0.35:
                        h = _norm_rows(rng.standard_normal((1, d)))[0]
                    else:
                        h = _find_hole(R, rng, n_sample=4500, n_cand=3)
                    X0[j] = h
                    R = np.concatenate([R, h[None, :]], axis=0)
                keep = np.ones(n, dtype=bool); keep[idx] = False
                sigma = float(10 ** rng.uniform(-2.8, -1.1))
                X0[keep] += rng.standard_normal((np.count_nonzero(keep), d)) * sigma
                X0 = _norm_rows(X0)
            elif u < 0.66:
                # worst-k kill
                k = 2 if rng.random() < 0.4 else (3 if rng.random() < 0.6 else (4 if rng.random() < 0.7 else 6))
                k = min(k, 8)
                rm, _ = _row_scores(base)
                order = np.argsort(-rm + rng.random(n) * 0.002)
                idx = order[:k]
                keep = np.ones(n, dtype=bool); keep[idx] = False
                R = base[keep]
                for j in idx:
                    h = _find_hole(R, rng, n_sample=6000, n_cand=3)
                    X0[j] = h
                    R = np.concatenate([R, h[None, :]], axis=0)
                sigma = float(10 ** rng.uniform(-2.8, -1.0))
                X0[keep] += rng.standard_normal((n - k, d)) * sigma
                X0 = _norm_rows(X0)
            elif u < 0.80:
                # crossover between two elites
                if len(elites) >= 2:
                    ia = int(rng.integers(0, len(elites))); ib = int(rng.integers(0, len(elites)))
                    while ib == ia and len(elites) > 1:
                        ib = int(rng.integers(0, len(elites)))
                    A = elites[ia]; B = elites[ib]
                    Ba = _align(B, A)
                    mask = rng.random(n) < 0.5
                    X0[mask] = A[mask]
                    X0[~mask] = Ba[~mask]
                    sigma = float(10 ** rng.uniform(-2.5, -1.2))
                    X0 += rng.standard_normal((n, d)) * sigma
                    X0 = _norm_rows(X0)
                else:
                    sigma = float(10 ** rng.uniform(-2.0, -0.7))
                    X0 = base + rng.standard_normal((n, d)) * sigma
                    X0 = _norm_rows(X0)
            elif u < 0.90:
                # subset rotation
                k = int(rng.integers(8, 22))
                idx = rng.choice(n, size=k, replace=False)
                Qm, _ = np.linalg.qr(rng.standard_normal((d, d)))
                if float(np.linalg.det(Qm)) < 0:
                    Qm[:, 0] = -Qm[:, 0]
                alpha = float(rng.uniform(0.4, 1.0))
                rot = X0[idx] @ Qm.T
                X0[idx] = _norm_rows((1 - alpha) * X0[idx] + alpha * rot + rng.standard_normal((k, d)) * 0.02)
                sigma = float(10 ** rng.uniform(-2.8, -1.4))
                X0 += rng.standard_normal((n, d)) * sigma * 0.3
                X0 = _norm_rows(X0)
            else:
                # fresh random
                X0 = _norm_rows(rng.standard_normal((n, d)))
        except Exception:
            continue
        # choose quench
        if tleft < 6:
            steps = 900
        elif tleft < 25:
            steps = 1500
        else:
            steps = 1800
            if rng.random() < 0.18:
                steps = 2800
        try:
            rr = rng.random()
            if rr < 0.45:
                T0 = float(10 ** rng.uniform(-2.4, -1.5))
                T1 = float(10 ** rng.uniform(-4.3, -3.1))
                lr = float(10 ** rng.uniform(-2.9, -2.0))
                X1, s1 = _quench_adam(X0, steps=steps, T0=T0, T1=T1, lr=lr, deadline=HOP_END)
            elif rr < 0.75:
                T0 = float(10 ** rng.uniform(-2.5, -1.6))
                T1 = float(10 ** rng.uniform(-4.2, -3.1))
                lr = float(10 ** rng.uniform(-1.4, -0.7))
                mu = float(rng.uniform(0.82, 0.94))
                # sometimes Riesz pre-condition
                if rng.random() < 0.3:
                    sp = float(rng.uniform(2.0, 7.0))
                    lrr = float(10 ** rng.uniform(-1.6, -0.9))
                    Xr = _quench_riesz(X0, steps=max(250, steps // 3), s_pow=sp, lr=lrr, deadline=HOP_END)
                    X0 = Xr
                X1, s1 = _quench_mom(X0, steps=steps, T0=T0, T1=T1, lr=lr, mu=mu, deadline=HOP_END)
            else:
                # Riesz + Adam finish
                sp = float(rng.uniform(2.0, 8.0))
                lrr = float(10 ** rng.uniform(-1.6, -0.9))
                Xr = _quench_riesz(X0, steps=max(300, steps // 2), s_pow=sp, lr=lrr, deadline=HOP_END)
                T0 = float(10 ** rng.uniform(-2.6, -1.9))
                T1 = float(10 ** rng.uniform(-4.2, -3.2))
                lr = float(10 ** rng.uniform(-2.9, -2.1))
                X1, s1 = _quench_adam(Xr, steps=steps, T0=T0, T1=T1, lr=lr, deadline=HOP_END)
        except Exception:
            continue
        if not np.isfinite(s1):
            continue
        if s1 < 0.265:
            # insert into elites (keep 6, diverse enough)
            try:
                dup = False
                for e in elites:
                    # cheap distance: mean abs corr
                    dd = float(np.abs((X1 * e).sum(axis=1).mean()))
                    if dd > 0.9995 and abs(s1 - _max_cos(e)) < 5e-7:
                        dup = True
                        break
                if not dup:
                    elites.append(_norm_rows(X1)); escores.append(float(s1))
                    od2 = np.argsort(np.array(escores))
                    elites = [elites[i] for i in od2[:6]]
                    escores = [escores[i] for i in od2[:6]]
                elif float(s1) < min(escores):
                    bi = int(np.argmin(np.array(escores)))
                    elites[bi] = _norm_rows(X1); escores[bi] = float(s1)
            except Exception:
                pass
        if s1 < best_score:
            best_score = float(s1); best_X = _norm_rows(X1).copy()
    # final jam: tighten top elites with low-T + LP
    try:
        od = np.argsort(np.array(escores))
        elites = [elites[i] for i in od]
        escores = [escores[i] for i in od]
    except Exception:
        elites = [best_X]; escores = [best_score]
    cand_list = []
    for e, s in zip(elites[:3], escores[:3]):
        cand_list.append((s, e.copy()))
    cand_list.append((best_score, best_X.copy()))
    # low-T Adam tightening
    tightened = []
    for s, Xc in cand_list:
        if time.time() > HARD_END - 14:
            break
        try:
            Xt, st = _quench_adam(Xc.copy(), steps=1600, T0=8e-4, T1=1.5e-4, lr=0.0025, deadline=HARD_END - 12)
            tightened.append((st, Xt))
        except Exception:
            tightened.append((s, Xc))
    # LP polish best two
    tightened.sort(key=lambda t: t[0])
    final_best = tightened[0][1] if tightened else best_X
    final_score = tightened[0][0] if tightened else best_score
    for k in range(min(2, len(tightened))):
        if time.time() > HARD_END - 4:
            break
        try:
            s0, X0c = tightened[k]
            Xp = _lp_polish(X0c, seconds=4.0, deadline=HARD_END - 1.0)
            sp = _max_cos(Xp)
            if sp < final_score:
                final_score = sp; final_best = Xp
        except Exception:
            continue
    # also LP on global best if different
    if time.time() < HARD_END - 3:
        try:
            Xp = _lp_polish(best_X, seconds=3.0, deadline=HARD_END - 0.5)
            sp = _max_cos(Xp)
            if sp < final_score:
                final_score = sp; final_best = Xp
        except Exception:
            pass
    final_best = _norm_rows(final_best)
    # safety: return the better of final and best_X
    try:
        sb = _max_cos(best_X)
        if sb < final_score:
            final_best = best_X
    except Exception:
        pass
    return final_best
