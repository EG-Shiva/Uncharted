import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _norm_rows(X):
    X = np.asarray(X, dtype=np.float64)
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[~np.isfinite(n)] = 1.0
    n[n == 0] = 1.0
    return X / n

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _topk_vec(X, k=60):
    n = X.shape[0]
    iu = np.triu_indices(n, 1)
    G = X @ X.T
    c = G[iu]
    s = np.sort(c)[::-1]
    return s[:k].copy(), float(s[0])

def _lex_better(new_top, old_top, tol=2e-12):
    for a, b in zip(new_top, old_top):
        if a < b - tol:
            return True
        if a > b + tol:
            return False
    return False

def _quench_lse(X, steps, T0, T1, lr, mu, rng=None, noise0=0.0, deadline=None, track=True):
    n, d = X.shape
    V = np.zeros_like(X)
    lt0 = np.log(T0); lt1 = np.log(T1)
    if track:
        bv = _max_cos(X); bX = X.copy()
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
        if noise0 > 0 and rng is not None:
            amp = noise0 * (1.0 - frac)
            if amp > 0:
                Z = rng.standard_normal((n, d)) * amp
                Z -= np.sum(Z * X, axis=1, keepdims=True) * X
                Gt = Gt + Z
        V = mu * V + Gt
        X = X - lr * V
        X = _norm_rows(X)
        if track and ((s % 200 == 0) or s == steps - 1):
            vc = _max_cos(X)
            if vc < bv:
                bv = vc; bX = X.copy()
    if track:
        return bX, bv
    return X, _max_cos(X)

def _quench_power(X, steps, thr_off, pw, lr, mu, deadline=None):
    n, d = X.shape
    V = np.zeros_like(X)
    bv = _max_cos(X); bX = X.copy()
    for s in range(steps):
        if deadline is not None and (s % 250 == 0) and time.time() > deadline:
            break
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        thr = m - thr_off
        M = np.clip((G - thr) / max(1e-12, (1.0 - thr)), 0, 1)
        W = np.power(M, pw - 1.0) * pw
        np.fill_diagonal(W, 0.0)
        Grad = W @ X
        dot = np.sum(Grad * X, axis=1, keepdims=True)
        Gt = Grad - dot * X
        med = float(np.median(np.linalg.norm(Gt, axis=1)) + 1e-12)
        Gt = Gt / med
        V = mu * V + Gt
        X = X - lr * V
        X = _norm_rows(X)
        if (s % 200 == 0) or s == steps - 1:
            vc = _max_cos(X)
            if vc < bv:
                bv = vc; bX = X.copy()
    return bX, bv

def _quench_worst(X, steps, lr, mu, topB=8, rng=None, deadline=None):
    n, d = X.shape
    V = np.zeros_like(X)
    bv = _max_cos(X); bX = X.copy()
    iu0 = np.triu_indices(n, 1)
    for s in range(steps):
        if deadline is not None and (s % 250 == 0) and time.time() > deadline:
            break
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        # pick topB pairs
        cuv = G[iu0]
        if topB < len(cuv):
            idx = np.argpartition(-cuv, topB)[:topB]
            I = iu0[0][idx]; J = iu0[1][idx]
            cs = cuv[idx]
        else:
            I = iu0[0]; J = iu0[1]; cs = cuv
        # weights focused on max
        T = 0.0006
        w = np.exp(np.clip((cs - m) / T, -50, 0))
        Grad = np.zeros_like(X)
        for k in range(len(I)):
            i = I[k]; j = J[k]; wk = w[k]
            Grad[i] += wk * X[j]
            Grad[j] += wk * X[i]
        dot = np.sum(Grad * X, axis=1, keepdims=True)
        Gt = Grad - dot * X
        gn = np.linalg.norm(Gt, axis=1, keepdims=True) + 1e-12
        mk = gn[:, 0] > 1.0
        if np.any(mk):
            Gt[mk] *= (1.0 / gn[mk])
        V = mu * V + Gt
        X = X - lr * V
        X = _norm_rows(X)
        if (s % 150 == 0) or s == steps - 1:
            vc = _max_cos(X)
            if vc < bv:
                bv = vc; bX = X.copy()
    return bX, bv

def _quench_riesz(X, steps, s_pow, lr, mu, deadline=None):
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

def _refine_single(s, R, steps=180, T=0.004, lr=0.12, mu=0.8):
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
    Z = _norm_rows(rng.standard_normal(size=(n_sample, d)))
    S = Z @ R.T
    scores = S.max(axis=1)
    idx = np.argpartition(scores, n_cand)[:n_cand]
    best_s = None; best_v = 1e100
    for k in idx:
        s0 = Z[k]
        s1 = _refine_single(s0, R, steps=160, T=0.005, lr=0.12)
        s1 = _refine_single(s1, R, steps=160, T=0.0015, lr=0.07)
        v = float((R @ s1).max())
        if v < best_v:
            best_v = v; best_s = s1
    best_s = best_s / (np.linalg.norm(best_s) + 1e-12)
    return best_s

def _lp_polish(x, seconds, deadline=None):
    x = _norm_rows(x.copy())
    n, d = x.shape
    try:
        best = _max_cos(x)
    except Exception:
        return x
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    nv = n * d + 1
    while time.time() - start < seconds:
        if deadline is not None and time.time() > deadline:
            break
        if radius < 1e-14:
            break
        G = x @ x.T
        cuv = G[iu]
        m = float(cuv.max())
        sel = cuv > m - max(20 * radius, 1e-9)
        if np.count_nonzero(sel) == 0:
            break
        I = iu[0][sel]; J = iu[1][sel]; cs = cuv[sel]
        p = len(I)
        if p > 1300:
            kk = np.argpartition(-cs, 1300)[:1300]
            I = I[kk]; J = J[kk]; cs = cs[kk]; p = 1300
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

def _greedy_squeeze(X, rng, iters, deadline, step0=0.005):
    X = X.copy()
    cur, cur_top = _topk_vec(X, 60)
    cur_max = float(cur_top[0])
    n, d = X.shape
    step = step0
    for it in range(iters):
        if time.time() > deadline:
            break
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        rm = G.max(axis=1)
        # pick one of 6 worst points
        order = np.argsort(-rm)
        cand = order[:6]
        i = int(cand[rng.integers(0, len(cand))])
        xi = X[i].copy()
        # random tangent step
        z = rng.standard_normal(d)
        z -= float(z @ xi) * xi
        zn = float(np.linalg.norm(z))
        if zn == 0 or not np.isfinite(zn):
            continue
        z /= zn
        amp = step * (0.3 + rng.random())
        trial = xi * np.cos(amp) + z * np.sin(amp)
        Xt = X.copy(); Xt[i] = trial / (np.linalg.norm(trial) + 1e-12)
        new_top, new_max = _topk_vec(Xt, 60)
        new_max = float(new_max)
        if (new_max < cur_max - 1e-14) or (abs(new_max - cur_max) <= 1e-12 and _lex_better(new_top, cur_top)):
            X = Xt; cur_top = new_top; cur_max = new_max
            step = min(step * 1.05, 0.01)
        else:
            if rng.random() < 0.05:
                step *= 0.9
            if step < 1e-6:
                step = step0 * 0.5
    return X, cur_max

def solve():
    t0 = time.time()
    HARD_END = t0 + 174.0
    HOP_END = HARD_END - 26.0
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
    # seed elites with quick quenches from random + loaded
    elites = []
    elite_scores = []
    elite_tops = []
    for a in cands:
        try:
            s = _max_cos(a)
            tp, _ = _topk_vec(a, 60)
            elites.append(a.copy()); elite_scores.append(s); elite_tops.append(tp)
        except Exception:
            pass
    # few fresh starts
    for f in range(4):
        if time.time() > HOP_END - 50:
            break
        X0 = _norm_rows(rng.standard_normal((n, d)))
        try:
            Xq, sq = _quench_lse(X0, steps=700, T0=float(10**rng.uniform(-2.2, -1.6)),
                                 T1=float(10**rng.uniform(-3.8, -3.1)),
                                 lr=float(10**rng.uniform(-1.3, -0.8)),
                                 mu=float(rng.uniform(0.82, 0.92)), rng=rng, noise0=0.0, deadline=HOP_END, track=True)
            tp, _ = _topk_vec(Xq, 60)
            elites.append(_norm_rows(Xq)); elite_scores.append(float(sq)); elite_tops.append(tp)
        except Exception:
            continue
    od = np.argsort(np.array(elite_scores))
    keep = min(5, len(od))
    elites = [elites[i] for i in od[:keep]]
    elite_scores = [elite_scores[i] for i in od[:keep]]
    elite_tops = [elite_tops[i] for i in od[:keep]]
    best_X = elites[0].copy(); best_score = float(elite_scores[0]); best_top = elite_tops[0].copy()
    it = 0
    while time.time() < HOP_END:
        it += 1
        tleft = HOP_END - time.time()
        if tleft < 1.0:
            break
        if rng.random() < 0.55 or len(elites) == 1:
            base = best_X
        else:
            base = elites[int(rng.integers(0, len(elites)))]
        X0 = base.copy()
        u = rng.random()
        if u < 0.26:
            # Levy heavy-tail global shake
            sigma = float(10 ** rng.uniform(-2.6, -0.7))
            Z = rng.standard_cauchy((n, d)) * sigma * 0.3 + rng.standard_normal((n, d)) * sigma
            # clip extremes
            Z = np.clip(Z, -0.8, 0.8)
            X0 = _norm_rows(base + Z)
        elif u < 0.58:
            # cap kill
            k = int(rng.integers(4, 12))
            direction = rng.standard_normal(d)
            direction /= (np.linalg.norm(direction) + 1e-12)
            if rng.random() < 0.5:
                direction = base[int(rng.integers(0, n))].copy() + rng.standard_normal(d) * 0.4
                direction /= (np.linalg.norm(direction) + 1e-12)
            proj = base @ direction
            idx = np.argpartition(-proj, k)[:k]
            R = np.delete(base, idx, axis=0)
            for j in idx:
                if rng.random() < 0.35:
                    h = _norm_rows(rng.standard_normal((1, d)))[0]
                else:
                    try:
                        h = _find_hole(R, rng, n_sample=4500, n_cand=3)
                    except Exception:
                        h = _norm_rows(rng.standard_normal((1, d)))[0]
                X0[j] = h
                R = np.concatenate([R, h[None, :]], axis=0)
            keepm = np.ones(n, dtype=bool); keepm[idx] = False
            sigma = float(10 ** rng.uniform(-2.8, -1.1))
            X0[keepm] += rng.standard_normal((np.count_nonzero(keepm), d)) * sigma
            X0 = _norm_rows(X0)
        elif u < 0.80:
            # hub kill worst rows
            k = 1 if rng.random() < 0.4 else (2 if rng.random() < 0.7 else (3 if rng.random() < 0.8 else 5))
            try:
                G = base @ base.T
                np.fill_diagonal(G, -2.0)
                rm = G.max(axis=1)
                order = np.argsort(-rm + rng.random(n) * 1e-4)
                idx = order[:k]
            except Exception:
                idx = rng.choice(n, size=k, replace=False)
            keepm = np.ones(n, dtype=bool); keepm[idx] = False
            R = base[keepm]
            for j in idx:
                try:
                    h = _find_hole(R, rng, n_sample=6000, n_cand=3)
                except Exception:
                    h = _norm_rows(rng.standard_normal((1, d)))[0]
                X0[j] = h
                R = np.concatenate([R, h[None, :]], axis=0)
            sigma = float(10 ** rng.uniform(-2.8, -1.0))
            X0[keepm] += rng.standard_normal((n - k, d)) * sigma
            X0 = _norm_rows(X0)
        else:
            # contact-split: force top pairs apart
            try:
                G = base @ base.T
                np.fill_diagonal(G, -2.0)
                iu = np.triu_indices(n, 1)
                cuv = G[iu]
                tb = min(6, len(cuv))
                idx = np.argpartition(-cuv, tb)[:tb]
                Is = iu[0][idx]; Js = iu[1][idx]
            except Exception:
                Is = np.array([0]); Js = np.array([1])
            sigma = float(10 ** rng.uniform(-1.8, -0.6))
            X0 = base + rng.standard_normal((n, d)) * (sigma * 0.3)
            for a, b in zip(Is, Js):
                # push a,b apart along difference + noise
                diff = X0[a] - X0[b]
                diff /= (np.linalg.norm(diff) + 1e-12)
                amp = float(rng.uniform(0.05, 0.25))
                X0[a] = X0[a] + diff * amp + rng.standard_normal(d) * sigma * 0.5
                X0[b] = X0[b] - diff * amp + rng.standard_normal(d) * sigma * 0.5
            X0 = _norm_rows(X0)
        # choose quench cascade
        if tleft < 6:
            steps1, steps2 = 500, 400
        elif tleft < 25:
            steps1, steps2 = 800, 600
        else:
            steps1, steps2 = 1000, 800
            if rng.random() < 0.25:
                steps1, steps2 = 1400, 1000
        try:
            r = rng.random()
            if r < 0.30:
                s_pow = float(rng.uniform(2.0, 6.0))
                Xr = _quench_riesz(X0, steps=max(250, steps1 // 3), s_pow=s_pow,
                                   lr=float(10 ** rng.uniform(-1.6, -0.9)),
                                   mu=float(rng.uniform(0.8, 0.92)), deadline=HOP_END)
                X1, _ = _quench_lse(Xr, steps=steps1,
                                    T0=float(10 ** rng.uniform(-2.4, -1.7)),
                                    T1=float(10 ** rng.uniform(-4.2, -3.2)),
                                    lr=float(10 ** rng.uniform(-1.3, -0.8)),
                                    mu=float(rng.uniform(0.82, 0.93)),
                                    rng=rng, noise0=float(10 ** rng.uniform(-5.0, -3.5)) if rng.random() < 0.4 else 0.0,
                                    deadline=HOP_END, track=True)
            elif r < 0.60:
                X1, _ = _quench_lse(X0, steps=steps1,
                                    T0=float(10 ** rng.uniform(-2.4, -1.5)),
                                    T1=float(10 ** rng.uniform(-4.3, -3.1)),
                                    lr=float(10 ** rng.uniform(-1.4, -0.7)),
                                    mu=float(rng.uniform(0.82, 0.94)),
                                    rng=rng, noise0=float(10 ** rng.uniform(-5.0, -3.5)) if rng.random() < 0.5 else 0.0,
                                    deadline=HOP_END, track=True)
                # power squeeze
                X1, _ = _quench_power(X1, steps=max(200, steps2 // 3),
                                      thr_off=float(rng.uniform(0.005, 0.03)),
                                      pw=int(rng.integers(3, 8)),
                                      lr=float(10 ** rng.uniform(-1.6, -1.0)),
                                      mu=0.85, deadline=HOP_END)
            else:
                X1, _ = _quench_lse(X0, steps=steps1,
                                    T0=float(10 ** rng.uniform(-2.5, -1.8)),
                                    T1=float(10 ** rng.uniform(-3.9, -3.0)),
                                    lr=float(10 ** rng.uniform(-1.4, -0.8)),
                                    mu=float(rng.uniform(0.82, 0.93)),
                                    rng=rng, noise0=0.0, deadline=HOP_END, track=True)
            # final worst-pair descent sharpening
            X2, s2 = _quench_worst(X1, steps=steps2,
                                   lr=float(10 ** rng.uniform(-1.5, -0.9)),
                                   mu=float(rng.uniform(0.75, 0.9)),
                                   topB=int(rng.integers(4, 24)), rng=rng, deadline=HOP_END)
        except Exception:
            continue
        if not np.all(np.isfinite(X2)) or not np.isfinite(s2):
            continue
        if s2 > 0.26:
            continue
        try:
            new_top, new_max = _topk_vec(X2, 60)
            new_max = float(new_max)
        except Exception:
            continue
        # accept best by max then lexicographic
        improved = False
        if new_max < best_score - 1e-14:
            improved = True
        elif abs(new_max - best_score) <= 1e-12 and _lex_better(new_top, best_top):
            improved = True
        if improved:
            best_X = _norm_rows(X2.copy()); best_score = new_max; best_top = new_top.copy()
        # elite archive: keep diverse within window
        if new_max < max(elite_scores) + 2e-4 or len(elites) < 5:
            # check duplicate (chordal)
            dup = False
            for e in elites:
                try:
                    dd = float(np.abs(e * X2).sum(axis=1).mean())
                    if dd > 0.9999 and abs(_max_cos(e) - new_max) < 1e-9:
                        dup = True; break
                except Exception:
                    pass
            if not dup:
                elites.append(_norm_rows(X2.copy())); elite_scores.append(new_max); elite_tops.append(new_top.copy())
                od2 = np.argsort(np.array(elite_scores))
                elites = [elites[i] for i in od2[:5]]
                elite_scores = [elite_scores[i] for i in od2[:5]]
                elite_tops = [elite_tops[i] for i in od2[:5]]
    # greedy lexicographic squeeze on best + elites
    try:
        best_X, best_score = _greedy_squeeze(best_X, rng, iters=2500, deadline=HARD_END - 16.0, step0=0.004)[:2]
    except Exception:
        pass
    # final LP polish best (and runner-up if time)
    try:
        Xp = _lp_polish(best_X, seconds=9.0, deadline=HARD_END - 6.0)
        sp = _max_cos(Xp)
        if sp < best_score and np.all(np.isfinite(Xp)):
            best_X = Xp; best_score = sp
    except Exception:
        pass
    # polish second elite briefly and take better
    try:
        if len(elites) > 1 and time.time() < HARD_END - 8.0:
            e1 = elites[1]
            Xp2 = _lp_polish(e1, seconds=4.0, deadline=HARD_END - 2.0)
            sp2 = _max_cos(Xp2)
            if sp2 < best_score and np.all(np.isfinite(Xp2)):
                best_X = Xp2; best_score = sp2
    except Exception:
        pass
    best_X = _norm_rows(best_X)
    return best_X
