import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _norm(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return X / n

def _maxcos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _quench(X, iters, a0, a1, lr0, lr1, rng=None):
    X = _norm(X.copy())
    m = np.zeros_like(X)
    v = np.zeros_like(X)
    b1, b2 = 0.9, 0.999
    eps = 1e-8
    b1p, b2p = 1.0, 1.0
    for it in range(1, iters + 1):
        f = it / float(iters)
        alpha = a0 + (a1 - a0) * f
        lr = lr0 + (lr1 - lr0) * f
        G = X @ X.T
        np.fill_diagonal(G, -10.0)
        cmax = float(G.max())
        W = np.exp(alpha * (G - cmax))
        np.fill_diagonal(W, 0.0)
        s = float(W.sum())
        if not np.isfinite(s) or s <= 0:
            break
        W /= s
        Gr = W @ X
        proj = (Gr * X).sum(axis=1, keepdims=True)
        Gt = Gr - proj * X
        b1p *= b1; b2p *= b2
        m *= b1; m += (1.0 - b1) * Gt
        v *= b2; v += (1.0 - b2) * Gt * Gt
        mh = m / (1.0 - b1p); vh = v / (1.0 - b2p)
        step = lr * mh / (np.sqrt(vh) + eps)
        X = X - step
        X = _norm(X)
    return X

def _rand_rot(rng, d, sigma):
    A = rng.standard_normal((d, d))
    A = A - A.T
    M = np.eye(d) + sigma * A
    Q, R = np.linalg.qr(M)
    # fix signs for determinism (make diag R positive)
    s = np.sign(np.diag(R))
    s[s == 0] = 1.0
    Q = Q * s[None, :]
    return Q

def _greedy_fill(kept, k, rng, ntry=2500):
    d = kept.shape[1]
    cur = kept.copy()
    for _ in range(k):
        C = rng.standard_normal((ntry, d))
        C /= np.linalg.norm(C, axis=1, keepdims=True) + 1e-15
        if len(cur) > 0:
            dots = C @ cur.T
            w = dots.max(axis=1)
            bi = int(np.argmin(w))
            y = C[bi]
        else:
            y = C[0]
        # quick local push off nearest
        step = 0.05
        for __ in range(30):
            dd = cur @ y
            j = int(np.argmax(dd)) if len(cur) > 0 else -1
            if j < 0:
                break
            g = cur[j] - float(cur[j] @ y) * y
            nn = float(np.linalg.norm(g))
            if nn < 1e-12:
                break
            g /= nn
            y = y - step * g
            y /= np.linalg.norm(y) + 1e-15
            step *= 0.97
        cur = np.vstack([cur, y / (np.linalg.norm(y) + 1e-15)])
    return cur

def _lp_polish(X, deadline):
    X = _norm(X.copy())
    n, d = X.shape
    best = _maxcos(X)
    bx = X.copy()
    radius = 2.0e-3
    iu = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        if time.time() > deadline - 0.3:
            break
        G = X @ X.T
        c = G[iu]
        m = float(c.max())
        thr = m - max(20.0 * radius, 1e-9)
        idx = np.where(c > thr)[0]
        if idx.size == 0:
            break
        if idx.size > 1400:
            o = np.argsort(c[idx])[-1400:]
            idx = idx[o]
        sel = np.zeros_like(c, dtype=bool)
        sel[idx] = True
        I = iu[0][sel]; J = iu[1][sel]
        cs = c[sel]
        p = len(I)
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], axis=1).ravel()
        vals = np.concatenate([X[J], X[I], -np.ones((p, 1))], axis=1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((X.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv); obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-cs, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = X + res.x[:-1].reshape(n, d)
        y = _norm(y)
        vv = _maxcos(y)
        if vv < best:
            best = vv; bx = y.copy(); x_old = X; X = y
            radius = min(radius * 1.5, 0.05)
        else:
            if vv <= best + 1e-15:
                X = y
            radius *= 0.5
    return bx, best

def solve():
    t0 = time.time()
    HARD = t0 + 171.0
    EXPL = t0 + 148.0
    rng = np.random.default_rng(777)
    def clean(a):
        try:
            a = np.asarray(a, dtype=float)
        except Exception:
            return None
        if a.shape != (116, 6) or not np.all(np.isfinite(a)):
            return None
        n = np.linalg.norm(a, axis=1, keepdims=True)
        if np.any(n < 1e-12):
            return None
        return a / n
    cands = []
    for f in ("best.npy", "record.npy"):
        try:
            c = clean(np.load(f))
            if c is not None:
                cands.append(c)
        except Exception:
            pass
    if not cands:
        cands.append(_norm(rng.standard_normal((116, 6))))
    vals = [_maxcos(c) for c in cands]
    bi = int(np.argmin(vals))
    best_x = cands[bi].copy()
    best_v = float(vals[bi])
    cur_x = best_x.copy()
    cur_v = best_v
    n, d = 116, 6
    it = 0
    T0, T1 = 0.0025, 0.00008
    while time.time() < EXPL:
        it += 1
        if time.time() > HARD - 25.0:
            break
        frac = (time.time() - t0) / max(EXPL - t0, 1e-9)
        frac = min(1.0, max(0.0, frac))
        Temp = T0 + (T1 - T0) * frac
        # base: mostly wandering current, sometimes best
        if rng.random() < 0.75:
            base = cur_x
        else:
            base = best_x
        trial = base.copy()
        op = rng.random()
        if op < 0.18:
            # global jitter + quench
            sig = float(rng.uniform(0.015, 0.07))
            trial = trial + sig * rng.standard_normal(trial.shape)
            trial = _norm(trial)
            trial = _quench(trial, iters=int(rng.integers(300, 500)),
                            a0=25, a1=200, lr0=0.025, lr1=0.004)
        elif op < 0.45:
            # cap deletion around random center + greedy refill (topology change)
            c = int(rng.integers(0, n))
            dots = trial @ trial[c]
            k = int(rng.integers(8, 25))
            order = np.argsort(-dots)
            kill = order[:k]
            keep_mask = np.ones(n, dtype=bool)
            keep_mask[kill] = False
            kept = trial[keep_mask]
            filled = _greedy_fill(kept, k, rng, ntry=2500)
            trial = filled
            # small noise to avoid exact symmetry
            trial = trial + 0.003 * rng.standard_normal(trial.shape)
            trial = _norm(trial)
            trial = _quench(trial, iters=int(rng.integers(350, 600)),
                            a0=20, a1=260, lr0=0.03, lr1=0.003)
        elif op < 0.63:
            # coherent subset rotation
            c = int(rng.integers(0, n))
            dots = trial @ trial[c]
            k = int(rng.integers(15, 40))
            order = np.argsort(-dots)
            sub = order[:k]
            sig = float(rng.uniform(0.15, 0.6))
            Q = _rand_rot(rng, d, sig)
            trial[sub] = trial[sub] @ Q.T
            trial = _norm(trial)
            # optionally jitter subset a bit
            trial[sub] += 0.01 * rng.standard_normal((k, d))
            trial = _norm(trial)
            trial = _quench(trial, iters=int(rng.integers(350, 600)),
                            a0=25, a1=240, lr0=0.028, lr1=0.003)
        elif op < 0.80:
            # hub neighbourhood kick (attack overloaded points)
            G = trial @ trial.T
            np.fill_diagonal(G, -2.0)
            mc = float(G.max())
            cnt = (G > mc - 1e-7).sum(axis=1)
            order = np.argsort(-cnt)
            k = int(rng.integers(6, 14))
            idx = order[:k].copy()
            # randomize one
            if rng.random() < 0.4:
                idx[int(rng.integers(0, k))] = int(rng.integers(0, n))
            sig = float(rng.uniform(0.08, 0.28))
            trial[idx] += sig * rng.standard_normal((k, d))
            trial = _norm(trial)
            trial = _quench(trial, iters=int(rng.integers(350, 600)),
                            a0=30, a1=260, lr0=0.03, lr1=0.003)
        else:
            # random slab rebirth along random direction
            u = rng.standard_normal(d); u /= np.linalg.norm(u) + 1e-15
            proj = trial @ u
            k = int(rng.integers(10, 26))
            order = np.argsort(proj)  # take top or bottom randomly
            if rng.random() < 0.5:
                kill = order[-k:]
            else:
                kill = order[:k]
            keep_mask = np.ones(n, dtype=bool)
            keep_mask[kill] = False
            kept = trial[keep_mask]
            filled = _greedy_fill(kept, k, rng, ntry=2000)
            trial = filled
            trial = _norm(trial)
            trial = _quench(trial, iters=int(rng.integers(350, 600)),
                            a0=20, a1=240, lr0=0.03, lr1=0.003)
        v = _maxcos(trial)
        if v < best_v - 1e-12:
            best_v = v
            best_x = trial.copy()
            cur_x = trial.copy()
            cur_v = v
        elif v < cur_v or rng.random() < np.exp(-(v - cur_v) / max(Temp, 1e-12)):
            if v < cur_v + 0.03:
                cur_x = trial.copy()
                cur_v = v
        # periodic return to best basin with kick
        if it % 40 == 0:
            cur_x = best_x.copy()
            cur_v = best_v
            # escape kick so next iteration starts elsewhere
            c = int(rng.integers(0, n))
            dots = cur_x @ cur_x[c]
            order = np.argsort(-dots)[:12]
            cur_x[order] += 0.05 * rng.standard_normal((12, d))
            cur_x = _norm(cur_x)
            cur_v = _maxcos(cur_x)
        if it % 25 == 0 and time.time() > EXPL - 5:
            break
    # final squeeze: strong quench then LP
    if time.time() < HARD - 3:
        best_x = _quench(best_x, iters=600, a0=40, a1=400, lr0=0.02, lr1=0.002)
        best_v = _maxcos(best_x)
    if time.time() < HARD - 2:
        remain = HARD - 1.0 - time.time()
        dl = time.time() + min(max(remain, 2.0), 20.0)
        bx, bv = _lp_polish(best_x, dl)
        if bv < best_v:
            best_x, best_v = bx, bv
    best_x = _norm(best_x)
    return best_x
