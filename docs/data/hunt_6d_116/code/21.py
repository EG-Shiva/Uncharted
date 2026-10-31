import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _norm(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return x / n

def _maxcos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(np.max(g))

def _smooth(x, steps, T0, T1, lr0, lr1):
    x = _norm(x)
    n, d = x.shape
    for s in range(steps):
        f = s / max(steps - 1, 1)
        T = T0 * (T1 / T0) ** f
        lr = lr0 * (1 - f) + lr1 * f
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        M = float(np.max(G))
        E = np.exp((G - M) / T)
        np.fill_diagonal(E, 0.0)
        S = float(np.sum(E))
        if not np.isfinite(S) or S <= 0:
            break
        Grad = (E @ x) / S
        dot = np.sum(Grad * x, axis=1, keepdims=True)
        Gt = Grad - dot * x
        gn = float(np.max(np.linalg.norm(Gt, axis=1)))
        if gn > 0:
            mx = 0.05
            if gn * lr > mx:
                Gt = Gt * (mx / (gn * lr))
        x = x - lr * Gt
        x = _norm(x)
    return x

def _lp_polish(x, deadline):
    x = _norm(x)
    n, d = x.shape
    best = _maxcos(x)
    bx = x.copy()
    radius = 2.5e-3
    iu = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        G = x @ x.T
        c = G[iu]
        m = float(np.max(c))
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
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], axis=1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
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
        y = _norm(y)
        v = _maxcos(y)
        if v < best:
            best = v; bx = y.copy(); x = y
            radius = min(radius * 1.5, 0.05)
        else:
            # keep best but shrink; also step to y if equal to allow sliding
            if v <= best + 1e-15:
                x = y
            radius *= 0.5
    return bx, best

def _deep_hole(others, rng, ntry=3000):
    d = others.shape[1]
    C = rng.standard_normal((ntry, d))
    C /= np.linalg.norm(C, axis=1, keepdims=True)
    dots = C @ others.T
    w = np.max(dots, axis=1)
    y = C[int(np.argmin(w))].copy()
    step = 0.05
    for _ in range(60):
        dd = others @ y
        j = int(np.argmax(dd))
        g = others[j] - float(others[j] @ y) * y
        n = float(np.linalg.norm(g))
        if n < 1e-12:
            break
        g /= n
        y = y - step * g
        y /= np.linalg.norm(y)
        step *= 0.97
    return y / np.linalg.norm(y)

def solve():
    t0 = time.time()
    HARD_END = t0 + 171.0
    EXPL_END = t0 + 142.0
    rng = np.random.default_rng(20260707)
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
    # quick initial squeeze if time allows
    if HARD_END - time.time() > 10:
        bx, bv = _lp_polish(best_x, min(time.time() + 4.0, HARD_END - 5.0))
        if bv < best_v:
            best_x, best_v = bx, bv
            cur_x, cur_v = bx.copy(), bv
    while time.time() < EXPL_END:
        it += 1
        if time.time() > HARD_END - 15.0:
            break
        # periodically restart from best
        if it % 12 == 0:
            base = best_x
            base_v = best_v
        else:
            base = cur_x
            base_v = cur_v
        G0 = base @ base.T
        np.fill_diagonal(G0, -2.0)
        m0 = float(np.max(G0))
        per = np.sum(G0 > (m0 - 1e-7), axis=1)
        pm = np.max(G0, axis=1)
        mode = it % 6
        X = base.copy()
        if mode == 0:
            X = X + rng.normal(scale=0.02, size=(n, d))
            X = _norm(X)
            X = _smooth(X, steps=150, T0=0.025, T1=0.004, lr0=2.0, lr1=0.25)
            lp_b = 1.1
        elif mode == 1:
            X = X + rng.normal(scale=0.08, size=(n, d))
            X = _norm(X)
            X = _smooth(X, steps=260, T0=0.025, T1=0.003, lr0=2.0, lr1=0.2)
            lp_b = 1.7
        elif mode == 2:
            # hub-targeted hard kick
            noise = rng.normal(scale=0.012, size=(n, d))
            top = np.argsort(-per)[:8]
            noise[top] += rng.normal(scale=0.22, size=(8, d))
            ratt = np.where(per <= 0.5)[0]
            if ratt.size > 0:
                noise[ratt] += rng.normal(scale=0.08, size=(ratt.size, d))
            X = _norm(X + noise)
            X = _smooth(X, steps=280, T0=0.03, T1=0.003, lr0=2.0, lr1=0.2)
            lp_b = 1.8
        elif mode == 3:
            # rattler void reinsertion
            ratt = np.where(per <= 0.5)[0]
            if ratt.size == 0:
                ratt = np.argsort(pm)[:3]
            k = int(min(len(ratt), 1 + (it % 2)))
            idx = rng.choice(ratt, size=k, replace=False) if len(ratt) >= k else ratt
            for j in idx:
                others = np.delete(X, int(j), axis=0)
                X[int(j)] = _deep_hole(others, rng, ntry=3000)
            X = X + rng.normal(scale=0.012, size=(n, d))
            X = _norm(X)
            X = _smooth(X, steps=260, T0=0.025, T1=0.003, lr0=2.0, lr1=0.2)
            lp_b = 1.7
        elif mode == 4:
            # worst-pair surgery: randomize one end of worst pairs
            flat = np.argmax(G0)
            a = flat // n; b = flat % n
            choices = [int(a), int(b)]
            # add 1-2 more random overloaded
            top = np.argsort(-per)[:10]
            extra = int(rng.integers(1, 3))
            choices += [int(v) for v in rng.choice(top, size=extra, replace=False)]
            choices = list(dict.fromkeys(choices))[:4]
            for j in choices:
                if rng.random() < 0.5:
                    others = np.delete(X, int(j), axis=0)
                    X[int(j)] = _deep_hole(others, rng, ntry=2500)
                else:
                    v = rng.standard_normal(d); v /= np.linalg.norm(v)
                    X[int(j)] = v
            X = X + rng.normal(scale=0.01, size=(n, d))
            X = _norm(X)
            X = _smooth(X, steps=280, T0=0.03, T1=0.003, lr0=2.2, lr1=0.2)
            lp_b = 1.8
        else:
            # full rebirth of few points
            k = int(rng.integers(2, 6))
            idx = rng.choice(n, size=k, replace=False)
            R = rng.standard_normal((k, d)); R /= np.linalg.norm(R, axis=1, keepdims=True)
            X[idx] = R
            X = X + rng.normal(scale=0.01, size=(n, d))
            X = _norm(X)
            X = _smooth(X, steps=300, T0=0.03, T1=0.003, lr0=2.0, lr1=0.2)
            lp_b = 1.8
        vs = _maxcos(X)
        if vs > best_v + 0.025:
            continue
        dl = min(time.time() + lp_b, EXPL_END + 1.0, HARD_END - 10.0)
        if dl <= time.time() + 0.05:
            break
        Xp, vp = _lp_polish(X, dl)
        if vp < best_v - 5e-13:
            best_v = vp; best_x = Xp.copy()
            cur_x, cur_v = Xp.copy(), vp
            continue
        # wandering acceptance for cur
        frac = (time.time() - t0) / max(EXPL_END - t0, 1.0)
        Temp = 0.003 * (0.0002 / 0.003) ** min(max(frac, 0.0), 1.0)
        if vp < cur_v or rng.random() < np.exp(-(vp - cur_v) / max(Temp, 1e-9)):
            # only accept if not too bad
            if vp < cur_v + 0.01:
                cur_x, cur_v = Xp.copy(), vp
        elif it % 12 == 11:
            # occasional reset of cur to best to avoid drift
            cur_x, cur_v = best_x.copy(), best_v
    # final polish
    rem = HARD_END - time.time()
    if rem > 4.0:
        fb = min(rem - 1.5, 26.0)
        Xf, vf = _lp_polish(best_x, time.time() + fb)
        if vf <= best_v:
            best_x, best_v = Xf, vf
    best_x = _norm(best_x)
    return best_x
