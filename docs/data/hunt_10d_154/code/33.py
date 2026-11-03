import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _norm(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return x / n

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _relax(x, steps, lr, T0, T1):
    n, d = x.shape
    m_ad = np.zeros_like(x)
    v_ad = np.zeros_like(x)
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    b1p, b2p = 1.0, 1.0
    if steps > 1:
        logr = np.log(max(T1, 1e-12) / max(T0, 1e-12)) / (steps - 1)
    else:
        logr = 0.0
    for t in range(1, steps + 1):
        T = T0 * np.exp(logr * (t - 1))
        if T < 8e-5:
            T = 8e-5
        G = x @ x.T
        np.fill_diagonal(G, -10.0)
        mv = float(G.max())
        E = np.exp((G - mv) / T)
        np.fill_diagonal(E, 0.0)
        S = E.sum() * 0.5
        if S <= 0 or not np.isfinite(S):
            break
        Grad = (E @ x) / S
        dot = (Grad * x).sum(axis=1, keepdims=True)
        Gt = Grad - dot * x
        b1p *= beta1
        b2p *= beta2
        m_ad = beta1 * m_ad + (1.0 - beta1) * Gt
        v_ad = beta2 * v_ad + (1.0 - beta2) * (Gt * Gt)
        mhat = m_ad / (1.0 - b1p)
        vhat = v_ad / (1.0 - b2p)
        step = mhat / (np.sqrt(vhat) + eps)
        sn = np.linalg.norm(step, axis=1, keepdims=True) + 1e-12
        step = step / np.maximum(sn / 1.0, 1.0)
        x = x - lr * step
        x = _norm(x)
    return x

def _two_stage(x, lr=0.02):
    x = _relax(x, 550, lr, 0.03, 0.007)
    x = _relax(x, 550, lr * 0.8, 0.007, 0.0012)
    return x

def _lp_polish(x, seconds, deadline=1e18):
    x = _norm(x)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        if time.time() > deadline:
            break
        c = (x @ x.T)[iu]
        m = float(c.max())
        sel = c > m - max(20.0 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        if p == 0:
            break
        if p > 4000:
            # subsample worst to keep LP tractable
            idx = np.argpartition(-c[sel], 4000)[:4000]
            I, J = I[idx], J[idx]
            p = 4000
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv)
            obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-c[sel][:p] if p != int(sel.sum()) else -c[sel], A_eq=A_eq, b_eq=np.zeros(n),
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
            x, best, radius = y, v, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def _greedy_hole(prop, victim, rng, C=6000):
    d = prop.shape[1]
    cand = _norm(rng.normal(size=(C, d)))
    D = cand @ prop.T
    if victim is not None and victim >= 0:
        D[:, victim] = -2.0
    md = D.max(axis=1)
    K = 8
    idx = np.argpartition(md, K)[:K]
    # randomize among top-K for diversity
    r = rng.random()
    if r < 0.5:
        pick = idx[np.argmin(md[idx])]
    else:
        pick = idx[rng.integers(K)]
    bc = cand[pick].copy()
    bv = float(md[pick])
    # one local deepening
    pert = _norm(bc[None, :] + rng.normal(scale=0.05, size=(2000, d)))
    D2 = pert @ prop.T
    if victim is not None and victim >= 0:
        D2[:, victim] = -2.0
    m2 = D2.max(axis=1)
    j = int(np.argmin(m2))
    if float(m2[j]) < bv:
        bc = pert[j]
    return bc

def _cap_rotate(X, rng):
    n, d = X.shape
    Y = X.copy()
    u = rng.normal(size=d)
    u /= np.linalg.norm(u) + 1e-12
    dots = Y @ u
    thr = float(rng.uniform(-0.1, 0.65))
    S = np.where(dots > thr)[0]
    if len(S) < 4 or len(S) > 70:
        return None
    a = rng.normal(size=d)
    a = a - (a @ u) * u
    # make a orthogonal to nothing special, just normalize
    na = np.linalg.norm(a)
    if na < 1e-8:
        return None
    a /= na
    b = rng.normal(size=d)
    b = b - (b @ a) * a - (b @ u) * u
    nb = np.linalg.norm(b)
    if nb < 1e-8:
        return None
    b /= nb
    theta = float(rng.uniform(0.15, 0.5)) * (1.0 if rng.random() < 0.5 else -1.0)
    ct, st = np.cos(theta), np.sin(theta)
    Xs = Y[S]
    pa = Xs @ a
    pb = Xs @ b
    Xs = Xs + (st * pa - (1 - ct) * 0 + 0)[:, None] * 0  # placeholder
    # exact rotation in plane (a,b):
    Xs = Xs + (st * pa)[:, None] * b.reshape(1, d) - (st * pb)[:, None] * a.reshape(1, d) + ((ct - 1.0) * pa)[:, None] * a.reshape(1, d) + ((ct - 1.0) * pb)[:, None] * b.reshape(1, d)
    Y[S] = Xs
    Y += 0.008 * rng.normal(size=(n, d))
    Y = _norm(Y)
    return Y

def solve():
    t0 = time.time()
    DEAD = t0 + 172.0
    EXPL_END = t0 + 132.0
    rng = np.random.default_rng(0)
    n, d = 154, 10
    best = None
    best_val = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            if a.shape == (n, d):
                a = _norm(a.astype(float))
                v = _max_cos(a)
                if v < best_val:
                    best_val = v
                    best = a.copy()
        except Exception:
            pass
    if best is None:
        best = _norm(rng.normal(size=(n, d)))
        best_val = _max_cos(best)
    cur = best.copy()
    cur_val = best_val
    it = 0
    while time.time() < EXPL_END and time.time() < DEAD - 25:
        it += 1
        elapsed = time.time() - t0
        frac = elapsed / max(EXPL_END - t0, 1e-9)
        T_acc = 0.002 * (1.0 - frac) + 0.0002
        mode = it % 4
        Xtry = None
        if mode == 0:
            # coherent cap rotation (breaks inter-cap contacts)
            for _ in range(4):
                Xtry = _cap_rotate(cur, rng)
                if Xtry is not None:
                    break
            if Xtry is None:
                Xtry = _norm(cur + 0.10 * rng.normal(size=(n, d)))
        elif mode == 1:
            # delete worst k and greedy refill (change point count intermediately)
            G = cur @ cur.T
            np.fill_diagonal(G, -2.0)
            rowmax = G.max(axis=1)
            k = int(rng.integers(4, 10))
            worst = np.argsort(-rowmax)[:k]
            keep_mask = np.ones(n, dtype=bool)
            keep_mask[worst] = False
            kept = cur[keep_mask].copy()
            newpts = []
            existing = kept.copy()
            for ii in range(k):
                bc = _greedy_hole(existing, -1, rng, C=5000)
                newpts.append(bc)
                existing = np.vstack([existing, bc[None, :]])
            Y = cur.copy()
            Y[worst] = np.array(newpts)
            Y += 0.005 * rng.normal(size=(n, d))
            Xtry = _norm(Y)
        elif mode == 2:
            # hub-to-hole: relocate high-contact hubs
            G = cur @ cur.T
            np.fill_diagonal(G, -2.0)
            M = float(G.max())
            deg = (G > M - 0.01).sum(axis=1)
            order = np.argsort(-deg)
            k = int(rng.integers(3, 6))
            hubs = order[:k]
            # also include a rowmax-worst to break dual-critical
            rowmax = G.max(axis=1)
            hubs = np.union1d(hubs, np.argsort(-rowmax)[:2])[:k+2]
            Y = cur.copy()
            for h in hubs:
                # victim = h so hole ignores itself
                bc = _greedy_hole(Y, int(h), rng, C=7000)
                Y[int(h)] = bc
            Y += 0.006 * rng.normal(size=(n, d))
            Xtry = _norm(Y)
        else:
            sig = float(rng.uniform(0.06, 0.14))
            Xtry = _norm(cur + sig * rng.normal(size=(n, d)))

        # two-stage annealed softmax relaxation
        Xref = _two_stage(Xtry, lr=0.02)
        val = _max_cos(Xref)
        # filter: only polish if reasonably close
        thresh = cur_val + 0.010
        if val < best_val - 1e-9:
            # immediate confirm with short LP
            sec = min(2.0, max(1.0, (DEAD - 22 - time.time()) / 12.0))
            if sec < 0.8:
                best, best_val = Xref.copy(), val
                cur, cur_val = Xref.copy(), val
                continue
            Xpol = _lp_polish(Xref, sec, deadline=DEAD - 20)
            vpol = _max_cos(Xpol)
            if vpol < best_val:
                best, best_val = Xpol.copy(), vpol
            if vpol < cur_val or rng.random() < 0.9:
                cur, cur_val = Xpol.copy(), vpol
            elif val < cur_val:
                cur, cur_val = Xref.copy(), val
        elif val < thresh:
            sec = min(1.4, max(0.9, (DEAD - 22 - time.time()) / 15.0))
            if sec < 0.7 or time.time() > EXPL_END - 2:
                # SA on unpolished to save time
                if val < cur_val or rng.random() < np.exp((cur_val - val) / T_acc):
                    cur, cur_val = Xref.copy(), val
                continue
            # polish better of Xref (skip quick push to save time)
            Xpol = _lp_polish(Xref, sec, deadline=DEAD - 20)
            vpol = _max_cos(Xpol)
            if vpol < best_val:
                best, best_val = Xpol.copy(), vpol
                cur, cur_val = Xpol.copy(), vpol
            elif vpol < cur_val or rng.random() < np.exp((cur_val - vpol) / T_acc):
                cur, cur_val = Xpol.copy(), vpol
        else:
            # far: occasional SA acceptance to diversify without LP
            if rng.random() < 0.02 and val < cur_val + 0.03:
                cur, cur_val = Xref.copy(), val
        # periodic restart from best if cur drifts too far
        if cur_val > best_val + 0.015 and (it % 10 == 0):
            cur, cur_val = best.copy(), best_val
        # rare full random injection
        if it % 25 == 0 and time.time() < EXPL_END - 10:
            Xr = _norm(rng.normal(size=(n, d)))
            Xr = _two_stage(Xr, lr=0.025)
            vr = _max_cos(Xr)
            if vr < cur_val + 0.005:
                sec = 1.0
                Xp = _lp_polish(Xr, sec, deadline=DEAD - 20)
                vp = _max_cos(Xp)
                if vp < best_val:
                    best, best_val = Xp.copy(), vp
                    cur, cur_val = Xp.copy(), vp

    remaining = (t0 + 172.0) - time.time() - 1.0
    if remaining > 3:
        try:
            best = _lp_polish(best, remaining, deadline=t0 + 173.0)
            best_val = _max_cos(best)
        except Exception:
            pass
    best = _norm(best)
    return best
