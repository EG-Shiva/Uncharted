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

def _relax_softmax(x, steps, lr, T0, T1):
    n, d = x.shape
    m_ad = np.zeros_like(x)
    v_ad = np.zeros_like(x)
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    b1p = 1.0
    b2p = 1.0
    if steps > 1:
        logr = np.log(max(T1, 1e-12) / max(T0, 1e-12)) / (steps - 1)
    else:
        logr = 0.0
    for t in range(1, steps + 1):
        T = T0 * np.exp(logr * (t - 1))
        if T < 1e-12:
            T = 1e-12
        G = x @ x.T
        np.fill_diagonal(G, -10.0)
        mv = G.max()
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
        x = x - lr * step
        x = _norm(x)
    return x

def _max_sharpen(x, steps=120, lr0=0.008):
    n = x.shape[0]
    for s in range(steps):
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        # top pair
        k = np.argmax(G)
        i = k // n
        j = k % n
        cij = G[i, j]
        # tangential push apart
        gi = x[j] - cij * x[i]
        gj = x[i] - cij * x[j]
        lr = lr0 / (1.0 + s * 0.02)
        x[i] = x[i] - lr * gi
        x[j] = x[j] - lr * gj
        # renormalize those two
        ni = np.linalg.norm(x[i]); nj = np.linalg.norm(x[j])
        if ni == 0 or not np.isfinite(ni):
            continue
        if nj == 0 or not np.isfinite(nj):
            continue
        x[i] /= ni; x[j] /= nj
    return x

def _hole_pick(prop, victim, rng, C=8000, K=8):
    d = prop.shape[1]
    cand = rng.normal(size=(C, d))
    cand = _norm(cand)
    Dots = cand @ prop.T
    Dots[:, victim] = -2.0
    md = Dots.max(axis=1)
    idx = np.argpartition(md, K)[:K]
    if rng.random() < 0.5:
        pick = idx[np.argmin(md[idx])]
    else:
        pick = idx[rng.integers(K)]
    return cand[pick]

def _lp_polish(x, seconds):
    x = _norm(x)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        c = (x @ x.T)[iu]
        m = c.max()
        sel = c > m - max(20 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        if p == 0:
            break
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
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
            x, best, radius = y, v, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def solve():
    t0 = time.time()
    rng = np.random.default_rng(12345)
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
    # exploration deadline: leave ~30s for final LP
    EXPL = t0 + 138.0
    HARD = t0 + 170.0
    it = 0
    while time.time() < EXPL and time.time() < HARD - 32.0:
        it += 1
        elapsed = time.time() - t0
        frac = elapsed / max(EXPL - t0, 1.0)
        T_acc = 0.0016 * (1.0 - frac) + 0.0002
        if T_acc < 0.0002:
            T_acc = 0.0002
        r = rng.random()
        prop = cur.copy()
        Gc = cur @ cur.T
        np.fill_diagonal(Gc, -2.0)
        mcur = float(Gc.max())
        delta = 0.002 + rng.random() * 0.005
        cnt = (Gc > mcur - delta).sum(axis=1)
        rattlers = np.where(cnt == 0)[0]
        if r < 0.55:
            nk = int(rng.integers(1, 4))
            probs = cnt.astype(float) + 0.5
            probs /= probs.sum()
            victims = rng.choice(n, size=nk, replace=False, p=probs)
            for vv in victims:
                prop[vv] = _hole_pick(prop, vv, rng, C=7000, K=8)
            # also refill one rattler if exists
            if len(rattlers) > 0 and rng.random() < 0.6:
                vr = int(rattlers[rng.integers(len(rattlers))])
                if vr not in victims:
                    prop[vr] = _hole_pick(prop, vr, rng, C=7000, K=8)
            prop = _relax_softmax(prop, steps=int(450 + rng.integers(300)), lr=0.015, T0=0.012, T1=0.0012)
        elif r < 0.78:
            # rattler-focused refill + light jammed kick
            nr = int(rng.integers(1, 3))
            victims = []
            if len(rattlers) > 0:
                rr = rng.choice(rattlers, size=min(nr, len(rattlers)), replace=False)
                victims.extend([int(v) for v in rr])
            # add one jammed
            probs = cnt.astype(float) + 0.5
            probs /= probs.sum()
            vj = int(rng.choice(n, p=probs))
            if vj not in victims:
                victims.append(vj)
            for vv in victims:
                prop[vv] = _hole_pick(prop, vv, rng, C=9000, K=10)
            prop = _relax_softmax(prop, steps=int(450 + rng.integers(300)), lr=0.014, T0=0.010, T1=0.0010)
        else:
            sigma = 0.02 + rng.random() * 0.06
            # bias noise to high-contact points
            w = (cnt.astype(float) + 1.0)
            w /= w.mean()
            noise = rng.normal(size=prop.shape) * (sigma * w[:, None])
            prop = prop + noise
            prop = _norm(prop)
            prop = _relax_softmax(prop, steps=int(500 + rng.integers(300)), lr=0.015, T0=0.012, T1=0.0010)
        # direct max sharpening
        prop = _max_sharpen(prop, steps=100, lr0=0.006)
        prop = _norm(prop)
        prop_val = _max_cos(prop)
        # Metropolis acceptance on cur
        if prop_val < cur_val or rng.random() < np.exp((cur_val - prop_val) / T_acc):
            cur = prop.copy()
            cur_val = prop_val
        if prop_val < best_val:
            best_val = prop_val
            best = prop.copy()
        # early exit if time
        if time.time() > EXPL:
            break
    # final polish on best
    remain = HARD - time.time()
    lp_time = min(32.0, max(8.0, remain - 2.0))
    if lp_time > 3.0 and remain > 5.0:
        try:
            best = _lp_polish(best, lp_time)
            best_val = _max_cos(best)
        except Exception:
            pass
    best = _norm(best)
    return best
