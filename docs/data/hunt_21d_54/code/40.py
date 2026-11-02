import time
import numpy as np

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _proj(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1.0
    y = x / n
    y = np.nan_to_num(y)
    n2 = np.linalg.norm(y, axis=1, keepdims=True)
    n2[n2 == 0] = 1.0
    return y / n2

def _quench(y, steps, lr0, T0_, T1_, mu=0.85):
    n, d = y.shape
    vel = np.zeros_like(y)
    for s in range(steps):
        frac = s / steps
        T = T0_ + (T1_ - T0_) * frac
        if T < 1e-12:
            T = 1e-12
        G = y @ y.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        E = (G - m) / T
        np.clip(E, -20.0, 0.0, out=E)
        W = np.exp(E)
        np.fill_diagonal(W, 0.0)
        rs = W.sum(axis=1, keepdims=True) + 1e-12
        grad = W @ y
        dot = np.einsum('ij,ij->i', grad, y)
        grad -= dot[:, None] * y
        grad /= rs
        vel = mu * vel + grad
        lr = lr0 * (1.0 - 0.6 * frac)
        y = y - lr * vel
        nrm = np.linalg.norm(y, axis=1, keepdims=True)
        nrm[nrm == 0] = 1.0
        y /= nrm
    return y

def _contact_counts(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    m = float(G.max())
    cnts = (G > m - 1e-9).sum(axis=1)
    rowmax = G.max(axis=1)
    return cnts, rowmax, m, G

def _kick(y, idx, rng, strength):
    for i in idx:
        d = y.shape[1]
        v = rng.normal(size=d)
        v -= (v @ y[i]) * y[i]
        nv = np.linalg.norm(v)
        if nv < 1e-12:
            continue
        v /= nv
        ang = strength * (0.5 + rng.random())
        # keep angle < 1.2
        if ang > 1.2:
            ang = 1.2
        y[i] = np.cos(ang) * y[i] + np.sin(ang) * v
    return y

def _refine(x, deadline):
    x = _proj(x)
    bv = _max_cos(x)
    best = x.copy()
    step = 2e-3
    delta = 0.004
    it = 0
    while time.time() < deadline:
        it += 1
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        mask = G > m - delta
        W = mask.astype(float)
        np.fill_diagonal(W, 0.0)
        cnts = W.sum(axis=1, keepdims=True)
        # if too few contacts widen delta
        if float(W.sum()) < 50:
            delta = min(delta * 2.0, 0.01)
            continue
        grad = W @ x
        dot = (grad * x).sum(axis=1, keepdims=True)
        grad -= dot * x
        grad /= (cnts + 1e-12)
        gn = np.linalg.norm(grad) + 1e-12
        # scale to keep move ~ step
        if gn > 1e-12:
            grad = grad / gn * np.sqrt(float(x.shape[0]))
        y = x - step * grad
        y = _proj(y)
        v = _max_cos(y)
        if v < bv:
            x = y
            best = y.copy()
            bv = v
            step = min(step * 1.3, 5e-3)
            delta = max(delta * 0.97, 2e-4)
        else:
            step *= 0.5
            if step < 1e-12:
                # shrink delta, reset step
                delta *= 0.5
                if delta < 5e-6:
                    break
                step = 2e-4
                # if stuck, nudge x = best
                x = best.copy()
        if it % 200 == 0 and time.time() >= deadline:
            break
    return best

def solve():
    T0 = time.time()
    HARD = T0 + 172.0
    rng = np.random.default_rng(7)
    try:
        b0 = np.load("best.npy")
    except Exception:
        b0 = None
    try:
        r0 = np.load("record.npy")
    except Exception:
        r0 = None
    cands = []
    if b0 is not None:
        try:
            b0 = _proj(np.nan_to_num(np.asarray(b0, dtype=float)))
            if b0.shape == (54, 21):
                cands.append(b0)
        except Exception:
            pass
    if r0 is not None:
        try:
            r0 = _proj(np.nan_to_num(np.asarray(r0, dtype=float)))
            if r0.shape == (54, 21):
                cands.append(r0)
        except Exception:
            pass
    if not cands:
        cands.append(_proj(rng.normal(size=(54, 21))))
    best = min(cands, key=_max_cos).copy()
    bv = _max_cos(best)
    n, d = 54, 21

    # reserve final refine time
    # basin hopping until HARD-22
    hop_deadline = HARD - 22.0
    it = 0
    strengths = [0.03, 0.08, 0.15, 0.25, 0.40, 0.55, 0.06, 0.12, 0.20, 0.35]
    while time.time() < hop_deadline:
        it += 1
        u = rng.random()
        y = None
        if u < 0.62:
            # kick subset + quench
            cnts, rowmax, m, G = _contact_counts(best)
            order_low = np.argsort(cnts)
            order_high = np.argsort(-cnts)
            s = strengths[(it - 1) % len(strengths)]
            # number to kick grows with strength
            if s < 0.1:
                nk = int(rng.integers(3, 8))
            elif s < 0.25:
                nk = int(rng.integers(6, 15))
            else:
                nk = int(rng.integers(10, 28))
            targets = set()
            targets.update(order_low[:max(1, nk // 3)].tolist())
            targets.update(order_high[:max(1, nk // 3)].tolist())
            while len(targets) < nk:
                targets.add(int(rng.integers(0, n)))
            targets = list(targets)
            y = best.copy()
            # random strength jitter
            sj = s * (0.7 + 0.6 * rng.random())
            y = _kick(y, targets, rng, sj)
            # tiny global tangent jitter
            gj = sj * 0.12
            noise = rng.normal(size=(n, d)) * gj
            noise -= (noise * y).sum(axis=1, keepdims=True) * y
            y = _proj(y + noise)
            steps = int(rng.choice([700, 1200, 1800, 2400]))
            T0_ = float(rng.choice([0.012, 0.02, 0.03]))
            T1_ = float(rng.choice([0.0012, 0.002, 0.0035]))
            lr0 = float(rng.choice([0.22, 0.30, 0.38]))
            mu = 0.85
            y = _quench(y, steps=steps, lr0=lr0, T0_=T0_, T1_=T1_, mu=mu)
        elif u < 0.85:
            # worst-point void reinsertion + short quench
            _, rowmax, _, _ = _contact_counts(best)
            order = np.argsort(-rowmax)
            iw = int(order[int(rng.integers(0, min(6, n)))])
            y = best.copy()
            K = 2000
            C = rng.normal(size=(K, d))
            C /= (np.linalg.norm(C, axis=1, keepdims=True) + 1e-12)
            D = C @ y.T
            # ignore self column iw
            D[:, iw] = -2.0
            vals = D.max(axis=1)
            bi = int(np.argmin(vals))
            y[iw] = C[bi]
            y = _proj(y)
            y = _quench(y, steps=int(rng.choice([700, 1200, 1800])), lr0=0.30, T0_=0.015, T1_=0.0015, mu=0.85)
        else:
            # partial random restart: keep some points, randomize rest
            keep = int(rng.choice([30, 36, 42, 48]))
            perm = rng.permutation(n)
            y = _proj(rng.normal(size=(n, d)))
            # copy keep points from best
            y[perm[:keep]] = best[perm[:keep]]
            y = _proj(y)
            y = _quench(y, steps=1800, lr0=0.32, T0_=0.025, T1_=0.002, mu=0.86)
        if y is None:
            continue
        v = _max_cos(y)
        if v < bv:
            best = y.copy()
            bv = v
        # safety time check
        if time.time() >= hop_deadline:
            break

    # final squeeze with remaining time
    try:
        if time.time() < HARD - 2.0:
            best = _refine(best, HARD - 1.0)
    except Exception:
        pass
    best = _proj(best)
    return best
