import time
import numpy as np

def _mc(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(np.max(g))

def _proj(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return x / n

def _quench(y, steps, lr0, T0, T1, mu=0.80, rng=None):
    n, d = y.shape
    vel = np.zeros((n, d))
    for s in range(steps):
        frac = s / max(1, steps - 1)
        T = T0 * (1 - frac) + T1 * frac
        if T < 1e-4:
            T = 1e-4
        G = y @ y.T
        np.fill_diagonal(G, -2.0)
        m = float(np.max(G))
        E = (G - m) / T
        np.clip(E, -25, 0, out=E)
        W = np.exp(E)
        np.fill_diagonal(W, 0.0)
        rs = W.sum(axis=1, keepdims=True) + 1e-12
        grad = W @ y
        dot = np.einsum('ij,ij->i', grad, y)
        grad -= dot[:, None] * y
        grad /= rs
        vel = mu * vel + (1 - mu) * grad if s > 0 else grad
        lr = lr0 * (1.0 - 0.6 * frac)
        y = y - lr * vel
        y = _proj(y)
    return y

def _squeeze(y, steps, lr0=0.02):
    # minimax: push apart only near-max pairs
    n, d = y.shape
    for s in range(steps):
        G = y @ y.T
        np.fill_diagonal(G, -2.0)
        m = float(np.max(G))
        margin = max(0.002 * (1 - s / steps), 0.0002)
        thr = m - margin
        # binary active
        A = (G > thr).astype(float)
        rs = A.sum(axis=1, keepdims=True) + 1e-12
        grad = A @ y
        dot = np.einsum('ij,ij->i', grad, y)
        grad -= dot[:, None] * y
        grad /= rs
        gn = np.linalg.norm(grad, axis=1, keepdims=True) + 1e-12
        # only move points that are active
        has = (rs[:, 0] > 0.5)
        step = lr0 * (1 - 0.5 * s / steps)
        y[has] -= step * grad[has] / np.maximum(1.0, gn[has])
        y = _proj(y)
    return y

def _best_void(Yex, rng, K=3000, refine=60):
    n, d = Yex.shape
    C = rng.normal(size=(K, d))
    C /= (np.linalg.norm(C, axis=1, keepdims=True) + 1e-12)
    D = C @ Yex.T
    mx = D.max(axis=1)
    bi = int(np.argmin(mx))
    c = C[bi:bi+1].copy()
    # local hill climb: minimise softmax max to Yex
    for s in range(refine):
        T = max(0.01 * (1 - s / refine), 0.001)
        G = (c @ Yex.T)[0]
        m = float(G.max())
        E = (G - m) / T
        np.clip(E, -25, 0, out=E)
        W = np.exp(E)
        W /= (W.sum() + 1e-12)
        grad = (W @ Yex)[None, :]
        dot = float(grad[0] @ c[0])
        grad = grad - dot * c
        c = c - 0.1 * grad
        c /= (np.linalg.norm(c, axis=1, keepdims=True) + 1e-12)
    return c[0]

def solve():
    T0 = time.time()
    LIM = 170.0
    def left():
        return LIM - (time.time() - T0)
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
        b0 = _proj(np.nan_to_num(np.asarray(b0, dtype=float)))
        if b0.shape == (54, 21):
            cands.append(b0)
    if r0 is not None:
        r0 = _proj(np.nan_to_num(np.asarray(r0, dtype=float)))
        if r0.shape == (54, 21):
            cands.append(r0)
    if not cands:
        cands.append(_proj(rng.normal(size=(54, 21))))
    best = min(cands, key=_mc).copy()
    bv = _mc(best)
    n, d = 54, 21

    # quick initial squeeze of starting point
    best = _squeeze(best.copy(), steps=400, lr0=0.02)
    v = _mc(best)
    if v < bv:
        bv = v
    else:
        # keep squeezed anyway if not worse by much (it tightens)
        if v < bv + 1e-9:
            best = best.copy()
            bv = v if v < bv else bv

    it = 0
    # precompute contact info helper
    while left() > 12:
        it += 1
        mode = it % 5
        base = best
        y = base.copy()
        if mode in (0, 1, 3):
            # delete-and-reinsert
            G = y @ y.T
            np.fill_diagonal(G, -2.0)
            rowmax = G.max(axis=1)
            if mode == 0:
                victim = int(np.argmax(rowmax))
            elif mode == 1:
                # random among top-8 worst
                order = np.argsort(-rowmax)
                victim = int(order[int(rng.integers(0, 8))])
            else:
                # random high-contact count point
                m = float(G.max())
                cnt = (G > m - 1e-9).sum(axis=1)
                order = np.argsort(-cnt)
                victim = int(order[int(rng.integers(0, 8))])
            Yex = np.delete(y, victim, axis=0)
            nv = _best_void(Yex, rng, K=2500, refine=50)
            y[victim] = nv
            # small jitter to neighbours of victim to break local graph
            G2 = y @ y.T
            np.fill_diagonal(G2, -2.0)
            m2 = float(G2.max())
            # neighbours close to victim
            drow = G2[victim]
            nbr = np.where(drow > m2 - 0.02)[0]
            for j in nbr[:6]:
                if j == victim:
                    continue
                t = rng.normal(size=d)
                t -= (t @ y[j]) * y[j]
                t /= (np.linalg.norm(t) + 1e-12)
                ang = float(rng.uniform(0.005, 0.05))
                y[j] = np.cos(ang) * y[j] + np.sin(ang) * t
            y = _proj(y)
            steps = int(rng.integers(1200, 2800))
            lr0 = float(rng.uniform(0.08, 0.35))
            T0s = float(rng.uniform(0.008, 0.03))
            T1s = float(rng.uniform(0.0008, 0.003))
            y = _quench(y, steps=steps, lr0=lr0, T0=T0s, T1=T1s, mu=0.82)
            y = _squeeze(y, steps=250, lr0=0.015)
        elif mode == 2:
            # collective jitter + quench (topology change via noise)
            sig = float(rng.choice([0.01, 0.03, 0.06, 0.12]))
            nz = rng.normal(size=(n, d)) * sig
            nz -= (nz * y).sum(axis=1, keepdims=True) * y
            y = _proj(y + nz)
            steps = int(rng.integers(1500, 3000))
            lr0 = float(rng.uniform(0.10, 0.30))
            y = _quench(y, steps=steps, lr0=lr0, T0=0.015, T1=0.001, mu=0.85)
            y = _squeeze(y, steps=200, lr0=0.015)
        else:
            # subset kick: kick 4-10 random points strongly
            k = int(rng.integers(4, 11))
            idx = rng.choice(n, size=k, replace=False)
            for i in idx:
                t = rng.normal(size=d)
                t -= (t @ y[i]) * y[i]
                t /= (np.linalg.norm(t) + 1e-12)
                ang = float(rng.uniform(0.05, 0.30))
                y[i] = np.cos(ang) * y[i] + np.sin(ang) * t
            y = _proj(y)
            y = _quench(y, steps=int(rng.integers(1500, 2500)), lr0=float(rng.uniform(0.08, 0.28)),
                        T0=float(rng.uniform(0.01, 0.025)), T1=0.0012, mu=0.82)
            y = _squeeze(y, steps=250, lr0=0.015)
        v = _mc(y)
        if v < bv - 1e-12:
            best, bv = y, v
        # opportunistic: also try short squeeze of best occasionally
        if it % 25 == 0 and left() > 20:
            ys = _squeeze(best.copy(), steps=600, lr0=0.01)
            vs = _mc(ys)
            if vs < bv:
                best, bv = ys, vs

    # final intensive squeeze
    if left() > 1.5:
        for _ in range(3):
            ys = _squeeze(best.copy(), steps=800, lr0=0.008)
            vs = _mc(ys)
            if vs < bv:
                best, bv = ys, vs
            else:
                break
        # final quench at low T to settle
        if left() > 3:
            ys = _quench(best.copy(), steps=2000, lr0=0.06, T0=0.005, T1=0.0005, mu=0.85)
            ys = _squeeze(ys, steps=500, lr0=0.008)
            vs = _mc(ys)
            if vs < bv:
                best, bv = ys, vs
    best = _proj(best)
    return best
