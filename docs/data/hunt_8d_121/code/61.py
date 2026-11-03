import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def solve():
    t0 = time.time()
    DEADLINE = t0 + 168.0
    n, d = 121, 8
    THR_HINGE = 0.460
    THR_DEL = 0.44

    def _normed(a):
        a = np.asarray(a, dtype=float)
        nr = np.linalg.norm(a, axis=1, keepdims=True)
        nr = np.maximum(nr, 1e-12)
        return a / nr

    # load incumbent
    cands = []
    B0 = None
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn).astype(float)
            if a.shape == (n, d):
                a = _normed(a)
                if np.all(np.isfinite(a)):
                    cands.append(a)
        except Exception:
            pass
    if not cands:
        rng0 = np.random.RandomState(0)
        cands = [_normed(rng0.randn(n, d))]
    vals0 = [_max_cos(a) for a in cands]
    Xbest = cands[int(np.argmin(vals0))].copy()
    Vbest = float(min(vals0))
    try:
        B0 = _normed(np.load("best.npy").astype(float))
        if B0.shape != (n, d):
            B0 = Xbest.copy()
    except Exception:
        B0 = Xbest.copy()

    # ---------- Adam optimizers on sphere ----------
    def adam_hinge(Yinit, t, steps, lr):
        Y = _normed(Yinit.copy())
        nn = Y.shape[0]
        iu = np.triu_indices(nn, 1)
        Ia, Ja = iu[0], iu[1]
        m = np.zeros_like(Y)
        v = np.zeros_like(Y)
        b1, b2, eps = 0.9, 0.999, 1e-8
        tt = float(t)
        for it in range(1, steps + 1):
            if time.time() > DEADLINE - 35:
                break
            G = Y @ Y.T
            c = G[Ia, Ja]
            diff = c - tt
            pos = diff > 0
            if np.any(pos):
                w = np.zeros_like(c)
                w[pos] = 2.0 * diff[pos]
                W = np.zeros((nn, nn))
                W[Ia, Ja] = w
                W[Ja, Ia] = w
                GradY = W @ Y
            else:
                GradY = np.zeros_like(Y)
                # feasible: nothing to do, early exit
                break
            Gt = GradY - (np.sum(GradY * Y, axis=1, keepdims=True)) * Y
            # clip extreme grads
            gn = np.linalg.norm(Gt) + 1e-12
            if gn > 1e4:
                Gt *= (1e4 / gn)
            m = b1 * m + (1 - b1) * Gt
            v = b2 * v + (1 - b2) * (Gt * Gt)
            mhat = m / (1 - b1 ** it)
            vhat = v / (1 - b2 ** it)
            step = lr * mhat / (np.sqrt(vhat) + eps)
            # limit step norm
            sn = np.linalg.norm(step) + 1e-12
            maxsn = 0.1 * np.sqrt(float(nn))
            if sn > maxsn:
                step *= (maxsn / sn)
            Y = Y - step
            Y = _normed(Y)
        return Y

    def adam_lse(Yinit, beta, steps, lr):
        Y = _normed(Yinit.copy())
        nn = Y.shape[0]
        m = np.zeros_like(Y)
        v = np.zeros_like(Y)
        b1, b2, eps = 0.9, 0.999, 1e-8
        bb = float(beta)
        for it in range(1, steps + 1):
            if time.time() > DEADLINE - 35:
                break
            G = Y @ Y.T
            # max over off-diag
            np.fill_diagonal(G, -2.0)
            gm = float(np.max(G))
            A = bb * (G - gm)
            np.clip(A, -700, 0, out=A)
            E = np.exp(A)
            np.fill_diagonal(E, 0.0)
            S = float(np.sum(np.triu(E, 1)))
            if not np.isfinite(S) or S <= 0:
                break
            EY = E @ Y
            GradY = EY / S
            Gt = GradY - (np.sum(GradY * Y, axis=1, keepdims=True)) * Y
            gn = np.linalg.norm(Gt) + 1e-12
            if gn > 1e4:
                Gt *= (1e4 / gn)
            m = b1 * m + (1 - b1) * Gt
            v = b2 * v + (1 - b2) * (Gt * Gt)
            mhat = m / (1 - b1 ** it)
            vhat = v / (1 - b2 ** it)
            step = lr * mhat / (np.sqrt(vhat) + eps)
            sn = np.linalg.norm(step) + 1e-12
            maxsn = 0.1 * np.sqrt(float(nn))
            if sn > maxsn:
                step *= (maxsn / sn)
            Y = Y - step
            Y = _normed(Y)
        return Y

    def find_hole(base, rng, nsample=20000, refine_steps=80):
        C = rng.randn(nsample, d)
        C /= np.maximum(np.linalg.norm(C, axis=1, keepdims=True), 1e-12)
        D = C @ base.T
        cm = D.max(axis=1)
        k = int(np.argmin(cm))
        bc = C[k].copy()
        cur = float(cm[k])
        for st in range(refine_steps):
            sig = 0.03 * (1.0 - st / float(refine_steps)) + 0.001
            prop = bc + sig * rng.randn(d)
            pn = np.linalg.norm(prop)
            if pn < 1e-12:
                continue
            prop /= pn
            md = float((prop @ base.T).max())
            if md < cur:
                bc = prop
                cur = md
        return bc

    def lp_polish_timed(x_init, seconds):
        x = _normed(x_init.copy())
        cur = _max_cos(x)
        radius = 1e-3
        start = time.time()
        nv = n * d + 1
        Iall = np.triu_indices(n, 1)[0]
        Jall = np.triu_indices(n, 1)[1]
        while (time.time() - start < seconds and time.time() < DEADLINE - 1.5
               and radius > 1e-13):
            G = x @ x.T
            c_all = G[Iall, Jall]
            m = float(c_all.max())
            win = max(20.0 * radius, 2e-9)
            sel = c_all > m - win
            I = Iall[sel]; J = Jall[sel]; c = c_all[sel]
            p = len(I)
            if p == 0:
                break
            if p > 2500:
                idx = np.argpartition(-c, 2500)[:2500]
                I = I[idx]; J = J[idx]; c = c[idx]
                p = 2500
            rows = np.repeat(np.arange(p), 2 * d + 1)
            cols = np.concatenate([I[:, None] * d + np.arange(d),
                                   J[:, None] * d + np.arange(d),
                                   np.full((p, 1), n * d)], axis=1).ravel()
            vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], axis=1).ravel()
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(),
                               (np.repeat(np.arange(n), d), np.arange(n * d))),
                              shape=(n, nv)).tocsr()
            obj = np.zeros(nv); obj[-1] = 1.0
            try:
                res = linprog(obj, A_ub=A_ub, b_ub=-c, A_eq=A_eq,
                              b_eq=np.zeros(n),
                              bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                              method="highs")
            except Exception:
                radius *= 0.5
                continue
            if res.status != 0:
                radius *= 0.5
                continue
            y = _normed(x + res.x[:-1].reshape(n, d))
            vv = _max_cos(y)
            if vv < cur:
                x, cur = y, vv
                radius = min(radius * 1.5, 0.05)
            else:
                radius *= 0.5
        return x

    # ---------- overpopulate + deflate, 3 seeds ----------
    extras = [6, 7, 5]
    endpoints = []
    for si in range(3):
        if time.time() > DEADLINE - 40:
            break
        rng = np.random.RandomState(1000 + si * 7919)
        kex = extras[si % len(extras)]
        cur = B0.copy()
        # augment with kex hole points greedily
        for e in range(kex):
            try:
                bc = find_hole(cur, rng, nsample=18000, refine_steps=70)
            except Exception:
                bc = rng.randn(d); bc /= np.linalg.norm(bc)
            cur = np.vstack([cur, bc[None, :]])
            cur = _normed(cur)
        # wholesale rearrangement: hinge at 0.460
        cur = adam_hinge(cur, THR_HINGE, steps=700, lr=0.02)
        # LogSumExp annealing
        cur = adam_lse(cur, 80.0, steps=250, lr=0.02)
        cur = adam_lse(cur, 220.0, steps=250, lr=0.014)
        cur = adam_lse(cur, 550.0, steps=250, lr=0.010)
        cur = adam_hinge(cur, THR_HINGE, steps=200, lr=0.01)
        # deflate one at a time to 121
        while cur.shape[0] > n:
            if time.time() > DEADLINE - 38:
                break
            G = cur @ cur.T
            np.fill_diagonal(G, -2.0)
            P = np.maximum(0.0, G - THR_DEL)
            P = P * P
            scores = P.sum(axis=1)
            idx = int(np.argmax(scores))
            # delete worst
            mask = np.ones(cur.shape[0], dtype=bool)
            mask[idx] = False
            cur = cur[mask].copy()
            # cheap re-optimise 150 Adam steps (hinge)
            cur = adam_hinge(cur, THR_HINGE, steps=150, lr=0.015)
            if time.time() > DEADLINE - 38:
                break
        if cur.shape[0] != n:
            continue
        # final LSE snap to actually minimize max
        cur = adam_lse(cur, 300.0, steps=300, lr=0.012)
        cur = adam_lse(cur, 800.0, steps=300, lr=0.008)
        try:
            v = _max_cos(cur)
        except Exception:
            continue
        if v < 0.462:
            endpoints.append((float(v), cur.copy()))

    # pick best endpoint
    best_end = None
    best_end_v = 1e9
    for v, X in endpoints:
        if v < best_end_v:
            best_end_v = v
            best_end = X.copy()

    # LP polish single best endpoint at most 30 s
    if best_end is not None:
        # also consider incumbent
        if best_end_v < Vbest or True:
            try:
                t_lp = min(30.0, max(5.0, DEADLINE - time.time() - 2.0))
                if t_lp > 3:
                    best_end = lp_polish_timed(best_end, t_lp)
                    best_end_v = _max_cos(best_end)
            except Exception:
                pass
            if best_end_v < Vbest and np.all(np.isfinite(best_end)):
                Xbest = _normed(best_end.copy())
                Vbest = float(best_end_v)

    # safety: return incumbent if no improvement
    Xbest = _normed(Xbest)
    # tiny shrink not needed; ensure valid
    return Xbest
