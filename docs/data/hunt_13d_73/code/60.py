import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def solve():
    t0 = time.time()
    HARD_END = t0 + 168.0
    SEARCH_END = t0 + 148.0
    rng = np.random.default_rng(0)
    n, d = 73, 13
    iu0, iu1 = np.triu_indices(n, 1)

    Xbest = None
    best = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            v = np.load(fn)
            v = np.asarray(v, dtype=np.float64)
            if v.shape == (n, d):
                v = v / np.linalg.norm(v, axis=1, keepdims=True)
                if np.all(np.isfinite(v)):
                    c = _max_cos(v)
                    if c < best:
                        best = c
                        Xbest = v.copy()
        except Exception:
            pass
    if Xbest is None:
        Xbest = rng.standard_normal(size=(n, d))
        Xbest /= np.linalg.norm(Xbest, axis=1, keepdims=True)
        best = _max_cos(Xbest)

    def adam_lse(X0, s0, s1, steps, lr0, lr1):
        X = X0.copy()
        m = np.zeros((n, d))
        v = np.zeros((n, d))
        b1, b2, eps = 0.9, 0.999, 1e-8
        b1p, b2p = b1, b2
        l0, l1 = np.log(lr0), np.log(lr1)
        WX = np.empty((n, d))
        for k in range(steps):
            if (k & 127) == 0 and time.time() > SEARCH_END:
                break
            frac = k / (steps - 1) if steps > 1 else 1.0
            s = s0 + (s1 - s0) * frac
            lr = np.exp(l0 + (l1 - l0) * frac)
            G = X @ X.T
            c = G[iu0, iu1]
            mx = float(c.max())
            e = np.exp(s * (c - mx))
            se = float(e.sum())
            if not np.isfinite(se) or se <= 0:
                w = np.zeros_like(c)
                w[int(np.argmax(c))] = 1.0
            else:
                w = e / se
            WX.fill(0.0)
            np.add.at(WX, iu0, w[:, None] * X[iu1])
            np.add.at(WX, iu1, w[:, None] * X[iu0])
            sp = np.sum(WX * X, axis=1)
            Gt = WX - sp[:, None] * X
            m *= b1
            m += (1 - b1) * Gt
            v *= b2
            v += (1 - b2) * (Gt * Gt)
            mhat = m / (1 - b1p)
            vhat = v / (1 - b2p)
            b1p *= b1
            b2p *= b2
            step = mhat / (np.sqrt(vhat) + eps)
            X = X - lr * step
            nr = np.linalg.norm(X, axis=1, keepdims=True)
            nr[nr == 0] = 1.0
            X /= nr
        return X

    def lp_polish(X, cur, t_deadline, max_iters=40, r0=2e-3, cap=1800):
        X = X / np.linalg.norm(X, axis=1, keepdims=True)
        radius = r0
        it = 0
        # precompute arange
        ar = np.arange(d)
        rep = np.repeat(np.arange(n), d)
        flatX_base = None
        while it < max_iters and time.time() < t_deadline and radius > 1e-12:
            it += 1
            Cfull = X @ X.T
            c = Cfull[iu0, iu1]
            m = float(c.max())
            thresh = m - max(12.0 * radius, 3e-7)
            sel = c > thresh
            cnt = int(np.count_nonzero(sel))
            if cnt == 0:
                radius *= 0.5
                continue
            if cnt > cap:
                idx = np.where(sel)[0]
                order = np.argsort(-c[idx])[:cap]
                mask = np.zeros_like(sel)
                mask[idx[order]] = True
                sel = mask
            I, J = iu0[sel], iu1[sel]
            p = len(I)
            csel = c[sel]
            nv = n * d + 1
            try:
                rows = np.repeat(np.arange(p), 2 * d + 1)
                cI = I[:, None] * d + ar
                cJ = J[:, None] * d + ar
                ct = np.full((p, 1), n * d)
                cols = np.concatenate([cI, cJ, ct], axis=1).ravel()
                vals = np.concatenate([X[J], X[I], -np.ones((p, 1))], axis=1).ravel()
                A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
                A_eq = coo_matrix((X.ravel(), (rep, np.arange(n * d))), shape=(n, nv)).tocsr()
                obj = np.zeros(nv)
                obj[-1] = 1.0
                res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq, b_eq=np.zeros(n),
                              bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
            except Exception:
                radius *= 0.5
                continue
            if res.status != 0:
                radius *= 0.5
                continue
            t_pred = float(res.x[-1])
            y = X + res.x[:-1].reshape(n, d)
            nr = np.linalg.norm(y, axis=1, keepdims=True)
            nr[nr == 0] = 1.0
            y /= nr
            vv = _max_cos(y)
            if vv < cur - 1e-12:
                X, cur = y, vv
                # expand slightly on success
                if t_pred < cur - 1e-12:
                    radius = min(radius * 1.4, 0.01)
                else:
                    radius = min(radius * 1.15, 0.01)
            else:
                # no improvement: check predicted reduction
                if t_pred >= m - 1e-12:
                    radius *= 0.5
                else:
                    radius *= 0.5
        return X, cur

    def largest_hole(X, exclude=None, n_cand=4000):
        C = rng.standard_normal(size=(n_cand, d))
        C /= np.linalg.norm(C, axis=1, keepdims=True)
        if exclude is not None:
            S = C @ X[exclude].T if len(exclude) > 0 else C @ X.T
        else:
            S = C @ X.T
        m = S.max(axis=1)
        return C[int(np.argmin(m))].copy()

    # 1) deep LP polish of incumbent to jammed point
    try:
        Xp, vp = lp_polish(Xbest, best, min(SEARCH_END - 60, t0 + 45), max_iters=60, r0=2e-3)
        if vp < best:
            best, Xbest = vp, Xp
        else:
            # keep jammed X even if equal (more contacts)
            # only keep if not worse by >1e-9
            if vp <= best + 1e-9:
                # check contacts count to decide
                pass
    except Exception:
        pass

    # quick anneal to help LP escape tiny stall
    if time.time() < SEARCH_END - 50:
        try:
            Xa = adam_lse(Xbest, 100.0, 800.0, 800, 0.01, 0.0015)
            va = _max_cos(Xa)
            Xap, vap = lp_polish(Xa, va, time.time() + 12, max_iters=25, r0=1.5e-3)
            if vap < best:
                best, Xbest = vap, Xap
            elif va < best:
                best, Xbest = va, Xa
        except Exception:
            pass

    # 2) basin hops with topology changes
    hop = 0
    sigmas = [0.002, 0.005, 0.01, 0.02, 0.03, 0.007]
    while time.time() < SEARCH_END and hop < 60:
        hop += 1
        mode = hop % 4
        X0 = Xbest.copy()
        try:
            if mode == 0:
                sigma = sigmas[(hop // 4) % len(sigmas)]
                if hop % 8 == 0:
                    sigma = float(10 ** rng.uniform(-2.7, -1.5))
                X0 = X0 + sigma * rng.standard_normal(size=(n, d))
                X0 /= np.linalg.norm(X0, axis=1, keepdims=True)
            elif mode == 1:
                # kill worst point -> hole
                G = X0 @ X0.T
                np.fill_diagonal(G, -2.0)
                rm = G.max(axis=1)
                wi = int(np.argmax(rm))
                # occasionally kill 2 worst
                kill = [wi]
                if hop % 3 == 0:
                    # second worst distinct
                    rm2 = rm.copy()
                    rm2[wi] = -2
                    kill.append(int(np.argmax(rm2)))
                keep = [i for i in range(n) if i not in kill]
                for k in kill:
                    h = largest_hole(X0[keep], n_cand=4000)
                    X0[k] = h
                    keep.append(k)
            elif mode == 2:
                # attack max pair: move one endpoint to hole
                G = X0 @ X0.T
                np.fill_diagonal(G, -2.0)
                flat = int(np.argmax(G[iu0, iu1]))
                a, b = int(iu0[flat]), int(iu1[flat])
                victim = a if rng.random() < 0.5 else b
                keep = [i for i in range(n) if i != victim]
                X0[victim] = largest_hole(X0[keep], n_cand=4000)
                # small shake of rest
                X0 = X0 + 0.001 * rng.standard_normal(size=(n, d))
                X0 /= np.linalg.norm(X0, axis=1, keepdims=True)
            else:
                # cluster shake: random subset stronger
                k = int(rng.integers(4, 14))
                idx = rng.choice(n, size=k, replace=False)
                sigma = float(10 ** rng.uniform(-2.3, -1.3))
                X0[idx] = X0[idx] + sigma * rng.standard_normal(size=(k, d))
                X0 /= np.linalg.norm(X0, axis=1, keepdims=True)

            # coarse repolish
            s0 = float(rng.choice([80.0, 120.0, 200.0]))
            s1 = float(rng.choice([600.0, 900.0]))
            steps = int(rng.choice([400, 600, 800]))
            lr0 = float(10 ** rng.uniform(-1.9, -1.5))
            lr1 = float(10 ** rng.uniform(-2.9, -2.5))
            Y = adam_lse(X0, s0, s1, steps, lr0, lr1)
            vy = _max_cos(Y)
            # short LP to jam
            budget = 7.0
            Yp, vyp = lp_polish(Y, vy, min(time.time() + budget, SEARCH_END), max_iters=18, r0=1.2e-3)
            if vyp < best - 1e-12:
                best, Xbest = vyp, Yp.copy()
                # extended squeeze on new best
                Yq, vq = lp_polish(Xbest, best, min(time.time() + 8, SEARCH_END), max_iters=25, r0=6e-4)
                if vq < best:
                    best, Xbest = vq, Yq
            elif vy < best - 1e-12:
                # adam alone improved but LP didn't; keep adam then extended LP
                Yq, vq = lp_polish(Y, vy, min(time.time() + 8, SEARCH_END), max_iters=25, r0=8e-4)
                if vq < best:
                    best, Xbest = vq, Yq
                else:
                    best, Xbest = vy, Y
        except Exception:
            continue

    # 3) final squeeze: adam + deep LP with tiny radius
    if time.time() < HARD_END - 15:
        try:
            Y = adam_lse(Xbest, 500.0, 1500.0, 600, 0.004, 0.0007)
            vy = _max_cos(Y)
            if vy < best:
                best, Xbest = vy, Y
        except Exception:
            pass
    if time.time() < HARD_END - 5:
        try:
            Xf, vf = lp_polish(Xbest, best, HARD_END - 2, max_iters=80, r0=5e-4, cap=2000)
            if vf < best:
                best, Xbest = vf, Xf
        except Exception:
            pass

    Xbest = Xbest / np.linalg.norm(Xbest, axis=1, keepdims=True)
    Xbest = np.where(np.isfinite(Xbest), Xbest, 1.0 / np.sqrt(13))
    nr = np.linalg.norm(Xbest, axis=1, keepdims=True)
    nr[nr == 0] = 1.0
    Xbest = Xbest / nr
    return Xbest
