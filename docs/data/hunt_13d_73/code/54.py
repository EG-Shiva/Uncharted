import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def _lp_polish(X, seconds, hard_end):
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    n, d = X.shape
    best = _max_cos(X)
    radius = 5e-4
    start = time.time()
    iu0, iu1 = np.triu_indices(n, 1)
    while time.time() - start < seconds and time.time() < hard_end and radius > 1e-14:
        C = X @ X.T
        c = C[iu0, iu1]
        m = float(c.max())
        sel = c > m - max(20.0 * radius, 2e-7)
        if np.count_nonzero(sel) > 2000:
            idx = np.where(sel)[0]
            order = np.argsort(-c[idx])[:2000]
            mask = np.zeros_like(sel)
            mask[idx[order]] = True
            sel = mask
        I, J = iu0[sel], iu1[sel]
        p = len(I)
        if p == 0:
            radius *= 0.5
            continue
        csel = c[sel]
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cI = I[:, None] * d + np.arange(d)
        cJ = J[:, None] * d + np.arange(d)
        ct = np.full((p, 1), n * d)
        cols = np.concatenate([cI, cJ, ct], axis=1).ravel()
        vals = np.concatenate([X[J], X[I], -np.ones((p, 1))], axis=1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((X.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
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
        y = X + res.x[:-1].reshape(n, d)
        nrm = np.linalg.norm(y, axis=1, keepdims=True)
        nrm[nrm == 0] = 1.0
        y /= nrm
        if not np.all(np.isfinite(y)):
            radius *= 0.5
            continue
        vv = _max_cos(y)
        if vv < best:
            X, best = y, vv
            radius = min(radius * 1.5, 0.02)
        else:
            radius *= 0.5
    return X, best

def solve():
    t0 = time.time()
    HARD = t0 + 175.0
    EXPLORE_END = t0 + 143.0
    rng = np.random.RandomState(0)
    n, d = 73, 13
    # load record and best
    X_record = None
    X_bestf = None
    v_record = 1e9
    v_bestf = 1e9
    try:
        v = np.asarray(np.load("record.npy"), dtype=np.float64)
        if v.shape == (n, d):
            v = v / np.linalg.norm(v, axis=1, keepdims=True)
            if np.all(np.isfinite(v)):
                X_record = v.copy()
                v_record = _max_cos(v)
    except Exception:
        pass
    try:
        v = np.asarray(np.load("best.npy"), dtype=np.float64)
        if v.shape == (n, d):
            v = v / np.linalg.norm(v, axis=1, keepdims=True)
            if np.all(np.isfinite(v)):
                X_bestf = v.copy()
                v_bestf = _max_cos(v)
    except Exception:
        pass
    if X_record is None and X_bestf is None:
        X_record = rng.randn(n, d)
        X_record /= np.linalg.norm(X_record, axis=1, keepdims=True)
        v_record = _max_cos(X_record)
    if X_record is None:
        X_record = X_bestf.copy()
        v_record = v_bestf
    if X_bestf is None:
        X_bestf = X_record.copy()
        v_bestf = v_record
    # global best starts from better of the two
    if v_bestf <= v_record:
        Xbest = X_bestf.copy(); best = float(v_bestf)
    else:
        Xbest = X_record.copy(); best = float(v_record)
    Xrec0 = X_record.copy()

    targets = [0.2068, 0.2060, 0.2050, 0.2035, 0.2020]
    kicks = [0.0, 0.05, 0.08, 0.10, 0.12, 0.15, 0.06, 0.09, 0.13, 0.07, 0.11, 0.14]

    trial = 0
    # exploration loop
    while time.time() < EXPLORE_END and time.time() < HARD - 25:
        kick = kicks[trial % len(kicks)]
        # after one cycle, randomize kick in 0.05-0.15
        if trial >= len(kicks):
            kick = 0.05 + 0.10 * rng.rand()
        # base: alternate record and current best
        if trial % 2 == 0:
            base = Xrec0
        else:
            base = Xbest
        if kick == 0.0:
            X = base.copy()
        else:
            pert = rng.randn(n, d)
            X = base + kick * pert
            nr = np.linalg.norm(X, axis=1, keepdims=True)
            nr[nr == 0] = 1.0
            X /= nr
        # Adam hyper for this trial
        steps = 2200
        # shrink steps if little time left
        tleft = EXPLORE_END - time.time()
        if tleft < 8:
            steps = max(400, int(tleft * 250))
        lr0 = 0.012 * (0.7 + 0.6 * rng.rand())
        lr1 = 0.002
        # tangent Adam state
        m = np.zeros((n, d))
        vv = np.zeros((n, d))
        b1, b2, eps = 0.9, 0.999, 1e-8
        b1p, b2p = b1, b2
        l0, l1 = np.log(lr0), np.log(lr1)
        ti = 0
        tcur = targets[0]
        for k in range(steps):
            if time.time() > EXPLORE_END:
                break
            frac = k / (steps - 1) if steps > 1 else 1.0
            lr = float(np.exp(l0 + (l1 - l0) * frac))
            G = X @ X.T
            # exact max (diag excluded)
            # use copy for max to avoid destroying G needed for grad? compute max first
            # set diag temporarily
            # faster: compute max of off-diag via G with diag -2
            # we already have G; compute mval
            # to avoid extra alloc, fill diag after grad? do max via masked
            # simple:
            mval = float(np.max(G - 2.0 * np.eye(n)))
            # hinge matrix
            # V = max(G - tcur, 0), zero diag
            # in-place to save allocs
            V = G - tcur
            # threshold: keep positive only
            # np.maximum with out would need extra; just do:
            np.maximum(V, 0.0, out=V)
            # zero diag (diag of G is 1, so V diag = 1-tcur >0, must zero)
            np.fill_diagonal(V, 0.0)
            # gradient Euclidean = 2 V @ X
            Grad = (2.0 * V) @ X
            # tangent projection
            sprod = np.sum(Grad * X, axis=1, keepdims=True)
            Grad -= sprod * X
            # Adam
            m = b1 * m + (1 - b1) * Grad
            vv = b2 * vv + (1 - b2) * (Grad * Grad)
            mhat = m / (1.0 - b1p)
            vhat = vv / (1.0 - b2p)
            b1p *= b1
            b2p *= b2
            step = mhat / (np.sqrt(vhat) + eps)
            X = X - lr * step
            nr = np.linalg.norm(X, axis=1, keepdims=True)
            nr[nr == 0] = 1.0
            X /= nr
            if not np.all(np.isfinite(X)):
                break
            # target lowering check every 50 steps (and at end)
            if (k % 50 == 49) or (k == steps - 1):
                # recompute exact max cheaply (we have mval from next iter? use current)
                # mval above is pre-update; compute fresh every check
                cur_max = _max_cos(X)
                # screen: global best update using exact Gram
                if cur_max < best:
                    best = float(cur_max)
                    Xbest = X.copy()
                if cur_max < tcur and ti + 1 < len(targets):
                    ti += 1
                    tcur = targets[ti]
                    # reset momentum partially to allow new direction
                    m *= 0.5
                    vv *= 0.5
            else:
                # cheap screen every 200 steps already covered; also quick best check using mval? skip
                pass
        trial += 1
        if trial > 400:
            break
        if time.time() > EXPLORE_END:
            break

    # gated LP polish only if raw max below published record
    RECORD = 0.207434422294
    # small shrink not needed; LP keeps feasibility
    if best < 0.2074344:
        remain = HARD - time.time() - 2.0
        remain = min(max(remain, 0.0), 28.0)
        if remain > 3.0:
            try:
                Xp, vp = _lp_polish(Xbest, seconds=remain, hard_end=HARD - 1.0)
                if vp < best and np.all(np.isfinite(Xp)):
                    Xbest, best = Xp, vp
            except Exception:
                pass
    nr = np.linalg.norm(Xbest, axis=1, keepdims=True)
    nr[nr == 0] = 1.0
    Xbest = Xbest / nr
    return Xbest
