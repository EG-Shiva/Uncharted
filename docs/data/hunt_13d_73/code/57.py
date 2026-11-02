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
    BUDGET = 150.0  # total for polish loop, leave margin for 180s limit
    n, d = 73, 13
    # load best.npy only (record as fallback)
    Xcur = None
    best_file = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            v = np.load(fn)
            v = np.asarray(v, dtype=np.float64)
            if v.shape == (n, d):
                vn = np.linalg.norm(v, axis=1, keepdims=True)
                vn[vn == 0] = 1.0
                v = v / vn
                if np.all(np.isfinite(v)):
                    c = _max_cos(v)
                    if c < best_file:
                        best_file = c
                        Xcur = v.copy()
        except Exception:
            pass
    if Xcur is None:
        rng = np.random.default_rng(0)
        Xcur = rng.standard_normal(size=(n, d))
        Xcur /= np.linalg.norm(Xcur, axis=1, keepdims=True)
        best_file = _max_cos(Xcur)
    Xcur = Xcur / np.linalg.norm(Xcur, axis=1, keepdims=True)
    best = float(best_file)
    Xbest = Xcur.copy()

    iu0, iu1 = np.triu_indices(n, 1)
    K = len(iu0)
    N = n * d

    radius = 3.0e-5
    R0 = 3.0e-5
    R1 = 1.0e-8
    # envelope decay factor
    log_ratio = np.log(R1 / R0)  # negative

    it = 0
    # preallocate arange for cols
    ar_d = np.arange(d)
    rep_n = np.repeat(np.arange(n), d)

    while True:
        elapsed = time.time() - t0
        if elapsed > BUDGET:
            break
        # envelope schedule
        envelope = R0 * np.exp(log_ratio * (elapsed / BUDGET))
        # don't let radius exceed envelope by much
        if radius > envelope * 1.5:
            radius = envelope * 1.5
        if radius < R1:
            radius = R1
        # current max and slacks
        G = Xcur @ Xcur.T
        cfull = G[iu0, iu1]
        m = float(cfull.max())
        # check equiripple condition: 200th smallest slack
        # slacks = m - c
        # partial sort for 200th
        if K >= 250:
            part = np.partition(cfull, K - 201)
            # largest 200 values
            top200 = part[K - 200:]
            slack200 = m - float(top200.min())
            if slack200 < 1e-10 and elapsed > 5:
                # achieved many-contact tie; keep pushing with tiny radius
                # drop radius to minimum to squeeze further
                pass
        # select active set within 5e-6
        thresh = 5.0e-6
        sel = cfull > (m - thresh)
        nsel = int(np.count_nonzero(sel))
        if nsel == 0:
            # should not happen
            sel = np.zeros(K, dtype=bool)
            sel[int(np.argmax(cfull))] = True
            nsel = 1
        if nsel > 2500:
            # cap by tightest
            idx = np.where(sel)[0]
            order = np.argsort(-cfull[idx])[:2500]
            mask = np.zeros(K, dtype=bool)
            mask[idx[order]] = True
            sel = mask
            nsel = 2500
        I = iu0[sel]
        J = iu1[sel]
        csel = cfull[sel]
        p = nsel

        # build LP: min t s.t. Xj.di + Xi.dj - t <= -csel, Xi.di=0, |d|<=radius
        # sparse A_ub (p x N+1)
        # each row has 2d+1 nonzeros
        rows = np.repeat(np.arange(p, dtype=np.int64), 2 * d + 1)
        cI = (I[:, None] * d + ar_d)  # p x d
        cJ = (J[:, None] * d + ar_d)
        ct = np.full((p, 1), N, dtype=np.int64)
        cols = np.concatenate([cI, cJ, ct], axis=1).ravel()
        vals = np.concatenate([Xcur[J], Xcur[I], -np.ones((p, 1))], axis=1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, N + 1)).tocsr()
            # A_eq n x N+1
            A_eq = coo_matrix((Xcur.ravel(), (rep_n, np.arange(N, dtype=np.int64))), shape=(n, N + 1)).tocsr()
            obj = np.zeros(N + 1)
            obj[-1] = 1.0
            bounds = [(-radius, radius)] * N + [(None, None)]
            res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=bounds, method="highs",
                          options={"presolve": True, "time_limit": 8.0})
        except Exception:
            radius *= 0.5
            it += 1
            continue
        if res.status != 0 or res.x is None:
            radius *= 0.5
            it += 1
            # avoid infinite shrink loop timeout
            if radius < 1e-12:
                radius = R1
            continue
        delta = res.x[:-1].reshape(n, d)
        t_pred = float(res.x[-1])
        # clamp crazy deltas (should already be in box)
        # line search
        alphas = (1.0, 0.7, 0.5, 0.3, 0.2, 0.1, 0.05)
        best_alpha_val = best
        best_alpha_X = None
        for a in alphas:
            Y = Xcur + a * delta
            nr = np.linalg.norm(Y, axis=1, keepdims=True)
            nr[nr == 0] = 1.0
            Y /= nr
            # exact max
            G2 = Y @ Y.T
            np.fill_diagonal(G2, -2.0)
            vv = float(G2.max())
            if vv < best_alpha_val:
                best_alpha_val = vv
                best_alpha_X = Y
                # don't break: look for minimal over alphas? take first improving
                # full step usually best; break early for speed but check full first
                break
        # if no improving alpha with coarse list, try full evaluation to find min anyway
        if best_alpha_X is None:
            # evaluate all to see closest (for diagnostics) but no accept
            # shrink trust region
            radius *= 0.5
            # if predicted improvement was nil and radius tiny, add envelope decay
            it += 1
            continue
        else:
            # accept exact decrease
            Xcur = best_alpha_X
            best = best_alpha_val
            Xbest = Xcur.copy()
            # on success keep radius (envelope will decay it); slight growth allowed but capped by envelope
            # allow small growth to accelerate if large improvement
            # improvement = m - best ; if large keep radius
            radius = min(radius * 1.05, 5e-5)
            it += 1
            # early exit if top-200 slacks < 1e-10 and no further predicted gain?
            # continue until budget to push lower
            continue

    Xbest = Xbest / np.linalg.norm(Xbest, axis=1, keepdims=True)
    return Xbest
