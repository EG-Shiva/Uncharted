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
    HARD_END = t0 + 172.0
    POLISH_END = t0 + 140.0
    n, d = 73, 13
    # load best.npy only (fallback to record.npy)
    X = None
    try:
        v = np.asarray(np.load("best.npy"), dtype=np.float64)
        if v.shape == (n, d):
            v = v / np.linalg.norm(v, axis=1, keepdims=True)
            if np.all(np.isfinite(v)):
                X = v.copy()
    except Exception:
        X = None
    if X is None:
        try:
            v = np.asarray(np.load("record.npy"), dtype=np.float64)
            if v.shape == (n, d):
                v = v / np.linalg.norm(v, axis=1, keepdims=True)
                if np.all(np.isfinite(v)):
                    X = v.copy()
        except Exception:
            pass
    if X is None:
        rng = np.random.RandomState(0)
        X = rng.randn(n, d)
        X /= np.linalg.norm(X, axis=1, keepdims=True)
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    best = _max_cos(X)
    Xbest = X.copy()
    best_val = best

    iu0, iu1 = np.triu_indices(n, 1)
    K = len(iu0)

    radius = 3.0e-5
    RMIN = 1.0e-8
    BAND = 5.0e-6
    CAP = 2500

    consec_fail = 0
    consec_stat = 0
    it = 0

    # pre-allocate for A_eq row/col (rows depend on X values only)
    eq_rows = np.repeat(np.arange(n), d)
    eq_cols = np.arange(n * d)

    while time.time() < POLISH_END and time.time() < HARD_END - 5.0:
        it += 1
        # time guard for LP solve (need ~2s margin)
        if time.time() > POLISH_END - 0.5:
            break
        if radius < RMIN:
            radius = RMIN
        # Gram and slacks
        C = X @ X.T
        cfull = C[iu0, iu1]
        m = float(np.max(cfull))
        # check equiripple stopping: top 200 slacks < 1e-10
        # use partition for speed every few iters
        if it % 5 == 1:
            # full sort check occasionally
            part = np.partition(cfull, K - 200)
            # largest 200 are at end; 200th tightest = min of those
            thr200 = float(np.min(part[K-200:]))
            slack200 = m - thr200
            if slack200 < 1.0e-10:
                # verify with exact sort to be safe
                s = np.sort(cfull)
                if m - float(s[K-200]) < 1.0e-10:
                    break
        # active set: within BAND of max, cap 2500 tightest
        # slacks = m - c
        # sel = c > m - BAND
        thresh = m - BAND
        # count
        # fast mask
        sel = cfull > thresh
        cnt = int(np.count_nonzero(sel))
        if cnt == 0:
            # should not happen (max itself), shrink & continue
            radius *= 0.5
            consec_fail += 1
            if consec_fail > 20 and radius <= RMIN + 0:
                break
            continue
        if cnt > CAP:
            # take CAP tightest by argpartition on -cfull
            # argpartition smallest slack = largest c
            idx = np.where(sel)[0]
            # c values for these
            cv = cfull[idx]
            # top CAP
            o = np.argpartition(-cv, CAP-1)[:CAP]
            take = idx[o]
            mask = np.zeros(K, dtype=bool)
            mask[take] = True
            sel = mask
            cnt = CAP
        I = iu0[sel]
        J = iu1[sel]
        p = cnt
        csel = cfull[sel]

        nv = n * d + 1
        # Build A_ub sparse: rows repeat, cols [I*d+.., J*d+.., t]
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cI = I[:, None] * d + np.arange(d)
        cJ = J[:, None] * d + np.arange(d)
        ct = np.full((p, 1), n * d)
        cols = np.concatenate([cI, cJ, ct], axis=1).ravel()
        vals = np.concatenate([X[J], X[I], -np.ones((p, 1))], axis=1).ravel()
        b_ub = -csel
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((X.ravel(), (eq_rows, eq_cols)), shape=(n, nv)).tocsr()
            obj = np.zeros(nv)
            obj[-1] = 1.0
            bounds = [(-radius, radius)] * (n * d) + [(None, None)]
            res = linprog(obj, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=bounds, method="highs")
        except Exception:
            radius *= 0.5
            consec_fail += 1
            continue
        if res.status != 0 or res.x is None:
            radius *= 0.5
            consec_fail += 1
            if consec_fail > 30 and radius <= RMIN * 1.01:
                break
            continue
        # track dual multipliers (for diagnosis, no action needed)
        try:
            _duals = res.ineqlin.marginals if hasattr(res, "ineqlin") else None
        except Exception:
            _duals = None
        t_pred = float(res.x[-1])
        delta = res.x[:-1].reshape(n, d)
        # check predicted improvement
        if not np.all(np.isfinite(delta)) or not np.isfinite(t_pred):
            radius *= 0.5
            consec_fail += 1
            continue
        if t_pred >= m - 1e-12:
            # stationary for this linear model
            consec_stat += 1
            # shrink to try higher precision, but if persistent, stop
            radius *= 0.5
            if consec_stat >= 5 and radius <= 2e-7:
                # check if already equiripple; otherwise stop (jammed)
                # do exact slack check
                srt = np.sort(cfull)
                if m - float(srt[K-200]) < 5e-10:
                    break
                # if not collapsed but LP says optimal, further shrinking won't help
                # break to return best
                if consec_stat >= 10:
                    break
            consec_fail += 1
            continue
        consec_stat = 0
        # backtracking line search on true max after retraction
        # alphas
        alphas = (1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125)
        accepted = False
        best_alpha_val = None
        best_alpha_Y = None
        for alpha in alphas:
            if time.time() > HARD_END - 2.0:
                break
            Y = X + alpha * delta
            nr = np.linalg.norm(Y, axis=1, keepdims=True)
            nr[nr == 0] = 1.0
            Y /= nr
            if not np.all(np.isfinite(Y)):
                continue
            vv = _max_cos(Y)
            if vv < best_val:
                best_alpha_val = vv
                best_alpha_Y = Y
                accepted = True
                break  # take largest alpha that improves (exact decrease)
        if accepted:
            X = best_alpha_Y
            best_val = float(best_alpha_val)
            if best_val < best - 1e-15:
                Xbest = X.copy()
                best = best_val
            else:
                # still update X for progress even if tiny
                if best_val < best:
                    Xbest = X.copy()
                    best = best_val
            # scheduled decay 3e-5 -> 1e-8 : geometric ~0.96 per success
            radius = max(radius * 0.96, RMIN)
            consec_fail = 0
        else:
            # no alpha improved -> shrink trust radius
            radius *= 0.5
            consec_fail += 1
            if radius < RMIN:
                radius = RMIN
                if consec_fail > 25:
                    break
        # safety: if radius at floor and no progress for long, stop
        if it > 5000:
            break

    # return best exact iterate
    Xbest = Xbest / np.linalg.norm(Xbest, axis=1, keepdims=True)
    return Xbest
