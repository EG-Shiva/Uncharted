import time
import numpy as np
from scipy.optimize import linprog, lsq_linear
from scipy.sparse import csr_matrix, hstack

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _tangent_bases(X):
    n, d = X.shape
    k = d - 1
    T = np.empty((n, d, k), dtype=np.float64)
    for i in range(n):
        xi = X[i]
        kc = int(np.argmax(np.abs(xi)))
        den = xi[kc]
        if abs(den) < 1e-12:
            den = 1e-12 if den >= 0 else -1e-12
        M = np.zeros((d, k), dtype=np.float64)
        col = 0
        for j in range(d):
            if j == kc:
                continue
            M[j, col] = 1.0
            M[kc, col] = -xi[j] / den
            col += 1
        Q, _ = np.linalg.qr(M)
        T[i] = Q
    return T

def _build_Aact(T, X, I, J):
    q = len(I)
    k = T.shape[2]
    Nv = X.shape[0] * k
    Ti = T[I]
    Tj = T[J]
    Xi = X[I]
    Xj = X[J]
    Ai = np.einsum('qdk,qd->qk', Ti, Xj)
    Aj = np.einsum('qdk,qd->qk', Tj, Xi)
    A = np.zeros((q, Nv), dtype=np.float64)
    for r in range(q):
        A[r, int(I[r]) * k:int(I[r]) * k + k] = Ai[r]
        A[r, int(J[r]) * k:int(J[r]) * k + k] = Aj[r]
    return A, Ai, Aj

def _renorm(X):
    nr = np.linalg.norm(X, axis=1, keepdims=True)
    nr[~np.isfinite(nr)] = 1.0
    nr[nr < 1e-12] = 1.0
    return X / nr

def _try_steps(X, T, vflat, m_cur, alphas):
    n, k = X.shape[0], T.shape[2]
    dX = np.einsum('ndk,nk->nd', T, vflat.reshape(X.shape[0], k))
    best_v = m_cur
    best_X = None
    best_a = None
    for a in alphas:
        Y = _renorm(X + a * dX)
        if not np.all(np.isfinite(Y)):
            continue
        vv = _max_cos(Y)
        if vv < best_v - 1e-15:
            best_v = vv
            best_X = Y
            best_a = a
    return best_X, best_v, best_a

def _sqp_polish(X0, deadline, rng):
    X = X0.copy()
    n, d = X.shape
    k = d - 1
    Nv = n * k
    iu0, iu1 = np.triu_indices(n, 1)
    m_cur = _max_cos(X)
    best = m_cur
    best_X = X.copy()
    trust = 1e-3
    radii_floor = 3e-6
    it = 0
    alphas = (1.0, 0.5, 0.25, 0.125, 0.0625)
    while time.time() < deadline:
        it += 1
        if trust < radii_floor:
            trust = 1e-3
            # if we cycled without progress, do a tiny random tangent kick
            # to avoid infinite loop (keeps determinism via rng)
            if time.time() > deadline - 2.0:
                break
        # full cosine vector
        G = X @ X.T
        cfull = G[iu0, iu1]
        m = float(cfull.max())
        slack = m - cfull
        # active set: slack below max(1e-6, 10*trial step scale)
        thresh = max(1e-6, 10.0 * trust)
        # cap size for LP tractability
        cap = 1500
        if np.count_nonzero(slack < thresh) > cap:
            # take cap smallest slacks
            part = np.argpartition(slack, cap - 1)[:cap]
            # ensure the true max pair included
            order = part[np.argsort(slack[part])]
        else:
            order = np.where(slack < thresh)[0]
            order = order[np.argsort(slack[order])]
        q = len(order)
        if q == 0:
            trust *= 0.5
            continue
        # keep at least the tightest 60 even if thresh tiny
        if q < 60:
            extra = np.argpartition(slack, min(60, len(slack)) - 1)[:min(60, len(slack))]
            order = np.unique(np.concatenate([order, extra]))
            order = order[np.argsort(slack[order])][:1500]
            q = len(order)
        I = iu0[order]
        J = iu1[order]
        cact = cfull[order]
        T = _tangent_bases(X)
        Aact, _, _ = _build_Aact(T, X, I, J)
        # rank test on leading block (for diagnostics / null moves)
        try:
            qq = min(200, q)
            s = np.linalg.svd(Aact[:qq], compute_uv=False)
            smax = float(s[0]) if len(s) else 1.0
            smin = float(s[-1]) if len(s) else 1.0
            rank_def = (smin / max(1e-30, smax)) < 1e-8
        except Exception:
            rank_def = True
            smax, smin = 1.0, 0.0
        # --- LP minimax subproblem ---
        improved = False
        try:
            A_ub = hstack([csr_matrix(Aact), -np.ones((q, 1))], format='csr')
            b_ub = -cact
            c_obj = np.zeros(Nv + 1)
            c_obj[-1] = 1.0
            bounds = [(-trust, trust)] * Nv + [(None, None)]
            res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs',
                          options={'time_limit': 5.0})
            ok = (res.status == 0) and res.success is not False and np.all(np.isfinite(res.x))
        except Exception:
            ok = False
            res = None
        if ok:
            vflat = res.x[:-1]
            if not np.all(np.isfinite(vflat)):
                ok = False
            else:
                # clip to trust (numerical)
                vflat = np.clip(vflat, -trust, trust)
                bX, bV, ba = _try_steps(X, T, vflat, m_cur, alphas)
                if bX is not None:
                    X = bX
                    m_cur = bV
                    if bV < best - 1e-15:
                        best = bV
                        best_X = X.copy()
                    # second-order correction is exact renormalization above;
                    # expand trust slightly
                    trust = min(trust * 1.5, 2e-3)
                    improved = True
        if improved:
            continue
        # --- null-space / equalization move (lsq_linear + variance descent) ---
        # only if LP failed to improve; try to equalize top plateau
        try:
            qe = min(600, q)
            # focus on tightest qe
            Ae = Aact[:qe]
            ce = cact[:qe]
            mean = float(ce.mean())
            rhs = (mean - ce)
            # least-squares equalizer with trust bounds
            r2 = lsq_linear(Ae, rhs, bounds=(-trust, trust), lsmr_tol='auto', verbose=0)
            cand_vs = []
            if r2.status is not None and np.all(np.isfinite(r2.x)):
                cand_vs.append(r2.x / max(1e-30, np.linalg.norm(r2.x)) * min(trust * np.sqrt(Nv), np.linalg.norm(r2.x)) if np.linalg.norm(r2.x) > 1e-16 else r2.x)
                cand_vs[-1] = r2.x
            # variance gradient direction projected off exact contacts
            resid = ce - mean
            g = (Ae.T @ resid) / max(1, qe)
            gn = float(np.linalg.norm(g))
            if np.isfinite(gn) and gn > 1e-18:
                g = g / gn
                # project off exact-contact rows if few
                ex = np.where(slack < 1e-9)[0]
                if len(ex) > 0 and len(ex) < 60:
                    Ie = iu0[ex]; Je = iu1[ex]
                    Aex, _, _ = _build_Aact(T, X, Ie, Je)
                    try:
                        M = Aex @ Aex.T
                        b = Aex @ g
                        w, *_ = np.linalg.lstsq(M + 1e-12 * np.eye(len(ex)), b, rcond=None)
                        g = g - Aex.T @ w
                        ng = float(np.linalg.norm(g))
                        if np.isfinite(ng) and ng > 1e-16:
                            g = g / ng
                            cand_vs.append(g * trust * 0.7)
                            cand_vs.append(-g * trust * 0.7)
                    except Exception:
                        cand_vs.append(g * trust * 0.5)
                        cand_vs.append(-g * trust * 0.5)
                else:
                    cand_vs.append(-g * trust * 0.7)
                    cand_vs.append(g * trust * 0.7)
            else:
                # rank-deficient fallback: random null-ish direction
                rr = rng.standard_normal(Nv)
                rr /= max(1e-30, np.linalg.norm(rr))
                cand_vs.append(rr * trust * 0.5)
            # random null move if rank deficient (extra exploration)
            if rank_def:
                rr = rng.standard_normal(Nv)
                # project off exact rows
                try:
                    ex = np.where(slack < 1e-9)[0][:20]
                    if len(ex) > 0:
                        Ie = iu0[ex]; Je = iu1[ex]
                        Aex, _, _ = _build_Aact(T, X, Ie, Je)
                        M = Aex @ Aex.T
                        b = Aex @ rr
                        w, *_ = np.linalg.lstsq(M + 1e-12 * np.eye(len(ex)), b, rcond=None)
                        rr = rr - Aex.T @ w
                except Exception:
                    pass
                nrr = float(np.linalg.norm(rr))
                if np.isfinite(nrr) and nrr > 1e-16:
                    cand_vs.append(rr / nrr * trust * 0.8)
            touched = False
            # evaluate candidates with backtracking on true max; allow
            # variance-reducing sideways moves that do not raise max
            cur_var = float(np.var(ce))
            for cv in cand_vs:
                if time.time() >= deadline:
                    break
                if not np.all(np.isfinite(cv)):
                    continue
                cv = np.clip(cv, -trust, trust)
                bX, bV, ba = _try_steps(X, T, cv, m_cur, alphas)
                if bX is not None:
                    X = bX
                    m_cur = bV
                    if bV < best - 1e-15:
                        best = bV
                        best_X = X.copy()
                    touched = True
                    trust = min(trust * 1.2, 2e-3)
                    break
            if touched:
                continue
            # try sideways equalization: accept if max not worse and variance drops
            best_side = None
            best_side_var = cur_var
            for cv in cand_vs:
                for a in (1.0, 0.5):
                    Y = _renorm(X + a * np.einsum('ndk,nk->nd', T, cv.reshape(n, k)))
                    if not np.all(np.isfinite(Y)):
                        continue
                    vv = _max_cos(Y)
                    if vv <= m_cur + 1e-12:
                        G2 = Y @ Y.T
                        cc = G2[iu0, iu1]
                        # variance of top-qe
                        idx = np.argpartition(-cc, qe - 1)[:qe] if qe < len(cc) else np.arange(len(cc))
                        var2 = float(np.var(cc[idx]))
                        if var2 < best_side_var - 1e-18:
                            best_side_var = var2
                            best_side = (Y, vv)
            if best_side is not None:
                X, m_cur = best_side
                # do not update best unless strictly better, but keep moving
                # shrink trust a bit to stay in basin
                trust = max(trust * 0.8, radii_floor)
                continue
        except Exception:
            pass
        # no progress: shrink trust through schedule 1e-3,3e-4,1e-4,...
        trust *= 0.5
        if trust < radii_floor:
            trust = 1e-3
            # avoid hot loop: tiny deterministic jitter in tangent space
            try:
                if time.time() < deadline - 3.0:
                    Tj = _tangent_bases(X)
                    kick = rng.standard_normal(Nv) * 1e-4
                    dXk = np.einsum('ndk,nk->nd', Tj, kick.reshape(n, k))
                    X = _renorm(X + dXk)
                    m_cur = _max_cos(X)
                    if m_cur < best - 1e-15:
                        best = m_cur
                        best_X = X.copy()
            except Exception:
                pass
    return best_X, best

def solve():
    T0 = time.time()
    BUDGET = 168.0
    END = T0 + BUDGET
    POLISH_TIME = 60.0
    rng = np.random.default_rng(0)
    try:
        xb = np.load("best.npy").astype(float)
    except Exception:
        try:
            xb = np.load("record.npy").astype(float)
        except Exception:
            xb = rng.standard_normal(size=(94, 13))
    xb = _renorm(xb.astype(float))
    base_v = _max_cos(xb)
    deadline = min(T0 + POLISH_TIME, END - 8.0)
    try:
        Xp, vp = _sqp_polish(xb, deadline, rng)
    except Exception:
        Xp, vp = xb, base_v
    Xp = _renorm(Xp)
    vp = _max_cos(Xp)
    # keep only if beats incumbent
    try:
        x0 = np.load("best.npy").astype(float)
        x0 = _renorm(x0)
        v0 = _max_cos(x0)
    except Exception:
        x0, v0 = xb, base_v
    if vp < min(v0, base_v) - 1e-15 and np.all(np.isfinite(Xp)):
        out = _renorm(Xp)
    else:
        out = _renorm(x0 if v0 <= base_v else xb)
    # strict feasibility: normalize, no NaN/zero rows
    out = np.asarray(out, dtype=float)
    nr = np.linalg.norm(out, axis=1, keepdims=True)
    nr[nr == 0] = 1.0
    out = out / nr
    return out
