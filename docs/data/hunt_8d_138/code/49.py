import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos_full(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    f = int(np.argmax(G))
    n = x.shape[0]
    return float(G.flat[f]), (f // n, f % n)

def _max_cos(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _norm(x):
    return x / np.linalg.norm(x, axis=1, keepdims=True)

def _anneal_full(x, steps, beta0, beta1, lr0, lr1):
    n, d = x.shape
    for s in range(steps):
        f = s / max(steps - 1, 1)
        beta = beta0 + (beta1 - beta0) * f
        lr = lr0 + (lr1 - lr0) * f
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        m = G.max()
        E = np.exp(beta * (G - m))
        np.fill_diagonal(E, 0.0)
        S = E.sum() + 1e-300
        Grad = (E @ x) / S
        dot = (Grad * x).sum(axis=1, keepdims=True)
        Tan = Grad - dot * x
        x = x - lr * Tan
        x = x / np.linalg.norm(x, axis=1, keepdims=True)
    return x

def _lp_polish_restart(x, deadline, active_gap=1e-3, r0=1e-2, cap=3000):
    x = _norm(x)
    n, d = x.shape
    try:
        best = _max_cos(x)
    except Exception:
        return x
    radius = r0
    iu = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        Gfull = x @ x.T
        c = Gfull[iu]
        m = c.max()
        thr = m - active_gap
        sel = c > thr
        cnt = int(np.count_nonzero(sel))
        if cnt == 0:
            break
        if cnt > cap:
            idx = np.where(sel)[0]
            order = np.argsort(-c[idx])[:cap]
            mask = np.zeros_like(sel, dtype=bool)
            mask[idx[order]] = True
            sel = mask
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate(
            [I[:, None] * d + np.arange(d),
             J[:, None] * d + np.arange(d),
             np.full((p, 1), n * d)], 1
        ).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix(
                (x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))),
                shape=(n, nv)
            ).tocsr()
            obj = np.zeros(nv)
            obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-c[sel], A_eq=A_eq,
                          b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                          method="highs")
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
            x, best = y, v
            radius = min(radius * 1.5, 0.05)
            # restart trust radius occasionally: keep large while improving
            if radius < 1e-3:
                radius = 1e-3
        else:
            radius *= 0.5
    return x

def _tangent_bases(x):
    n, d = x.shape
    T = np.empty((n, d, d - 1))
    for i in range(n):
        # null space of x[i]: SVD of 1xd
        _, _, Vh = np.linalg.svd(x[i:i+1, :], full_matrices=True)
        # Vh[0] ~ x[i], rest is basis
        T[i] = Vh[1:].T
    return T

def _build_J(x, I, J, T):
    n, d = x.shape
    dm = d - 1
    p = len(I)
    Ncol = n * dm
    Jmat = np.empty((p, Ncol))
    for k in range(p):
        i = int(I[k]); j = int(J[k])
        a = T[i].T @ x[j]  # dm
        b = T[j].T @ x[i]
        row = np.zeros(Ncol)
        row[i*dm:(i+1)*dm] = a
        row[j*dm:(j+1)*dm] = b
        Jmat[k] = row
    return Jmat

def solve():
    rng = np.random.default_rng(0)
    t0 = time.time()
    deadline = t0 + 168.0
    n, d = 138, 8
    best = None
    best_val = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            v = np.load(fn).astype(float)
            if v.shape == (n, d):
                v = _norm(v)
                c = _max_cos(v)
                if c < best_val:
                    best_val = c
                    best = v.copy()
        except Exception:
            pass
    if best is None:
        best = _norm(rng.standard_normal(size=(n, d)))
        best_val = _max_cos(best)
    x0 = best.copy()
    m0, arg0 = _max_cos_full(x0)
    iu = np.triu_indices(n, 1)
    G0 = x0 @ x0.T
    c0 = G0[iu]

    # tangent bases once
    try:
        T = _tangent_bases(x0)
    except Exception:
        T = None

    candidates = []

    # --- floppy mode analysis at several tolerances ---
    if T is not None:
        try:
            for tol in (1e-7, 1e-5, 1e-4):
                if time.time() > deadline - 60:
                    break
                sel = c0 > (m0 - tol)
                I = iu[0][sel]; J = iu[1][sel]
                p = len(I)
                if p < 5 or p > 5000:
                    continue
                Jmat = _build_J(x0, I, J, T)
                # floppy via eigh of Gram
                GtG = Jmat.T @ Jmat  # 966x966
                # regularize tiny
                w, V = np.linalg.eigh(GtG)
                # w ascending; take bottom K
                K = min(20, V.shape[1])
                dm = d - 1
                for kk in range(K):
                    if time.time() > deadline - 60:
                        break
                    vec = V[:, kk]  # smallest first
                    Y = vec.reshape(n, dm)
                    DX = np.einsum('nij,nj->ni', T, Y)  # (n,d)
                    # normalize direction to unit overall norm
                    nr = np.linalg.norm(DX) + 1e-30
                    DX = DX / nr
                    # exact line search over log-spaced steps, both signs
                    for sgn in (1.0, -1.0):
                        for alpha in (3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 6e-2, 1e-1, 2e-1):
                            Xt = x0 + sgn * alpha * DX
                            Xt = _norm(Xt)
                            v, arg = _max_cos_full(Xt)
                            # allow continuation until inactive pair hits max:
                            # keep if not too degraded and arg changed or v improved
                            if v < m0 + 1e-3:
                                changed = (arg != arg0)
                                candidates.append((v, changed, sgn*alpha, kk, tol, Xt))
                # only need one tolerance for directions if enough candidates
                if len(candidates) > 200:
                    break
        except Exception:
            pass

    # rank candidates: prefer improved, then changed topology with low v
    candsel = []
    try:
        # sort by v
        candidates.sort(key=lambda t: t[0])
        # take diverse subset: first few overall + first few with changed=True
        seen = 0
        for tup in candidates:
            if tup[0] < m0 + 5e-4:
                candsel.append(tup)
                if len(candsel) >= 40:
                    break
    except Exception:
        candsel = []

    # polish a few most promising floppy trials with restarted LP
    polished = 0
    # pick at most 5 to polish: spread across alphas
    to_polish = []
    if candsel:
        # always include best 2
        to_polish = [candsel[0][5]]
        # add others with spacing in list
        step = max(1, len(candsel)//8)
        for k in range(0, len(candsel), step):
            Xt = candsel[k][5]
            # avoid duplicates
            dup = False
            for Y in to_polish:
                if np.linalg.norm(Xt-Y) < 1e-6:
                    dup = True; break
            if not dup:
                to_polish.append(Xt)
            if len(to_polish) >= 5:
                break
    for Xt in to_polish:
        if time.time() > deadline - 45:
            break
        try:
            # quick LSE anneal then restarted LP
            Xw = _anneal_full(Xt.copy(), steps=200, beta0=30.0, beta1=250.0, lr0=0.015, lr1=0.002)
            Xp = _lp_polish_restart(Xw, min(deadline-30, time.time()+6.0), active_gap=1e-3, r0=1e-2)
            v = _max_cos(Xp)
            if v < best_val - 1e-12:
                best_val = v
                best = Xp.copy()
                m0 = best_val
        except Exception:
            continue

    # --- drop-one-tight-contact experiments ---
    try:
        # tightest pairs of current incumbent (updated best)
        Gc = best @ best.T
        cc = Gc[iu]
        order = np.argsort(-cc)
        # try up to 6 distinct pairs involving different points
        used = set()
        trials = []
        for o in order:
            if len(trials) >= 12:
                break
            i = int(iu[0][o]); j = int(iu[1][o])
            if i in used or j in used:
                continue
            # only very tight
            if cc[o] < best_val - 1e-3:
                break
            used.add(i); used.add(j)
            trials.append((i, j))
        for (i, j) in trials:
            if time.time() > deadline - 12:
                break
            try:
                Xt = best.copy()
                # push i away from j tangentially and vice versa
                eps = 0.03
                pi = Xt[j] - np.dot(Xt[i], Xt[j]) * Xt[i]
                pj = Xt[i] - np.dot(Xt[j], Xt[i]) * Xt[j]
                ni = np.linalg.norm(pi) + 1e-30
                nj = np.linalg.norm(pj) + 1e-30
                Xt[i] = Xt[i] - eps * pi / ni
                Xt[j] = Xt[j] - eps * pj / nj
                Xt = _norm(Xt)
                # re-optimise remaining with LSE while keeping this pair apart:
                # add extra repulsion by doing a few steps where that pair is excluded from attraction?
                # simple: anneal, then LP polish with that pair forced <= m-1e-3 via initial separation;
                # we emulate constraint by re-annealing with a penalty pushing i,j apart
                for _ in range(3):
                    # one anneal burst
                    Xt = _anneal_full(Xt, steps=120, beta0=40.0, beta1=250.0, lr0=0.012, lr1=0.002)
                    # if dropped pair still maximal, push again
                    if np.dot(Xt[i], Xt[j]) > _max_cos(Xt) - 1e-6:
                        pi2 = Xt[j] - np.dot(Xt[i], Xt[j]) * Xt[i]
                        pj2 = Xt[i] - np.dot(Xt[j], Xt[i]) * Xt[j]
                        Xt[i] = Xt[i] - 0.01 * pi2 / (np.linalg.norm(pi2)+1e-30)
                        Xt[j] = Xt[j] - 0.01 * pj2 / (np.linalg.norm(pj2)+1e-30)
                        Xt = _norm(Xt)
                    else:
                        break
                Xp = _lp_polish_restart(Xt, min(deadline-5, time.time()+4.0), active_gap=1e-3, r0=1e-2)
                v = _max_cos(Xp)
                if v < best_val - 1e-12:
                    best_val = v
                    best = Xp.copy()
            except Exception:
                continue
    except Exception:
        pass

    # final polish of incumbent with restarted radius
    try:
        if time.time() < deadline - 1.0:
            best = _lp_polish_restart(best, deadline-0.3, active_gap=1e-3, r0=1e-2)
    except Exception:
        pass
    best = _norm(best)
    return best
