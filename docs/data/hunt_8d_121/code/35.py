import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _lp_polish(x, deadline, max_contacts=3000):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    iu = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        Gf = x @ x.T
        c = Gf[iu]
        m = float(c.max())
        sel = c > m - max(20.0 * radius, 1e-9)
        idx = np.where(sel)[0]
        if idx.size == 0:
            radius *= 0.5
            continue
        if idx.size > max_contacts:
            o = np.argsort(-c[idx])[:max_contacts]
            idx = idx[o]
        I = iu[0][idx]; J = iu[1][idx]
        csel = c[idx]
        p = len(I)
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, n * d + 1)).tocsr()
            A_eq = coo_matrix((x.ravel(),
                               (np.repeat(np.arange(n), d), np.arange(n * d))),
                              shape=(n, n * d + 1)).tocsr()
        except Exception:
            radius *= 0.5
            continue
        obj = np.zeros(n * d + 1); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq,
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
        y = y / np.linalg.norm(y, axis=1, keepdims=True)
        vv = _max_cos(y)
        if vv < best:
            x, best = y, vv
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def _stress_null_basis(X, slack=1e-5, pcap=3200):
    n, d = X.shape
    N = n * d
    G = X @ X.T
    iu = np.triu_indices(n, 1)
    c = G[iu]
    m = float(c.max())
    slackv = m - c
    order = np.argsort(slackv)
    sel = order[slackv[order] < slack]
    if sel.size < 400:
        sel = order[:400]
    if sel.size > pcap:
        sel = sel[:pcap]
    I = iu[0][sel]; J = iu[1][sel]
    p = len(I)
    # tangent-projected equilibrium operator B (N x p)
    # column k: P_i^perp(x_J) at block I, P_j^perp(x_I) at block J
    dot = np.sum(X[I] * X[J], axis=1)
    YJ = X[J] - dot[:, None] * X[I]  # at I
    YI = X[I] - dot[:, None] * X[J]  # at J
    B = np.zeros((N, p), dtype=float)
    # fill
    for k in range(p):
        i = int(I[k]); j = int(J[k])
        B[i*d:(i+1)*d, k] = YJ[k]
        B[j*d:(j+1)*d, k] = YI[k]
    # entropic mirror descent for min ||B lam||^2 over simplex
    lam = np.full(p, 1.0/p)
    # scale eta adaptively
    for it in range(1200):
        R = B @ lam  # N
        g = B.T @ R  # p
        # center to avoid drift
        gmin = float(g.min()); gmax = float(g.max())
        spread = gmax - gmin
        if spread < 1e-16:
            break
        eta = 8.0 / (spread + 1e-12)
        # cap exponent
        step = np.clip(-eta * (g - float(g.mean())), -8.0, 8.0)
        lam = lam * np.exp(step)
        s = float(lam.sum())
        if not np.isfinite(s) or s <= 0:
            lam = np.full(p, 1.0/p); break
        lam /= s
        if it % 300 == 299:
            pass
    uniform = 1.0/p
    keep = lam > 0.25 * uniform
    # ensure enough kept
    if int(keep.sum()) < 200:
        o = np.argsort(-lam)[:max(200, min(p, N))]
        keep = np.zeros(p, dtype=bool); keep[o] = True
    # if everything kept (balanced), keep all
    if int(keep.sum()) == p:
        pass
    Is = I[keep]; Js = J[keep]
    ps = len(Is)
    # build full Jacobian A (ps x N) unprojected + E (n x N)
    A = np.zeros((ps, N), dtype=float)
    for k in range(ps):
        i = int(Is[k]); j = int(Js[k])
        A[k, i*d:(i+1)*d] = X[j]
        A[k, j*d:(j+1)*d] = X[i]
    E = np.zeros((n, N), dtype=float)
    for i in range(n):
        E[i, i*d:(i+1)*d] = X[i]
    M = np.vstack([A, E])
    try:
        _, S, Vt = np.linalg.svd(M, full_matrices=False)
    except Exception:
        return None, None, m, p
    tol = 1e-6
    nullmask = S < tol
    Vnull = Vt[nullmask]
    # extra null dims if rows < cols
    r = M.shape[0]; cc = M.shape[1]
    if r < cc:
        Vextra = Vt[min(len(S), r):]
        if Vextra.size > 0:
            if Vnull.size == 0:
                Vnull = Vextra
            else:
                Vnull = np.vstack([Vnull, Vextra])
    # fallback: softest modes
    if Vnull.shape[0] == 0:
        idx = np.argsort(S)[:3]
        Vnull = Vt[idx]
    return Vnull, (Is, Js, lam[keep]), m, p

def solve():
    t0 = time.time()
    BUDGET = 170.0
    DEAD = t0 + BUDGET
    np.random.seed(1)
    rng = np.random.default_rng(12345)
    n, d = 121, 8
    cands = []
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn).astype(float)
            if a.shape == (n, d):
                a = a / np.linalg.norm(a, axis=1, keepdims=True)
                if np.all(np.isfinite(a)):
                    cands.append(a)
        except Exception:
            pass
    if not cands:
        q = rng.standard_normal((n, d)); q /= np.linalg.norm(q, axis=1, keepdims=True)
        cands = [q]
    vals = [_max_cos(a) for a in cands]
    Xbest = cands[int(np.argmin(vals))].copy()
    Vbest = float(min(vals))
    if DEAD - time.time() > 25:
        Xbest = _lp_polish(Xbest, time.time() + 2.5)
        Vbest = _max_cos(Xbest)

    thetas = [0.01, 0.02, 0.04, 0.08, 0.12, 0.20]
    outer = 0
    stall = 0
    while time.time() < DEAD - 12.0 and outer < 40:
        outer += 1
        # rigidity analysis at current best
        try:
            Vnull, kept, m0, pall = _stress_null_basis(Xbest, slack=1e-5, pcap=3000)
        except Exception:
            Vnull = None
        if Vnull is None or Vnull.shape[0] == 0:
            # random tangent kick fallback
            D0 = rng.standard_normal((n, d))
            D0 -= np.sum(D0 * Xbest, axis=1, keepdims=True) * Xbest
            Vnull = D0.reshape(1, -1) / (np.linalg.norm(D0) + 1e-12)
        ndim = Vnull.shape[0]
        # cap number of trial directions
        ndir = 6 if ndim > 1 else 4
        cand_steps = []
        for di in range(ndir):
            coeff = rng.standard_normal(ndim)
            coeff /= (np.linalg.norm(coeff) + 1e-12)
            # mix: mostly null combo, occasionally add tiny random to break symmetry
            D = (coeff @ Vnull).reshape(n, d)
            # re-project to tangent
            D -= np.sum(D * Xbest, axis=1, keepdims=True) * Xbest
            mx = float(np.abs(D).max()) if False else float(np.linalg.norm(D, axis=1).max())
            if mx < 1e-12:
                continue
            D /= mx  # max row norm =1
            for th in thetas:
                alpha = float(np.tan(th))  # so max angular ~ th
                for sgn in (1.0, -1.0):
                    Y = Xbest + sgn * alpha * D
                    Y /= np.linalg.norm(Y, axis=1, keepdims=True)
                    v = _max_cos(Y)
                    cand_steps.append((v, sgn*alpha, D.copy(), Y))
                    if len(cand_steps) > 120:
                        break
                if len(cand_steps) > 120:
                    break
        if not cand_steps:
            continue
        cand_steps.sort(key=lambda t: t[0])
        # monitor contact crossings for top candidates (slack 1e-4)
        # pick diverse set: best by value + ones with different contact counts
        picks = []
        seen = set()
        G0 = Xbest @ Xbest.T
        np.fill_diagonal(G0, -2.0)
        # evaluate contact count for top 20
        for (v, al, Dd, Y) in cand_steps[:20]:
            G = Y @ Y.T; np.fill_diagonal(G, -2.0)
            cnt = int((G > float(G.max()) - 1e-4).sum() // 2)
            key = cnt // 25
            if key not in seen or len(picks) < 4:
                picks.append((v, al, Dd, Y, cnt))
                seen.add(key)
            if len(picks) >= 5:
                break
        improved = False
        for (v, al, Dd, Y, cnt) in picks:
            if time.time() > DEAD - 10.0:
                break
            # only polish promising or topology-changed
            if v > Vbest + 0.004 and len(picks) > 2:
                # skip very bad unless it changes topology a lot
                continue
            secs = 0.7 if (DEAD - time.time()) > 30 else 0.45
            Yp = _lp_polish(Y, time.time() + secs)
            vp = _max_cos(Yp)
            if vp < Vbest - 1e-12:
                Xbest = Yp; Vbest = vp; improved = True
                break
        if improved:
            stall = 0
            continue
        else:
            stall += 1
        # if stalled, inject a coordinated cap re-arrangement transferring contacts:
        # move overloaded points slightly along floppy + small noise, then short polish
        if stall % 4 == 0 and time.time() < DEAD - 14.0:
            G = Xbest @ Xbest.T; np.fill_diagonal(G, -2.0)
            m = float(G.max())
            contact = (G > m - 1e-9)
            deg = contact.sum(axis=1)
            # overloaded: top 8, underlocked: bottom 8
            o_over = np.argsort(-deg)[:8]
            o_under = np.argsort(deg)[:8]
            coeff = rng.standard_normal(ndim); coeff /= (np.linalg.norm(coeff)+1e-12)
            D = (coeff @ Vnull).reshape(n, d)
            D -= np.sum(D * Xbest, axis=1, keepdims=True) * Xbest
            mx = float(np.linalg.norm(D, axis=1).max()) + 1e-12
            D /= mx
            Y = Xbest.copy()
            Y[o_over] += 0.03 * D[o_over] + rng.standard_normal((8, d)) * 0.008
            Y[o_under] += 0.05 * D[o_under] + rng.standard_normal((8, d)) * 0.012
            Y /= np.linalg.norm(Y, axis=1, keepdims=True)
            Yp = _lp_polish(Y, time.time() + 0.8)
            vp = _max_cos(Yp)
            if vp < Vbest - 1e-12:
                Xbest = Yp; Vbest = vp; stall = 0
    rem = DEAD - time.time()
    if rem > 1.5:
        Xbest = _lp_polish(Xbest, time.time() + max(0.5, rem - 0.8))
    Xbest = Xbest / np.linalg.norm(Xbest, axis=1, keepdims=True)
    return Xbest
