import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _lp_polish(x, deadline, max_pairs=3000):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    try:
        best = _max_cos(x)
    except Exception:
        return x
    radius = 1e-3
    iu = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        Gfull = x @ x.T
        c = Gfull[iu]
        m = c.max()
        thr = m - max(20.0 * radius, 1e-9)
        sel = c > thr
        cnt = int(np.count_nonzero(sel))
        if cnt == 0:
            break
        if cnt > max_pairs:
            idx = np.where(sel)[0]
            order = np.argsort(-c[idx])[:max_pairs]
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
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix(
            (x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))),
            shape=(n, nv)
        ).tocsr()
        obj = np.zeros(nv)
        obj[-1] = 1.0
        try:
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
        y = y / np.linalg.norm(y, axis=1, keepdims=True)
        v = _max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def _anneal_full(x, steps, beta0, beta1, lr0, lr1):
    n, d = x.shape
    for s in range(steps):
        f = s / max(steps - 1, 1)
        beta = beta0 + (beta1 - beta0) * f
        lr = lr0 + (lr1 - lr0) * f
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        m = G.max()
        W = np.exp(beta * (G - m))
        np.fill_diagonal(W, 0.0)
        Grad = W @ x
        dot = (Grad * x).sum(axis=1, keepdims=True)
        Tan = Grad - dot * x
        # normalize step by mean norm to keep stable
        x = x - lr * Tan / (np.mean(np.linalg.norm(Tan, axis=1)) + 1.0)
        x = x / np.linalg.norm(x, axis=1, keepdims=True)
    return x

# ---- symmetric orbit helpers ----
def _rot_tables(m, mults):
    k = np.arange(m)[:, None]  # m,1
    mu = np.array(mults, dtype=float)[None, :]  # 1,4
    ang = 2.0 * np.pi * k * mu / float(m)  # m,4
    return np.cos(ang), np.sin(ang)

def _expand_rot(S, C, Ss):
    # S: ns,8 ; C,Ss: m,4 -> X: ns*m,8
    ns = S.shape[0]
    m = C.shape[0]
    S0 = S[:, 0::2]  # ns,4
    S1 = S[:, 1::2]
    # ns,m,4
    X0 = S0[:, None, :] * C[None, :, :] - S1[:, None, :] * Ss[None, :, :]
    X1 = S0[:, None, :] * Ss[None, :, :] + S1[:, None, :] * C[None, :, :]
    X = np.empty((ns, m, 8), dtype=float)
    X[:, :, 0::2] = X0
    X[:, :, 1::2] = X1
    return X.reshape(ns * m, 8)

def _backprop_rot(Gr, C, Ss, ns, m):
    # Gr: ns*m,8 -> Gs: ns,8
    G = Gr.reshape(ns, m, 8)
    G0 = G[:, :, 0::2]  # ns,m,4
    G1 = G[:, :, 1::2]
    # dS0 = sum_k C*G0 + Ss*G1 ; dS1 = sum_k -Ss*G0 + C*G1
    dS0 = np.sum(C[None, :, :] * G0 + Ss[None, :, :] * G1, axis=1)
    dS1 = np.sum(-Ss[None, :, :] * G0 + C[None, :, :] * G1, axis=1)
    Gs = np.empty((ns, 8), dtype=float)
    Gs[:, 0::2] = dS0
    Gs[:, 1::2] = dS1
    return Gs

def _expand_shiftneg(S):
    # S: ns,8 -> X: ns*16,8  (8 shifts x +-1)
    ns = S.shape[0]
    X = np.empty((ns * 16, 8), dtype=float)
    pos = 0
    for sg in (1.0, -1.0):
        for k in range(8):
            # X_j = sg * S_{(j-k) mod 8}
            idx = (np.arange(8) - k) % 8
            X[pos:pos+ns] = sg * S[:, idx]
            pos += ns
    return X

def _backprop_shiftneg(Gr, ns):
    # Gr: ns*16,8
    G = Gr.reshape(2, 8, ns, 8)  # sg, k, s, j  (order sg outer? we filled sg outer)
    # our fill order: sg=0 block of 8*ns then sg=1 block. reshape as (2,8,ns,8)
    Gs = np.zeros((ns, 8), dtype=float)
    sgs = [1.0, -1.0]
    for si, sg in enumerate(sgs):
        for k in range(8):
            Gsk = G[si, k]  # ns,8  (j index)
            # dS_i += sg * G_{(i+k)%8}
            # so for each i, take G at j=(i+k)%8
            jidx = (np.arange(8)[None, :] + k) % 8  # not needed vectorized trick:
            # vectorized: dS[:,i] += sg*G[:,(i+k)%8]
            Gs += sg * Gsk[:, (np.arange(8) - k) % 8][:, ::-1][:, ::-1] if False else sg * np.take_along_axis(Gsk, np.tile(((np.arange(8)[None, :] + k) % 8), (ns, 1)), axis=1) if False else sg*0
            # simpler loop over i with roll
    # redo correctly with rolls (ns small so loop fine)
    Gs = np.zeros((ns, 8), dtype=float)
    pos = 0
    for sg in (1.0, -1.0):
        for k in range(8):
            Gsk = Gr[pos:pos+ns]  # ns,8
            pos += ns
            for i in range(8):
                Gs[:, i] += sg * Gsk[:, (i + k) % 8]
    return Gs

def _symm_adam_rot(m, mults, ns, nf, rng, steps_per_beta=220, lrs=0.05):
    betas = [30.0, 60.0, 100.0, 170.0, 260.0, 400.0]
    C, Ss = _rot_tables(m, mults)
    Ys = rng.normal(size=(ns, 8))
    Yf = rng.normal(size=(nf, 8)) if nf > 0 else None
    ms = np.zeros_like(Ys); vs = np.zeros_like(Ys)
    mf = np.zeros_like(Yf) if nf > 0 else None
    vf = np.zeros_like(Yf) if nf > 0 else None
    b1, b2, eps = 0.9, 0.999, 1e-8
    t = 0
    N = ns * m + nf
    bestX = None; bestv = 1e9
    for beta in betas:
        for _ in range(steps_per_beta):
            t += 1
            nYs = np.linalg.norm(Ys, axis=1, keepdims=True) + 1e-30
            S = Ys / nYs
            if nf > 0:
                nYf = np.linalg.norm(Yf, axis=1, keepdims=True) + 1e-30
                F = Yf / nYf
                Xr = _expand_rot(S, C, Ss)
                X = np.concatenate([Xr, F], axis=0)
            else:
                F = None
                X = _expand_rot(S, C, Ss)
            # softmax weights
            G = X @ X.T
            np.fill_diagonal(G, -2.0)
            mv = G.max()
            E = np.exp(beta * (G - mv))
            np.fill_diagonal(E, 0.0)
            Ssum = E.sum() + 1e-300
            # Grad_X = (E @ X)/Ssum
            Grad = (E @ X) / Ssum
            # split
            if nf > 0:
                Gr_rot = Grad[:ns*m]; Gr_free = Grad[ns*m:]
                Gs = _backprop_rot(Gr_rot, C, Ss, ns, m)
                Gf = Gr_free
            else:
                Gs = _backprop_rot(Grad, C, Ss, ns, m)
            # to Y grads (tangential / norm)
            dotS = (Gs * S).sum(axis=1, keepdims=True)
            GyS = (Gs - dotS * S) / nYs
            ms = b1 * ms + (1 - b1) * GyS
            vs = b2 * vs + (1 - b2) * (GyS * GyS)
            mhat = ms / (1 - b1 ** t); vhat = vs / (1 - b2 ** t)
            Ys = Ys - lrs * mhat / (np.sqrt(vhat) + eps)
            if nf > 0:
                dotF = (Gf * F).sum(axis=1, keepdims=True)
                GyF = (Gf - dotF * F) / nYf
                mf = b1 * mf + (1 - b1) * GyF
                vf = b2 * vf + (1 - b2) * (GyF * GyF)
                mhatf = mf / (1 - b1 ** t); vhatf = vf / (1 - b2 ** t)
                Yf = Yf - lrs * mhatf / (np.sqrt(vhatf) + eps)
            if (t % 120) == 0:
                # snapshot
                nYs2 = np.linalg.norm(Ys, axis=1, keepdims=True) + 1e-30
                S2 = Ys / nYs2
                Xr2 = _expand_rot(S2, C, Ss)
                if nf > 0:
                    nYf2 = np.linalg.norm(Yf, axis=1, keepdims=True) + 1e-30
                    X2 = np.concatenate([Xr2, Yf / nYf2], axis=0)
                else:
                    X2 = Xr2
                v = _max_cos(X2)
                if v < bestv:
                    bestv = v; bestX = X2.copy()
    # final
    nYs2 = np.linalg.norm(Ys, axis=1, keepdims=True) + 1e-30
    S2 = Ys / nYs2
    Xr2 = _expand_rot(S2, C, Ss)
    if nf > 0:
        nYf2 = np.linalg.norm(Yf, axis=1, keepdims=True) + 1e-30
        X2 = np.concatenate([Xr2, Yf / nYf2], axis=0)
    else:
        X2 = Xr2
    v = _max_cos(X2)
    if v < bestv:
        bestv = v; bestX = X2.copy()
    return bestX, bestv

def _symm_adam_shiftneg(ns, nf, rng, steps_per_beta=200, lrs=0.05):
    betas = [30.0, 70.0, 130.0, 220.0, 400.0]
    Ys = rng.normal(size=(ns, 8))
    Yf = rng.normal(size=(nf, 8)) if nf > 0 else None
    ms = np.zeros_like(Ys); vs = np.zeros_like(Ys)
    mf = np.zeros_like(Yf) if nf > 0 else None
    vf = np.zeros_like(Yf) if nf > 0 else None
    b1, b2, eps = 0.9, 0.999, 1e-8
    t = 0
    bestX = None; bestv = 1e9
    for beta in betas:
        for _ in range(steps_per_beta):
            t += 1
            nYs = np.linalg.norm(Ys, axis=1, keepdims=True) + 1e-30
            S = Ys / nYs
            Xr = _expand_shiftneg(S)
            if nf > 0:
                nYf = np.linalg.norm(Yf, axis=1, keepdims=True) + 1e-30
                F = Yf / nYf
                X = np.concatenate([Xr, F], axis=0)
            else:
                X = Xr
            G = X @ X.T
            np.fill_diagonal(G, -2.0)
            mv = G.max()
            E = np.exp(beta * (G - mv))
            np.fill_diagonal(E, 0.0)
            Ssum = E.sum() + 1e-300
            Grad = (E @ X) / Ssum
            if nf > 0:
                Gr_rot = Grad[:ns*16]; Gr_free = Grad[ns*16:]
                Gs = _backprop_shiftneg(Gr_rot, ns)
                Gf = Gr_free
            else:
                Gs = _backprop_shiftneg(Grad, ns)
            dotS = (Gs * S).sum(axis=1, keepdims=True)
            GyS = (Gs - dotS * S) / nYs
            ms = b1*ms+(1-b1)*GyS; vs = b2*vs+(1-b2)*(GyS*GyS)
            Ys = Ys - lrs * (ms/(1-b1**t)) / (np.sqrt(vs/(1-b2**t))+eps)
            if nf > 0:
                dotF = (Gf * F).sum(axis=1, keepdims=True)
                GyF = (Gf - dotF * F) / nYf
                mf = b1*mf+(1-b1)*GyF; vf = b2*vf+(1-b2)*(GyF*GyF)
                Yf = Yf - lrs * (mf/(1-b1**t)) / (np.sqrt(vf/(1-b2**t))+eps)
            if (t % 120) == 0:
                n2 = np.linalg.norm(Ys, axis=1, keepdims=True)+1e-30
                Xr2 = _expand_shiftneg(Ys/n2)
                X2 = np.concatenate([Xr2, Yf/np.linalg.norm(Yf,axis=1,keepdims=True)],axis=0) if nf>0 else Xr2
                v = _max_cos(X2)
                if v < bestv:
                    bestv=v; bestX=X2.copy()
    n2 = np.linalg.norm(Ys, axis=1, keepdims=True)+1e-30
    Xr2 = _expand_shiftneg(Ys/n2)
    X2 = np.concatenate([Xr2, Yf/np.linalg.norm(Yf,axis=1,keepdims=True)],axis=0) if nf>0 else Xr2
    v=_max_cos(X2)
    if v<bestv:
        bestv=v; bestX=X2.copy()
    return bestX, bestv

def solve():
    rng = np.random.default_rng(0)
    t0 = time.time()
    deadline = t0 + 168.0
    n, d = 138, 8
    best = None; best_val = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            v = np.load(fn).astype(float)
            if v.shape == (n, d):
                v = v / np.linalg.norm(v, axis=1, keepdims=True)
                c = _max_cos(v)
                if c < best_val:
                    best_val = c; best = v.copy()
        except Exception:
            pass
    if best is None:
        best = rng.normal(size=(n, d))
        best = best / np.linalg.norm(best, axis=1, keepdims=True)
        best_val = _max_cos(best)
    # quick tighten of loaded best (short)
    try:
        if time.time() < deadline - 100:
            best = _lp_polish(best, min(deadline-100, time.time()+2.0))
            best_val = _max_cos(best)
    except Exception:
        pass
    cands = []
    # symmetric configs: (m, mults, ns) with ns*m==138 except shiftneg
    configs = [
        (23, [1, 2, 3, 5], 6),
        (23, [1, 3, 4, 7], 6),
        (46, [1, 3, 5, 7], 3),
        (69, [1, 2, 4, 7], 2),
        (138, [1, 5, 7, 11], 1),
    ]
    for ci, (m, mults, ns) in enumerate(configs):
        if time.time() > deadline - 55:
            break
        try:
            r2 = np.random.default_rng(1000 + ci)
            X, v = _symm_adam_rot(m, mults, ns, 0, r2, steps_per_beta=200, lrs=0.05)
            # deterministic prune/pad if needed (numerical duplicates)
            if X.shape[0] != n:
                # pad or prune to 138
                if X.shape[0] > n:
                    # remove closest pairs greedily
                    cur = X.copy()
                    while cur.shape[0] > n:
                        Gc = cur @ cur.T
                        np.fill_diagonal(Gc, -2)
                        ij = np.unravel_index(int(np.argmax(Gc)), Gc.shape)
                        # drop one with larger second-max
                        cur = np.delete(cur, ij[1], axis=0)
                    X = cur; v = _max_cos(X)
                else:
                    need = n - X.shape[0]
                    cur = X.copy()
                    for _ in range(need):
                        cand = r2.normal(size=(20000, d))
                        cand /= np.linalg.norm(cand, axis=1, keepdims=True)
                        sm = (cand @ cur.T).max(axis=1)
                        cur = np.concatenate([cur, cand[np.argmin(sm):np.argmin(sm)+1]], axis=0)
                    X = cur; v = _max_cos(X)
            cands.append((v, X.copy()))
            if v < best_val - 1e-12:
                best_val = v; best = X.copy()
        except Exception:
            continue
    # shiftneg group: ns=8 ->128 +10 free
    if time.time() < deadline - 45:
        try:
            r2 = np.random.default_rng(9999)
            X, v = _symm_adam_shiftneg(8, 10, r2, steps_per_beta=180, lrs=0.05)
            cands.append((v, X.copy()))
            if v < best_val - 1e-12:
                best_val = v; best = X.copy()
        except Exception:
            pass
    # sort candidates
    cands.sort(key=lambda t: t[0])
    # symmetry breaking: take top up to 3
    top = cands[:3] if len(cands) else [(best_val, best)]
    for ti, (cv, X0) in enumerate(top):
        if time.time() > deadline - 30:
            break
        try:
            Xb = X0.copy()
            # stages of tangent noise release
            for amp, b0, b1, l0, l1, st in [(1e-3, 30.0, 300.0, 0.015, 0.001, 500),
                                           (1e-2, 30.0, 350.0, 0.015, 0.001, 600),
                                           (3e-2, 25.0, 400.0, 0.018, 0.001, 600)]:
                if time.time() > deadline - 20:
                    break
                rr = np.random.default_rng(5000 + ti * 10 + int(amp * 1e5))
                Nz = rr.normal(size=Xb.shape)
                Nz = Nz - (Nz * Xb).sum(axis=1, keepdims=True) * Xb
                Nz /= (np.linalg.norm(Nz, axis=1, keepdims=True) + 1e-30)
                Xb = Xb + amp * Nz
                Xb /= np.linalg.norm(Xb, axis=1, keepdims=True)
                Xb = _anneal_full(Xb, steps=st, beta0=b0, beta1=b1, lr0=l0, lr1=l1)
                # light LP
                Xb = _lp_polish(Xb, min(deadline - 12, time.time() + 1.5), max_pairs=2500)
                v = _max_cos(Xb)
                if v < best_val - 1e-12:
                    best_val = v; best = Xb.copy()
            # extra shake of this basin
            if time.time() < deadline - 18:
                y = Xb + 0.05 * rng.normal(size=Xb.shape)
                y /= np.linalg.norm(y, axis=1, keepdims=True)
                y = _anneal_full(y, steps=500, beta0=20.0, beta1=320.0, lr0=0.02, lr1=0.001)
                y = _lp_polish(y, min(deadline - 12, time.time() + 1.2), max_pairs=2500)
                v = _max_cos(y)
                if v < best_val - 1e-12:
                    best_val = v; best = y.copy()
        except Exception:
            continue
    try:
        best = _lp_polish(best, min(deadline - 0.5, time.time() + 7.0), max_pairs=4000)
    except Exception:
        pass
    best = best / np.linalg.norm(best, axis=1, keepdims=True)
    return best
