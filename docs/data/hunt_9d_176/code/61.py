import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def count_contacts(X, thresh):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return int(np.count_nonzero(G > thresh))

def smooth_polish(X, steps=600, a0=40.0, a1=400.0, lr=0.003):
    x = X.copy()
    n, d = x.shape
    m = np.zeros_like(x)
    vv = np.zeros_like(x)
    b1, b2, eps = 0.9, 0.999, 1e-8
    for t in range(steps):
        frac = t / max(steps - 1, 1)
        alpha = a0 * ((a1 / a0) ** frac)
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        M = float(np.max(G))
        E = np.exp(alpha * (G - M))
        np.fill_diagonal(E, 0.0)
        S = float(E.sum() * 0.5)
        if S <= 0 or not np.isfinite(S):
            break
        Grad = (E @ x) / S
        dot = np.sum(Grad * x, axis=1, keepdims=True)
        Gt = Grad - dot * x
        m = b1 * m + (1 - b1) * Gt
        vv = b2 * vv + (1 - b2) * (Gt * Gt)
        mhat = m / (1 - b1 ** (t + 1))
        vhat = vv / (1 - b2 ** (t + 1))
        step = mhat / (np.sqrt(vhat) + eps)
        np.clip(step, -3.0, 3.0, out=step)
        lrt = lr * (1.0 - 0.5 * frac)
        x = x - lrt * step
        nr = np.linalg.norm(x, axis=1, keepdims=True)
        nr[nr == 0] = 1.0
        x /= nr
        if not np.all(np.isfinite(x)):
            break
    nr = np.linalg.norm(x, axis=1, keepdims=True)
    nr[nr == 0] = 1.0
    return x / nr

def lp_polish(X, seconds, radius_init=3e-3, cap=2800):
    x = X / np.linalg.norm(X, axis=1, keepdims=True)
    n, d = x.shape
    best = max_cos(x)
    radius = radius_init
    start = time.time()
    iu0 = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        Gfull = x @ x.T
        c_all = Gfull[iu0]
        mval = float(c_all.max())
        thresh = mval - max(20 * radius, 1e-9)
        sel = c_all > thresh
        cnt = int(np.count_nonzero(sel))
        if cnt == 0:
            break
        if cnt > cap:
            idx_all = np.where(sel)[0]
            order = np.argpartition(-c_all[idx_all], cap)[:cap]
            keep = np.zeros_like(sel)
            keep[idx_all[order]] = True
            sel = keep
            cnt = cap
        I = iu0[0][sel]
        J = iu0[1][sel]
        c = c_all[sel]
        p = cnt
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))),
                              shape=(n, nv)).tocsr()
        except Exception:
            radius *= 0.5
            continue
        obj = np.zeros(nv)
        obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                          method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        nr = np.linalg.norm(y, axis=1, keepdims=True)
        nr[nr == 0] = 1.0
        y /= nr
        v = max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
        if time.time() - start >= seconds:
            break
    return x

def solve():
    t0 = time.time()
    HARD = 165.0
    rng = np.random.default_rng(42)
    try:
        xb = np.load("best.npy")
    except Exception:
        try:
            xb = np.load("record.npy")
        except Exception:
            xb = rng.standard_normal((176, 9))
    xb = np.asarray(xb, dtype=float)
    if xb.shape != (176, 9):
        xb = rng.standard_normal((176, 9))
    xb = xb / np.linalg.norm(xb, axis=1, keepdims=True)
    best_val = max_cos(xb)
    best_x = xb.copy()
    # also check record
    try:
        xr = np.load("record.npy")
        xr = np.asarray(xr, dtype=float)
        if xr.shape == (176, 9):
            xr = xr / np.linalg.norm(xr, axis=1, keepdims=True)
            vr = max_cos(xr)
            if vr < best_val:
                best_val, best_x = vr, xr.copy()
    except Exception:
        pass
    incumbent_val = best_val
    incumbent_x = best_x.copy()
    n, d = 176, 9
    dm = d - 1
    N = n * dm

    try:
        X0 = incumbent_x.copy()
        G0 = X0 @ X0.T
        np.fill_diagonal(G0, -2.0)
        m0 = float(np.max(G0))
        iu = np.triu_indices(n, 1)
        c_all = G0[iu]
        sel = c_all > (m0 - 1e-7)
        I = iu[0][sel]; J = iu[1][sel]
        p = len(I)
        if p < 10 or p > 5000:
            raise RuntimeError("bad p")
        # tangent bases
        Tb = np.empty((n, d, dm))
        rngT = np.random.default_rng(7)
        for i in range(n):
            xi = X0[i]
            M = rngT.standard_normal((d, dm))
            M -= xi[:, None] * (xi @ M)
            Q, _ = np.linalg.qr(M)
            Tb[i] = Q
        # sparse->dense Jacobian
        Jmat = np.zeros((p, N), dtype=float)
        XT = X0
        for k in range(p):
            i = int(I[k]); j = int(J[k])
            Jmat[k, i*dm:(i+1)*dm] = Tb[i].T @ XT[j]
            Jmat[k, j*dm:(j+1)*dm] = Tb[j].T @ XT[i]
        # rotation generators deflation
        Rmat = np.zeros((N, 36), dtype=float)
        col = 0
        for a in range(d):
            for b in range(a+1, d):
                for i in range(n):
                    xi = X0[i]
                    # E @ xi : +xi[b] at a, -xi[a] at b
                    v = np.zeros(d)
                    v[a] = xi[b]
                    v[b] = -xi[a]
                    Rmat[i*dm:(i+1)*dm, col] = Tb[i].T @ v
                col += 1
        Qr, _ = np.linalg.qr(Rmat)
        # Gram
        Gram = Jmat.T @ Jmat
        # symmetrize
        Gram = 0.5 * (Gram + Gram.T)
        w, V = np.linalg.eigh(Gram)
        # V columns ascending; take first 10
        nmodes = min(10, N)
        modes = []
        for k in range(nmodes):
            v = V[:, k].copy()
            # remove rotation component
            coeff = Qr.T @ v
            v = v - Qr @ coeff
            nv = float(np.linalg.norm(v))
            if nv < 1e-8:
                continue
            v /= nv
            modes.append(v)
            if len(modes) >= 10:
                break
        if len(modes) < 5:
            # fallback: take more eigenvectors
            for k in range(nmodes, min(20, N)):
                v = V[:, k].copy()
                coeff = Qr.T @ v
                v = v - Qr @ coeff
                nv = float(np.linalg.norm(v))
                if nv < 1e-8:
                    continue
                v /= nv
                modes.append(v)
                if len(modes) >= 8:
                    break
        amps = [0.02, 0.05, 0.1, 0.2]
        cands = []
        base_contacts = int(np.count_nonzero(c_all > (m0 - 1e-7)))
        # exact max evaluation only
        for mi, v in enumerate(modes[:10]):
            Vm = v.reshape(n, dm)
            # ambient direction
            DX = np.einsum('nik,nk->ni', Tb, Vm)
            for amp in amps:
                for sgn in (+1.0, -1.0):
                    if time.time() - t0 > HARD - 40:
                        break
                    Y = X0 + sgn * amp * DX
                    Y /= np.linalg.norm(Y, axis=1, keepdims=True)
                    Gy = Y @ Y.T
                    np.fill_diagonal(Gy, -2.0)
                    my = float(np.max(Gy))
                    # exploding guard
                    if my > m0 + 0.01:
                        continue
                    # near-max contacts w.r.t. own max within 1e-7? use global ref m0-1e-7 and own
                    cnt = int(np.count_nonzero(Gy > (my - 1e-7)))
                    cands.append((cnt, my, mi, sgn, amp, Y))
                if time.time() - t0 > HARD - 40:
                    break
            if time.time() - t0 > HARD - 40:
                break
        # retain three moves that most reduce contacts without exploding max
        cands.sort(key=lambda t: (t[0], t[1]))
        top3 = cands[:3]
        quenched = []
        for (cnt, my, mi, sgn, amp, Y) in top3:
            if time.time() - t0 > HARD - 35:
                break
            Yq = smooth_polish(Y, steps=600, a0=40.0, a1=400.0, lr=0.003)
            vy = max_cos(Yq)
            quenched.append((vy, cnt, Yq, (mi, sgn, amp)))
        if quenched:
            quenched.sort(key=lambda t: t[0])
            best_q_val, _, best_q, _ = quenched[0]
            # reserve one capped 25s LP polish for the best
            rem = HARD - (time.time() - t0) - 1.0
            lp_sec = min(25.0, max(0.0, rem))
            if lp_sec > 2.0:
                best_q_lp = lp_polish(best_q, seconds=lp_sec, radius_init=3e-3)
                v_lp = max_cos(best_q_lp)
                if v_lp < best_val - 1e-12:
                    best_val = v_lp
                    best_x = best_q_lp.copy()
                # also consider unpolished quenches
                for (vy, _, Yq, _) in quenched:
                    if vy < best_val - 1e-12:
                        best_val = vy
                        best_x = Yq.copy()
            else:
                for (vy, _, Yq, _) in quenched:
                    if vy < best_val - 1e-12:
                        best_val = vy
                        best_x = Yq.copy()
    except Exception:
        pass
    # hard deadline: return incumbent if no improvement or time low
    if best_val >= incumbent_val - 1e-12:
        # return incumbent (ensure normalized)
        incumbent_x = incumbent_x / np.linalg.norm(incumbent_x, axis=1, keepdims=True)
        return incumbent_x
    # enforce deadline
    if time.time() - t0 > HARD:
        incumbent_x = incumbent_x / np.linalg.norm(incumbent_x, axis=1, keepdims=True)
        return incumbent_x
    best_x = best_x / np.linalg.norm(best_x, axis=1, keepdims=True)
    return best_x
