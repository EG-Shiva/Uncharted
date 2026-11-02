import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _proj(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return x / n

def _lp_polish(x, seconds):
    x = _proj(x)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        G = x @ x.T
        c = G[iu]
        m = float(c.max())
        sel = c > m - max(20 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        if p == 0:
            break
        if p > 2500:
            idx = np.argpartition(-c[sel], 2500)[:2500]
            I, J = I[idx], J[idx]
            csv = c[sel][idx]
            p = 2500
        else:
            csv = c[sel]
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
            obj = np.zeros(nv); obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-csv, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                          method="highs", options={"time_limit": 2.0})
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y = _proj(y)
        v = _max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.02)
        else:
            radius *= 0.5
    return x

def _adam_screen(y, steps=200, lr0=0.02, T=0.003, mu=0.8):
    n, d = y.shape
    vel = np.zeros((n, d))
    for s in range(steps):
        G = y @ y.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        E = (G - m) / T
        np.clip(E, -20, 0, out=E)
        W = np.exp(E)
        np.fill_diagonal(W, 0.0)
        rs = W.sum(axis=1, keepdims=True) + 1e-12
        grad = W @ y
        dot = np.einsum('ij,ij->i', grad, y)
        grad -= dot[:, None] * y
        grad /= rs
        # clip
        gn = np.linalg.norm(grad) + 1e-12
        if gn > 5.0:
            grad *= 5.0 / gn
        vel = mu * vel + grad
        lr = lr0 * (1.0 - 0.6 * s / steps)
        y = y - lr * vel
        y = _proj(y)
    return y

def _tangent_bases(x):
    n, d = x.shape
    dm = d - 1
    Tb = np.empty((n, d, dm))
    Eye = np.eye(d)
    for i in range(n):
        xi = x[i]
        amax = int(np.argmax(np.abs(xi)))
        cols = [c for c in range(d) if c != amax]
        if len(cols) > dm:
            cols = cols[:dm]
        B = Eye[:, cols].copy()  # d x dm
        # orthogonalize against xi: B - xi*(xi^T B)
        coeff = xi @ B  # dm
        B -= np.outer(xi, coeff)
        Q, _ = np.linalg.qr(B)
        # Q is d x dm orthonormal, orthogonal to xi (up to sign)
        Tb[i] = Q
    return Tb

def _soft_modes(x, Tb, I, J, K, rng):
    n, d = x.shape
    dm = d - 1
    N = n * dm
    p = len(I)
    if p == 0:
        R = rng.normal(size=(K, N))
        R /= np.linalg.norm(R, axis=1, keepdims=True) + 1e-12
        return R
    pc = min(p, 800)
    if pc < p:
        # keep tightest: need values; recompute
        G = x @ x.T
        vals = G[I, J]
        idx = np.argpartition(-vals, pc)[:pc]
        I, J = I[idx], J[idx]
        p = pc
    Jmat = np.zeros((p, N), dtype=float)
    # T^T x : dm vectors
    # loop
    for k in range(p):
        i, j = int(I[k]), int(J[k])
        bi = Tb[i].T @ x[j]  # dm
        bj = Tb[j].T @ x[i]
        Jmat[k, i * dm:(i + 1) * dm] = bi
        Jmat[k, j * dm:(j + 1) * dm] = bj
    # nullspace via projection: V = R - C A^{-1} J
    R = rng.normal(size=(K, N))
    # normalize rows
    # C = R J^T  K x p
    C = R @ Jmat.T
    A = Jmat @ Jmat.T
    A += np.eye(p) * 1e-8
    try:
        Yt = np.linalg.solve(A, C.T)  # p x K
    except Exception:
        Yt = np.linalg.lstsq(A, C.T, rcond=None)[0]
    Y = Yt.T  # K x p
    V = R - Y @ Jmat
    nv = np.linalg.norm(V, axis=1, keepdims=True) + 1e-12
    V /= nv
    return V

def _config_dist(a, b):
    Ga = a @ a.T
    Gb = b @ b.T
    return float(np.linalg.norm(Ga - Gb))

def solve():
    T0 = time.time()
    DEADLINE = 152.0
    rng = np.random.default_rng(0)
    try:
        b0 = np.load("best.npy")
    except Exception:
        b0 = np.load("record.npy")
    try:
        r0 = np.load("record.npy")
    except Exception:
        r0 = b0
    b0 = _proj(np.nan_to_num(np.asarray(b0, dtype=float)))
    r0 = _proj(np.nan_to_num(np.asarray(r0, dtype=float)))
    if b0.shape != (54, 21):
        b0 = _proj(rng.normal(size=(54, 21)))
    if r0.shape != (54, 21):
        r0 = b0.copy()
    bv = _max_cos(b0); rv = _max_cos(r0)
    best = b0.copy() if bv <= rv else r0.copy()
    bv = min(bv, rv)
    n, d = 54, 21
    dm = d - 1

    def left():
        return DEADLINE - (time.time() - T0)

    # brief initial LP to sit at optimum (max 6s)
    if left() > 40:
        best = _lp_polish(best, min(6.0, left() - 35.0))
        bv = _max_cos(best)

    elite = [(bv, best.copy())]
    # contact threshold for Jacobian
    contact_tol = 1e-6
    Kmodes = 16
    it = 0
    amps = [0.01, 0.03, 0.07, 0.12, 0.20]
    while left() > 14:
        it += 1
        # pick base round-robin from elite for diversity
        base_idx = (it - 1) % len(elite)
        base = elite[base_idx][1]
        G = base @ base.T
        np.fill_diagonal(G, -2.0)
        m0 = float(G.max())
        iu0 = np.where(G > m0 - contact_tol)
        # keep only i<j
        mask = iu0[0] < iu0[1]
        I0, J0 = iu0[0][mask], iu0[1][mask]
        cnt0 = len(I0)
        if len(I0) == 0:
            Tb = _tangent_bases(base)
            V = rng.normal(size=(Kmodes, n * dm))
            V /= np.linalg.norm(V, axis=1, keepdims=True) + 1e-12
        else:
            Tb = _tangent_bases(base)
            try:
                V = _soft_modes(base, Tb, I0, J0, Kmodes, rng)
            except Exception:
                V = rng.normal(size=(Kmodes, n * dm))
                V /= np.linalg.norm(V, axis=1, keepdims=True) + 1e-12
        # try a few random combos
        ntry = 4
        cands = []
        for t in range(ntry):
            if time.time() - T0 > DEADLINE - 14:
                break
            coeff = rng.normal(size=Kmodes)
            coeff /= np.linalg.norm(coeff) + 1e-12
            u = coeff @ V  # N
            U = u.reshape(n, dm)  # tangent coeffs
            # map to ambient: dx_i = Tb_i @ U_i
            dx = np.einsum('idk,ik->id', Tb, U)
            # tangent already; remove any normal drift
            dx -= np.einsum('ij,ij->i', dx, base)[:, None] * base
            dn = np.linalg.norm(dx) + 1e-12
            dx /= dn
            a = amps[int(rng.integers(0, len(amps)))]
            # second-order prediction: max change approx a*lin + 0.5*a^2*quad
            # cheap: just step
            y = base + a * dx
            y = _proj(y)
            # cheap Adam screening 200 steps
            ys = _adam_screen(y.copy(), steps=200, lr0=0.02, T=0.003, mu=0.8)
            vs = _max_cos(ys)
            # contact change
            Gs = ys @ ys.T
            np.fill_diagonal(Gs, -2.0)
            ms = float(Gs.max())
            cnt = int(np.sum(Gs > ms - contact_tol) // 2)
            cands.append((vs, ys, cnt, ms))
        if not cands:
            break
        cands.sort(key=lambda t: t[0])
        vs_best, ys_best, cnt_best, ms_best = cands[0]
        # acceptance: material contact change or predicted decrease
        contact_change = abs(cnt_best - cnt0) / max(1, cnt0)
        # diversity vs elite
        dmin = min(_config_dist(ys_best, e[1]) for e in elite)
        accept_lp = (vs_best < bv + 0.004) and (contact_change > 0.03 or vs_best < bv + 1e-7)
        if dmin < 0.15:
            # too close to known basin, skip LP unless improving
            if vs_best >= bv - 1e-9:
                accept_lp = False
        if accept_lp and left() > 6:
            polished = _lp_polish(ys_best, min(1.5, left() - 10.0 if left() > 12 else left() - 2.0))
            vp = _max_cos(polished)
            if vp < bv - 1e-12:
                best = polished.copy(); bv = vp
                elite.append((vp, polished.copy()))
                elite.sort(key=lambda t: t[0])
                elite = elite[:5]
            else:
                # keep diverse near-miss in pool
                if dmin > 0.25 and len(elite) < 5:
                    elite.append((vp, polished.copy()))
                    elite.sort(key=lambda t: t[0])
                elif dmin > 0.4:
                    # replace worst if diverse
                    elite[-1] = (vp, polished.copy())
                    elite.sort(key=lambda t: t[0])
        # occasional: if screened beats best without LP (rare), LP it next iter anyway
        # stop if time low
        if left() <= 12:
            break

    # final squeeze of best with remaining time (leave 1s margin)
    rem = DEADLINE - (time.time() - T0) - 1.0
    # also respect 155 internal: cap at 8s
    if rem > 0.5:
        best = _lp_polish(best, min(rem, 8.0))
    best = _proj(best)
    return best
