import time
import numpy as np

def solve():
    t0 = time.time()
    T_END = t0 + 172.0
    T_EXP = T_END - 30.0
    rng = np.random.default_rng(0)
    np.random.seed(0)
    n, d = 453, 5
    Nvar = n * 4

    def max_cos(X):
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        return float(np.max(G))

    def tangent_bases(X):
        B = np.empty((n, d, 4), dtype=np.float64)
        for i in range(n):
            x = X[i]
            k = int(np.argmin(np.abs(x)))
            T = np.empty((d, 4), dtype=np.float64)
            c = 0
            for j in range(d):
                if j == k:
                    continue
                v = -x[j] * x
                v[j] += 1.0
                T[:, c] = v
                c += 1
            Q, _ = np.linalg.qr(T)
            B[i] = Q[:, :4]
        return B

    # load incumbent
    cands = []
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=np.float64)
            if a.shape == (453, 5):
                nn = np.linalg.norm(a, axis=1, keepdims=True)
                nn[nn == 0] = 1.0
                a = a / nn
                if np.all(np.isfinite(a)):
                    cands.append(a)
        except Exception:
            pass
    if len(cands) == 0:
        a = rng.standard_normal((n, d))
        a /= np.linalg.norm(a, axis=1, keepdims=True)
        cands.append(a)
    vals = sorted([(max_cos(c), c) for c in cands], key=lambda t: t[0])
    best_val = float(vals[0][0])
    best_X = vals[0][1].copy()
    cur_X = best_X.copy()
    cur_val = best_val

    iu_r, iu_c = np.triu_indices(n, 1)
    ar4 = np.arange(4)

    def build_R(X, I, J, B):
        p = int(I.size)
        aI = np.einsum('pdk,pd->pk', B[I], X[J])
        aJ = np.einsum('pdk,pd->pk', B[J], X[I])
        R = np.zeros((p, Nvar), dtype=np.float64)
        # fill rows: could be vectorized via advanced indexing
        # loop in python is slow for 1700; use row-wise scatter with broadcasting
        # use flat indexing
        rows = np.repeat(np.arange(p), 8)
        cols_I = (I[:, None] * 4 + ar4[None, :]).ravel()
        cols_J = (J[:, None] * 4 + ar4[None, :]).ravel()
        cols = np.concatenate([cols_I, cols_J])
        # need matching rows: first 4 per row are I, next 4 are J
        # rows above interleaves; rebuild correctly:
        # simpler loop in numba-free but p~1700 loop of 8 assigns is fine
        for k in range(p):
            bi = int(I[k]) * 4
            bj = int(J[k]) * 4
            R[k, bi:bi+4] = aI[k]
            R[k, bj:bj+4] = aJ[k]
        return R

    def retract(X, B, v, s):
        # v: (Nvar,) max-per-point norm =1
        V = np.empty_like(X)
        for i in range(n):
            V[i] = B[i] @ v[i*4:(i+1)*4]
        Y = X + s * V
        Y /= np.linalg.norm(Y, axis=1, keepdims=True)
        return Y

    def softmax_polish(X, steps, tau0, tau1, lr0, lr1, mu, t_dead):
        nonlocal best_val, best_X
        x = X.copy()
        v = np.zeros_like(x)
        ltau = np.log(tau1 / tau0)
        llr = np.log(lr1 / lr0)
        for it in range(steps):
            if it % 40 == 0 and time.time() > t_dead:
                break
            frac = it / max(steps - 1, 1)
            tau = tau0 * np.exp(ltau * frac)
            lr = lr0 * np.exp(llr * frac)
            G = x @ x.T
            np.fill_diagonal(G, -2.0)
            m = float(np.max(G))
            if m < best_val:
                best_val = m
                best_X = x.copy()
            P = np.exp((G - m) / tau)
            np.fill_diagonal(P, 0.0)
            Grad = P @ x
            Gt = Grad - np.sum(Grad * x, axis=1, keepdims=True) * x
            v = mu * v + (1.0 - mu) * Gt
            x = x - lr * v
            x /= np.linalg.norm(x, axis=1, keepdims=True)
        return x

    # ---------- null-space exploration ----------
    n_iter = 0
    step_sizes = [0.003, 0.008, 0.015, 0.030]
    K_CAND = 40
    while time.time() < T_EXP - 2.0 and n_iter < 40:
        n_iter += 1
        if time.time() > T_EXP - 2.0:
            break
        X = cur_X.copy()
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        c_all = G[iu_r, iu_c]
        m0 = float(np.max(c_all))
        slack = m0 - c_all
        # contacts slack <1e-9
        cidx = np.where(slack < 1e-9)[0]
        if cidx.size > 2500:
            part = np.argpartition(slack[cidx], 2500)[:2500]
            cidx = cidx[part]
        if cidx.size < 50:
            # under-contacted: take tightest 800 as contacts
            cidx = np.argsort(slack)[:800]
        p = int(cidx.size)
        I = iu_r[cidx]
        J = iu_c[cidx]
        # next shell 5e-7 to 1e-4
        shell_mask = (slack >= 5e-7) & (slack <= 1e-4)
        shell_idx = np.where(shell_mask)[0]
        if shell_idx.size > 800:
            # take smallest slack in shell
            part = np.argpartition(slack[shell_idx], 800)[:800]
            shell_idx = shell_idx[part]
        B = tangent_bases(X)
        # build rigidity matrix R (p x Nvar)
        try:
            R = build_R(X, I, J, B)
        except Exception:
            R = None
        Vcands = []
        if R is not None and p > 0:
            try:
                S = R @ R.T
                # regularize
                tr = float(np.trace(S)) / max(p, 1)
                reg = max(1e-9 * tr, 1e-12)
                S = S + reg * np.eye(p)
                # Cholesky factor for repeated solves
                try:
                    L = np.linalg.cholesky(S)
                    def solve_S(b):
                        y = np.linalg.solve(L, b)
                        y = np.linalg.solve(L.T, y)
                        return y
                    use_chol = True
                except Exception:
                    use_chol = False
                    # fallback to lstsq via solve
                    def solve_S(b):
                        try:
                            return np.linalg.solve(S, b)
                        except Exception:
                            return np.linalg.lstsq(S, b, rcond=None)[0]
                for _ in range(K_CAND):
                    if time.time() > T_EXP - 1.0:
                        break
                    r = rng.standard_normal(Nvar)
                    b = R @ r
                    try:
                        y = solve_S(b)
                    except Exception:
                        continue
                    vnull = r - R.T @ y
                    # check null residual
                    rn = float(np.linalg.norm(vnull))
                    if rn < 1e-9 or not np.isfinite(rn):
                        continue
                    # normalize to max per-point norm =1
                    seg = vnull.reshape(n, 4)
                    pn = np.sqrt(np.sum(seg * seg, axis=1))
                    mx = float(np.max(pn))
                    if mx < 1e-12 or not np.isfinite(mx):
                        continue
                    vnull = vnull / mx
                    # residual check
                    res = float(np.linalg.norm(R @ vnull) / max(1.0, float(np.linalg.norm(vnull))))
                    if res > 0.05:
                        continue
                    Vcands.append(vnull)
            except Exception:
                Vcands = []
        if len(Vcands) == 0:
            # fallback: random tangent directions (rattler-biased)
            for _ in range(K_CAND):
                r = rng.standard_normal(Nvar)
                seg = r.reshape(n, 4)
                pn = np.sqrt(np.sum(seg * seg, axis=1))
                mx = float(np.max(pn)) + 1e-12
                Vcands.append(r / mx)

        # trial retractions at several step sizes, both signs
        best_trial_val = m0
        best_trial_X = None
        best_trial_contacts = p
        # current contacts count
        for v in Vcands:
            if time.time() > T_EXP - 0.5:
                break
            for sgn in (1.0, -1.0):
                vv = sgn * v
                # quick linear shell drive estimate (optional): skip if clearly bad?
                for s in step_sizes:
                    if time.time() > T_EXP - 0.5:
                        break
                    try:
                        Y = retract(X, B, vv, s)
                    except Exception:
                        continue
                    Gy = Y @ Y.T
                    np.fill_diagonal(Gy, -2.0)
                    cy = Gy[iu_r, iu_c]
                    my = float(np.max(cy))
                    if not np.isfinite(my):
                        continue
                    # count contacts
                    ncon = int(np.sum((my - cy) < 1e-9))
                    accept = False
                    if my < best_trial_val - 1e-12:
                        accept = True
                    elif my <= m0 + 2e-12 and ncon > best_trial_contacts and my <= best_trial_val + 2e-12:
                        accept = True
                    elif my < m0 - 1e-12:
                        accept = True
                    if accept:
                        # require either improvement over m0 or new contact without raising
                        if my < m0 - 5e-13 or (my <= m0 + 2e-12 and ncon > p):
                            best_trial_val = my
                            best_trial_X = Y
                            best_trial_contacts = ncon
        if best_trial_X is not None:
            # accept step
            cur_X = best_trial_X
            cur_val = best_trial_val
            if cur_val < best_val:
                best_val = cur_val
                best_X = cur_X.copy()
            else:
                # equal max but more jammed: keep as cur for next iteration
                pass
            # short time-capped LP-like polish (softmax)
            rem = min(4.0, max(0.5, T_EXP - time.time() - 0.5))
            softmax_polish(cur_X, steps=300, tau0=0.004, tau1=0.0005,
                           lr0=4e-4, lr1=5e-5, mu=0.8, t_dead=time.time() + rem)
            # refresh cur to best if best improved else keep jammed cur?
            # keep cur as best if best is close, else continue from best
            if best_val < cur_val - 1e-9:
                cur_X = best_X.copy()
                cur_val = best_val
            else:
                # re-evaluate cur after polish? keep best as cur to not drift
                # if polish improved best, use it
                cur_X = best_X.copy()
                cur_val = best_val
            continue
        else:
            # no null step accepted: try rattler/hub hole surgery to change topology
            try:
                Gr = X @ X.T
                np.fill_diagonal(Gr, -2.0)
                rmax = Gr.max(axis=1)
                # contact degrees
                con_mat = (m0 - Gr) < 1e-9
                np.fill_diagonal(con_mat, False)
                deg = con_mat.sum(axis=1)
                # pick rattlers first, else worst point
                ratt = np.where(deg == 0)[0]
                if ratt.size > 0:
                    worst = int(ratt[int(rng.integers(0, len(ratt)))] )
                else:
                    worst = int(np.argmax(rmax))
                # largest hole search
                C = rng.standard_normal((8000, d))
                C /= np.linalg.norm(C, axis=1, keepdims=True)
                D = C @ X.T
                D[:, worst] = -2.0
                cmax = D.max(axis=1)
                bi = int(np.argmin(cmax))
                Y = X.copy()
                Y[worst] = C[bi]
                my = max_cos(Y)
                if my <= m0 + 5e-4:
                    # short anneal from Y
                    rem2 = min(3.0, max(0.5, T_EXP - time.time() - 0.5))
                    # quick softmax from Y
                    xh = Y.copy()
                    vh = np.zeros_like(xh)
                    for it in range(250):
                        if time.time() > T_EXP - 0.5:
                            break
                        tau = 0.003 * (0.3 ** (it / 249))
                        lr = 4e-4 * (0.2 ** (it / 249))
                        Gh = xh @ xh.T
                        np.fill_diagonal(Gh, -2.0)
                        mh = float(np.max(Gh))
                        if mh < best_val:
                            best_val = mh
                            best_X = xh.copy()
                        P = np.exp((Gh - mh) / tau)
                        np.fill_diagonal(P, 0.0)
                        Gr2 = P @ xh
                        Gt = Gr2 - np.sum(Gr2 * xh, axis=1, keepdims=True) * xh
                        vh = 0.8 * vh + 0.2 * Gt
                        xh = xh - lr * vh
                        xh /= np.linalg.norm(xh, axis=1, keepdims=True)
                    cur_X = best_X.copy()
                    cur_val = best_val
                    continue
            except Exception:
                pass
            # if still stuck, small random shake of hubs + continue
            try:
                Y = X.copy()
                sig = 0.004
                nz = rng.standard_normal(Y.shape) * sig
                dot = np.sum(nz * Y, axis=1, keepdims=True)
                nz = nz - dot * Y
                Y = Y + nz
                Y /= np.linalg.norm(Y, axis=1, keepdims=True)
                cur_X = Y
                cur_val = max_cos(cur_X)
                if cur_val < best_val:
                    best_val = cur_val
                    best_X = cur_X.copy()
            except Exception:
                pass
            # loop again (new random null subspace next iter)
            continue

    # ---------- final 30s polish / return incumbent ----------
    # reserve: polish best with softmax + greedy shave
    def greedy_shave(x, steps, t_dead):
        nonlocal best_val, best_X
        eps = 1e-3
        for it in range(steps):
            if it % 30 == 0 and time.time() > t_dead:
                break
            G = x @ x.T
            np.fill_diagonal(G, -2.0)
            m = float(G.max())
            if m < best_val:
                best_val = m
                best_X = x.copy()
            flat = int(np.argmax(G))
            i = flat // n
            j = flat % n
            cij = float(x[i] @ x[j])
            ti = x[j] - cij * x[i]
            ni = float(np.linalg.norm(ti))
            if ni < 1e-12:
                continue
            ti /= ni
            tj = x[i] - cij * x[j]
            nj = float(np.linalg.norm(tj))
            if nj < 1e-12:
                continue
            tj /= nj
            improved = False
            for scale in (eps, eps * 0.5, eps * 0.25):
                vi = x[i] - scale * ti
                vi /= np.linalg.norm(vi)
                vj = x[j] - scale * tj
                vj /= np.linalg.norm(vj)
                oi = x[i].copy(); oj = x[j].copy()
                x[i] = vi; x[j] = vj
                g2 = x @ x.T
                np.fill_diagonal(g2, -2.0)
                m2 = float(g2.max())
                if m2 < m - 1e-12:
                    if m2 < best_val:
                        best_val = m2
                        best_X = x.copy()
                    eps = min(scale * 1.08, 3e-3)
                    improved = True
                    break
                else:
                    x[i] = oi; x[j] = oj
            if not improved:
                eps *= 0.96
                if eps < 5e-07:
                    eps = 5e-07
        return x

    # final polish from best
    try:
        x = best_X.copy()
        # stage 1 softmax
        softmax_polish(x, steps=2500, tau0=0.005, tau1=0.0004, lr0=5e-4, lr1=4e-5,
                       mu=0.82, t_dead=T_END - 6.0)
        x = best_X.copy()
        # stage 2 shave interleaved if time
        if time.time() < T_END - 6.0:
            greedy_shave(x, steps=600, t_dead=T_END - 2.0)
        x = best_X.copy()
        if time.time() < T_END - 2.0:
            softmax_polish(x, steps=1500, tau0=0.0015, tau1=0.0002, lr0=2e-4, lr1=2e-5,
                           mu=0.85, t_dead=T_END - 0.5)
    except Exception:
        pass
    # ensure normalized, finite
    Xout = best_X.copy()
    Xout /= np.linalg.norm(Xout, axis=1, keepdims=True)
    # tiny shrink safety not needed; just return
    return Xout
