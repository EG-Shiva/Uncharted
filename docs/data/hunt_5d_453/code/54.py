import time
import numpy as np

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def _tangent_bases(X):
    n, d = X.shape
    B = np.empty((n, d, 4), dtype=np.float64)
    for i in range(n):
        x = X[i]
        k = int(np.argmin(np.abs(x)))
        T = np.empty((d, 4), dtype=np.float64)
        c = 0
        for j in range(d):
            if j == k:
                continue
            v = -x[j] * x.copy()
            v[j] += 1.0
            T[:, c] = v
            c += 1
        Q, _ = np.linalg.qr(T)
        B[i] = Q[:, :4]
    return B

def solve():
    t0 = time.time()
    T_END = t0 + 172.0
    NULL_END = t0 + 132.0
    POLISH_END = t0 + 168.0
    np.random.seed(0)
    rng = np.random.default_rng(0)
    n, d = 453, 5
    # load incumbent
    cands = []
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            cands.append(np.asarray(a, dtype=np.float64))
        except Exception:
            pass
    bestX = None
    bestV = float("inf")
    for cc in cands:
        try:
            if cc.shape != (453, 5):
                continue
            cc = cc / np.linalg.norm(cc, axis=1, keepdims=True)
            if not np.all(np.isfinite(cc)):
                continue
            vv = _max_cos(cc)
            if vv < bestV:
                bestV = vv
                bestX = cc.copy()
        except Exception:
            continue
    if bestX is None:
        bestX = rng.standard_normal((n, d))
        bestX /= np.linalg.norm(bestX, axis=1, keepdims=True)
        bestV = _max_cos(bestX)
    curX = bestX.copy()
    curV = float(bestV)
    iu_r, iu_c = np.triu_indices(n, 1)

    def retract(X, U, s):
        Y = X + s * U
        nr = np.sqrt(np.sum(Y * Y, axis=1, keepdims=True))
        nr[nr == 0] = 1.0
        return Y / nr

    def smooth_run(x, N, tau0, tau1, lr0, lr1, mu, deadline):
        v = np.zeros_like(x)
        try:
            ltau = np.log(tau1 / tau0)
        except Exception:
            ltau = 0.0
        try:
            llr = np.log(lr1 / lr0)
        except Exception:
            llr = 0.0
        for it in range(N):
            if it % 40 == 0 and time.time() > deadline:
                break
            frac = it / max(1, N - 1)
            tau = tau0 * np.exp(ltau * frac)
            lr = lr0 * np.exp(llr * frac)
            G = x @ x.T
            np.fill_diagonal(G, -2.0)
            m = float(np.max(G))
            with np.errstate(over='ignore', invalid='ignore'):
                P = np.exp((G - m) / tau)
            np.fill_diagonal(P, 0.0)
            P = np.where(np.isfinite(P), P, 0.0)
            Grad = P @ x
            dot = np.sum(Grad * x, axis=1, keepdims=True)
            Gt = Grad - dot * x
            v = mu * v + (1.0 - mu) * Gt
            # normalize gradient scale
            gn = float(np.sqrt(np.sum(Gt * Gt))) + 1e-30
            # keep lr as is (tuned); clip huge steps
            x = x - lr * v / max(1.0, gn * 0.05)
            xn = np.sqrt(np.sum(x * x, axis=1, keepdims=True))
            xn[xn == 0] = 1.0
            x = x / xn
        return x

    # ---------- second-order null-space exploration ----------
    try:
        # build rigidity matrix for contacts slack<1e-9
        G0 = curX @ curX.T
        np.fill_diagonal(G0, -2.0)
        m0 = float(np.max(G0))
        c_all = G0[iu_r, iu_c]
        slack = m0 - c_all
        order = np.argsort(slack)
        thr = 1e-9
        tight_idx = np.where(slack < thr)[0]
        if tight_idx.size > 2500:
            tight_idx = order[:2500]
        if tight_idx.size < 400:
            tight_idx = order[:1500]
        p = int(tight_idx.size)
        I = iu_r[tight_idx]
        J = iu_c[tight_idx]
        B = _tangent_bases(curX)
        Nvar = n * 4
        # dense R p x Nvar
        R = np.zeros((p, Nvar), dtype=np.float64)
        # vectorized fill via einsum per block is fine
        # compute aI, aJ
        BI = B[I]  # p x5x4
        BJ = B[J]
        XJ = curX[J]
        XI = curX[I]
        aI = np.einsum('pdk,pd->pk', BI, XJ)
        aJ = np.einsum('pdk,pd->pk', BJ, XI)
        ar = np.arange(p)
        # fill rows
        # loop is okay for 1700 rows (fast enough) but vectorize with advanced indexing
        # Use flat assignment
        colsI = (I[:, None] * 4 + np.arange(4)[None, :])
        colsJ = (J[:, None] * 4 + np.arange(4)[None, :])
        # scatter
        R[ar[:, None], colsI] = aI
        R[ar[:, None], colsJ] = aJ
        # symmetric Gram M = R^T R
        M = R.T @ R
        # symmetrize
        M = 0.5 * (M + M.T)
        w, V = np.linalg.eigh(M)  # ascending
        Kdim = 90
        Kdim = min(Kdim, Nvar - 1)
        Vn = V[:, :Kdim]  # Nvar x K
        # shell pairs: slack in [5e-7, 1e-4]
        shell_mask = (slack >= 5e-7) & (slack <= 1e-4)
        shell_idx = np.where(shell_mask)[0]
        # cap to tightest 800 of shell
        if shell_idx.size > 800:
            # tightest = smallest slack among shell
            sub = slack[shell_idx]
            part = np.argpartition(sub, 800)[:800]
            shell_idx = shell_idx[part]
        SI = iu_r[shell_idx]
        SJ = iu_c[shell_idx]
        # precompute shell rows sparse (q x Nvar) dense for up to 800x1812 = 1.4M fine
        q = int(shell_idx.size)
        Rshell = None
        if q > 0:
            Rshell = np.zeros((q, Nvar), dtype=np.float64)
            BSI = B[SI]
            BSJ = B[SJ]
            XSJ = curX[SJ]
            XSI = curX[SI]
            bI = np.einsum('pdk,pd->pk', BSI, XSJ)
            bJ = np.einsum('pdk,pd->pk', BSJ, XSI)
            colsSI = (SI[:, None] * 4 + np.arange(4)[None, :])
            colsSJ = (SJ[:, None] * 4 + np.arange(4)[None, :])
            aqr = np.arange(q)
            Rshell[aqr[:, None], colsSI] = bI
            Rshell[aqr[:, None], colsSJ] = bJ
        # contact count baseline
        base_contacts = int(p)
        # null-space trial loop
        steps_grid = np.array([0.25, 0.6, 1.1, 2.0], dtype=np.float64)
        n_outer = 0
        # need U mapping helper
        def avec_to_U(a):
            A = a.reshape(n, 4)
            # U_i = B_i @ A_i
            U = np.einsum('ndk,nk->nd', B, A)
            # project again for safety (remove radial)
            dot = np.sum(U * curX, axis=1, keepdims=True)
            U -= dot * curX
            return U
        # for curvature refresh, rebuild B/R occasionally (every accepted step)
        while time.time() < NULL_END - 1.0 and n_outer < 25:
            n_outer += 1
            if time.time() > NULL_END - 1.0:
                break
            # sample directions in null space
            n_rand = 36
            n_targ = 16 if q > 0 else 0
            n_dir = n_rand + n_targ
            Alphas = np.empty((n_dir, Kdim), dtype=np.float64)
            for k2 in range(n_rand):
                Alphas[k2] = rng.standard_normal(Kdim)
            # targeted: pick random shell rows, project
            if n_targ > 0:
                # projection of shell rows onto null space: Vn.T @ Rshell.T  -> K x q
                # compute on the fly per pick to save time
                picks = rng.integers(0, q, size=n_targ)
                for t_i, pk in enumerate(picks):
                    rrow = Rshell[pk]  # Nvar
                    proj = Vn.T @ rrow  # K
                    pn = float(np.linalg.norm(proj))
                    if pn < 1e-12:
                        Alphas[n_rand + t_i] = rng.standard_normal(Kdim)
                    else:
                        # direction decreasing that shell pair: -proj, mixed with random
                        rnd = rng.standard_normal(Kdim)
                        # orthogonalize a bit? just mix
                        alpha = -proj / pn * 0.75 + rnd / (float(np.linalg.norm(rnd)) + 1e-12) * 0.25 * np.sqrt(float(Kdim))
                        Alphas[n_rand + t_i] = alpha
            # normalize alphas
            for k2 in range(n_dir):
                nn = float(np.linalg.norm(Alphas[k2]))
                if nn == 0 or not np.isfinite(nn):
                    Alphas[k2] = rng.standard_normal(Kdim)
                    nn = float(np.linalg.norm(Alphas[k2])) + 1e-30
                Alphas[k2] /= nn
            # map to U's (n_dir x n x d) -> evaluate stepwise
            # to save memory, iterate dirs
            best_improve = 0.0
            best_Xtry = None
            best_Vtry = curV
            best_cont_try = base_contacts
            # topology candidate (no raise, more contacts)
            topo_X = None
            topo_C = base_contacts
            topo_V = curV
            for di in range(n_dir):
                if time.time() > NULL_END - 0.5:
                    break
                a = Vn @ Alphas[di]  # Nvar
                # normalize a to unit norm
                an = float(np.linalg.norm(a)) + 1e-30
                a = a / an
                U = avec_to_U(a)
                # scale U to unit Frobenius/sqrt(n)? keep as is (norm ~1)
                for s in steps_grid:
                    Xtry = retract(curX, U, float(s))
                    # quick max
                    G = Xtry @ Xtry.T
                    np.fill_diagonal(G, -2.0)
                    mt = float(np.max(G))
                    if mt < best_Vtry - 1e-12:
                        # count contacts for tie-break? compute cheap count
                        # only count if improving
                        best_Vtry = mt
                        best_Xtry = Xtry
                        # estimate contacts
                        try:
                            cc = G[iu_r, iu_c]
                            cnt = int(np.sum(cc >= mt - 1e-9))
                        except Exception:
                            cnt = base_contacts
                        best_cont_try = cnt
                    elif mt <= curV + 2e-9:
                        # topology candidate: more contacts?
                        # avoid full count for every s (expensive); only if mt close
                        try:
                            cc = G[iu_r, iu_c]
                            cnt = int(np.sum(cc >= mt - 1e-9))
                        except Exception:
                            cnt = 0
                        if cnt > topo_C + 20 and mt <= topo_V + 2e-9:
                            # require meaningful increase
                            topo_C = cnt
                            topo_X = Xtry
                            topo_V = mt
            improved = False
            if best_Xtry is not None and best_Vtry < curV - 3e-10:
                curX = best_Xtry
                curV = float(best_Vtry)
                base_contacts = int(best_cont_try)
                improved = True
                if curV < bestV:
                    bestV = float(curV)
                    bestX = curX.copy()
                # short polish after improvement (time-capped 2.5s)
                try:
                    dl = min(time.time() + 2.5, NULL_END, POLISH_END)
                    curX = smooth_run(curX, 250, 0.004, 0.0006, 0.004, 0.0005, 0.86, dl)
                    nv = _max_cos(curX)
                    if nv < curV:
                        curV = nv
                    if nv < bestV:
                        bestV = nv
                        bestX = curX.copy()
                except Exception:
                    pass
                # rebuild B, R, Vn after move (refresh rigidity)
                try:
                    if time.time() < NULL_END - 3.0:
                        G0 = curX @ curX.T
                        np.fill_diagonal(G0, -2.0)
                        m0 = float(np.max(G0))
                        c_all = G0[iu_r, iu_c]
                        slack = m0 - c_all
                        order = np.argsort(slack)
                        tight_idx = np.where(slack < 1e-9)[0]
                        if tight_idx.size > 2500:
                            tight_idx = order[:2500]
                        if tight_idx.size < 400:
                            tight_idx = order[:1500]
                        p = int(tight_idx.size)
                        I = iu_r[tight_idx]
                        J = iu_c[tight_idx]
                        B = _tangent_bases(curX)
                        R = np.zeros((p, Nvar), dtype=np.float64)
                        BI = B[I]; BJ = B[J]
                        XJ = curX[J]; XI = curX[I]
                        aI = np.einsum('pdk,pd->pk', BI, XJ)
                        aJ = np.einsum('pdk,pd->pk', BJ, XI)
                        ar = np.arange(p)
                        colsI = (I[:, None] * 4 + np.arange(4)[None, :])
                        colsJ = (J[:, None] * 4 + np.arange(4)[None, :])
                        R[ar[:, None], colsI] = aI
                        R[ar[:, None], colsJ] = aJ
                        M = R.T @ R
                        M = 0.5 * (M + M.T)
                        w, V = np.linalg.eigh(M)
                        Vn = V[:, :Kdim]
                        base_contacts = p
                        shell_mask = (slack >= 5e-7) & (slack <= 1e-4)
                        shell_idx = np.where(shell_mask)[0]
                        if shell_idx.size > 800:
                            sub = slack[shell_idx]
                            part = np.argpartition(sub, 800)[:800]
                            shell_idx = shell_idx[part]
                        SI = iu_r[shell_idx]; SJ = iu_c[shell_idx]
                        q = int(shell_idx.size)
                        if q > 0:
                            Rshell = np.zeros((q, Nvar), dtype=np.float64)
                            BSI = B[SI]; BSJ = B[SJ]
                            XSJ = curX[SJ]; XSI = curX[SI]
                            bI = np.einsum('pdk,pd->pk', BSI, XSJ)
                            bJ = np.einsum('pdk,pd->pk', BSJ, XSI)
                            colsSI = (SI[:, None] * 4 + np.arange(4)[None, :])
                            colsSJ = (SJ[:, None] * 4 + np.arange(4)[None, :])
                            aqr = np.arange(q)
                            Rshell[aqr[:, None], colsSI] = bI
                            Rshell[aqr[:, None], colsSJ] = bJ
                        else:
                            Rshell = None
                except Exception:
                    pass
            elif topo_X is not None:
                # accept topology change without raising max
                curX = topo_X
                curV = float(topo_V)
                base_contacts = int(topo_C)
                # short polish capped
                try:
                    dl = min(time.time() + 1.5, NULL_END, POLISH_END)
                    curX = smooth_run(curX, 150, 0.003, 0.0006, 0.003, 0.0005, 0.86, dl)
                    nv = _max_cos(curX)
                    if nv < curV:
                        curV = nv
                    if nv < bestV:
                        bestV = nv
                        bestX = curX.copy()
                    else:
                        # keep topology move even if polish slightly worse but <= old best+tol
                        if curV <= bestV + 2e-9:
                            pass
                except Exception:
                    pass
                # rebuild rigidity after topology move
                try:
                    if time.time() < NULL_END - 3.0:
                        G0 = curX @ curX.T
                        np.fill_diagonal(G0, -2.0)
                        m0 = float(np.max(G0))
                        c_all = G0[iu_r, iu_c]
                        slack = m0 - c_all
                        order = np.argsort(slack)
                        tight_idx = np.where(slack < 1e-9)[0]
                        if tight_idx.size > 2500:
                            tight_idx = order[:2500]
                        if tight_idx.size < 400:
                            tight_idx = order[:1500]
                        p = int(tight_idx.size)
                        I = iu_r[tight_idx]
                        J = iu_c[tight_idx]
                        B = _tangent_bases(curX)
                        R = np.zeros((p, Nvar), dtype=np.float64)
                        BI = B[I]; BJ = B[J]
                        XJ = curX[J]; XI = curX[I]
                        aI = np.einsum('pdk,pd->pk', BI, XJ)
                        aJ = np.einsum('pdk,pd->pk', BJ, XI)
                        ar = np.arange(p)
                        colsI = (I[:, None] * 4 + np.arange(4)[None, :])
                        colsJ = (J[:, None] * 4 + np.arange(4)[None, :])
                        R[ar[:, None], colsI] = aI
                        R[ar[:, None], colsJ] = aJ
                        M = R.T @ R
                        M = 0.5 * (M + M.T)
                        w, V = np.linalg.eigh(M)
                        Vn = V[:, :Kdim]
                        base_contacts = p
                        shell_mask = (slack >= 5e-7) & (slack <= 1e-4)
                        shell_idx = np.where(shell_mask)[0]
                        if shell_idx.size > 800:
                            sub = slack[shell_idx]
                            part = np.argpartition(sub, 800)[:800]
                            shell_idx = shell_idx[part]
                        SI = iu_r[shell_idx]; SJ = iu_c[shell_idx]
                        q = int(shell_idx.size)
                        if q > 0:
                            Rshell = np.zeros((q, Nvar), dtype=np.float64)
                            BSI = B[SI]; BSJ = B[SJ]
                            XSJ = curX[SJ]; XSI = curX[SI]
                            bI = np.einsum('pdk,pd->pk', BSI, XSJ)
                            bJ = np.einsum('pdk,pd->pk', BSJ, XSI)
                            colsSI = (SI[:, None] * 4 + np.arange(4)[None, :])
                            colsSJ = (SJ[:, None] * 4 + np.arange(4)[None, :])
                            aqr = np.arange(q)
                            Rshell[aqr[:, None], colsSI] = bI
                            Rshell[aqr[:, None], colsSJ] = bJ
                        else:
                            Rshell = None
                except Exception:
                    pass
            else:
                # no move found: randomize shell targeting width or break
                # small random kick in null space to try new region (without raising much)
                try:
                    alpha = rng.standard_normal(Kdim)
                    alpha /= (float(np.linalg.norm(alpha)) + 1e-30)
                    a = Vn @ alpha
                    a /= (float(np.linalg.norm(a)) + 1e-30)
                    U = avec_to_U(a)
                    Xtry = retract(curX, U, 0.4)
                    mt = _max_cos(Xtry)
                    if mt <= curV + 5e-06 and time.time() < NULL_END - 2.0:
                        curX = smooth_run(Xtry, 120, 0.004, 0.0008, 0.003, 0.0005, 0.86, min(time.time()+1.0, NULL_END))
                        nv = _max_cos(curX)
                        if nv < bestV:
                            bestV = nv
                            bestX = curX.copy()
                        curV = nv
                except Exception:
                    pass
                # if still stuck, continue to polish phase
                if n_outer >= 6 and not improved and topo_X is None:
                    # perturb Vn sampling next round will differ; keep going a bit
                    pass
    except Exception:
        pass

    # ---------- final 30s polishing of incumbent ----------
    try:
        x = bestX.copy()
        # short diverse anneals, always keep best
        # phase A
        if time.time() < POLISH_END:
            x = smooth_run(x, 600, 0.004, 0.0004, 0.004, 0.0003, 0.86, POLISH_END - 6.0)
            vv = _max_cos(x)
            if vv < bestV:
                bestV = vv
                bestX = x.copy()
            else:
                x = bestX.copy()
        # phase B: tiny shakes + re-anneal (basin hopping, elitist)
        it = 0
        while time.time() < POLISH_END - 0.8:
            it += 1
            # shake strength decaying
            sigma = 0.002 if (it % 2 == 0) else 0.005
            y = bestX.copy()
            nz = rng.standard_normal((n, d)) * sigma
            nz -= np.sum(nz * y, axis=1, keepdims=True) * y
            y = y + nz
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            y = smooth_run(y, 160, 0.003, 0.0005, 0.003, 0.0004, 0.86, POLISH_END - 0.4)
            vv = _max_cos(y)
            if vv < bestV:
                bestV = vv
                bestX = y.copy()
            if POLISH_END - time.time() < 1.0:
                break
            if it > 40:
                break
    except Exception:
        pass
    # normalize strictly
    try:
        bestX = bestX / np.linalg.norm(bestX, axis=1, keepdims=True)
        bestX = np.where(np.isfinite(bestX), bestX, 0.0)
        # fix any zero rows
        nr = np.linalg.norm(bestX, axis=1)
        for i in range(n):
            if nr[i] < 1e-12 or not np.isfinite(nr[i]):
                v = rng.standard_normal(d)
                v /= (np.linalg.norm(v) + 1e-30)
                bestX[i] = v
        bestX = bestX / np.linalg.norm(bestX, axis=1, keepdims=True)
    except Exception:
        pass
    return np.asarray(bestX, dtype=np.float64)
