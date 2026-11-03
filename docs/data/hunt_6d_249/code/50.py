import time
import numpy as np
from scipy.optimize import minimize, linprog
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def solve():
    t0 = time.time()
    T_END = t0 + 172.0
    FINAL_RESERVE = 28.0
    rng = np.random.default_rng(7)
    n, d = 249, 6
    I0, J0 = np.triu_indices(n, 1)
    P = len(I0)

    # load incumbent
    cands = []
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=float)
            if a.shape == (n, d):
                a = a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-12)
                if np.all(np.isfinite(a)):
                    cands.append(a)
        except Exception:
            pass
    if not cands:
        a = rng.standard_normal(size=(n, d))
        a /= np.linalg.norm(a, axis=1, keepdims=True)
        cands.append(a)
    vals = [_max_cos(a) for a in cands]
    bestX = cands[int(np.argmin(vals))].copy()
    bestVal = float(np.min(vals))
    bestX /= np.maximum(np.linalg.norm(bestX, axis=1, keepdims=True), 1e-12)

    # ---------- symmetry configs ----------
    def mk_flip(k):
        D = np.eye(6)
        for i in range(k, 6):
            D[i, i] = -1.0
        E = np.zeros((6, k))
        for i in range(k):
            E[i, i] = 1.0
        return D, E
    def mk_swap_halves():
        D = np.zeros((6, 6))
        D[0:3, 3:6] = np.eye(3)
        D[3:6, 0:3] = np.eye(3)
        E = np.zeros((6, 3))
        E[0:3, :] = np.eye(3)
        E[3:6, :] = np.eye(3)
        return D, E
    def mk_swap01():
        D = np.eye(6)
        D[[0, 1]] = D[[1, 0]]
        E = np.zeros((6, 5))
        E[0, 0] = 1.0; E[1, 0] = 1.0
        E[2, 1] = 1.0; E[3, 2] = 1.0; E[4, 3] = 1.0; E[5, 4] = 1.0
        return D, E
    def mk_c3():
        Pm = np.zeros((6, 6))
        # cycle 0->1->2->0 : column action P e0 = e1 etc.
        Pm[1, 0] = 1; Pm[2, 1] = 1; Pm[0, 2] = 1
        Pm[4, 3] = 1; Pm[5, 4] = 1; Pm[3, 5] = 1
        return Pm

    configs = []
    for k in (3, 2, 1):
        D, E = mk_flip(k)
        configs.append({"name": f"flip{k}", "kind": "inv", "D": D, "E": E, "nrep": 124, "dfix": k})
    Dsh, Esh = mk_swap_halves()
    configs.append({"name": "swaphalf", "kind": "inv", "D": Dsh, "E": Esh, "nrep": 124, "dfix": 3})
    Ds01, Es01 = mk_swap01()
    configs.append({"name": "swap01", "kind": "inv", "D": Ds01, "E": Es01, "nrep": 124, "dfix": 5})
    Pm = mk_c3()
    configs.append({"name": "c3double", "kind": "c3", "P": Pm, "P2": Pm @ Pm, "nrep": 83, "dfix": 0})

    def build_Xraw(cfg, z):
        if cfg["kind"] == "inv":
            nr = cfg["nrep"]; df = cfg["dfix"]
            Y = z[:nr*6].reshape(nr, 6)
            w = z[nr*6:].reshape(df) if df > 0 else None
            Xraw = np.empty((n, 6))
            Xraw[:nr] = Y
            Xraw[nr:2*nr] = Y @ cfg["D"].T
            if df > 0:
                Xraw[2*nr] = (cfg["E"] @ w)
            else:
                Xraw[2*nr] = Y[0]  # never used
            return Xraw, Y, w
        else:
            nr = cfg["nrep"]
            Y = z.reshape(nr, 6)
            Xraw = np.empty((n, 6))
            Xraw[:nr] = Y
            Xraw[nr:2*nr] = Y @ cfg["P"].T
            Xraw[2*nr:] = Y @ cfg["P2"].T
            # 83*3=249 exactly
            return Xraw, Y, None

    def expand_normalized(cfg, z):
        Xraw, _, _ = build_Xraw(cfg, z)
        rn = np.maximum(np.linalg.norm(Xraw, axis=1, keepdims=True), 1e-12)
        # guard fixed row zero
        if not np.all(np.isfinite(rn)):
            rn = np.ones_like(rn)
        Xn = Xraw / rn
        # if fixed row was degenerate, replace
        if cfg["kind"] == "inv":
            nr = cfg["nrep"]
            if float(np.linalg.norm(Xraw[2*nr])) < 1e-9:
                v = rng.standard_normal(6); v /= np.linalg.norm(v)
                # project to fixed subspace: average with D
                v = v + cfg["D"] @ v
                if np.linalg.norm(v) < 1e-9:
                    v = np.ones(6); v = v + cfg["D"] @ v
                v /= np.linalg.norm(v)
                Xn[2*nr] = v
        Xn /= np.maximum(np.linalg.norm(Xn, axis=1, keepdims=True), 1e-12)
        return Xn

    def sym_fg(z, cfg, tau, creg=0.08):
        Xraw, _, _ = build_Xraw(cfg, z)
        r = np.sqrt(np.sum(Xraw*Xraw, axis=1))
        r = np.maximum(r, 1e-12)
        Xn = Xraw / r[:, None]
        G = Xn @ Xn.T
        g = G[I0, J0]
        gm = float(np.max(g))
        e = np.exp((g - gm) / tau)
        S = float(np.sum(e))
        Lsoft = gm + tau*np.log(S)
        w_ = e / S
        W = np.zeros((n, n))
        W[I0, J0] = w_
        W = W + W.T
        A = W @ Xn
        proj = np.sum(A*Xn, axis=1)
        G0 = (A - proj[:, None]*Xn) / r[:, None]
        diff = r - 1.0
        L = Lsoft + creg*np.sum(diff*diff)
        Gr = G0 + (2.0*creg*diff)[:, None]*Xn
        # map to dz
        if cfg["kind"] == "inv":
            nr = cfg["nrep"]; df = cfg["dfix"]
            G1 = Gr[:nr] + Gr[nr:2*nr] @ cfg["D"]
            dz = np.empty_like(z)
            dz[:nr*6] = G1.ravel()
            if df > 0:
                dz[nr*6:] = (cfg["E"].T @ Gr[2*nr])
            return float(L), dz
        else:
            G_a = Gr[:83]; G_b = Gr[83:166]; G_c = Gr[166:249]
            dY = G_a + G_b @ cfg["P"] + G_c @ cfg["P2"]
            return float(L), dY.ravel()

    def sym_anneal(cfg, z0, taus, iters, creg=0.08):
        z = z0.copy()
        for tau, mi in zip(taus, iters):
            if time.time() > T_END - FINAL_RESERVE:
                break
            def fg(zz, tau=float(tau)):
                return sym_fg(zz, cfg, tau, creg)
            try:
                res = minimize(fg, z, jac=True, method='L-BFGS-B',
                               options={'maxiter': int(mi), 'maxfun': int(mi*4),
                                        'ftol': 1e-14, 'gtol': 1e-7, 'maxcor': 8})
                z = res.x
            except Exception:
                break
        return z

    def full_anneal(Xinit, taus, iters, creg=0.08):
        xf = Xinit.ravel().copy()
        for tau, mi in zip(taus, iters):
            if time.time() > T_END - FINAL_RESERVE:
                break
            tau = float(tau)
            def fg(xflat, tau=tau):
                X = xflat.reshape(n, d)
                r = np.sqrt(np.sum(X*X, axis=1))
                r = np.maximum(r, 1e-12)
                Xn = X / r[:, None]
                G = Xn @ Xn.T
                g = G[I0, J0]
                gm = float(np.max(g))
                e = np.exp((g-gm)/tau)
                S = float(np.sum(e))
                Lsoft = gm + tau*np.log(S)
                w_ = e/S
                W = np.zeros((n, n))
                W[I0, J0] = w_
                W = W + W.T
                A = W @ Xn
                proj = np.sum(A*Xn, axis=1)
                G0 = (A - proj[:, None]*Xn)/r[:, None]
                diff = r-1.0
                L = Lsoft + creg*np.sum(diff*diff)
                Gr = G0 + (2.0*creg*diff)[:, None]*Xn
                return float(L), Gr.ravel()
            try:
                res = minimize(fg, xf, jac=True, method='L-BFGS-B',
                               options={'maxiter': int(mi), 'maxfun': int(mi*4),
                                        'ftol': 1e-14, 'gtol': 1e-7, 'maxcor': 8})
                xf = res.x
            except Exception:
                break
            try:
                Xm = xf.reshape(n, d)
                rn = np.maximum(np.linalg.norm(Xm, axis=1, keepdims=True), 1e-12)
                xf = (Xm/rn).ravel()
            except Exception:
                pass
        Xo = xf.reshape(n, d)
        Xo /= np.maximum(np.linalg.norm(Xo, axis=1, keepdims=True), 1e-12)
        return Xo

    def tangent_bases(X):
        T = np.empty((n, 6, 5))
        E = np.eye(6)
        for i in range(n):
            a = X[i]
            s = -1.0 if a[0] >= 0 else 1.0
            v = a.copy(); v[0] -= s
            nv = float(np.linalg.norm(v))
            if nv < 1e-12:
                T[i] = E[:, 1:]
            else:
                w = (v/nv).reshape(6, 1)
                H = E - 2.0*(w @ w.T)
                T[i] = H[:, 1:]
        return T

    def lp_jam(Xinit, deadline, radius_init=2.5e-3, slack=2.5e-3, max_it=40):
        nonlocal bestX, bestVal
        X = Xinit.copy()
        X /= np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
        mcur = _max_cos(X)
        if mcur < bestVal:
            bestVal = mcur; bestX = X.copy()
        radius = float(radius_init)
        Nv = n*5
        it = 0
        fails = 0
        while it < max_it and time.time() < deadline:
            it += 1
            if radius < 1e-12:
                break
            G = X @ X.T
            np.fill_diagonal(G, -2.0)
            m = float(np.max(G))
            gfull = G[I0, J0]
            sel = np.where(gfull > m - slack)[0]
            if len(sel) == 0:
                sel = np.argsort(gfull)[-800:]
            if len(sel) > 3500:
                part = np.argpartition(gfull[sel], -3500)[-3500:]
                sel = sel[part]
            Ic = I0[sel]; Jc = J0[sel]; cc = gfull[sel]
            K = len(sel)
            T = tangent_bases(X)
            TI = T[Ic]; TJ = T[Jc]
            XJ = X[Jc]; XI = X[Ic]
            Ai = np.sum(TI * XJ[:, :, None], axis=1)
            Aj = np.sum(TJ * XI[:, :, None], axis=1)
            colsA = np.concatenate([(Ic[:, None]*5+np.arange(5)).ravel(),
                                    (Jc[:, None]*5+np.arange(5)).ravel(),
                                    np.full(K, Nv)])
            r1 = np.repeat(np.arange(K), 5)
            rows = np.concatenate([r1, r1, np.arange(K)])
            data = np.concatenate([Ai.ravel(), Aj.ravel(), -np.ones(K)])
            try:
                Aub = coo_matrix((data, (rows, cols)), shape=(K, Nv+1)).tocsr()
            except Exception:
                radius *= 0.5; continue
            bub = -cc
            cvec = np.zeros(Nv+1); cvec[Nv] = 1.0
            bnds = [(-radius, radius)]*Nv + [(None, None)]
            try:
                res = linprog(cvec, A_ub=Aub, b_ub=bub, bounds=bnds, method='highs',
                              options={'presolve': True})
            except Exception:
                radius *= 0.5; fails += 1
                if fails > 5: break
                continue
            if res.status != 0 or res.x is None:
                radius *= 0.5; fails += 1
                if fails > 8: break
                continue
            fails = 0
            y = res.x[:Nv].reshape(n, 5)
            tpred = float(res.x[Nv])
            step = np.einsum('nkm,nm->nk', T, y)
            Xn = X + step
            Xn /= np.maximum(np.linalg.norm(Xn, axis=1, keepdims=True), 1e-12)
            mnew = _max_cos(Xn)
            # accept if improves or predicted improves
            if mnew < m + 1e-12:
                X = Xn
                if mnew < bestVal:
                    bestVal = mnew; bestX = X.copy()
                if tpred < m - 1e-9:
                    radius = min(radius*1.3, 8e-3)
                else:
                    radius *= 0.7
                # adapt slack near end
                if it % 10 == 0:
                    slack = max(slack*0.6, 4e-4)
            else:
                radius *= 0.5
                fails += 1
                if fails > 6:
                    # rattler kick: jitter worst point
                    try:
                        G2 = X @ X.T; np.fill_diagonal(G2, -2)
                        deg = (G2 > m - 1e-3).sum(axis=1)
                        # kick a low-degree point
                        idx = int(np.argmin(deg + rng.random(n)*0.5))
                        X[idx] = X[idx] + 0.002*rng.standard_normal(6)
                        X[idx] /= np.linalg.norm(X[idx])
                    except Exception:
                        pass
                    fails = 0
                    radius = max(radius, 5e-4)
        return X

    # ---------- symmetric search then break ----------
    taus_sym = [0.02, 0.01, 0.005]
    iters_sym = [30, 30, 40]
    taus_break = [0.005, 0.002, 0.001]
    iters_break = [50, 60, 70]

    broken_cands = []
    # budget symmetric phase
    sym_deadline = t0 + 95.0
    for cfg in configs:
        if time.time() > min(sym_deadline, T_END - FINAL_RESERVE - 30):
            break
        # two random symmetric starts per config (one for c3 to save time -> 2 as well)
        nstart = 2 if cfg["name"] != "c3double" else 2
        for s in range(nstart):
            if time.time() > min(sym_deadline, T_END - FINAL_RESERVE - 30):
                break
            nr = cfg["nrep"]; df = cfg["dfix"]
            if cfg["kind"] == "inv":
                Y0 = rng.standard_normal(size=(nr, 6))
                w0 = rng.standard_normal(size=(df,)) if df > 0 else np.zeros(0)
                z0 = np.concatenate([Y0.ravel(), w0])
            else:
                z0 = rng.standard_normal(size=(nr*6,))
            try:
                zopt = sym_anneal(cfg, z0, taus_sym, iters_sym, creg=0.08)
                Xsym = expand_normalized(cfg, zopt)
                vsym = _max_cos(Xsym)
            except Exception:
                continue
            # retain family even if slightly worse: break symmetry
            for sig in (0.004,):
                if time.time() > T_END - FINAL_RESERVE - 15:
                    break
                try:
                    Xb0 = Xsym + sig*rng.standard_normal(size=(n, 6))
                    Xb0 /= np.maximum(np.linalg.norm(Xb0, axis=1, keepdims=True), 1e-12)
                    Xb = full_anneal(Xb0, taus_break, iters_break, creg=0.06)
                    vb = _max_cos(Xb)
                    broken_cands.append((vb, Xb.copy(), cfg["name"]))
                    if vb < bestVal:
                        bestVal = vb; bestX = Xb.copy()
                except Exception:
                    pass
            # also try zero-jitter release (exact symmetric start, symmetry breaks via numerics)
            if time.time() < T_END - FINAL_RESERVE - 20:
                try:
                    Xb0 = Xsym + 1e-4*rng.standard_normal(size=(n, 6))
                    Xb0 /= np.maximum(np.linalg.norm(Xb0, axis=1, keepdims=True), 1e-12)
                    Xb = full_anneal(Xb0, [0.002, 0.001, 0.0005], [60, 70, 60], creg=0.05)
                    vb = _max_cos(Xb)
                    broken_cands.append((vb, Xb.copy(), cfg["name"]+"_fine"))
                    if vb < bestVal:
                        bestVal = vb; bestX = Xb.copy()
                except Exception:
                    pass

    # also polish incumbent lightly so we never regress
    # final selection: top broken + incumbent
    broken_cands = sorted(broken_cands, key=lambda t: t[0])[:3]
    final_pool = [(bestVal, bestX.copy())] + [(v, X.copy()) for (v, X, _) in broken_cands]
    # deduplicate roughly and fine anneal best two
    taus_fine = [0.002, 0.001, 0.0005, 0.00025]
    iters_fine = [70, 90, 90, 70]
    cand_final = []
    for (vv, XX) in final_pool[:3]:
        if time.time() > T_END - FINAL_RESERVE + 8:
            break
        try:
            Xf = full_anneal(XX, taus_fine, iters_fine, creg=0.04)
            vf = _max_cos(Xf)
            cand_final.append((vf, Xf))
            if vf < bestVal:
                bestVal = vf; bestX = Xf.copy()
        except Exception:
            pass
    # LP polishing on the very best
    try:
        order = sorted(cand_final + [(bestVal, bestX)], key=lambda t: t[0])
        Xlp0 = order[0][1]
        deadline = T_END - 3.0
        if time.time() < deadline - 5:
            lp_jam(Xlp0, deadline)
    except Exception:
        pass

    bestX /= np.maximum(np.linalg.norm(bestX, axis=1, keepdims=True), 1e-12)
    # tiny shrink guard (no-op, ensures finite)
    bestX = np.where(np.isfinite(bestX), bestX, 0.0)
    rn = np.linalg.norm(bestX, axis=1, keepdims=True)
    rn = np.maximum(rn, 1e-12)
    bestX = bestX / rn
    return bestX
