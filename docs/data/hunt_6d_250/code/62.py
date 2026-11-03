import time
import numpy as np
from scipy.optimize import minimize, linprog
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def _joint_opt(S_idx, Xfull, betas, maxiters, t_deadline=None):
    import time as _tm
    n, d = Xfull.shape
    k = len(S_idx)
    if k == 0:
        return Xfull
    S_idx = np.asarray(S_idx, dtype=np.int64)
    mask = np.ones(n, dtype=bool)
    mask[S_idx] = False
    F_idx = np.where(mask)[0]
    XF = Xfull[F_idx]  # (nf,d)
    nf = XF.shape[0]
    # precompute triu indices for SxS
    iu0, iu1 = np.triu_indices(k, 1)
    has_ss = iu0.size > 0
    Y0 = Xfull[S_idx].copy()
    Yflat = Y0.ravel().copy()
    for bi, beta in enumerate(betas):
        if t_deadline is not None and _tm.time() > t_deadline - 1.0:
            break
        mi = maxiters[bi] if bi < len(maxiters) else maxiters[-1]
        def fg(v):
            Y = v.reshape(k, d)
            nrm = np.sqrt(np.sum(Y*Y, axis=1))
            nrm = np.maximum(nrm, 1e-12)
            XS = Y / nrm[:, None]
            # build g
            if has_ss:
                gss = np.sum(XS[iu0]*XS[iu1], axis=1)
            else:
                gss = np.empty(0, dtype=np.float64)
            Gsf = XS @ XF.T  # k x nf
            gsf = Gsf.ravel()
            if has_ss:
                g = np.concatenate([gss, gsf])
            else:
                g = gsf
            M = float(np.max(g))
            D = beta*(g - M)
            # D<=0 so safe
            E = np.exp(D)
            Ssum = float(np.sum(E))
            if Ssum <= 0 or not np.isfinite(Ssum):
                Ssum = 1e-300
            loss = M + np.log(Ssum)/beta
            W = E / Ssum
            if has_ss:
                pss = gss.shape[0]
                Wss = W[:pss]
                Wsf = W[pss:].reshape(k, nf)
            else:
                Wsf = W.reshape(k, nf)
                Wss = None
            GradXS = Wsf @ XF  # k,d
            if has_ss:
                # symmetric matrix
                Wmat = np.zeros((k, k), dtype=np.float64)
                Wmat[iu0, iu1] = Wss
                Wmat[iu1, iu0] = Wss
                GradXS += Wmat @ XS
            dot = np.sum(GradXS*XS, axis=1, keepdims=True)
            GY = (GradXS - dot*XS) / nrm[:, None]
            return loss, GY.ravel()
        try:
            res = minimize(fg, Yflat, jac=True, method="L-BFGS-B",
                           options={"maxiter": int(mi), "ftol": 1e-20,
                                    "gtol": 1e-9, "maxfun": int(mi)*5,
                                    "maxcor": 10})
            Yflat = res.x
        except Exception:
            break
    Y = Yflat.reshape(k, d)
    nrm = np.linalg.norm(Y, axis=1, keepdims=True)
    nrm = np.maximum(nrm, 1e-12)
    XS = Y / nrm
    Xnew = Xfull.copy()
    Xnew[S_idx] = XS
    return Xnew

def _pair_refine(p1, p2, B, beta=5000.0, maxiter=60):
    # B: (nb,6) blockers, optimise 2 vectors minimising LSE of max to B and each other
    Y0 = np.stack([p1, p2], axis=0)
    nb = B.shape[0]
    def fg(v):
        Y = v.reshape(2, 6)
        nrm = np.sqrt(np.sum(Y*Y, axis=1))
        nrm = np.maximum(nrm, 1e-12)
        X = Y / nrm[:, None]
        d0 = B @ X[0]
        d1 = B @ X[1]
        d01 = float(X[0] @ X[1])
        g = np.empty(2*nb+1, dtype=np.float64)
        g[:nb] = d0
        g[nb:2*nb] = d1
        g[2*nb] = d01
        M = float(np.max(g))
        D = beta*(g-M)
        np.clip(D, -30.0, 0.0, out=D)
        E = np.exp(D)
        Ssum = float(np.sum(E))
        if Ssum <= 0 or not np.isfinite(Ssum):
            Ssum = 1e-300
        loss = M + np.log(Ssum)/beta
        W = E/Ssum
        w0 = W[:nb]; w1 = W[nb:2*nb]; w01 = float(W[2*nb])
        gx0 = w01*X[1] + (w0 @ B)
        gx1 = w01*X[0] + (w1 @ B)
        GX = np.stack([gx0, gx1], axis=0)
        Xc = X
        dot = np.sum(GX*Xc, axis=1, keepdims=True)
        GY = (GX - dot*Xc)/nrm[:, None]
        return loss, GY.ravel()
    try:
        res = minimize(fg, Y0.ravel(), jac=True, method="L-BFGS-B",
                       options={"maxiter": int(maxiter), "ftol": 1e-20,
                                "gtol": 1e-10, "maxfun": int(maxiter)*5,
                                "maxcor": 8})
        Y = res.x.reshape(2, 6)
    except Exception:
        Y = Y0
    nrm = np.linalg.norm(Y, axis=1, keepdims=True)
    nrm = np.maximum(nrm, 1e-12)
    return Y/nrm

def _lp_polish5(X, Mcur, t_deadline, max_iters=5, window=1e-5):
    import time as _tm
    n, d = X.shape
    iu0, iu1 = np.triu_indices(n, 1)
    ar = np.arange(d)
    radius = 2e-3
    cur = X.copy()
    curM = float(Mcur)
    for it in range(max_iters):
        if _tm.time() > t_deadline - 1.2:
            break
        if radius < 1e-13:
            break
        Gall = cur @ cur.T
        c_all = Gall[iu0, iu1]
        m = float(np.max(c_all))
        # keep exact curM in sync
        if m < curM:
            curM = m
        else:
            m = curM
        sel = c_all >= m - window
        cnt = int(np.count_nonzero(sel))
        if cnt == 0:
            break
        if cnt > 4000:
            # take top 4000
            idx = np.argpartition(-c_all, 4000)[:4000]
            I = iu0[idx]; J = iu1[idx]; c_sel = c_all[idx]
        else:
            I = iu0[sel]; J = iu1[sel]; c_sel = c_all[sel]
        p = I.shape[0]
        nv = n*d+1
        try:
            cols = np.concatenate([I[:, None]*d+ar, J[:, None]*d+ar,
                                   np.full((p,1), n*d)], axis=1).ravel()
            rows = np.repeat(np.arange(p), 2*d+1)
            vals = np.concatenate([cur[J], cur[I], -np.ones((p,1))], axis=1).ravel()
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((cur.ravel(), (np.repeat(np.arange(n), d),
                                              np.arange(n*d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv); obj[-1] = 1.0
            bounds = [(-radius, radius)]*(n*d)+[(None, float(m))]
            res = linprog(obj, A_ub=A_ub, b_ub=-c_sel, A_eq=A_eq,
                          b_eq=np.zeros(n), bounds=bounds, method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0 or res.x is None:
            radius *= 0.5
            continue
        delta = res.x[:-1].reshape(n, d)
        if not np.all(np.isfinite(delta)):
            radius *= 0.5
            continue
        improved = False
        for alpha in (1.0, 0.5, 0.25, 0.125):
            Y = cur + float(alpha)*delta
            nrm = np.linalg.norm(Y, axis=1, keepdims=True)
            nrm = np.maximum(nrm, 1e-12)
            Y = Y/nrm
            v = _max_cos(Y)
            if v < curM - 1e-15:
                cur = Y; curM = v
                improved = True
                break
        if improved:
            radius = min(radius*1.5, 0.012)
        else:
            radius *= 0.5
    return cur, curM

def solve():
    import time as _tm
    t0 = _tm.time()
    BUDGET = 168.0
    LOOP_END = t0 + 155.0
    n, d = 250, 6
    best = None; best_val = np.inf
    for fn in ("best.npy", "record.npy"):
        try:
            z = np.asarray(np.load(fn), dtype=np.float64)
            if z.shape == (n, d):
                zn = np.linalg.norm(z, axis=1, keepdims=True)
                zn = np.maximum(zn, 1e-12)
                z = z/zn
                if np.all(np.isfinite(z)):
                    v = _max_cos(z)
                    if v < best_val:
                        best_val = v; best = z.copy()
        except Exception:
            continue
    if best is None:
        rng0 = np.random.default_rng(0)
        best = rng0.standard_normal((n, d))
        best /= np.maximum(np.linalg.norm(best, axis=1, keepdims=True), 1e-12)
        best_val = _max_cos(best)
    Xbest = best.copy(); Mbest = float(best_val)
    iu0_all, iu1_all = np.triu_indices(n, 1)
    outer = 0
    # betas for main joint optimisation
    BETAS = [1e3, 3e3, 1e4, 3e4]
    # first: try direct joint opt of S once (basin equalisation)
    first_done = False
    while _tm.time() < LOOP_END and outer < 60:
        if _tm.time() > LOOP_END - 6.0:
            break
        # find bottleneck pair (a,b)
        try:
            G = Xbest @ Xbest.T
            np.fill_diagonal(G, -2.0)
            flat = np.argmax(G)
            a = int(flat // n); b = int(flat % n)
            if a == b:
                # fallback via triu
                gv = (Xbest @ Xbest.T)[iu0_all, iu1_all]
                k = int(np.argmax(gv)); a = int(iu0_all[k]); b = int(iu1_all[k])
            Mcur = float(G[a, b])
            Mbest = float(np.max(G))
        except Exception:
            break
        # build S: union of points with cos > M-1e-4 with a or b
        try:
            Ga = Xbest @ Xbest[a]
            Gb = Xbest @ Xbest[b]
            thr = Mbest - 1e-4
            maskS = (Ga >= thr) | (Gb >= thr)
            S_idx = np.where(maskS)[0]
            # ensure a,b included
            sset = set(S_idx.tolist()); sset.add(a); sset.add(b)
            S_idx = np.array(sorted(sset), dtype=np.int64)
            # cap to 50, pad to >=22
            if S_idx.size > 50:
                scores = np.maximum(Ga[S_idx], Gb[S_idx])
                # keep a,b + top 48
                order = np.argsort(-scores)
                keep = set([a, b])
                for o in order:
                    keep.add(int(S_idx[o]))
                    if len(keep) >= 50:
                        break
                S_idx = np.array(sorted(keep), dtype=np.int64)
            if S_idx.size < 22:
                # pad with nearest to a/b
                scores_all = np.maximum(Ga, Gb)
                order_all = np.argsort(-scores_all)
                keep = set(S_idx.tolist())
                for o in order_all:
                    keep.add(int(o))
                    if len(keep) >= 22:
                        break
                S_idx = np.array(sorted(keep), dtype=np.int64)
        except Exception:
            S_idx = np.array([a, b], dtype=np.int64)
        # 1) direct joint reoptimisation (only first time per basin)
        if not first_done:
            first_done = True
            try:
                Xcand = _joint_opt(S_idx, Xbest, BETAS, [120, 120, 120, 90], LOOP_END)
                vcand = _max_cos(Xcand)
                if vcand < Mbest - 1e-15:
                    Xcand2, v2 = _lp_polish5(Xcand, vcand, LOOP_END, 5, 1e-5)
                    if v2 < Mbest - 1e-15:
                        Xbest = Xcand2; Mbest = float(v2)
                    else:
                        Xbest = Xcand; Mbest = float(vcand)
                    first_done = False  # new basin, allow again
                    outer += 1
                    continue
            except Exception:
                pass
        # 2) deep-hole pair surgery for a,b
        seed = 12345 + outer*7919
        rng = np.random.default_rng(seed)
        # blockers = all except a,b
        keep_mask = np.ones(n, dtype=bool); keep_mask[a] = False; keep_mask[b] = False
        Bmat = Xbest[keep_mask]  # 248x6
        Npairs = 20000
        try:
            P1 = rng.standard_normal((Npairs, d)); P1 /= np.maximum(np.linalg.norm(P1, axis=1, keepdims=True), 1e-12)
            P2 = rng.standard_normal((Npairs, d)); P2 /= np.maximum(np.linalg.norm(P2, axis=1, keepdims=True), 1e-12)
        except Exception:
            break
        try:
            D1 = P1 @ Bmat.T  # 20000x248
            m1 = D1.max(axis=1)
            D2 = P2 @ Bmat.T
            m2 = D2.max(axis=1)
            m12 = np.sum(P1*P2, axis=1)
            score = np.maximum(np.maximum(m1, m2), m12)
            best20_idx = np.argpartition(score, 20)[:20]
            # sort them
            best20_idx = best20_idx[np.argsort(score[best20_idx])]
            P1b = P1[best20_idx]; P2b = P2[best20_idx]
        except Exception:
            outer += 1
            continue
        # refine best 20 pairs with small LSE opt
        refined = []
        for k2 in range(20):
            if _tm.time() > LOOP_END - 5.0:
                break
            try:
                q = _pair_refine(P1b[k2], P2b[k2], Bmat, 5000.0, 60)
                # score refined
                d0 = float(np.max(Bmat @ q[0]))
                d1 = float(np.max(Bmat @ q[1]))
                d01 = float(q[0] @ q[1])
                sc = d0 if d0 > d1 else d1
                if d01 > sc: sc = d01
                refined.append((sc, q[0].copy(), q[1].copy()))
            except Exception:
                continue
        if len(refined) == 0:
            outer += 1
            continue
        refined.sort(key=lambda t: t[0])
        # try full S joint opt for each refined pair (in order), accept if improves
        improved_this_round = False
        # limit full joints to maybe 6 best to save time, but ensure we test at least promising ones
        # we test all 20 with quick eval first, then full joint on top 6
        # quick eval global max without joint
        quick_order = refined  # already sorted by blocker score
        n_full = min(6, len(quick_order))
        # also consider: if blocker score already worse than Mbest, joint unlikely to help, but still try top ones
        for k2 in range(n_full):
            if _tm.time() > LOOP_END - 4.0:
                break
            sc, q0, q1 = quick_order[k2]
            try:
                Xcand0 = Xbest.copy()
                Xcand0[a] = q0/np.maximum(np.linalg.norm(q0), 1e-12)
                Xcand0[b] = q1/np.maximum(np.linalg.norm(q1), 1e-12)
                # short joint optimisation of S starting from this
                Xcand = _joint_opt(S_idx, Xcand0, [3e3, 1e4], [80, 80], LOOP_END)
                v = _max_cos(Xcand)
                if v < Mbest - 1e-15:
                    Xc2, v2 = _lp_polish5(Xcand, v, LOOP_END, 5, 1e-5)
                    if v2 < Mbest - 1e-15:
                        Xbest = Xc2; Mbest = float(v2)
                    else:
                        Xbest = Xcand; Mbest = float(v)
                    improved_this_round = True
                    first_done = False
                    break
            except Exception:
                continue
        if improved_this_round:
            outer += 1
            continue
        # 3) opposite tangent / difference-direction displacements of a,b + short joint
        try:
            Xa = Xbest[a].copy(); Xb = Xbest[b].copy()
            # difference direction tangents
            diff = Xa - Xb
            ta_d = diff - float(diff @ Xa)*Xa
            ntd = float(np.linalg.norm(ta_d))
            if ntd > 1e-12: ta_d /= ntd
            else: ta_d = np.zeros(6)
            tb_d = -diff - float((-diff) @ Xb)*Xb if False else ((-diff) - float(((-diff) @ Xb))*Xb)
            # actually tb diff = Xb-Xa projected
            db = Xb - Xa
            tb_d = db - float(db @ Xb)*Xb
            ntd2 = float(np.linalg.norm(tb_d))
            if ntd2 > 1e-12: tb_d /= ntd2
            cand_list = []
            for eps in (0.003, 0.008, 0.015):
                y1 = Xa + eps*ta_d; y1 /= max(np.linalg.norm(y1), 1e-12)
                y2 = Xb + eps*tb_d; y2 /= max(np.linalg.norm(y2), 1e-12)
                cand_list.append((y1, y2))
            # random opposite tangents: 5 dirs
            for rr in range(5):
                r = rng.standard_normal(6)
                ta = r - float(r @ Xa)*Xa; na = float(np.linalg.norm(ta))
                if na < 1e-12: continue
                ta /= na
                tb = r - float(r @ Xb)*Xb; nb2 = float(np.linalg.norm(tb))
                if nb2 < 1e-12: continue
                tb /= nb2
                for eps in (0.005, 0.012):
                    y1 = Xa + eps*ta; y1 /= max(np.linalg.norm(y1), 1e-12)
                    y2 = Xb - eps*tb; y2 /= max(np.linalg.norm(y2), 1e-12)
                    cand_list.append((y1, y2))
                    y1b = Xa - eps*ta; y1b /= max(np.linalg.norm(y1b), 1e-12)
                    y2b = Xb + eps*tb; y2b /= max(np.linalg.norm(y2b), 1e-12)
                    cand_list.append((y1b, y2b))
            # try each with short joint (limit to ~8 to save time: pick random subset)
            # evaluate quick global max first, keep best 4
            scored = []
            for (y1, y2) in cand_list:
                Xtmp = Xbest.copy(); Xtmp[a] = y1; Xtmp[b] = y2
                vq = _max_cos(Xtmp)
                scored.append((vq, y1, y2))
            scored.sort(key=lambda t: t[0])
            for vq, y1, y2 in scored[:4]:
                if _tm.time() > LOOP_END - 3.0:
                    break
                Xcand0 = Xbest.copy(); Xcand0[a] = y1; Xcand0[b] = y2
                Xcand = _joint_opt(S_idx, Xcand0, [3e3, 1e4], [70, 70], LOOP_END)
                v = _max_cos(Xcand)
                if v < Mbest - 1e-15:
                    Xc2, v2 = _lp_polish5(Xcand, v, LOOP_END, 5, 1e-5)
                    if v2 < Mbest - 1e-15:
                        Xbest = Xc2; Mbest = float(v2)
                    else:
                        Xbest = Xcand; Mbest = float(v)
                    improved_this_round = True
                    first_done = False
                    break
            if improved_this_round:
                outer += 1
                continue
        except Exception:
            pass
        outer += 1
    # final safety normalise
    out = Xbest.copy()
    nr = np.linalg.norm(out, axis=1, keepdims=True)
    nr = np.maximum(nr, 1e-12)
    out = out/nr
    out[~np.isfinite(out)] = 0.0
    nr2 = np.linalg.norm(out, axis=1, keepdims=True)
    nr2[nr2 == 0] = 1.0
    out = out/nr2
    for i in range(n):
        if not np.all(np.isfinite(out[i])) or float(np.linalg.norm(out[i])) < 0.5:
            out[i, :] = 0.0; out[i, 0] = 1.0
    nr3 = np.linalg.norm(out, axis=1, keepdims=True)
    out = out/np.maximum(nr3, 1e-12)
    return out
