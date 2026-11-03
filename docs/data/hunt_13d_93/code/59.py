import time
import numpy as np

def solve():
    t0 = time.time()
    BUDGET = 168.0
    deadline = t0 + BUDGET
    n, d = 93, 13
    rng = np.random.default_rng(7)
    iu0, iu1 = np.triu_indices(n, 1)
    P = len(iu0)

    def _norm(X):
        nr = np.linalg.norm(X, axis=1, keepdims=True)
        nr[nr == 0] = 1.0
        return X / nr

    def _maxcos(X):
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        return float(G.max())

    def _rowmax(X):
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        return G.max(axis=1)

    # load baseline
    gbest = None
    gval = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=float)
            if a.shape != (n, d):
                continue
            if not np.all(np.isfinite(a)):
                continue
            a = _norm(a)
            v = _maxcos(a)
            if v < gval:
                gval = v
                gbest = a.copy()
        except Exception:
            pass
    if gbest is None:
        gbest = _norm(rng.standard_normal((n, d)))
        gval = _maxcos(gbest)

    # ---------- softmax Adam ----------
    def softmax_adam(X0, taus, steps_list, lr0, lr1=None, track=True):
        X = _norm(X0.copy())
        best = _maxcos(X)
        bestX = X.copy()
        m = np.zeros_like(X)
        v = np.zeros_like(X)
        b1, b2, eps = 0.9, 0.999, 1e-8
        tt = 0
        total = sum(steps_list)
        done = 0
        for ti, tau in enumerate(taus):
            steps = steps_list[ti]
            for s in range(steps):
                if time.time() > deadline:
                    return bestX, best
                frac = (done + s) / max(1, total - 1)
                if lr1 is None:
                    lr = lr0
                else:
                    lr = lr0 * (lr1 / lr0) ** frac
                G = X @ X.T
                c = G[iu0, iu1]
                mx = float(c.max())
                E = np.exp((c - mx) / tau)
                S = float(E.sum())
                if not np.isfinite(S) or S <= 0:
                    break
                W = np.zeros((n, n), dtype=np.float64)
                W[iu0, iu1] = E / S
                W[iu1, iu0] = E / S
                grad = W @ X
                dot = np.einsum('ij,ij->i', grad, X)[:, None]
                gt = grad - dot * X
                ng = float(np.linalg.norm(gt))
                if not np.isfinite(ng) or ng < 1e-16:
                    break
                tt += 1
                m *= b1
                m += (1.0 - b1) * gt
                v *= b2
                v += (1.0 - b2) * (gt * gt)
                mh = m / (1.0 - b1 ** tt)
                vh = v / (1.0 - b2 ** tt)
                upd = mh / (np.sqrt(vh) + eps)
                d2 = np.einsum('ij,ij->i', upd, X)[:, None]
                upd = upd - d2 * X
                X = _norm(X - lr * upd)
                if track and (tt % 10 == 0 or s == steps - 1):
                    vv = _maxcos(X)
                    if vv < best:
                        best = vv
                        bestX = X.copy()
            done += steps
            # reset moments per temperature
            m.fill(0.0); v.fill(0.0); tt = 0
        vv = _maxcos(X)
        if vv < best:
            best = vv
            bestX = X.copy()
        return bestX, best

    def topk_adam(X0, K, steps, lr0, lr1=None):
        X = _norm(X0.copy())
        best = _maxcos(X)
        bestX = X.copy()
        m = np.zeros_like(X)
        v = np.zeros_like(X)
        b1, b2, eps = 0.9, 0.999, 1e-8
        tt = 0
        K = int(max(1, min(K, P)))
        grad = np.empty_like(X)
        for s in range(steps):
            if time.time() > deadline:
                break
            frac = s / max(1, steps - 1)
            lr = lr0 * ((lr1 / lr0) ** frac) if lr1 is not None else lr0
            G = X @ X.T
            c = G[iu0, iu1]
            if K < P:
                idx = np.argpartition(-c, K)[:K]
            else:
                idx = np.arange(P)
            I = iu0[idx]; J = iu1[idx]
            grad.fill(0.0)
            np.add.at(grad, I, X[J])
            np.add.at(grad, J, X[I])
            grad /= float(K)
            dot = np.einsum('ij,ij->i', grad, X)[:, None]
            gt = grad - dot * X
            ng = float(np.linalg.norm(gt))
            if not np.isfinite(ng) or ng < 1e-16:
                break
            tt += 1
            m *= b1; m += (1.0 - b1) * gt
            v *= b2; v += (1.0 - b2) * (gt * gt)
            mh = m / (1.0 - b1 ** tt)
            vh = v / (1.0 - b2 ** tt)
            upd = mh / (np.sqrt(vh) + eps)
            d2 = np.einsum('ij,ij->i', upd, X)[:, None]
            upd = upd - d2 * X
            X = _norm(X - lr * upd)
            if (s % 15 == 0) or s == steps - 1:
                vv = _maxcos(X)
                if vv < best:
                    best = vv
                    bestX = X.copy()
        return bestX, best

    def find_void(others, n_cand=25000):
        C = rng.standard_normal((n_cand, d))
        C /= np.linalg.norm(C, axis=1, keepdims=True)
        # chunked
        worst = np.empty(n_cand)
        ch = 8000
        for a in range(0, n_cand, ch):
            b = min(a + ch, n_cand)
            worst[a:b] = (C[a:b] @ others.T).max(axis=1)
        k = int(np.argmin(worst))
        v = C[k].copy()
        # quick refine: greedy tangent push
        lr = 0.05
        bestm = float(worst[k])
        bestv = v.copy()
        for _ in range(120):
            if time.time() > deadline:
                break
            dots = others @ v
            mx = float(dots.max())
            # softmax gradient for single point
            tau = 0.004
            E = np.exp((dots - mx) / tau)
            E /= E.sum()
            g = E @ others
            gv = float(g @ v)
            gt = g - gv * v
            nn = float(np.linalg.norm(gt))
            if nn < 1e-13:
                break
            gt /= nn
            v2 = v - lr * gt
            v2 /= np.linalg.norm(v2)
            m2 = float((others @ v2).max())
            if m2 < mx:
                v = v2
                if m2 < bestm:
                    bestm = m2
                    bestv = v.copy()
                lr = min(lr * 1.05, 0.08)
            else:
                lr *= 0.5
                if lr < 1e-6:
                    break
        return bestv / np.linalg.norm(bestv)

    # Phase A: iterated shake + anneal + cloud balance
    sigmas = [0.003, 0.006, 0.010, 0.016, 0.005, 0.012]
    si = 0
    # quick first polish of baseline with topK to squeeze cloud
    try:
        if time.time() < deadline - 20:
            Xt, vt = topk_adam(gbest, 500, 250, 0.006, 0.001)
            if vt < gval:
                gval = vt; gbest = Xt.copy()
            Xt2, vt2 = softmax_adam(gbest, [0.002, 0.0008], [200, 200], 0.008, 0.002)
            if vt2 < gval:
                gval = vt2; gbest = Xt2.copy()
    except Exception:
        pass

    while time.time() < t0 + 115:
        if time.time() > deadline - 25:
            break
        sig = sigmas[si % len(sigmas)]; si += 1
        Xp = gbest + sig * rng.standard_normal((n, d))
        Xp = _norm(Xp)
        # anneal: high->low tau
        try:
            Xa, va = softmax_adam(Xp, [0.008, 0.003, 0.0012], [180, 180, 180], 0.010, 0.002)
        except Exception:
            continue
        # cloud balancing with large K then small K
        try:
            # choose K based on iteration: alternate 600 / 250
            Kbig = 600 if (si % 2 == 0) else 350
            Xb, vb = topk_adam(Xa, Kbig, 220, 0.005, 0.001)
            # fine hard max squeeze
            Xc, vc = topk_adam(Xb, 40, 120, 0.003, 0.0008)
        except Exception:
            continue
        # pick best among stages
        cand, cval = (Xa, va)
        if vb < cval:
            cand, cval = Xb, vb
        if vc < cval:
            cand, cval = Xc, vc
        if cval < gval - 1e-12:
            gval = cval
            gbest = cand.copy()
        # also opportunistic: if within 2e-6, try extra fine softmax
        if cval < gval + 5e-6 and time.time() < t0 + 100:
            try:
                Xf, vf = softmax_adam(cand, [0.0006], [150], 0.004, 0.001)
                if vf < gval:
                    gval = vf; gbest = Xf.copy()
            except Exception:
                pass

    # Phase B: void surgery on worst points (topology change)
    surg_tries = 0
    while time.time() < t0 + 148 and surg_tries < 12:
        if time.time() > deadline - 18:
            break
        surg_tries += 1
        rm = _rowmax(gbest)
        # delete 1-3 worst
        kdel = 1 if (surg_tries % 3 != 0) else (2 if (surg_tries % 3 == 1) else 3)
        # also sometimes delete a random near-max participant
        G = gbest @ gbest.T
        np.fill_diagonal(G, -2.0)
        mx = float(G.max())
        # pairs within 1e-6 of max
        tight_mask = G > mx - 2e-6
        tight_count = tight_mask.sum(axis=1)
        # score to delete: combine rowmax and tight participation
        score = rm + 0.002 * tight_count
        # add tiny random tie break
        score = score + 1e-6 * rng.standard_normal(n)
        dell = np.argsort(-score)[:kdel].tolist()
        keep = np.ones(n, dtype=bool)
        keep[np.array(dell)] = False
        Xcur = gbest[keep].copy()
        ok = True
        for _ in range(kdel):
            if time.time() > deadline - 16:
                ok = False; break
            vnew = find_void(Xcur)
            Xcur = _norm(np.vstack([Xcur, vnew]))
        if not ok:
            continue
        try:
            Xp2, vp2 = softmax_adam(Xcur, [0.005, 0.0015], [220, 220], 0.008, 0.002)
            Xt2, vt2 = topk_adam(Xp2, 300, 180, 0.004, 0.001)
        except Exception:
            continue
        best_c, best_v = (Xp2, vp2) if vp2 < vt2 else (Xt2, vt2)
        if best_v < gval:
            gval = best_v; gbest = best_c.copy()

    # Phase C: final polish + LP if time
    try:
        if time.time() < deadline - 8:
            Xf, vf = topk_adam(gbest, 120, 300, 0.003, 0.0005)
            if vf < gval:
                gval = vf; gbest = Xf.copy()
        if time.time() < deadline - 6:
            Xf2, vf2 = softmax_adam(gbest, [0.0005], [300], 0.003, 0.0008)
            if vf2 < gval:
                gval = vf2; gbest = Xf2.copy()
    except Exception:
        pass

    # LP trust-region polish (light)
    try:
        from scipy.optimize import linprog
        from scipy.sparse import coo_matrix
        start_lp = time.time()
        x = _norm(gbest.copy())
        best = _maxcos(x)
        xb = x.copy()
        radius = 6e-4
        ar = np.arange(d)
        iu = np.triu_indices(n, 1)
        while time.time() - start_lp < 9 and time.time() < deadline - 2 and radius > 1e-14:
            Gf = x @ x.T
            call = Gf[iu]
            mm = float(call.max())
            sel = call > mm - max(20.0 * radius, 5e-7)
            cnt = int(sel.sum())
            if cnt == 0:
                radius *= 0.5; continue
            if cnt > 2500:
                idx = np.argpartition(-call, 2500)[:2500]
                msk = np.zeros_like(sel); msk[idx] = True; sel = msk
            I = iu[0][sel]; J = iu[1][sel]
            p = len(I)
            csub = call[sel]
            nv = n * d + 1
            rows = np.repeat(np.arange(p), 2 * d + 1)
            cols = np.concatenate([I[:, None] * d + ar, J[:, None] * d + ar, np.full((p, 1), n * d)], axis=1).ravel()
            vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], axis=1).ravel()
            try:
                A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
                A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
                obj = np.zeros(nv); obj[-1] = 1.0
                res = linprog(obj, A_ub=A_ub, b_ub=-csub, A_eq=A_eq, b_eq=np.zeros(n),
                              bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
            except Exception:
                radius *= 0.5; continue
            if res.status != 0:
                radius *= 0.5; continue
            y = _norm(x + res.x[:-1].reshape(n, d))
            vv = _maxcos(y)
            if vv < best:
                best = vv; xb = y.copy(); x = y
                radius = min(radius * 1.5, 0.008)
            else:
                radius *= 0.5
        if best < gval:
            gval = best; gbest = xb.copy()
    except Exception:
        pass

    gbest = _norm(gbest)
    return gbest
