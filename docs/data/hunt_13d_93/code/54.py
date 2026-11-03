import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _norm(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return X / n

def _maxcos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def solve():
    np.random.seed(0)
    t0 = time.time()
    LIMIT = 172.0
    n, d = 93, 13
    iu0, iu1 = np.triu_indices(n, 1)
    Pfull = len(iu0)
    ar_d = np.arange(d)

    # load best
    best = None
    best_val = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=float)
            if a.shape == (n, d) and np.all(np.isfinite(a)):
                a = _norm(a)
                v = _maxcos(a)
                if v < best_val:
                    best_val = v
                    best = a.copy()
        except Exception:
            pass
    if best is None:
        rng0 = np.random.RandomState(0)
        best = _norm(rng0.randn(n, d))
        best_val = _maxcos(best)
    bestX = best.copy()
    cur_best = float(best_val)

    # analyse best
    G0 = bestX @ bestX.T
    np.fill_diagonal(G0, -2.0)
    rowmax = G0.max(axis=1)
    order_loose = np.argsort(rowmax)  # increasing: loosest first
    curmax = float(rowmax.max())
    c_all0 = G0[iu0, iu1]
    slack0 = curmax - c_all0
    # contacts per point at 1e-7, 1e-5
    for thr in (1e-9, 1e-7, 1e-5):
        pass
    # hub = most contacts at 1e-7
    mask7 = slack0 < 1e-7
    cnt7 = np.zeros(n, dtype=int)
    np.add.at(cnt7, iu0[mask7], 1)
    np.add.at(cnt7, iu1[mask7], 1)
    hub = int(np.argmax(cnt7))
    # if all zero, hub = worst rowmax
    if cnt7.max() == 0:
        hub = int(np.argmax(rowmax))

    ks = [1, 2, 3, 5, 8, 13, 19]

    # build trial list: (k, variant) variant 0 = loosest, 1 = hub swap
    trials = []
    for k in ks:
        trials.append((k, 0))
        trials.append((k, 1))
    ntrials = len(trials)

    # ---------- helpers (closure) ----------
    def void_insert(Y, rng, n_sample=60000, n_best=20):
        m = Y.shape[0]
        # sample
        S = rng.standard_normal((n_sample, d))
        S = _norm(S)
        # score in chunks
        scores = np.empty(n_sample, dtype=np.float64)
        CH = 10000
        for s in range(0, n_sample, CH):
            e = min(s + CH, n_sample)
            D = S[s:e] @ Y.T
            scores[s:e] = D.max(axis=1)
        # best 20
        if n_best >= n_sample:
            idx = np.argsort(scores)
        else:
            part = np.argpartition(scores, n_best)[:n_best]
            # sort them for determinism
            part = part[np.argsort(scores[part])]
            idx = part
        C = S[idx].copy()  # (B,d)
        B = C.shape[0]
        # softmax descent vectorized, tau 0.008 -> 0.001
        taus = [0.008, 0.004, 0.002, 0.001]
        lr = 0.12
        for tau in taus:
            for _ in range(25):
                D = C @ Y.T  # (B,m)
                mx = D.max(axis=1, keepdims=True)
                E = np.exp((D - mx) / tau)
                E /= (E.sum(axis=1, keepdims=True) + 1e-300)
                Gr = E @ Y  # (B,d)
                dot = np.einsum('ij,ij->i', Gr, C)
                Gt = Gr - dot[:, None] * C
                C = C - lr * Gt
                C = _norm(C)
                # time check abort
                if time.time() - t0 > LIMIT - 2:
                    break
        # pick best by true max
        Df = C @ Y.T
        sc = Df.max(axis=1)
        bi = int(np.argmin(sc))
        return C[bi].copy(), float(sc[bi])

    def adam_pnorm(X, shift=0.20, steps_list=(40, 40, 40), lr=0.004):
        # X (nn,d) normalized in/out
        nn = X.shape[0]
        if nn < 2:
            return X
        ju0, ju1 = np.triu_indices(nn, 1)
        Pp = len(ju0)
        m_ad = np.zeros_like(X)
        v_ad = np.zeros_like(X)
        b1, b2, eps = 0.9, 0.999, 1e-8
        tstep = 0
        for p in steps_list:
            for s in range(p):
                if time.time() - t0 > LIMIT - 2:
                    return _norm(X)
                G = X @ X.T
                c = G[ju0, ju1]
                relu = c - shift
                np.maximum(relu, 0.0, out=relu)
                # weights dL/dc
                # M = mean(relu^p)
                # avoid overflow: relu<=1
                rp_pow = relu ** p if p < 200 else np.exp(p * np.log(np.maximum(relu, 1e-300)))
                M = float(np.mean(rp_pow) + 1e-300)
                if not np.finite(M) or M <= 0:
                    continue
                # rp-1
                with np.errstate(over='ignore', under='ignore', divide='ignore', invalid='ignore'):
                    rpm1 = relu ** (p - 1)
                scale = (M ** (1.0 / p - 1.0)) / float(Pp)
                if not np.finite(scale):
                    continue
                w = rpm1 * scale
                # grad
                Wm = np.zeros((nn, nn), dtype=np.float64)
                Wm[ju0, ju1] = w
                Wm[ju1, ju0] = w
                grad = Wm @ X
                # tangent project
                dot = np.einsum('ij,ij->i', grad, X)
                grad -= dot[:, None] * X
                # adam
                tstep += 1
                m_ad = b1 * m_ad + (1 - b1) * grad
                v_ad = b2 * v_ad + (1 - b2) * (grad * grad)
                mh = m_ad / (1 - b1 ** tstep)
                vh = v_ad / (1 - b2 ** tstep)
                X = X - lr * mh / (np.sqrt(vh) + eps)
                X = _norm(X)
        return _norm(X)

    def topk_descent(X, K=150, steps=250, lr=0.003):
        nn = X.shape[0]
        ju0, ju1 = np.triu_indices(nn, 1)
        Pp = len(ju0)
        K = min(K, Pp)
        m_ad = np.zeros_like(X)
        v_ad = np.zeros_like(X)
        b1, b2, eps = 0.9, 0.999, 1e-8
        best_l = _norm(X)
        best_v = _maxcos(best_l)
        for s in range(steps):
            if time.time() - t0 > LIMIT - 2:
                break
            G = X @ X.T
            c = G[ju0, ju1]
            # top K
            if K < Pp:
                thr_idx = np.argpartition(-c, K - 1)[:K]
            else:
                thr_idx = np.arange(Pp)
            w = np.zeros(Pp, dtype=np.float64)
            w[thr_idx] = 1.0 / float(K)
            Wm = np.zeros((nn, nn), dtype=np.float64)
            Wm[ju0, ju1] = w
            Wm[ju1, ju0] = w
            grad = Wm @ X
            dot = np.einsum('ij,ij->i', grad, X)
            grad -= dot[:, None] * X
            tstep = s + 1
            m_ad = b1 * m_ad + (1 - b1) * grad
            v_ad = b2 * v_ad + (1 - b2) * (grad * grad)
            mh = m_ad / (1 - b1 ** tstep)
            vh = v_ad / (1 - b2 ** tstep)
            X = _norm(X - lr * mh / (np.sqrt(vh) + eps))
            if (s % 25 == 0) or (s == steps - 1):
                v = _maxcos(X)
                if v < best_v:
                    best_v = v
                    best_l = X.copy()
        return _norm(best_l), float(best_v)

    def lp_polish(X, max_iter=6, radius0=1.2e-3):
        nn, dd = X.shape
        nv = nn * dd + 1
        radius = float(radius0)
        cur = _maxcos(X)
        Xc = X.copy()
        for it in range(max_iter):
            if time.time() - t0 > LIMIT - 2:
                break
            G = Xc @ Xc.T
            c_all = G[iu0[: (nn*(nn-1)//2)] if nn != n else iu0, iu1[: (nn*(nn-1)//2)] if nn != n else iu1] if nn != n else G[iu0, iu1]
            # for nn==n use iu0/iu1 else build indices
            if nn != n:
                ju0_, ju1_ = np.triu_indices(nn, 1)
                c_all = G[ju0_, ju1_]
                Iall, Jall = ju0_, ju1_
            else:
                Iall, Jall = iu0, iu1
            cmax = float(c_all.max())
            # select active: slack < 5e-4 or top 800
            thresh = 5e-4
            sel = c_all > cmax - thresh
            cnt = int(sel.sum())
            if cnt < 300:
                K2 = min(300, len(c_all))
                idx = np.argpartition(-c_all, K2 - 1)[:K2]
                sel = np.zeros(len(c_all), dtype=bool)
                sel[idx] = True
            elif cnt > 900:
                K2 = 900
                idx = np.argpartition(-c_all, K2 - 1)[:K2]
                sel = np.zeros(len(c_all), dtype=bool)
                sel[idx] = True
            I = Iall[sel]; J = Jall[sel]
            csub = c_all[sel]
            pnum = len(I)
            # build A_ub
            try:
                rows = np.empty(pnum * (2 * dd + 1), dtype=np.int64)
                cols = np.empty(pnum * (2 * dd + 1), dtype=np.int64)
                data = np.empty(pnum * (2 * dd + 1), dtype=np.float64)
                # block i
                r1 = np.repeat(np.arange(pnum), dd)
                c1 = (I[:, None] * dd + ar_d[None, :]).ravel()
                rows[0:pnum*dd] = r1
                cols[0:pnum*dd] = c1
                data[0:pnum*dd] = Xc[J].reshape(-1)
                r2 = np.repeat(np.arange(pnum), dd)
                c2 = (J[:, None] * dd + ar_d[None, :]).ravel()
                rows[pnum*dd:2*pnum*dd] = r2
                cols[pnum*dd:2*pnum*dd] = c2
                data[pnum*dd:2*pnum*dd] = Xc[I].reshape(-1)
                rows[2*pnum*dd:2*pnum*dd+pnum] = np.arange(pnum)
                cols[2*pnum*dd:2*pnum*dd+pnum] = nv - 1
                data[2*pnum*dd:2*pnum*dd+pnum] = -1.0
                A_ub = coo_matrix((data, (rows, cols)), shape=(pnum, nv)).tocsr()
                b_ub = -csub
                # A_eq: X_i . D_i =0
                er = np.repeat(np.arange(nn), dd)
                ec = (np.repeat(np.arange(nn), dd) * dd + np.tile(ar_d, nn))
                ed = Xc.reshape(-1)
                A_eq = coo_matrix((ed, (er, ec)), shape=(nn, nv)).tocsr()
                b_eq = np.zeros(nn)
                obj = np.zeros(nv); obj[-1] = 1.0
                bounds = [(-radius, radius)] * (nn * dd) + [(-1.0, float(cmax))]
                res = linprog(obj, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                              bounds=bounds, method="highs", options={"presolve": True})
            except Exception:
                radius *= 0.5
                continue
            if res.status != 0 or res.x is None or not np.all(np.isfinite(res.x)):
                radius *= 0.5
                if radius < 1e-9:
                    radius = 5e-6
                    break
                continue
            yv = res.x[:-1]
            t_pred = float(res.x[-1])
            pred = cmax - t_pred
            if pred <= 1e-13:
                radius *= 0.5
                if radius < 2e-9:
                    break
                continue
            Dd = yv.reshape(nn, dd)
            Xt = _norm(Xc + Dd)
            av = _maxcos(Xt)
            if av < cmax - 1e-15:
                Xc = Xt
                cur = av
                # expand slightly if at bound
                if np.abs(yv).max() > 0.9 * radius:
                    radius = min(radius * 1.3, 5e-3)
            else:
                # backtrack
                found = False
                for al in (0.5, 0.25, 0.125):
                    Xb = _norm(Xc + al * Dd)
                    vb = _maxcos(Xb)
                    if vb < cmax - 1e-15:
                        Xc = Xb; cur = vb; found = True
                        break
                if not found:
                    radius *= 0.5
                    if radius < 1e-9:
                        break
        return _norm(Xc), float(_maxcos(Xc))

    # ---------- main trial loop ----------
    for ti, (k, var) in enumerate(trials):
        now = time.time()
        remain = LIMIT - (now - t0)
        if remain < 5:
            break
        left_trials = ntrials - ti
        budget = remain / left_trials * 0.92  # strict per-trial
        t_start = time.time()
        # determine delete set
        loosest = list(order_loose[:k])
        if var == 1:
            if hub not in loosest:
                # replace last (least loose among them) with hub
                loosest = list(order_loose[:k])
                loosest[-1] = int(hub)
            else:
                # already includes hub -> skip duplicate
                continue
        delset = np.array(sorted(loosest), dtype=int)
        keep = np.array([i for i in range(n) if i not in set(delset.tolist())], dtype=int)
        Y = bestX[keep].copy()
        rng = np.random.RandomState(1000 + ti * 77 + k * 13 + var)
        ok = True
        # greedy reinsert one at a time
        for ins in range(k):
            if time.time() - t_start > budget * 0.72:
                ok = False
                break
            if time.time() - t0 > LIMIT - 4:
                ok = False
                break
            xv, _ = void_insert(Y, rng)
            Y = np.vstack([Y, xv.reshape(1, d)])
            Y = _norm(Y)
            # shifted p-norm adam short
            Y = adam_pnorm(Y, shift=0.20, steps_list=(30, 30, 30), lr=0.004)
        if not ok:
            continue
        if Y.shape[0] != n:
            continue
        # finish: top-K hard-max
        if time.time() - t_start > budget * 0.85:
            continue
        Y2, v2 = topk_descent(Y, K=150, steps=200, lr=0.003)
        # short LP polish if time
        if time.time() - t_start < budget * 0.95 and time.time() - t0 < LIMIT - 3:
            try:
                Y3, v3 = lp_polish(Y2, max_iter=5, radius0=1.0e-3)
            except Exception:
                Y3, v3 = Y2, v2
        else:
            Y3, v3 = Y2, v2
        if v3 < cur_best - 1e-12 and np.all(np.isfinite(Y3)):
            cur_best = float(v3)
            bestX = _norm(Y3).copy()
            # update order/hub for next trials? keep original order per spec (based on best.npy)
            # but update best so future deletes use new best? spec says load best.npy; we keep using current bestX keep set?
            # refresh loosest to new best for remaining trials
            Gnew = bestX @ bestX.T
            np.fill_diagonal(Gnew, -2.0)
            rowmax = Gnew.max(axis=1)
            order_loose = np.argsort(rowmax)
            c_all0 = Gnew[iu0, iu1]
            curmax = float(c_all0.max())
            slack0 = curmax - c_all0
            mask7 = slack0 < 1e-7
            cnt7 = np.zeros(n, dtype=int)
            np.add.at(cnt7, iu0[mask7], 1)
            np.add.at(cnt7, iu1[mask7], 1)
            hub = int(np.argmax(cnt7)) if cnt7.max() > 0 else int(np.argmax(rowmax))
        # enforce per-trial budget
        # loop continues
        if time.time() - t0 > LIMIT - 3:
            break

    bestX = _norm(bestX)
    return bestX
