import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

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
            v = -x[j] * x
            v[j] += 1.0
            T[:, c] = v
            c += 1
        Q, _ = np.linalg.qr(T)
        B[i] = Q[:, :4]
    return B

def solve():
    t0 = time.time()
    T_END = t0 + 172.0
    EXP_END = t0 + 108.0
    LP_CUTOFF = t0 + 166.0
    rng = np.random.default_rng(0)
    n, d = 453, 5

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
            nn = np.linalg.norm(cc, axis=1, keepdims=True)
            nn[nn == 0] = 1.0
            cc = cc / nn
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
    incumbentX = bestX.copy()
    incumbentV = float(bestV)

    iu_r, iu_c = np.triu_indices(n, 1)

    def anneal_fast(X, steps, tau0, tau1, lr0, lr1, mu=0.85):
        X = X.copy()
        v = np.zeros_like(X)
        # log_interp
        lt0 = np.log(max(tau0, 1e-12)); lt1 = np.log(max(tau1, 1e-12))
        ll0 = np.log(max(lr0, 1e-12)); ll1 = np.log(max(lr1, 1e-12))
        m = 0.0
        for it in range(steps):
            if (it & 31) == 0 and time.time() > EXP_END:
                break
            frac = it / max(steps - 1, 1)
            tau = float(np.exp(lt0 + (lt1 - lt0) * frac))
            lr = float(np.exp(ll0 + (ll1 - ll0) * frac))
            G = X @ X.T
            np.fill_diagonal(G, -2.0)
            m = float(np.max(G))
            E = (G - m) / tau
            # E<=0
            np.clip(E, -50.0, 0.0, out=E)
            P = np.exp(E)
            Grad = P @ X
            dot = np.sum(Grad * X, axis=1, keepdims=True)
            Gt = Grad - dot * X
            rs = np.sum(P, axis=1, keepdims=True) + 1e-12
            Gt = Gt / rs
            v = mu * v + (1.0 - mu) * Gt
            X = X - lr * v * 8.0
            X /= (np.linalg.norm(X, axis=1, keepdims=True) + 1e-30)
        return X

    def best_holes(Xref, k, n_cand=22000):
        C = rng.standard_normal((n_cand, d))
        C /= (np.linalg.norm(C, axis=1, keepdims=True) + 1e-30)
        D = C @ Xref.T
        md = np.max(D, axis=1)
        holes = []
        mdc = md.copy()
        for _ in range(k):
            bi = int(np.argmin(mdc))
            h = C[bi].copy()
            for __ in range(50):
                dots = Xref @ h
                ci = int(np.argmax(dots))
                xc = Xref[ci]
                cv = float(dots[ci])
                tt = xc - cv * h
                tn = float(np.linalg.norm(tt))
                if tn < 1e-12:
                    break
                tt /= tn
                h = h - 0.01 * tt
                nh = float(np.linalg.norm(h))
                if not np.isfinite(nh) or nh == 0:
                    break
                h /= nh
            holes.append(h)
            dh = C @ h
            mdc = np.maximum(mdc, dh)
            mdc[bi] = 10.0
        return np.array(holes)

    def row_max_deg(X):
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        rm = np.max(G, axis=1)
        m = float(np.max(rm))
        # degree with tol 1.5e-5 (near contacts)
        deg = np.sum(G > (m - 1.5e-5), axis=1)
        return rm, deg, m, G

    # contact analysis on incumbent
    try:
        rm0, deg0, m0, G0 = row_max_deg(bestX)
    except Exception:
        rm0 = np.zeros(n); deg0 = np.zeros(n, dtype=int)
    order_hub = np.argsort(-deg0 - 0.5 * (rm0 - np.min(rm0)) / (np.max(rm0) - np.min(rm0) + 1e-12))
    rattlers = np.where(deg0 <= 1)[0]
    # deletion choices: top hubs + a couple rattlers / random mid
    del_list = []
    for idx in order_hub[:5]:
        del_list.append(int(idx))
    for idx in rattlers[:2]:
        if int(idx) not in del_list:
            del_list.append(int(idx))
    # add one mid-degree for diversity
    mid = int(np.argsort(deg0)[n // 2])
    if mid not in del_list:
        del_list.append(mid)
    del_list = del_list[:7]

    trials = []  # (val, X)

    # ---- Cycle A: 453 -> 452 (delete hub) -> anneal -> reinsert hole -> anneal ----
    tau_sets = [(0.004, 0.0004, 0.006, 0.0006), (0.002, 0.0002, 0.0045, 0.00045),
                (0.006, 0.0006, 0.007, 0.0008), (0.003, 0.00025, 0.005, 0.0005)]
    for di, victim in enumerate(del_list):
        if time.time() > EXP_END - 2.0:
            break
        try:
            mask = np.ones(n, dtype=bool); mask[victim] = False
            Xkeep = bestX[mask].copy()  # 452
            # shake
            sig = 0.008 if (di % 2 == 0) else 0.014
            Nn = rng.standard_normal(Xkeep.shape)
            Nn -= np.sum(Nn * Xkeep, axis=1, keepdims=True) * Xkeep
            Xkeep = Xkeep + sig * Nn
            Xkeep /= (np.linalg.norm(Xkeep, axis=1, keepdims=True) + 1e-30)
            ts = tau_sets[di % len(tau_sets)]
            Xkeep_o = anneal_fast(Xkeep, 320, ts[0], ts[1], ts[2], ts[3], 0.85)
            if time.time() > EXP_END - 1.0:
                # still try reinsert quickly
                pass
            h = best_holes(Xkeep_o, 1, n_cand=18000)[0]
            Xnew = np.vstack([Xkeep_o, h.reshape(1, -1)])
            Nn2 = rng.standard_normal(Xnew.shape)
            Nn2 -= np.sum(Nn2 * Xnew, axis=1, keepdims=True) * Xnew
            Xnew = Xnew + 0.004 * Nn2
            Xnew /= (np.linalg.norm(Xnew, axis=1, keepdims=True) + 1e-30)
            ts2 = tau_sets[(di + 1) % len(tau_sets)]
            Xnew_o = anneal_fast(Xnew, 320, ts2[0], ts2[1], ts2[2], ts2[3], 0.86)
            vv = _max_cos(Xnew_o)
            if np.isfinite(vv):
                trials.append((float(vv), Xnew_o.copy()))
                if vv < bestV - 1e-12:
                    bestV = float(vv); bestX = Xnew_o.copy()
        except Exception:
            continue

    # ---- Cycle B: 453 -> 454 (insert hole) -> anneal -> delete weakest -> anneal ----
    if time.time() < EXP_END - 2.0:
        try:
            holesB = best_holes(bestX, 2, n_cand=22000)
        except Exception:
            holesB = []
        for hi, h in enumerate(holesB):
            if time.time() > EXP_END - 2.0:
                break
            try:
                X454 = np.vstack([bestX, h.reshape(1, -1)])
                Nn = rng.standard_normal(X454.shape)
                Nn -= np.sum(Nn * X454, axis=1, keepdims=True) * X454
                X454 = X454 + 0.007 * Nn
                X454 /= (np.linalg.norm(X454, axis=1, keepdims=True) + 1e-30)
                ts = tau_sets[(hi + 2) % len(tau_sets)]
                X454_o = anneal_fast(X454, 340, ts[0], ts[1], ts[2], ts[3], 0.85)
                # row max on 454
                G4 = X454_o @ X454_o.T
                np.fill_diagonal(G4, -2.0)
                rm4 = np.max(G4, axis=1)
                # two removal choices: argmin and second argmin (least stressed)
                order_rm = np.argsort(rm4)
                victimsB = [int(order_rm[0])]
                if order_rm.size > 1:
                    # second choice distinct
                    victimsB.append(int(order_rm[1]))
                for vi, vb in enumerate(victimsB):
                    if time.time() > EXP_END - 1.0 and vi > 0:
                        break
                    mask4 = np.ones(X454_o.shape[0], dtype=bool); mask4[vb] = False
                    X453 = X454_o[mask4].copy()
                    # tiny shake except removed region
                    Nn3 = rng.standard_normal(X453.shape)
                    Nn3 -= np.sum(Nn3 * X453, axis=1, keepdims=True) * X453
                    X453 = X453 + 0.003 * Nn3
                    X453 /= (np.linalg.norm(X453, axis=1, keepdims=True) + 1e-30)
                    ts3 = tau_sets[(hi + vi + 1) % len(tau_sets)]
                    X453_o = anneal_fast(X453, 300, ts3[0], ts3[1], ts3[2], ts3[3], 0.86)
                    vv = _max_cos(X453_o)
                    if np.isfinite(vv):
                        trials.append((float(vv), X453_o.copy()))
                        if vv < bestV - 1e-12:
                            bestV = float(vv); bestX = X453_o.copy()
            except Exception:
                continue

    # keep only best two candidates for LP polishing (plus diversity)
    trials_sorted = sorted(trials, key=lambda t: t[0])
    # deduplicate roughly: keep best two with different values
    lp_cands = []
    for vv, XX in trials_sorted[:6]:
        if len(lp_cands) >= 2:
            break
        # require some distance from already kept
        ok = True
        for _, YY in lp_cands:
            if np.linalg.norm(XX - YY) < 1e-6:
                ok = False; break
        if ok:
            lp_cands.append((vv, XX))
    # if no trial beat incumbent, still polish top2 if close
    # ---- LP polish best two ----
    def lp_polish(Xinit, radius0=6e-5, max_solves=6):
        X = Xinit.copy()
        curV = _max_cos(X)
        best_lX = X.copy(); best_lV = curV
        Nvar = X.shape[0] * 4
        c_obj = np.zeros(Nvar + 1); c_obj[-1] = 1.0
        ar4 = np.arange(4)
        radius = float(radius0)
        for _ in range(max_solves):
            if time.time() > LP_CUTOFF or time.time() > T_END - 6.0:
                break
            if radius < 1e-12:
                break
            nn_ = X.shape[0]
            iu_r2, iu_c2 = np.triu_indices(nn_, 1)
            G = X @ X.T
            c_all = G[iu_r2, iu_c2]
            m = float(np.max(c_all))
            if m < best_lV and np.isfinite(m):
                best_lV = m; best_lX = X.copy()
            slack_all = m - c_all
            tight = np.where(slack_all < 1.5e-5)[0]
            if tight.size == 0:
                tight = np.argsort(slack_all)[:500]
            if tight.size > 2000:
                part = np.argpartition(slack_all[tight], 2000)[:2000]
                tight = tight[part]
            I = iu_r2[tight]; J = iu_c2[tight]
            p = int(I.size)
            b_ub = slack_all[tight].astype(np.float64)
            B = _tangent_bases(X)
            aI = np.einsum('pdk,pd->pk', B[I], X[J])
            aJ = np.einsum('pdk,pd->pk', B[J], X[I])
            cols2 = np.empty((p, 9), dtype=np.int64)
            cols2[:, 0:4] = I[:, None] * 4 + ar4[None, :]
            cols2[:, 4:8] = J[:, None] * 4 + ar4[None, :]
            cols2[:, 8] = Nvar
            vals2 = np.empty((p, 9), dtype=np.float64)
            vals2[:, 0:4] = aI
            vals2[:, 4:8] = aJ
            vals2[:, 8] = -1.0
            rows = np.repeat(np.arange(p), 9)
            try:
                A_ub = coo_matrix((vals2.ravel(), (rows, cols2.ravel())), shape=(p, Nvar + 1)).tocsr()
            except Exception:
                radius *= 0.5; continue
            rem = min(LP_CUTOFF - time.time(), T_END - 6.0 - time.time(), 9.0)
            if rem < 0.5:
                break
            bounds = [(-radius, radius)] * Nvar + [(None, None)]
            try:
                res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs", options={"time_limit": max(0.5, rem)})
            except Exception:
                radius *= 0.5; continue
            if res.status != 0 or res.x is None:
                radius *= 0.5; continue
            U = np.asarray(res.x[:Nvar], dtype=np.float64).reshape(nn_, 4)
            if not np.all(np.isfinite(U)):
                radius *= 0.5; continue
            DX = np.einsum('ndk,nk->nd', B, U)
            Xt = X + DX
            Xt /= (np.linalg.norm(Xt, axis=1, keepdims=True) + 1e-30)
            mt = _max_cos(Xt)
            if mt < m - 1e-13 and np.isfinite(mt):
                X = Xt
                radius = min(radius * 1.4, 3e-4)
            else:
                radius *= 0.5
        return best_lX, best_lV

    # polish up to two
    for vv, XX in lp_cands:
        if time.time() > LP_CUTOFF - 1.0 or time.time() > T_END - 7.0:
            break
        # only polish promising: within 2e-4 of incumbent best (or better)
        if vv > bestV + 2e-4:
            continue
        try:
            pX, pV = lp_polish(XX, radius0=6e-5, max_solves=6)
            if np.isfinite(pV) and pV < bestV - 1e-13:
                bestV = float(pV); bestX = pX.copy()
        except Exception:
            continue

    # always retain incumbent unless verified lower
    if not np.isfinite(bestV) or bestV > incumbentV:
        bestX = incumbentX.copy(); bestV = float(incumbentV)
    else:
        # verify
        try:
            vv = _max_cos(bestX)
            if not np.isfinite(vv) or vv > incumbentV + 1e-12:
                # if our tracked bestV disagrees, recheck
                if vv > incumbentV:
                    bestX = incumbentX.copy()
        except Exception:
            bestX = incumbentX.copy()
    bestX /= (np.linalg.norm(bestX, axis=1, keepdims=True) + 1e-30)
    return bestX
