import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _norm(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return x / n

def _max_cos(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def solve():
    t0 = time.time()
    HARD_END = t0 + 172.0
    rng = np.random.default_rng(0)
    n0, d0 = 453, 5

    cands = []
    try:
        b = np.load("best.npy")
        cands.append(np.asarray(b, dtype=np.float64))
    except Exception:
        pass
    try:
        r = np.load("record.npy")
        cands.append(np.asarray(r, dtype=np.float64))
    except Exception:
        pass

    iu_tr, iu_tc = np.triu_indices(n0, 1)
    bestV = float("inf")
    bestX = None
    for cc in cands:
        try:
            if cc.shape != (n0, d0):
                f = cc.ravel()
                if f.size < n0 * d0:
                    continue
                cc = f[:n0*d0].reshape(n0, d0)
            cc = _norm(cc)
            if not np.all(np.isfinite(cc)):
                continue
            G = cc @ cc.T
            vv = float(np.max(G[iu_tr, iu_tc]))
            if vv < bestV:
                bestV = vv
                bestX = cc.copy()
        except Exception:
            continue
    if bestX is None:
        bestX = rng.normal(size=(n0, d0))
        bestX = _norm(bestX)
        bestV = _max_cos(bestX)
    n, d = n0, d0
    globX = bestX.copy()
    globV = float(bestV)

    def find_hole(Xref, exclude, n_cand=15000, refine=40):
        mask = np.ones(n, dtype=bool)
        try:
            mask[exclude] = False
        except Exception:
            mask[np.asarray(exclude)] = False
        Xe = Xref[mask]
        C = rng.normal(size=(n_cand, d))
        C /= np.linalg.norm(C, axis=1, keepdims=True) + 1e-30
        D = C @ Xe.T
        md = np.max(D, axis=1)
        bi = int(np.argmin(md))
        h = C[bi].copy()
        for _ in range(refine):
            dots = Xe @ h
            ci = int(np.argmax(dots))
            xc = Xe[ci]
            c_ = float(dots[ci])
            tt = xc - c_ * h
            tn = float(np.linalg.norm(tt))
            if tn < 1e-12:
                break
            tt /= tn
            h = h - 0.01 * tt
            h /= np.linalg.norm(h) + 1e-30
        return h

    def anneal_run(x, N, tau0, tau1, lr0, lr1, mu, deadline):
        nonlocal globX, globV
        v = np.zeros_like(x)
        ltau = np.log(tau1 / tau0)
        llr = np.log(lr1 / lr0)
        for it in range(N):
            if (it & 31) == 0 and time.time() > deadline:
                break
            frac = it / max(1, N - 1)
            tau = tau0 * np.exp(ltau * frac)
            lr = lr0 * np.exp(llr * frac)
            G = x @ x.T
            np.fill_diagonal(G, -2.0)
            m = float(np.max(G))
            if m < globV:
                globV = m
                globX = x.copy()
            # softmax weights
            # (G-m)<=0
            P = np.exp((G - m) / tau)
            np.fill_diagonal(P, 0.0)
            Grad = P @ x
            dot = np.sum(Grad * x, axis=1, keepdims=True)
            Gt = Grad - dot * x
            rn = np.sqrt(np.sum(Gt * Gt, axis=1))
            mn = float(np.max(rn))
            if not np.isfinite(mn) or mn < 1e-14:
                break
            Gt /= mn
            v = mu * v + (1.0 - mu) * Gt
            x = x - lr * v
            x /= np.linalg.norm(x, axis=1, keepdims=True) + 1e-30
            if m > globV + 0.03:
                # diverged, reset to best
                x = globX.copy()
                v *= 0.3
        # final check
        try:
            mm = _max_cos(x)
            if mm < globV:
                globV = mm
                globX = x.copy()
        except Exception:
            pass
        return x

    def micro_peel(x, best, deadline, max_outer=400):
        n_, d_ = x.shape
        s_list = [1e-3, 3e-4, 1e-4, 3e-5, 1e-5, 3e-6, 1e-6, 3e-7, 1e-7, 5e-8]
        outer = 0
        while time.time() < deadline and outer < max_outer:
            outer += 1
            G = x @ x.T
            np.fill_diagonal(G, -2.0)
            flat = int(np.argmax(G))
            ii = flat // n_
            jj = flat % n_
            m = float(G[ii, jj])
            if m < best:
                best = m
                if m < globV:
                    globV = m
                    globX = x.copy()
            improved = False
            for k, partner in ((ii, jj), (jj, ii)):
                xk = x[k]
                xp = x[partner]
                c = float(np.dot(xk, xp))
                proj = xp - c * xk
                pn = float(np.linalg.norm(proj))
                if pn < 1e-14 or not np.isfinite(pn):
                    continue
                unit = proj / pn
                Gtmp = G.copy()
                Gtmp[k, :] = -2.0
                Gtmp[:, k] = -2.0
                m_excl = float(np.max(Gtmp))
                # if even m_excl already >= best, moving k alone cannot beat best
                # still can improve current m
                for s in s_list:
                    yk = xk - s * unit
                    nn = float(np.linalg.norm(yk))
                    if nn == 0 or not np.finite(nn):
                        continue
                    yk = yk / nn
                    dv = x @ yk
                    dv[k] = -2.0
                    max1 = float(np.max(dv))
                    newm = max1 if max1 > m_excl else m_excl
                    if newm < m - 1e-12 and newm < best + 1e-12:
                        # accept if strictly improves current m
                        # require also improvement over m
                        x[k] = yk
                        m = newm
                        if newm < best:
                            best = newm
                        improved = True
                        break
                if improved:
                    break
            if improved:
                continue
            # no directed improvement -> try tiny random on endpoints
            found = False
            for k in (ii, jj):
                Gtmp = G.copy()
                Gtmp[k, :] = -2.0
                Gtmp[:, k] = -2.0
                m_excl = float(np.max(Gtmp))
                xk = x[k].copy()
                for sigma in (1e-6, 3e-7):
                    for _ in range(60):
                        z = rng.normal(scale=sigma, size=d_)
                        z = z - float(np.dot(z, xk)) * xk
                        yk = xk + z
                        nn = float(np.linalg.norm(yk))
                        if nn == 0 or not np.finite(nn):
                            continue
                        yk = yk / nn
                        dv = x @ yk
                        dv[k] = -2.0
                        max1 = float(np.max(dv))
                        newm = max1 if max1 > m_excl else m_excl
                        if newm < best - 1e-12:
                            x[k] = yk
                            best = newm
                            found = True
                            break
                    if found:
                        break
                if found:
                    break
            if found:
                continue
            break
        return x, best

    def tiny_lp(x, best, deadline, r_init=4e-6):
        iu_r = iu_tr
        iu_c = iu_tc
        ar = np.arange(d)
        radius = float(r_init)
        try:
            G0 = x @ x.T
            best = float(np.max(G0[iu_r, iu_c]))
        except Exception:
            return x, best
        while time.time() < deadline and radius > 1e-13:
            if deadline - time.time() < 2.0:
                break
            G = x @ x.T
            c = G[iu_r, iu_c]
            m = float(np.max(c))
            delta = max(12.0 * radius, 4e-7)
            sel = c > m - delta
            cnt = int(np.count_nonzero(sel))
            if cnt == 0:
                radius *= 0.5
                continue
            if cnt > 4000:
                idx = np.argpartition(-c, 4000)[:4000]
                I = iu_r[idx]; J = iu_c[idx]; csel = c[idx]
                keep = csel > m - delta
                if int(np.count_nonzero(keep)) >= 10:
                    I = I[keep]; J = J[keep]; csel = csel[keep]
                else:
                    I = I[:600]; J = J[:600]; csel = csel[:600]
            else:
                I = iu_r[sel]; J = iu_c[sel]; csel = c[sel]
            p = int(I.size)
            if p == 0:
                radius *= 0.5
                continue
            nv = n * d + 1
            rows = np.repeat(np.arange(p), 2 * d + 1)
            cols = np.concatenate([I[:, None]*d+ar, J[:, None]*d+ar, np.full((p,1), n*d)], axis=1).ravel()
            vals = np.concatenate([x[J], x[I], -np.ones((p,1))], axis=1).ravel()
            try:
                A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
                A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n*d))), shape=(n, nv)).tocsr()
                obj = np.zeros(nv); obj[-1] = 1.0
                res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq, b_eq=np.zeros(n),
                              bounds=[(-radius, radius)]*(n*d)+[(None, None)], method="highs")
            except Exception:
                radius *= 0.5
                continue
            if res.status != 0:
                radius *= 0.5
                continue
            y = x + res.x[:-1].reshape(n, d)
            y = _norm(y)
            if not np.all(np.isfinite(y)):
                radius *= 0.5
                continue
            Gy = y @ y.T
            vv = float(np.max(Gy[iu_r, iu_c]))
            if vv < best - 1e-12:
                x = y; best = vv
                if vv < globV:
                    globV = vv; globX = x.copy()
                radius = min(radius*1.4, 2e-5)
            else:
                radius *= 0.5
        return x, best

    # ---- main basin hopping ----
    # quick initial peel to squeeze starting point
    try:
        gg, gv = micro_peel(globX.copy(), globV, deadline=min(t0+8.0, HARD_END-160.0), max_outer=120)
        if gv < globV:
            globV = gv; globX = gg.copy()
    except Exception:
        pass

    trial = 0
    T_HOP_END = t0 + 138.0
    while time.time() < T_HOP_END and time.time() < HARD_END - 30.0:
        trial += 1
        if time.time() > HARD_END - 32.0:
            break
        try:
            base = globX
            G = base @ base.T
            np.fill_diagonal(G, -2.0)
            pm = np.max(G, axis=1)
            order = np.argsort(-pm)
            mode = trial % 5
            y = base.copy()
            if mode == 0:
                # perturb worst 35 moderately
                k = 35
                worst = order[:k]
                y = y + rng.normal(scale=0.002, size=y.shape)
                y[worst] = y[worst] + rng.normal(scale=0.018, size=(k, d))
                y = _norm(y)
                y = anneal_run(y, N=550, tau0=0.004, tau1=0.0004, lr0=0.008, lr1=0.001, mu=0.85, deadline=min(time.time()+3.0, T_HOP_END))
            elif mode == 1:
                # small global + worst 15 small
                k = 15
                worst = order[:k]
                y = y + rng.normal(scale=0.0012, size=y.shape)
                y[worst] = y[worst] + rng.normal(scale=0.008, size=(k, d))
                y = _norm(y)
                y = anneal_run(y, N=500, tau0=0.002, tau1=0.00025, lr0=0.005, lr1=0.0008, mu=0.88, deadline=min(time.time()+3.0, T_HOP_END))
            elif mode == 2:
                # hole reinsertion for top 1-2 worst
                nw = 1 if (trial % 2 == 0) else 2
                worst = order[:nw]
                y = y + rng.normal(scale=0.001, size=y.shape)
                y = _norm(y)
                for w in worst:
                    try:
                        h = find_hole(y, [w], n_cand=15000, refine=40)
                        y[w] = h
                    except Exception:
                        pass
                y = _norm(y)
                y = anneal_run(y, N=650, tau0=0.005, tau1=0.00035, lr0=0.009, lr1=0.0009, mu=0.82, deadline=min(time.time()+3.5, T_HOP_END))
            elif mode == 3:
                # larger kick to worst 50 to change topology
                k = 50
                worst = order[:k]
                y = y + rng.normal(scale=0.003, size=y.shape)
                y[worst] = y[worst] + rng.normal(scale=0.035, size=(k, d))
                y = _norm(y)
                y = anneal_run(y, N=700, tau0=0.006, tau1=0.0005, lr0=0.010, lr1=0.0012, mu=0.80, deadline=min(time.time()+3.5, T_HOP_END))
            else:
                # fine shake around best
                y = y + rng.normal(scale=0.0015, size=y.shape)
                y = _norm(y)
                y = anneal_run(y, N=400, tau0=0.0015, tau1=0.0002, lr0=0.004, lr1=0.0006, mu=0.90, deadline=min(time.time()+2.5, T_HOP_END))
            # quick peel of result if promising
            try:
                vcur = _max_cos(y)
            except Exception:
                continue
            if vcur < globV + 0.004:
                try:
                    y2, v2 = micro_peel(y, vcur, deadline=min(time.time()+0.9, T_HOP_END), max_outer=80)
                    if v2 < globV:
                        globV = v2; globX = y2.copy()
                except Exception:
                    pass
        except Exception:
            continue

    # intensify: short anneals from best with tiny noise
    while time.time() < HARD_END - 14.0 and time.time() < t0 + 155.0:
        try:
            y = globX + rng.normal(scale=0.0008, size=globX.shape)
            y = _norm(y)
            y = anneal_run(y, N=350, tau0=0.0012, tau1=0.00018, lr0=0.003, lr1=0.0005, mu=0.9, deadline=min(time.time()+2.0, HARD_END-14.0))
            vcur = _max_cos(y)
            if vcur < globV + 0.002:
                y2, v2 = micro_peel(y, vcur, deadline=min(time.time()+0.7, HARD_END-14.0), max_outer=60)
                if v2 < globV:
                    globV = v2; globX = y2.copy()
            else:
                break
        except Exception:
            break

    x = globX.copy()
    best = float(globV)
    # final squeeze: peel then LP then peel
    if time.time() < HARD_END - 2.0:
        try:
            x, best = micro_peel(x, best, deadline=min(HARD_END-9.0, time.time()+6.0), max_outer=500)
            globX = x.copy(); globV = best; best = float(best)
        except Exception:
            pass
    if time.time() < HARD_END - 2.0:
        try:
            x, best = tiny_lp(x, best, deadline=HARD_END-1.0, r_init=4e-6)
        except Exception:
            pass
    if time.time() < HARD_END - 0.6:
        try:
            x, best = micro_peel(x, best, deadline=HARD_END-0.2, max_outer=300)
        except Exception:
            pass
    x = _norm(x)
    # guard: keep global best if final worse due to numerics
    try:
        fv = _max_cos(x)
        if fv > globV and np.all(np.isfinite(globX)):
            x = _norm(globX.copy())
    except Exception:
        pass
    return x
