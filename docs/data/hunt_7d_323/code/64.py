import time
import numpy as np

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _contact_set(X, m, tol=1e-7):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    iu = np.triu_indices(X.shape[0], 1)
    sel = G[iu] > (m - tol)
    return set(zip(iu[0][sel].tolist(), iu[1][sel].tolist()))

def solve():
    t0 = time.time()
    LIMIT = 168.0
    rng = np.random.default_rng(1)
    # load start
    Xbest = None
    mbest = np.inf
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn).astype(np.float64)
            if a.shape == (323, 7):
                a = a / np.linalg.norm(a, axis=1, keepdims=True)
                v = _max_cos(a)
                if v < mbest:
                    mbest = v
                    Xbest = a.copy()
        except Exception:
            pass
    if Xbest is None:
        Xbest = rng.standard_normal(size=(323, 7))
        Xbest /= np.linalg.norm(Xbest, axis=1, keepdims=True)
        mbest = _max_cos(Xbest)
    m0 = float(mbest)
    n, d = Xbest.shape
    Xcur = Xbest.copy()
    mcur = float(mbest)
    set_best = _contact_set(Xbest, mbest, tol=1e-6)

    tau_soft_reg = 0.003
    tau_soft_rel = 0.0012
    eps_s = 0.004
    tau_s = 0.0010
    thr_r_global = m0 - 0.002

    b1, b2, eps_ad = 0.9, 0.999, 1e-8

    def one_phase(Xinit, steps, lr, use_reg, forced_pairs, delta, w_var=0.8, w_rat_scale=100.0, w_hinge_scale=2000.0):
        X = Xinit.copy()
        m_ad = np.zeros_like(X)
        v_ad = np.zeros_like(X)
        b1t = 1.0
        b2t = 1.0
        thr_h = m0 - delta
        # pre convert forced to arrays
        if forced_pairs is not None and len(forced_pairs) > 0:
            Fp = np.array([p[0] for p in forced_pairs], dtype=np.int64)
            Fq = np.array([p[1] for p in forced_pairs], dtype=np.int64)
            nF = len(Fp)
        else:
            nF = 0
        for t in range(1, steps + 1):
            if time.time() - t0 > LIMIT - 12:
                break
            # normalize (should already be)
            # G
            G = X @ X.T
            np.fill_diagonal(G, -2.0)
            m = float(G.max())
            # soft max grad
            if use_reg:
                tau = tau_soft_reg
            else:
                tau = tau_soft_rel
            # E
            E = np.exp((G - m) / tau)
            np.fill_diagonal(E, 0.0)
            sE = E.sum()
            if sE <= 0:
                sE = 1.0
            W = E / sE
            g_soft = 2.0 * (W @ X)
            ns = float(np.linalg.norm(g_soft))
            if ns < 1e-14:
                ns = 1.0
            g_soft_n = g_soft / ns
            if not use_reg:
                g = g_soft_n
            else:
                # variance term
                thr_s = m - eps_s
                Z = (G - thr_s) / tau_s
                Z = np.clip(Z, -20.0, 20.0)
                S = 1.0 / (1.0 + np.exp(-Z))
                np.fill_diagonal(S, 0.0)
                s_cnt = S.sum(axis=1)
                mean_s = s_cnt.mean()
                a_coef = 2.0 * (s_cnt - mean_s) / n  # (n,)
                Q = S * (1.0 - S) / tau_s
                A = a_coef[:, None] * Q
                g_var_raw = (A + A.T) @ X
                nv = float(np.linalg.norm(g_var_raw))
                if nv < 1e-14:
                    g_var_n = np.zeros_like(X)
                else:
                    g_var_n = g_var_raw / nv
                # hinge grad raw
                g_h = np.zeros_like(X)
                if nF > 0:
                    gvals = G[Fp, Fq]
                    exc = gvals - thr_h
                    pos = exc > 0
                    if np.any(pos):
                        coeff = 2.0 * exc[pos] / nF  # per pair
                        # accumulate
                        # use np.add.at for correctness
                        np.add.at(g_h, Fp[pos], coeff[:, None] * X[Fq[pos]])
                        np.add.at(g_h, Fq[pos], coeff[:, None] * X[Fp[pos]])
                # rattler grad raw
                # w_i and argmax
                jstar = np.argmax(G, axis=1)
                wrow = G[np.arange(n), jstar]
                deficit = thr_r_global - wrow
                g_r = np.zeros_like(X)
                mask = deficit > 0
                if np.any(mask):
                    idx = np.where(mask)[0]
                    coeff_r = -2.0 * deficit[idx] / n
                    # grad_i += coeff*X_j, grad_j += coeff*X_i
                    g_r[idx] += coeff_r[:, None] * X[jstar[idx]]
                    # second part accumulation (may collide)
                    np.add.at(g_r, jstar[idx], coeff_r[:, None] * X[idx])
                g = g_soft_n * 1.0 + g_var_n * w_var + g_h * w_hinge_scale + g_r * w_rat_scale
            # tangent projection
            dot = (g * X).sum(axis=1, keepdims=True)
            g = g - dot * X
            # adam
            m_ad = b1 * m_ad + (1 - b1) * g
            v_ad = b2 * v_ad + (1 - b2) * (g * g)
            b1t *= b1
            b2t *= b2
            mhat = m_ad / (1.0 - b1t)
            vhat = v_ad / (1.0 - b2t)
            step = lr * mhat / (np.sqrt(vhat) + eps_ad)
            # limit step norm to avoid explosion
            # simple global clip
            sn = float(np.linalg.norm(step))
            max_sn = 0.02
            if sn > max_sn:
                step = step * (max_sn / sn)
            X = X - step
            X /= np.linalg.norm(X, axis=1, keepdims=True)
        return X

    # basin hopping cycles
    n_cycles = 7
    for cyc in range(n_cycles):
        if time.time() - t0 > LIMIT - 25:
            break
        # identify tight edges of Xcur
        Gc = Xcur @ Xcur.T
        np.fill_diagonal(Gc, -2.0)
        mc = float(Gc.max())
        iu0 = np.triu_indices(n, 1)
        cvals = Gc[iu0]
        # tight threshold: top ~2500
        # use m - 5e-7 as tight
        tight_mask = cvals > (mc - 5e-7)
        ti = iu0[0][tight_mask]
        tj = iu0[1][tight_mask]
        ntight = len(ti)
        if ntight < 50:
            # take top 1500
            order = np.argpartition(cvals, -1500)[-1500:]
            ti = iu0[0][order]
            tj = iu0[1][order]
            ntight = len(ti)
        # contact counts for hub bias
        cnt = np.zeros(n, dtype=np.int64)
        # counts at 1e-6
        cmask = cvals > (mc - 1e-6)
        np.add.at(cnt, iu0[0][cmask], 1)
        np.add.at(cnt, iu0[1][cmask], 1)
        hubs = np.argsort(-cnt)[:24]
        hub_set = set(hubs.tolist())
        # build forced list: half from hub-incident tight edges, half random tight
        nF_want = 180 + int(rng.integers(0, 70))
        delta = float(rng.uniform(1e-3, 5e-3))
        forced = []
        # hub incident candidates
        hub_cand_idx = [k for k in range(ntight) if (ti[k] in hub_set or tj[k] in hub_set)]
        rng.shuffle(hub_cand_idx)
        # random order for all
        all_idx = np.arange(ntight)
        # shuffle via rng
        perm = rng.permutation(ntight)
        # take
        take_hub = min(len(hub_cand_idx), nF_want // 2)
        for k in range(take_hub):
            ii = hub_cand_idx[k]
            forced.append((int(ti[ii]), int(tj[ii])))
        need = nF_want - len(forced)
        # fill from perm
        for ii in perm:
            if need <= 0:
                break
            e = (int(ti[ii]), int(tj[ii]))
            if e in forced:
                continue
            forced.append(e)
            need -= 1
        # regularized phase
        lr_reg = float(rng.uniform(0.0025, 0.005))
        Xreg = one_phase(Xcur, steps=550, lr=lr_reg, use_reg=True, forced_pairs=forced, delta=delta)
        if time.time() - t0 > LIMIT - 20:
            # evaluate anyway
            pass
        # pure relaxation
        Xnew = one_phase(Xreg, steps=220, lr=0.003, use_reg=False, forced_pairs=None, delta=delta)
        mnew = _max_cos(Xnew)
        # diversity: contact set diff
        set_new = _contact_set(Xnew, mnew, tol=1e-6)
        # symmetric difference ratio
        union = len(set_best.union(set_new))
        if union == 0:
            diff_ratio = 0.0
        else:
            diff_ratio = len(set_best.symmetric_difference(set_new)) / max(1, len(set_best))
        # also compare to current set?
        # acceptance for best
        if mnew < mbest:
            # improvement (even tiny) -> take
            Xbest = Xnew.copy()
            mbest = float(mnew)
            set_best = set_new
            Xcur = Xnew.copy()
            mcur = float(mnew)
        elif mnew < mcur + 0.002 and diff_ratio > 0.04:
            # move to new basin (diversify) even if slightly worse than best but within 0.002 of current/best?
            # require within 0.002 of best to avoid drift
            if mnew < mbest + 0.002:
                Xcur = Xnew.copy()
                mcur = float(mnew)
            # else stay
        else:
            # stay at cur, but add small kick to avoid repeating same forced set? keep cur as is
            # if stuck, restart cur from best with small noise every 3 cycles
            if cyc % 3 == 2:
                noise = rng.standard_normal(size=(n, d)) * 0.005
                Xcur = Xbest + noise
                Xcur /= np.linalg.norm(Xcur, axis=1, keepdims=True)
                mcur = _max_cos(Xcur)
        # early exit if clearly better? continue to polish
    # final polish: pure relaxation longer from best
    if time.time() - t0 < LIMIT - 8:
        Xpol = one_phase(Xbest, steps=450, lr=0.0025, use_reg=False, forced_pairs=None, delta=0.002)
        mpol = _max_cos(Xpol)
        if mpol < mbest:
            Xbest = Xpol
            mbest = mpol
    # tiny greedy hill climb on worst pairs (fast, no LP)
    # do a few random tangent kicks on worst points
    try:
        remaining = LIMIT - (time.time() - t0) - 2.0
        if remaining > 4:
            # identify worst points
            G = Xbest @ Xbest.T
            np.fill_diagonal(G, -2.0)
            worst = G.max(axis=1)
            order_pts = np.argsort(-worst)[:40]
            rng2 = np.random.default_rng(2)
            best_local = mbest
            Xloc = Xbest.copy()
            t_end = time.time() + min(remaining - 1, 10)
            # simple stochastic hill climb
            while time.time() < t_end:
                k = int(order_pts[rng2.integers(0, len(order_pts))])
                step_scale = 0.002
                trial = Xloc[k] + rng2.standard_normal(size=(d,)) * step_scale
                trial /= np.linalg.norm(trial)
                old = Xloc[k].copy()
                Xloc[k] = trial
                v = _max_cos(Xloc)
                if v < best_local:
                    best_local = v
                    # update worst order occasionally
                    if rng2.random() < 0.05:
                        G2 = Xloc @ Xloc.T
                        np.fill_diagonal(G2, -2.0)
                        worst2 = G2.max(axis=1)
                        order_pts = np.argsort(-worst2)[:40]
                else:
                    Xloc[k] = old
            if best_local < mbest:
                Xbest = Xloc
                mbest = best_local
    except Exception:
        pass
    Xbest = Xbest / np.linalg.norm(Xbest, axis=1, keepdims=True)
    return Xbest
