import numpy as np
import time as _time

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def _norm(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return X / n

def solve():
    import time as _tm
    import numpy as _np
    np = _np
    t0 = _tm.time()
    DEADLINE = t0 + 168.0
    rng = np.random.default_rng(12345)
    n, d = 59, 22
    dm = d - 1
    N = n * dm

    cands = []
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn).astype(np.float64)
            if a.shape == (59, 22) and np.all(np.isfinite(a)):
                a = _norm(a)
                cands.append(a)
        except Exception:
            pass
    if not cands:
        cands.append(_norm(rng.standard_normal((59, 22))))
    vals = [_max_cos(c) for c in cands]
    best = cands[int(np.argmin(vals))].copy()
    best_val = float(min(vals))
    M0 = best_val

    try:
        from scipy.optimize import linprog
        HAS_LP = True
    except Exception:
        HAS_LP = False

    def adam_sm(X0, steps, b0, b1, lr0, lr1):
        X = X0.copy()
        m = np.zeros_like(X)
        v = np.zeros_like(X)
        b1m, b2m, eps = 0.9, 0.999, 1e-8
        lb0, lb1 = np.log(b0), np.log(b1)
        ll0, ll1 = np.log(lr0), np.log(lr1)
        cur_best = X.copy()
        cur_val = _max_cos(X)
        p1 = 1.0; p2 = 1.0
        for t in range(1, steps + 1):
            if (t & 127) == 0 and _tm.time() > DEADLINE - 3:
                break
            f = t / steps
            beta = float(np.exp(lb0 + (lb1 - lb0) * f))
            lr = float(np.exp(ll0 + (ll1 - ll0) * f))
            G = X @ X.T
            np.fill_diagonal(G, -2.0)
            Mx = float(np.max(G))
            G -= Mx
            G *= beta
            np.exp(G, out=G)
            np.fill_diagonal(G, 0.0)
            S = float(np.sum(G))
            if not np.isfinite(S) or S <= 0:
                continue
            Gr = (G @ X) / S
            dot = np.sum(Gr * X, axis=1, keepdims=True)
            Gr -= dot * X
            p1 *= b1m; p2 *= b2m
            m *= b1m; m += 0.1 * Gr
            v *= b2m; v += 0.001 * (Gr * Gr)
            mc = m / (1.0 - p1)
            vc = v / (1.0 - p2)
            X -= lr * mc / (np.sqrt(vc) + eps)
            nn = np.linalg.norm(X, axis=1, keepdims=True)
            nn[nn == 0] = 1.0
            X /= nn
            if (t % 400 == 0) or (t == steps):
                cv = _max_cos(X)
                if cv < cur_val:
                    cur_val = cv
                    cur_best = X.copy()
        fv = _max_cos(X)
        if fv < cur_val:
            cur_val = fv
            cur_best = X.copy()
        return cur_best, cur_val

    def tangent_bases(X):
        Bs = np.empty((n, d, dm), dtype=np.float64)
        for i in range(n):
            xi = X[i]
            k = int(np.argmax(np.abs(xi)))
            P = np.empty((d, dm), dtype=np.float64)
            c = 0
            for j in range(d):
                if j == k:
                    continue
                e = np.zeros(d)
                e[j] = 1.0
                P[:, c] = e - xi[j] * xi
                c += 1
            Q, _ = np.linalg.qr(P)
            # qr reduced gives d x dm
            if Q.shape != (d, dm):
                Q = Q[:, :dm]
            Bs[i] = Q
        return Bs

    def active_pairs(X, thresh, cap):
        G = X @ X.T
        iu, ju = np.triu_indices(n, 1)
        g = G[iu, ju]
        Mx = float(np.max(g))
        if cap is not None:
            order = np.argsort(-g)
            # take those above Mx-thresh, but at least cap tightest
            mask_thr = g[order] > (Mx - thresh)
            cnt_thr = int(np.sum(mask_thr))
            take = max(cnt_thr, min(cap, len(order)))
            # but don't take too loose: limit to Mx-0.003 if take==cap
            sel = order[:take]
            # filter very loose
            if take == cap:
                keep = g[sel] > (Mx - 0.003)
                # keep at least 300 tightest
                if int(np.sum(keep)) < 300:
                    sel = order[:300]
                else:
                    sel = sel[keep]
            ii = iu[sel]; jj = ju[sel]
            gg = g[sel]
        else:
            sel = np.where(g > (Mx - thresh))[0]
            ii = iu[sel]; jj = ju[sel]
            gg = g[sel]
        return ii, jj, gg, Mx

    def solve_standard_lp(X, delta, thresh=1e-5):
        Bs = tangent_bases(X)
        ii, jj, gg, Mx = active_pairs(X, thresh, cap=None)
        P = len(ii)
        if P == 0 or P > 2000:
            return None, None, None, None
        A = np.zeros((P, N + 1), dtype=np.float64)
        for r in range(P):
            i = int(ii[r]); j = int(jj[r])
            ai = Bs[i].T @ X[j]
            aj = Bs[j].T @ X[i]
            A[r, i * dm:(i + 1) * dm] = ai
            A[r, j * dm:(j + 1) * dm] = aj
            A[r, N] = -1.0
        b = -gg
        c = np.zeros(N + 1); c[N] = 1.0
        bounds = [(-delta, delta)] * N + [(-2.0, 2.0)]
        try:
            res = linprog(c, A_ub=A, b_ub=b, bounds=bounds, method='highs')
        except Exception:
            return None, None, None, None
        if res.status != 0:
            return None, None, None, None
        u = np.asarray(res.x[:N], dtype=np.float64)
        topt = float(res.x[N])
        duals = None
        try:
            if hasattr(res, 'ineqlin') and res.ineqlin is not None and hasattr(res.ineqlin, 'marginals'):
                duals = np.asarray(res.ineqlin.marginals, dtype=np.float64).ravel()
            elif hasattr(res, 'marginals'):
                duals = np.asarray(res.marginals, dtype=np.float64).ravel()
        except Exception:
            duals = None
        return u, topt, (ii, jj, gg), duals

    def solve_forced_lp(X, Bs, ii, jj, gg, Sset, Mref, drop, rise, delta):
        P = len(ii)
        A = np.zeros((P, N + 1), dtype=np.float64)
        b = np.empty(P, dtype=np.float64)
        is_sel = np.zeros(P, dtype=bool)
        if len(Sset) > 0:
            is_sel[list(Sset)] = True
        target_low = Mref - drop
        for r in range(P):
            i = int(ii[r]); j = int(jj[r])
            ai = Bs[i].T @ X[j]
            aj = Bs[j].T @ X[i]
            A[r, i * dm:(i + 1) * dm] = ai
            A[r, j * dm:(j + 1) * dm] = aj
            if is_sel[r]:
                A[r, N] = 0.0
                b[r] = target_low - float(gg[r])
            else:
                A[r, N] = -1.0
                b[r] = -float(gg[r])
        c = np.zeros(N + 1); c[N] = 1.0
        bounds = [(-delta, delta)] * N + [(-2.0, Mref + rise)]
        try:
            res = linprog(c, A_ub=A, b_ub=b, bounds=bounds, method='highs')
        except Exception:
            return None, None
        if res.status != 0:
            return None, None
        u = np.asarray(res.x[:N], dtype=np.float64)
        topt = float(res.x[N])
        return u, topt

    def apply_u(X, Bs, u):
        U = u.reshape(n, dm)
        Xn = np.empty_like(X)
        for i in range(n):
            Xn[i] = X[i] + Bs[i] @ U[i]
        return _norm(Xn)

    # ---- get LP marginals / dominant contacts from incumbent ----
    dom_ii = dom_jj = dom_gg = None
    duals0 = None
    if HAS_LP and _tm.time() < DEADLINE - 30:
        try:
            _, _, act, du = solve_standard_lp(best, delta=0.005, thresh=1e-5)
            if act is not None:
                dom_ii, dom_jj, dom_gg = act[0], act[1], act[2]
                duals0 = du
        except Exception:
            pass
    if dom_ii is None:
        G0 = best @ best.T
        iu, ju = np.triu_indices(n, 1)
        g0 = G0[iu, ju]
        order = np.argsort(-g0)[:600]
        dom_ii, dom_jj, dom_gg = iu[order], ju[order], g0[order]
        duals0 = None

    # rank dominant contacts by |dual| or tightness
    P0 = len(dom_ii)
    if duals0 is not None and len(duals0) == P0 and np.all(np.isfinite(duals0)):
        w = np.abs(duals0)
        if np.max(w) <= 0 or not np.all(np.isfinite(w)):
            w = (dom_gg - np.min(dom_gg) + 1e-12)
        rank = np.argsort(-w)
    else:
        rank = np.argsort(-dom_gg)
        w = None

    # ---- forced contact-breaking trials ----
    n_trials = 0
    max_trials = 14
    # predefine subset sizes 20-50
    trial_rng = np.random.default_rng(777)
    while _tm.time() < DEADLINE - 14 and n_trials < max_trials:
        n_trials += 1
        # select 20-50 dominant contacts
        ksel = int(trial_rng.integers(20, 51))
        # candidate pool: top 120 by rank + some random
        pool = rank[:min(120, P0)]
        if len(pool) < ksel:
            pool = rank[:P0]
        # weighted sample: prefer larger dual/tighter
        if w is not None:
            pw = np.abs(w[pool]).astype(np.float64)
            pw[~np.isfinite(pw)] = 0.0
            pw += 1e-12
            pw /= pw.sum()
            # mix with uniform to diversify across trials
            if (n_trials % 3) == 0:
                sel_idx = trial_rng.choice(len(pool), size=ksel, replace=False)
            else:
                sel_idx = trial_rng.choice(len(pool), size=ksel, replace=False, p=pw)
        else:
            # random subset of tightest + random pairs to diversify
            if (n_trials % 2) == 0:
                sel_idx = trial_rng.choice(len(pool), size=ksel, replace=False)
            else:
                # mix top-40 + random from rest
                a = trial_rng.choice(min(40, len(pool)), size=min(ksel // 2, 20), replace=False)
                b = trial_rng.choice(P0, size=ksel - len(a), replace=False)
                sel_idx = np.unique(np.concatenate([pool[a], b]))[:ksel]
                if len(sel_idx) < ksel:
                    extra = trial_rng.choice(P0, size=ksel - len(sel_idx), replace=False)
                    sel_idx = np.unique(np.concatenate([sel_idx, extra]))[:ksel]
        # map pool indices to positions in dom arrays -> but forced LP uses its own active set;
        # we map by pair identity: build set of pair tuples to force
        force_pairs = set()
        for s in sel_idx:
            p = int(pool[int(s)]) if (n_trials % 2 == 0 or w is not None) else int(sel_idx[int(np.where(sel_idx==s)[0][0])] if False else 0)
            force_pairs.add(int(p))
        # simpler: if we used mixed indexing above, rebuild:
        # Actually recompute S positions directly:
        # build list of forced pair keys
        forced_keys = set()
        if w is not None or (n_trials % 2 == 0):
            for s in sel_idx:
                pp = int(pool[int(s)])
                forced_keys.add((int(dom_ii[pp]), int(dom_jj[pp])))
        else:
            for pp in sel_idx:
                # sel_idx already holds absolute indices in dom arrays
                forced_keys.add((int(dom_ii[int(pp)]), int(dom_jj[int(pp)])))

        drop = 2e-4
        rise = 1e-4
        # vary drop/rise slightly per trial for diversity
        if (n_trials % 4) == 1:
            drop = 2.5e-4; rise = 1.2e-4
        elif (n_trials % 4) == 2:
            drop = 2e-4; rise = 0.8e-4
        elif (n_trials % 4) == 3:
            drop = 3e-4; rise = 1.5e-4

        Xc = best.copy()
        n_inner = 6
        delta_f = 0.012 if (n_trials % 2 == 0) else 0.020
        ok = True
        for inner in range(n_inner):
            if _tm.time() > DEADLINE - 12:
                ok = False
                break
            Bs = tangent_bases(Xc)
            ii, jj, gg, Mx = active_pairs(Xc, thresh=2e-3, cap=1200)
            # find Sset indices within current active set by key match
            key_to_pos = {(int(ii[r]), int(jj[r])): r for r in range(len(ii))}
            Sset = []
            for key in forced_keys:
                if key in key_to_pos:
                    Sset.append(key_to_pos[key])
                else:
                    rk = (key[1], key[0])
                    if rk in key_to_pos:
                        Sset.append(key_to_pos[rk])
            if len(Sset) == 0:
                # forced pairs drifted out of active set -> success (broken); break to polish
                break
            # cap Sset to 50
            if len(Sset) > 50:
                Sset = list(trial_rng.choice(Sset, size=50, replace=False))
            u, topt = solve_forced_lp(Xc, Bs, ii, jj, gg, Sset, M0, drop, rise, delta_f)
            if u is None:
                ok = False
                break
            Xc = apply_u(Xc, Bs, u)
            # shrink trust slightly
            delta_f *= 0.92
            if _max_cos(Xc) > M0 + 8e-4:
                # diverged too far; still polish but stop forcing
                break
        if not ok:
            continue
        # short Adam polish after forced move
        Yp, Yv = adam_sm(Xc, steps=1100, b0=110, b1=900, lr0=0.004, lr1=0.00035)
        if Yv < best_val - 1e-12:
            best, best_val = Yp.copy(), Yv
            M0 = best_val
            # refresh dominant set from new incumbent
            if HAS_LP and _tm.time() < DEADLINE - 20:
                try:
                    _, _, act, du = solve_standard_lp(best, delta=0.005, thresh=1e-5)
                    if act is not None:
                        dom_ii, dom_jj, dom_gg = act[0], act[1], act[2]
                        duals0 = du
                        P0 = len(dom_ii)
                        if duals0 is not None and len(duals0) == P0 and np.all(np.isfinite(duals0)):
                            w = np.abs(duals0)
                            rank = np.argsort(-w)
                        else:
                            rank = np.argsort(-dom_gg); w = None
                except Exception:
                    pass
        # also try a second polish seed with tiny jitter if close
        if _tm.time() > DEADLINE - 12:
            break

    # final intensification on best
    if _tm.time() < DEADLINE - 4:
        Y, vy = adam_sm(best, steps=2200, b0=850, b1=2400, lr0=0.0012, lr1=0.0001)
        if vy < best_val:
            best, best_val = Y, vy
    if _tm.time() < DEADLINE - 2:
        Y, vy = adam_sm(best, steps=900, b0=2000, b1=4000, lr0=0.0005, lr1=0.00005)
        if vy < best_val:
            best, best_val = Y, vy
    best = _norm(best)
    return best
