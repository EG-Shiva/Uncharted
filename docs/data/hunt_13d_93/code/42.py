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
    return float(np.max(G))

def _rowmax(G):
    return G.max(axis=1)

def _refine_cands(C, Others, steps=110):
    # C: (B,d) candidates, Others: (m,d)
    V = C.copy()
    B = V.shape[0]
    for s in range(steps):
        f = s / max(1, steps - 1)
        tau = 0.008 * ((0.001 / 0.008) ** f)
        lr = 0.03 * ((0.005 / 0.03) ** f)
        D = V @ Others.T  # B x m
        m0 = D.max(axis=1, keepdims=True)
        W = np.exp((D - m0) / tau)
        W /= (W.sum(axis=1, keepdims=True) + 1e-300)
        Gg = W @ Others
        dot = (Gg * V).sum(axis=1, keepdims=True)
        Gt = Gg - dot * V
        nrm = np.linalg.norm(Gt, axis=1, keepdims=True) + 1e-12
        V = V - lr * (Gt / nrm)
        V /= (np.linalg.norm(V, axis=1, keepdims=True) + 1e-300)
    return V

def _pnorm_adam(X, iu0, iu1, steps_per_stage=50, p_list=(32, 64, 128), shift=0.20, lr0=0.012, lr1=0.002):
    n, d = X.shape
    m = np.zeros_like(X)
    v = np.zeros_like(X)
    b1, b2, eps = 0.9, 0.999, 1e-8
    total = len(p_list) * steps_per_stage
    tglob = 0
    for p in p_list:
        m[:] = 0.0
        v[:] = 0.0
        tt = 0
        pm1 = float(p - 1.0)
        for s in range(steps_per_stage):
            frac = tglob / max(1, total - 1)
            lr = lr0 * ((lr1 / lr0) ** frac)
            G = X @ X.T
            c = G[iu0, iu1]
            pos = c > shift
            gt = None
            if np.any(pos):
                sc = c[pos] - shift
                smax = float(sc.max())
                if smax > 0 and np.isfinite(smax):
                    ratio = sc / smax
                    np.clip(ratio, 1e-300, 1.0, out=ratio)
                    epos = np.exp(pm1 * np.log(ratio))
                    esum = float(epos.sum())
                    if esum > 0 and np.isfinite(esum):
                        w = np.zeros_like(c)
                        w[pos] = epos / esum
                        W = np.zeros((n, n), dtype=np.float64)
                        W[iu0, iu1] = w
                        W[iu1, iu0] = w
                        grad = W @ X
                        dot = (grad * X).sum(axis=1, keepdims=True)
                        gt = grad - dot * X
            tglob += 1
            if gt is None:
                continue
            if not np.all(np.isfinite(gt)):
                continue
            tt += 1
            m *= b1
            m += (1.0 - b1) * gt
            v *= b2
            v += (1.0 - b2) * (gt * gt)
            mh = m / (1.0 - b1 ** tt)
            vh = v / (1.0 - b2 ** tt)
            step = lr * mh / (np.sqrt(vh) + eps)
            dot2 = (step * X).sum(axis=1, keepdims=True)
            step_t = step - dot2 * X
            X = X - step_t
            X = _norm(X)
    return X

def _hardK(X, iu0, iu1, iters=260, K=150, step0=0.0015):
    best = _maxcos(X)
    step = float(step0)
    mtot = len(iu0)
    K = int(min(K, mtot))
    grad = np.empty_like(X)
    for _ in range(iters):
        G = X @ X.T
        c = G[iu0, iu1]
        if K < mtot:
            idx = np.argpartition(-c, K)[:K]
        else:
            idx = np.arange(mtot)
        I = iu0[idx]
        J = iu1[idx]
        grad.fill(0.0)
        np.add.at(grad, I, X[J])
        np.add.at(grad, J, X[I])
        grad /= float(len(idx))
        dot = (grad * X).sum(axis=1, keepdims=True)
        gt = grad - dot * X
        ng = float(np.linalg.norm(gt))
        if ng == 0 or not np.isfinite(ng):
            step *= 0.5
            if step < 1e-13:
                break
            continue
        Xt = X - step * gt
        Xt = _norm(Xt)
        G2 = Xt @ Xt.T
        np.fill_diagonal(G2, -2.0)
        vv = float(G2.max())
        if vv < best:
            X = Xt
            best = vv
            step = min(step * 1.25, 0.008)
        else:
            step *= 0.5
            if step < 1e-13:
                break
    return X, best

def _lp_polish_short(x, seconds=2.0, r_init=2e-4):
    x = _norm(x.copy())
    n, d = x.shape
    best = _maxcos(x)
    radius = float(r_init)
    start = time.time()
    iu = np.triu_indices(n, 1)
    ar = np.arange(d)
    it = 0
    while time.time() - start < seconds and radius > 1e-13 and it < 4:
        it += 1
        Gfull = x @ x.T
        c_all = Gfull[iu]
        mm = float(c_all.max())
        thresh = max(25.0 * radius, 5e-8)
        sel = c_all > mm - thresh
        cnt = int(sel.sum())
        if cnt == 0:
            radius *= 0.5
            continue
        if cnt > 1500:
            idx = np.argpartition(-c_all, 1500)[:1500]
            mask = np.zeros_like(sel)
            mask[idx] = True
            sel = mask
        I = iu[0][sel]
        J = iu[1][sel]
        p = len(I)
        csub = c_all[sel]
        nv = n * d + 1
        try:
            rows = np.repeat(np.arange(p), 2 * d + 1)
            cols = np.concatenate([I[:, None] * d + ar, J[:, None] * d + ar, np.full((p, 1), n * d)], axis=1).ravel()
            vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], axis=1).ravel()
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            b_ub = -csub
            # equality: x_i . dx_i = 0
            rows_eq = np.repeat(np.arange(n), d)
            cols_eq = np.arange(n * d)
            vals_eq = x.ravel()
            A_eq = coo_matrix((vals_eq, (rows_eq, cols_eq)), shape=(n, nv)).tocsr()
            b_eq = np.zeros(n)
            c_obj = np.zeros(nv)
            c_obj[-1] = 1.0
            bounds = [(-radius, radius)] * (n * d) + [(mm - 0.01, mm + 0.005)]
            res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs',
                          options={'time_limit': max(0.5, seconds - (time.time() - start))})
            if not res.success:
                radius *= 0.5
                continue
            dx = res.x[:n * d].reshape(n, d)
            tval = float(res.x[-1])
            # trust region check
            if np.max(np.abs(dx)) > radius * 1.0001:
                radius *= 0.5
                continue
            xn = _norm(x + dx)
            vv = _maxcos(xn)
            if vv < best - 1e-12:
                x = xn
                best = vv
                radius = min(radius * 1.5, 5e-4)
            else:
                radius *= 0.5
        except Exception:
            radius *= 0.5
            continue
    return x, best

def solve():
    t0 = time.time()
    DEADLINE = 172.0
    rng = np.random.default_rng(12345)
    n, d = 93, 13
    iu0, iu1 = np.triu_indices(n, 1)
    best = None
    best_val = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            if a.shape != (93, 13):
                continue
            a = _norm(np.asarray(a, dtype=np.float64))
            vv = _maxcos(a)
            if vv < best_val:
                best_val = vv
                best = a.copy()
        except Exception:
            pass
    if best is None:
        best = _norm(rng.standard_normal((93, 13)))
        best_val = _maxcos(best)
    incumbent = best.copy()
    inc_val = float(best_val)

    # contact analysis
    G0 = incumbent @ incumbent.T
    np.fill_diagonal(G0, -2.0)
    gmax = float(G0.max())
    rowmax = G0.max(axis=1)
    c9 = (G0 >= gmax - 1e-9).sum(axis=1)  # includes? diag is -2 so fine
    # rattlers: zero contacts (no pair within 1e-9 of max)
    rattlers = np.where(c9 == 0)[0]
    # sort rattlers by increasing row-max (loosest first)
    if len(rattlers) > 0:
        order = np.argsort(rowmax[rattlers])
        rattlers_sorted = rattlers[order]
    else:
        # fallback: sort all by rowmax
        rattlers_sorted = np.argsort(rowmax)[:19]
    # hub: max contacts
    hub = int(np.argmax(c9))
    # ensure we have 19 entries for deletion ordering; pad if needed
    if len(rattlers_sorted) < 19:
        # pad with next-loosest points
        all_sorted = np.argsort(rowmax)
        extra = [int(i) for i in all_sorted if int(i) not in set(rattlers_sorted.tolist()) and int(i) != hub]
        need = 19 - len(rattlers_sorted)
        rattlers_sorted = np.array(list(rattlers_sorted) + extra[:need], dtype=int)
    else:
        rattlers_sorted = np.array(rattlers_sorted[:19], dtype=int)

    ks = [1, 2, 3, 5, 8, 13, 19]
    trials = []
    for k in ks:
        trials.append((k, 'rattler'))
    for k in ks:
        trials.append((k, 'hub'))
    # reorder to interleave small first: sort by k, rattler before hub
    trials = sorted(trials, key=lambda z: (z[0], 0 if z[1] == 'rattler' else 1))
    # move 1,2,3 first etc already sorted

    for (k, mode) in trials:
        remain = (t0 + DEADLINE) - time.time()
        # reserve 8s for final safety
        if remain < 9:
            break
        # estimate budget per remaining trial
        # strict time budget per trial
        idx_rem = trials.index((k, mode))
        n_rem = len(trials) - idx_rem
        budget = min(14.0, max(4.0, (remain - 6.0) / max(1, n_rem)))
        t_trial_start = time.time()
        try:
            if mode == 'rattler':
                del_idx = [int(x) for x in rattlers_sorted[:k]]
            else:
                if k == 1:
                    del_idx = [int(hub)]
                else:
                    # delete hub + k-1 loosest rattlers (excluding hub if present)
                    base_r = [int(x) for x in rattlers_sorted if int(x) != hub][:k - 1]
                    # if not enough (hub was in list), pad
                    if len(base_r) < k - 1:
                        all_sorted = np.argsort(rowmax)
                        for ci in all_sorted:
                            ci = int(ci)
                            if ci != hub and ci not in base_r:
                                base_r.append(ci)
                            if len(base_r) >= k - 1:
                                break
                    del_idx = base_r + [int(hub)]
            del_set = set(del_idx)
            keep_mask = np.ones(n, dtype=bool)
            keep_mask[np.array(del_idx, dtype=int)] = False
            # order of reinsertion: shuffled deterministically per trial
            order_ins = del_idx.copy()
            # deterministic shuffle using trial-dependent seed
            rs = np.random.default_rng(1000 + k * 10 + (0 if mode == 'rattler' else 1))
            rs.shuffle(order_ins)
            X = incumbent.copy()
            # current stack of fixed points
            stack = X[keep_mask].copy()
            # greedy reinsert one at a time
            for cpos in order_ins:
                if time.time() - t_trial_start > budget * 0.65:
                    break
                if time.time() > t0 + DEADLINE - 7:
                    break
                m_cur = stack.shape[0]
                # sample 60000 random unit vectors
                C = rng.standard_normal((60000, d))
                C /= (np.linalg.norm(C, axis=1, keepdims=True) + 1e-300)
                D = C @ stack.T
                worst = D.max(axis=1)
                # best 20 candidates
                if 20 < len(worst):
                    bi = np.argpartition(worst, 20)[:20]
                else:
                    bi = np.arange(len(worst))
                Bc = C[bi].copy()
                # refine best 20 with softmax descent
                Bc = _refine_cands(Bc, stack, steps=110)
                # score refined
                Dr = Bc @ stack.T
                wr = Dr.max(axis=1)
                best_r = int(np.argmin(wr))
                vv = Bc[best_r].copy()
                vv /= (np.linalg.norm(vv) + 1e-300)
                X[int(cpos)] = vv
                stack = np.vstack([stack, vv[None, :]])
                # shifted p-norm Adam after each insertion (few hundred steps total)
                # use shorter for large stacks to save time
                X = _pnorm_adam(X, iu0, iu1, steps_per_stage=40, p_list=(32, 64, 128), shift=0.20, lr0=0.012, lr1=0.002)
                # update stack view for next insertion (re-sync inserted rows)
                # rebuild stack from X excluding not-yet-inserted
                # simple: update last row already; for Adam drift of all points, rebuild
                # reconstruct: keep all currently inserted + original keep
                # find which del_idx have been inserted so far
                # we track inserted set
                # rebuild stack as X rows in order: keep + inserted-so-far
                # (Adam may have moved keep points too, so refresh)
                # determine inserted so far
                # use position in order_ins
                done = set(order_ins[:order_ins.index(cpos) + 1])
                # stack = all rows except not-done deletions
                not_done = set(del_idx) - done
                mask2 = np.ones(n, dtype=bool)
                if len(not_done) > 0:
                    mask2[np.array(list(not_done), dtype=int)] = False
                stack = X[mask2].copy()
                if time.time() - t_trial_start > budget * 0.70:
                    break
            # finish each k-trial with top-K hard-max descent and short LP polish
            if time.time() > t0 + DEADLINE - 6:
                break
            # top-K K=150 hard-max descent
            X, vb = _hardK(X, iu0, iu1, iters=220, K=150, step0=0.0015)
            # short trust-region LP polish within remaining budget
            lp_sec = min(2.5, max(0.8, budget - (time.time() - t_trial_start) - 0.5))
            if lp_sec > 0.6 and (time.time() - t_trial_start) < budget - 1.0:
                try:
                    X, vb2 = _lp_polish_short(X, seconds=lp_sec, r_init=2e-4)
                    if vb2 < vb:
                        vb = vb2
                    else:
                        vb = _maxcos(X)
                except Exception:
                    vb = _maxcos(X)
            else:
                vb = _maxcos(X)
            if vb < inc_val - 1e-12:
                incumbent = _norm(X.copy())
                inc_val = float(vb)
                # refresh rattler ordering based on new incumbent for future trials?
                # keep original ordering per spec, but update rowmax for padding only
                # recompute to allow hub update for later hub-trials
                G1 = incumbent @ incumbent.T
                np.fill_diagonal(G1, -2.0)
                g1 = float(G1.max())
                rm1 = G1.max(axis=1)
                c91 = (G1 >= g1 - 1e-9).sum(axis=1)
                hub = int(np.argmax(c91))
                # do not reorder rattlers_sorted to stay faithful to initial diagnosis,
                # but if improved a lot, re-sort remaining? keep as is.
        except Exception:
            continue
        # strict per-trial budget enforcement: continue to next
        continue

    incumbent = _norm(incumbent)
    # final safety: if not better than original best, return original best
    # (incumbent is at least as good)
    return incumbent
