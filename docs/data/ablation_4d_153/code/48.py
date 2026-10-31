import time
import numpy as np

def _norm(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return x / n

def _max_cos(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def _smooth_opt(x, steps, tau0, tau1, lr0, lr1, mom=0.9):
    n, d = x.shape
    x = _norm(x)
    vel = np.zeros_like(x)
    lt0, lt1 = np.log(tau0), np.log(tau1)
    ll0, ll1 = np.log(lr0), np.log(lr1)
    for t in range(steps):
        f = t / max(1, steps - 1)
        tau = float(np.exp(lt0 + (lt1 - lt0) * f))
        lr = float(np.exp(ll0 + (ll1 - ll0) * f))
        G = x @ x.T
        np.fill_diagonal(G, -np.inf)
        m = float(np.max(G))
        W = np.exp((G - m) / tau)
        np.fill_diagonal(W, 0.0)
        row = np.sum(W, axis=1, keepdims=True)
        row[row < 1e-12] = 1e-12
        grad = (W @ x) / row
        dot = np.sum(grad * x, axis=1, keepdims=True)
        grad = grad - dot * x
        gn = np.linalg.norm(grad, axis=1, keepdims=True)
        big = gn > 2.0
        if np.any(big):
            grad = grad * np.where(big, 2.0 / np.maximum(gn, 1e-12), 1.0)
        vel = mom * vel + (1.0 - mom) * grad
        bc = 1.0 - mom ** (t + 1)
        step = vel / bc
        x = x - lr * step
        x = _norm(x)
    return x

def _lp_polish_deadline(x, deadline, cap=2500, r0=1e-3):
    from scipy.optimize import linprog
    from scipy.sparse import coo_matrix
    x = _norm(x)
    n, d = x.shape
    best = _max_cos(x)
    best_x = x.copy()
    radius = r0
    iu = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        G = x @ x.T
        c = G[iu]
        m = float(c.max())
        thresh = m - max(20.0 * radius, 1e-9)
        idx = np.where(c > thresh)[0]
        if idx.size == 0:
            break
        if idx.size > cap:
            order = np.argpartition(c[idx], -cap)[-cap:]
            idx = idx[order]
        I = iu[0][idx]; J = iu[1][idx]
        csel = c[idx]
        p = idx.size
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cI = I[:, None] * d + np.arange(d)
        cJ = J[:, None] * d + np.arange(d)
        cols = np.concatenate([cI, cJ, np.full((p, 1), n * d)], axis=1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], axis=1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y = _norm(y)
        v = _max_cos(y)
        if v < best:
            x = y; best = v; best_x = y.copy()
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return best_x, best

def _largest_hole_inserts(x0, n_insert, rng, n_samples=20000):
    x = x0.copy()
    d = x.shape[1]
    for _ in range(n_insert):
        S = rng.standard_normal((n_samples, d))
        S /= np.linalg.norm(S, axis=1, keepdims=True)
        # max dot to existing
        Md = S @ x.T
        mrow = Md.max(axis=1)
        bi = int(np.argmin(mrow))
        x = np.vstack([x, S[bi:bi+1]])
        x = _norm(x)
    return x

def _propose_deletions(X, rng):
    m = X.shape[0]
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    mval = float(G.max())
    # degrees with eps
    eps_deg = 0.002
    deg = np.count_nonzero(G > mval - eps_deg, axis=1)
    # sum of near-maximal cosines
    thr = mval - 0.005
    Gc = np.where(G > thr, G, 0.0)
    ssum = Gc.sum(axis=1)
    # hole coverage: nearest neighbour dot
    nn = G.max(axis=1)
    i_deg = int(np.argmax(deg + rng.random(m) * 1e-9))
    i_sum = int(np.argmax(ssum + rng.random(m) * 1e-12))
    i_hole = int(np.argmax(nn + rng.random(m) * 1e-12))
    i_rand = int(rng.integers(0, m))
    out = []
    for v in (i_deg, i_sum, i_hole, i_rand):
        if v not in out:
            out.append(v)
    return out

def _deg_hash(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    mval = float(G.max())
    deg = np.count_nonzero(G > mval - 1e-4, axis=1)
    deg = np.sort(deg)
    return tuple(int(v) for v in deg)

def solve():
    t0 = time.time()
    HARD_END = t0 + 172.0
    rng = np.random.default_rng(0)
    n, d = 153, 4
    # load incumbent
    best_x = None; best_v = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=float)
            if a.shape == (n, d):
                a = _norm(a)
                if np.all(np.isfinite(a)):
                    v = _max_cos(a)
                    if v < best_v:
                        best_v = v; best_x = a.copy()
        except Exception:
            pass
    if best_x is None:
        best_x = _norm(rng.standard_normal((n, d)))
        best_v = _max_cos(best_x)
    incumbent_x = best_x.copy()
    incumbent_v = float(best_v)
    orig_n = n

    # --- overshoot: insert 15 -> 168 (within 165-170 / 12-17) ---
    N_INSERT = 15
    N_BIG = n + N_INSERT
    try:
        Xbig = _largest_hole_inserts(incumbent_x, N_INSERT, rng, n_samples=16000)
    except Exception:
        Xbig = np.vstack([incumbent_x, _norm(rng.standard_normal((N_INSERT, d)))])
    # annealed softmax on oversized
    # budget-aware
    Xbig = _smooth_opt(Xbig, steps=700, tau0=0.012, tau1=0.0008, lr0=0.15, lr1=0.02)
    if time.time() >= HARD_END - 120:
        # not enough time, skip beam and just polish incumbent
        Xp, vp = _lp_polish_deadline(incumbent_x, HARD_END - 1.0)
        if vp < incumbent_v:
            return _norm(Xp)
        return _norm(incumbent_x)
    # short LP polish on oversized
    Xbig, vbig = _lp_polish_deadline(Xbig, min(HARD_END - 110, time.time() + 12.0), cap=2500, r0=8e-4)

    BEAM = 5
    # beam entries: dict X, is_ins (bool array), flag_deleted_original
    is_ins0 = np.array([False]*orig_n + [True]*N_INSERT)
    beam = [(Xbig, is_ins0, False, vbig)]

    n_del_needed = N_BIG - n  # 15
    # prune in 15 single-deletion stages
    for step in range(n_del_needed):
        if time.time() >= HARD_END - 65:
            break
        cands = []  # (v, X, is_ins, flag)
        for (Xp_, is_ins_, flag_, vp_) in beam:
            idxs = _propose_deletions(Xp_, rng)
            # ensure at least one original deletion is proposed: add best original candidate
            mcur = Xp_.shape[0]
            # find original indices (is_ins==False)
            orig_idx = np.where(~is_ins_)[0]
            if orig_idx.size > 0 and not flag_:
                # force one: highest degree among originals
                Gf = Xp_ @ Xp_.T
                np.fill_diagonal(Gf, -2.0)
                mv = float(Gf.max())
                degf = np.count_nonzero(Gf > mv - 0.002, axis=1)
                fo = int(orig_idx[np.argmax(degf[orig_idx])])
                if fo not in idxs:
                    idxs.append(fo)
            for di in idxs:
                Xc = np.delete(Xp_, di, axis=0)
                is_c = np.delete(is_ins_, di, axis=0)
                flag_c = bool(flag_ or (not bool(is_ins_[di])))
                vc = _max_cos(Xc)
                cands.append((vc, Xc, is_c, flag_c))
        # select diverse top BEAM
        cands.sort(key=lambda t: t[0])
        new_beam = []
        seen = set()
        for (vc, Xc, is_c, flag_c) in cands:
            try:
                h = _deg_hash(Xc)
            except Exception:
                h = None
            if h is not None and h in seen:
                continue
            if h is not None:
                seen.add(h)
            new_beam.append((Xc, is_c, flag_c, vc))
            if len(new_beam) >= BEAM:
                break
        if len(new_beam) == 0:
            # fallback take top BEAM
            new_beam = [(Xc, is_c, flag_c, vc) for (vc, Xc, is_c, flag_c) in cands[:BEAM]]
        # ensure at least one branch with flag True survives
        has_flag = any(f for (_, _, f, _) in new_beam)
        if not has_flag:
            # find best flagged in cands not already taken
            for (vc, Xc, is_c, flag_c) in cands:
                if flag_c:
                    # replace worst
                    new_beam[-1] = (Xc, is_c, flag_c, vc)
                    break
        beam = new_beam
        # re-optimise after every 3 deletions (and at end)
        do_reopt = ((step + 1) % 3 == 0) or (step == n_del_needed - 1)
        # if beam members not yet size 153, only softmax; if size 153 later full polish
        if do_reopt and time.time() < HARD_END - 60:
            reopted = []
            for (Xc, is_c, flag_c, vc) in beam:
                try:
                    Xr = _smooth_opt(Xc, steps=220, tau0=0.006, tau1=0.0009, lr0=0.06, lr1=0.012)
                except Exception:
                    Xr = Xc
                vr = _max_cos(Xr)
                reopted.append((Xr, is_c, flag_c, vr))
            # short LP every 6 steps if time allows
            if ((step + 1) % 6 == 0) and time.time() < HARD_END - 70:
                lp2 = []
                for (Xr, is_c, flag_c, vr) in reopted:
                    dl = min(HARD_END - 60, time.time() + 2.5)
                    try:
                        Xp2, vp2 = _lp_polish_deadline(Xr, dl, cap=2000, r0=5e-4)
                    except Exception:
                        Xp2, vp2 = Xr, vr
                    lp2.append((Xp2, is_c, flag_c, vp2))
                reopted = lp2
            # re-sort by value, keep diversity
            reopted.sort(key=lambda t: t[3])
            # dedup hash again
            filt = []; seen2 = set()
            for (Xr, is_c, flag_c, vr) in reopted:
                try:
                    h = _deg_hash(Xr)
                except Exception:
                    h = None
                if h is not None and h in seen2:
                    continue
                if h is not None:
                    seen2.add(h)
                filt.append((Xr, is_c, flag_c, vr))
            # refill if dedup removed too many (allow dup)
            if len(filt) < len(reopted):
                # add back best removed
                for item in reopted:
                    if item not in filt and len(filt) < BEAM:
                        filt.append(item)
                    if len(filt) >= BEAM:
                        break
            beam = filt[:BEAM]
            # re-enforce flagged branch
            has_flag = any(f for (_, _, f, _) in beam)
            if not has_flag:
                for (Xr, is_c, flag_c, vr) in reopted:
                    if flag_c:
                        beam[-1] = (Xr, is_c, flag_c, vr)
                        break

    # After reaching 153, fully LP-polish every beam survivor
    survivors = []
    for (Xc, is_c, flag_c, vc) in beam:
        if Xc.shape[0] != n:
            # fix size by greedy delete/add
            if Xc.shape[0] > n:
                # delete worst redundant until n
                while Xc.shape[0] > n:
                    G = Xc @ Xc.T
                    np.fill_diagonal(G, -2.0)
                    nn = G.max(axis=1)
                    di = int(np.argmax(nn))
                    Xc = np.delete(Xc, di, axis=0)
            else:
                need = n - Xc.shape[0]
                Xc = _largest_hole_inserts(Xc, need, rng, n_samples=8000)
        survivors.append(Xc)
    # polish each with remaining time shared
    time_left = HARD_END - time.time()
    # reserve 2s for final
    per = max(4.0, (time_left - 2.0) / max(1, len(survivors)))
    polished = []
    for Xc in survivors:
        if time.time() >= HARD_END - 2.0:
            polished.append((Xc, _max_cos(Xc)))
            continue
        dl = min(HARD_END - 1.0, time.time() + per)
        # brief softmax then LP
        try:
            Xs = _smooth_opt(Xc, steps=250, tau0=0.004, tau1=0.0006, lr0=0.05, lr1=0.01)
        except Exception:
            Xs = Xc
        try:
            Xp, vp = _lp_polish_deadline(Xs, dl, cap=2500, r0=1e-3)
        except Exception:
            Xp, vp = Xs, _max_cos(Xs)
        polished.append((Xp, vp))
    polished.sort(key=lambda t: t[1])
    best_found = polished[0] if len(polished) > 0 else (incumbent_x, incumbent_v)
    if best_found[1] < incumbent_v:
        return _norm(best_found[0])
    # also try polishing incumbent with leftover time
    if time.time() < HARD_END - 3.0:
        try:
            Xp, vp = _lp_polish_deadline(incumbent_x, HARD_END - 1.0, cap=2500, r0=1e-3)
            if vp < incumbent_v:
                return _norm(Xp)
        except Exception:
            pass
    return _norm(incumbent_x)
