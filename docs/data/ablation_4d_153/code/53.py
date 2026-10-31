import time
import numpy as np

def _norm(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return x / n

def _max_cos(x):
    try:
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        return float(np.max(G))
    except Exception:
        return 1.0

def _smooth_opt(x, steps, tau0, tau1, lr0, lr1, mom=0.88):
    n, d = x.shape
    vel = np.zeros_like(x)
    try:
        lt0, lt1 = np.log(tau0), np.log(tau1)
        ll0, ll1 = np.log(lr0), np.log(lr1)
    except Exception:
        return x
    for t in range(steps):
        try:
            f = t / max(1, steps - 1)
            tau = float(np.exp(lt0 + (lt1 - lt0) * f))
            lr = float(np.exp(ll0 + (ll1 - ll0) * f))
            G = x @ x.T
            np.fill_diagonal(G, -np.inf)
            rmax = np.max(G, axis=1, keepdims=True)
            W = np.exp((G - rmax) / max(tau, 1e-12))
            np.fill_diagonal(W, 0.0)
            row = np.sum(W, axis=1, keepdims=True)
            np.maximum(row, 1e-12, out=row)
            grad = (W @ x) / row
            dot = np.sum(grad * x, axis=1, keepdims=True)
            grad -= dot * x
            gn = np.linalg.norm(grad, axis=1, keepdims=True)
            big = gn > 2.0
            if np.any(big):
                grad = grad * np.where(big, 2.0 / np.maximum(gn, 1e-12), 1.0)
            vel = mom * vel + (1.0 - mom) * grad
            bc = 1.0 - mom ** (t + 1)
            step = vel / max(bc, 1e-12)
            x = x - lr * step
            nr = np.linalg.norm(x, axis=1, keepdims=True)
            nr[nr == 0] = 1.0
            x = x / nr
        except Exception:
            break
    return x

def _find_hole(X, rng, S=4000):
    try:
        R = rng.standard_normal(size=(S, 4))
        R /= np.linalg.norm(R, axis=1, keepdims=True)
        D = R @ X.T
        s = np.max(D, axis=1)
        return R[int(np.argmin(s))]
    except Exception:
        v = rng.standard_normal(size=(4,))
        n = np.linalg.norm(v)
        if n == 0:
            v = np.array([1.0, 0, 0, 0])
        else:
            v = v / n
        return v

def _lp_polish(x, deadline, cap=2500, r0=1e-3):
    try:
        from scipy.optimize import linprog
        from scipy.sparse import coo_matrix
    except Exception:
        return x, _max_cos(x)
    try:
        x = _norm(np.asarray(x, dtype=float))
    except Exception:
        return x, 1.0
    n, d = x.shape
    try:
        best = _max_cos(x)
    except Exception:
        best = 1.0
    radius = r0
    iu = np.triu_indices(n, 1)
    it = 0
    while True:
        it += 1
        if it > 300:
            break
        try:
            if time.time() >= deadline:
                break
            if radius < 1e-14:
                break
            G = x @ x.T
            c = G[iu]
            m = float(np.max(c))
            thresh = m - max(20.0 * radius, 1e-9)
            idx = np.where(c > thresh)[0]
            if idx.size == 0:
                break
            if idx.size > cap:
                try:
                    order = np.argpartition(c[idx], -cap)[-cap:]
                    idx = idx[order]
                except Exception:
                    idx = idx[:cap]
            I = iu[0][idx]
            J = iu[1][idx]
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
            obj = np.zeros(nv)
            obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
            if res.status != 0:
                radius *= 0.5
                continue
            y = x + res.x[:-1].reshape(n, d)
            y = _norm(y)
            if not np.all(np.isfinite(y)):
                radius *= 0.5
                continue
            v = _max_cos(y)
            if v < best:
                x = y
                best = v
                radius = min(radius * 1.5, 0.05)
            else:
                radius *= 0.5
        except Exception:
            try:
                radius *= 0.5
            except Exception:
                break
            continue
    return x, best

def _loads_and_max(X, tau=1e-4):
    N = X.shape[0]
    try:
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        m = float(np.max(G))
        # per-point near-contact load
        diff = m - G
        # only near pairs contribute; cap exponent
        with np.errstate(over='ignore', under='ignore'):
            E = np.exp(-diff / tau)
        np.fill_diagonal(E, 0.0)
        # zero out far pairs for stability (exp underflows anyway)
        # keep all, sum is fine
        L = np.sum(E, axis=1)
        # total near load within 1e-4 style: sum over upper triangle
        iu = np.triu_indices(N, 1)
        diffu = diff[iu]
        with np.errstate(over='ignore', under='ignore'):
            Eu = np.exp(-diffu / tau)
        # restrict to within ~10*tau for secondary score stability
        # far terms are ~0 anyway
        s = float(np.sum(Eu))
        return m, L, s, G
    except Exception:
        return 1.0, np.zeros(N), 1e9, None

def _degree_hash_from_G(G, m, tol=5e-7):
    try:
        B = G > (m - tol)
        # diagonal is -2 so not counted
        deg = np.sum(B, axis=1)
        return tuple(sorted(np.asarray(deg, dtype=int).tolist()))
    except Exception:
        return tuple()

def _procrustes_perm_dist(A, B, max_iter=2):
    # returns approx distance after O(4) alignment + permutation; large if fail
    try:
        from scipy.optimize import linear_sum_assignment
        N = A.shape[0]
        R = np.eye(4)
        assign = np.arange(N)
        for _ in range(max_iter):
            Brot = B @ R
            # cost = -dot (maximise total dot)
            C = -(A @ Brot.T)
            # linear_sum_assignment minimises
            r_ind, c_ind = linear_sum_assignment(C)
            # r_ind should be 0..N-1; build ordered B
            Bord = np.zeros_like(B)
            # r_ind[i] corresponds to A row r_ind[i] matched to B col c_ind[i]
            # reorder B to A order
            inv = np.zeros(N, dtype=int)
            # c_ind is B index for each A in order of r_ind
            # need Bord[r] = B[c]
            for rr, cc in zip(r_ind, c_ind):
                Bord[rr] = B[cc]
            M = Bord.T @ A
            U, _, Vt = np.linalg.svd(M)
            R = U @ Vt
        # final error
        Brot = B @ R
        C = -(A @ Brot.T)
        r_ind, c_ind = linear_sum_assignment(C)
        dots = np.sum(A[r_ind] * Brot[c_ind], axis=1)
        dots = np.clip(dots, -1.0, 1.0)
        # mean angular error
        err = float(np.mean(1.0 - dots))
        return err
    except Exception:
        return 1e9

def solve():
    t0 = time.time()
    HARD_END = t0 + 175.0
    BRANCH_STOP = t0 + 160.0
    rng = np.random.default_rng(0)
    n, d = 153, 4
    best_x = None
    best_v = 1e9
    for fname in ("best.npy", "record.npy"):
        try:
            a = np.load(fname)
            a = np.asarray(a, dtype=float)
            if a.shape == (n, d) and np.all(np.isfinite(a)):
                nr = np.linalg.norm(a, axis=1, keepdims=True)
                if np.any(nr == 0):
                    continue
                a = a / nr
                v = _max_cos(a)
                if v < best_v:
                    best_v = v
                    best_x = a.copy()
        except Exception:
            continue
    if best_x is None:
        try:
            x = rng.standard_normal(size=(n, d))
            x = _norm(x)
            best_x = x
            best_v = _max_cos(x)
        except Exception:
            best_x = np.tile(np.eye(4)[:1], (n, 1)).astype(float)
            best_x = _norm(best_x + 1e-6 * np.ones_like(best_x))
            best_v = _max_cos(best_x)
    incumbent_x = best_x.copy()
    incumbent_v = float(best_v)

    all_leaves = []
    overs_list = [6, 9, 12]
    tau_load = 1e-4

    for K in overs_list:
        try:
            if time.time() >= BRANCH_STOP:
                break
            # build oversized code: incumbent + K holes
            try:
                cur = incumbent_x.copy()
                for _ in range(K):
                    h = _find_hole(cur, rng, S=4000)
                    # tiny jitter
                    try:
                        h = h + 1e-3 * rng.standard_normal(size=(4,))
                        h = h / np.linalg.norm(h)
                    except Exception:
                        pass
                    cur = np.vstack([cur, h.reshape(1, 4)])
                    cur = _norm(cur)
            except Exception:
                continue
            Nbig = cur.shape[0]
            # briefly optimize oversized with smooth minimax
            try:
                cur = _smooth_opt(cur, 350, 0.025, 0.0025, 0.07, 0.004, mom=0.88)
            except Exception:
                pass
            # one short LP polish
            try:
                if time.time() < BRANCH_STOP - 0.3:
                    dl = min(time.time() + 1.5, BRANCH_STOP)
                    if dl > time.time() + 0.15:
                        cur, _ = _lp_polish(cur, dl, cap=2200, r0=8e-4)
            except Exception:
                pass
            # beam prune down to 153
            beam = [cur]
            curN = Nbig
            while curN > n:
                try:
                    if time.time() >= BRANCH_STOP:
                        # greedy trim to 153 in one shot
                        new_beam = []
                        for Xb in beam:
                            try:
                                Nc = Xb.shape[0]
                                need = Nc - n
                                if need <= 0:
                                    new_beam.append(Xb)
                                    continue
                                m0, L0, _, _ = _loads_and_max(Xb, tau=tau_load)
                                drop = np.argsort(-L0)[:need]
                                mask = np.ones(Nc, dtype=bool)
                                mask[drop] = False
                                new_beam.append(Xb[mask])
                            except Exception:
                                continue
                        beam = new_beam[:3] if len(new_beam) > 0 else beam
                        break
                    cands = []
                    for bi, Xb in enumerate(beam):
                        try:
                            Nc = Xb.shape[0]
                            m0, L0, s0, G0 = _loads_and_max(Xb, tau=tau_load)
                            # top deletion choices by load
                            D = min(5, Nc)
                            order = np.argsort(-L0)[:D]
                            # add one random diverse choice if room and time
                            for k in order:
                                try:
                                    mask = np.ones(Nc, dtype=bool)
                                    mask[int(k)] = False
                                    Xd = Xb[mask]
                                    m2, L2, s2, G2 = _loads_and_max(Xd, tau=tau_load)
                                    dh = _degree_hash_from_G(G2, m2) if G2 is not None else tuple()
                                    cands.append((float(m2), float(s2), dh, Xd))
                                except Exception:
                                    continue
                        except Exception:
                            continue
                    if len(cands) == 0:
                        break
                    # sort by (max cosine, sum load)
                    try:
                        cands.sort(key=lambda t: (t[0], t[1]))
                    except Exception:
                        pass
                    # keep beam of 3 with diversity of degree hash
                    next_beam = []
                    seen_hash = set()
                    for m2, s2, dh, Xd in cands:
                        try:
                            if dh in seen_hash:
                                # allow duplicate only if we still need to fill and few options
                                continue
                            seen_hash.add(dh)
                            next_beam.append(Xd)
                            if len(next_beam) >= 3:
                                break
                        except Exception:
                            continue
                    # if too few due to hash collisions, fill from sorted list
                    if len(next_beam) < min(3, len(cands)):
                        for m2, s2, dh, Xd in cands:
                            if len(next_beam) >= 3:
                                break
                            try:
                                # check not already included (by identity)
                                dup = False
                                for y in next_beam:
                                    if y.shape == Xd.shape and np.may_share_memory(y, Xd):
                                        dup = True
                                        break
                                if not dup:
                                    # avoid exact duplicate hash overload? allow now
                                    already = any(np.array_equal(y, Xd) for y in next_beam)
                                    if not already:
                                        next_beam.append(Xd)
                            except Exception:
                                continue
                    if len(next_beam) == 0:
                        break
                    beam = next_beam
                    curN -= 1
                    # cap pair lists: beam size <=3 enforced
                    if len(beam) > 3:
                        beam = beam[:3]
                except Exception:
                    break
            # collect leaves that are exactly 153
            for Xb in beam:
                try:
                    if Xb.shape[0] == n and Xb.shape[1] == d and np.all(np.isfinite(Xb)):
                        nr = np.linalg.norm(Xb, axis=1, keepdims=True)
                        if np.any(nr == 0):
                            continue
                        Xb = Xb / nr
                        all_leaves.append(Xb)
                    elif Xb.shape[0] > n:
                        # greedy cut
                        Nc = Xb.shape[0]
                        m0, L0, _, _ = _loads_and_max(Xb, tau=tau_load)
                        need = Nc - n
                        drop = np.argsort(-L0)[:need]
                        mask = np.ones(Nc, dtype=bool)
                        mask[drop] = False
                        Xc = Xb[mask]
                        Xc = _norm(Xc)
                        all_leaves.append(Xc)
                except Exception:
                    continue
        except Exception:
            continue

    # deduplicate by sorted degree sequence before polish
    uniq = []
    seen_deg = set()
    # also track (deghash, rounded max) to keep distinct maxima
    for Xb in all_leaves:
        try:
            m = _max_cos(Xb)
            G = Xb @ Xb.T
            np.fill_diagonal(G, -2.0)
            dh = _degree_hash_from_G(G, m)
            key = (dh, round(float(m), 7))
            if key in seen_deg:
                continue
            seen_deg.add(key)
            uniq.append(Xb)
        except Exception:
            continue
    # further dedup by Procrustes+assignment distance for very close pairs
    try:
        kept = []
        for Xb in uniq:
            try:
                is_dup = False
                mb = _max_cos(Xb)
                for Y in kept:
                    try:
                        my = _max_cos(Y)
                        if abs(mb - my) > 2e-7:
                            continue
                        # quick descriptor: sorted row-max
                        Gb = Xb @ Xb.T
                        np.fill_diagonal(Gb, -2.0)
                        rb = np.sort(np.max(Gb, axis=1))
                        Gy = Y @ Y.T
                        np.fill_diagonal(Gy, -2.0)
                        ry = np.sort(np.max(Gy, axis=1))
                        if float(np.linalg.norm(rb - ry)) > 1e-6:
                            continue
                        dist = _procrustes_perm_dist(Xb, Y, max_iter=2)
                        if dist < 1e-6:
                            is_dup = True
                            break
                    except Exception:
                        continue
                if not is_dup:
                    kept.append(Xb)
                if time.time() >= BRANCH_STOP:
                    # stop expensive dedup
                    # append remaining without check
                    pass
            except Exception:
                continue
        # if we broke early, kept may be partial; ensure all uniq considered
        if len(kept) > 0:
            uniq = kept
    except Exception:
        pass

    # LP-polish each beam leaf, keep best
    final_best_x = incumbent_x.copy()
    final_best_v = float(incumbent_v)
    try:
        # limit number polished to fit time
        # sort uniq by current max so best first
        scored = []
        for Xb in uniq:
            try:
                scored.append((_max_cos(Xb), Xb))
            except Exception:
                continue
        try:
            scored.sort(key=lambda t: t[0])
        except Exception:
            pass
        # cap to at most 9 leaves, take best 6 if many
        if len(scored) > 6:
            scored = scored[:6]
        remaining = HARD_END - 1.0 - time.time()
        if remaining < 0:
            remaining = 0
        per = 0
        if len(scored) > 0 and remaining > 0.5:
            per = min(8.0, remaining / len(scored))
        for si, (v0, Xb) in enumerate(scored):
            try:
                if time.time() >= HARD_END - 1.0:
                    break
                dl = min(time.time() + per if per > 0 else time.time() + 2.0, HARD_END - 1.0)
                if dl <= time.time() + 0.05:
                    # no time, just evaluate
                    v = _max_cos(Xb)
                    if v < final_best_v:
                        final_best_v = v
                        final_best_x = Xb.copy()
                    continue
                Xp, vp = _lp_polish(Xb, dl, cap=2500, r0=1e-3)
                # also try tiny smooth if LP didn't improve? keep cheap
                vchk = _max_cos(Xp)
                if vchk < final_best_v:
                    final_best_v = float(vchk)
                    final_best_x = Xp.copy()
                else:
                    # still consider unpolished if better than final? already sorted
                    if v0 < final_best_v:
                        final_best_v = float(v0)
                        final_best_x = Xb.copy()
            except Exception:
                continue
    except Exception:
        pass

    # final safety: ensure valid
    try:
        final_best_x = np.asarray(final_best_x, dtype=float)
        if final_best_x.shape != (n, d) or not np.all(np.isfinite(final_best_x)):
            final_best_x = incumbent_x.copy()
            final_best_v = float(incumbent_v)
        else:
            nr = np.linalg.norm(final_best_x, axis=1, keepdims=True)
            nr[nr == 0] = 1.0
            final_best_x = final_best_x / nr
            fv = _max_cos(final_best_x)
            # if somehow worse than incumbent due to numerical, return incumbent
            if fv > incumbent_v and np.all(np.isfinite(incumbent_x)):
                final_best_x = incumbent_x.copy()
    except Exception:
        try:
            final_best_x = incumbent_x.copy()
        except Exception:
            rng2 = np.random.default_rng(1)
            final_best_x = _norm(rng2.standard_normal(size=(n, d)))
    return final_best_x
