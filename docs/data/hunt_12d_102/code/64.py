import time
import numpy as np
from scipy.optimize import linprog, nnls

def _max_cos(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _tangent_bases(x):
    n, d = x.shape
    dm = d - 1
    U = np.empty((n, d, dm))
    Eye = np.eye(d)
    for i in range(n):
        xi = x[i]
        # QR of matrix with xi as first column -> remaining columns span tangent
        M = np.empty((d, d))
        M[:, 0] = xi
        M[:, 1:] = Eye[:, 1:]
        # ensure full rank: if xi nearly parallel to e1, use different ordering
        # QR is enough generically; fix by replacing col with e0 if needed
        Q, _ = np.linalg.qr(M)
        if np.dot(Q[:, 0], xi) < 0:
            Q[:, 0] = -Q[:, 0]
        # re-orthogonalize rest against xi explicitly for accuracy
        # Q[:,1:] is already orthogonal to Q[:,0]~xi
        U[i] = Q[:, 1:]
    return U

def _lp_polish(x, deadline):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best_val = _max_cos(x)
    best_x = x.copy()
    radius = 1e-3
    iu_r, iu_c = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-13:
        G = x @ x.T
        c_full = G[iu_r, iu_c]
        m = c_full.max()
        thr = m - max(20.0 * radius, 1e-9)
        sel = c_full > thr
        p = int(sel.sum())
        if p == 0:
            radius *= 0.5
            continue
        if p > 2500:
            idx = np.argpartition(-c_full, 2500)[:2500]
            mask = np.zeros_like(sel)
            mask[idx] = True
            sel = sel & mask
            p = 2500
        I = iu_r[sel]; J = iu_c[sel]
        c_sel = c_full[sel]
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate(
            [I[:, None] * d + np.arange(d),
             J[:, None] * d + np.arange(d),
             np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            from scipy.sparse import coo_matrix
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix(
                (x.ravel(),
                 (np.repeat(np.arange(n), d), np.arange(n * d))),
                shape=(n, nv)).tocsr()
            obj = np.zeros(nv); obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-c_sel,
                          A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                          method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        yn = np.linalg.norm(y, axis=1, keepdims=True)
        if np.any(~np.isfinite(yn)) or np.any(yn < 1e-12):
            radius *= 0.5
            continue
        y /= yn
        v = _max_cos(y)
        if v < best_val:
            best_val = v
            best_x = y.copy()
            x = y
            radius = min(radius * 1.5, 0.02)
            if time.time() >= deadline:
                break
        else:
            radius *= 0.5
    return best_x

def _lse_quench(x, temps=(0.01, 0.005, 0.002, 0.001, 0.0005), steps_per=(120, 120, 120, 120, 140)):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    for ti, T in enumerate(temps):
        st = steps_per[ti] if ti < len(steps_per) else 120
        lr = 0.012 if T >= 0.004 else 0.006
        if T < 0.001:
            lr = 0.004
        for _ in range(st):
            G = x @ x.T
            np.fill_diagonal(G, -np.inf)
            m = G.max()
            # stable exp
            E = np.exp((G - m) / T)
            np.fill_diagonal(E, 0.0)
            S = E.sum()
            if not np.isfinite(S) or S == 0:
                break
            Grad = (E @ x) / S
            dot = np.sum(Grad * x, axis=1, keepdims=True)
            Gt = Grad - dot * x
            # clip
            nm = np.linalg.norm(Gt, axis=1, keepdims=True)
            # normalize-ish: limit step
            step = lr * Gt
            # per-row cap 0.03
            sn = np.linalg.norm(step, axis=1, keepdims=True) + 1e-16
            sc = np.minimum(1.0, 0.03 / sn)
            step = step * sc
            x = x - step
            x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x

def solve():
    import time as _time
    import numpy as _np
    t0 = _time.time()
    HARD_END = t0 + 172.0
    EXPLORE_END = t0 + 148.0
    rng = _np.random.default_rng(7)
    try:
        base = _np.load("best.npy").astype(float)
    except Exception:
        base = _np.load("record.npy").astype(float)
    n, d = 102, 12
    dm = d - 1
    if base.shape != (n, d):
        tmp = rng.standard_normal((n, d))
        try:
            rec = _np.load("record.npy").astype(float)
            m0 = min(rec.shape[0], n)
            tmp[:m0, :min(rec.shape[1], d)] = rec[:m0, :min(rec.shape[1], d)]
        except Exception:
            pass
        base = tmp
    base = base / _np.linalg.norm(base, axis=1, keepdims=True)
    best_val = _max_cos(base)
    best = base.copy()
    cur = base.copy()
    cur_val = best_val

    iu_r, iu_c = _np.triu_indices(n, 1)

    def contact_set(x, thr=1e-7):
        G = x @ x.T
        c = G[iu_r, iu_c]
        m = c.max()
        sel = c > m - thr
        return set(zip(iu_r[sel].tolist(), iu_c[sel].tolist())), m, c

    S0, m0v, _ = contact_set(best, 1e-7)
    if len(S0) == 0:
        S0, m0v, _ = contact_set(best, 1e-4)

    # helpers for null/stress on current cur
    nslide = 0
    # step sizes to cycle
    step_cycle = [0.03, 0.06, 0.09, 0.12, 0.15, 0.05, 0.10, 0.14]

    while _time.time() < EXPLORE_END:
        nslide += 1
        if _time.time() > HARD_END - 20:
            break
        # --- build contacts of cur ---
        Gc = cur @ cur.T
        cfull = Gc[iu_r, iu_c]
        mc = float(cfull.max())
        # Jacobian contacts within 1e-7
        thrJ = 1e-7
        selJ = cfull > mc - thrJ
        if int(selJ.sum()) < 60:
            selJ = cfull > mc - 1e-4
        if int(selJ.sum()) > 900:
            # keep top 900
            k = 900
            idx = _np.argpartition(-cfull, k)[:k]
            m2 = _np.zeros_like(selJ)
            m2[idx] = True
            selJ = selJ & m2
        Ij = iu_r[selJ]; Jj = iu_c[selJ]
        pJ = len(Ij)
        if pJ < 10:
            # random kick + quench to escape degeneracy
            cur = cur + 0.05 * rng.standard_normal((n, d))
            cur /= _np.linalg.norm(cur, axis=1, keepdims=True)
            cur = _lse_quench(cur)
            cur_val = _max_cos(cur)
            continue
        # tangent bases
        try:
            U = _tangent_bases(cur)
        except Exception:
            U = None
        if U is None:
            cur = best + 0.02 * rng.standard_normal((n, d))
            cur /= _np.linalg.norm(cur, axis=1, keepdims=True)
            cur = _lse_quench(cur)
            cur_val = _max_cos(cur)
            continue
        N = n * dm
        # Build A (pJ x N)
        A = _np.zeros((pJ, N), dtype=float)
        # vectorized per pair loop (pJ<=900, fine)
        for k in range(pJ):
            i = int(Ij[k]); j = int(Jj[k])
            A[k, i*dm:(i+1)*dm] = U[i].T @ cur[j]
            A[k, j*dm:(j+1)*dm] = U[j].T @ cur[i]
        # --- NNLS stress per point ---
        # map pair index to weight
        pair_w = _np.zeros(pJ)
        # build per-point adjacency
        # for speed, build list
        try:
            lam = 20.0
            sl = _np.sqrt(lam)
            # need per point deg
            for i in range(n):
                # find ks involving i
                ks = _np.where((Ij == i) | (Jj == i))[0]
                deg = len(ks)
                if deg == 0:
                    continue
                # build C (dm x deg)
                C = _np.empty((dm, deg))
                for di, k in enumerate(ks):
                    a = int(Ij[k]); b = int(Jj[k])
                    oth = b if a == i else a
                    cij = float(cur[i] @ cur[oth])
                    t = cur[oth] - cij * cur[i]
                    C[:, di] = U[i].T @ t
                M = _np.vstack([C, sl*_np.ones((1, deg))])
                bb = _np.zeros(dm+1); bb[-1] = sl
                w, _ = nnls(M, bb)
                # accumulate
                for di, k in enumerate(ks):
                    pair_w[k] += w[di]
        except Exception:
            pair_w = _np.ones(pJ)
        # order by stress
        order = _np.argsort(-pair_w)
        # choose 20-80 targets: take top 80, subsample
        ntarget = int(rng.integers(20, 81))
        ntop = min(80, pJ)
        # pick from top-120 with bias to top
        cand_pool = order[:min(120, pJ)]
        # ensure ntarget <= pool
        ntarget = min(ntarget, len(cand_pool))
        # take highest ntarget with some randomness: shuffle top then pick?
        # deterministic-ish: take top ntarget but randomly replace 30%
        tgt_idx = cand_pool[:ntarget].copy()
        if len(cand_pool) > ntarget and nslide % 2 == 0:
            # replace ~1/3 with random from rest of pool
            nr = max(1, ntarget // 3)
            repl = rng.choice(cand_pool[ntarget:], size=min(nr, len(cand_pool)-ntarget), replace=False)
            tgt_idx[-len(repl):] = repl
            rng.shuffle(tgt_idx)
        # near-null subspace: smallest K eigenvectors of A^T A
        K = 28
        K = min(K, N-1, pJ)
        try:
            Gram = A.T @ A  # N x N
            # eigh
            ew, ev = _np.linalg.eigh(Gram)
            # smallest K
            Vk = ev[:, :K]  # N x K
        except Exception:
            # fallback random subspace
            Vk = rng.standard_normal((N, K))
            # orthonormalize
            Vk, _ = _np.linalg.qr(Vk)
        # B matrices
        A_tgt = A[tgt_idx]  # nt x N
        Bt = A_tgt @ Vk  # nt x K
        # monitor set: pairs within 0.005 of max (cap 2000)
        selM = cfull > mc - 0.005
        if int(selM.sum()) > 2000:
            idxm = _np.argpartition(-cfull, 2000)[:2000]
            msk = _np.zeros_like(selM); msk[idxm] = True
            selM = selM & msk
        Im = iu_r[selM]; Jm = iu_c[selM]
        pM = len(Im)
        # build Am for monitor (reuse A rows where overlap? build fresh)
        # To save time, build Am only if pM moderate
        AmVk_max = None
        if pM > 0 and pM <= 2500:
            Am = _np.zeros((pM, N))
            for k in range(pM):
                i = int(Im[k]); j = int(Jm[k])
                Am[k, i*dm:(i+1)*dm] = U[i].T @ cur[j]
                Am[k, j*dm:(j+1)*dm] = U[j].T @ cur[i]
            Bm = Am @ Vk  # pM x K
        else:
            Bm = None
        # sample random combos
        T = 1500
        Cr = rng.standard_normal((K, T))
        Cr /= _np.linalg.norm(Cr, axis=0, keepdims=True) + 1e-16
        Pt = Bt @ Cr  # nt x T
        # score: max over targets (want very negative) + 0.5*max over monitors
        max_t = Pt.max(axis=0)  # T
        if Bm is not None:
            Pm = Bm @ Cr
            max_m = Pm.max(axis=0)
        else:
            max_m = _np.zeros(T)
        score = max_t + 0.5 * _np.maximum(0, max_m)
        # also reward deactivation count: count targets with pred < -2e-4
        # adjust slightly
        cnt = (Pt < -2e-4).sum(axis=0)
        # prefer larger count: subtract small bonus
        score = score - 1e-4 * cnt
        bi = int(_np.argmin(score))
        cstar = Cr[:, bi]  # K
        z_unit = Vk @ cstar  # N
        # scale to angular size
        s0 = step_cycle[(nslide-1) % len(step_cycle)]
        # add jitter
        s0 = float(s0 * (0.8 + 0.4 * rng.random()))
        s0 = min(0.16, max(0.025, s0))
        z = z_unit * (s0 * _np.sqrt(n))  # total norm
        # cap per-point max to 0.35 rad
        Z = z.reshape(n, dm)
        # dx full: U[i] @ Z[i]
        DX = _np.einsum('nij,nj->ni', U, Z)
        pn = _np.linalg.norm(DX, axis=1)
        mx = pn.max()
        if mx > 0.35:
            DX *= (0.35 / mx)
            pn = _np.linalg.norm(DX, axis=1)
        # exponential map slide
        x_slide = cur.copy()
        for i in range(n):
            ni = float(pn[i])
            if ni < 1e-12:
                continue
            x_slide[i] = _np.cos(ni) * cur[i] + _np.sin(ni) * (DX[i] / ni)
        x_slide /= _np.linalg.norm(x_slide, axis=1, keepdims=True)
        v_slide = _max_cos(x_slide)
        # monitor newly approaching: if slide blew up too much, still quench? skip if >0.30
        if v_slide > 0.32:
            # too violent, shrink and retry next iter
            cur = cur  # stay
            continue
        # --- log-sum-exp continuation T 0.01 -> 0.0005 ---
        x_q = _lse_quench(x_slide)
        v_q = _max_cos(x_q)
        # contact graph distance from incumbent S0
        Sq, _, _ = contact_set(x_q, 1e-7)
        if len(Sq) == 0:
            Sq, _, _ = contact_set(x_q, 1e-4)
        inter = len(Sq & S0)
        union = len(Sq | S0)
        dist = 1.0 - (inter / union if union > 0 else 0.0)
        changed_frac = dist  # Jaccard distance
        # gating for LP
        do_lp = (v_q < 0.2685) or (changed_frac >= 0.10)
        v_end = v_q
        x_end = x_q
        if do_lp:
            if _time.time() > HARD_END - 18:
                do_lp = False
            else:
                dl = min(_time.time() + 2.0, EXPLORE_END + 1.0, HARD_END - 12.0)
                if dl - _time.time() > 0.4:
                    x_p = _lp_polish(x_q, dl)
                    v_p = _max_cos(x_p)
                    # recompute distance after LP
                    Sp, _, _ = contact_set(x_p, 1e-7)
                    if len(Sp) == 0:
                        Sp = Sq
                        interp = inter; unionp = union
                    else:
                        interp = len(Sp & S0)
                        unionp = len(Sp | S0)
                    distp = 1.0 - (interp / unionp if unionp > 0 else 0.0)
                    # accept LP result as endpoint
                    x_end = x_p; v_end = v_p; dist = distp
        # acceptance: objective + graph distance
        improved = v_end < best_val - 1e-12
        if improved:
            best = x_end.copy(); best_val = v_end
            cur = x_end.copy(); cur_val = v_end
            # update S0? keep original incumbent for distance? spec says from incumbent.
            # keep S0 fixed to original record basin to enforce diversity? Update slowly:
            # if big improvement, reset S0 to new best's graph
            if changed_frac > 0.25:
                S0 = Sq if do_lp is False else Sp
        else:
            # wander: accept diverse endpoints even if slightly worse
            # threshold relaxes if dist large
            tol = 0.0015 + 0.003 * dist
            if v_end < cur_val + tol and dist > 0.12:
                cur = x_end.copy(); cur_val = v_end
            elif v_end < best_val + 0.004 and dist > 0.30:
                cur = x_end.copy(); cur_val = v_end
            elif nslide % 9 == 0 and v_end < best_val + 0.002:
                # occasional random restart from best with shake to avoid stagnation
                if rng.random() < 0.3:
                    cur = best.copy(); cur_val = best_val
        # early exit if time
        if _time.time() >= EXPLORE_END:
            break
    # final long polish of best
    remain = HARD_END - _time.time()
    if remain > 4.0:
        best = _lp_polish(best, _time.time() + remain - 1.5)
        best_val = _max_cos(best)
    best = best / _np.linalg.norm(best, axis=1, keepdims=True)
    return best
