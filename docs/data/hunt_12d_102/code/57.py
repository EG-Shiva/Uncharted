import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _normed(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return x / n

def _lp_polish(x, seconds, t_end):
    x = _normed(x)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and time.time() < t_end:
        if radius < 1e-14:
            break
        c = (x @ x.T)[iu]
        mcur = c.max()
        sel = c > mcur - max(20 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p, nv = len(I), n * d + 1
        if p == 0:
            break
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c[sel], A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y = _normed(y)
        vv = _max_cos(y)
        if vv < best:
            x, best, radius = y, vv, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x, best

def _tangent_bases(x):
    n, d = x.shape
    nb = d - 1
    B = np.empty((n, d, nb))
    for i in range(n):
        xi = x[i]
        k = int(np.argmax(np.abs(xi)))
        idx = [j for j in range(d) if j != k]
        M = np.zeros((d, nb))
        for c, j in enumerate(idx):
            M[j, c] = 1.0
            M[:, c] -= xi[j] * xi
        Q, _ = np.linalg.qr(M)
        B[i] = Q
    return B

def _build_Jt(x, B, I, J):
    n, d = x.shape
    nb = d - 1
    N = n * nb
    p = len(I)
    Jt = np.empty((p, N), dtype=float)
    for r in range(p):
        i = int(I[r]); j = int(J[r])
        row = np.zeros(N)
        row[i*nb:(i+1)*nb] = B[i].T @ x[j]
        row[j*nb:(j+1)*nb] = B[j].T @ x[i]
        Jt[r] = row
    return Jt

def _stress_weights(x, I, J, iters=200):
    p = len(I)
    n, d = x.shape
    if p == 0:
        return np.ones(0)
    try:
        A = np.zeros((n*d, p), dtype=float)
        for k in range(p):
            i = int(I[k]); j = int(J[k])
            c = float(x[i] @ x[j])
            ti = x[j] - c * x[i]
            tj = x[i] - c * x[j]
            A[i*d:(i+1)*d, k] = ti
            A[j*d:(j+1)*d, k] = tj
        kk = 5.0
        w = np.ones(p, dtype=float) / p
        # estimate step
        alpha = 1e-2
        for _ in range(iters):
            Aw = A @ w
            s = w.sum()
            grad = A.T @ Aw + (kk**2) * (s - 1.0)
            # simple adaptive step: normalize by max abs
            mx = np.abs(grad).max()
            st = alpha / (mx + 1e-12)
            # cap step so w doesn't jump too far
            st = min(st, 0.5)
            w = w - st * grad
            w[w < 0] = 0
            sw = w.sum()
            if sw <= 0 or not np.isfinite(sw):
                w = np.ones(p)/p
                break
            # renormalize slightly toward sum 1 is already in grad; keep as is
        if not np.all(np.isfinite(w)) or w.sum() <= 0:
            raise ValueError("bad w")
        return w
    except Exception:
        return np.ones(p, dtype=float)

def _lse_adam(x_init, steps, T0, T1, lr, rng_scale=1.0):
    x = x_init.copy()
    n, d = x.shape
    m = np.zeros_like(x)
    v = np.zeros_like(x)
    b1, b2, eps = 0.9, 0.999, 1e-8
    lT0 = np.log(T0); lT1 = np.log(T1)
    for s in range(1, steps+1):
        lt = lT0 + (lT1 - lT0) * (s / steps)
        T = float(np.exp(lt))
        G = x @ x.T
        np.fill_diagonal(G, -10.0)
        M = G.max()
        W = np.exp((G - M) / T)
        np.fill_diagonal(W, 0.0)
        # normalize to avoid huge grad: divide by sum
        S = W.sum()
        if S == 0 or not np.isfinite(S):
            break
        # gradient of log-sum-exp: (W@x)/(S*T)
        Grad = (W @ x) / (S * T + 1e-30)
        dot = np.sum(Grad * x, axis=1, keepdims=True)
        Gt = Grad - dot * x
        # clip
        m = b1 * m + (1.0 - b1) * Gt
        v = b2 * v + (1.0 - b2) * (Gt * Gt)
        c1 = 1.0 - b1 ** s
        c2 = 1.0 - b2 ** s
        upd = (m / c1) / (np.sqrt(v / c2) + eps)
        x = x - lr * upd
        x = _normed(x)
    return x

def _contact_set(x, tol):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    m = G.max()
    iu = np.triu_indices(x.shape[0], 1)
    c = G[iu]
    sel = c > m - tol
    s = set(zip(iu[0][sel].tolist(), iu[1][sel].tolist()))
    return s, m

def solve():
    t0 = time.time()
    T_END = t0 + 172.0
    rng = np.random.default_rng(7)
    try:
        xb = np.array(np.load("best.npy"), dtype=float)
    except Exception:
        try:
            xb = np.array(np.load("record.npy"), dtype=float)
        except Exception:
            xb = rng.standard_normal((102, 12))
    xb = _normed(xb)
    if xb.shape != (102, 12):
        tmp = rng.standard_normal((102, 12))
        tmp = _normed(tmp)
        xb = tmp
    n, d = 102, 12
    nb = d - 1
    N = n * nb
    K = 30
    best_val = _max_cos(xb)
    x_best = xb.copy()
    x_cur = xb.copy()
    cur_val = best_val
    best_set, _ = _contact_set(x_best, 1e-7)
    cur_set = set(best_set)
    # cached nullspace structures for x_cur
    Bcur = None; Z = None; wstress = None
    Ic = Jc = None; Imon = Jmon = None; Qmon = None
    need_recompute = True
    lp_spent = 0.0
    trial = 0
    # initial quick LP-free; go to loop
    while time.time() < T_END - 16.0 and trial < 200:
        trial += 1
        if need_recompute:
            if time.time() > T_END - 25.0:
                break
            try:
                Gc = x_cur @ x_cur.T
                np.fill_diagonal(Gc, -2.0)
                mcur = float(Gc.max())
                iu = np.triu_indices(n, 1)
                cfull = Gc[iu]
                sel_c = cfull > mcur - 1e-7
                Ic = iu[0][sel_c]; Jc = iu[1][sel_c]
                if len(Ic) < 50:
                    # widen a bit
                    sel_c = cfull > mcur - 1e-6
                    Ic = iu[0][sel_c]; Jc = iu[1][sel_c]
                if len(Ic) > 2000:
                    # keep tightest 2000
                    ordt = np.argsort(-cfull[sel_c])[:2000]
                    Ic = Ic[ordt]; Jc = Jc[ordt]
                Bcur = _tangent_bases(x_cur)
                Jt = _build_Jt(x_cur, Bcur, Ic, Jc)
                # SVD for near-null
                try:
                    _, S, Vt = np.linalg.svd(Jt, full_matrices=False)
                    # smallest K
                    Z = Vt[-K:].T.copy()  # N x K
                except Exception:
                    Z = None
                wstress = _stress_weights(x_cur, Ic, Jc, iters=150)
                # monitoring set: pairs within 0.02 of max
                sel_m = cfull > mcur - 0.02
                Imon = iu[0][sel_m]; Jmon = iu[1][sel_m]
                if len(Imon) > 4000:
                    # keep top 4000
                    idx = np.argsort(-cfull[sel_m])[:4000]
                    Imon = Imon[idx]; Jmon = Jmon[idx]
                if Z is not None:
                    Jmon_t = _build_Jt(x_cur, Bcur, Imon, Jmon)
                    Qmon = Jmon_t @ Z  # m x K
                    # map contact rows to mon rows: build dict
                else:
                    Qmon = None
                need_recompute = False
            except Exception:
                need_recompute = False
                Z = None
                # fallback: random kick + quench
        # ---- propose slide ----
        try:
            if Z is None or Qmon is None or Bcur is None:
                # fallback random slide
                x_slide = x_cur + 0.05 * rng.standard_normal((n, d))
                x_slide = _normed(x_slide)
            else:
                p = len(Ic)
                order = np.argsort(-wstress)
                # choose 20-80 high stress
                s_cnt = int(rng.integers(20, 81))
                s_cnt = min(s_cnt, p)
                # take top 2*s pool then subsample for diversity
                pool = order[:min(p, max(s_cnt*2, s_cnt+10))]
                if len(pool) > s_cnt:
                    sel_idx = rng.choice(pool, size=s_cnt, replace=False)
                else:
                    sel_idx = pool
                # map selected contacts to mon indices: since Imon superset of Ic (0.02>1e-7), find positions
                # build lookup dict for mon
                # for speed, use search: mon key -> pos
                # do it via dictionary (few thousand)
                mon_dict = { (int(Imon[k]), int(Jmon[k])): k for k in range(len(Imon)) }
                sel_mon = []
                for k in sel_idx:
                    key = (int(Ic[k]), int(Jc[k]))
                    if key in mon_dict:
                        sel_mon.append(mon_dict[key])
                if len(sel_mon) == 0:
                    sel_mon = list(range(min(s_cnt, len(Imon))))
                sel_mon = np.array(sel_mon, dtype=int)
                R = 300
                Arand = rng.standard_normal((K, R))
                Arand /= np.linalg.norm(Arand, axis=0, keepdims=True) + 1e-30
                D = Qmon @ Arand  # m x R
                Dsel = D[sel_mon, :]  # s x R
                # deactivation: derivative < -0.02 per unit norm
                deact = (Dsel < -0.02).sum(axis=0)
                viol = D.max(axis=0)  # worst increase
                # also consider second-order: mean increase of top 20
                # score
                lam = 60.0
                score = deact.astype(float) - lam * np.maximum(viol, 0)
                # bonus for making selected strongly negative on average
                mean_sel = Dsel.mean(axis=0)
                score -= 5.0 * np.maximum(mean_sel, 0)
                bi = int(np.argmax(score))
                a_best = Arand[:, bi].copy()
                # randomize sign check: ensure it actually deactivates; if best deact==0, try opposite?
                # pick theta 0.03-0.15
                theta = float(rng.uniform(0.03, 0.15))
                scale = np.sqrt(n) * theta
                # guard against huge violation: predicted max
                v = float(viol[bi])
                if v > 1e-12:
                    # keep predicted increase < 0.03
                    allow = 0.035 / (v * scale + 1e-12) * scale
                    # allow is scale cap
                    cap = 0.035 / (v + 1e-12)
                    if scale * v > 0.035:
                        scale = cap
                        theta = scale / np.sqrt(n)
                coeff = Z @ a_best * scale  # N
                dx = np.empty((n, d))
                for i in range(n):
                    dx[i] = Bcur[i] @ coeff[i*nb:(i+1)*nb]
                x_slide = _normed(x_cur + dx)
            # ---- log-sum-exp continuation 0.01 -> 0.0005 ----
            # time-aware steps
            remain = T_END - time.time()
            if remain < 20:
                break
            # 3-stage quench
            x_q = _lse_adam(x_slide, 250, 0.01, 0.003, 0.004)
            x_q = _lse_adam(x_q, 250, 0.003, 0.001, 0.003)
            x_q = _lse_adam(x_q, 300, 0.001, 0.0005, 0.002)
            vq = _max_cos(x_q)
            qset, _ = _contact_set(x_q, 1e-7)
            inter = len(qset & best_set)
            union = len(qset | best_set)
            dist = 1.0 - (inter / union) if union > 0 else 0.0
            # acceptance
            improved = vq < best_val - 1e-12
            qualifies_lp = (vq < 0.2685) or (dist >= 0.10)
            v_final = vq
            x_final = x_q
            if qualifies_lp and (time.time() < T_END - 16.0) and (lp_spent < 28.0):
                lp_bud = 4.0
                # smaller budget if many trials
                lp0 = time.time()
                x_lp, v_lp = _lp_polish(x_q, lp_bud, T_END - 12.0)
                lp_spent += time.time() - lp0
                # update dist for lp point
                lpset, _ = _contact_set(x_lp, 1e-7)
                inter2 = len(lpset & best_set)
                union2 = len(lpset | best_set)
                dist2 = 1.0 - (inter2 / union2) if union2 > 0 else 0.0
                if v_lp < v_final:
                    v_final = v_lp
                    x_final = x_lp
                    dist = dist2
                    qset = lpset
            if v_final < best_val - 1e-12:
                x_best = _normed(x_final.copy())
                best_val = float(v_final)
                best_set = set(qset)
                x_cur = x_best.copy()
                cur_val = best_val
                cur_set = set(best_set)
                need_recompute = True
            else:
                # accept as new cur based on objective + distance ( Basin hopping )
                # allow wandering to different topology even if slightly worse
                accept = False
                if v_final < cur_val:
                    accept = True
                elif dist >= 0.10 and v_final < cur_val + 0.004 and v_final < 0.269:
                    # diverse basin, not too bad -> accept with prob 0.7
                    if rng.random() < 0.7:
                        accept = True
                elif dist >= 0.25 and v_final < cur_val + 0.008:
                    if rng.random() < 0.4:
                        accept = True
                if accept:
                    x_cur = _normed(x_final.copy())
                    cur_val = float(v_final)
                    cur_set = set(qset)
                    need_recompute = True
                else:
                    # stay, but occasionally force diversity: if stuck, perturb cur slightly
                    if trial % 7 == 0:
                        # small random nudge of cur without recompute? keep same Z to save time
                        pass
                    # reuse Z for next trial (no recompute) to try different direction
                    need_recompute = False
                    # if cur far from best and not improving, reset cur to best occasionally
                    if trial % 12 == 0 and cur_val > best_val + 0.002:
                        x_cur = x_best.copy()
                        cur_val = best_val
                        need_recompute = True
        except Exception:
            continue
    # final polish of best with remaining time
    try:
        rem = T_END - time.time()
        if rem > 4.0:
            bud = min(rem - 1.0, 14.0)
            x_best, best_val = _lp_polish(x_best, bud, T_END - 0.5)
    except Exception:
        pass
    return _normed(x_best)
