import time
import numpy as np
from scipy.optimize import linprog, minimize
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _contact_mask(x, tol=2e-7):
    n = x.shape[0]
    iu0, iu1 = np.triu_indices(n, 1)
    g = (x @ x.T)[iu0, iu1]
    m = float(g.max())
    return (g > m - tol), m

def _jaccard(m1, m2):
    inter = np.count_nonzero(m1 & m2)
    union = np.count_nonzero(m1 | m2)
    if union == 0:
        return 0.0
    return 1.0 - inter / union

def _lp_polish(x, deadline, r0=0.004, cap=1500):
    x = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
    n, d = x.shape
    try:
        best = _max_cos(x)
    except Exception:
        return x, 1.0
    radius = r0
    iu0, iu1 = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        try:
            G = x @ x.T
            c = G[iu0, iu1]
            m = float(c.max())
            sel = c > m - max(20.0 * radius, 1e-9)
            if np.count_nonzero(sel) == 0:
                sel[int(np.argmax(c))] = True
            cnt = int(np.count_nonzero(sel))
            if cnt > cap:
                idx = np.argsort(-c)[:cap]
                sel2 = np.zeros_like(sel, dtype=bool)
                sel2[idx] = True
                sel = sel2
            I = iu0[sel]; J = iu1[sel]
            csel = c[sel]
            p = len(I)
            nv = n * d + 1
            rows = np.repeat(np.arange(p), 2 * d + 1)
            cols = np.empty((p, 2 * d + 1), dtype=int)
            ar = np.arange(d)
            cols[:, :d] = I[:, None] * d + ar
            cols[:, d:2 * d] = J[:, None] * d + ar
            cols[:, 2 * d] = n * d
            cols = cols.ravel()
            vals = np.empty((p, 2 * d + 1), dtype=float)
            vals[:, :d] = x[J]
            vals[:, d:2 * d] = x[I]
            vals[:, 2 * d] = -1.0
            vals = vals.ravel()
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv); obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
            if res.status != 0:
                radius *= 0.5
                continue
            y = x + res.x[:-1].reshape(n, d)
            y /= np.maximum(np.linalg.norm(y, axis=1, keepdims=True), 1e-12)
            v = _max_cos(y)
            if v < best:
                x, best = y, v
                radius = min(radius * 1.5, 0.05)
            else:
                radius *= 0.5
        except Exception:
            radius *= 0.5
            continue
    return x, best

def _gd(x0, steps, T0, T1, lr0, mu, rng, noise0=0.0, active=None, deadline=None):
    x = x0.copy()
    n, d = x.shape
    if active is None:
        act = np.ones(n, dtype=bool)
    else:
        act = np.asarray(active, dtype=bool)
    v = np.zeros_like(x)
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    best_val = float(G.max())
    best = x.copy()
    lratio = np.log(T1 / T0) if T0 > 0 and T1 > 0 else 0.0
    for k in range(steps):
        if deadline is not None and (k & 255) == 0 and time.time() > deadline:
            break
        frac = k / max(1, steps - 1)
        T = T0 * np.exp(lratio * frac)
        if T < 1e-12:
            T = 1e-12
        lr = lr0 * (1.0 - 0.55 * frac)
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        E = np.exp((G - m) / T)
        np.fill_diagonal(E, 0.0)
        S = float(E.sum() * 0.5)
        if S < 1e-300 or not np.isfinite(S):
            S = 1e-300
        grad = (E @ x) / S
        grad -= (grad * x).sum(axis=1, keepdims=True) * x
        if not np.all(act):
            grad[~act] = 0.0
            v[~act] = 0.0
        v = mu * v + (1.0 - mu) * grad
        x[act] = x[act] - lr * v[act]
        if noise0 > 0:
            amp = noise0 * (1.0 - frac)
            if amp > 0:
                nz = rng.randn(n, d) * amp
                nz -= (nz * x).sum(axis=1, keepdims=True) * x
                nz[~act] = 0.0
                x[act] = x[act] + nz[act]
        x[act] /= np.maximum(np.linalg.norm(x[act], axis=1, keepdims=True), 1e-12)
        if m < best_val:
            best_val = m
            best = x.copy()
    return best, best_val

def _refine_hole(y0, Xf, iters=220):
    y = y0 / max(np.linalg.norm(y0), 1e-12)
    dots = Xf @ y
    vy = float(dots.max())
    step = 0.05
    for it in range(iters):
        dots = Xf @ y
        j = int(np.argmax(dots))
        dj = float(dots[j])
        g = Xf[j] - dj * y
        ng = float(np.linalg.norm(g))
        if ng < 1e-13:
            break
        g /= ng
        improved = False
        for s in (step, step * 0.3):
            yn = y - s * g
            yn /= max(np.linalg.norm(yn), 1e-12)
            vn = float((Xf @ yn).max())
            if vn < vy:
                y, vy = yn, vn
                improved = True
                break
        if not improved:
            step *= 0.92
            if step < 1e-5:
                break
        else:
            step = min(step * 1.05, 0.05)
    return y, vy

def solve():
    t0 = time.time()
    DEADLINE = t0 + 172.0
    SEARCH_END = t0 + 126.0
    rng = np.random.RandomState(20261102)
    n = 163
    try:
        bx = np.load("best.npy")
    except Exception:
        try:
            bx = np.load("record.npy")
        except Exception:
            bx = rng.randn(n, 4)
    bx = np.asarray(bx, dtype=float)
    if bx.shape != (n, 4) or not np.all(np.isfinite(bx)):
        bx = rng.randn(n, 4)
    bx = bx / np.maximum(np.linalg.norm(bx, axis=1, keepdims=True), 1e-12)
    best_x = bx.copy()
    best_val = _max_cos(best_x)
    best_mask, _ = _contact_mask(best_x)
    elite = [(best_val, best_x.copy())]
    ks = [6, 10, 14, 18]
    trial = 0
    # precompute stressed order of incumbent (updated when best improves)
    def stressed_order(X):
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        deg = (G > m - 2e-7).sum(axis=1)
        # tie-break by sum of top dots (stress)
        order = np.argsort(deg + 1e-6 * rng.rand(len(deg)))
        return order, deg, G, m
    while time.time() < SEARCH_END:
        if time.time() > DEADLINE - 45:
            break
        trial += 1
        k = ks[(trial - 1) % len(ks)]
        # vary deletion size slightly on later cycles
        if trial > 12 and (trial % 3) == 0:
            k = int(k + rng.randint(-2, 3))
            k = max(4, min(20, k))
        order, deg, Gfull, mfull = stressed_order(best_x)
        # most stressed = highest degree (8-9 contacts); order ascending so take tail
        del_idx = order[-k:]
        # occasionally mix in a random mid-degree to diversify orders
        if (trial % 4) == 0 and k >= 6:
            swap_n = min(2, k // 3)
            cand_mid = order[len(order)//2:len(order)-k]
            if len(cand_mid) >= swap_n:
                pick = rng.choice(len(cand_mid), swap_n, replace=False)
                del_idx[:swap_n] = cand_mid[pick]
        keep_mask = np.ones(n, dtype=bool)
        keep_mask[del_idx] = False
        x_rem = best_x[keep_mask].copy()
        nr = x_rem.shape[0]
        # annealed softmax on remainder to push max well below full target
        try:
            x_rem_p, v_rem = _gd(x_rem, steps=1400, T0=0.008, T1=0.0004, lr0=0.035, mu=0.86,
                                 rng=rng, noise0=0.001, deadline=SEARCH_END)
        except Exception:
            x_rem_p, v_rem = x_rem, _max_cos(x_rem)
        # short LP on remainder (cheaper, fewer points) to under-press
        if time.time() < SEARCH_END - 2:
            try:
                x_rem_p, v_rem = _lp_polish(x_rem_p, deadline=min(SEARCH_END, time.time() + 1.2), r0=0.004)
            except Exception:
                pass
        cur_fixed = x_rem_p.copy()
        # sequential greedy reinsertion
        ok = True
        inserted = []
        for ins in range(k):
            if time.time() > SEARCH_END - 1.0:
                ok = False
                break
            Y = rng.randn(20000, 4)
            Y /= np.maximum(np.linalg.norm(Y, axis=1, keepdims=True), 1e-12)
            D = Y @ cur_fixed.T
            vals = D.max(axis=1)
            # best 20 holes
            if len(vals) <= 20:
                cand = np.arange(len(vals))
            else:
                cand = np.argpartition(vals, 20)[:20]
            # refine each
            best_y = None
            best_vy = 1e9
            for ci in cand:
                try:
                    yr, vr = _refine_hole(Y[ci], cur_fixed, iters=180)
                except Exception:
                    continue
                if vr < best_vy:
                    best_vy = vr
                    best_y = yr
            if best_y is None:
                ok = False
                break
            cur_fixed = np.vstack([cur_fixed, best_y.reshape(1, 4)])
            inserted.append(cur_fixed.shape[0] - 1)
            # locally relax inserted point + neighbours with cosine > 0.75
            try:
                dots_ins = cur_fixed[:-1] @ best_y
                nbr = np.where(dots_ins > 0.75)[0]
                act = np.zeros(cur_fixed.shape[0], dtype=bool)
                act[nr + ins] = True
                # nbr indices refer to pre-insert array, same indexing for first part
                act[nbr] = True
                # always include a couple of worst neighbours even if threshold empty
                if int(act.sum()) < 3:
                    worst = np.argsort(-dots_ins)[:3]
                    act[worst] = True
                cur_fixed, _ = _gd(cur_fixed, steps=260, T0=0.004, T1=0.0004, lr0=0.02, mu=0.85,
                                   rng=rng, noise0=0.0005, active=act, deadline=SEARCH_END)
            except Exception:
                pass
        if not ok:
            continue
        # cur_fixed is now 163 x 4 (order permuted); global light anneal
        cur = cur_fixed.copy()
        cur /= np.maximum(np.linalg.norm(cur, axis=1, keepdims=True), 1e-12)
        try:
            cur_b, cur_v = _gd(cur, steps=2200, T0=0.008 if (trial % 2) else 0.012, T1=0.0004,
                               lr0=0.035, mu=0.87, rng=rng, noise0=0.0012, deadline=SEARCH_END)
        except Exception:
            cur_b, cur_v = cur, _max_cos(cur)
        # elite / Jaccard gating for full LP polish
        try:
            cmask, _ = _contact_mask(cur_b)
            jd = _jaccard(cmask, best_mask)
        except Exception:
            jd = 1.0
        do_lp = (cur_v < best_val) or (jd > 0.10 and cur_v < best_val + 0.003)
        if do_lp and time.time() < DEADLINE - 8:
            try:
                lp_x, lp_v = _lp_polish(cur_b, deadline=min(DEADLINE - 2.0, time.time() + 4.5), r0=0.003)
                if lp_v < cur_v:
                    cur_b, cur_v = lp_x, lp_v
            except Exception:
                pass
        # keep several distinct reinsertions in elite archive
        if cur_v < best_val + 0.004:
            # check distinctness vs elite
            try:
                cm2, _ = _contact_mask(cur_b)
                distinct = True
                for ev, ex in elite:
                    try:
                        em, _ = _contact_mask(ex)
                        if _jaccard(cm2, em) < 0.05 and abs(cur_v - ev) < 2e-7:
                            distinct = False
                            break
                    except Exception:
                        continue
                if distinct:
                    elite.append((cur_v, cur_b.copy()))
                    elite.sort(key=lambda z: z[0])
                    elite = elite[:6]
            except Exception:
                pass
        if cur_v < best_val - 1e-12:
            best_val = cur_v
            best_x = cur_b.copy()
            best_mask, _ = _contact_mask(best_x)
            elite[0] = (best_val, best_x.copy())
        # restart with different orders: occasionally seed next trial from non-best elite
        if len(elite) > 1 and (trial % 6) == 5:
            # promote a diverse elite to incumbent for one surgery cycle without losing best
            # do this by swapping best_x temporarily: use second elite as base next iter
            # implement by rotating elite
            elite.sort(key=lambda z: z[0])
            # keep best, but set best_x to diverse member for next trial then restore afterwards?
            # simpler: replace best_x with diverse if its Jaccard is large (basin hop), keep true best in elite[0]
            div_idx = int(rng.randint(1, len(elite)))
            best_x = elite[div_idx][1].copy()
            best_mask, _ = _contact_mask(best_x)
            # note best_val stays as elite[0][0]; will be restored when improved or after a few trials
            # restore true best every few hops
            if (trial % 18) == 0:
                best_x = elite[0][1].copy()
                best_val = elite[0][0]
                best_mask, _ = _contact_mask(best_x)
    # final: ensure true best is returned, fully polish if time
    elite.sort(key=lambda z: z[0])
    best_val_e, best_x_e = elite[0]
    best_x, best_val = best_x_e.copy(), float(best_val_e)
    if time.time() < DEADLINE - 6:
        try:
            px, pv = _lp_polish(best_x, deadline=DEADLINE - 1.5, r0=0.002)
            if pv < best_val:
                best_x, best_val = px, pv
        except Exception:
            pass
    best_x = best_x / np.maximum(np.linalg.norm(best_x, axis=1, keepdims=True), 1e-12)
    return best_x
