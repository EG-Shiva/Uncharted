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

def _adam_smooth(X, steps, tau0, tau1, lr0, lr1):
    n, d = X.shape
    x = X.copy()
    m = np.zeros_like(x)
    v = np.zeros_like(x)
    b1, b2, eps = 0.9, 0.999, 1e-8
    tt = 0
    ltau = np.log(tau1 / tau0) if tau0 != tau1 else 0.0
    llr = np.log(lr1 / lr0) if lr0 != lr1 else 0.0
    for s in range(steps):
        f = s / max(1, steps - 1)
        tau = tau0 * np.exp(ltau * f)
        lr = lr0 * np.exp(llr * f)
        if tau < 1e-12:
            tau = 1e-12
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        mg = float(G.max())
        W = np.exp((G - mg) / tau)
        np.fill_diagonal(W, 0.0)
        rs = W.sum(axis=1, keepdims=True) + 1e-12
        grad = (W @ x) / rs
        dot = (grad * x).sum(axis=1, keepdims=True)
        gt = grad - dot * x
        tt += 1
        m = b1 * m + (1.0 - b1) * gt
        vv = b2 * v + (1.0 - b2) * (gt * gt)
        v = vv
        mh = m / (1.0 - b1 ** tt)
        vh = vv / (1.0 - b2 ** tt)
        step = lr * mh / (np.sqrt(vh) + eps)
        dot2 = (step * x).sum(axis=1, keepdims=True)
        step_t = step - dot2 * x
        x = x - step_t
        xn = np.linalg.norm(x, axis=1, keepdims=True)
        xn[xn == 0] = 1.0
        x /= xn
    return x

def _adam_subset(X, idx, steps, tau0, tau1, lr0, lr1):
    x = X.copy()
    m = np.zeros_like(x)
    v = np.zeros_like(x)
    b1, b2, eps = 0.9, 0.999, 1e-8
    tt = 0
    ltau = np.log(tau1 / tau0) if tau0 != tau1 else 0.0
    llr = np.log(lr1 / lr0) if lr0 != lr1 else 0.0
    idx = np.asarray(idx, dtype=int)
    for s in range(steps):
        f = s / max(1, steps - 1)
        tau = tau0 * np.exp(ltau * f)
        lr = lr0 * np.exp(llr * f)
        if tau < 1e-12:
            tau = 1e-12
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        mg = float(G.max())
        W = np.exp((G - mg) / tau)
        np.fill_diagonal(W, 0.0)
        rs = W.sum(axis=1, keepdims=True) + 1e-12
        grad = (W @ x) / rs
        dot = (grad * x).sum(axis=1, keepdims=True)
        gt = grad - dot * x
        tt += 1
        m = b1 * m + (1.0 - b1) * gt
        vv = b2 * v + (1.0 - b2) * (gt * gt)
        v = vv
        mh = m / (1.0 - b1 ** tt)
        vh = vv / (1.0 - b2 ** tt)
        step = lr * mh / (np.sqrt(vh) + eps)
        dot2 = (step * x).sum(axis=1, keepdims=True)
        step_t = step - dot2 * x
        x[idx] = x[idx] - step_t[idx]
        xn = np.linalg.norm(x[idx], axis=1, keepdims=True)
        xn[xn == 0] = 1.0
        x[idx] /= xn
    return x

def _find_hole(others, rng, n_cand=30000, steps=150):
    d = others.shape[1]
    C = rng.standard_normal((n_cand, d))
    C /= np.linalg.norm(C, axis=1, keepdims=True)
    D = C @ others.T
    worst = D.max(axis=1)
    vv = C[int(np.argmin(worst))].copy()
    lr = 0.05
    tau = 0.005
    for it in range(steps):
        dots = others @ vv
        m0 = float(dots.max())
        w = np.exp((dots - m0) / tau)
        s = float(w.sum())
        if s <= 0 or not np.isfinite(s):
            break
        w /= s
        g = w @ others
        gt = g - float(g @ vv) * vv
        nrm = float(np.linalg.norm(gt))
        if nrm < 1e-12 or not np.isfinite(nrm):
            break
        gt /= nrm
        v2 = vv - lr * gt
        n2 = float(np.linalg.norm(v2))
        if n2 < 1e-12:
            lr *= 0.5
            continue
        v2 /= n2
        m2 = float((others @ v2).max())
        if m2 < m0:
            vv = v2
            lr = min(lr * 1.05, 0.08)
        else:
            lr *= 0.5
            if lr < 1e-6:
                break
            if it % 50 == 49:
                tau *= 0.7
                if tau < 8e-4:
                    tau = 8e-4
    nvv = float(np.linalg.norm(vv))
    if nvv == 0 or not np.isfinite(nvv):
        vv = C[0]
        nvv = float(np.linalg.norm(vv))
    vv /= nvv
    return vv

def solve():
    t0 = time.time()
    DEADLINE = 172.0
    SURG_END = t0 + 96.0
    rng = np.random.default_rng(0)
    n, d = 93, 13
    dm = d - 1
    N = n * dm
    iu0, iu1 = np.triu_indices(n, 1)
    Pfull = len(iu0)
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
        best = _norm(rng.standard_normal((n, d)))
        best_val = _maxcos(best)
    incumbent = best.copy()
    inc_val = float(best_val)
    # ---------- surgery phase: balanced dense+rattler transplant ----------
    k_cycle = [4, 6, 8, 5, 7, 9]
    trial = 0
    while time.time() < SURG_END:
        if (t0 + DEADLINE) - time.time() < 80:
            break
        trial += 1
        try:
            base = incumbent
            G = base @ base.T
            np.fill_diagonal(G, -2.0)
            M = float(G.max())
            flat = int(np.argmax(G))
            a = flat // n; b = flat % n
            rowmax = G.max(axis=1)
            # contacts at 1e-7
            thr = M - 1e-7
            contacts = (G > thr).sum(axis=1)
            # overloaded ranking
            order_load = np.argsort(-(contacts * 10.0 + (rowmax - M + 1.0)))
            # rattlers: rowmax < M-1e-7
            ratt = np.where(rowmax < M - 1e-7)[0]
            k = k_cycle[(trial - 1) % len(k_cycle)]
            # split k into dense + rattler
            n_r = 0
            if len(ratt) > 0:
                n_r = min(len(ratt), k // 2)
                if trial % 3 == 0:
                    n_r = min(len(ratt), max(1, k - 2))
            n_d = k - n_r
            if n_d < 1:
                n_d = 1
                n_r = k - 1
            # pick dense centre: most loaded among {a,b} else top
            ca = int(contacts[a]); cb = int(contacts[b])
            centre = a if ca >= cb else b
            if trial % 4 == 3:
                centre = int(order_load[0])
            # neighbours of centre by cosine
            nn = np.argsort(-G[centre])
            dense = [int(centre)]
            for cand in nn:
                cand = int(cand)
                if cand == centre:
                    continue
                if cand not in dense:
                    dense.append(cand)
                if len(dense) >= n_d:
                    break
            # rattler choice: random subset, prefer lowest rowmax
            if n_r > 0 and len(ratt) > 0:
                # bias to lowest rowmax + random
                rs = np.asarray(ratt)
                # sort by rowmax ascending
                rs_sorted = rs[np.argsort(rowmax[rs])]
                # take pool of lowest 2*n_r then sample
                pooln = min(len(rs_sorted), max(n_r * 2, n_r + 2))
                poolr = rs_sorted[:pooln].tolist()
                # add some random rattlers for diversity
                if len(rs) > pooln and trial % 2 == 0:
                    extra = rng.choice(rs, size=min(n_r, len(rs)), replace=False).tolist()
                    poolr = list(dict.fromkeys(poolr + [int(x) for x in extra]))
                rng.shuffle(poolr)
                chosen_r = []
                for c in poolr:
                    c = int(c)
                    if c not in dense and c not in chosen_r:
                        chosen_r.append(c)
                    if len(chosen_r) >= n_r:
                        break
                if len(chosen_r) < n_r:
                    # fill with random non-dense
                    cand_all = [i for i in range(n) if i not in dense and i not in chosen_r]
                    rng.shuffle(cand_all)
                    for c in cand_all:
                        chosen_r.append(int(c))
                        if len(chosen_r) >= n_r:
                            break
            else:
                chosen_r = []
            chosen = dense + chosen_r
            chosen = [int(x) for x in chosen[:k]]
            # sequential hole insertion
            keep_mask = np.ones(n, dtype=bool)
            keep_mask[np.array(chosen, dtype=int)] = False
            stack = base[keep_mask].copy()
            order_ins = chosen.copy()
            rng.shuffle(order_ins)
            X = base.copy()
            for c in order_ins:
                vv = _find_hole(stack, rng, n_cand=30000, steps=150)
                X[int(c)] = vv
                stack = np.vstack([stack, vv[None, :]])
                if time.time() > t0 + DEADLINE - 75:
                    break
            sig = float(10 ** rng.uniform(-3.0, -2.15))
            X[chosen] += sig * rng.standard_normal((len(chosen), d))
            X[chosen] /= np.linalg.norm(X[chosen], axis=1, keepdims=True)
            # frozen then full anneal; vary params by trial
            if k <= 5:
                X = _adam_subset(X, chosen, steps=220, tau0=0.008, tau1=0.0006, lr0=0.015, lr1=0.002)
            else:
                X = _adam_subset(X, chosen, steps=160, tau0=0.010, tau1=0.0008, lr0=0.015, lr1=0.002)
            if trial % 2 == 0:
                X = _adam_smooth(X, steps=420, tau0=0.009, tau1=0.0005, lr0=0.012, lr1=0.0012)
            else:
                X = _adam_smooth(X, steps=320, tau0=0.013, tau1=0.0007, lr0=0.015, lr1=0.0015)
            vv2 = _maxcos(X)
            if vv2 < inc_val - 1e-12:
                incumbent = X.copy()
                inc_val = float(vv2)
        except Exception:
            continue
    # ---------- high-precision SQP polish ----------
    X = incumbent.copy()
    cur = float(_maxcos(X))
    bestX = X.copy()
    bestVal = float(cur)
    # tangent helpers
    D0 = np.zeros((d, dm))
    for bb in range(dm):
        D0[bb + 1, bb] = 1.0
    ar12 = np.arange(dm)
    c_obj = np.zeros(N + 1)
    c_obj[-1] = 1.0
    radius = 3e-3
    rmax = 1.0e-2
    rmin = 1e-12
    while time.time() - t0 < DEADLINE - 2.5:
        try:
            G = X @ X.T
            c_all = G[iu0, iu1]
            cur = float(c_all.max())
            # active set
            thr = max(1e-3, 3.0 * radius)
            sel = c_all > cur - thr
            cnt = int(sel.sum())
            if cnt < 800:
                idx = np.argpartition(-c_all, 800)[:800]
                sel = np.zeros(Pfull, dtype=bool)
                sel[idx] = True
                cnt = 800
            if cnt > 2500:
                idx = np.argpartition(-c_all, 2500)[:2500]
                sel = np.zeros(Pfull, dtype=bool)
                sel[idx] = True
                cnt = 2500
            Ic = iu0[sel]; Jc = iu1[sel]
            csub = c_all[sel]
            p = cnt
            # Householder tangent bases
            s = np.sign(X[:, 0])
            s[s == 0] = 1.0
            U = X.copy()
            U[:, 0] -= s
            denom = np.einsum('ij,ij->i', U, U)
            Barr = np.empty((n, d, dm))
            Barr[:] = D0[None, :, :]
            mask = denom > 1e-12
            if np.any(mask):
                Um = U[mask]
                denm = denom[mask]
                Uc = Um[:, 1:]
                prod = Um[:, :, None] * Uc[:, None, :]
                Barr[mask] -= 2.0 * prod / denm[:, None, None]
            BI = Barr[Ic]
            BJ = Barr[Jc]
            XI = X[Ic]
            XJ = X[Jc]
            Ai = np.einsum('pab,pa->pb', BI, XJ)
            Aj = np.einsum('pab,pa->pb', BJ, XI)
            rows = np.repeat(np.arange(p), 2 * dm + 1)
            cols = np.empty((p, 2 * dm + 1), dtype=np.int64)
            cols[:, :dm] = Ic[:, None] * dm + ar12[None, :]
            cols[:, dm:2 * dm] = Jc[:, None] * dm + ar12[None, :]
            cols[:, 2 * dm] = N
            vals = np.empty((p, 2 * dm + 1))
            vals[:, :dm] = Ai
            vals[:, dm:2 * dm] = Aj
            vals[:, 2 * dm] = -1.0
            A_ub = coo_matrix((vals.ravel(), (rows, cols.ravel())), shape=(p, N + 1)).tocsr()
            b_ub = -csub
            bounds = [(-radius, radius)] * N + [(-1.0, cur)]
            res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
            if not res.success:
                radius *= 0.5
                if radius < rmin:
                    radius = 1e-4
                    # tiny kick
                    kick = rng.standard_normal(X.shape) * 1e-5
                    kick -= (np.sum(kick * X, axis=1, keepdims=True)) * X
                    X = _norm(X + kick)
                continue
            y = res.x[:N]
            t_pred = float(res.x[N])
            if t_pred >= cur - 1e-12:
                radius *= 0.5
                if radius < rmin:
                    radius = 1e-4
                continue
            Y = y.reshape(n, dm)
            dx = np.einsum('ndm,nm->nd', Barr, Y)
            Xt = X + dx
            Xt /= np.linalg.norm(Xt, axis=1, keepdims=True)
            Gt = Xt @ Xt.T
            np.fill_diagonal(Gt, -2.0)
            curt = float(Gt.max())
            if curt < cur - 1e-12:
                # good step
                denom_pred = cur - t_pred
                denom_act = cur - curt
                ratio = denom_act / denom_pred if denom_pred > 0 else 0.0
                X = Xt
                cur = curt
                if curt < bestVal:
                    bestVal = float(curt)
                    bestX = Xt.copy()
                if ratio > 0.6:
                    radius = min(radius * 1.4, rmax)
                elif ratio > 0.15:
                    radius = min(radius * 1.1, rmax)
                else:
                    radius *= 0.8
            else:
                radius *= 0.5
                if radius < 5e-7 and curt < bestVal + 5e-6:
                    # accept tiny non-improving to keep moving? no
                    pass
                if radius < rmin:
                    # restart kick from best
                    X = bestX.copy()
                    kick = rng.standard_normal(X.shape) * 2e-4
                    kick -= (np.sum(kick * X, axis=1, keepdims=True)) * X
                    X = _norm(X + kick)
                    radius = 5e-4
        except Exception:
            try:
                radius *= 0.5
            except Exception:
                pass
            continue
    bestX = _norm(bestX)
    # tiny shrink guard (strict feasibility not needed but keep normalized)
    return bestX
