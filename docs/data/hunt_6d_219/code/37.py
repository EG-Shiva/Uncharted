import time
import numpy as np
from scipy.optimize import minimize, linprog
from scipy.sparse import coo_matrix

def _norm(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n = np.maximum(n, 1e-12)
    return X / n

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def solve():
    t0 = time.time()
    BUDGET = 168.0
    n, d = 219, 6
    iu0, iu1 = np.triu_indices(n, 1)
    rng = np.random.RandomState(42)

    x_best = None
    best_val = 1e9
    for fn in ["best.npy", "record.npy"]:
        try:
            v = np.load(fn)
            v = np.asarray(v, dtype=float)
            if v.shape == (n, d):
                v = _norm(v)
                c = _max_cos(v)
                if c < best_val:
                    best_val = c
                    x_best = v.copy()
        except Exception:
            pass
    if x_best is None:
        x_best = _norm(rng.randn(n, d))
        best_val = _max_cos(x_best)

    def run_lbfgs(x0, beta, maxiter):
        y0 = x0.ravel().copy()
        def fun(y):
            Y = y.reshape(n, d)
            rn = np.maximum(np.linalg.norm(Y, axis=1, keepdims=True), 1e-12)
            X = Y / rn
            G = X @ X.T
            g = G[iu0, iu1]
            m = float(np.max(g))
            e = np.exp(beta * (g - m))
            S = float(np.sum(e)) + 1e-300
            loss = m + np.log(S) / beta
            w = e / S
            W = np.zeros((n, n))
            W[iu0, iu1] = w
            W = W + W.T
            GX = W @ X
            dot = (GX * X).sum(axis=1, keepdims=True)
            GY = (GX - dot * X) / rn
            return float(loss), GY.ravel()
        try:
            res = minimize(fun, y0, jac=True, method="L-BFGS-B",
                           options={"maxiter": maxiter, "ftol": 1e-12, "gtol": 1e-7, "maxfun": 20000})
            Y = res.x.reshape(n, d)
            return _norm(Y)
        except Exception:
            return _norm(x0)

    def lp_polish(x, deadline, r_init=1.0e-3, max_steps=5):
        x = _norm(x.copy())
        best = _max_cos(x)
        bx = x.copy()
        radius = r_init
        steps = 0
        while time.time() < deadline and steps < max_steps and time.time() - t0 < BUDGET - 1.5:
            steps += 1
            if radius < 1e-14:
                break
            Gf = x @ x.T
            c = Gf[iu0, iu1]
            m = float(c.max())
            thr = m - max(18.0 * radius, 1e-9)
            sel = c > thr
            I = iu0[sel]; J = iu1[sel]; csel = c[sel]
            p = len(I)
            if p == 0:
                break
            if p > 3500:
                k2 = 3500
                part = np.argpartition(-csel, k2 - 1)[:k2]
                I, J, csel = I[part], J[part], csel[part]
                p = k2
            rows = np.repeat(np.arange(p), 2 * d + 1)
            cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d),
                                   np.full((p, 1), n * d)], 1).ravel()
            vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
            try:
                A_ub = coo_matrix((vals, (rows, cols)), shape=(p, n * d + 1)).tocsr()
                A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))),
                                  shape=(n, n * d + 1)).tocsr()
                obj = np.zeros(n * d + 1); obj[-1] = 1.0
                res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq, b_eq=np.zeros(n),
                              bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
            except Exception:
                radius *= 0.5
                continue
            if res.status != 0:
                radius *= 0.5
                continue
            y = _norm(x + res.x[:-1].reshape(n, d))
            vv = _max_cos(y)
            if vv < best:
                x, best = y, vv
                bx = y.copy()
                radius = min(radius * 1.5, 0.03)
            else:
                radius *= 0.5
        return bx, best

    def refine_hole(c0, Xk, steps=120, beta=400.0, lr=0.15):
        c = c0 / max(np.linalg.norm(c0), 1e-12)
        best_c = c.copy()
        # value = max dot
        bv = float(np.max(Xk @ c))
        cur = c.copy()
        for it in range(steps):
            s = Xk @ cur
            m = float(np.max(s))
            e = np.exp(beta * (s - m))
            S = float(np.sum(e)) + 1e-300
            w = e / S
            g = w @ Xk  # gradient of softmax wrt cur
            gt = g - float(g @ cur) * cur
            gn = float(np.linalg.norm(gt))
            if not np.isfinite(gn) or gn < 1e-14:
                break
            st = lr / (1.0 + it * 0.05)
            nxt = cur - (st / gn) * gt * min(1.0, gn * 5.0 + 0.2)
            # scale step adaptively; simpler: fixed tangent step
            nxt = cur - (st * 0.3 / max(gn, 1e-12)) * gt
            nxt = nxt / max(np.linalg.norm(nxt), 1e-12)
            vn = float(np.max(Xk @ nxt))
            if vn < m + 2e-4:
                cur = nxt
                if vn < bv:
                    bv = vn
                    best_c = nxt.copy()
            else:
                lr *= 0.7
                if lr < 1e-4:
                    break
        return best_c, bv

    def greedy_holes(Xkeep, k_new, rs):
        # Xkeep: (m,d) unit
        cur_keep = Xkeep.copy()
        outs = []
        for _ in range(k_new):
            S = 3500
            C = rs.randn(S, d)
            C = C / np.maximum(np.linalg.norm(C, axis=1, keepdims=True), 1e-12)
            M = C @ cur_keep.T
            mx = M.max(axis=1)
            # take few best
            bi = np.argsort(mx)[:6]
            bc = None; bv = 1e9
            for b in bi:
                cc, vv = refine_hole(C[b], cur_keep, steps=90)
                if vv < bv:
                    bv = vv; bc = cc
            outs.append(bc)
            cur_keep = np.vstack([cur_keep, bc.reshape(1, d)])
        return np.array(outs)

    def row_stats(X):
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        rm = G.max(axis=1)
        m = float(rm.max())
        deg = (G > m - 1e-3).sum(axis=1)
        return rm, deg, m, G

    # walking state for topology exploration
    x_cur = x_best.copy()
    cur_val = best_val

    hop_deadline = t0 + BUDGET - 14.0
    it = 0
    # pre-allocate random states deterministically
    while time.time() < hop_deadline and time.time() - t0 < BUDGET - 14:
        it += 1
        if it > 80:
            break
        r = rng.rand()
        base = x_cur.copy() if (rng.rand() < 0.7) else x_best.copy()
        rm, deg, mglob, Gfull = row_stats(base)
        order_max = np.argsort(-rm)
        order_deg = np.argsort(-deg)

        if r < 0.55:
            # kill-and-fill void re-insertion (main basin changer)
            k_kill = int(rng.randint(2, 7))
            # victims: mostly worst row_max, sometimes high degree
            cand = np.concatenate([order_max[:18], order_deg[:10]])
            cand = np.unique(cand)
            # score victims
            perm = rng.permutation(len(cand))
            cand = cand[perm]
            # bias to worst: sort subset by rm
            # pick k_kill among cand weighted by rm
            scores = rm[cand] + 0.002 * deg[cand]
            pick = np.argsort(-scores)[:k_kill]
            # with prob, ensure not picking the rattler (min rm)
            ratt = int(np.argmin(rm))
            victims = cand[pick].tolist()
            if ratt in victims and len(victims) > 1:
                victims.remove(ratt)
                # replace with next
                for c in cand:
                    if c not in victims and c != ratt:
                        victims.append(int(c)); break
                    if len(victims) >= k_kill:
                        break
                victims = victims[:k_kill]
            keep_mask = np.ones(n, dtype=bool)
            keep_mask[np.array(victims, dtype=int)] = False
            Xkeep = base[keep_mask]
            rs = np.random.RandomState(10000 + it * 97)
            try:
                newpts = greedy_holes(Xkeep, len(victims), rs)
            except Exception:
                continue
            Xn0 = base.copy()
            Xn0[np.array(victims, dtype=int)] = newpts
            beta = 250.0 if rng.rand() < 0.5 else 450.0
            Xn = run_lbfgs(Xn0, beta, maxiter=60)
            vn = _max_cos(Xn)
            if vn < min(cur_val, best_val) + 0.0025:
                Xn2, vn2 = lp_polish(Xn, hop_deadline, r_init=6e-4, max_steps=4)
                if vn2 < vn:
                    Xn, vn = Xn2, vn2
            # acceptance: improve best or sideways walk
            if vn < best_val:
                best_val = vn; x_best = Xn.copy()
                x_cur = Xn.copy(); cur_val = vn
            elif vn < cur_val + 0.0015 and rng.rand() < 0.45:
                x_cur = Xn.copy(); cur_val = vn
            elif vn < best_val + 0.001 and rng.rand() < 0.15:
                x_cur = Xn.copy(); cur_val = vn
        elif r < 0.8:
            # cluster kick around most jammed region to shatter core
            seed = int(order_max[0] if rng.rand() < 0.6 else rng.choice(order_max[:20]))
            row = Gfull[seed]
            nn = np.argsort(-row)[:60]
            k2 = int(rng.randint(12, 38))
            idx = np.unique(nn[:k2])
            cur = base.copy()
            pert = rng.randn(len(idx), d)
            pert -= (np.sum(pert * cur[idx], axis=1, keepdims=True)) * cur[idx]
            sig = 0.25 + 0.45 * rng.rand()
            cur[idx] = _norm(cur[idx] + sig * pert)
            gj = rng.randn(n, d)
            gj -= (np.sum(gj * cur, axis=1, keepdims=True)) * cur
            cur = _norm(cur + 0.004 * gj)
            beta = 200.0 if rng.rand() < 0.5 else 350.0
            Xn = run_lbfgs(cur, beta, maxiter=55)
            vn = _max_cos(Xn)
            if vn < min(cur_val, best_val) + 0.002:
                Xn2, vn2 = lp_polish(Xn, hop_deadline, r_init=6e-4, max_steps=4)
                if vn2 < vn:
                    Xn, vn = Xn2, vn2
            if vn < best_val:
                best_val = vn; x_best = Xn.copy()
                x_cur = Xn.copy(); cur_val = vn
            elif vn < cur_val and rng.rand() < 0.6:
                x_cur = Xn.copy(); cur_val = vn
        else:
            # rattler-integration: move rattler + neighbours into balanced position
            ratt = int(np.argmin(rm))
            # neighbours closest to rattler (largest dot with rattler)
            dots = base @ base[ratt]
            dots[ratt] = -2
            nb = np.argsort(-dots)[:int(rng.randint(6, 14))]
            cur = base.copy()
            # place rattler at perturbed hole w.r.t. others
            mask = np.ones(n, dtype=bool); mask[ratt] = False
            Xrest = cur[mask]
            rs = np.random.RandomState(20000 + it * 131)
            C = rs.randn(2500, d); C = _norm(C)
            mx = (C @ Xrest.T).max(axis=1)
            best_c = C[int(np.argmin(mx))]
            bc, _ = refine_hole(best_c, Xrest, steps=80)
            cur[ratt] = bc
            # jitter neighbours
            pert = rng.randn(len(nb), d)
            pert -= (np.sum(pert * cur[nb], axis=1, keepdims=True)) * cur[nb]
            cur[nb] = _norm(cur[nb] + (0.15 + 0.25 * rng.rand()) * pert)
            Xn = run_lbfgs(cur, 300.0, maxiter=60)
            vn = _max_cos(Xn)
            if vn < min(cur_val, best_val) + 0.002:
                Xn2, vn2 = lp_polish(Xn, hop_deadline, r_init=6e-4, max_steps=4)
                if vn2 < vn:
                    Xn, vn = Xn2, vn2
            if vn < best_val:
                best_val = vn; x_best = Xn.copy()
                x_cur = Xn.copy(); cur_val = vn
            elif vn < cur_val + 0.001 and rng.rand() < 0.4:
                x_cur = Xn.copy(); cur_val = vn

    # final sharpen best with escalating beta + LP
    try:
        for beta, mi in [(180.0, 70), (400.0, 70), (800.0, 50)]:
            if time.time() - t0 > BUDGET - 6:
                break
            Xn = run_lbfgs(x_best, beta, maxiter=mi)
            vn = _max_cos(Xn)
            if vn < best_val:
                best_val = vn; x_best = Xn.copy()
    except Exception:
        pass
    try:
        if time.time() - t0 < BUDGET - 2:
            Xp, vp = lp_polish(x_best, t0 + BUDGET - 1.0, r_init=1.0e-3, max_steps=10)
            if vp < best_val:
                best_val = vp; x_best = Xp.copy()
    except Exception:
        pass
    x_best = _norm(x_best)
    return x_best
