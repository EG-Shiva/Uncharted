import time
import numpy as np
from scipy.optimize import minimize, linprog
from scipy.sparse import coo_matrix

def solve():
    t0 = time.time()
    BUDGET = 168.0
    rng = np.random.RandomState(12345)
    n, d = 219, 6
    iu0, iu1 = np.triu_indices(n, 1)

    def _norm(X):
        q = np.linalg.norm(X, axis=1, keepdims=True)
        q = np.maximum(q, 1e-12)
        return X / q

    def _max_cos(X):
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        return float(np.max(G))

    def _per_max(X):
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        return np.max(G, axis=1)

    # load incumbent
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
    x_inc = x_best.copy()
    inc_val = best_val

    def _get_duals(res):
        try:
            il = getattr(res, "ineqlin", None)
            if il is not None:
                if hasattr(il, "marginals"):
                    return np.asarray(il.marginals, dtype=float)
                if isinstance(il, dict) and "marginals" in il:
                    return np.asarray(il["marginals"], dtype=float)
            # dict-style result
            try:
                dd = dict(res)
                if "ineqlin" in dd:
                    il2 = dd["ineqlin"]
                    if hasattr(il2, "marginals"):
                        return np.asarray(il2.marginals, dtype=float)
                    if isinstance(il2, dict) and "marginals" in il2:
                        return np.asarray(il2["marginals"], dtype=float)
            except Exception:
                pass
        except Exception:
            pass
        return None

    def lp_polish(x, radius=1e-3, window=None, topk=3000, get_dual=False, I_fix=None, J_fix=None, c_fix=None):
        # minimize t s.t. c_ij + xj.dxi + xi.dxj - t <= 0
        # vars [dx (n*d), t]
        try:
            Gf = x @ x.T
            c = Gf[iu0, iu1]
            m = float(np.max(c))
            if I_fix is None:
                if window is None:
                    window = max(20.0 * radius, 5e-4)
                sel = c > m - window
                I = iu0[sel]; J = iu1[sel]; cs = c[sel]
                if len(cs) > topk:
                    part = np.argpartition(-cs, topk - 1)[:topk]
                    I, J, cs = I[part], J[part], cs[part]
            else:
                I, J, cs = I_fix, J_fix, c_fix
                m = float(np.max(c_fix)) if len(c_fix) > 0 else float(np.max(c))
            p = len(I)
            if p == 0:
                return x, _max_cos(x), None
            nv = n * d + 1
            # A_ub: p x nv
            rows = np.repeat(np.arange(p), 2 * d + 1)
            cI = (I * d + np.arange(d)[None, :]).reshape(p, d)
            cJ = (J * d + np.arange(d)[None, :]).reshape(p, d)
            cols = np.concatenate([cI, cJ, np.full((p, 1), n * d)], axis=1).ravel()
            vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], axis=1).ravel()
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            b_ub = -cs  # c + lin - t <= 0  => lin - t <= -c
            # A_eq: x_i . dx_i = 0
            er = np.repeat(np.arange(n), d)
            ec = np.arange(n * d)
            A_eq = coo_matrix((x.ravel(), (er, ec)), shape=(n, nv)).tocsr()
            b_eq = np.zeros(n)
            cobj = np.zeros(nv)
            cobj[-1] = 1.0
            bounds = [(-radius, radius)] * (n * d) + [(m - 0.02, m + 0.02)]
            res = linprog(cobj, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
            if res.status != 0:
                return x, _max_cos(x), None
            dx = res.x[:n * d].reshape(n, d)
            y = _norm(x + dx)
            v = _max_cos(y)
            du = _get_duals(res) if get_dual else None
            if get_dual and du is not None:
                du = np.asarray(du).ravel()
                if du.size != p:
                    du = None
                else:
                    du = np.abs(du)
            # if dual requested but unavailable, proxy with softmax
            if get_dual and du is None:
                # proxy stress: exp weight
                bb = 2000.0
                du = np.exp(bb * (cs - m))
            return y, v, (du, I, J, cs, m) if get_dual else None
        except Exception:
            return x, _max_cos(x), None

    def run_egal(x0, beta, Wmat, Uvec, maxiter):
        y0 = x0.ravel().copy()
        # pre-zero diag handled via Wmat diag 0
        def fun(y):
            Y = y.reshape(n, d)
            rn = np.linalg.norm(Y, axis=1, keepdims=True)
            rn = np.maximum(rn, 1e-12)
            X = Y / rn
            G = X @ X.T
            np.fill_diagonal(G, -2.0)
            mi = np.max(G, axis=1)  # (n,)
            # E = W * exp(beta(G-mi))
            D = G - mi[:, None]
            # D<=0 (except diag very negative)
            E = Wmat * np.exp(beta * D)
            # avoid zero row sum
            S = np.sum(E, axis=1) + 1e-300  # (n,)
            loss_vec = mi + np.log(S) / beta
            loss = float(np.sum(Uvec * loss_vec))
            P = (Uvec[:, None] * E) / S[:, None]
            # symmetrize
            Ws = P + P.T
            GX = Ws @ X
            dot = np.sum(GX * X, axis=1, keepdims=True)
            GY = (GX - dot * X) / rn
            return loss, GY.ravel()
        try:
            res = minimize(fun, y0, jac=True, method="L-BFGS-B",
                           options={"maxiter": maxiter, "ftol": 1e-12, "gtol": 1e-7, "maxfun": 20000})
            Y = res.x.reshape(n, d)
            return _norm(Y)
        except Exception:
            return _norm(x0)

    # initial stress extraction on incumbent (window 1e-6)
    def extract_stress(x):
        try:
            Gf = x @ x.T
            c = Gf[iu0, iu1]
            m = float(np.max(c))
            sel = c > m - 1e-6
            I = iu0[sel]; J = iu1[sel]; cs = c[sel]
            if len(I) == 0:
                return None, m
            if len(I) > 3500:
                part = np.argpartition(-cs, 3500 - 1)[:3500]
                I, J, cs = I[part], J[part], cs[part]
            _, _, info = lp_polish(x, radius=0.004, get_dual=True, I_fix=I, J_fix=J, c_fix=cs)
            if info is None:
                return None, m
            du, II, JJ, cc, mm = info
            return (du, II, JJ, cc, mm), m
        except Exception:
            return None, _max_cos(x)

    stress_info, m0 = extract_stress(x_best)
    # build initial Wmat
    eps = 0.05
    Wmat = np.full((n, n), eps, dtype=float)
    np.fill_diagonal(Wmat, 0.0)
    zero_count = {}
    if stress_info is not None:
        du, II, JJ, cc, mm = stress_info
        mx = float(np.max(du)) if du.size else 1.0
        if mx < 1e-12:
            mx = 1.0
        for k in range(len(II)):
            i, j = int(II[k]), int(JJ[k])
            s = float(du[k]) / mx
            Wmat[i, j] = eps + s
            Wmat[j, i] = eps + s
            if s < 1e-3:
                zero_count[(min(i, j), max(i, j))] = 1
            else:
                zero_count[(min(i, j), max(i, j))] = 0

    # row weights based on degree (boost low-degree / rattler)
    def row_weights(x):
        pm = _per_max(x)
        mg = float(np.max(pm))
        # contacts within 1e-9 ~ use 1e-7 for robustness
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        deg = np.sum(G > mg - 1e-7, axis=1)
        U = np.ones(n)
        for i in range(n):
            if deg[i] <= 1:
                U[i] = 4.0
            elif deg[i] < 6:
                U[i] = 2.2
            elif deg[i] < 8:
                U[i] = 1.4
        # also boost current worst points slightly
        # normalize to mean 1 to keep beta scale stable
        U = U / (np.mean(U) + 1e-12)
        return U, deg, pm, mg

    betas = [100.0, 200.0, 400.0, 800.0, 1500.0, 3000.0]
    x_cur = x_best.copy()
    cur_val = best_val
    # light egal start from incumbent
    outer = 0
    # time allocation: loop over betas repeatedly with perturbations
    # alternate L-BFGS + LP polish, recompute stresses
    total_cycles = 0
    while time.time() - t0 < BUDGET - 18.0:
        outer += 1
        for bi, beta in enumerate(betas):
            if time.time() - t0 > BUDGET - 18.0:
                break
            Uvec, deg, pm, mg = row_weights(x_cur)
            # refresh Wmat active set: add pairs with slack <1e-4
            try:
                Gc = x_cur @ x_cur.T
                np.fill_diagonal(Gc, -2.0)
                mg2 = float(np.max(Gc))
                # add entering pairs
                sel_add = (Gc > mg2 - 1e-4)
                # set W for those at least eps+small if not already large
                # find indices where Wmat==eps but should be boosted slightly
                # keep cheap: for pairs with slack<1e-4 and Wmat<eps+1e-9, set to eps+0.15
                ii, jj = np.where(np.triu(sel_add, 1))
                for a, b in zip(ii.tolist(), jj.tolist()):
                    if Wmat[a, b] < eps + 1e-9:
                        Wmat[a, b] = eps + 0.15
                        Wmat[b, a] = eps + 0.15
            except Exception:
                pass
            mit = 120 if beta < 1000 else 80
            # 200-500 steps requested: use two chained calls to simulate more steps
            try:
                y = run_egal(x_cur, beta, Wmat, Uvec, maxiter=mit)
            except Exception:
                y = x_cur
            v = _max_cos(y)
            if v < best_val - 1e-12:
                best_val = v
                x_best = y.copy()
            # short LP polish
            try:
                yp, vp, _ = lp_polish(y, radius=8e-4, window=6e-3, topk=3000, get_dual=False)
                if vp < v + 2e-4:
                    y, v = yp, vp
                    if vp < best_val - 1e-12:
                        best_val = vp
                        x_best = yp.copy()
            except Exception:
                pass
            # accept move if not too bad (allow sideways to change basin)
            if v < cur_val + 0.0012:
                x_cur = y.copy()
                cur_val = v
            # periodically recompute stresses and drop zero-stress
            if (bi % 2 == 1) or beta >= 1500:
                si, _ = extract_stress(x_cur)
                if si is not None:
                    du2, I2, J2, c2, m2 = si
                    mx2 = float(np.max(du2)) if du2.size else 1.0
                    if mx2 < 1e-12:
                        mx2 = 1.0
                    # update Wmat
                    seen = set()
                    for k in range(len(I2)):
                        i, j = int(I2[k]), int(J2[k])
                        key = (min(i, j), max(i, j))
                        seen.add(key)
                        s = float(du2[k]) / mx2
                        if s < 1e-3:
                            zero_count[key] = zero_count.get(key, 0) + 1
                        else:
                            zero_count[key] = 0
                        if zero_count.get(key, 0) >= 2:
                            # drop: back to eps
                            Wmat[i, j] = eps
                            Wmat[j, i] = eps
                        else:
                            Wmat[i, j] = eps + s
                            Wmat[j, i] = eps + s
                    # pairs previously in zero_count but no longer active: decay
                    # (keep W as eps)
            total_cycles += 1
            if time.time() - t0 > BUDGET - 25.0:
                break
        # end beta sweep: topology-changing kick focused on rattler/hubs
        if time.time() - t0 > BUDGET - 18.0:
            break
        try:
            Uvec, deg, pm, mg = row_weights(x_cur)
            # rattler = argmin pm; hubs = argsort stress row sum or degree
            ratt = int(np.argmin(pm))
            # hub score: degree + per-max
            order_worst = np.argsort(-pm)
            hubs = order_worst[:6].tolist()
            # random tangent kick
            P = rng.randn(n, d)
            P -= (np.sum(P * x_cur, axis=1, keepdims=True)) * x_cur
            pn = np.linalg.norm(P, axis=1, keepdims=True) + 1e-12
            P = P / pn
            step = np.full((n, 1), 0.002)
            step[ratt] = 0.06 + 0.02 * rng.rand()
            for h in hubs[:3]:
                step[h] = max(float(step[h]), 0.025)
            # random subset medium kick to redistribute stress
            Q = rng.choice(n, size=14, replace=False)
            step[Q] = np.maximum(step[Q], 0.012)
            x_cur = _norm(x_cur + step * P)
            # quick egal low-beta to fall into different basin (few iters)
            Uvec2, _, _, _ = row_weights(x_cur)
            yk = run_egal(x_cur, 120.0, Wmat, Uvec2, maxiter=40)
            vk = _max_cos(yk)
            if vk < best_val - 1e-12:
                best_val = vk
                x_best = yk.copy()
            if vk < cur_val + 0.002:
                x_cur = yk
                cur_val = vk
            else:
                # if kick made it much worse, revert to best with small noise
                if rng.rand() < 0.4:
                    Pp = rng.randn(n, d)
                    Pp -= (np.sum(Pp * x_best, axis=1, keepdims=True)) * x_best
                    Pp /= (np.linalg.norm(Pp, axis=1, keepdims=True) + 1e-12)
                    x_cur = _norm(x_best + 0.006 * Pp)
                    cur_val = _max_cos(x_cur)
            # stall reset
            if cur_val > best_val + 0.003:
                x_cur = x_best.copy()
                cur_val = best_val
        except Exception:
            pass
        if outer > 12:
            break

    # final tie-down: LP polishes with shrinking radius on best
    try:
        x_f = x_best.copy()
        for rad in [1e-3, 5e-4, 2e-4, 1e-4]:
            if time.time() - t0 > BUDGET - 6.0:
                break
            y, v, _ = lp_polish(x_f, radius=rad, window=max(20 * rad, 8e-4), topk=3500, get_dual=False)
            if v < best_val - 1e-13:
                best_val = v
                x_f = y
            elif v < best_val + 1e-6:
                # keep sideways if degrees improve? just keep best
                pass
        # also one egal high-beta finish
        if time.time() - t0 < BUDGET - 8.0:
            Uf, _, _, _ = row_weights(x_f if 'x_f' in locals() else x_best)
            yb = run_egal(x_best, 3000.0, Wmat, Uf, maxiter=60)
            vb = _max_cos(yb)
            if vb < best_val:
                best_val = vb
                x_best = yb
            else:
                x_best = x_f
                best_val = _max_cos(x_best)
        else:
            x_best = x_f
            best_val = _max_cos(x_best)
    except Exception:
        pass

    # fallback
    try:
        if best_val > inc_val - 1e-12:
            # keep exact incumbent if none beats it (still return best which equals inc)
            x_best = x_inc.copy()
            best_val = inc_val
    except Exception:
        pass
    x_best = _norm(np.asarray(x_best, dtype=float))
    # final safety: replace non-finite
    if not np.all(np.isfinite(x_best)):
        x_best = _norm(np.asarray(x_inc, dtype=float))
    return x_best
