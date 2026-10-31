import numpy as np
import time

def solve():
    import numpy as np
    import time
    from scipy.optimize import minimize, linprog
    N = 26
    t_start = time.time()
    TIME_LIMIT = 118.0
    rng = np.random.RandomState(0)

    iu = np.triu_indices(N, 1)
    M = len(iu[0])
    # pattern for A_ub: each row has two ones
    # precompute for fast build
    pairs = list(zip(iu[0].tolist(), iu[1].tolist()))

    def lp_radii(xc, yc):
        # clip
        xc = np.clip(xc, 0.0, 1.0)
        yc = np.clip(yc, 0.0, 1.0)
        b = np.minimum(np.minimum(xc, 1.0 - xc), np.minimum(yc, 1.0 - yc))
        # if outside, b negative -> infeasible, return None
        if np.any(b < -1e-12):
            return None, -1.0
        b = np.maximum(b, 0.0)
        dx = xc[:, None] - xc[None, :]
        dy = yc[:, None] - yc[None, :]
        D = np.sqrt(dx * dx + dy * dy)
        dvec = D[iu]
        # build A_ub
        # dense M x N
        A = np.zeros((M, N), dtype=float)
        # vectorised fill using advanced indexing
        rows = np.arange(M)
        A[rows, iu[0]] = 1.0
        A[rows, iu[1]] = 1.0
        c = -np.ones(N)
        bounds = [(0.0, float(bi)) for bi in b]
        try:
            res = linprog(c, A_ub=A, b_ub=dvec, bounds=bounds, method='highs')
        except Exception:
            return None, -1.0
        if not res.success:
            return None, -1.0
        r = np.asarray(res.x).ravel()
        # clean tiny negatives
        r = np.maximum(r, 0.0)
        s = float(np.sum(r))
        return r, s

    # penalty objective with analytic grad
    def make_fun(W):
        def fun_grad(p):
            x = p[0:N]
            y = p[N:2*N]
            r = p[2*N:3*N]
            # border
            e1 = r - x
            e2 = x + r - 1.0
            e3 = r - y
            e4 = y + r - 1.0
            v1 = np.maximum(0.0, e1)
            v2 = np.maximum(0.0, e2)
            v3 = np.maximum(0.0, e3)
            v4 = np.maximum(0.0, e4)
            loss_b = np.sum(v1*v1 + v2*v2 + v3*v3 + v4*v4)
            # overlap matrix
            Dx = x[:, None] - x[None, :]
            Dy = y[:, None] - y[None, :]
            D = np.sqrt(Dx*Dx + Dy*Dy + 1e-16)
            E = r[:, None] + r[None, :] - D
            # zero diagonal
            # E diag = 2r -1e-8 >0 would create self penalty, remove
            np.fill_diagonal(E, -1.0)
            V = np.maximum(0.0, E)
            # loss overlap sum_{i<j} V^2 = 0.5*sum V^2 (diag zero)
            loss_o = 0.5 * np.sum(V*V)
            loss = -np.sum(r) + W * (loss_b + loss_o)

            # grads
            gx = -2.0*v1 + 2.0*v2
            gy = -2.0*v3 + 2.0*v4
            gr = (2.0*v1 + 2.0*v2 + 2.0*v3 + 2.0*v4) - 1.0

            # overlap grads
            # InvD zero diag
            InvD = 1.0 / (D + 1e-12)
            np.fill_diagonal(InvD, 0.0)
            # V already zero diag
            # common term
            # gx_o = 2 * sum_j V*(-Dx*InvD)
            Tx = V * (Dx * InvD)
            Ty = V * (Dy * InvD)
            gx_o = -2.0 * np.sum(Tx, axis=1)
            gy_o = -2.0 * np.sum(Ty, axis=1)
            gr_o = 2.0 * np.sum(V, axis=1)
            gx = gx + W * gx_o if False else gx  # placeholder
            # actually border grad already includes W factor? need multiply
            # recompute correctly: border part *W, overlap part *W
            # gx currently is unweighted border derivative, gr includes -1
            # split:
            # border derivatives:
            gx_b = -2.0*v1 + 2.0*v2
            gy_b = -2.0*v3 + 2.0*v4
            gr_b = 2.0*v1 + 2.0*v2 + 2.0*v3 + 2.0*v4
            gx_full = W * (gx_b + gx_o)
            gy_full = W * (gy_b + gy_o)
            gr_full = -np.ones(N) + W * (gr_b + gr_o)
            g = np.concatenate([gx_full, gy_full, gr_full])
            return loss, g
        return fun_grad

    def run_lbfgs(p0, W, maxiter):
        fg = make_fun(W)
        bounds = [(0.0, 1.0)]*(2*N) + [(0.008, 0.30)]*N
        def f(p):
            l, g = fg(p)
            return l, g
        try:
            res = minimize(f, p0, method='L-BFGS-B', jac=True, bounds=bounds,
                           options={'maxiter': maxiter, 'ftol': 1e-9, 'gtol': 1e-6})
            return res.x
        except Exception:
            return p0

    best_sum = -1.0
    best_xc = None
    best_yc = None
    best_r = None

    def try_update(xc, yc):
        nonlocal best_sum, best_xc, best_yc, best_r
        r, s = lp_radii(xc, yc)
        if r is None:
            return False
        if s > best_sum:
            best_sum = s
            best_xc = xc.copy()
            best_yc = yc.copy()
            best_r = r.copy()
            return True
        return False

    n_restart = 10
    for ri in range(n_restart):
        if time.time() - t_start > 55.0:
            break
        if ri == 0:
            # jittered 5x5 grid + 1
            xs = []
            ys = []
            for ix in range(5):
                for iy in range(5):
                    xs.append((ix + 0.5) / 5.0 + rng.randn()*0.01)
                    ys.append((iy + 0.5) / 5.0 + rng.randn()*0.01)
            xs.append(0.85 + rng.randn()*0.02)
            ys.append(0.25 + rng.randn()*0.02)
            xc0 = np.clip(np.array(xs), 0.05, 0.95)
            yc0 = np.clip(np.array(ys), 0.05, 0.95)
            r0 = np.full(N, 0.09) + rng.randn(N)*0.01
            r0 = np.clip(r0, 0.02, 0.15)
        else:
            xc0 = rng.uniform(0.05, 0.95, size=N)
            yc0 = rng.uniform(0.05, 0.95, size=N)
            r0 = rng.uniform(0.04, 0.12, size=N)
        p0 = np.concatenate([xc0, yc0, r0])
        # stage 1 moderate penalty
        p1 = run_lbfgs(p0, W=2000.0, maxiter=250)
        # stage 2 strong penalty
        p2 = run_lbfgs(p1, W=20000.0, maxiter=250)
        xc = np.clip(p2[0:N], 0.0, 1.0)
        yc = np.clip(p2[N:2*N], 0.0, 1.0)
        try_update(xc, yc)

    if best_xc is None:
        # fallback grid
        k = 6
        r = 0.5 / k
        pts = []
        for i in range(k):
            for j in range(k):
                if len(pts) < N:
                    pts.append([(2*i+1)*r, (2*j+1)*r, r*0.999])
        return np.array(pts)

    # hill climbing with LP evaluations
    # sigma schedule
    n_iter = 6000
    sigma0 = 0.06
    sigma1 = 0.003
    cur_x = best_xc.copy()
    cur_y = best_yc.copy()
    # local rng for proposals
    for it in range(n_iter):
        if time.time() - t_start > TIME_LIMIT:
            break
        frac = it / float(n_iter)
        sigma = sigma0 * (1.0 - frac) + sigma1 * frac
        # perturb 1 or 2 circles
        if rng.rand() < 0.8:
            idx = [rng.randint(N)]
        else:
            idx = [rng.randint(N), rng.randint(N)]
        prop_x = cur_x.copy()
        prop_y = cur_y.copy()
        for j in idx:
            prop_x[j] += rng.randn() * sigma
            prop_y[j] += rng.randn() * sigma
        prop_x = np.clip(prop_x, 0.0, 1.0)
        prop_y = np.clip(prop_y, 0.0, 1.0)
        # quick bound check: need positive border
        # evaluate LP
        rprop, sprop = lp_radii(prop_x, prop_y)
        if rprop is None:
            continue
        if sprop > best_sum + 1e-12:
            # accept as best
            best_sum = sprop
            best_xc = prop_x.copy()
            best_yc = prop_y.copy()
            best_r = rprop.copy()
            cur_x = prop_x
            cur_y = prop_y
        elif sprop > float(np.sum(lp_radii(cur_x, cur_y)[1] if False else -1e9)):
            # Metropolis-like: accept if better than current (need current sum)
            pass
        # also random walk: with small prob accept non-best to explore?
        # simple: if better than current, move current
        # compute current sum quickly? we maintain cur_sum
        # maintain cur_sum variable
        # To avoid extra LP, keep cur_sum
        # (implement cur_sum tracking)
        # For simplicity, if sprop improves over best*0.995, move current with prob
        # We'll just move current if sprop is close: allow exploration
        # Use greedy for current as well:
        # We need cur_sum; let's track
        # (initialise before loop - patch)
        # Instead do: if sprop > cur_sum, move
        # We don't have cur_sum, compute via best? Let's maintain.
        # Workaround: keep cur_sum variable
        pass

    # Second hill climbing loop with proper current tracking (fast)
    cur_x = best_xc.copy()
    cur_y = best_yc.copy()
    _, cur_sum = lp_radii(cur_x, cur_y)
    if cur_sum is None or cur_sum < 0:
        cur_sum = best_sum
    for it in range(n_iter):
        if time.time() - t_start > TIME_LIMIT:
            break
        frac = it / float(n_iter)
        sigma = sigma0 * (1.0 - frac) + sigma1 * frac
        idx = rng.randint(N)
        prop_x = cur_x.copy()
        prop_y = cur_y.copy()
        prop_x[idx] += rng.randn() * sigma
        prop_y[idx] += rng.randn() * sigma
        prop_x[idx] = np.clip(prop_x[idx], 0.0, 1.0)
        prop_y[idx] = np.clip(prop_y[idx], 0.0, 1.0)
        rprop, sprop = lp_radii(prop_x, prop_y)
        if rprop is None:
            continue
        if sprop > cur_sum:
            cur_x = prop_x
            cur_y = prop_y
            cur_sum = sprop
            if sprop > best_sum:
                best_sum = sprop
                best_xc = prop_x.copy()
                best_yc = prop_y.copy()
                best_r = rprop.copy()
        elif rng.rand() < 0.02:
            # occasional random move to escape (only if not too bad)
            if sprop > cur_sum - 0.02:
                cur_x = prop_x
                cur_y = prop_y
                cur_sum = sprop

    # final safety shrink
    xc = np.clip(best_xc, 0.0, 1.0)
    yc = np.clip(best_yc, 0.0, 1.0)
    r = np.maximum(np.asarray(best_r).ravel(), 1e-9)
    # ensure feasibility by scaling check
    # compute required scale due to numerics: shrink by 1-1e-7
    shrink = 1.0 - 5e-7
    r = r * shrink
    # guarantee r>0 and inside: if any b < r due to numerics, adjust
    b = np.minimum(np.minimum(xc, 1.0 - xc), np.minimum(yc, 1.0 - yc))
    # if any r > b, scale that r down
    for i in range(N):
        if r[i] > b[i]:
            r[i] = b[i] * (1.0 - 1e-9)
            if r[i] <= 0:
                r[i] = 1e-9
                # nudge centre inside
                xc[i] = np.clip(xc[i], 2e-9, 1.0 - 2e-9)
                yc[i] = np.clip(yc[i], 2e-9, 1.0 - 2e-9)
    # pairwise exact fix: iteratively reduce overlapping radii by tiny factor
    # single global factor to resolve any residual overlap
    dx = xc[:, None] - xc[None, :]
    dy = yc[:, None] - yc[None, :]
    D = np.sqrt(dx*dx + dy*dy) + np.eye(N)
    # find max violation ratio
    max_ratio = 1.0
    for i in range(N):
        for j in range(i+1, N):
            d = float(np.sqrt((xc[i]-xc[j])**2 + (yc[i]-yc[j])**2))
            if d < 1e-12:
                continue
            need = r[i] + r[j]
            if need > d:
                ratio = d / need
                if ratio < max_ratio:
                    max_ratio = ratio
    if max_ratio < 1.0:
        r = r * (max_ratio * (1.0 - 1e-9))
    # ensure >0
    r = np.maximum(r, 1e-7)
    out = np.column_stack([xc, yc, r])
    return out
