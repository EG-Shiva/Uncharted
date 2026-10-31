import numpy as np
import time

def solve():
    n = 26
    t0 = time.time()
    DEADLINE = 120.0  # leave margin for 130s limit

    # fallback feasible grid (parent-like) to guarantee validity
    k = int(np.ceil(np.sqrt(n)))
    r0 = 0.5 / k
    pts = []
    for i in range(k):
        for j in range(k):
            if len(pts) < n:
                pts.append([(2*i+1)*r0, (2*j+1)*r0, r0*0.999])
    best_arr = np.array(pts, dtype=float)
    # ensure feasible scaling for fallback
    def uniform_scale(X, Y, R):
        R = np.maximum(R, 1e-12)
        wall_ratio = np.minimum(np.minimum(X, 1.0-X), np.minimum(Y, 1.0-Y)) / R
        # pairwise
        Dx = X[:, None] - X[None, :]
        Dy = Y[:, None] - Y[None, :]
        D = np.sqrt(Dx*Dx + Dy*Dy)
        S = R[:, None] + R[None, :]
        # ignore diagonal
        with np.errstate(divide='ignore', invalid='ignore'):
            ratio = D / S
        np.fill_diagonal(ratio, np.inf)
        a = np.min(wall_ratio)
        b = np.min(ratio)
        alpha = min(1.0, a, b)
        alpha = alpha * (1.0 - 1e-7) - 1e-12
        if alpha <= 0 or not np.isfinite(alpha):
            alpha = 0.5
        return R * alpha, alpha

    Xb, Yb, Rb = best_arr[:,0], best_arr[:,1], best_arr[:,2]
    Rb_s, _ = uniform_scale(Xb, Yb, Rb)
    best_arr = np.column_stack([Xb, Yb, Rb_s])
    best_sum = float(np.sum(Rb_s))

    rng = np.random.default_rng(0)

    # loss+grad factory
    def make_fun(n_):
        def fun(v, W):
            X = v[0:n_]
            Y = v[n_:2*n_]
            R = v[2*n_:3*n_]
            loss = -np.sum(R)
            gx = np.zeros(n_)
            gy = np.zeros(n_)
            gr = -np.ones(n_)
            pen = 0.0
            # wall penalties
            v1 = R - X
            m1 = v1 > 0
            if np.any(m1):
                pen += np.sum(v1[m1]**2)
                gx[m1] += -2.0*v1[m1]
                gr[m1] += 2.0*v1[m1]
            v2 = X + R - 1.0
            m2 = v2 > 0
            if np.any(m2):
                pen += np.sum(v2[m2]**2)
                gx[m2] += 2.0*v2[m2]
                gr[m2] += 2.0*v2[m2]
            v3 = R - Y
            m3 = v3 > 0
            if np.any(m3):
                pen += np.sum(v3[m3]**2)
                gy[m3] += -2.0*v3[m3]
                gr[m3] += 2.0*v3[m3]
            v4 = Y + R - 1.0
            m4 = v4 > 0
            if np.any(m4):
                pen += np.sum(v4[m4]**2)
                gy[m4] += 2.0*v4[m4]
                gr[m4] += 2.0*v4[m4]
            # pairwise
            Dx = X[:, None] - X[None, :]
            Dy = Y[:, None] - Y[None, :]
            D = np.sqrt(Dx*Dx + Dy*Dy + 1e-20)
            S = R[:, None] + R[None, :]
            V = S - D
            np.fill_diagonal(V, 0.0)
            # need D diag not used; set to 1 to avoid div issues
            np.fill_diagonal(D, 1.0)
            Mask = V > 0
            if np.any(Mask):
                pen += 0.5*np.sum(V[Mask]**2)
                # Q = -2*V/D masked
                Q = np.zeros((n_, n_), dtype=float)
                Q[Mask] = -2.0*V[Mask]/D[Mask]
                Hx = Q*Dx
                Hy = Q*Dy
                gx += np.sum(Hx, axis=1)
                gy += np.sum(Hy, axis=1)
                G = np.zeros((n_, n_), dtype=float)
                G[Mask] = 2.0*V[Mask]
                gr += np.sum(G, axis=1)
            loss = loss + W*pen
            grad = np.concatenate([W*gx, W*gy, gr])
            # gr already includes -1 + W*part? careful: gr currently = -1 + grad_pen_R
            # gx,gy currently grad_pen only, so grad = [W*gx_pen, W*gy_pen, -1+W*gr_pen]
            # Above gx,gy,gr: gx,gy are pen grads, gr = -1 + pen grad. So:
            # we did gx+=pen, gr=-1+pen. So final:
            # grad_x = W*gx, grad_y=W*gy, grad_r = gr_pen*W -1
            # Our gr = -1 + pen_grad, so need to convert:
            # Actually we added pen contributions to gr on top of -1, so gr = -1+pen_grad.
            # Then final grad_r should be -1 + W*pen_grad = -1 + W*(gr+1) = W*gr + (W-1)*(-?) Let's fix:
            # Simpler: reconstruct
            # We have pen_grad_r = gr + 1 (since gr=-1+pen). So final = -1 + W*(gr+1)
            # Instead recompute cleanly:
            # Let's just adjust:
            # grad vector built as [W*gx, W*gy, -1+W*(gr+1)]
            # Implement:
            g = np.empty_like(v)
            g[0:n_] = W*gx
            g[n_:2*n_] = W*gy
            # gr = -1 + pen_grad  => pen_grad = gr+1
            g[2*n_:3*n_] = -1.0 + W*(gr + 1.0)
            return loss, g
        return fun

    fun = make_fun(n)
    from scipy.optimize import minimize, linprog

    Ws = [1e2, 1e3, 1e4, 1e5, 1e6]
    bounds = [(0.0, 1.0)]*(2*n) + [(0.03, 0.20)]*n

    restart = 0
    max_restarts = 200
    while restart < max_restarts:
        if time.time() - t0 > DEADLINE - 8:
            break
        # init
        if restart == 0:
            # perturbed 5x5 + one extra
            X0 = np.zeros(n); Y0 = np.zeros(n); R0 = np.full(n, 0.095)
            idx = 0
            for i in range(5):
                for j in range(5):
                    X0[idx] = (i+0.5)/5.0 + rng.normal(0, 0.005)
                    Y0[idx] = (j+0.5)/5.0 + rng.normal(0, 0.005)
                    idx += 1
            X0[25] = 0.9 + rng.normal(0, 0.01)
            Y0[25] = 0.1 + rng.normal(0, 0.01)
            X0 = np.clip(X0, 0.05, 0.95); Y0 = np.clip(Y0, 0.05, 0.95)
        elif restart == 1:
            # hex-like grid 6x5 subset
            X0 = np.zeros(n); Y0 = np.zeros(n)
            R0 = np.full(n, 0.09)
            pts_l = []
            for ix in range(6):
                for iy in range(5):
                    if len(pts_l) < n:
                        xx = (ix+0.5)/6.0
                        yy = (iy+0.5)/5.0
                        if iy % 2 == 1:
                            xx += 0.5/6.0*0.5
                            if xx > 0.95: xx -= 1.0/6.0
                        pts_l.append((xx, yy))
            arr = np.array(pts_l)
            X0 = arr[:,0] + rng.normal(0, 0.005, n)
            Y0 = arr[:,1] + rng.normal(0, 0.005, n)
            X0 = np.clip(X0, 0.05, 0.95); Y0 = np.clip(Y0, 0.05, 0.95)
            R0 = 0.085 + 0.02*rng.random(n)
        else:
            X0 = rng.random(n)*0.9+0.05
            Y0 = rng.random(n)*0.9+0.05
            R0 = 0.085 + 0.025*rng.random(n)
        v = np.concatenate([X0, Y0, R0])
        # annealed LBFGS
        for W in Ws:
            if time.time() - t0 > DEADLINE - 5:
                break
            try:
                res = minimize(fun, v, args=(W,), jac=True, method='L-BFGS-B',
                               bounds=bounds,
                               options={'maxiter': 300, 'ftol': 1e-9, 'gtol': 1e-7})
                v = res.x
            except Exception:
                break
        X = np.clip(v[0:n], 0.0, 1.0)
        Y = np.clip(v[n:2*n], 0.0, 1.0)
        R = np.clip(v[2*n:3*n], 0.01, 0.25)
        # candidate via uniform scale (safety)
        try:
            Rs, _ = uniform_scale(X, Y, R)
            s_uni = float(np.sum(Rs))
            if s_uni > best_sum:
                best_sum = s_uni
                best_arr = np.column_stack([X, Y, Rs])
        except Exception:
            pass
        # LP polish for optimal radii given centres
        try:
            wall = np.minimum(np.minimum(X, 1.0-X), np.minimum(Y, 1.0-Y))
            wall = np.maximum(wall, 0.0)
            m = n*(n-1)//2
            A = np.zeros((m, n), dtype=float)
            b = np.zeros(m, dtype=float)
            kk = 0
            for i in range(n):
                xi = X[i]; yi = Y[i]
                for j in range(i+1, n):
                    dx = xi - X[j]; dy = yi - Y[j]
                    A[kk, i] = 1.0; A[kk, j] = 1.0
                    b[kk] = np.sqrt(dx*dx + dy*dy)
                    kk += 1
            c = -np.ones(n)
            bnds = [(0.0, float(w)) for w in wall]
            rlp = linprog(c, A_ub=A, b_ub=b, bounds=bnds, method='highs')
            if rlp.success:
                Ropt = np.array(rlp.x, dtype=float)
                Ropt = np.maximum(Ropt, 0.0) * (1.0 - 1e-7)
                # tiny absolute guard
                Ropt = np.maximum(Ropt - 1e-13, 0.0)
                # verify quickly
                s_lp = float(np.sum(Ropt))
                if s_lp > best_sum:
                    # double-check feasibility with margin
                    ok = True
                    for i in range(n):
                        if not (Ropt[i] <= X[i]+1e-12 and Ropt[i] <= 1-X[i]+1e-12 and Ropt[i] <= Y[i]+1e-12 and Ropt[i] <= 1-Y[i]+1e-12):
                            # allow 1e-9 slack due to shrink
                            if Ropt[i] > X[i]+1e-9 or Ropt[i] > 1-X[i]+1e-9 or Ropt[i] > Y[i]+1e-9 or Ropt[i] > 1-Y[i]+1e-9:
                                ok = False
                    if ok:
                        best_sum = s_lp
                        best_arr = np.column_stack([X, Y, Ropt])
        except Exception:
            pass
        restart += 1

    # final safety shrink to guarantee strict feasibility 1e-12
    try:
        Xf = best_arr[:,0]; Yf = best_arr[:,1]; Rf = best_arr[:,2]
        Rf_s, _ = uniform_scale(Xf, Yf, Rf)
        # uniform_scale already includes 1e-7 shrink; ensure positive
        best_arr = np.column_stack([Xf, Yf, Rf_s])
    except Exception:
        pass
    return best_arr
