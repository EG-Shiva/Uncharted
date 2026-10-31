import numpy as np
from scipy.optimize import minimize, linprog
import time

def solve():
    np.random.seed(0)
    n = 26
    r_lo = 0.02
    r_hi = 0.18
    t_start = time.time()
    TIME_BUDGET = 115.0

    def sigmoid(t):
        t = np.clip(t, -15.0, 15.0)
        return 1.0 / (1.0 + np.exp(-t))

    def decode(p):
        A = p[0:n]
        B = p[n:2*n]
        C = p[2*n:3*n]
        rs = sigmoid(C)
        R = r_lo + (r_hi - r_lo) * rs
        sx = sigmoid(A)
        sy = sigmoid(B)
        # x = r + (1-2r)*sx  guarantees r<=x<=1-r
        X = R + (1.0 - 2.0 * R) * sx
        Y = R + (1.0 - 2.0 * R) * sy
        return X, Y, R, sx, sy, rs

    def encode(Xd, Yd, Rd):
        Rd = np.clip(Rd, r_lo + 1e-6, r_hi - 1e-6)
        rs = (Rd - r_lo) / (r_hi - r_lo)
        rs = np.clip(rs, 0.02, 0.98)
        C = np.log(rs / (1.0 - rs))
        sx = (Xd - Rd) / (1.0 - 2.0 * Rd)
        sy = (Yd - Rd) / (1.0 - 2.0 * Rd)
        sx = np.clip(sx, 0.02, 0.98)
        sy = np.clip(sy, 0.02, 0.98)
        A = np.log(sx / (1.0 - sx))
        B = np.log(sy / (1.0 - sy))
        return np.concatenate([A, B, C])

    def make_fun(P):
        def fun(p):
            X, Y, R, sx, sy, rs = decode(p)
            # pairwise
            DX = X[:, None] - X[None, :]
            DY = Y[:, None] - Y[None, :]
            D = np.sqrt(DX**2 + DY**2)
            # avoid div by zero on diagonal
            Dsafe = np.where(np.eye(n, dtype=bool), 1.0, np.maximum(D, 1e-12))
            S = R[:, None] + R[None, :]
            V = S - D
            V = np.maximum(V, 0.0)
            np.fill_diagonal(V, 0.0)
            pen = np.sum(V**2) * 0.5  # each pair twice -> /2
            F = -np.sum(R) + P * pen
            # gradients w.r.t X,Y,R
            # Gr = -1 + P*sum_j 2V_ij
            Gr = -np.ones(n) + P * np.sum(2.0 * V, axis=1)
            # Gx = P*sum_j 2V*(-DX/D)
            W = np.zeros((n, n))
            mask = V > 0
            W[mask] = 2.0 * V[mask] / Dsafe[mask]
            # Gx_i = P*sum_j W_ij*(-DX_ij)
            Gx = P * np.sum(W * (-DX), axis=1)
            Gy = P * np.sum(W * (-DY), axis=1)
            # chain to A,B,C
            drdc = (r_hi - r_lo) * rs * (1.0 - rs)
            dxda = (1.0 - 2.0 * R) * sx * (1.0 - sx)
            dydb = (1.0 - 2.0 * R) * sy * (1.0 - sy)
            dxdr = (1.0 - 2.0 * sx)
            dydr = (1.0 - 2.0 * sy)
            dFda = Gx * dxda
            dFdb = Gy * dydb
            dFdc = Gr * drdc + Gx * dxdr * drdc + Gy * dydr * drdc
            g = np.concatenate([dFda, dFdb, dFdc])
            return F, g
        return fun

    def feasible_shrink(X, Y, R):
        # returns scaling factor and shrunk radii
        DX = X[:, None] - X[None, :]
        DY = Y[:, None] - Y[None, :]
        D = np.sqrt(DX**2 + DY**2)
        b = np.minimum(np.minimum(X, 1.0 - X), np.minimum(Y, 1.0 - Y))
        # border factor
        fb = np.min(b / np.maximum(R, 1e-12))
        fp = 1e9
        for i in range(n):
            for j in range(i+1, n):
                denom = R[i] + R[j]
                if denom <= 0:
                    continue
                fp = min(fp, D[i, j] / denom)
        f = min(1.0, fb, fp)
        # safety margin for strict 1e-12 check
        f_safe = f * (1.0 - 1e-7) - 1e-12
        if f_safe <= 0:
            f_safe = f * 0.99
        return f_safe, R * f_safe, b, D

    def lp_polish(X, Y):
        try:
            DX = X[:, None] - X[None, :]
            DY = Y[:, None] - Y[None, :]
            D = np.sqrt(DX**2 + DY**2)
            b = np.minimum(np.minimum(X, 1.0 - X), np.minimum(Y, 1.0 - Y))
            lb = 1e-5
            if np.any(b < lb + 1e-12):
                return None
            m = n*(n-1)//2
            A_ub = np.zeros((m, n))
            b_ub = np.zeros(m)
            k = 0
            for i in range(n):
                for j in range(i+1, n):
                    A_ub[k, i] = 1.0
                    A_ub[k, j] = 1.0
                    b_ub[k] = D[i, j]
                    k += 1
            # quick infeasibility: if any D < 2*lb -> infeasible
            if np.any(b_ub < 2*lb - 1e-12):
                return None
            c = -np.ones(n)
            bounds = [(lb, float(bi)) for bi in b]
            res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
            if not res.success:
                return None
            return res.x
        except Exception:
            return None

    best_sum = -1.0
    best_arr = None

    def consider(X, Y, R):
        nonlocal best_sum, best_arr
        f_safe, Rs, _, _ = feasible_shrink(X, Y, R)
        s = float(np.sum(Rs))
        if s > best_sum:
            best_sum = s
            best_arr = np.column_stack([X, Y, Rs])

    # build initial guesses list on the fly
    n_restarts = 40
    P_list = [100.0, 1000.0, 10000.0]

    for restart in range(n_restarts):
        if time.time() - t_start > TIME_BUDGET:
            break
        # --- init centres ---
        if restart == 0:
            # 6x5 grid first 26
            xs = (np.arange(6) + 0.5) / 6.0
            ys = (np.arange(5) + 0.5) / 5.0
            X0 = []
            Y0 = []
            for yy in ys:
                for xx in xs:
                    X0.append(xx)
                    Y0.append(yy)
                    if len(X0) >= n:
                        break
                if len(X0) >= n:
                    break
            X0 = np.array(X0)
            Y0 = np.array(Y0)
            R0 = np.full(n, 0.10)
            # small deterministic jitter
            rs = np.random.RandomState(restart)
            X0 += rs.randn(n) * 0.005
            Y0 += rs.randn(n) * 0.005
            X0 = np.clip(X0, 0.06, 0.94)
            Y0 = np.clip(Y0, 0.06, 0.94)
            p = encode(X0, Y0, R0)
        elif restart == 1:
            # 5x5 + extra staggered
            xs5 = (np.arange(5) + 0.5) / 5.0
            ys5 = (np.arange(5) + 0.5) / 5.0
            X0 = []
            Y0 = []
            for yy in ys5:
                for xx in xs5:
                    X0.append(xx)
                    Y0.append(yy)
            X0 = np.array(X0[:25])
            Y0 = np.array(Y0[:25])
            # 26th in centre of a cell
            X0 = np.append(X0, 0.5)
            Y0 = np.append(Y0, 0.4)
            rs = np.random.RandomState(restart)
            X0 += rs.randn(n) * 0.01
            Y0 += rs.randn(n) * 0.01
            X0 = np.clip(X0, 0.06, 0.94)
            Y0 = np.clip(Y0, 0.06, 0.94)
            R0 = np.full(n, 0.10) + rs.randn(n) * 0.005
            R0 = np.clip(R0, 0.06, 0.13)
            p = encode(X0, Y0, R0)
        else:
            rs = np.random.RandomState(1000 + restart)
            X0 = rs.uniform(0.07, 0.93, size=n)
            Y0 = rs.uniform(0.07, 0.93, size=n)
            # hex-ish perturbation for some restarts
            if restart % 3 == 0:
                # jittered 5x6 hex
                nx, ny = 5, 6
                pts = []
                for iy in range(ny):
                    for ix in range(nx):
                        xx = (ix + 0.5 + (0.5 if iy % 2 == 1 else 0.0)) / (nx + 0.5)
                        yy = (iy + 0.5) / ny
                        pts.append((xx, yy))
                        if len(pts) >= n:
                            break
                    if len(pts) >= n:
                        break
                pts = np.array(pts)
                X0 = pts[:, 0] + rs.randn(n) * 0.015
                Y0 = pts[:, 1] + rs.randn(n) * 0.015
                X0 = np.clip(X0, 0.06, 0.94)
                Y0 = np.clip(Y0, 0.06, 0.94)
            R0 = 0.09 + rs.rand(n) * 0.04  # 0.09-0.13
            p = encode(X0, Y0, R0)

        # --- annealed penalty optimisation ---
        cur = p
        for P in P_list:
            if time.time() - t_start > TIME_BUDGET:
                break
            fun = make_fun(P)
            try:
                res = minimize(fun, cur, jac=True, method='L-BFGS-B',
                               options={'maxiter': 250, 'ftol': 1e-10, 'gtol': 1e-6})
                cur = res.x
            except Exception:
                break
        X, Y, R, _, _, _ = decode(cur)
        # evaluate raw (shrunk)
        consider(X, Y, R)
        # LP polish for these centres
        if time.time() - t_start > TIME_BUDGET:
            break
        Rlp = lp_polish(X, Y)
        if Rlp is not None:
            consider(X, Y, Rlp)
            # one more local move: re-encode polished radii and continue a bit
            # small random kick to centres then re-opt at high P
            try:
                if restart % 2 == 0 and (time.time() - t_start < TIME_BUDGET - 5):
                    rs2 = np.random.RandomState(2000 + restart)
                    Xk = X + rs2.randn(n) * 0.003
                    Yk = Y + rs2.randn(n) * 0.003
                    # keep inside
                    Xk = np.clip(Xk, Rlp + 1e-6, 1.0 - Rlp - 1e-6)
                    Yk = np.clip(Yk, Rlp + 1e-6, 1.0 - Rlp - 1e-6)
                    cur2 = encode(Xk, Yk, np.clip(Rlp, r_lo+1e-6, r_hi-1e-6))
                    funH = make_fun(10000.0)
                    res2 = minimize(funH, cur2, jac=True, method='L-BFGS-B',
                                    options={'maxiter': 150, 'ftol': 1e-10, 'gtol': 1e-6})
                    X2, Y2, R2, _, _, _ = decode(res2.x)
                    consider(X2, Y2, R2)
                    Rlp2 = lp_polish(X2, Y2)
                    if Rlp2 is not None:
                        consider(X2, Y2, Rlp2)
            except Exception:
                pass

    if best_arr is None:
        # fallback grid
        k = int(np.ceil(np.sqrt(n)))
        r = 0.5 / k
        pts = []
        for i in range(k):
            for j in range(k):
                if len(pts) < n:
                    pts.append([(2*i+1)*r, (2*j+1)*r, r*0.999])
        return np.array(pts)
    # final strict safety: already shrunk, add one more tiny shrink
    best_arr[:, 2] *= (1.0 - 1e-9)
    # ensure >0 and inside
    best_arr[:, 2] = np.maximum(best_arr[:, 2], 1e-6)
    return best_arr
