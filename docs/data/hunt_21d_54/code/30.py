import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _proj(x):
    nrm = np.linalg.norm(x, axis=1, keepdims=True)
    nrm[nrm == 0] = 1.0
    return x / nrm

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _lp_polish(x, seconds):
    x = _proj(x)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        G = x @ x.T
        c = G[iu]
        m = float(c.max())
        sel = c > m - max(20 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        if p > 2500:
            idx = np.argpartition(-c[sel], 2500)[:2500]
            I, J = I[idx], J[idx]
            csv = c[sel][idx]
            p = 2500
        else:
            csv = c[sel]
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))),
                          shape=(n, nv)).tocsr()
        obj = np.zeros(nv)
        obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-csv, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                          method="highs", options={"time_limit": 3.0})
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y = _proj(y)
        v = _max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.02)
        else:
            radius *= 0.5
    return x

def solve():
    T0 = time.time()
    LIMIT = 168.0
    SUCCESS = 0.0666634  # need strictly below; LP will squeeze further
    rng = np.random.default_rng(7)
    try:
        b0 = np.load("best.npy")
    except Exception:
        b0 = np.load("record.npy")
    b0 = _proj(np.nan_to_num(np.asarray(b0, dtype=float)))
    if b0.shape != (54, 21):
        b0 = _proj(rng.normal(size=(54, 21)))
    try:
        r0 = _proj(np.asarray(np.load("record.npy"), dtype=float))
        if r0.shape == (54, 21):
            if _max_cos(r0) < _max_cos(b0):
                b0 = r0.copy()
    except Exception:
        pass
    n, d = 54, 21
    global_best = b0.copy()
    global_val = _max_cos(global_best)
    success_x = None

    targets = [0.066, 0.064, 0.062, 0.060]
    alpha = 5e-4
    tau = 0.008
    b1, b2, eps = 0.9, 0.999, 1e-8
    lr0 = 0.01

    def time_left():
        return LIMIT - (time.time() - T0)

    # reserve for LP if success, else use almost all for homotopy
    HOMO_DEADLINE = 138.0  # seconds after T0 to stop homotopy if no success
    start_t = time.time()

    restart = 0
    # precompute mask for upper diag
    iu = np.triu_indices(n, 1)

    while (time.time() - T0) < HOMO_DEADLINE and time_left() > 12:
        restart += 1
        # --- build start point ---
        if restart == 1:
            x0 = global_best.copy()
        elif restart <= 6:
            sigmas = [0.02, 0.05, 0.12, 0.25, 0.01]
            sg = sigmas[(restart - 2) % len(sigmas)]
            y = global_best.copy()
            nz = rng.normal(size=(n, d))
            nz -= (nz * y).sum(axis=1, keepdims=True) * y
            y = y + sg * nz
            x0 = _proj(y)
        else:
            if restart % 3 == 0:
                x0 = _proj(rng.normal(size=(n, d)))
            else:
                sg = float(rng.uniform(0.03, 0.22))
                y = global_best.copy()
                nz = rng.normal(size=(n, d))
                nz -= (nz * y).sum(axis=1, keepdims=True) * y
                y = y + sg * nz
                x0 = _proj(y)

        X = x0.copy()
        m_ad = np.zeros((n, d))
        v_ad = np.zeros((n, d))
        cur = 0
        # steps per restart adapts to time
        max_steps = 4000
        if restart > 10:
            max_steps = 3000
        # inner Adam
        for step in range(1, max_steps + 1):
            # hard time check every 200 steps
            if step % 200 == 0:
                if (time.time() - T0) > HOMO_DEADLINE or time_left() < 12:
                    break
            t = targets[cur]
            G = X @ X.T
            np.fill_diagonal(G, -2.0)
            cur_max = float(G.max())
            # track global best cheaply every 100 steps + at end of chunk
            if step % 100 == 0 or step == max_steps:
                if cur_max < global_val:
                    global_val = cur_max
                    global_best = X.copy()
                if cur_max < SUCCESS:
                    success_x = X.copy()
                    break
            # feasibility -> anneal downward
            if cur_max <= t + 2e-12 and cur < len(targets) - 1:
                cur += 1
                m_ad[:] = 0.0
                v_ad[:] = 0.0
                continue
            # build symmetric weight S
            # penalty part
            # use only upper then symmetrize for speed; compute full for simplicity
            R = G - t
            np.fill_diagonal(R, 0.0)
            # relu
            np.maximum(R, 0.0, out=R)
            # softmax part: exp((G-m)/tau) upper
            # compute E only upper to save? full is fine (54x54)
            # E max is 0 at max pair
            E = (G - cur_max) / tau
            # clip low end
            np.clip(E, -20.0, 0.0, out=E)
            np.fill_diagonal(E, -20.0)
            W = np.exp(E)
            # zero lower triangle double count? we want symmetric softmax over all ordered pairs;
            # using full symmetric W is fine up to factor 2.
            s = W.sum() + 1e-30
            W /= s
            # S = 2*R + alpha*W  (R already relu, zero diag; W zero-ish diag)
            S = 2.0 * R + alpha * W
            # gradient
            grad = S @ X
            # tangent projection
            dot = np.einsum('ij,ij->i', grad, X)
            grad -= dot[:, None] * X
            # Adam
            m_ad = b1 * m_ad + (1 - b1) * grad
            v_ad = b2 * v_ad + (1 - b2) * (grad * grad)
            bc1 = 1 - b1 ** step
            bc2 = 1 - b2 ** step
            mhat = m_ad / bc1
            vhat = v_ad / bc2
            # lr decay slightly
            lr = lr0 / (1.0 + step * 0.00015)
            X = X - lr * mhat / (np.sqrt(vhat) + eps)
            X = _proj(X)
        # end inner
        # final check of this restart
        try:
            vm = _max_cos(X)
        except Exception:
            continue
        if vm < global_val:
            global_val = vm
            global_best = X.copy()
        if vm < SUCCESS:
            success_x = X.copy()
            break
        # if we already have success from inner, break
        if success_x is not None:
            break
        # update base for perturbations if improved a lot? already via global_best
        # diversify: occasionally perturb with larger noise inside homotopy is done via restarts
    # --- spend remaining budget ---
    if success_x is not None:
        # polish success; keep time for return
        secs = time_left() - 4.0
        if secs < 5:
            secs = 5
        secs = min(secs, 30.0)
        try:
            px = _lp_polish(success_x, secs)
            pv = _max_cos(px)
            if pv < global_val:
                global_best = px
                global_val = pv
            elif pv < SUCCESS:
                # keep polished success even if slightly worse than success_x? keep better
                if pv < _max_cos(success_x):
                    global_best = px
                    global_val = pv
                else:
                    # success_x is better
                    if _max_cos(success_x) < global_val:
                        global_best = success_x
                        global_val = _max_cos(success_x)
        except Exception:
            # fallback to success iterate
            sv = _max_cos(success_x)
            if sv < global_val:
                global_best = success_x
    # ensure valid output
    global_best = _proj(np.nan_to_num(global_best))
    # tiny shrink guard not needed (normalization only)
    return global_best
