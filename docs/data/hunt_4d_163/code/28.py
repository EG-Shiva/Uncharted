import time
import numpy as np
from scipy.optimize import linprog, minimize
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _lp_polish(x, deadline, r0=0.004, cap=1300):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    try:
        best = _max_cos(x)
    except Exception:
        return x
    radius = r0
    iu0, iu1 = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        G = x @ x.T
        c = G[iu0, iu1]
        m = float(c.max())
        thresh = max(20.0 * radius, 1e-9)
        sel = c > m - thresh
        cnt = int(np.count_nonzero(sel))
        if cnt == 0:
            sel[np.argmax(c)] = True
            cnt = 1
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
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv); obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        try:
            y = x + res.x[:-1].reshape(n, d)
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            v = _max_cos(y)
        except Exception:
            radius *= 0.5
            continue
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def _anneal(x0, steps, T0, T1, lr, mu, rng, active=None, noise0=0.0, deadline=None):
    x = x0.copy()
    n, d = x.shape
    if active is None:
        act = np.ones(n, dtype=bool)
    else:
        act = active
    vel = np.zeros_like(x)
    G0 = x @ x.T
    np.fill_diagonal(G0, -2.0)
    bv = float(G0.max())
    bx = x.copy()
    if T0 <= 0 or T1 <= 0:
        T0 = max(T0, 1e-12); T1 = max(T1, 1e-12)
        lratio = 0.0
    else:
        lratio = np.log(T1 / T0)
    for k in range(steps):
        if deadline is not None and (k & 63) == 0 and time.time() > deadline:
            break
        frac = k / max(1, steps - 1)
        T = T0 * np.exp(lratio * frac)
        if T < 5e-5:
            T = 5e-5
        lr_k = lr * (1.0 - 0.5 * frac)
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        E = np.exp((G - m) / T)
        np.fill_diagonal(E, 0.0)
        S = float(E.sum() * 0.5)
        if not np.isfinite(S) or S < 1e-300:
            S = 1e-300
        Grad = (E @ x) / S
        Grad -= (Grad * x).sum(axis=1, keepdims=True) * x
        if not np.all(act):
            Grad[~act] = 0.0
        vel = mu * vel + (1.0 - mu) * Grad
        if not np.all(act):
            vel[~act] = 0.0
        x[act] = x[act] - lr_k * vel[act]
        if noise0 > 0:
            amp = noise0 * (1.0 - frac)
            if amp > 0:
                na = int(act.sum())
                nz = rng.randn(na, d) * amp
                xa = x[act]
                nz -= (nz * xa).sum(axis=1, keepdims=True) * xa
                xa = xa + nz
                xa /= np.maximum(np.linalg.norm(xa, axis=1, keepdims=True), 1e-12)
                x[act] = xa
        xn = np.linalg.norm(x[act], axis=1, keepdims=True)
        xn = np.maximum(xn, 1e-12)
        x[act] /= xn
        if m < bv:
            bv = m
            bx = x.copy()
    return bx, bv

def _lbfgs(x_init, deadline, taus=(0.003, 0.001, 0.0004), iters=(80, 80, 100)):
    x = x_init / np.maximum(np.linalg.norm(x_init, axis=1, keepdims=True), 1e-12)
    n, d = x.shape
    iu0, iu1 = np.triu_indices(n, 1)
    Np = len(iu0)
    y = x.ravel().copy()
    for tau, maxiter in zip(taus, iters):
        if time.time() > deadline - 0.25:
            break
        def fg(yflat, _tau=tau):
            Y = yflat.reshape(n, d)
            rn = np.linalg.norm(Y, axis=1, keepdims=True)
            rn = np.maximum(rn, 1e-12)
            X = Y / rn
            g = (X @ X.T)[iu0, iu1]
            m = float(g.max())
            e = np.exp((g - m) / _tau)
            s = float(e.sum())
            if s <= 0 or not np.isfinite(s):
                return float(m), np.zeros_like(yflat)
            L = m + _tau * np.log(s / Np)
            w = e / s
            W = np.zeros((n, n))
            W[iu0, iu1] = w
            W = W + W.T
            GX = W @ X
            dot = (GX * X).sum(axis=1, keepdims=True)
            GY = (GX - dot * X) / rn
            return float(L), GY.ravel()
        try:
            res = minimize(fg, y, method="L-BFGS-B", jac=True,
                           options={"maxiter": int(maxiter), "maxfun": int(maxiter * 6),
                                    "ftol": 1e-14, "gtol": 1e-9})
            y = res.x
        except Exception:
            break
    Y = y.reshape(n, d)
    Y /= np.maximum(np.linalg.norm(Y, axis=1, keepdims=True), 1e-12)
    return Y

def _deep_hole(others, rng, nc=10000):
    d = others.shape[1]
    cand = rng.randn(nc, d)
    cand /= np.linalg.norm(cand, axis=1, keepdims=True)
    chunk = 4000
    m_all = np.empty(nc)
    for a in range(0, nc, chunk):
        b = min(nc, a + chunk)
        m_all[a:b] = (cand[a:b] @ others.T).max(axis=1)
    bi = int(np.argmin(m_all))
    best_v = float(m_all[bi])
    best_c = cand[bi].copy()
    rad = 0.25
    for _ in range(3):
        prop = rng.randn(2500, d) * (rad / 2.0)
        prop -= (prop @ best_c)[:, None] * best_c[None, :]
        prop = best_c[None, :] + prop
        prop /= np.linalg.norm(prop, axis=1, keepdims=True)
        m = (prop @ others.T).max(axis=1)
        j = int(np.argmin(m))
        if float(m[j]) < best_v:
            best_v = float(m[j])
            best_c = prop[j].copy()
        rad *= 0.45
    return best_c

def _degrees_top(cur, K=480):
    n = cur.shape[0]
    G = cur @ cur.T
    np.fill_diagonal(G, -2.0)
    iu0, iu1 = np.triu_indices(n, 1)
    c = G[iu0, iu1]
    idx = np.argsort(-c)[:K]
    deg = np.zeros(n, dtype=int)
    rowmax = G.max(axis=1)
    for k in idx:
        i = int(iu0[k]); j = int(iu1[k])
        deg[i] += 1; deg[j] += 1
    return deg, rowmax, G

def solve():
    t0 = time.time()
    HARD_END = t0 + 174.0
    POLISH_START = HARD_END - 13.0
    rng = np.random.RandomState(12345)
    try:
        xb = np.load("best.npy").astype(np.float64)
    except Exception:
        xb = None
    try:
        xr = np.load("record.npy").astype(np.float64)
    except Exception:
        xr = None
    cands = []
    if xb is not None:
        xb = xb / np.linalg.norm(xb, axis=1, keepdims=True)
        cands.append(xb)
    if xr is not None:
        xr = xr / np.linalg.norm(xr, axis=1, keepdims=True)
        cands.append(xr)
    if len(cands) == 0:
        tmp = rng.randn(163, 4)
        tmp /= np.linalg.norm(tmp, axis=1, keepdims=True)
        cands.append(tmp)
    best = min(cands, key=lambda z: _max_cos(z)).copy()
    best_val = _max_cos(best)
    n, d = best.shape
    best = _lp_polish(best, min(HARD_END - 2, time.time() + 2.0), r0=0.003)
    best_val = _max_cos(best)
    cur = best.copy(); cur_val = best_val
    it = 0
    # Metropolis temperature for basin hopping
    Tt = 3e-6
    while time.time() < POLISH_START:
        it += 1
        x = cur.copy()
        deg, rowmax, Gc = _degrees_top(cur, K=480)
        order_worst = np.argsort(-rowmax)
        order_deg_hi = np.argsort(-deg)
        order_deg_lo = np.argsort(deg)
        r = rng.rand()
        if r < 0.40:
            # degree-balanced hub relocation: remove overloaded hubs to deep holes
            k = int(rng.choice([2, 3, 3, 4, 5]))
            victims = []
            # mix highest-degree and highest-rowmax
            pool_hi = list(order_deg_hi[:14]) + list(order_worst[:10])
            rng.shuffle(pool_hi)
            for v in pool_hi:
                v = int(v)
                if v not in victims:
                    victims.append(v)
                if len(victims) >= k:
                    break
            if rng.rand() < 0.3:
                # add a random victim to diversify
                rv = int(rng.randint(0, n))
                if rv not in victims:
                    victims[0] = rv
            # light global shake first
            x += rng.randn(n, d) * (0.002 + rng.rand() * 0.006)
            x /= np.linalg.norm(x, axis=1, keepdims=True)
            used = set()
            for victim in victims:
                if time.time() > POLISH_START:
                    break
                if victim in used:
                    continue
                used.add(victim)
                mask = np.ones(n, dtype=bool); mask[victim] = False
                others = x[mask]
                nc = 8000 + int(rng.randint(0, 8000))
                hc = _deep_hole(others, rng, nc=nc)
                # small jitter toward low-degree neighbourhood mean? just insert
                x[victim] = hc
            # anneal global
            T0 = 0.006 + rng.rand() * 0.008
            T1 = 0.0005 + rng.rand() * 0.0008
            steps = 200 + int(rng.randint(0, 150))
            lr = 0.025 + rng.rand() * 0.025
            mu = 0.75 + rng.rand() * 0.14
            x, _ = _anneal(x, steps, T0, T1, lr, mu, rng, active=None, noise0=0.0, deadline=POLISH_START)
            if time.time() > POLISH_START:
                v = _max_cos(x)
                if v < best_val:
                    best, best_val = x, v
                break
            # occasional L-BFGS refinement
            if rng.rand() < 0.6:
                x = _lbfgs(x, min(POLISH_START, time.time() + 1.2), taus=(0.0025, 0.0009, 0.0004), iters=(60, 60, 80))
            x = _lp_polish(x, min(POLISH_START, time.time() + 0.9), r0=0.004)
        elif r < 0.70:
            # large subgraph basin jump: BFS around worst point, big shake, frozen-bulk anneal
            K = int(rng.choice([24, 32, 40, 55]))
            if rng.rand() < 0.8:
                w = int(order_worst[0]) if rng.rand() < 0.7 else int(order_worst[rng.randint(0, 8)])
            else:
                w = int(rng.randint(0, n))
            # BFS by strong dots
            Gw = Gc[w]
            neigh = np.argsort(-Gw)
            S = [w]; seen = {w}
            for cand in list(neigh[:K * 2]):
                if len(S) >= K:
                    break
                c2 = int(cand)
                if c2 not in seen:
                    S.append(c2); seen.add(c2)
            for o in list(order_worst[:8]):
                if len(S) >= K:
                    break
                o = int(o)
                if o not in seen:
                    S.append(o); seen.add(o)
            while len(S) < K:
                rr = int(rng.randint(0, n))
                if rr not in seen:
                    S.append(rr); seen.add(rr)
            S = np.array(S, dtype=int)
            act = np.zeros(n, dtype=bool); act[S] = True
            sigma = 0.03 + rng.rand() * 0.10
            x[S] += rng.randn(len(S), d) * sigma
            x[S] /= np.linalg.norm(x[S], axis=1, keepdims=True)
            # strong pair-breaking on worst pairs touching S
            Gtmp = x @ x.T
            np.fill_diagonal(Gtmp, -2.0)
            iu0, iu1 = np.triu_indices(n, 1)
            c = Gtmp[iu0, iu1]
            topk = np.argsort(-c)[:8]
            step = 0.05 + rng.rand() * 0.08
            Sset = set(S.tolist())
            for tk in topk:
                i = int(iu0[tk]); j = int(iu1[tk])
                if (i not in Sset) and (j not in Sset):
                    continue
                dot = float(x[i] @ x[j])
                ti = x[j] - dot * x[i]
                tj = x[i] - dot * x[j]
                if act[i] and np.linalg.norm(ti) > 1e-12:
                    x[i] -= step * ti / max(1e-12, np.linalg.norm(ti)) * 0.2
                if act[j] and np.linalg.norm(tj) > 1e-12:
                    x[j] -= step * tj / max(1e-12, np.linalg.norm(tj)) * 0.2
            x /= np.linalg.norm(x, axis=1, keepdims=True)
            T0 = 0.008 + rng.rand() * 0.010
            T1 = 0.0004 + rng.rand() * 0.0007
            steps = 200 + int(rng.randint(0, 150))
            lr = 0.03 + rng.rand() * 0.03
            mu = 0.75 + rng.rand() * 0.15
            x, _ = _anneal(x, steps, T0, T1, lr, mu, rng, active=act, noise0=0.003 + rng.rand() * 0.005, deadline=POLISH_START)
            if time.time() > POLISH_START:
                v = _max_cos(x)
                if v < best_val:
                    best, best_val = x, v
                break
            # global short sweep cold
            x, _ = _anneal(x, 100 + int(rng.randint(0, 60)), 0.003, 0.0005, 0.02, 0.8, rng, active=None, noise0=0.0, deadline=POLISH_START)
            if rng.rand() < 0.4:
                x = _lbfgs(x, min(POLISH_START, time.time() + 0.9), taus=(0.0015, 0.0005), iters=(60, 80))
            x = _lp_polish(x, min(POLISH_START, time.time() + 0.9), r0=0.004 + rng.rand() * 0.003)
        else:
            # fresh-basin probe: Thomson-like random + Coulomb pre-opt, or heavy global noise
            if rng.rand() < 0.25:
                # partly fresh: keep 70% of cur, replace 30% random
                nr = int(rng.choice([20, 35, 50]))
                # replace lowest-value? replace random + high-degree
                repl = set()
                for o in order_deg_hi[:nr]:
                    if len(repl) >= nr:
                        break
                    if rng.rand() < 0.7:
                        repl.add(int(o))
                while len(repl) < nr:
                    repl.add(int(rng.randint(0, n)))
                repl = list(repl)
                x[repl] = rng.randn(len(repl), d)
                x[repl] /= np.linalg.norm(x[repl], axis=1, keepdims=True)
            else:
                x += rng.randn(n, d) * (0.010 + rng.rand() * 0.020)
                x /= np.linalg.norm(x, axis=1, keepdims=True)
            T0 = 0.010 + rng.rand() * 0.010
            T1 = 0.0006 + rng.rand() * 0.0008
            steps = 220 + int(rng.randint(0, 160))
            lr = 0.03 + rng.rand() * 0.025
            mu = 0.75 + rng.rand() * 0.14
            x, _ = _anneal(x, steps, T0, T1, lr, mu, rng, active=None, noise0=0.002, deadline=POLISH_START)
            if time.time() > POLISH_START:
                v = _max_cos(x)
                if v < best_val:
                    best, best_val = x, v
                break
            if rng.rand() < 0.5:
                x = _lbfgs(x, min(POLISH_START, time.time() + 1.0), taus=(0.003, 0.001), iters=(70, 90))
            x = _lp_polish(x, min(POLISH_START, time.time() + 0.9), r0=0.004)
        try:
            v = _max_cos(x)
        except Exception:
            continue
        if v < best_val - 1e-12:
            best = x.copy(); best_val = v
            cur = x.copy(); cur_val = v
            Tt = max(Tt * 0.95, 8e-7)
        else:
            # Metropolis move to different basin
            if v < cur_val or rng.rand() < np.exp(-(v - cur_val) / Tt):
                cur = x.copy(); cur_val = v
            # slowly cool Tt, reheat if stuck
            if it % 15 == 0:
                Tt *= 0.92
                if Tt < 8e-7:
                    Tt = 2e-6
    # final deep polish of best
    try:
        best = _lp_polish(best, HARD_END - 1.0, r0=0.002)
        best_val = _max_cos(best)
        best = _lbfgs(best, HARD_END - 0.6, taus=(0.0006, 0.00025), iters=(90, 120))
        best = _lp_polish(best, HARD_END - 0.2, r0=0.0015)
    except Exception:
        pass
    best /= np.linalg.norm(best, axis=1, keepdims=True)
    return best
