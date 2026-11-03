import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _norm(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return x / n

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _sym_max_abs(y):
    g = y @ y.T
    np.fill_diagonal(g, 0.0)
    return float(np.abs(g).max())

def _relax_sym(y, steps, lr, T0, T1, deadline=1e18):
    y = _norm(y)
    m_ad = np.zeros_like(y)
    v_ad = np.zeros_like(y)
    b1, b2, eps = 0.9, 0.999, 1e-8
    b1p = 1.0
    b2p = 1.0
    if steps > 1:
        logr = np.log(max(T1, 1e-12) / max(T0, 1e-12)) / (steps - 1)
    else:
        logr = 0.0
    for t in range(1, steps + 1):
        if (t % 200 == 0) and time.time() > deadline:
            break
        T = T0 * np.exp(logr * (t - 1))
        if T < 1e-12:
            T = 1e-12
        G = y @ y.T
        np.fill_diagonal(G, 0.0)
        A = np.abs(G)
        mv = float(A.max())
        E = np.exp((A - mv) / T)
        np.fill_diagonal(E, 0.0)
        S = float(E.sum() * 0.5)
        if not np.isfinite(S) or S <= 0:
            break
        Sg = np.sign(G) * E
        Grad = (Sg @ y) / S
        dot = (Grad * y).sum(axis=1, keepdims=True)
        Gt = Grad - dot * y
        b1p *= b1
        b2p *= b2
        m_ad = b1 * m_ad + (1.0 - b1) * Gt
        v_ad = b2 * v_ad + (1.0 - b2) * (Gt * Gt)
        mhat = m_ad / (1.0 - b1p)
        vhat = v_ad / (1.0 - b2p)
        step = mhat / (np.sqrt(vhat) + eps)
        y = y - lr * step
        y = _norm(y)
    return y

def _relax_asym(x, steps, lr, T0, T1, deadline=1e18):
    x = _norm(x)
    m_ad = np.zeros_like(x)
    v_ad = np.zeros_like(x)
    b1, b2, eps = 0.9, 0.999, 1e-8
    b1p = 1.0
    b2p = 1.0
    if steps > 1:
        logr = np.log(max(T1, 1e-12) / max(T0, 1e-12)) / (steps - 1)
    else:
        logr = 0.0
    for t in range(1, steps + 1):
        if (t % 200 == 0) and time.time() > deadline:
            break
        T = T0 * np.exp(logr * (t - 1))
        if T < 8e-4:
            T = 8e-4
        G = x @ x.T
        np.fill_diagonal(G, -10.0)
        mv = float(G.max())
        E = np.exp((G - mv) / T)
        np.fill_diagonal(E, 0.0)
        S = float(E.sum() * 0.5)
        if not np.isfinite(S) or S <= 0:
            break
        Grad = (E @ x) / S
        dot = (Grad * x).sum(axis=1, keepdims=True)
        Gt = Grad - dot * x
        b1p *= b1
        b2p *= b2
        m_ad = b1 * m_ad + (1.0 - b1) * Gt
        v_ad = b2 * v_ad + (1.0 - b2) * (Gt * Gt)
        mhat = m_ad / (1.0 - b1p)
        vhat = v_ad / (1.0 - b2p)
        step = mhat / (np.sqrt(vhat) + eps)
        x = x - lr * step
        x = _norm(x)
    return x

def _lp_polish(x, seconds, deadline=1e18):
    x = _norm(x)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    itn = 0
    while time.time() - start < seconds and time.time() < deadline:
        if radius < 1e-14 or itn >= 40:
            break
        itn += 1
        c = (x @ x.T)[iu]
        m = float(c.max())
        thr = max(20.0 * radius, 1e-9)
        sel = c > m - thr
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        if p == 0:
            break
        if p > 2500:
            order = np.argsort(-c[sel])[:2500]
            I, J = I[order], J[order]
            cvals = c[sel][order]
        else:
            cvals = c[sel]
            p = len(I)
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv)
            obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-cvals, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y = _norm(y)
        v = _max_cos(y)
        if v < best:
            x, best, radius = y, v, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def _build_pool(rng):
    P = []
    for i in range(10):
        for s in (1.0, -1.0):
            v = np.zeros(10); v[i] = s; P.append(v)
    for i in range(10):
        for j in range(i + 1, 10):
            for si in (1.0, -1.0):
                for sj in (1.0, -1.0):
                    v = np.zeros(10); v[i] = si / np.sqrt(2.0); v[j] = sj / np.sqrt(2.0); P.append(v)
    for mask in range(1024):
        v = np.ones(10)
        for k in range(10):
            if (mask >> k) & 1:
                v[k] = -1.0
        v /= np.sqrt(10.0)
        P.append(v)
    for _ in range(800):
        idx = rng.choice(10, size=4, replace=False)
        signs = rng.choice(np.array([-1.0, 1.0]), size=4)
        v = np.zeros(10); v[idx] = signs / 2.0
        P.append(v)
    for _ in range(600):
        idx = rng.choice(10, size=3, replace=False)
        signs = rng.choice(np.array([-1.0, 1.0]), size=3)
        v = np.zeros(10); v[idx] = signs / np.sqrt(3.0)
        P.append(v)
    for _ in range(300):
        idx = rng.choice(10, size=5, replace=False)
        signs = rng.choice(np.array([-1.0, 1.0]), size=5)
        th = rng.random() * np.pi
        a = np.cos(th) / np.sqrt(3.0); b = np.sin(th) / np.sqrt(2.0)
        v = np.zeros(10)
        v[idx[:3]] = a * signs[:3]
        v[idx[3:]] = b * signs[3:]
        nrm = np.linalg.norm(v)
        if nrm > 1e-12:
            v /= nrm
            P.append(v)
    R = rng.standard_normal(size=(1200, 10))
    R = R / np.linalg.norm(R, axis=1, keepdims=True)
    for v in R:
        P.append(v)
    for _ in range(800):
        v = rng.integers(-2, 3, size=10).astype(float)
        nrm = np.linalg.norm(v)
        if nrm < 1e-12:
            continue
        v /= nrm
        P.append(v)
    P = np.array(P, dtype=float)
    P = P / np.linalg.norm(P, axis=1, keepdims=True)
    return P

def solve():
    t0 = time.time()
    DEAD = t0 + 172.0
    SYM_END = t0 + 55.0
    REL_END = t0 + 140.0
    rng = np.random.default_rng(77)
    n, d = 154, 10
    m = 77
    best = None
    best_val = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            if a.shape == (n, d):
                a = _norm(a.astype(float))
                v = _max_cos(a)
                if v < best_val:
                    best_val = v
                    best = a.copy()
        except Exception:
            pass
    if best is None:
        best = _norm(rng.standard_normal(size=(n, d)))
        best_val = _max_cos(best)
    # structured pool
    try:
        P = _build_pool(rng)
    except Exception:
        P = _norm(rng.standard_normal(size=(2500, 10)))
    Mp = P.shape[0]
    # greedy selection of 77 axes minimizing max abs
    try:
        first = int(rng.integers(Mp))
        sel = [first]
        sel_flag = np.zeros(Mp, dtype=bool)
        sel_flag[first] = True
        Y = P[sel].copy()
        for k in range(1, m):
            if time.time() > SYM_END - 30.0:
                while len(sel) < m:
                    cand = int(rng.integers(Mp))
                    if not sel_flag[cand]:
                        sel.append(cand); sel_flag[cand] = True
                Y = P[np.array(sel)].copy()
                break
            D = np.abs(P @ Y.T).max(axis=1)
            D[sel_flag] = 2.0
            bi = int(np.argmin(D))
            sel.append(bi); sel_flag[bi] = True
            Y = P[np.array(sel)].copy()
    except Exception:
        Y = _norm(rng.standard_normal(size=(m, d)))
    Y = _norm(Y.copy())
    targets = [0.42, 0.415, 0.41, 0.405, 0.40, 0.395]
    for ti, tgt in enumerate(targets):
        if time.time() > SYM_END:
            break
        try:
            Y = _relax_sym(Y, steps=450, lr=0.015, T0=0.012, T1=0.0016, deadline=SYM_END)
        except Exception:
            pass
        if time.time() > SYM_END:
            break
        for sw in range(14):
            if time.time() > SYM_END:
                break
            try:
                G = Y @ Y.T
                np.fill_diagonal(G, 0.0)
                A = np.abs(G)
                cur_max = float(A.max())
                cnt = (A > tgt).sum(axis=1)
                rowmax = A.max(axis=1)
                if (cnt.max() == 0 and cur_max < tgt):
                    break
                victim = int(np.argmax(cnt * 10.0 + rowmax))
                if cnt[victim] == 0:
                    victim = int(np.argmax(rowmax))
                keep = np.delete(np.arange(m), victim)
                Yrem = Y[keep]
                scores = np.abs(P @ Yrem.T).max(axis=1)
                scores[sel_flag] = 2.0
                bi = int(np.argmin(scores))
                if scores[bi] < rowmax[victim] - 1e-12:
                    Y[victim] = P[bi]
                    sel_flag[bi] = True
                    if sw % 3 == 2:
                        Y = _relax_sym(Y, steps=180, lr=0.010, T0=0.008, T1=0.0016, deadline=SYM_END)
                else:
                    order = np.argpartition(scores, 5)[:5]
                    improved = False
                    for cand in order:
                        cand = int(cand)
                        old = Y[victim].copy()
                        Y[victim] = P[cand]
                        nv = _sym_max_abs(Y)
                        if nv < cur_max - 1e-12:
                            sel_flag[cand] = True
                            improved = True
                            break
                        Y[victim] = old
                    if not improved:
                        break
            except Exception:
                break
    # expand antipodal to 154
    try:
        Y = _norm(Y)
        Xsym = np.vstack([Y, -Y])
        Xsym = _norm(Xsym)
        sym_val = _max_cos(Xsym)
    except Exception:
        Xsym = None
        sym_val = 1e9
    cur = None
    cur_val = 1e9
    if Xsym is not None:
        try:
            cur = _relax_asym(Xsym.copy(), steps=700, lr=0.015, T0=0.012, T1=0.0012, deadline=REL_END)
            cur_val = _max_cos(cur)
        except Exception:
            cur = Xsym.copy(); cur_val = sym_val
    else:
        cur = best.copy(); cur_val = best_val
    # release-stage basin hopping on 154 points
    trial = 0
    Tacc0 = 0.0015
    while time.time() < REL_END and time.time() < DEAD - 28.0:
        trial += 1
        try:
            frac = (time.time() - t0) / max(REL_END - t0, 1.0)
            Tacc = Tacc0 * (1.0 - 0.7 * frac)
            if Tacc < 0.0003:
                Tacc = 0.0003
            r = rng.random()
            if r < 0.5:
                sigma = 0.02 + rng.random() * 0.06
            else:
                sigma = 0.06 + rng.random() * 0.10
            prop = cur + sigma * rng.standard_normal(size=cur.shape)
            prop = _norm(prop)
            steps = 550 + int(rng.integers(250))
            prop = _relax_asym(prop, steps=steps, lr=0.015, T0=0.011, T1=0.0011, deadline=REL_END)
            val = _max_cos(prop)
            if val < cur_val - 1e-12:
                cur = prop; cur_val = val
            else:
                if rng.random() < np.exp((cur_val - val) / Tacc):
                    cur = prop; cur_val = val
            if trial % 4 == 0 and val < best_val - 1e-12:
                try:
                    p2 = _relax_asym(prop.copy(), steps=500, lr=0.008, T0=0.005, T1=0.0008, deadline=REL_END)
                    v2 = _max_cos(p2)
                    if v2 < val:
                        cur = p2; cur_val = v2
                except Exception:
                    pass
        except Exception:
            break
        if trial > 40:
            break
    # choose overall best candidate
    if cur_val < best_val - 1e-12:
        final = cur.copy()
        final_val = cur_val
    else:
        final = best.copy()
        final_val = best_val
        if cur_val < final_val:
            final = cur.copy(); final_val = cur_val
    remain = DEAD - time.time()
    if remain > 8.0:
        lp_time = min(max(6.0, remain - 3.0), 30.0)
        try:
            final = _lp_polish(final, lp_time, deadline=DEAD - 2.0)
            final_val = _max_cos(final)
        except Exception:
            pass
        if final_val > best_val + 1e-9:
            final = best.copy()
    final = _norm(final)
    nr = np.linalg.norm(final, axis=1)
    bad = ~(np.isfinite(nr) & (nr > 1e-12))
    if np.any(bad):
        rb = rng.standard_normal(size=(int(np.sum(bad)), d))
        final[bad] = _norm(rb)
    final = _norm(final)
    return final
