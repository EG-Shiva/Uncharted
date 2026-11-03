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

def _contact_counts(x, mcur, delta):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return (g > mcur - delta).sum(axis=1), mcur

def _relax(x, steps, lr, T0, T1, beta1=0.9, beta2=0.999, eps=1e-8):
    n, d = x.shape
    m_ad = np.zeros_like(x)
    v_ad = np.zeros_like(x)
    b1p = 1.0
    b2p = 1.0
    if steps > 1:
        logr = np.log(max(T1, 1e-12) / max(T0, 1e-12)) / (steps - 1)
    else:
        logr = 0.0
    for t in range(1, steps + 1):
        T = T0 * np.exp(logr * (t - 1))
        if T < 5e-5:
            T = 5e-5
        G = x @ x.T
        np.fill_diagonal(G, -10.0)
        mv = G.max()
        # truncate far pairs to reduce noise, keep 3*T band at least
        # use full softmax but shift; truncation via mask
        E = np.exp((G - mv) / T)
        np.fill_diagonal(E, 0.0)
        S = E.sum() * 0.5
        if S <= 0 or not np.isfinite(S):
            break
        Grad = (E @ x) / S
        dot = (Grad * x).sum(axis=1, keepdims=True)
        Gt = Grad - dot * x
        # clip extreme grads
        b1p *= beta1
        b2p *= beta2
        m_ad = beta1 * m_ad + (1.0 - beta1) * Gt
        v_ad = beta2 * v_ad + (1.0 - beta2) * (Gt * Gt)
        mhat = m_ad / (1.0 - b1p)
        vhat = v_ad / (1.0 - b2p)
        step = mhat / (np.sqrt(vhat) + eps)
        # limit step norm per point
        sn = np.linalg.norm(step, axis=1, keepdims=True) + 1e-12
        # allow Adam step but cap at 1.0
        step = step / np.maximum(sn / 1.0, 1.0)
        x = x - lr * step
        x = _norm(x)
    return x

def _greedy_hole(prop, victim, rng, C=8000, K=8):
    d = prop.shape[1]
    cand = rng.normal(size=(C, d))
    cand = _norm(cand)
    Dots = cand @ prop.T
    Dots[:, victim] = -2.0
    md = Dots.max(axis=1)
    idx = np.argpartition(md, K)[:K]
    if rng.random() < 0.4:
        pick = idx[np.argmin(md[idx])]
    else:
        pick = idx[rng.integers(K)]
    return cand[pick]

def _lp_polish(x, seconds, deadline=1e18):
    x = _norm(x)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        if time.time() > deadline:
            break
        c = (x @ x.T)[iu]
        m = c.max()
        sel = c > m - max(20 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        if p == 0:
            break
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv)
        obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c[sel], A_eq=A_eq, b_eq=np.zeros(n),
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

def solve():
    t0 = time.time()
    rng = np.random.default_rng(12345)
    n, d = 154, 10
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
        best = _norm(rng.normal(size=(n, d)))
        best_val = _max_cos(best)
    # quick LP snap on loaded best if time allows (gets true local opt)
    # do not spend too much here
    try:
        if time.time() - t0 < 20:
            best = _lp_polish(best, 2.0, deadline=t0 + 170.0)
            best_val = _max_cos(best)
    except Exception:
        pass
    cur = best.copy()
    cur_val = best_val
    EXPL_END = t0 + 132.0
    DEAD = t0 + 170.0
    it = 0
    snap_counter = 0
    while time.time() < EXPL_END:
        it += 1
        elapsed = time.time() - t0
        frac = elapsed / max(EXPL_END - t0, 1.0)
        T_acc = 0.0018 * (1.0 - frac) + 0.00018
        if T_acc < 0.00015:
            T_acc = 0.00015
        r = rng.random()
        prop = cur.copy()
        if r < 0.55:
            # jammed hub kick to hole
            nk = int(rng.integers(1, 4))  # 1-3
            if rng.random() < 0.12:
                nk = int(rng.integers(4, 6))  # occasional big kick 4-5
            G = cur @ cur.T
            np.fill_diagonal(G, -2.0)
            mcur = float(G.max())
            delta = 0.002 + rng.random() * 0.005
            cnt = (G > mcur - delta).sum(axis=1)
            probs = cnt.astype(float) + 0.5
            probs /= probs.sum()
            victims = rng.choice(n, size=nk, replace=False, p=probs)
            for vv in victims:
                prop[vv] = _greedy_hole(prop, vv, rng, C=7000, K=8)
            # two-stage relax: hot spread then cold focus
            prop = _relax(prop, steps=550, lr=0.016, T0=0.014, T1=0.0025)
            prop = _relax(prop, steps=350, lr=0.010, T0=0.004, T1=0.0009)
        elif r < 0.75:
            # rattler refill: move points with no near contacts
            G = cur @ cur.T
            np.fill_diagonal(G, -2.0)
            mcur = float(G.max())
            ratt = np.where((G > mcur - 0.006).sum(axis=1) == 0)[0]
            if len(ratt) == 0:
                # fallback to overloaded
                cnt = (G > mcur - 0.004).sum(axis=1)
                ratt = np.argsort(-cnt)[:3]
            nk = min(len(ratt), int(rng.integers(1, 3)))
            if nk <= 0:
                nk = 1
                victims = rng.choice(n, size=1, replace=False)
            else:
                # pick random subset of rattlers, bias to random
                victims = rng.choice(ratt, size=nk, replace=False) if len(ratt) >= nk else ratt
            for vv in victims:
                prop[vv] = _greedy_hole(prop, vv, rng, C=9000, K=10)
            # also small jitter to neighbours of victims? no
            prop = _relax(prop, steps=600, lr=0.014, T0=0.010, T1=0.0010)
        elif r < 0.90:
            sigma = 0.02 + rng.random() * 0.06
            prop = cur + rng.normal(scale=sigma, size=cur.shape)
            prop = _norm(prop)
            prop = _relax(prop, steps=650, lr=0.015, T0=0.012, T1=0.0012)
        else:
            # fine shake + cold polish
            sigma = 0.005 + rng.random() * 0.012
            prop = cur + rng.normal(scale=sigma, size=cur.shape)
            prop = _norm(prop)
            prop = _relax(prop, steps=450, lr=0.009, T0=0.006, T1=0.0007)
        val = _max_cos(prop)
        if val < cur_val - 1e-12:
            cur = prop
            cur_val = val
        else:
            if rng.random() < np.exp((cur_val - val) / T_acc):
                cur = prop
                cur_val = val
        if val < best_val - 2e-7:
            # squeeze new record: extra cold relax
            prop2 = _relax(prop, steps=900, lr=0.007, T0=0.004, T1=0.0006)
            val2 = _max_cos(prop2)
            if val2 < val:
                prop, val = prop2, val2
                cur, cur_val = prop.copy(), val
            # short LP snap to get true basin value (cheap, 1.2s)
            snap_counter += 1
            try:
                if time.time() < EXPL_END - 5:
                    prop_s = _lp_polish(prop, 1.2, deadline=EXPL_END)
                    vals = _max_cos(prop_s)
                    if vals < val:
                        prop, val = prop_s, vals
                        cur, cur_val = prop.copy(), val
            except Exception:
                pass
            if val < best_val:
                best = prop.copy()
                best_val = val
        if time.time() > EXPL_END:
            break
    # final polish with remaining time
    remain = DEAD - (time.time() - t0) - 2.0
    if remain > 4:
        lp_time = min(max(5.0, remain), 36.0)
        try:
            best = _lp_polish(best, lp_time, deadline=DEAD)
        except Exception:
            pass
    best = _norm(best)
    best[~np.isfinite(best)] = 0.0
    nr = np.linalg.norm(best, axis=1, keepdims=True)
    bad = (nr.squeeze() < 1e-12)
    if np.any(bad):
        rb = rng.normal(size=(int(np.sum(bad)), d))
        best[bad] = _norm(rb)
    best = _norm(best)
    return best
