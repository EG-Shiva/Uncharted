import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _lp_polish(X, seconds, p_cap=1000):
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    n, d = X.shape
    best = _max_cos(X)
    best_x = X.copy()
    radius = 1.0e-3
    start = time.time()
    iu0, iu1 = np.triu_indices(n, 1)
    itc = 0
    while time.time() - start < seconds and radius > 1e-14 and itc < 25:
        itc += 1
        G = X @ X.T
        c_all = G[iu0, iu1]
        m = float(c_all.max())
        window = max(20.0 * radius, 1e-9)
        sel = c_all > m - window
        cnt = int(np.count_nonzero(sel))
        if cnt == 0:
            break
        if cnt > p_cap:
            idx = np.where(sel)[0]
            part = np.argpartition(-c_all[idx], p_cap - 1)[:p_cap]
            keep = np.zeros_like(sel)
            keep[idx[part]] = True
            sel = keep
        I = iu0[sel]; J = iu1[sel]
        csel = c_all[sel]
        p = len(I)
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([(I[:, None] * d + np.arange(d)), (J[:, None] * d + np.arange(d)), np.full((p, 1), n * d)], axis=1).ravel()
        vals = np.concatenate([X[J], X[I], -np.ones((p, 1))], axis=1).ravel()
        remain = seconds - (time.time() - start)
        if remain <= 0.05:
            break
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((X.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv); obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs",
                          options={"time_limit": max(0.1, remain)})
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = X + res.x[:-1].reshape(n, d)
        yn = np.linalg.norm(y, axis=1, keepdims=True)
        yn[yn == 0] = 1.0
        y = y / yn
        v = _max_cos(y)
        if v < best:
            best = v; best_x = y.copy(); X = y
            radius = min(radius * 1.5, 0.03)
        else:
            radius *= 0.5
    return best_x

def _unrestricted_anneal(X0, steps, tau0, tau1, lr0, lr1, t_end=1e20):
    X = X0.copy()
    n, d = X.shape
    best = X.copy()
    G0 = X @ X.T
    np.fill_diagonal(G0, -2.0)
    best_val = float(G0.max())
    ltau0 = np.log(tau0); ltau1 = np.log(tau1)
    for s in range(steps):
        if s % 200 == 0 and time.time() > t_end:
            break
        frac = s / max(steps - 1, 1)
        tau = float(np.exp(ltau0 + (ltau1 - ltau0) * frac))
        lr = lr0 + (lr1 - lr0) * frac
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        # softmax weights
        W = np.exp((G - m) / tau)
        np.fill_diagonal(W, 0.0)
        A = W @ X
        S = (W * G).sum(axis=1)
        cnt = W.sum(axis=1, keepdims=True) + 1.0
        F = (A - S[:, None] * X) / cnt
        X = X - lr * F
        X /= np.linalg.norm(X, axis=1, keepdims=True)
        if m < best_val:
            best_val = m
            best = X.copy()
    return best, best_val

def _expand_c13(Rrep, Rfree):
    K, D = Rrep.shape
    Gsz = 13
    X3 = np.empty((K, Gsz, D))
    for t in range(Gsz):
        X3[:, t, :] = np.roll(Rrep, t, axis=1)
    Xsym = X3.reshape(K * Gsz, D)
    return np.vstack([Xsym, Rfree])

def _sym_anneal_c13(Rrep0, Rfree0, steps, tau0, tau1, lr0, lr1, t_end=1e20):
    Rrep = Rrep0.copy(); Rfree = Rfree0.copy()
    K, D = Rrep.shape
    Gsz = 13
    best_val = 1e9
    best_Rrep = Rrep.copy(); best_Rfree = Rfree.copy()
    ltau0 = np.log(tau0); ltau1 = np.log(tau1)
    for s in range(steps):
        if s % 200 == 0 and time.time() > t_end:
            break
        frac = s / max(steps - 1, 1)
        tau = float(np.exp(ltau0 + (ltau1 - ltau0) * frac))
        lr = lr0 + (lr1 - lr0) * frac
        X = _expand_c13(Rrep, Rfree)
        n = X.shape[0]
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        if m < best_val:
            best_val = m
            best_Rrep = Rrep.copy(); best_Rfree = Rfree.copy()
        W = np.exp((G - m) / tau)
        np.fill_diagonal(W, 0.0)
        A = W @ X
        S = (W * G).sum(axis=1)
        cnt = W.sum(axis=1, keepdims=True) + 1.0
        F = (A - S[:, None] * X) / cnt
        Xn = X - lr * F
        Xn /= np.linalg.norm(Xn, axis=1, keepdims=True)
        Xsym_new = Xn[:K * Gsz].reshape(K, Gsz, D)
        acc = np.zeros_like(Rrep)
        for t in range(Gsz):
            acc += np.roll(Xsym_new[:, t, :], -t, axis=1)
        Rrep = acc / Gsz
        Rrep /= np.linalg.norm(Rrep, axis=1, keepdims=True)
        Rfree = Xn[K * Gsz:]
        Rfree /= np.linalg.norm(Rfree, axis=1, keepdims=True)
    Xb = _expand_c13(best_Rrep, best_Rfree)
    return Xb, best_val, best_Rrep, best_Rfree

def _expand_c12(Rrep, Rfree):
    K, D = Rrep.shape
    Gsz = 12
    X3 = np.empty((K, Gsz, D))
    for t in range(Gsz):
        X3[:, t, :12] = np.roll(Rrep[:, :12], t, axis=1)
        X3[:, t, 12] = Rrep[:, 12]
    Xsym = X3.reshape(K * Gsz, D)
    return np.vstack([Xsym, Rfree])

def _sym_anneal_c12(Rrep0, Rfree0, steps, tau0, tau1, lr0, lr1, t_end=1e20):
    Rrep = Rrep0.copy(); Rfree = Rfree.copy()
    K, D = Rrep.shape
    Gsz = 12
    best_val = 1e9
    best_Rrep = Rrep.copy(); best_Rfree = Rfree.copy()
    ltau0 = np.log(tau0); ltau1 = np.log(tau1)
    for s in range(steps):
        if s % 200 == 0 and time.time() > t_end:
            break
        frac = s / max(steps - 1, 1)
        tau = float(np.exp(ltau0 + (ltau1 - ltau0) * frac))
        lr = lr0 + (lr1 - lr0) * frac
        X = _expand_c12(Rrep, Rfree)
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        if m < best_val:
            best_val = m
            best_Rrep = Rrep.copy(); best_Rfree = Rfree.copy()
        W = np.exp((G - m) / tau)
        np.fill_diagonal(W, 0.0)
        A = W @ X
        S = (W * G).sum(axis=1)
        cnt = W.sum(axis=1, keepdims=True) + 1.0
        F = (A - S[:, None] * X) / cnt
        Xn = X - lr * F
        Xn /= np.linalg.norm(Xn, axis=1, keepdims=True)
        Xsym_new = Xn[:K * Gsz].reshape(K, Gsz, D)
        acc12 = np.zeros((K, 12))
        acc1 = np.zeros((K,))
        for t in range(Gsz):
            acc12 += np.roll(Xsym_new[:, t, :12], -t, axis=1)
            acc1 += Xsym_new[:, t, 12]
        Rrep = np.empty_like(Rrep)
        Rrep[:, :12] = acc12 / Gsz
        Rrep[:, 12] = acc1 / Gsz
        Rrep /= np.linalg.norm(Rrep, axis=1, keepdims=True)
        Rfree = Xn[K * Gsz:]
        Rfree /= np.linalg.norm(Rfree, axis=1, keepdims=True)
    Xb = _expand_c12(best_Rrep, best_Rfree)
    return Xb, best_val, best_Rrep, best_Rfree

def solve():
    t0 = time.time()
    LIM = 168.0
    n, d = 123, 13
    rng = np.random.default_rng(0)
    best = None; best_val = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, float)
            if a.shape == (n, d):
                a = a / np.linalg.norm(a, axis=1, keepdims=True)
                v = _max_cos(a)
                if v < best_val:
                    best_val = v; best = a.copy()
        except Exception:
            pass
    if best is None:
        best = rng.standard_normal((n, d))
        best /= np.linalg.norm(best, axis=1, keepdims=True)
        best_val = _max_cos(best)
    # ---- symmetric C13 search ----
    try:
        sym_best = None; sym_val = 1e9
        for restart in range(3):
            if time.time() - t0 > LIM - 90:
                break
            Rrep0 = rng.standard_normal((9, d)); Rrep0 /= np.linalg.norm(Rrep0, axis=1, keepdims=True)
            Rfree0 = rng.standard_normal((6, d)); Rfree0 /= np.linalg.norm(Rfree0, axis=1, keepdims=True)
            Xb, vb, _, _ = _sym_anneal_c13(Rrep0, Rfree0, steps=900, tau0=0.03, tau1=0.003,
                                          lr0=0.6, lr1=0.06, t_end=t0 + LIM - 70)
            # short unrestricted release of symmetric result
            Xrel, vr = _unrestricted_anneal(Xb, steps=600, tau0=0.01, tau1=0.002, lr0=0.4, lr1=0.05, t_end=t0 + LIM - 60)
            if vr < sym_val:
                sym_val = vr; sym_best = Xrel.copy()
            if vb < sym_val:
                pass
            if sym_val < best_val - 1e-12:
                pass
        if sym_best is not None and sym_val < best_val:
            best = sym_best.copy(); best_val = sym_val
        # keep sym_best for breaking even if worse (different basin)
        saved_sym = sym_best if sym_best is not None else None
        saved_sym_val = sym_val if sym_best is not None else 1e9
    except Exception:
        saved_sym = None; saved_sym_val = 1e9
    # ---- symmetric C12 search ----
    try:
        sym12_best = None; sym12_val = 1e9
        for restart in range(3):
            if time.time() - t0 > LIM - 70:
                break
            Rrep0 = rng.standard_normal((10, d)); Rrep0 /= np.linalg.norm(Rrep0, axis=1, keepdims=True)
            Rfree0 = rng.standard_normal((3, d)); Rfree0 /= np.linalg.norm(Rfree0, axis=1, keepdims=True)
            Xb, vb, _, _ = _sym_anneal_c12(Rrep0, Rfree0, steps=900, tau0=0.03, tau1=0.003,
                                          lr0=0.6, lr1=0.06, t_end=t0 + LIM - 55)
            Xrel, vr = _unrestricted_anneal(Xb, steps=600, tau0=0.01, tau1=0.002, lr0=0.4, lr1=0.05, t_end=t0 + LIM - 45)
            if vr < sym12_val:
                sym12_val = vr; sym12_best = Xrel.copy()
        if sym12_best is not None and sym12_val < best_val:
            best = sym12_best.copy(); best_val = sym12_val
        if sym12_best is not None and (saved_sym is None or sym12_val < saved_sym_val):
            saved_sym = sym12_best.copy(); saved_sym_val = sym12_val
    except Exception:
        pass
    # ---- symmetry-breaking perturbations of best symmetric ----
    try:
        if saved_sym is not None and time.time() - t0 < LIM - 40:
            for k, sig in enumerate([0.004, 0.010, 0.020]):
                if time.time() - t0 > LIM - 40:
                    break
                Xp = saved_sym + sig * rng.standard_normal(saved_sym.shape)
                Xp /= np.linalg.norm(Xp, axis=1, keepdims=True)
                Xr, vr = _unrestricted_anneal(Xp, steps=1200, tau0=0.012, tau1=0.0015, lr0=0.5, lr1=0.04, t_end=t0 + LIM - 35)
                if vr < best_val:
                    best_val = vr; best = Xr.copy()
    except Exception:
        pass
    # ---- small basin hopping around overall best ----
    try:
        hops = 0
        while time.time() - t0 < LIM - 30 and hops < 8:
            hops += 1
            sig = 0.003 if hops % 2 == 1 else 0.007
            Xp = best + sig * rng.standard_normal(best.shape)
            Xp /= np.linalg.norm(Xp, axis=1, keepdims=True)
            Xr, vr = _unrestricted_anneal(Xp, steps=700, tau0=0.008, tau1=0.0015, lr0=0.35, lr1=0.04, t_end=t0 + LIM - 28)
            if vr < best_val - 2e-12:
                best_val = vr; best = Xr.copy()
    except Exception:
        pass
    # ---- final LP polish ----
    try:
        remain = LIM - (time.time() - t0) - 0.8
        if remain > 1.5:
            sec = float(np.clip(remain, 2.0, 22.0))
            pol = _lp_polish(best, sec)
            pv = _max_cos(pol)
            if pv < best_val:
                best = pol; best_val = pv
    except Exception:
        pass
    best = best / np.linalg.norm(best, axis=1, keepdims=True)
    return best
