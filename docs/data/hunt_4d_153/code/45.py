import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(np.max(g))

def lp_polish(x, seconds):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = max_cos(x)
    radius, start = 1e-3, time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        c = (x @ x.T)[iu]
        m = float(np.max(c))
        sel = c > m - max(20 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p, nv = len(I), n * d + 1
        if p == 0:
            break
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv); obj[-1] = 1.0
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
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        v = max_cos(y)
        if v < best:
            x, best, radius = y, v, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def anneal_softmax(x, steps, a0, a1, lr0, lr1):
    n, d = x.shape
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    for s in range(steps):
        frac = s / max(steps - 1, 1)
        alpha = a0 * (a1 / a0) ** frac
        lr = lr0 * (lr1 / lr0) ** frac
        G = x @ x.T
        np.fill_diagonal(G, -np.inf)
        m = float(np.max(G))
        W = np.exp(alpha * (G - m))
        np.fill_diagonal(W, 0.0)
        rs = np.maximum(W.sum(axis=1, keepdims=True), 1e-300)
        A = (W @ x) / rs
        dot = np.sum(A * x, axis=1, keepdims=True)
        T = A - dot * x
        x = x - lr * T
        x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x

def anneal_weighted(x, new_idx, lam, steps, a0, a1, lr0, lr1):
    n, d = x.shape
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    core = np.ones(n, dtype=bool); core[new_idx] = False
    for s in range(steps):
        frac = s / max(steps - 1, 1)
        alpha = a0 * (a1 / a0) ** frac
        lr = lr0 * (lr1 / lr0) ** frac
        G = x @ x.T
        np.fill_diagonal(G, -np.inf)
        m = float(np.max(G))
        W = np.exp(alpha * (G - m))
        np.fill_diagonal(W, 0.0)
        if lam < 1.0:
            W[core, new_idx] *= lam
        rs = np.maximum(W.sum(axis=1, keepdims=True), 1e-300)
        A = (W @ x) / rs
        dot = np.sum(A * x, axis=1, keepdims=True)
        T = A - dot * x
        x = x - lr * T
        x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x

def contact_counts(x, thr=1e-9):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    m = float(np.max(G))
    cnt = np.sum(G > m - thr, axis=1)
    return cnt, m, G

def find_voids(x152, n_sample, k, rng):
    n, d = x152.shape
    Z = rng.standard_normal(size=(n_sample, d))
    Z /= np.linalg.norm(Z, axis=1, keepdims=True)
    # chunked max dot
    vals = np.empty(n_sample)
    CH = 8000
    for a in range(0, n_sample, CH):
        b = min(n_sample, a + CH)
        D = Z[a:b] @ x152.T
        vals[a:b] = D.max(axis=1)
    order = np.argsort(vals)  # smallest = deepest void
    picks = []
    for idx in order:
        z = Z[idx]
        ok = True
        for q in picks:
            if float(np.dot(z, q)) > 0.90:
                ok = False; break
        if ok:
            picks.append(z)
            if len(picks) >= k:
                break
    return picks, vals[order[:k]]

def solve():
    t0 = time.time()
    DEADLINE = 172.0
    rng = np.random.default_rng(12345)
    # load fallback record
    best = None; best_val = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            y = np.load(fn)
            y = y / np.linalg.norm(y, axis=1, keepdims=True)
            if y.shape == (153, 4) and np.all(np.isfinite(y)):
                v = max_cos(y)
                if v < best_val:
                    best_val = v; best = y.copy()
        except Exception:
            pass
    if best is None:
        z = rng.standard_normal(size=(153, 4))
        z /= np.linalg.norm(z, axis=1, keepdims=True)
        best = z; best_val = max_cos(best)
    champion = best.copy(); champ_val = best_val

    def remain():
        return DEADLINE - (time.time() - t0)

    # --- identify rattler ---
    try:
        cnt, m, G = contact_counts(champion)
        rattler = int(np.argmin(cnt))
    except Exception:
        rattler = 0

    # --- N=152: delete rattler, polish ---
    x152 = np.delete(champion, rattler, axis=0)
    x152 = x152 / np.linalg.norm(x152, axis=1, keepdims=True)
    if remain() > 8:
        x152 = lp_polish(x152, seconds=min(4.0, remain() - 4.0))

    # --- find several voids ---
    try:
        voids, _ = find_voids(x152, n_sample=90000, k=6, rng=rng)
    except Exception:
        voids = []

    # --- insertion branch: ~half budget ---
    # adiabatic lambda ramp 0->1 in 8 stages, beta 30->250, lr 0.005->0.0005
    NSTAGE = 8
    lams = np.linspace(0.0, 1.0, NSTAGE)
    B0, B1 = 30.0, 250.0
    L0, L1 = 0.005, 0.0005
    for vi, v in enumerate(voids):
        if remain() < 22:  # reserve for deletion branch + final
            break
        # start from polished 152 core + void point
        y = np.vstack([x152, v.reshape(1, 4)])
        y = y / np.linalg.norm(y, axis=1, keepdims=True)
        new_idx = 152
        # per-stage beta/lr slices of global schedule
        for si, lam in enumerate(lams):
            if remain() < 10:
                break
            frac0 = si / NSTAGE; frac1 = (si + 1) / NSTAGE
            a0 = B0 * (B1 / B0) ** frac0
            a1 = B0 * (B1 / B0) ** frac1
            r0 = L0 * (L1 / L0) ** frac0
            r1 = L0 * (L1 / L0) ** frac1
            y = anneal_weighted(y, new_idx, float(lam), steps=220, a0=a0, a1=a1, lr0=r0, lr1=r1)
            # short LP to open cavity gradually
            y = lp_polish(y, seconds=0.35)
        # final slightly stronger anneal at full coupling + polish
        if remain() > 6:
            y = anneal_softmax(y, steps=300, a0=80.0, a1=600.0, lr0=0.004, lr1=0.001)
            y = lp_polish(y, seconds=1.2)
        try:
            vv = max_cos(y)
        except Exception:
            continue
        if vv < champ_val - 1e-12 and np.all(np.isfinite(y)):
            champ_val = vv; champion = y.copy()

    # --- N=154 symmetric variant: duplicate rattler ---
    if remain() > 18:
        try:
            cnt2, _, _ = contact_counts(champion)
            r2 = int(np.argmin(cnt2))
            pr = champion[r2].copy()
            # small tangent offset
            off = rng.standard_normal(size=(4,)); off -= float(np.dot(off, pr)) * pr
            nrm = np.linalg.norm(off)
            if nrm < 1e-12:
                off = np.array([1.0, 0.0, 0.0, 0.0]); off -= float(np.dot(off, pr)) * pr; nrm = np.linalg.norm(off)
            off /= nrm
            eps = 0.02
            pnew = pr + eps * off
            pnew /= np.linalg.norm(pnew)
            x154 = np.vstack([champion, pnew.reshape(1, 4)])
            x154 = x154 / np.linalg.norm(x154, axis=1, keepdims=True)
            x154 = anneal_softmax(x154, steps=600, a0=40.0, a1=700.0, lr0=0.02, lr1=0.002)
            x154 = lp_polish(x154, seconds=min(2.0, remain() - 14.0))
            # candidate deletions: overloaded + worst-max + random
            cnt154, _, G154 = contact_counts(x154)
            np.fill_diagonal(G154, -2.0)
            pmax = G154.max(axis=1)
            order_cnt = np.argsort(-cnt154)
            order_max = np.argsort(-pmax)
            cands = []
            for i in order_cnt[:12]:
                if int(i) not in cands: cands.append(int(i))
            for i in order_max[:8]:
                if int(i) not in cands: cands.append(int(i))
            # fill to ~20 with random
            while len(cands) < 20:
                r = int(rng.integers(0, 154))
                if r not in cands: cands.append(r)
            cands = cands[:20]
            scored = []
            for ci in cands:
                if remain() < 10:
                    break
                xd = np.delete(x154, ci, axis=0)
                xd = xd / np.linalg.norm(xd, axis=1, keepdims=True)
                # quick screen without polish
                v0 = max_cos(xd)
                # polish short
                xp = lp_polish(xd, seconds=0.6)
                vp = max_cos(xp)
                scored.append((vp, v0, ci, xp))
            if scored:
                scored.sort(key=lambda t: t[0])
                # take top 3 for longer polish
                for k in range(min(3, len(scored))):
                    if remain() < 8:
                        break
                    vp, v0, ci, xp = scored[k]
                    xp2 = lp_polish(xp, seconds=1.2)
                    vp2 = max_cos(xp2)
                    if vp2 < champ_val - 1e-12 and np.all(np.isfinite(xp2)):
                        champ_val = vp2; champion = xp2.copy()
                # also if best scored already beats champ, keep it
                if scored[0][0] < champ_val - 1e-12:
                    champ_val = scored[0][0]; champion = scored[0][3].copy()
        except Exception:
            pass

    # --- extra quick insertion tries if time left (different voids, no per-stage LP) ---
    while remain() > 14 and len(voids) > 0:
        try:
            # random jittered void from 152 core
            z = rng.standard_normal(size=(4,)); z /= np.linalg.norm(z)
            y = np.vstack([x152, z.reshape(1, 4)])
            y = y / np.linalg.norm(y, axis=1, keepdims=True)
            y = anneal_weighted(y, 152, 1.0, steps=500, a0=30.0, a1=500.0, lr0=0.005, lr1=0.001)
            y = lp_polish(y, seconds=1.0)
            vv = max_cos(y)
            if vv < champ_val - 1e-12 and np.all(np.isfinite(y)):
                champ_val = vv; champion = y.copy()
            else:
                break
        except Exception:
            break

    # --- final long polish of champion ---
    if remain() > 5:
        try:
            champion = lp_polish(champion, seconds=min(remain() - 1.0, 10.0))
            champ_val = max_cos(champion)
        except Exception:
            pass
    champion = champion / np.linalg.norm(champion, axis=1, keepdims=True)
    champion = np.where(np.isfinite(champion), champion, 0.0)
    nrm = np.linalg.norm(champion, axis=1, keepdims=True)
    nrm = np.maximum(nrm, 1e-12)
    champion = champion / nrm
    # safety: if somehow worse than fallback best by a lot due to NaN, return best
    try:
        if max_cos(champion) > best_val + 1e-9:
            return best
    except Exception:
        return best
    return champion
