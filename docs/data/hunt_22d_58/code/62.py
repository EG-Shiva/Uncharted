import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def adam_quench_full(x, steps, b0, b1, lr0, lr1):
    n, d = x.shape
    m1 = np.zeros_like(x)
    m2 = np.zeros_like(x)
    b_1 = 0.9; b_2 = 0.999; eps = 1e-8
    lb0, lb1 = np.log(b0), np.log(b1)
    ll0, ll1 = np.log(lr0), np.log(lr1)
    for s in range(steps):
        f = s / max(1, steps - 1)
        beta = float(np.exp(lb0 + (lb1 - lb0) * f))
        lr = float(np.exp(ll0 + (ll1 - ll0) * f))
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        gmax = float(G.max())
        W = np.exp(beta * (G - gmax))
        np.fill_diagonal(W, 0.0)
        sW = float(W.sum())
        if not np.isfinite(sW) or sW == 0:
            break
        W /= sW
        Grad = W @ x
        dot = np.sum(Grad * x, axis=1, keepdims=True)
        Gt = Grad - dot * x
        m1 = b_1 * m1 + (1 - b_1) * Gt
        m2 = b_2 * m2 + (1 - b_2) * (Gt * Gt)
        c1 = 1.0 - b_1 ** (s + 1)
        c2 = 1.0 - b_2 ** (s + 1)
        step = lr * (m1 / c1) / (np.sqrt(m2 / c2) + eps)
        sdot = np.sum(step * x, axis=1, keepdims=True)
        step = step - sdot * x
        sn = np.linalg.norm(step, axis=1, keepdims=True)
        over = sn > 0.05
        scale = np.ones_like(sn)
        scale[over] = 0.05 / (sn[over] + 1e-12)
        step = step * scale
        x = x - step
        nr = np.linalg.norm(x, axis=1, keepdims=True)
        nr[nr == 0] = 1.0
        x = x / nr
    return x

def lp_polish(x, seconds, t_deadline=None):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = max_cos(x)
    radius = 5e-4
    start = time.time()
    iu0 = np.triu_indices(n, 1)
    while True:
        if time.time() - start >= seconds:
            break
        if t_deadline is not None and time.time() >= t_deadline:
            break
        if radius < 1e-13:
            break
        G = x @ x.T
        c = G[iu0]
        m = float(c.max())
        window = max(20.0 * radius, 5e-5)
        order = np.argsort(-c)
        sel_order = []
        for idx in order:
            if c[idx] > m - window and len(sel_order) < 700:
                sel_order.append(int(idx))
            elif c[idx] <= m - window:
                break
            if len(sel_order) >= 700:
                break
        if len(sel_order) == 0:
            break
        sel = np.array(sel_order)
        I = iu0[0][sel]; J = iu0[1][sel]
        p = len(sel); nv = n * d + 1
        try:
            rows = np.repeat(np.arange(p), 2 * d + 1)
            cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
            vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        except Exception:
            break
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c[sel], A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs",
                          options={"time_limit": 5.0})
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        nr = np.linalg.norm(y, axis=1, keepdims=True)
        nr[nr == 0] = 1.0
        y = y / nr
        v = max_cos(y)
        if v < best - 1e-15:
            x = y; best = v
            radius = min(radius * 1.5, 0.02)
        else:
            radius *= 0.5
    return x

# ---- symmetric two-orbit machinery ----
def precompute_tables(m, n):
    # m,n: (11,) ints 1..14 ; P=29
    P = 29
    J = 11
    D = np.arange(1, 29)  # 28 lags
    WA = np.cos(2 * np.pi * m[:, None] * D[None, :] / P)  # 11x28
    WB = np.cos(2 * np.pi * n[:, None] * D[None, :] / P)
    K = np.arange(P)[:, None]  # k rows
    L = np.arange(P)[None, :]  # l cols
    cos0 = np.empty((J, P, P))
    sin0 = np.empty((J, P, P))
    for j in range(J):
        E = n[j] * np.broadcast_to(L, (P, P)) - m[j] * np.broadcast_to(K, (P, P))
        A = 2 * np.pi * E / P
        cos0[j] = np.cos(A)
        sin0[j] = np.sin(A)
    return WA, WB, cos0, sin0

def sym_value(u, v, phi, psi, WA, WB, cos0, sin0):
    # returns true max and components
    eu = np.exp(u - u.max()); R = eu / eu.sum()
    ev = np.exp(v - v.max()); S = ev / ev.sum()
    r = np.sqrt(R); s = np.sqrt(S)
    CA = R @ WA; CB = S @ WB
    delta = psi - phi
    cd = np.cos(delta); sd = np.sin(delta)
    t = r * s
    tcd = t * cd; tsd = t * sd
    CAB = np.tensordot(tcd, cos0, axes=([0], [0])) - np.tensordot(tsd, sin0, axes=([0], [0]))
    mval = max(float(CA.max()), float(CB.max()), float(CAB.max()))
    return mval, R, S, r, s, CA, CB, CAB

def sym_quench(m, n, u, v, phi, psi, steps, b0, b1, lr0, lr1, WA, WB, cos0, sin0):
    J = 11
    mu = np.zeros(J); mv = np.zeros(J); mph = np.zeros(J); mps = np.zeros(J)
    vu = np.zeros(J); vv = np.zeros(J); vph = np.zeros(J); vps = np.zeros(J)
    b_1 = 0.9; b_2 = 0.999; eps = 1e-8
    lb0, lb1 = np.log(b0), np.log(b1)
    ll0, ll1 = np.log(lr0), np.log(lr1)
    P = 29
    for st in range(steps):
        f = st / max(1, steps - 1)
        beta = float(np.exp(lb0 + (lb1 - lb0) * f))
        lr = float(np.exp(ll0 + (ll1 - ll0) * f))
        # softmax radii
        meu = u - u.max(); eu = np.exp(meu); R = eu / eu.sum()
        mev = v - v.max(); ev = np.exp(mev); S = ev / ev.sum()
        r = np.sqrt(np.maximum(R, 1e-16)); s = np.sqrt(np.maximum(S, 1e-16))
        t = r * s
        delta = psi - phi
        cd = np.cos(delta); sd = np.sin(delta)
        tcd = t * cd; tsd = t * sd
        CA = R @ WA  # 28
        CB = S @ WB
        CAB = np.tensordot(tcd, cos0, axes=([0], [0])) - np.tensordot(tsd, sin0, axes=([0], [0]))
        # global softmax weights
        gmax = max(float(CA.max()), float(CB.max()), float(CAB.max()))
        eA = np.exp(beta * (CA - gmax))
        eB = np.exp(beta * (CB - gmax))
        eAB = np.exp(beta * (CAB - gmax))
        tot = eA.sum() + eB.sum() + eAB.sum()
        if not np.isfinite(tot) or tot == 0:
            break
        wA = eA / tot; wB = eB / tot; wAB = eAB / tot
        WAtot = float(wA.sum()); WBtot = float(wB.sum()); WABtot = float(wAB.sum())
        # helpers
        WAwA = WA @ wA  # 11
        WBwB = WB @ wB
        wACA = float(wA @ CA); wBCB = float(wB @ CB)
        sumWABC = float((wAB * CAB).sum())
        Acoef = np.einsum('kl,jkl->j', wAB, cos0)
        Bcoef = np.einsum('kl,jkl->j', wAB, sin0)
        sumCos = cd * Acoef - sd * Bcoef
        sumSin = sd * Acoef + cd * Bcoef
        gA = R * (WAwA - wACA)
        gB = S * (WBwB - wBCB)
        gABu = 0.5 * (t * sumCos - R * sumWABC)
        gABv = 0.5 * (t * sumCos - S * sumWABC)
        gu = gA + gABu
        gv = gB + gABv
        gphi = t * sumSin
        gpsi = -gphi
        # Adam updates (descent)
        for grad, mbuf, vbuf in ((gu, 'u', None),):
            pass
        # manual Adam
        mu = b_1 * mu + (1 - b_1) * gu
        vu = b_2 * vu + (1 - b_2) * (gu * gu)
        mv = b_1 * mv + (1 - b_1) * gv
        vv = b_2 * vv + (1 - b_2) * (gv * gv)
        mph = b_1 * mph + (1 - b_1) * gphi
        vph = b_2 * vph + (1 - b_2) * (gphi * gphi)
        mps = b_1 * mps + (1 - b_1) * gpsi
        vps = b_2 * vps + (1 - b_2) * (gpsi * gpsi)
        c1 = 1 - b_1 ** (st + 1); c2 = 1 - b_2 ** (st + 1)
        su = lr * (mu / c1) / (np.sqrt(vu / c2) + eps)
        sv = lr * (mv / c1) / (np.sqrt(vv / c2) + eps)
        sph = lr * (mph / c1) / (np.sqrt(vph / c2) + eps)
        sps = lr * (mps / c1) / (np.sqrt(vps / c2) + eps)
        # clip
        su = np.clip(su, -0.2, 0.2); sv = np.clip(sv, -0.2, 0.2)
        sph = np.clip(sph, -0.3, 0.3); sps = np.clip(sps, -0.3, 0.3)
        u = u - su; v = v - sv; phi = phi - sph; psi = psi - sps
    return u, v, phi, psi

def build_X_from_sym(r, s, phi, psi, m, n):
    P = 29; J = 11
    X = np.zeros((58, 22))
    kk = np.arange(P)
    for j in range(J):
        angA = phi[j] + 2 * np.pi * m[j] * kk / P
        X[:P, 2 * j] = r[j] * np.cos(angA)
        X[:P, 2 * j + 1] = r[j] * np.sin(angA)
        angB = psi[j] + 2 * np.pi * n[j] * kk / P
        X[P:, 2 * j] = s[j] * np.cos(angB)
        X[P:, 2 * j + 1] = s[j] * np.sin(angB)
    nr = np.linalg.norm(X, axis=1, keepdims=True); nr[nr == 0] = 1
    return X / nr

def solve():
    t0 = time.time()
    TIME_LIMIT = 165.0
    DEADLINE = t0 + TIME_LIMIT
    rng = np.random.default_rng(29)
    n_pts, dim = 58, 22
    # load incumbents
    best_rec = None; best_rec_val = 1e9
    for f in ("best.npy", "record.npy"):
        try:
            vv = np.load(f).astype(float)
            if vv.shape != (58, 22):
                continue
            vv = vv / np.linalg.norm(vv, axis=1, keepdims=True)
            c = max_cos(vv)
            if c < best_rec_val:
                best_rec_val = c; best_rec = vv.copy()
        except Exception:
            continue
    if best_rec is None:
        z = rng.standard_normal((n_pts, dim)); z /= np.linalg.norm(z, axis=1, keepdims=True)
        best_rec = z; best_rec_val = max_cos(z)

    # ---- Phase 1: symmetric two-orbit search (budget ~55s) ----
    sym_deadline = t0 + 55.0
    # keep top few symmetric configs
    sym_bank = []  # (val, r,s,phi,psi,m,n)
    n_trials = 0
    # quick estimate: short trials 700 steps
    while time.time() < sym_deadline - 2.0:
        n_trials += 1
        # random multiplier assignments: 11 distinct out of 1..14 per orbit
        try:
            m = rng.choice(np.arange(1, 15), size=11, replace=False).astype(int)
            nn = rng.choice(np.arange(1, 15), size=11, replace=False).astype(int)
        except Exception:
            m = rng.integers(1, 15, size=11); nn = rng.integers(1, 15, size=11)
        WA, WB, cos0, sin0 = precompute_tables(m, nn)
        u = 0.3 * rng.standard_normal(11)
        vv0 = 0.3 * rng.standard_normal(11)
        phi = 2 * np.pi * rng.random(11)
        psi = 2 * np.pi * rng.random(11)
        # short annealed trial
        try:
            u2, v2, phi2, psi2 = sym_quench(m, nn, u, vv0, phi, psi, 650, 25.0, 350.0, 0.08, 0.008, WA, WB, cos0, sin0)
            mval, R, S, r, s, CA, CB, CAB = sym_value(u2, v2, phi2, psi2, WA, WB, cos0, sin0)
        except Exception:
            continue
        if not np.isfinite(mval):
            continue
        sym_bank.append((float(mval), r.copy(), s.copy(), phi2.copy(), psi2.copy(), m.copy(), nn.copy()))
        sym_bank.sort(key=lambda z: z[0])
        sym_bank = sym_bank[:4]
        if n_trials > 120:
            break
        if time.time() > sym_deadline - 2.0:
            break
    # refine top bank with longer quench
    refined = []
    for (val0, r0, s0, phi0, psi0, m0, n0) in list(sym_bank):
        if time.time() > sym_deadline + 8.0:
            break
        # convert r,s to logits
        u0 = np.log(np.maximum(r0 * r0, 1e-12)); u0 = u0 - u0.mean()
        v0 = np.log(np.maximum(s0 * s0, 1e-12)); v0 = v0 - v0.mean()
        WA, WB, cos0, sin0 = precompute_tables(m0, n0)
        try:
            u2, v2, phi2, psi2 = sym_quench(m0, n0, u0, v0, phi0, psi0, 1200, 40.0, 600.0, 0.05, 0.005, WA, WB, cos0, sin0)
            mval, R, S, r, s, CA, CB, CAB = sym_value(u2, v2, phi2, psi2, WA, WB, cos0, sin0)
            refined.append((float(mval), r, s, phi2, psi2, m0, n0))
        except Exception:
            refined.append((val0, r0, s0, phi0, psi0, m0, n0))
    if refined:
        refined.sort(key=lambda z: z[0])
        sym_bank = refined
    # build expanded symmetric seeds
    sym_seeds = []
    for (val0, r0, s0, phi0, psi0, m0, n0) in sym_bank[:3]:
        try:
            X = build_X_from_sym(r0, s0, phi0, psi0, m0, n0)
            sym_seeds.append((max_cos(X), X))
        except Exception:
            continue
    # ---- Phase 2: release symmetry (Direction-2 surgery seeds) ----
    # global best starts as record
    global_best = best_rec.copy(); global_val = float(best_rec_val)
    # also release each symmetric seed with a short unconstrained quench + keep as alternate basin
    alt_best = None; alt_val = 1e9
    for sv, SX in sym_seeds:
        if time.time() > DEADLINE - 60:
            break
        try:
            Y = adam_quench_full(SX.copy(), 700, 80, 700, 0.004, 0.0005)
            Y = adam_quench_full(Y, 300, 900, 2500, 0.001, 0.0003)
            v = max_cos(Y)
            if v < alt_val:
                alt_val = float(v); alt_best = Y.copy()
            if v < global_val - 1e-12:
                global_val = float(v); global_best = Y.copy()
        except Exception:
            continue
    if alt_best is None:
        # fallback: perturbed record as alt basin
        alt_best = global_best.copy(); alt_val = global_val
    # quick LP squeeze of global best to get true local value (2s)
    try:
        if time.time() < DEADLINE - 50:
            global_best = lp_polish(global_best, 2.0, DEADLINE - 40)
            global_val = max_cos(global_best)
    except Exception:
        pass
    # basin hopping mixing both basins: alternate seeds
    hop_end = DEADLINE - 20.0
    it = 0
    while time.time() < hop_end:
        it += 1
        if time.time() > DEADLINE - 25:
            break
        # choose base: alternate between global and alt to preserve diversity
        if it % 3 == 0 and alt_best is not None:
            base = alt_best; base_val = alt_val; is_alt = True
        else:
            base = global_best; base_val = global_val; is_alt = False
        y = base.copy()
        mode = it % 5
        if mode == 0:
            y = y + 0.004 * rng.standard_normal(y.shape)
            y /= np.linalg.norm(y, axis=1, keepdims=True)
        elif mode == 1:
            y = y + 0.009 * rng.standard_normal(y.shape)
            y /= np.linalg.norm(y, axis=1, keepdims=True)
        elif mode == 2:
            k = 2; idx = rng.choice(n_pts, size=k, replace=False)
            y = y + 0.003 * rng.standard_normal(y.shape)
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            rr = rng.standard_normal((k, dim)); rr /= np.linalg.norm(rr, axis=1, keepdims=True)
            y[idx] = rr
        elif mode == 3:
            G = base @ base.T; np.fill_diagonal(G, -2.0)
            ij = int(np.argmax(G)); i0 = ij // n_pts
            row = G[i0]; top = np.argsort(-row)[:3]
            cand = [int(i0)] + [int(t_) for t_ in top[:2]]
            cand = np.array(cand)
            y[cand] = y[cand] + 0.03 * rng.standard_normal((len(cand), dim))
            y /= np.linalg.norm(y, axis=1, keepdims=True)
        else:
            k = 3; idx = rng.choice(n_pts, size=k, replace=False)
            rr = rng.standard_normal((k, dim)); rr /= np.linalg.norm(rr, axis=1, keepdims=True)
            y[idx] = rr
        try:
            if mode in (0, 1):
                y = adam_quench_full(y, 800, 80, 600, 0.004, 0.0006)
                y = adam_quench_full(y, 250, 800, 2000, 0.001, 0.0003)
            elif mode == 2:
                y = adam_quench_full(y, 900, 100, 800, 0.003, 0.0004)
            elif mode == 3:
                y = adam_quench_full(y, 800, 60, 500, 0.004, 0.0005)
            else:
                y = adam_quench_full(y, 900, 80, 700, 0.0035, 0.0004)
        except Exception:
            continue
        v = max_cos(y)
        if is_alt:
            if v < alt_val - 1e-12:
                # short LP to confirm
                try:
                    y2 = lp_polish(y, 1.0, DEADLINE - 15)
                    v2 = max_cos(y2)
                    if v2 < v:
                        y, v = y2, v2
                except Exception:
                    pass
                alt_val = float(v); alt_best = y.copy()
                if v < global_val - 1e-12:
                    global_val = float(v); global_best = y.copy()
        else:
            if v < global_val - 1e-12:
                try:
                    y2 = lp_polish(y, 1.0, DEADLINE - 15)
                    v2 = max_cos(y2)
                    if v2 < v:
                        y, v = y2, v2
                except Exception:
                    pass
                global_val = float(v); global_best = y.copy()
        if it > 60:
            break
    # final polish
    remain = DEADLINE - time.time() - 1.0
    if remain > 4:
        try:
            polished = lp_polish(global_best, min(remain - 1.0, 16.0), DEADLINE - 1.0)
            vp = max_cos(polished)
            if vp < global_val:
                global_best = polished; global_val = vp
        except Exception:
            pass
    global_best = global_best / np.linalg.norm(global_best, axis=1, keepdims=True)
    global_best = np.nan_to_num(global_best, nan=0.0, posinf=0.0, neginf=0.0)
    nr = np.linalg.norm(global_best, axis=1, keepdims=True); nr[nr == 0] = 1.0
    global_best = global_best / nr
    return global_best
