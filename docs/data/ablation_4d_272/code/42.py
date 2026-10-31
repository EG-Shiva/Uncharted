import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _normr(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return x / n

def max_cos(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def lp_polish_deadline(x, deadline):
    x = _normr(x.copy())
    n, d = x.shape
    best = max_cos(x)
    radius = 1e-3
    iu0 = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        G = x @ x.T
        c_all = G[iu0]
        m = float(c_all.max())
        thresh = m - max(20.0 * radius, 1e-9)
        sel = c_all > thresh
        cnt = int(np.count_nonzero(sel))
        if cnt == 0:
            radius *= 0.5
            continue
        if cnt > 5000:
            idx = np.where(sel)[0]
            order = np.argpartition(-c_all[idx], 5000 - 1)[:5000]
            keep = np.zeros_like(sel, dtype=bool)
            keep[idx[order]] = True
            sel = keep
        c = c_all[sel]
        I = iu0[0][sel]; J = iu0[1][sel]
        p = len(I)
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        except Exception:
            radius *= 0.5
            continue
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y = _normr(y)
        v = max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x, best

def repulse_gd(x, steps, a0, a1, s0, s1, mu=0.85, deadline=None):
    n, d = x.shape
    vel = np.zeros_like(x)
    for it in range(steps):
        if deadline is not None and time.time() > deadline:
            break
        frac = it / max(steps - 1, 1)
        alpha = a0 * (a1 / a0) ** frac if a0 != a1 else a0
        smax = s0 * (s1 / s0) ** frac if s0 != s1 else s0
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        W = np.exp(np.clip(alpha * (G - m), -50, 0))
        np.fill_diagonal(W, 0.0)
        grad = W @ x
        dot = np.sum(grad * x, axis=1, keepdims=True)
        gt = grad - dot * x
        vel = mu * vel + gt
        nrm = float(np.linalg.norm(vel, axis=1).max())
        if nrm > 1e-14:
            x = x - (smax / nrm) * vel
            x = _normr(x)
    return x

# ---- quaternion helpers ----
def qmul_ab(a, b):
    # a (m,4), b (4,) or (m,4) -> (m,4)
    a = np.asarray(a, dtype=np.float64); b = np.asarray(b, dtype=np.float64)
    if b.ndim == 1:
        bw, bx, by, bz = b[0], b[1], b[2], b[3]
        aw, ax, ay, az = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    else:
        aw, ax, ay, az = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
        bw, bx, by, bz = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    w = aw * bw - ax * bx - ay * by - az * bz if isinstance(bw, np.ndarray) else aw * bw - ax * bx - ay * by - az * bz
    x = aw * bx + ax * bw + ay * bz - az * by
    y = aw * by - ax * bz + ay * bw + az * bx
    z = aw * bz + ax * by - ay * bx + az * bw
    return np.stack([w, x, y, z], axis=1)

def qconj_a(a):
    a = np.asarray(a, dtype=np.float64)
    c = a.copy()
    c[:, 1:] *= -1
    return c

def qconj_v(v):
    v = np.asarray(v, dtype=np.float64)
    c = v.copy()
    c[1:] *= -1
    return c

def build_templates():
    T = {}
    Q8 = []
    for i in range(4):
        for s in (-1.0, 1.0):
            v = [0.0, 0.0, 0.0, 0.0]; v[i] = s; Q8.append(v)
    T['Q8'] = np.array(Q8, dtype=np.float64)
    def dicyclic(n):
        # order 4n? we want 16 and 32: n=4 ->16, n=8->32
        m = 2 * n  # number of powers? a = cos(pi/n)+i sin(pi/n), order 2n
        a = np.array([np.cos(np.pi / n), np.sin(np.pi / n), 0, 0], dtype=np.float64)
        # powers of a
        els = []
        cur = np.array([1.0, 0, 0, 0], dtype=np.float64)
        for _ in range(2 * n):
            els.append(cur.copy())
            # cur = cur * a
            aw, ax, ay, az = cur
            bw, bx, by, bz = a
            cur = np.array([aw*bw-ax*bx, aw*bx+ax*bw, ay*bw+az*bx if False else aw*by-ax*bz+ay*bw+az*bx, aw*bz+ax*by-ay*bx+az*bw])
            # generic formula with ay=az=by=bz=0 simplifies but use full
        j = np.array([0, 0, 1, 0], dtype=np.float64)
        els2 = []
        for e in els:
            aw, ax, ay, az = e
            bw, bx, by, bz = j
            w = aw*bw-ax*bx-ay*by-az*bz
            x = aw*bx+ax*bw+ay*bz-az*by
            y = aw*by-ax*bz+ay*bw+az*bx
            z = aw*bz+ax*by-ay*bx+az*bw
            els2.append(np.array([w, x, y, z]))
        G = np.array(els + els2, dtype=np.float64)
        G /= np.linalg.norm(G, axis=1, keepdims=True)
        return G
    T['D16'] = dicyclic(4)
    T['D32'] = dicyclic(8)
    H = []
    for i in range(4):
        for s in (-1.0, 1.0):
            v = [0.0, 0.0, 0.0, 0.0]; v[i] = s; H.append(v)
    for s0 in (-0.5, 0.5):
        for s1 in (-0.5, 0.5):
            for s2 in (-0.5, 0.5):
                for s3 in (-0.5, 0.5):
                    H.append([s0, s1, s2, s3])
    T['T24'] = np.array(H, dtype=np.float64)
    O = [h for h in H]
    sq = 1.0 / np.sqrt(2.0)
    for p1 in range(4):
        for p2 in range(p1 + 1, 4):
            for s1 in (-sq, sq):
                for s2 in (-sq, sq):
                    v = [0.0, 0.0, 0.0, 0.0]; v[p1] = s1; v[p2] = s2
                    O.append(v)
    O = np.array(O, dtype=np.float64)
    # dedup to 48
    # round and unique
    Or = np.round(O, 8)
    _, idx = np.unique(Or, axis=0, return_index=True)
    O = O[np.sort(idx)]
    if len(O) >= 48:
        # keep Hurwitz 24 + first 24 extras
        # ensure Hurwitz included
        O = O[:48] if len(O) == 48 else O
        if len(O) != 48:
            # fallback: take 48 with smallest lexicographic? just truncate/pad
            O = O[:48]
    T['O48'] = O
    return T

def optimize_partition(tmpl_list, rng, steps, a0=20.0, a1=400.0, lr=0.03, deadline=None):
    K = len(tmpl_list)
    Lr = rng.standard_normal((K, 4))
    Sr = rng.standard_normal((K, 4))
    mL = np.zeros_like(Lr); vL = np.zeros_like(Lr)
    mS = np.zeros_like(Sr); vS = np.zeros_like(Sr)
    b1, b2, eps = 0.9, 0.999, 1e-8
    # index slices
    sizes = [t.shape[0] for t in tmpl_list]
    N = sum(sizes)
    best_val = 2.0
    best_Ln = None; best_Sn = None
    for it in range(steps):
        if deadline is not None and time.time() > deadline:
            break
        frac = it / max(steps - 1, 1)
        alpha = a0 * (a1 / a0) ** frac
        Ln = Lr / np.linalg.norm(Lr, axis=1, keepdims=True)
        Sn = Sr / np.linalg.norm(Sr, axis=1, keepdims=True)
        # build X
        Xs = []
        Ts = []
        Us = []
        for k in range(K):
            g = tmpl_list[k]
            t = qmul_ab(g, Sn[k])
            x = qmul_ab(np.broadcast_to(Ln[k], t.shape), t)
            # qmul(L,t): L is (4,), t (m,4): need row-wise
            # our qmul_ab expects a (m,4), b (4,): computes a*b, but we need L*t = qmul(L,t)
            # so recompute correctly: L*t
            # implement L*t manually:
            # use swapped: compute via formula L*t
            Lw, Lx, Ly, Lz = Ln[k]
            tw, tx, ty, tz = t[:, 0], t[:, 1], t[:, 2], t[:, 3]
            w = Lw*tw - Lx*tx - Ly*ty - Lz*tz
            xx = Lw*tx + Lx*tw + Ly*tz - Lz*ty
            yy = Lw*ty - Lx*tz + Ly*tw + Lz*tx
            zz = Lw*tz + Lx*ty - Ly*tx + Lz*tw
            x = np.stack([w, xx, yy, zz], axis=1)
            Xs.append(x); Ts.append(t)
            u = qmul_ab(np.broadcast_to(Ln[k], g.shape), g) if False else None
            # u = L*g
            gw, gx, gy, gz = g[:, 0], g[:, 1], g[:, 2], g[:, 3]
            w2 = Lw*gw - Lx*gx - Ly*gy - Lz*gz
            x2 = Lw*gx + Lx*gw + Ly*gz - Lz*gy
            y2 = Lw*gy - Lx*gz + Ly*gw + Lz*gx
            z2 = Lw*gz + Lx*gy - Ly*gx + Lz*gw
            u = np.stack([w2, x2, y2, z2], axis=1)
            Us.append(u)
        X = np.concatenate(Xs, axis=0)
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        E = np.exp(np.clip(alpha * (G - m), -50, 0))
        np.fill_diagonal(E, 0.0)
        ssum = float(E.sum() / 2.0) + 1e-300
        Wm = E / ssum
        grad = Wm @ X
        # tangent project
        dot = np.sum(grad * X, axis=1, keepdims=True)
        gt = grad - dot * X
        # split per orbit
        gL = np.zeros_like(Ln); gS = np.zeros_like(Sn)
        off = 0
        for k in range(K):
            mk = sizes[k]
            gk = gt[off:off+mk]
            tk = Ts[k]; uk = Us[k]
            # grad_L = sum qmul(gx, conj(t))
            tc = qconj_a(tk)
            # qmul(gk, tc)
            aw, ax, ay, az = gk[:, 0], gk[:, 1], gk[:, 2], gk[:, 3]
            bw, bx, by, bz = tc[:, 0], tc[:, 1], tc[:, 2], tc[:, 3]
            w = aw*bw - ax*bx - ay*by - az*bz
            xx = aw*bx + ax*bw + ay*bz - az*by
            yy = aw*by - ax*bz + ay*bw + az*bx
            zz = aw*bz + ax*by - ay*bx + az*bw
            gL[k] = np.array([w.sum(), xx.sum(), yy.sum(), zz.sum()])
            # grad_S = sum qmul(conj(u), gx)
            uc = qconj_a(uk)
            aw2, ax2, ay2, az2 = uc[:, 0], uc[:, 1], uc[:, 2], uc[:, 3]
            bw2, bx2, by2, bz2 = gk[:, 0], gk[:, 1], gk[:, 2], gk[:, 3]
            w = aw2*bw2 - ax2*bx2 - ay2*by2 - az2*bz2
            xx = aw2*bx2 + ax2*bw2 + ay2*bz2 - az2*by2
            yy = aw2*by2 - ax2*bz2 + ay2*bw2 + az2*bx2
            zz = aw2*bz2 + ax2*by2 - ay2*bx2 + az2*bw2
            gS[k] = np.array([w.sum(), xx.sum(), yy.sum(), zz.sum()])
            off += mk
        # to raw via normalization jacobian
        Lnorm = np.linalg.norm(Lr, axis=1, keepdims=True)
        Snorm = np.linalg.norm(Sr, axis=1, keepdims=True)
        gLr = (gL - np.sum(gL * Ln, axis=1, keepdims=True) * Ln) / Lnorm
        gSr = (gS - np.sum(gS * Sn, axis=1, keepdims=True) * Sn) / Snorm
        # Adam update (gradient descent)
        mL = b1 * mL + (1 - b1) * gLr
        vL = b2 * vL + (1 - b2) * gLr * gLr
        mS = b1 * mS + (1 - b1) * gSr
        vS = b2 * vS + (1 - b2) * gSr * gSr
        bc = 1 - b1 ** (it + 1); bv = 1 - b2 ** (it + 1)
        Lr -= lr * (mL / bc) / (np.sqrt(vS * 0 + vL / bv) + eps)
        Sr -= lr * (mS / bc) / (np.sqrt(vS / bv) + eps)
        if (it % 100 == 0) or (it == steps - 1):
            vv = max_cos(X)
            if vv < best_val:
                best_val = vv
                best_Ln = Ln.copy(); best_Sn = Sn.copy()
    if best_Ln is None:
        Ln = Lr / np.linalg.norm(Lr, axis=1, keepdims=True)
        Sn = Sr / np.linalg.norm(Sr, axis=1, keepdims=True)
        best_Ln, best_Sn = Ln, Sn
        Xs = []
        for k in range(K):
            g = tmpl_list[k]
            t = qmul_ab(g, best_Sn[k])
            Lw, Lx, Ly, Lz = best_Ln[k]
            tw, tx, ty, tz = t[:, 0], t[:, 1], t[:, 2], t[:, 3]
            w = Lw*tw - Lx*tx - Ly*ty - Lz*tz
            xx = Lw*tx + Lx*tw + Ly*tz - Lz*ty
            yy = Lw*ty - Lx*tz + Ly*tw + Lz*tx
            zz = Lw*tz + Lx*ty - Ly*tx + Lz*tw
            Xs.append(np.stack([w, xx, yy, zz], axis=1))
        Xb = np.concatenate(Xs, axis=0)
        best_val = max_cos(Xb)
        return Xb, best_val
    Xs = []
    for k in range(K):
        g = tmpl_list[k]
        t = qmul_ab(g, best_Sn[k])
        Lw, Lx, Ly, Lz = best_Ln[k]
        tw, tx, ty, tz = t[:, 0], t[:, 1], t[:, 2], t[:, 3]
        w = Lw*tw - Lx*tx - Ly*ty - Lz*tz
        xx = Lw*tx + Lx*tw + Ly*tz - Lz*ty
        yy = Lw*ty - Lx*tz + Ly*tw + Lz*tx
        zz = Lw*tz + Lx*ty - Ly*tx + Lz*tw
        Xs.append(np.stack([w, xx, yy, zz], axis=1))
    Xb = np.concatenate(Xs, axis=0)
    return Xb, float(max_cos(Xb))

def optimize_hopf(nlist, rng, steps, deadline=None):
    T = len(nlist)
    N = sum(nlist)
    assign = np.repeat(np.arange(T), nlist)
    eta_raw = rng.standard_normal(T) * 0.5
    ang1 = rng.uniform(0, 2*np.pi, size=N)
    ang2 = rng.uniform(0, 2*np.pi, size=N)
    # init lattice: overwrite with even spacing + random offset
    off = 0
    for t, nt in enumerate(nlist):
        o1 = rng.uniform(0, 2*np.pi); o2 = rng.uniform(0, 2*np.pi)
        p = 1
        # find coprime
        while np.gcd(p, nt) != 1 and p < nt:
            p += 1
        for i in range(nt):
            ang1[off+i] = 2*np.pi*i/nt + o1
            ang2[off+i] = 2*np.pi*i*p/nt + o2
        off += nt
    m_e = np.zeros(T); v_e = np.zeros(T)
    m_a = np.zeros(N); v_a = np.zeros(N)
    m_b = np.zeros(N); v_b = np.zeros(N)
    b1, b2, eps = 0.9, 0.999, 1e-8
    lr = 0.03
    best_val = 2.0; best_X = None
    for it in range(steps):
        if deadline is not None and time.time() > deadline:
            break
        frac = it / max(steps-1, 1)
        alpha = 20.0 * (500.0/20.0) ** frac
        eta = np.pi/2 * (1/(1+np.exp(-eta_raw)))  # (0,pi/2), avoid ends
        eta = np.clip(eta, 0.05, np.pi/2-0.05)
        ce = np.cos(eta)[assign]; se = np.sin(eta)[assign]
        c1 = np.cos(ang1); s1 = np.sin(ang1); c2 = np.cos(ang2); s2 = np.sin(ang2)
        X = np.stack([ce*c1, ce*s1, se*c2, se*s2], axis=1)
        G = X @ X.T; np.fill_diagonal(G, -2.0)
        m = float(G.max())
        E = np.exp(np.clip(alpha*(G-m), -50, 0)); np.fill_diagonal(E, 0.0)
        ssum = float(E.sum()/2.0)+1e-300
        Wm = E/ssum
        grad = Wm @ X
        dot = np.sum(grad*X, axis=1, keepdims=True)
        gt = grad - dot*X
        # chain
        # dx/deta, dx/da, dx/db
        deta_x = np.stack([-se*c1, -se*s1, ce*c2, ce*s2], axis=1)
        da_x = np.stack([-ce*s1, ce*c1, np.zeros(N), np.zeros(N)], axis=1)
        db_x = np.stack([np.zeros(N), np.zeros(N), -se*s2, se*c2], axis=1)
        g_a = np.sum(gt*da_x, axis=1)
        g_b = np.sum(gt*db_x, axis=1)
        g_e_point = np.sum(gt*deta_x, axis=1)
        g_e = np.zeros(T)
        for t in range(T):
            g_e[t] = g_e_point[assign==t].sum()
        # eta_raw chain: deta/draw = pi/2 * sig*(1-sig)
        sig = 1/(1+np.exp(-eta_raw))
        g_eraw = g_e * (np.pi/2*sig*(1-sig))
        m_e = b1*m_e+(1-b1)*g_eraw; v_e = b2*v_e+(1-b2)*g_eraw*g_eraw
        m_a = b1*m_a+(1-b1)*g_a; v_a = b2*v_a+(1-b2)*g_a*g_a
        m_b = b1*m_b+(1-b1)*g_b; v_b = b2*v_b+(1-b2)*g_b*g_b
        bc = 1-b1**(it+1); bv = 1-b2**(it+1)
        eta_raw -= lr*(m_e/bc)/(np.sqrt(v_e/bv)+eps)
        ang1 -= lr*(m_a/bc)/(np.sqrt(v_a/bv)+eps)
        ang2 -= lr*(m_b/bc)/(np.sqrt(v_b/bv)+eps)
        if it % 100 == 0 or it == steps-1:
            vv = max_cos(X)
            if vv < best_val:
                best_val = vv; best_X = X.copy()
    return best_X, float(best_val)

def solve():
    t0 = time.time()
    BUDGET = 168.0
    rng = np.random.default_rng(20261031)
    try:
        bx = np.load("best.npy")
    except Exception:
        try:
            bx = np.load("record.npy")
        except Exception:
            bx = rng.standard_normal((272, 4))
    bx = np.asarray(bx, dtype=np.float64)
    if bx.shape != (272, 4):
        bx = np.resize(bx, (272, 4))
    bx = _normr(bx)
    best_x = bx.copy(); best_v = max_cos(best_x)
    try:
        rx = np.load("record.npy")
        rx = np.asarray(np.load("record.npy"), dtype=np.float64)
        rx = _normr(rx)
        rv = max_cos(rx)
        if rv < best_v:
            best_x, best_v = rx.copy(), rv
    except Exception:
        pass
    TPL = build_templates()
    # partitions of 272
    parts = []
    parts.append([('D16', 17)])
    parts.append([('Q8', 34)])
    parts.append([('D32', 8), ('D16', 1)])
    parts.append([('T24', 10), ('D16', 2)])
    parts.append([('T24', 11), ('Q8', 1)])
    parts.append([('O48', 5), ('D16', 2)])
    parts.append([('O48', 4), ('D32', 2), ('D16', 1)])
    sym_best_x = None; sym_best_v = 2.0
    # time for symmetric search: until t0+108
    sym_deadline = t0 + 108.0
    for pi, part in enumerate(parts):
        if time.time() > sym_deadline:
            break
        tmpl_list = []
        ok = True
        for name, cnt in part:
            base = TPL[name]
            for _ in range(cnt):
                tmpl_list.append(base)
        tot = sum(t.shape[0] for t in tmpl_list)
        if tot != 272:
            continue
        # shuffle order deterministically
        steps = 700 if tot > 0 else 700
        # reduce steps if little time
        remain = sym_deadline - time.time()
        if remain < 8:
            break
        if remain < 20:
            steps = 350
        try:
            Xc, vc = optimize_partition(tmpl_list, rng, steps=steps, deadline=sym_deadline)
        except Exception:
            continue
        if vc < sym_best_v:
            sym_best_v = vc; sym_best_x = Xc.copy()
        if vc < best_v:
            best_x, best_v = Xc.copy(), vc
    # Hopf families
    if time.time() < sym_deadline:
        for nlist in ([68, 68, 68, 68], [34]*8):
            if time.time() > sym_deadline:
                break
            try:
                Xh, vh = optimize_hopf(nlist, rng, steps=600, deadline=sym_deadline)
            except Exception:
                continue
            if vh < sym_best_v:
                sym_best_v = vh; sym_best_x = Xh.copy()
            if vh < best_v:
                best_x, best_v = Xh.copy(), vh
    # final stage: symmetry breaking + full polish (last ~60s, at least 20s)
    # choose base
    base = best_x.copy()
    # tiny symmetry-breaking perturbation in last phase
    # if base came from symmetric candidate, break it; else small kick to escape basin
    noise_scale = 1e-4 if sym_best_x is not None and best_v < 0.905 else 3e-3
    # always do: create a few perturbed variants and polish the best
    t_final_start = time.time()
    T_END = t0 + BUDGET
    # quick annealed full GD on base before LP
    try:
        if T_END - time.time() > 25:
            y = base.copy()
            # small noise to break exact symmetry
            y = y + rng.standard_normal(y.shape) * noise_scale
            y = _normr(y)
            y = repulse_gd(y, steps=500, a0=30, a1=600, s0=0.02, s1=0.002, deadline=T_END-18)
            v = max_cos(y)
            if v < best_v:
                best_x, best_v = y, v
                base = y.copy()
    except Exception:
        pass
    # iterative LP + GD kicks with remaining time
    try:
        while time.time() < T_END - 2:
            # LP polish with available time chunk
            chunk = min(12.0, (T_END - 2) - time.time())
            if chunk < 1.5:
                chunk = (T_END - 2) - time.time()
            if chunk <= 0.2:
                break
            yp, vp = lp_polish_deadline(base, time.time() + chunk)
            if vp < best_v:
                best_x, best_v = yp.copy(), vp
            base = yp.copy()
            if time.time() >= T_END - 3:
                break
            # small GD escape attempt
            y2 = base + rng.standard_normal(base.shape) * 1e-3
            y2 = _normr(y2)
            y2 = repulse_gd(y2, steps=200, a0=80, a1=800, s0=0.01, s1=0.001, deadline=T_END-2)
            v2 = max_cos(y2)
            if v2 < best_v:
                best_x, best_v = y2.copy(), v2
                base = y2.copy()
            else:
                # keep LP result as base; add tiny random tangent kick occasionally
                if rng.random() < 0.3:
                    base = base + rng.standard_normal(base.shape) * 5e-4
                    base = _normr(base)
    except Exception:
        pass
    best_x = _normr(best_x)
    return best_x
