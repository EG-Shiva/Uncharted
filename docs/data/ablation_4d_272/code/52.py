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

def Ml(q):
    w, x, y, z = q
    return np.array([[w, -x, -y, -z],
                     [x, w, -z, y],
                     [y, z, w, -x],
                     [z, -y, x, w]], dtype=np.float64)

def Mr(q):
    w, x, y, z = q
    return np.array([[w, -x, -y, -z],
                     [x, w, z, -y],
                     [y, -z, w, x],
                     [z, y, -x, w]], dtype=np.float64)

def qmul(a, b):
    a = np.asarray(a, dtype=np.float64); b = np.asarray(b, dtype=np.float64)
    aw, ax, ay, az = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bw, bx, by, bz = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    w = aw*bw - ax*bx - ay*by - az*bz
    x = aw*bx + ax*bw + ay*bz - az*by
    y = aw*by - ax*bz + ay*bw + az*bx
    z = aw*bz + ax*by - ay*bx + az*bw
    return np.stack([w, x, y, z], axis=-1)

def build_templates():
    T = {}
    Q8 = []
    for i in range(4):
        for s in (-1.0, 1.0):
            v = [0.0, 0.0, 0.0, 0.0]; v[i] = s; Q8.append(v)
    T[8] = np.array(Q8, dtype=np.float64)
    # Dic16 via closure
    a = np.array([np.cos(np.pi/4), np.sin(np.pi/4), 0, 0], dtype=np.float64)
    j = np.array([0, 0, 1, 0], dtype=np.float64)
    els = [np.array([1., 0, 0, 0])]
    frontier = els[:]
    gens = [a, j]
    # BFS
    seen = [els[0]]
    def add(q):
        for s in seen:
            if np.linalg.norm(s - q) < 1e-9 or np.linalg.norm(s + q) < 1e-9:
                # check exact match (allow antipode distinct? keep both)
                pass
        return False
    # simple closure loop
    S = [np.array([1., 0, 0, 0])]
    for _ in range(20):
        new = []
        for s in S:
            for g in gens:
                for q in (qmul(s, g), qmul(g, s)):
                    found = False
                    for e in S + new:
                        if np.linalg.norm(e - q) < 1e-8:
                            found = True; break
                    if not found:
                        new.append(q)
        if not new:
            break
        S.extend(new)
        if len(S) > 32:
            break
    S = np.array(S)
    # S should be 16
    if len(S) != 16:
        # fallback analytic: powers of a + j*powers
        els = []
        cur = np.array([1., 0, 0, 0])
        for k in range(8):
            els.append(cur.copy())
            cur = qmul(cur, a)
        jel = [qmul(j, e) for e in els]
        S = np.array(els + jel)
    # normalize
    S = S / np.linalg.norm(S, axis=1, keepdims=True)
    T[16] = S[:16]
    # binary tetrahedral 24
    H = [v for v in Q8]
    for s0 in (-0.5, 0.5):
        for s1 in (-0.5, 0.5):
            for s2 in (-0.5, 0.5):
                for s3 in (-0.5, 0.5):
                    H.append([s0, s1, s2, s3])
    T[24] = np.array(H, dtype=np.float64)
    # binary octahedral 48
    O = [list(v) for v in H]
    v2 = 1.0/np.sqrt(2.0)
    for i in range(4):
        for jj in range(i+1, 4):
            for si in (-v2, v2):
                for sj in (-v2, v2):
                    v = [0.0, 0.0, 0.0, 0.0]; v[i] = si; v[jj] = sj
                    O.append(v)
    T[48] = np.array(O, dtype=np.float64)
    return T

def symmetric_optimize(partition, templates, rng, steps=900, lr=0.02):
    # partition: list of sizes
    norb = len(partition)
    Gs = [templates[s] for s in partition]
    # params uL,uM (4,) per orbit
    uL = rng.standard_normal((norb, 4))
    uM = rng.standard_normal((norb, 4))
    # normalize to avoid zero
    uL /= np.linalg.norm(uL, axis=1, keepdims=True)+1e-15
    uM /= np.linalg.norm(uM, axis=1, keepdims=True)+1e-15
    # adam states
    mL = np.zeros_like(uL); vL = np.zeros_like(uL)
    mM = np.zeros_like(uM); vM = np.zeros_like(uM)
    b1, b2, eps = 0.9, 0.999, 1e-8
    # precompute basis Ml(e), Mr(e)
    MlE = [Ml(np.eye(4)[k]) for k in range(4)]
    MrE = [Mr(np.eye(4)[k]) for k in range(4)]
    t = 0
    for it in range(steps):
        frac = it / max(steps-1, 1)
        alpha = 20.0 * (500.0/20.0)**frac
        # build points
        qL = uL / (np.linalg.norm(uL, axis=1, keepdims=True)+1e-15)
        qM = uM / (np.linalg.norm(uM, axis=1, keepdims=True)+1e-15)
        Plist = []
        Alist = []
        for j in range(norb):
            A = Ml(qL[j]) @ Mr(qM[j])
            Alist.append(A)
            Pj = Gs[j] @ A.T
            Plist.append(Pj)
        P = np.vstack(Plist)
        n = P.shape[0]
        G = P @ P.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        # weights upper triangle softmax
        # compute exp in stable way using full matrix (symmetric)
        E = np.exp(np.clip(alpha*(G - m), -60, 0.0))
        np.fill_diagonal(E, 0.0)
        S = float(E.sum())/2.0 + 1e-300
        W = E / (2.0*S)  # so that sum_{i<j} w =1? Actually E symmetric, sum/2 = S
        # gradient D = W @ P scaled? dL/dP = (W+W^T)/2? W symmetric, D = 2*? Let's derive:
        # L = m + log(sum_{i<j} exp)/alpha, dL = sum w_ij d(dot_ij)
        # d(dot_ij)/dP_i = P_j. So D_i = sum_j w_ij P_j where w_ij = E_ij/(2S)? Wait E_ij for i!=j, S=sum_{i<j}E.
        # E symmetric, sum_{all}=2S, W=E/(2S) gives row sum = sum_j w_ij. Good.
        D = W @ P
        # backprop per orbit
        grad_qL = np.zeros((norb, 4))
        grad_qM = np.zeros((norb, 4))
        off = 0
        for j in range(norb):
            mj = Gs[j].shape[0]
            Dj = D[off:off+mj]
            Gj = Gs[j]
            GA = Dj.T @ Gj  # 4x4 = grad_A
            A = Alist[j]
            # grad wrt qL,qM
            MrM = Mr(qM[j]); MlL = Ml(qL[j])
            for a in range(4):
                dA_L = MlE[a] @ MrM
                grad_qL[j, a] = float(np.sum(GA * dA_L))
                dA_M = MlL @ MrE[a]
                grad_qM[j, a] = float(np.sum(GA * dA_M))
            off += mj
        # chain to u
        for j in range(norb):
            nl = float(np.linalg.norm(uL[j])+1e-15)
            q = qL[j]
            g = grad_qL[j]
            grad_qL[j] = (g - float(g @ q)*q)/nl
            nm = float(np.linalg.norm(uM[j])+1e-15)
            q2 = qM[j]
            g2 = grad_qM[j]
            grad_qM[j] = (g2 - float(g2 @ q2)*q2)/nm
        t += 1
        # adam update
        mL = b1*mL + (1-b1)*grad_qL
        vL = b2*vL + (1-b2)*(grad_qL**2)
        mM = b1*mM + (1-b1)*grad_qM
        vM = b2*vM + (1-b2)*(grad_qM**2)
        mLh = mL/(1-b1**t); vLh = vL/(1-b2**t)
        mMh = mM/(1-b1**t); vMh = vM/(1-b2**t)
        uL -= lr * mLh/(np.sqrt(vLh)+eps)
        uM -= lr * mMh/(np.sqrt(vMh)+eps)
    # final build
    qL = uL / (np.linalg.norm(uL, axis=1, keepdims=True)+1e-15)
    qM = uM / (np.linalg.norm(uM, axis=1, keepdims=True)+1e-15)
    Plist = []
    for j in range(norb):
        A = Ml(qL[j]) @ Mr(qM[j])
        Plist.append(Gs[j] @ A.T)
    P = _normr(np.vstack(Plist))
    return P

def hopf_optimize(nlist, rng, steps=700, lr=0.02):
    # nlist: points per torus
    K = len(nlist)
    eta = rng.uniform(0.4, 1.17, size=K)
    delta = rng.uniform(0, 2*np.pi, size=K)
    wint = np.array([1 + (k % 3) for k in range(K)], dtype=int)  # 1,2,3 winding
    uL = rng.standard_normal((K, 4)); uL /= np.linalg.norm(uL, axis=1, keepdims=True)+1e-15
    uM = rng.standard_normal((K, 4)); uM /= np.linalg.norm(uM, axis=1, keepdims=True)+1e-15
    m_eta = np.zeros(K); v_eta = np.zeros(K)
    m_de = np.zeros(K); v_de = np.zeros(K)
    mL = np.zeros_like(uL); vL = np.zeros_like(uL)
    mM = np.zeros_like(uM); vM = np.zeros_like(uM)
    b1, b2, eps = 0.9, 0.999, 1e-8
    MlE = [Ml(np.eye(4)[k]) for k in range(4)]
    MrE = [Mr(np.eye(4)[k]) for k in range(4)]
    t = 0
    thetas = [2*np.pi*np.arange(n)/n for n in nlist]
    for it in range(steps):
        frac = it / max(steps-1, 1)
        alpha = 20.0*(400.0/20.0)**frac
        qL = uL/(np.linalg.norm(uL, axis=1, keepdims=True)+1e-15)
        qM = uM/(np.linalg.norm(uM, axis=1, keepdims=True)+1e-15)
        Xlist = []; Alist = []
        for k in range(K):
            th = thetas[k]; w = wint[k]
            c = np.cos(eta[k]); s = np.sin(eta[k])
            x0 = c*np.cos(th); x1 = c*np.sin(th)
            x2 = s*np.cos(w*th+delta[k]); x3 = s*np.sin(w*th+delta[k])
            Xb = np.stack([x0, x1, x2, x3], axis=1)
            A = Ml(qL[k]) @ Mr(qM[k])
            Alist.append(A); Xlist.append((Xb, A))
        Plist = [Xb @ A.T for (Xb, A) in Xlist]
        # for eta/delta deriv need base derivs too; store
        P = np.vstack(Plist)
        G = P @ P.T; np.fill_diagonal(G, -2.0)
        m = float(G.max())
        E = np.exp(np.clip(alpha*(G-m), -60, 0.0)); np.fill_diagonal(E, 0.0)
        S = float(E.sum())/2.0+1e-300
        W = E/(2.0*S)
        D = W @ P
        # backprop
        grad_eta = np.zeros(K); grad_de = np.zeros(K)
        grad_qL = np.zeros((K, 4)); grad_qM = np.zeros((K, 4))
        off = 0
        for k in range(K):
            n = nlist[k]; th = thetas[k]; w = wint[k]
            Dj = D[off:off+n]
            Xb, A = Xlist[k]
            # grad_A
            GA = Dj.T @ Xb
            MrM = Mr(qM[k]); MlL = Ml(qL[k])
            for a in range(4):
                grad_qL[k, a] = float(np.sum(GA*(MlE[a] @ MrM)))
                grad_qM[k, a] = float(np.sum(GA*(MlL @ MrE[a])))
            # grad_x base = Dj @ A
            Gx = Dj @ A  # n x4
            c = np.cos(eta[k]); s = np.sin(eta[k])
            # d/deta
            dxe0 = -s*np.cos(th); dxe1 = -s*np.sin(th)
            dxe2 = c*np.cos(w*th+delta[k]); dxe3 = c*np.sin(w*th+delta[k])
            grad_eta[k] = float(np.sum(Gx[:, 0]*dxe0+Gx[:, 1]*dxe1+Gx[:, 2]*dxe2+Gx[:, 3]*dxe3))
            # d/ddelta
            dxd2 = -s*np.sin(w*th+delta[k]); dxd3 = s*np.cos(w*th+delta[k])
            grad_de[k] = float(np.sum(Gx[:, 2]*dxd2+Gx[:, 3]*dxd3))
            off += n
        # chain u
        for k in range(K):
            nl = float(np.linalg.norm(uL[k])+1e-15)
            q = qL[k]; g = grad_qL[k]
            grad_qL[k] = (g-float(g@q)*q)/nl
            nm = float(np.linalg.norm(uM[k])+1e-15)
            q2 = qM[k]; g2 = grad_qM[k]
            grad_qM[k] = (g2-float(g2@q2)*q2)/nm
        t += 1
        m_eta = b1*m_eta+(1-b1)*grad_eta; v_eta = b2*v_eta+(1-b2)*(grad_eta**2)
        m_de = b1*m_de+(1-b1)*grad_de; v_de = b2*v_de+(1-b2)*(grad_de**2)
        mL = b1*mL+(1-b1)*grad_qL; vL = b2*vL+(1-b2)*(grad_qL**2)
        mM = b1*mM+(1-b1)*grad_qM; vM = b2*vM+(1-b2)*(grad_qM**2)
        eta -= lr*(m_eta/(1-b1**t))/(np.sqrt(v_eta/(1-b2**t))+eps)
        delta -= lr*(m_de/(1-b1**t))/(np.sqrt(v_de/(1-b2**t))+eps)
        uL -= lr*mL/(1-b1**t)/(np.sqrt(vL/(1-b2**t))+eps)
        uM -= lr*mM/(1-b1**t)/(np.sqrt(vM/(1-b2**t))+eps)
        eta = np.clip(eta, 0.08, 1.49)
    qL = uL/(np.linalg.norm(uL, axis=1, keepdims=True)+1e-15)
    qM = uM/(np.linalg.norm(uM, axis=1, keepdims=True)+1e-15)
    Plist = []
    for k in range(K):
        th = thetas[k]; w = wint[k]
        c = np.cos(eta[k]); s = np.sin(eta[k])
        Xb = np.stack([c*np.cos(th), c*np.sin(th), s*np.cos(w*th+delta[k]), s*np.sin(w*th+delta[k])], axis=1)
        A = Ml(qL[k]) @ Mr(qM[k])
        Plist.append(Xb @ A.T)
    return _normr(np.vstack(Plist))

def repulse_gd(x, steps, a0, a1, s0, s1, mu=0.85, deadline=None):
    x = _normr(x.copy())
    vel = np.zeros_like(x)
    for it in range(steps):
        if deadline is not None and time.time() > deadline:
            break
        frac = it/max(steps-1, 1)
        alpha = a0*(a1/a0)**frac if a0 != a1 else a0
        smax = s0*(s1/s0)**frac if s0 != s1 else s0
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        W = np.exp(np.clip(alpha*(G-m), -50, 0))
        np.fill_diagonal(W, 0.0)
        grad = W @ x
        dot = np.sum(grad*x, axis=1, keepdims=True)
        gt = grad-dot*x
        vel = mu*vel+gt
        nrm = float(np.linalg.norm(vel, axis=1).max())
        if nrm > 1e-14:
            x = x-(smax/nrm)*vel
            x = _normr(x)
    return x

def lp_polish_deadline(x, deadline, cap=5000):
    x = _normr(x.copy())
    best = max_cos(x)
    radius = 1e-3
    iu0 = np.triu_indices(x.shape[0], 1)
    n, d = x.shape
    ar = np.arange(d)
    while time.time() < deadline and radius > 1e-14:
        G = x @ x.T
        c_all = G[iu0]
        m = float(c_all.max())
        thresh = m-max(20.0*radius, 1e-9)
        sel = c_all > thresh
        cnt = int(np.count_nonzero(sel))
        if cnt == 0:
            radius *= 0.5; continue
        if cnt > cap:
            idx = np.where(sel)[0]
            order = np.argpartition(-c_all[idx], cap-1)[:cap]
            keep = np.zeros_like(sel, dtype=bool); keep[idx[order]] = True
            sel = keep
        c = c_all[sel]
        I = iu0[0][sel]; J = iu0[1][sel]
        p = len(I)
        nv = n*d+1
        rows = np.repeat(np.arange(p), 2*d+1)
        cols = np.concatenate([I[:, None]*d+ar[None, :], J[:, None]*d+ar[None, :], np.full((p, 1), n*d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n*d))), shape=(n, nv)).tocsr()
        except Exception:
            radius *= 0.5; continue
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)]*(n*d)+[(None, None)], method="highs")
        except Exception:
            radius *= 0.5; continue
        if res.status != 0:
            radius *= 0.5; continue
        y = x+res.x[:-1].reshape(n, d)
        y = _normr(y)
        v = max_cos(y)
        if v < best-1e-15:
            x, best = y, v
            radius = min(radius*1.5, 0.05)
        else:
            radius *= 0.5
    return x, best

def solve():
    t0 = time.time()
    BUDGET = 170.0
    T_END = t0+BUDGET
    rng = np.random.default_rng(7)
    n, d = 272, 4
    cands = []
    try:
        bx = np.load("best.npy")
        bx = np.asarray(bx, dtype=np.float64)
        if bx.shape == (n, d):
            cands.append(_normr(bx))
    except Exception:
        pass
    try:
        rx = np.load("record.npy")
        rx = np.asarray(rx, dtype=np.float64)
        if rx.shape == (n, d):
            cands.append(_normr(rx))
    except Exception:
        pass
    if not cands:
        cands.append(_normr(rng.standard_normal((n, d))))
    vals = [(max_cos(c), c) for c in cands]
    vals.sort(key=lambda t: t[0])
    best_v, best_x = vals[0][0], vals[0][1].copy()
    templates = build_templates()
    # --- symmetric phase, keep symmetry unbroken ---
    partitions = [
        [48]*4+[16]*5,
        [48]*2+[24]*4+[16]*5,
        [8]*34,
    ]
    sym_deadline = t0+55.0
    for part in partitions:
        if time.time() > sym_deadline:
            break
        try:
            P = symmetric_optimize(part, templates, rng, steps=650, lr=0.025)
            v = max_cos(P)
            if v < best_v:
                best_v, best_x = v, P.copy()
            # store even if worse for diversity? keep pool
            vals.append((v, P.copy()))
        except Exception:
            continue
    # Hopf-Clifford family
    if time.time() < t0+70.0:
        try:
            Ph = hopf_optimize([34]*8, rng, steps=600, lr=0.02)
            vh = max_cos(Ph)
            vals.append((vh, Ph.copy()))
            if vh < best_v:
                best_v, best_x = vh, Ph.copy()
        except Exception:
            pass
        try:
            Ph2 = hopf_optimize([68]*4, rng, steps=500, lr=0.02)
            vh2 = max_cos(Ph2)
            vals.append((vh2, Ph2.copy()))
            if vh2 < best_v:
                best_v, best_x = vh2, Ph2.copy()
        except Exception:
            pass
    # re-select incumbent (likely still record/best)
    vals.sort(key=lambda t: t[0])
    # keep top pool for restarts
    pool = [c.copy() for _, c in vals[:3]]
    # ensure best_x is current best
    best_v = max_cos(best_x)
    for c in pool:
        vc = max_cos(c)
        if vc < best_v-1e-15:
            best_v, best_x = vc, c.copy()
    # --- final symmetry-breaking: tiny perturbation + full polish ---
    # first polish incumbent hard
    remain = T_END-time.time()
    # quick GD + LP on best
    if remain > 25:
        try:
            x1 = repulse_gd(best_x, steps=800, a0=15.0, a1=180.0, s0=0.008, s1=0.0007, mu=0.85, deadline=T_END-25)
            v1 = max_cos(x1)
            if v1 < best_v:
                best_v, best_x = v1, x1.copy()
        except Exception:
            pass
    if time.time() < T_END-22:
        try:
            x2, v2 = lp_polish_deadline(best_x, min(T_END-18, time.time()+8))
            if v2 < best_v:
                best_v, best_x = v2, x2.copy()
            else:
                best_x = x2.copy() if v2 <= best_v+1e-12 else best_x
                best_v = min(best_v, v2)
        except Exception:
            pass
    # perturbation restarts to change basin slightly
    k = 0
    while time.time() < T_END-4:
        k += 1
        try:
            scale = 0.002 if (k % 2 == 0) else 0.006
            pert = best_x + scale*rng.standard_normal(best_x.shape)
            pert = _normr(pert)
            pert = repulse_gd(pert, steps=350, a0=25.0, a1=160.0, s0=0.005, s1=0.0007, mu=0.85, deadline=T_END-3)
            v = max_cos(pert)
            # tiny LP if promising
            if v < best_v+2e-4 and time.time() < T_END-6:
                pert2, v2 = lp_polish_deadline(pert, min(T_END-2, time.time()+2.5))
                v, pert = v2, pert2
            if v < best_v-1e-15:
                best_v, best_x = v, pert.copy()
        except Exception:
            continue
        if k > 30:
            break
    # final tiny LP lock
    if time.time() < T_END-1.5:
        try:
            xf, vf = lp_polish_deadline(best_x, T_END-0.3)
            if vf < best_v:
                best_v, best_x = vf, xf.copy()
        except Exception:
            pass
    return _normr(best_x)
