import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def build_code_from_seeds(sa, sb, cosA, sinA, cosB, sinB):
    # sa,sb (22,) unit; cosA/sinA (29,11), cosB/sinB (29,11)
    n29 = cosA.shape[0]
    nb = cosA.shape[1]
    XA = np.empty((n29, nb*2))
    XB = np.empty((n29, nb*2))
    for b in range(nb):
        cA = cosA[:, b]; sA = sinA[:, b]
        cB = cosB[:, b]; sB = sinB[:, b]
        sax = sa[2*b]; say = sa[2*b+1]
        sbx = sb[2*b]; sby = sb[2*b+1]
        XA[:, 2*b] = cA*sax - sA*say
        XA[:, 2*b+1] = sA*sax + cA*say
        XB[:, 2*b] = cB*sbx - sB*sby
        XB[:, 2*b+1] = sB*sbx + cB*sby
    return np.vstack([XA, XB])

def sym_optimize(rng, kvals, opt_phi, steps, beta0, beta1, lr0, lr1, deadline=None):
    theta0 = 2*np.pi/29.0
    nb = 11
    n29 = 29
    J = np.arange(n29)[:, None]  # 29x1
    K = np.asarray(kvals, dtype=float)[None, :]  # 1x11
    base_ang = J * K * theta0  # 29x11
    # init
    ua = rng.standard_normal(22)
    ub = rng.standard_normal(22)
    if opt_phi:
        phi = rng.uniform(0, 2*np.pi, size=nb)
    else:
        phi = np.zeros(nb)
    dim = 44 + (nb if opt_phi else 0)
    m = np.zeros(dim)
    v = np.zeros(dim)
    b1 = 0.9; b2 = 0.999; eps = 1e-8
    lb0, lb1 = np.log(beta0), np.log(beta1)
    ll0, ll1 = np.log(lr0), np.log(lr1)
    best_val = 1e9
    best_tuple = None
    # precompute nothing else
    for s in range(steps):
        if deadline is not None and time.time() >= deadline:
            break
        f = s / max(1, steps-1)
        beta = float(np.exp(lb0 + (lb1-lb0)*f))
        lr = float(np.exp(ll0 + (ll1-ll0)*f))
        na = float(np.linalg.norm(ua)); nb_ = float(np.linalg.norm(ub))
        if na == 0: na = 1.0
        if nb_ == 0: nb_ = 1.0
        sa = ua/na; sb = ub/nb_
        cosA = np.cos(base_ang); sinA = np.sin(base_ang)
        angB = base_ang + phi[None, :]
        cosB = np.cos(angB); sinB = np.sin(angB)
        X = build_code_from_seeds(sa, sb, cosA, sinA, cosB, sinB)
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        curmax = float(G.max())
        if curmax < best_val:
            best_val = curmax
            best_tuple = (ua.copy(), ub.copy(), phi.copy(), curmax)
        # softmax weights
        W = np.exp(beta*(G-curmax))
        np.fill_diagonal(W, 0.0)
        sw = float(W.sum())
        if not np.isfinite(sw) or sw <= 0:
            break
        W /= sw
        GradX = W @ X  # 58x22
        GradA = GradX[:n29, :]
        GradB = GradX[n29:, :]
        # reconstruct XB for phi grad (need X values)
        XB = X[n29:, :]
        g_sa = np.zeros(22); g_sb = np.zeros(22); g_phi = np.zeros(nb)
        for b in range(nb):
            cA = cosA[:, b]; sA = sinA[:, b]
            cB = cosB[:, b]; sB = sinB[:, b]
            gAx = GradA[:, 2*b]; gAy = GradA[:, 2*b+1]
            # Rot(-a) g
            dsax = float(np.sum(cA*gAx + sA*gAy))
            dsay = float(np.sum(-sA*gAx + cA*gAy))
            g_sa[2*b] = dsax; g_sa[2*b+1] = dsay
            gBx = GradB[:, 2*b]; gBy = GradB[:, 2*b+1]
            dsbx = float(np.sum(cB*gBx + sB*gBy))
            dsby = float(np.sum(-sB*gBx + cB*gBy))
            g_sb[2*b] = dsbx; g_sb[2*b+1] = dsby
            if opt_phi:
                xx = XB[:, 2*b]; xy = XB[:, 2*b+1]
                g_phi[b] = float(np.sum(gBx*(-xy) + gBy*(xx)))
        # to ua, ub
        dot_a = float(np.dot(g_sa, sa)); dot_b = float(np.dot(g_sb, sb))
        g_ua = (g_sa - dot_a*sa)/na
        g_ub = (g_sb - dot_b*sb)/nb_
        if opt_phi:
            g = np.concatenate([g_ua, g_ub, g_phi])
        else:
            g = np.concatenate([g_ua, g_ub])
        # adam
        m = b1*m + (1-b1)*g
        v = b2*v + (1-b2)*(g*g)
        c1 = 1.0 - b1**(s+1); c2 = 1.0 - b2**(s+1)
        mh = m/c1; vh = v/c2
        step = lr*mh/(np.sqrt(vh)+eps)
        # apply
        ua[:22] = ua - step[:22]
        ub[:] = ub - step[22:44] if opt_phi else ub - step[22:44]
        if opt_phi:
            phi = phi - step[44:]
    if best_tuple is None:
        return None, 1e9
    ua_b, ub_b, phi_b, bv = best_tuple
    na = np.linalg.norm(ua_b); nb_ = np.linalg.norm(ub_b)
    sa = ua_b/max(na,1e-12); sb = ub_b/max(nb_,1e-12)
    cosA = np.cos(base_ang); sinA = np.sin(base_ang)
    angB = base_ang + phi_b[None,:]
    Xb = build_code_from_seeds(sa, sb, cosA, sinA, np.cos(angB), np.sin(angB))
    return Xb, float(max_cos(Xb))

def adam_quench(x, steps, b0, b1, lr0, lr1):
    n, d = x.shape
    m1 = np.zeros_like(x); m2 = np.zeros_like(x)
    b_1 = 0.9; b_2 = 0.999; eps = 1e-8
    lb0, lb1 = np.log(b0), np.log(b1)
    ll0, ll1 = np.log(lr0), np.log(lr1)
    for s in range(steps):
        f = s / max(1, steps-1)
        beta = float(np.exp(lb0+(lb1-lb0)*f))
        lr = float(np.exp(ll0+(ll1-ll0)*f))
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        gmax = float(G.max())
        W = np.exp(beta*(G-gmax))
        np.fill_diagonal(W, 0.0)
        sW = float(W.sum())
        if not np.isfinite(sW) or sW == 0:
            break
        W /= sW
        Grad = W @ x
        dot = np.sum(Grad*x, axis=1, keepdims=True)
        Gt = Grad - dot*x
        m1 = b_1*m1 + (1-b_1)*Gt
        m2 = b_2*m2 + (1-b_2)*(Gt*Gt)
        c1 = 1.0-b_1**(s+1); c2 = 1.0-b_2**(s+1)
        step = lr*(m1/c1)/(np.sqrt(m2/c2)+eps)
        sdot = np.sum(step*x, axis=1, keepdims=True)
        step = step - sdot*x
        sn = np.linalg.norm(step, axis=1, keepdims=True)
        over = sn > 0.05
        scale = np.ones_like(sn); scale[over] = 0.05/(sn[over]+1e-12)
        step = step*scale
        x = x - step
        nr = np.linalg.norm(x, axis=1, keepdims=True); nr[nr==0]=1.0
        x = x/nr
    return x

def lp_polish(x, seconds, t_deadline=None):
    x = x/np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = max_cos(x)
    radius = 6e-4
    start = time.time()
    iu0 = np.triu_indices(n, 1)
    itlp = 0
    while True:
        if time.time()-start >= seconds:
            break
        if t_deadline is not None and time.time() >= t_deadline:
            break
        if radius < 1e-13:
            break
        itlp += 1
        if itlp > 30:
            break
        G = x @ x.T
        c = G[iu0]
        m = float(c.max())
        window = max(20.0*radius, 4e-5)
        order = np.argsort(-c)
        sel_list = []
        for idx in order:
            if c[idx] > m-window and len(sel_list) < 600:
                sel_list.append(int(idx))
            elif c[idx] <= m-window:
                break
            if len(sel_list) >= 600:
                break
        if len(sel_list) == 0:
            break
        sel = np.array(sel_list)
        I = iu0[0][sel]; J = iu0[1][sel]
        p = len(sel); nv = n*d+1
        try:
            rows = np.repeat(np.arange(p), 2*d+1)
            cols = np.concatenate([I[:,None]*d+np.arange(d), J[:,None]*d+np.arange(d), np.full((p,1), n*d)],1).ravel()
            vals = np.concatenate([x[J], x[I], -np.ones((p,1))],1).ravel()
            A_ub = coo_matrix((vals,(rows,cols)),shape=(p,nv)).tocsr()
            A_eq = coo_matrix((x.ravel(),(np.repeat(np.arange(n),d),np.arange(n*d))),shape=(n,nv)).tocsr()
        except Exception:
            break
        obj = np.zeros(nv); obj[-1]=1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c[sel], A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius,radius)]*(n*d)+[(None,None)], method="highs",
                          options={"time_limit":2.0})
        except Exception:
            radius*=0.5; continue
        if res.status != 0:
            radius*=0.5; continue
        y = x + res.x[:-1].reshape(n,d)
        y = y/np.linalg.norm(y,axis=1,keepdims=True)
        v = max_cos(y)
        if v < best-1e-15:
            x=y; best=v; radius=min(radius*1.5,0.02)
        else:
            radius*=0.5
    return x

def solve():
    t0 = time.time()
    TIME_LIMIT = 165.0
    DEAD = t0+TIME_LIMIT
    rng = np.random.default_rng(29)
    n,d = 58,22
    best=None; best_val=1e9
    for f in ("best.npy","record.npy"):
        try:
            v=np.load(f).astype(float)
            if v.shape!=(58,22): continue
            v=v/np.linalg.norm(v,axis=1,keepdims=True)
            c=max_cos(v)
            if c<best_val: best_val=c; best=v.copy()
        except Exception:
            continue
    if best is None:
        z=rng.standard_normal((n,d)); z/=np.linalg.norm(z,axis=1,keepdims=True)
        best=z; best_val=max_cos(z)
    # symmetry search budget ~ up to 40s but leave time
    sym_dead = min(t0+45.0, DEAD-90.0)
    # if very little time, skip
    cands=[]
    if time.time() < sym_dead:
        n_rest = 8
        for r in range(n_rest):
            if time.time() >= sym_dead: break
            if r < 2:
                kvals = list(range(1,12))  # 1..11
            else:
                kvals = sorted(rng.choice(np.arange(1,15), size=11, replace=False).tolist())
            opt_phi = (r % 2 == 1)
            # vary beta/lr slightly
            try:
                Xsym, vsym = sym_optimize(rng, kvals, opt_phi, steps=1300,
                                          beta0=25, beta1=1200,
                                          lr0=0.03, lr1=0.003,
                                          deadline=min(sym_dead, DEAD-80))
            except Exception:
                continue
            if Xsym is None: continue
            cands.append((vsym, Xsym, tuple(kvals), opt_phi))
        # also try phi=0 fixed with random kvals already included; add one more sharp anneal on best?
        cands.sort(key=lambda t: t[0])
    # release + quench top sym candidates
    # quench budget
    for idx, (vsym, Xsym, kv, op) in enumerate(cands[:3]):
        if time.time() >= DEAD-30: break
        try:
            y = Xsym.copy()
            # small random break of symmetry to allow release
            y = y + 0.0015*rng.standard_normal(y.shape)
            y /= np.linalg.norm(y,axis=1,keepdims=True)
            y = adam_quench(y, 700, 80, 600, 0.004, 0.0006)
            if time.time() >= DEAD-25: 
                v=max_cos(y)
            else:
                y = adam_quench(y, 250, 800, 2200, 0.001, 0.0003)
                v=max_cos(y)
            if v < best_val-1e-12:
                # quick LP squeeze 1s
                try:
                    y2 = lp_polish(y, 1.2, DEAD-5)
                    v2 = max_cos(y2)
                    if v2 < v: y,v = y2,v2
                except Exception:
                    pass
                best_val=v; best=y.copy()
        except Exception:
            continue
    # light hop around incumbent if no improvement and time remains (a couple quenches)
    if time.time() < DEAD-30:
        for _ in range(2):
            if time.time() >= DEAD-28: break
            try:
                y = best.copy() + 0.005*rng.standard_normal(best.shape)
                y /= np.linalg.norm(y,axis=1,keepdims=True)
                y = adam_quench(y, 600, 80, 600, 0.0035, 0.0005)
                v=max_cos(y)
                if v < best_val-1e-12:
                    best_val=v; best=y.copy()
                    break
            except Exception:
                break
    remain = DEAD - time.time() - 1.0
    if remain > 4:
        try:
            polished = lp_polish(best, min(remain-1.0, 14.0), DEAD-1.0)
            vp = max_cos(polished)
            if vp < best_val: best=polished; best_val=vp
        except Exception:
            pass
    best = best/np.linalg.norm(best,axis=1,keepdims=True)
    best = np.nan_to_num(best,nan=0.0,posinf=0.0,neginf=0.0)
    nr=np.linalg.norm(best,axis=1,keepdims=True); nr[nr==0]=1.0
    best=best/nr
    return best
