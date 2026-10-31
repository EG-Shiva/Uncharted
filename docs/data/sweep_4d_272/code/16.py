import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _norm(x):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    x = np.where(np.isfinite(x), x, 0.0)
    n = np.linalg.norm(x, axis=1, keepdims=True)
    bad = (n[:, 0] < 1e-12)
    if np.any(bad):
        x[bad] = np.array([1.0, 0.0, 0.0, 0.0])
        x = x / np.linalg.norm(x, axis=1, keepdims=True)
    return x

def _small_rot(rng, scale):
    R = np.eye(4)
    pairs = [(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)]
    for (a,b) in pairs:
        th = scale * rng.standard_normal()
        c, s = np.cos(th), np.sin(th)
        G = np.eye(4)
        G[a,a]=c; G[a,b]=-s; G[b,a]=s; G[b,b]=c
        R = G @ R
    return R

def _find_void(others, rng, n_cand=8000):
    cand = rng.standard_normal((n_cand, 4))
    cand /= np.linalg.norm(cand, axis=1, keepdims=True)
    s = cand @ others.T
    scores = s.max(axis=1)
    bi = int(np.argmin(scores))
    b = cand[bi].copy()
    bs = float(scores[bi])
    step = 0.05
    for _ in range(200):
        prop = b + step * rng.standard_normal((32, 4))
        prop /= np.linalg.norm(prop, axis=1, keepdims=True)
        sc = (prop @ others.T).max(axis=1)
        j = int(np.argmin(sc))
        if sc[j] < bs:
            b, bs = prop[j], float(sc[j])
            step *= 1.05
        else:
            step *= 0.96
        if step < 1e-5:
            break
    return b

def _adam_anneal(x, steps, tau0, tau1, lr0, lr1, rng):
    n, d = x.shape
    m = np.zeros_like(x)
    v = np.zeros_like(x)
    b1, b2, eps = 0.9, 0.999, 1e-8
    for k in range(steps):
        frac = k/(steps-1) if steps>1 else 0.0
        tau = tau0*(tau1/tau0)**frac
        lr = lr0+(lr1-lr0)*frac
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        gm = float(G.max())
        np.subtract(G, gm, out=G)
        np.divide(G, tau, out=G)
        np.exp(G, out=G)
        np.fill_diagonal(G, 0.0)
        S = float(G.sum())
        if not np.isfinite(S) or S==0:
            break
        G /= S
        grad = G @ x
        dot = (grad*x).sum(axis=1, keepdims=True)
        g = grad - dot*x
        m *= b1; m += (1-b1)*g
        v *= b2; v += (1-b2)*(g*g)
        bc1 = 1.0 - b1**(k+1); bc2 = 1.0 - b2**(k+1)
        mh = m/bc1; vh = v/bc2
        step = mh/(np.sqrt(vh)+eps)
        np.clip(step, -2.0, 2.0, out=step)
        x = x - lr*step
        x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x

def _lp_polish(x, seconds, deadline):
    x = _norm(x)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time()-start < seconds and time.time() < deadline and radius > 1e-14:
        c = (x @ x.T)[iu]
        mc = float(c.max())
        sel = c > mc - max(20*radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p, nv = len(I), n*d+1
        if p==0:
            break
        rows = np.repeat(np.arange(p), 2*d+1)
        cols = np.concatenate([I[:,None]*d+np.arange(d), J[:,None]*d+np.arange(d), np.full((p,1), n*d)],1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p,1))],1).ravel()
        A_ub = coo_matrix((vals,(rows,cols)),shape=(p,nv)).tocsr()
        A_eq = coo_matrix((x.ravel(),(np.repeat(np.arange(n),d),np.arange(n*d))),shape=(n,nv)).tocsr()
        obj = np.zeros(nv); obj[-1]=1.0
        try:
            res = linprog(obj,A_ub=A_ub,b_ub=-c[sel],A_eq=A_eq,b_eq=np.zeros(n),
                          bounds=[(-radius,radius)]*(n*d)+[(None,None)],method="highs")
        except Exception:
            radius*=0.5; continue
        if res.status!=0:
            radius*=0.5; continue
        y = x + res.x[:-1].reshape(n,d)
        y = _norm(y)
        vv = _max_cos(y)
        if vv < best:
            x,best,radius = y,vv,min(radius*1.5,0.05)
        else:
            radius*=0.5
    return x

def solve():
    rng = np.random.default_rng(12345)
    t0 = time.time()
    deadline = t0 + 172.0
    try:
        x0 = np.load("best.npy")
    except Exception:
        x0 = np.load("record.npy")
    x0 = _norm(np.asarray(x0,float))
    if x0.shape != (272,4):
        tmp = rng.standard_normal((272,4))
        x0 = _norm(tmp)
    try:
        xr = _norm(np.asarray(np.load("record.npy"),float))
        if xr.shape==(272,4):
            if _max_cos(xr) < _max_cos(x0):
                x0 = xr.copy()
    except Exception:
        pass
    x_best = x0.copy()
    best_val = _max_cos(x_best)
    cur = x_best.copy(); cur_val = best_val

    sigmas = [0.02,0.05,0.01,0.08,0.03,0.12,0.015]
    trial = 0
    # explore until ~135s, reserve rest for LP
    while time.time() < t0+128 and time.time() < deadline-40:
        trial+=1
        sigma = sigmas[trial % len(sigmas)]
        # start from cur (random walk), periodically reset to best
        base = x_best if (trial%9==0) else cur
        x = base + sigma*rng.standard_normal((272,4))
        x = _norm(x)
        mode = trial % 3
        if mode==0:
            # rigid cluster rotation (collective move)
            seed = int(rng.integers(0,272))
            dots = x @ x[seed]
            C = int(8 + rng.integers(0,20))
            cl = np.argsort(-dots)[:C]
            R = _small_rot(rng, sigma*1.5+0.02)
            x[cl] = x[cl] @ R.T
            x = _norm(x)
        elif mode==1:
            # kill 1-3 worst points -> voids
            G = x @ x.T; np.fill_diagonal(G,-2.0)
            pw = G.max(axis=1)
            nk = 1 + (trial % 3)
            order = np.argsort(-pw)
            for k in range(nk):
                wi = int(order[k]) if rng.random()<0.7 else int(rng.integers(0,272))
                others = np.delete(x, wi, axis=0)
                x[wi] = _find_void(others, rng, n_cand=6000)
        else:
            # single void + small cluster rotation
            G = x @ x.T; np.fill_diagonal(G,-2.0)
            idx = int(G.argmax()); ii = idx//272
            others = np.delete(x, ii, axis=0)
            x[ii] = _find_void(others, rng, n_cand=6000)

        # anneal params cycle
        if trial%4==0:
            tau0,tau1=0.02,0.002; lr0,lr1=0.012,0.001; steps=900
        elif trial%4==1:
            tau0,tau1=0.008,0.0008; lr0,lr1=0.006,0.0006; steps=1100
        elif trial%4==2:
            tau0,tau1=0.03,0.003; lr0,lr1=0.008,0.0008; steps=700
        else:
            tau0,tau1=0.012,0.0012; lr0,lr1=0.01,0.001; steps=900
        # cut steps if low time
        if time.time() > t0+100:
            steps = min(steps,500)
        x = _adam_anneal(x, steps, tau0, tau1, lr0, lr1, rng)
        val = _max_cos(x)
        if val < best_val:
            x_best, best_val = x.copy(), val
            cur, cur_val = x.copy(), val
        elif val < cur_val:
            cur, cur_val = x.copy(), val
        else:
            if rng.random()<0.15:
                cur, cur_val = x.copy(), val
            # occasional reset if wandering too high
            if cur_val > best_val + 0.005 and trial%5==0:
                cur, cur_val = x_best.copy(), best_val
        if time.time() >= deadline-42:
            break

    # final snap: LP polish best
    remain = deadline - time.time()
    if remain > 5:
        x_best = _lp_polish(x_best, min(remain-1, 38), deadline-0.5)
    x_best = _norm(x_best)
    return x_best
