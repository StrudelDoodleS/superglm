"""v2: switch line x = a + max(1, sqrt(a)); CF stop on per-coefficient relative change of h."""
import sys, math, numpy as np
sys.path.insert(0, sys.argv[1])
import jet_check as J
from jet_check import jmul, jinv, jlog, jlgamma, U

def jvar(a0):
    f=np.zeros(J.K+1); f[0]=a0
    if J.K>=1: f[1]=1.0
    return f
J.jvar=jvar

def log_q_cf2(a0, x):
    K=J.K; A=jvar(a0); tiny=1e-300
    b=-A.copy(); b[0]+=x+1.0
    c=np.zeros(K+1); c[0]=1/tiny; d=jinv(b); h=d.copy(); i=0
    while True:
        i+=1
        an=i*A; an[0]=-i*(i-a0)
        b=b.copy(); b[0]+=2.0
        d=jmul(an,d)+b; c=b+jmul(an,jinv(c)); d=jinv(d); de=jmul(d,c)
        hn=jmul(h,de)
        done = i>2 and np.all(np.abs(hn-h) <= U*np.abs(hn))
        h=hn
        if done: break
    lp=A*math.log(x)-jlgamma(a0); lp[0]-=x
    return lp+jlog(h), i

def log_pq2(a0,x):
    if x <= a0 + max(1.0, math.sqrt(a0)):
        lp,n=J.log_p_series(a0,x); return lp, J.jlog1mexp(lp), ("series",n)
    lq,n=log_q_cf2(a0,x); return J.jlog1mexp(lq), lq, ("cf",n)

pts = J.POINTS + [(0.01,0.9),(0.01,1.02),(0.5,1.49),(0.5,1.51),(30.0,30.99),(30.0,36.0),(1e3,1e3+1),
                  (1e3,1e3+32),(1e3,1e3+33),(1e4,1e4+1),(1e4,1e4+20),(1e4,1e4+99),(1e4,1e4+101)]
worst=np.zeros(J.K+1)
for a,x in pts:
    lp,lq,(m,n)=log_pq2(a,x)
    for which,jet in (("P",lp),("Q",lq)):
        rv=np.array([float(v) for v in J.ref(a,x,which)])
        e=np.where(rv!=0, np.abs(jet-rv)/np.where(rv!=0,np.abs(rv),1), np.abs(jet))
        worst=np.maximum(worst,e)
        if e.max()>1e-12: print(f"  a={a:<7g} x={x:<9g} {m:6s} n={n:<4d} {which} relerr "+" ".join(f"{v:.1e}" for v in e))
print("v2 worst relerr by order over", len(pts), "points x {P,Q}:", " ".join(f"{w:.1e}" for w in worst))
print("term counts value-only K=0 / K=2 / K=4:")
for a,x in [(0.1,3.0),(10.0,8.0),(10.0,12.0),(100.0,100.0),(100.0,111.0),(1e3,1e3),(1e3,1050.0),(1e4,1e4),(1e4,1.015e4)]:
    ns=[]
    for K in (0,2,4):
        J.K=K; _,_,(m,n)=log_pq2(a,x); ns.append(n)
    J.K=4
    print(f"  a={a:<7g} x={x:<9g} {m:6s} {ns[0]:4d} {ns[1]:4d} {ns[2]:4d}")
