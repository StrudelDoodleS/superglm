import sys, math, numpy as np
sys.path.insert(0, sys.argv[1])
import jet_check as J
from jet_check import jmul, jinv, jlog, jvar, jlgamma, U

def log_q_cf_fixed(a0, x, extra):
    """Same Lentz CF but keep iterating `extra` steps after the stop test first passes."""
    K=J.K; A=jvar(a0); tiny=1e-300
    b=-A.copy(); b[0]+=x+1.0
    c=np.zeros(K+1); c[0]=1/tiny; d=jinv(b); h=d.copy(); i=0; passed=None
    while True:
        i+=1
        an=i*A; an[0]=-i*(i-a0)
        b=b.copy(); b[0]+=2.0
        d=jmul(an,d)+b; c=b+jmul(an,jinv(c)); d=jinv(d); de=jmul(d,c); h=jmul(h,de)
        dm1=de.copy(); dm1[0]-=1
        if passed is None and np.all(np.abs(dm1)<=U) and i>2: passed=i
        if passed is not None and i>=passed+extra: break
    lp=A*math.log(x)-jlgamma(a0); lp[0]-=x
    return lp+jlog(h), i

for a,x in [(1e3,1e3+1),(1e4,1e4+1),(1e4,1e4+20),(1e4,1e4+100),(0.01,1.02)]:
    r=J.ref(a,x,"Q"); rv=np.array([float(v) for v in r])
    for extra in (0,50,500,5000):
        lq,n=log_q_cf_fixed(a,x,extra)
        e=np.abs(lq-rv)/np.abs(rv)
        print(f"a={a:<6g} x={x:<8g} extra={extra:<5d} n={n:<5d} relerr by order "+" ".join(f"{v:.1e}" for v in e))
    lp,n=J.log_p_series(a,x)
    lqs=J.jlog1mexp(lp); e=np.abs(lqs-rv)/np.abs(rv)
    print(f"   series-complement n={n}: "+" ".join(f"{v:.1e}" for v in e))
