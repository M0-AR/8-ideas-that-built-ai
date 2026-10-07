"""03 — A network that dreams (Boltzmann machine / RBM, Hinton & Sejnowski 1985; CD Hinton 2002).

RBM: 64 visible (8x8 pixels) + 64 hidden, binary stochastic units.
Energy E(v,h) = -vW h - b v - c h. Learning: CD-1
  dW = <v0 p(h|v0)> - <v1 p(h|v1)>  ("awake minus asleep").
Reports: redraw MSE, energies (real vs noise), inpaint bottom-half + judge,
dreams from noise + judge confidence. Numpy + sklearn (data + judge) only.
"""
import numpy as np
SEED=0

def load_digits01(n_train=1297):
    from sklearn.datasets import load_digits
    from sklearn.model_selection import train_test_split
    D=load_digits(); X=D.data/16.0; y=D.target
    Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=len(X)-n_train,random_state=0,stratify=y)
    return (Xtr,ytr),(Xte,yte)

def sigm(z): return 1/(1+np.exp(-z))

class RBM:
    def __init__(self,nv=64,nh=64,seed=0):
        rng=np.random.default_rng(seed)
        self.W=rng.normal(0,0.01,(nv,nh)); self.bv=np.zeros(nv); self.bh=np.zeros(nh)
    def p_h(self,v): return sigm(v@self.W+self.bh)
    def p_v(self,h): return sigm(h@self.W.T+self.bv)
    def energy(self,v,h=None):
        if h is None: h=(self.p_h(v)>0.5).astype(float)
        return float((-(v@self.W*h.T).sum() if v.ndim==1 else -(v@self.W*h.swapaxes(0,1) if False else 0)) ) if False else float(-(v@self.W@h.T).sum()-self.bv@v-self.bh@h if v.ndim==1 else (-(v@self.W*h).sum(1)-v@self.bv-(h@self.bh)).mean())
    def train_cd1(self,X,epochs=15,lr=0.1,batch=64,seed=0):
        rng=np.random.default_rng(seed); n=len(X); mse0=None
        for ep in range(epochs):
            perm=rng.permutation(n)
            for i in range(0,n,batch):
                v0=X[perm[i:i+batch]]
                ph0=self.p_h(v0); h0=(rng.random(ph0.shape)<ph0).astype(float)
                pv1=self.p_v(h0); v1=(rng.random(pv1.shape)<pv1).astype(float)
                ph1=self.p_h(v1)
                self.W+=lr*((v0.T@ph0)-(v1.T@ph1))/len(v0)
                self.bv+=lr*(v0-v1).mean(0); self.bh+=lr*(ph0-ph1).mean(0)
            mse=self.redraw_mse(X[:300])
            if ep==0: mse0=mse
        return mse0, mse
    def redraw_mse(self,X):
        v1=self.p_v(self.p_h(X)); return float(((X-v1)**2).mean())
    def dream(self,rng,steps=50,n=1):
        v=(rng.random((n,64))<0.5).astype(float)
        for _ in range(steps):
            h=(rng.random((n,64))<self.p_h(v)).astype(float)
            v=(rng.random((n,64))<self.p_v(h)).astype(float)
        return v

def judge_score(Xtr,ytr,Xte):
    from sklearn.linear_model import LogisticRegression
    clf=LogisticRegression(max_iter=1000).fit(Xtr,ytr)
    prob=clf.predict_proba(Xte); pred=prob.argmax(1)
    conf=prob.max(1); return clf,pred,conf

def run(seed=0,epochs=30):
    rng=np.random.default_rng(seed)
    (Xtr,ytr),(Xte,yte)=load_digits01()
    Xb=(Xtr>0.5).astype(float); Xt=(Xte>0.5).astype(float)
    rbm=RBM(seed=seed)
    mse0,mse1=rbm.train_cd1(Xb,epochs=epochs,seed=seed)
    e_real=np.mean([rbm.energy(v) for v in Xb[:50]])
    noise=(rng.random((50,64))<0.5).astype(float)
    e_noise=np.mean([rbm.energy(v) for v in noise])
    # inpaint: hold top half, resample bottom half
    clf,_,_=judge_score(Xtr,ytr,Xte)
    blank=Xte.copy(); blank[:,32:]=0
    acc_blank=float(clf.score(blank,yte))
    repaired=[]
    for v in Xt[:200]:
        vv=v.copy()
        for _ in range(30):
            h=(rng.random(64)<rbm.p_h(vv[None,:])[0]).astype(float)
            pv=rbm.p_v(h[None,:])[0]
            new=(rng.random(64)<pv)
            vv=np.concatenate([v[:32],new[32:]]).astype(float)
        repaired.append(vv)
    repaired=np.array(repaired)
    # map back to [0,1] intensities for judge (binarized-> judge trained on intensities; use 0/1 as proxy)
    acc_fix=float(clf.score(repaired,yte[:200]))
    # dreams
    dreams=rbm.dream(rng,steps=60,n=100)
    _,dpred,dconf=judge_score(Xtr,ytr,dreams)
    confident=int((dconf>0.9).sum()); digits_found=int(len(set(dpred[dconf>0.9].tolist())))
    return {"mse0":mse0,"mse1":mse1,"e_real":e_real,"e_noise":e_noise,
            "acc_blank":acc_blank,"acc_fix":acc_fix,
            "dreams_confident":confident,"dreams_digits":digits_found,"rbm":rbm}

if __name__=="__main__":
    r=run()
    print(f"redraw MSE {r['mse0']:.4f} -> {r['mse1']:.4f}")
    print(f"energy real {r['e_real']:.1f} noise {r['e_noise']:.1f}")
    print(f"judge blank {r['acc_blank']*100:.1f}% -> repaired {r['acc_fix']*100:.1f}%")
    print(f"dreams: {r['dreams_confident']}/100 confident, {r['dreams_digits']}/10 digits")
