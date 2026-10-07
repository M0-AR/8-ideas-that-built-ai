"""07 — Dark knowledge / distillation (Hinton, Vinyals & Dean 2015).

Teacher (big, ~85k weights -> here 64-256-128-10) reads ~98% test.
Student 35x smaller (64-32-10, ~2.4k) never sees any '3' (126 pics removed).
- hard labels: 0/57 threes (calls many 9).
- soft targets (T=8): teacher's 'slightly-three' hints on 8s/5s/9s teach the unseen digit.
Numpy MLPs + softmax/T.
"""
import numpy as np
SEED=0
def load():
    from sklearn.datasets import load_digits
    from sklearn.model_selection import train_test_split
    D=load_digits(); X=D.data/16.0; y=D.target
    return train_test_split(X,y,test_size=500,random_state=0,stratify=y)

def softmax(z): z=z-z.max(1,keepdims=True); e=np.exp(z); return e/e.sum(1,keepdims=True)

def train(Xtr,ytr,Xte,yte,hid=(256,128),epochs=30,lr=0.2,seed=0,soft=None,T=8.0,remove3=False):
    rng=np.random.default_rng(seed)
    if remove3:
        keep=ytr!=3; Xtr,ytr=Xtr[keep],ytr[keep]
    n=len(Xtr)
    dims=[64]+list(hid)+[10]
    W=[rng.normal(0,np.sqrt(2/(dims[i]+dims[i+1])),(dims[i],dims[i+1])) for i in range(len(dims)-1)]
    b=[np.zeros(dims[i+1]) for i in range(len(dims)-1)]
    Yh=np.zeros((n,10)); Yh[np.arange(n),ytr]=1
    if soft is not None:  # soft = teacher probs at T on Xtr
        Yt=soft
    for ep in range(epochs):
        perm=rng.permutation(n)
        for i in range(0,n,128):
            xb=Xtr[perm[i:i+128]]; m=len(xb)
            A=[xb]
            for j in range(len(W)-1):
                A.append(np.maximum(A[-1]@W[j]+b[j],0))
            Z=A[-1]@W[-1]+b[-1]; O=softmax(Z)
            if soft is None: tgt=Yh[perm[i:i+128]]
            else: tgt=Yt[perm[i:i+128]]
            d=(O-tgt)/m
            grads=[]
            dd=d
            for j in reversed(range(len(W))):
                gW=A[j].T@dd; gb=dd.sum(0)
                if j>0:
                    dd=(dd@W[j].T)*(A[j]>0)
                W[j]-=lr*gW; b[j]-=lr*gb
    def predict(X):
        A=X
        for j in range(len(W)-1): A=np.maximum(A@W[j]+b[j],0)
        return softmax(A@W[-1]+b[-1])
    return W,b,predict

def run(epochs=30,T=8.0):
    Xtr,Xte,ytr,yte=load()
    Wt,bt,pt=train(Xtr,ytr,Xte,yte,hid=(256,128),epochs=epochs,seed=0)
    acc_t=float((pt(Xte).argmax(1)==yte).mean())
    # teacher soft on no-3 subset
    keep=ytr!=3; Xs=Xtr[keep]; ys=ytr[keep]
    # teacher logits at T: recompute via predict then sharpen? approximate soft = softmax(log(p)*? ) — use pt probs tempered:
    p_teacher=pt(Xs)
    # re-temper: pT_i ∝ p_i^(1/T) (equivalent to dividing logits by T up to constant)
    pT=(p_teacher**(1.0/T)); pT/=pT.sum(1,keepdims=True)
    _,_,ps_hard=train(Xtr,ytr,Xte,yte,hid=(32,),epochs=epochs,seed=1,remove3=True)
    _,_,ps_soft=train(Xtr,ytr,Xte,yte,hid=(32,),epochs=epochs,seed=1,remove3=True,soft=pT,T=T)
    mask3=yte==3; mno3=~mask3
    def stats(p):
        P=p(Xte); pr=P.argmax(1)
        return float((pr[mno3]==yte[mno3]).mean()), float((pr[mask3]==3).mean()), pr[mask3]
    acc_other_h,acc3_h,pred3_h=stats(ps_hard)
    acc_other_s,acc3_s,pred3_s=stats(ps_soft)
    # example: teacher view of a '2' and of a '9' (find first test 2 and 9)
    i2=np.where(yte==2)[0][0]; i9=np.where(yte==9)[0][0]
    ex2=pT[np.where(keep)[0].tolist().index(np.where((Xtr==Xte[i2]).all(1))[0][0])] if False else pt(Xte[i2:i2+1])
    return {"acc_teacher":acc_t,"hard_other":acc_other_h,"hard_3":acc3_h,
            "soft_other":acc_other_s,"soft_3":acc3_s,
            "n_threes_test":int(mask3.sum()),"pred3_hard":pred3_h[:10].tolist(),"pred3_soft":pred3_s[:10].tolist(),
            "T":T,"n_params_teacher":64*256+256+256*128+128+128*10+10,
            "n_params_student":64*32+32+32*10+10}

if __name__=="__main__":
    r=run()
    print(f"teacher {r['acc_teacher']*100:.1f}% params={r['n_params_teacher']} student params={r['n_params_student']}")
    print(f"hard student: other {r['hard_other']*100:.1f}% threes {r['hard_3']*100:.1f}% (n={r['n_threes_test']})")
    print(f"soft student: other {r['soft_other']*100:.1f}% threes {r['soft_3']*100:.1f}%")
