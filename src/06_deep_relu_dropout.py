"""06 — Deep networks: vanishing blame, ReLU (Nair & Hinton ICML 2010), dropout (2012).

A) 8 hidden layers of sigmoid: blame shrinks ~4x per layer -> first layers learn nothing (~37%).
   Same 8 layers with ReLU: blame survives -> ~95%.
B) Spoiled labels (~1/3 wrong): wide net memorizes (99.9% agreement w/ spoiled, ~70% clean test).
   Same + dropout(p=0.5): 91% agreement, ~84% clean test (+14pts).
Numpy MLPs on sklearn digits. Small epochs for <1min total.
"""
import numpy as np
SEED=0
def load():
    from sklearn.datasets import load_digits
    from sklearn.model_selection import train_test_split
    D=load_digits(); X=D.data/16.0; y=D.target
    Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=500,random_state=0,stratify=y)
    return Xtr,ytr,Xte,yte

def onehot(y,k=10):
    O=np.zeros((len(y),k)); O[np.arange(len(y)),y]=1; return O

def train_mlp(Xtr,ytr,Xte,yte,hidden=(64,)*8,act="sigmoid",drop=0.0,epochs=30,lr=0.1,seed=0,spoil=None):
    rng=np.random.default_rng(seed)
    if spoil is not None:
        ytr=np.array([rng.integers(10) if rng.random()<spoil else t for t in ytr])
    Ytr=onehot(ytr); n,d=Xtr.shape; k=10
    dims=[d]+list(hidden)+[k]
    W=[rng.normal(0,np.sqrt(2/sum(dims[i:i+2])),(dims[i],dims[i+1])) for i in range(len(dims)-1)]
    b=[np.zeros(dims[i+1]) for i in range(len(dims)-1)]
    act_fn=(lambda z: 1/(1+np.exp(-z))) if act=="sigmoid" else (lambda z: np.maximum(z,0))
    def fwd(X,train=True):
        A=[X]; masks=[]
        for i in range(len(W)-1):
            Z=A[-1]@W[i]+b[i]; H=act_fn(Z)
            if drop>0 and train:
                m=(rng.random(H.shape)>(drop)).astype(float)/(1-drop); H*=m; masks.append(m)
            else: masks.append(None)
            A.append(H)
        Z=A[-1]@W[-1]+b[-1]; Z-=Z.max(1,keepdims=True); E=np.exp(Z); O=E/E.sum(1,keepdims=True)
        return A,O,masks
    blame_ratio=None
    for ep in range(epochs):
        perm=rng.permutation(n)
        for i in range(0,n,128):
            xb=Xtr[perm[i:i+128]]; yb=Ytr[perm[i:i+128]]; m=len(xb)
            A,O,masks=fwd(xb,True)
            d=O.copy(); d[np.arange(m),yb.argmax(1)]-=1; d/=m
            # output blame size
            out_blame=float(np.abs(d).mean())
            grads=[]
            dd=d
            for iL in reversed(range(len(W))):
                gW=A[iL].T@dd; gb=dd.sum(0)
                grads.append(float(np.abs(dd).mean()))
                if iL>0:
                    dd=(dd@W[iL].T)
                    if act=="sigmoid":
                        H=A[iL]; dd*=H*(1-H)
                    else: dd*=(A[iL]>0)
                    if masks[iL-1] is not None: dd*=masks[iL-1]
                W[iL]-=lr*gW; b[iL]-=lr*gb
            if ep==0 and blame_ratio is None:
                blame_ratio=grads[-1]/max(grads[0],1e-12)  # first-layer / last-layer
        # end epoch
    def acc(X,y):
        _,O,_=fwd(X,False); return float((O.argmax(1)==y).mean())
    # final blame sizes per layer (one batch, no dropout)
    A,O,_=fwd(Xtr[:256],False); d=O.copy(); Yb=onehot(ytr[:256]); d[np.arange(256),Yb.argmax(1)]-=1; d/=256
    sizes=[]; dd=d
    for iL in reversed(range(len(W))):
        sizes.append(float(np.abs(dd).mean())); 
        if iL>0:
            dd=dd@W[iL].T
            dd*= (A[iL]*(1-A[iL]) if act=="sigmoid" else (A[iL]>0))
    sizes=list(reversed(sizes))
    return {"acc_test":acc(Xte,yte),"acc_spoiled_train":acc(Xtr,ytr),
            "blame_first":sizes[0],"blame_last":sizes[-1],"blame_ratio":sizes[0]/max(sizes[-1],1e-12)}

def run(epochs=30):
    Xtr,ytr,Xte,yte=load()
    sig=train_mlp(Xtr,ytr,Xte,yte,act="sigmoid",epochs=epochs,lr=0.1)
    rel=train_mlp(Xtr,ytr,Xte,yte,act="relu",epochs=epochs,lr=0.1)
    mem=train_mlp(Xtr,ytr,Xte,yte,hidden=(256,256),act="relu",epochs=epochs,lr=0.1,spoil=1/3,seed=1)
    do=train_mlp(Xtr,ytr,Xte,yte,hidden=(256,256),act="relu",drop=0.5,epochs=epochs,lr=0.1,spoil=1/3,seed=1)
    return {"sig":sig,"relu":rel,"mem":mem,"do":do}

if __name__=="__main__":
    r=run()
    print(f"sigmoid8 blame first {r['sig']['blame_first']:.2e} last {r['sig']['blame_last']:.2e} ratio {r['sig']['blame_ratio']:.2e} acc {r['sig']['acc_test']*100:.1f}%")
    print(f"relu8    blame first {r['relu']['blame_first']:.2e} last {r['relu']['blame_last']:.2e} ratio {r['relu']['blame_ratio']:.2e} acc {r['relu']['acc_test']*100:.1f}%")
    print(f"memorize spoiled-train {r['mem']['acc_spoiled_train']*100:.1f}% clean-test {r['mem']['acc_test']*100:.1f}%")
    print(f"dropout  spoiled-train {r['do']['acc_spoiled_train']*100:.1f}% clean-test {r['do']['acc_test']*100:.1f}%")
