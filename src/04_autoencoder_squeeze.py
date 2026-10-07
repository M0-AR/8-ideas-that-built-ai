"""04 — Squeezing pictures (autoencoder, Hinton & Salakhutdinov Science 2006).

64 -> 32 -> 2 -> 32 -> 64. Target = input. Bottleneck 2 numbers = map point.
Training = backprop through decoder, through bottleneck, into encoder.
Compares vs PCA (straight lines): MSE + 1NN same-digit rate. Numpy + sklearn (data/PCA/1NN).
"""
import numpy as np
SEED=0
def sigm(z): return 1/(1+np.exp(-z))

def load(n_train=1297):
    from sklearn.datasets import load_digits
    from sklearn.model_selection import train_test_split
    D=load_digits(); X=D.data/16.0; y=D.target
    return train_test_split(X,y,test_size=len(X)-n_train,random_state=0,stratify=y)

def train_ae(Xtr,seed=0,epochs=60,lr=0.5,batch=64,h1=32,h2=2):
    rng=np.random.default_rng(seed)
    W1=rng.normal(0,np.sqrt(1/64),(64,h1)); b1=np.zeros(h1)
    W2=rng.normal(0,np.sqrt(1/h1),(h1,h2)); b2=np.zeros(h2)
    W3=rng.normal(0,np.sqrt(1/h2),(h2,h1)); b3=np.zeros(h1)
    W4=rng.normal(0,np.sqrt(1/h1),(h1,64)); b4=np.zeros(64)
    n=len(Xtr)
    for ep in range(epochs):
        perm=rng.permutation(n)
        for i in range(0,n,batch):
            xb=Xtr[perm[i:i+batch]]; m=len(xb)
            a1=xb@W1+b1; h1a=np.maximum(a1,0)
            a2=h1a@W2+b2; code=a2  # linear bottleneck (map coords)
            a3=code@W3+b3; h3=np.maximum(a3,0)
            a4=h3@W4+b4; out=sigm(a4)
            do=(out-xb)/m*sigm(a4)*(1-sigm(a4)) if False else (out-xb)/m*out*(1-out)
            dW4=h3.T@do; db4=do.sum(0)
            dh3=do@W4.T*(a3>0)
            dW3=code.T@dh3; db3=dh3.sum(0)
            dcode=dh3@W3.T
            dh1a=dcode@W2.T
            dW2=h1a.T@dcode; db2=dcode.sum(0)
            dh1=dh1a*(a1>0)
            dW1=xb.T@dh1; db1=dh1.sum(0)
            for P,g in [(W4,dW4),(W3,dW3),(W2,dW2),(W1,dW1)]: P-=lr*g
            b4-=lr*db4; b3-=lr*db3; b2-=lr*db2; b1-=lr*db1
    return (W1,b1,W2,b2,W3,b3,W4,b4)

def encode(X,P):
    W1,b1,W2,b2,_,_,_,_=P; return np.maximum(X@W1+b1,0)@W2+b2
def decode(C,P):
    _,_,_,_,W3,b3,W4,b4=P; return sigm(np.maximum(C@W3+b3,0)@W4+b4)

def run(seed=0,epochs=60):
    from sklearn.decomposition import PCA
    from sklearn.neighbors import KNeighborsClassifier
    Xtr,Xte,ytr,yte=load()
    P=train_ae(Xtr,seed=seed,epochs=epochs)
    Zte=encode(Xte,P); rec=decode(Zte,P)
    mse_ae=float(((Xte-rec)**2).mean())
    pca=PCA(n_components=2).fit(Xtr); Zp=pca.transform(Xte)
    rec_p=pca.inverse_transform(Zp); mse_pca=float(((Xte-rec_p)**2).mean())
    knn=lambda Ztr,Zte_: float(KNeighborsClassifier(1).fit(Ztr,ytr).score(Zte_,yte))
    Ztr=encode(Xtr,P); Zptr=pca.transform(Xtr)
    return {"mse_ae":mse_ae,"mse_pca":mse_pca,"nn_ae":knn(Ztr,Zte),"nn_pca":knn(Zptr,Zp),
            "Zte":Zte,"yte":yte,"P":P,"Xte":Xte}

if __name__=="__main__":
    r=run()
    print(f"AE MSE {r['mse_ae']:.4f} vs PCA {r['mse_pca']:.4f}")
    print(f"1NN same-digit: AE {r['nn_ae']*100:.1f}% vs PCA {r['nn_pca']*100:.1f}%")
