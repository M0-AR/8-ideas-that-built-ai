"""05 — A map of data (t-SNE, van der Maaten & Hinton JMLR 2008).

Neighbors in 64-D become probabilities; same on sheet; pull/push by KL.
Heavy-tailed Student-t on sheet lets clusters spread (the 't').
Numpy exact t-SNE, 1000 digits, PCA-30 pre-step, early exaggeration.
Reports 1NN same-digit: raw 64-D vs sheet vs PCA-2. Warns: trust neighbors,
not island sizes/gaps.
"""
import numpy as np
SEED=0

def load1000():
    from sklearn.datasets import load_digits
    from sklearn.model_selection import train_test_split
    D=load_digits(); X=D.data/16.0; y=D.target
    from sklearn.decomposition import PCA
    X30=PCA(n_components=30,random_state=0).fit_transform(X)
    idx=np.random.default_rng(0).choice(len(X),1000,replace=False)
    return X[idx],y[idx],X30[idx]

def joint_P(X,perp=30):
    n=len(X); D2=((X[:,None,:]-X[None,:,:])**2).sum(-1)
    P=np.zeros((n,n))
    for i in range(n):
        beta=1.0; lo,hi=-50,50
        for _ in range(50):
            w=np.exp(-D2[i]*beta); w[i]=0; s=w.sum()
            if s==0: P[i]=0; break
            p=w/s; H=-(p[p>0]*np.log2(p[p>0]+1e-12)).sum()
            if abs(H-np.log2(perp))<1e-5: P[i]=p; break
            if H>np.log2(perp): lo=beta; beta=(beta*2 if hi==-50 else (beta+hi)/2)
            else: hi=beta; beta=(beta/2 if lo==-50 else (beta+lo)/2)
        else: P[i]=p
    P=(P+P.T)/(2*n); return np.maximum(P,1e-12)

def tsne(X30,steps=300,seed=0,lr=200.0):
    rng=np.random.default_rng(seed); n=len(X30)
    P=joint_P(X30)*4.0  # early exaggeration
    Y=rng.normal(0,1e-4,(n,2)); dY=np.zeros_like(Y)
    for it in range(steps):
        D2=((Y[:,None,:]-Y[None,:,:])**2).sum(-1)
        Q=(1+D2)**-1; np.fill_diagonal(Q,0); Q/=Q.sum()
        G=4*((P-Q)*(1+D2)**-1)[:,:,None]*(Y[:,None,:]-Y[None,:,:])
        g=G.sum(1)
        mom=0.5 if it<150 else 0.8
        dY=mom*dY-lr*g; Y+=dY
        Y-=Y.mean(0)
        if it==150: P/=4.0
    return Y

def knn1(Z,y):
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.model_selection import cross_val_predict, StratifiedKFold
    # use 5-fold CV on the 1000 to avoid train-test leak optimism; simpler: leave-one-out approx via 5fold
    pred=cross_val_predict(KNeighborsClassifier(1),Z,y,cv=5)
    return float((pred==y).mean())

def run(steps=300):
    from sklearn.decomposition import PCA
    X,y,X30=load1000()
    Y=tsne(X30,steps=steps)
    Zp=PCA(n_components=2,random_state=0).fit_transform(X)
    return {"nn_raw":knn1(X,y),"nn_tsne":knn1(Y,y),"nn_pca":knn1(Zp,y),"Y":Y,"y":y}

if __name__=="__main__":
    r=run()
    print(f"1NN same-digit: raw {r['nn_raw']*100:.1f}% tsne {r['nn_tsne']*100:.1f}% pca2 {r['nn_pca']*100:.1f}%")
    print("note: trust who is next to whom; do NOT trust island sizes or gaps")
