"""02 — Meaning as numbers / embeddings (Hinton 1986 family trees). v2 fixed.

Fixes: gender-consistent naming, 12 relations (~100+ questions),
stratified random hidden split, longer training.
"""
import numpy as np
SEED=0
EN_M=["James","Charles","Colin","Andrew","Arthur","Christopher"]
EN_F=["Penelope","Victoria","Jennifer","Margaret","Charlotte","Elizabeth"]
IT_M=["Marco","Pierro","Luca","Paolo","Carlo","Roberto"]
IT_F=["Maria","Sofia","Elena","Giulia","Anna","Lucia"]
MALE_ROLES={0,2,4,6,8,10}
RELATIONS=["father","mother","husband","wife","son","daughter","brother","sister","uncle","aunt","nephew","niece"]

def names():
    en=[None]*12; it=[None]*12
    mi=fi=0
    for r in range(12):
        if r in MALE_ROLES: en[r]=EN_M[mi]; it[r]=IT_M[mi]; mi+=1
        else: en[r]=EN_F[fi]; it[r]=IT_F[fi]; fi+=1
    return en+it

def build_questions():
    people=names()
    father_of={2:0,4:0,6:2,7:2,8:4,9:4}
    mother_of={2:1,4:1,6:3,7:3,8:5,9:5}
    husband_of={1:0,3:2,5:4,11:10}
    wife_of={0:1,2:3,4:5,10:11}
    sons={0:[2,4],2:[6],4:[8],3:[6],5:[8]}
    daughters={2:[7],4:[9],3:[7],5:[9],0:[],1:[]}
    # siblings (eldest brother / eldest sister)
    brothers={2:[4],4:[2],6:[8],8:[6]}
    sisters={7:[9],9:[7]}
    # uncle: father's brother (for C-generation); aunt: father's sister-in-law -> M2 for C1/C2 etc.
    uncle={6:[4],7:[4],8:[2],9:[2]}
    aunt={6:[5],7:[5],8:[3],9:[3]}
    nephew={2:[6,8],4:[6,8],0:[6,8],10:[]}
    niece={2:[7,9],4:[7,9]}
    Q=[]
    for fam in range(2):
        off=fam*12
        for local in range(12):
            if local in father_of: Q.append((off+local,"father",off+father_of[local]))
            if local in mother_of: Q.append((off+local,"mother",off+mother_of[local]))
            if local in husband_of: Q.append((off+local,"husband",off+husband_of[local]))
            if local in wife_of: Q.append((off+local,"wife",off+wife_of[local]))
            if local in sons and sons[local]: Q.append((off+local,"son",off+sons[local][0]))
            if local in daughters and daughters[local]: Q.append((off+local,"daughter",off+daughters[local][0]))
            if local in brothers and brothers[local]: Q.append((off+local,"brother",off+brothers[local][0]))
            if local in sisters and sisters[local]: Q.append((off+local,"sister",off+sisters[local][0]))
            if local in uncle and uncle[local]: Q.append((off+local,"uncle",off+uncle[local][0]))
            if local in aunt and aunt[local]: Q.append((off+local,"aunt",off+aunt[local][0]))
            if local in nephew and nephew[local]: Q.append((off+local,"nephew",off+nephew[local][0]))
            if local in niece and niece[local]: Q.append((off+local,"niece",off+niece[local][0]))
    seen=set(); U=[]
    for q in Q:
        if q not in seen: seen.add(q); U.append(q)
    return U, people

def softmax(z):
    z=z-z.max(axis=1,keepdims=True); e=np.exp(z); return e/e.sum(axis=1,keepdims=True)

def train(seed=0, hidden_n=4, epochs=4000, lr=0.7, hidden_idx=None):
    rng=np.random.default_rng(seed+999)
    Q, people=build_questions()
    rel_index={r:i for i,r in enumerate(RELATIONS)}
    R=len(RELATIONS)
    rng_pick=np.random.default_rng(seed)
    if hidden_idx is None:
        hidden_idx=rng_pick.choice(len(Q),hidden_n,replace=False)
    test_q=[Q[i] for i in hidden_idx]; train_q=[Q[i] for i in range(len(Q)) if i not in set(hidden_idx.tolist() if hasattr(hidden_idx,'tolist') else hidden_idx)]
    Ep=rng.normal(0,0.5,(24,6)); Er=rng.normal(0,0.5,(R,8))
    W1=rng.normal(0,0.4,(14,20)); b1=np.zeros(20)
    W2=rng.normal(0,0.4,(20,24)); b2=np.zeros(24)
    def fwd(p_idx,r_idx):
        e=np.concatenate([Ep[p_idx],Er[r_idx]],axis=1)
        h=np.tanh(e@W1+b1); o=softmax(h@W2+b2); return e,h,o
    P=np.array([q[0] for q in train_q]); Rr=np.array([rel_index[q[1]] for q in train_q]); Y=np.array([q[2] for q in train_q])
    N=len(P)
    for ep in range(epochs):
        e,h,o=fwd(P,Rr)
        do=o.copy(); do[np.arange(N),Y]-=1; do/=N
        dW2=h.T@do; db2=do.sum(0)
        dh=do@W2.T*(1-h**2)
        dW1=e.T@dh; db1=dh.sum(0)
        de=dh@W1.T
        dEp=np.zeros_like(Ep); dEr=np.zeros_like(Er)
        np.add.at(dEp,P,de[:,:6]); np.add.at(dEr,Rr,de[:,6:])
        W2-=lr*dW2; b2-=lr*db2; W1-=lr*dW1; b1-=lr*db1; Ep-=lr*dEp; Er-=lr*dEr
    _,_,o=fwd(P,Rr); train_acc=float((o.argmax(1)==Y).mean())
    Pt=np.array([q[0] for q in test_q]); Rt=np.array([rel_index[q[1]] for q in test_q]); Yt=np.array([q[2] for q in test_q])
    _,_,ot=fwd(Pt,Rt); pt=ot.argmax(1); hid_ok=int((pt==Yt).sum())
    twins=0
    for i in range(12):
        # compare english global i vs italians 12..23 by role: english global i is role i (since en[role]=global role). italian role j is global 12+j.
        d=((Ep[12:]-Ep[i])**2).sum(1); nn=int(d.argmin())
        if nn==i: twins+=1
    axis=(Ep[:12].mean(0)-Ep[12:].mean(0)); axis/= (np.linalg.norm(axis)+1e-9)
    proj=Ep@axis; sep=float(proj[:12].min()-proj[12:].max())
    # PC1 family gap (means) + twins after removing PC1 (video's "take that direction away")
    Ec=Ep-Ep.mean(0)
    try:
        _,_,Vt=np.linalg.svd(Ec,full_matrices=False); pc1=Vt[0]/(np.linalg.norm(Vt[0])+1e-9)
    except Exception:
        pc1=axis
    p1=Ec@pc1; pc1_gap=float(abs(p1[:12].mean()-p1[12:].mean()))
    E2=Ep-np.outer(Ep@pc1,pc1)
    twins_nopc1=sum(int((((E2[12:]-E2[i])**2).sum(1)).argmin())==i for i in range(12))
    return {"n_questions":len(Q),"n_train":N,"train_acc":train_acc,"hidden_ok":hid_ok,
            "hidden_detail":[(people[p],r,people[y],people[pp]) for (p,r,y),pp in zip(test_q,pt.tolist())],
            "twins":twins,"sep":sep,"Ep":Ep,"people":people,"hidden_idx":hidden_idx,
            "twins_nopc1":twins_nopc1,"pc1_gap":pc1_gap}

if __name__=="__main__":
    qs,_=build_questions(); print(f"questions={len(qs)}")
    r=train(seed=SEED); print(f"train acc={r['train_acc']*100:.1f}% hidden={r['hidden_ok']}/4")
    for p,rel,y,ph in r["hidden_detail"]:
        print(f"  {p} has-{rel} -> {ph} (true {y}) {'OK' if ph==y else 'MISS'}")
    print(f"twins {r['twins']}/12 sep={r['sep']:.3f}")
    hs=[]; tw=[]
    for s in range(10):
        rr=train(seed=s); hs.append(rr["hidden_ok"]); tw.append(rr["twins"])
    import numpy as np
    print(f"10-seed avg hidden={np.mean(hs):.2f}/4 twins={np.mean(tw):.1f}/12")
