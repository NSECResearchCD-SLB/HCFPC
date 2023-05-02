import numpy as np

# X(s,f) C(k,f) W(s,k) 

def kmeans(c,x,i=10000,m=1.5,p=0.5,m2=2,e=0.05,debug=False):
    flag=False
    if debug==True :
        print("Kmeans started with iterations=",i)
        flag=True
    n=compute_n(c,x,p,m2,debug=flag)
    s=x.shape[0]
    f=x.shape[1]
    k=c.shape[0]
    w=compute_weights(c.copy(),x.copy(),m,n,debug=flag)
    pc=c
    while(i>0):
        #n=compute_n(c,x,p,m2,debug=flag)
        c=compute_centroids(w.copy(),x.copy(),m,debug=flag)
        w=compute_weights(c.copy(),x.copy(),m,n,debug=flag)
        i=i-1
        if np.linalg.norm(pc-c) < e:
            i=0
        pc=c
        if debug==True :
            print("Iterations left ",i)
            print("Centroids")
            print(c)
            print("Weights")
            print(w)
    return w,c

def compute_weights(c,x,m,n,debug=False):
    s=x.shape[0]
    k=c.shape[0]
    w=np.zeros([s,k])
    for i in range(s):
        for j in range(k):
            tmp=np.linalg.norm(x[i]-c[j])/n[j]
            w[i][j]=1/(1+(tmp**(1/(m-1))))
            if debug==True :
                print("Values i=",i," j=",j)
                print("tmp ",tmp)
                print("Current Weights")
                print(w)
    return w

def compute_centroids(w,x,m,debug=False):
    s=x.shape[0]
    f=x.shape[1]
    k=w.shape[1]
    c=np.zeros([k,f])
    for i in range(k):
        denominator=np.zeros(f)
        neumerator=np.zeros(f)
        for j in range(s):
            neumerator=neumerator+((w[j][i]**m)*x[j])
            denominator=denominator+(w[j][i]**m)
            if debug==True :
                print("Values i=",i," j=",j)
                print(" N=",neumerator,"D=",denominator)
        c[i]=neumerator/denominator
        if debug==True :
            print("Values i=",i)
            print("Centroids")
            print(c)
    return c

#################################################

def compute_n(c,x,p,m2,debug=False):
    s=x.shape[0]
    k=c.shape[0]
    d=np.zeros([s,k])
    w=compute_fuzzy_weights(c,x,m2)
    for i in range(s):
        for j in range(k):
            d[i][j]=distance(x[i],c[j])
    n=((w*d).sum(axis=0))/(w.sum(axis=0))
    n=p*n
    if debug==True :
        print("d\n",d,"\nw\n",w,"\nn\n",n)
    return n

def compute_fuzzy_weights(c,x,m,debug=False):
    s=x.shape[0]
    k=c.shape[0]
    w=np.zeros([s,k])
    for i in range(s):
        for j in range(k):
            tmp=np.linalg.norm(x[i]-c[j])
            tmp_sum=0
            for l in range(k):
                tmp1=np.linalg.norm(x[i]-c[l])
                tmp2=(tmp/tmp1)**(2/(m-1))
                tmp_sum=tmp_sum+tmp2
            w[i][j]=1/tmp_sum
            if debug==True :
                print("Values i=",i," j=",j)
                print("tmp ",tmp)
                print("tmp_sum ",tmp_sum)
                print("Current Weights")
                print(w)
    return w

def distance(a,b,p=2,debug=False):
    tmp=np.abs(a-b)
    tmp=np.power(tmp,p)
    tmp=np.sum(tmp)
    tmp=np.power(tmp,1/p)
    if debug==True :
        print("p=",p," distance=",tmp)
    return tmp

#################################################









