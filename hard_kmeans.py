import numpy as np

# X(s,f) C(k,f) W(s,k) 

def kmeans(c,x,i=10000,e=0.05,debug=False):
    flag=False
    if debug==True :
        print("Kmeans started with iterations=",i)
        flag=True
    s=x.shape[0]
    f=x.shape[1]
    k=c.shape[0]
    w=compute_weights(c.copy(),x.copy(),debug=flag)
    pc=c
    while(i>0):
        c=compute_centroids(w.copy(),x.copy(),debug=flag)
        w=compute_weights(c.copy(),x.copy(),debug=flag)
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

def compute_weights(c,x,debug=False):
    s=x.shape[0]
    k=c.shape[0]
    w=np.zeros([s,k])
    d=np.zeros([s,k])
    for i in range(s):
        for j in range(k):
            d[i][j]=np.linalg.norm(x[i]-c[j])
        pos=np.argmin(d[i])
        if debug==True :
            print("Values ",d[i])
        w[i][pos]=1
        if debug==True :
            print("Current Weights")
            print(w)
    if debug==True :
        print("Distance matrix")
        print(d)
        print("Weight matrix")
        print(w)
    return w

def compute_centroids(w,x,m=1,debug=False):
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








