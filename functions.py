import numpy as np
import copy

#####################################################
# Functions Used By Other Funtions Not To Be Called #
#####################################################

def distance(a,b,p=2,debug=False):
    tmp=np.abs(a-b)
    tmp=np.power(tmp,p)
    tmp=np.sum(tmp)
    tmp=np.power(tmp,1/p)
    if debug==True :
        print("p=",p," distance=",tmp)
    return tmp

#######################################################################
# Execute Below Functions Sequentially To Get Cluster Representatives #
# Pass On The Recieved Matrices / Matrix To The Next As Main Input    #
#######################################################################

def distance_matrix(a,debug=False):
    length=a.shape[0]
    d_matrix=np.zeros([length,length])
    for i in range(length):
        for j in range(length):
            d_matrix[i][j]=distance(a[i],a[j])
    if debug==True :
        print("Distance matrix")
        print(d_matrix)
    return d_matrix

def similarity_matrix(a,debug=False):
    s_matrix=np.subtract(1,np.divide(a,a.max()))
    if debug==True :
        print("Similarity matrix")
        print(s_matrix)
    return s_matrix

def noise_estimation(a,p1,p2,p3,debug=False):
    s=a.copy()
    noise=list()
    for i in range(a.shape[0]):
        s[i][i]=0
    if debug==True :
        print(s)
    for i in range(a.shape[0]):
        if s[i].max() < p1 :
            noise.append(i)
    for i in range(a.shape[0]):
        if s[i].sum()/a.shape[0] < p2 :
            noise.append(i)
    for i in range(a.shape[0]):
        for j in range(a.shape[0]):
            if s[i][j] < p3 :
                s[i][j] = 0
            else :
                s[i][j] = 1
    if debug==True :
        print(noise)
        print(s)
    noise=list(set(noise))
    return s,noise

def cluster_represent(a,debug=False):
    length=a.shape[0]
    m_matrix=a.copy()
    represent=list()
    neighbour=np.ones(length)
    if debug==True :
        print("Original matrix")
        print(a)
    while(np.max(m_matrix) != 0):
        for i in range(length):
            neighbour[i]=np.count_nonzero(m_matrix[i])
        pos=np.argmax(neighbour)
        represent.append(pos)
        for i in range(length):
            if m_matrix[i][pos]==1 :
                tmp=i
                for j in range(length):
                    m_matrix[j][tmp]=0
                    m_matrix[tmp][j]=0
            m_matrix[i][pos]=0
            m_matrix[pos][i]=0
        if debug==True :
            print("Neighbour")
            print(neighbour)
            print("Selected")
            print(pos)
            print("Modified matrix")
            print(m_matrix)
            print("Represent List")
            print(represent)
    if debug==True :
        print("Final represent List")
        print(represent)
    return represent

def centroid_find(r,x,b=0.0001,debug=False):
    tmp=x[r]
    c=tmp+b
    if debug==True :
        print("Representatives=",r," Bias=",b)
        print("Centroids")
        print(c)
    return c

#####################################################
# The Below Are Functions To Aid After Kmeans Phase #
#####################################################

def cluster_consensus(a,threshold,debug=False):
    matrix=a.copy()
    k=a.shape[1]
    noise=list()
    clusters=list()
    for i in range(k):
        clusters.append(list())
    for i in range(a.shape[0]):
        pos=matrix[i].argmax()
        s=matrix[i].max()
        matrix[i]=0
        if s >= threshold :
            matrix[i][pos]=1
            clusters[pos].append(i)
        else :
            noise.append(i)
        if debug==True :
            print("Matrix")
            print(matrix)
            print("Sample no ",i)
            print("Noise ",noise)
            print("Position ",pos)
            print("Sum ",s," Threshold ",threshold)
            print("Clusters")
            print(clusters)
    noise=list(set(noise))
    return clusters,noise,matrix

def cluster_merge(c1,c2,debug=False):
    k=len(c1)
    clusters=list()
    noise=list()
    tmp1=list()
    tmp2=list()
    for i in range(k):
        score=0
        pos=0
        for j in range(k):
            tmp_score=len(set.intersection(set(c1[i]),set(c2[j])))
            if tmp_score>score :
                pos=j
                score=tmp_score
        clusters.append(list(set.intersection(set(c1[i]),set(c2[pos]))))
        if debug==True :
            print("Iteration ",i)
            print("Position ",pos," Score ",score)
            print("Clusters")
            print(clusters)
    for i in range(k):
        tmp1.extend(c1[i])
        tmp2.extend(clusters[i])
        if debug==True :
            print("Iteration ",i)
            print("tmp1 ",tmp1," tmp2 ",tmp2)
    noise=list(set.symmetric_difference(set(tmp1),set(tmp2)))
    if debug==True :
        print("Noise ",noise)
    return clusters,noise

def noise_filter(m,n,debug=False):
    not_n=list(set.difference(set(n),set(m)))
    if debug==True :
        print("Marked=",m)
        print("Suspected=",n)
        print("Not Noise=",not_n)
    return not_n

def reassign(clusters,x,left,debug=False):
    k=len(clusters)
    new_clusters=copy.deepcopy(clusters)
    f=x.shape[1]
    centroids=np.zeros([k,f])
    for i in range(k):
        s=np.zeros(f)
        n=0
        for j in clusters[i]:
            s=s+x[j]
            n=n+1
        centroids[i]=s/n
        if debug==True :
            print("Centroids iter=",i)
            print(centroids)
    for i in left:
        cluster_no=0
        distance=(np.abs(centroids[0]-x[i])**2).sum()**(1/2)
        for j in range(k):
            tmp_d=(np.abs(centroids[j]-x[i])**2).sum()**(1/2)
            if tmp_d < distance :
                cluster_no=j
                distance=tmp_d
        new_clusters[cluster_no].append(i)
        if debug==True :
            print("Taking unassigned point ",i)
            print(clusters)
            print(new_clusters)
    return new_clusters

def cluster_matrix_make(r,s,debug=False):
    c=[-1]*s
    for i in range(len(r)):
        for j in r[i] :
            c[j]=i
    return c

#########################################
# Miscellaneous Functions For Extra Use #
#########################################

def unfuzzify(w,debug=False):
    s=w.shape[0]
    new_w=np.zeros(w.shape)
    for i in range(s):
        pos=np.argmax(w[i])
        new_w[i][pos]=1
    if debug==True :
        print("Original weights")
        print(w)
        print("Unfuzzified weights")
        print(new_w)
    return new_w






