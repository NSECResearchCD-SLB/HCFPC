
##########################################################################
#                                                                        #
#   █████████    █████████  ███████████   █████ ███████████  ███████████ #
#  ███░░░░░███  ███░░░░░███░░███░░░░░███ ░░███ ░░███░░░░░███░█░░░███░░░█ #
# ░███    ░░░  ███     ░░░  ░███    ░███  ░███  ░███    ░███░   ░███  ░  #
# ░░█████████ ░███          ░██████████   ░███  ░██████████     ░███     #
#  ░░░░░░░░███░███          ░███░░░░░███  ░███  ░███░░░░░░      ░███     #
#  ███    ░███░░███     ███ ░███    ░███  ░███  ░███            ░███     #
# ░░█████████  ░░█████████  █████   █████ █████ █████           █████    #
#  ░░░░░░░░░    ░░░░░░░░░  ░░░░░   ░░░░░ ░░░░░ ░░░░░           ░░░░░     #
#                                                                        #
#                                 ########################################
# This File Is The Custom Script #
# For Bootstrapping All The      #
# Functions Together And Running #
# The Whole Algorithm With The   #
# Various Versions Of Kmeans     #
#                                #
##################################

###############################
import sys
import pandas as pd
import numpy as np
import sklearn.metrics
###############################
DATA=sys.argv[1]
df=pd.read_csv("../data/"+DATA+".csv")
labels=df.iloc[:,-1]
data=df.iloc[:,:-1]
###############################
tmp1=float(sys.argv[2])
tmp2=float(sys.argv[3])
tmp3=float(sys.argv[4])
tmp4=float(sys.argv[5])
tmp5=int(sys.argv[6])
tmp6=int(sys.argv[7])
tmp7=float(sys.argv[8])
###############################
exec(open("../functions.py").read())
y=np.array(labels)
x=np.array(data)
x=x/x.max(axis=0)
d=distance_matrix(x)
s=similarity_matrix(d)
m,n=noise_estimation(s,tmp1,tmp2,tmp3)
r=cluster_represent(m)
c=centroid_find(r[:tmp6],x)
###############################
exec(open("../possibilistic_kmeans.py").read())
weights0,centroids0=kmeans(c,x,i=tmp5,e=tmp7)
clusters0,noise0,weights_clipped0=cluster_consensus(weights0,tmp4)
noise=list(set.intersection(set(noise0),set(n)))
###############################
noise_free_indices=list(set.difference(set(range(x.shape[0])),set(noise)))
new_x=x[noise_free_indices]
new_y=y[noise_free_indices]
###############################
exec(open("../hard_kmeans.py").read())
weights1,centroids1=kmeans(c,new_x,i=tmp5,e=tmp7)
clusters1,noise1,weights_clipped1=cluster_consensus(weights1,0)
prediction1=cluster_matrix_make(clusters1,new_x.shape[0])
###############################
exec(open("../fuzzy_kmeans.py").read())
weights2,centroids2=kmeans(c,new_x,i=tmp5,e=tmp7)
clusters2,noise2,weights_clipped2=cluster_consensus(weights2,0)
prediction2=cluster_matrix_make(clusters2,new_x.shape[0])
###############################
exec(open("../possibilistic_kmeans.py").read())
weights3,centroids3=kmeans(c,new_x,i=tmp5,e=tmp7)
clusters3,noise3,weights_clipped3=cluster_consensus(weights3,0)
prediction3=cluster_matrix_make(clusters3,new_x.shape[0])
###############################
"""
THIS SECTION DENOTES THE COUSENSUS OF CLUSTERS
"""
tmp,left=cluster_merge(clusters1,clusters2)
clusters_tmp,tmp=cluster_merge(tmp,clusters3)
left.extend(tmp)
clusters=reassign(clusters_tmp,new_x,left)
prediction=cluster_matrix_make(clusters,new_x.shape[0])
###############################
ret=list()
ret.append(sklearn.metrics.rand_score(new_y,prediction))
ret.append(sklearn.metrics.davies_bouldin_score(new_x,prediction))
ret.append(sklearn.metrics.silhouette_score(new_x,prediction))
sys.exit(ret)
###############################
pass
