
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

#IMPORTS

import os
import time
import pandas as pd
import numpy as np
import sklearn.metrics
from jqmcvi import base

#FUNCTIONS

def sleep():
    time.sleep(1)
    #os.system("clear")

#START

sleep()
print("""
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
#                                                    BY  AISHWARYA  ...  #
##########################################################################
""")
sleep()

###############################
print("   _____ ______ ___     ____  ______ ____ _   __ ______             ")
print("  / ___//_  __//   |   / __ \/_  __//  _// | / // ____/             ")
print("  \__ \  / /  / /| |  / /_/ / / /   / / /  |/ // / __               ")
print(" ___/ / / /  / ___ | / _, _/ / /  _/ / / /|  // /_/ /  _    _    _  ")
print("/____/ /_/  /_/  |_|/_/ |_| /_/  /___//_/ |_/ \____/  (_)  (_)  (_) ")
print("                                                                    ")
###############################
DATA=input("ENTER DATA NAME : ")
df=pd.read_csv("../CANCER_GENE_EXPRESSION_DATASETS/"+DATA+".csv",header=None)
labels=df.iloc[:,-1]
original_data=df.iloc[:,:-1]
data=original_data.T
print(df)
###############################
print("ENTER NOISE PARAMETER MAX CUT")
tmp1=float(input())
print("ENTER NOISE PARAMETER AVG CUT")
tmp2=float(input())
print("ENTER NEIGHBOUR PARAMETER FOR NOISE STAGE")
tmp3=float(input())
print("ENTER NEIGHBOUR PARAMETER FOR CLUSTERING STAGE")
tmp3_1=float(input())
print("ENTER MEMBERSHIP THRESHOLD FOR POSSIBILISTIC NOISE FILTER")
tmp4=float(input())
print("ENTER MAX ITERATION NO")
tmp5=int(input())
print("ENTER MAX CLUSTER NO")
tmp6=int(input())
print("ENTER CONVERGE PARAMETER")
tmp7=float(input())
###############################
start_time = time.time()
###############################
exec(open("../functions.py").read())
x=np.array(data)
x=x/x.max(axis=0)
d=distance_matrix(x)
s=similarity_matrix(d)
m,n=noise_estimation(s,tmp1,tmp2,tmp3)
print("MARKED \n",n,"\nNO : ",len(n))
r=cluster_represent(m)
nrep=r
print("FOUND REPRESENTATIVES IN NOISE STAGE\n",r)
c=centroid_find(r[:tmp6],x)
print("SELECTED REPRESENTATIVES IN NOISE STAGE\n",r[:tmp6])
nsrep=r[:tmp6]
###############################
exec(open("../possibilistic_kmeans.py").read())
weights0,centroids0=kmeans(c,x,i=tmp5,e=tmp7)
clusters0,noise0,weights_clipped0=cluster_consensus(weights0,tmp4)
print("NOISE FOUND\n",noise0,"\nNO : ",len(noise0))
noise=list(set.intersection(set(noise0),set(n)))
print("ABSOLUTE NOISE\n",noise,"\nNO : ",len(noise))
###############################
noise_free_indices=list(set.difference(set(range(x.shape[0])),set(noise)))
new_x=x[noise_free_indices]
###############################
new_x=new_x/new_x.max(axis=0)
d=distance_matrix(new_x)
s=similarity_matrix(d)
m,n=noise_estimation(s,0,0,tmp3_1)
r=cluster_represent(m)
rep=r
print("FOUND REPRESENTATIVES\n",r)
c=centroid_find(r[:tmp6],new_x)
print("SELECTED REPRESENTATIVES\n",r[:tmp6])
srep=r[:tmp6]
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
#RANKING#
#########
average_cluster_centroid=list()
for i in range(np.unique(prediction).size):
    counter=0
    arr=np.zeros(new_x.shape[1])
    for j in range(len(prediction)):
        if prediction[j] == i :
            counter=counter+1
            arr=arr+new_x[j]
    average_cluster_centroid.append(arr/counter)
average_cluster_centroid=np.array(average_cluster_centroid)
dist_centroid=list()
for i in range(len(prediction)):
    dist_centroid.append(np.linalg.norm(new_x[i]-average_cluster_centroid[prediction[i]]))
###############################
logging=pd.DataFrame()
#print("NOISE FREE",noise_free_indices,len(noise_free_indices))
#print("prediction",prediction,len(prediction))
#print("distance",dist_centroid,len(dist_centroid))
logging["NOISE_FREE_GENE_INDICES"]=noise_free_indices
logging["CLUSTER_NO"]=prediction
logging["Distance_FROM_CLUSTER_CENTROID"]=dist_centroid
logging.to_csv("../data/"+DATA+"_log"+".csv")
###############################
stop_time = time.time()
###############################
print("""\n\n\n\n\n
██████╗ ███████╗███████╗██╗   ██╗██╗  ████████╗███████╗
██╔══██╗██╔════╝██╔════╝██║   ██║██║  ╚══██╔══╝██╔════╝
██████╔╝█████╗  ███████╗██║   ██║██║     ██║   ███████╗
██╔══██╗██╔══╝  ╚════██║██║   ██║██║     ██║   ╚════██║
██║  ██║███████╗███████║╚██████╔╝███████╗██║   ███████║
╚═╝  ╚═╝╚══════╝╚══════╝ ╚═════╝ ╚══════╝╚═╝   ╚══════╝
\n\n""")
try:
    print("    DAVIS BOULDIN INDEX ",sklearn.metrics.davies_bouldin_score(new_x,prediction))
    print("        SILHOUTTE INDEX ",sklearn.metrics.silhouette_score(new_x,prediction))
    print("              DUNN INDEX",base.dunn_fast(new_x,prediction))
except :
    print("ONLY ONE CLUSTER FORMED TRY WITH DIFFERENT PARAMETERS")
print("\n\n")
print("""
--------------------------PARAMETERS---------------------\n""")
print("NOISE PARAMETER MAX CUT ",tmp1)
print("NOISE PARAMETER AVG CUT ",tmp2)
print("  NEIGHBOUR PARAMETER N ",tmp3)
print("  NEIGHBOUR PARAMETER C ",tmp3_1)
print("POSSIBILISTIC NOISE CUT ",tmp4)
print("       MAX ITERATION NO ",tmp5)
print("             CLUSTER NO ",len(r[:tmp6]))
print("   CONVERGENT PARAMETER ",tmp7)
###############################
exec(open("../evaluators.py").read())
print("     Full data DT score ",ten_fold_dt(original_data,labels))
print("    Full data SVM score ",ten_fold_svm(original_data,labels))
print("    Full data KNN score ",ten_fold_knn(original_data,labels))
if (len(noise0)>0) :
    print("    Noise data DT score ",ten_fold_dt(original_data.iloc[:,noise0],labels))
    print("   Noise data SVM score ",ten_fold_svm(original_data.iloc[:,noise0],labels))
    print("   Noise data KNN score ",ten_fold_knn(original_data.iloc[:,noise0],labels))
else :
    print("    Noise data DT score NO NOISE WAS FOUND !")
    print("   Noise data SVM score NO NOISE WAS FOUND !")
    print("   Noise data KNN score NO NOISE WAS FOUND !")
print("   N Reps data DT score ",ten_fold_dt(original_data.iloc[:,nrep],labels))
print("  N Reps data SVM score ",ten_fold_svm(original_data.iloc[:,nrep],labels))
print("  N Reps data KNN score ",ten_fold_knn(original_data.iloc[:,nrep],labels))
print("  N SReps data DT score ",ten_fold_dt(original_data.iloc[:,nsrep],labels))
print(" N SReps data SVM score ",ten_fold_svm(original_data.iloc[:,nsrep],labels))
print(" N Sreps data KNN score ",ten_fold_knn(original_data.iloc[:,nsrep],labels))
print("     Reps data DT score ",ten_fold_dt(original_data.iloc[:,rep],labels))
print("    Reps data SVM score ",ten_fold_svm(original_data.iloc[:,rep],labels))
print("    Reps data KNN score ",ten_fold_knn(original_data.iloc[:,rep],labels))
print("    SReps data DT score ",ten_fold_dt(original_data.iloc[:,srep],labels))
print("   SReps data SVM score ",ten_fold_svm(original_data.iloc[:,srep],labels))
print("   Sreps data KNN score ",ten_fold_knn(original_data.iloc[:,srep],labels))
print(" centroid hybr DT score ",ten_fold_dt(average_cluster_centroid.T,labels))
print("centroid hybr SVM score ",ten_fold_svm(average_cluster_centroid.T,labels))
print("centroid hybr KNN score ",ten_fold_knn(average_cluster_centroid.T,labels))
print(" centroid hard DT score ",ten_fold_dt(centroids1.T,labels))
print("centroid hard SVM score ",ten_fold_svm(centroids1.T,labels))
print("centroid hard KNN score ",ten_fold_knn(centroids1.T,labels))
print(" centroid fuzy DT score ",ten_fold_dt(centroids2.T,labels))
print("centroid fuzy SVM score ",ten_fold_svm(centroids2.T,labels))
print("centroid fuzy KNN score ",ten_fold_knn(centroids2.T,labels))
print(" centroid posi DT score ",ten_fold_dt(centroids3.T,labels))
print("centroid posi SVM score ",ten_fold_svm(centroids3.T,labels))
print("centroid posi KNN score ",ten_fold_knn(centroids3.T,labels))
print("         EXECUTION TIME ",stop_time-start_time," SECONDS")
input()
#sleep()
###############################
pass
