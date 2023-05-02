
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
import random
from jqmcvi import base

#FUNCTIONS

def sleep():
    time.sleep(1)
    os.system("clear")

#START
sleep()
print("""
  ██████ ▄▄▄█████▓ ▄▄▄       ██▀███  ▄▄▄█████▓ ██▓ ███▄    █   ▄████ 
▒██    ▒ ▓  ██▒ ▓▒▒████▄    ▓██ ▒ ██▒▓  ██▒ ▓▒▓██▒ ██ ▀█   █  ██▒ ▀█▒
░ ▓██▄   ▒ ▓██░ ▒░▒██  ▀█▄  ▓██ ░▄█ ▒▒ ▓██░ ▒░▒██▒▓██  ▀█ ██▒▒██░▄▄▄░
  ▒   ██▒░ ▓██▓ ░ ░██▄▄▄▄██ ▒██▀▀█▄  ░ ▓██▓ ░ ░██░▓██▒  ▐▌██▒░▓█  ██▓
▒██████▒▒  ▒██▒ ░  ▓█   ▓██▒░██▓ ▒██▒  ▒██▒ ░ ░██░▒██░   ▓██░░▒▓███▀▒
▒ ▒▓▒ ▒ ░  ▒ ░░    ▒▒   ▓▒█░░ ▒▓ ░▒▓░  ▒ ░░   ░▓  ░ ▒░   ▒ ▒  ░▒   ▒ 
░ ░▒  ░ ░    ░      ▒   ▒▒ ░  ░▒ ░ ▒░    ░     ▒ ░░ ░░   ░ ▒░  ░   ░ 
░  ░  ░    ░        ░   ▒     ░░   ░   ░       ▒ ░   ░   ░ ░ ░ ░   ░ 
 ███▄ ░  █ ▒█████   █   ░ █░   ░   ▐██▌   ▐██▌ ░ ▐██▌      ░       ░ 
 ██ ▀█   █▒██▒  ██▒▓█░ █ ░█░       ▐██▌   ▐██▌   ▐██▌                
▓██  ▀█ ██▒██░  ██▒▒█░ █ ░█        ▐██▌   ▐██▌   ▐██▌                
▓██▒  ▐▌██▒██   ██░░█░ █ ░█        ▓██▒   ▓██▒   ▓██▒                
▒██░   ▓██░ ████▓▒░░░██▒██▓        ▒▄▄    ▒▄▄    ▒▄▄                 
░ ▒░   ▒ ▒░ ▒░▒░▒░ ░ ▓░▒ ▒         ░▀▀▒   ░▀▀▒   ░▀▀▒                
░ ░░   ░ ▒░ ░ ▒ ▒░   ▒ ░ ░         ░  ░   ░  ░   ░  ░                
   ░   ░ ░░ ░ ░ ▒    ░   ░            ░      ░      ░                
         ░    ░ ░      ░           ░      ░      ░                   
""")
###############################
DATA=input("ENTER DATA NAME : ")
df=pd.read_csv("../data/"+DATA+".csv")
labels=df.iloc[:,-1]
data=df.iloc[:,:-1]
print(df)
###############################
print("ENTER THE NUMBER OF CLUSTERS")
k=int(input())
print("ENTER ITERATION NO")
i=int(input())
print("ENTER CONVERGENT PARAMETER")
e=float(input())
###############################
exec(open("../functions.py").read())
y=np.array(labels)
x=np.array(data)
x=x/x.max(axis=0)
r=random.choices(range(x.shape[0]),k=k)
c=centroid_find(r,x)
print("\n\n\n")
###############################
# FUZZY K MEANS
###############################
start_time = time.time()
###############################
exec(open("../fuzzy_kmeans.py").read())
weights,centroids=kmeans(c,x,i=i,e=e)
clusters,noise,weights_clipped=cluster_consensus(weights,0)
prediction=cluster_matrix_make(clusters,x.shape[0])
###############################
stop_time = time.time()
###############################
print("""
██████╗ ███████╗███████╗██╗   ██╗██╗  ████████╗███████╗
██╔══██╗██╔════╝██╔════╝██║   ██║██║  ╚══██╔══╝██╔════╝
██████╔╝█████╗  ███████╗██║   ██║██║     ██║   ███████╗
██╔══██╗██╔══╝  ╚════██║██║   ██║██║     ██║   ╚════██║
██║  ██║███████╗███████║╚██████╔╝███████╗██║   ███████║
╚═╝  ╚═╝╚══════╝╚══════╝ ╚═════╝ ╚══════╝╚═╝   ╚══════╝
                                                  FUZZY
\n""")
print("               ACCURACY ",sklearn.metrics.rand_score(y,prediction))
print("    DAVIS BOULDIN INDEX ",sklearn.metrics.davies_bouldin_score(x,prediction))
print("        SILHOUTTE INDEX ",sklearn.metrics.silhouette_score(x,prediction))
print("              DUNN INDEX",base.dunn_fast(x,prediction))
print("    CHOOSEN CLUSTERN NO ",k)
print("           ITERATION NO ",i)
print("     CONVERGE PARAMETER ",e)
print("         EXECUTION TIME ",stop_time-start_time," SECONDS")
###############################
# HARD K MEANS
###############################
start_time = time.time()
###############################
exec(open("../hard_kmeans.py").read())
weights,centroids=kmeans(c,x,i=i,e=e)
clusters,noise,weights_clipped=cluster_consensus(weights,0)
prediction=cluster_matrix_make(clusters,x.shape[0])
###############################
stop_time = time.time()
###############################
print("""
██████╗ ███████╗███████╗██╗   ██╗██╗  ████████╗███████╗
██╔══██╗██╔════╝██╔════╝██║   ██║██║  ╚══██╔══╝██╔════╝
██████╔╝█████╗  ███████╗██║   ██║██║     ██║   ███████╗
██╔══██╗██╔══╝  ╚════██║██║   ██║██║     ██║   ╚════██║
██║  ██║███████╗███████║╚██████╔╝███████╗██║   ███████║
╚═╝  ╚═╝╚══════╝╚══════╝ ╚═════╝ ╚══════╝╚═╝   ╚══════╝
                                                   HARD
\n""")
print("               ACCURACY ",sklearn.metrics.rand_score(y,prediction))
print("    DAVIS BOULDIN INDEX ",sklearn.metrics.davies_bouldin_score(x,prediction))
print("        SILHOUTTE INDEX ",sklearn.metrics.silhouette_score(x,prediction))
print("              DUNN INDEX",base.dunn_fast(x,prediction))
print("    CHOOSEN CLUSTERN NO ",k)
print("           ITERATION NO ",i)
print("     CONVERGE PARAMETER ",e)
print("         EXECUTION TIME ",stop_time-start_time," SECONDS")
###############################
# POSSIBILISTIC K MEANS
###############################
start_time = time.time()
###############################
exec(open("../possibilistic_kmeans.py").read())
weights,centroids=kmeans(c,x,i=i,e=e)
clusters,noise,weights_clipped=cluster_consensus(weights,0)
prediction=cluster_matrix_make(clusters,x.shape[0])
###############################
stop_time = time.time()
###############################
print("""
██████╗ ███████╗███████╗██╗   ██╗██╗  ████████╗███████╗
██╔══██╗██╔════╝██╔════╝██║   ██║██║  ╚══██╔══╝██╔════╝
██████╔╝█████╗  ███████╗██║   ██║██║     ██║   ███████╗
██╔══██╗██╔══╝  ╚════██║██║   ██║██║     ██║   ╚════██║
██║  ██║███████╗███████║╚██████╔╝███████╗██║   ███████║
╚═╝  ╚═╝╚══════╝╚══════╝ ╚═════╝ ╚══════╝╚═╝   ╚══════╝
                                          POSSIBILISTIC
\n""")
print("               ACCURACY ",sklearn.metrics.rand_score(y,prediction))
print("    DAVIS BOULDIN INDEX ",sklearn.metrics.davies_bouldin_score(x,prediction))
print("        SILHOUTTE INDEX ",sklearn.metrics.silhouette_score(x,prediction))
print("              DUNN INDEX",base.dunn_fast(x,prediction))
print("    CHOOSEN CLUSTERN NO ",k)
print("           ITERATION NO ",i)
print("     CONVERGE PARAMETER ",e)
print("         EXECUTION TIME ",stop_time-start_time," SECONDS")
###############################
input()
#sleep()
###############################
pass
