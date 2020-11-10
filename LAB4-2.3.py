# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 01:52:18 2020

@author: 75965
"""

from sklearn.cluster import KMeans

kmeans = KMeans(M) 
kmeans.fit(X_scaled)
C_kmeans = kmeans.cluster_centers_

print(kmeans.cluster_centers_.shape)