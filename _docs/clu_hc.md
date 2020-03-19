---
title: Decision Tree (ID3)
subtitle: This document will cover something cool.
author: NobodySpeak
tags: [mlmindmap]
permalink:
---


# 1.   Introduction 
Hierarchical clustering is an unsupervised learning algorithm that is used to group similar objects (data points) in clusters. It doesn’t require a predefined number of clusters as this algorithm outputs a tree diagram called dendrogram which we can then cut to obtain clusters.

# 2.   Key Terms 
**Cluster** is a collection of data points aggregated together because of certain similarities.

**Cluster Centroid** (or simply **centroid**) is the mean of a cluster, its values are the coordinate-wise average of the data points in this cluster.

In hierarchical clustering, different distance measure can be used (such as Manhattan or L1, Euclidian or L2 distances, etc.). After selecting the distance measure we need to specify from where distance is computed. This is determined by the **linkage criteria**:

- **Single Linkage** - distance is computed between the two MOST similar parts of clusters (two closest points). Single linkage suffers from chaining meaning that clusters can be too spread out, and not compact enough.

- **Complete Linkage** - distance is computed between the two LEAST similar parts of clusters (two most distant points). Complete linkage avoids chaining but suffers from crowding meaning that Clusters are compact, but not far enough apart.

- **Average Linkage** - distance is computed between clusters’ centroids. This is a balanced approach: clusters tend to be relatively compact and relatively far apart.

<p align="center">
    <img src="uploads/doc/clustering/clu_hc_LinkageTypes.PNG" height="300" width="500">
</p>

# 3.   Data Representation and Preparation
Each observation from a data set is represented by a vector in a P-dimensional space and P is a number of predictors in data set. Hierarchical clustering utilizes the notion of distance between data points and each data point weights equally. In order to calculate the distance, we need our data to be numerical. For this reason, categorical values should be handled (either excluded from the list of predictors or replaced with numerical values). Also, we need to normalize our data in order to avoid the effects of incomparable units and different scaling.

# 4.   Approaches
There are two types of hierarchical clustering, Divisive and Agglomerative. 

In **divisive** or **top-down** method all observations are assigned to one cluster and then we partition the cluster into two least similar clusters recursively until each cluster contains exactly one observation.

In **agglomerative** or **bottom-up** method each observation is assigned to its own cluster and then we join two most similar clusters into one cluster recursively until there is only one cluster containing all observations.

These approaches are illustrated in Figure 2. In some settings, divisive algorithms can perform better than agglomerative algorithms, but they are conceptually more complex. For this reason, we will discuss the agglomerative clustering algorithm in this article.

<p align="center">
    <img src="uploads/doc/clustering/clu_hc_HCTypes.PNG" height="400" width="450">
</p>

# 5.   Algorithm
We can summarize the agglomerative clustering algorithm discussed in the Approaches section in the following steps:

- Step 1: Assign each observation to its own cluster

- Step 2: Calculate distances between clusters

- Step 3: Merge two most similar clusters based on linkage criteria

- Step 4: If Number of Clusters > 1: Go to Step 2

# 6.   Example
Let’s perform agglomerative hierarchical clustering on the following data points:

<p align="center">
    <img src="uploads/doc/clustering/clu_hc_DataPoints.PNG" height="300" width="300">
</p>

First, let’s fill matrix of distances between all data points. For simplicity of calculations, we will use Manhattan (L1) distance measure. Remember, that in agglomerative clustering we assign each observation to its own cluster.

<p align="center">
    <img src="uploads/doc/clustering/clu_hc_DataPoints_2.PNG" height="400" width="450">
</p>

The lower half of the distance matrix is not filled as this matrix is symmetric. As you can see clusters E and F are the most similar clusters at this point as well as clusters B and C. We can join them in any order. Let’s begin with E and F, we join them into one cluster.

Note, that we need to recompute distances according to our new cluster structure. In this example, we will use single linkage criteria. For this reason, the distance between clusters (B,C) and (A) equals the distance between POINTS B and A as they are the closest points of two clusters. Similarly, the distance between clusters (B,C) and (D) equals the distance between POINTS C and D as they are the closest points of two clusters.

<p align="center">
    <img src="uploads/doc/clustering/clu_hc_DataPoints_3.PNG" height="400" width="450">
</p>

The steps 2 and 3 of the algorithm are repeated until we have just one class:

<p align="center">
    <img src="uploads/doc/clustering/clu_hc_DataPoints_4.PNG" height="1100" width="450">
    <img src="uploads/doc/clustering/clu_hc_DataPoints_5.PNG" height="400" width="450">
</p>

Note that when we have clusters (A,B,C), (D) and (E,F) we can join (A,B,C) and (E,F) as well as (D) and (E,F). We proceeded with latter to keep cluster equivalent in terms of size. The dendrogram of this clustering can be found below. The scale at the left-hand side denotes the distance between clusters at which the join appeared. We can perform a cut of this dendrogram at the particular distance in order to achieve the clustering we want. For example, if we cut at a distance of 2 we will have clusters (A), (B,C), (D) and (E,F).

<p align="center">
    <img src="uploads/doc/clustering/clu_hc_dendrogram.PNG" height="300" width="400">
</p>

This was an example of hierarchical clustering with a single-link method, where the distance is computed between the two MOST similar parts of clusters (two closest points). If we were to use complete-link, the distance would be computed between the two LEAST similar parts of clusters (two most distant points), and if average-link, the distance would be computed between clusters’ centroids.