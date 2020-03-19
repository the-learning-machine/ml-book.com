---
title: K-Means
subtitle: This document will cover something cool.
author: victor
tags: [mlmindmap]
permalink:
---

# 1.   Introduction 
Labels are an essential ingredient to supervised algorithms like Support Vector Machines, which learns a hypothesis function to predict labels given features. K-means clustering is a type of unsupervised learning, which is used when you have unlabeled data (i.e., data without defined categories/groups/response variables).


# 2.   Key Terms 
- **Cluster** is a collection of data points aggregated together because of certain similarities.
- **Cluster Centroid** (or simply **centroid**) is the mean of a cluster, its values are the coordinate-wise average of the data points in this cluster.
- **Within-Cluster Variance** is the coordinate-wise squared deviations from the cluster centroid of all the observations belonging to that cluster:

$$
W(C_k) = \sum_{x_i\in{C_i}}{}\sum_{j=1}{}P(x_{ij} - \overline{x}_k)^2=\sum_{x_i\inC_i}\mid\mid{x_i-\overline{x}_k}\mid\mid^2
$$

<p align="center">
In the expression above $$x_{ij}$$ denotes *j-th* predictor of observation $$x_i$$. $$C_k$$ denotes a set of points belonging to cluster *k* and $$x_k$$ denotes a centroid of cluster *k*.
</p>

- **Total Within-Cluster Variance** is a within-cluster variance summed up across all clusters:

$$
W(C) = \sum_{k=1}{K}\sum_{x_i\in{C_x}}{}\mid\mid{x_i - \overline{x}_k}\mid\mid^2
$$

Note that the notation, $$\mid\mid{x_i - \overline{x}_k}\mid\mid$$ means the euclidean distance between vectors $$x_i$$ and $$x_k$$. 

# 3.   Data Representation and Preparation
In the formulas above $$x_i$$ represents a vector in a P-dimensional space and P is a number of predictors in data set. As you can see from the formulas above, *K-Means* algorithm utilizes the notion of distance between data points and each data point weights equally. In order to calculate the distance, we need our data to be numerical. For this reason, categorical values should be handled (either excluded from the list of predictors or replaced with numerical values). Also, we need to normalize our data in order to avoid the effects of incomparable units and different scaling.

# 4.   Algorithm 
*K-Means* algorithm finds cluster centers that minimize the total within-cluster variance $$W(C)$$. This is achieved in several steps:
- **Step 1**:  Randomly generate K centroids $$x_1,x_2...x_n$$
- **Step 2**:  Assign data points to the cluster of the closest centroid:

$$
c_i:=argmin_j\mid\mid{x_i - \overline{x}_k}\mid\mid^2

- **Step 3**:  Compute the mean of each cluster
- **Step 4**: Reassign centroids to respective clusters’ means computed in Step 3
- **Step 5**: If the **stop criterion** is not satisfied: Go to Step 2

**Stop criterion** can be one of the following:
1. Cluster re-assignation results in same clusters
2. A specified number of iterations is reached
3. Reassigned centroids are located close (need to specify the distance) to the previous centroids

In order to achieve **global optima**, the algorithm should be run multiple times and clusters’ realization that is observed more often will be our global optima.
 
Example: In Figure 1, you can see a K-means algorithm. Training examples are shown as dots, and cluster centroids $$(K)$$ are shown as crosses. $$(a)$$ is an original dataset. $$(b)$$ is a random initial cluster centroids. $$(c-f)$$ is an illustration of running two iterations of k-means. In each iteration, we assign each training example to the closest cluster centroid (shown by ”painting” the training examples the same color as the cluster centroid to which is assigned); then we move each cluster centroid to the mean of the points assigned to it.

<p align="center">
    <img src="/uploads/doc/clustering/Clustering_1.PNG" height="550" width="550">
</p>

# 5.   Choosing K
There are three most common ways of selecting the number of clusters K.
1. Utilize our domain knowledge or any other insight about the data. For instance, we want to cluster flower and we know that our data contains exactly 3 types of flowers. Another example is when we want to cluster cars sold last year. In this case, K will be the number of all car manufacturers available on the market.
2. Run the algorithms several times for different values of K and select such K that results in the smallest value of total within-cluster variance.
3. Perform cross-validation and select such K that performs the best on a hold-out dataset.