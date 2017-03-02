# -*- coding: utf-8 -*-
"""
Created on Mon Feb 20 10:21:08 2017

@author: Team-9
"""
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

# Load data
df = pd.read_csv('ArgentinaPlayers.csv')

##Question 4 -- k-means clustering
# Initialize the model with 2 parameters -- number of clusters and random state.
kmeans_model = KMeans(n_clusters=5, random_state=1)

# Get only the numeric columns from games.
good_columns = df._get_numeric_data()

# Fit the model using the good columns.
kmeans_model.fit(good_columns)

# Get the cluster assignments.
labels = kmeans_model.labels_
labels = pd.DataFrame(labels)
labels.columns = ['clusters']

#Add labels column in df in order to see who player is in which cluster
clusters_table = pd.concat([df, labels], axis=1)

# Create a PCA model.
pca_2 = PCA(2)
# Fit the PCA model on the numeric columns from earlier.
plot_columns = pca_2.fit_transform(good_columns)
# Make a scatter plot of each game, shaded according to cluster assignment.
plt.scatter(x=plot_columns[:,0], y=plot_columns[:,1], c=labels)

# question 5
# data per cluster
cluster_0 = clusters_table.loc[lambda df: df.clusters == 0]
cluster_1 = clusters_table.loc[lambda df: df.clusters == 1]
cluster_2 = clusters_table.loc[lambda df: df.clusters == 2]
cluster_3 = clusters_table.loc[lambda df: df.clusters == 3]
cluster_4 = clusters_table.loc[lambda df: df.clusters == 4]

# table structure
C0_summary = cluster_0.describe().T # attacking midfielder/forward
C0_summary.sort_values(by = 'mean', ascending = False, inplace = True) # sort by mean value

C1_summary = cluster_1.describe().T # goalkeeper
C1_summary.sort_values(by = 'mean', ascending = False, inplace = True) 

C2_summary = cluster_2.describe().T # central defender/ defensive midfielder
C2_summary.sort_values(by = 'mean', ascending = False, inplace = True) 

C3_summary = cluster_3.describe().T # fullback/wingback
C3_summary.sort_values(by = 'mean', ascending = False, inplace = True) 

C4_summary = cluster_4.describe().T # striker
C4_summary.sort_values(by = 'mean', ascending = False, inplace = True)

# question 6
# calculate centroids (including all attributes)
centroid_all = kmeans_model.cluster_centers_
centroid_all = pd.DataFrame(centroid_all, columns = df.columns[2:])

# calculate centroids (using just 7 attributes for the new player)
centroid = centroid_all[[0,11,19,20,26,1,30]]


# initiate player data frame with zero for unknown attributes
plyr = pd.DataFrame(np.zeros((1, 7)), columns = df.columns[[2,13,21,22,28,3,32]])
plyr['Crossing'] = 45
plyr['Sprint Speed'] = 40
plyr['Long Shots'] = 35
plyr['Aggression'] = 45
plyr['Marking'] = 60
plyr['Finishing'] = 40
plyr['GK Handling'] = 15

distance = []
for i in range(len(centroid)):
    dst = []
    for j in range(plyr.shape[1]):
        if plyr.iloc[0, j] != 0:
            d = np.power(plyr.iloc[0, j] - centroid.iloc[i, j], 2)
            dst.append(d)
    distance.append(np.sqrt(sum(dst)))

# print cluster number
distance.index(min(distance))