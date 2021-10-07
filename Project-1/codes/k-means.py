import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from numpy.linalg import norm
from sklearn.metrics import silhouette_samples

np.set_printoptions(threshold=sys.maxsize)
#%%
class Kmeans:

    def __init__(self, n_clusters, max_iter=100, random_state=np.random.randint(0, 1000, size=1)):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.random_state = random_state

    def comp_sse(self, X, labels, centroids):
        distance = np.zeros(X.shape[0])
        for k in range(self.n_clusters):
        # calculating the distance of each datapoint present from the 
        # centroid of the cluster it is present in to find how the
        # datapoints are distributed
            distance[labels == k] = norm(X[labels == k] - centroids[k], axis=1)
        return np.sum(np.square(distance))
    
    def comp_centroids(self, X, labels):
        centroids = np.zeros((self.n_clusters, X.shape[1]))
        for k in range(self.n_clusters):
        # taking the mean of all the datapoints assigned to a cluster
        # to find out our new centroids.
            centroids[k, :] = np.mean(X[labels == k, :], axis=0)
        return centroids
    
    def closest_cluster(self, distance):
        # returning the index or the centroid which is nearest
        # to each datapoint in the dataset in the form of an array for
        # all the datapoints
        return np.argmin(distance, axis=1)
    
    def comp_distance(self, X, centroids):
        # creating a 2D array capable of holding |n_clusters| different
        # values for each datapoint so that we will know which data point
        # is how far from each centroid
        distance = np.zeros((X.shape[0], self.n_clusters))
        for k in range(self.n_clusters):
        # finding the vector norm of every data point with each of the 
        # centroid
            arr = X - centroids[k,:]
            row_norm = norm(arr, axis=1)
            distance[:, k] = np.square(row_norm)
        return distance
    
    def init_centroids(self, X):
        np.random.RandomState(self.random_state)
        # randomising all examples in the dataset
        random_idx = np.random.permutation(X.shape[0])
        # going from 0 to |n_clusters| in the random_idx array to 
        # choose |n_clusters| number of random examples from the dataset
        # and to choose them as our centroids
        cent_idx = random_idx[:self.n_clusters]
        centroids = X[cent_idx]
        return centroids
    
    def fit(self, X):
        # initializing centroids with random datapoints
        self.centroids = self.init_centroids(X)
        # running for |max_iter| number of times to finalize the centroids
        for i in range(self.max_iter):
            o_centroids = self.centroids
        # finding the distance of every datapoint from each centroid
            distance = self.comp_distance(X, o_centroids)
        # finding which centroid is closer to which datapoint
            self.labels = self.closest_cluster(distance)
        # finding new centroids after allocating all the datapoints to
        # the respective clusters
            self.centroids = self.comp_centroids(X, self.labels)
        # if the previous found centroids are same as the newly found one
        # break the loop beacuse we have found our optimal centroids.
            if np.all(o_centroids == self.centroids):
                break
        # calculating what's the total sum of squared error for each 
        # centroid obtained
        self.error = self.comp_sse(X, self.labels, self.centroids)
    
    def predict(self, X):
        distance = self.comp_distance(X, self.centroids)
        return self.closest_cluster(distance)
#%%
dataset = pd.read_csv('football_data.csv')

#%%
toKeep=np.array([3,7,8,11,12,13,15,16,17,22,23])
for i in range(25,89):
    toKeep=np.append(toKeep,i)

X = dataset.iloc[:,toKeep].values
header=dataset.iloc[:,toKeep].columns


for j in np.array([3,4,X.shape[1]-1]):
    for i in range(X.shape[0]):
        if type(X[i,j]) is float:
            pass
        elif X[i,j][-1]=='M':
            X[i,j]=float(X[i,j][1:-1])*1000000
        elif X[i,j][-1]=='K':
            X[i,j]=float(X[i,j][1:-1])*1000
        else:
            X[i,j]=0

for j in range(10,12):
    for i in range(X.shape[0]):
       if type(X[i,j]) is not float:
           X[i,j]=float(X[i,j][-4:])
           
for i in range (X.shape[0]):
    if type(X[i,12]) is not float:
        X[i,12]=float(X[i,12].split("'")[0])*12+float(X[i,12].split("'")[1])
        
        
for i in range(X.shape[0]):
    if type(X[i,13]) is not float:
        X[i,13]=float(X[i,13][:-3])
        
    
for i in range(40,X.shape[1]-1):
    X[:,i].astype(float)

for j in range(14,40):
    for i in range (X.shape[0]):
        if type(X[i,j]) is not float:
            X[i,j]=float(X[i,j].split("+")[0])+float(X[i,j].split("+")[1])


X=X.astype(float)


#%%
#for i in range(75):
#   print(np.isnan(X[:,i]).sum())

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values = np.nan, strategy = 'mean')
for j in range(X.shape[1]):
    imputer = imputer.fit(X[:, j:j+1])
    X[:, j:j+1] = imputer.transform(X[:, j:j+1])
    
#%%

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X2= sc_X.fit_transform(X)

# Standardize the data
X_std = StandardScaler().fit_transform(X)
#%%
for l,j in enumerate([3,5,7]):
    n_iter = 8
    fig, ax = plt.subplots(4, 2, figsize=(14, 24))
    fig1, ay = plt.subplots(4, 2, figsize=(14, 24))
    fig2, az = plt.subplots(4, 2, figsize=(14, 24))
    fig3, au = plt.subplots(4, 2, figsize=(14, 24))
    ax = np.ravel(ax)
    ay = np.ravel(ay)
    az = np.ravel(az)
    au = np.ravel(au)
    for i in range(n_iter):
        km = Kmeans(n_clusters=j)
        km.fit(X_std)
        centroids = km.centroids
        
        ''' Graph : Overall vs Gkdiving  '''
        
        ax[i].scatter(X_std[km.labels == 0, 69], X_std[km.labels == 0, 1], c='green', label='cluster 1')
        ax[i].scatter(X_std[km.labels == 1, 69], X_std[km.labels == 1, 1], c='blue', label='cluster 2')
        ax[i].scatter(X_std[km.labels == 2, 69], X_std[km.labels == 2, 1], c='orange', label='cluster 3')
        if(j == 5):
            ax[i].scatter(X_std[km.labels == 3, 69], X_std[km.labels == 3, 1], c='pink', label='cluster 4')
            ax[i].scatter(X_std[km.labels == 4, 69], X_std[km.labels == 4, 1], c='yellow', label='cluster 5')
        
        if(j == 7):
            ax[i].scatter(X_std[km.labels == 3, 69], X_std[km.labels == 3, 1], c='pink', label='cluster 4')
            ax[i].scatter(X_std[km.labels == 4, 69], X_std[km.labels == 4, 1], c='yellow', label='cluster 5')
            ax[i].scatter(X_std[km.labels == 5, 69], X_std[km.labels == 5, 1], c='grey', label='cluster 6')
            ax[i].scatter(X_std[km.labels == 6, 69], X_std[km.labels == 6, 1], c='indigo', label='cluster 7')

        ax[i].scatter(centroids[:, 69], centroids[:, 1], c='red', marker='*', s=300, label='centroid')
        ax[i].set_xlim([-6, 6])
        ax[i].set_ylim([-6, 6])
        ax[i].legend(loc='upper left')
        ax[i].set_title(f'{km.error:.4f}')
        ax[i].set_aspect('equal')
        
        ''' Graph : Overall vs ST '''
        
        ay[i].scatter(X_std[km.labels == 0, 15], X_std[km.labels == 0, 1], c='green', label='cluster 1')
        ay[i].scatter(X_std[km.labels == 1, 15], X_std[km.labels == 1, 1], c='blue', label='cluster 2')
        ay[i].scatter(X_std[km.labels == 2, 15], X_std[km.labels == 2, 1], c='orange', label='cluster 3')
        if(j == 5):
            ay[i].scatter(X_std[km.labels == 3, 15], X_std[km.labels == 3, 1], c='pink', label='cluster 4')
            ay[i].scatter(X_std[km.labels == 4, 15], X_std[km.labels == 4, 1], c='yellow', label='cluster 5')
        
        if(j == 7):
            ay[i].scatter(X_std[km.labels == 3, 15], X_std[km.labels == 3, 1], c='pink', label='cluster 4')
            ay[i].scatter(X_std[km.labels == 4, 15], X_std[km.labels == 4, 1], c='yellow', label='cluster 5')
            ay[i].scatter(X_std[km.labels == 5, 15], X_std[km.labels == 5, 1], c='grey', label='cluster 6')
            ay[i].scatter(X_std[km.labels == 6, 15], X_std[km.labels == 6, 1], c='indigo', label='cluster 7')


        ay[i].scatter(centroids[:, 15], centroids[:, 1], c='red', marker='*', s=300, label='centroid')
        ay[i].set_xlim([-6, 6])
        ay[i].set_ylim([-6, 6])
        ay[i].legend(loc='upper left')
        ay[i].set_title(f'{km.error:.4f}')
        ay[i].set_aspect('equal')
        
        
        ''' Graph : Overall vs CM  '''
        
        az[i].scatter(X_std[km.labels == 0, 27], X_std[km.labels == 0, 1], c='green', label='cluster 1')
        az[i].scatter(X_std[km.labels == 1, 27], X_std[km.labels == 1, 1], c='blue', label='cluster 2')
        az[i].scatter(X_std[km.labels == 2, 27], X_std[km.labels == 2, 1], c='orange', label='cluster 3')
        if(j == 5):
            az[i].scatter(X_std[km.labels == 3, 27], X_std[km.labels == 3, 1], c='pink', label='cluster 4')
            az[i].scatter(X_std[km.labels == 4, 27], X_std[km.labels == 4, 1], c='yellow', label='cluster 5')
        
        if(j == 7):
            az[i].scatter(X_std[km.labels == 3, 27], X_std[km.labels == 3, 1], c='pink', label='cluster 4')
            az[i].scatter(X_std[km.labels == 4, 27], X_std[km.labels == 4, 1], c='yellow', label='cluster 5')
            az[i].scatter(X_std[km.labels == 5, 27], X_std[km.labels == 5, 1], c='grey', label='cluster 6')
            az[i].scatter(X_std[km.labels == 6, 27], X_std[km.labels == 6, 1], c='indigo', label='cluster 7')


        az[i].scatter(centroids[:, 27], centroids[:, 1], c='red', marker='*', s=300, label='centroid')
        az[i].set_xlim([-6, 6])
        az[i].set_ylim([-6, 6])
        az[i].legend(loc='upper left')
        az[i].set_title(f'{km.error:.4f}')
        az[i].set_aspect('equal')
        
        ''' Graph : Overall vs CB  '''
        
        au[i].scatter(X_std[km.labels == 0, 37], X_std[km.labels == 0, 1], c='green', label='cluster 1')
        au[i].scatter(X_std[km.labels == 1, 37], X_std[km.labels == 1, 1], c='blue', label='cluster 2')
        au[i].scatter(X_std[km.labels == 2, 37], X_std[km.labels == 2, 1], c='orange', label='cluster 3')
        if(j == 5):
            au[i].scatter(X_std[km.labels == 3, 37], X_std[km.labels == 3, 1], c='pink', label='cluster 4')
            au[i].scatter(X_std[km.labels == 4, 37], X_std[km.labels == 4, 1], c='yellow', label='cluster 5')
        
        if(j == 7):
            au[i].scatter(X_std[km.labels == 3, 37], X_std[km.labels == 3, 1], c='pink', label='cluster 4')
            au[i].scatter(X_std[km.labels == 4, 37], X_std[km.labels == 4, 1], c='yellow', label='cluster 5')
            au[i].scatter(X_std[km.labels == 5, 37], X_std[km.labels == 5, 1], c='grey', label='cluster 6')
            au[i].scatter(X_std[km.labels == 6, 37], X_std[km.labels == 6, 1], c='indigo', label='cluster 7')


        au[i].scatter(centroids[:, 37], centroids[:, 1], c='red', marker='*', s=300, label='centroid')
        au[i].set_xlim([-6, 6])
        au[i].set_ylim([-6, 6])
        au[i].legend(loc='upper left')
        au[i].set_title(f'{km.error:.4f}')
        au[i].set_aspect('equal')
        
        
    
    fig.tight_layout()
    fig.suptitle(f'Clustering done using k = {j}, Overall vs Gkdiving', fontsize=16, fontweight='semibold', y=1.05)
    fig1.tight_layout()
    fig1.suptitle(f'Clustering done using k = {j}, Overall vs ST', fontsize=16, fontweight='semibold', y=1.05)
    fig2.tight_layout()
    fig2.suptitle(f'Clustering done using k = {j}, Overall vs CM', fontsize=16, fontweight='semibold', y=1.05)
    fig3.tight_layout()
    fig3.suptitle(f'Clustering done using k = {j}, Overall vs CB', fontsize=16, fontweight='semibold', y=1.05)
#%%
sse = []
list_k = list(range(1, 9))

for k in list_k:
    km = Kmeans(n_clusters=k)
    km.fit(X_std)
    sse.append(km.error)

# Plot sse against k
plt.figure(figsize=(6, 6))
plt.plot(list_k, sse, '-o')
plt.xlabel(r'Number of clusters *k*')
plt.ylabel('Sum of squared distance');
#%%
for i, k in enumerate([3,5,7]):
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(18, 7)
    
    # Run the Kmeans algorithm
    km = Kmeans(n_clusters=k)
    km.fit(X_std)
    labels = km.predict(X_std)
    centroids = km.centroids
    
    # Get silhouette samples
    silhouette_vals = silhouette_samples(X_std, labels)

    # Silhouette plot
    y_ticks = []
    y_lower, y_upper = 0, 0
    for i, cluster in enumerate(np.unique(labels)):
        cluster_silhouette_vals = silhouette_vals[labels == cluster]
        cluster_silhouette_vals.sort()
        y_upper += len(cluster_silhouette_vals)
        ax1.barh(range(y_lower, y_upper), cluster_silhouette_vals, edgecolor='none', height=1)
        ax1.text(-0.03, (y_lower + y_upper) / 2, str(i + 1))
        y_lower += len(cluster_silhouette_vals)

    # The silhouette_score gives the average value for all the samples.
    # This gives a perspective into the density and separation of the formed
    # clusters
    
    avg_score = np.mean(silhouette_vals)
    
    print("For n_clusters =", k, "The average silhouette_score is :", avg_score)
    
    ax1.axvline(avg_score, linestyle='--', linewidth=2, color='green')
    ax1.set_yticks([])
    ax1.set_xlim([-0.1, 1])
    ax1.set_xlabel('Silhouette coefficient values')
    ax1.set_ylabel('Cluster labels')
    ax1.set_title('Silhouette plot for the various clusters', y=1.02);
    
    # Scatter plot of data colored with labels
    ax2.scatter(X_std[:, 69], X_std[:, 3], c=labels)
    ax2.scatter(centroids[:, 69], centroids[:, 3], marker='*', c='r', s=250)
    ax2.set_xlim([-2, 6])
    ax2.set_ylim([-2, 6])
    ax2.set_xlabel('gkdiving')
    ax2.set_ylabel('value')
    ax2.set_title('Visualization of clustered data', y=1.02)
    ax2.set_aspect('equal')
    plt.tight_layout()
    plt.suptitle(f'Silhouette analysis using k = {k}', fontsize=16, fontweight='semibold', y=1.05)
#%%
