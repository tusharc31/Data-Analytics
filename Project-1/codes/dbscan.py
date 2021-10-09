import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

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
temp=X
toKeep=np.array([])
for i in range(0,6):
    toKeep=np.append(toKeep,i)
for i in range(12,14):
    toKeep=np.append(toKeep,i)
toKeep=np.append(toKeep,[15,19,27,37])
for i in range(40,75):
    toKeep=np.append(toKeep,i)
toKeep=toKeep.astype(int)

X=X[:,toKeep]
header=header[toKeep]



#%%
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X2= sc_X.fit_transform(X)

#%%
#plt.scatter(X[:,0],X[:,1])
#plt.xlabel('Age')
#plt.ylabel('Overall')
#plt.show()

#plt.scatter(X[:,0],X[:,2])
#plt.xlabel('Age')
#plt.ylabel('Potential')
#plt.show()

#plt.scatter(X[:,12],X[:,13])
#plt.xlabel('Height')
#plt.ylabel('Weight')
#plt.show()

#plt.scatter(X[:,1],X[:,4])
#plt.xlabel('Overall')
#plt.ylabel('Wage')
#plt.show()

#plt.scatter(X[:,1],X[:,3])
#plt.xlabel('Overall')
#plt.ylabel('Value')
#plt.show()

#plt.scatter(X[:,1],X[:,74])
#plt.xlabel('Overall')
#plt.ylabel('Release Clause')
#plt.show()

#for i in range(14,74):
#    plt.scatter(X[:,i],X[:,3])
#    plt.xlabel(header[i])
#    plt.ylabel('Value')
#    plt.show()
#%%

def freq(labels):
    mp=dict()
    for i in labels:
        if i in mp.keys():
            mp[i]+=1
        else:
            mp[i]=1
    return [k for k, v in sorted(mp.items(),reverse=True, key=lambda item: item[1])]


#%%
def draw(labels,intraClass,interClass):
    
    mp=freq(labels)
    
    for arg in [5,8,10,11,45]:
        plt.figure()
        
        color=['black','green','red','blue','orange','yellow','pink']    

        for i in mp:
            plt.scatter(X[labels==i][:,1],X[labels==i][:,arg],color=color[i+1],label=str(i)+"="+str(np.sum(labels==i)))
            
        plt.xlabel(header[1])
        plt.ylabel(header[arg])
        plt.legend()
        plt.title(str(np.max(labels)))
        plt.show()
#%%
from sklearn.decomposition import PCA 
from sklearn.neighbors import NearestNeighbors
from kneebow.rotor import Rotor
from sklearn.cluster import DBSCAN 
from sklearn.metrics import pairwise 
stats=[]

for n_components in [10,20]:
    pca = PCA(n_components = n_components)
    X3= pca.fit_transform(X2) 
    
    for n_neighbors in range(5,15):
      
        neigh = NearestNeighbors(n_neighbors=n_neighbors)
        nbrs = neigh.fit(X3)
        distances, indices = nbrs.kneighbors(X3)
        distances=np.mean(distances,axis=1)
        distances = np.sort(distances, axis=0)
        
        plt.plot(distances)
        
        
        temp=np.append(indices[:,0:1],distances.reshape(distances.shape[0],1),axis=1)      
        rotor = Rotor()
        rotor.fit_rotate(temp)
        elbow_idx = rotor.get_elbow_index()
        
        plt.title("elbow_idx"+str(elbow_idx)+" eps=" + str(distances[elbow_idx]))
        plt.show()
        
        db_default = DBSCAN(eps = distances[elbow_idx], min_samples = n_neighbors).fit(X3) 
        labels = db_default.labels_ 
        
        if np.max(labels)>4:
            continue

        n_clusters=np.max(labels)+1
        
        intraClass=0
        interClass=0
        for i in range(n_clusters):
            intraClass+=np.mean(pairwise.pairwise_distances(X2[labels==i],metric='euclidean'))
            for j in range(i+1,n_clusters):
                interClass+=np.mean(pairwise.pairwise_distances(X2[labels==i],X2[labels==j],metric='euclidean'))
        intraClass=intraClass/n_clusters
        interClass=interClass/((n_clusters*(n_clusters-1))/2)
    
        stats.append([interClass/intraClass,interClass,intraClass,distances[elbow_idx],n_neighbors,n_components,n_clusters])

        print(str(intraClass)+"  "+str(interClass))
        draw(labels,intraClass,interClass)        
        
        
#%%
stats=sorted(stats,reverse=True)