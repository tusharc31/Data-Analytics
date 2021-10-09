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

np.random.shuffle(X)
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X2= sc_X.fit_transform(X)
graph_data=X
X=X2

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
from sklearn.decomposition import PCA 
from sklearn.cluster import AgglomerativeClustering 
import scipy.cluster.hierarchy as sch 
from sklearn.metrics import pairwise 

#%%
print("--------------------")
 
elements=np.array([])
components=np.array([])
clusters=np.array([])

for n_elements in range(2500,5500,500):
    for n_components in [5,8,10,15]:
        
        
        pca = PCA(n_components = n_components) 
        X2= pca.fit_transform(X) 
        
        print("PCA done")
        
        dendrogram = sch.dendrogram(sch.linkage(X2[:n_elements,:],method='ward'))
        plt.title("components="+str(n_components)+" elements="+str(n_elements))
        plt.show()
        
        elements=np.append(elements,n_elements)
        components=np.append(components,n_components)
        print("dendrogram done")

#%%
color=['green','red','blue','orange','yellow','pink']   

elements=elements.astype(int)
components=components.astype(int)

stats=[]
for k in range(clusters.shape[0]):
    
    
    n_elements=elements[k]
    n_components=components[k]
    n_clusters=clusters[k]
    
    
    pca = PCA(n_components = n_components) 
    X2= pca.fit_transform(X) 
        
    hc=AgglomerativeClustering(n_clusters=n_clusters,affinity='euclidean', linkage='ward')
    y_hc=hc.fit_predict(X2[:n_elements,:])
    
    
    data=graph_data[:n_elements,:]
    
    intraClass=0
    interClass=0
    for i in range(n_clusters):
        intraClass+=np.mean(pairwise.pairwise_distances(X[:n_elements,:][y_hc==i],metric='euclidean'))
        for j in range(i+1,n_clusters):
            interClass+=np.mean(pairwise.pairwise_distances(X[:n_elements,:][y_hc==i],X[:n_elements,:][y_hc==j],metric='euclidean'))
    intraClass=intraClass/n_clusters
    interClass=interClass/((n_clusters*(n_clusters-1))/2)
    
    for i in [5,8,10,11,45]:
        plt.figure()
        for j in range(n_clusters):
            plt.scatter(data[y_hc==j][:,1],data[y_hc==j][:,i],color=color[j])
        plt.xlabel(header[1])
        plt.ylabel(header[i])
        plt.title("clusters="+str(n_clusters)+" PCA_Parameter="+str(n_components))
        plt.show()
        
    stats.append([interClass/intraClass,n_clusters,n_components,n_elements])
    print(str(intraClass)+"  "+str(interClass))
    
#%%

link=np.array([[0,1,0.1,2],[3,2,0.2,3]])
sch.dendrogram(link)
plt.show()