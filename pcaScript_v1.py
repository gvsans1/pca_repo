# -*- coding: utf-8 -*-

#%%
import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import seaborn as sns

#%%

'''Import some data to play with'''
iris = datasets.load_iris()

#%%

'''Create Dataset'''
X = pd.DataFrame(data = iris.data, columns = iris.feature_names)
y = pd.DataFrame(data = iris.target, columns = ['target'])
nms = pd.DataFrame(data = iris.target_names, columns = ['targetNames'])

df = y.join(X)
df = df.merge(nms, left_on=df.target, right_index=True)


#%%

'''Standardize Data'''

X_normalized = StandardScaler().fit_transform(X)
X_normalized = pd.DataFrame(X_normalized, columns = X.columns)

#%%
'''Exploratory analysis'''

#Correlation Matrix
cm = X_normalized.corr()

#Extract Eigenvalues and Eigenvectors
#Instantiate PCA class
pca = PCA(n_components = len(X.columns))
pcMatrix = pca.fit_transform(X_normalized)

#Call class attributes
eigenvalues = pca.explained_variance_
eigenvectors = pca.components_
explaiedVariance = pca.explained_variance_ratio_

#Create summary table
eValsTable = pd.DataFrame(data = [eigenvalues, explaiedVariance * 100, np.cumsum(explaiedVariance) * 100], columns = ['PC {}'.format(i) for i in range(1,len(eigenvalues)+1)]).T
eValsTable.columns = ['Eigenvalues', 'Explained variance (%)','Cumulative explained variance (%)']

#Plot Eigenvalues
eValsTable['Eigenvalues'].plot.bar(title='Eigenvalues of the correlation matrix').axhline(y=1, color='r', linestyle='--', lw=.5)

#Plot Scree-Plot
eValsTable[['Explained variance (%)','Cumulative explained variance (%)']].plot(title='Scree-plot of PC explained variance')

#Loadings Matrix - Squared correlation of Original Variables and PCs (e.g. variable explained per PC)
loadingMatrix = pd.DataFrame((eigenvectors.T * np.sqrt(eigenvalues))**2, columns = ['PC {}'.format(i) for i in range(1,len(eigenvalues)+1)])
loadingMatrix.set_index(X_normalized.columns)

#Cumulative Loadings Matrix - Squared correlation of Original Variables and PCs (e.g. variable explained per PC)
cumulativeLoadingMatrix = np.cumsum(loadingMatrix, axis=1)


#%%
'''PCs Selection'''

#Select Principal Components
goldenRulePCs = len([i for i in eigenvalues if i>1])
print(goldenRulePCs)

#
if not(goldenRulePCs % 2 == 0):
    pcsToExtract = goldenRulePCs + 1
else:
    pcsToExtract = goldenRulePCs

#%%
'''PCs Extraction'''
pcaExtracted = PCA(n_components = pcsToExtract)
pcMatrixExtracted = pcaExtracted.fit_transform(X_normalized)
pcMatrixExtracted = pd.DataFrame(pcMatrixExtracted)
pcMatrixExtracted.columns = ['PC {}'.format(i) for i in range(1,len(pcMatrixExtracted.columns)+1)]

#%%
'''Plot results'''
pcMatrixExtracted = pd.DataFrame(df['targetNames']).join(pcMatrixExtracted)

# Use the 'hue' argument to provide a factor variable
sns.lmplot( x="PC 1", y="PC 2", data=pcMatrixExtracted, fit_reg=False, hue='targetNames', legend=True)

#%%
'''Post-Estimation Diagnostics: Squared Cosines Analysis'''

cosines = pd.DataFrame()
cosines['squaredCosine1'] = (pcMatrixExtracted['PC 1']**2) / (X_normalized**2).sum(axis=1)
cosines['squaredCosine2'] = (pcMatrixExtracted['PC 2']**2) / (X_normalized**2).sum(axis=1)