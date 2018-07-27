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

#Call class attributes and put them in Data Frames
eigenvalues = pd.DataFrame(data=pca.explained_variance_, columns = ['Eigenvalues']).set_index([['PC_{}'.format(i) for i in range(1,len(X_normalized.columns)+1)]])
eigenvectors = pd.DataFrame(data =pca.components_, columns = ['PC_{}'.format(i) for i in range(1,len(eigenvalues)+1)]).set_index(X_normalized.columns)
explainedVariance = pd.DataFrame(data=pca.explained_variance_ratio_*100, columns = ['Explained Variance (%)']).set_index([['PC_{}'.format(i) for i in range(1,len(eigenvalues)+1)]])
cumulativeExplainedVariance = pd.DataFrame(data= np.cumsum(pca.explained_variance_ratio_) * 100, columns = ['Cumulative Explained Variance (%)']).set_index([['PC_{}'.format(i) for i in range(1,len(eigenvalues)+1)]])

#Create summary table
eValsTable = eigenvalues.merge(explainedVariance, left_index=True, right_index=True).merge(cumulativeExplainedVariance, left_index=True, right_index=True)

#Plot Eigenvalues
eValsTable['Eigenvalues'].plot.bar(title='Eigenvalues of the correlation matrix').axhline(y=1, color='r', linestyle='--', lw=.5)

#Plot Scree-Plot
eValsTable[['Explained Variance (%)','Cumulative Explained Variance (%)']].plot(title='Scree-plot of PC explained variance')

#Loadings Matrix - Squared correlation of Original Variables and PCs (e.g. variable explained per PC)
loadingMatrix = pd.DataFrame((eigenvectors.T * np.sqrt(eigenvalues))**2, columns = ['PC_{}'.format(i) for i in range(1,len(eigenvalues)+1)])
loadingMatrix.set_index(X_normalized.columns)

#Cumulative Loadings Matrix - Squared correlation of Original Variables and PCs (e.g. variable explained per PC)
cumulativeLoadingMatrix = np.cumsum(loadingMatrix, axis=1)


#%%
'''PCs Selection'''

#Select Principal Components
goldenRulePCs = len([i for i in eigenvalues['Eigenvalues'] if i>1])
print(goldenRulePCs)

#
if not(goldenRulePCs % 2 == 0):
    pcsToExtract = goldenRulePCs + 1
else:
    pcsToExtract = goldenRulePCs

#%%
'''PCs Extraction'''
#Number of PCs is dictated by Golden Rule, +1 if number is odd
pcaExtracted = PCA(n_components = pcsToExtract)

pcMatrixExtracted = pcaExtracted.fit_transform(X_normalized)
pcMatrixExtracted = pd.DataFrame(pcMatrixExtracted)
pcMatrixExtracted.columns = ['PC_{}'.format(i) for i in range(1,len(pcMatrixExtracted.columns)+1)]

#%%
'''Plot results'''
pcMatrixExtracted = pd.DataFrame(df['targetNames']).join(pcMatrixExtracted)

# Use the 'hue' argument to provide a factor variable
sns.lmplot( x="PC_1", y="PC_2", data=pcMatrixExtracted, fit_reg=False, hue='targetNames', legend=True)

#%%
'''Post-Estimation Diagnostics: Squared Cosines Analysis'''

cosines = pd.DataFrame()
cosines['squaredCosine1'] = (pcMatrixExtracted['PC_1']**2) / (X_normalized**2).sum(axis=1)
cosines['squaredCosine2'] = (pcMatrixExtracted['PC_2']**2) / (X_normalized**2).sum(axis=1)
