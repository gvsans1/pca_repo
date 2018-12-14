# -*- coding: utf-8 -*-

# %%
import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import seaborn as sns

# %%

"""
Import some data to play with
"""
iris = datasets.load_iris()

# %%

"""
Create Dataset
"""
# Create dataframe with flower properties
X_vars_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)

# Create dataframe with target variable
y_var_df = pd.DataFrame(data=iris.target, columns=['target'])
target_names_df = pd.DataFrame(data=iris.target_names,
                               columns=['TARGET_NAMES'])

# Join X and y in a single df. Merge target names
df = y_var_df.join(X_vars_df)
df = df.merge(target_names_df, left_on=df.target, right_index=True)

# %%

"""
Standardize X Data
"""
X_normalized = StandardScaler().fit_transform(X_vars_df)
X_normalized = pd.DataFrame(X_normalized, columns=X_vars_df.columns)

# %%
"""
Exploratory analysis
"""
# Obtain the correlation matrix
correlation_matrix = X_normalized.corr()

# Extract eigenvalues and eigenvectors from the correlation matrix
# Instantiate PCA object selecting all possible dimensions,
# i.e. len(original_variables)
pca = PCA(n_components=len(X_vars_df.columns))
pc_matrix = pca.fit_transform(X_normalized)

# Extract eigenvalues and eigenvectors from class properties and store in dfs
# --Get index for eigenvalues
eigenvalues_index = \
    ['PC_{}'.format(i) for i in range(1, len(X_normalized.columns) + 1)]

# --Get actual eigenvalues df
eigenvalues_df = \
    pd.DataFrame(data=pca.explained_variance_, columns=['Eigenvalues']
                 ).set_index([eigenvalues_index])

# --Get columns for eigenvectors
eigenvectors_cols = \
    ['PC_{}'.format(i) for i in range(1, len(eigenvalues_df) + 1)]

# --Get actual eigenvectors df
eigenvectors = \
    pd.DataFrame(data=pca.components_, columns=eigenvectors_cols
                 ).set_index(X_normalized.columns)

# Compute explained variance from eigenvalues
# --Define index for explained variance
explained_variance_index = \
    ['PC_{}'.format(i) for i in range(1, len(eigenvalues_df) + 1)]

# --Obtain explained variance
explained_variance = \
    pd.DataFrame(data=(pca.explained_variance_ratio_ * 100),
                 columns=['Explained Variance (%)']
                 ).set_index([explained_variance_index])

# --Define index for cumulative explained variance
cumulative_explained_variance_index = \
    ['PC_{}'.format(i) for i in range(1, len(eigenvalues_df) + 1)]


# --Obtain cumulative explained variance
cumulative_explained_variance = \
    pd.DataFrame(data=np.cumsum(pca.explained_variance_ratio_) * 100,
                 columns=['Cumulative Explained Variance (%)']
                 ).set_index([cumulative_explained_variance_index])

# Create summary table of explained variance by merging
# explained and cumulative variance
variance_summary_table = \
    eigenvalues_df.merge(
            explained_variance,
            left_index=True,
            right_index=True
            ).merge(
                    cumulative_explained_variance,
                    left_index=True,
                    right_index=True)

# Support analysis by plotting
# --Plot Eigenvalues
variance_summary_table['Eigenvalues'].plot.bar(
    title='Eigenvalues of the correlation matrix').axhline(y=1,color='r',
                                                           linestyle='--',
                                                           lw=.5)

# --Plot Scree-Plot
variance_summary_table[
    ['Explained Variance (%)', 'Cumulative Explained Variance (%)']].plot(
    title='Scree-plot of PC explained variance')

# Compute the Loadings Matrix
# This is defined as the squared correlation between original variables and PCs
# (i.e. the variable explained per PC)
loading_matrix_columns = \
    ['PC_{}'.format(i) for i in range(1, len(eigenvalues_df) + 1)]

loading_matrix = pd.DataFrame(
        data=(eigenvectors.T * np.sqrt(eigenvalues_df)) ** 2,
        columns=loading_matrix_columns)

loading_matrix.set_index(X_normalized.columns, inplace=True)

# Compute the Cumulative Loadings Matrix
cumulative_loading_matrix = np.cumsum(loading_matrix, axis=1)

# %%
"""
Perform PCs Selection
"""

# Select Principal Components according to the 'Golden Rule'
# The Golden Rule suggests to take all components with an eigenvalue>1,
# i.e. with a variance higher than the one of the original variable
golden_rule_pcs = len([i for i in eigenvalues_df['Eigenvalues'] if i > 1])
print('INFO: Golden rule suggests {} PCs'.format(golden_rule_pcs))

# Define how many PCs to extract. Add one dimension if golden rule suggests
# an odd number of PCs (for plotting convenience further on)

if not (golden_rule_pcs % 2 == 0):
    pcs_to_extract = golden_rule_pcs + 1

else:
    pcs_to_extract = golden_rule_pcs

# %%
"""
Perfrom PCs Extraction based on Selection above
"""
# Instantiate PCA object with selected number of PCs to extract
# Number of PCs is dictated by Golden Rule, +1 if number is odd
pca_extracted_object = PCA(n_components=pcs_to_extract)
pc_matrix_extracted_df = pca_extracted_object.fit_transform(X_normalized)

# Store results in a dataframe
pc_matrix_extracted_df = pd.DataFrame(pc_matrix_extracted_df)

# --Amend column names dynamically, based on axis=1 shape of dataframe
pc_matrix_extracted_df.columns = ['PC_{}'.format(i) for i in
                             range(1, len(pc_matrix_extracted_df.columns) + 1)]

# %%
"""
Plot results
"""

# -- Merge target names and principal components
pc_matrix_extracted_df = \
    pd.DataFrame(df['TARGET_NAMES']).join(pc_matrix_extracted_df)

# Plot the pc_matrix_extracted_df along the extracted principal components
# Use the 'hue' argument to provide a factor variable
sns.lmplot(x="PC_1",
           y="PC_2",
           data=pc_matrix_extracted_df,
           fit_reg=False,
           hue='TARGET_NAMES',
           legend=True)

# %%
"""
Post-Estimation Diagnostics: Squared Cosines Analysis

A Squared cosine is defined as the square root of a PC,
divided by the sum of the squared original variables
"""

# Instantiate an empty dataframe to store cosines
cosines = pd.DataFrame()

# Peform computations by looping over all extracted principal components
extracted_pcs_list = [i for i in pc_matrix_extracted_df if i.startswith('PC')]

for pc in extracted_pcs_list:
    cosines['squared_cosine_{}'.format(pc)] = \
        (pc_matrix_extracted_df[pc] ** 2) / \
        (X_normalized ** 2).sum(axis=1)
