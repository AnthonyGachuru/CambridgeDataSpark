"""
PART 2

This file should contain the code that you wrote to explore the data set
Number-and-types-of-applications-by-all-account-customers-2017-06.csv

The exploratory data analysis can typically done in a notebook.
Export the code and copy it in this file for your final submission.
You will also receive feedback on code aesthetic and PEP8 compliance.
"""

# coding: utf-8

# ### Second part
# 
# * In this second dataset (`Number-and-types-of-applications-by-all-account-customers-2017-06.csv`) 
#   you will be performing clustering analysis. This dataset does not have categorical data.
# 
# * what can you see in the data? how do you interpret what you see?

# In[59]:


#for data exploration
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency


# In[60]:


# for plotting
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[61]:


# for feature engineering
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


# In[62]:


# for clustering
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.metrics.cluster import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage


# In[63]:


applications = pd.read_csv('data/Number-and-types-of-applications-by-all-account-customers-2017-06.csv', sep=',', index_col='Account Customer')


# In[64]:


print("applications shape is {}".format(applications.shape))
applications.describe()


# In[65]:


applications.head()


# In[66]:


# the variables are counts of applications. We are looking to cluster the data. I do not see that there will be outliers.
# NaN: Check for Nan.
# Optional (if time). Like to do a count of unique values in column.
# Outliers: In this case I will not be removing outiers simply because outliers are believed to be a factor or data collation.
#           Here, each is simply the count of the number of applications and larger values may clump together
# Standardisation: I am going to run two algorithms. One with standardised data (first) and then one without (second).
# Co-dependency: Check relationships between variables.
# Variance Thresholding: <optional> interesting here in that 
# Clustering: Then run clustering algorhythm


# ## Check for NaN values
# 
# *** Tasks ***
# 
# * `NaN` - check for whether there are any "Not-a-numbers"
# * `Null` - check for nulls in the dataset

# In[67]:


beforenanshape = applications.shape
applications.dropna(inplace = True)
afternanshape = applications.shape
print("Running the nan process has removed {} rows.".format(beforenanshape[0] - afternanshape[0]))


# In[68]:


nullcheck = pd.isnull(applications).sum()
#print(type(nullcheck))
#print(nullcheck)
if (nullcheck.sum() > 0):
    print("Need to remove nulls from dataset")
    # <optional> add code in that checks shape. Removes nulls. Checks shape again and outputs message.


# ## Drop the Total Row

# In[69]:


del applications['Total']


# ## Standardise data

# In[70]:


# Initialise the scaler
scaler = StandardScaler() 

# Apply auto-scaling (or any other type of scaling) and cast to DataFrame 
applications = pd.DataFrame(
                    scaler.fit_transform(applications), 
                    columns = applications.columns, 
                    index = applications.index)

# Print the first rows
applications.head()


# #### Run a boxplot to look at standardised data

# In[71]:


plt.figure(figsize=(17, 7))
sns.boxplot(data=applications)


# *** Notes ***
# * There are still a lot of outliers after standardisation. I am going to run a clustering of the data anyway with the outliers in. 
# * I may run an alternative with calculating outliers

# ## Relationships between features
# 
# *** Tasks ***
# 
# * First run pairplots between variables to see is there is co-variation
# * Then run a heatmap of correlations.

# In[72]:


sns.pairplot(applications)


# #### There appear to be a few variables that are co-varying. So best to run the heatmap

# In[73]:


def runcovarianceheatmap(dataframeoffeatures):
    dfcorrelationmatrix = dataframeoffeatures.corr()

    plt.figure(figsize=(9,7))
    sns.heatmap(data=dfcorrelationmatrix, 
            linewidths=0.5, 
            cmap="RdBu", 
            vmin=-1, 
            vmax=1, 
            annot=True)

    plt.xticks(rotation=270);


# In[74]:


runcovarianceheatmap(applications)


# #### Notes from the heatmap:
# * OS(P) seems to have strong relationships with DFL and TP
# * OS(NPP) seems to have no strong relationships with anything
# * OC1 and OS(W) seem to have a strong relationship
# * OC1 and OC2 are strongly covariant
# 
# *** Tasks ***
# * Remove OC1 from the dataset
# * Remove OS(P) from the dataset

# In[75]:


applications_nocovariance = applications.copy(deep=True)
# remove columns listed above
del applications_nocovariance['OC1']
del applications_nocovariance['OS(P)']
print(applications.columns)
print(applications_nocovariance.columns)


# In[76]:


runcovarianceheatmap(applications_nocovariance)


# ## Cluster data
# 
# *** Tasks ***
# Note:- We have no way of knowing how many clusters we need. There are some 10,000 observations so potentially there are a fair few clusters.
# So I am going to take the following approach
# * 1 Compute cluster
# * 2 check silhouette score
# * 2a print out number of clusters and silhouette score

# In[77]:


for numofclusters in range(2,11,1):
    kmeans = KMeans(n_clusters = numofclusters)
    kmeans.fit(applications_nocovariance)
    cluster_assignment = kmeans.predict(applications_nocovariance)
    silscore = silhouette_score(applications_nocovariance, cluster_assignment)
    print("For cluster size {} we get a silhouette score of {}.".format(numofclusters, silscore))


# #### From the values above it seems 2 or 7 are sensible as
# *  2 has a really high silhouette score
# * 3 through 7 have roughly equivalent scores before we drop above 0.05 so therefore let us look at 7 
#   which for 100 observations seems sensible
# 
# *** Tasks ***
# * write a function that takes a dataframe and the number of clusters and prints a clustering diagram

# In[78]:


# This function generates a pairplot enhanced with the result of k-means
def pairplot_cluster(df, cols, cluster_assignment):
    """
    df: dataframe that contains the data to plot 
    cols: columns to consider for the plot
    cluster_assignments: cluster asignment returned by the clustering algorithm
    """
    # seaborn will color the samples according to the column cluster
    df['cluster'] = cluster_assignment 
    sns.pairplot(df, vars=cols, hue='cluster')
    df.drop('cluster', axis=1, inplace=True)


# In[79]:


def showclusters(data, numofclusters, listofcolstodisplay):
    kmeans = KMeans(n_clusters = numofclusters)
    kmeans.fit(data)
    cluster_assignment = kmeans.predict(data)
    pairplot_cluster(data, listofcolstodisplay, cluster_assignment)


# In[80]:


showclusters(applications_nocovariance, 2, ['FR','DFL','TP','DLG'])


# In[81]:


showclusters(applications_nocovariance, 7, ['FR','DFL','TP','DLG'])


# In[82]:


showclusters(applications_nocovariance, 7, ['OS(W)','OS(NPW)','OS(NPP)','SIMS','OC2'])


# In[83]:


showclusters(applications_nocovariance, 7, ['DLG','OC2','OS(NPW)','OS(W)'])


# ## Thoughts and conclusions
# * Seven clusters works well.
# * The OC2 vs DLG has a really good represenation of the clusters as they start to split up.
# * Clusters I can see
# *    PINK - Low value of OC2 and high value of OS(NPW) see OC2 vs OS(NPW)
# *    MAROON - Low value of OS(NPW) and any value of DLG!
# *    GREEN - High value OS(W) and 0 (or low) value of OS(NPW)
# *    BLUE - low value of OC2 and 0 to 20 on OS (NPW)

# ### Looking at the clusters
# 
# So we have 7 clusters and it is time to have a look at them
# 
# *** Tasks ***
# 
# * Look at centroids of clusters
# * Split clusters up (into different dataframes) and have a look at what describe tells us
# * <optional> Remember the data is standardised so probably need to un standardise it

# In[84]:


# centroids of clusters
kmeans = KMeans(n_clusters = 7)
kmeans.fit(applications_nocovariance)
cluster_assignment = kmeans.predict(applications_nocovariance)
centers_df = pd.DataFrame(data=kmeans.cluster_centers_, 
                          columns=applications_nocovariance.columns)
print(centers_df)


# In[85]:


print(type(cluster_assignment))
applications_nocovariance['cluster_assignment'] = cluster_assignment


# In[86]:


applications_nocovariance.describe()


# In[87]:


applications_nocovariance[cluster_assignment == 0].describe()


# In[88]:


applications_nocovariance[cluster_assignment == 1].describe()


# In[89]:


applications_nocovariance[cluster_assignment == 2].describe()


# In[90]:


applications_nocovariance[cluster_assignment == 3].describe()


# In[91]:


applications_nocovariance[cluster_assignment == 4].describe()


# In[92]:


applications_nocovariance[cluster_assignment == 5].describe()


# In[93]:


applications_nocovariance[cluster_assignment == 6].describe()


# ## Trying DBScan because one all the data are clustered together
# 

# In[94]:


del applications_nocovariance['cluster_assignment']


# In[95]:


dbscan = DBSCAN(eps=0.3)  # use the default epsilon for now

dbscan_assignment = dbscan.fit_predict(applications_nocovariance)


# In[104]:


np.unique(dbscan.labels_)


# In[106]:


applications_nocovariance['dbscan_assignment'] = dbscan.labels_


# In[109]:


for label in np.unique(dbscan.labels_):
    print(applications_nocovariance[dbscan_assignment == label].describe())

