
# coding: utf-8

# In[58]:


# * from exploring the data, can you build a few key insights 
# that could be shared with a 
# data-curious executive who has not looked at the data?


# In[59]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
get_ipython().run_line_magic('matplotlib', 'inline')


# In[60]:


# read in bank client data
bank_client_data = pd.read_csv('data/bank_client_data.csv', sep=':', index_col='ID')
last_contact_data = pd.read_csv('data/last_contact_data.csv', sep='=', index_col='ID')
campaign_data = pd.read_csv('data/other_data.csv', sep=' ', index_col='ID')
outcome_data = pd.read_csv('data/outcome_data.csv', sep=' ', index_col='ID')
social_data = pd.read_csv('data/social_data.csv', sep=' ', index_col='ID')


# In[61]:


campaign_data.head()


# ### Using describe to look at the variables we can categorise them as:
# 1. bank_client_data 
# numerical variables: age
# categorical variables : job; marital; education; default; housing; loan
# 2. last_contact_data
# numerical variables: duration
# categorical variables: contact; monthl day_of_week
# 3. campaign_data
# numerical variables: campaign; pdays; previous
# categorical: poutcome

# In[62]:


bank_client_data.head()
print(np.sum(bank_client_data['education'] == 'unknown'))
print(np.sum(bank_client_data['default'] == 'unknown'))
print(np.sum(bank_client_data['job'] == 'unknown'))
print(np.sum(bank_client_data['marital'] == 'unknown'))


# ### Going out on a bit of a limb here
# 
# 1. For education I am going to sort by job and age and "fill in" the education levels using ffill as logically education
# standards have changed over time (so age is relevant) and also education probably has a bearing on job to that will also work.
# 2. For default I have two options
#    (a) remove the rows. But that removes 20% of the dataset and I am unhappy with that.
#    (b) fill in the rows but this seems dangerous. However there are only 3 "yes" defaulters in the whole set. So we can either assume all the "unknown" were yes. Or go with the most common value and assume "no" I will fill it in with "no" therefore. However if all but 3 values are "no" it will offer no value.
#    
#    After consideration I will drop default.

# In[63]:


del bank_client_data['default']
bank_client_data.columns


# ### Filling in the "unknowns" in the eduction column
# 

# In[64]:


# Task 1: Sort the dataset in job then age order
bank_client_data.sort_values(['job', 'age'], ascending=[True, True], inplace=True)
# Task 2: Turn unknowns to "NA"
bank_client_data['education'].replace('unknown', np.nan, inplace=True)
# Task 3: Impute the values using bfill (more like the older than younger age)
bank_client_data['education'].fillna(method='bfill', inplace=True)
print(np.sum(bank_client_data['education'] == 'unknown'))
#print(np.sum(bank_client_data['default'] == 'unknown'))


# In[65]:


#removing all other unknowns for job, housing and loan because there is no reasonable way to impute a value
print(bank_client_data.describe())
bank_client_data.replace('unknown', np.nan, inplace=True)
bank_client_data.dropna(inplace=True)
print(bank_client_data.describe())


# ### Campaign data. 
# 
# In the campaign data pays field we have a mixing of data types in effect. The 999 means "never contacted before" whilst the other numbers represents days since last contact.
# The data needs splitting into "those we have not contacted before" and "those we have".
# We will then split the analysis accordingly.
# 
# *** Tasks ***
# 
# 1. split the campaign dataset into two
# 2. join the respective datasets to it to create separate analyses

# In[66]:


previouscontact = campaign_data[campaign_data['pdays'] != 999] #1515 value


# In[67]:


nopreviouscontact = campaign_data[campaign_data['pdays'] == 999] #1515 value


# In[68]:


print(np.unique(nopreviouscontact['poutcome']))
print(np.unique(nopreviouscontact['pdays']))


# ### The no previous contact information has no need for the poutcome or pdays columns
# 
# 1. Since there has been no contact the poutcome and pdays variables have no meaning.

# In[69]:


del nopreviouscontact['poutcome'] # as if no previous contact how can there have been an outcome
del nopreviouscontact['pdays'] # as all values are 999. Therefore will not add to the mix of information.


# ## Join the datasets together

# In[70]:


bank_client_data_with_y = bank_client_data.join(outcome_data)


# In[71]:


previouscontact_with_y = previouscontact.join(outcome_data)


# In[72]:


nopreviouscontact_with_y = nopreviouscontact.join(outcome_data)


# In[73]:


last_contact_data_with_y = last_contact_data.join(outcome_data)


# # Exploring Categorical variables though contingency tables

# In[74]:


def contingencytable(df, columnstoignore, targetcolumn):
    for col in df.columns:
        if col in (columnstoignore):
            continue
        totals = pd.crosstab(df[targetcolumn],df[col]).reset_index()
        row_percentages = pd.crosstab(df[targetcolumn],
            df[col]).apply(lambda row: row/row.sum(),axis=1) #.reset_index()
        col_percentages = pd.crosstab(df[targetcolumn],
            df[col]).apply(lambda cola: cola/cola.sum(),axis=0) #.reset_index()
        ## pret totals for action
        totals_lessy = totals.drop(targetcolumn, axis=1)
        #totals_lessy_andAll = totals_lessy.drop(2, axis='index')

        chi2, p, dof, expected = chi2_contingency(totals_lessy)
        
        if p < 0.05:
            print("Significant contingency table found. P value of {} found for col {}".format(p,col))
            differences = totals_lessy - expected
            print("Differences contingency table: \n {}".format(differences))
            #print(expected)
            #print(totals_lessy)
        print("\n")


# In[75]:


contingencytable(bank_client_data_with_y, ['age','y'], 'y')


# In[76]:


contingencytable(previouscontact_with_y,['pdays','y','campaign','previous'],'y')


# In[77]:


contingencytable(last_contact_data_with_y,['duration','y'],'y')


# # Exploring numerical values of interest

# In[78]:


plt.figure(figsize=(10, 4))
sns.boxplot(data=last_contact_data['duration'])


# In[79]:


sns.boxplot(data=previouscontact['pdays'])


# In[80]:


sns.boxplot(data=previouscontact['campaign'])


# In[81]:


sns.boxplot(data=previouscontact['previous'])


# In[82]:


sns.boxplot(data=bank_client_data['age'])


# ### Considering boxplots:
# #### Question: which variables require transformation or outliers removing
# 1. duration, in last_contact_data, requires transformation due to the very high values
# 2. pday, campaign and previous do have "outliers" but they are in no way extraordinary values
# 3. age - well age does go to 98 (we all hope)

# In[83]:


# this function takes a Series and filters out all elements that are outside
# the range [mu-k*sigma , mu+k*sigma]
def remove_outliers(column_of_data, k=3):
    mu       = np.mean(column_of_data) # get the mean
    sigma    = np.std(column_of_data)  # get the standard deviation
    filtered = column_of_data[(mu - k*sigma < column_of_data) & (column_of_data < mu + k*sigma)]
    return filtered


# In[84]:


print(len(last_contact_data))
filtereddata = remove_outliers(last_contact_data['duration'])
print(len(filtereddata))
print(filtereddata.head())
del last_contact_data['duration']

last_contact_data = last_contact_data.join(filtereddata)
last_contact_data.dropna(inplace=True)


# In[85]:


print(len(last_contact_data))


# In[86]:


last_contact_data.columns


# In[87]:


plt.figure(figsize=(10, 4))
sns.boxplot(data=last_contact_data['duration'])


# ### No previous contact
# 
# We need to create a dataframe of numerical columns.
# Then we need to standardise them and run PCA on them.

# In[88]:


campaign_data.head()


# In[89]:


previouscontact_df = pd.DataFrame(data=bank_client_data['age'])
previouscontact_df = previouscontact_df.join(last_contact_data['duration'],how='inner')
previouscontact_df = previouscontact_df.join(previouscontact[['campaign','pdays','previous']], how='inner')


# In[90]:


previouscontact_df.head()


# In[91]:


previouscontact_df.describe()


# In[92]:


previouscontact_df.apply(pd.to_numeric)


# In[93]:


# Initialise the scaler
scaler = MinMaxScaler() 

# Apply auto-scaling (or any other type of scaling) and cast to DataFrame 
previouscontact_scale = pd.DataFrame(
                    scaler.fit_transform(previouscontact_df), 
                    columns = previouscontact_df.columns, 
                    index = previouscontact_df.index)

# Print the first rows
previouscontact_scale.head()


# In[94]:


previouscontact_scale.describe()


# ## Run PCA

# In[95]:


pca = PCA()
previouscontact_pca = pd.DataFrame(
                        pca.fit_transform(previouscontact_scale), 
                        columns = ['PC'+str(i+1) for i in range(previouscontact_scale.shape[1])], 
                        index   = previouscontact_scale.index)


# ### Plot the PCA components in a descending bar plot

# In[96]:


import seaborn as sns

pca_full = PCA()
pca_full.fit(previouscontact_pca)

plt.figure(figsize=(8, 6))
sns.barplot(x=np.arange(0,pca_full.n_components_)+1, y=pca_full.explained_variance_ratio_)
plt.xticks(np.arange(0, pca_full.n_components_, 10), np.arange(0, pca_full.n_components_, 10))

plt.figure(figsize=(8, 6))
sns.barplot(np.arange(pca_full.n_components_)+1, np.cumsum(pca_full.explained_variance_ratio_))
plt.xticks(np.arange(0, pca_full.n_components_, 10), np.arange(0, pca_full.n_components_, 10))


# ## Equally weighted components

# In[97]:


# Create the PCA loadings matrix and show the loadings
previouscontact_pca_loadings = pd.DataFrame(pca.components_, 
                        columns=previouscontact_scale.columns,
                        index=previouscontact_pca.columns)


# In[98]:


print(previouscontact_pca_loadings)


# In[99]:


previouscontact_pca_loadingsT = previouscontact_pca_loadings.transpose()


# In[100]:


previouscontact_pca_loadingsT


# ### This is the end of the previous contact data analysis. Given where we are, it would seem best to go back and remove some of the low variance variables from the dataset before continuing with the analysis.

# In[101]:


# remove the ys used in the contingency tables
nopreviouscontact_df = pd.DataFrame(data=bank_client_data['age'])
nopreviouscontact_df = nopreviouscontact_df.join(last_contact_data['duration'],how='inner')
nopreviouscontact_df = nopreviouscontact_df.join(nopreviouscontact[['campaign','previous']], how='inner')


# In[102]:


# Initialise the scaler
scaler = MinMaxScaler() 

# Apply auto-scaling (or any other type of scaling) and cast to DataFrame 
nopreviouscontact_scale = pd.DataFrame(
                    scaler.fit_transform(nopreviouscontact_df), 
                    columns = nopreviouscontact_df.columns, 
                    index = nopreviouscontact_df.index)

# Print the first rows
nopreviouscontact_scale.head()


# In[103]:


pca = PCA()
nopreviouscontact_pca = pd.DataFrame(
                        pca.fit_transform(nopreviouscontact_scale), 
                        columns = ['PC'+str(i+1) for i in range(nopreviouscontact_scale.shape[1])], 
                        index   = nopreviouscontact_scale.index)


# In[104]:


pca_full = PCA()
pca_full.fit(nopreviouscontact_pca)

plt.figure(figsize=(8, 6))
sns.barplot(x=np.arange(0,pca_full.n_components_)+1, y=pca_full.explained_variance_ratio_)
plt.xticks(np.arange(0, pca_full.n_components_, 10), np.arange(0, pca_full.n_components_, 10))

plt.figure(figsize=(8, 6))
sns.barplot(np.arange(pca_full.n_components_)+1, np.cumsum(pca_full.explained_variance_ratio_))
plt.xticks(np.arange(0, pca_full.n_components_, 10), np.arange(0, pca_full.n_components_, 10))


# In[105]:


# Create the PCA loadings matrix and show the loadings
nopreviouscontact_pca_loadings = pd.DataFrame(pca.components_, 
                        columns=nopreviouscontact_scale.columns,
                        index=nopreviouscontact_pca.columns)


# In[106]:


nopreviouscontact_pca_loadingsT = nopreviouscontact_pca_loadings.transpose()
nopreviouscontact_pca_loadingsT


# In[107]:


previouscontact_pca_loadingsT


# ### Under Standard Scaling we got:
# For the Data Hungry Executive
# 
# Nuggets of information
# 
# 1. Whether a person has been contacted or before is a good split of the data as different features impact the dataset
# 1a. For the No previous contact group the first principal component is dominated by the duration of call to the customer.
# 1b. For the people whom have been contacted before the first principal component is influenced by the amount of time since last contact, the duration of the call and their age. The second principal component, also important, is influcend by age and duration.
# 
# These features need exploring in more detail to determine what they could tell us. However, they certainly seem to explain lots of what is going on so we can be hopeful models will bear fruit and insight.
# 
# 2. The contingency table analysis tells us that some of the levels within the categorical factors have a great affect on the outcome. As an example, taken in isolation, blue collar workers were less likely to take up the product proportionately. Again, to explain these effects further, a proper model would be need to see how that level of that feature (blue collar of jon) worked with all the other features together.

# ### What would I do next?
# 
# Well, we have some data with an output variable so it would be interesting to create a linear regression; for which we would need one hot coding of the categorical variables. See below for the one hot encoding.

# In[108]:


# join all the tables of data together
# bank_data
listofcolumns = ['job','marital','education','housing','loan']
for col in listofcolumns:
    onecolofdata = bank_client_data[col]

    onehotencoding = pd.get_dummies(onecolofdata)
    
    onehotencoding.columns = [col + "_" + entry for entry in onehotencoding.columns]
    #del(bank_client_data[col])

    bank_nonan = bank_client_data.join(onehotencoding, rsuffix='_' + col)

listoflastcols = ['contact','month','day_of_week'] #contact is cellular or phone
for col in listoflastcols:
    onecolofdata = last_contact_data[col]

    onehotencoding = pd.get_dummies(onecolofdata)
    
    onehotencoding.columns = [col + "_" + entry for entry in onehotencoding.columns]

    last_contact_data = last_contact_data.join(onehotencoding, rsuffix='_' + col)

outcome_data
listofoutcomecols = ['y']
for col in listofoutcomecols:
    onecolofdata = outcome_data[col]

    onehotencoding = pd.get_dummies(onecolofdata)
    
    onehotencoding.columns = [col + "_" + entry for entry in onehotencoding.columns]

    outcome_data = outcome_data.join(onehotencoding, rsuffix='_' + col)

