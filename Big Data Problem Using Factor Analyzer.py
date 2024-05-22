#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Importing important Libraries
import pandas as pd 
import matplotlib.pyplot as plt
from factor_analyzer import FactorAnalyzer
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity
from factor_analyzer import calculate_kmo
import seaborn as sns


# In[2]:


#Read The CSV file From The Path: C:\\Users\\lenovo\\OneDrive\Desktop\\BigDataAssignment\\FactorAnalysis.csv and print it
df=pd.read_csv("C:\\Users\\lenovo\\OneDrive\Desktop\\BigDataAssignment\\FactorAnalysis.csv")
print(df)


# In[3]:


# showing the type of the data in every column
print(df.dtypes)


# In[4]:


# showing the dimesion of the dataset
df.shape


# In[5]:


# returning list with column names
df.columns.values.tolist()


# In[6]:


# showing some data 
# read the first 20 row of the data
df.head(20)


# In[7]:


# read the last 20 row of the data
df.tail(20)


# ## from the data shown above we notice that data has some problems so it need to be cleaned first

# In[8]:


# we start by putting the columns of the dataset inside variable called cols to start process and clean the data
cols = df.columns


# In[9]:


# this code will return boolean value if there any missing values in each column of the dataset
df[cols].isna().any()


# In[10]:


# from the above code we know now that each column has missing data inside it 
# so now we need to know how many missing values in each column
df[cols].isnull().sum()


# In[11]:


# now after we know how many missing values in each column we will start to handle it
# we will follow two ways to handle missing data : 1 - drop it , 2 - replace it
# first drop it even it is not best way for dealing with small datasets
df.dropna(subset=['Appearance' , 'Communication' , 'Company Fit' , 'Experience' ,'Letter' , 'Academic record' , 'Job Fit'] , inplace=True)


# In[12]:


# second replcae it by the mean value and median value and zero
# Replacing by mean in the column Likeability 
rep_mean = df['Likeability'].mean()
df['Likeability'].fillna(rep_mean , inplace = True)


# In[13]:


# Replacing by median in the column Organization 
rep_median = df['Organization'].median()
df['Organization'].fillna(rep_median , inplace = True)


# In[14]:


# Replacing the rest of columns by zero value
df.fillna(0 , inplace = True)


# In[15]:


# check now weather if there is any missing values
df[cols].isnull().sum()


# In[16]:


# the code above shown there no missing data anymore 
# after checking the data visually there is no wrong formats or outliers
# now we will check the duplicated records if there any 
print(df.duplicated().to_string())


# In[17]:


# as we can see there is some duplicates then we will remove it
df.drop_duplicates(inplace = True)


# In[18]:


# know we will check again if there are any duplicated values
print(df.duplicated())


# ## now data is very clean and ready to be used

# In[19]:


# first we will the dataframe structure , including the number of rows and columns and data type of column and amount of memory used by dataframe using info function
df.info()


# In[20]:


# we will calculate the bartlett's test of sphericity which is stastical test that check weather the observed variables in adataset are uncorrelated or not
# chi square value => measure of the difference between observed correlation matrix and identity matrix
# p_value => probabillity of obtaining a chi square value as extereme as one observed
chi_square_value , p_value = calculate_bartlett_sphericity(df)
chi_square_value , p_value


# In[21]:


#in this code we will calculate kmo_all and kmo_model using the calculate_kmo function
# first kmo is stand fot kaiser-meyer-olkin and it measure of samoling adequacy of each variable in the dataframe
# kmo_all => is the ovreall kmo measure of all variables
# kmo_mode => is the KMO measure of the subset of variables that are suitable for factor analysis
kmo_all,kmo_model = calculate_kmo(df)
print(kmo_model)
print(kmo_all)


# In[22]:


# since kmo_model in the above code was greater than 0.6 then we can perform factor analysis
# in this code we perform factor analysis on the data using 12 factors like the number of variables in our dataset
fa = FactorAnalyzer(n_factors=12, rotation=None)
fa.fit(df)


# In[23]:


# know we will check the eigen values to know the best number of factors 
ev , v = fa.get_eigenvalues()
ev


# In[24]:


# since eigen values less than 1 should be dropped 
# then we have only 4 factors to perform on the data
# we will use here also varimax rotation method to simplfy the factor structure and make it easier to interpret
fa = FactorAnalyzer(n_factors=4, rotation="varimax")
fa.fit(df)


# In[25]:


# know we will print loading matrix which show the realtionship between the observed variables and underlying factors
loadings = pd.DataFrame(fa.loadings_,index = df.columns)
# Printloadings
loadings


# In[26]:


# know we will make some visualliztions
# visualization by screeplot by get eigenvalue
# this graph will asure that best number of factors is 4
ev,v=fa.get_eigenvalues()
plt.scatter(range(1,len(ev)+1),ev)
plt.plot(range(1, len(ev) + 1),ev)
plt.title('screeplot')
plt.xlabel('factor')
plt.ylabel('eigenvalue')
plt.grid()
plt.show()


# In[27]:


# Heatmap of Factor Loadings to show us which column in belong to which factor
sns.heatmap(loadings, annot=True, cmap='coolwarm')
plt.title('Factor Loadings Heatmap')
plt.show()


# In[28]:


# Bar Plot of Factor Loadings to show us which column in belong to which factor
loadings_abs = loadings.abs()  
loadings_abs.plot(kind='bar')
plt.title('Factor Loadings (Absolute Values)')
plt.show()


# In[29]:


#At the end my conclusion and observations on the result :
# columns that affected by factor 0 : Academic record , Experience , Job Fit , Potential
# columns that affected by factor 1 : Communication , Company Fit , Organization
# columns that affected by factor 2 : Appearance , Likeability , Self-Confidence
# columns that affected by factor 3 : Letter , Resume


# In[ ]:




