#!/usr/bin/env python
# coding: utf-8

# # DESCRIPTION
# 
# ### Background of Problem Statement :
# 
# #### The US Census Bureau has published California Census Data which has 10 types of metrics such as the population, median income, median housing price, and so on for each block group in California. The dataset also serves as an input for project scoping and tries to specify the functional and nonfunctional requirements for it.
# 
# ## Problem Objective :
# 
# #### The project aims at building a model of housing prices to predict median house values in California using the provided dataset. This model should learn from the data and be able to predict the median housing price in any district, given all the other metrics.
# 
# #### Districts or block groups are the smallest geographical units for which the US Census Bureau publishes sample data (a block group typically has a population of 600 to 3,000 people). There are 20,640 districts in the project dataset.
# 
# ### Domain: Finance and Housing
# 
# ### Analysis Tasks to be performed:
# 
# 1. Build a model of housing prices to predict median house values in California using the provided dataset.
# 
# 2. Train the model to learn from the data to predict the median housing price in any district, given all the other metrics.
# 
# 3. Predict housing prices based on median_income and plot the regression chart for it.
# 
# ##### 1. Load the data :
# 
# Read the “housing.csv” file from the folder into the program.
# Print first few rows of this data.
# Extract input (X) and output (Y) data from the dataset.
# 
# #### 2. Handle missing values :
# 
# Fill the missing values with the mean of the respective column.
# 
# #### 3. Encode categorical data :
# 
# Convert categorical column in the dataset to numerical data.
# 
# #### 4. Split the dataset : 
# 
# Split the data into 80% training dataset and 20% test dataset.
# 
# #### 5. Standardize data :
# 
# Standardize training and test datasets.
# 
# #### 6. Perform Linear Regression : 
# 
# Perform Linear Regression on training data.
# Predict output for test dataset using the fitted model.
# Print root mean squared error (RMSE) from Linear Regression.
#             [ HINT: Import mean_squared_error from sklearn.metrics ]
# 
# ### 7. Bonus exercise: Perform Linear Regression with one independent variable :
# 
# Extract just the median_income column from the independent variables (from X_train and X_test).
# 
# Perform Linear Regression to predict housing values based on median_income.
# 
# Predict output for test dataset using the fitted model.
# Plot the fitted model for training data as well as for test data to check if the fitted model satisfies the test data.
# 
# Dataset Description :
# 
# 
# Field	Description
# longitude	(signed numeric - float) : Longitude value for the block in California, USA
# 
# latitude	(numeric - float ) : Latitude value for the block in California, USA
# 
# housing_median_age	(numeric - int ) : Median age of the house in the block
# 
# total_rooms	(numeric - int ) : Count of the total number of rooms (excluding bedrooms) in all houses in the block
# 
# total_bedrooms	(numeric - float ) : Count of the total number of bedrooms in all houses in the block
# 
# population	(numeric - int ) : Count of the total number of population in the block
# 
# households	(numeric - int ) : Count of the total number of households in the block
# 
# median_income	(numeric - float ) : Median of the total household income of all the houses in the block
# 
# ocean_proximity	(numeric - categorical ) : Type of the landscape of the block [ Unique Values : 'NEAR BAY', '<1H OCEAN', 'INLAND', 'NEAR OCEAN', 'ISLAND'  ]
# 
# median_house_value	(numeric - int ) : Median of the household prices of all the houses in the block

# In[426]:


import pandas as pd
import matplotlib.pyplot as plt


# In[427]:


housing = pd.read_csv("/Users/anirudhagarwal/Desktop/Anirudh/OneDrive/Purdue DS Course/Course 5/My Project/P 2/housing.csv")


# In[428]:


housing.head(5)    # To check all the columns and also to check whether the data has been loaded properly


# In[429]:


housing1 = housing.copy()


# In[430]:


housing1.describe().round(2)


# In[431]:


housing1.info()


# In[432]:


hou= housing1.copy()


# In[433]:


#Extract input (X) and output (Y) data from the dataset.
X = hou.iloc[:, :-1]                # all parameters except median_house_value are independent variables
Y = hou.iloc[:, [-1]]               # median_ house_value is dependent a variable


# In[434]:


X


# In[435]:


Y


# In[436]:


# To check the number of missing values in all columns
hou.isna().sum()


# ### total_bedrooms has 207 missing values as shown above.
# ### I will replace these values with the column's mean value

# In[437]:


hou['total_bedrooms'] = hou['total_bedrooms'].fillna(value = hou['total_bedrooms'].mean())


# In[438]:


hou.isna().sum()        # There are no blanks anymore


# ### To convert categorical column of the data into numerical data
# #### From the data , we can see that only ocean_proximity column has categories like 
# #### NEAR BAY, INLAD, ISLAND , NEAR OCEAN AND <1HOCEAN.

# In[439]:


housing1[['ocean_proximity']]


# In[440]:


from sklearn.preprocessing import LabelEncoder


# In[441]:


hou['ocean_proximity'].nunique()


# In[442]:


le = LabelEncoder()
hou['ocean_proximity']=le.fit_transform(hou['ocean_proximity'])


# In[443]:


hou[['ocean_proximity']]  # Now the data is converted into numeric values


# ### Split the data into 80% training dataset and 20% test dataset.

# In[444]:


from sklearn.model_selection import train_test_split


# In[445]:


train_set, test_set = train_test_split(hou, test_size=0.2, random_state=123)


# In[446]:


train_set                    #16512 rows = 80% of data


# In[447]:


test_set                   #4128 rows = 20% of data


# ### Standardize training and test datasets

# In[448]:


from sklearn.preprocessing import StandardScaler 


# In[449]:


#For training data set
scaler = StandardScaler()
train_scaled = scaler.fit_transform(train_set)
train_scaled = pd.DataFrame(train_scaled, columns = train_set.columns)
train_scaled


# In[450]:


# For test data set
test_scaled = scaler.fit_transform(test_set)
test_scaled = pd.DataFrame(test_scaled, columns = test_set.columns)
test_scaled


# # Perform Linear Regression : 
# 
# ### Perform Linear Regression on training data.
# ### Predict output for test dataset using the fitted model.
# ### Print root mean squared error (RMSE) from Linear Regression.

# In[451]:


from sklearn.linear_model import LinearRegression
LR1 = LinearRegression()


# In[452]:


#To classify X,Y variables from both training and testing data sets
X_train = train_scaled.iloc[:, :-1]
Y_train = train_scaled.iloc[:, [-1]]
X_test = test_scaled.iloc[:, :-1]
Y_test = test_scaled.iloc[:, [-1]]


# In[453]:


# To check correlation amongst variables
corr_values = train_scaled.corr().round(2)
corr_values


# In[454]:


# To visualise the correlation with the help of graphs
import seaborn as sns
sns.set(style='whitegrid')
cols = ['latitude', 'total_rooms', 'population', 'households', 'median_income','total_bedrooms','median_house_value']
plt.figure(figsize=(10,15))
sns.pairplot(train_scaled[cols], height=2)


# In[483]:


#Using heatmap
sns.heatmap(train_scaled[cols].corr().values, cbar=True, annot=True, yticklabels=cols, xticklabels=cols, cmap="YlGnBu")


# In[456]:


LR1.fit(X_train,Y_train)


# In[457]:


Y_train_pred = LR1.predict(X_train)
Y_train_pred


# In[458]:


from sklearn.metrics import mean_squared_error


# In[459]:


mean_squared_error(Y_train, Y_train_pred)            # The error between original and predicted data is 36%


# In[460]:


LR1.intercept_


# In[461]:


LR1.coef_


# In[462]:


ind = np.array(X_train.columns).reshape(-1,1)
pd.DataFrame(LR1.coef_.reshape(-1,1), index=ind, columns=['LR coef'])


# In[463]:


LR1.score(X_train,Y_train)                 # To get the R^2 value
                                           # The model accounts for 63% variation in the data


# In[464]:


# To predict output for test dataset using the fitted model.
Y_test_pred = LR1.predict(X_test)
Y_test_pred


# ### To print root mean squared error (RMSE) from Linear Regression.

# In[465]:


from sklearn.metrics import mean_squared_error


# In[466]:


mean_squared_error(Y_test, Y_test_pred)


# ## Bonus exercise: Perform Linear Regression with one independent variable :
# #### Extract just the median_income column from the independent variables (from X_train and X_test).

# In[468]:


X_train[["median_income"]]


# In[469]:


X_test[["median_income"]]


# ### Perform Linear Regression to predict housing values based on median_income.

# In[470]:


from sklearn.linear_model import LinearRegression
LR2 = LinearRegression()


# In[471]:


med_income_train = X_train[['median_income']]


# In[472]:


LR2.fit(med_income_train, Y_train)


# In[473]:


LR2.intercept_, LR2.coef_               # y = B0 + B1x


# In[482]:


LR2.score(med_income_train,Y_train)                    # value of R^2


# In[475]:


med_income_test = X_test[["median_income"]]


# In[476]:


Y_train_pred2 = LR2.predict(med_income_train)
Y_train_pred2


# In[477]:


Y_test_pred2 = LR2.predict(med_income_test)
Y_test_pred2


# ### Plot the fitted model for training data as well as for test data to check if the fitted model satisfies the test data.

# In[478]:


plt.figure(figsize = (10,7))
plt.plot(med_income_train, Y_train, 'bo', label = 'Original Data')
plt.plot(med_income_train, Y_train_pred2, 'r-', label='Fitted Line' )
plt.xlabel('Median Income', fontsize=20)
plt.ylabel('Median House Value', fontsize=20)
plt.title("Training Data Set",fontsize=25)
plt.legend(loc=2)


# In[479]:


LR2.score(med_income_train,Y_train)           # To get the R^2 value
                                              # The model accounts for 47% variation in the data


# In[485]:


plt.figure(figsize = (10,7))
plt.plot(med_income_test, Y_test, 'bo', label = 'Original Data')
plt.plot(med_income_test, Y_test_pred2, 'r-', label='Fitted Line' )
plt.xlabel('Median Income', fontsize=20)
plt.ylabel('Median House Value', fontsize=20)
plt.title("Testing Data Set",fontsize=25)
plt.legend(loc=2)


# In[481]:


LR2.score(med_income_test,Y_test)               # To get the R^2 value
                                                # The model accounts for 46% variation in the data

