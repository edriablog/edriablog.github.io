+++
title = 'Kaggle Flood Prediction Regression Competition'
date = 2024-05-16T16:37:31-07:00
draft = false
+++

## Overview
- Goal: predict the probability of a region flooding based on various factors.
- Outcome: For each ```id``` row in the test set, you must predict the value of ```FloodProbability```.
### Data
- Flood Prediction Factors dataset
- Various Factors:
    - Monsoon Intensity
    - Topography Drainage
    - River Management
    - Deforestation
    - Urbanization
    - Climate Change
    - DamsQuality
    - Siltation
    - Agricultural Practices
    - Encroachments
    - Ineffective Disaster Preparedness
    - Drainage Systems
    - Coastal Vulnerability
    - Landslides
    - Watersheds
    - Deteriorating Infrastructure
    - Population Score
    - Wetland Loss
    - Inadequate Planning
    - Political Factors
- Target: Flood Probability
## Start Project
### Import Libraries
```python
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from pandas.plotting import bootstrap_plot
import plotly.express as px

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter("ignore")

import catboost
import xgboost
from catboost import CatBoostRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score,make_scorer
from xgboost import XGBRegressor
```

### Explore Data Techniques
#### Load Data
```python
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
```
Similarly, we get the path of the files so that we can load the data in files by using the ```pd.read_csv``` command.

```python
train_data = pd.read_csv("/kaggle/input/playground-series-s4e5/train.csv")
test_data = pd.read_csv("/kaggle/input/playground-series-s4e5/test.csv")
```
### Check Data basics
#### check table shape and type:
```python
print(train_data.shape)
print(test_data.shape)
```
where you get the following printing message:
> (1117957, 22)

> (745305, 21)

```python
df.dtypes #will look at the data types for each column
``` 


#### check first/last few rows of table:
```python
train_data.head() #will show the first 5 rows
df.tail() #wills how the last 5 rows
```
#### check header items of table:
```python
train_data.keys() #show the header keys for the table, and will print the following message:
```
output:
```
Index(['id', 'MonsoonIntensity', 'TopographyDrainage', 'RiverManagement',
       'Deforestation', 'Urbanization', 'ClimateChange', 'DamsQuality',
       'Siltation', 'AgriculturalPractices', 'Encroachments',
       'IneffectiveDisasterPreparedness', 'DrainageSystems',
       'CoastalVulnerability', 'Landslides', 'Watersheds',
       'DeterioratingInfrastructure', 'PopulationScore', 'WetlandLoss',
       'InadequatePlanning', 'PoliticalFactors', 'FloodProbability'],
      dtype='object')
```
in the format of: Index([headers in 'strings' separated with ,], datatype='object')

also the following command
```python
df.columns.values   # return array of column names
df.columns.values.tolist   # return column names to list
```

#### check table info:
```python
data.info()
```
it will provide a table print statement:
```
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 1117957 entries, 0 to 1117956
Data columns (total 21 columns):
 #   Column                           Non-Null Count    Dtype  
---  ------                           --------------    -----  
 0   MonsoonIntensity                 1117957 non-null  int64  
 1   TopographyDrainage               1117957 non-null  int64  
 2   RiverManagement                  1117957 non-null  int64  
 3   Deforestation                    1117957 non-null  int64  
 4   Urbanization                     1117957 non-null  int64  
 5   ClimateChange                    1117957 non-null  int64  
 6   DamsQuality                      1117957 non-null  int64  
 7   Siltation                        1117957 non-null  int64  
 8   AgriculturalPractices            1117957 non-null  int64  
 9   Encroachments                    1117957 non-null  int64  
 10  IneffectiveDisasterPreparedness  1117957 non-null  int64  
 11  DrainageSystems                  1117957 non-null  int64  
 12  CoastalVulnerability             1117957 non-null  int64  
 13  Landslides                       1117957 non-null  int64  
 14  Watersheds                       1117957 non-null  int64  
 15  DeterioratingInfrastructure      1117957 non-null  int64  
 16  PopulationScore                  1117957 non-null  int64  
 17  WetlandLoss                      1117957 non-null  int64  
 18  InadequatePlanning               1117957 non-null  int64  
 19  PoliticalFactors                 1117957 non-null  int64  
 20  FloodProbability                 1117957 non-null  float64
dtypes: float64(1), int64(20)
memory usage: 179.1 MB
```
#### Check mean,std,min percentiles,max in each column
```python
data.describe() #check mean,std,min percentiles,max in each column
```
or you can transpose the output analysis table by ```.T```
```python
data.describe().T
```
### Clean Data
#### Check if data has null cells
```python
data.isnull().any() # will print the following boolean for each header whether its column will have any null cell / missing value
```
output:
```
MonsoonIntensity                   False
TopographyDrainage                 False
RiverManagement                    False
Deforestation                      False
Urbanization                       False
ClimateChange                      False
DamsQuality                        False
Siltation                          False
AgriculturalPractices              False
Encroachments                      False
IneffectiveDisasterPreparedness    False
DrainageSystems                    False
CoastalVulnerability               False
Landslides                         False
Watersheds                         False
DeterioratingInfrastructure        False
PopulationScore                    False
WetlandLoss                        False
InadequatePlanning                 False
PoliticalFactors                   False
FloodProbability                   False
dtype: bool
```
other similar ways of checking
```python
df.isnull() #check missing values
df.notnull() #check non-missing values
df.isnull().values.any() #check if any missing values
df.isnull().sum().sum() #check how many missing values in data
df.isnull().sum() #check how many missing by variable
```
Also some other ways of check columns or rows:
```python
df[df['MonsoonIntensity'].notnull()] #Return the rows where MonsoonIntensity is not null
df[df['DamsQuality'].notnull() & df['Watershed'].notnull()] #returns rows where both columns not null
```
#### Drop irrelevant Columns or Nulls
```python
data = train_data.drop('id',axis=1) #will drop irrelevant header id in the vertical direction (axis=1).
```
```python
dfnew = df.dropna() #drop missing values and assign the data to dfnew
dfnew = dfnew.reset_index(drop=True) #resets index back to 0
```
Threshold Dropping:
```python
dfthreshold = df.dropna(thresh=10) #drop rows that contain less than 10 non-missing values (i.e. if rows full of missing values and after we dropped the missing values, recheck if the row has less than 10 values left, if not, then just drop the row due to too little valuable data)
```
also we don't have to save into new table frames each time, we can just drop the missing value and update the original variable using ```inplace```:
```python
df.dropna(thresh=10, inplace=True) #save changes to original data frame
```

### Data Visualization Techniques
- Related plotting libraries
```py
# mostly used 3 basic libraries
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
#pandas' own plotting library
from pandas.plotting import bootstrap_plot
```
#### Seaborn KDE plot density distribution plot
[seaborn.kdeplot](https://seaborn.pydata.org/generated/seaborn.kdeplot.html)
> seaborn is a statistical data visualization package based on matplotlib. KDE plot is a kernel density estimate plot for visualizing the distribution of observations in a dataset, analogous to a histogram. KDE represents the data using a continuous probability density curve in one or more dimensions. 

> seaborn.kdeplot Plot univariate or bivariate distributions using kernel density estimation.

```py
# single univariate plot for continuous probability density curve
sns.kdeplot(data['FloodProbability'], fill=True,gridsize=100)
plt.title('FloodProbability')
plt.grid()
plt.show()
```
#### Seaborn violinplot / boxplot /swarmplot etc.for each feature
[seaborn.violinplot](https://seaborn.pydata.org/generated/seaborn.violinplot.html#seaborn-violinplot)
> Violin plot shows the distribution of data points after grouping by one (or more) variables. Unlike a box plot, each violin is drawn using a kernel density estimate of the underlying distribution.

[seaborn.boxplot](https://seaborn.pydata.org/generated/seaborn.boxplot.html)
> A box plot (or box-and-whisker plot) shows the distribution of quantitative data in a way that facilitates comparisons between variables or across levels of a categorical variable. The box shows the quartiles of the dataset while the whiskers extend to show the rest of the distribution, except for points that are determined to be “outliers” using a method that is a function of the inter-quartile range.

```py
# multi-plots for both discrete and continuous variable
features = data.keys()   # get feature for violin lot
print(features)     #feature is a Pandas Index class, which is iterable. Other iterable classes include MultiIndex, Series etc.

fig, axes = plt.subplots(nrows=7, ncols=3, figsize=(15, 25))  #align 21 plots 1 for each column, total of 21 axes in 2 directions
axes = axes.flatten() 

for i, feature in enumerate(features):
    sns.violinplot(y=data[feature], ax=axes[i]) #or replace by: sns.boxplot, sns.swarmplot
    axes[i].set_title(feature)
    axes[i].set_xlabel('')
plt.tight_layout()
plt.show()
```
**Note:** when using swarmplot, it will get really slow with very large data samples. In this case we can try ```sns.swarmplot(y=data[features[i]].sample(500))``` for visualization purpose.
#### Seaborn barplot using bivariate variables
```py
sns.barplot(x='Encroachments', y='FloodProbability', data=data)
```
#### seaborn heatmap
```py
plt.figure(figsize=(25, 25))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()
```

#### Use plotly.express for interactive plots
```py
fig = px.scatter_3d(data.sample(200000,random_state=42),x='RiverManagement',y='WetlandLoss',z='AgriculturalPractices',color='FloodProbability')
fig.show()
```
**Note:** original data has > 1 million examples, which are too much to show in the 3D plot, resulting in blank plot. We can offset this by diminishing number of data size using sample, or we can do ```train_test_split``` now since eventually we will need the split sets for validation (and because for this competition we already have separate test set).

```py
df_train, df_val = train_test_split(data,test_size=0.2,random_state=42)
```
#### Side Note on Pandas Indexing
- When try ```df[header]```, Pandas will go column-first
    - cell allocation with ```df[col header][row header]```, still, column-first
- If want to get row info, use ```df.loc[header]```, in this case will go through row first
    - Similarly, we may find ```df.drop(axis=?)``` for ```axis=0``` in row and ```axis=1``` in column, where looks like row first
- therefore, a little trick may be helpful: **```df[]``` will go column-first, and ```df.xxx()``` will start axis=0 as rows.**

### Feature Engineering
Add one column of sum of all features, was thinking about maybe accumulative/aggregated factor
```py
data['fsum'] = data.iloc[:, :-2].sum(axis=1) 
test = test_data.drop('id',axis=1)
test['fsum'] = test.iloc[:, :-1].sum(axis=1) 
```
Now we construct modeling set ```X``` and target set ```y```:
```py
X= data.drop(['FloodProbability'], axis=1)
y= data['FloodProbability']
#split for cross validation
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
```
### Modeling
We use CatBoostRegressor for our training
```py
cat = CatBoostRegressor(random_seed=12,
                        iterations=1500,
                        depth=7,
                        colsample_bylevel=1.0,
                        verbose=True)
cat.fit(X_train,y_train)
```

### Submission
```py
id = test_data['id']
submission = pd.DataFrame({'id':id, 'FloodProbability': prediction})
submission.to_csv('submission.csv', index=False)
```
