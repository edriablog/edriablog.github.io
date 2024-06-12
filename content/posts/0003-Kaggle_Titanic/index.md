+++
title = 'Titanic Competition Notebook'
date = 2024-05-14T16:50:31-07:00
draft = false
+++

## Project Intro
### Data Set
- Ground Truth: train.csv (891 entries)
- Test set: test.csv (418 entries)
#### Data Dictionary
| Variable | Definition | Key |
| :------- | :-------- | :----------- |
| survival | survival | 0 = No, 1 = Yes |
| pclass | Ticket Class | 1 = 1st (Upper), 2 = 2nd (Middle), 3 = 3rd (Lower) |
sex | Sex | Male, Female |
| age | age in years | fractional if < 1, if estimated then in form xx.5 |
| sibsp | # of siblings / spouses aboard the Titanic | to define family relations, siblings = brother, sister, stepbrother, stepsister |
| parch | # of parents / children aboard the Titanic | to define family relations, parent = mother, father; child = daughter, son, stepdaughter, stepson |
| ticket | Ticket number | some string |
| fare | Passenger fare | price float
| cabin | Cabin number | some string
| embarked | Port of Embarkation | C = Cherbourg, Q = Queenstown, S = Southampton |

### Submission
- csv file with 418 entries + 1 head row
- PassengerID + Survived (prediction binary: 1=survived, 0=deceased)
### Evaluation
- predict 0 or 1 value for the test set passenger
- percentage of passengers with correct prediction -- accuracy

### Goal
Goal: we want to find patterns in train.csv that help us predict whether the passengers in test.csv survived.

## Data
### Get Data First
#### Know where data located
```python
import numpy as np 
import pandas as pd

import os

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
```
> This code block will print out all filws under the input directory:
/kaggle/input/titanic/train.csv
/kaggle/input/titanic/test.csv
/kaggle/input/titanic/gender_submission.csv

#### Load Training Data
```python
train_data = pd.read_csv("/kaggle/input/titanic/train.csv")

train_data.head()
```
> This will return the first 5 rows of train.csv

#### Load Test Data
```python
test_data = pd.read_csv("/kaggle/input/titanic/test.csv")
test_data.head()
```

Similarly in code, we can now load test data. The test data has deliberately covered the true label for survived data.

What we need is to predict the survival column for test data and achieve ideal accuracy.

### Explore Data
Let's see the percentage of survival rates with different gender
```python
women = train_data.loc[train_data.Sex == 'female']["Survived"]
rate_women = sum(women)/len(women)

print("% of women who survived:", rate_women)
```
We can check shape and type of ```women```:
```python
print(type(women))
print(women.shape)
```
> type: <class 'pandas.core.series.Series'>
> shape: (314,)

To desect the code above:
```python
train_data.Sex == 'female'
```
is a boolean, will return False of True for the Sex column of train_data, whether it shows 'female', and it actually returns a Series of False and True 1D Array.

This code then serves as a "**mask**" and applied towards ```train_data``` dataset by: ```train_data.loc```prompt. 

```python
train_data.loc[train_data.Sex == 'female']
```
This will generate a **sub-table** of the original table, now only have rows with ```Sex column == female```. ($312 \times 12$ 312 rows and 12 columns, still have all columns, but now from 891 data rows, we only have 312 rows now. i.e. 312 women are selected out.)

Then we applied to column ```Survived``` column only, now instead of a table with all 12 columns, now we have 1 column Series, therefore, the shape is (314,) its a 1D-Array Pandas Series.

## Modeling and Output
### Random Forest Model
Model constructed of 100 trees that will individually consider each passenger's data and vote on whether the individual survived. Then use the majority vote rule, the outcome with the most votes wins. That is, out of the 100 trees to predict this single passenger, and 50 of the trees believe survived then we predict survived = 1.
![alt text](https://i.imgur.com/AC9Bq63.png)

This model uses 4 columns of data: ```Pclass```, ```Sex```, ```SibSp```, ```Parch```. Code below:

```python
from sklearn.ensemble import RandomForestClassifier

y = train_data["Survived"]

features = ["Pclass", "Sex", "SibSp", "Parch"]
X = pd.get_dummies(train_data[features])
X_test = pd.get_dummies(test_data[features])
```
> We get ```X``` is the training data with 4 features. ```X_test``` is the test data with 4 features. ```y```is the ground truth label (output) for training data.

Then we create model using ```RandomFOrestClassifier``` with different parameters. Then we train the data with the model (forward propagation part) using ```model.fit(train data input,train data ground truth)```. And after we train the data, we then predict the outcome on test data ```model.predict(test data)```.

```python
model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
model.fit(X, y)
predictions = model.predict(X_test)
```
We finally need to generate a table for submission into .csv file. We will use Pandas commands ```DataFrame``` to generate a table with 2 columns, with ```PassengerId``` same as ```test_data```'s ```PassengerId``` column, and also ```predictions```from the ```model.predict``` which is under the ```Survived``` header.

```python
output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})
output.to_csv('submission.csv', index=False)
print("Your submission was successfully saved!")
```
```output.to_csv``` is the command for storing into csv files.

> where did I saved my file?

## Save and Submit
### Save Version
- click Save version, and change ```version name```
- choose ```Commit``` i.e. Save and Run all
- Then press ```Save```
