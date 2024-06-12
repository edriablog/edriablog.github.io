+++
title = 'Pandas Tutorial EP01'
date = 2024-06-12T15:22:31-07:00
draft = false
+++

## Data Filtering
### Choose column values
#### values and logical operation
```py
df = world[(world['area'] >= 3000000) |\
             (world['population'] >= 25000000)]
```
- mask: **`df['col'] > val`**
    - for certain column, compare value
    - will create a mask for the table, generate a col with "true" or "false"
- apply mask: **```df[df['col']>val]```**
    - now apply ```df[mask]``` get a new table with only the rows has true value from the mask
- multiple mask and logic operand: **```(df['col1'] > val1) | (df['col2'] < val2) ```**
    - logical and, or, not, xor, any, all as follows:
    - and: ```&``` --> ```(mask1) & (mask2)```
    - or: ```|``` --> ```(mask1) | (mask2)```
    - not: ```~``` --> ```~mask1```
    - xor: ```df.any()``` --> non-zero, non-empty
    - all: ```df.all()```
#### show column of interest
```py
df_selected = df[['name','area','population']]
```
or **any dataframe followed with double brackets** ```[[...]]```:
```py
df = world[(world['area'] >= 3000000) |\
             (world['population'] >= 25000000)
             ][['name','area','population']]
```
- Differences between df[['col']] and df['col]!!
    - ```df[['col']]``` **double brackets** returns a **dataframe** with the specific col
    - `df['col']` **single bracket** returns a **pandas Series** containing the column, but it is a **Series** data type, not a **dataframe** data type!

#### use `isin()` to find instances of df1 in df2
```py
df = customers[~customers['id'].isin(orders['customerId'])][['name']]
```
- create mask `df1['col1'].isin(df2['col2'])`
    - also can add negate `~` to indicate `not` at the front
- apply mask `df1[mask]`
- get the column of interest using double braskets `df1[mask][['col of interest']]`
- change name of column for output

#### use `df.rename()` to change column name
```py
df = customers[~customers['id'].isin(orders['customerId'])][['name']].rename(columns={'name':'Customers'})
```
-`df.rename(columns={'old_name': 'new_name'})`
- see some other features of [`df.rename()`](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.rename.html)

#### `df.drop_duplicates()` and `df.sort_values('col')`
```py
    df = views[views['author_id']==views['viewer_id']
            ][['author_id']
            ].rename(columns={'author_id': 'id'}
            ).sort_values('id', ascending=True
            ).drop_duplicates()
```
### String Methods
#### check string length of a cell
```py 
df = tweets[tweets['content'].str.len() > 15][['tweet_id']]
```
- mask: `df['col'].str.len() > some_value`
    - built-in `.str.len()` can be applied either to a dataframe: `df.str.len()` 
    - or a Series: `some_series.str.len()`
- apply mask: `df[mask]`
- choose certain column for representation/output: `df[[...]]`



