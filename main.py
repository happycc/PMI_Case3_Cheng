# -*- coding: utf-8 -*-
'''
This is the main script of data analysis for the Philip Morris Case study 3
Author: Cheng CHEN (cheng.chen@nestle.com)
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import scipy.stats as stats
import calendar

from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error


#%%##########################################################
############# Step 1. LOAD AND PREPARE DATA #################
#############################################################

# Read data
df_surrounding = pd.read_json('./UseCase_3_Datasets/Surroundings.json')
df_sale = pd.read_csv('./UseCase_3_Datasets/sales_granular.csv')


'''
There are duplicated store_code in both surrounding and sale data (turned out to be the issue for only one code 11028)
Closer inspection on code 11028 shows that the 4x duplications in df_sale is the same, so we only keep one (the last occurence)
the 2x duplicated 11028 in df_surrounding is not exactly the same but similar, so we also only keep one (the last occurence)
As a general rule, we only keep the last occurence of each store_code in case of duplication
'''
for df in [df_surrounding, df_sale]:
    code_count = df['store_code'].value_counts(); 
    duplicate_codes = code_count.index[code_count>1]
    for code in duplicate_codes:        
        df.drop(np.where(df['store_code']==code)[0][0:-1],inplace=True)

'''
Flatten the surrounding data. The original structure is in a nested json structure.
A new data frame is constructed using relevant information of amenities, where each row 
corresponds to one amenity of one store_code
'''
print('*** Flattening surrounding data, may take a few minutes... ', end='')
df_surrounding = flatten_surrounding(df_surrounding)
print('done')


# Reducing/aggregating surrounding data.
'''
At this point, df_surrouding is a long data frame, each store code spans multiple rows, 
where each row corresponds to one surrounding amenity. Now we need to reduce this data frame
by aggregating the information of amenities per store_code. The reduced data frame will have
one line per store_code. Three types of information will be aggregated per store_code:
a) number of amenities per category; b) administrative (canton/commune); c) longitude/latitude
'''

# a) For each store_code, count the number of amenities in each category
df_amenity_count = df_surrounding.pivot_table(index='store_code', columns='category',\
                                              values='place_id', aggfunc='count', fill_value=0)
# Add a prefix "cat" to the column names so they can be easily retrived later in the bigger data frame
df_amenity_count.columns = 'cat: ' + df_amenity_count.columns

# Add a new column which is the total number of amenities per POS
df_amenity_count['cat: total_count'] = df_amenity_count.sum(axis=1)

# Add new columns for rating statistics
df_amenity_count['cat: mean_rating'] = df_surrounding.groupby('store_code').mean()['rating']
df_amenity_count['cat: mean_rating_total'] = df_surrounding.groupby('store_code').mean()['rating_total']
df_amenity_count['cat: sum_rating'] = df_surrounding.groupby('store_code').sum()['rating']
df_amenity_count['cat: sum_rating_total'] = df_surrounding.groupby('store_code').sum()['rating_total']

# b) For each store_code, get the country/canton/commune/locality using the most common value of its surrouding amenities
getPredominantField = lambda code,field : df_surrounding.loc[df_surrounding['store_code']==code,field].value_counts().index[0]
dict_region = dict()
store_codes = df_surrounding['store_code'].unique()
dict_region['country'] = [getPredominantField(code,'country') for code in store_codes]
dict_region['canton'] = [getPredominantField(code,'canton') for code in store_codes]
dict_region['commune'] = [getPredominantField(code,'commune') for code in store_codes]
dict_region['locality'] = [getPredominantField(code,'locality') for code in store_codes]
df_region = pd.DataFrame(dict_region, index=store_codes); df_region.index.name = 'store_code'

# c) For each store_code, get the longitude/latitude by averaging the long/lat of its surrounding amenities
df_longlat = df_surrounding.groupby('store_code').mean()[['longitude','latitude']]

# And the reduced/agregated surrounding data frame is the combination of the three (a+b+c)
df_surrounding = pd.concat([df_amenity_count, df_region, df_longlat], axis=1)

# Put store_code from column to index in df_sale. From now on, both df_surrounding and df_sale has store_code as index
df_sale.set_index('store_code', inplace=True)

# Only keep store_ids which can be found in both surrouding and sale dataframe
code_common = set(df_surrounding.index).intersection(set(df_sale.index))
indices = [code in code_common for code in df_surrounding.index]
df_surrounding = df_surrounding.loc[indices,:]
indices = [code in code_common for code in df_sale.index]
df_sale = df_sale.loc[indices,:]

# clean up
del [df, code_count, duplicate_codes, code, indices, df_amenity_count, df_region, dict_region, store_codes, df_longlat, code_common]



#%%#########################################################
####### Step 2a. EXPLORATORY ANALYSIS ON SURROUNDINGS ######
############################################################

# Histogram of total number of amenities per POS
fig,ax = plt.subplots(1,1,figsize=(8,6),dpi=150)
ax.hist(df_surrounding['cat: total_count'],bins=25,edgecolor='black')
ax.grid(axis='y');
ax.set(xlabel='total number of surrouding amenities per POS')
ax.set(ylabel='count')
plt.tight_layout()
fig.savefig('./output/count_amenities_per_POS.png',transparent=True)


# Set canton/commune/locality to "others" if less than 5 POS in that area
min_POS = 5
counts = df_surrounding['canton'].value_counts()
df_surrounding['canton'] = [x if counts[x]>=min_POS else 'others ({})'.format(sum(counts<min_POS)) for x in df_surrounding['canton']]
counts = df_surrounding['commune'].value_counts()
df_surrounding['commune'] = [x if counts[x]>=min_POS else 'others ({})'.format(sum(counts<min_POS)) for x in df_surrounding['commune']]
counts = df_surrounding['locality'].value_counts()
df_surrounding['locality'] = [x if counts[x]>=min_POS else 'others ({})'.format(sum(counts<min_POS)) for x in df_surrounding['locality']]

# Histogram of canton/commune/locality per POS
fig,axes = plt.subplots(1,3,figsize=(20,5),dpi=150)

ax = axes[0]; 
counts = df_surrounding['canton'].value_counts()
ax.bar(np.arange(0,len(counts)), counts)
ax.set_xticks(np.arange(len(counts))); ax.set_xticklabels(counts.index, rotation=90)
ax.grid(axis='y')
ax.set_title('Histogram of POS canton (others: <{} POS)'.format(min_POS))

ax = axes[1]; 
counts = df_surrounding['commune'].value_counts();
ax.bar(np.arange(0,len(counts)), counts)
ax.set_xticks(np.arange(len(counts))); ax.set_xticklabels(counts.index, rotation=90)
ax.grid(axis='y')
ax.set_title('Histogram of POS commune (others: <{} POS)'.format(min_POS))

ax = axes[2]; 
counts = df_surrounding['locality'].value_counts();
ax.bar(np.arange(0,len(counts)), counts)
ax.set_xticks(np.arange(len(counts))); ax.set_xticklabels(counts.index, rotation=90)
ax.grid(axis='y')
ax.set_title('Histogram of POS locality (others: <{} POS)'.format(min_POS))
plt.tight_layout()

fig.savefig('./output/hist_region_per_POS.png',transparent=True)

# clean up
del [fig, axes, ax, counts, min_POS]



#%%#########################################################
########### Step 2b. EXPLORATORY ANALYSIS ON SALE ##########
############################################################

# First, aggregate the sum of sales per day, as we probably do not want to go to details by hours
dates = list(map(lambda x : datetime.strptime(x,'%m/%d/%y %H:%M').date(),df_sale.columns))
df_sale = df_sale.groupby(by=dates, axis=1).sum()

# Note that there are zero values in sales, which naturally means zero sales.
# Therefore, for the empty values (which happens a lot in the data), we assume it means that 
# data is not available (for example the POS is not open on that day), not means zero sale
n_zero = np.sum(np.sum(df_sale==0))

# Show the evolution of available POS by date
nnna = np.sum(~np.isnan(df_sale),axis=0)
fig = plt.figure(figsize=(10,5),dpi=150)
plt.plot(nnna); plt.xlabel('date'); plt.ylabel('number of POS with available data')
plt.grid(); plt.tight_layout(); plt.savefig('./output/nb_POS_date.png',transparent=True)

# Only take 2017 data
df_sale = df_sale.loc[:,[x.year==2017 for x in df_sale.columns]]

# Show the evolution of available POS by date
fig,axes = plt.subplots(1,2,figsize=(15,5),dpi=150)
nnna = np.sum(~np.isnan(df_sale),axis=0)
ax = axes[0]; ax.plot(nnna); ax.set(xlabel='date'); ax.set(ylabel='number of POS with available data'); ax.grid(); 

# Histogram of number of days where value is available by POS
ax = axes[1];
nnna = np.sum(~np.isnan(df_sale),axis=1)
ax.hist(nnna,bins=20,edgecolor='black'); ax.set(xlabel='nb of days with sales data in 2017'); ax.set(ylabel='count of POS'); ax.grid(axis='y'); 
plt.tight_layout(); plt.savefig('./output/nb_POS_date_2017.png',transparent=True)

# Remove POS which have less than 5 days data
df_sale = df_sale.loc[nnna>=5,:]

# Average sale per day (only count POS which have data on the day)
daily_mean = df_sale.mean()

# Average daily sales by weekday, all POS confounded
weekday = [x.weekday() for x in daily_mean.index]
mean_bywd = daily_mean.groupby(by=weekday).mean()
fig = plt.figure(figsize=(6,4),dpi=150)
plt.bar(np.arange(len(mean_bywd)), mean_bywd)
plt.xticks(np.arange(7), [calendar.day_abbr[x] for x in mean_bywd.index])
plt.grid(axis='y'); plt.ylabel('average sale per POS')
plt.tight_layout(); plt.savefig('./output/sale_weekday.png',transparent=True)


# Construct the variable indicating the sales performance per POS:
'''
- Taking total sale amount of each POS is not desirable, because each POS might be open 
  on different days and we do not want this to influence
- Taking the daily average sale of each POS on days where data is available is not desirable either,
  because each POS might be open on different days and there is global day variation
  which is not part of the intrinsic sale performance of POS#   
- Instead, we take another approach. For each POS, we sum up total sales on days where this POS has info,
  and then sum up total average sales on THOSE DAYS, and then compute the ratio. This is a global
  performance indicator of each POS compared to the average of all POS. The higher the better.
'''
daily_mean = pd.DataFrame([daily_mean]*len(df_sale))
daily_mean.index = df_sale.index
daily_mean = daily_mean.multiply(~np.isnan(df_sale)*1.0)
sale_index = df_sale.sum(axis=1).divide(daily_mean.sum(axis=1))

# plot the histogram of sale_index
# before taking log
fig,axes = plt.subplots(1,2,figsize=(15,5),dpi=150)
ax = axes[0]; ax.hist(sale_index, bins=20, edgecolor='black')
ax.set(xlabel='sale_index'); ax.set(ylabel='count'); ax.grid(axis='y'); 

# take the log as the distribution is highly log-normal
sale_index = np.log10(sale_index + 0.001)  

# after taking log
ax = axes[1]; ax.hist(sale_index, bins=20, edgecolor='black')
ax.set(xlabel='log10(sale_index)'); ax.set(ylabel='count'); ax.grid(axis='y')
plt.tight_layout(); plt.savefig('./output/hist_sale_index.png',transparent=True)

df_sale['sale_index'] = sale_index

# clean up
del [dates, n_zero, nnna, fig, axes, ax, daily_mean, weekday, mean_bywd]



#%%#########################################################
############## Step 3. SIMPLE CORRELATIONS #################
############################################################

# Here we look at the simple correlations between sale_index and the predictors

df = pd.merge(df_surrounding,df_sale[['sale_index']],how='inner',left_index=True,right_index=True)

# Calculate the correlation and p-values between sale_index and each category of amenities
cat_cols = list(filter(lambda x : 'cat:' in x, df.columns))
r = np.array([stats.pearsonr(df[cat], df['sale_index']) for cat in cat_cols])
# Sort the categories according to strength of correlations, and plot
idx = abs(r[:,0]).argsort()[-1::-1]
r = r[idx,:]; cat_cols = [cat_cols[id] for id in idx]

# Plot the Pearson's correlation between the count of each category of amenity and sale
fig = plt.figure(figsize=(20,7),dpi=150)
plt.plot(r[:,0],'g.'); plt.grid()
plt.xticks(np.arange(len(cat_cols)), cat_cols, rotation=90)
plt.ylabel('Pearson''s correlation with sale_index')
plt.gca().yaxis.label.set_color('green'); plt.gca().tick_params(axis='y', colors='green')
ax = plt.gca().twinx(); ax.plot(r[:,1], ':w'); ax.set(ylabel='p-value'); 
ax.yaxis.label.set_color('white'); ax.tick_params(axis='y', colors='white')
plt.tight_layout(); plt.savefig('./output/r_category_sale_index.png',transparent=True)

# Scatter plot of some interesting amenity categories
cat_cols = ['cat: pharmacy','cat: subway_station', 'cat: mean_rating', 'cat: police']
fig,axes = plt.subplots(1,len(cat_cols),figsize=(18,5),dpi=150)
for index in range(len(cat_cols)):
    cat = cat_cols[index]
    ax = axes[index]; ax.scatter(df[cat], df['sale_index'], marker='.')
    ax.set_title('{}'.format(cat))
    ax.set(xlabel='Nb of '+cat); ax.set(ylabel='sale_index')
plt.tight_layout(); plt.savefig('./output/scatter_cat_sale.png',transparent=True)   

# Correlation of sale_index with canton/commune/locality
df.boxplot('sale_index',by='canton',figsize=(8,6),rot=90)
plt.savefig('./output/box_canton_sale.png',transparent=True)   
df.boxplot('sale_index',by='commune',figsize=(10,6),rot=90)
plt.savefig('./output/box_commune_sale.png',transparent=True)   
df.boxplot('sale_index',by='locality',figsize=(6,6),rot=90)
plt.savefig('./output/box_locality_sale.png',transparent=True)   

# Correlation of sale_index with longitude/latitude, by geographic plotting (or knn)
cat_cols = ['longitude','latitude']
fig,axes = plt.subplots(1,len(cat_cols),figsize=(12,5),dpi=150)
for index in range(len(cat_cols)):
    cat = cat_cols[index]
    ax = axes[index]; ax.scatter(df[cat], df['sale_index'],  marker='.')
    ax.grid();
    ax.set(xlabel=cat); ax.set(ylabel='sale_index')
plt.tight_layout(); plt.savefig('./output/scatter_longlat_sale.png',transparent=True)   

del[cat_cols, fig, axes, ax, cat, idx, index, r]



#%%#########################################################
############# Step 4. RANDOM FOREST MODELING ###############
############################################################

# shuffle/permute df, so that the order is randomized. We do this here instead of asking random split in k-fold cv,
# to keep random but stable results across different cv runs.
df = df.sample(frac=1)

# Split the data by 5-fold cross-validation
K = 5; kf = KFold(n_splits=K)

# Decompose data: 3 different x (features) types + 1 output variable (y)
cat_cols = list(filter(lambda x : 'cat:' in x, df.columns))
x_amenities = df[cat_cols]
x_admin = pd.get_dummies(df[['canton','commune','locality']])
x_longlat = df[['longitude','latitude']]
y = df['sale_index']

# Try different types of x (features), also the combination of all features
xs = [x_amenities, x_admin, x_longlat, pd.concat([x_amenities,x_admin,x_longlat],axis=1)]
labels = ['amenity_counts', 'administrative_region','long_lat','all features']  # text label to be used in plot
rf = RandomForestRegressor(n_estimators=1000, max_features = 0.1, min_samples_leaf = 5)

# Modeling using different features (amenities, admin. long/lat)
for index in range(len(xs)):
    x = xs[index]; label=labels[index]
    ypred = np.zeros(len(df))  # predicted y value
    ypred_trivial = np.zeros(len(df))  # trivial prediction using the mean    
    splits = list(kf.split(x,y))
    for index in range(len(splits)):
        train_id,test_id = splits[index]
        xtrain = x.iloc[train_id,:]; ytrain = y.iloc[train_id]; xtest = x.iloc[test_id,:];
        rf.fit(xtrain, ytrain)
        ypred[test_id] = rf.predict(xtest)
        ypred_trivial[test_id] = ytrain.mean()        

    rmse = np.sqrt(mean_squared_error(y,ypred))
    r = np.corrcoef(y, ypred)[0,1]
    # scatter plot: ground-truth vs. predicted
    fig = plt.figure(figsize=[6,6],dpi=150)
    plt.scatter(y, ypred, marker='.', alpha=0.6); plt.grid()
    plt.xlim([-0.8,0.8]); plt.ylim([-0.6,0.6]); 
    plt.xlabel('ground-truth sale_index'); plt.ylabel('predicted sale_index'); 
    plt.title('Prediction using {}. rmse={:0.2f},r={:0.2f}'.format(label,rmse,r))
    plt.plot([-0.8,0.8], [-0.8,0.8], ls=":", c=".3")
    plt.tight_layout(); plt.savefig('./output/gt_predicted_{}.png'.format(label),transparent=True)

# scatter plot: ground-truth vs. predicted
rmse = np.sqrt(mean_squared_error(y,ypred_trivial))
r = np.corrcoef(y, ypred_trivial)[0,1]
fig = plt.figure(figsize=[6,6],dpi=150)
plt.scatter(y, ypred_trivial, marker='.', alpha=0.6); plt.grid()
plt.xlim([-0.8,0.8]); plt.ylim([-0.6,0.6]); 
plt.xlabel('ground-truth sale_index'); plt.ylabel('predicted sale_index'); 
plt.title('Prediction using trivial mean. rmse={:0.2f},r={:0.2f}'.format(rmse,r))
plt.plot([-0.8,0.8], [-0.8,0.8], ls=":", c=".3")
plt.tight_layout(); plt.savefig('./output/gt_trivial_predicted_{}.png'.format(label),transparent=True)


# plot feature importance for amenities
rf.fit(x_amenities, y)
importances = rf.feature_importances_
idx = importances.argsort(); idx = idx[-1::-1]
importances = importances[idx]; cats = [cat_cols[x] for x in idx]
fig = plt.figure(figsize=(20,7),dpi=150)
plt.plot(np.arange(len(cats)),importances,'.'); plt.grid()
plt.xticks(np.arange(len(cats)), cats, rotation=90)
plt.ylabel('Feature importance')
plt.tight_layout(); plt.savefig('./output/feature_importance.png',transparent=True)

# Scatter plot of some interesting features
cat_cols = ['cat: mean_rating','cat: mean_rating_total', 'cat: sum_rating_total', 'cat: total_count']
fig,axes = plt.subplots(1,len(cat_cols),figsize=(18,5),dpi=150)
for index in range(len(cat_cols)):
    cat = cats[index]
    ax = axes[index]; ax.scatter(df[cat], df['sale_index'], marker='.')
    ax.set_title('{}'.format(cat))
    ax.set(xlabel='Nb of '+cat); ax.set(ylabel='sale_index')
plt.tight_layout(); plt.savefig('./output/scatter_feature_sale.png',transparent=True)   

# clean up
del [df_surrounding, df_sale, cat_cols, x_amenities, x_admin, x_longlat, y, xs, labels, rf, K, kf, x, ypred, splits, index, train_id, test_id, xtrain, ytrain, xtest, fig, ypred_trivial, label, rmse, r, idx, importances, cat, cats]


