"""
This script contains the grid search procedure for hyperparameter tuning to the lgb model utilised
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import lightgbm as lgb
from sklearn.metrics import mean_squared_log_error as rmsler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, mean_squared_error

Holidays = pd.read_csv('holidays_events.csv', 
                       parse_dates=['date'], 
                       index_col=['date'])
Oil_prices = pd.read_csv('oil.csv', 
                       parse_dates=['date'], 
                       index_col=['date'])
Train = pd.read_csv('train.csv',
                       parse_dates=['date'],  
                       index_col=['id'])

print('data loaded', flush = True)
group_dfs = []
lags = [7, 14, 28]

def day_feature_engineering(df):
    df['day'] = df['date'].dt.day
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year
    df['dayofweek'] = df['date'].dt.dayofweek
    return df

def add_lag_features(df, lags):
    df = df.copy()
    for lag in lags:
        df[f'lag_{lag}'] = df['sales'].shift(lag)

        df[f'lag_{lag}'] = df[f'lag_{lag}'].fillna(df[f'lag_{lag}'].shift(lag))
        df[f'lag_{lag}'] = df[f'lag_{lag}'].fillna(df[f'lag_{lag}'].shift(lag))
    return df

def target_encode_family_to_integer(df):
    unique_families = sorted(df['family'].unique())
    family_to_integer = {family: idx for idx, family in enumerate(unique_families)}

    df['family'] = df['family'].map(family_to_integer)
    return df

print('functions loaded', flush = True)

training_start = "2016-01-01"
quake = "2016-04-16"

All_data = pd.concat([Train], ignore_index=True)

last_day_of_month_tr = All_data['date'] + pd.offsets.MonthEnd(0)
All_data.loc[:, 'payday'] = np.where(All_data['date'].dt.day.isin([1, 15]) | (All_data['date'] == last_day_of_month_tr), 1, 0)

All_data = All_data.merge(Oil_prices, on='date', how='left')
All_data.dcoilwtico = All_data.dcoilwtico.ffill().bfill()

national_hols = Holidays.loc[(Holidays['locale'] == 'National') & (~Holidays['transferred'])]

# Create a column indicating national holidays in Train
All_data['is_holiday'] = All_data.index.isin(national_hols.index).astype(int)

# we make a new df which has all of the mean sales in for each product, and stitch these together 
mean_sales = All_data.groupby(['store_nbr', 'family']).sales.mean().reset_index()
All_data = pd.merge(All_data, mean_sales, on=['store_nbr', 'family'], suffixes=('', '_mean'), how='left')

sorted_df = All_data.sort_values(by = ['store_nbr', 'family'])
for (store_nbr, family), group_df in sorted_df.groupby(['store_nbr', 'family']):
    # Apply lag features to the group
    group_df_with_lags = add_lag_features(group_df, lags)
    
    # Append the modified group DataFrame to the list
    group_dfs.append(group_df_with_lags)

# Concatenate all group DataFrames back into one DataFrame
processed_df = pd.concat(group_dfs)

# Now processed_df contains the original DataFrame with lag features added group by group
pd.DataFrame(processed_df)
df_encoded = target_encode_family_to_integer(processed_df)
df_encoded = day_feature_engineering(df_encoded)

print('df fully defined and appended to', flush = True)
print(df_encoded.columns, flush = True)

train_data_w_quake = df_encoded.loc[(df_encoded['date'] > training_start)]
train_data = train_data_w_quake.loc[~((train_data_w_quake['date'] >= "2016-04-16") & (train_data_w_quake['date'] <= "2016-05-16"))]

features = ['store_nbr', 'family', 'onpromotion', 'payday', 'dcoilwtico', 'day', 'month', 'year', 'dayofweek', 'is_holiday', 
            'sales_mean', 'lag_7', 'lag_14', 'lag_28']
target = 'sales'

X_train = train_data[features]
y_train = train_data[target]

rmse_scorer = make_scorer(mean_squared_error, greater_is_better=False, squared=False)
param_grid = {
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'num_leaves': [20, 31, 50, 100],
    'feature_fraction': [0.7, 0.8, 0.9, 1.0],
    'lambda_l1': [0.2],
    'lambda_l2': [0.2],
    'min_child_samples': [10, 20, 30, 50],
    'bagging_fraction': [0.5, 0.6, 0.7, 0.8, 0.9],
    'bagging_freq': [1, 5, 10]
}
print('grid search to begin', flush = True)
model = lgb.LGBMRegressor(objective='tweedie', metric='rmse', boosting_type='gbdt', verbose=1)
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, scoring=rmse_scorer, cv=5, verbose=2, n_jobs=-1)

# Fit the grid search to the data
grid_search.fit(X_train, y_train)

# Best parameters
best_params = grid_search.best_params_
print("Best parameters found: ", best_params, flush = True)