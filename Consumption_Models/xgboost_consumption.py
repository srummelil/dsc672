# imports
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import numpy as np
import random
import pandas as pd
import xgboost as xgb

df = pd.read_csv('../processed/consumption_master_with_SF.csv')

# i guess i have to bring in consumption weather and join it to the consumption dataset

c_weather = pd.read_csv('../raw/powercity_weather_consumption.csv')

# create datetime index to join both tables
c_weather['Hour_str'] = [str(x-1) if len(str(x-1))==2 else '0'+str(x-1) for x in c_weather['Hour']]
c_weather['datetime_str'] = ['1900-'+str(x)+'-'+str(y)+' '+str(z)+':00:00' for x, y, z in zip(c_weather.Month, c_weather.Day, c_weather.Hour_str)]
c_weather['datetime'] = pd.to_datetime(c_weather.datetime_str, infer_datetime_format=True)

c_weather = c_weather.set_index(pd.DatetimeIndex(c_weather['datetime']))
c_weather.drop(['datetime', 'datetime_str', 'Hour_str', 'Year', 'City'], axis=1, inplace=True)

for field in ['Month', 'Day', 'Hour']:
    c_weather[field] = c_weather[field].astype(str)

# join to main consumption data
df = df.set_index(pd.DatetimeIndex(df['Time']))
df.drop('Time', inplace=True, axis=1)

model_df = df.merge(c_weather, left_index=True, right_index=True, how='inner')

model_df['Total_Consumption'] = df.FOOD_SVC_TOTAL + df.GROCERY_TOTAL + df.HEALTH_CARE_TOTAL + df.K12_TOTAL + \
                                df.LODGING_TOTAL + df.OFFICE_TOTAL + df.RESIDENTIAL_TOTAL + df.SA_RTL_TOTAL

columns_to_drop = ['FOOD_SERVICE',
                   'GROCERY',
                   'HEALTH_CARE',
                   'K12_SCHOOLS',
                   'LODGING',
                   'OFFICE',
                   'RESIDENTIAL',
                   'STAND_ALONE_RETAIL',
                   'FOOD_SVC_TOTAL',
                   'GROCERY_TOTAL',
                   'HEALTH_CARE_TOTAL',
                   'K12_TOTAL',
                   'LODGING_TOTAL',
                   'OFFICE_TOTAL',
                   'RESIDENTIAL_TOTAL',
                   'SA_RTL_TOTAL',
                   'ELECTRIC_CAR']

model_df.drop(columns_to_drop, inplace=True, axis=1)

# convert school day boolean to string
model_df.School_Day = model_df.School_Day.astype(str)

model_df_dummies = pd.get_dummies(model_df)

X = model_df_dummies.drop('Total_Consumption', axis=1)
y = model_df_dummies['Total_Consumption']


dmatrix = xgb.DMatrix(data=X, label=y)

best_test_rmse = 1000000
best_train_rmse = None
best_params = None
for i in range(30):
    params = None
    params = {"objective": "reg:linear",
              'n_estimators': random.randint(75, 175),
              'colsample_bytree': round(np.random.uniform(0.1, 0.75), 2),
              'learning_rate': round(np.random.uniform(0.01, 0.3), 2),
              'max_depth': random.randint(12, 25),
              'alpha': 5}

    print(params)
    cv_results = xgb.cv(dtrain=dmatrix,
                        params=params,
                        nfold=5,
                        num_boost_round=50,
                        early_stopping_rounds=7,
                        metrics="rmse",
                        as_pandas=True,
                        seed=123,
                        verbose_eval=250)
    print(f'Finished round {i}')
    current_test_rmse = cv_results['test-rmse-mean'].iloc[-1]

    if current_test_rmse < best_test_rmse:
        best_test_rmse = current_test_rmse
        best_train_rmse = cv_results['train-rmse-mean'].iloc[-1]
        best_params = params

print(best_test_rmse)
print(best_train_rmse)
print(params)
