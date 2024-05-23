#!/usr/bin/env python
# coding: utf-8

# In[89]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error
import pmdarima as pm


# In[90]:


# load the dataset
transactions = pd.read_csv('transactions.csv')
holidays_events = pd.read_csv('holidays_events.csv')
oil = pd.read_csv('oil.csv')
stores = pd.read_csv('stores.csv')
train = pd.read_csv('train.csv')


# In[91]:


# Convert date columns to datetime
train['date'] = pd.to_datetime(train['date'])
holidays_events['date'] = pd.to_datetime(holidays_events['date'])
oil['date'] = pd.to_datetime(oil['date'])

# Filter data for top stores
top_stores = [44, 45, 47]
train_top_stores = train[train['store_nbr'].isin(top_stores)]

# Aggregate sales by store, date, and family
#aggregated_sales = train_top_stores.groupby(['store_nbr', 'date', 'family']).agg({'sales': 'sum'}).reset_index()

# Merge datasets
sales_holidays = pd.merge(train_top_stores, holidays_events, on='date', how='left')
sales_holidays_oil = pd.merge(sales_holidays, oil, on='date', how='left')
sales_holidays_oil['dcoilwtico'].fillna(method='bfill', inplace=True)

# Apply interpolation safely using transform
sales_holidays_oil['sales'] = sales_holidays_oil.groupby(['store_nbr', 'family'])['sales'].transform(lambda x: x.interpolate(method='linear'))


# Split the data into training and forecasting sets
train_end_date = '2017-08-01'
forecast_start_date = '2017-08-02'
forecast_end_date = '2017-08-15'

train_data = sales_holidays_oil[sales_holidays_oil['date'] <= train_end_date]
forecast_data = sales_holidays_oil[(sales_holidays_oil['date'] >= forecast_start_date) & (sales_holidays_oil['date'] <= forecast_end_date)]


# In[92]:


train_data.head()


# In[93]:


forecast_data.head()


# In[94]:


# Identify top 3 families by total sales
top_families = train_data.groupby('family')['sales'].sum().nlargest(3).index
# Filter data for store 44
store = 44
train_store44 = train_data[train_data['store_nbr'] == store]
forecast_store44 = forecast_data[forecast_data['store_nbr'] == store]
# Plot sales data for top 5 families in store 44
plt.figure(figsize=(14, 10))
for family in top_families:
    family_data = train_store44[train_store44['family'] == family]
    plt.plot(family_data['date'], family_data['sales'], label=family)

plt.title('Sales Data for Top 3 Families in Store 44')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.legend()
plt.grid(True)
plt.show()


# In[95]:


# Select the top 1 family
top_family = top_families[0]

# Filter data for the top 1 family
train_family_data = train_store44[train_store44['family'] == top_family]
forecast_family_data = forecast_store44[forecast_store44['family'] == top_family]


# In[96]:


train_family_data.head()


# In[97]:


forecast_family_data.head()


# In[43]:


# Determine the best ARIMA order for the top family
def find_best_arima_order(data):
    model = pm.auto_arima(data['sales'],
                          seasonal=True,
                          m=12,  # Assuming monthly seasonality
                          stepwise=True,
                          suppress_warnings=True)
    return model.order, model.seasonal_order

order, seasonal_order = find_best_arima_order(train_family_data)
print(f"Store {store}, Family {top_family} - Best ARIMA order: {order}, Seasonal order: {seasonal_order}")

# Train ARIMAX model for the top family
exog_vars = ['onpromotion','dcoilwtico']
model = SARIMAX(train_family_data['sales'], exog=train_family_data[exog_vars], order=order, seasonal_order=seasonal_order)
model_fit = model.fit(disp=False)

# Forecast for the forecasting period
forecast = model_fit.predict(start=len(train_family_data), end=len(train_family_data)+len(forecast_family_data)-1, exog=forecast_family_data[exog_vars])

# Calculate RMSE for the top family
rmse = np.sqrt(mean_squared_error(forecast_family_data['sales'], forecast))
print(f"RMSE for Store {store}, Family {top_family}: {rmse}")


# In[45]:


# Plot Actual vs Forecast for the top family
plt.figure(figsize=(14, 7))
plt.plot(forecast_family_data['date'], forecast_family_data['sales'], label='Actual')
plt.plot(forecast_family_data['date'], forecast, label='Forecast')
plt.title(f'Sales Forecast for Store {store}, Family {top_family}')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.legend()
plt.grid(True)
plt.show()


# In[98]:


# Train ARIMAX model for the top family
exog_vars = ['onpromotion','dcoilwtico']
model = SARIMAX(train_family_data['sales'], exog=train_family_data[exog_vars], order=(2, 1, 2), seasonal_order=(2, 0, 2, 7))
model_fit = model.fit(disp=False)

# Forecast for the forecasting period
forecast = model_fit.predict(start=len(train_family_data), end=len(train_family_data)+len(forecast_family_data)-1, exog=forecast_family_data[exog_vars])

# Calculate RMSE for the top family
rmse = np.sqrt(mean_squared_error(forecast_family_data['sales'], forecast))
print(f"RMSE for Store {store}, Family {top_family}: {rmse}")
# Plot Actual vs Forecast for the top family
plt.figure(figsize=(14, 7))
plt.plot(forecast_family_data['date'], forecast_family_data['sales'], label='Actual')
plt.plot(forecast_family_data['date'], forecast, label='Forecast')
plt.title(f'Sales Forecast for Store {store}, Family {top_family}')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.legend()
plt.grid(True)
plt.show()


# In[99]:


# Extract residuals
residuals = model_fit.resid

# Plot ACF and PACF of residuals
plt.figure(figsize=(12, 6))
plt.subplot(121)
plot_acf(residuals, lags=30, ax=plt.gca())
plt.title('ACF of Residuals')
plt.subplot(122)
plot_pacf(residuals, lags=30, ax=plt.gca())
plt.title('PACF of Residuals')
plt.tight_layout()
plt.show()


# In[101]:


# Perform Ljung-Box test for autocorrelation
lb_test = sm.stats.acorr_ljungbox(residuals, lags=[1], return_df=True)
print('Ljung-Box test:\n', lb_test)


# In[102]:


# Plot histogram and Q-Q plot of residuals
plt.figure(figsize=(12, 6))
plt.subplot(121)
sns.histplot(residuals, kde=True, bins=30)
plt.title('Histogram of Residuals')
plt.subplot(122)
sm.qqplot(residuals, line='s', ax=plt.gca())
plt.title('Q-Q Plot of Residuals')
plt.tight_layout()
plt.show()


# In[66]:


# Train ARIMAX model for the top family
exog_vars = ['onpromotion','dcoilwtico']
model = SARIMAX(train_family_data['sales'], exog=train_family_data[exog_vars], order=(2, 1, 3), seasonal_order=(2, 0, 2, 7))
model_fit = model.fit(disp=False)

# Forecast for the forecasting period
forecast = model_fit.predict(start=len(train_family_data), end=len(train_family_data)+len(forecast_family_data)-1, exog=forecast_family_data[exog_vars])

# Calculate RMSE for the top family
rmse = np.sqrt(mean_squared_error(forecast_family_data['sales'], forecast))
print(f"RMSE for Store {store}, Family {top_family}: {rmse}")
# Plot Actual vs Forecast for the top family
plt.figure(figsize=(14, 7))
plt.plot(forecast_family_data['date'], forecast_family_data['sales'], label='Actual')
plt.plot(forecast_family_data['date'], forecast, label='Forecast')
plt.title(f'Sales Forecast for Store {store}, Family {top_family}')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.legend()
plt.grid(True)
plt.show()


# In[88]:


# Train ARIMAX model for the top family
exog_vars = ['onpromotion','dcoilwtico']
model = SARIMAX(train_family_data['sales'], exog=train_family_data[exog_vars], order=(3, 1, 2), seasonal_order=(2, 0, 2, 7))
model_fit = model.fit(disp=False)

# Forecast for the forecasting period
forecast = model_fit.predict(start=len(train_family_data), end=len(train_family_data)+len(forecast_family_data)-1, exog=forecast_family_data[exog_vars])

# Calculate RMSE for the top family
rmse = np.sqrt(mean_squared_error(forecast_family_data['sales'], forecast))
print(f"RMSE for Store {store}, Family {top_family}: {rmse}")
# Plot Actual vs Forecast for the top family
plt.figure(figsize=(14, 7))
plt.plot(forecast_family_data['date'], forecast_family_data['sales'], label='Actual')
plt.plot(forecast_family_data['date'], forecast, label='Forecast')
plt.title(f'Sales Forecast for Store {store}, Family {top_family}')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.legend()
plt.grid(True)
plt.show()


# In[ ]:





# In[104]:


# Filter data for store 45
store = 45
train_store45 = train_data[train_data['store_nbr'] == store]
forecast_store45 = forecast_data[forecast_data['store_nbr'] == store]
# Plot sales data for top 5 families in store 45
plt.figure(figsize=(14, 10))
for family in top_families:
    family_data = train_store45[train_store45['family'] == family]
    plt.plot(family_data['date'], family_data['sales'], label=family)

plt.title('Sales Data for Top 3 Families in Store 45')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.legend()
plt.grid(True)
plt.show()


# In[79]:


# Select the top 1 family
top_family = top_families[0]

# Filter data for the top 1 family
train_family_data = train_store45[train_store45['family'] == top_family]
forecast_family_data = forecast_store45[forecast_store45['family'] == top_family]


# In[80]:


# Train ARIMAX model for the top family
exog_vars = ['onpromotion','dcoilwtico']
model = SARIMAX(train_family_data['sales'], exog=train_family_data[exog_vars], order=(3, 1, 2), seasonal_order=(2, 0, 2, 7))
model_fit = model.fit(disp=False)

# Forecast for the forecasting period
forecast = model_fit.predict(start=len(train_family_data), end=len(train_family_data)+len(forecast_family_data)-1, exog=forecast_family_data[exog_vars])

# Calculate RMSE for the top family
rmse = np.sqrt(mean_squared_error(forecast_family_data['sales'], forecast))
print(f"RMSE for Store {store}, Family {top_family}: {rmse}")
# Plot Actual vs Forecast for the top family
plt.figure(figsize=(14, 7))
plt.plot(forecast_family_data['date'], forecast_family_data['sales'], label='Actual')
plt.plot(forecast_family_data['date'], forecast, label='Forecast')
plt.title(f'Sales Forecast for Store {store}, Family {top_family}')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.legend()
plt.grid(True)
plt.show()


# In[77]:


# Filter data for store 47
store = 47
train_store47 = train_data[train_data['store_nbr'] == store]
forecast_store47 = forecast_data[forecast_data['store_nbr'] == store]
# Plot sales data for top 5 families in store 47
plt.figure(figsize=(14, 10))
for family in top_families:
    family_data = train_store47[train_store47['family'] == family]
    plt.plot(family_data['date'], family_data['sales'], label=family)

plt.title('Sales Data for Top 3 Families in Store 47')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.legend()
plt.grid(True)
plt.show()


# In[81]:


# Select the top 1 family
top_family = top_families[0]

# Filter data for the top 1 family
train_family_data = train_store47[train_store47['family'] == top_family]
forecast_family_data = forecast_store47[forecast_store47['family'] == top_family]
# Train ARIMAX model for the top family
exog_vars = ['onpromotion','dcoilwtico']
model = SARIMAX(train_family_data['sales'], exog=train_family_data[exog_vars], order=(3, 1, 2), seasonal_order=(2, 0, 2, 7))
model_fit = model.fit(disp=False)

# Forecast for the forecasting period
forecast = model_fit.predict(start=len(train_family_data), end=len(train_family_data)+len(forecast_family_data)-1, exog=forecast_family_data[exog_vars])

# Calculate RMSE for the top family
rmse = np.sqrt(mean_squared_error(forecast_family_data['sales'], forecast))
print(f"RMSE for Store {store}, Family {top_family}: {rmse}")
# Plot Actual vs Forecast for the top family
plt.figure(figsize=(14, 7))
plt.plot(forecast_family_data['date'], forecast_family_data['sales'], label='Actual')
plt.plot(forecast_family_data['date'], forecast, label='Forecast')
plt.title(f'Sales Forecast for Store {store}, Family {top_family}')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.legend()
plt.grid(True)
plt.show()


# In[82]:


# Select the top 1 family
top_family = top_families[0]

# Filter data for the top 1 family
train_family_data = train_store47[train_store47['family'] == top_family]
forecast_family_data = forecast_store47[forecast_store47['family'] == top_family]
# Train ARIMAX model for the top family
exog_vars = ['onpromotion','dcoilwtico']
model = SARIMAX(train_family_data['sales'], exog=train_family_data[exog_vars], order=(3, 1, 3), seasonal_order=(2, 0, 2, 7))
model_fit = model.fit(disp=False)

# Forecast for the forecasting period
forecast = model_fit.predict(start=len(train_family_data), end=len(train_family_data)+len(forecast_family_data)-1, exog=forecast_family_data[exog_vars])

# Calculate RMSE for the top family
rmse = np.sqrt(mean_squared_error(forecast_family_data['sales'], forecast))
print(f"RMSE for Store {store}, Family {top_family}: {rmse}")
# Plot Actual vs Forecast for the top family
plt.figure(figsize=(14, 7))
plt.plot(forecast_family_data['date'], forecast_family_data['sales'], label='Actual')
plt.plot(forecast_family_data['date'], forecast, label='Forecast')
plt.title(f'Sales Forecast for Store {store}, Family {top_family}')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.legend()
plt.grid(True)
plt.show()


# In[83]:


# Select the top 1 family
top_family = top_families[0]

# Filter data for the top 1 family
train_family_data = train_store47[train_store47['family'] == top_family]
forecast_family_data = forecast_store47[forecast_store47['family'] == top_family]
# Train ARIMAX model for the top family
exog_vars = ['onpromotion','dcoilwtico']
model = SARIMAX(train_family_data['sales'], exog=train_family_data[exog_vars], order=(2, 1, 3), seasonal_order=(2, 0, 2, 7))
model_fit = model.fit(disp=False)

# Forecast for the forecasting period
forecast = model_fit.predict(start=len(train_family_data), end=len(train_family_data)+len(forecast_family_data)-1, exog=forecast_family_data[exog_vars])

# Calculate RMSE for the top family
rmse = np.sqrt(mean_squared_error(forecast_family_data['sales'], forecast))
print(f"RMSE for Store {store}, Family {top_family}: {rmse}")
# Plot Actual vs Forecast for the top family
plt.figure(figsize=(14, 7))
plt.plot(forecast_family_data['date'], forecast_family_data['sales'], label='Actual')
plt.plot(forecast_family_data['date'], forecast, label='Forecast')
plt.title(f'Sales Forecast for Store {store}, Family {top_family}')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.legend()
plt.grid(True)
plt.show()


# In[ ]:




