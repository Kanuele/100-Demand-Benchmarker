import numpy as np
import pandas as pd

import pyarrow as pa  # pyarrow to use parquet files
import pyarrow.parquet as pq

# from prophet import Prophet
from time import time

# import matplotlib.pyplot as plt

# import statsmodels.api as sm

import Functions.ImportFunctions as import_functions
import Functions.ExportResults as export_functions
import Functions.DemandCleansing as dc
import Functions.ForecastAlgorithms as FFA
import Functions.ForecastError as FE

# ------------ Import------------ ------------ ------------ ------------
start_time_script = time()
import_file = import_functions.import_file
demand_imported = import_file("ExampleData/Forecasting_beer.parquet")

# ------------ Take Colnames

col_names = set(list(demand_imported.columns))
values = set(["Date", "demand_quantity"])
col_names = list(col_names - values)

demand = demand_imported.copy()

# ------------ Cleanse ------------ ------------ ------------ ------------

## Aggregate to month level
demand = dc.aggregate(demand)

demand = demand.reset_index()

## Fill missing periods

demand = dc.combine_attributes(demand)
demand.set_index(["Date"], inplace=True)
demand = dc.iterate_combinations(demand)
# demand = dc.split_column(demand, 'combined', ' // ', col_names)

# ------------ Forecast ------------ ------------ ------------ ------------

# Prepare for forecasting
demand.columns = ["ds", "d", "ticker"]
demand_by_ticker = demand.groupby("ticker")
ticker_list = list(demand_by_ticker.groups.keys())


# df_test = demand[demand["ticker"] == ticker_list[0]]
# df_forecast = FFA.create_forecast(df_test, model=FFA.moving_average, extra_periods=24, n=5)
# test

# df = demand.loc[demand["ticker"] == ticker_list[0]]
# df_forecast = create_forecast(df, FFA.simple_ex_smoothing, extra_periods=24, alpha=3)
# df_errors = FE.KPI(df_forecast)
# d = [37, 60, 85, 112, 132, 145, 179, 198, 150, 132]
# df = FFA.moving_average(d, extra_periods=24, n=3)
# df["ticker"] = "A"
# df["model"] = "moving_average"
# df_errors = FE.KPI(df)

# ------------- Forecasting ------------ ------------ ------------ ------------
# Start time
start_time_loop = time()
# Create an empty dataframe
for_loop_forecast = pd.DataFrame()
for_loop_errors = pd.DataFrame()
# Loop through each ticker
for ticker in ticker_list:  # 1.33 seconds
    # Get the data for the ticker
    group = demand_by_ticker.get_group(ticker)

    # Make first forecast
    df_forecast = FFA.create_forecast(group, FFA.moving_average, extra_periods=24, n=3)
    df_errors = FE.KPI(df_forecast)
    for_loop_forecast = pd.concat((for_loop_forecast, df_forecast))
    for_loop_errors = pd.concat((for_loop_errors, df_errors))

    # second forecast
    df_forecast = FFA.create_forecast(
        group, FFA.simple_ex_smoothing, extra_periods=24, alpha=0.3
    )
    df_errors = FE.KPI(df_forecast)
    # Add the forecast results to the dataframe
    for_loop_forecast = pd.concat((for_loop_forecast, df_forecast))
    for_loop_errors = pd.concat((for_loop_errors, df_errors))

    # Third forecast
    df_forecast = FFA.create_forecast(
        group, FFA.double_ex_smoothing, extra_periods=24, alpha=0.3, beta=0.4
    )
    df_errors = FE.KPI(df_forecast)
    # Add the forecast results to the dataframe
    for_loop_forecast = pd.concat((for_loop_forecast, df_forecast))
    for_loop_errors = pd.concat((for_loop_errors, df_errors))

print("The time used for the for-loop forecast is ", time() - start_time_loop)

# Take a look at the data
demand.info()
demand.loc[demand["ticker"] == ticker]
for_loop_forecast.loc[for_loop_forecast["ticker"] == ticker]["d"].info()
for_loop_forecast.info()
for_loop_forecast.head()
for_loop_forecast.tail()

for_loop_errors.loc[for_loop_errors["ticker"] == ticker].iloc[
    for_loop_errors.loc[for_loop_errors["ticker"] == ticker]["MAE_rel"].argmin()
]


# ------------ Try out area

# for_loop_forecast = dc.split_column(for_loop_forecast, "ticker", " // ", col_names)
# add dates to the empty dataframe


# ----- Export

for_loop_forecast.to_csv("ExampleData/Forecasting_beer_forecast.csv")
for_loop_errors.to_csv("ExampleData/Forecasting_beer_errors.csv")
print("The time used for the script is ", time() - start_time_script)
