import numpy as np
import pandas as pd

import pyarrow as pa  # pyarrow to use parquet files
import pyarrow.parquet as pq

from prophet import Prophet
from time import time

# import statsmodels.api as sm

import Functions.ImportFunctions as import_functions
import Functions.ExportResults as export_functions
import Functions.DemandCleansing as dc
import Functions.ForecastAlgorithms as FFA

# ------------ Import------------ ------------ ------------ ------------

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

# Prepare for forecasting
demand.columns = ["ds", "y", "ticker"]
demand_by_ticker = demand.groupby("ticker")
ticker_list = list(demand_by_ticker.groups.keys())

# df_test = demand[demand["ticker"] == ticker_list[0]]
# test


# ------------- Forecasting ------------ ------------ ------------ ------------
# Start time
start_time = time()
# Create an empty dataframe
for_loop_forecast = pd.DataFrame()
# Loop through each ticker
for ticker in ticker_list:  # 1.33 seconds
    # Get the data for the ticker
    group = demand_by_ticker.get_group(ticker)
    # Make forecast
    forecast = FFA.create_forecast(
        group, model=FFA.moving_average, extra_periods=24, n=3
    )
    # Add the forecast results to the dataframe
    for_loop_forecast = pd.concat((for_loop_forecast, forecast))

print("The time used for the for-loop forecast is ", time() - start_time)

# Take a look at the data
for_loop_forecast.head()
for_loop_forecast.tail()

# ------------ Try out area

for_loop_forecast = dc.split_column(for_loop_forecast, "ticker", " // ", col_names)
# add dates to the empty dataframe


# ----- Export
demand_filtered.to_csv("ExampleData/Forecasting_beer_filtered.csv")

for_loop_forecast.to_csv("ExampleData/Forecasting_beer_forecast.csv")
