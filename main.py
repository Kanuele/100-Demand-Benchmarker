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
import Functions.ForecastOptimizer as FO

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

# ------------ Forecast ------------ ------------ ------------ ------------

# Prepare for forecasting
demand.columns = ["ds", "d", "ticker"]
demand_by_ticker = demand.groupby("ticker")
ticker_list = list(demand_by_ticker.groups.keys())


# df_test = demand[demand["ticker"] == ticker_list[0]]
# df_forecast = FFA.create_forecast(df_test, model=FFA.moving_average, extra_periods=24, n=5)
# test
# test_df = demand.loc[demand["ticker"] == ticker_list[0]]
# test_df = pd.pivot_table(
#     test_df,
#     values="d",
#     index="ds",
#     columns="ticker",
#     fill_value=0,
# )

# test_opti = FO.return_optimal_forecast(test_df, extra_periods=24, measure="MAE_abs")

# ------------- Forecasting ------------ ------------ ------------ ------------
# Start time
start_time_loop = time()

# Simple Moving Average

df_moving_averages = [
    FFA.create_forecast(
        demand_by_ticker.get_group(ticker), FFA.moving_average, extra_periods=24, n=3
    )
    for ticker in ticker_list
]
df_errors_moving_averages = pd.DataFrame(
    [
        {"ticker": df.loc[:, "ticker"][1], "model": "moving average", **FE.KPI(df)}
        for df in df_moving_averages
    ]
)

df_simple_ex_smoothing = [
    FFA.create_forecast(
        demand_by_ticker.get_group(ticker),
        FFA.simple_ex_smoothing,
        extra_periods=24,
        alpha=0.3,
    )
    for ticker in ticker_list
]

df_errors_simple_ex_smoothing = pd.DataFrame(
    [
        {"ticker": df.loc[:, "ticker"][1], "model": "simple_ex_smoothing", **FE.KPI(df)}
        for df in df_simple_ex_smoothing
    ]
)

df_double_ex_smoothing = [
    FFA.create_forecast(
        demand_by_ticker.get_group(ticker),
        FFA.double_ex_smoothing,
        extra_periods=24,
        alpha=0.3,
        beta=0.4,
        phi=0.97,
    )
    for ticker in ticker_list
]

df_errors_double_ex_smoothing = pd.DataFrame(
    [
        {"ticker": df.loc[:, "ticker"][1], "model": "double_ex_smoothing", **FE.KPI(df)}
        for df in df_double_ex_smoothing
    ]
)


forecasts = pd.concat(
    df_moving_averages + df_simple_ex_smoothing + df_double_ex_smoothing
).reset_index(drop=True)

errors = pd.concat(
    [
        df_errors_moving_averages,
        df_errors_simple_ex_smoothing,
        df_errors_double_ex_smoothing,
    ]
).reset_index(drop=True)

print(
    "The time used for the for-loop forecast is ", time() - start_time_loop
)  # 3,88 seconds using list comprehension # 4,39 seconds using list comprehension for fc and errors

# start_time_loop = time()
# optimals = [
#     FO.return_optimal_forecast(
#         demand_by_ticker.get_group(ticker), extra_periods=24, measure="MAE_abs"
#     )
#     for ticker in ticker_list
# ]
# optimal_forecast = pd.concat(optimals)

# optimal_forecast["model"].unique()
# print("The time used for the list comprehension optimals is ", time() - start_time_loop)


# ------------ Try out area

# for_loop_forecast = dc.split_column(for_loop_forecast, "ticker", " // ", col_names)
# add dates to the empty dataframe


# ----- Export

# for_loop_forecast.to_csv("ExampleData/Forecasting_beer_forecast.csv")
# for_loop_errors.to_csv("ExampleData/Forecasting_beer_errors.csv")
# print("The time used for the script is ", time() - start_time_script)
