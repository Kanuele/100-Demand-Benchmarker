import numpy as np
import pandas as pd

# import polars as pl

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

demand = demand.copy().reset_index()

## Fill missing periods

demand = dc.combine_attributes(demand)
demand.set_index(["Date"], inplace=True)
demand = dc.iterate_combinations(demand)
# demand = dc.split_column(demand, 'combined', ' // ', col_names)
# Split demand_aggregated to train and test


# ------------ Forecast ------------ ------------ ------------ ------------
models_map = {
    "double exp. smoothing": FFA.double_ex_smoothing,
    "simple exp. smoothing": FFA.simple_ex_smoothing,
    "moving average": FFA.moving_average,
}

# Prepare for forecasting
demand.columns = ["ds", "d", "ticker"]
demand_by_ticker = demand.groupby("ticker")
ticker_list = list(demand_by_ticker.groups.keys())


# Split groups into train and test dataset

test_size = 12
test = pd.concat(
    [demand_by_ticker.get_group(ticker).iloc[-test_size:] for ticker in ticker_list],
    axis=0,
    ignore_index=True,
)
train = pd.concat(
    [demand_by_ticker.get_group(ticker).iloc[:-test_size] for ticker in ticker_list],
    axis=0,
    ignore_index=True,
)
train_by_ticker, test_by_ticker = train.groupby("ticker"), test.groupby("ticker")
train_ticker_list, test_ticker_list = list(train_by_ticker.groups.keys()), list(
    test_by_ticker.groups.keys()
)

# Optimize get optimal parameters for each ticker
optim_ex_post = pd.concat(
    [
        FO.return_optimal_forecast(
            train_by_ticker.get_group(ticker), extra_periods=24, measure="MAE_abs"
        )
        for ticker in train_ticker_list
    ],
    axis=0,
    ignore_index=True,
)


opti_by_ticker = optim_ex_post.groupby(["ticker"])
opti_ticker_list = list(opti_by_ticker.groups.keys())

header = [
    i
    for i in optim_ex_post.columns
    if i not in ["d", "e", "f", "ds", "level", "trend", "season", "ds"]
]

optim_parameters_dict = pd.concat(
    [
        opti_by_ticker.get_group(ticker).reset_index().loc[0, header]
        for ticker in opti_ticker_list
    ],
    axis=1,
    ignore_index=True,
).T.set_index("ticker")

# ---- Forecast Test using parameters from optimization ----

optim_tests = [
    FFA.create_forecast(
        test_by_ticker.get_group(ticker),
        models_map[optim_parameters_dict.loc[ticker, "model"]],
        extra_periods=24,
        alpha=optim_parameters_dict.loc[ticker, "alpha"],
        # beta=new_optim.loc[ticker, "beta"],
        # phi=0.97,
    )
    for ticker in test_ticker_list
]
# how to validate, that the optimal_ex_post is also best for test?

optim_tests_errors = pd.DataFrame(
    [
        {
            "ticker": df.loc[0, "ticker"],
            "model": df.loc[0, "model"],
            **FE.KPI(df),
        }
        for df in optim_tests
    ]
)


optim_tests = pd.concat(optim_tests).reset_index(drop=True)


# Run forecast using optimal parameters

optim = [
    FFA.create_forecast(
        demand_by_ticker.get_group(ticker),
        models_map[optim_parameters_dict.loc[ticker, "model"]],
        extra_periods=24,
        alpha=optim_parameters_dict.loc[ticker, "alpha"],
        # beta=new_optim.loc[ticker, "beta"],
        # phi=0.97,
    )
    for ticker in ticker_list
]
optim_errors = pd.DataFrame(
    [
        {
            "ticker": df.loc[0, "ticker"],
            "model": df.loc[0, "model"],
            **FE.KPI(df),
        }
        for df in optim
    ]
)
optim = pd.concat(optim).reset_index(drop=True)

# how to validate, that the optimal_ex_post is also best for test?


# ------------ Try out area

# for_loop_forecast = dc.split_column(for_loop_forecast, "ticker", " // ", col_names)
# add dates to the empty dataframe


# ----- Export

# for_loop_forecast.to_csv("ExampleData/Forecasting_beer_forecast.csv")
# for_loop_errors.to_csv("ExampleData/Forecasting_beer_errors.csv")
# print("The time used for the script is ", time() - start_time_script)
