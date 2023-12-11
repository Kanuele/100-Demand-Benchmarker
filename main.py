import numpy as np
import pandas as pd

import polars as pl

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
# import_file = import_functions.import_file
# print("Using polars")
# start_time = time()
# demand_imported = import_file("ExampleData/Forecasting_beer.parquet")
# demand_imported = pl.read_parquet("ExampleData/Forecasting_beer.parquet")
demand_imported = pl.read_csv(
    "ExampleData/Forecasting_beer_subset.csv", separator=";"
).with_columns(
    pl.col("Date").str.to_date("%Y-%m-%d"), pl.col("demand_quantity").cast(pl.Float64)
)  # Missing periods: February 2013 and January 2014 for Fancy Beer IPA 1 pint | Negative Periods: Fancy Beer IPA 500 ml April 2014 | Empty Period: Fancy Beer IPA 500 ml June 2014

# ------------ Take Colnames
# values = set(["Date", "demand_quantity"])
# col_names = list(set(list(demand_imported.columns)) - values)


demand_imported.filter(pl.col("Product Name") == "Fancy Beer IPA 1 pint").sort("Date")


# demand = demand_imported.copy()
df = demand_imported.clone().rename({"demand_quantity": "d", "Date": "ds"})

# ------------ Cleanse ------------ ------------ ------------ ------------

## Aggregate to month level in polars
## Fill missing periods

df = (
    df.pipe(dc.aggregate_polars)
    .pipe(dc.combine_attributes_polars)
    .set_sorted("ds")
    .upsample(time_column="ds", every="1mo", by="ticker", maintain_order=True)
    .with_columns(pl.col("ticker").forward_fill())
    .with_columns(pl.col("d").fill_null(0))  # Fill up missing values with 0
)

df_pd = (
    df.to_pandas()
)  # hat fortlaufenden index, nicht wie bei Pandas und hat monats anfang als datum
# df_pd.loc[df_pd["ticker"] == ticker_list[1]]
# demand.loc[demand["ticker"] == ticker_list[1]]

# demand = dc.split_column(demand, 'combined', ' // ', col_names)
# Split demand_aggregated to train and test
# end_time = time() - start_time
# print(f"Polars used: {end_time} s ")

# ------------ Forecast ------------ ------------ ------------ ------------
models_map = {
    "double exp. smoothing": FFA.double_ex_smoothing,
    "simple exp. smoothing": FFA.simple_ex_smoothing,
    "moving average": FFA.moving_average,
}

# Prepare for forecasting
# demand.columns = ["ds", "d", "ticker"]
demand_by_ticker = df_pd.groupby("ticker")
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
n_timeseries = len(train_ticker_list)
data_points = len(train)
print("Optimize get optimal parameters for each combination")
start_time = time()
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

end_time = time() - start_time
print(
    f"Optimization time on training set: {end_time} s for {n_timeseries} timeseries and {data_points} data points"
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
n_timeseries_df = len(ticker_list)
data_points_df = len(demand)
print("Run forecast using optimal parameters")
start_time = time()
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
end_time = time() - start_time
print(
    f"Best-fit time on whole dataset: {end_time} s for {n_timeseries_df} timeseries and {data_points_df} data points"
)
# how to validate, that the optimal_ex_post is also best for test?
optim.head()
optim_errors.head()

optim_parameters_dict.iloc[1]
optim_errors.iloc[1]
# ------------ Try out area

# for_loop_forecast = dc.split_column(for_loop_forecast, "ticker", " // ", col_names)
# add dates to the empty dataframe


# ----- Export

# for_loop_forecast.to_csv("ExampleData/Forecasting_beer_forecast.csv")
# for_loop_errors.to_csv("ExampleData/Forecasting_beer_errors.csv")
# print("The time used for the script is ", time() - start_time_script)
