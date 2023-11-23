import numpy as np
import pandas as pd

import pyarrow as pa #pyarrow to use parquet files
import pyarrow.parquet as pq

# import statsmodels.api as sm

import Functions.ImportFunctions as import_functions
import Functions.ExportResults as export_functions
import Functions.DemandCleansing as dc
# from Functions.DemandCleansing import aggregate as dc_aggregate
import Functions.ForecastAlgorithms as FFA
# import Functions.DemandCleansing as demand_cleansing

# Import

import_file = import_functions.import_file
demand  = import_file('ExampleData/Forecasting_beer.parquet')

# Take Colnames

col_names = set(list(demand.columns))
values = set(["Date", "demand_quantity"])
col_names = list(col_names - values)


# Cleanse

## Aggregate to month level
aggregated_df = dc.aggregate(demand)

demand = aggregated_df.reset_index()

## Fill missing periods

demand = dc.combine_attributes(demand)
demand.set_index(["Date"], inplace=True)
demand = dc.iterate_combinations(demand)
demand = dc.split_column(demand, 'combined', ' // ', col_names)




forecast = FFA.moving_average(demand_filtered["demand_quantity"], extra_periods=12, n=3) # apend the results to the dataframe
# ----- Export
demand_filtered.to_csv("ExampleData/Forecasting_beer_filtered.csv")

demand_sub.to_csv("ExampleData/Forecasting_beer_sub.csv")