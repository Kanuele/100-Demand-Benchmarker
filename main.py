import numpy as np
import pandas as pd

import pyarrow as pa #pyarrow to use parquet files
import pyarrow.parquet as pq

import statsmodels.api as sm

import Functions.ImportFunctions as import_functions
import Functions.ExportResults as export_functions
from Functions.DemandCleansing import aggregate as dc_aggregate
# import Functions.DemandCleansing as demand_cleansing


import_file = import_functions.import_file

demand  = import_file('ExampleData/Forecasting_beer.parquet')

aggregated_df = dc_aggregate(demand)
demand_filtered = aggregated_df.xs(("Zapopan", "Mexico", "Fancy Beer Ale 1 pint"))#.reset_index()

# ----- Export
demand_filtered.to_csv("ExampleData/Forecasting_beer_filtered.csv")