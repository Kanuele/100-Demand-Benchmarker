import numpy as np
import pandas as pd

import pyarrow as pa #pyarrow to use parquet files
import pyarrow.parquet as pq

# import statsmodels.api as sm

import Functions.ImportFunctions as import_functions
import Functions.ExportResults as export_functions
import Functions.DemandCleansing as dc
import Functions.ForecastAlgorithms as FFA

# Import

import_file = import_functions.import_file
demand_imported  = import_file('ExampleData/Forecasting_beer.parquet')
demand_imported = import_file('ExampleData/Forecasting_beer_filtered.csv')

# Take Colnames

col_names = set(list(demand_imported.columns))
values = set(["Date", "demand_quantity"])
col_names = list(col_names - values)

demand = demand_imported.copy()

# Cleanse

## Aggregate to month level
demand = dc.aggregate(demand)

demand = demand.reset_index()

## Fill missing periods

demand = dc.combine_attributes(demand)
demand.set_index(["Date"], inplace=True)
demand = dc.iterate_combinations(demand)
demand = dc.split_column(demand, 'combined', ' // ', col_names)


# test_df = demand.loc[(demand["Product Name"] == "Fancy Beer Ale 1 pint") & (demand["Store_city"] == "Zapopan")].copy()
# test_df = dc.combine_attributes(test_df)
# test_df_missing_period = test_df.drop([39, 44, 54]) # april 2016, september 2016, july 2017
# test_df_missing_period = test_df_missing_period.set_index("Date") 

# test_df_missing_period = dc.iterate_combinations(test_df_missing_period)

# test_df_missing_period = test_df_missing_period.asfreq('M', fill_value=0.0)
# test_df_missing_period['combined'] = test_df_missing_period['combined'].mask(test_df_missing_period['combined'] == 0).ffill()



# test_df_missing_period = dc.fill_with_zeros(test_df_missing_period)

# df_demand = test_df.copy()

demand_pivotted = pd.pivot_table(df_demand, values="demand_quantity", index="Date", columns=col_names, fill_value=0)



forecast = FFA.moving_average(test_df["demand_quantity"], extra_periods=12, n=3) # apend the results to the dataframe
# ----- Export
demand_filtered.to_csv("ExampleData/Forecasting_beer_filtered.csv")

demand_sub.to_csv("ExampleData/Forecasting_beer_sub.csv")