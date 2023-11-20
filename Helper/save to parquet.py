import numpy as np
import pandas as pd

import pyarrow as pa #pyarrow to use parquet files
import pyarrow.parquet as pq

import Functions.ImportFunctions as import_functions

import_file = import_functions.import_file

demand = import_file("ExampleData/Forecasting_beer.xlsx")

demand_schema = pa.schema([
    ('Date', pa.date32()),
    ('Store_city', pa.string()),
    ('Store_Country', pa.string()),
    ('Product Name', pa.string()),
    ('Product Group', pa.string()),
    ('Product Family', pa.string()),
    ('demand_quantity', pa.int32())
])

table = pa.Table.from_pandas(demand, schema=demand_schema)
pq.write_table(table, 'ExampleData/Forecasting_beer.parquet')