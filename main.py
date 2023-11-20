import numpy as np
import pandas as pd

import pyarrow as pa #pyarrow to use parquet files
import pyarrow.parquet as pq

import Functions.ImportFunctions as import_functions

import_file = import_functions.import_file

demand  = import_file('ExampleData/Forecasting_beer.parquet')