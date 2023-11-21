import pandas as pd

# Group times series to month level
def aggregate_to_month(df):
    df['Date'] = pd.to_datetime(df['Date'])
    #df.set_index('Date', inplace=True)
    header = df.columns.values
    header = list(header[header != "demand_quantity"])
    df = df.groupby(header).agg({"demand_quantity": "sum"})#.reset_index()
    return df

def aggregate_all_attributes(df):
    df['Date'] = pd.to_datetime(df['Date'])
    #df.set_index('Date', inplace=True)
    header = df.columns.values
    header = list(header[header != ["demand_quantity"]])
    header.remove("Date")
    df = df.groupby(header).agg({"demand_quantity": "sum"})#.reset_index()
    return df

demand_grouped = aggregate_to_month(demand)

demand_grouped["demand_quantity"].fillna(0, inplace=True) # Fill up missing values with 0
demand_grouped["demand_quantity"] = demand_grouped["demand_quantity"].clip(lower=0) # set negative values to 0

demand_all_attr = aggregate_all_attributes(demand)

# check if per time series all month are present

# Remove outliers


import pandas as pd
from darts import TimeSeries

# Create a TimeSeries, specifying the time and value columns
series = TimeSeries.from_dataframe(demand, "Date", "demand_quantity")

# Set aside the last 36 months as a validation series
train, val = series[:-36], series[-36:]