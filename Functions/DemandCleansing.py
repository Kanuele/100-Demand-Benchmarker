import pandas as pd

# Group times series to month level
def aggregate_to_month(df):
    df['Date'] = pd.to_datetime(df['Date'])
    #df.set_index('Date', inplace=True)
    header = df.columns.values
    header = list(header[header != "demand_quantity"])
    df = df.groupby(header).agg({"demand_quantity": "sum"})#.reset_index()
    return df

demand_grouped = aggregate_to_month(demand)

demand_grouped["demand_quantity"].fillna(0, inplace=True) # Fill up missing values with 0
demand_grouped["demand_quantity"] = demand_grouped["demand_quantity"].clip(lower=0) # set negative values to 0

# check if per time series all month are present

# Remove outliers