import pandas as pd

# Group times series to month level
def aggregate_to_month(df):
    df['Date'] = pd.to_datetime(df['Date'])
    #df.set_index('Date', inplace=True)
    header = df.columns.values
    header = list(header[header != "demand_quantity"])
    df = df.groupby(header).agg({"demand_quantity": "sum"})#.reset_index()
    return df

def aggregate_all_attributes(df): #hiervon ausgehend Darstellung der Gruppen in Pareto und selection der verwendeten attribute
    df['Date'] = pd.to_datetime(df['Date'])
    #df.set_index('Date', inplace=True)
    header = df.columns.values
    header = list(header[header != ["demand_quantity"]])
    header.remove("Date")
    df = df.groupby(header).agg({"demand_quantity": "sum"})#.reset_index()
    return df

def aggregate(df): #erweitern durch Auswahl der Attribute, die aggregiert werden sollen
    df['Date'] = pd.to_datetime(df['Date'])
    header = df.columns.values
    header = list(header[header != "demand_quantity"])
    header.remove("Date")
    header.append("Date")
    df = (df.groupby([pd.Grouper(key="Date", freq="M"), 'Store_city', 'Store_Country', 'Product Name', 'Product Group', 'Product Family'])
        .sum()
        .reset_index("Date")
        .sort_index())
    return df

'''

demand_grouped = aggregate(demand)

demand_filtered = demand_grouped.xs(("Zapopan", "Mexico", "Fancy Beer Ale 1 pint"))#.reset_index()#.set_index("Date") # select one time series

# check that all months from start to previous months are available


demand_grouped["demand_quantity"].fillna(0, inplace=True) # Fill up missing values with 0
demand_grouped["demand_quantity"] = demand_grouped["demand_quantity"].clip(lower=0) # set negative values to 0

demand_all_attr = aggregate_all_attributes(demand)

# check if per time series all month are present

# Remove outliers
'''