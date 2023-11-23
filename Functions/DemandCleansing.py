import pandas as pd
import numpy as np

# Group times series to month level
# def aggregate_to_month(df):
#     df['Date'] = pd.to_datetime(df['Date'])
#     #df.set_index('Date', inplace=True)
#     header = df.columns.values
#     header = list(header[header != "demand_quantity"])
#     df = df.groupby(header).agg({"demand_quantity": "sum"})#.reset_index()
#     return df

# def aggregate_all_attributes(df): #hiervon ausgehend Darstellung der Gruppen in Pareto und selection der verwendeten attribute
#     df['Date'] = pd.to_datetime(df['Date'])
#     #df.set_index('Date', inplace=True)
#     header = df.columns.values
#     header = list(header[header != ["demand_quantity"]])
#     header.remove("Date")
#     df = df.groupby(header).agg({"demand_quantity": "sum"})#.reset_index()
#     return df

def aggregate(df): #erweitern durch Auswahl der Attribute, die aggregiert werden sollen
    df['Date'] = pd.to_datetime(df['Date'])
    header = df.columns.values
    header = list(header[header != "demand_quantity"])
    header.remove("Date")
    df = (df.groupby([pd.Grouper(key="Date", freq="M")] + header) # https://copyprogramming.com/howto/groupby-pandas-throwing-valueerror-grouper-and-axis-must-be-same-length does it help?
        .sum()
        .reset_index("Date")
        .sort_index())
    return df

# Combine all attributes
def combine_attributes(df, date_name = "Date", key_figures = "demand_quantity"):
    col_names = set(list(df.columns))
    values = set([date_name, key_figures])
    col_names = list(col_names - values)

    df["combined"] = pd.Series(df[col_names].values.tolist()).str.join(' // ')
    df = df.drop(col_names, axis=1)
    return df

# fills missing periods with 0 for a series --> need to iterate over all time series
def fill_with_zeros(df):
    df = df.asfreq('M', fill_value=0.0)
    df['combined'] = df.loc[:,'combined'].mask(df['combined'] == 0).ffill()
    df.reset_index(inplace=True)
    return df

# # Fill missing periods
def iterate_combinations(df):
    combinations = df["combined"].unique()
    new_df = pd.DataFrame()

    for combination in combinations:
        subset = df.loc[df["combined"] == combination]
        subset = fill_with_zeros(subset)
        new_df = pd.concat([new_df, subset])

    return new_df


def split_column(df, column_name, delimiter, col_names):
    new_columns = df[column_name].str.split(delimiter, expand=True)
    new_column_names = col_names[:new_columns.shape[1]]
    df[new_column_names] = new_columns
    df.drop(column_name, axis=1, inplace=True)
    return df



# # check that all months from start to previous months are available


# demand_grouped["demand_quantity"].fillna(0, inplace=True) # Fill up missing values with 0
# demand_grouped["demand_quantity"] = demand_grouped["demand_quantity"].clip(lower=0) # set negative values to 0

# demand_all_attr = aggregate_all_attributes(demand)

# # check if per time series all month are present

# # Remove outliers
# '''