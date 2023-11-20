import pandas as pd

# Assuming your DataFrame is named 'df' and the column you want to group is named 'date'
def date_to_month(df):
    df['month'] = pd.DatetimeIndex(df['date']).month
    df['year'] = pd.DatetimeIndex(df['date']).year
    df['month_year'] = pd.to_datetime(df['date']).dt.to_period('M')
    return df

# Group times series to month level

# set negative values to 0

# Fill up missing values with 0

# Remove outliers