import pandas as pd
import numpy as np


def KPI(df, ticker="ticker", model="model", demand="d", error="e"):
    """Calculates the normalized bias of a forecast.
    Average of errors in df divided by average of demand in df for the same periods as the errors.

    Args:
        df (_type_): Dataframe with times_series name ticker, forecast model, demand, forecast and error.
        ticker (str, optional): column name of ticker in df. Defaults to "ticker".
        model (str, optional): column name of forecast model in df. Defaults to "model".
        demand (str, optional): column name of demand in df. Defaults to "y".
        error (str, optional): column name of error in df . Defaults to "e".

    Returns:
        _type_: the value of the normalized bias.
    """
    kpis = pd.DataFrame()
    kpis["ticker"] = df[ticker].unique()
    kpis["model"] = df[model].unique()

    BIAS_abs = df[error].mean()

    dem_ave = df.loc[df[error].notnull(), demand].mean()
    if dem_ave == 0:
        BIAS_rel = 0
    else:
        BIAS_rel = df[error].mean() / dem_ave

    MAPE = (df[error].abs() / df[demand]).mean()

    MAE_abs = df[error].abs().mean()
    MAE_rel = MAE_abs / dem_ave

    RMSE_abs = np.sqrt((df[error] ** 2).mean())
    RMSE_rel = RMSE_abs / dem_ave

    MSE = (df[error] ** 2).mean()

    kpis["BIAS_abs"] = BIAS_abs
    kpis["BIAS_rel"] = BIAS_rel
    kpis["MAPE"] = MAPE
    kpis["MAE_abs"] = MAE_abs
    kpis["MAE_rel"] = MAE_rel
    kpis["RMSE_abs"] = RMSE_abs
    kpis["RMSE_rel"] = RMSE_rel
    kpis["MSE"] = MSE
    return kpis
