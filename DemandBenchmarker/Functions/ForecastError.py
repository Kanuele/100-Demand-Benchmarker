def KPI(d, demand="d", error="e"):
    """Calculates the normalized bias of a forecast.
    Average of errors in df divided by average of demand in df for the same periods as the errors.

    Args:
        df (_type_): Dataframe with times_series name ticker, forecast model, demand, forecast and error.
        demand (str, optional): column name of demand in df. Defaults to "y".
        error (str, optional): column name of error in df . Defaults to "e".

    Returns:
        _type_: the value of the normalized bias.
    """
    kpis = pd.DataFrame()
    # kpis["ticker"] = df[ticker].unique()
    # kpis["model"] = df[model].unique()

    BIAS_abs = d[error].mean()

    dem_ave = d.loc[d[error].notnull(), demand].mean()
    if dem_ave == 0:
        BIAS_rel = 0
    else:
        BIAS_rel = d[error].mean() / dem_ave

    MAPE = (d[error].abs() / d[demand]).mean()

    MAE_abs = d[error].abs().mean()
    MAE_rel = MAE_abs / dem_ave

    RMSE_abs = np.sqrt((d[error] ** 2).mean())
    RMSE_rel = RMSE_abs / dem_ave

    MSE = (d[error] ** 2).mean()

    return {
        "BIAS_abs": BIAS_abs,
        "BIAS_rel": BIAS_rel,
        "MAPE": MAPE,
        "MAE_abs": MAE_abs,
        "MAE_rel": MAE_rel,
        "RMSE_abs": RMSE_abs,
        "RMSE_rel": RMSE_rel,
        "MSE": MSE,
    }
