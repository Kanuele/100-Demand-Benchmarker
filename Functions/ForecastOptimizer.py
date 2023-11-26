import pandas as pd
import numpy as np

import Functions.ForecastAlgorithms as FFA
import Functions.ForecastError as FE


def exp_smooth_opti(d, extra_periods=6, measure="MAE_abs"):
    """_summary_

    Args:
        d (_type_): _description_
        extra_periods (int, optional): _description_. Defaults to 6.
    """

    dfs = pd.DataFrame()  # contains all the DataFrames returned by the different models

    for alpha in np.arange(0.05, 0.65, 0.05):
        df = FFA.simple_ex_smoothing(d, extra_periods=extra_periods, alpha=alpha)
        df["model"] = "simple exp. smoothing"
        df["alpha"] = alpha

        KPI_value = FE.KPI(df).get(measure)
        df[f"{measure}"] = KPI_value
        dfs = pd.concat((dfs, df)).reset_index(drop=True)

        for beta in np.arange(0.05, 0.65, 0.05):
            df = FFA.double_ex_smoothing(
                d, extra_periods=extra_periods, alpha=alpha, beta=beta
            )
            df["model"] = "double exp. smoothing"
            df["alpha"] = alpha
            df["beta"] = beta

            KPI_value = FE.KPI(df).get(measure)
            df[f"{measure}"] = KPI_value
            dfs = pd.concat((dfs, df)).reset_index(drop=True)

    value = dfs[measure].nsmallest(1).iloc[0]

    return dfs[dfs[measure] == value].reset_index(drop=True)


def return_optimal_forecast(df, extra_periods, measure):
    future = FFA.create_future_periods(df, periods=extra_periods, freq="M")

    # append future to df
    past_future = pd.concat((df, future)).reset_index(drop=True)

    df = df.set_index(["ds"])
    df_pivotted = pd.pivot_table(
        df, values="d", index="ds", columns="ticker", fill_value=0
    )
    optimized = exp_smooth_opti(
        df_pivotted, extra_periods=extra_periods, measure="MAE_abs"
    )
    # forecast = model(df_pivotted, **parameters)
    optimized["ticker"] = past_future["ticker"]
    optimized["ds"] = past_future["ds"]
    return optimized


# test
# d = [28, 19, 18, 13, 19, 16, 19, 18, 13, 16, 16, 11, 18, 15, 13, 15, 13, 11, 13, 10, 12]
# df = exp_smooth_opti(d, extra_periods=6, measure="MAE_abs")
