import pandas as pd
import numpy as np

import Functions.ForecastAlgorithms as FFA
import Functions.ForecastError as FE


def exp_smooth_opti(d, extra_periods=6, KPI="MAE_abs"):
    """_summary_

    Args:
        d (_type_): _description_
        extra_periods (int, optional): _description_. Defaults to 6.
    """

    params = []  # contains all the different parameter sets
    KPIs = []  # contains the result of each model
    dfs = pd.DataFrame()  # contains all the DataFrames returned by th different models

    for alpha in np.arange(0.05, 0.65, 0.05):
        df = FFA.simple_ex_smoothing(d, extra_periods=extra_periods, alpha=alpha)
        df["ticker"] = "test"
        df["model"] = "simple exp. smoothing"
        df["alpha"] = alpha
        params.append({"simple exp. smoothing": [alpha]})

        KPI_value = FE.KPI(df).iloc[0][KPI]
        KPIs.append(KPI_value)
        df[f"{KPI}"] = KPI_value
        dfs = pd.concat((dfs, df)).reset_index(drop=True)

        for beta in np.arange(0.05, 0.65, 0.05):
            df = FFA.double_ex_smoothing(
                d, extra_periods=extra_periods, alpha=alpha, beta=beta
            )
            df["ticker"] = "test"
            df["model"] = "double exp. smoothing"
            df["alpha"] = alpha
            df["beta"] = beta
            params.append({"double exp. smoothing": [alpha, beta]})

            KPI_value = FE.KPI(df).iloc[0][KPI]
            KPIs.append(KPI_value)
            df[f"{KPI}"] = KPI_value
            dfs = pd.concat((dfs, df)).reset_index(drop=True)
    value = dfs[KPI].nsmallest(1).iloc[0]

    return dfs[dfs[KPI] == value]


# test
d = [28, 19, 18, 13, 19, 16, 19, 18, 13, 16, 16, 11, 18, 15, 13, 15, 13, 11, 13, 10, 12]
df = exp_smooth_opti(d, extra_periods=6, KPI="MAE_abs")
