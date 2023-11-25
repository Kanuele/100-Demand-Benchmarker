import numpy as np
import pandas as pd

# from statsmodels.tsa.holtwinters import SimpleExpSmoothing, ExponentialSmoothing
# from sklearn.linear_model import LinearRegression


class ForecastAlgorithms:
    def single_exponential_smoothing(self, series, alpha):
        model = SimpleExpSmoothing(series)
        model_fit = model.fit(smoothing_level=alpha)
        return model_fit.forecast()

    def double_exponential_smoothing(self, series, alpha, beta):
        model = ExponentialSmoothing(series, trend="add")
        model_fit = model.fit(smoothing_level=alpha, smoothing_slope=beta)
        return model_fit.forecast()

    def triple_exponential_smoothing(self, series, alpha, beta, gamma, m):
        model = ExponentialSmoothing(
            series, trend="add", seasonal="add", seasonal_periods=m
        )
        model_fit = model.fit(
            smoothing_level=alpha, smoothing_slope=beta, smoothing_seasonal=gamma
        )
        return model_fit.forecast()

    def simple_average(self, series):
        return np.mean(series)

    def croston_forecast(self, series, extra_periods=1, alpha=0.4):
        # Croston's method requires intermittent demand series, which is not covered here.
        # Placeholder for completeness.
        pass

    def regression(self, x, y):
        model = LinearRegression()
        model.fit(x, y)
        return model.predict(x)


# Create an empty dataframe
def create_future_periods(df, periods=24, freq="M"):
    """_summary_

    Args:
        df (_type_): _description_
        periods (int, optional): _description_. Defaults to 24.
        freq (str, optional): _description_. Defaults to 'M'.

    Returns:
        _type_: _description_
    """
    future = pd.DataFrame()
    future["ds"] = pd.date_range(
        start=df["ds"].max() + pd.DateOffset(months=1), periods=periods, freq=freq
    )
    future["ticker"] = df["ticker"]
    return future


def naive(d, extra_periods=24, n=1):
    """Naive forecasting algorithm to predict the last period into the future

    Args:
        df (_type_): _description_
        extra_periods (int, optional): Number of forecast periods into the future. Defaults to 24.
        n (int, optional): _description_. Defaults to 1.
    """
    # Historical period length
    cols = len(d)
    # Append np.nan into the demand array to cover future periods
    d = np.append(d, [np.nan] * extra_periods)
    # Define the forecast array
    f = np.full(cols + extra_periods, np.nan)

    # Create all the t+1 forecast until end of hirostical period
    for t in range(n, cols):
        if t == 0:
            continue
        f[t] = d[t - 1]

    # Forecast for all extra periods
    f[t + 1 :] = d[t]

    # Return DataFrame with the demand, forecast and error
    df = pd.DataFrame.from_dict({"d": d, "f": f, "e": d - f})
    return df


def moving_average(d, extra_periods=24, n=3):
    """Generate a forecast using the moving average method. The forecast for each period is the average of the demand in n previous periods.

    Args:
        d (_Dataframe_): A time series that contains the historical demand (can be a list or a NumPy array)
        extra_periods (int, optional): The number or periods we want to forecast into the future. Defaults to 24.
        n (int, optional): The number of periods we will average. Defaults to 3. If n = 1, then the forecast is the same as the naive method.

    Returns:
        _DataFrame_: Returns a dataframe with the historical demand, forecast and error (demand - forecast).
    """
    # Historical period length
    cols = len(d)
    # Append np.nan into the demand array to cover future periods
    d = np.append(d, [np.nan] * extra_periods)
    # Define the forecast array
    f = np.full(cols + extra_periods, np.nan)

    # Create all the t+1 forecast until end of hirostical period
    for t in range(n, cols):
        f[t] = np.mean(d[t - n : t])

    # Forecast for all extra periods
    f[t + 1 :] = np.mean(d[t - n + 1 : t + 1])

    # Return DataFrame with the demand, forecast and error
    df = pd.DataFrame.from_dict({"d": d, "f": f, "e": d - f})

    return df


def create_forecast(df, model, extra_periods, n):
    future = create_future_periods(df, periods=extra_periods, freq="M")

    # append future to df
    past_future = pd.concat((df, future)).reset_index(drop=True)

    df = df.set_index(["ds"])
    df_pivotted = pd.pivot_table(
        df, values="d", index="ds", columns="ticker", fill_value=0
    )

    forecast = model(df_pivotted, extra_periods, n)
    past_future["f"] = forecast["f"]
    past_future["e"] = forecast["e"]
    return past_future
