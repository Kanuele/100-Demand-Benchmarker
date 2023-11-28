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


def simple_ex_smoothing(d, extra_periods=24, alpha=0.3):
    """Generate a forecast using the single exponential smoothing method. The forecast for each period is the average of the demand in n previous periods.

    Args:
        d (_Dataframe_): A time series that contains the historical demand (can be a list or a NumPy array)
        extra_periods (int, optional): The number or periods we want to forecast into the future. Defaults to 24.
        alpha (float, optional): The smoothing factor. Best to be between 0.05 and 0.5. Defaults to 0.3.

    Returns:
        _DataFrame_: Returns a dataframe with the historical demand, forecast and error (demand - forecast).
    """

    # Initialize arrays
    cols = len(d)
    d = np.append(d, [np.nan] * extra_periods)
    f = np.full(cols + extra_periods, np.nan)

    # Initialize first forecast
    f[1] = d[0]

    # Create all the t+1 forecast until end of hirostical period
    for t in range(2, cols + 1):
        f[t] = alpha * d[t - 1] + (1 - alpha) * f[t - 1]

    # Forecast for all extra periods
    for t in range(cols + 1, cols + extra_periods):
        # Update the forecast as previous forecast
        f[t] = f[t - 1]

    df = pd.DataFrame.from_dict({"d": d, "f": f, "e": d - f})

    return df


def double_ex_smoothing(d, extra_periods=24, alpha=0.3, beta=0.4, phi=0.97):
    """Generate a forecast using the single exponential smoothing method. The forecast for each period is the average of the demand in n previous periods.

    Args:
        d (_Dataframe_): A time series that contains the historical demand (can be a list or a NumPy array)
        extra_periods (int, optional): The number or periods we want to forecast into the future. Defaults to 24.
        alpha (float, optional): The smoothing factor. Best to be between 0.05 and 0.5. Defaults to 0.3.
        beta (float, optional): The trend smoothing factor. Best to be between 0.05 and 0.5. Defaults to 0.4.

    Returns:
        _DataFrame_: Returns a dataframe with the historical demand, forecast and error (demand - forecast).
    """

    # Initialize arrays
    cols = len(d)
    d = np.append(d, [np.nan] * extra_periods)
    f, a, b = np.full((3, cols + extra_periods), np.nan)

    # Initialize first forecast
    # Should be improved by several methods to be chosen
    a[0] = d[0]
    b[0] = d[1] - d[0]

    # Create all the t+1 forecast until end of hirostical period
    for t in range(1, cols):
        f[t] = a[t - 1] + phi * b[t - 1]
        a[t] = alpha * d[t] + (1 - alpha) * (a[t - 1] + phi * b[t - 1])
        b[t] = beta * (a[t] - a[t - 1]) + (1 - beta) * phi * b[t - 1]

    # Forecast for all extra periods
    for t in range(cols, cols + extra_periods):
        # Update the forecast as previous forecast
        f[t] = a[t - 1] + phi * b[t - 1]
        a[t] = f[t]
        b[t] = phi * b[t - 1]

    df = pd.DataFrame.from_dict({"d": d, "f": f, "e": d - f, "level": a, "trend": b})

    return df


def initialize_seasonal_factors_mul(s, d, slen, cols):
    """_summary_

    Args:
        s (_type_): _description_
        d (_type_): _description_
        slen (_type_): _description_
        cols (_type_): _description_

    Returns:
        _type_: _description_
    """
    for i in range(slen):
        s[i] = np.mean(d[i:cols:slen])  # Season average
    s /= np.mean(s[:slen])  # Scale all seasonal indices by dividing by their mean
    return s


def triple_ex_smoothing_mul(
    d, extra_periods=1, alpha=0.4, beta=0.4, gamma=0.3, phi=0.9, slen=12
):
    """_summary_

    Args:
        d (_type_): _description_
        extra_periods (int, optional): _description_. Defaults to 24.
        alpha (float, optional): _description_. Defaults to 0.4.
        beta (float, optional): _description_. Defaults to 0.4.
        gamma (float, optional): _description_. Defaults to 0.3.
        phi (float, optional): _description_. Defaults to 0.97.
        slen (int, optional): _description_. Defaults to 12.
    """

    cols = len(d)
    d = np.append(d, [np.nan] * extra_periods)

    # component initialization
    f, a, b, s = np.full((4, cols + extra_periods), np.nan)
    s = initialize_seasonal_factors_mul(s, d, slen, cols)

    a[0] = d[0] / s[0]
    b[0] = d[1] / s[1] - d[0] / s[0]

    # Create the forecast for the first season

    for t in range(1, slen):
        f[t] = (a[t - 1] + phi * b[t - 1]) * s[t - 1]

        a[t] = alpha * d[t] / s[t] + (1 - alpha) * (a[t - 1] + phi * b[t - 1])

        b[t] = beta * (a[t] - a[t - 1]) + (1 - beta) * phi * b[t - 1]
    # create all the t+1 forecast until end of historical period
    for t in range(slen, cols):
        f[t] = (a[t - 1] + phi * b[t - 1]) * s[t - slen]
        a[t] = alpha * d[t] / s[t - slen] + (1 - alpha) * (a[t - 1] + phi * b[t - 1])
        b[t] = beta * (a[t] - a[t - 1]) + (1 - beta) * phi * b[t - 1]
        s[t] = gamma * d[t] / a[t] + (1 - gamma) * s[t - slen]

    # Forecast for all extra periods
    for t in range(cols, cols + extra_periods):
        f[t] = (a[t - 1] + phi * b[t - 1]) * s[t - slen]
        a[t] = f[t] / s[t - slen]
        b[t] = phi * b[t - 1]
        s[t] = s[t - slen]

    df = pd.DataFrame.from_dict(
        {"d": d, "f": f, "e": d - f, "level": a, "trend": b, "season": s}
    )

    return df


def create_forecast(df, model, **parameters):
    extra_periods = parameters["extra_periods"]
    future = create_future_periods(df, periods=extra_periods, freq="M")

    # append future to df
    past_future = pd.concat((df, future)).reset_index(drop=True)

    df = df.set_index(["ds"])
    df_pivotted = pd.pivot_table(
        df, values="d", index="ds", columns="ticker", fill_value=0
    )
    forecast = model(df_pivotted, **parameters)
    past_future["f"] = forecast["f"]
    past_future["e"] = forecast["e"]
    past_future["model"] = model.__name__
    return past_future


d = [14, 10, 6, 2, 18, 8, 4, 1, 16, 9, 5, 3, 18, 11, 4, 2, 17, 9, 5, 1]
df = triple_ex_smoothing_mul(
    d, slen=12, extra_periods=4, alpha=0.3, beta=0.2, gamma=0.2, phi=0.9
)
KPI(df)
