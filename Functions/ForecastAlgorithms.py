import numpy as np
import pandas as pd
#from statsmodels.tsa.holtwinters import SimpleExpSmoothing, ExponentialSmoothing
#from sklearn.linear_model import LinearRegression

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
        model = ExponentialSmoothing(series, trend="add", seasonal="add", seasonal_periods=m)
        model_fit = model.fit(smoothing_level=alpha, smoothing_slope=beta, smoothing_seasonal=gamma)
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
def create_future_periods(df, periods=24, freq='M'):
  future = pd.DataFrame()
  future["ds"] = pd.date_range(start=df["ds"].max()+ pd.DateOffset(months=1), periods=periods, freq=freq)
  future["ticker"] = df["ticker"]
  return future
    
def moving_average(d, extra_periods=1, n=3):

    # Historical period length
    cols = len(d)
    # Append np.nan into the demand array to cover future periods
    d = np.append(d, [np.nan] * extra_periods)
    # Define the forecast array
    f = np.full(cols + extra_periods, np.nan)

    # Create all the t+1 forecast until end of hirostical period
    for t in range(n, cols):
        f[t] = np.mean(d[t - n:t])

    # Forecast for all extra periods
    f[t + 1:] = np.mean(d[t-n+1:t+1])

    # Return DataFrame with the demand, forecast and error
    df = pd.DataFrame.from_dict({"Demand": d, "Forecast": f, "Error": d - f})

    return df

def create_forecast(df, model, extra_periods, n): 
    future = create_future_periods(df, periods=extra_periods, freq='M')

    # append future to df
    past_future = pd.concat((df, future)).reset_index(drop=True)

    df = df.set_index(["ds"])
    df_pivotted = pd.pivot_table(df, values="y", index="ds", columns="ticker", fill_value=0)

    forecast = model(df_pivotted, extra_periods, n)
    past_future["fc"] = forecast["Forecast"]
    past_future["e"] = forecast["Error"]
    return past_future