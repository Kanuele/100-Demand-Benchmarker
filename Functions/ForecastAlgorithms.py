import numpy as np
from statsmodels.tsa.holtwinters import SimpleExpSmoothing, ExponentialSmoothing
from sklearn.linear_model import LinearRegression

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