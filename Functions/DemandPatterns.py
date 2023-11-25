import pandas as pd


def identify_pattern(data):
    # Calculate the first difference of the data
    diff = data.diff().dropna()

    # Calculate the autocorrelation of the first difference
    autocorr = diff.autocorr()

    # Check if the data follows a sporadic pattern
    if autocorr < 0.2:
        return "sporadic"

    # Check if the data follows a phase-in pattern
    if autocorr > 0.2 and autocorr < 0.5:
        return "phase-in"

    # Check if the data follows a phase-out pattern
    if autocorr > 0.5 and autocorr < 0.8:
        return "phase-out"

    # Check if the data follows a trend pattern
    if autocorr > 0.8 and autocorr < 0.95:
        return "trend"

    # Check if the data follows a seasonal pattern
    if autocorr > 0.95 and autocorr < 1:
        return "seasonal"

    # Check if the data follows a seasonal-trend pattern
    if autocorr == 1:
        return "seasonal-trend"

    # If none of the above patterns are identified, return "unknown"
    return "unknown"


# Example usage
data = pd.Series([1, 2, 3, 4, 5, 4, 3, 2, 1])
pattern = identify_pattern(data)
print(pattern)
