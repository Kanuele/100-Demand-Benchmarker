import matplotlib.pyplot as plt

from main import *

test_df = forecasts.loc[forecasts["ticker"] == ticker_list[100]]
plot_df = test_df.pivot(index="ds", columns="model", values="f")
col = list(test_df["d"])[0 : len(plot_df)]

plot_df.insert(0, "d", col)


plot_df.plot(
    figsize=(8, 3),
    xticks=plot_df.index,
    ylabel="volume",
    xlabel="date",
    title=f"Volume forecast for {ticker_list[1]}",
    style=["-", "--", "-.", ":"],
)

plt.show()


# Data science for supply chain forecasting test area

# d = [28,19, 18, 13, 19, 16, 19, 18, 13, 16, 16, 11, 18, 15, 13, 15, 13, 11, 13, 10, 12]
# df = FFA.double_ex_smoothing(d, extra_periods=4, alpha=0.4, beta=0.4, phi=0.9)
# FE.KPI(df)
# df[["d", "f"]].plot(figsize=(8, 5), title="Double Exponential Smoothing", style=["-", "--"])
# plt.show()
