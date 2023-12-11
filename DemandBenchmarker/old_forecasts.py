# start_time_loop = time()

# # Simple Moving Average

# df_moving_averages = [
#     FFA.create_forecast(
#         demand_by_ticker.get_group(ticker), FFA.moving_average, extra_periods=24, n=3
#     )
#     for ticker in ticker_list
# ]
# df_errors_moving_averages = pd.DataFrame(
#     [
#         {"ticker": df.loc[:, "ticker"][1], "model": "moving average", **FE.KPI(df)}
#         for df in df_moving_averages
#     ]
# )

# df_simple_ex_smoothing = [
#     FFA.create_forecast(
#         demand_by_ticker.get_group(ticker),
#         FFA.simple_ex_smoothing,
#         extra_periods=24,
#         alpha=0.3,
#     )
#     for ticker in ticker_list
# ]

# df_errors_simple_ex_smoothing = pd.DataFrame(
#     [
#         {"ticker": df.loc[:, "ticker"][1], "model": "simple_ex_smoothing", **FE.KPI(df)}
#         for df in df_simple_ex_smoothing
#     ]
# )

# df_double_ex_smoothing = [
#     FFA.create_forecast(
#         demand_by_ticker.get_group(ticker),
#         FFA.double_ex_smoothing,
#         extra_periods=24,
#         alpha=0.3,
#         beta=0.4,
#         phi=0.97,
#     )
#     for ticker in ticker_list
# ]

# df_errors_double_ex_smoothing = pd.DataFrame(
#     [
#         {"ticker": df.loc[:, "ticker"][1], "model": "double_ex_smoothing", **FE.KPI(df)}
#         for df in df_double_ex_smoothing
#     ]
# )


# forecasts = pd.concat(
#     df_moving_averages + df_simple_ex_smoothing + df_double_ex_smoothing
# ).reset_index(drop=True)

# errors = pd.concat(
#     [
#         df_errors_moving_averages,
#         df_errors_simple_ex_smoothing,
#         df_errors_double_ex_smoothing,
#     ]
# ).reset_index(drop=True)

# print(
#     "The time used for the for-loop forecast is ", time() - start_time_loop
# )  # 3,88 seconds using list comprehension # 4,39 seconds using list comprehension for fc and errors

# # start_time_loop = time()
# # optimals = [
# #     FO.return_optimal_forecast(
# #         demand_by_ticker.get_group(ticker), extra_periods=24, measure="MAE_abs"
# #     )
# #     for ticker in ticker_list
# # ]
# # optimal_forecast = pd.concat(optimals)

# # optimal_forecast["model"].unique()
# # print("The time used for the list comprehension optimals is ", time() - start_time_loop)
