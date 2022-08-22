# # %%
# import numpy as np
# from sklearn.linear_model import LinearRegression
# from sklearn.preprocessing import PolynomialFeatures
# import matplotlib.pyplot as plt
# import csv
# import pandas
#
# # %%
# input_path = './data/data.csv'
# df_data = pandas.read_csv(input_path)
#
# print(df_data)
#
# # %%
# print(df_data.columns)
#
# # %%
# rename_map = dict()
# current_columns = df_data.columns
# new_columns = ["model", "L2", "L3", "tdp", "frequency", "turbo", "cores","threads", "process",  "y"]
# for k, v in zip(current_columns, new_columns):
#     rename_map[k] = v
# rename_map
#
# # %%
# named_data = df_data.rename(columns=rename_map)
# named_data
#
# # %%
# named_data = named_data.dropna()
#
#
# def convert_cache_label(text, label=1):
#     # print(f"{text} {label}")
#     text = text.strip()
#     if text.startswith("+"):
#         text = text[1:]
#     if "+" not in text and label == 2:
#         return 0.0
#     else:
#         size_text = text.split("+")[label - 1].strip()
#         if size_text.replace("KB", "").replace("MB", "").strip() == "":
#             return 0.0
#         if "MB" in size_text:
#             return float(size_text.replace("MB", ""))
#         else:
#             return float(size_text.replace("KB", "")) / 1000
#
#
# named_data["cache_l2"] = (float)named_data["L2"]
# # named_data["cache_l2"] = named_data["cache"].apply(lambda x: convert_cache_label(x, 2))
# # named_data["tdp"] = named_data["tdp"].apply(lambda x: int(x))
# # named_data["process"] = named_data["process"].apply(lambda x: int(x))
#
# # named_data
#
#
# # a = named_data["cache"].apply(lambda x: )
# # a[a["cache"] != "<class 'str'>"]
# # a.columns
#
# # %%
# # def convert_frequency(frequency, bottom=True):
# #     # print(frequency)
# #     frequency = frequency.strip().split("‑")
# #     if len(frequency) < 2:
# #         return frequency
# #     if bottom == True:
# #         return frequency[0]
# #     else:
# #         return frequency[1]
# #
# #
# # named_data["min_frequency"] = named_data["frequency"].apply(lambda x: convert_frequency(x, True))
# # named_data["max_frequency"] = named_data["frequency"].apply(lambda x: convert_frequency(x, False))
# # named_data
#
#
# # %%
# # def convert_cores(cores, flag=True):
# #     # print(f"{cores} {type(cores)}")
# #     core_thread = str(cores).split(" ")
# #     if len(core_thread) == 1:
# #         return int(core_thread[0])
# #     if cores == True:
# #         return int((core_thread[0]).strip())
# #     else:
# #         return int((core_thread[1]).strip())
# #
# #
# # named_data["cores"] = named_data["cores"].apply(lambda x: convert_cores(x, True))
# # named_data["threads"] = named_data["cores"].apply(lambda x: convert_cores(x, False))
#
# # named_data["y1"] = 1/named_data["y1"]
# # named_data["y2"] = 1/named_data["y2"]
#
# named_data
#
# # %%
# data = named_data[["cores", "threads", "tdp", "cache_l1", "cache_l2", "min_frequency", "max_frequency", "y1", "y2"]]
# data
#
# # %%
# # Polynomial Fit
# import numpy as np
# from sklearn.preprocessing import PolynomialFeatures
# from sklearn.linear_model import LinearRegression
#
# # data = data.sort_values("cache_l1")
#
# x = np.array(data["cache_l2"].values)
# y = np.array(data["y1"].values)
#
# print(f"count of data x is {len(x)}")
# print(f"count of data y is {len(y)}")
#
# threshold = int(len(x) * 0.8)
#
# x_train = x[:threshold]
# y_train = y[:threshold]
#
# x_test = x[threshold:]
# y_test = y[threshold:]
#
# print(f"count of train is {len(x_train)}")
# print(f"count of test is {len(y_test)}")
#
# data[["cache_l1", "y1"]]
# # x_train
#
# # %%
# poly = PolynomialFeatures(degree=2)
# poly.fit(x_train.reshape(-1, 1))
# x2 = poly.transform(x_train.reshape(-1, 1))
#
# lin_reg = LinearRegression()
# lin_reg.fit(x2, y_train)
# y_predict = lin_reg.predict(x2)
#
# print("prediction completed.")
#
# # %%
# import matplotlib.pyplot as plt
#
# plt.scatter(x_train, y_train)
# plt.plot(np.sort(x_train), y_predict[np.argsort(x_train)], color="r")
#
# plt.xlabel("CPU L1 Cache (MB)")
# plt.ylabel("7-Zip Score")
#
# plt.show()
#
#
#
