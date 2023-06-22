import math, copy
# TODO: Not sure what these are for
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('~/Desktop/Projects/AI/supervised-machine-learning/linear_regression_models/deeplearning.mplstyle') # TODO: Relative path worked before but now matplotlib seems to be unhappy |  Low priority - understand its use
from lab_utils_uni import plt_intuition, plt_stationary, plt_update_onclick, soup_bowl
# import lab_utils_uni

x_feature_values = [1.0, 1.2, 1.4, 1.6, 1.8, 2.0] # size in 1000 sqft
x_train = np.array(x_feature_values)

y_target_values = [300.0, 350.0, 380.0, 460.0, 485.0, 500.0] # price in 1000s of dollars
y_train = np.array(y_target_values)

def compute_model_output(x, w, b): # Calculating f(x) = wx + b for training data
    m = len(x_feature_values)
    f_wb = np.zeros(m) # TODO: Find out whether preallocation is faster or appending

    for i in range(m):
        f_wb[i] = w * x[i] + b

    return f_wb

# def compute_cost(x, y, w, b): # Calculation cost function J to find ideal value for w
#     m = len(x)

#     cost_sum = 0
#     for i in range(m):
#         f_wb = w * x[i] + b
#         cost = (f_wb - y[i]) ** 2
#         cost_sum += cost
#     total_cost = (1 / (2*m)) * cost_sum

#     return total_cost

w = 200
b = 100

temp_f_wb = compute_model_output(x_train, w, b)
plt.plot(x_train, temp_f_wb, c='purple', label='Prediction Graph') # TODO: Find diff between plot and scatter functions
plt.scatter(x_train, y_train, marker='*', c='black', label='Training Data')
plt.title('Housing Prices Model')
plt.ylabel('Price (in 1000 dollars)')
plt.xlabel('Size (in 1000 sqft)')
plt.legend()
plt.show()