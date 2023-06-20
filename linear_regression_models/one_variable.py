import numpy as np
import matplotlib.pyplot as plt
from lab_content.lab_utils_uni import plt_intuition, plt_stationary, plt_update_onclick, soup_bowl
plt.style.use('./deeplearning.mplstyle') # TODO: Low priority - understand its use

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