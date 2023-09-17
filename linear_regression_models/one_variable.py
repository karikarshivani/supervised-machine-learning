import math, copy # TODO: Not sure what these are for
from datetime import date
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('linear_regression_models/lib/deeplearning.mplstyle')
# from lab_utils_uni import plt_gradients, plt_intuition, plt_stationary, plt_update_onclick, soup_bowl
import lib.lab_utils_uni as lab

x_feature_values = [1.0, 1.2, 1.4, 1.6, 1.8, 2.0] # size in 1000 sqft
x_train = np.array(x_feature_values)

y_target_values = [300.0, 350.0, 380.0, 460.0, 485.0, 500.0] # price in 1000s of dollars
y_train = np.array(y_target_values)

w_init = 200
b_init = 100

# Temporary gradient descent settings
iterations = 10000
tmp_alpha = 1.0e-2 # TODO: Understand the format

def compute_gradient(x, y, w, b):
    """
    Computes the gradient for linear regression

    Args:
      x (ndarray (m,)): Data, m examples
      y (ndarray (m,)): target values
      w,b (scalar)    : model parameters

    Returns
      dj_dw (scalar): The gradient of the cost w.r.t. the parameters w
      dj_db (scalar): The gradient of the cost w.r.t. the parameter b     
    """
    m = len(x)
    dj_dw = 0
    dj_db = 0

    for i in range(m):
        f_wb = w * x[i] + b
        dj_dw_i = (f_wb - y [i]) * x[i]
        dj_db_i = f_wb - y[i]
        dj_dw += dj_dw_i
        dj_db += dj_db_i

    dj_dw = dj_dw / m
    dj_db = dj_db / m

    return dj_dw, dj_db

def compute_cost(x, y, w, b): # Calculation cost function J to find ideal value for w
    m = len(x)

    cost = 0
    for i in range(m):
        f_wb = w * x[i] + b
        cost = (f_wb - y[i]) ** 2
    total_cost = (1 / (2*m)) * cost

    return total_cost

def gradient_descent(x, y, w_in, b_in, alpha, num_iters, cost_function, gradient_function):
    """
    Performs gradient descent to fit w,b. Updates w,b by taking 
    num_iters gradient steps with learning rate alpha
    
    Args:
      x (ndarray (m,))  : Data, m examples 
      y (ndarray (m,))  : target values
      w_in,b_in (scalar): initial values of model parameters  
      alpha (float)     : Learning rate
      num_iters (int)   : number of iterations to run gradient descent
      cost_function     : function to call to produce cost
      gradient_function : function to call to produce gradient
      
    Returns:
      w (scalar): Simultaneously updated value of parameter after running gradient descent
      b (scalar): Simultaneously updated value of parameter after running gradient descent
      J_history (List): History of cost values
      p_history (list): History of parameters [w,b] 
    """
    
    # An array to store cost J and parameters at each iteration primarily for graphing later
    J_history = []
    p_history = []
    b = b_in
    w = w_in

    for i in range(num_iters):
        # Calculate the gradient and update the parameters using gradient_function
        dj_dw, dj_db = gradient_function(x, y, w, b)

        # Simultaneously update parameters
        b = b - alpha * dj_db
        w = w - alpha * dj_dw

        # Save cost J at each iteration
        if i < 100000: # prevent resource exhaustion
            J_history.append(cost_function(x, y, w, b))
            p_history.append([w, b])

        # # TODO: Understand the context behind the commented code below before using | Are there typos and are the values hardcoded?
        # # Print cost every at intervals 10 times or as many iterations if < 10
        # if i % math.ceil(num_iters/10) == 0:
        #     print(f"Iteration {i:4}: Cost {J_history[-1]:0.2e}",
        #           f"dj_dw: {dj_dw: 0.3e}, dj_db {dj_db: 0.3e}",
        #           f"w: {w: 0.3e}, b: {b: 0.5e}")


    return w, b, J_history, p_history

def compute_model_output(x, w, b): # Calculating f(x) = wx + b for training data
    m = len(x_feature_values)
    f_wb = np.zeros(m) # TODO: Find out whether preallocation is faster or appending

    for i in range(m):
        f_wb[i] = w * x[i] + b

    return f_wb

w_final, b_final, J_hist, p_hist = gradient_descent(x_train, y_train, w_init, b_init, tmp_alpha, iterations, compute_cost, compute_gradient)

print(f"Final values w: {w_final}, b: {b_final}") # You can use {w_final:8.4f} to limit digits after decimal to 4 (8 is the padded length of the float)

temp_f_wb = compute_model_output(x_train, w_final, b_final)
# lab.plt_gradients(x_train, y_train, compute_cost, compute_gradient)
# plt.show()
plt.plot(x_train, temp_f_wb, c='purple', label='Prediction Graph') # TODO: Find diff between plot and scatter functions
plt.scatter(x_train, y_train, marker='*', c='black', label='Training Data')
plt.title('Housing Prices Model')
plt.ylabel('Price (in 1000 dollars)')
plt.xlabel('Size (in 1000 sqft)')
plt.legend()
plt.savefig(f"linear_regression_models/output/OneVariable-{date.isoformat(date.today())}.png")
plt.show()