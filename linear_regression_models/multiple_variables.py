import copy, math
import numpy as np
from lib.lab_utils_multi import load_house_data, run_gradient_descent, norm_plot, plt_equal_scale, plot_cost_i_w
from lib.lab_utils_common import dlc
import matplotlib.pyplot as plt
plt.style.use('linear_regression_models/lib/deeplearning.mplstyle')
np.set_printoptions(precision=2)  # reduced display precision on numpy arrays

# Note: rows are separate examples (m) | columns are individual features (n)
# four features: sqft, bedrooms, floors, age)
X_train = np.array([
    [2104, 5, 1, 45],
    [1416, 3, 2, 40],
    [852, 2, 1, 35]
])

# target: price
Y_train = np.array([
    460,
    232,
    178
])

w_init = np.array([
    0.39133535,
    18.75376741,
    -53.36032453,
    -26.42131618
])

b_init = 785.1811367994083

def compute_cost(X, y, w, b):
    """
    Compute cost

    Args:
        X (ndarray (m,n)): (m) examples with (n) features
        y (ndarray (m,)) : target values
        w (ndarray (n,)) : model parameters
        b (scalar)       : model parameter

    Returns:
        cost (scalar): difference between expected outcome
    """

    m = X.shape[0]
    cost = 0.0

    for i in range(m):
        f_wb_i = np.dot(X[i], w) + b
        cost += (f_wb_i - y[i])**2

    cost /= 2 * m

    return cost

cost = compute_cost(X_train, Y_train, w_init, b_init)
print(f"Cost at optimal w: {cost}")

def compute_gradient(X, y, w, b):
    """
    Compute gradient
    Args:
        X (ndarray (m,n)): (m) examples with (n) features
        y (ndarray (m,)) : target values
        w (ndarray (n,)) : model parameters
        b (scalar)       : model parameter

    Returns:
        dj_dw (ndarray (n,)): Gradient of the cost wrt w
        dj_db (scalar)      : Gradient of the cost wrt b
    """

    m,n = X.shape # m examples, n features
    dj_dw = np.zeros((n,))
    dj_db = 0.

    for i in range(m):
        err = (np.dot(X[i], w) + b) - y[i]
        for j in range(n):
            dj_dw[j] = dj_dw[j] + err * X[i, j]
        dj_db = dj_db + err
    dj_dw /= m
    dj_db /= m

    return dj_db, dj_dw

tmp_dj_db, tmp_dj_dw = compute_gradient(X_train, Y_train, w_init, b_init)
print(f"dj_db at initial w,b: {tmp_dj_db}")
print(f"dj_dw at initial w,b: {tmp_dj_dw}")

def gradient_descent(X, y, w_in, b_in, cost_function, gradient_function, alpha, num_iters):
    """
    Performs batch gradient descent to learn w and b.
    Updates w and b by taking num_iters gradient steps with learning rate alpha

    Args:
        X (ndarray (m,n))  : (m) examples with (n) features
        y (ndarray (m,))   : target values
        w_in (ndarray (n,)): initial model parameters
        b_in (scalar)      : initial model parameter
        aplha (float)      : learning rate
        num_iters (int)    : number of iterations to run gradient descent
        cost_function      : function to compute cost
        gradient_function  : function to compute gradient

    Returns:
        w (ndarray (n,))   : updated values of parameters
        b (scalar)         : updated value of parameter
    """

    # An array to store cost J and w's at each iteration primarily for graphing later
    J_history = []
    w = copy.deepcopy(w_in) # avoid modifying global w within function # TODO: understand and consider renaming as alternative
    b = b_in

    for i in range(num_iters):

        # Calculate the gradient and update the parameters
        dj_db, dj_dw = gradient_function(X, y, w, b)

        # Update parameters using w, b, alpha and gradient
        w -= alpha * dj_dw
        b -= alpha * dj_db

        # Save cost J at each iteration
        if i < 100000: # prevent resource exhaustion
            J_history.append(cost_function(X, y, w, b))

        # Print cost 10 times for num_iters > 0 or num_iters times
        if i % math.ceil(num_iters/10) == 0:
            print(f"Iteration {i:4d}: Cost {J_history[-1]:8.2f}")

    return w, b, J_history

