import copy, math
import numpy as np
import matplotlib.pyplot as plt
from lib.lab_utils_multi import load_house_data, run_gradient_descent, norm_plot, plt_equal_scale, plot_cost_i_w
plt.style.use('linear_regression_models/lib/deeplearning.mplstyle')
np.set_printoptions(precision=2)  # reduced display precision on numpy arrays

# Note: rows are separate examples (m) | columns are individual features (n)
# four features: sqft, bedrooms, floors, age)

## DUMMY DATA
    # X_train = np.array([
    #     [2104, 5, 1, 45],
    #     [1416, 3, 2, 40],
    #     [852, 2, 1, 35]
    # ])

    # # target: price
    # Y_train = np.array([
    #     460,
    #     232,
    #     178
    # ])

## LAB DATA

X_train, Y_train = load_house_data()
X_feature_names = ['size (sqft)', 'bedrooms (count)', 'floors (count)', 'age (years)']
Y_target = "price (1000's)"

## Plot dataset against price

fig, ax = plt.subplots(1, 4, figsize=(12,3), sharey=True)
for i in range(len(ax)):
    ax[i].scatter(X_train[:,i], Y_train)
    ax[i].set_xlabel(X_feature_names[i])
ax[0].set_ylabel(Y_target)
# plt.show()

def z_score_normalize_feaures(X):
    """
    computes X, z-score normalized by column

    Args:
        X (ndarray (m,n))       : input data, m examples, n features

    Returns:
        X_norm (ndarray (m,n))  : input normalized by column
        mu (ndarray (n,))       : mean of each feature
        sigma (ndarray (n,))    : standard deviation of each feature
    """

    mu = np.mean(X, axis=0) # In a 2D array | axis=0 : column mean, axis=1 : row mean
    sigma = np.std(X, axis=0)

    X_norm = (X - mu) / sigma

    return (X_norm, mu, sigma)

X_norm, X_mu, X_sigma = z_score_normalize_feaures(X_train)

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
        cost (scalar): difference between expected outcome (ŷ - y)
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


## ALTERNATE GRADIENT DESCENT FROM LAB LIB TO TEST APPROPRIATE LEARNING RATE

w_norm, b_norm, hist = run_gradient_descent(X_norm, Y_train, 10, alpha=1.0e-1)

# Adjust alpha as needed to find a rate that leads to convergence (left graph)
# Change iterations from 10 to more if needed


## Prediction test with one example:

x_house = np.array([1200, 3, 1, 40]) # ref: ['size (sqft)', 'bedrooms (count)', 'floors (count)', 'age (years)']
x_house_norm = (x_house - X_mu) / X_sigma # normalize features using z-score
print(x_house_norm)
x_house_predict = np.dot(x_house_norm, w_norm) + b_norm
print(f"Predicted price = ${x_house_predict*1000:0.0f}") ## TODO: Find out why my prediction is different (Lab: $318709)



