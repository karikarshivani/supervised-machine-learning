import time
import numpy as np

# NumPy routines which allocate memory and fill arrays with value
a = np.zeros(4)
print(f"np.zeros(4) :   a = {a}, a shape = {a.shape}, a data type = {a.dtype}")

a = np.random.random_sample(4)
print(f"np.random.random_sample(4): a = {a}, a shape = {a.shape}, a data type = {a.dtype}")

# Note: One value in an np array set as a float will convert all the other elements to a float for consistency
b = np.array([5, 4, 3.0, 2, 1])
print(f"b: {b}")

c = np.array([5, 4, 3, 2, 1])
print(f"c: {c}")

# Slicing
print(c[1:4:2]) # start:stop:step
print(c[2:])

# Single vector operations
d = -c
print(d)
e = d**2
print(e)
f = np.sum(e) # Other functions: mean, 
print(f)

x_vector = [1000, 3, 1, 15] # sqft, # of bedrooms, # of floors, age of house
w_vector = [0.1, 4, 10, -2] # random weights for each feature
b = 80 # random base price | TODO: bias?

# f(w,b) = w_vector * x_vector + b
f = np.dot(w_vector, x_vector) + b

