import time
import numpy as np

# NumPy routines which allocate memory and fill arrays with value
a = np.zeros(4)
print(f"np.zeros(4) :   a = {a}, a shape = {a.shape}, a data type = {a.dtype}")

a = np.random.random_sample(4)
print(f"np.random.random_sample(4): a = {a}, a shape = {a.shape}, a data type = {a.dtype}")

## VECTORS

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
e = d ** 2
print(e)
f = e * 2 # Other arithmetic functions: +, -, /, %,
print(f)
g = np.sum(f) # Other functions: mean, 
print(g)

# Multiple vector operations (Requires vectors of same length)
h = e + f # Other arithmetic functions: -, *, /, %,
print(h)

x_vector = [1000, 3, 1, 15] # sqft, # of bedrooms, # of floors, age of house
w_vector = [0.1, 4, 10, -2] # random weights for each feature
b = 80 # random base price | TODO: bias?

# f(w,b) = w_vector * x_vector + b
f = np.dot(w_vector, x_vector) + b # dot function multiplies elements of two vectors and sums up the products

## MATRICES

# (rows, columns)

a = np.zeros((2, 5))
print(a)

b = np.random.random_sample((4, 4))
print(b)

c = np.arange(10).reshape(2, 5)
print(c)

print(c[1, 0])
print(c[0])

d = np.arange(10).reshape(-1, 2) # -1 automatically calculates the number of rows required (as long the length is divisible by the column integer)
print(d)

e = np.arange(18).reshape(3, 6)
print(e)

print(e[0, 3:6:1])
print(e[:, 4])