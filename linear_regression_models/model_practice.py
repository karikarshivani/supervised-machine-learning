import numpy as np

X_training_data = np.array[
    [2104, 5, 1, 45],
    [1416, 3, 2, 40],
    [852, 2, 1, 35]
]

Y_training_data = np.array[460, 232, 178]

# f_wb(x) = w.x + b

def predict(w, x, b):
    y_target = np.dot(w,x) + b
    return y_target

y = predict([5,2], [4,1], 2)


print(y)
print("end")