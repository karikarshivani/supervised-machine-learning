import numpy as np
import matplotlib.pyplot as plot
# plt.style.use('./deeplearning.mplstyle')

x_feature_values = [1.0, 2.0] # size in 1000 sqft
x_train = np.array(x_feature_values)

y_target_values = [300.0, 500.0] # price in 1000s of dollars
y_train = np.array(y_target_values)

plot.scatter(x_train, y_train, marker='*', c='pink')
plot.title('Housing Prices')
plot.ylabel('Price (in 1000 dollars)')
plot.xlabel('Size (in 1000 sqft)')
plot.show()