## GLOSSARY:

`x`   Training example feature (input data)

`y`   Training example target (output data)

`m`   Number of training examples

`w`   Weight parameter

`b`   Bias parameter

$\alpha$ Alpha (learning rate between 0 and 1)

$\bar{x}$<sub>j</sub><sup>(i)</sup>     Vector of x (list of multiple training features)

$\bar{w}$<sub>j</sub><sup>(i)</sup>     Vector of w (list of weights associated with $\bar{x}$)

|        x<sub>1</sub>        |        x<sub>2</sub>        |        x<sub>3</sub>        |        x<sub>4</sub>        |
| --------------------------- | --------------------------- | --------------------------- | --------------------------- |
| x<sub>1</sub><sup>(1)</sup> | x<sub>2</sub><sup>(1)</sup> | x<sub>3</sub><sup>(1)</sup> | x<sub>4</sub><sup>(1)</sup> |
| x<sub>1</sub><sup>(2)</sup> | x<sub>2</sub><sup>(2)</sup> | x<sub>3</sub><sup>(2)</sup> | x<sub>4</sub><sup>(2)</sup> |
| x<sub>1</sub><sup>(3)</sup> | x<sub>2</sub><sup>(3)</sup> | x<sub>3</sub><sup>(3)</sup> | x<sub>4</sub><sup>(3)</sup> |
| x<sub>1</sub><sup>(4)</sup> | x<sub>2</sub><sup>(4)</sup> | x<sub>3</sub><sup>(4)</sup> | x<sub>4</sub><sup>(4)</sup> |



## FORMULAE:

### Line graph / Linear model
f<sub>w,b</sub>(x<sup>(i)</sup>) = wx<sup>(i)</sup> + b # This gives you the prediction for example i using parameters w and b

### Cost function J
J(w,b) = $\frac{1}{2m}$ $\sum^{m-1}_{i=0}$ (f<sub>w,b</sub>(x<sup>(i)</sup>) - y<sup>(i)</sup>)<sup>2</sup>

Notes:
(f<sub>w,b</sub>(x<sup>(i)</sup>) - y<sup>(i)</sup>)<sup>2</sup> # This gives you the squared difference between the target value (y) and the prediction (f(x))

$\sum^{m-1}_{i=0}$ # The summation range is from 0 to m-1

### Gradient Descent
Note: This is batch gradient descent (uses the entire training data set) and not gradient descent for subsets of training data.

$\frac{dJ(w,b)}{dw}$ = $\frac{1}{m}$ $\sum^{m-1}_{i=0}$ (f<sub>w,b</sub>(x<sup>(i)</sup>) - y<sup>(i)</sup>)x<sup>(i)</sup> # where d is $\delta$
$\frac{dJ(w,b)}{db}$ = $\frac{1}{m}$ $\sum^{m-1}_{i=0}$ (f<sub>w,b</sub>(x<sup>(i)</sup>) - y<sup>(i)</sup>) # where d is $\delta$ 

Repeat the following until convergence (close to 0):
w = w - $\alpha$ $\frac{dJ(w,b)}{dw}$ # where d is $\delta$
b = b - $\alpha$ $\frac{dJ(w,b)}{db}$ # where d is $\delta$ 

`Note: Alpha determines how big of step is taken during gradient descent. The partial derivatives for all parameters are calculated before updating the values for them. Even with a fixed/constant learning rate (alpha), the gradient descent steps become smaller due to the partial derivatives reducing as they approach a local minimum. In other words, the slope may be steep for high cost function values so the partial derivative is high - leading to a bigger step during gradient descent. However, the slope gradually becomes gentle as it approaches the local minimum, the partial derivate reduces the length of the step.`

_Note: Set the learning rate extremely small (such as 0.001) to find out if the cost(J) decreases with every iteration. If even a small learning rate doesn't lead to cost (J) decreasing, review the code for bugs. Increase the learning rate (~3 times each time) to make it more efficient without causing a learning rate too large for incorrect learning curve for cost(J)

## DEFINITIONS:

### Feature Scaling

Features can be different ranges of values which can lead to concentrated scatter plots. These values can be scaled with their relation to other values of that feature. The aim is to get the values of the features roughly between -1 to 1. If the actual values of the features are too large or too small, they can be scaled - however, a range of 3 usually works so not every feature needs to be scaled.

#### Methods:
1. Divide by maximum: $\frac{x<sub>1</sub>}{max}$ | This will provide a range from ~0 to 1
2. Mean normalization: $\frac{x<sub>1</sub> - mean}{max - min}$ | This will surround the values around 0 (both negative and positive values)
3. Z-score normalization: $\frac{x<sub>1</sub> - $\mu$}{$\sigma$}$ | An ideal option that can be implemented using standard deviation

