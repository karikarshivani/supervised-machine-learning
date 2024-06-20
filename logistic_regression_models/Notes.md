## FORMULAE:

### Sigmoid Function
g(z) = $\frac{1}{1 + e^-.^z}$ # This creates a curve and outputs between 0 and 1. _Note: Ignore the period used as a workaround for superscript parsing._

### Probability
f<sub>w,b</sub>(x) = P(y = 1 | x; w, b) # Probability that y is 1 given input array x, parameter array w, and parameter b.
P(y=0) + P(y=1) = 1

### Gradient Descent
Note: This is batch gradient descent (uses the entire training data set) and not gradient descent for subsets of training data.

$\frac{dJ(w,b)}{dw}$ = $\frac{1}{m}$ $\sum^{m-1}_{i=0}$ (f<sub>w,b</sub>(x<sup>(i)</sup>) - y<sup>(i)</sup>)x<sub>j<sup>(i)</sup> # where d is $\delta$
$\frac{dJ(w,b)}{db}$ = $\frac{1}{m}$ $\sum^{m-1}_{i=0}$ (f<sub>w,b</sub>(x<sup>(i)</sup>) - y<sup>(i)</sup>) # where d is $\delta$ 

Repeat the following until convergence (close to 0):
w<sub>j</sub> = w<sub>j</sub> - $\alpha$ $\frac{dJ(w,b)}{dw}$ # where d is $\delta$
b = b - $\alpha$ $\frac{dJ(w,b)}{db}$ # where d is $\delta$ 