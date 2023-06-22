`x`   Training example feature (input data)
`y`   Training example target (output data)
`m`   Number of training examples
`w`   Weight parameter
`b`   Bias parameter




## Formulae:

### Line graph / Linear model
f<sub>w,b</sub>(x<sup>(i)</sup>) = wx<sup>(i)</sup> + b # This gives you the prediction for example i using parameters w and b

### Cost function J
J(w,b) = $\frac{1}{2m}$ $\sum^{m-1}_{i=0}$ (f<sub>w,b</sub>(x<sup>(i)</sup>) - y<sup>(i)</sup>)<sup>2</sup>

Notes:
(f<sub>w,b</sub>(x<sup>(i)</sup>) - y<sup>(i)</sup>)<sup>2</sup> # This gives you the squared difference between the target value (y) and the prediction (f(x))

$\sum^{m-1}_{i=0}$ # The summation range is from 0 to m-1

### Gradient Descent
Repeat the following until convergence (close to 0):
w = w - $\alpha$ $\frac{dJ(w,b)}{dw}$ # where d is $\delta$
b = b - $\alpha$ $\frac{dJ(w,b)}{db}$ # where d is $\delta$ 