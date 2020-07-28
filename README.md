# Logistic Regression using R


## Logistic Regression:

Logistic regression is a machine learning classification algorithm used to assign observations to a discrete set of classes. Given a feature vector X and a qualitative response Y taking values in the set C = {C1 , . . . , Ck }, a classification task is to build a function f : X → Y (classifier) that takes as input the feature vector X and predicts the value for Y , i.e. Y ∈ C. The model or function or classifier f is built using a set of training observations (X1 , y1), . . . , (Xn , Yn) for a given n.

In logistic regression, we relate p(y), the probability of Y belonging to a certain class (which ranges between 0 and 1), to the features X1 , . . . , Xk via the logistic (or logit) transformation given by

p(y) = S(β0 + β1 * x1 + . . . + βk * xk), Where S(w) is the logistic sigmoid function given by s(w) = 1/(1+e^(-w))

### Maximum Likelihood Estimation (MLE) of the Model

In logistic regression, our goal is to learn a set of parameters βT = (β0 , β1 , ... , βn ) using the available training data. For linear regression, the typical method used is the least squares estimation. Although we could use (non-linear) least squares to fit the logistic regression model, the more general method of maximum likelihood estimation (MLE) is preferred, since it has better statistical properties. The idea behind MLE is to choose the most likely values of the parameters β0 , . . . , βn given the observed sample

<div align="center">{(Xi1 , . . . , Xik , Yi), 1 ≤ i ≤ n}.</div>

In logistic regression, the probability model is based on the binomial distributions:

<div align="center">f(xi, pi) = f(yi, pi) = { pi, if yi = 1 && 1 - pi, if yi = 0}</div>

where xi = (x1 ,..., xk) is the vector of features and 0 < pi < 1 are the probabilities associated to the binomials in the model. In other words, the probability of the feature vector xi specifying the class yi = 1 occurs with probability pi , that is

<div align="center">p(yi = 1) = pi = (e^(β0+β1*x1i+...+βk*xki)/(1+e^(β0+β1*x1i+...+βk*xki)</div>

Given a dataset with n training examples and k features, then the conditional likelihood L(β) is given by

<div align="center">L(β) = <img src="https://latex.codecogs.com/gif.latex?\coprod_{i=1}^{n}P(xi)^{yi}((1-P(xi))^{1-yi})" title="\coprod_{i=1}^{n}P(xi)^{yi}((1-P(xi))^{1-yi})" /></div>

### Cost/Objective function

The cost function for logistic regression is the log of conditional likelihood and it is given by

<div align="center">The Log likelihood, l(β) = log(L(β)) = <img src="https://latex.codecogs.com/gif.latex?\sum_{i=1}^{n}[yi(\beta&space;_{0}&plus;\beta&space;_{i}xi)&space;-&space;log(1&plus;e^{\beta&space;_{0}&plus;\beta&space;_{i}xi})]" title="\sum_{i=1}^{n}[yi(\beta _{0}+\beta _{i}xi) - log(1+e^{\beta _{0}+\beta _{i}xi})]" /></div>

### Gradient function

The gradient function to find the local maxima is obtained using the following equation
<div align="center"><img src="https://latex.codecogs.com/gif.latex?\frac{\partial&space;l}{\partial&space;\beta&space;_{j}}&space;=&space;\sum_{i=1}^{n}{x_{j}}^{(i)}(y_{i}-p_{(i)})" title="\frac{\partial l}{\partial \beta _{j}} = \sum_{i=1}^{n}{x_{j}}^{(i)}(y_{i}-p_{(i)})" /></div>



## Implementation in R:

The goal of the project is to implement logistic regression classifier using gradient ascent. Gradient ascent is used to find the best weight and bias. The below algorithm is used to find the optimal weights.

<div>Gradient_Ascent()</div>
<div>	1. Set α ∈ [0,1]  (Set learning rate)</div>
<div>	2. Set ε > 0 (Tolerance Term)</div>
<div>	3. β0 <- initial value</div>
<div>	4. for t = 0, 1, ... do</div>
<div>	5. 		Compute the gradient: gt = ∇l(β(t))</div>
<div>	6.		Update the coefficients: β(t+1) <- β(t) + αgt</div>
<div>	7.		Iterate until: || β(t+1) − β(t) || < ε</div>
<div>	8. end for</div>
<div>	9. Return the final coefficients: β(t final)</div>

The feature variable x1 is normalized before weights are calculated and the following formulae is used
<div align="center"><img src="https://latex.codecogs.com/gif.latex?x_{i}&space;=&space;\frac{x_{i}&space;-&space;mean}{sd}" title="x_{i} = \frac{x_{i} - mean}{sd}" /></div>

### DataSet

### Accuracy

### Graph with different learning rate

### Confusion Matrix

## Contributors:

1. Bhuvaneshwaran Ravi (bravi19@ubishops.ca) 
2. Jayashree Srinivasan (jsrinivasan19@ubishops.ca)
3. Kameswaran Rangasamy (krangasamy19@ubishops.ca)
4. Serlin Tamilselvam (stamilselvam19@ubishops.ca)