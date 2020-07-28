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

<div align="center">L(β) = <a href="https://www.codecogs.com/eqnedit.php?latex=\coprod_{i=1}^{n}P(xi)^{yi}((1-P(xi))^{1-yi})" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\coprod_{i=1}^{n}P(xi)^{yi}((1-P(xi))^{1-yi})" title="\coprod_{i=1}^{n}P(xi)^{yi}((1-P(xi))^{1-yi})" /></a></div>


## Implementation in R:

The goal of the project is to implement logistic regression classifier using gradient ascent.

### Objective Function


## Contributors:

1. Bhuvaneshwaran Ravi (bravi19@ubishops.ca) 
2. Jayashree Srinivasan (jsrinivasan19@ubishops.ca)
3. Kameswaran Rangasamy (krangasamy19@ubishops.ca)
4. Serlin Tamilselvam (stamilselvam19@ubishops.ca)