
Bhuvaneshwaran Ravi, Jayashree Srinivasan, Kameswaran Rangasamy, Serlin Tamilselvam 
bravi19@ubishops.ca, jsrinivasan19@ubishops.ca, krangasamy19@ubishops.ca, stamilselvam19@ubishops.ca


# Logistic Regression using R

## Logistic Regression:

Logistic regression is a machine learning classification algorithm used to assign observations to a discrete set of classes. Given a feature vector X and a qualitative response Y taking values in the set C = {C1 , . . . , Ck }, a classification task is to build a function f : X → Y (classifier) that takes as input the feature vector X and predicts the value for Y , i.e. Y ∈ C. The model or function or classifier f is built using a set of training observations (X1 , y1), . . . , (Xn , Yn) for a given n.

In logistic regression, we relate p(y), the probability of Y belonging to a certain class (which ranges between 0 and 1), to the features X1 , . . . , Xk via the logistic (or logit) transformation given by

p(y) = S(β0 + β1 * x1 + . . . + βk * xk), Where S(w) is the logistic sigmoid function given by s(w) = 1/(1+e^(-w))


## Implementation in R:

The goal of the project is to implement logistic regression classifier using gradient ascent.

### Objective Function