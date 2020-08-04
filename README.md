# Logistic Regression using R


## Logistic Regression:

Logistic regression is a machine learning classification algorithm used to assign observations to a discrete set of classes. Given a feature vector X and a qualitative response Y taking values in the set C = {C1 , . . . , Ck }, a classification task is to build a function f : X → Y (classifier) that takes as input the feature vector X and predicts the value for Y , i.e. Y ∈ C. The model or function or classifier f is built using a set of training observations (X1 , y1), . . . , (Xn , Yn) for a given n.

In logistic regression, we relate p(y), the probability of Y belonging to a certain class (which ranges between 0 and 1), to the features X1 , . . . , Xk via the logistic (or logit) transformation given by

p(y) = S(β0 + β1 * x1 + . . . + βk * xk), Where S(w) is the logistic sigmoid function given by s(w) = 1/(1+e^(-w))

### Maximum Likelihood Estimation (MLE) of the Model

In logistic regression, our goal is to learn a set of parameters βT = (β0 , β1 , ... , βn ) using the available training data. For linear regression, the typical method used is the least squares estimation. Although we could use (non-linear) least squares to fit the logistic regression model, the more general method of maximum likelihood estimation (MLE) is preferred, since it has better statistical properties. The idea behind MLE is to choose the most likely values of the parameters β0 , . . . , βn given the observed sample

<div align="center">{(Xi1 , . . . , Xik , Yi), 1 ≤ i ≤ n}.</div>

In logistic regression, the probability model is based on the binomial distributions:

<div align="center">
	<img src="https://latex.codecogs.com/gif.latex?f(x_{i},p_{i})&space;=&space;f(y_{i},p_{i})&space;=&space;\left\{\begin{matrix}&space;p_{i}&space;&&space;if&space;&&space;y_{i}&space;=&space;1&space;\\&space;1-p_{i}&space;&&space;if&space;&&space;y_{i}&space;=&space;0&space;\end{matrix}\right." title="f(x_{i},p_{i}) = f(y_{i},p_{i}) = \left\{\begin{matrix} p_{i} & if & y_{i} = 1 \\ 1-p_{i} & if & y_{i} = 0 \end{matrix}\right."/>
</div>

where xi = (x1 ,..., xk) is the vector of features and 0 < pi < 1 are the probabilities associated to the binomials in the model. In other words, the probability of the feature vector xi specifying the class yi = 1 occurs with probability pi , that is

<div align="center">
	<img src="https://latex.codecogs.com/gif.latex?p(y_{i}&space;=&space;1)&space;=&space;p_{i}&space;=&space;\frac{e^{\beta_{0}&plus;\beta_{1}x_{1}^{(i)}&plus;...&plus;\beta_{k}x_{k}^{(i)}}}{1&plus;e^{\beta_{0}&plus;\beta_{1}x_{1}^{(i)}&plus;...&plus;\beta_{k}x_{k}^{(i)}}}&space;=&space;\frac{e^{x_{i}^{T}\beta}}{1&plus;e^{x_{i}^{T}\beta}}" title="p(y_{i} = 1) = p_{i} = \frac{e^{\beta_{0}+\beta_{1}x_{1}^{(i)}+...+\beta_{k}x_{k}^{(i)}}}{1+e^{\beta_{0}+\beta_{1}x_{1}^{(i)}+...+\beta_{k}x_{k}^{(i)}}} = \frac{e^{x_{i}^{T}\beta}}{1+e^{x_{i}^{T}\beta}}" />
</div>

Given a dataset with n training examples and k features, then the conditional likelihood L(β) is given by

<div align="center">
	<img src="https://latex.codecogs.com/gif.latex?L(\beta)&space;=&space;\coprod_{i=1}^{n}P(xi)^{yi}((1-P(xi))^{1-yi})" title="L(\beta) = \coprod_{i=1}^{n}P(xi)^{yi}((1-P(xi))^{1-yi})" />
</div>

### Cost/Objective function/Log Likelihood

The cost function for logistic regression is the log of conditional likelihood and it is given by

<div align="center">
	<img src="https://latex.codecogs.com/gif.latex?The&space;Log&space;likelihood,&space;l(\beta&space;)&space;=&space;log(L(\beta&space;))&space;=&space;\sum_{i=1}^{n}[yi(\beta&space;_{0}&plus;\beta&space;_{i}xi)&space;-&space;log(1&plus;e^{\beta&space;_{0}&plus;\beta&space;_{i}xi})]" title="The Log likelihood, l(\beta ) = log(L(\beta )) = \sum_{i=1}^{n}[yi(\beta _{0}+\beta _{i}xi) - log(1+e^{\beta _{0}+\beta _{i}xi})]" />
</div>

### Gradient function

The gradient function to find the local maxima is obtained using the following equation
<div align="center">
	<img src="https://latex.codecogs.com/gif.latex?\frac{\partial&space;l}{\partial&space;\beta&space;_{j}}&space;=&space;\sum_{i=1}^{n}{x_{j}}^{(i)}(y_{i}-p_{(i)})" title="\frac{\partial l}{\partial \beta _{j}} = \sum_{i=1}^{n}{x_{j}}^{(i)}(y_{i}-p_{(i)})" />
</div>


## Implementation in R:

The goal of the project is to implement logistic regression classifier using gradient ascent. Gradient ascent is used to find the best weight and bias. The below algorithm is used to find the optimal weights.

<div><strong>Gradient_Ascent()</strong></div>
<blockquote>
	<div>	1. Set α ∈ [0,1]  (Set learning rate)</div>
	<div>	2. Set ε > 0 (Tolerance Term)</div>
	<div>	3. β0 <- initial value</div>
	<div>	4. for t = 0, 1, ... do</div>
	<blockquote>
		<div>	5. 		Compute the gradient: gt = ∇l(β(t))</div>
		<div>	6.		Update the coefficients: β(t+1) <- β(t) + αgt</div>
		<div>	7.		Iterate until: || β(t+1) − β(t) || < ε</div>
	</blockquote>
	<div>	8. end for</div>
	<div>	9. Return the final coefficients: β(t final)</div>
</blockquote>

The feature variable x1 is normalized before weights are calculated and the following formulae is used to do so
<div align="center"><img src="https://latex.codecogs.com/gif.latex?x_{i}&space;=&space;\frac{x_{i}&space;-&space;mean}{sd}" title="x_{i} = \frac{x_{i} - mean}{sd}" /></div>


### DataSet

Data available at <a href="https://web.stanford.edu/~hastie/ElemStatLearn/datasets/SAheart.data" target="_blank">https://web.stanford.edu/~hastie/ElemStatLearn/datasets/SAheart.data.</a> This data set is a retrospective sample of males in a heart-disease high-risk region of the Western Cape, South Africa. Many of the coronary heart disease (CHD) positive men have undergone blood pressure reduction treatment and other programs to reduce their risk factors after their CHD event. In some cases the measurements were made after these treatments. The class label indicates if the person has a coronary heart disease (negative or positive) and is hidden for our analysis. Individuals are described by the following nine variables. The continuous variables are systolic blood pressure (sbp), cumulative tobacco (tobacco), low density lipoprotein cholesterol (ldl), adiposity, obesity and current alcohol consumption (alcohol). The integer variables are type-A behavior (typea) and age at onset (age). Finally, the binary variable indicates the presence or not of heart disease in the family history(famhist).


### Accuracy

The gradient ascent algorithm to find optimal weights is performed on SA heart dataset. Out of 9 different features available, low density lipoprotein cholesterol(ldl) is selected as a feature to train the model and Coronary heart disease(chd) is predicted. First 100 data is used to train the model and next 362 data is used for testing the accuracy of the model.

#### PARAMETER VALUES


<ul>
	<li>Learning Rate (α)  =  0.001</li>
	<li>Tolerance Term (ε) =  1e-5</li>
	<li>Max Iteration      =  10000</li>
</ul>
<br>

An accuracy of <strong>63%</strong> is obtained on train data(100).	

<br>
An accuracy of <strong>67.67%</strong> is obtained on test data(362).


### Graph with different learning rate

The regression plot is drawn on train data with different values of learning rates. The learning rates used are 1, 0.9, 0.1, 0.001, 1e-5 and 1e-10.

<div align="center">
	<strong>REGRESSION PLOT</strong>
</div>

<div align="center">
	<img src="/plots/regression_line_for_different_alphas.png">
</div>

### Gradient Convergence Analysis

By using the log likelihood function the convergence of gradient ascent is tested. The convergence is tested for various values of learning rate(1, 0.9, 0.1, 0.001, 1e-5 and 1e-10) and the maximum number of iteration is set to 100000. The loglikelihood curve for different iteration is plotted below and it is found that for learning rate values of 0.001 and 1e-5 the curve is constant after sometime. The plots are as follows:

<div align="center">
	<img width="50%" src="/plots/iteration_vs_loglikelihood_for_lr_1.png">
	<img width="50%" src="/plots/iteration_vs_loglikelihood_for_lr_0.9.png">
	<img width="50%" src="/plots/iterationvsloglikelihood_for_lr_0.1.png">
	<img width="50%" src="/plots/iteration_vs_loglikelihood_for_lr_0.9.png">
	<img width="50%" src="/plots/iterationvsloglikelihood_for_lr_1e-5.png">
</div>


### Confusion Matrix

#### Confusion matrix on train data

<div align="center">
	<img align="center" src="/plots/confusion_matrix_train_data.png">
</div>

#### Confusion matrix on test data

<div align="center">
	<img align="center" src="/plots/confusion_matrix_test_data.png">
</div>

## Contributors:

1. <a href="https://github.com/BhuvaneshRavi" target="_blank">Bhuvaneshwaran Ravi</a>
2. <a href="https://github.com/jsri16" target="_blank">Jayashree Srinivasan</a>
3. <a href="https://github.com/rangakamesh" target="_blank">Kameswaran Rangasamy</a>
4. <a href="https://github.com/serlintamilselvam" target="_blank">Serlin Tamilselvam</a>