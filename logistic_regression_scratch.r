# LOAD DATA ----
saHeartData <- read.table("http://www-stat.stanford.edu/~tibs/ElemStatLearn/datasets/SAheart.data", sep=",",head=T,row.names=1)
maxIteration <- 50000
bestlearningRate <- 0.001

# NORMALISE INPUT ----
normalizeInput <- function(x) {
  xMean <- mean(x)
  standardDeviation <- sd(x)
  if(all(standardDeviation == 0)) return(x) # if all the values are the same
  return ((x - xMean) / standardDeviation)
}

# SIGMOID FUNCTION (Probability) ----
sigmoid <- function(w) {
  wPrediction <- 1/(1 + exp(-w))
}

# PLOT LOGLIKELIHOOD GRAPH
plotLogLikelihood <- function(trainLogLikelihood, alphaValue) {
  
  xPlotData <- seq(1, length(trainLogLikelihood), 1)
  plot(x=xPlotData, y=trainLogLikelihood,
       type = "l",
       col="green",
       xlab="Iteration",
       ylab="Log likelihood",
       main="No. of iteration vs Log likelihood")
  legend(x = "topright",
         legend = alphaValue, 
         cex = .8,
         title = "Learning Rate",
         pch = 15, 
         col = "green")

}

# DEFINING COST FUNCTION TO CALCULATE WEIGHTTS (LOG LIKELIHOOD) ----
logLikelihood <- function(x, y, beta) {
  logW <- x %*% t(beta)
  likelihood <- y*logW - log((1+exp(logW)))
  return (sum(likelihood))
}

# GRADIENT FUNCTION ----
grad <- function(x, y, beta) {
  w <- x %*% t(beta) # Multiply matrix x with weights(beta)
  scores <- sigmoid(w)
  gradient <- (t(x) %*% (y-scores))
  return(t(gradient))
}

gradientAscend <- function(x, y, learningRate = bestlearningRate, 
                           noOfIterations = 500, 
                           toleranceTerm=1e-5) {
  
  # Add x_0 = 1 as the first column
  x0 <- if(is.vector(x)) length(x) else nrow(x)
  if(is.vector(x) || (!all(x[,1] == 1))) x <- cbind(rep(1, x0), x)
  if(is.vector(y)) y <- matrix(y)
  
  noOfFeatures <- ncol(x)
  localTrainLogLikelihood <- c(0:0)
  # Initialize the beta(Weights)
  newBeta <- matrix(rep(0, noOfFeatures), nrow=1)
  
  for (i in 1:noOfIterations) {
    previousBeta <- newBeta
    localTrainLogLikelihood[i] <- logLikelihood(x, y, newBeta)
    newBeta <- previousBeta + learningRate * grad(x, y, previousBeta)
    if(all(is.na(newBeta))) {
      return (previousBeta)
    }
    if(all(abs(newBeta - previousBeta) < toleranceTerm)) {
      break;
    }
  }
  plotLogLikelihood(localTrainLogLikelihood, learningRate)
  return (list("newBeta" = newBeta, "logLikelihood" = localTrainLogLikelihood))
}

# COMPARE PREDICTED VALUE AND ACTUAL VALUE ----
predictionAccuracy <- function(y,yProbs,title="Train") {
  yPred <- round(yProbs)
  if(is.vector(y)) y <- matrix(y)
  count <- 0
  for (i in 1:nrow(y)) {
    if(y[i][ncol(y)] == yPred[i][ncol(yPred)]) {
      count = count + 1
    }
  }
  predCount <- (count/nrow(y))*100
  cat(sprintf("%s Accuracy: %f \n", title, predCount))
  return (yPred)
}


# PREDICT PROBABILITY
predictProb <- function(x,betaMax) {
  scaledXData <- if(is.vector(x)) length(x) else nrow(x)
  if(is.vector(scaledXData) || (!all(scaledXData[,1] == 1))) scaledXData <- cbind(rep(1, scaledXData), x)
  
  predictionCalculation <- scaledXData %*% betaMax
  predictedProbabilityValues <- sigmoid(predictionCalculation)
  return (predictedProbabilityValues)
}


# PLOT REGRESSION GRAPH FOR DIFFERENT LEARNING RATE ----
regressionGraph <- function(x, y, varyAlpha, colorsArray) {
  
  multipleBetas <- lapply(varyAlpha, 
                          function(alpha) 
                            matrix(gradientAscend(x = x, 
                                                  y = y, 
                                                  learningRate = alpha, 
                                                  noOfIterations = maxIteration)$newBeta))
  plot(x,jitter(y, 1),
       pch = 19,
       xlab="Low Density Lipoprotein Cholesterol",
       ylab="Coronary Heart Disease(0 - Negative, 1 - Positive)",
       main="Regression Line with different learning rates",
       ylim=c(-0.25,2))
  abline(h=.5, lty=2)
  xPlotData <- seq(min(x), max(x), 0.01)

  for (i in 1:length(multipleBetas)) {
    cat(sprintf("Learning rate = %.10f \n", varyAlpha[i]))
    yPredOnDiffAlphaValue <- predictProb(x, multipleBetas[[i]])
    dummy <- predictionAccuracy(y,yPredOnDiffAlphaValue,"Train")
    print("______________________________________________________");
    yPredGraphDataDiffBeta <- predictProb(xPlotData, multipleBetas[[i]])
    lines(xPlotData, yPredGraphDataDiffBeta, col = colorsArray[i], lwd = 2)
  }

  legend(x = "topright", y = 2.1,
         legend = varyAlpha, cex = .8,
         title = "Learning Rates",
         pch = 15, col = colorsArray)
}


# SPLIT TEST AND TRAIN DATA ----

#TRAIN DATA
xTrainData <- normalizeInput(saHeartData[1:100, 3]) # Normalize xTrainData
yTrainData <- saHeartData[1:100,10]


# COMPUTE BETA MAX WITH DIFFERENT LEARNING RATE
multipleAlpha <- c(1, 0.9, 0.1, 0.001, 1e-5, 1e-10)
randColors <- c("Yellow2", "Blue", "Brown", "Orange", "Green4", "Red")
regressionGraph(xTrainData, yTrainData, multipleAlpha, randColors)


# COMPUTE BETA MAX
gradientAscendOutput <- gradientAscend(x=xTrainData, y=yTrainData, noOfIterations=maxIteration)
betaMax <- matrix(gradientAscendOutput$newBeta)
trainProbs <- predictProb(xTrainData, betaMax)

# PREDICT ON TRAINED DATA
cat(sprintf("\n\n\tGradient Ascent converged for Learning Rate = %f\n",bestlearningRate))
trainYPred <- predictionAccuracy(yTrainData,trainProbs,"Train")
# print("CONFUSION MATRIX FOR TRAIN DATA:");
# library(caret)
# confusionMatrix(table(trainYPred,yTrainData))


# PREDICT ON TEST DATA
xTestData <- normalizeInput(saHeartData[101:dim(saHeartData)[1], 3]) # Normalize xTestData
yTestData <- saHeartData[101:dim(saHeartData)[1], 10]
testProbs <- predictProb(xTestData,betaMax)
testYPred <- predictionAccuracy(yTestData,testProbs,"Test")


# CONFUSION MATRIX ON PREDICTED OUTPUT
# print("CONFUSION MATRIX FOR TEST DATA:");
# library(caret)
# confusionMatrix(table(testYPred, yTestData))


# CODE TO PLOT CONFUSION MATRIX ----

# library(cvms)
# library(broom)    # tidy()
# library(tibble)   # tibble()
# d_binomial <- tibble("target" = yTrainData,
#                      "prediction" = trainYPred)
# 
# d_binomial
# basic_table <- table(d_binomial)
# basic_table
# cfm <- broom::tidy(basic_table)
# cfm
# plot_confusion_matrix(cfm,
#                       targets_col = "target",
#                       predictions_col = "prediction",
#                       counts_col = "n",
#                       palette = "Red")