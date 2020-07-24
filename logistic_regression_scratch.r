# LOAD DATA ----
saHeartData <- read.table("http://www-stat.stanford.edu/~tibs/ElemStatLearn/datasets/SAheart.data", sep=",",head=T,row.names=1)
maxIteration <- 100000

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

# DEFINING COST FUNCTION TO CALCULATE WEIGHTTS (LOG LIKELIHOOD) ----


# GRADIENT FUNCTION ----
grad <- function(x, y, beta) {
  w <- x %*% t(beta) # Multiply matrix x with weights(beta)
  scores <- sigmoid(w)
  gradient <- (t(x) %*% (y-scores))
  return(t(gradient))
}

gradientAscend <- function(x, y, learningRate=0.001, noOfIterations=500, toleranceTerm=1e-10) {
  
  # Add x_0 = 1 as the first column
  x0 <- if(is.vector(x)) length(x) else nrow(x)
  if(is.vector(x) || (!all(x[,1] == 1))) x <- cbind(rep(1, x0), x)
  if(is.vector(y)) y <- matrix(y)
  
  noOfFeatures <- ncol(x)
  
  # Initialize the beta(Weights)
  newBeta <- matrix(rep(0, noOfFeatures), nrow=1)
  
  for (i in 1:noOfIterations) {
    previousBeta <- newBeta
    newBeta <- previousBeta + learningRate * grad(x, y, previousBeta)
    if(all(is.na(newBeta))) {
      return (previousBeta)
    }
    if(all(abs(newBeta - previousBeta) < toleranceTerm)) {
      break;
    }
  }
  cat("In Iteration %f BETA DIFFERENCE IS %f\n", i, abs(newBeta - previousBeta))
  return (newBeta)
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


# SPLIT TEST AND TRAIN DATA ----

#TRAIN DATA
xTrainData <- normalizeInput(saHeartData[1:100, 3]) # Normalize xTrainData
yTrainData <- saHeartData[1:100,10]

# COMPUTE BETA MAX
betaMax <- matrix(gradientAscend(x=xTrainData, y=yTrainData, noOfIterations=maxIteration))
trainProbs <- predictProb(xTrainData, betaMax)

# COMPUTE BETA MAX WITH DIFFERENT LEARNING RATE ----
# Finish this part of code tomo (24/7/2020)
# multipleAlpha <- c(0.9, 0.1, 0.001, 1e-5, 1e-7, 1e-10)
# betaMaxArray <- matrix(nrow = length(multipleAlpha), ncol = 1)
# dim(betaMaxArray)
# for (i in 1:length(multipleAlpha)) {
#   betaMaxArray[i][1] <-  matrix(gradientAscend(x=xTrainData, 
#                                                y=yTrainData, 
#                                                learningRate = multipleAlpha[i], 
#                                                noOfIterations=maxIteration,))
# }

# PREDICT ON TRAINED DATA
trainYPred <- predictionAccuracy(yTrainData,trainProbs,"Train")
print("CONFUSION MATRIX FOR TRAIN DATA:");
library(caret)
confusionMatrix(table(trainYPred,yTrainData))


# PLOT REGRESSION GRAPH
plot(xTrainData,jitter(yTrainData, 1), 
     pch = 19, 
     xlab="Low Density Lipoprotein Cholesterol", 
     ylab="Coronary Heart Disease(0 - Negative, 1 - Positive)", 
     main="Logistic Regression on SA Heart Train Data")
abline(h=.5, lty=2)
xPlotData <- seq(min(xTrainData), max(xTrainData), 0.01)
yPredGraphData <- predictProb(xPlotData, betaMax)
lines(xPlotData, yPredGraphData, col = "blue")


# PREDICT ON TEST DATA
xTestData <- normalizeInput(saHeartData[101:dim(saHeartData)[1], 3]) # Normalize xTestData
yTestData <- saHeartData[101:dim(saHeartData)[1], 10]
testProbs <- predictProb(xTestData,betaMax)
testYPred <- predictionAccuracy(yTestData,testProbs,"Train")


# CONFUSION MATRIX ON PREDICTED OUTPUT
print("CONFUSION MATRIX FOR TEST DATA:");
library(caret)
confusionMatrix(table(testYPred,yTestData))

